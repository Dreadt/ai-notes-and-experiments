import argparse
import math
import os
import socket
from dataclasses import dataclass
from typing import Any


def _can_reach_host(host: str, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


if not os.environ.get("HF_ENDPOINT") and not _can_reach_host("huggingface.co") and _can_reach_host("hf-mirror.com"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT Qwen3-0.6B with Accelerate")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-name", default="shibing624/alpaca-zh")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--validation-split-percentage", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/qwen3_0.6b_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb-project", default="qwen3-0.6b-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["autodl", "accelerate", "multi-gpu", "qwen3-0.6b", "sft"],
    )
    return parser.parse_args()


def can_reach_wandb(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection(("api.wandb.ai", 443), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_wandb_mode(requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "online" if can_reach_wandb() else "offline"


def can_reach_host(host: str, timeout: float = 2.0) -> bool:
    return _can_reach_host(host, timeout)


def maybe_enable_hf_mirror() -> str | None:
    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"]
    if can_reach_host("huggingface.co"):
        return None
    if can_reach_host("hf-mirror.com"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        return os.environ["HF_ENDPOINT"]
    return None


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_example_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()
    output = (example.get("output") or example.get("response") or "").strip()
    if user_input:
        user_text = f"{instruction}\n\n{user_input}" if instruction else user_input
    else:
        user_text = instruction
    if not user_text or not output:
        return []
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]


def preprocess_example(example: dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> dict[str, list[int]]:
    messages = build_example_messages(example)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full = tokenizer(full_text, max_length=max_length, truncation=True)
    prompt = tokenizer(prompt_text, max_length=max_length, truncation=True)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]

    if all(label == IGNORE_INDEX for label in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForSFT:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def load_and_prepare_datasets(args: argparse.Namespace, tokenizer: AutoTokenizer) -> DatasetDict:
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    if args.dataset_split in dataset:
        train_dataset = dataset[args.dataset_split]
    else:
        train_dataset = dataset["train"]

    split_dataset = train_dataset.train_test_split(
        test_size=args.validation_split_percentage / 100.0,
        seed=args.seed,
    )

    if args.max_train_samples:
        split_dataset["train"] = split_dataset["train"].select(range(min(args.max_train_samples, len(split_dataset["train"]))))
    if args.max_eval_samples:
        split_dataset["test"] = split_dataset["test"].select(range(min(args.max_eval_samples, len(split_dataset["test"]))))

    processed = split_dataset.map(
        lambda example: preprocess_example(example, tokenizer, args.max_length),
        remove_columns=split_dataset["train"].column_names,
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing dataset",
    )

    processed = processed.filter(lambda example: len(example["input_ids"]) > 0, desc="Dropping empty examples")
    return processed


def evaluate(model: AutoModelForCausalLM, dataloader: DataLoader, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    losses = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        gathered = accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].size(0)))
        losses.append(gathered)

    loss_tensor = torch.cat(losses)
    mean_loss = loss_tensor.mean().item()
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    model.train()
    return {"eval/loss": mean_loss, "eval/perplexity": perplexity}


def save_checkpoint(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    step: int,
) -> None:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(checkpoint_dir)
        # Save optimizer/scheduler/RNG state for real training resumption.
        accelerator.save_state(os.path.join(checkpoint_dir, "accelerate_state"))


def main() -> None:
    args = parse_args()
    mixed_precision = "bf16" if args.bf16 else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if args.wandb_mode != "disabled" else None,
        project_dir=args.output_dir,
    )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    hf_endpoint = maybe_enable_hf_mirror()
    wandb_mode = resolve_wandb_mode(args.wandb_mode)
    if accelerator.is_main_process:
        if hf_endpoint:
            print(f"Resolved HF endpoint: {hf_endpoint}")
        print(f"Resolved W&B mode: {wandb_mode}")

    if args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = wandb_mode
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "name": args.wandb_name,
                    "tags": args.wandb_tags + [f"wandb:{wandb_mode}"],
                    "dir": os.getcwd(),
                }
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else None,
    )

    datasets = load_and_prepare_datasets(args, tokenizer)
    collator = DataCollatorForSFT(tokenizer)
    train_dataloader = DataLoader(
        datasets["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        datasets["test"],
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.logging_steps == 0:
                    log_payload = {
                        "train/loss": loss.detach().float().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step / max(len(train_dataloader), 1)),
                    }
                    accelerator.log(log_payload, step=global_step)

                if global_step % args.eval_steps == 0:
                    metrics = evaluate(model, eval_dataloader, accelerator)
                    accelerator.log(metrics, step=global_step)

                if global_step % args.save_steps == 0:
                    save_checkpoint(accelerator, model, tokenizer, args.output_dir, global_step)

                if global_step >= max_train_steps:
                    break

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(args.output_dir)

    final_metrics = evaluate(model, eval_dataloader, accelerator)
    accelerator.log(final_metrics, step=global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
