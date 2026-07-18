import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--output", default="results_hf.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def load_prompts(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    warmup_inputs = tokenizer(
        [prompts[0]["prompt"]],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.inference_mode():
        _ = model.generate(**warmup_inputs, **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    outputs = []
    start = time.perf_counter()

    for batch in batches(prompts, args.batch_size):
        texts = [row["prompt"] for row in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        batch_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - batch_start

        generated = generated.detach().cpu()
        for idx, row in enumerate(batch):
            input_len = int(input_lengths[idx])
            output_ids = generated[idx][input_len:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "output": output_text,
                    "input_tokens": input_len,
                    "output_tokens": int(output_ids.numel()),
                    "batch_latency_sec": batch_elapsed,
                }
            )

    total_elapsed = time.perf_counter() - start
    total_input_tokens = sum(item["input_tokens"] for item in outputs)
    total_output_tokens = sum(item["output_tokens"] for item in outputs)

    result = {
        "engine": "huggingface_transformers",
        "model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(outputs),
        "total_wall_time_sec": total_elapsed,
        "avg_latency_per_prompt_sec": total_elapsed / max(len(outputs), 1),
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "output_tokens_per_sec": total_output_tokens / total_elapsed if total_elapsed else math.nan,
        "end_to_end_tokens_per_sec": (total_input_tokens + total_output_tokens) / total_elapsed if total_elapsed else math.nan,
        "results": outputs,
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
