import argparse
import json
import math
import time
from pathlib import Path

from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    # vLLM expects this property on the tokenizer base class, but some
    # tokenizer/code combinations in this environment expose only
    # `all_special_tokens`.
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--output", default="results_vllm.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def load_prompts(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    _ = llm.generate([prompts[0]["prompt"]], sampling_params=sampling_params, use_tqdm=False)

    outputs = []
    start = time.perf_counter()

    for batch in batches(prompts, args.batch_size):
        texts = [row["prompt"] for row in batch]
        input_token_ids = tokenizer(texts, add_special_tokens=True)["input_ids"]

        batch_start = time.perf_counter()
        generated = llm.generate(texts, sampling_params=sampling_params, use_tqdm=False)
        batch_elapsed = time.perf_counter() - batch_start

        for idx, row in enumerate(batch):
            item = generated[idx]
            prompt_tokens = len(input_token_ids[idx])
            output_text = item.outputs[0].text
            output_token_count = len(item.outputs[0].token_ids)
            outputs.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "output": output_text,
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_token_count,
                    "batch_latency_sec": batch_elapsed,
                }
            )

    total_elapsed = time.perf_counter() - start
    total_input_tokens = sum(item["input_tokens"] for item in outputs)
    total_output_tokens = sum(item["output_tokens"] for item in outputs)

    result = {
        "engine": "vllm",
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
