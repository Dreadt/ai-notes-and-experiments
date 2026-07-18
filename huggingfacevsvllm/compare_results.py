import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare_results.py <hf.json> <vllm.json>")

    left = load(sys.argv[1])
    right = load(sys.argv[2])

    print("Comparison")
    print(f"- left engine: {left['engine']}")
    print(f"- right engine: {right['engine']}")
    print(f"- prompt count: {left['prompt_count']} vs {right['prompt_count']}")
    print(f"- batch size: {left['batch_size']} vs {right['batch_size']}")
    print(f"- max_new_tokens: {left['max_new_tokens']} vs {right['max_new_tokens']}")
    print()

    left_tps = left["output_tokens_per_sec"]
    right_tps = right["output_tokens_per_sec"]
    left_lat = left["avg_latency_per_prompt_sec"]
    right_lat = right["avg_latency_per_prompt_sec"]

    print(f"Output tokens/sec: {left_tps:.2f} vs {right_tps:.2f}")
    if left_tps > 0:
        print(f"Speedup (right/left): {right_tps / left_tps:.2f}x")
    print(f"Avg latency/prompt: {left_lat:.4f}s vs {right_lat:.4f}s")
    if right_lat > 0:
        print(f"Latency ratio (left/right): {left_lat / right_lat:.2f}x")


if __name__ == "__main__":
    main()
