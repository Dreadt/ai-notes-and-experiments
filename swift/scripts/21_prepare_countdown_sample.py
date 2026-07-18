import json
import os
import random
from pathlib import Path

from modelscope import MsDataset


ROOT = Path("/root/autodl-tmp/ai-notes-and-experiments/swift")
OUT = Path(os.environ.get("COUNTDOWN_OUTPUT_PATH", ROOT / "data" / "countdown_sample_256.jsonl"))
DATASET_ID = os.environ.get("COUNTDOWN_DATASET_ID", "zouxuhong/Countdown-Tasks-3to4")
SAMPLE_SIZE = int(os.environ.get("COUNTDOWN_SAMPLE_SIZE", "256"))
SEED = int(os.environ.get("COUNTDOWN_SAMPLE_SEED", "42"))


def normalize_row(row):
    row = dict(row)
    nums = row.get("nums")
    target = row.get("target")
    if nums is None or target is None:
        return None

    nums_text = ", ".join(str(x) for x in nums)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are solving a countdown arithmetic task. "
                    "Use each provided number exactly once. "
                    "You may use +, -, *, /, and parentheses. "
                    "Finish with the exact format `#### <expression>`."
                ),
            },
            {
                "role": "user",
                "content": f"Using the numbers [{nums_text}], make the target {target}.",
            },
        ],
        "nums": nums,
        "target": target,
    }


def main():
    random.seed(SEED)
    dataset = MsDataset.load(DATASET_ID, split="train")
    rows = list(dataset)
    random.shuffle(rows)

    normalized = []
    for row in rows:
        item = normalize_row(row)
        if item is not None:
            normalized.append(item)
        if len(normalized) >= SAMPLE_SIZE:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"dataset_id={DATASET_ID}")
    print(f"sample_size={len(normalized)}")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
