import json
import sys
from pathlib import Path


def try_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python 50_summarize_json_metrics.py <output_dir>")

    root = Path(sys.argv[1])
    candidates = sorted(root.rglob("trainer_state.json"))
    if not candidates:
        print("No trainer_state.json found.")
        return

    for path in candidates:
        data = try_load_json(path)
        print(f"== {path} ==")
        if not data:
            print("Unreadable JSON")
            continue
        print(f"global_step: {data.get('global_step')}")
        print(f"best_metric: {data.get('best_metric')}")
        history = data.get("log_history", [])
        if history:
            print(f"log_history_entries: {len(history)}")
            print(f"last_log_keys: {sorted(history[-1].keys())}")
        else:
            print("log_history_entries: 0")


if __name__ == "__main__":
    main()
