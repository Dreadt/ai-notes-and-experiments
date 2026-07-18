#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv_sys/bin/python"

"${PYTHON_BIN}" - <<'PY'
import art
import art.local
from art.local import LocalBackend
from art.pipeline_trainer.trainer import PipelineTrainer

print("art import ok")
print("art version module:", art.__file__)
print("LocalBackend:", LocalBackend)
print("PipelineTrainer:", PipelineTrainer)
PY
