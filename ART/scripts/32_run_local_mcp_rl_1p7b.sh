#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv_sys/bin/python"
RUNTIME_LIB_DIR="${ROOT_DIR}/.runtime_libs/lib"
CACHE_ROOT="${ROOT_DIR}/.cache_runtime"

export PATH="${ROOT_DIR}/.venv_sys/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export ART_ART_PATH="${ART_ART_PATH:-${ROOT_DIR}/.art_local}"
export ART_BASE_MODEL="${ART_BASE_MODEL:-Qwen/Qwen3-1.7B}"
export ART_MODEL_PREFIX="${ART_MODEL_PREFIX:-qwen3-1p7b-local-mcp-rl}"
export ART_MAX_STEPS="${ART_MAX_STEPS:-1}"
export ART_ROLLOUTS_PER_SCENARIO="${ART_ROLLOUTS_PER_SCENARIO:-4}"
export ART_MAX_TOKENS="${ART_MAX_TOKENS:-48}"
export ART_ROLLOUT_TEMPERATURE="${ART_ROLLOUT_TEMPERATURE:-1.0}"
export ART_EVAL_TEMPERATURE="${ART_EVAL_TEMPERATURE:-0.2}"
export ART_MIN_BATCH_SIZE="${ART_MIN_BATCH_SIZE:-2}"
export ART_DISCARD_QUEUE_MULTIPLIER="${ART_DISCARD_QUEUE_MULTIPLIER:-200}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export ART_VLLM_RUNTIME_CACHE_DIR="${ART_VLLM_RUNTIME_CACHE_DIR:-${CACHE_ROOT}/art_vllm_runtime}"
mkdir -p "${UV_CACHE_DIR}" "${ART_VLLM_RUNTIME_CACHE_DIR}"

if [[ -f "${RUNTIME_LIB_DIR}/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${RUNTIME_LIB_DIR}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
  export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" art_experiment/local_mcp_rl.py
