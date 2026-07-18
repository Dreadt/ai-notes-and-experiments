#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
PATCH_DIR="/root/autodl-tmp/ai-notes-and-experiments/swift/pythonpath_patch"

export USE_HF=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"

swift rollout \
  --model "${MODEL_ID}" \
  --vllm_tensor_parallel_size 1 \
  --vllm_data_parallel_size 1 \
  --port "${PORT}"
