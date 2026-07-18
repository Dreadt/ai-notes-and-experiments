#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/ai-notes-and-experiments/swift"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/data/countdown_sample_256.jsonl}"
PLUGIN_PATH="${ROOT_DIR}/plugins/countdown_plugin.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/countdown_colocate_1p7b_smoke}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
PATCH_DIR="${ROOT_DIR}/pythonpath_patch"

export USE_HF=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ID}" \
  --dataset "${DATASET_PATH}" \
  --external_plugins "${PLUGIN_PATH}" \
  --reward_funcs countdown_correct countdown_format \
  --reward_weights 1.0 0.2 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps 60 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --max_length 384 \
  --max_completion_length 64 \
  --num_generations 2 \
  --generation_batch_size 2 \
  --eval_steps 20 \
  --save_steps 20 \
  --logging_steps 1 \
  --beta 0.0 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.50 \
  --output_dir "${OUTPUT_DIR}"
