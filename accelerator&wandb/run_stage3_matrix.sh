#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb"
cd "$ROOT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p logs

run_experiment() {
  local run_name="$1"
  local output_dir_name="${2:-$1}"
  if [[ $# -ge 2 ]]; then
    shift
  fi
  shift

  local output_dir="outputs/${output_dir_name}"
  local log_file="logs/${run_name}.log"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${run_name}" | tee -a "$log_file"

  accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset-name shibing624/alpaca-zh \
    --output-dir "$output_dir" \
    --num-train-epochs 2 \
    --per-device-train-batch-size 2 \
    --per-device-eval-batch-size 2 \
    --gradient-accumulation-steps 8 \
    --logging-steps 10 \
    --eval-steps 50 \
    --save-steps 50 \
    --bf16 \
    --wandb-mode offline \
    --wandb-project qwen3-0.6b-sft \
    --wandb-name "$run_name" \
    "$@" 2>&1 | tee -a "$log_file"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] finished ${run_name}" | tee -a "$log_file"
}

run_experiment "exp-stage3-full-lr2e5" "exp-stage3-full-lr2e5-resumed" \
  --model-name outputs/exp-stage3-full-lr2e5/checkpoint-1050 \
  --max-train-samples 48818 \
  --max-eval-samples 500 \
  --learning-rate 2e-5 \
  --eval-steps 100 \
  --save-steps 500 \
  --wandb-tags autodl accelerate multi-gpu qwen3-0.6b sft stage3 full-data lr-2e-5

run_experiment "exp-stage3-lr1e5" \
  --max-train-samples 10000 \
  --max-eval-samples 500 \
  --learning-rate 1e-5 \
  --eval-steps 100 \
  --save-steps 500 \
  --wandb-tags autodl accelerate multi-gpu qwen3-0.6b sft stage3 lr-sweep lr-1e-5

run_experiment "exp-stage3-lr3e5" \
  --max-train-samples 10000 \
  --max-eval-samples 500 \
  --learning-rate 3e-5 \
  --eval-steps 100 \
  --save-steps 500 \
  --wandb-tags autodl accelerate multi-gpu qwen3-0.6b sft stage3 lr-sweep lr-3e-5
