# Qwen3-0.6B SFT Experiment

This project is a minimal AutoDL-friendly supervised fine-tuning setup for `Qwen/Qwen3-0.6B` using:

- `accelerate` for multi-GPU training
- Hugging Face model/tokenizer save format
- `wandb` online logging with offline fallback
- assistant-only loss masking with `-100`

## Install

```bash
cd /root/autodl-tmp/qwen3-0.6b-sft-exp
pip install -r requirements.txt
```

## Train

```bash
accelerate launch --config_file accelerate_config.yaml train.py \
  --dataset-name shibing624/alpaca-zh \
  --dataset-split train \
  --output-dir outputs/qwen3_0.6b_sft \
  --max-train-samples 2000 \
  --max-eval-samples 200 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --wandb-project qwen3-0.6b-sft \
  --wandb-mode auto
```

If `huggingface.co` is blocked on AutoDL, the script automatically switches to `https://hf-mirror.com` when that mirror is reachable.

Manual override:

```bash
export HF_ENDPOINT=https://hf-mirror.com
accelerate launch --config_file accelerate_config.yaml train.py ...
```

## W&B

Auto mode tries online logging first. If the machine cannot reach W&B, the script falls back to offline mode and stores runs locally under `wandb/`.

Manual modes:

```bash
WANDB_API_KEY=... accelerate launch --config_file accelerate_config.yaml train.py --wandb-mode online
accelerate launch --config_file accelerate_config.yaml train.py --wandb-mode offline
accelerate launch --config_file accelerate_config.yaml train.py --wandb-mode disabled
```

Offline sync:

```bash
wandb sync wandb/offline-run-*
```

## What To Study

1. Hugging Face save/load format: `config.json`, `model.safetensors`, tokenizer files.
2. SFT data construction: prompt tokens masked with `-100`, assistant tokens supervised.
3. Cross-entropy loss: `-log p(target_token | previous_tokens)`.
4. Multi-GPU basics: process rank, gradient accumulation, saving only on the main process.
