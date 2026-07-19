# SWIFT GRPO Experiment Design

This directory is a runnable experiment scaffold for learning the `SWIFT` RL architecture on this machine.

Target machine:

- `2 x RTX 3090 24GB`
- `torch 2.8.0+cu128`
- `vllm 0.10.2`
- `ms-swift 4.4.1`

## Goal

Use a very small text-only GRPO task to understand:

- what `rollout` is
- what `GRPO` is optimizing
- how `SWIFT` connects training and `vLLM`
- how to set up a two-GPU experiment that is likely to run on this machine

## Experiment Structure

### Stage 0: Environment Prep

Install `ms-swift`, verify CUDA visibility, and keep model downloads on `ModelScope`.

### Stage 1: Smoke Test

Single-GPU, colocate mode, tiny model.

Purpose:

- prove the dataset format works
- prove the reward plugin works
- prove `swift rlhf` can start

Recommended model:

- `Qwen/Qwen3-0.6B`

### Stage 2: Main Architecture Experiment

Two-GPU, external rollout server mode on a reference dataset.

Architecture:

- `GPU 0`: `swift rollout` server
- `GPU 1`: `swift rlhf` trainer

Purpose:

- understand training/inference separation
- reduce memory coupling
- match the official `server` mode architecture

Recommended model:

- `Qwen/Qwen3-0.6B`
- if stable, upgrade to `Qwen/Qwen3-1.7B`

Recommended reference dataset:

- `zouxuhong/Countdown-Tasks-3to4`

### Stage 3: Optional Stress Test

Two-GPU, colocate mode.

Purpose:

- compare with server mode
- observe memory pressure and throughput tradeoffs

This was expected to be more fragile than Stage 2, but on this machine `Qwen/Qwen3-1.7B` in `2GPU colocate` did complete a 60-step smoke run.

## Files

- `data/arithmetic_grpo_train.jsonl`: tiny local GRPO dataset
- `plugins/arithmetic_plugin.py`: exact-match and format rewards
- `scripts/00_install_ms_swift.sh`: install commands
- `scripts/10_check_env.sh`: sanity checks
- `scripts/20_run_smoke_colocate_1gpu.sh`: single-GPU smoke test
- `scripts/21_prepare_countdown_sample.py`: create a local small subset from the reference dataset
- `scripts/22_run_countdown_smoke_1gpu.sh`: single-GPU reference-dataset smoke test
- `scripts/30_start_rollout_server.sh`: external rollout server on one GPU
- `scripts/31_run_grpo_server_mode.sh`: trainer on another GPU
- `scripts/32_run_countdown_server_mode.sh`: two-GPU server-mode run on the reference dataset
- `scripts/40_run_colocate_2gpu_optional.sh`: optional two-GPU colocate run

## Why This Design

This design matches the current `SWIFT` docs:

- `GRPO` uses multiple sampled completions per prompt and normalizes rewards within the group.
- `SWIFT` supports `vllm_mode colocate` and `vllm_mode server`.
- `server` mode launches rollout separately with `swift rollout`.

On this machine, the tested path is:

1. run Stage 1 with `Qwen/Qwen3-0.6B`
2. run Stage 2 with `Qwen/Qwen3-0.6B`
3. run `Qwen/Qwen3-1.7B` in `1GPU colocate`
4. run `Qwen/Qwen3-1.7B` in `2GPU colocate`

Current status from experiments on Saturday, July 18, 2026:

- `0.6B 2GPU server-mode`: succeeded
- `1.7B 1GPU colocate`: succeeded
- `1.7B 1GPU colocate + 1024 sample dataset`: succeeded
- `1.7B 2GPU colocate`: training completed through `60/60` steps and saved `checkpoint-60`
- `1.7B 2GPU colocate + 1024 sample dataset`: training completed through `160/160` steps and saved `checkpoint-160`
- `1.7B 2GPU server-mode`: failed during rollout communicator initialization
- `4B 2GPU server-mode`: failed during rollout communicator initialization

Interpretation:

- the current blocker is specifically the `server-mode` cross-process communication path for larger models on this stack
- true multi-GPU `SWIFT` is still practical here through `colocate` with tensor parallelism
- the remaining issue in `2GPU colocate` is teardown stability after training completes, not training execution itself

## Dataset Design

There are now two dataset layers in this directory:

### A. Toy local dataset

The task is intentionally small:

- short arithmetic prompts
- deterministic answers
- easy exact-match reward
- no external API or judge model

This keeps the experiment focused on the RL pipeline rather than data complexity.

Each row contains:

- `messages`: required by `SWIFT`
- `solution`: required by the custom reward

### B. Reference dataset path

The main practical reference is:

- `zouxuhong/Countdown-Tasks-3to4`

Use it in two phases:

1. sample a small local subset first
2. run the full remote dataset only after the subset pipeline is stable

This is the recommended path because it is the official-style GRPO practice dataset and still uses verifiable rewards.

## Reward Design

Two rewards are registered in `plugins/arithmetic_plugin.py`:

- `arith_exact`: reward `1.0` when the final answer matches the dataset `solution`
- `arith_format`: reward `1.0` when the completion contains a final answer in `#### answer` format

This mirrors the common GRPO pattern:

- one reward for correctness
- one reward for output format

For the reference countdown dataset, the preferred setup is to use the dataset's existing verifiable structure rather than the toy reward plugin whenever possible.

## Pass Criteria

The experiment counts as successful when all of the following are true:

1. `swift rollout` starts without crashing
2. `swift rlhf` starts and performs at least a few optimizer steps
3. checkpoints are written under the output directory
4. reward metrics are logged
5. at least some completions get non-zero reward

## Recommended Runtime Settings

Start conservative:

- `num_generations=4`
- `generation_batch_size=4`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1`
- `max_completion_length=128`
- `tuner_type=lora`

If memory is tight, reduce:

- `vllm_gpu_memory_utilization`
- `num_generations`
- `max_completion_length`

If `colocate` OOMs, reduce rollout pressure first before switching architectures:

- lower `vllm_gpu_memory_utilization`
- lower `num_generations`
- lower `max_completion_length`
- keep `offload_optimizer=true`
- keep `offload_model=true`

On this machine, switching back to `server` mode is not currently a reliable fallback for `1.7B` or `4B`.

## Important Notes

- Prefer `ModelScope` instead of direct Hugging Face in this environment.
- Use a dedicated Python env for `SWIFT`. Current local Python is `3.12`; if installation/runtime issues appear, use Python `3.10` or `3.11`.
- In `GRPO`, `num_generations` must be divisible by the total sampling batch size arrangement.
- For custom rewards, keep fields like `solution` outside `messages`.

## Suggested Run Order

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/swift
bash scripts/10_check_env.sh
bash scripts/20_run_smoke_colocate_1gpu.sh
python scripts/21_prepare_countdown_sample.py
bash scripts/22_run_countdown_smoke_1gpu.sh
bash scripts/30_start_rollout_server.sh
bash scripts/32_run_countdown_server_mode.sh
```

## References

- SWIFT installation: https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html
- SWIFT GRPO: https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html
- SWIFT command parameters: https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
- SWIFT custom dataset: https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html
- SWIFT reward plugin guide: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md
