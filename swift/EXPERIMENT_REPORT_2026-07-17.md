# SWIFT Countdown GRPO Experiment Report

Date: Friday, July 17, 2026

## Scope

This experiment verifies that a practical `SWIFT` text RL workflow can run end-to-end on this machine with:

- reference dataset: `zouxuhong/Countdown-Tasks-3to4`
- model: `Qwen/Qwen3-0.6B`
- RL algorithm: `GRPO`
- inference backend: `vLLM`
- architecture: `2GPU server-mode`

The goal was not to maximize final task accuracy. The goal was to prove the architecture, reward path, and rollout-trainer separation.

## Dataset

Local sample used for training:

- `data/countdown_sample_256.jsonl`
- 256 rows sampled from `zouxuhong/Countdown-Tasks-3to4`

Each sample was converted into:

- `messages`
- `nums`
- `target`

The task asks the model to use all given numbers exactly once and produce a final expression in `#### <expression>` format.

## Reward Setup

Custom plugin:

- `plugins/countdown_plugin.py`

Registered rewards:

- `countdown_correct`
- `countdown_format`

Reward meaning:

- `countdown_correct`: reward `1.0` only if the final expression is valid, uses the exact required numbers, and evaluates to the target
- `countdown_format`: reward `1.0` if the model outputs a parseable final expression

Training used reward weights:

- correctness: `1.0`
- format: `0.2`

## Runs

### 1GPU smoke test

Output:

- `outputs/countdown_smoke_1gpu/v1-20260718-013029`

Result:

- succeeded
- entered real training
- wrote checkpoints
- logged non-zero rewards

Quick metric summary:

- reward rows: `26`
- rows with positive correctness reward: `9`
- max reward: `0.70000005`
- average reward: `0.2288`

### 2GPU main architecture run

Architecture:

- `GPU 0`: `swift rollout`
- `GPU 1`: `swift rlhf`

Output:

- `outputs/countdown_server_mode_2gpu/v0-20260718-015658`

Result:

- succeeded
- trainer connected to external vLLM rollout server
- completed full training to `global_step 1024/1024`
- wrote periodic checkpoints every `20` steps and final `checkpoint-1024`

Final runtime metrics:

- train runtime: `478.4928s`
- train steps per second: `2.14`
- train samples per second: `0.535`
- reported trainer-side memory: `1.72 GiB`
- trainable LoRA params: `5.0463M` of `601.0962M` total (`0.8395%`)

Reward summary from `logging.jsonl`:

- reward rows: `256`
- rows with positive correctness reward: `1`
- rows with full format reward: `233`
- max correctness mean: `0.25`
- max total reward: `0.45000002`
- average total reward: `0.1941`
- average format mean: `0.9658`
- average correctness mean: `0.001`

## Interpretation

This run proves the `SWIFT + vLLM + GRPO` pipeline is operational on local multi-GPU hardware.

What worked:

- rollout server mode is stable on this machine
- reward plugin registration works
- verifiable-reward dataset path works
- checkpoints, logs, and trainer state are all written correctly
- server mode keeps trainer-side memory low because rollout memory sits on another GPU

What did not yet improve much:

- the model mostly learned or preserved output formatting
- correctness reward remained close to zero in the 2GPU full run

This is expected for a very small base model, one short epoch, and conservative settings. The experiment should be treated as an architecture validation run, not a performance run.

## RL Concept Mapping

Observed in this experiment:

- `Rollout`: vLLM generates multiple sampled completions for each prompt
- `num_generations=4`: each prompt forms a group of 4 sampled answers
- `GRPO`: SWIFT compares rewards inside that group and updates the policy using relative advantage, without a separate value model
- `Reward plugin`: converts model outputs into scalar rewards the trainer can optimize

In practice here, the rollout side mostly produced correctly formatted expressions, but only rarely produced mathematically correct answers. That is why format reward stayed high while correctness reward stayed near zero.

## Practical Next Steps

Recommended next experiments:

1. Increase training signal quality:
   - raise correctness weight
   - reduce format weight
   - increase epochs
2. Increase model capacity:
   - try `Qwen/Qwen3-1.7B`
3. Increase task stability:
   - start with easier Countdown subsets
   - shorten prompt wording
4. Compare architectures:
   - `2GPU server-mode` vs `1GPU colocate`
5. Compare learning behavior:
   - `num_generations=4` vs `8`

### 2GPU colocate smoke test for `Qwen/Qwen3-1.7B`

Date:

- Saturday, July 18, 2026

Command shape:

- `torchrun --nproc_per_node 2`
- `vllm_mode=colocate`
- `vllm_tensor_parallel_size=2`
- `vllm_gpu_memory_utilization=0.30`
- `offload_optimizer=true`
- `offload_model=true`

Output:

- `outputs/countdown_colocate_2gpu_1p7b_smoke/v0-20260718-125704`

Result:

- training itself succeeded
- completed full `60/60` steps
- saved `checkpoint-20`, `checkpoint-40`, `checkpoint-60`
- logged non-zero format reward and intermittent full correctness reward
- final trainer summary was written
- process exited non-cleanly after completion because `rank1` ended with `SIGSEGV` during teardown

Final runtime metrics:

- train runtime: `285.7s`
- train steps per second: `0.21`
- trainer-reported iteration speed: about `4.76s/it`
- reported memory: `10.82 GiB`
- last checkpoint: `outputs/countdown_colocate_2gpu_1p7b_smoke/v0-20260718-125704/checkpoint-60`

Interpretation:

- this proves multi-GPU `SWIFT` is workable on this machine through `colocate`
- the larger-model blocker is not "multi-GPU in general"
- the unstable part is the current `server-mode` stack for `1.7B+`, which fails much earlier during rollout communicator setup
- the post-training `SIGSEGV` is a shutdown-path bug or compatibility issue, but it does not invalidate the training artifacts that were already saved

### 2GPU colocate main run for `Qwen/Qwen3-1.7B` with 1024 samples

Date:

- Saturday, July 18, 2026

Command shape:

- `torchrun --nproc_per_node 2`
- `vllm_mode=colocate`
- `vllm_tensor_parallel_size=2`
- `vllm_gpu_memory_utilization=0.30`
- `offload_optimizer=true`
- `offload_model=true`

Dataset:

- `data/countdown_sample_1024.jsonl`

Output:

- `outputs/countdown_colocate_2gpu_1p7b_1k/v0-20260718-133335`

Result:

- training itself succeeded
- completed full `160/160` steps
- saved `checkpoint-40`, `checkpoint-80`, `checkpoint-120`, `checkpoint-160`
- logged repeated non-zero format reward
- logged intermittent partial and full correctness reward, including `rewards/CountdownCorrect/mean = 1`
- final trainer summary was written
- process exited non-cleanly after completion because `rank1` ended with `SIGABRT` during teardown

Final runtime metrics:

- train runtime: `592.9s`
- train steps per second: `0.27`
- trainer-reported iteration speed: about `3.705s/it`
- reported memory: `11.04 GiB`
- last checkpoint: `outputs/countdown_colocate_2gpu_1p7b_1k/v0-20260718-133335/checkpoint-160`

Interpretation:

- this is the strongest validation so far that `2GPU colocate` is the practical multi-GPU route on this machine for `Qwen/Qwen3-1.7B`
- compared with the 1GPU `1.7B + 1024 sample` run, the 2GPU run remained memory-stable and improved per-iteration speed materially
- the remaining defect is concentrated in shutdown after successful training, not in rollout, reward computation, checkpointing, or optimization

## Important Files

- `README.md`
- `scripts/21_prepare_countdown_sample.py`
- `scripts/22_run_countdown_smoke_1gpu.sh`
- `scripts/30_start_rollout_server.sh`
- `scripts/32_run_countdown_server_mode.sh`
- `scripts/50_summarize_json_metrics.py`
- `plugins/countdown_plugin.py`
- `pythonpath_patch/sitecustomize.py`
