# Hugging Face vs vLLM: First Benchmark

This experiment compares `transformers` inference against `vLLM` inference on the same model and prompt set.

## Goal

Use `Qwen/Qwen3-4B` as the target model and answer a narrow first question:

`vLLM` is faster than plain Hugging Face `generate()` by how much under low concurrency and light batching?

This first experiment focuses on:

- single GPU
- fixed prompt set
- fixed generation length
- greedy decoding
- throughput and latency only

It does not yet cover:

- high concurrency
- long-context stress
- multi-turn chat
- serving mode

## Files

- `prompts.jsonl`: benchmark prompt set
- `benchmark_hf.py`: `transformers` baseline
- `benchmark_vllm.py`: `vLLM` baseline
- `compare_results.py`: summary utility

## Environment

Recommended:

```bash
pip install "torch>=2.8" "transformers>=4.53" accelerate
pip install "vllm>=0.10"
```

`Qwen/Qwen3-3B` is not currently a public Hugging Face model id. For this reason, the first experiment uses `Qwen/Qwen3-4B`. If that is unavailable in your environment, replace it with another small Qwen model such as `Qwen/Qwen3-1.7B` or `Qwen/Qwen2.5-3B-Instruct`, but keep the model identical across both runs.

## Experiment Design

### Controlled variables

- same model id
- same prompts
- same `max_new_tokens`
- same `temperature=0.0`
- same device: one GPU

### Main metrics

- total wall time
- average latency per prompt
- output tokens per second
- end-to-end tokens per second

### Why this is a good first experiment

This isolates the runtime stack difference:

- Hugging Face path: `AutoModelForCausalLM.generate()`
- vLLM path: `LLM.generate()`

At this stage, any observed speedup is mainly from scheduler, KV-cache handling, and kernel/runtime differences rather than from prompt engineering or decoding randomness.

## Run

### 1. Hugging Face baseline

```bash
python benchmark_hf.py \
  --model Qwen/Qwen3-4B \
  --prompts prompts.jsonl \
  --max-new-tokens 128 \
  --batch-size 4 \
  --output results_hf.json
```

### 2. vLLM baseline

```bash
python benchmark_vllm.py \
  --model Qwen/Qwen3-4B \
  --prompts prompts.jsonl \
  --max-new-tokens 128 \
  --batch-size 4 \
  --output results_vllm.json
```

### 3. Compare

```bash
python compare_results.py results_hf.json results_vllm.json
```

## Suggested first matrix

Run these four settings first:

1. `batch_size=1`, `max_new_tokens=64`
2. `batch_size=1`, `max_new_tokens=128`
3. `batch_size=4`, `max_new_tokens=64`
4. `batch_size=4`, `max_new_tokens=128`

Keep everything else fixed.

## Notes

- Plain Hugging Face here means local eager-style generation, not TGI.
- For a fair comparison, do one warmup pass before timed runs.
- Decode and file-write time are excluded from the main measured region as much as practical.
