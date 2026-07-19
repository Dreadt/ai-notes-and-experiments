# HF vs vLLM 推理对比实验报告

日期：2026年7月18日，星期六

## 一、任务背景

本实验的目标是比较同一模型在两种本地推理后端下的性能差异：

- `Hugging Face transformers.generate()`
- `vLLM LLM.generate()`

核心问题是：

> 在模型、提示词、生成长度都保持一致的前提下，`vLLM` 相比普通 `Hugging Face` 推理到底快多少，原因是什么？

本次实验关注的是推理效率，不讨论回答质量、线上服务稳定性或多轮对话能力。

## 二、实验目标

在严格控制变量的前提下，对比两种推理后端的：

- 吞吐率
- 单条样本平均延迟
- 端到端 token 生成效率

## 三、实验配置

### 1. 模型

主实验模型：

- `Qwen/Qwen3-4B`

补充实验模型：

- `Qwen/Qwen3-0.6B`

### 2. 提示词数据

- 文件：`prompts.jsonl`
- 样本数：`8`

### 3. 控制变量

两组实验中保持一致的内容包括：

- 相同模型
- 相同提示词集合
- 相同 `batch_size`
- 相同 `max_new_tokens`
- 相同确定性生成设置

### 4. 对比后端

- `huggingface_transformers`
- `vllm`

### 5. 主要指标

- `avg_latency_per_prompt_sec`
- `output_tokens_per_sec`
- `end_to_end_tokens_per_sec`

## 四、实验矩阵

### Qwen3-4B 主实验

共测量四组配置：

1. `batch_size=1`, `max_new_tokens=64`
2. `batch_size=1`, `max_new_tokens=128`
3. `batch_size=4`, `max_new_tokens=64`
4. `batch_size=4`, `max_new_tokens=128`

### Qwen3-0.6B 补充实验

共测量一组配置：

1. `batch_size=1`, `max_new_tokens=64`

## 五、实验结果对比

### 1. Qwen3-4B 对比结果

| 配置 | HF 输出 tok/s | vLLM 输出 tok/s | vLLM 加速比 | HF 平均延迟 | vLLM 平均延迟 | 延迟改善倍数 |
|---|---:|---:|---:|---:|---:|---:|
| `bs=1, t=64` | 33.82 | 80.76 | 2.39x | 1.8925s | 0.7925s | 2.39x |
| `bs=1, t=128` | 33.92 | 80.62 | 2.38x | 3.7740s | 1.5876s | 2.38x |
| `bs=4, t=64` | 127.36 | 307.61 | 2.42x | 0.5290s | 0.2081s | 2.54x |
| `bs=4, t=128` | 124.01 | 315.48 | 2.54x | 1.0594s | 0.4057s | 2.61x |

### 2. Qwen3-0.6B 补充结果

| 配置 | HF 输出 tok/s | vLLM 输出 tok/s | vLLM 加速比 | HF 平均延迟 | vLLM 平均延迟 | 延迟改善倍数 |
|---|---:|---:|---:|---:|---:|---:|
| `bs=1, t=64` | 45.87 | 236.71 | 5.16x | 1.3951s | 0.2704s | 5.16x |

## 六、结果分析

### 1. vLLM 在所有测试中都快于 Hugging Face

在 `Qwen3-4B` 的四组实验中：

- 吞吐率提升大致稳定在 `2.38x` 到 `2.54x`
- 平均延迟改善大致稳定在 `2.38x` 到 `2.61x`

说明 `vLLM` 的优势不是偶然现象，而是在不同生成长度、不同 batch 条件下都比较稳定。

### 2. batch 增大后，vLLM 的优势更明显

对于 `Qwen3-4B`：

- `bs=1` 时，vLLM 大约快 `2.38x` 到 `2.39x`
- `bs=4` 时，vLLM 提升到 `2.42x` 到 `2.54x`

这说明当请求可以组成 batch 时，vLLM 能更充分地利用 GPU。

### 3. 小模型上的相对增益更大

在 `Qwen3-0.6B bs=1 t=64` 的补充实验中：

- HF：`45.87 tok/s`
- vLLM：`236.71 tok/s`
- 加速比：`5.16x`

这表明在小模型场景下，框架开销、调度方式和缓存管理对整体速度的影响更大，而 vLLM 在这些环节优化得更充分。

## 七、为什么 vLLM 更快

### 1. 调度机制更适合大模型推理

`Hugging Face generate()` 更偏向通用推理接口，使用方便，但不是专门为高吞吐推理优化的调度器。

`vLLM` 则是围绕 LLM 推理场景设计的，能够更高效地组织 prefill 和 decode 阶段。

### 2. KV Cache 管理更高效

自回归生成过程中，KV Cache 的分配和复用直接影响推理速度。

`vLLM` 在 KV Cache 管理上更激进、更高效，因此可以减少显存碎片和不必要的数据搬运。

### 3. batching 利用率更高

从实验结果可以看到，随着 `batch_size` 增大：

- HF 的吞吐率会提高
- vLLM 的吞吐率提高得更多

这说明 vLLM 更擅长把一批请求转化成高 GPU 利用率的推理过程。

### 4. vLLM 是专门面向推理优化的运行时

`transformers` 是通用框架，既支持训练，也支持微调，也支持推理。

`vLLM` 的核心目标就是提升大模型推理效率，因此它在运行时层面的优化更集中、更有针对性。

## 八、实验结论

本实验得到的结论可以概括为：

1. `vLLM` 在相同模型、相同提示词和相同生成长度下，明显快于普通 `Hugging Face` 推理。
2. 在 `Qwen3-4B` 上，vLLM 相比 HF 的速度提升大约为 `2.4x` 到 `2.5x`。
3. 在 `Qwen3-0.6B` 上，观测到的速度提升达到 `5.16x`。
4. 当 batch 增大时，vLLM 的优势进一步放大。

因此，从当前实验结果看：

> 如果目标是做高效推理，尤其是后续要支撑 RL rollout 或批量生成任务，`vLLM` 比普通 `Hugging Face` 更适合作为推理后端。

## 九、局限性

本实验尚未覆盖：

- 高并发在线服务
- 超长上下文场景
- 更大的 batch
- streaming 输出
- 多 GPU 推理服务
- 随机采样下的质量差异

因此本次结论应理解为：

> 在当前这组受控的本地基准实验中，`vLLM` 的推理性能明显优于普通 `Hugging Face`。

## 十、结果文件

- `results_hf_qwen3_0p6b_bs1_t64.json`
- `results_vllm_qwen3_0p6b_bs1_t64.json`
- `results_hf_qwen3_4b_bs1_t64.json`
- `results_vllm_qwen3_4b_bs1_t64.json`
- `results_hf_qwen3_4b_bs1_t128.json`
- `results_vllm_qwen3_4b_bs1_t128.json`
- `results_hf_qwen3_4b_bs4_t64.json`
- `results_vllm_qwen3_4b_bs4_t64.json`
- `results_hf_qwen3_4b_bs4_t128.json`
- `results_vllm_qwen3_4b_bs4_t128.json`
