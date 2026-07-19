# ms-swift 框架实验报告

日期：2026年7月18日，星期六

## 一、实验背景

本实验的目标是基于 `ms-swift` 框架完成一个可运行的文本强化学习实验，重点理解以下问题：

1. `SWIFT` 如何组织文本 RL 训练流程
2. `GRPO` 在 `SWIFT` 中是如何运行的
3. `rollout` 与 `trainer` 的关系是什么
4. 单卡与多卡、`server-mode` 与 `colocate` 两种架构的差异是什么
5. 在当前机器环境下，哪种配置是实际可用的

本次实验不以最终任务精度为核心目标，而以“跑通训练链路、验证多种架构、明确可行配置”为核心目标。

## 二、实验环境

硬件环境：

- `2 x RTX 3090 24GB`

软件环境：

- `ms-swift 4.4.1`
- `torch 2.8.0+cu128`
- `vllm 0.10.2`
- Python `3.12`

模型来源：

- 使用 `ModelScope`

## 三、实验任务设计

### 1. 主任务

本次 RL 任务采用可验证奖励（verifiable reward）路线，使用 Countdown 类型数据构造数学表达式生成任务。

核心数据集：

- `zouxuhong/Countdown-Tasks-3to4`

本地样本集：

- `countdown_sample_256.jsonl`
- `countdown_sample_1024.jsonl`

### 2. 奖励函数

自定义奖励插件：

- `plugins/countdown_plugin.py`

奖励包含两类：

- `countdown_correct`
  - 模型生成的最终表达式必须使用指定数字，并且计算结果等于目标值
- `countdown_format`
  - 模型输出必须符合可解析格式

这对应了 `GRPO` 里常见的两类奖励信号：

- 正确性奖励
- 格式奖励

### 3. 算法

使用：

- `GRPO`

在实验中，`GRPO` 的直观含义是：

- 每个 prompt 采样多个回答
- 对这些回答按奖励做组内比较
- 根据相对优劣更新策略
- 不依赖单独的 value model

## 四、实验配置路线

本次实验主要覆盖了四条路线：

1. `0.6B + 2GPU server-mode`
2. `1.7B + 1GPU colocate`
3. `1.7B + 2GPU colocate`
4. `1.7B/4B + 2GPU server-mode`

其中：

- `server-mode`：训练器与 rollout 服务分离
- `colocate`：训练器与 rollout 在同一作业中协同运行

## 五、实验结果

### 1. 0.6B + 2GPU server-mode

模型：

- `Qwen/Qwen3-0.6B`

输出目录：

- [countdown_server_mode_2gpu/v0-20260718-015658](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/countdown_server_mode_2gpu/v0-20260718-015658)

结果：

- 成功
- 训练完整跑通
- 成功写出大量 checkpoint，最终保存到 `checkpoint-1024`
- trainer 与 rollout server 的分离架构在 `0.6B` 下是可用的

结论：

- `0.6B` 可以成功跑通 `2GPU server-mode`
- 这证明 `SWIFT + vLLM + GRPO + rollout server` 的基本架构是成立的

### 2. 1.7B + 1GPU colocate smoke

模型：

- `Qwen/Qwen3-1.7B`

输出目录：

- [countdown_colocate_1p7b_smoke/v0-20260718-123621](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/countdown_colocate_1p7b_smoke/v0-20260718-123621)

结果：

- 成功
- 完成 `60/60`
- 成功写出 checkpoint
- 出现了 `CountdownCorrect = 1.0` 的正确奖励

结论：

- `1.7B` 在单卡 `colocate` 下可以稳定运行

### 3. 1.7B + 1GPU colocate + 1024 sample

输出目录：

- [countdown_colocate_1p7b_1k/v0-20260718-124009](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/countdown_colocate_1p7b_1k/v0-20260718-124009)

结果：

- 成功
- 完成 `160/160`
- 成功保存 `checkpoint-40`、`checkpoint-120`、`checkpoint-160`
- 日志中出现多次非零 reward，且存在 `CountdownCorrect = 1.0`

结论：

- 单卡 `1.7B` 不仅能 smoke run，也能在更大样本集上完成主实验

### 4. 1.7B + 2GPU colocate smoke

输出目录：

- [countdown_colocate_2gpu_1p7b_smoke/v0-20260718-125704](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/countdown_colocate_2gpu_1p7b_smoke/v0-20260718-125704)

结果：

- 训练主体成功
- 完成 `60/60`
- 成功写出 `checkpoint-20`、`checkpoint-40`、`checkpoint-60`
- 出现 `CountdownCorrect = 1.0`
- 训练结束后退出阶段出现异常

说明：

- 异常发生在训练完成之后
- checkpoint 与最终日志已经落盘

结论：

- `1.7B 2GPU colocate` 的训练链路是成功的
- 问题集中在收尾退出阶段，不在训练主体

### 5. 1.7B + 2GPU colocate + 1024 sample

输出目录：

- [countdown_colocate_2gpu_1p7b_1k/v0-20260718-133335](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/countdown_colocate_2gpu_1p7b_1k/v0-20260718-133335)

结果：

- 成功完成 `160/160`
- 成功写出 `checkpoint-40`、`checkpoint-80`、`checkpoint-120`、`checkpoint-160`
- 最终指标写出成功
- 日志中多次出现非零 reward
- 出现了 `CountdownCorrect = 1.0`
- 训练结束后收尾阶段依然存在退出异常

关键指标：

- `train_runtime = 592.9s`
- `train_steps_per_second = 0.27`
- 日志内显存约 `11.04 GiB`

结论：

- 这是目前最强的一组实验证据
- 说明 `1.7B 2GPU colocate` 在当前机器上是可用的主实验方案

### 6. 0.6B + 2GPU colocate probe

输出目录：

- [tmp_colocate_2gpu_0p6b_probe/v0-20260718-140359](/root/autodl-tmp/ai-notes-and-experiments/swift/outputs/tmp_colocate_2gpu_0p6b_probe/v0-20260718-140359)

结果：

- 成功完成 `5/5`
- 成功写出 `checkpoint-5`
- 日志正常
- 训练结束后依然出现退出异常

结论：

- `0.6B 2GPU colocate` 同样可以成功训练
- 多卡 `colocate` 的核心问题依然是退出阶段，而不是训练过程

### 7. 1.7B / 4B + 2GPU server-mode

结果：

- 失败

失败位置：

- 并不是训练跑到中间崩溃
- 而是在 rollout communicator / NCCL 初始化阶段就失败

典型判断：

- `1.7B` 与 `4B` 的问题出现在 `server-mode` 通信路径
- 不是数据集、奖励函数或 `GRPO` 逻辑错误

结论：

- `1.7B` 和 `4B` 在当前环境下不适合走 `2GPU server-mode`

## 六、总体结论

### 1. SWIFT/GRPO 是否成功

结论：

- 成功

更准确地说：

- `SWIFT + GRPO + vLLM` 文本强化学习训练链路已经跑通
- 单卡与多卡 `colocate` 都已经成功验证

### 2. 哪些实验是成功的

成功的实验包括：

- `0.6B 2GPU server-mode`
- `0.6B 2GPU colocate`
- `1.7B 1GPU colocate`
- `1.7B 1GPU colocate + 1024 sample`
- `1.7B 2GPU colocate`
- `1.7B 2GPU colocate + 1024 sample`

### 3. 哪些实验没有成功

未成功的实验包括：

- `1.7B 2GPU server-mode`
- `4B 2GPU server-mode`

### 4. 当前最可行的多卡方案

在这台机器上，当前最实际可用的多卡方案是：

- `2GPU colocate`

而不是：

- `2GPU server-mode`

## 七、问题分析

### 1. 为什么 `0.6B server-mode` 可以而 `1.7B server-mode` 不行

最合理的解释是：

- `1.7B/4B` 对显存和跨进程 NCCL 通信的压力更大
- `server-mode` 的 rollout server 与 trainer 分离架构对通信链路更敏感
- 因此 `0.6B` 能跑通，但 `1.7B` 开始暴露兼容性问题

### 2. 为什么 `colocate` 能训练成功但退出异常

当前现象表明：

- 训练主体是成功的
- 异常发生在训练完成之后
- 错误形态可能是 `SIGABRT` 或 `SIGSEGV`

这说明问题更可能来自：

- `torch distributed`
- `NCCL`
- `vLLM colocate`
- `ms-swift`

这几者在退出阶段的资源回收顺序或兼容性问题，而不是 RL 训练逻辑本身。

### 3. 这个问题是否影响实验成功

影响不大。

因为以下关键结果都已经成立：

- step 跑完
- checkpoint 落盘
- logging.jsonl 落盘
- reward 正常记录

所以从实验验证角度，这些实验仍然应判定为成功。

## 八、最终建议

### 1. 作为学习和实验路线

推荐优先使用：

- `1.7B + 1GPU colocate`
- `1.7B + 2GPU colocate`

这两条路线最适合继续理解：

- `rollout`
- `GRPO`
- 奖励函数
- 多卡资源使用方式

### 2. 作为框架理解结论

可以明确得到：

- `SWIFT` 能跑通文本 RL demo
- `GRPO` 能在可验证奖励任务上实际运行
- `vLLM` 在 RL rollout 中是可用推理后端

### 3. 作为工程结论

在当前机器与当前版本组合下：

- 推荐方案：`colocate`
- 不推荐方案：`1.7B+ server-mode`

### 4. 如果后续要继续优化

后续方向可以分成两类：

1. 训练侧继续扩展
   - 增大样本量
   - 提高模型规模
   - 调整 reward 权重
2. 工程侧继续排障
   - 研究 `colocate` 退出异常
   - 尝试版本矩阵排查 `torch / vllm / ms-swift` 的兼容性

## 九、报告结论摘要

一句话总结：

> 本次 `ms-swift` 实验已经成功跑通 `SWIFT + GRPO + vLLM` 的文本强化学习流程，当前机器上最可行的多卡路线是 `2GPU colocate`，而 `1.7B/4B` 的 `2GPU server-mode` 目前不可用。
