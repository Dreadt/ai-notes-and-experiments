# AutoDL 多卡 SFT 学习报告

## 1. 任务背景

本次学习任务的目标是：

- 在 AutoDL 环境下使用 `Hugging Face Accelerate` 做多卡训练
- 以 `Qwen3-0.6B` 为底座模型进行 SFT
- 使用 `wandb` 记录训练曲线与实验过程
- 理解 Hugging Face 模型目录格式、SFT 数据格式、loss 计算方式，以及多卡训练的工程基础

本次实验目录：

- [/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb)


## 2. Accelerator 的作用

`accelerate` 的核心作用，是把原本复杂的分布式训练流程统一封装起来，让我们可以用接近单卡 PyTorch 的写法完成多卡训练。

它主要解决了这几个工程问题：

### 2.1 统一多卡启动方式

传统多卡训练通常需要手动处理：

- `torch.distributed`
- 进程数与 rank
- DDP 初始化
- 不同进程上的模型、数据、日志同步

而 `accelerate` 允许我们通过：

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

直接启动多卡训练。

### 2.2 自动封装模型、优化器、数据加载器

在 `train.py` 中，我们使用：

```python
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

这一步会把：

- 模型放到正确设备
- 多进程通信包装好
- DataLoader 自动按进程切分
- 保证每张卡拿到正确的数据分片

### 2.3 统一梯度累积与反向传播

使用：

```python
with accelerator.accumulate(model):
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
```

它处理了：

- 多卡同步梯度
- 梯度累积
- 不同精度训练下的 backward 兼容

### 2.4 控制主进程行为

多卡训练中，不能让每个进程都重复：

- 保存模型
- 打印日志
- 写 `wandb`

因此我们使用：

```python
if accelerator.is_main_process:
    ...
```

只让主进程负责保存和日志输出。

### 2.5 让代码从单卡平滑迁移到多卡

这是 `accelerate` 最大的工程价值：  
我们先写一个最小单卡训练脚本，再通过少量改动切到多卡，而不需要完全重写训练框架。


## 3. 我们如何实现多卡训练

### 3.1 硬件环境

本次 AutoDL 环境中可见 GPU 为：

- `NVIDIA GeForce RTX 3090`
- `NVIDIA GeForce RTX 3090`

也就是 2 卡、每卡 24GB。

### 3.2 核心配置文件

多卡训练配置文件：

- [accelerate_config.yaml](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/accelerate_config.yaml)

其中关键项包括：

- `distributed_type: MULTI_GPU`
- `num_processes: 2`
- `mixed_precision: bf16`

### 3.3 训练脚本结构

训练脚本：

- [train.py](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/train.py)

它完成了这些事情：

1. 加载模型与 tokenizer
2. 加载 `alpaca-zh` 数据集
3. 将 `instruction/input/output` 转成 Qwen 对话模板
4. 构造 `assistant-only loss mask`
5. 用 `accelerator.prepare(...)` 包装训练对象
6. 在多卡上训练、评估、保存 checkpoint
7. 记录 `wandb`

### 3.4 数据并行的实现思路

本次实现的本质是 **数据并行**：

- 每张卡各自持有一份模型副本
- 每张卡处理不同 batch
- 反向传播后同步梯度
- 最终参数更新保持一致

这就是 DDP 范式，而 `accelerate` 帮我们封装了其中大部分细节。


## 4. 学习路径：我们是怎么一步步做起来的

这次不是一开始就直接全量双卡训练，而是按工程上更稳的路径推进。

### 4.1 第一步：搭最小项目骨架

我们先创建了实验工程，包含：

- [train.py](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/train.py)
- [requirements.txt](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/requirements.txt)
- [accelerate_config.yaml](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/accelerate_config.yaml)
- [README.md](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/README.md)

这一步的目标不是追求效果，而是先把工程结构固定下来。

### 4.2 第二步：补齐依赖并验证环境

开始时环境缺少：

- `transformers`
- `datasets`
- `accelerate`
- `wandb`

随后完成安装并验证 import。

### 4.3 第三步：验证 tokenizer 与 loss mask

我们先没有直接训练，而是先验证：

- Qwen tokenizer 能否正确加载
- `chat_template` 能否正常工作
- `labels=-100` 的 masking 是否只监督 assistant 回复

这一步非常关键，因为 SFT 中最容易出问题的就是：

- prompt 和 answer 切分错误
- 全句都参与 loss
- tokenizer 模板不匹配

### 4.4 第四步：单进程冒烟实验

我们先跑了一次极小规模训练：

- 输出目录：[outputs/smoke](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/smoke)

这一步验证了整条链：

- 数据集加载
- tokenization
- forward / backward
- HF 格式保存

### 4.5 第五步：双卡冒烟实验

然后我们切到：

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

并完成了双卡冒烟：

- 输出目录：[outputs/smoke-multigpu](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/smoke-multigpu)

这一步证明：

- 2 卡启动正常
- 多进程训练可用
- 主进程保存正常

### 4.6 第六步：验证 wandb 离线记录

由于 AutoDL 网络环境不稳定，我们优先支持 `wandb offline`。

我们完成了离线记录验证：

- 典型离线目录：
  - [wandb/offline-run-20260711_095442-kir5no1s](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/wandb/wandb/offline-run-20260711_095442-kir5no1s)
  - [wandb/offline-run-20260711_104248-9ypjhswf](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/wandb/offline-run-20260711_104248-9ypjhswf)

这说明即使训练时不联网，也可以：

- 本地落盘
- 训练结束后再 `wandb sync`


## 5. 我们做了哪些正式实验

### 5.1 Stage 1：小规模双卡正式实验

配置：

- 数据集：`shibing624/alpaca-zh`
- 训练样本：`2000`
- 验证样本：`200`
- 双卡
- `bf16`

输出：

- [outputs/exp-stage1](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage1)

意义：

- 从冒烟过渡到正式实验
- 验证更大数据下的稳定性

### 5.2 Stage 2：标准化中等规模实验

配置：

- 训练样本：`10000`
- 验证样本：`500`
- epoch：`2`
- 双卡
- `bf16`

输出：

- [outputs/exp-stage2](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage2)

结果：

- `eval/loss = 1.696`
- `eval/perplexity = 5.4521`

意义：

- 得到一组可用的基线结果
- 说明当前 SFT 链路是稳定可复用的

### 5.3 Stage 3：全量数据与学习率对比实验

我们设计了一个 batch 脚本：

- [run_stage3_matrix.sh](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/run_stage3_matrix.sh)

计划顺序运行三组：

1. 全量数据，`lr=2e-5`
2. `10k` 数据，`lr=1e-5`
3. `10k` 数据，`lr=3e-5`

目标是同时完成：

- 数据规模扩展实验
- 学习率对比实验


## 6. 我们遇到的问题与解决过程

### 6.1 Hugging Face 原站不可达

问题：

- `huggingface.co` 在当前环境中不可直接访问

解决：

- 自动切到 `HF_ENDPOINT=https://hf-mirror.com`

工程意义：

- 学会处理 AutoDL 上常见的镜像下载问题

### 6.2 accelerate 配置字段兼容性问题

问题：

- 旧配置里存在 `tee` 字段，当前版本 `accelerate` 不识别

解决：

- 删除不兼容配置项

工程意义：

- 多卡框架版本与配置文件格式必须匹配

### 6.3 多卡总步数显示不准确

问题：

- 早期版本训练脚本按 prepare 前 DataLoader 长度计算步数，导致双卡下进度条不准确

解决：

- 改为在 `accelerator.prepare(...)` 之后重新计算 `num_update_steps_per_epoch`

工程意义：

- 分布式训练中，很多“看起来没问题”的数值实际上会因为数据切分而变化

### 6.4 磁盘空间不足

问题：

- 全量实验运行到中途，在保存 checkpoint 时触发：
  - `No space left on device`

直接原因：

- `/root/autodl-tmp` 分区被中间 checkpoint 打满

处理方法：

1. 删除低价值 smoke 产物
2. 删除全量实验的旧 checkpoint，只保留关键续跑点
3. 降低 `save_steps`

工程意义：

- 在大模型训练里，磁盘空间管理和显存管理同样重要

### 6.5 断点续训不完整

问题：

- 早期 `checkpoint` 仅保存 HF 模型权重，没有保存优化器、调度器、RNG 状态
- 因此 `resume-from-checkpoint` 不能真正恢复完整训练状态

解决：

- 在新的 `save_checkpoint()` 中加入：

```python
accelerator.save_state(...)
```

工程意义：

- “能保存模型”不等于“能断点续训”
- 真正的训练恢复必须保存训练状态而不只是参数


## 7. 多卡训练的学习路径总结

如果把这次学习抽象成一条通用路径，可以总结为：

### 7.1 先学单卡最小闭环

先掌握：

- 模型加载
- tokenizer
- 数据集预处理
- loss 计算
- 模型保存

### 7.2 再引入 accelerate

重点理解：

- `accelerate config`
- `accelerate launch`
- `accelerator.prepare`
- `accelerator.backward`
- `accelerator.is_main_process`

### 7.3 然后学工程化问题

这次真正有价值的，不只是“跑起来”，而是理解这些问题：

- 数据镜像与下载
- 多卡步数与日志
- W&B 在线/离线
- checkpoint 设计
- 磁盘空间管理
- 断点恢复

### 7.4 最后才是调参实验

只有在链路稳定之后，才值得做：

- 数据规模对比
- 学习率 sweep
- epoch 对比
- loss / perplexity 曲线分析


## 8. 本次学习的收获

通过这次实验，我们不仅完成了“Qwen3-0.6B 的 AutoDL 多卡 SFT”，还建立了一个更完整的工程认知：

1. `accelerate` 的价值不在于“更快”，而在于把多卡工程复杂度压到可控范围内
2. 多卡训练的重点不是启动命令本身，而是数据切分、主进程控制、保存逻辑和恢复逻辑
3. `wandb` 在 AutoDL 上更适合优先按离线模式设计
4. 真正阻碍实验推进的，往往不是模型本身，而是工程边界：
   - 网络
   - 存储
   - 恢复机制
   - 版本兼容性


## 8.1 Qwen 模型目前放在哪里

这次实验里，Qwen 模型实际存在两类位置：

### 1. 底座模型缓存

下载下来的原始 `Qwen/Qwen3-0.6B` 底座模型缓存位置在：

- [/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B](/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B)

这个目录是 Hugging Face Hub 的本地缓存目录，里面通常包含：

- `blobs/`
- `snapshots/`
- `refs/`

这里保存的是“原始预训练模型”的本地缓存，不是我们训练后的结果目录。

### 2. 微调后的模型输出

目前已经成功保存下来的微调模型目录有：

- [outputs/exp-stage1](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage1)
- [outputs/exp-stage2](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage2)

这两个目录中都包含标准 Hugging Face 保存格式，例如：

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`

### 3. Stage 3 的中间续跑点

全量实验当前使用的中间模型续跑点在：

- [outputs/exp-stage3-full-lr2e5/checkpoint-1050](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage3-full-lr2e5/checkpoint-1050)

这个目录保存的是训练进行到 `step 1050` 时的模型权重，可作为继续 fine-tune 的起点。

### 4. 如何理解这些位置

可以把它们理解成：

- `~/.cache/huggingface/hub/...`
  - 原始 Qwen 底座模型缓存
- `outputs/exp-stage1`, `outputs/exp-stage2`
  - 已完成实验得到的微调模型
- `outputs/exp-stage3-full-lr2e5/checkpoint-1050`
  - 当前全量实验的中途续跑权重


## 9. 后续建议

后续建议按下面顺序继续学习：

1. 完成 Stage 3 三组实验并做对比表
2. 增加真正可恢复的断点续训实验
3. 给训练脚本加入 `save_total_limit` 或自动清理旧 checkpoint
4. 试 LoRA 版本，比较全参数 SFT 与参数高效微调
5. 补一份单独的：
   - Hugging Face 文件格式说明
   - SFT 数据格式与 loss 数学含义说明


## 10. 当前结论

截至本报告撰写时，我们已经完成：

- 单卡冒烟
- 双卡冒烟
- `wandb` 离线记录
- Stage 1 正式训练
- Stage 2 基线训练
- Stage 3 全量实验的中途续跑与工程修复

也就是说，这次学习已经从“能否跑起来”推进到了“如何把一个真实的多卡训练工程跑稳、跑通、跑可复现”。
