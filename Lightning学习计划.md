# Lightning 学习计划：从单卡训练到多卡分布式训练

本文面向 Lightning 初学者，目标是学会：

1. 理解 Lightning 的核心抽象：`LightningModule`、`Trainer`、`LightningDataModule`
2. 能把普通 PyTorch 训练代码改写成 Lightning 风格
3. 能用 Lightning 在单卡 GPU 上稳定训练模型
4. 能用 Lightning 进行多卡训练，重点掌握 DDP
5. 能处理 checkpoint、日志、混合精度、梯度累积等常见训练需求

## 先建立一个正确认知

Lightning 的核心价值不是“替你写模型”，而是把原本分散在训练脚本里的工程逻辑组织起来。官方文档对 `Trainer` 的描述很准确：当你把 PyTorch 代码组织进 `LightningModule` 后，`Trainer` 会自动处理训练循环、设备放置、回调、验证/测试等通用工作。

对初学者来说，最重要的学习顺序不是先研究各种高级策略，而是：

1. 先会写 `LightningModule`
2. 再会用 `Trainer.fit()`
3. 再整理数据到 `LightningDataModule`
4. 再把单卡脚本切到多卡
5. 最后再学性能优化和更高级的分布式策略

## 完整学习链条

### 阶段 0：补齐前置知识

如果你对下面内容还不熟，需要先补：

1. PyTorch `Dataset` / `DataLoader`
2. `nn.Module`
3. forward、loss、optimizer、scheduler 的基本关系
4. GPU 训练基础
5. 基本命令行运行方式

建议标准：

- 你能独立写一个普通 PyTorch 的 MNIST/CIFAR-10 训练脚本
- 你知道 `model.train()`、`model.eval()`、`optimizer.step()`、`loss.backward()` 在做什么

如果 PyTorch 基础还不稳，先看官方 Lightning 教程列表中的 PyTorch 入门教程，再进入 Lightning 本体。

## 阶段 1：理解 Lightning 的最小闭环

目标：先把 Lightning 的最小训练流程跑通。

你需要掌握：

1. `LightningModule` 负责什么
2. `training_step()`、`validation_step()`、`test_step()` 的职责
3. `configure_optimizers()` 怎么返回优化器和调度器
4. `Trainer.fit()`、`Trainer.validate()`、`Trainer.test()` 的调用方式
5. `self.log()` 的基本用法

建议实践：

1. 用 MNIST 或 CIFAR-10 写一个最小分类模型
2. 不要一开始就上复杂项目
3. 只保留最核心的训练/验证逻辑

这一阶段的输出标准：

- 你能独立写出一个 `LightningModule`
- 你能运行单卡训练
- 你能看到 loss 和 val metric 的变化

### 阶段 1 推荐教程

1. 官方快速入门：Lightning in 15 minutes  
   https://lightning.ai/docs/pytorch/stable/starter/introduction.html
2. 官方 Trainer 文档  
   https://lightning.ai/docs/pytorch/stable/common/trainer.html
3. 官方 LightningModule 说明  
   https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

## 阶段 2：把“能跑”变成“结构清晰”

目标：学习把数据逻辑和训练逻辑拆开，形成标准项目结构。

你需要掌握：

1. `LightningDataModule` 的意义
2. `prepare_data()`、`setup()`、`train_dataloader()`、`val_dataloader()`、`test_dataloader()`
3. 训练、验证、测试的职责边界
4. checkpoint 的保存与恢复
5. 日志系统与 callback 的基本配置

建议实践：

1. 把阶段 1 的数据部分重构成 `LightningDataModule`
2. 增加 `ModelCheckpoint`
3. 增加 `EarlyStopping`
4. 学会从 checkpoint 恢复训练

这一阶段的输出标准：

- 你能把数据处理从模型中解耦
- 你能保存最佳模型
- 你能中断后继续训练

### 阶段 2 推荐教程

1. 官方 DataModule 文档  
   https://lightning.ai/docs/pytorch/stable/data/datamodule.html
2. 官方 checkpoint 文档  
   https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
3. 官方验证与测试文档  
   https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html

## 阶段 3：形成 Lightning 工程化训练习惯

目标：学会日常训练脚本常用配置。

你需要掌握：

1. `accelerator`、`devices` 的设置方式
2. 混合精度训练，如 `precision="16-mixed"` 或 bf16
3. 梯度累积 `accumulate_grad_batches`
4. 梯度裁剪 `gradient_clip_val`
5. 日志器，如 TensorBoard / CSV / WandB
6. 随机种子、可复现性、训练配置管理

建议实践：

1. 在单卡上试一次 fp16/bf16 训练
2. 用梯度累积模拟更大的 batch size
3. 对比纯 fp32 与混合精度的速度/显存差异

这一阶段的输出标准：

- 你能说明为什么训练脚本需要这些参数
- 你能调通常见训练配置，而不是只会跑默认值

### 阶段 3 推荐教程

1. 官方训练技巧文档  
   https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
2. 官方加速训练文档  
   https://lightning.ai/docs/pytorch/stable/advanced/speed.html

## 阶段 4：进入多卡训练

目标：掌握 Lightning 下最常见、最重要的多卡训练方式：DDP。

### 你要先理解的概念

多卡训练不要一上来就背所有策略。先抓住最关键的区分：

1. 数据并行：每张卡放一份模型，输入不同 batch，最后同步梯度
2. 模型并行：模型被拆到多张卡上，通常用于超大模型
3. 对大多数常规 CV/NLP 任务，优先学 DDP，不要先学 FSDP/DeepSpeed

Lightning 官方文档也明确给出倾向：如果是一般的多 GPU 训练，优先使用常规 DDP；模型特别大时，再考虑 FSDP、DeepSpeed 等模型并行/分片方案。

### 这一阶段必须掌握的内容

1. `Trainer(accelerator="gpu", devices=2, strategy="ddp")`
2. 不显式写 `.cuda()`、`.to(device)` 的 Lightning 思维
3. DDP 下每个进程各持有一个 device
4. 日志、随机种子、sampler、checkpoint 在多卡下的常见行为
5. Notebook 与命令行启动方式的差异

### 一个最小多卡例子

```python
from lightning import Trainer

trainer = Trainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    precision="16-mixed",
    max_epochs=10,
)

trainer.fit(model, datamodule=datamodule)
```

如果你在脚本中运行，推荐使用命令行方式启动训练。官方文档也建议将训练主逻辑放进 `main()`。

### Notebook 和脚本的差异

这是初学者最容易踩坑的地方：

1. 如果你在普通 Python 脚本里训练，多卡优先用 `ddp`
2. 如果你在 Jupyter/Colab/Kaggle 里训练，多卡通常要用 `ddp_notebook`
3. 不要把 Notebook 里的启动方式和终端脚本混用

### 阶段 4 推荐教程

1. 官方 GPU training（Intermediate）  
   https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
2. 官方 GPU FAQ  
   https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html
3. 官方 notebooks 使用说明  
   https://lightning.ai/docs/pytorch/stable/common/notebooks.html
4. 官方 Strategy 概念文档  
   https://lightning.ai/docs/pytorch/stable/extensions/strategy.html

## 阶段 5：从“会用多卡”到“能稳定用多卡”

目标：学会处理多卡训练的实际问题。

你需要掌握：

1. 有效 batch size 的计算  
   `global_batch_size = per_device_batch_size x devices x accumulate_grad_batches`
2. DDP 下梯度同步会带来的通信开销
3. 多卡不一定一定更快，小模型可能反而受通信影响
4. DataLoader 的 `num_workers`、pin memory、数据瓶颈
5. 多卡下 metric 聚合和日志行为
6. 恢复训练、保存最优 checkpoint、测试最佳权重

建议实践：

1. 单卡和双卡各跑一次同一个实验
2. 记录每 epoch 时间、显存、吞吐
3. 对比不同 batch size 的收敛表现
4. 练习从 `ckpt_path="best"` 测试最佳模型

这一阶段的输出标准：

- 你能解释为什么有时双卡不比单卡快
- 你能独立排查常见多卡训练问题

## 阶段 6：按需进入高级分布式

当你已经熟练使用 DDP 后，再决定是否学习：

1. FSDP：大模型参数分片
2. DeepSpeed：更大规模模型训练优化
3. Tensor Parallelism：超大模型层级切分

如果你目前只是学习 Lightning 架构和多卡训练，先把 DDP 用熟，比过早进入 FSDP/DeepSpeed 更重要。

### 阶段 6 推荐文档

1. 官方 FSDP 文档  
   https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html
2. 官方 DeepSpeed 文档  
   https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html
3. 官方模型并行总览  
   https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html

## 一个建议的 4 周学习计划

### 第 1 周：Lightning 基础

目标：

1. 理解 `LightningModule`
2. 跑通单卡训练
3. 熟悉 `Trainer.fit()/validate()/test()`

任务：

1. 阅读 Lightning in 15 minutes
2. 用 MNIST 写一个最小分类例子
3. 加入验证集和 `self.log()`
4. 阅读 Trainer 文档中的基础用法

交付标准：

- 你能从零写出一个最小 Lightning 训练脚本

### 第 2 周：工程化与可恢复训练

目标：

1. 学会 `LightningDataModule`
2. 学会 checkpoint、callback、logger

任务：

1. 把第 1 周代码重构成 Module + DataModule
2. 加 `ModelCheckpoint`
3. 加 `EarlyStopping`
4. 练习断点恢复训练

交付标准：

- 你能清楚划分模型逻辑、数据逻辑和训练控制逻辑

### 第 3 周：单卡优化与训练技巧

目标：

1. 掌握混合精度
2. 掌握梯度累积和梯度裁剪
3. 建立训练配置意识

任务：

1. 加入 `precision`
2. 试验 `accumulate_grad_batches`
3. 对比不同配置的显存和训练速度
4. 学会 `ckpt_path="best"` 做测试

交付标准：

- 你能独立配置一份比较像样的训练脚本

### 第 4 周：多卡训练实战

目标：

1. 掌握 DDP 多卡训练
2. 学会脚本与 Notebook 的区别
3. 能解释多卡训练的常见问题

任务：

1. 在 2 卡环境跑通 `strategy="ddp"`
2. 对比 1 卡和 2 卡的速度
3. 计算 global batch size
4. 记录多卡训练中的日志和 checkpoint 行为

交付标准：

- 你能独立把一个 Lightning 单卡项目切换到多卡

## 推荐的学习材料清单

### 必看主线

1. Lightning in 15 minutes  
   https://lightning.ai/docs/pytorch/stable/starter/introduction.html
2. Trainer 文档  
   https://lightning.ai/docs/pytorch/stable/common/trainer.html
3. LightningDataModule 文档  
   https://lightning.ai/docs/pytorch/stable/data/datamodule.html
4. GPU training（Intermediate）  
   https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html

### 训练必备

1. Checkpoint 文档  
   https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
2. Validation/Test 文档  
   https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
3. Training tricks 文档  
   https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
4. Speed up training 文档  
   https://lightning.ai/docs/pytorch/stable/advanced/speed.html

### 教程集合

1. 官方教程总览  
   https://lightning.ai/docs/pytorch/stable/tutorials.html
2. 官方 notebooks 列表  
   https://lightning.ai/docs/pytorch/stable/notebooks.html
3. 官方教程仓库  
   https://github.com/Lightning-AI/tutorials

## 学习时最容易踩的坑

1. 一开始就学太多策略  
   对初学者，多卡先学 `ddp`，不要同时钻进 `fsdp`、`deepspeed`、`spawn`、`notebook` 全套。
2. 把 Lightning 当成“自动建模工具”  
   Lightning 不替你设计模型，它主要整理训练工程结构。
3. 在多卡下手动到处写 `.cuda()`  
   在 Lightning 里通常不需要自己手动搬运模型和 batch 到设备。
4. 在 Notebook 里照搬脚本的 DDP 用法  
   交互式环境和脚本环境的策略选择不同。
5. 只看代码不做实验  
   多卡训练一定要亲自比较 1 卡、2 卡、混合精度、梯度累积的实际效果。

## 推荐你的第一个完整练习项目

建议用 CIFAR-10 做一次完整项目，按下面顺序推进：

1. 先写普通 PyTorch 版本
2. 再改成 `LightningModule`
3. 再抽成 `LightningDataModule`
4. 加 checkpoint 和 early stopping
5. 单卡跑通
6. 切到双卡 DDP
7. 加混合精度
8. 比较单卡与双卡性能

如果你能把这个项目完整做完，Lightning 的核心使用方式基本就建立起来了。

## 你现在最适合的学习策略

如果你是“Lightning 初学者”，最好的路线不是到处找零散视频，而是：

1. 以官方文档主线学习
2. 用一个小项目贯穿所有阶段
3. 每学一个概念就立刻做实验
4. 先单卡、后多卡
5. 先 DDP、后高级策略

这样学习成本最低，也最容易真正掌握。

## 参考来源

本文的学习路径主要基于 Lightning 官方文档当前稳定版页面整理，尤其参考了以下入口：

1. Lightning 文档首页  
   https://lightning.ai/docs/pytorch/stable/index.html
2. Lightning in 15 minutes  
   https://lightning.ai/docs/pytorch/stable/starter/introduction.html
3. Trainer 文档  
   https://lightning.ai/docs/pytorch/stable/common/trainer.html
4. DataModule 文档  
   https://lightning.ai/docs/pytorch/stable/data/datamodule.html
5. GPU training（Intermediate）  
   https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
6. GPU FAQ  
   https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html
