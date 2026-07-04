# 从 PyTorch 到 Lightning，再到 DDP：我的 MNIST 训练学习笔记

## 1. 学习背景

在学习深度学习训练框架时，我希望解决两个问题：

1. 如何从纯 PyTorch 训练代码过渡到 Lightning 架构
2. 如何在 Lightning 中进一步完成多卡 DDP 训练

为了把问题尽量收敛，我选择了 `MNIST` 作为最小实验对象，并按下面的顺序进行：

1. 先写一份纯 PyTorch 版训练脚本
2. 再改写成 Lightning 版训练脚本
3. 最后在 Lightning 基础上尝试单卡与双卡 DDP 的对照实验

这样做的好处是，学习路径非常清晰，不会一开始就陷入复杂工程细节。

## 2. 为什么我要学 Lightning

在纯 PyTorch 中，虽然训练逻辑足够灵活，但也会带来一个明显问题：训练工程代码容易分散。

一个最简单的训练脚本往往也要自己处理：

- 模型定义
- 数据加载
- 训练循环
- 验证循环
- 测试循环
- optimizer 配置
- checkpoint 保存
- device 放置

当模型稍微复杂、实验稍微变多时，这些逻辑会越来越难维护。

Lightning 的核心价值不是“替你写模型”，而是把训练工程中的共性逻辑抽象出来，让代码结构更稳定、更清晰。

## 3. 本次实验用到的代码

这次学习过程中，我整理了以下几份脚本：

- 纯 PyTorch 版：
  [train_mnist.py](/root/autodl-tmp/Lightning-train-learning/pytorch-MNIST/train_mnist.py)
- Lightning 版：
  [train_mnist_lightning.py](/root/autodl-tmp/Lightning-train-learning/Lightning-MNIST/train_mnist_lightning.py)
- Lightning 单卡对照版：
  [train_mnist_single_gpu.py](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/train_mnist_single_gpu.py)
- Lightning 双卡 DDP 版：
  [train_mnist_ddp.py](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/train_mnist_ddp.py)

另外，我还专门整理了一份固定 global batch size 的对照实验记录：

- [compare_global_batch_size.md](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/compare_global_batch_size.md)

## 4. 纯 PyTorch 版训练脚本拆解

在纯 PyTorch 版本中，一个标准训练脚本通常包含下面几个部分：

### 4.1 模型定义

我使用的是一个简单的多层全连接网络：

- 输入为 `28 x 28` 的灰度图
- 先展平
- 再通过两层隐藏层
- 最后输出 `10` 类分类结果

这样的结构足够简单，适合作为训练框架学习样例。

### 4.2 数据加载

数据使用 `torchvision.datasets.MNIST`，并使用：

```python
transforms.ToTensor()
transforms.Normalize((0.1307,), (0.3081,))
```

其中：

- `0.1307` 是 MNIST 训练集像素均值
- `0.3081` 是 MNIST 训练集像素标准差

标准化后的数据通常更有利于训练稳定。

### 4.3 训练循环

在纯 PyTorch 中，训练的核心步骤是：

1. `model.train()`
2. 前向计算
3. 计算 loss
4. `optimizer.zero_grad()`
5. `loss.backward()`
6. `optimizer.step()`

验证和测试则对应：

1. `model.eval()`
2. `torch.no_grad()`
3. 只做前向和指标统计

### 4.4 纯 PyTorch 的特点

优点：

- 非常直接
- 能清楚看见每一步训练逻辑

缺点：

- 训练、验证、测试、保存模型等逻辑都要自己维护
- 稍大一点的项目很容易变乱

## 5. Lightning 版训练脚本拆解

Lightning 版本最重要的变化，不是模型本身，而是代码的组织方式。

### 5.1 LightningModule

`LightningModule` 主要负责：

- 模型结构定义
- forward
- training_step
- validation_step
- test_step
- configure_optimizers

这相当于把纯 PyTorch 中分散的训练逻辑统一收进一个对象里。

### 5.2 LightningDataModule

`LightningDataModule` 主要负责：

- 数据准备
- 数据集切分
- train/val/test dataloader

这相当于把纯 PyTorch 中的数据部分独立出来。

### 5.3 Trainer

Lightning 最关键的控制器是 `Trainer`。

在纯 PyTorch 中，我需要自己写 epoch 循环；而在 Lightning 中，训练入口变成了：

```python
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule, ckpt_path="best")
```

这意味着训练循环、验证调度、设备管理、回调执行等大量共性逻辑都交给了 Lightning。

## 6. PyTorch 与 Lightning 的一一对照

这是我学习过程中最重要的一部分。

### 6.1 模型结构

PyTorch 和 Lightning 中，模型结构本身没有本质变化。

换句话说，Lightning 并不改变你“怎么设计模型”，它只是改变你“怎么组织训练代码”。

### 6.2 数据加载

PyTorch 中，我通过一个 `build_dataloaders()` 函数组织数据。

Lightning 中，这部分被迁移到 `LightningDataModule`：

- `prepare_data()`
- `setup()`
- `train_dataloader()`
- `val_dataloader()`
- `test_dataloader()`

### 6.3 训练逻辑

PyTorch 中，我自己写一个 `run_epoch()` 函数处理训练和验证。

Lightning 中，这部分被拆成：

- `training_step()`
- `validation_step()`
- `test_step()`

每个 step 只处理“一个 batch 的逻辑”，而 epoch 管理交给 `Trainer`。

### 6.4 优化器

PyTorch 中：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

Lightning 中：

```python
def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
```

Lightning 会自动在训练循环中调用优化器步骤。

### 6.5 checkpoint

PyTorch 中，我需要自己比较验证精度并手动保存：

```python
if val_acc > best_val_acc:
    torch.save(model.state_dict(), save_path)
```

Lightning 中，这部分交给 `ModelCheckpoint`：

```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
)
```

这让代码更整洁，也更适合实验管理。

## 7. 单卡训练实践

在单卡训练阶段，我主要确认了三件事：

1. 纯 PyTorch 脚本能正常训练
2. Lightning 脚本能得到与 PyTorch 接近的结果
3. 数据处理、模型结构和优化器一致时，两者结果应基本对齐

本次实验里，纯 PyTorch 与 Lightning 单卡版本都可以正常跑通，测试精度也比较接近。

这一步说明：Lightning 并不会改变模型本身能力，它主要改变的是工程组织方式。

## 8. 多卡 DDP 的基本原理

在进入多卡训练之前，我先明确了一个核心概念：

`DDP = Distributed Data Parallel`

它的基本思想是：

1. 每张 GPU 上各放一份模型副本
2. 每张卡处理不同的数据 batch
3. 每一步反向传播后，对梯度做同步
4. 再分别更新参数

所以 DDP 的本质不是“一个模型拆到两张卡”，而是“多张卡各自训练同一个模型副本，然后同步梯度”。

## 9. Lightning 中从单卡到 DDP 的最小变化

Lightning 下从单卡切换到 DDP，最关键的变化其实很少。

### 9.1 Trainer 配置变化

单卡：

```python
Trainer(
    accelerator="gpu",
    devices=1,
)
```

双卡 DDP：

```python
Trainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",
)
```

### 9.2 日志同步

在 DDP 中，每张卡对应一个独立进程。如果日志不做同步，就可能只记录某一个进程的局部指标。

因此在 DDP 脚本中，需要使用：

```python
self.log(..., sync_dist=True)
```

### 9.3 GPU 数量检查

如果脚本要求 `devices=2`，那当前进程就必须真的能看到 `2` 张 GPU。

否则 DDP 无法启动。

## 10. Global Batch Size 的概念

这是我在做单卡 / 双卡对照实验时学到的最重要概念之一。

在分布式训练中，真正影响优化过程的不是单个进程的 batch，而是：

```text
global_batch_size = per_device_batch_size × devices × accumulate_grad_batches
```

例如：

- 单卡 `batch_size=128`，global batch = `128`
- 双卡 `batch_size=128, devices=2`，global batch = `256`

如果直接拿这两组实验比较，其实不公平，因为优化条件已经不同。

因此，做单卡和双卡对照时，必须固定 global batch size。

## 11. 固定 Global Batch Size 的实验对照

为了公平比较，我固定：

```text
global_batch_size = 256
```

对应地：

- 单卡：`batch_size=256`
- 双卡 DDP：每卡 `batch_size=128`，`devices=2`

实验结果显示：

- 两者验证和测试精度已经比较接近
- 但双卡 DDP 并没有更快，反而略慢

这说明一个很重要的问题：

多卡训练并不保证一定加速。

对于像 MNIST 这样的小数据集、小模型，DDP 的通信和进程开销可能会超过它带来的并行收益。

## 12. 我在这次实验中踩到的坑

### 12.1 默认数据下载源过慢

最开始直接通过 `torchvision.datasets.MNIST(download=True)` 下载数据时，速度很慢。

后来我把数据手动下载到共享目录，后续所有脚本都复用这份数据，问题才解决。

### 12.2 双卡不一定比单卡快

这是很多初学者最容易误解的地方。

只有当模型足够大、batch 足够大、计算足够重时，多卡才更容易体现加速优势。

### 12.3 DDP 测试阶段要谨慎看指标

Lightning 在多卡 `test()` 阶段会使用 `DistributedSampler`，为了保持 batch 对齐，可能复制部分样本。

所以更严谨的做法是：

1. 用 DDP 训练得到 best checkpoint
2. 再切回单卡加载该 checkpoint 做最终测试

## 13. 我目前对 Lightning 的理解

经过这次实践，我对 Lightning 的理解变得更具体了。

我现在认为：

1. Lightning 不是为了替代 PyTorch，而是建立在 PyTorch 之上的训练工程抽象
2. 它最核心的价值是让训练逻辑更结构化
3. 它尤其适合在后期需要加入 checkpoint、logger、callback、混合精度、多卡训练时使用

如果只是写一个非常短的实验脚本，纯 PyTorch 更直接；但只要项目稍微像样一点，Lightning 的结构优势就会明显体现出来。

## 14. 这次学习的结论

通过这次从 PyTorch 到 Lightning，再到 DDP 的学习实验，我得到几点结论：

1. 纯 PyTorch 更适合理解训练底层流程
2. Lightning 更适合组织训练工程代码
3. 从单卡切换到 DDP，Lightning 的改动很少，但理解分布式原理仍然重要
4. 比较单卡和双卡时，一定要固定 global batch size
5. 小模型和小数据集上，双卡 DDP 未必值得

## 15. 下一步学习计划

接下来我准备继续学习以下内容：

1. 混合精度训练
2. 梯度累积与 global batch size 的进一步关系
3. 使用单卡测试 best checkpoint 的规范评估方式
4. 在更大一些的数据集或模型上观察 DDP 的真实收益
5. 进一步学习 Lightning 中的 callback、logger 和配置管理

## 16. 参考文件

- [Lightning学习计划.md](/root/autodl-tmp/Lightning-train-learning/Lightning学习计划.md)
- [setup_and_run.md](/root/autodl-tmp/Lightning-train-learning/setup_and_run.md)
- [train_mnist.py](/root/autodl-tmp/Lightning-train-learning/pytorch-MNIST/train_mnist.py)
- [train_mnist_lightning.py](/root/autodl-tmp/Lightning-train-learning/Lightning-MNIST/train_mnist_lightning.py)
- [train_mnist_single_gpu.py](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/train_mnist_single_gpu.py)
- [train_mnist_ddp.py](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/train_mnist_ddp.py)
- [compare_global_batch_size.md](/root/autodl-tmp/Lightning-train-learning/DDP-MNIST/compare_global_batch_size.md)
