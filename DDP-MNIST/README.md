# Lightning MNIST: 单卡版与 DDP 多卡版对照

这个目录里的目标不是再写一套新模型，而是直接基于前面的 Lightning MNIST 脚本，给出一份最小的“单卡版 -> 多卡 DDP 版”对照。

## 文件说明

- `train_mnist_single_gpu.py`
  单卡训练版本。
- `train_mnist_ddp.py`
  多卡 DDP 训练版本。

两份脚本的模型结构、数据处理、优化器都保持一致，重点只看 `Trainer` 配置和 DDP 相关差异。

## 最关键的差异

### 1. `Trainer` 配置不同

单卡版：

```python
trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    ...
)
```

DDP 版：

```python
trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=args.devices,
    strategy="ddp",
    ...
)
```

这里真正决定“多卡分布式”的是：

```python
devices=2
strategy="ddp"
```

### 2. DDP 版日志需要 `sync_dist=True`

单卡版：

```python
self.log("val_acc", acc, on_step=False, on_epoch=True)
```

DDP 版：

```python
self.log("val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
```

因为 DDP 下每张卡是一个独立进程，不加 `sync_dist=True` 时，日志指标通常只是当前进程的局部值，不是全局聚合结果。

### 3. DDP 版必须能看到多张 GPU

在 `train_mnist_ddp.py` 里额外加了检查：

```python
if torch.cuda.device_count() < args.devices:
    raise RuntimeError(...)
```

这是因为你请求 `devices=2` 时，当前进程必须真的能访问到两张 GPU。

## 是否需要访问另一张卡？

需要。

如果你要真正运行 `DDP` 多卡训练，训练进程必须同时“看得到”多张 GPU。例如：

- `CUDA_VISIBLE_DEVICES=0,1`
- 或者机器本身就给了你 2 张及以上 GPU

如果当前环境只能看到 1 张卡，那么：

- 单卡版可以正常运行
- DDP 脚本可以作为学习示例阅读
- 但 `devices=2, strategy="ddp"` 不能真正启动

## 当前环境结论

你当前这台机器上，我检查到：

```bash
nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3090
```

也就是说当前只看到 `1` 张 GPU，因此现在不能实际跑 `2` 卡 DDP。

## 运行方式

单卡版：

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_single_gpu.py --epochs 5 --batch-size 128
```

双卡 DDP 版：

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_ddp.py --epochs 5 --batch-size 128 --devices 2
```

## 学习建议

建议按这个顺序看：

1. 先看 `train_mnist_single_gpu.py`
2. 再看 `train_mnist_ddp.py`
3. 只比较下面三类差异：
   `Trainer(...)`
   `self.log(..., sync_dist=True)`
   GPU 数量检查

如果这三处你能看明白，Lightning 下从单卡切到 DDP 的最核心变化就已经抓住了。
