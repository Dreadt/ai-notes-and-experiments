# 固定 Global Batch Size 的单卡 / 双卡 DDP 对照

## 为什么要固定 global batch size

比较单卡和双卡时，不能只看命令里写的 `batch_size`。

在 Lightning DDP 里：

```text
global_batch_size = per_device_batch_size × devices × accumulate_grad_batches
```

如果你写：

- 单卡：`batch_size=128`
- 双卡：`batch_size=128, devices=2`

那么它们的全局 batch 实际上分别是：

- 单卡：`128`
- 双卡：`256`

这不是公平对照，因为优化过程已经变了。

## 本次对照设置

我固定：

```text
global_batch_size = 256
```

对应换算为：

- 单卡：`batch_size=256`
- 双卡 DDP：`per_device_batch_size=128, devices=2`

其余关键参数保持一致：

- `epochs=1`
- `num_workers=2`
- `seed=42`
- 同一份数据目录：`/root/autodl-tmp/Lightning-train-learning/shared-data`
- 同一模型结构、同一优化器、同一学习率

## 运行命令

单卡：

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_single_gpu.py \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 2 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data \
  --save-dir ./compare_global256_single
```

双卡 DDP：

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_ddp.py \
  --epochs 1 \
  --batch-size 128 \
  --devices 2 \
  --num-workers 2 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data \
  --save-dir ./compare_global256_ddp
```

## 实际结果

### 单卡，global batch = 256

- train_loss: `0.403`
- train_acc: `0.879`
- val_loss: `0.185`
- val_acc: `0.9433`
- test_loss: `0.1648`
- test_acc: `0.9484`
- elapsed: `8.682s`

checkpoint:

- `compare_global256_single/best-single-epoch=00-val_acc=0.9433.ckpt`

### 双卡 DDP，global batch = 256

- train_loss: `0.398`
- train_acc: `0.881`
- val_loss: `0.182`
- val_acc: `0.9428`
- test_loss: `0.1578`
- test_acc: `0.9503`
- elapsed: `10.054s`

checkpoint:

- `compare_global256_ddp/best-ddp-epoch=00-val_acc=0.9428.ckpt`

## 怎么理解这组结果

结论很直接：

1. 固定 global batch size 后，单卡和双卡的收敛结果已经很接近。
2. 在这个小模型上，双卡 DDP 没有更快，反而更慢。
3. 这是正常现象，因为 MNIST 模型太小，计算量不大，DDP 的进程启动、通信同步和分布式开销占比反而更明显。

本次时间对比：

- 单卡：`8.682s`
- 双卡 DDP：`10.054s`

也就是说这次双卡没有带来吞吐优势。

## 为什么双卡可能更慢

常见原因：

1. 模型太小，GPU 计算不够“重”
2. 每步都要做梯度同步，通信成本抵消了并行收益
3. 训练轮数太少，DDP 初始化开销占比大
4. 数据集太小，难以体现多卡优势

MNIST 就是一个很典型的“不适合用来证明 DDP 加速”的例子。

## 一个重要注意点

Lightning 在双卡 `trainer.test()` 时给了明确提示：

- 多设备测试会使用 `DistributedSampler`
- 为了让每个进程 batch 对齐，可能复制部分样本

所以：

1. 上面的训练/验证对照是合理的
2. 但 `DDP` 脚本里直接跑出的 `test_acc` 不是最严格的最终评估方式
3. 更严谨的做法是：
   先用 DDP 训练保存 checkpoint
   再用 `devices=1` 单卡加载这个 checkpoint 做最终测试

## 你现在应该记住什么

最关键的不是“双卡一定更快”，而是：

1. 对照实验必须先固定 `global batch size`
2. DDP 比较适合更大的模型、更大的 batch、更多训练步
3. 小模型和小数据集上，DDP 可能完全不划算

如果你继续学 Lightning 多卡训练，下一步最值得做的是：

1. 再做一组 `global batch = 128` 的公平对照
2. 把 `trainer.test()` 改成“训练完后用单卡加载 best checkpoint 测试”
3. 换一个比 MNIST 更重一点的数据集或模型，再观察多卡收益
