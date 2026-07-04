# 环境准备与运行说明

## 1. 推荐环境

当前机器上已经验证可运行的环境是系统 Python 环境：

```bash
python --version
pip install "lightning>=2.4,<3.0"
```

已确认可用的关键版本：

```bash
torch==2.8.0+cu128
torchvision==0.23.0+cu128
lightning==2.6.5
```

## 2. 可选：创建虚拟环境

```bash
cd /root/autodl-tmp/Lightning-train-learning
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

说明：

- 如果 `venv` 不继承系统包，它会重新下载 `torch`，耗时会比较长。
- 如果你只是复现当前结果，直接用系统环境更省时间。

## 3. 先准备共享数据目录

为了避免 `torchvision` 默认下载源过慢，建议先手动把 `MNIST` 下载到共享目录：

```bash
mkdir -p /root/autodl-tmp/Lightning-train-learning/shared-data/MNIST/raw
cd /root/autodl-tmp/Lightning-train-learning/shared-data/MNIST/raw
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
```

后面的所有脚本都统一使用：

```bash
--data-dir /root/autodl-tmp/Lightning-train-learning/shared-data
```

## 4. 运行 PyTorch 版

```bash
cd /root/autodl-tmp/Lightning-train-learning/pytorch-MNIST
python train_mnist.py --epochs 2 --batch-size 128 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data
```

## 5. 运行 Lightning 单卡版

```bash
cd /root/autodl-tmp/Lightning-train-learning/Lightning-MNIST
python train_mnist_lightning.py --epochs 2 --batch-size 128 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data \
  --accelerator gpu --devices 1
```

## 6. 运行 Lightning DDP 单卡对照版

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_single_gpu.py --epochs 2 --batch-size 128 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data
```

## 7. 运行 Lightning DDP 双卡版

```bash
cd /root/autodl-tmp/Lightning-train-learning/DDP-MNIST
python train_mnist_ddp.py --epochs 2 --batch-size 128 \
  --data-dir /root/autodl-tmp/Lightning-train-learning/shared-data \
  --devices 2
```

## 8. 当前目录中的训练脚本

- `pytorch-MNIST/train_mnist.py`
- `Lightning-MNIST/train_mnist_lightning.py`
- `DDP-MNIST/train_mnist_single_gpu.py`
- `DDP-MNIST/train_mnist_ddp.py`
