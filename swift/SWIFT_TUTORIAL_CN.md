# SWIFT 实践教程

## 1. 安装环境

建议先准备独立环境：

```bash
conda create -n swift310 python=3.10 -y
conda activate swift310
pip install ms-swift -U
```

对应脚本：[scripts/00_install_ms_swift.sh](/root/autodl-tmp/ai-notes-and-experiments/swift/scripts/00_install_ms_swift.sh)

## 2. 准备数据集

我们这里用了两层数据：

- 本地小数据集：`data/arithmetic_grpo_train.jsonl`
- 参考数据集：`zouxuhong/Countdown-Tasks-3to4`

先生成本地参考样本：

```bash
python scripts/21_prepare_countdown_sample.py
```

对应脚本：[scripts/21_prepare_countdown_sample.py](/root/autodl-tmp/ai-notes-and-experiments/swift/scripts/21_prepare_countdown_sample.py)

## 3. 选择数据

推荐先跑小样本：

- `data/countdown_sample_256.jsonl`

数据字段核心是：

- `messages`
- `nums`
- `target`

## 4. 开始训练

单卡 smoke：

```bash
bash scripts/22_run_countdown_smoke_1gpu.sh
```

对应脚本：[scripts/22_run_countdown_smoke_1gpu.sh](/root/autodl-tmp/ai-notes-and-experiments/swift/scripts/22_run_countdown_smoke_1gpu.sh)

多卡建议优先 `colocate`，因为我们实测 `1.7B/4B server-mode` 在当前环境不稳定。

## 5. 训练参数

我们实验里比较稳的一组参数是：

- `--rlhf_type grpo`
- `--tuner_type lora`
- `--torch_dtype bfloat16`
- `--num_generations 4`
- `--generation_batch_size 4`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 1`
- `--learning_rate 5e-6`
- `--max_length 512`
- `--max_completion_length 128`
- `--reward_funcs countdown_correct countdown_format`
- `--reward_weights 1.0 0.2`

## 6. 查看结果

重点看这些文件：

- `outputs/.../logging.jsonl`
- `outputs/.../checkpoint-*`
- `outputs/.../images/*.png`

## 7. 实践结论

- `0.6B` 适合先跑通流程
- `1.7B` 适合做主实验
- 当前机器上 `2GPU colocate` 比 `2GPU server-mode` 更稳

