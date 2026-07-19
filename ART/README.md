# ART 本地实验

这个目录用于验证 `OpenPipe ART` 在当前双卡服务器上的最小可运行流程，目标不是马上做出高质量 RL 结果，而是先跑通 `register -> rollout -> PipelineTrainer.train()` 这条链路，并理解 ART 的本地执行架构。

## 当前设计

- ART 版本：`openpipe-art==0.5.18`
- Python 环境：`/root/autodl-tmp/ai-notes-and-experiments/ART/.venv_sys`
- 环境策略：`--system-site-packages`
- 原因：复用系统里已经可用的 `torch / vllm / transformers`，避免 ART 安装过程覆盖系统训练环境
- 基础模型：`Qwen/Qwen3-0.6B`
- Backend：`art.local.LocalBackend`
- 训练方式：`PipelineTrainer`
- GPU 划分：dedicated mode
- 训练卡：`GPU 0`
- 推理卡：`GPU 1`
- 当前已落地的实验：
  - `local_smoke.py`：binary bitstring reward smoke
  - `local_tool_smoke.py`：tool-use style reward smoke
  - `local_mcp_rl.py`：real MCP server + ART local RL smoke

## 为什么必须用 dedicated mode

`PipelineTrainer + LocalBackend` 在 ART 里只支持 dedicated mode。也就是训练和推理必须分配到不同 GPU 上，否则本地异步 rollout 和训练会互相阻塞。

当前机器是 `2 x RTX 3090 24GB`，因此最自然的配置是：

```text
trainer_gpu_ids=[0]
inference_gpu_ids=[1]
```

## 实验内容

这个 smoke 实验现在使用一个“二进制前缀猜测”任务：

- 用户提示词要求模型输出固定长度的二进制串
- rollout 一次生成多个候选
- 奖励函数综合考虑：
  - shared prefix
  - 按位匹配率
  - 长度正确 bonus
  - 非法长度 penalty
- 训练步数默认为很小的 `2` 步
- scenario 不再只用单一 secret，而是包含 `4` 个变体，降低 reward collapse 风险

这个实验的意义是：

- 验证 ART 的本地注册和服务拉起流程
- 验证 OpenAI-compatible 推理接口是否可用
- 验证 `Trajectory / TrajectoryGroup / PipelineTrainer` 的最小闭环
- 为后续换成更有实践意义的数据或 tool-use 场景提供脚手架

## 目录结构

```text
ART/
├── README.md
├── art_experiment/
│   ├── local_smoke.py
│   ├── local_tool_smoke.py
│   ├── local_mcp_rl.py
│   └── mcp_client_smoke.py
├── mcp_servers/
│   └── arithmetic_server.py
└── scripts/
    ├── 00_setup_venv_sys.sh
    ├── 10_verify_imports.sh
    └── 20_run_local_smoke.sh
    └── 30_run_local_tool_smoke.sh
    └── 31_run_local_mcp_rl.sh
```

## 运行方法

先验证导入：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/10_verify_imports.sh
```

再跑 smoke 实验：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/20_run_local_smoke.sh
```

跑 tool-use smoke：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/30_run_local_tool_smoke.sh
```

跑真实 MCP•RL smoke：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/31_run_local_mcp_rl.sh
```

## 默认运行参数

可以通过环境变量覆盖：

- `ART_BASE_MODEL`：默认 `Qwen/Qwen3-0.6B`
- `ART_MAX_STEPS`：默认 `2`
- `ART_ROLLOUTS_PER_SCENARIO`：默认 `8`
- `ART_MAX_TOKENS`：默认 `24`
- `ART_ROLLOUT_TEMPERATURE`：默认 `1.1`
- `ART_EVAL_TEMPERATURE`：默认 `0.3`
- `ART_MIN_BATCH_SIZE`：默认 `2`
- `ART_DISCARD_QUEUE_MULTIPLIER`：默认 `500`
- `ART_ART_PATH`：默认当前目录下 `.art_local`
- `WANDB_MODE`：脚本默认设为 `offline`

## 已知风险

- `LocalBackend` 运行时仍可能触发 `vllm` 的额外依赖要求
- dedicated mode 下推理只支持单卡，因此这里固定 `inference_gpu_ids=[1]`
- 本实验是架构验证，不代表 reward 设计或训练效果已经足够好

## 2026-07-18 实测状态

截至 `2026-07-18`，这个目录里的实验已经完成了下面这些验证：

- `.venv_sys` 环境可正常导入 `art`
- `LocalBackend` 与 `PipelineTrainer` 导入成功
- dedicated mode 配置已按 `trainer_gpu_ids=[0]`、`inference_gpu_ids=[1]` 写入
- `local_smoke.py` 已能越过 ART 内部的 `megatron` 目标模块推断路径
- `unsloth` 训练上下文已成功初始化
- ART managed `vLLM runtime` 已成功安装到本地大盘缓存
- 本地 `art-vllm-runtime-server` 已成功拉起
- `model.register(backend)` 已成功
- `PipelineTrainer.train()` 已真实开始执行

当前这套 smoke 实验已经完成了两轮不同阶段的验证：

### 第一轮：链路打通但任务塌缩

- 旧版任务过于简单，ART 触发 `MODEL COLLAPSE DETECTED`
- 触发原因：连续 `400` 个 `TrajectoryGroup` 的 reward variance 为 `0`
- 停止位置：`step=0`

### 第二轮：任务改造后可完整训练

`2026-07-18 20:23 UTC` 左右重跑后，实验已完整结束 `2` 个训练 step：

- 模型目录：`.art_local/art-local-smoke/models/qwen3-0.6b-local-smoke-20260718-202328`
- 最终状态：
  - `trained=25`
  - `queued=80`
  - `discarded total=11`
  - `discarded 0_var=11`
  - `train step=2 reward=-0.106`
  - `train avg_std=0.013`
  - `val reward=-0.088`

这意味着：

- ART 本地训练主链路已经不仅是“启动成功”，而是完成了真实训练与评估
- 通过增加 scenario 多样性和 reward 分层，`0_var` 丢弃量显著下降
- 这套实验现在适合作为后续升级到 reasoning / tool-use 任务的最小脚手架

### 第三轮：tool-use 风格 smoke 成功

在 bitstring 任务稳定后，又新增了一个更接近 agent / tool-use 的最小实验：

- 任务形式：给定一个工具名和两个操作数，模型输出 `TOOL=<tool> RESULT=<n>`
- 候选工具：`add / sub / mul / max`
- reward：按“格式可解析 / 工具是否正确 / 结果是否正确 / 数值距离”分层

`2026-07-18 21:01 UTC` 左右的成功重跑结果：

- 模型目录：`.art_local/art-local-tool-smoke/models/qwen3-0.6b-local-tool-smoke-20260718-210123`
- 最终状态：
  - `trained=23`
  - `queued=80`
  - `discarded total=8`
  - `discarded 0_var=8`
  - `train step=2 reward=0.292`
  - `train avg_std=0.218`
  - `val reward=0.450`

这说明：

- ART 不仅能跑“纯 synthetic reward”任务，也能跑一个最小的 tool-use style reward 环境
- 对 `0.6B` 模型，宽松格式 + 分层 reward 明显比严格结构化输出更稳

### 第四轮：真实 MCP•RL 本地实验完成

这轮实验不再只是“模拟工具奖励”，而是接入了一个真实的本地 MCP server：

- server 实现：`mcp_servers/arithmetic_server.py`
- transport：`stdio`
- 工具发现：运行时通过 `list_tools()` 获取
- 实际工具调用：reward 过程中通过 `call_tool()` 验证工具返回值

这轮实验经历了两个版本：

#### 版本 A：`TOOL + A + B + RESULT`

- 能跑通，但动作空间太大
- 终态大致为：
  - `trained=4`
  - `queued=24`
  - `discarded total=183`
  - `discarded 0_var=183`
  - `train step=2 reward=-0.037`
  - `val reward=-0.200`

#### 版本 B：`TOOL + RESULT`

- 保留真实 MCP `call_tool()` reward
- 去掉参数复述要求，只训练“工具选择 + 结果预测”
- `2026-07-18` 的最终成功结果：
  - 模型目录：`.art_local/art-local-mcp-rl/models/qwen3-0.6b-local-mcp-rl-20260719-001149`
  - `trained=7`
  - `queued=16`
  - `discarded total=12`
  - `discarded 0_var=12`
  - `train step=2 reward=0.190`
  - `train avg_std=0.061`
  - `val reward=0.415`

这说明：

- 当前目录里已经完成了一个真正意义上的“本地 MCP•RL 最小实验”
- ART 本地双卡链路可以和真实 MCP server 协同工作
- 对小模型来说，动作空间设计仍然比“是否接了 MCP”更关键
