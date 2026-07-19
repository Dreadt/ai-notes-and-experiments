# ART 详细实践教程

日期：`2026-07-19`

这份教程基于当前目录里已经跑通的本地实验来写，目标不是介绍 ART 的所有功能，而是让你在这台机器上按实际可行路径复现：

- 环境安装
- 导入验证
- local smoke
- tool-use smoke
- 真实 MCP-RL smoke

## 1. 目录说明

当前目录结构：

```text
ART/
├── README.md
├── REPORT.md
├── ART_EXPERIMENT_REPORT_CN_2026-07-19.md
├── ART_TUTORIAL_CN_2026-07-19.md
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
    ├── 20_run_local_smoke.sh
    ├── 30_run_local_tool_smoke.sh
    └── 31_run_local_mcp_rl.sh
```

## 2. 环境准备

### 2.1 为什么用 `.venv_sys`

这里没有单独重装一整套 `torch / vllm / transformers`，而是采用：

- `python venv --system-site-packages`

原因：

- 复用系统里已经可用的深度学习环境
- 降低 ART 安装对现有训练环境的破坏

### 2.2 创建环境

运行：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/00_setup_venv_sys.sh
```

脚本内容见：

- [scripts/00_setup_venv_sys.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/00_setup_venv_sys.sh)

它会做两件事：

- 创建 `.venv_sys`
- 安装 `openpipe-art==0.5.18`

## 3. 验证导入

先不要急着训练，先确认核心模块能导入：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/10_verify_imports.sh
```

脚本见：

- [scripts/10_verify_imports.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/10_verify_imports.sh)

预期你至少能看到这些模块导入成功：

- `art`
- `LocalBackend`
- `PipelineTrainer`

如果这一步失败，就先不要往后跑。

## 4. GPU 架构理解

本地 ART 实验这里采用的是 dedicated mode。

也就是：

- `GPU 0` 给 trainer
- `GPU 1` 给 inference / rollout

原因：

- `PipelineTrainer + LocalBackend` 需要训练和推理解耦
- 如果放在同一张卡上，本地异步 rollout 会和训练互相卡住

所以这套实验的默认理解是：

```text
trainer_gpu_ids=[0]
inference_gpu_ids=[1]
```

## 5. 第一个实验：local smoke

### 5.1 作用

这是最小架构验证实验。

目标：

- 跑通 `register -> rollout -> train`
- 验证本地 ART 训练闭环
- 不依赖真实工具或 MCP

### 5.2 启动命令

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/20_run_local_smoke.sh
```

脚本见：

- [scripts/20_run_local_smoke.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/20_run_local_smoke.sh)

对应 Python 文件：

- [art_experiment/local_smoke.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_smoke.py)

### 5.3 默认参数

脚本里默认会设置：

- `ART_BASE_MODEL=Qwen/Qwen3-0.6B`
- `ART_MAX_STEPS=2`
- `ART_ROLLOUTS_PER_SCENARIO=8`
- `ART_MAX_TOKENS=24`
- `ART_ROLLOUT_TEMPERATURE=1.1`
- `ART_EVAL_TEMPERATURE=0.3`
- `ART_MIN_BATCH_SIZE=2`

### 5.4 你会看到什么

如果成功，说明这些动作已经发生：

- ART 本地 runtime 被拉起
- model 注册成功
- rollout 开始
- `PipelineTrainer.train()` 开始执行

### 5.5 常见问题

如果出现 `MODEL COLLAPSE DETECTED`：

- 通常不是环境挂了
- 而是 reward 没有区分度
- 需要提高 scenario 多样性，或者重写 reward

## 6. 第二个实验：tool-use smoke

### 6.1 作用

这一步不是纯文本 reward，而是更接近 agent / tool-use 的任务。

目标：

- 让模型学会“选工具 + 给结果”
- 但还不接真实 MCP

### 6.2 启动命令

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/30_run_local_tool_smoke.sh
```

脚本见：

- [scripts/30_run_local_tool_smoke.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/30_run_local_tool_smoke.sh)

对应 Python 文件：

- [art_experiment/local_tool_smoke.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_tool_smoke.py)

### 6.3 任务形式

模型会面对一种最小工具任务，输出形如：

```text
TOOL=<tool> RESULT=<n>
```

reward 不是只看最终对错，而是分层：

- 能不能解析
- 工具是否对
- 结果是否对
- 数值是否接近

### 6.4 为什么要这样设计

因为如果一开始就要求非常严格的结构化输出，小模型很容易：

- 全部输出低分
- reward variance 过低
- 训练直接塌缩

所以正确策略是：

- 先给平滑 reward
- 再逐步收紧格式

## 7. 第三个实验：真实 MCP-RL

### 7.1 作用

这是当前目录里最重要的实验。

它和 `tool-use smoke` 的区别是：

- 不再模拟工具调用
- 而是接入真实 MCP server

### 7.2 先理解组成

涉及三个文件：

- MCP server：
  [mcp_servers/arithmetic_server.py](/root/autodl-tmp/ai-notes-and-experiments/ART/mcp_servers/arithmetic_server.py)
- client smoke：
  [art_experiment/mcp_client_smoke.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/mcp_client_smoke.py)
- ART 训练脚本：
  [art_experiment/local_mcp_rl.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_mcp_rl.py)

### 7.3 启动命令

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/31_run_local_mcp_rl.sh
```

脚本见：

- [scripts/31_run_local_mcp_rl.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/31_run_local_mcp_rl.sh)
- [scripts/32_run_local_mcp_rl_1p7b.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/32_run_local_mcp_rl_1p7b.sh)
- [scripts/33_run_local_mcp_rl_4b.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/33_run_local_mcp_rl_4b.sh)

如果要直接复现更大模型，可以分别运行：

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/32_run_local_mcp_rl_1p7b.sh
```

```bash
cd /root/autodl-tmp/ai-notes-and-experiments/ART
bash scripts/33_run_local_mcp_rl_4b.sh
```

### 7.4 这一步真正验证了什么

这一步验证的是：

- ART rollout 期间是否能接上真实 MCP server
- 是否能通过 `list_tools()` 获取工具
- 是否能通过 `call_tool()` 取得真实工具结果
- reward 是否能建立在真实环境反馈之上

这就是 `MCP-RL` 的最小形态。

### 7.5 实践建议

第一次做真实 MCP-RL 时，不要把动作空间设计得太复杂。

更好的做法是：

- 先让模型只预测 `TOOL + RESULT`
- 先确认 reward 可训练
- 再慢慢加参数复述、多轮调用、严格格式

### 7.6 当前已经验证成功的模型

截至 `2026-07-19`，这三档模型都已经在当前机器上跑通最小 `MCP-RL`：

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`

其中：

- `1.7B` 成功结果目录：
  [qwen3-1p7b-local-mcp-rl-20260719-150631](/root/autodl-tmp/ai-notes-and-experiments/ART/.art_local/art-local-mcp-rl/models/qwen3-1p7b-local-mcp-rl-20260719-150631)
- `4B` 成功结果目录：
  [qwen3-4b-local-mcp-rl-20260719-150901](/root/autodl-tmp/ai-notes-and-experiments/ART/.art_local/art-local-mcp-rl/models/qwen3-4b-local-mcp-rl-20260719-150901)

两个关键结果：

- `1.7B`: `train reward=0.325`, `val reward=0.415`, `step_seconds=26.43`
- `4B`: `train reward=0.300`, `val reward=0.530`, `step_seconds=31.72`

要注意：

- `4B` 看起来“启动很慢”，主要不是报错，而是在做 `vLLM` 的 `torch.compile` 和 `CUDA graph capture`
- 只要后面出现 `model registered` 和 `training start`，就说明链路已经越过最容易失败的阶段

## 8. 常用环境变量

这些脚本都支持通过环境变量覆盖默认参数：

- `ART_BASE_MODEL`
- `ART_MAX_STEPS`
- `ART_ROLLOUTS_PER_SCENARIO`
- `ART_MAX_TOKENS`
- `ART_ROLLOUT_TEMPERATURE`
- `ART_EVAL_TEMPERATURE`
- `ART_MIN_BATCH_SIZE`
- `ART_DISCARD_QUEUE_MULTIPLIER`
- `ART_ART_PATH`

例如：

```bash
ART_MAX_STEPS=4 ART_MAX_TOKENS=64 bash scripts/31_run_local_mcp_rl.sh
```

## 9. 结果怎么看

跑完以后，重点关注：

- `.art_local/.../models/...`
- 训练过程里的 `trained / queued / discarded`
- `train reward`
- `train avg_std`
- `val reward`

判断标准：

- 如果 `train()` 能完整结束，说明链路通了
- 如果 `discarded 0_var` 极高，说明任务设计或 reward 设计有问题
- 如果 `val reward` 提升，说明当前任务可训练

## 10. 一条推荐复现路径

如果你现在从零开始，推荐顺序是：

1. 安装环境  
```bash
bash scripts/00_setup_venv_sys.sh
```

2. 验证导入  
```bash
bash scripts/10_verify_imports.sh
```

3. 跑最小 smoke  
```bash
bash scripts/20_run_local_smoke.sh
```

4. 跑 tool-use smoke  
```bash
bash scripts/30_run_local_tool_smoke.sh
```

5. 跑真实 MCP-RL  
```bash
bash scripts/31_run_local_mcp_rl.sh
```

这样做的好处是：

- 每一步都能单独定位问题
- 不会把环境错误、reward 错误、MCP 接入错误混在一起

## 11. 当前最重要的经验

这次 ART 实验最重要的实践经验不是“怎么安装”，而是下面三条：

- `ART` 很适合 agent / tool-use / environment feedback 类型任务
- 小模型下，动作空间设计比框架本身更关键
- 做 `MCP-RL` 时，先做最小闭环，再慢慢增加复杂度

一句话总结：

- 先跑通
- 再做稳
- 最后再做难
