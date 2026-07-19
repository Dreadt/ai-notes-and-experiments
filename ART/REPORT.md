# ART 实验状态报告

日期：`2026-07-18`

## 目标

在 `/root/autodl-tmp/ai-notes-and-experiments/ART` 中搭建一套可复现的 `OpenPipe ART` 本地实验，优先验证：

- 本地 backend 是否可用
- dedicated mode 双卡划分是否可配置
- `register -> rollout -> PipelineTrainer` 的链路是否能跑通

## 已完成内容

已创建：

- `README.md`
- `scripts/00_setup_venv_sys.sh`
- `scripts/10_verify_imports.sh`
- `scripts/20_run_local_smoke.sh`
- `art_experiment/local_smoke.py`

已验证成功：

- `openpipe-art==0.5.18` 在 `.venv_sys` 中可导入
- `LocalBackend` 可导入
- `PipelineTrainer` 可导入
- `TrainableModel` + dedicated mode 配置可构造

## 实跑结论

### 第一阶段

`model.register(backend)` 初始失败，报错为：

- `ModuleNotFoundError: No module named 'megatron'`

处理：

- 补装 `megatron-core==0.16.0`
- 补装 `megatron-bridge==0.4.0`
- 同时发现 ART 在本地 smoke 阶段对完整 `megatron` 栈存在过早导入
- 为 `local_smoke.py` 增加了一个只用于 smoke 的 `model_support shim`

### 第二阶段

越过 `megatron` 后，进入：

- `UnslothService`

随后失败，报错为：

- `ModuleNotFoundError: No module named 'unsloth'`

处理：

- 补装 `unsloth==2026.3.3`
- 补装 `unsloth-zoo==2026.3.1`
- 补装 `bitsandbytes==0.45.2`

### 第三阶段

进入 `unsloth` 初始化后再次失败，核心错误是：

```text
GLIBCXX_3.4.32 not found
AttributeError: 'NoneType' object has no attribute 'cdequantize_blockwise_fp32'
```

这表示：

- `bitsandbytes` 本地二进制库没有成功加载
- `unsloth` 依赖 `bitsandbytes` 的底层函数指针
- 因此训练上下文在初始化时中断

### 第四阶段

经过下面这些修正之后，ART 本地 smoke 已经成功进入真实训练流程：

- 使用本地 `conda` 运行库提供 `GLIBCXX_3.4.32`
- 将 `bitsandbytes` 升级到支持 `cuda128` 的 `0.49.2`
- 将 `.venv_sys` 中的 `transformers / trl / accelerate / peft` 覆写到 ART 预期版本
- 将 `ART` 的基础模型切到本地缓存快照路径
- 在 smoke 配置里关闭 `unsloth` 的 4bit 自动模型映射
- 安装 `uv`
- 将 `ART_VLLM_RUNTIME_CACHE_DIR` 和 `UV_CACHE_DIR` 重定向到 `/root/autodl-tmp`

在 `2026-07-18` 的第一次最终实跑中，出现了下面这个结果：

- `model.register(backend)` 成功
- managed `vLLM runtime` 安装成功
- 本地 `art-vllm-runtime-server` 成功拉起
- `PipelineTrainer.train()` 已开始运行
- 随后 ART 主动停止训练，原因是：
  - `MODEL COLLAPSE DETECTED`
  - `400` 个 trajectory groups 的 reward variance 为 `0`

这不是环境失败，而是当前 smoke 任务设计过于简单，导致 rollout 几乎没有奖励差异，ART 将其判定为退化策略并触发保护停止。

### 第五阶段

随后对 smoke 任务做了两类改造：

- 将运行脚本默认参数从旧值同步到新值：
  - `ART_ROLLOUTS_PER_SCENARIO=8`
  - `ART_MAX_TOKENS=24`
  - `ART_ROLLOUT_TEMPERATURE=1.1`
  - `ART_EVAL_TEMPERATURE=0.3`
- 将任务从单一 secret 改成 `4` 个 secret 变体，并把 reward 改成“前缀 + 按位匹配 + 长度 bonus - 长度 penalty”

在 `2026-07-18 20:23 UTC` 开始的第二次重跑中，训练完整结束 `2` 个 step，模型目录为：

- `.art_local/art-local-smoke/models/qwen3-0.6b-local-smoke-20260718-202328`

终态指标：

- `trained=25`
- `queued=80`
- `discarded total=11`
- `discarded 0_var=11`
- `train step=2 reward=-0.106`
- `train avg_std=0.013`
- `val reward=-0.088`

从 `history.jsonl` 可确认：

- `step=1`
  - `train/reward=-0.1073`
  - `train/reward_std_dev=0.0132`
  - `data/step_num_trajectories=40`
- `step=2`
  - `train/reward=-0.1006`
  - `train/reward_std_dev=0.0108`
  - `data/step_num_trajectories=160`
- `val/reward=-0.0884`

### 第六阶段

为了把实验从“纯格式猜测任务”推进到更接近 agentic RL 的方向，又新增了一个 `tool-use smoke`：

- 脚本：`art_experiment/local_tool_smoke.py`
- 启动脚本：`scripts/30_run_local_tool_smoke.sh`
- 初版设计：
  - 输出格式要求过严：`TOOL + ARGS + RESULT`
  - 结果：虽然能训练，但大部分 rollout 被判成同质低分
  - 一次完整实跑终态约为：
    - `trained=3`
    - `discarded=799`
    - `val reward=-0.200`

随后进行了第二次改造：

- 放宽输出格式为：`TOOL=<tool> RESULT=<n>`
- reward 改成更平滑的分层形式：
  - 可解析奖励
  - 工具正确奖励
  - 结果正确奖励
  - 数值接近奖励
  - 工具与结果同时正确的 bonus

在 `2026-07-18 21:01 UTC` 开始的第二次重跑中，最终结果为：

- 模型目录：`.art_local/art-local-tool-smoke/models/qwen3-0.6b-local-tool-smoke-20260718-210123`
- 终态指标：
  - `trained=23`
  - `queued=80`
  - `discarded total=8`
  - `discarded 0_var=8`
  - `train step=1 reward=0.253`
  - `train step=2 reward=0.292`
  - `train avg_std=0.218`
  - `val reward=0.450`

这个结果的重要性在于：

- 双卡 dedicated mode 下的 ART 本地训练不仅能跑通，而且已经能支持一个“近似 tool-use”的 reward 环境
- 对小模型而言，reward shaping 和输出格式设计比“是否是 tool-use”本身更关键
- 这给后续迁移到真正的 MCP / ART agent 环境提供了明确经验：先放宽动作空间和解析规则，再逐步收紧

### 第七阶段

为了把“tool-use style smoke”升级成真正的 `MCP•RL`，新增了一个真实本地 MCP server 实验：

- MCP server：`mcp_servers/arithmetic_server.py`
- client smoke：`art_experiment/mcp_client_smoke.py`
- ART 训练脚本：`art_experiment/local_mcp_rl.py`
- 启动脚本：`scripts/31_run_local_mcp_rl.sh`

这个实验的关键区别是：

- 不是在 reward 函数里手写 Python 工具逻辑
- 而是通过 MCP `stdio` transport 启动 server
- 运行时先 `list_tools()`
- 训练和评估时真实调用 `call_tool()`
- 再根据返回值给 reward

#### 第一次实跑

首版动作格式要求模型输出：

- `TOOL=<tool> A=<a> B=<b> RESULT=<n>`

这版能完整训练，但质量较差，终态约为：

- `trained=4`
- `queued=24`
- `discarded total=183`
- `discarded 0_var=183`
- `train step=2 reward=-0.037`
- `val reward=-0.200`

问题判断：

- 真实 MCP 接入本身没有问题
- 问题主要出在小模型同时要完成：
  - 选对工具
  - 复述参数
  - 预测结果
- 动作空间过大，导致太多同质低分 trajectory group

#### 第二次实跑

随后将动作格式放宽为：

- `TOOL=<tool> RESULT=<n>`

但 reward 仍然保持真实 MCP 调用：

- 训练时仍通过 `call_tool()` 验证正确工具在给定参数上的真实返回
- 也就是去掉了“参数复述负担”，但没有退回成纯模拟奖励

在 `Saturday, July 18, 2026` 这次最终成功重跑中，模型目录为：

- `.art_local/art-local-mcp-rl/models/qwen3-0.6b-local-mcp-rl-20260719-001149`

终态指标：

- `trained=7`
- `queued=16`
- `discarded total=12`
- `discarded 0_var=12`
- `train step=2 reward=0.190`
- `train avg_std=0.061`
- `val reward=0.415`

这个结果说明：

- 真实 MCP server 已经被成功纳入 ART 本地 RL 闭环
- 当前机器上不只是“能接 MCP”，而是已经跑完了一个最小 `MCP•RL` 训练实验
- 对 `0.6B` 小模型，简化动作空间后，真实 MCP reward 也能产生可训练信号

## 当前判断

截至 `2026-07-18`，这个 ART 实验目录已经完成了“本地链路打通 + 两类最小训练闭环验证”：

- 训练环境已可运行
- 本地推理服务已可拉起
- `PipelineTrainer` 已完成真实训练 step
- dedicated mode 下的双卡分工已被实际验证
- 一个 synthetic bitstring reward 任务已稳定完成
- 一个 tool-use style reward 任务已稳定完成
- 一个 real MCP server reward 任务已稳定完成

当前最主要的问题已经从“环境兼容性”转成了“任务设计与奖励设计”：

- 当前 reward variance 已经足够支撑最小训练闭环
- 但 reward 绝对值仍偏低，说明任务对模型仍不够“可学”
- tool-use style 与 real MCP 任务已经能给出正 reward，但仍属于架构验证实验，而不是效果型实验

## 下一步建议

如果要继续把这个实验升级成更有价值的 ART 验证，优先级应是：

1. 用有参考价值的真实偏好数据替换 synthetic bitstring 任务，例如数学推理偏好或格式约束数据。
2. 把 tool-use smoke 扩展成更大 scenario 集合，避免 `cycle` 下的超小样本重复训练。
3. 将当前 arithmetic MCP server 升级成更真实的 domain server，例如天气、文件系统或数据库查询。
4. 把单步 `TOOL + RESULT` 任务扩展成多步 agent 轨迹，例如“先查再算再总结”。
5. 等 smoke 任务稳定后，再迁移到 ART 官方 `MCP•RL` notebook 那种更完整的 agentic RL 任务。
