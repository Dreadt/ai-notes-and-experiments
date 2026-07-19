# ART 架构实验报告

日期：`2026-07-19`

## 1. 实验目的

本次实验的目标不是直接做出高质量 agent，而是回答下面四个更基础的问题：

- `ART` 能不能在本地双卡机器上稳定跑起来
- `register -> rollout -> train -> eval` 这条强化学习链路能不能闭环
- `ART` 能不能从 synthetic reward 逐步推进到真实工具调用，再推进到真实 `MCP-RL`
- 在当前 `2 x RTX 3090 24GB` 环境下，`0.6B / 1.7B / 4B` 三档模型分别能做到什么程度

## 2. 实验环境

- 框架：`openpipe-art==0.5.18`
- Python 环境：`/root/autodl-tmp/ai-notes-and-experiments/ART/.venv_sys`
- 基础训练方式：`PipelineTrainer`
- backend：`art.local.LocalBackend`
- 推理后端：`vLLM`
- 机器：`2 x RTX 3090 24GB`

GPU 采用 dedicated mode：

- `trainer_gpu_ids=[0]`
- `inference_gpu_ids=[1]`

原因很直接：

- 训练和 rollout 同时进行时，如果不拆卡，训练显存和推理显存会互相挤压
- `ART + LocalBackend` 这类本地 agent RL 任务更适合把 trainer 和 inference 明确分开

## 3. 实验设计

实验按难度分三层推进：

1. `local_smoke.py`
   - 纯 synthetic reward
   - 验证最小 RL 闭环

2. `local_tool_smoke.py`
   - tool-use 风格任务
   - 还没有接真实 MCP，但已经进入“模型输出动作，环境按工具语义打分”的阶段

3. `local_mcp_rl.py`
   - 接入真实本地 MCP server
   - reward 基于真实 `list_tools()` / `call_tool()`

对应文件：

- [art_experiment/local_smoke.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_smoke.py)
- [art_experiment/local_tool_smoke.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_tool_smoke.py)
- [art_experiment/local_mcp_rl.py](/root/autodl-tmp/ai-notes-and-experiments/ART/art_experiment/local_mcp_rl.py)
- [mcp_servers/arithmetic_server.py](/root/autodl-tmp/ai-notes-and-experiments/ART/mcp_servers/arithmetic_server.py)

这个设计的意义是：

- 先分离“环境能不能跑”和“任务能不能学”
- 再分离“本地 reward 能不能训”和“真实 MCP 接入会不会出问题”
- 最后才讨论模型规模和训练长度

## 4. 实验过程

### 4.1 环境与导入验证

先完成最基础检查：

- `art` 能否导入
- `LocalBackend` 能否导入
- `PipelineTrainer` 能否导入
- dedicated mode 能否正常构造

对应脚本：

- [scripts/00_setup_venv_sys.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/00_setup_venv_sys.sh)
- [scripts/10_verify_imports.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/10_verify_imports.sh)

结果：

- 基础环境搭建成功
- ART 本地 backend 可用
- 双卡 dedicated mode 可用

### 4.2 synthetic reward 阶段

最初的 `local_smoke.py` 任务虽然能启动，但很快暴露出一个典型问题：

- `MODEL COLLAPSE DETECTED`

根因不是框架坏了，而是任务太简单，reward variance 太低，导致大量轨迹几乎同分。

后续做了两类改动：

- scenario 变得更多样
- reward 从单点判断改成分层评分

结果：

- `local_smoke.py` 可以稳定完成真实训练与评估
- `0_var` 丢弃下降
- 说明 ART 主训练链路是通的

这一阶段的核心结论是：

- ART 对“能否学起来”比对“能否启动”更敏感
- reward 设计不好，框架能跑也没用

### 4.3 tool-use 风格阶段

在 `local_smoke.py` 稳定后，加入 `local_tool_smoke.py`。

任务形式是：

- 给定工具名和参数
- 模型输出带工具语义的动作
- 环境按解析成功、工具是否正确、结果是否正确来给 reward

早期问题：

- 输出格式要求太严
- 小模型既要学格式，又要学动作，又要学结果

后续调整：

- 放宽动作结构
- 提高 reward 的可学习性

结果：

- `local_tool_smoke.py` 完成训练与评估
- `train reward` 可以转正
- `val reward` 有明显提升

这一阶段说明：

- ART 不是只能做 synthetic reward
- 一旦进入 tool-use 任务，动作空间设计会迅速成为主瓶颈

### 4.4 真实 MCP-RL 阶段

最终实验是 `local_mcp_rl.py`。

与前两阶段相比，关键变化是：

- reward 不再来自本地硬编码逻辑
- 而是通过 MCP `stdio` transport 启动本地 server
- 运行时真实调用 `list_tools()`
- 训练时真实调用 `call_tool()`

这一步才是严格意义上的最小 `MCP-RL`。

MCP server 使用：

- [arithmetic_server.py](/root/autodl-tmp/ai-notes-and-experiments/ART/mcp_servers/arithmetic_server.py)

它提供了简单算术工具：

- `add`
- `sub`
- `mul`
- `max2`

模型要学到的能力不是“背答案”，而是：

- 识别该调用哪个工具
- 输出能被环境解析的动作
- 利用真实工具返回值完成任务

## 5. 关键工程问题与修复

### 5.1 动作空间过大

最早的设计要求模型同时输出：

- 工具名
- 参数复述
- 结果

这种 `TOOL + A + B + RESULT` 设计对小模型负担太大。

后续收缩为：

- `TOOL + RESULT`

效果明显更稳定。

结论：

- 小模型做 agent RL 时，优先保证 reward 可训练，再考虑动作格式精细化

### 5.2 1.7B 首次失败不是显存问题

`1.7B` 最早失败时，根因不是 OOM，而是模型路径解析到了不完整的 Hugging Face `manual` snapshot。

后续修改了模型解析逻辑：

- 同时检查 Hugging Face cache 和 ModelScope cache
- 只接受包含完整权重文件的 snapshot

修复后：

- `1.7B` 可以稳定完成 `MCP-RL` 最小闭环

### 5.3 4B 启动慢的主要原因

`4B` 最容易让人误判成“卡住”，但实际主要耗时在：

- `vLLM` 的 `torch.compile`
- `CUDA graph capture`

也就是说：

- `4B` 的额外时间主要花在推理后端初始化
- 不是训练逻辑本身坏了

## 6. 实验结果

### 6.1 最小 MCP-RL 闭环结果

#### `Qwen/Qwen3-1.7B`

- 启动脚本：
  [32_run_local_mcp_rl_1p7b.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/32_run_local_mcp_rl_1p7b.sh)
- 结果目录：
  [qwen3-1p7b-local-mcp-rl-20260719-150631](/root/autodl-tmp/ai-notes-and-experiments/ART/.art_local/art-local-mcp-rl/models/qwen3-1p7b-local-mcp-rl-20260719-150631)
- 结果：
  - `training_step=1`
  - `train reward=0.325`
  - `val reward=0.415`
  - `step_seconds=26.43`

#### `Qwen/Qwen3-4B`

- 启动脚本：
  [33_run_local_mcp_rl_4b.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/33_run_local_mcp_rl_4b.sh)
- 结果目录：
  [qwen3-4b-local-mcp-rl-20260719-150901](/root/autodl-tmp/ai-notes-and-experiments/ART/.art_local/art-local-mcp-rl/models/qwen3-4b-local-mcp-rl-20260719-150901)
- 结果：
  - `training_step=1`
  - `train reward=0.300`
  - `val reward=0.530`
  - `step_seconds=31.72`

### 6.2 4B 长线训练结果

在最小闭环成功后，又进一步跑了长线 `4B` 版本：

- 启动脚本：
  [34_run_local_mcp_rl_4b_long.sh](/root/autodl-tmp/ai-notes-and-experiments/ART/scripts/34_run_local_mcp_rl_4b_long.sh)
- 结果目录：
  [qwen3-4b-long-mcp-rl-20260719-152735](/root/autodl-tmp/ai-notes-and-experiments/ART/.art_local/art-local-mcp-rl/models/qwen3-4b-long-mcp-rl-20260719-152735)

长线配置：

- `steps=4`
- `rollouts_per_scenario=4`
- `max_tokens=64`
- `min_batch_size=2`

最终结果：

- `training_step=4`
- `completed_eval_steps=[1,2,3,4]`
- `train reward=0.264`
- `val reward=0.540`
- `step_seconds=26.83`
- `discarded total=42`
- `discarded 0_var=42`
- `num_scenarios=19`
- `num_trajectories=76`
- `trainer_tokens=3840`
- `num_gradient_steps=60`
- `time/cum/wall_s=96.38`

从这个结果看：

- 长线训练是稳定跑完的
- `val reward` 维持在比单步实验更高的位置
- 但 `0_var` 丢弃仍然明显，说明任务多样性和 reward 结构还有继续优化空间

## 7. 当前结论

### 7.1 ART 已经在本地双卡环境完成闭环验证

截至 `2026-07-19`，当前目录下已经完成：

- `local_smoke.py`
- `local_tool_smoke.py`
- `local_mcp_rl.py`
- `Qwen3-1.7B` 最小 `MCP-RL`
- `Qwen3-4B` 最小 `MCP-RL`
- `Qwen3-4B` 长线 `MCP-RL`

因此可以明确下结论：

- ART 架构实验成功
- 本地双卡 dedicated mode 成功
- 真实 MCP server 接入成功
- `MCP-RL` 最小闭环已经在 `1.7B` 和 `4B` 上实测通过

### 7.2 ART 的定位不是“只做 MCP-RL”

从这次实验看得很清楚：

- `local_smoke.py` 是 synthetic reward
- `local_tool_smoke.py` 是 tool-use style reward
- `local_mcp_rl.py` 才是 MCP-RL

所以更准确的结论是：

- ART 是面向 agent / environment feedback 的通用 RL 框架
- `MCP-RL` 是它非常适合的一类任务，但不是它唯一的用途

### 7.3 真正的瓶颈不是“能不能跑”，而是“任务能不能学”

当前阶段最关键的经验不是框架安装，而是：

- reward variance 要足够
- 动作空间不能一开始就设计得过大
- 小模型对格式负担非常敏感
- 大模型虽然能跑更远，但初始化和 rollout 成本明显更高

## 8. 后续建议

如果继续推进，优先级建议如下：

1. 先扩展 scenario 多样性  
2. 再优化 reward shaping，减少 `0_var` 丢弃  
3. 再继续拉长 `4B` 训练步数  
4. 最后再考虑更复杂工具集、多轮 agent 任务或更大模型  

原因很明确：

- 现在已经不是“ART 能不能跑”的问题
- 而是“当前 MCP 任务设计，是否足够让模型持续学到东西”的问题

## 9. 一句话总结

截至 `2026-07-19`，`ART + 本地双 3090 + 真实 MCP server` 这条链路已经完整跑通，并且已经在 `Qwen3-1.7B` 和 `Qwen3-4B` 上完成最小 `MCP-RL`，同时完成了一轮 `4B` 长线训练验证。当前的主要工作重点，应从“能否运行”转向“如何提高任务可学性和训练效率”。
