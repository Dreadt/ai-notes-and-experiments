# SFT 数据格式与 Loss 公式详解

## 1. 这份说明解决什么问题

做 SFT 时最常见的两个问题是：

1. 数据到底应该长什么样
2. loss 到底在算什么

本次实验使用：

- 模型：`Qwen/Qwen3-0.6B`
- 任务：监督微调（SFT）
- 数据集：`shibing624/alpaca-zh`

因此这份说明会结合我们这次的实际脚本来讲：

- SFT 数据格式
- assistant-only supervision
- `labels = -100` 的含义
- 交叉熵 loss 的公式与解释


## 2. 我们这次使用的数据格式

本次数据集字段是：

- `instruction`
- `input`
- `output`

典型样本例如：

```json
{
  "instruction": "保持健康的三个提示。",
  "input": "",
  "output": "以下是保持健康的三个提示：..."
}
```

这是一种很常见的单轮指令数据格式。


## 3. 为什么这种格式不能直接喂给模型

模型并不直接理解 JSON 字段名。  
模型实际看到的是一串 token。

所以在进入模型之前，需要完成两步转换：

### 3.1 先把结构化字段转成对话消息

我们在脚本中会构造成：

```python
[
  {"role": "user", "content": instruction_or_instruction_plus_input},
  {"role": "assistant", "content": output}
]
```

### 3.2 再通过 chat template 转成文本

例如大致会变成：

```text
<user>
保持健康的三个提示。
<assistant>
以下是保持健康的三个提示：...
```

然后再被 tokenizer 编码成 token id。


## 4. SFT 数据在训练时真正长什么样

训练时最核心的三个张量是：

- `input_ids`
- `attention_mask`
- `labels`

### 4.1 `input_ids`

含义：

- 模型真正看到的 token 序列

### 4.2 `attention_mask`

含义：

- 哪些位置是真实 token
- 哪些位置是 padding

### 4.3 `labels`

含义：

- 哪些 token 位置要参与监督
- 哪些位置不参与监督


## 5. 为什么 SFT 不是“所有 token 都算 loss”

如果把整句都监督，模型就会同时学习：

- 用户提问部分
- assistant 回答部分

这通常不是我们想要的。

更常见的目标是：

- 只让模型学习 assistant 的回答
- 不要求模型去“预测用户输入”

因此本次脚本采用的是：

- prompt 部分全部 mask 掉
- 只对 assistant 回答部分计算 loss


## 6. `labels = -100` 是什么意思

在我们的训练脚本里，`IGNORE_INDEX = -100`。

作用是：

- 把不需要监督的位置设成 `-100`
- PyTorch 的交叉熵会自动忽略这些位置

也就是说：

- `labels == -100` 的 token 不参与 loss
- `labels != -100` 的 token 才参与 loss

这就是为什么它能实现：

- 只监督 assistant answer
- 忽略 prompt
- 忽略 padding


## 7. 本次脚本如何构造 assistant-only supervision

本次脚本的大致逻辑是：

1. 构造完整对话 `messages`
2. 构造只包含 prompt 的 `prompt_messages`
3. 分别做 chat template
4. 计算 prompt 长度
5. 把 prompt 对应位置的 `labels` 设成 `-100`

也就是：

```python
labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
```

这句代码的含义非常重要：

- prompt 长度之前：不学
- prompt 之后：开始学 assistant 输出


## 8. Loss Function 的核心形式

SFT 对因果语言模型的本质，仍然是 next-token prediction。

标准形式可以写成：

\[
\mathcal{L} = - \frac{1}{N}\sum_{t \in M}\log p_\theta(y_t \mid x, y_{<t})
\]

其中：

- \(\mathcal{L}\)：整体 loss
- \(N\)：有效监督 token 数量
- \(M\)：参与监督的位置集合
- \(y_t\)：第 \(t\) 个正确目标 token
- \(p_\theta(y_t \mid x, y_{<t})\)：模型给正确 token 的概率


## 9. 这个公式怎么理解

可以拆成几层理解。

### 9.1 `p(...)`

表示：

- 模型认为“下一个 token 是正确答案 token”的概率

例如：

- 正确 token 是“健康”
- 模型给它的概率是 `0.8`

那说明模型预测得比较好。

### 9.2 `log`

对概率取对数，是为了把很多 token 的概率乘积变成求和，更方便优化。

### 9.3 `-log`

前面的负号表示：

- 正确 token 的概率越高，loss 越小
- 正确 token 的概率越低，loss 越大

例如：

- 如果模型给正确 token 概率是 `0.9`
  - `-log(0.9)` 很小
- 如果模型给正确 token 概率是 `0.01`
  - `-log(0.01)` 很大

所以它天然符合“预测越准，惩罚越小”的目标。


## 10. `-log` 为什么有意义

直觉上可以这样理解：

- 模型越确信正确答案，罚得越轻
- 模型越不相信正确答案，罚得越重

这让训练的方向非常明确：

- 提高正确 token 的概率
- 降低错误 token 的相对概率


## 11. `N` 和 `M` 为什么重要

公式里不是对所有 token 求和，而是对：

\[
t \in M
\]

也就是只对参与监督的位置求和。

在本次实验里，这些位置主要是：

- assistant 回复 token

不包括：

- user prompt
- padding

因此：

- `M` 决定了“哪些地方要学”
- `-100` 的设计本质上就是在定义 `M`


## 12. 为什么 causal LM 的 labels 不需要手动左移

很多初学者容易误以为要自己手动 shift labels。

但在 Hugging Face 的 `AutoModelForCausalLM` 中，常见做法是：

- `input_ids` 和 `labels` 长度一致
- shift 通常由模型内部处理

所以我们只需要关心：

- 哪些位置监督
- 哪些位置忽略

而不是自己手写序列平移逻辑。


## 13. 这次实验里 loss 学到的到底是什么

本次实验中，模型学习的是：

- 在给定 user 指令和上下文的条件下
- 生成符合训练数据风格的 assistant 回答

换句话说：

- 它学的是“回答方式”
- 而不是重新预训练整个语言系统

这也是为什么 SFT 数据质量非常重要。


## 14. SFT 数据格式的常见几种形态

除了我们这次用的 `instruction/input/output`，还常见：

### 14.1 直接文本格式

```json
{"text": "<user>...\n<assistant>..."}
```

优点：

- 简单

缺点：

- 不方便做精细 mask

### 14.2 多轮对话格式

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

优点：

- 更适合 chat 模型

缺点：

- 预处理稍复杂

### 14.3 指令三元组格式

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

这就是我们这次使用的格式。

优点：

- 清晰
- 易转为单轮 chat
- 适合初学 SFT


## 15. 结合本次实验，最应该记住什么

如果只记住最核心的 5 点：

1. SFT 本质上还是 causal LM 的 next-token prediction
2. 数据不会直接以 JSON 给模型，必须先变成文本再 token 化
3. `labels=-100` 的位置不会参与 loss
4. assistant-only supervision 是更常见的 SFT 做法
5. loss 的核心就是让正确 token 概率更高


## 16. 一句话总结

SFT 数据格式决定“模型看到了什么”，  
`labels` 决定“模型该学什么”，  
交叉熵 loss 决定“模型如何朝正确答案逼近”。
