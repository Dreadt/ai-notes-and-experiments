# Hugging Face 文件格式说明

## 1. 这份说明解决什么问题

在 Hugging Face 训练和推理流程里，我们经常会看到一个“模型目录”，比如：

- [outputs/exp-stage1](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage1)
- [outputs/exp-stage2](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage2)

这类目录可以被：

```python
AutoTokenizer.from_pretrained(path)
AutoModelForCausalLM.from_pretrained(path)
```

直接加载。

所以学习 Hugging Face 文件格式，本质上是理解：

1. 一个可加载模型目录通常有哪些文件
2. 每个文件分别负责什么
3. 为什么 `from_pretrained()` 只给一个目录就能工作


## 2. 我们这次实验中的 Hugging Face 模型目录

以：

- [outputs/exp-stage2](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage2)

为例，它是本次训练后保存下来的一个标准 Hugging Face 模型目录。

常见文件包括：

- `config.json`
- `model.safetensors`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`


## 3. 主要文件的作用

### 3.1 `config.json`

作用：

- 定义模型结构配置
- 告诉 Transformers 这是什么模型、层数多少、hidden size 多大等

你可以理解为：

- 这是“模型骨架说明书”

没有它，框架不知道该用什么结构去承载权重。


### 3.2 `model.safetensors`

作用：

- 保存模型参数
- 是真正的权重文件

你可以理解为：

- 这是“训练出来的数值本体”

为什么现在更常见是 `safetensors` 而不是 `pytorch_model.bin`：

- 更安全
- 不依赖 Python pickle
- 加载速度和兼容性更好


### 3.3 `generation_config.json`

作用：

- 保存生成相关配置
- 例如 `eos_token_id`、采样行为默认参数等

它不是训练必须文件，但在推理阶段很常见。


### 3.4 `tokenizer.json`

作用：

- 保存 tokenizer 的核心数据
- 包括词表、分词规则、编码逻辑

它是 tokenizer 的主要内容文件。


### 3.5 `tokenizer_config.json`

作用：

- 保存 tokenizer 的附加配置
- 包括一些特殊 token 设置、预处理行为等

它和 `tokenizer.json` 通常一起使用。


### 3.6 `chat_template.jinja`

作用：

- 保存 chat 模板
- 用于把结构化消息：

```python
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

转换成模型实际训练/推理时看到的文本

这在 Qwen 这种 chat 模型上非常重要。


## 4. 文件之间是如何配合工作的

当我们执行：

```python
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
```

内部实际上在做两件事：

### 4.1 先根据 `config.json` 构造模型结构

比如：

- 多少层
- hidden size
- attention 配置

### 4.2 再把 `model.safetensors` 中的参数加载进去

也就是：

- 先造壳子
- 再往壳子里灌权重

对于 tokenizer 也是类似：

- 读取 `tokenizer.json`
- 加载 `tokenizer_config.json`
- 如果有 chat 模板，再读取 `chat_template.jinja`


## 5. Hugging Face 缓存目录和训练输出目录的区别

这次实验里，我们同时接触到了两类目录。

### 5.1 原始模型缓存目录

例如：

- [/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B](/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B)

这是 Hugging Face Hub 下载后的本地缓存。

特点：

- 是底座模型的原始缓存
- 可能包含 `snapshots/`、`blobs/`、`refs/`
- 主要用于重复加载时不必重新下载

### 5.2 训练输出目录

例如：

- [outputs/exp-stage1](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage1)
- [outputs/exp-stage2](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage2)

这是我们训练后通过：

```python
save_pretrained(...)
```

保存出来的目录。

特点：

- 直接面向下游加载
- 更像“训练结果包”


## 6. checkpoint 目录是什么

这次实验中还有一类目录：

- [outputs/exp-stage3-full-lr2e5/checkpoint-1050](/root/autodl-tmp/ai-notes-and-experiments/accelerator&wandb/outputs/exp-stage3-full-lr2e5/checkpoint-1050)

这种目录的含义是：

- 训练到某个 step 时保存的中间模型

它通常包含和最终模型目录类似的文件：

- `config.json`
- `model.safetensors`
- tokenizer 文件

区别是：

- 它不是最终结果
- 而是中途快照


## 7. 为什么一个 checkpoint 不能天然等于“可续训”

这是这次实验中我们踩到的一个关键点。

只保存：

- `config.json`
- `model.safetensors`
- tokenizer 文件

只能说明：

- 你能从这个点重新加载模型权重继续 fine-tune

但不意味着你能真正“恢复训练状态”。

真正完整的断点恢复还需要保存：

- optimizer state
- scheduler state
- 随机数状态
- 分布式训练状态

因此我们后来在训练脚本中补了：

```python
accelerator.save_state(...)
```


## 8. 常见目录可以怎么理解

可以把 Hugging Face 相关目录分成 3 层：

### 8.1 Hub 缓存层

比如：

- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`

作用：

- 保存下载下来的原始模型

### 8.2 训练产物层

比如：

- `outputs/exp-stage1`
- `outputs/exp-stage2`

作用：

- 保存训练后的最终模型

### 8.3 中间快照层

比如：

- `outputs/exp-stage3-full-lr2e5/checkpoint-1050`

作用：

- 保存中间 step 的模型快照


## 9. 这部分最应该记住什么

如果只记 4 点，最重要的是：

1. `config.json` 决定模型结构
2. `model.safetensors` 保存模型参数
3. tokenizer 文件决定文本如何变成 token
4. “模型权重目录”不等于“完整可恢复训练状态目录”


## 10. 一句话总结

Hugging Face 文件格式的核心思想是：

- 用一个目录，把“模型结构 + 参数 + tokenizer + 生成配置”打包起来  
- 从而让 `from_pretrained()` 和 `save_pretrained()` 成为统一的加载/保存接口
