# 06. 训练原理和配置取舍

目标：解释本项目的 LoRA 监督微调为什么这样实现，以及常见训练参数应该如何取舍。操作步骤见 [03. 训练操作手册](03_train.md)。

## 1. 本项目训练的是什么

Qwen3-4B Instruct 是因果语言模型。监督微调时，模型仍然做下一 token 预测：

```text
给定前面的 token，预测下一个 token
```

区别在于训练样本是对话格式，项目只让模型为 assistant 回复部分承担 loss。system 和 user 只是上下文，不作为要学习复述的目标。

在代码里，这件事发生在 `trainer/dataset.py`：

1. 原始样本先标准化成 `messages`。
2. tokenizer 的 chat template 把 `messages` 渲染成模型实际看到的文本。
3. 代码用“逐步渲染 + 前缀差分”找出每条消息对应的 token span。
4. assistant span 的 label 保留原 token id。
5. system/user span 的 label 设成 `-100`，PyTorch loss 会忽略这些位置。

所以模型学到的是：在当前 system 和 user 上下文下，assistant 应该如何回答。

## 2. 为什么要求最后一条消息是 assistant

监督微调必须有可学习的目标。如果一条样本最后没有 assistant 回复，模型就只有提示词，没有标准答案。

本项目的数据处理会尽量保证：

- 每条样本至少有一条 assistant 消息。
- 最后一条消息是 assistant。
- assistant 内容不为空。
- 截断后仍然保留至少一部分 assistant label。

如果截断后所有 label 都是 `-100`，训练代码会报错，而不是默默训练一条没有监督信号的样本。

## 3. chat template 的作用

模型训练和推理都不应该直接拼普通字符串，而应使用 tokenizer 的 chat template。它会把对话转成模型熟悉的格式，例如：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

如果模型目录自带 chat template，项目优先使用模型自带模板。如果缺失，`model/tokenizer.py` 会补一个兼容 Qwen 风格的默认模板。

这保证了训练时和推理时看到的格式一致。格式不一致时，常见现象是 loss 能降，但真实推理效果很差。

## 4. LoRA 在改什么

LoRA 不直接更新基座模型的大权重矩阵，而是在指定线性层旁边加一组低秩增量：

```text
W' = W + scale * B * A
scale = alpha / r
```

其中：

- `W` 是冻结的原始权重。
- `A` 和 `B` 是可训练的小矩阵。
- `r` 是低秩维度，越大容量越强，但显存和文件也更大。
- `alpha` 控制 LoRA 增量的缩放，常见起点是 `2 * r`。
- `dropout` 给 LoRA 分支加随机丢弃，小数据时可帮助抑制过拟合。

本项目默认只注入 attention 四件套：

```yaml
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
```

这是稳妥起点：参数少、显存压力低、对指令跟随和表达风格通常已经有效。若任务需要更强的知识重排或领域表达，可再加入：

```yaml
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

加入 MLP 三件套会增加训练参数和显存，也更容易过拟合，需要配合验证集观察。

## 5. adapter、checkpoint 和合并模型的区别

adapter：

- 只包含 LoRA 增量权重。
- 文件小，适合保存多个任务版本。
- 推理时必须同时加载基座模型。
- 可以继续训练。

checkpoint：

- 是训练过程中的断点。
- 除 adapter 状态外，还可能包含 optimizer、scheduler、Trainer state。
- 适合训练中断后无缝恢复。

合并模型：

- 把 LoRA 增量合回基座模型，导出普通 Transformers 模型目录。
- 部署更简单，但文件大。
- 合并后通常不再作为 LoRA 继续训练的首选来源。

简单判断：

| 需求 | 使用 |
| --- | --- |
| 中断后接着同一次实验跑 | `RESUME_FROM_CHECKPOINT` |
| 用旧 adapter 做新一轮训练 | `ADAPTER_PATH` |
| 交付给不支持 PEFT 的部署环境 | 合并模型 |
| 保留多个任务版本 | adapter |

## 6. batch size 和梯度累积

有效 batch size：

```text
per_device_train_batch_size * gradient_accumulation_steps * GPU 数
```

默认单卡：

```text
1 * 8 * 1 = 8
```

`per_device_train_batch_size` 决定每次前向/反向实际放进显存的样本数。`gradient_accumulation_steps` 决定累计多少次小 batch 后再更新一次参数。

所以显存不足时，不应先加大 `per_device_train_batch_size`。更稳妥的方式是保持它为 1，通过 `gradient_accumulation_steps` 调整有效 batch。

经验取舍：

| 现象 | 可尝试 |
| --- | --- |
| loss 波动很大 | 增大 `gradient_accumulation_steps` |
| 训练太慢 | 在显存允许时增大 `per_device_train_batch_size` |
| 小数据过拟合 | 降低 epoch 或学习率，不一定要增大 batch |

## 7. max_length 为什么影响很大

`max_length` 是单条样本最多保留的 token 数。它同时影响：

- 显存占用。
- 训练速度。
- 能否保留完整问题和答案。
- 长样本是否会被截断。

本项目 tokenizer 默认右截断。如果右截断导致所有 assistant label 都被裁掉，代码会回退到保留尾部 token，尽量保住答案部分。但这只是兜底，不应依赖它处理大量超长样本。

建议：

- 冒烟训练用 `512`。
- 普通指令数据可从 `1024` 或 `2048` 开始。
- 长上下文任务先统计长度分布，再决定是否提高到 `4096`。
- 如果大量样本超过 `max_length`，优先清洗或拆分数据。

## 8. 学习率、warmup 和 scheduler

LoRA 只训练少量参数，学习率通常可以高于全量微调。本项目默认：

```yaml
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.03
max_grad_norm: 1.0
```

含义：

- `learning_rate` 控制每次参数更新幅度。
- `warmup_ratio` 让学习率在训练初期逐步升高，降低一开始发散的概率。
- `cosine` 会在训练后期逐步降低学习率。
- `max_grad_norm` 做梯度裁剪，限制异常梯度尖峰。

调参建议：

| 现象 | 优先调整 |
| --- | --- |
| loss 变 `nan` 或发散 | 降到 `1.0e-4` 或 `5.0e-5` |
| loss 下降很慢 | 保持数据没问题后，再尝试 `2.0e-4` |
| 小数据记忆过强 | 降低学习率和 epoch |
| 训练初期不稳定 | 增大 `warmup_ratio` 到 `0.05` |

## 9. bf16、fp16 和 gradient checkpointing

`bf16` 和 `fp16` 都是混合精度训练，目的是省显存和提速。

优先级：

1. 支持 bf16 的 GPU 用 `bf16: true`、`fp16: false`。
2. 不支持 bf16 的老 GPU 尝试 `bf16: false`、`fp16: true`。
3. 如果半精度不稳定，再退回 fp32，但显存压力会明显增加。

`gradient_checkpointing: true` 会减少显存占用，代价是反向传播时重算部分激活，训练会慢一些。

训练时项目会关闭 `use_cache`。KV cache 对推理有用，但训练时会占额外显存，并且和 gradient checkpointing 不适合同时使用。

## 10. eval_loss 怎么看

训练 loss 只说明模型越来越贴合训练集。验证集的 `eval_loss` 才能帮助判断是否泛化。

常见模式：

| train loss | eval loss | 解释 |
| --- | --- | --- |
| 下降 | 下降 | 正常学习 |
| 下降 | 上升 | 可能过拟合 |
| 不降 | 不降 | 数据、学习率或 LoRA 容量可能有问题 |
| 很低 | 很高 | 训练集和验证集分布差异大，或训练集被记忆 |

没有验证集时，至少固定一组人工 prompt，每次训练后用相同推理参数比较输出。

## 11. 推荐的实验顺序

1. 固定数据预处理逻辑，确保样本能稳定转成 `messages`。
2. 用小步数冒烟训练，先验证链路和显存。
3. 用少量真实数据训练几十步，确认 loss 会下降。
4. 加验证集，开始记录 `eval_loss`。
5. 调整学习率、epoch、`max_length`。
6. 最后再调整 LoRA `r` 和 `target_modules`。

这个顺序的好处是每次只改变一个主要变量。否则模型效果变好或变差时，很难判断是数据、长度、学习率、LoRA 容量还是训练步数造成的。
