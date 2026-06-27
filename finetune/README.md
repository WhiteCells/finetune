# Qwen3-4B LoRA 微调完整教程

本文档配套当前目录下的 LoRA 微调工程，目标是在本地
`/home/cells/dev/model/qwen3-4B/models/Qwen3-4B-Instruct-2507`
基座模型上完成监督微调、断点续训、TensorBoard 或 wandb 记录、LoRA 推理，以及
LoRA 权重合并。

工程默认面向 Python 3.11、PyTorch 2.6+、Transformers、PEFT、Accelerate
和可选 DeepSpeed。所有训练代码会按模块拆分，避免把加载模型、数据集、collator、
trainer、保存逻辑全部塞进 `train.py`。

## 1. LoRA 原理介绍

LoRA，全称 Low-Rank Adaptation，是一种参数高效微调方法。它的核心思想是：
在大模型原始权重保持冻结的前提下，只训练一组低秩矩阵，用很少的新增参数近似完整
权重更新。

对一个线性层来说，原始前向计算可以写成：

$$
y = Wx
$$

完整微调会直接更新 `W`。LoRA 不直接改动 `W`，而是引入一个低秩增量：

$$
y = Wx + \mathrm{scale} \cdot BAx
$$

其中：

- `W`：原始预训练权重，训练时冻结。
- `A`：下投影矩阵，形状通常为 `[r, in_features]`。
- `B`：上投影矩阵，形状通常为 `[out_features, r]`。
- `r`：LoRA rank，控制低秩空间大小。
- `scale = alpha / r`：缩放系数，控制 LoRA 更新强度。
- `dropout`：可选，用于降低过拟合。

LoRA 的优势：

- 显存占用低：只训练少量 LoRA 参数，优化器状态也更小。
- 训练速度快：反向传播主要发生在 adapter 参数上。
- 易于多任务管理：每个任务保存一个 adapter，不需要复制完整基座模型。
- 可合并部署：训练完成后可将 LoRA 增量合并进基座权重，导出完整模型。

LoRA 的限制：

- 它不会像全参微调那样拥有最高自由度。
- 如果数据质量差，LoRA 同样会学习到错误模式。
- `r`、`alpha`、目标模块等参数不合理时，可能出现欠拟合或过拟合。

## 2. Qwen3-4B 模型结构简介

当前工程默认使用本仓库中的 Qwen3-4B Instruct 模型：

```text
models/Qwen3-4B-Instruct-2507
```

根据本地 `config.json`，模型关键结构如下：

| 配置项 | 当前值 | 说明 |
| --- | ---: | --- |
| `architectures` | `Qwen3ForCausalLM` | 因果语言模型结构 |
| `model_type` | `qwen3` | Transformers 识别的模型类型 |
| `num_hidden_layers` | `36` | Transformer decoder 层数 |
| `hidden_size` | `2560` | 隐层维度 |
| `intermediate_size` | `9728` | MLP 中间层维度 |
| `num_attention_heads` | `32` | Query attention heads |
| `num_key_value_heads` | `8` | KV heads，使用 GQA |
| `head_dim` | `128` | 单个 attention head 的维度 |
| `vocab_size` | `151936` | 词表大小 |
| `max_position_embeddings` | `262144` | 最大位置长度配置 |
| `rope_theta` | `5000000` | RoPE 位置编码参数 |
| `torch_dtype` | `bfloat16` | 模型建议 dtype |
| `tie_word_embeddings` | `true` | 输入输出 embedding 共享 |

Qwen3-4B 属于 decoder-only causal language model。训练时给定一段 token 序列，
模型预测下一个 token。指令微调时，一般只对 assistant 回复部分计算 loss，
用户指令、系统提示和上下文部分会被 mask 成 `-100`，这样模型学习的是“如何回答”，
而不是机械复读 prompt。

常见 LoRA 注入模块包括：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

如果显存紧张，可以只注入 attention 投影层：

```text
q_proj, k_proj, v_proj, o_proj
```

如果希望效果更强，可以同时注入 MLP：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## 3. PEFT 工作流程

PEFT 是 Hugging Face 提供的参数高效微调库。LoRA 是 PEFT 支持的核心方法之一。

本工程的 PEFT 工作流程如下：

1. 加载 tokenizer。
2. 加载 Qwen3-4B 基座模型。
3. 根据 `config/lora.yaml` 构造 `LoraConfig`。
4. 调用 `get_peft_model()` 将 LoRA adapter 注入目标模块。
5. 冻结基座模型权重，只训练 LoRA 参数。
6. 构建训练数据集和 data collator。
7. 使用 Transformers `Trainer` 训练。
8. 保存 LoRA adapter checkpoint。
9. 推理时加载基座模型和 LoRA adapter。
10. 如需完整模型，调用 `merge_and_unload()` 合并权重。

LoRA adapter 保存目录通常只包含 adapter 配置和 adapter 权重，例如：

```text
outputs/qwen3-4b-lora/checkpoint-1000/
├── adapter_config.json
├── adapter_model.safetensors
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
└── training_args.bin
```

`adapter_model.safetensors` 不是完整模型，它必须和基座模型配合使用。

## 4. 数据集格式说明

本工程支持三种输入格式：

### 4.1 Alpaca 格式

适合单轮指令微调。

```json
{
  "instruction": "请解释什么是 LoRA。",
  "input": "",
  "output": "LoRA 是一种参数高效微调方法..."
}
```

字段说明：

- `instruction`：用户任务描述，必填。
- `input`：任务额外输入，可为空字符串。
- `output`：期望 assistant 输出，必填。

### 4.2 ShareGPT 格式

适合多轮对话数据。

```json
{
  "conversations": [
    {"from": "human", "value": "你是谁？"},
    {"from": "gpt", "value": "我是一个 AI 助手。"},
    {"from": "human", "value": "你能做什么？"},
    {"from": "gpt", "value": "我可以回答问题、编写代码和整理资料。"}
  ]
}
```

也支持常见的 `messages` 格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业、谨慎的助手。"},
    {"role": "user", "content": "介绍一下杭州。"},
    {"role": "assistant", "content": "杭州是浙江省省会..."}
  ]
}
```

### 4.3 自定义 JSONL 格式

如果你的数据不是标准 Alpaca 或 ShareGPT，可以在 `data/preprocess.py`
中将它转换为统一的 `messages` 格式：

```json
{
  "messages": [
    {"role": "user", "content": "原始问题"},
    {"role": "assistant", "content": "目标答案"}
  ]
}
```

最终训练阶段推荐统一使用一行一个 JSON 对象的 JSONL 文件。

## 5. JSONL 示例

`data/example.jsonl` 会提供可直接跑通流程的示例数据。示例内容类似：

```jsonl
{"instruction":"请用一句话解释 LoRA。","input":"","output":"LoRA 是一种通过训练低秩适配矩阵来高效微调大语言模型的方法。"}
{"instruction":"把下面的话改写得更正式。","input":"这个方案挺靠谱，可以试试。","output":"该方案具备较高可行性，建议进一步验证并尝试实施。"}
{"messages":[{"role":"system","content":"你是一个严谨的中文助手。"},{"role":"user","content":"什么是梯度累积？"},{"role":"assistant","content":"梯度累积是在多个 mini-batch 上累加梯度，再统一执行一次优化器更新的方法。"}]}
{"conversations":[{"from":"human","value":"给我三个提升数据质量的建议。"},{"from":"gpt","value":"第一，删除重复和低质量样本；第二，统一标注规范；第三，抽样人工复核。"}]}
```

JSONL 要求：

- 每行必须是合法 JSON。
- 不要在文件末尾写逗号。
- `output`、`assistant` 或 `gpt` 内容不能为空。
- 训练集中不要混入模型不应该学习的隐私信息、错误答案或不可遵守指令。

## 6. 数据预处理流程

`data/preprocess.py` 的职责是将不同原始格式转换为训练代码可统一读取的 JSONL。

推荐流程：

1. 准备原始数据，如 `raw/alpaca.json`、`raw/sharegpt.json` 或业务导出的 JSONL。
2. 运行预处理脚本。
3. 输出统一 JSONL。
4. 抽样检查 20 到 100 条数据。
5. 使用训练脚本加载处理后的 JSONL。

示例命令：

```bash
python data/preprocess.py \
  --input data/raw.jsonl \
  --output data/train.jsonl \
  --format auto
```

`--format` 支持：

- `auto`：自动识别 Alpaca、ShareGPT、messages 等格式。
- `alpaca`：强制按 `instruction/input/output` 解析。
- `sharegpt`：强制按 `conversations` 解析。
- `messages`：强制按 OpenAI 风格 `messages` 解析。

训练时，dataset 会自动拼接 prompt。对 Alpaca 格式，默认模板为：

```text
<|im_start|>system
你是一个专业、可靠、简洁的中文助手。<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

如果 `input` 为空，则不会额外插入空上下文。

## 7. 环境安装

建议使用 Python 3.11 的虚拟环境：

```bash
cd /home/cells/dev/model/qwen3-4B/finetune
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你使用 uv：

```bash
cd /home/cells/dev/model/qwen3-4B/finetune
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

检查 CUDA 与 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 8. requirements 说明

`requirements.txt` 会包含：

| 依赖 | 作用 |
| --- | --- |
| `torch>=2.6` | 深度学习框架，负责模型训练与 CUDA 计算 |
| `transformers` | 加载 Qwen3 模型、tokenizer、Trainer、训练参数 |
| `peft` | 注入、训练、保存和加载 LoRA adapter |
| `accelerate` | 设备放置、分布式训练、混合精度支持 |
| `datasets` | 可选的数据处理工具 |
| `pyyaml` | 读取 YAML 配置文件 |
| `safetensors` | 安全高效地保存模型权重 |
| `tensorboard` | 本地训练曲线可视化 |
| `wandb` | 可选的云端实验跟踪 |
| `deepspeed` | 可选的 ZeRO 优化和多卡训练 |
| `tqdm` | 命令行进度条 |
| `numpy` | 统计指标与数值处理 |

安装 DeepSpeed 时，如果你的机器没有完整 CUDA 编译环境，可能会失败。单卡 LoRA
训练可以先不启用 DeepSpeed。

## 9. 每个目录作用

最终工程结构如下：

```text
finetune/
├── README.md
├── requirements.txt
├── config/
├── data/
├── model/
├── trainer/
├── utils/
├── train.py
├── inference.py
├── merge_lora.py
└── scripts/
```

目录说明：

| 路径 | 作用 |
| --- | --- |
| `config/` | 存放 LoRA、训练和 DeepSpeed 配置 |
| `data/` | 存放示例数据和数据预处理脚本 |
| `model/` | 模型、tokenizer、LoRA adapter 加载与注入 |
| `trainer/` | Dataset、DataCollator、Trainer 和 metrics |
| `utils/` | 日志、随机种子、保存工具 |
| `scripts/` | 可直接运行的 shell 脚本 |
| `outputs/` | 训练输出目录，运行后生成 |
| `logs/` | 日志目录，运行后生成 |

## 10. 每个 Python 文件作用

| 文件 | 作用 |
| --- | --- |
| `data/preprocess.py` | 将 Alpaca、ShareGPT、messages、自定义 JSONL 转成统一 JSONL |
| `model/loader.py` | 根据配置加载 Qwen3-4B causal LM |
| `model/tokenizer.py` | 加载 tokenizer，设置 pad token 和 chat template 相关行为 |
| `model/lora.py` | 读取 LoRA YAML，构造 `LoraConfig`，注入 adapter |
| `trainer/dataset.py` | 读取 JSONL，拼接 prompt，生成 `input_ids`、`attention_mask`、`labels` |
| `trainer/collator.py` | 动态 padding batch，并把 label padding 设置为 `-100` |
| `trainer/trainer.py` | 构建 Transformers `Trainer` 和 `TrainingArguments` |
| `trainer/metrics.py` | 计算 loss 派生指标，如 perplexity |
| `utils/logger.py` | 统一日志格式 |
| `utils/seed.py` | 设置 Python、NumPy、PyTorch 随机种子 |
| `utils/save.py` | 保存 tokenizer、配置快照和 LoRA adapter |
| `train.py` | 主训练入口，只负责串联配置、模型、数据和 trainer |
| `inference.py` | 加载基座模型和 LoRA adapter 后生成文本 |
| `merge_lora.py` | 调用 `merge_and_unload()` 合并 LoRA 到完整模型 |

## 11. 每个配置项解释

### 11.1 `config/lora.yaml`

| 配置项 | 示例值 | 说明 |
| --- | --- | --- |
| `r` | `16` | LoRA 低秩维度，越大可训练容量越强，显存也越高 |
| `alpha` | `32` | LoRA 缩放参数，实际缩放通常是 `alpha / r` |
| `dropout` | `0.05` | LoRA dropout，数据少时可适当增大 |
| `bias` | `none` | 是否训练 bias，常用 `none` |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | 注入 LoRA 的模块名 |
| `task_type` | `CAUSAL_LM` | 任务类型，因果语言模型固定为 `CAUSAL_LM` |
| `inference_mode` | `false` | 训练时为 `false`，推理 adapter 配置中通常为 `true` |

### 11.2 `config/train.yaml`

| 配置项 | 示例值 | 说明 |
| --- | --- | --- |
| `model_name_or_path` | `../models/Qwen3-4B-Instruct-2507` | 基座模型路径 |
| `train_file` | `data/example.jsonl` | 训练数据 JSONL |
| `eval_file` | `null` | 验证集 JSONL，可为空 |
| `output_dir` | `outputs/qwen3-4b-lora` | checkpoint 和最终 adapter 输出目录 |
| `max_length` | `2048` | 单条样本最大 token 长度 |
| `per_device_train_batch_size` | `1` | 单卡 micro batch size |
| `per_device_eval_batch_size` | `1` | 验证 micro batch size |
| `gradient_accumulation_steps` | `8` | 梯度累积步数 |
| `learning_rate` | `2.0e-4` | LoRA 学习率 |
| `num_train_epochs` | `3` | 训练轮数 |
| `lr_scheduler_type` | `cosine` | 学习率调度器 |
| `warmup_ratio` | `0.03` | warmup 占总步数比例 |
| `logging_steps` | `10` | 日志记录间隔 |
| `save_steps` | `200` | checkpoint 保存间隔 |
| `eval_steps` | `200` | 验证间隔，需要 `eval_file` |
| `save_total_limit` | `3` | 最多保留 checkpoint 数 |
| `fp16` | `false` | 是否启用 fp16 |
| `bf16` | `true` | 是否启用 bf16，推荐 Ampere 及以上 GPU |
| `gradient_checkpointing` | `true` | 是否启用梯度检查点，节省显存但降低速度 |
| `report_to` | `["tensorboard"]` | 日志后端，可加入 `wandb` |
| `deepspeed` | `null` | DeepSpeed 配置路径，可填 `config/ds_config.json` |
| `resume_from_checkpoint` | `null` | 断点续训 checkpoint 路径 |
| `seed` | `42` | 随机种子 |

### 11.3 `config/ds_config.json`

DeepSpeed 为可选配置。LoRA 单卡训练不一定需要 DeepSpeed；多卡、显存更紧张或需要
ZeRO 优化时可以启用。

常见配置项：

| 配置项 | 说明 |
| --- | --- |
| `zero_optimization.stage` | ZeRO 阶段，LoRA 常用 stage 2 |
| `bf16.enabled` | 是否启用 bf16 |
| `fp16.enabled` | 是否启用 fp16 |
| `gradient_accumulation_steps` | 与训练配置保持一致或设为 `auto` |
| `train_micro_batch_size_per_gpu` | 单 GPU micro batch size |
| `gradient_clipping` | 梯度裁剪阈值 |

## 12. 如何开始训练

进入工程目录：

```bash
cd /home/cells/dev/model/qwen3-4B/finetune
```

安装依赖后，直接运行：

```bash
bash scripts/train.sh
```

或手动执行：

```bash
python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml
```

训练输出默认保存到：

```text
outputs/qwen3-4b-lora
```

有效 batch size 计算方式：

$$
\mathrm{effective\_batch\_size} =
\mathrm{per\_device\_train\_batch\_size}
\times
\mathrm{gradient\_accumulation\_steps}
\times
\mathrm{num\_gpus}
$$

例如单卡：

$$
1 \times 8 \times 1 = 8
$$

## 13. 如何继续训练

继续训练分两种情况。

### 13.1 从 Trainer checkpoint 继续

如果 checkpoint 中包含 optimizer、scheduler 和 trainer state，可以继续完整训练状态：

```bash
python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-lora/checkpoint-1000
```

这种方式会恢复：

- LoRA adapter 权重
- optimizer 状态
- scheduler 状态
- global step
- Trainer state

### 13.2 从已有 LoRA adapter 继续训练

如果你只有 adapter 权重，也可以把它作为初始化 adapter 继续训：

```bash
python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --adapter-path outputs/old-adapter
```

这种方式不会恢复 optimizer 和 scheduler 状态，相当于加载旧 adapter 后开始新的训练。

## 14. 如何保存 checkpoint

保存行为由 `config/train.yaml` 控制：

```yaml
save_steps: 200
save_total_limit: 3
output_dir: outputs/qwen3-4b-lora
```

表示每 200 个 update step 保存一次 checkpoint，最多保留 3 个。

每个 checkpoint 通常包含：

```text
checkpoint-200/
├── adapter_config.json
├── adapter_model.safetensors
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
├── rng_state.pth
└── training_args.bin
```

训练完成后，最终 adapter 会保存在 `output_dir` 根目录。

## 15. 如何恢复 checkpoint

恢复 checkpoint 推荐使用命令行参数：

```bash
python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-lora/checkpoint-200
```

也可以在 `config/train.yaml` 中配置：

```yaml
resume_from_checkpoint: outputs/qwen3-4b-lora/checkpoint-200
```

命令行参数优先级高于 YAML。

注意：

- checkpoint 必须和当前基座模型兼容。
- 如果更改了 `target_modules` 或 `r`，旧 adapter 可能无法加载。
- 如果只想推理，不需要恢复 optimizer 状态，直接使用 `inference.py --adapter-path`。

## 16. TensorBoard 使用

训练配置中默认启用：

```yaml
report_to:
  - tensorboard
logging_dir: logs/qwen3-4b-lora
```

启动 TensorBoard：

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

常看指标：

- `train/loss`：训练 loss。
- `eval/loss`：验证 loss。
- `eval/perplexity`：困惑度，越低通常越好。
- `train/learning_rate`：学习率曲线。
- `train/grad_norm`：梯度范数，异常尖峰可能代表学习率过高或脏数据。

如果训练 loss 快速接近 0，而验证 loss 上升，通常是过拟合。

## 17. wandb 使用（可选）

如果希望使用 Weights & Biases：

```bash
wandb login
```

在 `config/train.yaml` 中设置：

```yaml
report_to:
  - tensorboard
  - wandb
run_name: qwen3-4b-lora
```

也可以通过环境变量设置项目名：

```bash
export WANDB_PROJECT=qwen3-4b-lora
export WANDB_NAME=qwen3-4b-exp001
```

如果不想启用 wandb：

```bash
export WANDB_DISABLED=true
```

## 18. 推理方法

LoRA 推理需要同时加载：

1. Qwen3-4B 基座模型。
2. LoRA adapter。

运行脚本：

```bash
bash scripts/infer.sh
```

或手动执行：

```bash
python inference.py \
  --model-name-or-path ../models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --prompt "请用一句话解释 LoRA。" \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9
```

常见生成参数：

| 参数 | 说明 |
| --- | --- |
| `max_new_tokens` | 最多生成 token 数 |
| `temperature` | 采样温度，越高越随机 |
| `top_p` | nucleus sampling，控制采样候选范围 |
| `top_k` | top-k sampling |
| `repetition_penalty` | 重复惩罚 |
| `do_sample` | 是否采样；关闭时更接近贪心或 beam |

如果要进行稳定评测，建议降低随机性：

```bash
--temperature 0.1 --top-p 0.8
```

如果要做创意生成，可以适当提高：

```bash
--temperature 0.8 --top-p 0.95
```

## 19. Merge LoRA 权重

训练得到的 LoRA adapter 默认不是完整模型。若要导出完整模型，可运行：

```bash
bash scripts/merge.sh
```

或手动执行：

```bash
python merge_lora.py \
  --model-name-or-path ../models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --output-dir outputs/qwen3-4b-merged
```

内部流程：

1. 加载基座模型。
2. 加载 LoRA adapter。
3. 调用 PEFT 的 `merge_and_unload()`。
4. 保存合并后的完整模型和 tokenizer。

合并后的模型目录可以像普通 Transformers 模型一样加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "outputs/qwen3-4b-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
```

注意：

- 合并会生成完整模型，磁盘占用明显大于 adapter。
- 合并后的模型不再需要 PEFT adapter。
- 如果继续训练，建议使用未合并的 adapter checkpoint。

## 20. 常见报错及解决方案

### 20.1 CUDA out of memory

现象：

```text
torch.cuda.OutOfMemoryError: CUDA out of memory
```

解决：

- 减小 `per_device_train_batch_size`。
- 增大 `gradient_accumulation_steps` 保持有效 batch size。
- 降低 `max_length`。
- 开启 `gradient_checkpointing`。
- 使用 `bf16` 或 `fp16`。
- 减少 LoRA `target_modules`，只训 attention 层。
- 减小 LoRA `r`。

### 20.2 Tokenizer 没有 pad token

现象：

```text
ValueError: Asking to pad but the tokenizer does not have a padding token
```

解决：

- 将 `tokenizer.pad_token` 设置为 `tokenizer.eos_token`。
- 将 `model.config.pad_token_id` 设置为 `tokenizer.pad_token_id`。

本工程的 `model/tokenizer.py` 会自动处理。

### 20.3 labels 全是 `-100`

现象：

- loss 可能为 `nan`。
- 模型没有学习信号。

原因：

- 数据中没有 assistant 回复。
- prompt 拼接模板错误。
- max_length 太短，把 assistant 内容截断了。

解决：

- 检查 JSONL 是否存在 `output`、`assistant` 或 `gpt` 内容。
- 增大 `max_length`。
- 抽样打印 tokenized labels，确认回复部分没有被全部 mask。

### 20.4 `target_modules` 找不到

现象：

```text
ValueError: Target modules {...} not found in the base model
```

解决：

- 打印模型 `named_modules()`，确认线性层名称。
- Qwen 系列常用 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- 如果只想稳妥起步，先使用 attention 四件套。

### 20.5 bf16 不支持

现象：

- 启用 bf16 后报 CUDA 或 dtype 相关错误。

解决：

- Ampere 及以上 NVIDIA GPU 通常支持 bf16。
- 老 GPU 可改为 `bf16: false`、`fp16: true`。
- CPU 或部分消费级环境不适合训练 4B 模型。

### 20.6 DeepSpeed 配置不匹配

现象：

- batch size 报错。
- fp16/bf16 配置冲突。

解决：

- 将 DeepSpeed 中相关值设为 `auto`。
- 保持 `train_micro_batch_size_per_gpu` 与训练配置一致。
- 不需要 DeepSpeed 时将 `deepspeed: null`。

### 20.7 `trust_remote_code` 相关错误

现象：

- 模型加载失败。
- 自定义结构无法识别。

解决：

- 加载 tokenizer 和 model 时设置 `trust_remote_code=True`。
- 使用较新的 Transformers 版本。

## 21. GPU 显存占用分析

LoRA 训练显存主要来自：

1. 基座模型权重。
2. LoRA 可训练参数。
3. optimizer 状态。
4. activation。
5. gradient。
6. 临时 buffer 和 CUDA kernel workspace。

对 4B 级别模型，粗略估算：

- bf16 模型权重：约 `4B * 2 bytes = 8GB`，实际会因 embedding、buffer、加载策略更高。
- LoRA 参数：通常几十 MB 到数百 MB。
- AdamW optimizer 状态：只针对 LoRA 参数，远小于全参微调。
- activation：和 `batch_size * max_length * hidden_size * layers` 强相关。

影响显存的关键配置：

| 配置 | 显存影响 |
| --- | --- |
| `max_length` | 很大，长度翻倍通常显存明显上升 |
| `per_device_train_batch_size` | 近似线性增加 |
| `gradient_checkpointing` | 降低 activation 显存，但训练变慢 |
| `target_modules` | 注入模块越多，LoRA 参数和梯度越多 |
| `r` | rank 越大，LoRA 参数越多 |
| `bf16/fp16` | 比 fp32 省约一半模型权重显存 |
| `use_cache` | 训练时应关闭，避免额外缓存 |

常见单卡起步建议：

| GPU 显存 | 建议 |
| ---: | --- |
| 16GB | `max_length=1024`，batch size 1，gradient checkpointing 开启，只训 attention |
| 24GB | `max_length=2048`，batch size 1，gradient checkpointing 开启 |
| 40GB | `max_length=4096`，可尝试更多 target modules |
| 80GB | 可以提高 batch size、max length 或 LoRA rank |

实际显存还取决于 CUDA、PyTorch、Transformers、attention 实现和数据长度分布。

## 22. 不同 LoRA 参数的影响

### 22.1 `r`

`r` 是低秩维度，直接控制 adapter 容量。

| r | 特点 |
| ---: | --- |
| 4 | 参数少，显存低，适合小数据或快速试验 |
| 8 | 稳健起步值 |
| 16 | 常用默认值，效果和成本平衡 |
| 32 | 容量更强，适合复杂任务 |
| 64+ | 成本更高，数据量不足时可能过拟合 |

### 22.2 `alpha`

`alpha` 控制 LoRA 更新强度。实际缩放通常是：

$$
\frac{\alpha}{r}
$$

常见组合：

| r | alpha | scale |
| ---: | ---: | ---: |
| 8 | 16 | 2 |
| 16 | 32 | 2 |
| 32 | 64 | 2 |

如果训练不稳定，可以降低 `alpha` 或学习率。

### 22.3 `dropout`

`dropout` 用于降低过拟合：

| dropout | 适用场景 |
| ---: | --- |
| 0.0 | 大数据集、希望最大拟合能力 |
| 0.05 | 常用默认值 |
| 0.1 | 数据少或容易过拟合 |
| 0.2+ | 可能明显降低拟合能力，慎用 |

### 22.4 `target_modules`

只训 attention：

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

优点是省显存、训练快；缺点是表达能力较弱。

attention + MLP：

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

优点是效果更强；缺点是显存和训练时间更高。

## 23. 如何提高训练效果

优先级从高到低：

1. 提升数据质量。高质量、无冲突、格式统一的数据比调参更重要。
2. 对齐训练目标。问答、摘要、代码、客服、多轮对话应使用对应数据模板。
3. 控制样本长度。过长样本会浪费显存，过短样本可能缺少上下文。
4. 只对 assistant 回复计算 loss。不要让模型学习复述 user prompt。
5. 设置合理学习率。LoRA 常见范围 `1e-4` 到 `3e-4`。
6. 使用验证集。只看训练 loss 容易误判。
7. 抽样人工评测。定期检查模型输出是否符合业务目标。
8. 避免脏数据。错误答案、乱码、重复样本会直接影响模型风格。
9. 合理混合数据。通用能力数据和领域数据比例要根据目标调整。
10. 保存多个 checkpoint。不要只保留最后一个。

如果模型回答变短：

- 检查训练集中答案是否普遍很短。
- 适当增加长答案样本。
- 推理时提高 `max_new_tokens`。

如果模型幻觉变多：

- 检查训练数据是否包含无法验证的断言。
- 添加“不知道时说明不确定”的样本。
- 降低推理温度。

如果模型格式不稳定：

- 增加严格格式样本。
- 在 system prompt 中声明输出格式。
- 使用更一致的数据模板。

## 24. 如何制作自己的数据集

推荐步骤：

1. 明确任务类型：问答、分类、抽取、总结、代码生成、客服对话等。
2. 设计统一字段：优先使用 `instruction/input/output` 或 `messages`。
3. 写清楚指令：避免含糊任务，例如“处理一下这个文本”。
4. 保证答案正确：宁可少一些，也不要混入错误答案。
5. 覆盖真实分布：包含常见问题、边界情况和困难样本。
6. 去重：删除完全重复和近似重复样本。
7. 脱敏：删除手机号、身份证、密钥、内部地址等敏感信息。
8. 切分训练和验证：例如 95% train，5% eval。
9. 抽样审查：训练前人工看一批样本。
10. 小规模试训：先用 100 到 1000 条样本跑通，再扩大训练。

Alpaca 样本模板：

```json
{
  "instruction": "请根据给定用户评论判断情绪。",
  "input": "物流很快，但是包装破损了。",
  "output": "混合情绪。用户认可物流速度，但对包装破损不满意。"
}
```

多轮 messages 样本模板：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业客服助手。"},
    {"role": "user", "content": "我的订单一直没有发货。"},
    {"role": "assistant", "content": "请提供订单号，我会帮你查询发货状态。"},
    {"role": "user", "content": "订单号是 123456。"},
    {"role": "assistant", "content": "我已收到订单号。建议先核对支付状态和预计发货时间。"}
  ]
}
```

数据质量检查清单：

- 每行 JSON 可解析。
- 所有样本都有 assistant 目标答案。
- 没有空答案。
- 没有大段重复。
- 没有明显乱码。
- 没有把系统内部规则暴露给模型的内容。
- 没有敏感个人信息。

## 25. 最佳实践

### 25.1 推荐起步配置

单卡 24GB 可先尝试：

```yaml
max_length: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
```

LoRA：

```yaml
r: 16
alpha: 32
dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

### 25.2 实验管理

- 每次实验固定 `seed`。
- 保存 `train.yaml` 和 `lora.yaml` 快照。
- 给 `output_dir` 使用有含义的名字，如 `outputs/legal-r16-lr2e-4`。
- 记录数据版本，不要只记录代码版本。
- 不同实验不要覆盖同一个输出目录。

### 25.3 训练观察

正常现象：

- 前几百步 loss 明显下降。
- 学习率先 warmup，再 cosine 下降。
- 验证 loss 稳定下降或在一定范围内波动。

异常现象：

- loss 为 `nan`：检查学习率、fp16/bf16、labels、脏数据。
- loss 完全不降：检查 labels 是否全 `-100`，LoRA 参数是否可训练。
- 训练很慢：检查是否开启了过长 `max_length` 或过多 target modules。
- 输出重复：降低学习率，增加高质量样本，推理时使用 repetition penalty。

### 25.4 训练完成后的评估

建议至少做三类评估：

- 自动指标：eval loss、perplexity。
- 固定题集：准备 50 到 200 条固定 prompt，比较不同 checkpoint。
- 人工评审：检查准确性、格式、语气、安全性和业务可用性。

### 25.5 部署建议

- 多任务场景优先保存 adapter，按任务动态加载。
- 单任务高频部署可 merge 成完整模型。
- 合并前保留原始 adapter，方便继续训练。
- 推理服务中固定 tokenizer、chat template 和 generation config。
- 对外服务前做安全测试和边界测试。

## 快速命令总览

安装：

```bash
cd /home/cells/dev/model/qwen3-4B/finetune
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

训练：

```bash
bash scripts/train.sh
```

断点续训：

```bash
python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-lora/checkpoint-200
```

推理：

```bash
bash scripts/infer.sh
```

合并：

```bash
bash scripts/merge.sh
```

TensorBoard：

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 推荐阅读顺序

第一次使用时，建议按下面顺序阅读和运行：

1. 先读 `config/lora.yaml` 和 `config/train.yaml`。
2. 查看 `data/example.jsonl`，确认数据格式。
3. 运行 `bash scripts/train.sh` 跑通最小训练。
4. 用 `bash scripts/infer.sh` 检查 adapter 是否可用。
5. 需要部署完整模型时运行 `bash scripts/merge.sh`。

这套工程的设计重点是清晰、可改、可恢复。先用小数据跑通闭环，再扩大数据和训练规模，
通常比一开始就追求大参数和长上下文更稳。
