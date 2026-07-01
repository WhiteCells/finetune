# Qwen3-4B LoRA 微调项目

面向 Qwen3-4B Instruct 的 LoRA 监督微调工程。代码按训练链路拆分为
数据预处理、tokenizer、模型加载、LoRA 注入、dataset/collator、Trainer、推理和
权重合并几部分。

默认基座模型路径以 `config/train.yaml` 的 `model_name_or_path` 为准，当前是：

```text
./models/Qwen3-4B-Instruct-2507
```

## 执行顺序

第一次使用建议按下面顺序走一遍：

1. 准备 Python 环境和依赖：见 [docs/01_environment.md](docs/01_environment.md)
2. 准备或转换训练数据：见 [docs/02_data_preprocess.md](docs/02_data_preprocess.md)
3. 修改训练和 LoRA 配置：见 [docs/03_train.md](docs/03_train.md)
4. 启动训练并观察日志：见 [docs/03_train.md](docs/03_train.md)
5. 理解训练原理和参数取舍：见 [docs/06_train_principles.md](docs/06_train_principles.md)
6. 加载 adapter 推理：见 [docs/04_inference_merge.md](docs/04_inference_merge.md)
7. 按需合并完整模型：见 [docs/04_inference_merge.md](docs/04_inference_merge.md)
8. 运行本地案例测试：见 [docs/05_testing.md](docs/05_testing.md)

## 快速命令

安装依赖：

```bash
uv sync
```

预处理数据：

```bash
uv run python data/preprocess.py \
  --input data/example.jsonl \
  --output /tmp/qwen3-example.normalized.jsonl \
  --format auto \
  --skip-invalid \
  --deduplicate
```

训练：

```bash
bash scripts/train.sh
```

断点续训：

```bash
RESUME_FROM_CHECKPOINT=outputs/qwen3-4b-lora/checkpoint-200 bash scripts/train.sh
```

加载已有 adapter 继续训练：

```bash
ADAPTER_PATH=outputs/old-adapter bash scripts/train.sh
```

推理：

```bash
bash scripts/infer.sh
```

合并 LoRA：

```bash
bash scripts/merge.sh
```

运行测试：

```bash
uv run python -m unittest discover -s tests -v
```

## 目录说明

```text
finetune/
├── config/              # 训练、LoRA、DeepSpeed 配置
├── data/                # 示例数据和预处理脚本
├── docs/                # 分步骤操作文档
├── model/               # tokenizer、基座模型、LoRA adapter 工具
├── scripts/             # 常用 shell 入口
├── tests/               # 本地案例测试
├── trainer/             # dataset、collator、Trainer、metrics
├── utils/               # 日志、保存、随机种子
├── train.py             # 训练主入口
├── inference.py         # LoRA adapter 推理入口
└── merge_lora.py        # LoRA 合并入口
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `config/train.yaml` | 训练参数、路径、精度、日志、断点恢复 |
| `config/lora.yaml` | LoRA rank、alpha、dropout、target modules |
| `data/preprocess.py` | 将 Alpaca、ShareGPT、messages、自定义格式转为统一 JSONL |
| `trainer/dataset.py` | 渲染 chat template，只对 assistant 回复计算 loss |
| `trainer/collator.py` | batch 动态 padding，labels padding 使用 `-100` |
| `trainer/trainer.py` | 构建 `TrainingArguments` 和 LoRA Trainer |
| `model/loader.py` | 加载 causal LM，处理 dtype、attention、gradient checkpointing |
| `model/lora.py` | 读取 LoRA 配置，注入或加载 adapter |
| `inference.py` | 加载基座模型和 LoRA adapter 生成文本 |
| `merge_lora.py` | 调用 `merge_and_unload()` 导出完整模型 |

训练操作手册见 [docs/03_train.md](docs/03_train.md)，训练机制说明见
[docs/06_train_principles.md](docs/06_train_principles.md)。

## 数据格式

推荐最终训练文件是一行一个 JSON 对象的 JSONL。支持以下常见格式：

Alpaca：

```json
{"instruction":"请解释 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法。"}
```

OpenAI messages：

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好。"}]}
```

ShareGPT：

```json
{"conversations":[{"from":"human","value":"你好"},{"from":"gpt","value":"你好。"}]}
```

训练阶段会统一标准化成：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## 训练产物

默认输出目录是：

```text
outputs/qwen3-4b-lora
```

常见产物：

```text
outputs/qwen3-4b-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── config_snapshots/
├── train.log
├── trainer_state.json
└── metrics.summary.json
```

checkpoint 会保存到：

```text
outputs/qwen3-4b-lora/checkpoint-200
```

adapter 不是完整模型，推理时必须同时加载基座模型。需要完整模型时再运行合并流程。
