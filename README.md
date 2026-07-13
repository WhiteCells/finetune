# Qwen3 LoRA 微调

这是一个面向 Qwen3 Instruct 模型的 LoRA 监督微调项目。项目保留训练真正需要的模块：数据规范化、chat template、LoRA、Trainer、adapter 推理和权重合并。

启动不需要编写项目命令行参数，也没有 shell 包装。修改 YAML 或 Python 文件顶部的配置后，直接运行对应 `.py` 文件。

## 最短流程

```bash
uv sync
uv run train.py
uv run inference.py
```

首次训练前，确认：

1. `config/train.yaml` 中的 `model_name_or_path` 指向已下载模型。
2. `train_file` 指向真实训练数据。
3. `output_dir` 是本次实验专用目录。
4. `inference.py` 的 `RUN_CONFIG.adapter_path` 指向训练完成后的 adapter。

## 入口和配置

| 文件 | 用途 | 修改位置 |
| --- | --- | --- |
| `train.py` | LoRA 训练 | `config/train.yaml`、`config/lora.yaml` |
| `data/preprocess.py` | 统一 JSON/JSONL 数据 | 文件顶部常量 |
| `inference.py` | 加载 adapter 推理 | `RUN_CONFIG` |
| `merge_lora.py` | 合并为完整模型 | `RUN_CONFIG` |
| `download_model.py` | 下载 Hugging Face 模型 | 文件顶部常量 |
| `main.py` | 最小基座模型推理示例 | `MODEL_PATH`、`PROMPT` |

## 文档

文档分为两个目录：

- [微调应用](docs/微调应用/README.md)：环境、数据、训练、推理、合并和测试。
- [公式细节](docs/公式细节/README.md)：监督损失、LoRA、优化器、显存和采样公式。

## 目录

```text
finetune/
├── config/       # 训练、LoRA、DeepSpeed 配置
├── data/         # 数据和预处理脚本
├── docs/
│   ├── 微调应用/
│   └── 公式细节/
├── model/        # tokenizer、模型加载、LoRA
├── tests/        # 不加载大模型的测试
├── trainer/      # dataset、collator、Trainer、metrics
├── utils/        # 日志、保存、随机种子
├── train.py
├── inference.py
├── merge_lora.py
└── download_model.py
```
