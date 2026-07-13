# 微调应用

这一组文档只说明如何使用当前项目。所有入口都是 Python 文件，不需要记忆项目命令行参数，也没有 shell 启动脚本。

从项目根目录按顺序执行：

```bash
uv sync
uv run download_model.py
uv run data/preprocess.py
uv run train.py
uv run inference.py
uv run merge_lora.py
```

预处理不是强制步骤：训练代码可以直接读取 Alpaca、ShareGPT、OpenAI `messages` 和简单 `prompt/response` 数据。需要统一格式、去重或跳过坏样本时，再运行 `data/preprocess.py`。

## 修改位置

| 目标 | 修改文件 |
| --- | --- |
| 基座模型、数据、训练超参数、断点续训 | `config/train.yaml` |
| LoRA rank、缩放和目标层 | `config/lora.yaml` |
| 原始数据和预处理输出位置 | `data/preprocess.py` 顶部常量 |
| 推理模型、adapter、问题和采样参数 | `inference.py` 的 `RUN_CONFIG` |
| 合并模型的三个目录 | `merge_lora.py` 的 `RUN_CONFIG` |
| 下载的 Hugging Face 仓库和目录 | `download_model.py` 顶部常量 |

## 文档索引

1. [环境和模型](01_环境和模型.md)
2. [数据准备](02_数据准备.md)
3. [训练](03_训练.md)
4. [推理和合并](04_推理和合并.md)
5. [测试和检查](05_测试和检查.md)

想理解 loss、LoRA 矩阵、优化器和采样公式，继续看 [公式细节](../公式细节/README.md)。
