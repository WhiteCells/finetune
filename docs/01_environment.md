# 01. 环境准备

目标：用 uv 准备能加载 Qwen3-4B、运行 LoRA 训练和推理的 Python 环境。

## 1. 进入项目目录

```bash
cd /home/cells/dev/model/qwen3-4B
```

## 2. 安装 uv

如果机器上还没有 uv，先安装 uv。安装完成后确认版本：

```bash
uv --version
```

## 3. 同步项目环境

项目使用 `.python-version` 固定 Python 3.11，并通过 `pyproject.toml` 管理依赖：

```bash
uv sync
```

uv 会创建 `.venv`，后续命令用 `uv run ...` 执行，不需要手动 `source .venv/bin/activate`。

如果当前系统的默认 uv 缓存目录不可写，可以把缓存放到项目目录：

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
```

项目自带的 `scripts/*.sh` 已经默认设置这个缓存目录。

如果当前机器没有 Python 3.11，可以让 uv 安装：

```bash
uv python install 3.11
uv sync
```

DeepSpeed 是可选依赖。普通单卡 LoRA 先不用安装；需要多卡或 ZeRO 时再执行：

```bash
uv sync --extra deepspeed
```

## 4. 下载或检查模型目录

默认配置使用：

```text
./models/Qwen3-4B-Instruct-2507
```

如果模型还没下载，可以运行：

```bash
uv run python download_model.py \
  --output-dir ./models/Qwen3-4B-Instruct-2507
```

确认目录存在：

```bash
ls -la ./models/Qwen3-4B-Instruct-2507
```

至少应能看到 `config.json`、tokenizer 文件和模型权重文件。

## 5. 检查 CUDA 和核心依赖

```bash
uv run python - <<'PY'
import torch
import transformers
import peft

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu 0:", torch.cuda.get_device_name(0))
PY
```

## 6. 常见环境问题

`ModuleNotFoundError: No module named 'torch'`

说明当前 uv 环境没同步完整依赖，重新执行 `uv sync`。

`torch.cuda.is_available()` 为 `False`

说明当前环境不能使用 CUDA。4B 模型训练通常需要 GPU；CPU 只适合做预处理和轻量测试。

DeepSpeed 安装失败

先把 `config/train.yaml` 中的 `deepspeed` 保持为 `null`，跑通普通 LoRA 训练后再执行
`uv sync --extra deepspeed`。
