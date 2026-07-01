# 04. 推理和 LoRA 合并

目标：用训练好的 LoRA adapter 做推理，并按需导出合并后的完整模型。

## 1. adapter 推理

训练完成后，默认 adapter 在：

```text
outputs/qwen3-4b-lora
```

运行：

```bash
bash scripts/infer.sh
```

手动命令：

```bash
uv run python inference.py \
  --model-name-or-path ./models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --prompt "请用一句话解释 LoRA。" \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9
```

## 2. 稳定评测参数

做固定题集评测时，建议降低随机性：

```bash
uv run python inference.py \
  --model-name-or-path ./models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --prompt "请用一句话解释 LoRA。" \
  --max-new-tokens 128 \
  --no-do-sample \
  --num-beams 1
```

关闭采样时，脚本不会再把 `temperature`、`top_p`、`top_k` 传给 `generate()`，可以减少
Transformers 的无关 warning。

## 3. 保存推理结果

```bash
OUTPUT_FILE=outputs/sample_answer.txt bash scripts/infer.sh
```

或：

```bash
uv run python inference.py \
  --model-name-or-path ./models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --prompt "请用一句话解释 LoRA。" \
  --output-file outputs/sample_answer.txt
```

## 4. 合并 LoRA

adapter 不是完整模型。如果部署方需要普通 Transformers 模型目录，运行合并：

```bash
bash scripts/merge.sh
```

手动命令：

```bash
uv run python merge_lora.py \
  --model-name-or-path ./models/Qwen3-4B-Instruct-2507 \
  --adapter-path outputs/qwen3-4b-lora \
  --output-dir outputs/qwen3-4b-merged
```

合并后的目录：

```text
outputs/qwen3-4b-merged
```

可以像普通模型一样加载：

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

## 5. 选择 adapter 还是合并模型

优先保存 adapter：

- 多任务共用同一个基座模型。
- 需要继续训练。
- 需要节省磁盘空间。

选择合并模型：

- 部署环境不想引入 PEFT。
- 单任务高频调用。
- 需要交付完整 Transformers 模型目录。

合并前保留原始 adapter，方便后续恢复训练或回滚。
