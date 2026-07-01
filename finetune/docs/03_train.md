# 03. 训练和断点恢复

目标：用 `config/train.yaml` 和 `config/lora.yaml` 启动 LoRA 训练，并能恢复 checkpoint。

## 1. 检查 LoRA 配置

打开 `config/lora.yaml`：

```yaml
r: 16
alpha: 32
dropout: 0.05
bias: "none"
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
task_type: "CAUSAL_LM"
inference_mode: false
```

起步建议先训 attention 四件套：`q_proj,k_proj,v_proj,o_proj`。如果数据量足够、显存足够，再考虑加入
`gate_proj,up_proj,down_proj`。

## 2. 检查训练配置

重点确认 `config/train.yaml` 中这些字段：

```yaml
model_name_or_path: "../models/Qwen3-4B-Instruct-2507"
train_file: "data/example.jsonl"
eval_file: null
output_dir: "outputs/qwen3-4b-lora"
max_length: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
```

有效 batch size：

```text
per_device_train_batch_size * gradient_accumulation_steps * GPU 数
```

单卡默认就是 `1 * 8 * 1 = 8`。

## 3. 启动训练

```bash
bash scripts/train.sh
```

等价手动命令：

```bash
uv run python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml
```

训练日志写入：

```text
outputs/qwen3-4b-lora/train.log
```

## 4. 观察训练

训练过程中重点看：

- `loss` 是否下降。
- `perplexity` 是否随 loss 下降。
- `grad_norm` 是否异常尖峰。
- 是否真的只有 LoRA 参数可训练。
- 验证集存在时，`eval_loss` 是否稳定。

启动 TensorBoard：

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 5. 从 checkpoint 继续训练

如果 checkpoint 保留了 optimizer、scheduler 和 trainer state，用：

```bash
RESUME_FROM_CHECKPOINT=outputs/qwen3-4b-lora/checkpoint-200 bash scripts/train.sh
```

或手动：

```bash
uv run python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-lora/checkpoint-200
```

## 6. 从已有 adapter 继续训练

如果只有 adapter 权重，没有 optimizer 状态，用：

```bash
ADAPTER_PATH=outputs/old-adapter bash scripts/train.sh
```

这会加载旧 adapter，然后按当前训练配置继续新的训练。

## 7. 常见训练问题

CUDA out of memory：

- 降低 `max_length`。
- 保持 batch size 1。
- 开启 `gradient_checkpointing`。
- 减少 `target_modules`。
- 降低 LoRA `r`。

loss 是 `nan`：

- 检查数据是否有空答案或异常长样本。
- 检查 labels 是否全是 `-100`。
- 降低学习率。
- 老 GPU 上把 `bf16: false`、`fp16: true`。

训练很慢：

- 降低 `max_length`。
- 减少 LoRA 注入模块。
- 确认没有在 CPU 上跑。
