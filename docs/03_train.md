# 03. 训练操作手册

目标：用 `config/train.yaml` 和 `config/lora.yaml` 启动 Qwen3-4B LoRA 监督微调，并能判断一次训练是否正常、如何续训、如何排查常见问题。

训练原理和参数取舍见 [06. 训练原理和配置取舍](06_train_principles.md)。本文只写操作步骤。

## 1. 训练入口会做什么

推荐入口：

```bash
bash scripts/train.sh
```

等价手动命令：

```bash
uv run python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml
```

执行链路：

1. `scripts/train.sh` 读取环境变量，拼出 `train.py` 命令。
2. `train.py` 读取 `config/train.yaml` 和 `config/lora.yaml`。
3. 检查基座模型、训练集、验证集、checkpoint、adapter 路径是否存在。
4. 加载 tokenizer 和 causal LM，并按配置设置 dtype、attention、gradient checkpointing。
5. 新建 LoRA adapter，或用 `adapter_path` 加载已有 adapter。
6. 构建训练集和可选验证集，只对 assistant 回复计算 loss。
7. 用 Transformers Trainer 训练、保存 checkpoint、adapter、日志和 metrics。

## 2. 启动前必须确认

先检查 `config/train.yaml`：

| 字段 | 当前默认值 | 必须确认什么 |
| --- | --- | --- |
| `model_name_or_path` | `./models/Qwen3-4B-Instruct-2507` | 目录存在，且包含模型权重和 tokenizer 文件 |
| `train_file` | `data/example.jsonl` | 文件存在，每条样本能转成 `messages`，最后有 assistant 回复 |
| `eval_file` | `null` | 没有验证集就保持 `null`；有验证集就填真实路径 |
| `output_dir` | `outputs/qwen3-4b-lora` | 正式实验建议每次换新目录，避免覆盖旧实验 |
| `logging_dir` | `logs/qwen3-4b-lora` | 建议和 `output_dir` 同步命名 |
| `max_length` | `2048` | 必须能容纳大部分问题和答案，否则会截断 |
| `bf16` / `fp16` | `true` / `false` | 支持 bf16 的 GPU 用 bf16；老卡可改成 fp16 |
| `gradient_checkpointing` | `true` | 显存紧张时保持开启 |

再检查 `config/lora.yaml`：

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

起步建议先只训 attention 四件套：`q_proj`、`k_proj`、`v_proj`、`o_proj`。如果训练 loss 能下降但任务能力仍明显不足，再考虑加入 `gate_proj`、`up_proj`、`down_proj`。

## 3. 先跑一次冒烟训练

冒烟训练只验证链路能跑通，不用于产出模型。建议临时新建一个训练配置，例如 `config/train.smoke.yaml`，内容从 `config/train.yaml` 复制后改这些字段：

```yaml
train_file: "data/example.jsonl"
eval_file: null
output_dir: "outputs/smoke-qwen3-4b-lora"
logging_dir: "logs/smoke-qwen3-4b-lora"
max_length: 512
max_steps: 2
logging_steps: 1
save_steps: 1
save_total_limit: 2
run_name: "smoke-qwen3-4b-lora"
```

启动：

```bash
TRAIN_CONFIG=config/train.smoke.yaml bash scripts/train.sh
```

冒烟训练通过的标准：

- 终端能打印 `trainable params`，并且可训练参数远小于总参数。
- 日志出现训练集样本数。
- 能完成 2 个 step，并在 `outputs/smoke-qwen3-4b-lora` 下生成 adapter 文件。
- `outputs/smoke-qwen3-4b-lora/train.log` 中没有路径、CUDA、数据格式错误。

## 4. 配正式训练参数

正式训练前，把 `config/train.yaml` 至少改成自己的数据和输出目录：

```yaml
train_file: "data/train.jsonl"
eval_file: "data/eval.jsonl"
output_dir: "outputs/qwen3-4b-lora-exp001"
logging_dir: "logs/qwen3-4b-lora-exp001"
run_name: "qwen3-4b-lora-exp001"
```

如果还没有验证集：

```yaml
eval_file: null
```

单卡默认有效 batch size：

```text
per_device_train_batch_size * gradient_accumulation_steps * GPU 数
= 1 * 8 * 1
= 8
```

显存不足时优先这样改：

```yaml
per_device_train_batch_size: 1
gradient_checkpointing: true
max_length: 1024
```

小数据集建议先少训：

```yaml
num_train_epochs: 1
max_steps: -1
learning_rate: 1.0e-4
```

想固定训练步数时，用 `max_steps` 覆盖 epoch：

```yaml
max_steps: 1000
num_train_epochs: 3
```

只要 `max_steps` 大于 0，Trainer 会优先按 step 数停止。

## 5. 启动正式训练

使用默认配置：

```bash
bash scripts/train.sh
```

指定配置：

```bash
TRAIN_CONFIG=config/train.yaml \
LORA_CONFIG=config/lora.yaml \
LOG_LEVEL=INFO \
bash scripts/train.sh
```

常用日志文件：

```text
outputs/qwen3-4b-lora-exp001/train.log
```

TensorBoard：

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 6. 训练中看什么

先看启动阶段：

- `运行环境`：确认 CUDA 可用、GPU 数正确。
- `Tokenizer 摘要`：确认有 `pad_token_id`、`eos_token_id` 和 chat template。
- `基础模型摘要`：确认加载的是 Qwen causal LM。
- `可训练参数摘要`：确认只训练 LoRA 参数。
- `训练集样本数` / `验证集样本数`：确认数据路径没有填错。

再看训练指标：

| 指标 | 正常现象 | 异常信号 |
| --- | --- | --- |
| `loss` | 总体下降，允许短期波动 | 长时间不降、突然变成 `nan` |
| `perplexity` | 随 loss 下降 | 极大或持续上升 |
| `grad_norm` | 有波动但不长期尖峰 | 经常异常尖峰，随后 loss 发散 |
| `learning_rate` | 先 warmup，再按调度变化 | 一直为 0 或明显不符合预期 |
| `eval_loss` | 有验证集时稳定或下降 | train loss 降但 eval loss 升，可能过拟合 |

## 7. 训练产物在哪里

默认产物在 `output_dir`：

```text
outputs/qwen3-4b-lora-exp001/
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-200/
├── config_snapshots/
├── metrics.summary.json
├── train.log
└── trainer_state.json
```

重点文件：

| 文件 | 作用 |
| --- | --- |
| `adapter_model.safetensors` | LoRA adapter 权重 |
| `adapter_config.json` | PEFT 加载 adapter 需要的配置 |
| `checkpoint-*` | 包含训练状态，可用于完整断点续训 |
| `trainer_state.json` | Trainer 的 step、日志、最优指标等状态 |
| `metrics.summary.json` | 最终 train/eval 指标汇总 |
| `config_snapshots/` | 本次实验使用的 train/lora 配置快照 |

adapter 不是完整模型。推理时需要同时加载基座模型和 adapter，见 [04. 推理和 LoRA 合并](04_inference_merge.md)。

## 8. 从 checkpoint 继续训练

适用场景：训练中断，但 `checkpoint-*` 目录还在，里面保留了模型、optimizer、scheduler、Trainer state。

```bash
RESUME_FROM_CHECKPOINT=outputs/qwen3-4b-lora-exp001/checkpoint-200 \
bash scripts/train.sh
```

手动命令：

```bash
uv run python train.py \
  --train-config config/train.yaml \
  --lora-config config/lora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-lora-exp001/checkpoint-200
```

注意：

- checkpoint 续训会恢复 optimizer 和 scheduler，适合接着同一次实验继续跑。
- 正常 checkpoint 续训不要同时设置 `ADAPTER_PATH`。
- 续训时 `train_file`、LoRA 结构、基座模型路径应与原实验保持一致。

## 9. 从已有 adapter 继续训练

适用场景：只有 adapter 权重，没有 optimizer 和 Trainer state，或者想把旧 adapter 当初始化权重继续做新实验。

```bash
ADAPTER_PATH=outputs/old-adapter \
TRAIN_CONFIG=config/train.yaml \
bash scripts/train.sh
```

这会以可训练方式加载旧 adapter，然后用当前 `config/train.yaml` 重新开始 optimizer 和 scheduler。

建议：

- 给新实验设置新的 `output_dir` 和 `logging_dir`。
- `config/lora.yaml` 仍会被读取，但 adapter 结构以 `ADAPTER_PATH` 里的配置为准。
- adapter 继续训练不要同时设置 `RESUME_FROM_CHECKPOINT`。

## 10. 常见问题

### 路径不存在

现象：

```text
FileNotFoundError: 基础模型路径不存在
```

处理：

- 检查 `model_name_or_path` 是否相对当前项目根目录。
- 检查模型是否已下载完整。
- 检查 `train_file` / `eval_file` 是否真实存在。

### CUDA out of memory

优先级从高到低：

1. 保持 `per_device_train_batch_size: 1`。
2. 降低 `max_length`，例如从 `2048` 改到 `1024` 或 `512`。
3. 保持 `gradient_checkpointing: true`。
4. 减少 LoRA `target_modules`。
5. 降低 LoRA `r`，例如从 `16` 改到 `8`。

### loss 变成 nan

处理：

- 检查数据是否有空答案、乱码、异常长样本。
- 检查是否有样本截断后没有 assistant 标签。
- 降低 `learning_rate`，例如从 `2.0e-4` 改到 `1.0e-4`。
- 老 GPU 上尝试 `bf16: false`、`fp16: true`。
- 保持 `max_grad_norm: 1.0`。

### 训练很慢

检查：

- 是否真的在 GPU 上跑，日志中 `cuda_available` 应为 `true`。
- `max_length` 是否过大。
- 是否把 MLP 三件套也加入了 `target_modules`。
- 数据是否有大量超长样本。

### 验证集没有运行

原因：

- `eval_file: null` 时，代码会自动把评估策略设为 `no`。
- 只有配置了真实 `eval_file`，`eval_steps` 和 `eval_loss` 才会生效。

### 训练后效果变差

可能原因：

- 学习率过高，先降到 `1.0e-4` 或 `5.0e-5`。
- 小数据训练 epoch 太多，先减少 `num_train_epochs`。
- 数据答案质量不稳定。
- 训练数据风格覆盖了原模型能力，导致目标外问题退化。

## 11. 推荐实验节奏

1. 用 `data/example.jsonl` 跑冒烟训练，确认链路可用。
2. 用 100 到 1000 条真实数据跑 `max_steps: 20` 到 `100`，检查 loss 和样例输出。
3. 加入验证集，固定一组测试 prompt，每次训练后用相同推理参数比较。
4. 再扩大训练数据、步数和 `max_length`。
5. 效果稳定后，再尝试提高 LoRA `r` 或增加 `target_modules`。
