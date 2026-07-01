# 02. 数据准备和预处理

目标：把原始数据转换成训练脚本稳定读取的 JSONL，并在训练前检查坏样本。

## 1. 确定原始格式

项目支持四类常见输入。

Alpaca：

```json
{"instruction":"请解释 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法。"}
```

ShareGPT：

```json
{"conversations":[{"from":"human","value":"你好"},{"from":"gpt","value":"你好。"}]}
```

OpenAI messages：

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好。"}]}
```

自定义单轮：

```json
{"prompt":"你好","response":"你好。"}
```

## 2. 运行预处理

```bash
uv run python data/preprocess.py \
  --input data/example.jsonl \
  --output /tmp/qwen3-example.normalized.jsonl \
  --format auto \
  --skip-invalid \
  --deduplicate
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--format auto` | 自动识别 Alpaca、ShareGPT、messages、自定义格式 |
| `--skip-invalid` | 遇到坏样本时跳过，并输出坏样本编号和原因 |
| `--deduplicate` | 对标准化后的样本去重 |
| `--system-prompt` | 给没有 system 消息的样本注入统一 system prompt |
| `--ensure-ascii` | 写出 JSONL 时转义非 ASCII 字符 |

## 3. 检查输出

```bash
head -n 3 /tmp/qwen3-example.normalized.jsonl
```

每行都应是这种结构：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## 4. 数据质量检查

训练前至少检查这些点：

- 每行都是合法 JSON。
- 每条样本最终都有 `assistant` 消息。
- 最后一条消息是 `assistant`，这样才有监督目标。
- 答案不为空。
- 没有明显重复、乱码、隐私数据、错误答案。
- 长样本比例可控，避免大量样本被 `max_length` 截断。

## 5. 推荐切分方式

小数据可以先只跑训练集；正式实验建议切出验证集：

```text
data/train.jsonl
data/eval.jsonl
```

然后修改 `config/train.yaml`：

```yaml
train_file: "data/train.jsonl"
eval_file: "data/eval.jsonl"
```

## 6. 坏样本处理

如果预处理输出：

```text
[WARN] 跳过第 12 条坏样本: 最后一条消息必须是 assistant，才能形成监督目标。
```

说明第 12 条样本不能直接用于监督微调。优先修数据；只有在确认坏样本可以丢弃时才使用
`--skip-invalid`。
