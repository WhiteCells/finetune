# 05. 案例测试

目标：用小案例逐步验证预处理、配置校验和推理辅助逻辑。这里的测试不会加载 4B 模型。

## 1. 运行全部测试

```bash
uv run python -m unittest discover -s tests -v
```

如果当前 Python 没有安装 `torch`、`transformers`、`peft`，训练和推理辅助测试会显示
`skipped`；预处理测试仍会运行。

## 2. 只跑预处理案例

```bash
uv run python -m unittest tests.test_preprocess -v
```

覆盖内容：

- 自动识别 Alpaca、ShareGPT、messages。
- Alpaca 样本标准化并注入 system prompt。
- 坏样本跳过时输出样本编号和原因。
- CLI 预处理同时执行去重和跳过坏样本。

## 3. 手动预处理案例

创建一个临时 JSONL：

```bash
cat >/tmp/qwen3-preprocess-case.jsonl <<'EOF'
{"instruction":"请解释 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法。"}
{"instruction":"请解释 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法。"}
{"instruction":"缺少答案"}
EOF
```

运行：

```bash
uv run python data/preprocess.py \
  --input /tmp/qwen3-preprocess-case.jsonl \
  --output /tmp/qwen3-preprocess-case.out.jsonl \
  --format auto \
  --skip-invalid \
  --deduplicate
```

预期结果：

```text
写出样本数: 1
跳过样本数: 1
去重丢弃数: 1
```

检查输出：

```bash
cat /tmp/qwen3-preprocess-case.out.jsonl
```

应只有一条标准化后的 `messages` 样本。

## 4. 配置静态检查

不加载模型，只检查配置文件能否被读取：

```bash
uv run python - <<'PY'
from trainer.trainer import load_train_config
from model.lora import load_lora_config

train_config = load_train_config("config/train.yaml")
lora_config = load_lora_config("config/lora.yaml")
print(train_config)
print(lora_config)
PY
```

这个命令需要安装训练依赖，因为 `trainer.trainer` 会导入 Transformers 和 torch。

## 5. 源码语法检查

```bash
uv run python -m compileall -q .
```

如果命令没有输出，说明 Python 文件语法层面通过。

## 6. 小规模闭环测试

安装完整依赖和准备 GPU 后，再跑真实闭环：

```bash
uv run python data/preprocess.py \
  --input data/example.jsonl \
  --output /tmp/qwen3-example.normalized.jsonl \
  --format auto
```

把 `config/train.yaml` 临时改成：

```yaml
train_file: "/tmp/qwen3-example.normalized.jsonl"
max_length: 512
max_steps: 2
save_steps: 1
logging_steps: 1
```

启动训练：

```bash
bash scripts/train.sh
```

再用生成脚本检查 adapter：

```bash
ADAPTER_PATH=outputs/qwen3-4b-lora MAX_NEW_TOKENS=64 bash scripts/infer.sh
```

如果这一步能跑通，再扩大训练数据和训练步数。
