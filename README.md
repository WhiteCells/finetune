```sh
uv run main.py --prompt "自我介绍"
```

微调脚本：

```sh
uv run finetune.py \
  --data-path data/train.jsonl \
  --output-dir outputs/qwen3-4b-lora
```

`train.jsonl` 可以是一行一个样本，支持两种常见格式：

```json
{"messages":[{"role":"system","content":"你是一个助理"},{"role":"user","content":"介绍一下杭州"},{"role":"assistant","content":"杭州是浙江省省会。"}]}
```

```json
{"prompt":"介绍一下杭州","response":"杭州是浙江省省会。"}
```

脚本默认使用 LoRA，只对 `assistant` 回复部分计算 loss。常用参数：

- `--max-length 2048`
- `--batch-size 1`
- `--gradient-accumulation-steps 8`
- `--learning-rate 2e-4`
- `--lora-target-modules q_proj,k_proj,v_proj,o_proj`

如果改了依赖，先执行：

```sh
uv sync
```
