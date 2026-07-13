# 公式细节

这一组文档解释项目为什么这样训练和生成，不重复操作步骤。

1. [监督微调损失](01_监督微调损失.md)
2. [LoRA 低秩适配](02_LoRA低秩适配.md)
3. [训练优化与显存](03_训练优化与显存.md)
4. [生成和采样](04_生成和采样.md)

## 代码对应关系

| 数学概念 | 对应代码 |
| --- | --- |
| 数据标准化和消息约束 | `data/preprocess.py` |
| chat template 和 tokenizer | `model/tokenizer.py` |
| assistant 标签掩码 | `trainer/dataset.py` |
| batch padding 和 `-100` | `trainer/collator.py` |
| LoRA 注入与 adapter 加载 | `model/lora.py` |
| Trainer、优化器参数和学习率调度 | `trainer/trainer.py` |
| 推理 prompt、采样和解码 | `inference.py` |

需要实际修改配置和运行脚本时，回到 [微调应用](../微调应用/README.md)。
