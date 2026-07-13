# LoRA 低秩适配

## 1. 从全量微调到低秩增量

线性层原始权重记为：

```math
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```

全量微调会直接训练整个 `W`。LoRA 冻结 `W`，只学习一个低秩更新：

```math
\Delta W = \frac{\alpha}{r}BA
```

其中：

```math
A \in \mathbb{R}^{r \times d_{\mathrm{in}}}
\qquad
B \in \mathbb{R}^{d_{\mathrm{out}} \times r}
```

前向计算变为：

```math
y = Wx + \frac{\alpha}{r}B(Ax)
```

训练期间梯度只更新 `A` 和 `B`，原始 `W` 保持冻结。

## 2. 参数量为何会下降

全量训练一个线性层需要：

```math
d_{\mathrm{out}}d_{\mathrm{in}}
```

个权重参数。LoRA 只需：

```math
r(d_{\mathrm{in}} + d_{\mathrm{out}})
```

个参数。二者比例为：

```math
\frac{r(d_{\mathrm{in}} + d_{\mathrm{out}})}
{d_{\mathrm{out}}d_{\mathrm{in}}}
```

若方阵维度为 `d x d`，比例进一步化为：

```math
\frac{2r}{d}
```

例如隐藏维度 `d = 4096`、rank `r = 16` 时，单个方阵投影的 LoRA 参数比例约为：

```math
\frac{2 \times 16}{4096} = 0.0078125
```

即约 `0.78%`。实际模型还包含 MLP、词嵌入和层归一化，项目只给选定层注入 LoRA，所以最终可训练占比通常更低。

## 3. `r` 和 `alpha`

`r` 是低秩空间的维度，决定 LoRA 分支的表达容量：

- `r` 小：参数和显存更少，适合简单风格、格式和小规模任务。
- `r` 大：可表达的权重更新更复杂，但训练参数、优化器状态和过拟合风险都会增加。

`alpha` 决定增量缩放：

```math
s = \frac{\alpha}{r}
```

项目默认：

```yaml
r: 16
alpha: 32
```

所以初始缩放为 `s = 2`。保持 `alpha / r` 大致稳定时，改变 rank 不会让 LoRA 分支的数值尺度突然发生巨大变化。

`dropout` 只作用于 LoRA 分支输入，训练时可写成：

```math
y = Wx + \frac{\alpha}{r}B(A(\operatorname{Dropout}(x)))
```

它有助于小数据下抑制过拟合；推理时 dropout 自动关闭。

## 4. 项目注入哪些层

默认 `config/lora.yaml`：

```yaml
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
```

它们是注意力层中的 query、key、value 和输出投影。默认选择的理由是以较少参数改变注意力读写方式，通常已经能适配指令风格和领域表达。

加入：

```text
gate_proj, up_proj, down_proj
```

会同时修改 MLP 分支，容量更大，也会增加显存、训练时间和数据质量要求。

## 5. adapter、合并和继续训练

adapter 文件保存的是 LoRA 相关参数和配置，而不是完整 `W`。推理时先加载基础模型 `W`，再加载 adapter 的 `A`、`B`。

合并时执行：

```math
W_{\mathrm{merged}} = W + \frac{\alpha}{r}BA
```

`merge_lora.py` 调用 PEFT 的 `merge_and_unload()` 完成这件事，然后把完整模型保存到 `output_dir`。

合并后方便部署，但 adapter 更适合继续训练和保存多个任务版本。原因很简单：adapter 保留了“增量”这一层结构，完整合并模型只剩最终权重，无法再区分基础权重和特定任务的更新。
