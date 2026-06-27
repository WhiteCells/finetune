"""训练与评估指标工具。"""

from __future__ import annotations

import math
from typing import Mapping


def compute_perplexity(loss_value: float | int | None) -> float | None:
    """根据 loss 计算 perplexity。

    Args:
        loss_value: 标量 loss 值。

    Returns:
        float | None: 若可计算则返回 perplexity，否则返回 `None`。
    """

    if loss_value is None:
        return None

    loss = float(loss_value)
    if math.isnan(loss) or math.isinf(loss):
        return None

    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def enrich_metrics_with_perplexity(
    metrics: Mapping[str, float | int],
) -> dict[str, float | int]:
    """为日志指标自动补充 perplexity。

    支持以下字段映射：

    - `loss` -> `perplexity`
    - `train_loss` -> `train_perplexity`
    - `eval_loss` -> `eval_perplexity`
    - `test_loss` -> `test_perplexity`

    Args:
        metrics: 原始指标字典。

    Returns:
        dict[str, float | int]: 补充 perplexity 后的新字典。
    """

    enriched_metrics = dict(metrics)
    key_mapping = {
        "loss": "perplexity",
        "train_loss": "train_perplexity",
        "eval_loss": "eval_perplexity",
        "test_loss": "test_perplexity",
    }

    for loss_key, perplexity_key in key_mapping.items():
        if loss_key not in metrics:
            continue

        perplexity = compute_perplexity(metrics[loss_key])
        if perplexity is not None:
            enriched_metrics[perplexity_key] = perplexity

    return enriched_metrics

