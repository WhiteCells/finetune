"""监督微调数据整理器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class SupervisedDataCollator:
    """用于监督微调的动态 padding collator。

    Attributes:
        tokenizer: 用于 padding 的 tokenizer。
        label_pad_token_id: 标签 padding 值，监督学习中通常使用 `-100`。
        pad_to_multiple_of: 可选的长度对齐倍数，常用于提升 Tensor Core 利用率。
        return_tensors: 返回张量类型，默认 `pt`。
    """

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(
        self,
        features: Sequence[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        """对 batch 内样本做动态 padding。

        Args:
            features: 单条样本特征列表，每项必须包含
                `input_ids`、`attention_mask` 和 `labels`。

        Returns:
            dict[str, torch.Tensor]: padding 后的 batch 张量字典。
        """

        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            encoded_inputs=model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        max_length = int(batch["input_ids"].shape[1])
        padded_labels: list[list[int]] = []

        for feature in features:
            labels = feature["labels"]
            pad_length = max_length - len(labels)

            if self.tokenizer.padding_side == "left":
                padded = [self.label_pad_token_id] * pad_length + labels
            else:
                padded = labels + [self.label_pad_token_id] * pad_length

            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

