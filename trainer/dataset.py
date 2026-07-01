"""监督微调数据集实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

from torch.utils.data import Dataset

from data.preprocess import PreprocessError
from data.preprocess import load_records
from data.preprocess import normalize_record
from model.tokenizer import TokenizerType
from model.tokenizer import apply_chat_template


IGNORE_INDEX = -100


@dataclass(slots=True)
class DatasetConfig:
    """数据集构建配置。

    Attributes:
        data_path: 训练或验证数据路径，支持 `.json` 和 `.jsonl`。
        max_length: 单条样本最大 token 长度。
        system_prompt: 当样本没有 system 消息时，自动注入的系统提示词。
        input_format: 数据格式，默认 `auto` 自动识别。
    """

    data_path: str
    max_length: int
    system_prompt: str = ""
    input_format: str = "auto"


@dataclass(slots=True)
class TokenizedSample:
    """单条样本的编码结果。

    Attributes:
        input_ids: 模型输入 token id 列表。
        attention_mask: attention mask 列表。
        labels: 训练监督标签列表，非监督位置使用 `-100`。
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

    def to_dict(self) -> dict[str, list[int]]:
        """将数据类转为 Trainer 可直接消费的字典。"""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


class SupervisedConversationDataset(Dataset[dict[str, list[int]]]):
    """用于 LoRA 指令微调的监督数据集。

    该数据集支持直接读取：

    - Alpaca：`instruction/input/output`
    - ShareGPT：`conversations`
    - OpenAI 风格：`messages`
    - 自定义：`prompt/response`、`input/output`

    数据集内部会先统一标准化为 `messages`，再使用 tokenizer 的 chat template
    渲染文本，并且只对 assistant 片段计算 loss。
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        config: DatasetConfig,
    ) -> None:
        """初始化数据集。

        Args:
            tokenizer: 已加载好的 tokenizer。
            config: 数据集构建配置。

        Raises:
            FileNotFoundError: 数据文件不存在时抛出。
            ValueError: 数据文件为空或 `max_length` 非法时抛出。
        """

        if config.max_length <= 0:
            raise ValueError("`max_length` 必须大于 0。")

        self.tokenizer = tokenizer
        self.config = config
        self.data_path = Path(config.data_path)
        self.records: list[dict[str, Any]] = load_records(self.data_path)

        if not self.records:
            raise ValueError(f"数据文件为空: {self.data_path}")

    def __len__(self) -> int:
        """返回样本数量。"""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """按索引返回单条训练样本。

        Args:
            index: 样本索引。

        Returns:
            dict[str, list[int]]: 包含 `input_ids`、`attention_mask` 和 `labels`
            的训练特征字典。
        """

        try:
            raw_record = self.records[index]
            normalized_record = normalize_record(
                record=raw_record,
                input_format=self.config.input_format,
                system_prompt=self.config.system_prompt,
            )
            messages = normalized_record["messages"]

            tokenized = tokenize_messages_for_training(
                tokenizer=self.tokenizer,
                messages=messages,
                max_length=self.config.max_length,
            )
        except (PreprocessError, ValueError) as error:
            raise ValueError(
                f"{self.data_path} 第 {index + 1} 条样本处理失败: {error}"
            ) from error
        return tokenized.to_dict()


def tokenize_messages_for_training(
    tokenizer: TokenizerType,
    messages: Sequence[dict[str, str]],
    max_length: int,
) -> TokenizedSample:
    """将标准化消息转换为监督微调样本。

    处理策略：

    1. 逐条消息递增渲染 chat template。
    2. 使用当前消息渲染结果与上一步结果做差分，定位每条消息对应的 token span。
    3. 对 `assistant` 片段保留原始标签，对 `system/user` 片段全部置为 `-100`。
    4. 若常规截断会把所有监督信号裁掉，则自动回退到保留尾部 token，
       尽量保住 assistant 回复。

    Args:
        tokenizer: 已加载 tokenizer。
        messages: 规范化后的对话消息序列。
        max_length: 单条样本最大 token 长度。

    Returns:
        TokenizedSample: 编码完成的监督样本。

    Raises:
        ValueError: 当样本在截断后完全失去监督信号时抛出。
    """

    full_input_ids, full_labels = build_token_spans(
        tokenizer=tokenizer,
        messages=messages,
    )

    if not full_input_ids:
        raise ValueError("样本编码后为空，请检查数据或 chat template。")

    input_ids, labels = truncate_sample(
        input_ids=full_input_ids,
        labels=full_labels,
        max_length=max_length,
        truncation_side=tokenizer.truncation_side or "right",
    )

    # 如果常规截断把所有 assistant 监督信号都裁掉了，则优先保留尾部 token。
    if all(label == IGNORE_INDEX for label in labels) and len(full_input_ids) > max_length:
        input_ids = full_input_ids[-max_length:]
        labels = full_labels[-max_length:]

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            "截断后样本不包含任何可训练标签，请增大 `max_length` 或清洗超长样本。"
        )

    attention_mask = [1] * len(input_ids)
    return TokenizedSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def build_token_spans(
    tokenizer: TokenizerType,
    messages: Sequence[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """按消息级别构造输入 token 与监督标签。

    该函数通过“逐步渲染 + 前缀差分”的方式，将整段对话拆成多个消息片段的 token。
    这样可以准确知道每条消息新增了哪些 token，并据此决定是否计算 loss。

    Args:
        tokenizer: tokenizer 实例。
        messages: 标准化后的消息序列。

    Returns:
        tuple[list[int], list[int]]:
            - 整条样本的 `input_ids`
            - 与之对齐的 `labels`
    """

    previous_ids: list[int] = []
    full_input_ids: list[int] = []
    full_labels: list[int] = []

    for end_index, message in enumerate(messages, start=1):
        partial_messages = messages[:end_index]
        rendered_text = apply_chat_template(
            tokenizer=tokenizer,
            messages=partial_messages,
            add_generation_prompt=False,
        )
        current_ids = tokenizer(
            rendered_text,
            add_special_tokens=False,
        )["input_ids"]

        if len(current_ids) < len(previous_ids):
            raise ValueError("chat template 渲染结果异常，当前 token 长度小于前缀长度。")

        message_ids = current_ids[len(previous_ids) :]
        previous_ids = current_ids

        if not message_ids:
            continue

        full_input_ids.extend(message_ids)

        if message["role"] == "assistant":
            # 仅对 assistant 片段计算 loss。
            full_labels.extend(message_ids)
        else:
            # system 与 user 片段全部 mask 掉。
            full_labels.extend([IGNORE_INDEX] * len(message_ids))

    return full_input_ids, full_labels


def truncate_sample(
    input_ids: Sequence[int],
    labels: Sequence[int],
    max_length: int,
    truncation_side: str,
) -> tuple[list[int], list[int]]:
    """按指定方向截断样本。

    Args:
        input_ids: 原始输入 token 序列。
        labels: 对齐后的标签序列。
        max_length: 最大长度。
        truncation_side: 截断方向，支持 `left` 和 `right`。

    Returns:
        tuple[list[int], list[int]]: 截断后的 `input_ids` 与 `labels`。
    """

    if len(input_ids) <= max_length:
        return list(input_ids), list(labels)

    if truncation_side == "left":
        return list(input_ids[-max_length:]), list(labels[-max_length:])

    return list(input_ids[:max_length]), list(labels[:max_length])
