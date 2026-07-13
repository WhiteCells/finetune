#!/usr/bin/env python3
"""数据预处理脚本。

该脚本用于将 Alpaca、ShareGPT、OpenAI messages 风格以及部分自定义
JSON/JSONL 数据统一转换为训练阶段可直接消费的 JSONL 格式。

输出格式统一为：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

脚本支持：

1. 自动识别输入文件是 JSON 还是 JSONL。
2. 自动识别 Alpaca、ShareGPT、messages、自定义 prompt/response 格式。
3. 可选追加默认 system prompt。
4. 可选去重。
5. 可选在遇到坏样本时跳过，而不是直接终止整个流程。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence


SUPPORTED_FORMATS: tuple[str, ...] = (
    "auto",
    "alpaca",
    "sharegpt",
    "messages",
    "custom",
)


ROLE_MAPPING: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


@dataclass(slots=True)
class PreprocessConfig:
    """预处理配置。

    Attributes:
        input_path: 原始输入数据文件路径，支持 `.json` 和 `.jsonl`。
        output_path: 转换后的 JSONL 输出路径。
        input_format: 输入数据格式，`auto` 表示自动识别。
        system_prompt: 需要自动注入的 system prompt，空字符串表示不注入。
        skip_invalid: 是否在遇到坏样本时跳过，而不是直接报错退出。
        deduplicate: 是否对标准化后的输出样本去重。
        ensure_ascii: 写出 JSONL 时是否转义非 ASCII 字符。
    """

    input_path: Path
    output_path: Path
    input_format: str
    system_prompt: str
    skip_invalid: bool
    deduplicate: bool
    ensure_ascii: bool


@dataclass(slots=True)
class PreprocessStats:
    """预处理统计信息。

    Attributes:
        total_records: 输入样本总数。
        written_records: 成功写出的样本数。
        skipped_records: 被跳过的样本数。
        duplicate_records: 被去重过滤掉的样本数。
    """

    total_records: int = 0
    written_records: int = 0
    skipped_records: int = 0
    duplicate_records: int = 0


class PreprocessError(ValueError):
    """数据预处理异常。"""


INPUT_PATH = Path("data/example.jsonl")
OUTPUT_PATH = Path("data/train.jsonl")
INPUT_FORMAT = "auto"
SYSTEM_PROMPT = ""
SKIP_INVALID = True
DEDUPLICATE = True
ENSURE_ASCII = False


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """从 `.json` 或 `.jsonl` 文件中读取原始样本。

    约定：

    - `.jsonl` 文件按行读取，每行一个 JSON 对象。
    - `.json` 文件既支持单个对象，也支持对象列表。

    Args:
        input_path: 输入文件路径。

    Returns:
        list[dict[str, Any]]: 原始样本列表。

    Raises:
        FileNotFoundError: 输入文件不存在。
        PreprocessError: 文件格式非法或内容不可解析。
    """

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        return load_jsonl_records(input_path)

    if input_path.suffix.lower() == ".json":
        return load_json_records(input_path)

    raise PreprocessError(
        f"不支持的文件后缀: {input_path.suffix}，仅支持 .json 和 .jsonl"
    )


def load_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    Args:
        input_path: JSONL 文件路径。

    Returns:
        list[dict[str, Any]]: 解析后的样本列表。

    Raises:
        PreprocessError: 某一行不是合法 JSON 对象时抛出。
    """

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PreprocessError(
                    f"JSONL 第 {line_number} 行解析失败: {error}"
                ) from error

            if not isinstance(record, dict):
                raise PreprocessError(
                    f"JSONL 第 {line_number} 行不是 JSON 对象，而是 {type(record).__name__}"
                )

            records.append(record)

    return records


def load_json_records(input_path: Path) -> list[dict[str, Any]]:
    """读取 JSON 文件。

    Args:
        input_path: JSON 文件路径。

    Returns:
        list[dict[str, Any]]: 解析后的样本列表。

    Raises:
        PreprocessError: 根节点既不是对象也不是对象列表时抛出。
    """

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        return [payload]

    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise PreprocessError(
                    f"JSON 第 {index} 个元素不是对象，而是 {type(item).__name__}"
                )
            records.append(item)
        return records

    raise PreprocessError(
        f"JSON 根节点必须是对象或对象列表，当前为 {type(payload).__name__}"
    )


def normalize_records(
    records: Sequence[dict[str, Any]],
    config: PreprocessConfig,
) -> tuple[list[dict[str, Any]], PreprocessStats]:
    """将原始样本标准化为统一 messages 结构。

    Args:
        records: 原始样本序列。
        config: 预处理配置。

    Returns:
        tuple[list[dict[str, Any]], PreprocessStats]:
            - 标准化后的样本列表
            - 处理统计信息
    """

    stats = PreprocessStats(total_records=len(records))
    normalized_records: list[dict[str, Any]] = []
    seen_serialized: set[str] = set()

    for index, record in enumerate(records):
        try:
            normalized = normalize_record(
                record=record,
                input_format=config.input_format,
                system_prompt=config.system_prompt,
            )

            if config.deduplicate:
                serialized = stable_serialize(normalized)
                if serialized in seen_serialized:
                    stats.duplicate_records += 1
                    continue
                seen_serialized.add(serialized)

            normalized_records.append(normalized)
            stats.written_records += 1
        except PreprocessError as error:
            if not config.skip_invalid:
                raise
            stats.skipped_records += 1
            print(f"[WARN] 跳过第 {index + 1} 条坏样本: {error}")

    return normalized_records, stats


def normalize_record(
    record: dict[str, Any],
    input_format: str,
    system_prompt: str,
) -> dict[str, Any]:
    """将单条原始样本转换为统一 messages 结构。

    Args:
        record: 原始样本对象。
        input_format: 输入格式，可为 `auto` 或显式格式。
        system_prompt: 需要自动注入的 system prompt。

    Returns:
        dict[str, Any]: 标准化后的样本。

    Raises:
        PreprocessError: 当样本字段不完整或格式无法识别时抛出。
    """

    if input_format not in SUPPORTED_FORMATS:
        supported_values = ", ".join(SUPPORTED_FORMATS)
        raise PreprocessError(
            f"不支持的输入格式: {input_format}。可选值: {supported_values}"
        )

    resolved_format = detect_format(record) if input_format == "auto" else input_format

    if resolved_format == "alpaca":
        messages = normalize_alpaca_record(record)
    elif resolved_format == "sharegpt":
        messages = normalize_sharegpt_record(record)
    elif resolved_format == "messages":
        messages = normalize_messages_record(record)
    elif resolved_format == "custom":
        messages = normalize_custom_record(record)
    else:
        raise PreprocessError(f"不支持的样本格式: {resolved_format}")

    messages = inject_system_prompt(messages=messages, system_prompt=system_prompt)
    validate_messages(messages)
    return {"messages": messages}


def detect_format(record: dict[str, Any]) -> str:
    """自动识别单条样本格式。

    Args:
        record: 原始样本对象。

    Returns:
        str: 识别到的格式名。

    Raises:
        PreprocessError: 无法识别格式时抛出。
    """

    if {"instruction", "output"}.issubset(record.keys()):
        return "alpaca"

    if "conversations" in record:
        return "sharegpt"

    if "messages" in record:
        return "messages"

    if {"prompt", "response"}.issubset(record.keys()):
        return "custom"

    if {"input", "output"}.issubset(record.keys()):
        return "custom"

    raise PreprocessError("无法自动识别样本格式，请修改 INPUT_FORMAT 后重试。")


def normalize_alpaca_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """标准化 Alpaca 样本。

    Args:
        record: Alpaca 样本，要求包含 `instruction` 和 `output`。

    Returns:
        list[dict[str, str]]: 统一 messages 列表。

    Raises:
        PreprocessError: 字段缺失或为空时抛出。
    """

    instruction = normalize_text(record.get("instruction"))
    input_text = normalize_text(record.get("input", ""))
    output_text = normalize_text(record.get("output"))

    if not instruction:
        raise PreprocessError("Alpaca 样本缺少 instruction。")
    if not output_text:
        raise PreprocessError("Alpaca 样本缺少 output。")

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]


def normalize_sharegpt_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """标准化 ShareGPT 样本。

    Args:
        record: ShareGPT 样本，要求包含 `conversations` 字段。

    Returns:
        list[dict[str, str]]: 统一 messages 列表。

    Raises:
        PreprocessError: 对话结构非法时抛出。
    """

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise PreprocessError("ShareGPT 样本缺少 conversations 列表。")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(conversations):
        if not isinstance(item, dict):
            raise PreprocessError(
                f"ShareGPT 第 {index} 个对话单元不是对象，而是 {type(item).__name__}"
            )

        raw_role = normalize_text(item.get("from"))
        content = normalize_text(item.get("value"))
        role = ROLE_MAPPING.get(raw_role.lower(), "")

        if not role:
            raise PreprocessError(f"ShareGPT 中存在未知角色: {raw_role!r}")
        if not content:
            raise PreprocessError("ShareGPT 对话内容为空。")

        messages.append({"role": role, "content": content})

    return messages


def normalize_messages_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """标准化 OpenAI 风格 messages 样本。

    Args:
        record: 包含 `messages` 的样本对象。

    Returns:
        list[dict[str, str]]: 统一 messages 列表。

    Raises:
        PreprocessError: messages 字段非法时抛出。
    """

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise PreprocessError("messages 样本缺少 messages 列表。")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise PreprocessError(
                f"messages 第 {index} 项不是对象，而是 {type(item).__name__}"
            )

        role = normalize_text(item.get("role")).lower()
        content = normalize_text(item.get("content"))

        if role not in {"system", "user", "assistant"}:
            raise PreprocessError(f"messages 中存在不支持的角色: {role!r}")
        if not content:
            raise PreprocessError("messages 中存在空内容。")

        messages.append({"role": role, "content": content})

    return messages


def normalize_custom_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """标准化自定义样本。

    当前支持两类常见自定义结构：

    1. `{"prompt": "...", "response": "..."}`
    2. `{"input": "...", "output": "..."}`

    Args:
        record: 自定义样本对象。

    Returns:
        list[dict[str, str]]: 统一 messages 列表。

    Raises:
        PreprocessError: 无法提取有效 user/assistant 内容时抛出。
    """

    prompt = normalize_text(record.get("prompt"))
    response = normalize_text(record.get("response"))
    input_text = normalize_text(record.get("input"))
    output_text = normalize_text(record.get("output"))

    if prompt and response:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    if input_text and output_text:
        return [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]

    raise PreprocessError("自定义样本无法识别为 prompt/response 或 input/output。")


def inject_system_prompt(
    messages: Sequence[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """为样本按需注入 system prompt。

    注入规则：

    - `system_prompt` 为空时，不做处理。
    - 样本首条消息已是 `system` 时，不重复插入。
    - 否则在最前面追加一条 `system` 消息。

    Args:
        messages: 原始消息序列。
        system_prompt: 需注入的 system 提示词。

    Returns:
        list[dict[str, str]]: 注入后的消息列表。
    """

    normalized_messages = list(messages)
    if not system_prompt:
        return normalized_messages

    if normalized_messages and normalized_messages[0]["role"] == "system":
        return normalized_messages

    return [{"role": "system", "content": system_prompt}, *normalized_messages]


def validate_messages(messages: Sequence[dict[str, str]]) -> None:
    """校验标准化后的 messages 是否可用于监督微调。

    校验重点：

    - 消息列表不能为空。
    - 至少存在一条 assistant 消息。
    - 最后一条消息必须是 assistant，保证存在监督目标。

    Args:
        messages: 标准化消息序列。

    Raises:
        PreprocessError: 当 messages 不符合训练约束时抛出。
    """

    if not messages:
        raise PreprocessError("标准化后的 messages 为空。")

    if not any(message["role"] == "assistant" for message in messages):
        raise PreprocessError("样本中不存在 assistant 消息。")

    if messages[-1]["role"] != "assistant":
        raise PreprocessError("最后一条消息必须是 assistant，才能形成监督目标。")


def normalize_text(value: Any) -> str:
    """将任意输入值标准化为干净文本。

    Args:
        value: 原始字段值。

    Returns:
        str: 去除首尾空白后的字符串；若值为空则返回空字符串。
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def stable_serialize(record: dict[str, Any]) -> str:
    """将标准化样本序列化为稳定字符串，用于去重。

    Args:
        record: 标准化后的样本对象。

    Returns:
        str: 键顺序稳定的 JSON 字符串。
    """

    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def write_jsonl(
    output_path: Path,
    records: Iterable[dict[str, Any]],
    ensure_ascii: bool,
) -> None:
    """将标准化样本写出为 JSONL 文件。

    Args:
        output_path: 输出 JSONL 文件路径。
        records: 需要写出的样本迭代器。
        ensure_ascii: 是否转义非 ASCII 字符。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            # 使用紧凑 JSON，便于后续训练和版本管理。
            line = json.dumps(record, ensure_ascii=ensure_ascii, separators=(",", ":"))
            file.write(f"{line}\n")


def iter_preview(records: Sequence[dict[str, Any]], limit: int = 2) -> Iterator[str]:
    """生成标准化结果的预览文本。

    Args:
        records: 标准化样本序列。
        limit: 最多预览多少条。

    Yields:
        str: 单条样本的格式化预览字符串。
    """

    for record in records[:limit]:
        yield json.dumps(record, ensure_ascii=False, indent=2)


def main() -> None:
    """脚本主入口。

    处理流程：

    修改本文件顶部的路径和选项后直接运行本脚本。
    """

    config = PreprocessConfig(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        input_format=INPUT_FORMAT,
        system_prompt=SYSTEM_PROMPT,
        skip_invalid=SKIP_INVALID,
        deduplicate=DEDUPLICATE,
        ensure_ascii=ENSURE_ASCII,
    )
    run_preprocess(config)


def run_preprocess(config: PreprocessConfig) -> PreprocessStats:
    """执行数据预处理并返回统计信息。"""

    raw_records = load_records(config.input_path)
    normalized_records, stats = normalize_records(
        records=raw_records,
        config=config,
    )
    write_jsonl(
        output_path=config.output_path,
        records=normalized_records,
        ensure_ascii=config.ensure_ascii,
    )

    print("[INFO] 预处理完成。")
    print(f"[INFO] 输入样本数: {stats.total_records}")
    print(f"[INFO] 写出样本数: {stats.written_records}")
    print(f"[INFO] 跳过样本数: {stats.skipped_records}")
    print(f"[INFO] 去重丢弃数: {stats.duplicate_records}")
    print(f"[INFO] 输出文件: {config.output_path}")

    for index, preview in enumerate(iter_preview(normalized_records), start=1):
        print(f"[INFO] 预览样本 {index}:")
        print(preview)

    return stats


if __name__ == "__main__":
    main()
