"""保存工具。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Mapping

import yaml
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase


def ensure_directory(path: str | Path) -> Path:
    """确保目录存在。

    Args:
        path: 目录路径。

    Returns:
        Path: 规范化后的目录路径对象。
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_for_serialization(value: Any) -> Any:
    """将对象递归转换为可序列化结构。

    处理规则：

    - dataclass 转为字典
    - `Path` 转为字符串
    - `dict/list/tuple/set` 递归处理
    - 其他值原样返回

    Args:
        value: 任意 Python 对象。

    Returns:
        Any: 适合 JSON/YAML 输出的对象。
    """

    if is_dataclass(value):
        return normalize_for_serialization(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): normalize_for_serialization(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [normalize_for_serialization(item) for item in value]

    return value


def save_json_file(data: Any, output_path: str | Path) -> Path:
    """将数据保存为 JSON 文件。

    Args:
        data: 待保存对象。
        output_path: 目标文件路径。

    Returns:
        Path: 实际输出路径。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = normalize_for_serialization(data)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            serializable_data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        file.write("\n")

    return path


def save_yaml_file(data: Any, output_path: str | Path) -> Path:
    """将数据保存为 YAML 文件。

    Args:
        data: 待保存对象。
        output_path: 目标文件路径。

    Returns:
        Path: 实际输出路径。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = normalize_for_serialization(data)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            serializable_data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    return path


def save_text_file(content: str, output_path: str | Path) -> Path:
    """将文本内容保存到文件。

    Args:
        content: 文本内容。
        output_path: 目标文件路径。

    Returns:
        Path: 实际输出路径。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def copy_file_if_exists(
    source_path: str | Path | None,
    output_path: str | Path,
) -> Path | None:
    """若源文件存在，则复制到目标位置。

    Args:
        source_path: 源文件路径；为空时直接返回 `None`。
        output_path: 目标文件路径。

    Returns:
        Path | None: 成功复制时返回目标路径，否则返回 `None`。
    """

    if source_path is None:
        return None

    source = Path(source_path)
    if not source.exists():
        return None

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def save_config_snapshots(
    output_dir: str | Path,
    train_config: Any,
    lora_config: Any,
    train_config_source: str | Path | None = None,
    lora_config_source: str | Path | None = None,
) -> dict[str, str | None]:
    """保存训练配置、LoRA 配置及其原始文件快照。

    Args:
        output_dir: 输出目录。
        train_config: 训练配置对象或字典。
        lora_config: LoRA 配置对象或字典。
        train_config_source: 原始 `train.yaml` 路径。
        lora_config_source: 原始 `lora.yaml` 路径。

    Returns:
        dict[str, str | None]: 各快照文件路径摘要。
    """

    output_path = ensure_directory(output_dir)
    snapshot_dir = ensure_directory(output_path / "config_snapshots")

    train_snapshot = save_yaml_file(
        train_config,
        snapshot_dir / "train.snapshot.yaml",
    )
    lora_snapshot = save_yaml_file(
        lora_config,
        snapshot_dir / "lora.snapshot.yaml",
    )

    original_dir = ensure_directory(snapshot_dir / "original")
    copied_train = copy_file_if_exists(
        train_config_source,
        original_dir / "train.yaml",
    )
    copied_lora = copy_file_if_exists(
        lora_config_source,
        original_dir / "lora.yaml",
    )

    return {
        "train_snapshot": str(train_snapshot),
        "lora_snapshot": str(lora_snapshot),
        "copied_train_source": str(copied_train) if copied_train else None,
        "copied_lora_source": str(copied_lora) if copied_lora else None,
    }


def save_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
) -> Path:
    """保存 tokenizer。

    Args:
        tokenizer: 需要保存的 tokenizer。
        output_dir: 输出目录。

    Returns:
        Path: tokenizer 保存目录。
    """

    output_path = ensure_directory(output_dir)
    tokenizer.save_pretrained(str(output_path))
    return output_path


def save_full_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    safe_serialization: bool = True,
) -> dict[str, str]:
    """保存完整模型与 tokenizer。

    该函数更适合 merge LoRA 之后的完整模型导出场景。

    Args:
        model: 已合并或完整权重模型。
        tokenizer: tokenizer 实例。
        output_dir: 输出目录。
        safe_serialization: 是否保存为 safetensors。

    Returns:
        dict[str, str]: 完整模型保存结果摘要。
    """

    output_path = ensure_directory(output_dir)
    model.save_pretrained(
        save_directory=str(output_path),
        safe_serialization=safe_serialization,
    )
    tokenizer.save_pretrained(str(output_path))
    return {
        "model_dir": str(output_path),
        "tokenizer_dir": str(output_path),
    }


def save_metrics(
    metrics: Mapping[str, Any],
    output_dir: str | Path,
    filename: str = "metrics.json",
) -> Path:
    """保存训练或评估指标。

    Args:
        metrics: 指标字典。
        output_dir: 输出目录。
        filename: 指标文件名。

    Returns:
        Path: 指标文件路径。
    """

    output_path = ensure_directory(output_dir)
    return save_json_file(metrics, output_path / filename)
