"""日志工具。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping


@dataclass(slots=True)
class LoggerConfig:
    """日志配置。

    Attributes:
        name: logger 名称。
        level: 日志级别，支持字符串或 logging 常量名。
        log_file: 可选日志文件路径。
        console: 是否输出到标准输出。
        propagate: 是否向上层 logger 传播。
    """

    name: str = "qwen3-finetune"
    level: str | int = "INFO"
    log_file: str | None = None
    console: bool = True
    propagate: bool = False


def resolve_log_level(level: str | int) -> int:
    """将字符串或整数日志级别解析为 logging 常量。

    Args:
        level: 日志级别，如 `INFO`、`DEBUG` 或数值常量。

    Returns:
        int: logging 模块可识别的级别值。

    Raises:
        ValueError: 输入字符串不是合法日志级别时抛出。
    """

    if isinstance(level, int):
        return level

    normalized_level = level.strip().upper()
    if not hasattr(logging, normalized_level):
        raise ValueError(f"不支持的日志级别: {level}")

    resolved_level = getattr(logging, normalized_level)
    if not isinstance(resolved_level, int):
        raise ValueError(f"非法日志级别: {level}")

    return resolved_level


def build_formatter() -> logging.Formatter:
    """构建统一日志格式器。

    Returns:
        logging.Formatter: 统一格式的 formatter。
    """

    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def clear_logger_handlers(logger: logging.Logger) -> None:
    """清除 logger 上已存在的 handlers。

    这样做可以避免多次初始化 logger 时重复输出同一条日志。

    Args:
        logger: 目标 logger。
    """

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logger(config: LoggerConfig) -> logging.Logger:
    """根据配置创建并初始化 logger。

    Args:
        config: 日志配置。

    Returns:
        logging.Logger: 初始化完成的 logger。
    """

    logger = logging.getLogger(config.name)
    logger.setLevel(resolve_log_level(config.level))
    logger.propagate = config.propagate

    # 先清除旧 handler，避免重复初始化后日志打印多次。
    clear_logger_handlers(logger)

    formatter = build_formatter()

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "qwen3-finetune") -> logging.Logger:
    """获取 logger。

    如果目标 logger 尚未初始化，则使用默认配置进行初始化。

    Args:
        name: logger 名称。

    Returns:
        logging.Logger: 可直接使用的 logger。
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    return setup_logger(LoggerConfig(name=name))


def log_kv(logger: logging.Logger, title: str, payload: Mapping[str, Any]) -> None:
    """按稳定 JSON 形式输出结构化字典。

    Args:
        logger: 目标 logger。
        title: 日志标题。
        payload: 需要记录的键值对数据。
    """

    formatted_payload = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    logger.info("%s:\n%s", title, formatted_payload)


def log_section(logger: logging.Logger, title: str) -> None:
    """输出简单分节标题。

    Args:
        logger: 目标 logger。
        title: 分节标题。
    """

    logger.info("=" * 20 + " %s " + "=" * 20, title)

