#!/usr/bin/env python3
"""Qwen3-4B LoRA 权重合并脚本。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import peft
import torch
from peft import PeftModel
from transformers import __version__ as transformers_version
from transformers import PreTrainedModel

from model.loader import ModelLoadConfig
from model.loader import load_causal_lm
from model.loader import resize_token_embeddings_if_needed
from model.loader import summarize_model
from model.lora import load_lora_adapter
from model.lora import validate_lora_adapter_path
from model.tokenizer import TokenizerLoadConfig
from model.tokenizer import TokenizerType
from model.tokenizer import load_tokenizer
from model.tokenizer import tokenizer_summary
from utils.logger import LoggerConfig
from utils.logger import get_logger
from utils.logger import log_kv
from utils.logger import log_section
from utils.logger import setup_logger
from utils.save import ensure_directory
from utils.save import save_full_model
from utils.save import save_json_file


@dataclass(slots=True)
class MergeArgs:
    """LoRA 合并命令行参数。

    Attributes:
        model_name_or_path: 基座模型路径或名称。
        adapter_path: LoRA adapter 目录路径。
        output_dir: 合并后的完整模型输出目录。
        cache_dir: 可选缓存目录。
        trust_remote_code: 是否允许加载自定义模型代码。
        use_fast_tokenizer: 是否优先使用 fast tokenizer。
        torch_dtype: 加载模型时使用的 dtype。
        attn_implementation: attention 实现方式。
        device_map: 模型设备映射。
        safe_serialization: 是否保存为 safetensors。
        log_level: 日志级别。
    """

    model_name_or_path: str
    adapter_path: str
    output_dir: str
    cache_dir: str | None
    trust_remote_code: bool
    use_fast_tokenizer: bool
    torch_dtype: str
    attn_implementation: str | None
    device_map: str
    safe_serialization: bool
    log_level: str


def parse_args(argv: Sequence[str] | None = None) -> MergeArgs:
    """解析命令行参数。

    Args:
        argv: 可选参数序列；为空时读取当前进程命令行。

    Returns:
        MergeArgs: 结构化合并参数。
    """

    parser = argparse.ArgumentParser(description="Qwen3-4B LoRA 合并脚本。")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="基座模型目录或模型名。",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="LoRA adapter 目录路径。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="合并后的完整模型输出目录。",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="可选缓存目录。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否允许 Transformers 加载自定义模型代码。",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否优先使用 fast tokenizer。",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="模型加载 dtype，如 bfloat16、float16、float32 或 auto。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="attention 实现方式，如 sdpa 或 flash_attention_2。",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="模型 device_map，默认 `auto`。",
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 safetensors 格式保存合并后的完整模型。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，如 INFO、DEBUG、WARNING。",
    )

    namespace = parser.parse_args(argv)
    return MergeArgs(
        model_name_or_path=namespace.model_name_or_path,
        adapter_path=namespace.adapter_path,
        output_dir=namespace.output_dir,
        cache_dir=namespace.cache_dir,
        trust_remote_code=namespace.trust_remote_code,
        use_fast_tokenizer=namespace.use_fast_tokenizer,
        torch_dtype=namespace.torch_dtype,
        attn_implementation=namespace.attn_implementation,
        device_map=namespace.device_map,
        safe_serialization=namespace.safe_serialization,
        log_level=namespace.log_level,
    )


def initialize_logger(output_dir: str, log_level: str) -> tuple[Path, object]:
    """初始化输出目录与 logger。

    Args:
        output_dir: 合并输出目录。
        log_level: 日志级别。

    Returns:
        tuple[Path, object]:
            - 输出目录路径
            - 已初始化 logger
    """

    output_path = ensure_directory(output_dir)
    logger = setup_logger(
        LoggerConfig(
            name="qwen3-finetune.merge",
            level=log_level,
            log_file=str(output_path / "merge.log"),
            console=True,
            propagate=False,
        )
    )
    return output_path, logger


def validate_args(args: MergeArgs) -> None:
    """校验合并参数。

    Args:
        args: 合并参数。

    Raises:
        FileNotFoundError: 基础模型或 adapter 路径不存在时抛出。
        ValueError: 输出目录非法时抛出。
    """

    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"基础模型路径不存在: {args.model_name_or_path}")

    validate_lora_adapter_path(args.adapter_path)

    output_path = Path(args.output_dir)
    if output_path.resolve() == Path(args.adapter_path).resolve():
        raise ValueError("`output_dir` 不能与 `adapter_path` 相同，避免覆盖 adapter。")


def log_environment(logger: object) -> None:
    """记录当前运行环境。

    Args:
        logger: 日志对象。
    """

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(device_index)
            for device_index in range(torch.cuda.device_count())
        ]

    payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers_version,
        "peft": getattr(peft, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
    }
    log_kv(logger, "运行环境", payload)


def configure_model_special_tokens(model: object, tokenizer: TokenizerType) -> None:
    """同步模型与 tokenizer 的 special token 配置。

    Args:
        model: 已加载模型。
        tokenizer: tokenizer 实例。
    """

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id


def merge_and_unload_model(model: PeftModel) -> PreTrainedModel:
    """执行 LoRA 权重合并。

    Args:
        model: 已加载 base model + LoRA adapter 的 PEFT 模型。

    Returns:
        PreTrainedModel: 合并后的完整模型。
    """

    # merge_and_unload 会把 LoRA 增量写回基础权重，并返回普通 Transformers 模型。
    merged_model = model.merge_and_unload()
    return merged_model


def save_merge_summary(
    output_dir: Path,
    args: MergeArgs,
    base_model_summary: dict[str, object],
    merged_model_summary: dict[str, object],
    tokenizer_info: dict[str, object],
) -> Path:
    """保存合并摘要信息。

    Args:
        output_dir: 输出目录。
        args: 合并参数。
        base_model_summary: 合并前基础模型摘要。
        merged_model_summary: 合并后模型摘要。
        tokenizer_info: tokenizer 摘要。

    Returns:
        Path: 摘要文件路径。
    """

    payload = {
        "merge_args": asdict(args),
        "base_model_summary": base_model_summary,
        "merged_model_summary": merged_model_summary,
        "tokenizer_summary": tokenizer_info,
    }
    return save_json_file(payload, output_dir / "merge_summary.json")


def run_merge(args: MergeArgs) -> dict[str, str]:
    """执行完整 LoRA 合并流程。

    Args:
        args: 合并参数。

    Returns:
        dict[str, str]: 合并后产物摘要。
    """

    validate_args(args)
    output_dir, logger = initialize_logger(args.output_dir, args.log_level)

    log_section(logger, "启动合并")
    log_environment(logger)
    log_kv(logger, "合并参数", asdict(args))

    tokenizer = load_tokenizer(
        TokenizerLoadConfig(
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
            model_max_length=None,
            padding_side="right",
            truncation_side="right",
        )
    )
    tokenizer_info = tokenizer_summary(tokenizer)
    log_kv(logger, "Tokenizer 摘要", tokenizer_info)

    base_model = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            gradient_checkpointing=False,
            use_cache=True,
            device_map=args.device_map,
        )
    )
    resize_token_embeddings_if_needed(model=base_model, tokenizer_size=len(tokenizer))
    configure_model_special_tokens(model=base_model, tokenizer=tokenizer)
    base_model_info = summarize_model(base_model)
    log_kv(logger, "基础模型摘要", base_model_info)

    peft_model = load_lora_adapter(
        model=base_model,
        adapter_path=args.adapter_path,
        is_trainable=False,
    )
    logger.info("LoRA adapter 已加载，准备执行 merge_and_unload()。")

    merged_model = merge_and_unload_model(peft_model)
    configure_model_special_tokens(model=merged_model, tokenizer=tokenizer)
    merged_model_info = summarize_model(merged_model)
    log_kv(logger, "合并后模型摘要", merged_model_info)

    save_result = save_full_model(
        model=merged_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        safe_serialization=args.safe_serialization,
    )
    summary_path = save_merge_summary(
        output_dir=output_dir,
        args=args,
        base_model_summary=base_model_info,
        merged_model_summary=merged_model_info,
        tokenizer_info=tokenizer_info,
    )

    logger.info("LoRA 权重合并完成。")
    logger.info("完整模型已保存到: %s", save_result["model_dir"])
    logger.info("合并摘要已保存到: %s", summary_path)
    return save_result


def main() -> None:
    """脚本主入口，负责统一异常处理。"""

    try:
        run_merge(parse_args())
    except Exception as error:  # noqa: BLE001
        logger = get_logger("qwen3-finetune.merge")
        logger.exception("合并失败: %s", error)
        raise


if __name__ == "__main__":
    main()
