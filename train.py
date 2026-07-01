#!/usr/bin/env python3
"""Qwen3-4B LoRA 微调主训练入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import peft
import torch
from transformers import __version__ as transformers_version

from model.loader import ModelLoadConfig
from model.loader import load_causal_lm
from model.loader import resize_token_embeddings_if_needed
from model.loader import summarize_model
from model.lora import load_lora_config
from model.lora import prepare_model_for_training
from model.lora import summarize_lora_config
from model.lora import summarize_trainable_parameters
from model.tokenizer import TokenizerLoadConfig
from model.tokenizer import load_tokenizer
from model.tokenizer import tokenizer_summary
from trainer.metrics import enrich_metrics_with_perplexity
from trainer.trainer import build_datasets
from trainer.trainer import build_trainer
from trainer.trainer import load_train_config
from trainer.trainer import train_config_to_dict
from utils.logger import LoggerConfig
from utils.logger import get_logger
from utils.logger import log_kv
from utils.logger import log_section
from utils.logger import setup_logger
from utils.save import ensure_directory
from utils.save import save_config_snapshots
from utils.save import save_metrics
from utils.save import save_tokenizer
from utils.seed import SeedConfig
from utils.seed import seed_everything


@dataclass(slots=True)
class CLIArgs:
    """命令行参数。

    Attributes:
        train_config: 训练配置文件路径。
        lora_config: LoRA 配置文件路径。
        resume_from_checkpoint: 可选断点续训路径，优先级高于 YAML 配置。
        adapter_path: 可选已有 LoRA adapter 路径，优先级高于 YAML 配置。
        log_level: 日志级别。
        local_rank: 分布式训练保留参数，供 `torchrun` 或部分启动器透传。
    """

    train_config: str
    lora_config: str
    resume_from_checkpoint: str | None
    adapter_path: str | None
    log_level: str
    local_rank: int


def parse_args(argv: Sequence[str] | None = None) -> tuple[CLIArgs, list[str]]:
    """解析命令行参数。

    Args:
        argv: 可选参数序列；为空时默认读取命令行。

    Returns:
        tuple[CLIArgs, list[str]]:
            - 结构化命令行参数
            - 未识别参数列表
    """

    parser = argparse.ArgumentParser(description="Qwen3-4B LoRA 微调主训练脚本。")
    parser.add_argument(
        "--train-config",
        default="config/train.yaml",
        help="训练配置文件路径，默认 `config/train.yaml`。",
    )
    parser.add_argument(
        "--lora-config",
        default="config/lora.yaml",
        help="LoRA 配置文件路径，默认 `config/lora.yaml`。",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="断点续训 checkpoint 路径，优先级高于 YAML 配置。",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="已有 LoRA adapter 路径，用于继续训练或以旧 adapter 初始化训练。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，如 INFO、DEBUG、WARNING。",
    )
    parser.add_argument(
        "--local-rank",
        default=-1,
        type=int,
        help="分布式训练保留参数，可忽略。",
    )

    namespace, unknown_args = parser.parse_known_args(argv)
    cli_args = CLIArgs(
        train_config=namespace.train_config,
        lora_config=namespace.lora_config,
        resume_from_checkpoint=namespace.resume_from_checkpoint,
        adapter_path=namespace.adapter_path,
        log_level=namespace.log_level,
        local_rank=namespace.local_rank,
    )
    return cli_args, list(unknown_args)


def initialize_logger(output_dir: str, log_level: str) -> tuple[Path, object]:
    """初始化日志目录与 logger。

    Args:
        output_dir: 训练输出目录。
        log_level: 日志级别。

    Returns:
        tuple[Path, object]:
            - 输出目录路径
            - 已初始化 logger
    """

    output_path = ensure_directory(output_dir)
    logger = setup_logger(
        LoggerConfig(
            name="qwen3-finetune.train",
            level=log_level,
            log_file=str(output_path / "train.log"),
            console=True,
            propagate=False,
        )
    )
    return output_path, logger


def resolve_runtime_overrides(
    cli_args: CLIArgs,
) -> tuple[object, object]:
    """加载配置并应用命令行覆盖项。

    Args:
        cli_args: 命令行参数。

    Returns:
        tuple[object, object]:
            - 训练配置对象
            - LoRA 配置对象
    """

    train_config = load_train_config(cli_args.train_config)
    lora_config = load_lora_config(cli_args.lora_config)

    # 命令行参数优先级高于 YAML，便于快速试验和断点续训。
    if cli_args.resume_from_checkpoint:
        train_config.resume_from_checkpoint = cli_args.resume_from_checkpoint
    if cli_args.adapter_path:
        train_config.adapter_path = cli_args.adapter_path

    return train_config, lora_config


def log_environment(logger: object) -> None:
    """记录当前运行环境信息。

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


def validate_runtime_paths(train_config: object) -> None:
    """校验训练过程中涉及的重要路径。

    Args:
        train_config: 训练配置对象。

    Raises:
        FileNotFoundError: 当关键路径不存在时抛出。
    """

    model_path = Path(train_config.model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(f"基础模型路径不存在: {model_path}")

    train_file_path = Path(train_config.train_file)
    if not train_file_path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file_path}")

    if train_config.eval_file:
        eval_file_path = Path(train_config.eval_file)
        if not eval_file_path.exists():
            raise FileNotFoundError(f"验证数据文件不存在: {eval_file_path}")

    if train_config.resume_from_checkpoint:
        checkpoint_path = Path(train_config.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"恢复 checkpoint 路径不存在: {checkpoint_path}")

    if train_config.adapter_path:
        adapter_path = Path(train_config.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter 路径不存在: {adapter_path}")


def configure_model_special_tokens(model: object, tokenizer: object) -> None:
    """同步模型与 tokenizer 的 special token 配置。

    Args:
        model: 已加载模型。
        tokenizer: 已加载 tokenizer。
    """

    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id


def save_pre_training_metadata(
    trainer: object,
    tokenizer: object,
    output_dir: Path,
    train_config: object,
    lora_config: object,
    train_config_path: str,
    lora_config_path: str,
    logger: object,
) -> None:
    """在训练开始前保存元数据。

    保存时机选择在 Trainer 构建完成后，这样可以借助 `is_world_process_zero()`
    避免多进程同时写文件。

    Args:
        trainer: 已构建 Trainer。
        tokenizer: tokenizer 实例。
        output_dir: 输出目录。
        train_config: 训练配置对象。
        lora_config: LoRA 配置对象。
        train_config_path: 原始训练配置文件路径。
        lora_config_path: 原始 LoRA 配置文件路径。
        logger: 日志对象。
    """

    if not trainer.is_world_process_zero():
        return

    snapshot_paths = save_config_snapshots(
        output_dir=output_dir,
        train_config=train_config_to_dict(train_config),
        lora_config=asdict(lora_config),
        train_config_source=train_config_path,
        lora_config_source=lora_config_path,
    )
    save_tokenizer(tokenizer=tokenizer, output_dir=output_dir)
    log_kv(logger, "配置快照", snapshot_paths)


def train() -> None:
    """执行完整的 LoRA 训练流程。"""

    cli_args, unknown_args = parse_args()
    train_config, lora_config = resolve_runtime_overrides(cli_args)
    validate_runtime_paths(train_config)

    output_dir, logger = initialize_logger(
        output_dir=train_config.output_dir,
        log_level=cli_args.log_level,
    )

    if unknown_args:
        logger.warning("检测到未识别参数，将忽略: %s", " ".join(unknown_args))

    log_section(logger, "启动训练")
    log_environment(logger)
    log_kv(logger, "命令行参数", asdict(cli_args))
    log_kv(logger, "训练配置", train_config_to_dict(train_config))
    log_kv(logger, "LoRA 配置", summarize_lora_config(lora_config))

    # 固定随机种子，提升实验可复现性。
    seed_value = seed_everything(SeedConfig(seed=train_config.seed))
    logger.info("随机种子已设置为: %s", seed_value)

    tokenizer = load_tokenizer(
        TokenizerLoadConfig(
            model_name_or_path=train_config.model_name_or_path,
            cache_dir=train_config.cache_dir,
            use_fast_tokenizer=train_config.use_fast_tokenizer,
            trust_remote_code=train_config.trust_remote_code,
            model_max_length=train_config.max_length,
            padding_side="right",
            truncation_side="right",
        )
    )
    log_kv(logger, "Tokenizer 摘要", tokenizer_summary(tokenizer))

    model = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=train_config.model_name_or_path,
            cache_dir=train_config.cache_dir,
            trust_remote_code=train_config.trust_remote_code,
            torch_dtype=train_config.torch_dtype,
            attn_implementation=train_config.attn_implementation,
            gradient_checkpointing=train_config.gradient_checkpointing,
            use_cache=False,
            device_map=None,
        )
    )
    resize_token_embeddings_if_needed(model=model, tokenizer_size=len(tokenizer))
    configure_model_special_tokens(model=model, tokenizer=tokenizer)
    log_kv(logger, "基础模型摘要", summarize_model(model))

    lora_model = prepare_model_for_training(
        model=model,
        lora_config=lora_config,
        adapter_path=train_config.adapter_path,
    )
    log_kv(logger, "可训练参数摘要", summarize_trainable_parameters(lora_model))

    if hasattr(lora_model, "print_trainable_parameters"):
        # PEFT 自带的统计输出对排查“是否真的只训 LoRA 参数”很有帮助。
        lora_model.print_trainable_parameters()

    train_dataset, eval_dataset = build_datasets(
        tokenizer=tokenizer,
        config=train_config,
    )
    logger.info("训练集样本数: %s", len(train_dataset))
    if eval_dataset is not None:
        logger.info("验证集样本数: %s", len(eval_dataset))
    else:
        logger.info("未提供验证集，将跳过评估。")

    trainer = build_trainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=train_config,
    )

    save_pre_training_metadata(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_config=train_config,
        lora_config=lora_config,
        train_config_path=cli_args.train_config,
        lora_config_path=cli_args.lora_config,
        logger=logger,
    )

    log_section(logger, "开始训练")
    train_result = trainer.train(
        resume_from_checkpoint=train_config.resume_from_checkpoint,
    )
    trainer.save_state()
    trainer.save_model(str(output_dir))

    train_metrics = enrich_metrics_with_perplexity(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics: dict[str, float | int] | None = None
    if eval_dataset is not None:
        log_section(logger, "开始评估")
        eval_metrics = enrich_metrics_with_perplexity(trainer.evaluate())
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if trainer.is_world_process_zero():
        summary_metrics: dict[str, object] = {
            "train": train_metrics,
            "eval": eval_metrics,
            "output_dir": str(output_dir),
            "resume_from_checkpoint": train_config.resume_from_checkpoint,
            "adapter_path": train_config.adapter_path,
        }
        metrics_path = save_metrics(
            metrics=summary_metrics,
            output_dir=output_dir,
            filename="metrics.summary.json",
        )
        log_kv(logger, "训练结果摘要", summary_metrics)
        logger.info("训练完成，结果摘要已保存到: %s", metrics_path)


def main() -> None:
    """脚本主入口，负责统一异常处理。"""

    try:
        train()
    except Exception as error:  # noqa: BLE001
        logger = get_logger("qwen3-finetune.train")
        logger.exception("训练失败: %s", error)
        raise


if __name__ == "__main__":
    main()
