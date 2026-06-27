"""Trainer 配置与构建工具。"""

from __future__ import annotations

import inspect
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from peft import PeftModel
from transformers import PreTrainedTokenizerBase
from transformers import Trainer
from transformers import TrainingArguments

from trainer.collator import SupervisedDataCollator
from trainer.dataset import DatasetConfig
from trainer.dataset import SupervisedConversationDataset
from trainer.metrics import enrich_metrics_with_perplexity


@dataclass(slots=True)
class TrainConfig:
    """训练配置数据类。

    该数据类与 `config/train.yaml` 对齐，负责承载训练、日志、断点恢复和
    mixed precision 等参数。
    """

    model_name_or_path: str
    train_file: str
    eval_file: str | None = None
    output_dir: str = "outputs/qwen3-4b-lora"
    logging_dir: str = "logs/qwen3-4b-lora"
    cache_dir: str | None = None
    max_length: int = 2048
    system_prompt: str = "你是一个专业、可靠、简洁的中文助手。"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    torch_dtype: str = "bfloat16"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    logging_strategy: str = "steps"
    eval_on_start: bool = False
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str = "qwen3-4b-lora"
    deepspeed: str | None = None
    resume_from_checkpoint: str | None = None
    adapter_path: str | None = None
    seed: int = 42


class LoRATrainer(Trainer):
    """为 LoRA 监督微调定制的 Trainer。

    当前主要增强点：

    - 对训练与评估日志自动补充 perplexity。
    """

    def log(self, logs: dict[str, float]) -> None:
        """在原始日志基础上自动补充 perplexity 后再写入。

        Args:
            logs: Trainer 产生的日志字典。
        """

        enriched_logs = enrich_metrics_with_perplexity(logs)
        super().log(enriched_logs)


def load_train_config(config_path: str | Path) -> TrainConfig:
    """从 YAML 文件加载训练配置。

    Args:
        config_path: `train.yaml` 路径。

    Returns:
        TrainConfig: 结构化训练配置对象。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 出现未知字段时抛出。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"训练配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    valid_fields = {field_item.name for field_item in fields(TrainConfig)}
    unknown_fields = sorted(set(payload) - valid_fields)
    if unknown_fields:
        raise ValueError(f"train.yaml 中存在未知字段: {', '.join(unknown_fields)}")

    report_to = payload.get("report_to")
    if isinstance(report_to, str):
        payload["report_to"] = [report_to]

    config = TrainConfig(**payload)
    validate_train_config(config)
    return config


def validate_train_config(config: TrainConfig) -> None:
    """校验训练配置合法性。

    Args:
        config: 训练配置。

    Raises:
        ValueError: 关键配置不合法时抛出。
    """

    if config.fp16 and config.bf16:
        raise ValueError("`fp16` 与 `bf16` 不能同时为 true。")
    if config.max_length <= 0:
        raise ValueError("`max_length` 必须大于 0。")
    if config.per_device_train_batch_size <= 0:
        raise ValueError("`per_device_train_batch_size` 必须大于 0。")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` 必须大于 0。")
    if config.learning_rate <= 0:
        raise ValueError("`learning_rate` 必须大于 0。")
    if config.num_train_epochs <= 0 and config.max_steps <= 0:
        raise ValueError("`num_train_epochs` 与 `max_steps` 至少需要一个为正数。")


def train_config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """将训练配置转换为普通字典。

    Args:
        config: 训练配置对象。

    Returns:
        dict[str, Any]: 便于日志或保存的配置字典。
    """

    return asdict(config)


def build_datasets(
    tokenizer: PreTrainedTokenizerBase,
    config: TrainConfig,
) -> tuple[SupervisedConversationDataset, SupervisedConversationDataset | None]:
    """根据训练配置构建训练集和验证集。

    Args:
        tokenizer: 已加载 tokenizer。
        config: 训练配置。

    Returns:
        tuple[SupervisedConversationDataset, SupervisedConversationDataset | None]:
            训练集和可选验证集。
    """

    dataset_config = DatasetConfig(
        data_path=config.train_file,
        max_length=config.max_length,
        system_prompt=config.system_prompt,
        input_format="auto",
    )
    train_dataset = SupervisedConversationDataset(
        tokenizer=tokenizer,
        config=dataset_config,
    )

    eval_dataset: SupervisedConversationDataset | None = None
    if config.eval_file:
        eval_dataset = SupervisedConversationDataset(
            tokenizer=tokenizer,
            config=DatasetConfig(
                data_path=config.eval_file,
                max_length=config.max_length,
                system_prompt=config.system_prompt,
                input_format="auto",
            ),
        )

    return train_dataset, eval_dataset


def build_training_arguments(
    config: TrainConfig,
    has_eval_dataset: bool,
) -> TrainingArguments:
    """构建 `TrainingArguments`。

    该函数会根据当前 Transformers 版本动态过滤参数，并自动处理
    `evaluation_strategy` 与 `eval_strategy` 的命名兼容问题。

    Args:
        config: 训练配置。
        has_eval_dataset: 当前是否提供了验证集。

    Returns:
        TrainingArguments: 供 Trainer 使用的训练参数对象。
    """

    signature = inspect.signature(TrainingArguments.__init__)
    supported_args = set(signature.parameters.keys())

    evaluation_strategy = config.evaluation_strategy if has_eval_dataset else "no"
    load_best_model_at_end = config.load_best_model_at_end and has_eval_dataset

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "logging_dir": config.logging_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "save_strategy": config.save_strategy,
        "logging_strategy": config.logging_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "remove_unused_columns": config.remove_unused_columns,
        "report_to": config.report_to,
        "run_name": config.run_name,
        "deepspeed": config.deepspeed,
        "seed": config.seed,
        "eval_on_start": config.eval_on_start,
        "save_safetensors": True,
        "optim": "adamw_torch",
    }

    if "evaluation_strategy" in supported_args:
        kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in supported_args:
        kwargs["eval_strategy"] = evaluation_strategy

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_args and value is not None
    }

    return TrainingArguments(**filtered_kwargs)


def build_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: SupervisedConversationDataset,
    eval_dataset: SupervisedConversationDataset | None,
    config: TrainConfig,
) -> LoRATrainer:
    """构建 LoRA 微调 Trainer。

    Args:
        model: 已注入 LoRA 的 PEFT 模型。
        tokenizer: tokenizer 实例。
        train_dataset: 训练数据集。
        eval_dataset: 可选验证数据集。
        config: 训练配置。

    Returns:
        LoRATrainer: 已完整配置的数据整理器和训练参数的 Trainer 对象。
    """

    training_args = build_training_arguments(
        config=config,
        has_eval_dataset=eval_dataset is not None,
    )
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)

    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return LoRATrainer(**trainer_kwargs)
