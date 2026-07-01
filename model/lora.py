"""LoRA 配置、注入与适配器加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig
from peft import PeftModel
from peft import TaskType
from peft import get_peft_model
from transformers import PreTrainedModel


SUPPORTED_TASK_TYPES: dict[str, TaskType] = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "SEQ_CLS": TaskType.SEQ_CLS,
    "TOKEN_CLS": TaskType.TOKEN_CLS,
    "QUESTION_ANS": TaskType.QUESTION_ANS,
    "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
}


@dataclass(slots=True)
class LoRAConfigData:
    """LoRA 配置数据类。

    Attributes:
        r: LoRA rank。
        alpha: LoRA alpha。
        dropout: LoRA dropout。
        bias: bias 训练方式。
        target_modules: 需要注入 LoRA 的线性层名称列表。
        task_type: PEFT 任务类型。
        inference_mode: 是否以推理模式创建 adapter。
    """

    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: list[str]
    task_type: str
    inference_mode: bool


def load_lora_config(config_path: str | Path) -> LoRAConfigData:
    """从 YAML 文件加载 LoRA 配置。

    Args:
        config_path: LoRA YAML 配置文件路径。

    Returns:
        LoRAConfigData: 结构化 LoRA 配置。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 配置字段缺失或取值非法时抛出。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA 配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    required_fields = {
        "r",
        "alpha",
        "dropout",
        "bias",
        "target_modules",
        "task_type",
        "inference_mode",
    }
    missing_fields = sorted(required_fields - set(payload))
    if missing_fields:
        raise ValueError(f"LoRA 配置缺少字段: {', '.join(missing_fields)}")

    target_modules = payload["target_modules"]
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("LoRA `target_modules` 必须是非空列表。")

    config = LoRAConfigData(
        r=int(payload["r"]),
        alpha=int(payload["alpha"]),
        dropout=float(payload["dropout"]),
        bias=str(payload["bias"]),
        target_modules=[str(module_name) for module_name in target_modules],
        task_type=str(payload["task_type"]).upper(),
        inference_mode=bool(payload["inference_mode"]),
    )
    validate_lora_config(config)
    return config


def validate_lora_config(config: LoRAConfigData) -> None:
    """校验 LoRA 配置的基础取值。

    Args:
        config: 已解析的 LoRA 配置。

    Raises:
        ValueError: 配置取值非法时抛出。
    """

    if config.r <= 0:
        raise ValueError("LoRA `r` 必须大于 0。")
    if config.alpha <= 0:
        raise ValueError("LoRA `alpha` 必须大于 0。")
    if not 0 <= config.dropout < 1:
        raise ValueError("LoRA `dropout` 必须在 [0, 1) 范围内。")
    if config.bias not in {"none", "all", "lora_only"}:
        raise ValueError("LoRA `bias` 必须是 none、all 或 lora_only。")


def build_peft_lora_config(config: LoRAConfigData) -> LoraConfig:
    """将项目 LoRA 配置转换为 PEFT 的 `LoraConfig`。

    Args:
        config: 项目层 LoRA 配置。

    Returns:
        LoraConfig: 可直接用于 `get_peft_model()` 的配置对象。

    Raises:
        ValueError: `task_type` 非法时抛出。
    """

    if config.task_type not in SUPPORTED_TASK_TYPES:
        supported_values = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(
            f"不支持的 task_type: {config.task_type}。可选值: {supported_values}"
        )

    return LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=config.target_modules,
        task_type=SUPPORTED_TASK_TYPES[config.task_type],
        inference_mode=config.inference_mode,
    )


def apply_lora_to_model(
    model: PreTrainedModel,
    config: LoRAConfigData,
) -> PeftModel:
    """向基础模型注入 LoRA adapter。

    Args:
        model: 基础 causal LM。
        config: LoRA 配置。

    Returns:
        PeftModel: 注入 LoRA 后的 PEFT 模型。
    """

    peft_config = build_peft_lora_config(config)
    lora_model = get_peft_model(model, peft_config)
    return lora_model


def load_lora_adapter(
    model: PreTrainedModel,
    adapter_path: str | Path,
    is_trainable: bool = False,
) -> PeftModel:
    """将已有 LoRA adapter 加载到基础模型上。

    Args:
        model: 已加载的基础模型。
        adapter_path: LoRA adapter 目录路径。
        is_trainable: 是否以可训练方式加载 adapter。

    Returns:
        PeftModel: 已加载 adapter 的 PEFT 模型。

    Raises:
        FileNotFoundError: adapter 目录不存在时抛出。
    """

    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {path}")

    return PeftModel.from_pretrained(
        model=model,
        model_id=str(path),
        is_trainable=is_trainable,
    )


def prepare_model_for_training(
    model: PreTrainedModel,
    lora_config: LoRAConfigData,
    adapter_path: str | None = None,
) -> PeftModel:
    """为训练场景准备 PEFT 模型。

    逻辑如下：

    - 若提供 `adapter_path`，则基于已有 adapter 继续训练。
    - 否则从 LoRA 配置新建 adapter 并注入模型。

    Args:
        model: 已加载基础模型。
        lora_config: LoRA 配置。
        adapter_path: 可选已有 adapter 路径。

    Returns:
        PeftModel: 可训练的 PEFT 模型。
    """

    if adapter_path:
        return load_lora_adapter(
            model=model,
            adapter_path=adapter_path,
            is_trainable=True,
        )

    return apply_lora_to_model(model=model, config=lora_config)


def summarize_lora_config(config: LoRAConfigData) -> dict[str, Any]:
    """返回 LoRA 配置摘要。

    Args:
        config: LoRA 配置。

    Returns:
        dict[str, Any]: 便于日志输出的配置摘要。
    """

    return {
        "r": config.r,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "bias": config.bias,
        "target_modules": config.target_modules,
        "task_type": config.task_type,
        "inference_mode": config.inference_mode,
    }


def summarize_trainable_parameters(model: PeftModel) -> dict[str, float | int]:
    """统计 LoRA 模型的总参数量和可训练参数量。

    Args:
        model: PEFT 模型。

    Returns:
        dict[str, float | int]: 参数量与训练占比摘要。
    """

    total_params = 0
    trainable_params = 0

    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total_params += parameter_count
        if parameter.requires_grad:
            trainable_params += parameter_count

    trainable_ratio = 0.0
    if total_params > 0:
        trainable_ratio = trainable_params / total_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
    }
