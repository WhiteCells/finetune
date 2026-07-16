"""基础模型加载工具。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM
from transformers import __version__ as transformers_version
from transformers import PreTrainedModel
from transformers.utils import is_flash_attn_2_available


SUPPORTED_DTYPES: dict[str, torch.dtype | str] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto",
}


@dataclass(slots=True)
class ModelLoadConfig:
    """基础模型加载配置。

    Attributes:
        model_name_or_path: 基座模型路径或模型名。
        cache_dir: 可选缓存目录。
        trust_remote_code: 是否允许 Transformers 加载自定义模型代码。
        torch_dtype: 目标 dtype，可传 `bf16`、`fp16`、`float32` 或 `auto`。
        attn_implementation: attention 实现，如 `sdpa`、`flash_attention_2`。
        use_cache: 是否启用 KV cache。训练时通常应关闭，推理时开启。
        device_map: 可选设备映射；训练默认 `None`，推理可用 `auto`。
    """

    model_name_or_path: str
    cache_dir: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = "sdpa"
    use_cache: bool = False
    device_map: str | dict[str, int | str] | None = None


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """将字符串 dtype 解析为 `torch.dtype`。

    Args:
        dtype_name: 配置文件中的 dtype 名称。

    Returns:
        torch.dtype | str: 对应 dtype；`auto` 原样交给 Transformers 自动判断。

    Raises:
        ValueError: 输入 dtype 不受支持时抛出。
    """

    normalized_dtype = dtype_name.strip().lower()
    if normalized_dtype not in SUPPORTED_DTYPES:
        supported_values = ", ".join(sorted(SUPPORTED_DTYPES))
        raise ValueError(f"不支持的 torch_dtype: {dtype_name}。可选值: {supported_values}")
    return SUPPORTED_DTYPES[normalized_dtype]


def model_dtype_kwarg_name() -> str:
    """返回当前 Transformers 版本推荐的模型 dtype 参数名。"""

    major_version = int(transformers_version.split(".", maxsplit=1)[0])
    if major_version >= 5:
        return "dtype"
    return "torch_dtype"


def validate_attention_backend(attn_implementation: str | None) -> None:
    """在加载权重前检查可选 attention 后端是否可用。"""

    if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
        raise RuntimeError(
            "`flash_attention_2` 需要安装 `flash-attn` 并使用兼容的 CUDA/PyTorch "
            "环境。当前环境未检测到可用后端；请安装并验证后再启用，或使用 `eager`。"
        )


def load_causal_lm(config: ModelLoadConfig) -> PreTrainedModel:
    """加载 Qwen3 因果语言模型。

    该函数会：

    1. 解析并设置目标 dtype。
    2. 加载基础模型。
    3. 关闭或开启 `use_cache`。

    梯度检查点统一由 Trainer 在训练开始时启用，避免加载器与 Trainer
    用不同参数重复配置同一个模型。

    Args:
        config: 模型加载配置。

    Returns:
        PreTrainedModel: 已完成基础训练设置的因果语言模型。
    """

    torch_dtype = resolve_torch_dtype(config.torch_dtype)
    validate_attention_backend(config.attn_implementation)

    model_kwargs = {
        "pretrained_model_name_or_path": config.model_name_or_path,
        "cache_dir": config.cache_dir,
        "trust_remote_code": config.trust_remote_code,
        model_dtype_kwarg_name(): torch_dtype,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # 训练时通常关闭 cache，避免额外显存占用并兼容 gradient checkpointing。
    model.config.use_cache = config.use_cache

    return model


def resize_token_embeddings_if_needed(
    model: PreTrainedModel,
    tokenizer_size: int,
) -> None:
    """在 tokenizer 词表大于模型 embedding 时同步扩容。

    Qwen 等模型的配置词表可能包含预留槽位，常见表现是模型 embedding
    大于 `len(tokenizer)`。这种情况下不能缩小 embedding，否则 PEFT 会把
    embedding 视为训练中被 resize 并随 adapter 一起保存，显著放大产物体积。

    Args:
        model: 已加载模型。
        tokenizer_size: 当前 tokenizer 词表大小。
    """

    current_embeddings = model.get_input_embeddings()
    if current_embeddings is None:
        return

    current_vocab_size = current_embeddings.num_embeddings
    if tokenizer_size > current_vocab_size:
        model.resize_token_embeddings(tokenizer_size)


def summarize_model(model: PreTrainedModel) -> dict[str, object]:
    """返回模型关键信息摘要。

    Args:
        model: 已加载模型。

    Returns:
        dict[str, object]: 可直接用于日志记录的摘要信息。
    """

    hidden_size = getattr(model.config, "hidden_size", None)
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    num_attention_heads = getattr(model.config, "num_attention_heads", None)

    return {
        "model_class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "vocab_size": getattr(model.config, "vocab_size", None),
        "use_cache": getattr(model.config, "use_cache", None),
    }
