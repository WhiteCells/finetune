#!/usr/bin/env python3
"""将 LoRA adapter 合并为普通 Transformers 模型。

修改 `RUN_CONFIG` 后，直接运行：

    uv run merge_lora.py
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

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
class MergeConfig:
    """LoRA 合并所需的三个路径。"""

    model_name_or_path: str
    adapter_path: str
    output_dir: str


RUN_CONFIG = MergeConfig(
    model_name_or_path="./models/Qwen3-4B-Instruct-2507",
    adapter_path="outputs/qwen3-4b-lora",
    output_dir="outputs/qwen3-4b-merged",
)


def initialize_logger(output_dir: str) -> tuple[Path, object]:
    """创建输出目录并初始化日志。"""

    output_path = ensure_directory(output_dir)
    logger = setup_logger(
        LoggerConfig(
            name="qwen3-finetune.merge",
            level="INFO",
            log_file=str(output_path / "merge.log"),
            console=True,
            propagate=False,
        )
    )
    return output_path, logger


def validate_config(config: MergeConfig) -> None:
    """校验基础模型、adapter 和输出目录。"""

    if not Path(config.model_name_or_path).exists():
        raise FileNotFoundError(f"基础模型路径不存在: {config.model_name_or_path}")

    adapter_path = validate_lora_adapter_path(config.adapter_path)
    if Path(config.output_dir).resolve() == adapter_path.resolve():
        raise ValueError("`output_dir` 不能与 `adapter_path` 相同。")


def configure_model_special_tokens(model: object, tokenizer: TokenizerType) -> None:
    """同步模型与 tokenizer 的 special token 配置。"""

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id


def run_merge(config: MergeConfig) -> dict[str, str]:
    """加载基座模型与 adapter，合并后保存完整模型。"""

    validate_config(config)
    output_dir, logger = initialize_logger(config.output_dir)
    log_section(logger, "启动合并")
    log_kv(logger, "合并配置", asdict(config))

    tokenizer = load_tokenizer(
        TokenizerLoadConfig(
            model_name_or_path=config.model_name_or_path,
            model_max_length=None,
        )
    )
    tokenizer_info = tokenizer_summary(tokenizer)
    log_kv(logger, "Tokenizer 摘要", tokenizer_info)

    base_model = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=config.model_name_or_path,
            use_cache=True,
            device_map="auto",
        )
    )
    resize_token_embeddings_if_needed(base_model, tokenizer_size=len(tokenizer))
    configure_model_special_tokens(base_model, tokenizer)
    base_model_info = summarize_model(base_model)
    log_kv(logger, "基础模型摘要", base_model_info)

    peft_model = load_lora_adapter(
        model=base_model,
        adapter_path=config.adapter_path,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    configure_model_special_tokens(merged_model, tokenizer)

    result = save_full_model(
        model=merged_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        safe_serialization=True,
    )
    summary_path = save_json_file(
        {
            "merge_config": asdict(config),
            "base_model_summary": base_model_info,
            "merged_model_summary": summarize_model(merged_model),
            "tokenizer_summary": tokenizer_info,
        },
        output_dir / "merge_summary.json",
    )
    logger.info("完整模型已保存到: %s", result["model_dir"])
    logger.info("合并摘要已保存到: %s", summary_path)
    return result


def main() -> None:
    """运行顶部定义的合并配置。"""

    try:
        run_merge(RUN_CONFIG)
    except Exception as error:  # noqa: BLE001
        logger = get_logger("qwen3-finetune.merge")
        logger.exception("合并失败: %s", error)
        raise


if __name__ == "__main__":
    main()
