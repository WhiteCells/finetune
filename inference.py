#!/usr/bin/env python3
"""加载 LoRA adapter 并生成回答。

修改 `RUN_CONFIG` 后，直接运行：

    uv run inference.py
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch

from model.loader import ModelLoadConfig
from model.loader import load_causal_lm
from model.loader import resize_token_embeddings_if_needed
from model.loader import summarize_model
from model.lora import load_lora_adapter
from model.lora import validate_lora_adapter_path
from model.tokenizer import TokenizerLoadConfig
from model.tokenizer import TokenizerType
from model.tokenizer import apply_chat_template
from model.tokenizer import build_messages
from model.tokenizer import load_tokenizer
from model.tokenizer import tokenizer_summary
from utils.logger import LoggerConfig
from utils.logger import get_logger
from utils.logger import log_kv
from utils.logger import log_section
from utils.logger import setup_logger
from utils.save import save_text_file
from utils.seed import seed_everything


@dataclass(slots=True)
class InferenceConfig:
    """一次推理需要的少量配置。"""

    model_name_or_path: str
    adapter_path: str
    prompt: str
    input_text: str = ""
    system_prompt: str = "你是一个专业、可靠、简洁的中文助手。"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    output_file: str | None = None


RUN_CONFIG = InferenceConfig(
    model_name_or_path="./models/Qwen3-4B-Instruct-2507",
    adapter_path="outputs/qwen3-4b-lora",
    prompt="请用一句话解释 LoRA。",
)


def initialize_logger() -> object:
    """初始化推理日志。"""

    return setup_logger(
        LoggerConfig(
            name="qwen3-finetune.inference",
            level="INFO",
            log_file=None,
            console=True,
            propagate=False,
        )
    )


def validate_config(config: InferenceConfig) -> None:
    """校验推理配置中的路径和生成参数。"""

    if not Path(config.model_name_or_path).exists():
        raise FileNotFoundError(f"基础模型路径不存在: {config.model_name_or_path}")

    validate_lora_adapter_path(config.adapter_path)

    if not config.prompt.strip():
        raise ValueError("`prompt` 不能为空。")
    if config.max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` 必须大于 0。")
    if config.temperature < 0:
        raise ValueError("`temperature` 不能小于 0。")
    if config.do_sample and config.temperature <= 0:
        raise ValueError("启用采样时 `temperature` 必须大于 0。")
    if not 0 < config.top_p <= 1.0:
        raise ValueError("`top_p` 必须在 (0, 1] 范围内。")


def configure_model_special_tokens(model: object, tokenizer: TokenizerType) -> None:
    """同步模型与 tokenizer 的 special token 配置。"""

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


def get_input_device(model: object) -> torch.device:
    """返回输入张量应放置的设备。"""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_prompt_text(
    tokenizer: TokenizerType,
    prompt: str,
    input_text: str,
    system_prompt: str,
) -> str:
    """将输入渲染成模型需要的 chat template 文本。"""

    messages = build_messages(
        instruction=prompt,
        input_text=input_text,
        output_text=None,
        system_prompt=system_prompt,
    )
    return apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
    )


def build_generation_kwargs(
    config: InferenceConfig,
    tokenizer: TokenizerType,
) -> dict[str, object]:
    """构建传给 `model.generate()` 的参数。"""

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generation_kwargs.update(
            {
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
        )
    return generation_kwargs


def generate_text(
    model: object,
    tokenizer: TokenizerType,
    prompt_text: str,
    generation_kwargs: dict[str, object],
) -> tuple[str, list[int]]:
    """执行生成并仅解码新增 token。"""

    encoded_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded_inputs = {
        key: value.to(get_input_device(model))
        for key, value in encoded_inputs.items()
    }

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded_inputs,
            **generation_kwargs,
        )

    prompt_length = int(encoded_inputs["input_ids"].shape[1])
    generated_text = tokenizer.decode(
        output_ids[0][prompt_length:],
        skip_special_tokens=True,
    ).strip()
    return generated_text, output_ids[0].tolist()


def maybe_save_output(output_file: str | None, content: str) -> str | None:
    """按需写出生成结果。"""

    if output_file is None:
        return None
    return str(save_text_file(content=content, output_path=output_file))


def run_inference(config: InferenceConfig) -> str:
    """执行完整的 adapter 推理流程。"""

    validate_config(config)
    logger = initialize_logger()
    log_section(logger, "启动推理")
    log_kv(logger, "推理配置", asdict(config))

    if config.seed is not None:
        seed_everything(config.seed)
        logger.info("推理随机种子已设置为: %s", config.seed)

    tokenizer = load_tokenizer(
        TokenizerLoadConfig(
            model_name_or_path=config.model_name_or_path,
            model_max_length=None,
            padding_side="left",
            truncation_side="left",
        )
    )
    log_kv(logger, "Tokenizer 摘要", tokenizer_summary(tokenizer))

    base_model = load_causal_lm(
        ModelLoadConfig(
            model_name_or_path=config.model_name_or_path,
            gradient_checkpointing=False,
            use_cache=True,
            device_map="auto",
        )
    )
    resize_token_embeddings_if_needed(base_model, tokenizer_size=len(tokenizer))
    configure_model_special_tokens(base_model, tokenizer)
    log_kv(logger, "基础模型摘要", summarize_model(base_model))

    model = load_lora_adapter(
        model=base_model,
        adapter_path=config.adapter_path,
        is_trainable=False,
    )
    prompt_text = build_prompt_text(
        tokenizer=tokenizer,
        prompt=config.prompt,
        input_text=config.input_text,
        system_prompt=config.system_prompt,
    )
    generation_kwargs = build_generation_kwargs(config=config, tokenizer=tokenizer)
    generated_text, output_ids = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        generation_kwargs=generation_kwargs,
    )

    saved_output_path = maybe_save_output(config.output_file, generated_text)
    if saved_output_path is not None:
        logger.info("生成结果已保存到: %s", saved_output_path)

    log_kv(
        logger,
        "生成结果摘要",
        {
            "prompt_length_chars": len(prompt_text),
            "generated_length_chars": len(generated_text),
            "output_token_count": len(output_ids),
        },
    )
    print(generated_text)
    return generated_text


def main() -> None:
    """运行顶部定义的推理配置。"""

    try:
        run_inference(RUN_CONFIG)
    except Exception as error:  # noqa: BLE001
        logger = get_logger("qwen3-finetune.inference")
        logger.exception("推理失败: %s", error)
        raise


if __name__ == "__main__":
    main()
