#!/usr/bin/env python3
"""Qwen3-4B LoRA 推理脚本。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from model.loader import ModelLoadConfig
from model.loader import load_causal_lm
from model.loader import resize_token_embeddings_if_needed
from model.loader import summarize_model
from model.lora import load_lora_adapter
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
class InferenceArgs:
    """推理命令行参数。

    Attributes:
        model_name_or_path: 基座模型路径或名称。
        adapter_path: LoRA adapter 路径。
        prompt: 用户主提示词。
        input_text: 可选额外输入上下文。
        system_prompt: 可选系统提示词。
        cache_dir: 可选缓存目录。
        trust_remote_code: 是否允许加载自定义模型代码。
        use_fast_tokenizer: 是否优先使用 fast tokenizer。
        torch_dtype: 推理 dtype。
        attn_implementation: attention 实现方式。
        device_map: 设备映射，默认 `auto`。
        max_new_tokens: 最大生成长度。
        temperature: 采样温度。
        top_p: nucleus sampling 参数。
        top_k: top-k sampling 参数。
        repetition_penalty: 重复惩罚。
        do_sample: 是否启用采样。
        num_beams: beam search beam 数。
        seed: 可选随机种子。
        output_file: 可选输出文件路径。
        log_level: 日志级别。
    """

    model_name_or_path: str
    adapter_path: str
    prompt: str
    input_text: str
    system_prompt: str
    cache_dir: str | None
    trust_remote_code: bool
    use_fast_tokenizer: bool
    torch_dtype: str
    attn_implementation: str | None
    device_map: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool
    num_beams: int
    seed: int | None
    output_file: str | None
    log_level: str


def parse_args(argv: Sequence[str] | None = None) -> InferenceArgs:
    """解析推理命令行参数。

    Args:
        argv: 可选参数序列；为空时读取进程命令行。

    Returns:
        InferenceArgs: 结构化推理参数。
    """

    parser = argparse.ArgumentParser(description="Qwen3-4B LoRA 推理脚本。")
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
        "--prompt",
        required=True,
        help="用户主提示词。",
    )
    parser.add_argument(
        "--input-text",
        default="",
        help="可选额外输入上下文，将与 prompt 一起组成 user 消息。",
    )
    parser.add_argument(
        "--system-prompt",
        default="你是一个专业、可靠、简洁的中文助手。",
        help="可选 system prompt。",
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
        help="推理 dtype，如 bfloat16、float16、float32 或 auto。",
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
        "--max-new-tokens",
        default=256,
        type=int,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="采样温度。",
    )
    parser.add_argument(
        "--top-p",
        default=0.9,
        type=float,
        help="top-p nucleus sampling 参数。",
    )
    parser.add_argument(
        "--top-k",
        default=50,
        type=int,
        help="top-k sampling 参数。",
    )
    parser.add_argument(
        "--repetition-penalty",
        default=1.0,
        type=float,
        help="重复惩罚系数。",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用采样生成。",
    )
    parser.add_argument(
        "--num-beams",
        default=1,
        type=int,
        help="beam search 的 beam 数。",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="可选随机种子。",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="可选输出文件路径，用于保存生成结果。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，如 INFO、DEBUG、WARNING。",
    )

    namespace = parser.parse_args(argv)
    return InferenceArgs(
        model_name_or_path=namespace.model_name_or_path,
        adapter_path=namespace.adapter_path,
        prompt=namespace.prompt,
        input_text=namespace.input_text,
        system_prompt=namespace.system_prompt,
        cache_dir=namespace.cache_dir,
        trust_remote_code=namespace.trust_remote_code,
        use_fast_tokenizer=namespace.use_fast_tokenizer,
        torch_dtype=namespace.torch_dtype,
        attn_implementation=namespace.attn_implementation,
        device_map=namespace.device_map,
        max_new_tokens=namespace.max_new_tokens,
        temperature=namespace.temperature,
        top_p=namespace.top_p,
        top_k=namespace.top_k,
        repetition_penalty=namespace.repetition_penalty,
        do_sample=namespace.do_sample,
        num_beams=namespace.num_beams,
        seed=namespace.seed,
        output_file=namespace.output_file,
        log_level=namespace.log_level,
    )


def initialize_logger(log_level: str) -> object:
    """初始化推理 logger。

    Args:
        log_level: 日志级别。

    Returns:
        object: 已配置 logger。
    """

    return setup_logger(
        LoggerConfig(
            name="qwen3-finetune.inference",
            level=log_level,
            log_file=None,
            console=True,
            propagate=False,
        )
    )


def validate_args(args: InferenceArgs) -> None:
    """校验推理参数。

    Args:
        args: 推理参数。

    Raises:
        FileNotFoundError: 基础模型或 adapter 路径不存在时抛出。
        ValueError: 生成参数非法时抛出。
    """

    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"基础模型路径不存在: {args.model_name_or_path}")

    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {args.adapter_path}")

    if not args.prompt.strip():
        raise ValueError("`--prompt` 不能为空。")
    if args.max_new_tokens <= 0:
        raise ValueError("`--max-new-tokens` 必须大于 0。")
    if args.temperature < 0:
        raise ValueError("`--temperature` 不能小于 0。")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("启用采样时 `--temperature` 必须大于 0。")
    if not 0 < args.top_p <= 1.0:
        raise ValueError("`--top-p` 必须在 (0, 1] 范围内。")
    if args.top_k < 0:
        raise ValueError("`--top-k` 不能小于 0。")
    if args.num_beams <= 0:
        raise ValueError("`--num-beams` 必须大于 0。")
    if args.repetition_penalty <= 0:
        raise ValueError("`--repetition-penalty` 必须大于 0。")


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


def get_input_device(model: object) -> torch.device:
    """获取输入张量应该放置的设备。

    对使用 `device_map="auto"` 的模型，通常把输入放到首个参数所在设备即可，
    Accelerate 会在后续层之间自动分发。

    Args:
        model: 已加载模型。

    Returns:
        torch.device: 输入设备。
    """

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
    """将 prompt、input 和 system prompt 渲染成最终推理文本。

    Args:
        tokenizer: tokenizer 实例。
        prompt: 用户主提示词。
        input_text: 额外输入上下文。
        system_prompt: 系统提示词。

    Returns:
        str: 应送入模型的完整 prompt 文本。
    """

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
    args: InferenceArgs,
    tokenizer: TokenizerType,
) -> dict[str, object]:
    """构建 `generate()` 所需参数。

    Args:
        args: 推理参数。
        tokenizer: tokenizer 实例。

    Returns:
        dict[str, object]: 可直接传给 `model.generate()` 的参数字典。
    """

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    return generation_kwargs


def generate_text(
    model: object,
    tokenizer: TokenizerType,
    prompt_text: str,
    generation_kwargs: dict[str, object],
) -> tuple[str, list[int]]:
    """执行文本生成。

    Args:
        model: 已加载 base model + LoRA adapter。
        tokenizer: tokenizer 实例。
        prompt_text: 完整输入 prompt 文本。
        generation_kwargs: 传给 `model.generate()` 的生成参数。

    Returns:
        tuple[str, list[int]]:
            - 解码后的新生成文本
            - 完整输出 token id
    """

    encoded_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_device = get_input_device(model)
    encoded_inputs = {
        key: value.to(input_device)
        for key, value in encoded_inputs.items()
    }

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded_inputs,
            **generation_kwargs,
        )

    prompt_length = int(encoded_inputs["input_ids"].shape[1])
    generated_ids = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
    return generated_text, output_ids[0].tolist()


def maybe_save_output(output_file: str | None, content: str) -> str | None:
    """按需将生成结果写入文件。

    Args:
        output_file: 目标文件路径；为空时跳过保存。
        content: 需要保存的文本内容。

    Returns:
        str | None: 若保存成功则返回文件路径字符串，否则返回 `None`。
    """

    if output_file is None:
        return None

    output_path = save_text_file(content=content, output_path=output_file)
    return str(output_path)


def run_inference(args: InferenceArgs) -> str:
    """执行完整推理流程。

    Args:
        args: 推理参数。

    Returns:
        str: 模型生成的最终文本。
    """

    validate_args(args)
    logger = initialize_logger(args.log_level)

    log_section(logger, "启动推理")
    log_kv(logger, "推理参数", asdict(args))

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("推理随机种子已设置为: %s", args.seed)

    tokenizer = load_tokenizer(
        TokenizerLoadConfig(
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
            model_max_length=None,
            padding_side="left",
            truncation_side="left",
        )
    )
    log_kv(logger, "Tokenizer 摘要", tokenizer_summary(tokenizer))

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
    resize_token_embeddings_if_needed(base_model, tokenizer_size=len(tokenizer))
    configure_model_special_tokens(base_model, tokenizer)
    log_kv(logger, "基础模型摘要", summarize_model(base_model))

    model = load_lora_adapter(
        model=base_model,
        adapter_path=args.adapter_path,
        is_trainable=False,
    )
    model.eval()

    prompt_text = build_prompt_text(
        tokenizer=tokenizer,
        prompt=args.prompt,
        input_text=args.input_text,
        system_prompt=args.system_prompt,
    )
    generation_kwargs = build_generation_kwargs(args=args, tokenizer=tokenizer)

    log_kv(
        logger,
        "生成参数",
        {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
        },
    )

    generated_text, output_ids = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        generation_kwargs=generation_kwargs,
    )

    saved_output_path = maybe_save_output(args.output_file, generated_text)
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
    """脚本主入口，负责统一异常处理。"""

    try:
        run_inference(parse_args())
    except Exception as error:  # noqa: BLE001
        logger = get_logger("qwen3-finetune.inference")
        logger.exception("推理失败: %s", error)
        raise


if __name__ == "__main__":
    main()
