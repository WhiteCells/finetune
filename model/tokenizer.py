"""Tokenizer 加载与消息格式化工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Sequence

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast


DEFAULT_CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}
{%- set loop_messages = messages[1:] -%}
{%- else -%}
{{- '<|im_start|>system\\n你是一个专业、可靠、简洁的中文助手。<|im_end|>\\n' -}}
{%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}
"""


TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast


@dataclass(slots=True)
class TokenizerLoadConfig:
    """Tokenizer 加载配置。

    Attributes:
        model_name_or_path: tokenizer 对应的模型目录或 Hugging Face 名称。
        cache_dir: 可选缓存目录。
        use_fast_tokenizer: 是否优先使用 fast tokenizer。
        trust_remote_code: 是否信任远程自定义代码。
        model_max_length: 可选 tokenizer 最大长度覆盖值。
        padding_side: padding 方向，默认右侧。
        truncation_side: 截断方向，默认右侧。
    """

    model_name_or_path: str
    cache_dir: str | None = None
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True
    model_max_length: int | None = None
    padding_side: str = "right"
    truncation_side: str = "right"


def load_tokenizer(config: TokenizerLoadConfig) -> TokenizerType:
    """加载并规范化 tokenizer。

    该函数会优先使用模型目录中的 tokenizer 配置，并在必要时补齐
    `pad_token`、`padding_side`、`truncation_side` 和 chat template，
    以便后续 dataset 与推理逻辑稳定运行。

    Args:
        config: tokenizer 加载配置。

    Returns:
        TokenizerType: 已完成必要规范化的 tokenizer 实例。
    """

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code,
    )

    # 保证 tokenizer 拥有 pad token，便于后续动态 padding。
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 明确 padding 与 truncation 方向，避免不同版本行为不一致。
    tokenizer.padding_side = config.padding_side
    tokenizer.truncation_side = config.truncation_side

    if config.model_max_length is not None:
        tokenizer.model_max_length = config.model_max_length

    # 优先使用模型自带 chat template；缺失时补一个兼容 Qwen 风格的默认模板。
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def build_messages(
    instruction: str,
    input_text: str,
    output_text: str | None = None,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    """根据 instruction/input/output 拼接统一消息结构。

    Args:
        instruction: 指令主体。
        input_text: 额外输入上下文。
        output_text: 可选 assistant 回复内容。训练时通常提供，推理时可为空。
        system_prompt: 可选系统提示词。

    Returns:
        list[dict[str, str]]: 规范化消息列表。
    """

    user_content = instruction.strip()
    normalized_input = input_text.strip()
    if normalized_input:
        user_content = f"{user_content}\n\n{normalized_input}"

    messages: list[dict[str, str]] = []

    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.append({"role": "user", "content": user_content})

    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text.strip()})

    return messages


def apply_chat_template(
    tokenizer: TokenizerType,
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """使用 tokenizer 的 chat template 渲染消息。

    Args:
        tokenizer: 已加载 tokenizer。
        messages: 对话消息列表。
        add_generation_prompt: 是否在末尾追加 assistant 生成起始标记。

    Returns:
        str: 渲染后的纯文本 prompt。
    """

    rendered_text = tokenizer.apply_chat_template(
        conversation=list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return str(rendered_text)


def get_special_token_ids(tokenizer: TokenizerType) -> dict[str, int | None]:
    """返回训练和生成中常用的 special token id。

    Args:
        tokenizer: tokenizer 实例。

    Returns:
        dict[str, int | None]: special token id 字典。
    """

    return {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def tokenizer_summary(tokenizer: TokenizerType) -> dict[str, Any]:
    """返回 tokenizer 关键摘要信息。

    Args:
        tokenizer: tokenizer 实例。

    Returns:
        dict[str, Any]: 便于日志打印的 tokenizer 摘要。
    """

    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }

