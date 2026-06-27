from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the local Qwen3-4B-Instruct-2507 model."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="User prompt. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Target device.",
    )
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def read_prompt(prompt: str | None) -> str:
    if prompt is not None:
        return prompt.strip()
    if sys.stdin.isatty():
        raise SystemExit("Provide --prompt or pipe text through stdin.")
    return sys.stdin.read().strip()


def main() -> None:
    args = parse_args()
    prompt = read_prompt(args.prompt)
    if not prompt:
        raise SystemExit("Provide --prompt or pipe a prompt through stdin.")

    model_path = str(args.model_path)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    response_ids = generated_ids[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(response)


if __name__ == "__main__":
    main()
