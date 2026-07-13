from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# MODEL_PATH = Path("models/Qwen3-4B-Instruct-2507")
# MODEL_PATH = Path("models/Qwen3-4B-Instruct-2507-FP8")
MODEL_PATH = Path("models/Qwen3.5-2B")
PROMPT = "请用三句话解释什么是 LoRA。"


def main() -> None:
    model_path = MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # inference_mode 会关闭梯度计算，减少推理时的显存占用。
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # generate 的前半部分是输入 token，只解码模型新增的内容。
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
