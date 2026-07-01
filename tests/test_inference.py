from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from inference import InferenceArgs
    from inference import build_generation_kwargs
    from inference import validate_args
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "peft", "yaml"}:
        raise
    InferenceArgs = None  # type: ignore[assignment]
    build_generation_kwargs = None  # type: ignore[assignment]
    validate_args = None  # type: ignore[assignment]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1


@unittest.skipUnless(InferenceArgs is not None, "需要安装项目依赖")
class InferenceHelperTests(unittest.TestCase):
    def make_args(
        self,
        model_path: str = "model",
        adapter_path: str = "adapter",
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> InferenceArgs:
        return InferenceArgs(
            model_name_or_path=model_path,
            adapter_path=adapter_path,
            prompt="请用一句话解释 LoRA。",
            input_text="",
            system_prompt="你是一个助手。",
            cache_dir=None,
            trust_remote_code=True,
            use_fast_tokenizer=True,
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            device_map="auto",
            max_new_tokens=64,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            do_sample=do_sample,
            num_beams=1,
            seed=None,
            output_file=None,
            log_level="INFO",
        )

    def test_generation_kwargs_omit_sampling_fields_when_sampling_disabled(self) -> None:
        args = self.make_args(do_sample=False)
        generation_kwargs = build_generation_kwargs(args, FakeTokenizer())

        self.assertFalse(generation_kwargs["do_sample"])
        self.assertNotIn("temperature", generation_kwargs)
        self.assertNotIn("top_p", generation_kwargs)
        self.assertNotIn("top_k", generation_kwargs)

    def test_validate_args_rejects_zero_temperature_when_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "base"
            adapter_path = Path(temp_dir) / "adapter"
            model_path.mkdir()
            adapter_path.mkdir()

            args = self.make_args(
                model_path=str(model_path),
                adapter_path=str(adapter_path),
                do_sample=True,
                temperature=0.0,
            )

            with self.assertRaisesRegex(ValueError, "temperature"):
                validate_args(args)


if __name__ == "__main__":
    unittest.main()
