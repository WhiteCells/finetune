from __future__ import annotations

import unittest

try:
    from trainer.trainer import TrainConfig
    from trainer.trainer import validate_train_config
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "peft", "yaml"}:
        raise
    TrainConfig = None  # type: ignore[assignment]
    validate_train_config = None  # type: ignore[assignment]


@unittest.skipUnless(TrainConfig is not None, "需要安装项目依赖")
class TrainConfigTests(unittest.TestCase):
    def test_rejects_invalid_warmup_ratio(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            warmup_ratio=1.5,
        )

        with self.assertRaisesRegex(ValueError, "warmup_ratio"):
            validate_train_config(config)

    def test_rejects_step_strategy_with_zero_save_steps(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            save_strategy="steps",
            save_steps=0,
        )

        with self.assertRaisesRegex(ValueError, "save_steps"):
            validate_train_config(config)


if __name__ == "__main__":
    unittest.main()
