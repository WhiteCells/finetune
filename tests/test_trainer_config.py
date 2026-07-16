from __future__ import annotations

import os
import inspect
import unittest

try:
    from trainer.trainer import build_training_arguments
    from trainer.trainer import LoRATrainer
    from trainer.trainer import TrainConfig
    from trainer.trainer import validate_train_config
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "peft", "yaml"}:
        raise
    build_training_arguments = None  # type: ignore[assignment]
    LoRATrainer = None  # type: ignore[assignment]
    TrainConfig = None  # type: ignore[assignment]
    validate_train_config = None  # type: ignore[assignment]


@unittest.skipUnless(TrainConfig is not None, "需要安装项目依赖")
class TrainConfigTests(unittest.TestCase):
    def test_trainer_log_accepts_newer_transformers_extra_arguments(self) -> None:
        signature = inspect.signature(LoRATrainer.log)
        parameter_kinds = {
            parameter.kind
            for parameter in signature.parameters.values()
        }

        self.assertIn(inspect.Parameter.VAR_POSITIONAL, parameter_kinds)
        self.assertIn(inspect.Parameter.VAR_KEYWORD, parameter_kinds)

    def test_rejects_invalid_warmup_ratio(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            warmup_ratio=1.5,
        )

        with self.assertRaisesRegex(ValueError, "warmup_ratio"):
            validate_train_config(config)

    def test_rejects_negative_gpu_id(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            gpu_id=-1,
        )

        with self.assertRaisesRegex(ValueError, "gpu_id"):
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

    def test_rejects_negative_warmup_steps(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            warmup_steps=-1,
        )

        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            validate_train_config(config)

    @unittest.skipUnless(build_training_arguments is not None, "需要安装项目依赖")
    def test_maps_warmup_ratio_to_warmup_steps(self) -> None:
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            bf16=False,
            warmup_steps=0,
            warmup_ratio=0.03,
            report_to=[],
        )

        training_args = build_training_arguments(
            config=config,
            has_eval_dataset=False,
        )

        self.assertEqual(training_args.warmup_steps, 0.03)

    @unittest.skipUnless(build_training_arguments is not None, "需要安装项目依赖")
    def test_sets_tensorboard_logging_dir_environment(self) -> None:
        original_logging_dir = os.environ.get("TENSORBOARD_LOGGING_DIR")
        config = TrainConfig(
            model_name_or_path="../models/Qwen3-4B-Instruct-2507",
            train_file="data/example.jsonl",
            bf16=False,
            logging_dir="logs/test-run",
            report_to=[],
        )

        try:
            build_training_arguments(config=config, has_eval_dataset=False)
            self.assertEqual(
                os.environ.get("TENSORBOARD_LOGGING_DIR"),
                "logs/test-run",
            )
        finally:
            if original_logging_dir is None:
                os.environ.pop("TENSORBOARD_LOGGING_DIR", None)
            else:
                os.environ["TENSORBOARD_LOGGING_DIR"] = original_logging_dir


if __name__ == "__main__":
    unittest.main()
