from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from data.preprocess import PreprocessConfig
from data.preprocess import detect_format
from data.preprocess import normalize_record
from data.preprocess import normalize_records
from data.preprocess import run_preprocess


class PreprocessTests(unittest.TestCase):
    def test_detects_common_formats(self) -> None:
        self.assertEqual(
            detect_format({"instruction": "解释 LoRA", "output": "一种微调方法"}),
            "alpaca",
        )
        self.assertEqual(
            detect_format(
                {
                    "conversations": [
                        {"from": "human", "value": "你好"},
                        {"from": "gpt", "value": "你好"},
                    ]
                }
            ),
            "sharegpt",
        )
        self.assertEqual(
            detect_format(
                {
                    "messages": [
                        {"role": "user", "content": "你好"},
                        {"role": "assistant", "content": "你好"},
                    ]
                }
            ),
            "messages",
        )

    def test_normalizes_alpaca_and_injects_system_prompt(self) -> None:
        normalized = normalize_record(
            record={
                "instruction": "把语气改正式",
                "input": "这个方案挺靠谱",
                "output": "该方案具备较高可行性。",
            },
            input_format="auto",
            system_prompt="你是一个严谨助手。",
        )

        self.assertEqual(normalized["messages"][0]["role"], "system")
        self.assertIn("严谨助手", normalized["messages"][0]["content"])
        self.assertEqual(normalized["messages"][1]["role"], "user")
        self.assertIn("这个方案挺靠谱", normalized["messages"][1]["content"])
        self.assertEqual(normalized["messages"][2]["role"], "assistant")

    def test_skip_invalid_records_reports_reason(self) -> None:
        config = PreprocessConfig(
            input_path=Path("unused.jsonl"),
            output_path=Path("unused.out.jsonl"),
            input_format="auto",
            system_prompt="",
            skip_invalid=True,
            deduplicate=False,
            ensure_ascii=False,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            records, stats = normalize_records(
                records=[
                    {"instruction": "解释 LoRA", "output": "参数高效微调方法。"},
                    {"instruction": "缺少答案"},
                ],
                config=config,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(stats.total_records, 2)
        self.assertEqual(stats.written_records, 1)
        self.assertEqual(stats.skipped_records, 1)
        self.assertIn("跳过第 2 条坏样本", stdout.getvalue())
        self.assertIn("无法自动识别样本格式", stdout.getvalue())

    def test_preprocess_deduplicates_and_skips_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "train.jsonl"
            valid_record = {
                "instruction": "请用一句话解释 LoRA。",
                "input": "",
                "output": "LoRA 是一种参数高效微调方法。",
            }
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(valid_record, ensure_ascii=False),
                        json.dumps(valid_record, ensure_ascii=False),
                        json.dumps({"instruction": "缺少 output"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = PreprocessConfig(
                input_path=input_path,
                output_path=output_path,
                input_format="auto",
                system_prompt="",
                skip_invalid=True,
                deduplicate=True,
                ensure_ascii=False,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                stats = run_preprocess(config)

            output_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(output_lines), 1)
            self.assertEqual(stats.written_records, 1)
            self.assertEqual(stats.skipped_records, 1)
            self.assertEqual(stats.duplicate_records, 1)
            self.assertIn("写出样本数: 1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
