from __future__ import annotations

import os
import unittest
from unittest.mock import patch

try:
    from train import configure_cuda_device
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "peft", "yaml"}:
        raise
    configure_cuda_device = None  # type: ignore[assignment]


@unittest.skipUnless(configure_cuda_device is not None, "需要安装项目依赖")
class ConfigureCudaDeviceTests(unittest.TestCase):
    def test_sets_visible_device_before_cuda_initialization(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("train.torch.cuda.is_initialized", return_value=False):
                configure_cuda_device(1)

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_rejects_changes_after_cuda_initialization(self) -> None:
        with patch("train.torch.cuda.is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA 已初始化"):
                configure_cuda_device(1)


if __name__ == "__main__":
    unittest.main()
