"""随机种子工具。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class SeedConfig:
    """随机种子配置。

    Attributes:
        seed: 随机种子值。
        deterministic: 是否启用更严格的确定性模式。
    """

    seed: int = 42
    deterministic: bool = False


def seed_everything(
    seed_or_config: int | SeedConfig,
    deterministic: bool | None = None,
) -> int:
    """统一设置 Python、NumPy 和 PyTorch 的随机种子。

    Args:
        seed_or_config: 整数随机种子，或 `SeedConfig` 配置对象。
        deterministic: 可选覆盖项。若传入，则优先于 `SeedConfig` 中的值。

    Returns:
        int: 最终生效的随机种子值。
    """

    if isinstance(seed_or_config, SeedConfig):
        seed = seed_or_config.seed
        final_deterministic = (
            seed_or_config.deterministic
            if deterministic is None
            else deterministic
        )
    else:
        seed = int(seed_or_config)
        final_deterministic = False if deterministic is None else deterministic

    os.environ["PYTHONHASHSEED"] = str(seed)

    # 统一设置 Python 标准库、NumPy 和 PyTorch 的随机种子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 根据需要启用确定性模式。该模式更可复现，但通常会牺牲部分速度。
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = final_deterministic
        torch.backends.cudnn.benchmark = not final_deterministic

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(final_deterministic, warn_only=True)

    return seed


def seed_worker(worker_id: int) -> None:
    """为 DataLoader worker 设置随机种子。

    该函数适合传给 `DataLoader(worker_init_fn=...)`，以保证多 worker
    场景下 NumPy 和 Python `random` 的随机态可复现。

    Args:
        worker_id: worker 编号。函数内部不直接使用该值，但保留该参数以符合
            `worker_init_fn` 调用约定。
    """

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    """构建带固定随机种子的 `torch.Generator`。

    Args:
        seed: 随机种子值。

    Returns:
        torch.Generator: 已设置随机种子的生成器实例。
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

