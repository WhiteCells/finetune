from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"
REPO_ID = DEFAULT_REPO_ID
OUTPUT_DIR = Path("models") / REPO_ID.split("/")[-1]
REVISION: str | None = None
TOKEN: str | None = None
MAX_WORKERS = 8


def main() -> None:
    """下载顶部配置指定的 Hugging Face 模型。"""

    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {output_dir}")
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(output_dir),
        revision=REVISION,
        token=TOKEN,
        max_workers=MAX_WORKERS,
    )
    print(f"Download complete: {downloaded_path}")


if __name__ == "__main__":
    main()
