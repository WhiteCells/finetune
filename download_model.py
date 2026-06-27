from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / DEFAULT_REPO_ID.split("/")[-1],
        help="Directory where the model files will be stored.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit to download.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of concurrent download workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {output_dir}")
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(output_dir),
        revision=args.revision,
        token=args.token,
        max_workers=args.max_workers,
    )
    print(f"Download complete: {downloaded_path}")


if __name__ == "__main__":
    main()
