from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

COLLECTIONS = {
    "train": ("Salesforce/GiftEvalPretrain", "GIFT_EVAL_PRETRAIN", "data/GiftEvalPretrain"),
    "test": ("Salesforce/GIFT_Eval", "GIFT_EVAL", "data/GIFT_Eval"),
}


def _download(kind: str, revision: str | None) -> None:
    repo_id, env_name, default = COLLECTIONS[kind]
    destination = Path(os.getenv(env_name, default)).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {destination}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the two GIFT-Eval collections used by MetaCaster"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--train-revision", default=None)
    parser.add_argument("--test-revision", default=None)
    args = parser.parse_args()
    if args.train or args.all:
        _download("train", args.train_revision)
    if args.test or args.all:
        _download("test", args.test_revision)


if __name__ == "__main__":
    main()
