"""기존 Validation/Test를 고정하고 Active Learning 라벨을 Train에만 추가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import SEED
from .split_dataset import validate_labeled_data, validate_no_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("data/processed/finbert"))
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/finbert_active"))
    args = parser.parse_args()

    names = ("train", "validation", "test")
    paths = {name: args.base_dir / f"{name}_seed{SEED}.csv" for name in names}
    base = {
        name: pd.read_csv(path, encoding="utf-8-sig", dtype={"target_stock": str, "original_id": str})
        for name, path in paths.items()
    }
    active_raw = pd.read_csv(
        args.active, encoding="utf-8-sig", dtype={"target_stock": str, "original_id": str}
    )
    active = validate_labeled_data(active_raw)
    active["input_file"] = args.active.name
    active["split"] = "train"
    active["review_status"] = "labeled"

    existing = pd.concat(base.values(), ignore_index=True)
    overlap = {
        "item_id": len(set(active["item_id"].astype(str)) & set(existing["item_id"].astype(str))),
        "normalized_text_hash": len(set(active["normalized_text_hash"].astype(str)) & set(existing["normalized_text_hash"].astype(str))),
        "split_group": len(set(active["split_group"].astype(str)) & set(existing["split_group"].astype(str))),
    }
    if any(overlap.values()):
        raise ValueError(f"기존 split과 Active Learning 데이터 중복: {overlap}")

    train = pd.concat([base["train"], active], ignore_index=True, sort=False)
    splits = {"train": train, "validation": base["validation"], "test": base["test"]}
    leakage = validate_no_leakage(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(args.output_dir / f"{name}_seed{SEED}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED, "base_train_rows": len(base["train"]), "active_added_rows": len(active),
        "train_rows": len(train), "validation_rows": len(base["validation"]), "test_rows": len(base["test"]),
        "overlap_with_existing": overlap, "split_leakage": leakage,
        "train_current_label": train["current_label"].value_counts().to_dict(),
        "train_future_label": train["future_label"].value_counts().to_dict(),
    }
    (args.output_dir / f"active_training_summary_seed{SEED}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
