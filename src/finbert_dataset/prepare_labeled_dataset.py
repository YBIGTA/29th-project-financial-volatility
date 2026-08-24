"""여러 라벨 완료 CSV를 검증·결합하고 고정 Train/Validation/Test split을 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ANALYSIS_POOL_FILE, OUTPUT_DIR, SEED, ensure_output_dir
from .split_dataset import (
    assign_groups,
    build_summary,
    validate_analysis_separation,
    validate_labeled_data,
    validate_no_leakage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="라벨 완료 CSV(복수 가능)")
    args = parser.parse_args()

    frames = []
    for path in args.input:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"target_stock": str, "original_id": str})
        frame["input_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)

    duplicate_items = data["item_id"].astype(str).duplicated(keep=False)
    if duplicate_items.any():
        examples = sorted(data.loc[duplicate_items, "item_id"].astype(str).unique())[:10]
        raise ValueError(f"입력 파일 사이 item_id 중복 {duplicate_items.sum()}행: {examples}")

    data = validate_labeled_data(data)
    assignment = assign_groups(data)
    data["split"] = data["split_group"].map(assignment)
    splits = {
        name: data[data["split"].eq(name)].copy().reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    leakage = validate_no_leakage(splits)

    if not ANALYSIS_POOL_FILE.exists():
        raise FileNotFoundError(f"Analysis pool이 없습니다: {ANALYSIS_POOL_FILE}")
    analysis = pd.read_csv(
        ANALYSIS_POOL_FILE,
        encoding="utf-8-sig",
        dtype={"target_stock": str, "original_id": str},
        parse_dates=["datetime"],
    )
    analysis_separation = validate_analysis_separation(splits, analysis)
    summary = build_summary(splits, leakage, analysis_separation)
    summary["input_files"] = [str(path.resolve()) for path in args.input]

    ensure_output_dir()
    combined = OUTPUT_DIR / f"labeled_combined_seed{SEED}.csv"
    data.to_csv(combined, index=False, encoding="utf-8-sig")
    for name, frame in splits.items():
        frame.to_csv(OUTPUT_DIR / f"{name}_seed{SEED}.csv", index=False, encoding="utf-8-sig")
    summary_path = OUTPUT_DIR / f"split_summary_seed{SEED}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결합: {combined} ({len(data):,}건)")
    for name, frame in splits.items():
        print(f"{name}: {len(frame):,}건")
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    main()
