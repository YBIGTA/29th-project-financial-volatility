"""2차 Active Learning용 경계 사례 400건과 무작위 대조군 100건을 선정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANALYSIS_START, DEVELOPMENT_START, OUTPUT_DIR, SEED
from .select_active_learning import predict
from .utils import load_unified_data, split_time_pools


LABELING_COLUMNS = [
    "labeling_id", "item_id", "source", "target_stock", "datetime", "text",
    "original_index", "original_id", "headline", "body", "normalized_text_hash",
    "duplicate_group", "selection_group", "current_label", "future_label",
    "review_status", "label_reason",
]


def take(frame: pd.DataFrame, count: int, score: str, group: str, used: set[str]) -> pd.DataFrame:
    candidates = frame[~frame["item_id"].astype(str).isin(used)].copy()
    candidates["month"] = pd.to_datetime(candidates["datetime"]).dt.strftime("%Y-%m")
    candidates = candidates.sort_values(score, ascending=True)
    candidates["rank_in_stratum"] = candidates.groupby(
        ["target_stock", "month"], dropna=False
    ).cumcount()
    selected = candidates.sort_values(
        ["rank_in_stratum", score], ascending=[True, True]
    ).head(count).copy()
    selected["selection_group"] = group
    used.update(selected["item_id"].astype(str))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finbert/model_a_active/best"))
    parser.add_argument(
        "--labeled", type=Path, nargs="+",
        default=[
            OUTPUT_DIR / f"labeled_combined_seed{SEED}.csv",
            OUTPUT_DIR / f"active_learning_2000_seed{SEED}_labeled_3class.csv",
        ],
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / f"active_learning_round2_500_seed{SEED}.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    labeled_frames = [
        pd.read_csv(path, encoding="utf-8-sig", dtype={"original_id": str}) for path in args.labeled
    ]
    labeled = pd.concat(labeled_frames, ignore_index=True, sort=False)
    development, _, _ = split_time_pools(load_unified_data())
    item_ids = set(labeled["item_id"].astype(str))
    hashes = set(labeled["normalized_text_hash"].astype(str))
    originals = set(labeled["source"].astype(str) + ":" + labeled["original_id"].astype(str))
    dev_originals = development["source"].astype(str) + ":" + development["original_id"].astype(str)
    eligible = development[
        ~development["item_id"].astype(str).isin(item_ids)
        & ~development["normalized_text_hash"].astype(str).isin(hashes)
        & ~dev_originals.isin(originals)
    ].copy()
    eligible = eligible.sort_values("item_id").drop_duplicates("normalized_text_hash", keep="first")
    scored = predict(eligible, args.checkpoint, args.batch_size, args.max_length)
    scored["negative_neutral_gap"] = (
        scored["model_future_p_negative"] - scored["model_future_p_neutral"]
    ).abs()
    scored["negative_positive_gap"] = (
        scored["model_future_p_negative"] - scored["model_future_p_positive"]
    ).abs()
    scored["negative_priority"] = -scored["model_future_p_negative"]

    used: set[str] = set()
    parts = [
        take(scored, 150, "negative_neutral_gap", "future_negative_vs_neutral_boundary", used),
        take(scored, 150, "negative_positive_gap", "future_negative_vs_positive_boundary", used),
        take(scored, 100, "negative_priority", "future_negative_high_probability", used),
    ]
    remaining = scored[~scored["item_id"].astype(str).isin(used)].copy()
    random_part = remaining.sample(n=100, random_state=SEED).copy()
    random_part["selection_group"] = "random_control"
    parts.append(random_part)
    selected_audit = pd.concat(parts, ignore_index=True)
    selected_audit.insert(0, "labeling_id", [f"active2_42_{i:04d}" for i in range(1, 501)])
    selected_audit["current_label"] = ""
    selected_audit["future_label"] = ""
    selected_audit["review_status"] = "unlabeled"
    selected_audit["label_reason"] = ""

    selected = selected_audit[LABELING_COLUMNS].copy()
    if len(selected) != 500 or selected["item_id"].duplicated().any():
        raise RuntimeError("선정 건수 또는 item_id 중복 오류")
    if selected["normalized_text_hash"].duplicated().any() or set(selected["normalized_text_hash"]) & hashes:
        raise RuntimeError("정규화 텍스트 중복 오류")
    dates = pd.to_datetime(selected["datetime"], errors="coerce")
    if not dates.between(DEVELOPMENT_START, ANALYSIS_START, inclusive="left").all():
        raise RuntimeError("Development 기간 밖 표본이 있습니다.")
    forbidden = [column for column in selected if column.startswith("model_") or "entropy" in column]
    if forbidden:
        raise RuntimeError(f"라벨링 CSV에 모델 정보가 남았습니다: {forbidden}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = {
        "seed": SEED, "labeled_excluded_rows": len(labeled), "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "selection_group": selected["selection_group"].value_counts().to_dict(),
        "source": selected["source"].value_counts().to_dict(),
        "target_stock": selected["target_stock"].astype(str).value_counts().to_dict(),
        "overlap_item_id": len(set(selected["item_id"].astype(str)) & item_ids),
        "overlap_text_hash": len(set(selected["normalized_text_hash"].astype(str)) & hashes),
        "model_columns_exposed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"저장: {args.output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
