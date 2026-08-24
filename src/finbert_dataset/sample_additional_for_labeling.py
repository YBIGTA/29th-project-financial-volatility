"""Development pool에서 기존 500건과 겹치지 않는 추가 라벨링 표본 1,500건을 만든다."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import (
    ADDITIONAL_EDAILY_SIZE,
    ADDITIONAL_FUTURE_NEGATIVE_SIZE,
    ADDITIONAL_RANDOM_SIZE,
    ADDITIONAL_SAMPLE_FILE,
    ADDITIONAL_SAMPLE_REPORT_FILE,
    ADDITIONAL_TOSS_SIZE,
    ANALYSIS_START,
    OUTPUT_DIR,
    SEED,
    ensure_output_dir,
)
from .sample_for_labeling import round_robin_sample
from .utils import load_unified_data, normalize_text, split_time_pools, stable_hash


FUTURE_TERMS = [
    "내일", "모레", "앞으로", "향후", "이후", "추후", "다음", "전망", "예상", "예측",
    "곧", "장차", "가능성", "것 같다", "거 같다", "듯하다", "할 듯", "될 듯", "갈 듯",
    "할 것", "될 것", "갈 것", "오를까", "내릴까", "보인다", "보일 것",
]

NEGATIVE_TERMS = [
    "하락", "급락", "폭락", "떡락", "내림", "내려가", "내려갈", "빠지", "빠질",
    "무너지", "하방", "약세", "조정", "매도", "저점", "고점", "악화", "적자", "감소",
    "부진", "위험", "망하", "망할", "박살", "꼬라박", "처박", "손절", "물리", "물림",
    "나락", "끝났", "상폐", "폭망", "실망", "위기",
]


OUTPUT_COLUMNS = [
    "item_id",
    "source",
    "target_stock",
    "datetime",
    "text",
    "current_label",
    "future_label",
    "original_index",
    "original_id",
    "headline",
    "body",
    "sampling_group",
    "candidate_score",
    "future_keyword_hits",
    "negative_keyword_hits",
    "normalized_text_hash",
    "duplicate_group",
]


def _find_existing_sample() -> Path:
    candidates = [
        OUTPUT_DIR / "labeling_sample_seed42_labeled_3class.csv",
        OUTPUT_DIR / "labeling_sample_seed42_labeled.csv",
        OUTPUT_DIR / "labeling_sample_seed42.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"기존 500건 표본을 찾지 못했습니다: {[str(path) for path in candidates]}")


def _load_existing_keys(path: Path) -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    existing = pd.read_csv(path, encoding="utf-8-sig", dtype={"target_stock": str, "original_id": str})
    required = {"item_id", "source", "original_id", "text"}
    missing = required.difference(existing.columns)
    if missing:
        raise ValueError(f"기존 표본 필수 컬럼 누락: {sorted(missing)}")
    normalized_hash = existing["text"].fillna("").astype(str).map(normalize_text).map(stable_hash)
    item_ids = set(existing["item_id"].astype(str))
    original_ids = set(existing["source"].astype(str) + ":" + existing["original_id"].astype(str))
    return existing, item_ids, original_ids, set(normalized_hash)


def _exclude_keys(
    frame: pd.DataFrame,
    item_ids: set[str],
    original_ids: set[str],
    text_hashes: set[str],
) -> pd.DataFrame:
    source_original = frame["source"].astype(str) + ":" + frame["original_id"].astype(str)
    mask = (
        ~frame["item_id"].astype(str).isin(item_ids)
        & ~source_original.isin(original_ids)
        & ~frame["normalized_text_hash"].isin(text_hashes)
    )
    return frame.loc[mask].copy()


def _keyword_count(text: str, terms: list[str]) -> int:
    normalized = normalize_text(text)
    return sum(len(re.findall(re.escape(term), normalized)) for term in terms)


def build_future_negative_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["future_keyword_hits"] = result["text"].map(lambda value: _keyword_count(value, FUTURE_TERMS))
    result["negative_keyword_hits"] = result["text"].map(lambda value: _keyword_count(value, NEGATIVE_TERMS))
    result = result[
        result["future_keyword_hits"].gt(0) & result["negative_keyword_hits"].gt(0)
    ].copy()
    repeated_or_multiple = (
        result["future_keyword_hits"].add(result["negative_keyword_hits"]).ge(3)
        | result["future_keyword_hits"].ge(2)
        | result["negative_keyword_hits"].ge(2)
    )
    result["candidate_score"] = (
        1 + 1 + repeated_or_multiple.astype(int)
    )
    return result.sort_values(
        ["candidate_score", "future_keyword_hits", "negative_keyword_hits", "datetime", "item_id"],
        ascending=[False, False, False, True, True],
    )


def _deduplicate_for_labeling(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["normalized_text_hash", "source", "target_stock", "item_id"])
        .drop_duplicates("item_id", keep="first")
        .drop_duplicates("normalized_text_hash", keep="first")
        .copy()
    )


def _sample_group_b(candidates: pd.DataFrame) -> pd.DataFrame:
    if len(candidates) < ADDITIONAL_FUTURE_NEGATIVE_SIZE:
        raise ValueError(
            f"Future Negative 후보가 {len(candidates)}건뿐이라 "
            f"{ADDITIONAL_FUTURE_NEGATIVE_SIZE}건을 추출할 수 없습니다."
        )
    # 높은 점수 후보를 우선하되 동일 점수 안에서는 source·종목·월·길이를 분산한다.
    selected_parts = []
    remaining_needed = ADDITIONAL_FUTURE_NEGATIVE_SIZE
    for score in sorted(candidates["candidate_score"].unique(), reverse=True):
        score_frame = candidates[candidates["candidate_score"].eq(score)]
        take = min(remaining_needed, len(score_frame))
        if take:
            selected_parts.append(
                round_robin_sample(
                    score_frame,
                    take,
                    ["source", "target_stock", "month", "length_bucket"],
                )
            )
            remaining_needed -= take
        if remaining_needed == 0:
            break
    if remaining_needed:
        raise RuntimeError(f"Future Negative 후보 표본이 {remaining_needed}건 부족합니다.")
    result = pd.concat(selected_parts, ignore_index=True)
    result["sampling_group"] = "future_negative_candidate"
    return result


def _sample_group_a(remaining: pd.DataFrame) -> pd.DataFrame:
    toss = round_robin_sample(
        remaining[remaining["source"].eq("toss_community")],
        ADDITIONAL_TOSS_SIZE,
        ["target_stock", "month", "length_bucket"],
    )
    edaily = round_robin_sample(
        remaining[remaining["source"].eq("edaily_news")],
        ADDITIONAL_EDAILY_SIZE,
        ["target_stock", "month", "length_bucket"],
    )
    result = pd.concat([toss, edaily], ignore_index=True)
    if len(result) != ADDITIONAL_RANDOM_SIZE:
        raise RuntimeError(f"일반 표본 수 오류: {len(result)}")
    result["sampling_group"] = "random"
    result["candidate_score"] = 0
    result["future_keyword_hits"] = 0
    result["negative_keyword_hits"] = 0
    return result


def build_additional_sample() -> tuple[pd.DataFrame, dict]:
    data = load_unified_data()
    development, _, _ = split_time_pools(data)
    existing_path = _find_existing_sample()
    existing, existing_items, existing_originals, existing_texts = _load_existing_keys(existing_path)
    available = _exclude_keys(development, existing_items, existing_originals, existing_texts)
    available = _deduplicate_for_labeling(available)

    candidate_pool = build_future_negative_candidates(available)
    group_b = _sample_group_b(candidate_pool)
    b_items = set(group_b["item_id"].astype(str))
    b_originals = set(group_b["source"].astype(str) + ":" + group_b["original_id"].astype(str))
    b_texts = set(group_b["normalized_text_hash"])
    remaining = _exclude_keys(available, b_items, b_originals, b_texts)
    group_a = _sample_group_a(remaining)

    sample = pd.concat([group_a, group_b], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["current_label"] = ""
    sample["future_label"] = ""
    sample = sample[OUTPUT_COLUMNS]

    if len(group_a) != 1000 or len(group_b) != 500 or len(sample) != 1500:
        raise RuntimeError("A/B/전체 표본 수가 목표와 다릅니다.")
    if (pd.to_datetime(sample["datetime"]) >= ANALYSIS_START).any():
        raise RuntimeError("추가 표본에 2026-05-14 이후 데이터가 포함됐습니다.")
    if sample["item_id"].duplicated().any() or sample["normalized_text_hash"].duplicated().any():
        raise RuntimeError("추가 1,500건 내부에 ID 또는 exact text 중복이 있습니다.")
    if set(sample["item_id"].astype(str)) & existing_items:
        raise RuntimeError("기존 500건과 item_id가 겹칩니다.")
    if set(sample["normalized_text_hash"]) & existing_texts:
        raise RuntimeError("기존 500건과 normalized text가 겹칩니다.")
    if sample["current_label"].ne("").any() or sample["future_label"].ne("").any():
        raise RuntimeError("라벨 컬럼이 비어 있지 않습니다.")

    lengths = sample["text"].str.len()
    report = {
        "seed": SEED,
        "existing_sample_file": str(existing_path),
        "existing_rows": int(len(existing)),
        "development_rows": int(len(development)),
        "available_after_existing_and_exact_dedup": int(len(available)),
        "future_negative_candidate_pool": int(len(candidate_pool)),
        "rows": int(len(sample)),
        "sampling_group": sample["sampling_group"].value_counts().to_dict(),
        "source": sample["source"].value_counts().to_dict(),
        "month": pd.to_datetime(sample["datetime"]).dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
        "target_stock": sample["target_stock"].astype(str).value_counts().to_dict(),
        "group_distributions": {
            group: {
                "source": frame["source"].value_counts().to_dict(),
                "month": pd.to_datetime(frame["datetime"]).dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
                "target_stock": frame["target_stock"].astype(str).value_counts().to_dict(),
            }
            for group, frame in sample.groupby("sampling_group", sort=True)
        },
        "text_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "candidate_keyword_hits": {
            "future_total": int(group_b["future_keyword_hits"].sum()),
            "negative_total": int(group_b["negative_keyword_hits"].sum()),
            "candidate_score": group_b["candidate_score"].value_counts().sort_index().to_dict(),
        },
        "validation": {
            "datetime_min": pd.to_datetime(sample["datetime"]).min().isoformat(),
            "datetime_max": pd.to_datetime(sample["datetime"]).max().isoformat(),
            "existing_item_id_intersection": 0,
            "existing_normalized_text_intersection": 0,
            "internal_duplicate_item_id": 0,
            "internal_duplicate_normalized_text": 0,
            "blank_current_label": int(sample["current_label"].eq("").sum()),
            "blank_future_label": int(sample["future_label"].eq("").sum()),
        },
    }
    return sample, report


def main() -> None:
    sample, report = build_additional_sample()
    ensure_output_dir()
    sample.to_csv(
        ADDITIONAL_SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    ADDITIONAL_SAMPLE_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"추가 표본 저장: {ADDITIONAL_SAMPLE_FILE}")
    print(f"요약 저장: {ADDITIONAL_SAMPLE_REPORT_FILE}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
