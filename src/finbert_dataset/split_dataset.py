"""라벨 완료 CSV를 중복 그룹 단위의 Train/Validation/Test로 고정 분할한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_POOL_FILE,
    ANALYSIS_START,
    DEVELOPMENT_START,
    OUTPUT_DIR,
    SEED,
    SPLIT_RATIOS,
    VALID_LABELS,
    ensure_output_dir,
)
from .utils import normalize_text, pairwise_intersections, stable_hash


FEATURE_COLUMNS = ["source", "target_stock", "current_label", "future_label"]


def validate_labeled_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {"item_id", "source", "target_stock", "datetime", "text", "current_label", "future_label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"입력 CSV 필수 컬럼 누락: {sorted(missing)}")

    result = data.copy()
    for column in ("current_label", "future_label"):
        result[column] = result[column].fillna("").astype(str).str.strip().str.lower()
        blank_count = int(result[column].eq("").sum())
        invalid = sorted(set(result[column]) - VALID_LABELS - {""})
        if blank_count or invalid:
            raise ValueError(f"{column}: 빈 라벨 {blank_count}건, 허용되지 않은 라벨 {invalid}")

    result["text"] = result["text"].fillna("").astype(str)
    empty_text = int(result["text"].str.strip().eq("").sum())
    if empty_text:
        raise ValueError(f"빈 text가 {empty_text}건 있습니다.")

    result["datetime"] = pd.to_datetime(result["datetime"], format="mixed", errors="coerce")
    invalid_datetime = int(result["datetime"].isna().sum())
    outside_development = int(
        (~result["datetime"].ge(DEVELOPMENT_START) | ~result["datetime"].lt(ANALYSIS_START)).sum()
    )
    if invalid_datetime or outside_development:
        raise ValueError(
            f"datetime 오류 {invalid_datetime}건, Development 기간 밖 데이터 {outside_development}건"
        )

    normalized_hash = result["text"].map(normalize_text).map(stable_hash)
    if "normalized_text_hash" in result:
        mismatch = int(result["normalized_text_hash"].fillna("").astype(str).ne(normalized_hash).sum())
        if mismatch:
            raise ValueError(f"normalized_text_hash가 현재 text와 다른 행이 {mismatch}건 있습니다.")
    result["normalized_text_hash"] = normalized_hash
    # 같은 원본 ID 또는 같은 정규화 문장으로 연결된 행은 하나의 연결요소로 묶는다.
    # A-B가 같은 ID이고 B-C가 같은 text인 연쇄 중복도 세 행 모두 같은 split에 남는다.
    positions = list(range(len(result)))
    parent = positions.copy()

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    if "original_id" in result:
        original_key = result["source"].astype(str) + ":original:" + result["original_id"].astype(str)
    else:
        original_key = result["source"].astype(str) + ":item:" + result["item_id"].astype(str)
    text_key = "text:" + result["normalized_text_hash"]

    for keys in (original_key, text_key):
        first_position: dict[str, int] = {}
        for position, key in enumerate(keys):
            if key in first_position:
                union(first_position[key], position)
            else:
                first_position[key] = position

    roots = [find(position) for position in positions]
    component_positions: dict[int, list[int]] = defaultdict(list)
    for position, root in enumerate(roots):
        component_positions[root].append(position)
    component_names = {}
    for root, members in component_positions.items():
        item_ids = sorted(result.iloc[members]["item_id"].astype(str))
        component_names[root] = "group:" + stable_hash("|".join(item_ids))
    result["split_group"] = [component_names[root] for root in roots]
    return result


def _target_counts(data: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    sizes = {split: len(data) * ratio for split, ratio in SPLIT_RATIOS.items()}
    features: dict[str, dict[str, dict[str, float]]] = {}
    for column in FEATURE_COLUMNS:
        totals = data[column].astype(str).value_counts().to_dict()
        features[column] = {
            split: {value: count * SPLIT_RATIOS[split] for value, count in totals.items()}
            for split in SPLIT_RATIOS
        }
    return sizes, features


def assign_groups(data: pd.DataFrame) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    groups = [(name, frame.index.tolist()) for name, frame in data.groupby("split_group", sort=True)]
    feature_totals = {column: data[column].astype(str).value_counts().to_dict() for column in FEATURE_COLUMNS}

    def rarity(indexes: list[int]) -> float:
        score = 0.0
        frame = data.loc[indexes]
        for column in FEATURE_COLUMNS:
            score += sum(1.0 / feature_totals[column][str(value)] for value in frame[column].astype(str))
        return score

    tie_breakers = {name: float(rng.random()) for name, _ in groups}
    groups.sort(key=lambda item: (-rarity(item[1]), -len(item[1]), tie_breakers[item[0]], item[0]))

    target_sizes, target_features = _target_counts(data)
    current_sizes = Counter()
    current_features = {split: {column: Counter() for column in FEATURE_COLUMNS} for split in SPLIT_RATIOS}
    assignment: dict[str, str] = {}

    for group_name, indexes in groups:
        frame = data.loc[indexes]
        candidates = []
        for split in SPLIT_RATIOS:
            projected_size = current_sizes[split] + len(frame)
            size_fill = projected_size / max(target_sizes[split], 1)
            overflow = max(0.0, projected_size - np.ceil(target_sizes[split]))
            feature_fills = []
            for column in FEATURE_COLUMNS:
                additions = frame[column].astype(str).value_counts()
                for value, count in additions.items():
                    projected = current_features[split][column][value] + int(count)
                    target = target_features[column][split].get(value, 0.0)
                    feature_fills.append(projected / max(target, 1.0))
            feature_fill = float(np.mean(feature_fills)) if feature_fills else 0.0
            candidates.append((size_fill * 8 + feature_fill + overflow * 100, split))

        _, chosen = min(candidates, key=lambda value: (value[0], list(SPLIT_RATIOS).index(value[1])))
        assignment[group_name] = chosen
        current_sizes[chosen] += len(frame)
        for column in FEATURE_COLUMNS:
            current_features[chosen][column].update(frame[column].astype(str))

    return assignment


def validate_no_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    checks = {
        "item_id": pairwise_intersections({name: frame["item_id"].astype(str) for name, frame in splits.items()}),
        "normalized_text_hash": pairwise_intersections(
            {name: frame["normalized_text_hash"].astype(str) for name, frame in splits.items()}
        ),
        "split_group": pairwise_intersections({name: frame["split_group"].astype(str) for name, frame in splits.items()}),
    }
    problems = {kind: pairs for kind, pairs in checks.items() if any(pairs.values())}
    if problems:
        raise RuntimeError(f"split leakage 발견: {problems}")
    return checks


def validate_analysis_separation(
    splits: dict[str, pd.DataFrame], analysis: pd.DataFrame
) -> dict[str, dict[str, int]]:
    development = pd.concat(splits.values(), ignore_index=True)
    if (development["datetime"] >= ANALYSIS_START).any():
        raise RuntimeError("Train/Validation/Test에 2026-05-14 이후 데이터가 포함됐습니다.")

    def original_keys(frame: pd.DataFrame) -> pd.Series:
        if "original_id" in frame:
            return frame["source"].astype(str) + ":" + frame["original_id"].astype(str)
        return frame["item_id"].astype(str)

    id_checks = pairwise_intersections(
        {
            **{name: original_keys(frame) for name, frame in splits.items()},
            "analysis": original_keys(analysis),
        }
    )
    analysis_id_problems = {
        pair: count for pair, count in id_checks.items() if "analysis" in pair and count
    }
    if analysis_id_problems:
        raise RuntimeError(f"Development/Analysis ID leakage 발견: {analysis_id_problems}")

    text_checks = pairwise_intersections(
        {
            **{name: frame["normalized_text_hash"].astype(str) for name, frame in splits.items()},
            "analysis": analysis["normalized_text_hash"].astype(str),
        }
    )
    return {"original_id": id_checks, "normalized_text_hash_report_only": text_checks}


def build_summary(splits: dict[str, pd.DataFrame], leakage: dict, analysis_separation: dict) -> dict:
    total = sum(len(frame) for frame in splits.values())
    summary = {
        "seed": SEED,
        "total_rows": total,
        "ratios": SPLIT_RATIOS,
        "development_internal_leakage": leakage,
        "analysis_separation": analysis_separation,
        "splits": {},
    }
    for name, frame in splits.items():
        summary["splits"][name] = {
            "rows": int(len(frame)),
            "ratio": len(frame) / total if total else 0.0,
            "source": frame["source"].value_counts().to_dict(),
            "target_stock": frame["target_stock"].astype(str).value_counts().to_dict(),
            "current_label": frame["current_label"].value_counts().to_dict(),
            "future_label": frame["future_label"].value_counts().to_dict(),
        }
    return summary


def print_summary(summary: dict) -> None:
    print(f"SEED={summary['seed']} | 전체 {summary['total_rows']}건")
    for name, values in summary["splits"].items():
        print(f"\n[{name}] {values['rows']}건 ({values['ratio']:.1%})")
        for column in FEATURE_COLUMNS:
            print(f"  {column}: {values[column]}")
    print("\nDevelopment 내부 leakage:", summary["development_internal_leakage"])
    print("Analysis 분리 검증:", summary["analysis_separation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Current/Future 라벨 입력이 완료된 CSV")
    args = parser.parse_args()

    data = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"target_stock": str})
    data = validate_labeled_data(data)
    assignment = assign_groups(data)
    data["split"] = data["split_group"].map(assignment)
    splits = {
        name: data[data["split"].eq(name)].copy().reset_index(drop=True)
        for name in SPLIT_RATIOS
    }
    leakage = validate_no_leakage(splits)
    if not ANALYSIS_POOL_FILE.exists():
        raise FileNotFoundError(
            f"Analysis pool이 없습니다: {ANALYSIS_POOL_FILE}. 먼저 prepare_pools를 실행하세요."
        )
    analysis = pd.read_csv(
        ANALYSIS_POOL_FILE,
        encoding="utf-8-sig",
        dtype={"target_stock": str, "original_id": str},
        parse_dates=["datetime"],
    )
    analysis_separation = validate_analysis_separation(splits, analysis)
    summary = build_summary(splits, leakage, analysis_separation)

    ensure_output_dir()
    for name, frame in splits.items():
        output = OUTPUT_DIR / f"{name}_seed{SEED}.csv"
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"저장: {output}")
    summary_path = OUTPUT_DIR / f"split_summary_seed{SEED}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
