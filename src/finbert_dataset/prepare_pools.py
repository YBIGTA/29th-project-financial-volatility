"""6개월 데이터를 과거 Development와 미래 Final Analysis 구간으로 분리한다."""

from __future__ import annotations

import json

import pandas as pd

from .config import (
    ANALYSIS_END_EXCLUSIVE,
    ANALYSIS_POOL_FILE,
    ANALYSIS_START,
    DEVELOPMENT_START,
    POOL_REPORT_FILE,
    SEED,
    ensure_output_dir,
)
from .utils import load_unified_data, normalize_text, pairwise_intersections, split_time_pools


def _span(frame: pd.DataFrame) -> dict[str, str | int | None]:
    return {
        "rows": int(len(frame)),
        "datetime_min": None if frame.empty else frame["datetime"].min().isoformat(),
        "datetime_max": None if frame.empty else frame["datetime"].max().isoformat(),
    }


def build_pool_report(
    all_data: pd.DataFrame,
    development: pd.DataFrame,
    analysis: pd.DataFrame,
    outside: pd.DataFrame,
) -> dict:
    source_report = {}
    for source in sorted(all_data["source"].unique()):
        source_report[source] = {
            "total_loaded": _span(all_data[all_data["source"].eq(source)]),
            "development": _span(development[development["source"].eq(source)]),
            "analysis": _span(analysis[analysis["source"].eq(source)]),
            "outside_experiment_window": _span(outside[outside["source"].eq(source)]),
        }

    id_overlap = pairwise_intersections(
        {
            "development": development["source"].astype(str) + ":" + development["original_id"].astype(str),
            "analysis": analysis["source"].astype(str) + ":" + analysis["original_id"].astype(str),
        }
    )
    text_overlap = pairwise_intersections(
        {
            "development": development["text"].map(normalize_text),
            "analysis": analysis["text"].map(normalize_text),
        }
    )
    if any(id_overlap.values()):
        raise RuntimeError(f"Development/Analysis 원본 ID leakage 발견: {id_overlap}")

    return {
        "seed": SEED,
        "boundaries": {
            "development": f"{DEVELOPMENT_START.isoformat()} <= datetime < {ANALYSIS_START.isoformat()}",
            "analysis": f"{ANALYSIS_START.isoformat()} <= datetime < {ANALYSIS_END_EXCLUSIVE.isoformat()}",
        },
        "total_loaded": _span(all_data),
        "development_pool": _span(development),
        "analysis_pool": _span(analysis),
        "outside_experiment_window": _span(outside),
        "sources": source_report,
        "cross_pool_original_id_overlap": id_overlap,
        "cross_pool_normalized_text_overlap": text_overlap,
    }


def print_pool_report(report: dict) -> None:
    print(f"SEED={SEED}")
    print("전체 로드:", report["total_loaded"])
    print("Development:", report["development_pool"])
    print("Analysis:", report["analysis_pool"])
    print("실험기간 외:", report["outside_experiment_window"])
    for source, values in report["sources"].items():
        print(f"\n[{source}]")
        print("  total:", values["total_loaded"])
        print("  development:", values["development"])
        print("  analysis:", values["analysis"])
    print("\nDevelopment/Analysis ID 교집합:", report["cross_pool_original_id_overlap"])
    print("Development/Analysis 동일 normalized text:", report["cross_pool_normalized_text_overlap"])


def main() -> None:
    data = load_unified_data()
    development, analysis, outside = split_time_pools(data)
    report = build_pool_report(data, development, analysis, outside)
    ensure_output_dir()
    analysis.to_csv(ANALYSIS_POOL_FILE, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    POOL_REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_pool_report(report)
    print(f"\nAnalysis pool 저장: {ANALYSIS_POOL_FILE}")
    print(f"기간 분리 리포트 저장: {POOL_REPORT_FILE}")


if __name__ == "__main__":
    main()
