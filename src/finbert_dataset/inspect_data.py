"""6개월 이데일리·토스 데이터의 schema, 분포와 중복 현황을 점검한다."""

from __future__ import annotations

import json

import pandas as pd

from .config import OUTPUT_DIR, SEED, ensure_output_dir
from .utils import load_unified_data, split_time_pools


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def build_report(data: pd.DataFrame) -> dict:
    reports = {}
    for source, frame in data.groupby("source", sort=True):
        normalized = frame["normalized_text_hash"]
        reports[source] = {
            "rows": int(len(frame)),
            "datetime_min": None if frame["datetime"].isna().all() else frame["datetime"].min().isoformat(),
            "datetime_max": None if frame["datetime"].isna().all() else frame["datetime"].max().isoformat(),
            "stock_counts": _counts(frame["target_stock"]),
            "monthly_counts": _counts(frame["month"]),
            "missing": {
                "item_id": int(frame["item_id"].isna().sum()),
                "target_stock": int(frame["target_stock"].isna().sum()),
                "datetime": int(frame["datetime"].isna().sum()),
                "text": int(frame["text"].str.strip().eq("").sum()),
                "headline": int(frame["headline"].str.strip().eq("").sum()),
                "body": int(frame["body"].str.strip().eq("").sum()),
            },
            "duplicate_item_id_rows": int(frame["item_id"].duplicated(keep=False).sum()),
            "duplicate_original_id_rows": int(frame["original_id"].duplicated(keep=False).sum()),
            "exact_duplicate_text_rows": int(normalized.duplicated(keep=False).sum()),
            "headline_equals_body_rows": int(
                (frame["headline"].str.strip().ne("") & frame["headline"].eq(frame["body"])).sum()
            ),
        }
    return {"seed": SEED, "total_rows": int(len(data)), "sources": reports}


def print_report(report: dict) -> None:
    print(f"SEED={report['seed']} | 전체 {report['total_rows']:,}건")
    for source, values in report["sources"].items():
        print(f"\n[{source}] {values['rows']:,}건 | {values['datetime_min']} ~ {values['datetime_max']}")
        print("  종목별:", values["stock_counts"])
        print("  월별:", values["monthly_counts"])
        print("  결측:", values["missing"])
        print(
            "  중복: item_id 행={duplicate_item_id_rows:,}, original_id 행={duplicate_original_id_rows:,}, "
            "정규화 text 행={exact_duplicate_text_rows:,}".format(**values)
        )
        print(f"  headline==body: {values['headline_equals_body_rows']:,}건")


def main() -> None:
    data = load_unified_data()
    report = build_report(data)
    development, analysis, outside = split_time_pools(data)
    report["time_pools"] = {
        "development": int(len(development)),
        "analysis": int(len(analysis)),
        "outside_experiment_window": int(len(outside)),
    }
    print_report(report)
    print("\n기간 분리:", report["time_pools"])
    ensure_output_dir()
    output = OUTPUT_DIR / f"inspection_report_seed{SEED}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트 저장: {output}")


if __name__ == "__main__":
    main()
