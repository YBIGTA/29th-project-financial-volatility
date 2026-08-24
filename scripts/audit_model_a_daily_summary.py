"""Audit the stock-specific Model A daily merged files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SESSIONS = ("intraday", "overnight")


def check_ohlc(frame: pd.DataFrame, path: Path, prefix: str = "") -> None:
    columns = [f"{prefix}{name}" for name in ("open", "high", "low", "close")]
    available = frame.dropna(subset=columns)
    invalid = (
        available[f"{prefix}high"].lt(
            available[[f"{prefix}open", f"{prefix}close"]].max(axis=1)
        )
        | available[f"{prefix}low"].gt(
            available[[f"{prefix}open", f"{prefix}close"]].min(axis=1)
        )
    )
    if invalid.any():
        raise ValueError(f"{path}: invalid {prefix}OHLC rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    source_audit = json.loads(
        (args.summary_dir / "aggregation_audit.json").read_text(encoding="utf-8")
    )
    results = {}

    for stock_name, source in source_audit.items():
        path = args.summary_dir / stock_name / "merged_daily.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        expected_stock_columns = ["date"]
        for session in SESSIONS:
            expected_stock_columns.extend(
                [
                    f"{session}_current_sentiment_index",
                    f"{session}_future_sentiment_index",
                    f"{session}_item_count",
                    f"{session}_no_posts",
                ]
            )
        expected_stock_columns.extend(
            [
                "open", "high", "low", "close", "volume",
                "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume",
            ]
        )
        if list(frame.columns) != expected_stock_columns:
            raise ValueError(f"{path}: unexpected columns")
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{path}: invalid date ordering")

        assigned_from_counts = 0
        for session in SESSIONS:
            count = frame[f"{session}_item_count"]
            no_posts = frame[f"{session}_no_posts"]
            if not count.eq(0).eq(no_posts.eq(1)).all():
                raise ValueError(f"{path}: inconsistent {session} no_posts flag")
            assigned_from_counts += int(count.sum())
            for head in ("current", "future"):
                score = frame[f"{session}_{head}_sentiment_index"]
                if not score.between(-1, 1).all():
                    raise ValueError(f"{path}: {session} {head} score outside [-1, 1]")
                if not score[count.eq(0)].eq(0).all():
                    raise ValueError(f"{path}: nonzero score in no-post rows")

        if assigned_from_counts != source["assigned_articles"]:
            raise ValueError(
                f"{path}: assigned count mismatch "
                f"({assigned_from_counts} != {source['assigned_articles']})"
            )
        check_ohlc(frame, path)
        check_ohlc(frame, path, "vix_")
        daily_path = (
            args.summary_dir.parent
            / "processed"
            / f"sentiment_index_daily_{stock_name}.csv"
        )
        daily = pd.read_csv(daily_path, encoding="utf-8-sig").set_index("date")
        merged_indexed = frame.set_index("date")
        comparison_columns = expected_stock_columns[1:9]
        difference = (
            merged_indexed.loc[daily.index, comparison_columns]
            - daily[comparison_columns]
        ).abs().max().max()
        if difference > 1e-9:
            raise ValueError(f"{path}: sentiment values do not tie to {daily_path}")
        results[stock_name] = {
            "rows": len(frame),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "assigned_articles": assigned_from_counts,
            "unassigned_articles": source["unassigned_articles"],
            "missing_stock_price_rows": int(frame["close"].isna().sum()),
            "missing_vix_rows": int(frame["vix_close"].isna().sum()),
            "max_sentiment_tie_error": float(difference),
        }

    market_dir = args.summary_dir / "market_index"
    market_results = {}
    market_frames = {}
    expected_columns = {
        "kospi": ["date", "open", "high", "low", "close", "volume"],
        "kosdaq": ["date", "open", "high", "low", "close", "volume"],
        "vix": ["date", "open", "high", "low", "close", "volume"],
        "vkospi": ["date", "open", "high", "low", "close"],
    }
    for name, columns in expected_columns.items():
        path = market_dir / f"{name}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if list(frame.columns) != columns:
            raise ValueError(f"{path}: unexpected columns")
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{path}: invalid dates")
        missing_rows = frame.drop(columns="date").isna().any(axis=1)
        if missing_rows.any():
            if name != "vix" or not frame.loc[missing_rows].drop(columns="date").isna().all().all():
                raise ValueError(f"{path}: unexpected partial missing values")
        check_ohlc(frame, path)
        market_frames[name] = frame
        market_results[name] = {
            "rows": len(frame),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "missing_market_rows": int(missing_rows.sum()),
        }

    merged_path = market_dir / "market_index_merged.csv"
    merged = pd.read_csv(merged_path, encoding="utf-8-sig")
    merged_dates = pd.to_datetime(merged["date"], errors="raise")
    if merged_dates.duplicated().any() or not merged_dates.is_monotonic_increasing:
        raise ValueError(f"{merged_path}: invalid dates")
    for name, source_frame in market_frames.items():
        source = source_frame.set_index("date")
        target = merged.set_index("date")
        for column in source.columns:
            target_column = f"{name}_{column}"
            difference = (target.loc[source.index, target_column] - source[column]).abs().max()
            if difference > 1e-9:
                raise ValueError(f"{merged_path}: mismatch in {target_column}")
    market_results["market_index_merged"] = {
        "rows": len(merged),
        "date_min": str(merged_dates.min().date()),
        "date_max": str(merged_dates.max().date()),
    }
    results["market_index"] = market_results
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
