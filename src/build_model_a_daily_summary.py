"""Aggregate Model A article scores by trading session and merge daily prices."""

from __future__ import annotations

import argparse
import json
import zipfile
from bisect import bisect_right
from pathlib import Path

import pandas as pd


STOCK_FILES = {
    "005930": "samsung_electronics",
    "000660": "sk_hynix",
    "035720": "kakao",
    "247540": "ecopro_bm",
}
HEADS = ("current", "future")
SESSIONS = ("intraday", "overnight")
LABELS = ("negative", "neutral", "positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored-dir", type=Path, default=Path("data/3개월_학습후_분석/processed")
    )
    parser.add_argument("--price-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--summary-dir", type=Path, default=Path("data/3개월_학습후_분석/정리")
    )
    parser.add_argument("--market-zip", type=Path)
    parser.add_argument("--vkospi", type=Path)
    parser.add_argument("--start-date", default="2026-05-14")
    parser.add_argument("--end-date", default="2026-08-14")
    return parser.parse_args()


def load_trading_dates(
    price: pd.DataFrame, timestamps: pd.Series, end_date: pd.Timestamp
) -> tuple[list, set]:
    price_dates = sorted(pd.to_datetime(price["date"]).dt.normalize().unique().tolist())
    price_date_set = set(price_dates)
    if price_dates and (timestamps >= price_dates[-1] + pd.Timedelta(hours=15, minutes=30)).any():
        provisional_date = price_dates[-1] + pd.offsets.BDay(1)
        if provisional_date <= end_date and provisional_date not in price_date_set:
            price_dates.append(provisional_date)
    return price_dates, price_date_set


def assign_session(timestamp: pd.Timestamp, trading_dates: list, trading_set: set):
    date = timestamp.normalize()
    minute = timestamp.hour * 60 + timestamp.minute
    if date in trading_set and 9 * 60 <= minute < 15 * 60 + 30:
        return date, "intraday"
    if date in trading_set and minute < 9 * 60:
        return date, "overnight"
    next_index = bisect_right(trading_dates, date)
    if next_index < len(trading_dates):
        return trading_dates[next_index], "overnight"
    return pd.NaT, "overnight"


def validate_article_scores(frame: pd.DataFrame, path: Path) -> None:
    required = ["item_id", "datetime", "target_stock"]
    for head in HEADS:
        required.extend(f"model_{head}_p_{label}" for label in LABELS)
        required.append(f"{head}_sentiment_score")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError(f"{path}: missing values in required Model A columns")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate item_id values")
    for head in HEADS:
        probability_columns = [f"model_{head}_p_{label}" for label in LABELS]
        probability_error = (frame[probability_columns].sum(axis=1) - 1).abs().max()
        formula = (
            frame[f"model_{head}_p_positive"].pow(2)
            - frame[f"model_{head}_p_negative"].pow(2)
        )
        formula_error = (frame[f"{head}_sentiment_score"] - formula).abs().max()
        if probability_error > 1e-5 or formula_error > 1e-5:
            raise ValueError(
                f"{path}: {head} validation failed "
                f"(probability_error={probability_error}, formula_error={formula_error})"
            )


def build_daily_index(
    scored: pd.DataFrame, price: pd.DataFrame, end_date: str
) -> tuple[pd.DataFrame, dict]:
    scored = scored.copy()
    scored["datetime"] = pd.to_datetime(scored["datetime"], errors="raise")
    trading_dates, price_date_set = load_trading_dates(
        price, scored["datetime"], pd.Timestamp(end_date)
    )
    trading_set = set(trading_dates)
    assignments = scored["datetime"].map(
        lambda value: assign_session(value, trading_dates, trading_set)
    )
    scored[["session_date", "session"]] = pd.DataFrame(
        assignments.tolist(), index=scored.index
    )
    unassigned = scored.loc[scored["session_date"].isna()].copy()
    assigned = scored.dropna(subset=["session_date"]).copy()

    aggregate_columns = []
    for head in HEADS:
        aggregate_columns.append(f"{head}_sentiment_score")
        aggregate_columns.extend(f"model_{head}_p_{label}" for label in LABELS)
    grouped = assigned.groupby(["session_date", "session"])[aggregate_columns].agg(
        ["mean", "size"]
    )

    result = pd.DataFrame({"date": trading_dates})
    for session in SESSIONS:
        if session in grouped.index.get_level_values("session"):
            session_values = grouped.xs(session, level="session")
        else:
            session_values = pd.DataFrame()

        for head in HEADS:
            source_score = (f"{head}_sentiment_score", "mean")
            destination_score = f"{session}_{head}_sentiment_index"
            if source_score in session_values.columns:
                result[destination_score] = result["date"].map(session_values[source_score])
            else:
                result[destination_score] = float("nan")
            result[destination_score] = result[destination_score].fillna(0.0)

            for label in LABELS:
                source_probability = (f"model_{head}_p_{label}", "mean")
                destination_probability = f"{session}_{head}_{label}_prob_mean"
                if source_probability in session_values.columns:
                    result[destination_probability] = result["date"].map(
                        session_values[source_probability]
                    )
                else:
                    result[destination_probability] = float("nan")
                result[destination_probability] = result[destination_probability].fillna(0.0)

        size_column = ("current_sentiment_score", "size")
        if size_column in session_values.columns:
            counts = result["date"].map(session_values[size_column])
        else:
            counts = pd.Series(0, index=result.index, dtype=float)
        result[f"{session}_item_count"] = counts.fillna(0).astype(int)
        result[f"{session}_no_posts"] = result[f"{session}_item_count"].eq(0).astype(int)

    result["price_available"] = result["date"].map(
        lambda date: int(pd.Timestamp(date) in price_date_set)
    )
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    audit = {
        "input_articles": len(scored),
        "assigned_articles": len(assigned),
        "unassigned_articles": len(unassigned),
        "unassigned_min_datetime": (
            str(unassigned["datetime"].min()) if not unassigned.empty else None
        ),
        "unassigned_max_datetime": (
            str(unassigned["datetime"].max()) if not unassigned.empty else None
        ),
        "daily_rows": len(result),
        "price_rows": len(price),
        "provisional_rows": int(result["price_available"].eq(0).sum()),
    }
    return result, audit


def merge_prices(
    daily: pd.DataFrame, price: pd.DataFrame, vix: pd.DataFrame
) -> pd.DataFrame:
    price = price.copy()
    vix = vix.rename(
        columns={
            "open": "vix_open",
            "high": "vix_high",
            "low": "vix_low",
            "close": "vix_close",
            "volume": "vix_volume",
        }
    )
    for frame in (daily, price, vix):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame["date"].duplicated().any():
            raise ValueError("Duplicate date encountered while merging daily data")
    merged = daily.merge(price, on="date", how="left", validate="one_to_one")
    merged = merged.merge(vix, on="date", how="left", validate="one_to_one")
    columns = ["date"]
    for session in SESSIONS:
        columns.extend(
            [
                f"{session}_current_sentiment_index",
                f"{session}_future_sentiment_index",
                f"{session}_item_count",
                f"{session}_no_posts",
            ]
        )
    columns.extend(
        [
            "open", "high", "low", "close", "volume",
            "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume",
        ]
    )
    merged = merged[columns].sort_values("date").reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def read_market_zip(zip_path: Path, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        return pd.read_csv(archive.open(f"market_index/{name}.csv"), encoding="utf-8-sig")


def filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    result = result.loc[result["date"].between(start, end)].copy()
    if result["date"].duplicated().any():
        raise ValueError("Duplicate market-index dates")
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result.sort_values("date").reset_index(drop=True)


def build_market_index_files(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.market_zip is None or args.vkospi is None:
        return {}
    frames = {
        "kospi": read_market_zip(args.market_zip, "kospi"),
        "kosdaq": read_market_zip(args.market_zip, "kosdaq"),
        "vix": read_market_zip(args.market_zip, "vix"),
        "vkospi": pd.read_csv(args.vkospi, encoding="utf-8-sig"),
    }
    frames = {
        name: filter_dates(frame, args.start_date, args.end_date)
        for name, frame in frames.items()
    }
    output_dir = args.summary_dir / "market_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.12g",
        )

    merged = None
    for name in ("kospi", "kosdaq", "vix", "vkospi"):
        renamed = frames[name].rename(
            columns={column: f"{name}_{column}" for column in frames[name].columns if column != "date"}
        )
        merged = renamed if merged is None else merged.merge(
            renamed, on="date", how="outer", validate="one_to_one"
        )
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(
        output_dir / "market_index_merged.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    frames["market_index_merged"] = merged
    return frames


def main() -> None:
    args = parse_args()
    market_frames = build_market_index_files(args)
    if "vix" in market_frames:
        vix = market_frames["vix"]
    else:
        vix_path = args.price_dir / "price_daily_vix.csv"
        vix = pd.read_csv(vix_path, encoding="utf-8-sig")
    audits = {}

    for stock_code, file_name in STOCK_FILES.items():
        scored_path = args.scored_dir / f"sentiment_scored_{file_name}.csv"
        price_path = args.price_dir / f"price_daily_{file_name}.csv"
        scored = pd.read_csv(
            scored_path,
            encoding="utf-8-sig",
            dtype={"target_stock": "string", "original_id": "string"},
        )
        validate_article_scores(scored, scored_path)
        if set(scored["target_stock"].unique()) != {stock_code}:
            raise ValueError(f"{scored_path}: unexpected stock code")
        price = pd.read_csv(price_path, encoding="utf-8-sig")
        price = filter_dates(price, args.start_date, args.end_date)

        daily, audit = build_daily_index(scored, price, args.end_date)
        daily_output = args.scored_dir / f"sentiment_index_daily_{file_name}.csv"
        daily.to_csv(daily_output, index=False, encoding="utf-8-sig", float_format="%.12g")

        merged = merge_prices(daily, price, vix)
        summary_output = args.summary_dir / file_name / "merged_daily.csv"
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(
            summary_output, index=False, encoding="utf-8-sig", float_format="%.12g"
        )
        audit.update(
            {
                "stock_code": stock_code,
                "daily_output": str(daily_output),
                "merged_output": str(summary_output),
                "missing_stock_price_rows": int(merged["close"].isna().sum()),
                "missing_vix_rows": int(merged["vix_close"].isna().sum()),
            }
        )
        audits[file_name] = audit
        print(
            f"{stock_code} {file_name}: articles={len(scored)}, "
            f"daily={len(daily)}, unassigned={audit['unassigned_articles']}"
        )

    audit_path = args.summary_dir / "aggregation_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
