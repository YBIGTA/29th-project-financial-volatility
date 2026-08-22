from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
STATISTICS_DIR = ANALYSIS_DIR / "statistics"

STOCK_FILES = {
    "005930": "analysis_samsung_electronics.csv",
    "000660": "analysis_sk_hynix.csv",
    "247540": "analysis_ecopro_bm.csv",
    "035720": "analysis_kakao.csv",
}


def main() -> None:
    all_rows = []

    for stock_code, filename in STOCK_FILES.items():
        path = ANALYSIS_DIR / filename

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = df.sort_values("date").reset_index(drop=True)

        df["previous_trading_date"] = df["date"].shift(1)

        df["calendar_gap_days"] = (
            df["date"]
            - df["previous_trading_date"]
        ).dt.days

        df["weekend_or_holiday_gap"] = (
            df["calendar_gap_days"] > 1
        )

        selected = df[
            [
                "date",
                "previous_trading_date",
                "calendar_gap_days",
                "weekend_or_holiday_gap",
                "overnight_sentiment_ffill",
                "overnight_item_count",
                "overnight_no_posts",
            ]
        ].copy()

        selected.insert(
            0,
            "stock_code",
            stock_code,
        )

        all_rows.append(selected)

    result = pd.concat(
        all_rows,
        ignore_index=True,
    )

    output_path = (
        STATISTICS_DIR
        / "overnight_holiday_audit.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        result.groupby(
            [
                "stock_code",
                "weekend_or_holiday_gap",
            ]
        )
        .agg(
            n=("date", "count"),
            mean_sentiment=(
                "overnight_sentiment_ffill",
                "mean",
            ),
            std_sentiment=(
                "overnight_sentiment_ffill",
                "std",
            ),
            mean_item_count=(
                "overnight_item_count",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        STATISTICS_DIR
        / "overnight_holiday_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(summary.to_string(index=False))
    print(f"\n상세 결과: {output_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    main()