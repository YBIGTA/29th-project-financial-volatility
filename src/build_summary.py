"""종목별 감성지수와 가격을 병합하고 분석용 그래프를 생성한다.

출력 위치: data/정리/<영문 종목명>/
- merged_daily.csv
- intraday_sentiment.png
- overnight_sentiment.png
- sentiment_vs_close.png
- candlestick.png
- close_price_line.png
"""
from __future__ import annotations

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    ROOT_DIR,
    SIX_MONTH_PROCESSED_DIR,
    SIX_MONTH_RAW_DIR,
    SIX_MONTH_SUMMARY_DIR,
    STOCK_ENGLISH_NAMES,
    STOCK_FILE_NAMES,
)


SUMMARY_DIR = ROOT_DIR / "data" / "정리"
INPUT_RAW_DIR = RAW_DIR
INPUT_PROCESSED_DIR = PROCESSED_DIR
CSV_COLUMNS = [
    "date",
    "intraday_sentiment_index",
    "intraday_item_count",
    "intraday_no_posts",
    "overnight_sentiment_index",
    "overnight_item_count",
    "overnight_no_posts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vix_open",
    "vix_high",
    "vix_low",
    "vix_close",
    "vix_volume",
]


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9E2EC", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _set_date_ticks(ax: plt.Axes, dates: pd.Series, step: int = 10) -> None:
    positions = np.arange(len(dates))
    ticks = positions[::step]
    if len(positions) and (not len(ticks) or ticks[-1] != positions[-1]):
        if len(ticks) and positions[-1] - ticks[-1] < max(2, step // 2):
            ticks[-1] = positions[-1]
        else:
            ticks = np.append(ticks, positions[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(dates.dt.strftime("%Y-%m-%d").iloc[ticks], rotation=45, ha="right")


def load_and_merge(code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = STOCK_FILE_NAMES[code]
    sentiment_path = INPUT_PROCESSED_DIR / f"sentiment_index_daily_{label}.csv"
    price_path = INPUT_RAW_DIR / f"price_daily_{label}.csv"
    vix_path = INPUT_RAW_DIR / "price_daily_vix.csv"

    sentiment = pd.read_csv(sentiment_path, encoding="utf-8-sig", parse_dates=["date"])
    price = pd.read_csv(price_path, encoding="utf-8-sig", parse_dates=["date"])
    vix = pd.read_csv(vix_path, encoding="utf-8-sig", parse_dates=["date"]).rename(
        columns={
            "open": "vix_open",
            "high": "vix_high",
            "low": "vix_low",
            "close": "vix_close",
            "volume": "vix_volume",
        }
    )
    merged = sentiment.merge(price, on="date", how="left", validate="one_to_one")
    merged = merged.merge(vix, on="date", how="left", validate="one_to_one")
    merged = merged[CSV_COLUMNS].sort_values("date").reset_index(drop=True)
    return merged, sentiment.sort_values("date").reset_index(drop=True)


def plot_sentiment(
    merged: pd.DataFrame,
    sentiment: pd.DataFrame,
    company: str,
    session: str,
    output_path,
) -> None:
    score_column = f"{session}_sentiment_index"
    no_posts_column = f"{session}_no_posts"
    values = merged[score_column].mask(sentiment[no_posts_column].eq(1))
    color = "#2563EB" if session == "intraday" else "#7C3AED"
    session_name = "Intraday" if session == "intraday" else "Overnight"

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(merged))
    ax.plot(x, values, color=color, linewidth=1.8, marker="o", markersize=3)
    ax.axhline(0, color="#475569", linewidth=1)
    ax.fill_between(x, 0, values, where=values.ge(0), color="#EF4444", alpha=0.12)
    ax.fill_between(x, 0, values, where=values.lt(0), color="#2563EB", alpha=0.12)
    _style_axis(ax, f"{company} - {session_name} Sentiment Index", "Sentiment (-1 to +1)")
    _set_date_ticks(ax, merged["date"])
    ax.set_xlabel("Trading date")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_candlestick(merged: pd.DataFrame, company: str, output_path) -> None:
    price = merged.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    x = np.arange(len(price))
    colors = np.where(price["close"].ge(price["open"]), "#DC2626", "#2563EB")

    fig, (ax, volume_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    for index, row in price.iterrows():
        color = colors[index]
        ax.vlines(index, row["low"], row["high"], color=color, linewidth=1)
        body_low = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        if body_height == 0:
            ax.hlines(row["close"], index - 0.3, index + 0.3, color=color, linewidth=1.5)
        else:
            ax.add_patch(
                Rectangle(
                    (index - 0.3, body_low),
                    0.6,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                )
            )

    volume_ax.bar(x, price["volume"], color=colors, width=0.65, alpha=0.65)
    _style_axis(ax, f"{company} - Daily Candlestick", "Price (KRW)")
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, axis="y", color="#D9E2EC", linewidth=0.7, alpha=0.7)
    volume_ax.spines[["top", "right"]].set_visible(False)
    _set_date_ticks(volume_ax, price["date"])
    volume_ax.set_xlabel("Trading date")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_close_line(merged: pd.DataFrame, company: str, output_path) -> None:
    price = merged.dropna(subset=["close"]).reset_index(drop=True)
    x = np.arange(len(price))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(x, price["close"], color="#0F766E", linewidth=2.2)
    ax.fill_between(x, price["close"], price["close"].min(), color="#14B8A6", alpha=0.12)
    _style_axis(ax, f"{company} - Daily Closing Price", "Close price (KRW)")
    _set_date_ticks(ax, price["date"])
    ax.set_xlabel("Trading date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_sentiment_vs_close(merged: pd.DataFrame, company: str, output_path) -> None:
    """장중·장외 감성지수와 종가를 하나의 이중축 차트에 표시한다."""
    x = np.arange(len(merged))
    fig, sentiment_ax = plt.subplots(figsize=(14, 6.5))
    price_ax = sentiment_ax.twinx()

    # 무게시 구간은 집계 단계에서 0으로 채운 값을 그대로 표시한다.
    intraday_line = sentiment_ax.plot(
        x,
        merged["intraday_sentiment_index"],
        color="#2563EB",
        linewidth=1.8,
        marker="o",
        markersize=2.8,
        label="Intraday Sentiment Index",
    )[0]
    overnight_line = sentiment_ax.plot(
        x,
        merged["overnight_sentiment_index"],
        color="#7C3AED",
        linewidth=1.8,
        marker="s",
        markersize=2.6,
        label="Overnight Sentiment Index",
    )[0]
    close_line = price_ax.plot(
        x,
        merged["close"],
        color="#0F766E",
        linewidth=2.2,
        alpha=0.9,
        label="Close Price",
    )[0]

    sentiment_ax.axhline(0, color="#475569", linewidth=0.9, alpha=0.8)
    sentiment_ax.set_ylim(-1.05, 1.05)
    sentiment_ax.set_title(
        f"{company} - Intraday / Overnight Sentiment Index vs Close Price",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    sentiment_ax.set_ylabel("Sentiment Index (-1 to +1)")
    price_ax.set_ylabel("Close Price (KRW)")
    sentiment_ax.set_xlabel("Trading date")
    sentiment_ax.grid(True, color="#D9E2EC", linewidth=0.7, alpha=0.7)
    sentiment_ax.spines["top"].set_visible(False)
    price_ax.spines["top"].set_visible(False)
    _set_date_ticks(sentiment_ax, merged["date"])
    sentiment_ax.legend(
        handles=[intraday_line, overnight_line, close_line],
        loc="upper left",
        ncols=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.91, top=0.9, bottom=0.17)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_stock_summary(code: str) -> None:
    label = STOCK_FILE_NAMES[code]
    company = STOCK_ENGLISH_NAMES[code]
    output_dir = SUMMARY_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, sentiment = load_and_merge(code)
    merged.to_csv(
        output_dir / "merged_daily.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    plot_sentiment(merged, sentiment, company, "intraday", output_dir / "intraday_sentiment.png")
    plot_sentiment(merged, sentiment, company, "overnight", output_dir / "overnight_sentiment.png")
    plot_sentiment_vs_close(merged, company, output_dir / "sentiment_vs_close.png")
    plot_candlestick(merged, company, output_dir / "candlestick.png")
    plot_close_line(merged, company, output_dir / "close_price_line.png")

    missing_price_rows = int(merged["close"].isna().sum())
    print(f"[summary] {company}: {len(merged)} rows, missing price rows={missing_price_rows}")


def main() -> None:
    global SUMMARY_DIR, INPUT_RAW_DIR, INPUT_PROCESSED_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--six-month", action="store_true")
    args = parser.parse_args()
    if args.six_month:
        SUMMARY_DIR = SIX_MONTH_SUMMARY_DIR
        INPUT_RAW_DIR = SIX_MONTH_RAW_DIR
        INPUT_PROCESSED_DIR = SIX_MONTH_PROCESSED_DIR

    for code in STOCK_FILE_NAMES:
        build_stock_summary(code)


if __name__ == "__main__":
    main()
