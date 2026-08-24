"""정형 데이터 수집: 종목 4개 + KOSPI/KOSDAQ/VIX 일봉(시가/고가/저가/종가/거래량).

FinanceDataReader로 최근 3개월치를 안정적으로 수집. (분봉은 불필요 판단으로 제외.)
"""
import argparse
import ast
import time
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

from config import (
    END_DATE,
    INDEXES,
    PRICE_FILE_NAMES,
    RAW_DIR,
    REQUEST_DELAY_SEC,
    START_DATE,
    STOCKS,
)


NAVER_INDEX_SYMBOLS = {"KS11": "KOSPI", "KQ11": "KOSDAQ"}


def fetch_naver_index(code: str) -> pd.DataFrame:
    response = requests.get(
        "https://api.finance.naver.com/siseJson.naver",
        params={
            "symbol": NAVER_INDEX_SYMBOLS[code],
            "requestType": 1,
            "startTime": START_DATE.strftime("%Y%m%d"),
            "endTime": END_DATE.strftime("%Y%m%d"),
            "timeframe": "day",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    rows = ast.literal_eval(response.text.strip())
    df = pd.DataFrame(rows[1:], columns=["date", "open", "high", "low", "close", "volume", "foreign_net"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_daily_ohlcv(code: str) -> pd.DataFrame:
    if code in NAVER_INDEX_SYMBOLS:
        return fetch_naver_index(code)

    df = pd.DataFrame()
    for attempt in range(1, 6):
        df = fdr.DataReader(code, START_DATE, END_DATE)
        if not df.empty:
            break
        if attempt < 5:
            print(f"[price] {code} 빈 응답, 재시도 {attempt}/5")
            time.sleep(attempt * 5)
    if df.empty:
        raise RuntimeError(f"{code}: {START_DATE} ~ {END_DATE} 데이터가 없습니다.")
    df.index.name = "date"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df["date"].dt.date >= START_DATE) & (df["date"].dt.date <= END_DATE)]
    return df[["date", "open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser()
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--indexes-only", action="store_true")
    target_group.add_argument("--stocks-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    if args.indexes_only:
        targets = INDEXES
    elif args.stocks_only:
        targets = STOCKS
    else:
        targets = {**STOCKS, **INDEXES}
    for code, name in targets.items():
        df = fetch_daily_ohlcv(code)
        out_path = args.output_dir / f"price_daily_{PRICE_FILE_NAMES[code]}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary.append((code, name, len(df), df["date"].min(), df["date"].max()))
        time.sleep(REQUEST_DELAY_SEC)

    print(f"{'code':<8}{'name':<10}{'rows':<8}{'from':<22}{'to'}")
    for code, name, n, start, end in summary:
        print(f"{code:<8}{name:<10}{n:<8}{str(start):<22}{str(end)}")


if __name__ == "__main__":
    main()
