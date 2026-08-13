"""정형 데이터 수집: 종목 4개 + KOSPI/KOSDAQ/VIX 일봉(시가/고가/저가/종가/거래량).

FinanceDataReader로 최근 3개월치를 안정적으로 수집. (분봉은 불필요 판단으로 제외.)
"""
import time

import FinanceDataReader as fdr
import pandas as pd

from config import END_DATE, INDEXES, RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCKS


def fetch_daily_ohlcv(code: str) -> pd.DataFrame:
    df = fdr.DataReader(code, START_DATE, END_DATE)
    df.index.name = "date"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["date", "open", "high", "low", "close", "volume"]]


def main():
    summary = []

    for code, name in {**STOCKS, **INDEXES}.items():
        df = fetch_daily_ohlcv(code)
        out_path = RAW_DIR / f"price_daily_{code}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary.append((code, name, len(df), df["date"].min(), df["date"].max()))
        time.sleep(REQUEST_DELAY_SEC)

    print(f"{'code':<8}{'name':<10}{'rows':<8}{'from':<22}{'to'}")
    for code, name, n, start, end in summary:
        print(f"{code:<8}{name:<10}{n:<8}{str(start):<22}{str(end)}")


if __name__ == "__main__":
    main()
