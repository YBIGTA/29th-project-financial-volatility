"""정형 데이터 수집: 종목 4개 + KOSPI/KOSDAQ/VIX 일봉, 그리고 종목별 최근 분봉(보너스).

- 일봉(시가/고가/저가/종가/거래량): FinanceDataReader로 최근 3개월치를 안정적으로 수집.
- 분봉: 네이버의 비공식 차트 API를 쓰는데, 종가·거래량만 나오고(시/고/저 없음)
  최근 영업일 며칠치만 조회 가능 — 3개월 연속 분봉은 무료로는 불가능해서 "가능한 만큼"만 모음.
  자세한 배경은 README의 "데이터 수집 범위의 현실적 한계" 참고.
"""
import re
import time

import FinanceDataReader as fdr
import pandas as pd
import requests

from config import END_DATE, INDEXES, RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}


def fetch_daily_ohlcv(code: str) -> pd.DataFrame:
    df = fdr.DataReader(code, START_DATE, END_DATE)
    df.index.name = "date"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_minute_bonus(code: str, count: int = 3000) -> pd.DataFrame:
    """네이버 fchart 분봉 API — 종가·거래량만, 최근 며칠치만 반환됨(플랫폼 제약)."""
    url = "https://fchart.stock.naver.com/sise.nhn"
    params = {"symbol": code, "timeframe": "minute", "count": count, "requestType": 0}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    text = resp.content.decode("euc-kr", errors="ignore")
    rows = re.findall(r'<item data="([^"]+)"', text)

    records = []
    for row in rows:
        parts = row.split("|")
        if len(parts) != 6:
            continue
        ts, o, h, l, close, volume = parts
        if close in ("null", ""):
            continue
        records.append(
            {
                "datetime": pd.to_datetime(ts, format="%Y%m%d%H%M"),
                "close": int(close),
                "volume": int(volume),
            }
        )
    return pd.DataFrame(records)


def main():
    summary = []

    for code, name in {**STOCKS, **INDEXES}.items():
        df = fetch_daily_ohlcv(code)
        out_path = RAW_DIR / f"price_daily_{code}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary.append((code, name, "daily", len(df), df["date"].min(), df["date"].max()))
        time.sleep(REQUEST_DELAY_SEC)

    for code, name in STOCKS.items():
        df = fetch_minute_bonus(code)
        out_path = RAW_DIR / f"price_minute_{code}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        if len(df):
            summary.append(
                (code, name, "minute(bonus)", len(df), df["datetime"].min(), df["datetime"].max())
            )
        else:
            summary.append((code, name, "minute(bonus)", 0, None, None))
        time.sleep(REQUEST_DELAY_SEC)

    print(f"{'code':<8}{'name':<10}{'type':<16}{'rows':<8}{'from':<22}{'to'}")
    for code, name, kind, n, start, end in summary:
        print(f"{code:<8}{name:<10}{kind:<16}{n:<8}{str(start):<22}{str(end)}")


if __name__ == "__main__":
    main()
