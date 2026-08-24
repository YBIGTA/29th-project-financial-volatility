"""비정형 데이터 수집: 이데일리 증권(categoryCode=16100) 뉴스.

robots.txt가 사실상 전면 허용(SEO 스크래퍼 몇 종만 차단, AI봇 명시 차단 없음)이라
이 크롤러도 직접 실행한다. "더보기" 버튼이 호출하는 내부 JSON API
(`/article/MoreList?categoryCode=16100&page=N&date=YYYYMMDD`)를 그대로 사용 —
날짜별로 그날의 증권 기사를 전량 가져온 뒤, 종목명이 제목/본문에 들어간 기사만 남긴다.
"""
import re
import time
from datetime import timedelta

import pandas as pd
import requests

from config import END_DATE, PROCESSED_DIR, REQUEST_DELAY_SEC, SIX_MONTH_RAW_DIR, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
API_URL = "https://www.edaily.co.kr/article/MoreList"
CATEGORY_STOCK = 16100
OUTPUT_DIR = SIX_MONTH_RAW_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_TAG_RE = re.compile(r"<[^>]+>")
EPOCH_MS_RE = re.compile(r"(\d+)")


def parse_confirm_date(value: str) -> pd.Timestamp:
    m = EPOCH_MS_RE.search(value or "")
    return pd.to_datetime(int(m.group(1)), unit="ms") if m else pd.NaT


def fetch_day(date_str: str, max_page: int = 30) -> list[dict]:
    items = []
    for page in range(1, max_page + 1):
        for attempt in range(1, 6):
            try:
                resp = requests.get(
                    API_URL,
                    params={"categoryCode": CATEGORY_STOCK, "page": page, "pagesize": 20, "date": date_str},
                    headers=HEADERS,
                    timeout=20,
                )
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                print(f"[edaily] {date_str} page={page} 요청 실패, 재시도 {attempt}/5")
                time.sleep(attempt * 5)
        data = resp.json()
        if not data:
            break
        items.extend(data)
        time.sleep(REQUEST_DELAY_SEC)
    return items


def crawl_all_days() -> pd.DataFrame:
    records = []
    day = START_DATE
    while day <= END_DATE:
        items = fetch_day(day.strftime("%Y%m%d"))
        for it in items:
            records.append(
                {
                    "news_id": it["NEWS_ID"],
                    "datetime": parse_confirm_date(it.get("ConfirmDate")),
                    "headline": it["HEADLINE"],
                    "body": HTML_TAG_RE.sub(" ", it.get("BODY_HTML") or ""),
                    "journalist": it.get("Journalist"),
                    "url": f"https://www.edaily.co.kr/News/Read?newsId={it['NEWS_ID']}&mediaCodeNo={it['MediaCodeNo']}",
                }
            )
        day += timedelta(days=1)

    if not records:
        return pd.DataFrame(columns=["news_id", "datetime", "headline", "body", "journalist", "url"])
    return pd.DataFrame(records).drop_duplicates(subset="news_id").sort_values("datetime")


def split_by_stock(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for code, name in STOCKS.items():
        mask = df["headline"].str.contains(name, na=False) | df["body"].str.contains(name, na=False)
        result[code] = df[mask].copy()
    return result


def main():
    all_df = crawl_all_days()
    all_df.to_csv(OUTPUT_DIR / "edaily_stock_all.csv", index=False, encoding="utf-8-sig")
    print(f"[edaily] 증권 섹션 전체 {len(all_df)}건 수집 ({START_DATE} ~ {END_DATE})")

    by_stock = split_by_stock(all_df)
    for code, name in STOCKS.items():
        df = by_stock[code]
        out_path = OUTPUT_DIR / f"edaily_news_{STOCK_FILE_NAMES[code]}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[edaily] {code} {name}: {len(df)}건 ({span})")


if __name__ == "__main__":
    main()
