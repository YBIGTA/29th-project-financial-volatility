"""비정형 데이터 수집: 네이버 증권 종목 뉴스.

네이버 모바일증권의 뉴스 JSON API를 사용. 페이지당 20건, 최대 100페이지(2000건)까지만
열람 가능한 플랫폼 제약이 있어 종목별로 실제 소급 기간이 다름(뉴스가 많은 종목일수록 짧음).
매일 실행하면 새 기사가 누적되어 시간이 지날수록 실제 커버 기간이 길어짐(nid 기준 중복 제거).
"""
import time

import pandas as pd
import requests

from config import (
    NAVER_MAX_PAGE,
    RAW_DIR,
    REQUEST_DELAY_SEC,
    STOCK_ENGLISH_NAMES,
    STOCK_FILE_NAMES,
    STOCKS,
    USER_AGENT,
)

HEADERS = {"User-Agent": USER_AGENT}
API_URL = "https://m.stock.naver.com/api/news/stock/{code}"


def fetch_news_page(code: str, page: int, page_size: int = 20) -> list[dict]:
    resp = requests.get(
        API_URL.format(code=code),
        params={"pageSize": page_size, "page": page},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    groups = resp.json()
    items = []
    for g in groups:
        items.extend(g.get("items", []))
    return items


def crawl_news(code: str, known_ids: set[str] | None = None, max_page: int = NAVER_MAX_PAGE) -> pd.DataFrame:
    """최신 페이지부터 읽고, 한 페이지 전체가 기존 ID이면 수집을 종료한다."""
    known_ids = known_ids or set()
    all_items = []
    for page in range(1, max_page + 1):
        items = fetch_news_page(code, page)
        if not items:
            break
        new_items = [item for item in items if str(item["id"]) not in known_ids]
        all_items.extend(new_items)
        if known_ids and not new_items:
            print(f"[news] {code}: 기존 데이터 도달(page={page}), 증분 수집 종료")
            break
        time.sleep(REQUEST_DELAY_SEC)
    else:
        print(f"[news] {code}: {max_page}페이지 상한 도달 — 실행 사이에 2,000건 이상 등록됐을 수 있습니다.")

    if not all_items:
        return pd.DataFrame(
            columns=["id", "code", "datetime", "office_name", "title", "body_snippet", "url"]
        )

    df = pd.DataFrame(all_items)
    df["code"] = STOCK_ENGLISH_NAMES[code]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M")
    df = df.rename(columns={"officeName": "office_name", "body": "body_snippet", "mobileNewsUrl": "url"})
    return df[["id", "code", "datetime", "office_name", "title", "body_snippet", "url"]]


def crawl_and_save(code: str) -> pd.DataFrame:
    out_path = RAW_DIR / f"news_{STOCK_FILE_NAMES[code]}.csv"

    if out_path.exists():
        old_df = pd.read_csv(out_path, encoding="utf-8-sig", dtype={"id": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"])
        known_ids = set(old_df["id"])
    else:
        old_df = pd.DataFrame()
        known_ids = set()

    new_df = crawl_news(code, known_ids=known_ids)
    new_df["id"] = new_df["id"].astype(str)

    if len(old_df):
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="id").sort_values("datetime")
    else:
        combined = new_df.sort_values("datetime")

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[news] {code}: 신규 {len(new_df)}건 추가")
    return combined


def main():
    for code, name in STOCKS.items():
        df = crawl_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[news] {code} {name}: {len(df)}건 누적 ({span})")


if __name__ == "__main__":
    main()
