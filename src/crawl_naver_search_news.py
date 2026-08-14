# 네이버 뉴스 통합검색(search.naver.com) 날짜 슬라이싱 크롤러
#
# *** 이 스크립트는 robots.txt가 전면 차단(User-agent: * Disallow: /)인 사이트를 대상으로
#     합니다. 다른 크롤러(paxnet/edaily/toss)와 다르게 이 파일은 세션 안에서 자동으로
#     실행하지 않고, 코드만 준비해뒀습니다 — 직접 실행 여부는 판단해서 결정하세요. ***
#
# 한 번의 검색 쿼리는 최대 2,000건까지만 나옴("검색결과는 2,000건까지 제공합니다"
# 문구로 실측 확인). 대신 ds(시작일)·de(종료일) 파라미터로 날짜 범위를 지정할 수 있어서,
# 하루 단위로 쪼개서 여러 번 검색하면 하루치가 2,000건을 넘지 않는 한 상한 문제를 피할 수 있음.
#
# 주의: 검색 결과 화면은 상대시각("N시간 전")만 보여주고 절대시각은 안 줘서, datetime은
# 정확한 시각이 아니라 "그 날짜의 정오"로만 채움 — 시간 단위 정밀도가 필요하면
# 기사 링크를 열어서 개별로 다시 가져와야 함. headline1/body1 클래스명도 네이버가 프론트엔드를
# 바꾸면 깨질 수 있으니, 실행 결과가 0건만 계속 나오면 구조가 바뀐 건지 먼저 확인할 것.
import re
import time
from datetime import timedelta

import pandas as pd
import requests

from config import END_DATE, RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
SEARCH_URL = "https://search.naver.com/search.naver"
MAX_PAGES_PER_DAY = 20  # 페이지당 10건 -> 하루 최대 200건까지, 2,000건 상한에 여유 있게

HEADLINE_RE = re.compile(
    r'href="([^"]+)"[^>]*data-heatmap-target="\.tit"[^>]*>.*?'
    r'sds-comps-text-type-headline1">(.*?)</span>',
    re.S,
)
BODY_RE = re.compile(r'sds-comps-text-type-body1">(.*?)</span>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
NO_MORE_RE = "표시할 검색결과가 없습니다"


def _strip(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def fetch_search_page(query: str, ds: str, de: str, start: int) -> list[dict]:
    resp = requests.get(
        SEARCH_URL,
        params={"where": "news", "query": query, "ds": ds, "de": de, "start": start},
        headers=HEADERS,
        timeout=10,
    )
    text = resp.text
    if NO_MORE_RE in text:
        return []

    headlines = HEADLINE_RE.findall(text)
    bodies = BODY_RE.findall(text)
    items = []
    for i, (url, title) in enumerate(headlines):
        body = bodies[i] if i < len(bodies) else ""
        items.append({"url": url, "title": _strip(title), "body": _strip(body)})
    return items


def crawl_day(query: str, day: pd.Timestamp) -> list[dict]:
    ds = de = day.strftime("%Y.%m.%d")
    all_items = []
    for page in range(MAX_PAGES_PER_DAY):
        start = page * 10 + 1
        items = fetch_search_page(query, ds, de, start)
        if not items:
            break
        all_items.extend(items)
        time.sleep(REQUEST_DELAY_SEC)
    for item in all_items:
        item["datetime"] = day + pd.Timedelta(hours=12)  # 절대시각 없음 -> 그 날짜 정오로 채움
    return all_items


def crawl_stock(code: str) -> pd.DataFrame:
    query = STOCKS[code]
    rows = []
    day = START_DATE
    while day <= END_DATE:
        rows.extend(crawl_day(query, pd.Timestamp(day)))
        day += timedelta(days=1)

    cols = ["url", "title", "body", "datetime"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols].drop_duplicates(subset="url")
    df["code"] = code
    return df.sort_values("datetime")


def main():
    for code, name in STOCKS.items():
        df = crawl_stock(code)
        out_path = RAW_DIR / f"news_search_{STOCK_FILE_NAMES[code]}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[naver_search] {code} {name}: {len(df)}건 ({span})")


if __name__ == "__main__":
    main()
