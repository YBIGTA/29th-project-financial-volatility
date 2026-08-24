"""비정형 데이터 수집: 팍스넷(Paxnet) 종목토론방.

robots.txt가 `Allow: /`라 일반 크롤링에 제약이 없고(Claude류 AI봇 차단 명시 없음),
네이버와 달리 페이지 상한이 없어 3개월 전까지도 그대로 소급 조회가 된다.
목록 페이지가 서버에서 완전히 렌더링돼 있어 초 단위 타임스탬프까지 그대로 가져온다.
"""
import re
import time

import pandas as pd
import requests

from config import PROCESSED_DIR, REQUEST_DELAY_SEC, SIX_MONTH_RAW_DIR, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
LIST_URL = "https://www.paxnet.co.kr/tbbs/list"
OUTPUT_DIR = SIX_MONTH_RAW_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAXNET_CODES = ("005930", "000660")

ROW_RE = re.compile(
    r'data-seq="(?P<seq>\d+)".*?'
    r'href="javascript:bbsWrtView\(\d+\);">(?P<title>.*?)</a>.*?'
    r"viewProfile\('(?P<author>[^']*)'\).*?"
    r'id="hitsNum_\d+"><span>조회 </span>(?P<views>\d+).*?'
    r'id="recmNum_\d+"><span>추천 </span>(?P<likes>\d+).*?'
    r'data-date-format="(?P<date>[A-Za-z].*?)"',
    re.S,
)


def fetch_board_page(code: str, page: int) -> list[dict]:
    for attempt in range(1, 6):
        try:
            resp = requests.get(
                LIST_URL,
                params={"tbbsType": "L", "id": code, "page": page},
                headers=HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            print(f"[paxnet] {code} page={page} 요청 실패, 재시도 {attempt}/5")
            time.sleep(attempt * 5)
    rows = []
    for m in ROW_RE.finditer(resp.text):
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        rows.append(
            {
                "seq": m.group("seq"),
                "code": code,
                "datetime": pd.to_datetime(m.group("date"), format="%a %b %d %H:%M:%S KST %Y"),
                "title": title,
                "author": m.group("author"),
                "views": int(m.group("views")),
                "likes": int(m.group("likes")),
                "url": f"https://www.paxnet.co.kr/tbbs/view?tbbsType=L&id={code}&viewSeq={m.group('seq')}",
            }
        )
    return rows


def crawl_board(code: str, max_page: int = 500) -> pd.DataFrame:
    all_rows = []
    for page in range(1, max_page + 1):
        rows = fetch_board_page(code, page)
        if not rows:
            break
        all_rows.extend(rows)
        if page % 50 == 0:
            print(f"[paxnet] {code}: {page}페이지, {len(all_rows)}건 수집 중")
        if rows[-1]["datetime"].date() < START_DATE:
            break
        time.sleep(REQUEST_DELAY_SEC)

    cols = ["seq", "code", "datetime", "title", "author", "views", "likes", "url"]
    if not all_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows)[cols]
    df = df[df["datetime"].dt.date >= START_DATE]
    return df.drop_duplicates(subset="seq").sort_values("datetime")


def crawl_and_save(code: str) -> pd.DataFrame:
    df = crawl_board(code)
    out_path = OUTPUT_DIR / f"paxnet_board_{STOCK_FILE_NAMES[code]}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def main():
    for code in PAXNET_CODES:
        name = STOCKS[code]
        df = crawl_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[paxnet board] {code} {name}: {len(df)}건 ({span})")


if __name__ == "__main__":
    main()
