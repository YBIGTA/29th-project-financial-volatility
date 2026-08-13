"""비정형 데이터 수집: 네이버페이 증권 주주토론방 게시글 목록.

목록 페이지(제목/작성자/날짜/조회수/공감/비공감)는 정적 HTML이라 requests+BeautifulSoup으로
바로 수집 가능. 단, 게시글 본문 상세페이지는 JS로 렌더링되는 SPA라 별도 렌더링 도구
(Selenium 등) 없이는 본문 전체 텍스트 수집이 어려움 — 1차 목표는 제목 기반 수집.

뉴스와 동일하게 페이지당 20건, 최대 100페이지(2000건) 제약이 있고, 게시글이 많은
종목일수록(삼성전자 등) 실제 소급 기간이 짧음. 매일 실행해 nid 기준으로 누적할 것.
"""
import re
import time
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import NAVER_MAX_PAGE, RAW_DIR, REQUEST_DELAY_SEC, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
LIST_URL = "https://finance.naver.com/item/board.naver"


def fetch_board_page(code: str, page: int) -> list[dict]:
    resp = requests.get(LIST_URL, params={"code": code, "page": page}, headers=HEADERS, timeout=10)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.select_one("table.type2")
    if table is None:
        return []  # 최대 열람 페이지를 넘어가면 "잘못된 접근입니다" 안내 페이지가 옴

    rows = []
    for tr in table.select("tr"):
        tds = tr.find_all("td")
        if len(tds) != 6:
            continue

        date_text = tds[0].get_text(strip=True)
        link = tds[1].find("a")
        if link is None:
            continue
        title = link.get("title") or link.get_text(strip=True)
        href = link.get("href", "")
        nid = parse_qs(urlparse(href).query).get("nid", [None])[0]
        author = tds[2].get_text(strip=True)
        views = tds[3].get_text(strip=True)
        likes = tds[4].get_text(strip=True)
        dislikes = tds[5].get_text(strip=True)

        if not nid or not re.match(r"\d{4}\.\d{2}\.\d{2}", date_text):
            continue

        rows.append(
            {
                "nid": nid,
                "code": code,
                "datetime": date_text,
                "title": title,
                "author": author,
                "views": views,
                "likes": likes,
                "dislikes": dislikes,
                "url": f"https://finance.naver.com{href}",
            }
        )
    return rows


def crawl_board(code: str, max_page: int = NAVER_MAX_PAGE) -> pd.DataFrame:
    all_rows = []
    for page in range(1, max_page + 1):
        rows = fetch_board_page(code, page)
        if not rows:
            break
        all_rows.extend(rows)
        time.sleep(REQUEST_DELAY_SEC)

    cols = ["nid", "code", "datetime", "title", "author", "views", "likes", "dislikes", "url"]
    if not all_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    return df[cols]


def crawl_and_save(code: str) -> pd.DataFrame:
    new_df = crawl_board(code)
    out_path = RAW_DIR / f"board_{code}.csv"

    new_df["nid"] = new_df["nid"].astype(str)

    if out_path.exists():
        old_df = pd.read_csv(out_path, encoding="utf-8-sig", dtype={"nid": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"])
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="nid").sort_values("datetime")
    else:
        combined = new_df.sort_values("datetime")

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    return combined


def main():
    for code, name in STOCKS.items():
        df = crawl_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[board] {code} {name}: {len(df)}건 누적 ({span})")


if __name__ == "__main__":
    main()
