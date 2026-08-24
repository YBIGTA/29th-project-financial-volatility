"""비정형 데이터 수집: 네이버페이 증권 주주토론방 게시글 목록.

목록 페이지(제목/작성자/날짜/조회수/공감/비공감)는 정적 HTML이라 requests+BeautifulSoup으로
바로 수집 가능. 단, 게시글 본문 상세페이지는 JS로 렌더링되는 SPA라 별도 렌더링 도구
(Selenium 등) 없이는 본문 전체 텍스트 수집이 어려움 — 1차 목표는 제목 기반 수집.

페이지당 약 20건이다. 100페이지에서 끝나는 것이 아니라 페이지 하단의 실제 다음 링크를
따라가면 101페이지 이후도 열리므로, 백필 모드에서는 세션과 Referer를 유지하면서 링크를
순차적으로 따라가 3개월 시작일까지 수집한다.
"""
import re
import time
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    NAVER_MAX_PAGE,
    RAW_DIR,
    REQUEST_DELAY_SEC,
    START_DATE,
    STOCK_ENGLISH_NAMES,
    STOCK_FILE_NAMES,
    STOCKS,
    USER_AGENT,
)

HEADERS = {"User-Agent": USER_AGENT}
LIST_URL = "https://finance.naver.com/item/board.naver"
BOARD_COLUMNS = ["nid", "code", "datetime", "title", "author", "views", "likes", "dislikes", "url"]
BACKFILL_CHECKPOINT_PAGES = 25
BACKFILL_COOLDOWN_MINUTES = 30
BACKFILL_RETRIES = 6


class BoardPageError(RuntimeError):
    """게시판 표가 없는 차단·오류 페이지를 받았을 때 발생한다."""


def _parse_board_page(html: str, code: str, current_url: str) -> tuple[list[dict], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.type2")
    if table is None:
        raise BoardPageError(f"게시판 표를 찾지 못했습니다: {current_url}")

    current_page = int(parse_qs(urlparse(current_url).query).get("page", ["1"])[0])
    next_url = None
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "board.naver" not in href:
            continue
        linked_page = parse_qs(urlparse(href).query).get("page", [None])[0]
        if linked_page and linked_page.isdigit() and int(linked_page) == current_page + 1:
            next_url = urljoin(current_url, href)
            break

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
                "code": STOCK_ENGLISH_NAMES[code],
                "datetime": date_text,
                "title": title,
                "author": author,
                "views": views,
                "likes": likes,
                "dislikes": dislikes,
                "url": f"https://finance.naver.com{href}",
            }
        )
    return rows, next_url


def fetch_board_page(code: str, page: int) -> list[dict]:
    resp = requests.get(LIST_URL, params={"code": code, "page": page}, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    current_url = f"{LIST_URL}?code={code}&page={page}"
    try:
        rows, _ = _parse_board_page(resp.text, code, current_url)
        return rows
    except BoardPageError:
        return []


def fetch_board_link_page(
    session: requests.Session, code: str, url: str, referer: str | None
) -> tuple[list[dict], str | None]:
    headers = {"Referer": referer} if referer else None
    resp = session.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return _parse_board_page(resp.text, code, url)


def crawl_board(code: str, known_ids: set[str] | None = None, max_page: int = NAVER_MAX_PAGE) -> pd.DataFrame:
    """최신 페이지부터 읽고, 한 페이지 전체가 기존 nid이면 수집을 종료한다."""
    known_ids = known_ids or set()
    all_rows = []
    for page in range(1, max_page + 1):
        rows = fetch_board_page(code, page)
        if not rows:
            break
        new_rows = [row for row in rows if str(row["nid"]) not in known_ids]
        all_rows.extend(new_rows)
        if known_ids and not new_rows:
            print(f"[board] {code}: 기존 데이터 도달(page={page}), 증분 수집 종료")
            break
        time.sleep(REQUEST_DELAY_SEC)
    else:
        print(f"[board] {code}: {max_page}페이지 상한 도달 — 실행 사이에 2,000건 이상 등록됐을 수 있습니다.")

    if not all_rows:
        return pd.DataFrame(columns=BOARD_COLUMNS)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    return df[BOARD_COLUMNS]


def _merge_and_save(out_path, existing: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if rows:
        new_df = pd.DataFrame(rows, columns=BOARD_COLUMNS)
        new_df["nid"] = new_df["nid"].astype(str)
        new_df["datetime"] = pd.to_datetime(new_df["datetime"], format="%Y.%m.%d %H:%M")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing
    combined = combined.drop_duplicates(subset="nid").sort_values("datetime")
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    return combined


def backfill_and_save(code: str) -> pd.DataFrame:
    """실제 다음 페이지 링크를 따라가며 START_DATE까지 수집하고 중간 저장한다."""
    out_path = RAW_DIR / f"board_{STOCK_FILE_NAMES[code]}.csv"
    if out_path.exists():
        combined = pd.read_csv(out_path, encoding="utf-8-sig", dtype={"nid": str})
        combined["datetime"] = pd.to_datetime(combined["datetime"])
    else:
        combined = pd.DataFrame(columns=BOARD_COLUMNS)
    known_ids = set(combined["nid"].astype(str))

    session = requests.Session()
    session.headers.update(HEADERS)
    current_url = f"{LIST_URL}?code={code}&page=1"
    referer = None
    page_count = 0
    buffer: list[dict] = []

    while current_url:
        for attempt in range(1, BACKFILL_RETRIES + 1):
            try:
                rows, next_url = fetch_board_link_page(session, code, current_url, referer)
                break
            except (requests.exceptions.RequestException, BoardPageError) as error:
                if attempt == BACKFILL_RETRIES:
                    combined = _merge_and_save(out_path, combined, buffer)
                    raise RuntimeError(f"{current_url}에서 {BACKFILL_RETRIES}회 연속 실패") from error
                print(
                    f"[board:backfill] {code}: 접근 실패({error.__class__.__name__}), "
                    f"{BACKFILL_COOLDOWN_MINUTES}분 후 같은 페이지 재시도 "
                    f"({attempt}/{BACKFILL_RETRIES})"
                )
                time.sleep(BACKFILL_COOLDOWN_MINUTES * 60)

        page_count += 1
        reached_start = False
        for row in rows:
            row_time = pd.to_datetime(row["datetime"], format="%Y.%m.%d %H:%M")
            if row_time.date() < START_DATE:
                reached_start = True
                continue
            if str(row["nid"]) not in known_ids:
                known_ids.add(str(row["nid"]))
                buffer.append(row)

        if page_count % BACKFILL_CHECKPOINT_PAGES == 0 or reached_start or not next_url:
            before = len(combined)
            combined = _merge_and_save(out_path, combined, buffer)
            print(
                f"[board:backfill] {code}: page={page_count}, "
                f"신규 {len(combined) - before}건 저장, 누적 {len(combined)}건"
            )
            buffer.clear()

        if reached_start:
            print(f"[board:backfill] {code}: 시작일 {START_DATE} 이전에 도달해 완료")
            break
        if not next_url:
            print(
                f"[board:backfill] {code}: page={page_count}에서 다음 링크가 없어 종료 "
                f"(시작일 {START_DATE} 도달 여부를 확인하세요)"
            )
        referer, current_url = current_url, next_url
        time.sleep(REQUEST_DELAY_SEC)

    return combined


def crawl_and_save(code: str) -> pd.DataFrame:
    out_path = RAW_DIR / f"board_{STOCK_FILE_NAMES[code]}.csv"

    if out_path.exists():
        old_df = pd.read_csv(out_path, encoding="utf-8-sig", dtype={"nid": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"])
        known_ids = set(old_df["nid"])
    else:
        old_df = pd.DataFrame()
        known_ids = set()

    new_df = crawl_board(code, known_ids=known_ids)
    new_df["nid"] = new_df["nid"].astype(str)

    if len(old_df):
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="nid").sort_values("datetime")
    else:
        combined = new_df.sort_values("datetime")

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[board] {code}: 신규 {len(new_df)}건 추가")
    return combined


def main():
    for code, name in STOCKS.items():
        df = crawl_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[board] {code} {name}: {len(df)}건 누적 ({span})")


def backfill_main():
    for code, name in STOCKS.items():
        df = backfill_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[board:backfill] {code} {name}: {len(df)}건 누적 ({span})")


if __name__ == "__main__":
    main()
