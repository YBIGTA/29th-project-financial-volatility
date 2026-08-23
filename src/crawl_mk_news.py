# 매일경제(mk.co.kr) 종목명 검색 크롤러
#
# *** 이 스크립트는 robots.txt가 ClaudeBot을 이름으로 명시해서 막아둔 사이트를 대상으로
#     합니다(GPTBot, anthropic-ai 등도 같이 차단, User-agent: * 는 Allow: /). 그래서 세션
#     안에서 자동으로 실행하지 않고, 코드만 준비해뒀습니다 — 직접 실행하세요. ***
#
# 이데일리는 진보 성향으로 분류돼서, 반대 성향(우파/중도우파) 경제지를 하나 더 넣어
# 뉴스 소스의 정치적 편향을 완화하려고 추가했습니다. 후보로 확인해본 한국경제(Cloudflare
# 봇 차단), 조선비즈(기사 목록이 JS로 나중에 그려지는 방식이라 일반 요청으론 안 보임)는
# 접근 자체가 안 돼서 제외했고, 매일경제만 남았습니다.
#
# mk.co.kr 검색창에 종목명을 넣고 "더보기"를 눌렀을 때 브라우저가 실제로 부르는 내부 API를
# 사용자가 개발자도구 Network 탭에서 직접 확인해서 알려준 것 -- robots.txt 때문에 이 사이트는
# 제가 직접 열어서 구조를 확인할 수 없어서, 화면에 보이는 텍스트를 그대로 옮겨 받아 구조를
# 추정해 만들었다. 그래서 팍스넷/이데일리보다 실제 HTML 구조 추정의 신뢰도가 낮다 --
# 실행해서 0건만 계속 나오면 BlockedOrChangedPageError 메시지부터 확인할 것.
#
#   GET https://www.mk.co.kr/_CP/243?word={검색어}&page={N}&highlight=Y&page_size=null&id=null
#
# 응답은 JSON이 아니라 HTML 조각이고, 기사 하나당 카테고리/제목/본문요약/날짜 링크 4개가
# 전부 같은 기사 URL(`https://www.mk.co.kr/news/{카테고리}/{숫자ID}`)을 가리키며 순서대로
# 반복되는 구조 (예: [증권](url) [제목](url) [본문요약...](url) [2026-08-23 17:40:12](url)).
# 정렬은 기본이 최신순이라, 팍스넷처럼 페이지를 넘기다 설정 기간보다 오래된 기사가 나오면
# 멈추는 방식으로 기간 전체를 확보한다.
import re
import time

import pandas as pd
import requests

from config import RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
SEARCH_API = "https://www.mk.co.kr/_CP/243"

NETWORK_RETRIES = 5
NETWORK_RETRY_WAIT_SEC = 30
MAX_PAGE = 200  # 안전장치 -- 이 이상 페이지를 넘겨야 한다면 뭔가 잘못된 것

BLOCK_RE = re.compile(
    r'<a[^>]*href="(?P<url>https://www\.mk\.co\.kr/news/[a-z]+/\d+)"[^>]*>\s*(?P<category>[^<]{1,20}?)\s*</a>'
    r'.*?<a[^>]*href="(?P=url)"[^>]*>\s*(?P<headline>[^<]+?)\s*</a>'
    r'.*?<a[^>]*href="(?P=url)"[^>]*>\s*(?P<body>[^<]+?)\s*</a>'
    r'.*?<a[^>]*href="(?P=url)"[^>]*>\s*(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*</a>',
    re.S,
)


class BlockedOrChangedPageError(RuntimeError):
    """페이지 구조가 바뀌었거나 접근이 막혔을 때 -- 조용히 0건으로 넘어가지 않고 여기서 멈춘다."""


def _get_with_retries(params: dict) -> requests.Response:
    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            resp = requests.get(SEARCH_API, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == NETWORK_RETRIES:
                raise
            print(f"[mk] 네트워크 오류({e.__class__.__name__}), {NETWORK_RETRY_WAIT_SEC}초 후 재시도 ({attempt}/{NETWORK_RETRIES})")
            time.sleep(NETWORK_RETRY_WAIT_SEC)


def fetch_search_page(query: str, page: int) -> list[dict]:
    resp = _get_with_retries({"word": query, "page": page, "highlight": "Y", "page_size": "null", "id": "null"})
    matches = BLOCK_RE.finditer(resp.text)
    rows = [
        {
            "news_id": m.group("url"),
            "url": m.group("url"),
            "category": m.group("category"),
            "headline": m.group("headline"),
            "body": m.group("body"),
            "datetime": pd.to_datetime(m.group("datetime")),
        }
        for m in matches
    ]
    if not rows and page == 1:
        raise BlockedOrChangedPageError(
            "1페이지인데 기사를 하나도 못 찾았습니다 -- 매일경제가 검색 페이지 구조를 바꿨거나 접근이 막혔을 수 있습니다. "
            "브라우저에서 같은 URL을 직접 열어서 화면이 정상인지 먼저 확인하세요."
        )
    return rows


def crawl_query(query: str) -> pd.DataFrame:
    rows: list[dict] = []
    for page in range(1, MAX_PAGE + 1):
        page_rows = fetch_search_page(query, page)
        if not page_rows:
            break
        rows.extend(page_rows)
        oldest = min(r["datetime"] for r in page_rows)
        if oldest.date() < START_DATE:
            break
        time.sleep(REQUEST_DELAY_SEC)

    cols = ["news_id", "url", "category", "headline", "body", "datetime"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols].drop_duplicates(subset="news_id")
    return df[df["datetime"].dt.date >= START_DATE].sort_values("datetime")


def _load_existing(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["news_id", "url", "category", "headline", "body", "datetime"])
    old = pd.read_csv(path, encoding="utf-8-sig")
    old["datetime"] = pd.to_datetime(old["datetime"], format="mixed")
    return old


def main():
    for code, name in STOCKS.items():
        out_path = RAW_DIR / f"mk_news_{STOCK_FILE_NAMES[code]}.csv"
        fresh = crawl_query(name)
        existing = _load_existing(out_path)
        frames = [df for df in (existing, fresh) if not df.empty]
        merged = pd.concat(frames, ignore_index=True) if frames else existing
        merged = merged.drop_duplicates(subset="news_id").sort_values("datetime")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        span = f"{merged['datetime'].min()} ~ {merged['datetime'].max()}" if len(merged) else "no data"
        print(f"[mk] {code} {name}: {len(merged)}건 ({span})")


if __name__ == "__main__":
    main()
