# 네이버 뉴스 통합검색(search.naver.com) 날짜 슬라이싱 크롤러
#
# *** 이 스크립트는 robots.txt가 전면 차단(User-agent: * Disallow: /)인 사이트를 대상으로
#     합니다. 다른 크롤러(paxnet/edaily/toss)와 다르게 이 파일은 세션 안에서 자동으로
#     실행하지 않고, 코드만 준비해뒀습니다 — 직접 실행 여부는 판단해서 결정하세요. ***
#
# 한 번의 검색 쿼리는 최대 2,000건까지만 나옴("검색결과는 2,000건까지 제공합니다"
# 문구로 실측 확인). 대신 ds(시작일)·de(종료일) 파라미터로 날짜 범위를 지정할 수 있어서,
# 하루 단위로 쪼개서 여러 번 검색하면 하루치가 2,000건을 넘지 않는 한 상한 문제를 피할 수 있음.
# 다만 예전 종목뉴스 API도 2,000건 상한을 다 못 채우고 6일치 정도에서 끝났던 걸 보면,
# 하루에 다 긁으려 욕심낼 필요가 없어서 하루 3페이지(30건)만 표본으로 가져온다 — 그래야
# 요청 총량이 줄어서 IP 차단 위험도 낮아지고, 90일 전체를 고르게 훑을 수 있음.
#
# 주의: 검색 결과 화면은 상대시각("N시간 전")만 보여주고 절대시각은 안 줘서, datetime은
# 정확한 시각이 아니라 "그 날짜의 정오"로만 채움 — 시간 단위 정밀도가 필요하면
# 기사 링크를 열어서 개별로 다시 가져와야 함. headline1/body1 클래스명도 네이버가 프론트엔드를
# 바꾸면 깨질 수 있으니, 실행 결과가 0건만 계속 나오면 구조가 바뀐 건지 먼저 확인할 것.
import random
import re
import time
from datetime import timedelta

import pandas as pd
import requests

from config import END_DATE, RAW_DIR, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
SEARCH_URL = "https://search.naver.com/search.naver"

# 예전 종목뉴스 API도 상한 2,000건을 다 못 채우고 실제로는 6일치 정도만 나왔던 걸 감안하면,
# 하루에 욕심내서 다 긁을 필요 없음 -> 하루 3페이지(30건)만 표본으로 가져오는 대신
# 90일 전체를 다 훑어서, 총 요청 수를 최대 7,200번 -> 최대 1,080번으로 줄인다.
MAX_PAGES_PER_DAY = 3

# 일반 검색은 트래픽에 훨씬 민감해서(막히면 이후 요청이 전부 조용히 0건으로 새버림)
# 다른 크롤러보다 요청 간격을 넉넉히 두고, 매번 똑같은 간격이면 그 자체도 패턴이라 흔들어준다.
SEARCH_DELAY_RANGE = (1.5, 3.5)


def _search_delay() -> float:
    return random.uniform(*SEARCH_DELAY_RANGE)

HEADLINE_RE = re.compile(
    r'href="([^"]+)"[^>]*data-heatmap-target="\.tit"[^>]*>.*?'
    r'sds-comps-text-type-headline1">(.*?)</span>',
    re.S,
)
BODY_RE = re.compile(r'sds-comps-text-type-body1">(.*?)</span>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
NO_MORE_RE = "표시할 검색결과가 없습니다"


class BlockedOrChangedPageError(RuntimeError):
    """네이버가 접근을 막았거나(비정상 트래픽 감지) 페이지 구조가 바뀐 것으로 의심될 때."""


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
    if not headlines:
        # "검색결과 없음" 문구도 없는데 기사도 안 잡히면 십중팔구 차단이거나 구조 변경.
        # 조용히 0건 처리하면 이후 전체가 빈 데이터로 새버리니 바로 멈춘다.
        raise BlockedOrChangedPageError(
            f"query={query!r} ds={ds} start={start}에서 기사를 하나도 못 찾았습니다. "
            "네이버가 일시적으로 접근을 막았거나(잠시 후 재시도) 페이지 구조가 바뀐 것 같습니다."
        )

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
        time.sleep(_search_delay())
    for item in all_items:
        item["datetime"] = day + pd.Timedelta(hours=12)  # 절대시각 없음 -> 그 날짜 정오로 채움
    return all_items


def _to_df(rows: list[dict], code: str) -> pd.DataFrame:
    cols = ["url", "title", "body", "datetime"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols].drop_duplicates(subset="url")
    df["code"] = code
    return df.sort_values("datetime")


def crawl_stock(code: str) -> tuple[pd.DataFrame, BlockedOrChangedPageError | None]:
    """중간에 막히면 그때까지 모은 것만이라도 반환한다(전부 버리지 않음)."""
    query = STOCKS[code]
    rows = []
    day = START_DATE
    while day <= END_DATE:
        try:
            rows.extend(crawl_day(query, pd.Timestamp(day)))
        except BlockedOrChangedPageError as e:
            return _to_df(rows, code), e
        day += timedelta(days=1)
    return _to_df(rows, code), None


def main():
    for code, name in STOCKS.items():
        new_df, error = crawl_stock(code)
        out_path = RAW_DIR / f"news_search_{STOCK_FILE_NAMES[code]}.csv"

        if out_path.exists():
            # 이전에 막혀서 중간까지만 저장된 파일이 있으면 이번 결과와 합쳐서(url 기준 중복제거)
            # 재실행할 때마다 처음부터 다시 긁지 않고 이어서 쌓이게 한다.
            old_df = pd.read_csv(out_path, encoding="utf-8-sig")
            df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset="url").sort_values("datetime")
        else:
            df = new_df

        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[naver_search] {code} {name}: {len(df)}건 누적 ({span})")
        if error:
            print(f"[naver_search] 여기서 중단합니다: {error}")
            print("네이버가 일시적으로 막았을 가능성이 높습니다. 시간을 두고(예: 30분~1시간) 다시 실행해보세요.")
            return


if __name__ == "__main__":
    main()
