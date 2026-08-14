"""비정형 데이터 수집: 토스증권 종목 커뮤니티(댓글).

robots.txt가 `Allow: /`라 직접 실행한다. 화면에는 안 보이지만 실제로는
`wts-cert-api.tossinvest.com`의 공개 REST API(`accessLevel: EXTERNAL_PUBLIC`,
로그인 불필요)로 댓글을 불러온다 — 브라우저에서 fetch를 가로채 확인함.

- subjectId는 6자리 종목코드가 아니라 **ISIN**(예: 삼성전자 KR7005930003)을 써야 함.
- 페이지당 11건 고정, `lastCommentId`로 다음 페이지 요청(무한 스크롤과 동일한 방식).
- 실주주 인증 여부(holding.shareHoldingStatus)도 같이 내려줘서 신뢰도 필터링 가능.
- commentId는 (모든 종목이 공유하는) 전역 순번이라, 시간과 거의 선형 비례한다.
  임의로 오래된 lastCommentId를 넣어도 그 근처 시점 댓글이 바로 반환됨 —
  즉 순차 페이지네이션 없이 원하는 시점으로 바로 "점프"가 가능하다.

인기 종목은 댓글 속도가 너무 빨라서(분당 여러 건) 3개월 전체를 순차 페이지네이션으로
모으려면 요청이 9만 건 이상 필요 — 대신 위 점프 기능을 이용해 `crawl_comments_sampled()`로
4종목 전부 표본 추출한다. 하루마다 장중/장외 각각 3~8곳을 무작위 시각으로 뽑아서(고정
시각이 아님) 점프하는 방식이라, 매일 다른 시간대가 샘플링되고 특정 시각에 쏠리지 않는다.
"""
import random
import time

import pandas as pd
import requests

from config import END_DATE, RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
API_URL = "https://wts-cert-api.tossinvest.com/api/v4/comments"

# 6자리 종목코드 -> ISIN (wts-cert-api의 subjectId 형식)
ISIN = {
    "005930": "KR7005930003",
    "000660": "KR7000660001",
    "247540": "KR7247540008",
    "035720": "KR7035720002",
}

# 댓글 속도가 너무 빨라 순차 페이지네이션으로 3개월을 못 채우는 종목 -> 시간대 점프 샘플링 사용
SAMPLED_CODES = ["005930", "000660", "247540", "035720"]
JUMP_MAX_ITERS = 6
JUMP_TOLERANCE_SEC = 3600  # 목표 시각과 1시간 이내로 맞으면 충분

# 하루 안에서 장중/장외 각각 몇 곳을 뽑을지 -> 매일 3~8곳 사이에서 무작위, 시각도 고정하지 않고 무작위
SAMPLES_PER_SESSION_RANGE = (3, 8)
INTRADAY_WINDOW = ("09:00", "15:30")
OVERNIGHT_WINDOW = ("15:30", "09:00")  # 다음날 09:00까지


def _random_times_in_window(day: pd.Timestamp, start_str: str, end_str: str, n: int, rng: random.Random) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{day.date()} {start_str}")
    end = pd.Timestamp(f"{day.date()} {end_str}")
    if end <= start:
        end += pd.Timedelta(days=1)
    span_minutes = (end - start).total_seconds() / 60
    return [start + pd.Timedelta(minutes=rng.uniform(0, span_minutes)) for _ in range(n)]


def _daily_sample_targets(start_date, end_date, seed: int | None = None) -> list[tuple[pd.Timestamp, str]]:
    rng = random.Random(seed)
    targets = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        n_intraday = rng.randint(*SAMPLES_PER_SESSION_RANGE)
        for t in _random_times_in_window(day, *INTRADAY_WINDOW, n_intraday, rng):
            targets.append((t, "intraday"))

        n_overnight = rng.randint(*SAMPLES_PER_SESSION_RANGE)
        for t in _random_times_in_window(day, *OVERNIGHT_WINDOW, n_overnight, rng):
            targets.append((t, "overnight"))
    return sorted(targets)


def fetch_page(isin: str, last_comment_id: int | None = None) -> dict:
    params = {"subjectType": "STOCK", "subjectId": isin, "commentSortType": "RECENT"}
    if last_comment_id is not None:
        params["lastCommentId"] = last_comment_id
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
    return resp.json()["result"]


def _page_date(results: list[dict]) -> pd.Timestamp:
    return pd.to_datetime(results[0]["createdAt"]).tz_localize(None)


def _estimate_rate_per_day(isin: str, anchor_id: int, anchor_date: pd.Timestamp, probe_offset: int = 3_000_000) -> float | None:
    """commentId 오프셋과 실제 날짜 차이로 '하루에 id가 몇 개 증가하는지' 추정한다."""
    probe_id = max(1, anchor_id - probe_offset)
    results = fetch_page(isin, probe_id)["results"]
    if not results:
        return None
    probe_date = _page_date(results)
    days = (anchor_date - probe_date).total_seconds() / 86400
    return probe_offset / days if days > 0 else None


def _jump_to_date(
    isin: str,
    target: pd.Timestamp,
    lo_id: int,
    lo_date: pd.Timestamp,
    hi_id: int,
    hi_date: pd.Timestamp,
) -> tuple[list[dict], int, pd.Timestamp] | None:
    """lo/hi 사이를 선형보간(secant)으로 좁혀가며 target 시각에 가장 가까운 페이지를 찾는다."""
    best = None
    for _ in range(JUMP_MAX_ITERS):
        if hi_date <= lo_date:
            break
        frac = min(max((target - lo_date).total_seconds() / (hi_date - lo_date).total_seconds(), 0.0), 1.0)
        guess_id = max(1, int(lo_id + frac * (hi_id - lo_id)))
        results = fetch_page(isin, guess_id)["results"]
        if not results:
            break
        guess_date = _page_date(results)
        best = (results, guess_id, guess_date)
        if abs((guess_date - target).total_seconds()) <= JUMP_TOLERANCE_SEC:
            break
        if guess_date < target:
            lo_id, lo_date = guess_id, guess_date
        else:
            hi_id, hi_date = guess_id, guess_date
        time.sleep(REQUEST_DELAY_SEC)
    return best


def crawl_comments(code: str, max_page: int = 500) -> pd.DataFrame:
    isin = ISIN[code]
    rows = []
    last_id = None
    for _ in range(max_page):
        page = fetch_page(isin, last_id)
        results = page["results"]
        if not results:
            break
        for c in results:
            rows.append(
                {
                    "comment_id": c["commentId"],
                    "code": code,
                    "datetime": pd.to_datetime(c["createdAt"]).tz_localize(None),
                    "author": c["author"]["nickname"],
                    "message": c["message"]["message"],
                    "is_holder": c.get("holding", {}).get("shareHoldingStatus") == "HOLDING",
                    "likes": c["statistic"]["likeCount"],
                    "reads": c["statistic"]["readCount"],
                    "replies": c["statistic"]["replyCount"],
                }
            )
        oldest = pd.to_datetime(results[-1]["createdAt"]).tz_localize(None)
        last_id = results[-1]["commentId"]
        if oldest.date() < START_DATE:
            break
        if not page["hasNext"]:
            break
        time.sleep(REQUEST_DELAY_SEC)

    cols = ["comment_id", "code", "datetime", "author", "message", "is_holder", "likes", "reads", "replies"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)[cols]
    df = df[df["datetime"].dt.date >= START_DATE]
    return df.drop_duplicates(subset="comment_id").sort_values("datetime")


def _rows_from_results(results: list[dict], code: str) -> list[dict]:
    return [
        {
            "comment_id": c["commentId"],
            "code": code,
            "datetime": pd.to_datetime(c["createdAt"]).tz_localize(None),
            "author": c["author"]["nickname"],
            "message": c["message"]["message"],
            "is_holder": c.get("holding", {}).get("shareHoldingStatus") == "HOLDING",
            "likes": c["statistic"]["likeCount"],
            "reads": c["statistic"]["readCount"],
            "replies": c["statistic"]["replyCount"],
        }
        for c in results
    ]


def crawl_comments_sampled(code: str) -> pd.DataFrame:
    """댓글이 너무 많아 순차 수집이 안 되는 종목용 — 하루 안 장중/장외 시각대로 점프해서 표본만 모은다."""
    isin = ISIN[code]
    recent = fetch_page(isin)["results"]
    if not recent:
        return pd.DataFrame()
    anchor_id, anchor_date = recent[0]["commentId"], _page_date(recent)

    rate = _estimate_rate_per_day(isin, anchor_id, anchor_date)
    if not rate:
        return pd.DataFrame()

    targets = _daily_sample_targets(START_DATE, END_DATE)

    rows = []
    seen_ids = set()
    # 최근 -> 과거 순으로 훑으면서, 직전에 찾은 지점을 다음 탐색의 상단(hi) 브래킷으로 재사용
    hi_id, hi_date = anchor_id, anchor_date
    for target, _session in reversed(targets):
        if target > anchor_date:
            continue
        guess_offset = max(1_000, int((hi_date - target).total_seconds() / 86400 * rate * 1.5))
        lo_id = max(1, hi_id - guess_offset)
        lo_results = fetch_page(isin, lo_id)["results"]
        if not lo_results:
            continue
        lo_date = _page_date(lo_results)
        time.sleep(REQUEST_DELAY_SEC)

        found = _jump_to_date(isin, target, lo_id, lo_date, hi_id, hi_date)
        if found:
            results, found_id, found_date = found
            for r in _rows_from_results(results, code):
                if r["comment_id"] not in seen_ids:
                    seen_ids.add(r["comment_id"])
                    rows.append(r)
            hi_id, hi_date = found_id, found_date  # 다음(더 과거) 탐색의 상단 브래킷으로 재사용
        time.sleep(REQUEST_DELAY_SEC)

    cols = ["comment_id", "code", "datetime", "author", "message", "is_holder", "likes", "reads", "replies"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[cols]
    df = df[df["datetime"].dt.date >= START_DATE]
    return df.drop_duplicates(subset="comment_id").sort_values("datetime")


def crawl_and_save(code: str, sampled: bool = False) -> pd.DataFrame:
    new_df = crawl_comments_sampled(code) if sampled else crawl_comments(code)
    new_df["comment_id"] = new_df["comment_id"].astype(str)
    out_path = RAW_DIR / f"toss_community_{STOCK_FILE_NAMES[code]}.csv"

    if out_path.exists():
        old_df = pd.read_csv(out_path, encoding="utf-8-sig", dtype={"comment_id": str})
        old_df["datetime"] = pd.to_datetime(old_df["datetime"])
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="comment_id").sort_values("datetime")
    else:
        combined = new_df.sort_values("datetime")

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    return combined


def main():
    for code, name in STOCKS.items():
        df = crawl_and_save(code, sampled=code in SAMPLED_CODES)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        mode = "표본" if code in SAMPLED_CODES else "순차"
        print(f"[toss:{mode}] {code} {name}: {len(df)}건 누적 ({span})")


if __name__ == "__main__":
    main()
