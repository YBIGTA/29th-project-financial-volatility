"""비정형 데이터 수집: 토스증권 종목 커뮤니티(댓글).

robots.txt가 `Allow: /`라 직접 실행한다. 화면에는 안 보이지만 실제로는
`wts-cert-api.tossinvest.com`의 공개 REST API(`accessLevel: EXTERNAL_PUBLIC`,
로그인 불필요)로 댓글을 불러온다 — 브라우저에서 fetch를 가로채 확인함.

- subjectId는 6자리 종목코드가 아니라 **ISIN**(예: 삼성전자 KR7005930003)을 써야 함.
- 페이지당 11건 고정, `lastCommentId`로 다음 페이지 요청(무한 스크롤과 동일한 방식).
- 실주주 인증 여부(holding.shareHoldingStatus)도 같이 내려줘서 신뢰도 필터링 가능.
- 인기 종목(삼성전자·SK하이닉스)은 댓글 속도가 매우 빨라(분당 여러 건) 3개월 전체를
  한 번에 모으려면 요청 수가 매우 많이 필요 — 네이버와 동일하게 날짜 컷오프 +
  안전장치용 최대 페이지 수로 제한하고, 매일 누적 실행을 전제로 설계했다.
"""
import time

import pandas as pd
import requests

from config import RAW_DIR, REQUEST_DELAY_SEC, START_DATE, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
API_URL = "https://wts-cert-api.tossinvest.com/api/v4/comments"

# 6자리 종목코드 -> ISIN (wts-cert-api의 subjectId 형식)
ISIN = {
    "005930": "KR7005930003",
    "000660": "KR7000660001",
    "247540": "KR7247540008",
    "035720": "KR7035720002",
}


def fetch_page(isin: str, last_comment_id: int | None = None) -> dict:
    params = {"subjectType": "STOCK", "subjectId": isin, "commentSortType": "RECENT"}
    if last_comment_id is not None:
        params["lastCommentId"] = last_comment_id
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
    return resp.json()["result"]


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


def crawl_and_save(code: str) -> pd.DataFrame:
    new_df = crawl_comments(code)
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
        df = crawl_and_save(code)
        span = f"{df['datetime'].min()} ~ {df['datetime'].max()}" if len(df) else "no data"
        print(f"[toss] {code} {name}: {len(df)}건 누적 ({span})")


if __name__ == "__main__":
    main()
