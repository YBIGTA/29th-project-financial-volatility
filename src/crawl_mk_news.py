# 매일경제(mk.co.kr) 증권 섹션 RSS 크롤러
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
# RSS는 최신 기사 몇십 건 정도만 나옵니다(과거로 깊이 못 감) — 이데일리처럼 날짜별로
# 전체 기간을 훑는 게 아니라 네이버 뉴스 API와 비슷한 "최근 스냅샷" 소스입니다. 6개월
# 전체가 필요하면 이 스크립트를 주기적으로 실행해서 누적시키거나, 매일경제의 날짜 범위
# 검색 URL을 찾아서 확장해야 합니다.
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from config import RAW_DIR, STOCK_FILE_NAMES, STOCKS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}
RSS_URL = "https://www.mk.co.kr/rss/30000001/"  # 증권 섹션

NETWORK_RETRIES = 5
NETWORK_RETRY_WAIT_SEC = 30


class FeedChangedError(RuntimeError):
    """RSS 구조가 바뀌어서 파싱이 안 될 때 — 조용히 0건으로 넘어가지 않고 여기서 멈춘다."""


def _get_with_retries(url: str) -> requests.Response:
    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == NETWORK_RETRIES:
                raise
            print(f"[mk] 네트워크 오류({e.__class__.__name__}), {NETWORK_RETRY_WAIT_SEC}초 후 재시도 ({attempt}/{NETWORK_RETRIES})")
            time.sleep(NETWORK_RETRY_WAIT_SEC)


def fetch_rss_items() -> list[dict]:
    resp = _get_with_retries(RSS_URL)
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise FeedChangedError(f"RSS를 XML로 파싱할 수 없습니다 — 매일경제가 피드 형식을 바꿨을 수 있습니다: {e}")

    items = root.findall(".//item")
    if not items:
        raise FeedChangedError("RSS에 <item>이 하나도 없습니다 — 피드 구조가 바뀌었는지 먼저 확인하세요.")

    rows = []
    for it in items:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        description = (it.findtext("description") or "").strip()
        pub_date = (it.findtext("pubDate") or "").strip()
        guid = (it.findtext("guid") or link).strip()
        rows.append(
            {
                "news_id": guid,
                "datetime": pd.to_datetime(pub_date, errors="coerce", utc=True).tz_localize(None) if pub_date else pd.NaT,
                "headline": title,
                "body": description,
                "url": link,
            }
        )
    return rows


def split_by_stock(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for code, name in STOCKS.items():
        mask = df["headline"].str.contains(name, na=False) | df["body"].str.contains(name, na=False)
        result[code] = df[mask].copy()
    return result


def _load_existing(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["news_id", "datetime", "headline", "body", "url"])
    old = pd.read_csv(path, encoding="utf-8-sig")
    old["datetime"] = pd.to_datetime(old["datetime"], format="mixed")
    return old


def main():
    rows = fetch_rss_items()
    all_df = pd.DataFrame(rows).drop_duplicates(subset="news_id").sort_values("datetime")
    print(f"[mk] 증권 섹션 RSS {len(all_df)}건 수집 ({all_df['datetime'].min()} ~ {all_df['datetime'].max()})")

    by_stock = split_by_stock(all_df)
    for code, name in STOCKS.items():
        out_path = RAW_DIR / f"mk_news_{STOCK_FILE_NAMES[code]}.csv"
        # RSS는 매번 최신 스냅샷만 주기 때문에, 이어붙여서 누적한다(중복은 news_id로 제거).
        merged = pd.concat([_load_existing(out_path), by_stock[code]], ignore_index=True)
        merged = merged.drop_duplicates(subset="news_id").sort_values("datetime")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        span = f"{merged['datetime'].min()} ~ {merged['datetime'].max()}" if len(merged) else "no data"
        print(f"[mk] {code} {name}: 누적 {len(merged)}건 ({span})")


if __name__ == "__main__":
    main()
