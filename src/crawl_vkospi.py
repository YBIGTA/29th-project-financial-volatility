"""공공데이터포털에서 VKOSPI 일봉 OHLC만 수집한다.

환경변수 ``DATA_GO_KR_SERVICE_KEY``에 공공데이터포털 일반 인증키를 설정한 뒤 실행한다.
결과는 ``data/raw/price_daily_vkospi.csv``에 저장한다.
"""

import argparse
import os
from datetime import date, timedelta

import pandas as pd
import requests

from config import SIX_MONTH_RAW_DIR


API_URL = (
    "https://apis.data.go.kr/1160100/service/GetMarketIndexInfoService/"
    "getDerivationProductMarketIndex"
)
DEFAULT_END_DATE = date.today()
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=180)
OUTPUT_PATH = SIX_MONTH_RAW_DIR / "price_daily_vkospi.csv"


def fetch_vkospi(service_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(
        API_URL,
        params={
            "serviceKey": service_key,
            "resultType": "json",
            "pageNo": 1,
            "numOfRows": 10000,
            "beginBasDt": start_date.replace("-", ""),
            "endBasDt": end_date.replace("-", ""),
        },
        timeout=30,
    )
    response.raise_for_status()

    try:
        payload = response.json()["response"]
        header = payload["header"]
        if header.get("resultCode") != "00":
            raise RuntimeError(header.get("resultMsg", "공공데이터 API 오류"))
        items = payload["body"].get("items", {}).get("item", [])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"예상하지 못한 API 응답입니다: {response.text[:300]}") from error

    if isinstance(items, dict):
        items = [items]

    candidates = pd.DataFrame(items)
    if candidates.empty:
        raise RuntimeError("조회 기간에 파생상품지수 데이터가 없습니다.")

    name_column = "idxNm"
    if name_column not in candidates.columns:
        raise RuntimeError(f"API 응답에 idxNm 컬럼이 없습니다: {candidates.columns.tolist()}")

    names = candidates[name_column].astype(str)
    selected = candidates[
        names.str.contains("VKOSPI", case=False, na=False)
        | (names.str.contains("코스피", na=False) & names.str.contains("변동성", na=False))
    ].copy()
    if selected.empty:
        available = ", ".join(sorted(names.dropna().unique())[:20])
        raise RuntimeError(f"응답에서 VKOSPI를 찾지 못했습니다. 조회된 지수: {available}")

    required = {"basDt": "date", "mkp": "open", "hipr": "high", "lopr": "low", "clpr": "close"}
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise RuntimeError(f"VKOSPI 응답에 필요한 컬럼이 없습니다: {missing}")

    result = selected[list(required)].rename(columns=required)
    result["date"] = pd.to_datetime(result["date"], format="%Y%m%d")
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column].astype(str).str.replace(",", ""), errors="raise")

    return result.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="VKOSPI 일봉 OHLC 전용 수집기")
    parser.add_argument("--start", default=DEFAULT_START_DATE.isoformat(), help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END_DATE.isoformat(), help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()

    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY 환경변수에 공공데이터포털 일반 인증키를 설정하세요.")

    vkospi = fetch_vkospi(service_key, args.start, args.end)
    vkospi.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print(f"[VKOSPI] {len(vkospi)}행 저장: {OUTPUT_PATH}")
    print(f"기간: {vkospi['date'].min().date()} ~ {vkospi['date'].max().date()}")
    print("컬럼: date, open, high, low, close")


if __name__ == "__main__":
    main()
