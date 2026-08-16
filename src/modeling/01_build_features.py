"""
1단계: 종목별 변동성 예측용 피처 테이블 생성 (v3 - 최종 데이터셋 반영)

[v2 -> v3 변경사항]
- 종목 데이터가 4개(삼성전자/하이닉스/카카오/에코프로비엠) 최종본으로 교체됨
  -> 4종목 전부 한 번에 처리하도록 루프 구조로 변경
- 시장 데이터(코스피/코스닥/VKOSPI)가 market_index_merged.csv 하나로 통합됨
  -> 기존의 개별 파일 병합 + VKOSPI EUC-KR 인코딩 처리 로직 제거 (더 이상 불필요)
- 종목 CSV에 intraday_item_count/no_posts, overnight_item_count/no_posts, vix_* 컬럼 추가됨
  -> no_posts는 4종목 전부 항상 0(변동 없음)이라 피처로서 의미가 없어 미사용
  -> item_count(게시글 수)는 이제 깔끔하게 존재하지만, 아직 팀 합의 전이라 미사용
     (화제성 피처로 추가할지는 별도 논의 필요)
  -> vix_*는 현재 설계(VKOSPI 기반)에서 사용 안 함, 그대로 무시

[예측 시점 정의] (전체 파이프라인 공통, 02/03번과 동일)
"t+1일 장 시작 직전" 시점에서 target_vol[t](=t+1일의 실현변동성)를 예측한다고 정의.
이 시점엔 t일 마감까지의 모든 정보(realized_vol[t], log_ret[t], vkospi_close[t] 등)
+ t일 마감~t+1일 개장의 overnight 감성까지 전부 확정되어 있음.
-> intraday_sentiment_index[t], overnight_sentiment_index[t] 모두 shift 없이
   t행 그대로 사용 가능.
"""
import pandas as pd
import numpy as np

DATA_DIR = "/mnt/user-data/uploads"
OUTPUT_DIR = "/home/claude/vol_project"
MARKET_CSV = f"{DATA_DIR}/market_index_merged.csv"

# 종목명 -> 파일명 매핑 (4종목 전부)
STOCKS = {
    "samsung_electronics": f"{DATA_DIR}/samsung_electronics_merged_daily.csv",
    "sk_hynix": f"{DATA_DIR}/sk_hynix_merged_daily.csv",
    "kakao": f"{DATA_DIR}/kakao_merged_daily.csv",
    "ecopro_bm": f"{DATA_DIR}/ecopro_bm_merged_daily.csv",
}


def load_market_data(market_csv: str) -> pd.DataFrame:
    mkt = pd.read_csv(market_csv)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.strftime("%Y-%m-%d")
    mkt["kospi_ret"] = np.log(mkt["kospi_close"] / mkt["kospi_close"].shift(1))
    mkt["kosdaq_ret"] = np.log(mkt["kosdaq_close"] / mkt["kosdaq_close"].shift(1))
    return mkt[["date", "kospi_ret", "kosdaq_ret", "vkospi_close"]]


def build_features(stock_csv: str, market: pd.DataFrame) -> pd.DataFrame:
    stock = pd.read_csv(stock_csv)
    stock["date"] = pd.to_datetime(stock["date"]).dt.strftime("%Y-%m-%d")

    # 가격 데이터 없는 행 제거 (감성지수만 있고 OHLCV가 아직 안 붙은 경우 대비)
    stock = stock.dropna(subset=["close"]).copy()

    # 병합
    df = stock.merge(market, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # 종목 자체 수익률 / 실현변동성
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(5).std()

    # lag 피처
    df["realized_vol_lag1"] = df["realized_vol"].shift(1)
    df["realized_vol_lag2"] = df["realized_vol"].shift(2)
    df["realized_vol_lag3"] = df["realized_vol"].shift(3)
    df["log_ret_lag1"] = df["log_ret"].shift(1)
    df["vkospi_lag1"] = df["vkospi_close"].shift(1)

    # 감성지수 (shift 없이 그대로 - 위 docstring 예측 시점 정의 참고)
    df["sentiment_intraday"] = df["intraday_sentiment_index"]
    df["sentiment_overnight"] = df["overnight_sentiment_index"]

    # 타겟: 다음날 실현변동성
    df["target_vol"] = df["realized_vol"].shift(-1)

    # 결측치 있는 행 제거 (rolling/lag/shift로 생긴 NaN + 시장데이터 휴장일 등으로
    # 인한 중간 NaN 전부 포함 - 첫/마지막 행만 있다고 가정하지 않음)
    df_clean = df.dropna(subset=[
        "realized_vol_lag3", "vkospi_lag1", "kospi_ret", "kosdaq_ret", "target_vol"
    ]).reset_index(drop=True)

    return df_clean


if __name__ == "__main__":
    market = load_market_data(MARKET_CSV)

    for stock_name, stock_csv in STOCKS.items():
        features = build_features(stock_csv, market)
        out_path = f"{OUTPUT_DIR}/features_{stock_name}.csv"
        features.to_csv(out_path, index=False)
        print(f"[{stock_name}] 사용 가능 행 수: {len(features)}, "
              f"기간: {features['date'].min()} ~ {features['date'].max()} -> {out_path}")
