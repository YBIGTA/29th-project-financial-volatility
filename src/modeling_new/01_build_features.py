"""
1단계: 종목별 변동성 예측용 피처 테이블 생성 (v8)

[v7 -> v8 변경사항] (교수/선배 피드백 반영)
- **(버그 수정, 최우선) 감성 시간 정렬 오류 수정**
  원래 연구 질문: "장외 감성 -> 당일(t) 장중 변동성", "장중 감성 -> 익일(t+1) 장중 변동성".
  그런데 기존 코드는 장외/장중 감성을 shift 없이 그대로 써서 사실상 "장외 감성도
  익일(t+1)을 예측"하는 구조가 되어 있었음 (당일 예측이어야 하는데 하루 밀려 있었음).
  -> target_vol[t] = vol[t+1] 구조는 유지하되(예측 시점 = "t+1일 장 시작 직전"),
     장외 감성만 sentiment_overnight = overnight_sentiment_index.shift(-1)로 수정.
     이러면 "t+1일 개장 직전에 확정되는 장외 감성"이 되어 원래 연구질문
     ("장외 감성(t+1일 개장 전) -> t+1일 장중 변동성")과 정확히 일치함.
     장중 감성(sentiment_intraday)은 원래도 맞았으므로 shift 없이 그대로 유지.
  주의: 이 수정으로 sentiment_overnight도 마지막 행에서 NaN이 발생 -> dropna 대상에 추가.

- **Target A/B 병렬 비교 구조 도입** (target을 교체하지 않고 둘 다 산출)
  Target A = target_vol_5d  (rolling_volatility_5d 기준, 팀 기존 합의안, close-to-close)
  Target B = target_vol_pk  (Parkinson volatility 기준, 당일 High-Low 기반 "장중" 변동성)
  두 target 모두 동일한 피처셋(realized_vol 계열)으로 02/03단계에서 나란히 모델링해서
  비교함 - 어느 게 더 좋다고 미리 정하지 않고, 연구질문과의 정합성 + 결과를 같이 보고
  최종 결정 (피드백 13,14번 권장안).

- **경로를 상대경로로 변경** (VS Code 등 로컬 환경에서 바로 실행 가능하도록)
  DATA_DIR = ./data, OUTPUT_DIR = ./outputs (스크립트 위치 기준)

[v5 -> v7 변경사항 요약]
- 데이터 3개월 -> 6개월치로 교체, holdout/MIN_TRAIN 비례 확대
- vkospi_close(레벨, 비정상) -> vkospi_change(변화율, 정상성 확인)로 교체

[예측 시점 정의] (전체 파이프라인 공통, 02/03번과 동일)
"t+1일 장 시작 직전" 시점에서 target_vol[t](=t+1일의 실현/장중 변동성)를 예측.
- sentiment_intraday[t] = t일 09:00~15:29 감성 (t일 마감 시점에 확정, 장 시작 전 이용 가능)
- sentiment_overnight[t] = overnight_sentiment_index[t+1] (t+1일 개장 직전에 확정되는
  15:30(t일)~08:59(t+1일) 감성) -> t+1일 장 시작 직전에 이용 가능, 미래정보 누수 없음
"""
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MARKET_CSV = DATA_DIR / "market_index_merged.csv"

# 종목명 -> 파일명 매핑 (4종목 전부, 6개월치)
STOCKS = {
    "samsung_electronics": DATA_DIR / "merged_daily_samsung_electronics.csv",
    "sk_hynix": DATA_DIR / "merged_daily_sk_hynix.csv",
    "kakao": DATA_DIR / "merged_daily_kakao.csv",
    "ecopro_bm": DATA_DIR / "merged_daily_ecopro_bm.csv",
}


def load_market_data(market_csv) -> pd.DataFrame:
    mkt = pd.read_csv(market_csv)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.strftime("%Y-%m-%d")
    mkt["kospi_ret"] = np.log(mkt["kospi_close"] / mkt["kospi_close"].shift(1))
    mkt["kosdaq_ret"] = np.log(mkt["kosdaq_close"] / mkt["kosdaq_close"].shift(1))
    # vkospi_close(레벨)는 ADF 검정 결과 비정상 -> vkospi_change(변화율, 정상성 확인)로 대체
    mkt["vkospi_change"] = np.log(mkt["vkospi_close"] / mkt["vkospi_close"].shift(1))
    return mkt[["date", "kospi_ret", "kosdaq_ret", "vkospi_change"]]


def build_features(stock_csv, market: pd.DataFrame) -> pd.DataFrame:
    stock = pd.read_csv(stock_csv)
    stock["date"] = pd.to_datetime(stock["date"]).dt.strftime("%Y-%m-%d")

    # 가격 데이터 없는 행 제거
    stock = stock.dropna(subset=["close"]).copy()

    df = stock.merge(market, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ---- 종목 자체 수익률 / 변동성 ----
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(5).std()

    # Parkinson 변동성 (당일 고가/저가 기반, Target B용)
    # sigma_P = |ln(High/Low)| / sqrt(4*ln2)
    df["parkinson_vol"] = np.abs(np.log(df["high"] / df["low"])) / np.sqrt(4 * np.log(2))

    # lag 피처 (Target A/B 공통, realized_vol 계열만 - 공정 비교를 위해 두 target 모두
    # 동일한 피처셋을 사용함. parkinson_vol 자체는 피처에 넣지 않음 - Target B로 쓸 때
    # 자기 자신의 정보를 피처로 흘려보내는 걸 방지)
    df["realized_vol_lag1"] = df["realized_vol"].shift(1)
    df["realized_vol_lag2"] = df["realized_vol"].shift(2)
    df["realized_vol_lag3"] = df["realized_vol"].shift(3)
    df["log_ret_lag1"] = df["log_ret"].shift(1)
    df["vkospi_change_lag1"] = df["vkospi_change"].shift(1)

    # ---- 감성지수 (v8: 장외 감성 시간정렬 버그 수정) ----
    df["sentiment_intraday"] = df["intraday_sentiment_index"]                 # 그대로 (원래도 맞았음)
    df["sentiment_overnight"] = df["overnight_sentiment_index"].shift(-1)     # t+1일 개장 전 감성으로 정렬

    # ---- Target A: rolling_volatility_5d 기준 (팀 기존 합의) ----
    df["target_vol_5d"] = df["realized_vol"].shift(-1)
    # ---- Target B: Parkinson volatility 기준 (연구질문의 "장중 변동성"과 직접 대응) ----
    df["target_vol_pk"] = df["parkinson_vol"].shift(-1)

    # 결측치 있는 행 제거 (rolling/lag/shift 전부 포함, sentiment_overnight의
    # shift(-1)로 생기는 마지막 행 NaN도 포함)
    df_clean = df.dropna(subset=[
        "realized_vol_lag3", "vkospi_change_lag1", "kospi_ret", "kosdaq_ret",
        "sentiment_overnight", "target_vol_5d", "target_vol_pk",
    ]).reset_index(drop=True)

    return df_clean


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    market = load_market_data(MARKET_CSV)

    for stock_name, stock_csv in STOCKS.items():
        features = build_features(stock_csv, market)
        out_path = OUTPUT_DIR / f"features_{stock_name}.csv"
        features.to_csv(out_path, index=False)
        print(f"[{stock_name}] 사용 가능 행 수: {len(features)}, "
              f"기간: {features['date'].min()} ~ {features['date'].max()} -> {out_path}")
