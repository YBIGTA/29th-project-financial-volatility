"""
2단계: GARCH 계열 - walk-forward 방식, 3단계 비교 (v5)

[v4 -> v5 변경사항] (팀 리뷰 반영, 8/17)
- 명칭 확정: "GARCH-X" 대신 "GARCH 예측값 + 감성지수 2단계 회귀"로 통일
  (arch 패키지가 진짜 GARCH-X(분산방정식에 외생변수 추가)를 지원 안 해서
   대안으로 회귀를 쓴다는 점을 명칭에서부터 명확히 함)
- 비교 단계를 2개 -> 3개로 확장. 기존엔 [GARCH(1,1) 자체가 이미 OLS 회귀를 거친 값]
  vs [+감성지수까지 넣은 회귀값]만 비교해서, 후자가 더 잘 나와도 그게 "감성지수 덕분"인지
  "OLS 회귀 보정 자체의 효과"인지 구분이 안 되는 문제가 있었음
  -> 회귀를 전혀 안 거친 "GARCH 원본 예측값"을 0단계로 추가해서 3단계로 비교:

    [1] GARCH 원본        : garch_vol 그 자체를 예측값으로 사용 (회귀 보정 없음)
    [2] GARCH+회귀(감성없음) : target_vol ~ garch_vol 로 OLS 회귀한 예측값
    [3] GARCH+회귀(감성포함) : target_vol ~ garch_vol + sentiment_* 로 OLS 회귀한 예측값

  [1]->[2] 차이 = "회귀 보정 자체의 효과", [2]->[3] 차이 = "감성지수 고유의 기여도"로
  분리해서 해석 가능해짐

[예측 시점 정의] "t+1일 장 시작 직전" (01번 docstring 참고)
"""
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model

VOL_DIR = "/home/claude/vol_project"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]

HOLDOUT_DAYS = 8
MIN_TRAIN = 30


def garch_walkforward_vol(returns_pct: pd.Series, min_train: int) -> pd.Series:
    """Expanding-window walk-forward one-step-ahead GARCH(1,1) 변동성 예측.
    인덱스 i의 예측값은 반드시 인덱스 0~i-1 데이터로만 학습 (데이터 누수 없음)."""
    forecasts = {}
    n = len(returns_pct)
    for i in range(min_train, n):
        train = returns_pct.iloc[:i]
        res = arch_model(train, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        var_fc = fc.variance.values[-1, 0]
        forecasts[i] = np.sqrt(var_fc) / 100
    return pd.Series(forecasts)


def evaluate(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual.values - pred.values
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    return {"RMSE": rmse, "MAE": mae, "n": len(actual)}


def run_pipeline(features_csv: str):
    df = pd.read_csv(features_csv)
    returns_pct = df["log_ret"] * 100

    n_total = len(df)
    n_trainval = n_total - HOLDOUT_DAYS

    garch_vol_all = garch_walkforward_vol(returns_pct, MIN_TRAIN)
    df = df.assign(garch_vol=np.nan)
    df.loc[garch_vol_all.index, "garch_vol"] = garch_vol_all.values

    tv = df.iloc[MIN_TRAIN:n_trainval].dropna(subset=["garch_vol"]).copy()
    ho = df.iloc[n_trainval:].copy()

    # ---------------------------------------------------------
    # [1] GARCH 원본 - 회귀 보정 없이 garch_vol을 그대로 예측값으로 사용
    # ---------------------------------------------------------
    metrics_ho_raw = evaluate(ho["target_vol"], ho["garch_vol"])

    # ---------------------------------------------------------
    # [2] GARCH+회귀(감성없음) - target_vol ~ garch_vol
    # ---------------------------------------------------------
    X_base = sm.add_constant(tv[["garch_vol"]])
    ols_base = sm.OLS(tv["target_vol"], X_base).fit()

    X_ho_base = sm.add_constant(ho[["garch_vol"]], has_constant="add")
    pred_ho_base = ols_base.predict(X_ho_base)
    metrics_ho_base = evaluate(ho["target_vol"], pred_ho_base)

    # ---------------------------------------------------------
    # [3] GARCH+회귀(감성포함) - target_vol ~ garch_vol + sentiment_*
    # ---------------------------------------------------------
    X_x = sm.add_constant(tv[["garch_vol", "sentiment_intraday", "sentiment_overnight"]])
    ols_x = sm.OLS(tv["target_vol"], X_x).fit()

    X_ho_x = sm.add_constant(ho[["garch_vol", "sentiment_intraday", "sentiment_overnight"]],
                              has_constant="add")
    pred_ho_x = ols_x.predict(X_ho_x)
    metrics_ho_x = evaluate(ho["target_vol"], pred_ho_x)

    return {
        "ols_base": ols_base, "ols_x": ols_x,
        "holdout_metrics_raw": metrics_ho_raw,
        "holdout_metrics_base": metrics_ho_base,
        "holdout_metrics_x": metrics_ho_x,
    }


if __name__ == "__main__":
    for stock_name in STOCKS:
        features_csv = f"{VOL_DIR}/features_{stock_name}.csv"
        result = run_pipeline(features_csv)

        print("=" * 60)
        print(f"[{stock_name}] 감성지수 계수 유의성 (p-value < 0.05면 유의)")
        print("=" * 60)
        print(result["ols_x"].pvalues[["sentiment_intraday", "sentiment_overnight"]])
        print(f"홀드아웃 [1] GARCH 원본          : {result['holdout_metrics_raw']}")
        print(f"홀드아웃 [2] GARCH+회귀(감성없음) : {result['holdout_metrics_base']}")
        print(f"홀드아웃 [3] GARCH+회귀(감성포함) : {result['holdout_metrics_x']}")
        print()

        with open(f"{VOL_DIR}/holdout_metrics_{stock_name}_garch.json", "w") as f:
            json.dump({
                "GARCH 원본": result["holdout_metrics_raw"],
                "GARCH+회귀(감성없음)": result["holdout_metrics_base"],
                "GARCH+회귀(감성포함)": result["holdout_metrics_x"],
            }, f, indent=2)

    print("전체 종목 GARCH 계열 실행 완료.")
