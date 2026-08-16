"""
2단계: GARCH(1,1) / GARCH-X 계열 - walk-forward 방식 (v3: 4종목 전체 루프)

[v2 -> v3 변경사항]
- STOCK_NAME 하나만 처리하던 구조 -> STOCKS 리스트 전체를 순회하도록 변경
- 로직 자체(walk-forward, 예측 시점 정의, GARCH-X 구현 방식)는 v2와 동일

[예측 시점 정의] "t+1일 장 시작 직전" (01번 docstring 참고)
[GARCH-X 구현 방식] arch 패키지 한계로 "GARCH forecast + sentiment 회귀"
  (encompassing regression) 방식 사용. 팀 합의 필요 (README 참고)
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

    X_base = sm.add_constant(tv[["garch_vol"]])
    ols_base = sm.OLS(tv["target_vol"], X_base).fit()

    X_x = sm.add_constant(tv[["garch_vol", "sentiment_intraday", "sentiment_overnight"]])
    ols_x = sm.OLS(tv["target_vol"], X_x).fit()

    ho = df.iloc[n_trainval:].copy()

    X_ho_base = sm.add_constant(ho[["garch_vol"]], has_constant="add")
    pred_ho_base = ols_base.predict(X_ho_base)

    X_ho_x = sm.add_constant(ho[["garch_vol", "sentiment_intraday", "sentiment_overnight"]],
                              has_constant="add")
    pred_ho_x = ols_x.predict(X_ho_x)

    metrics_ho_base = evaluate(ho["target_vol"], pred_ho_base)
    metrics_ho_x = evaluate(ho["target_vol"], pred_ho_x)

    return {
        "ols_base": ols_base, "ols_x": ols_x,
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
        print(f"홀드아웃 GARCH(1,1)    : {result['holdout_metrics_base']}")
        print(f"홀드아웃 GARCH-X(+감성) : {result['holdout_metrics_x']}")
        print()

        with open(f"{VOL_DIR}/holdout_metrics_{stock_name}_garch.json", "w") as f:
            json.dump({
                "GARCH(1,1)": result["holdout_metrics_base"],
                "GARCH-X": result["holdout_metrics_x"],
            }, f, indent=2)

    print("전체 종목 GARCH 계열 실행 완료.")
