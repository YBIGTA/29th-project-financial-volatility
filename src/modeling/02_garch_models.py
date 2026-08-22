"""
2단계: GARCH 계열 - Target A/B 병렬 비교 + Ridge 대안 + DM test (v8)

[v7 -> v8 변경사항]
- Target A(target_vol_5d)/B(target_vol_pk) 각각에 대해 동일한 절차로 전부 실행
  (01번 참고 - 01번에서 감성 시간정렬 버그도 같이 수정됨)
- 경로를 상대경로로 변경 (VS Code 등 로컬 실행 지원)
- (지난 버전) DM test, Ridge 비교 로직은 유지

[예측 시점 정의] "t+1일 장 시작 직전" (01번 docstring 참고)
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
VOL_DIR = SCRIPT_DIR / "outputs"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]
TARGETS = {"pk": "target_vol_pk", "5d": "target_vol_5d"}
# ↑ v9: Parkinson(pk)을 메인 target으로 확정 (GARCH의 1일-앞 조건부분산 추정과 정의상
#   더 가깝고, 연구질문의 "장중 변동성"과 직접 대응됨 - MODELING_README.md 참고).
#   5d(rolling_volatility_5d)는 폐기하지 않고 부록(로버스트니스 체크)으로 계속 산출함.

HOLDOUT_DAYS = 15
MIN_TRAIN = 50
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def garch_walkforward_vol(returns_pct: pd.Series, min_train: int) -> pd.Series:
    """Expanding-window walk-forward one-step-ahead GARCH(1,1) 변동성 예측."""
    forecasts = {}
    n = len(returns_pct)
    for i in range(min_train, n):
        train = returns_pct.iloc[:i]
        res = arch_model(train, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        var_fc = fc.variance.values[-1, 0]
        forecasts[i] = np.sqrt(var_fc) / 100
    return pd.Series(forecasts)


def evaluate(actual: pd.Series, pred) -> dict:
    err = actual.values - np.asarray(pred)
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    return {"RMSE": rmse, "MAE": mae, "n": len(actual)}


def fit_ridge(tv: pd.DataFrame, ho: pd.DataFrame, feature_cols: list, target_col: str):
    n_splits = max(2, min(3, len(tv) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, cv=tscv))
    model.fit(tv[feature_cols], tv[target_col])
    pred_ho = model.predict(ho[feature_cols])
    return pred_ho, model.named_steps["ridgecv"].alpha_


def diebold_mariano(actual: pd.Series, pred1, pred2) -> dict:
    actual = actual.values
    e1 = actual - np.asarray(pred1)
    e2 = actual - np.asarray(pred2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    if d_var == 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n}
    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n": n}


def run_pipeline(features_csv, target_col: str, garch_vol_cache: pd.Series = None):
    df = pd.read_csv(features_csv)
    returns_pct = df["log_ret"] * 100
    n_total = len(df)
    n_trainval = n_total - HOLDOUT_DAYS

    if garch_vol_cache is None:
        garch_vol_cache = garch_walkforward_vol(returns_pct, MIN_TRAIN)
    df = df.assign(garch_vol=np.nan)
    df.loc[garch_vol_cache.index, "garch_vol"] = garch_vol_cache.values

    tv = df.iloc[MIN_TRAIN:n_trainval].dropna(subset=["garch_vol"]).copy()
    ho = df.iloc[n_trainval:].copy()

    pred_ho_raw = ho["garch_vol"]
    metrics_ho_raw = evaluate(ho[target_col], pred_ho_raw)

    X_base = sm.add_constant(tv[["garch_vol"]])
    ols_base = sm.OLS(tv[target_col], X_base).fit()
    X_ho_base = sm.add_constant(ho[["garch_vol"]], has_constant="add")
    pred_ho_base = ols_base.predict(X_ho_base)
    metrics_ho_base = evaluate(ho[target_col], pred_ho_base)

    sent_cols = ["garch_vol", "sentiment_intraday", "sentiment_overnight"]
    X_x = sm.add_constant(tv[sent_cols])
    ols_x = sm.OLS(tv[target_col], X_x).fit()
    X_ho_x = sm.add_constant(ho[sent_cols], has_constant="add")
    pred_ho_x = ols_x.predict(X_ho_x)
    metrics_ho_x = evaluate(ho[target_col], pred_ho_x)

    pred_ho_ridge_base, alpha_base = fit_ridge(tv, ho, ["garch_vol"], target_col)
    metrics_ho_ridge_base = evaluate(ho[target_col], pred_ho_ridge_base)
    pred_ho_ridge_x, alpha_x = fit_ridge(tv, ho, sent_cols, target_col)
    metrics_ho_ridge_x = evaluate(ho[target_col], pred_ho_ridge_x)

    dm_results = {
        "원본 vs OLS(감성포함)": diebold_mariano(ho[target_col], pred_ho_raw, pred_ho_x),
        "원본 vs Ridge(감성포함)": diebold_mariano(ho[target_col], pred_ho_raw, pred_ho_ridge_x),
        "OLS vs Ridge(감성포함)": diebold_mariano(ho[target_col], pred_ho_x, pred_ho_ridge_x),
    }

    return {
        "garch_vol_cache": garch_vol_cache,
        "ols_x": ols_x,
        "holdout_metrics_raw": metrics_ho_raw,
        "holdout_metrics_base": metrics_ho_base,
        "holdout_metrics_x": metrics_ho_x,
        "holdout_metrics_ridge_base": metrics_ho_ridge_base,
        "holdout_metrics_ridge_x": metrics_ho_ridge_x,
        "ridge_alpha_base": alpha_base, "ridge_alpha_x": alpha_x,
        "dm_results": dm_results,
        "holdout_preds": pd.DataFrame({
            "date": ho["date"].values, "actual": ho[target_col].values,
            "pred_raw": np.asarray(pred_ho_raw), "pred_ols_base": np.asarray(pred_ho_base),
            "pred_ols_x": np.asarray(pred_ho_x),
        }),
    }


if __name__ == "__main__":
    for stock_name in STOCKS:
        features_csv = VOL_DIR / f"features_{stock_name}.csv"
        garch_cache = None  # GARCH(1,1)은 target과 무관(종목 수익률만 사용) -> 한 번만 계산해 재사용

        for tname, tcol in TARGETS.items():
            result = run_pipeline(features_csv, tcol, garch_vol_cache=garch_cache)
            garch_cache = result["garch_vol_cache"]

            print("=" * 60)
            print(f"[{stock_name}] Target={tname} ({tcol})")
            print("=" * 60)
            print("OLS 감성지수 계수 p-value:",
                  dict(result["ols_x"].pvalues[["sentiment_intraday", "sentiment_overnight"]]))
            print(f"홀드아웃 [1] GARCH 원본              : {result['holdout_metrics_raw']}")
            print(f"홀드아웃 [2] GARCH+OLS(감성없음)     : {result['holdout_metrics_base']}")
            print(f"홀드아웃 [3] GARCH+OLS(감성포함)     : {result['holdout_metrics_x']}")
            print(f"홀드아웃 [2b]GARCH+Ridge(감성없음), alpha={result['ridge_alpha_base']:.3g} : {result['holdout_metrics_ridge_base']}")
            print(f"홀드아웃 [3b]GARCH+Ridge(감성포함), alpha={result['ridge_alpha_x']:.3g} : {result['holdout_metrics_ridge_x']}")
            print("DM test:")
            for name, dm in result["dm_results"].items():
                print(f"  {name}: stat={dm['dm_stat']:.3f}, p={dm['p_value']:.3f}")
            print()

            with open(VOL_DIR / f"holdout_metrics_{stock_name}_garch_{tname}.json", "w") as f:
                json.dump({
                    "GARCH 원본": result["holdout_metrics_raw"],
                    "GARCH+회귀(감성없음)": result["holdout_metrics_base"],
                    "GARCH+회귀(감성포함)": result["holdout_metrics_x"],
                    "GARCH+Ridge(감성없음)": result["holdout_metrics_ridge_base"],
                    "GARCH+Ridge(감성포함)": result["holdout_metrics_ridge_x"],
                    "dm_results": result["dm_results"],
                }, f, indent=2)

            result["holdout_preds"].to_csv(
                VOL_DIR / f"garch_holdout_preds_{stock_name}_{tname}.csv", index=False)

    print("전체 종목 x Target A/B GARCH 계열 실행 완료.")
