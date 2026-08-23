"""
부록: current/future 감성지수(4변수) - 3개월 표본 모델링 (간단 버전)

[배경] (README "7. 부록" 섹션 참고)
팀 합의: 6개월/2변수(intraday·overnight) 결과를 메인으로 유지하고, 감성분석팀이
새로 산출한 current/future(현재상태 vs 미래전망 감성 분리) 4변수 버전은 3개월
표본으로 별도 부록에서 간단히 다룸. 표본이 작아(종목당 64행) 6개월 메인 파이프라인
수준의 정교함(HAC pooled test, Ridge 진단 등)은 유지하되 결과 해석은 "참고용"으로
가볍게 다룸 - 메인 결론(6개월 기준, 감성지수 유의성 없음)을 대체하지 않음.

[감성 변수 4종]
- intraday_current_sentiment_index / intraday_future_sentiment_index
- overnight_current_sentiment_index / overnight_future_sentiment_index
시간정렬 로직은 기존과 동일 (01번 스크립트 v8 로직 그대로 적용):
- intraday_* : shift 없음 (t일 마감에 확정, t+1일 예측에 사용)
- overnight_* : shift(-1) (t+1일 개장 전 확정되는 값으로 정렬)
current/future 둘 다 같은 시간대(intraday/overnight)면 같은 shift 규칙 적용.

[target] Parkinson(pk)만 사용 (메인에서 확정한 target)
[표본 규모] 종목당 64행(3개월, 2026-05-14~2026-08-14) -> holdout/CV를 비례 축소:
  HOLDOUT_DAYS=10, MIN_TRAIN=30, CV_SPLITS=3, CV_TEST_SIZE=5
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from arch import arch_model
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_appendix"
OUT_DIR = SCRIPT_DIR / "outputs_appendix"
OUT_DIR.mkdir(exist_ok=True)

STOCKS = {
    "samsung_electronics": DATA_DIR / "merged_daily_samsung_electronics.csv",
    "sk_hynix": DATA_DIR / "merged_daily_sk_hynix.csv",
    "kakao": DATA_DIR / "merged_daily_kakao.csv",
    "ecopro_bm": DATA_DIR / "merged_daily_ecopro_bm.csv",
}
MARKET_CSV = DATA_DIR / "market_index_merged.csv"

HOLDOUT_DAYS = 10
MIN_TRAIN = 30
CV_SPLITS = 3
CV_TEST_SIZE = 5
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

SENT_COLS = [
    "intraday_current", "intraday_future", "overnight_current", "overnight_future",
]

PARAM_DIST = {
    "max_depth": [2, 3],
    "n_estimators": [30, 50, 80, 100],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 3],
    "reg_lambda": [1, 5, 10],
}
N_ITER_SEARCH = 60  # 표본이 작아 v9 메인(150)보다 축소


# ---------------------------------------------------------------- 피처 빌드
def load_market(market_csv):
    mkt = pd.read_csv(market_csv)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.strftime("%Y-%m-%d")
    mkt["kospi_ret"] = np.log(mkt["kospi_close"] / mkt["kospi_close"].shift(1))
    mkt["kosdaq_ret"] = np.log(mkt["kosdaq_close"] / mkt["kosdaq_close"].shift(1))
    mkt["vkospi_change"] = np.log(mkt["vkospi_close"] / mkt["vkospi_close"].shift(1))
    return mkt[["date", "kospi_ret", "kosdaq_ret", "vkospi_change"]]


def build_features(stock_csv, market):
    stock = pd.read_csv(stock_csv)
    stock["date"] = pd.to_datetime(stock["date"]).dt.strftime("%Y-%m-%d")
    stock = stock.dropna(subset=["close"]).copy()
    df = stock.merge(market, on="date", how="left").sort_values("date").reset_index(drop=True)

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(5).std()
    df["parkinson_vol"] = np.abs(np.log(df["high"] / df["low"])) / np.sqrt(4 * np.log(2))

    df["realized_vol_lag1"] = df["realized_vol"].shift(1)
    df["realized_vol_lag2"] = df["realized_vol"].shift(2)
    df["realized_vol_lag3"] = df["realized_vol"].shift(3)
    df["log_ret_lag1"] = df["log_ret"].shift(1)
    df["vkospi_change_lag1"] = df["vkospi_change"].shift(1)

    # 감성 4변수 - 시간정렬 규칙은 기존과 동일 (intraday: 그대로, overnight: shift(-1))
    df["intraday_current"] = df["intraday_current_sentiment_index"]
    df["intraday_future"] = df["intraday_future_sentiment_index"]
    df["overnight_current"] = df["overnight_current_sentiment_index"].shift(-1)
    df["overnight_future"] = df["overnight_future_sentiment_index"].shift(-1)

    df["target_vol_pk"] = df["parkinson_vol"].shift(-1)

    df_clean = df.dropna(subset=[
        "realized_vol_lag3", "vkospi_change_lag1", "kospi_ret", "kosdaq_ret",
        "overnight_current", "overnight_future", "target_vol_pk",
    ]).reset_index(drop=True)
    return df_clean


# ---------------------------------------------------------------- GARCH
def garch_walkforward_vol(returns_pct, min_train):
    forecasts = {}
    for i in range(min_train, len(returns_pct)):
        train = returns_pct.iloc[:i]
        res = arch_model(train, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        forecasts[i] = np.sqrt(fc.variance.values[-1, 0]) / 100
    return pd.Series(forecasts)


def evaluate(actual, pred):
    err = np.asarray(actual) - np.asarray(pred)
    return {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean()), "n": len(actual)}


def diebold_mariano(actual, pred1, pred2):
    actual = np.asarray(actual)
    e1 = actual - np.asarray(pred1)
    e2 = actual - np.asarray(pred2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_var = d.var(ddof=1)
    if d_var == 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n}
    dm_stat = d.mean() / np.sqrt(d_var / n)
    p = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p), "n": n}


def run_garch(df):
    returns_pct = df["log_ret"] * 100
    n_total = len(df)
    n_trainval = n_total - HOLDOUT_DAYS
    garch_vol = garch_walkforward_vol(returns_pct, MIN_TRAIN)
    df = df.assign(garch_vol=np.nan)
    df.loc[garch_vol.index, "garch_vol"] = garch_vol.values

    tv = df.iloc[MIN_TRAIN:n_trainval].dropna(subset=["garch_vol"]).copy()
    ho = df.iloc[n_trainval:].copy()

    pred_raw = ho["garch_vol"]
    m_raw = evaluate(ho["target_vol_pk"], pred_raw)

    X_base = sm.add_constant(tv[["garch_vol"]])
    ols_base = sm.OLS(tv["target_vol_pk"], X_base).fit()
    pred_base = ols_base.predict(sm.add_constant(ho[["garch_vol"]], has_constant="add"))
    m_base = evaluate(ho["target_vol_pk"], pred_base)

    X_sent = sm.add_constant(tv[["garch_vol"] + SENT_COLS])
    ols_sent = sm.OLS(tv["target_vol_pk"], X_sent).fit()
    pred_sent = ols_sent.predict(sm.add_constant(ho[["garch_vol"] + SENT_COLS], has_constant="add"))
    m_sent = evaluate(ho["target_vol_pk"], pred_sent)

    dm_raw_vs_sent = diebold_mariano(ho["target_vol_pk"], pred_raw, pred_sent)
    dm_pure_sent = diebold_mariano(ho["target_vol_pk"], pred_base, pred_sent)

    return {
        "pvalues": {c: float(ols_sent.pvalues[c]) for c in SENT_COLS},
        "metrics_raw": m_raw, "metrics_base": m_base, "metrics_sent": m_sent,
        "dm_raw_vs_sent": dm_raw_vs_sent, "dm_pure_sent": dm_pure_sent,
        "holdout_preds": pd.DataFrame({
            "date": ho["date"].values, "actual": ho["target_vol_pk"].values,
            "pred_raw": np.asarray(pred_raw), "pred_base": np.asarray(pred_base),
            "pred_sent": np.asarray(pred_sent),
        }),
    }


# ---------------------------------------------------------------- XGBoost
FEATURES_BASE = [
    "realized_vol", "realized_vol_lag1", "realized_vol_lag2", "realized_vol_lag3",
    "log_ret_lag1", "vkospi_change", "vkospi_change_lag1", "kospi_ret", "kosdaq_ret",
]
FEATURES_SENT = FEATURES_BASE + SENT_COLS


def tune_params(train_val, feature_cols, target_col):
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST_SIZE)
    model = xgb.XGBRegressor(random_state=42)
    search = RandomizedSearchCV(model, PARAM_DIST, n_iter=N_ITER_SEARCH, cv=tscv,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42)
    search.fit(train_val[feature_cols], train_val[target_col])
    return {**{"random_state": 42}, **search.best_params_}, -search.best_score_


def walk_forward_predict(data, feature_cols, target_col, min_train, params):
    records = []
    for i in range(min_train, len(data)):
        train, test = data.iloc[:i], data.iloc[i:i + 1]
        model = xgb.XGBRegressor(**params)
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(test[feature_cols])[0]
        records.append({"date": test["date"].values[0], "actual": test[target_col].values[0], "predicted": pred})
    return pd.DataFrame(records)


def run_xgb(df):
    n_trainval = len(df) - HOLDOUT_DAYS
    train_val = df.iloc[:n_trainval].reset_index(drop=True)

    params_base, cv_rmse_base = tune_params(train_val, FEATURES_BASE, "target_vol_pk")
    params_sent, cv_rmse_sent = tune_params(train_val, FEATURES_SENT, "target_vol_pk")

    full_base = walk_forward_predict(df, FEATURES_BASE, "target_vol_pk", MIN_TRAIN, params_base)
    full_sent = walk_forward_predict(df, FEATURES_SENT, "target_vol_pk", MIN_TRAIN, params_sent)
    n_diag = n_trainval - MIN_TRAIN

    ho_base = full_base.iloc[n_diag:].reset_index(drop=True)
    ho_sent = full_sent.iloc[n_diag:].reset_index(drop=True)

    m_base = evaluate(ho_base["actual"], ho_base["predicted"])
    m_sent = evaluate(ho_sent["actual"], ho_sent["predicted"])
    dm = diebold_mariano(ho_base["actual"], ho_base["predicted"], ho_sent["predicted"])

    return {
        "params_base": params_base, "params_sent": params_sent,
        "cv_rmse_base": cv_rmse_base, "cv_rmse_sent": cv_rmse_sent,
        "metrics_base": m_base, "metrics_sent": m_sent, "dm_base_vs_sent": dm,
        "holdout_preds": pd.DataFrame({
            "date": ho_base["date"].values, "actual": ho_base["actual"].values,
            "pred_base": ho_base["predicted"].values, "pred_sent": ho_sent["predicted"].values,
        }),
    }


# ---------------------------------------------------------------- pooled 검정
def hac_lrv(d, maxlag):
    n = len(d)
    d_c = d - d.mean()
    lrv = (d_c @ d_c) / n
    for lag in range(1, maxlag + 1):
        if lag >= n:
            break
        w = 1 - lag / (maxlag + 1)
        lrv += 2 * w * (d_c[lag:] @ d_c[:-lag]) / n
    return max(lrv, 1e-12)


def pooled_by_date_test(pairs, maxlag=1):
    d_std = []
    for actual, pred1, pred2 in pairs:
        actual = np.asarray(actual)
        d = (actual - np.asarray(pred1)) ** 2 - (actual - np.asarray(pred2)) ** 2
        std = d.std(ddof=1)
        d_std.append(d / std if std > 0 else d * 0)
    d_by_date = np.mean(np.vstack(d_std), axis=0)
    n_dates = len(d_by_date)
    lrv = hac_lrv(d_by_date, maxlag)
    t_stat = d_by_date.mean() / np.sqrt(lrv / n_dates)
    t_p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_dates - 1))
    wins2 = int((d_by_date > 0).sum())
    sign_p = stats.binomtest(wins2, n_dates, 0.5).pvalue
    return {"n_dates": n_dates, "pooled_hac_t_stat": float(t_stat), "pooled_hac_t_p": float(t_p),
            "wins_model2": wins2, "sign_test_p": float(sign_p)}


if __name__ == "__main__":
    market = load_market(MARKET_CSV)
    garch_results, xgb_results = {}, {}

    for stock_name, stock_csv in STOCKS.items():
        df = build_features(stock_csv, market)
        print(f"[{stock_name}] 사용 가능 행 수: {len(df)}, 기간: {df['date'].min()} ~ {df['date'].max()}")
        df.to_csv(OUT_DIR / f"features_{stock_name}.csv", index=False)

        g = run_garch(df)
        garch_results[stock_name] = g
        print(f"  GARCH 원본={g['metrics_raw']['RMSE']:.4f}  "
              f"회귀(감성없음)={g['metrics_base']['RMSE']:.4f}  회귀(감성포함)={g['metrics_sent']['RMSE']:.4f}")
        print(f"  감성계수 p-value: { {k: round(v,3) for k,v in g['pvalues'].items()} }")
        g["holdout_preds"].to_csv(OUT_DIR / f"garch_holdout_preds_{stock_name}.csv", index=False)

        x = run_xgb(df)
        xgb_results[stock_name] = x
        print(f"  XGB base={x['metrics_base']['RMSE']:.4f}  XGB +sentiment={x['metrics_sent']['RMSE']:.4f}")
        x["holdout_preds"].to_csv(OUT_DIR / f"xgb_holdout_preds_{stock_name}.csv", index=False)
        print()

    # 4종목 통합 pooled 검정
    garch_pairs_raw_vs_sent = [(garch_results[s]["holdout_preds"]["actual"],
                                 garch_results[s]["holdout_preds"]["pred_raw"],
                                 garch_results[s]["holdout_preds"]["pred_sent"]) for s in STOCKS]
    garch_pairs_pure_sent = [(garch_results[s]["holdout_preds"]["actual"],
                               garch_results[s]["holdout_preds"]["pred_base"],
                               garch_results[s]["holdout_preds"]["pred_sent"]) for s in STOCKS]
    xgb_pairs = [(xgb_results[s]["holdout_preds"]["actual"],
                  xgb_results[s]["holdout_preds"]["pred_base"],
                  xgb_results[s]["holdout_preds"]["pred_sent"]) for s in STOCKS]

    pooled = {
        "garch_raw_vs_sent": pooled_by_date_test(garch_pairs_raw_vs_sent),
        "garch_pure_sent_contribution": pooled_by_date_test(garch_pairs_pure_sent),
        "xgb_base_vs_sent": pooled_by_date_test(xgb_pairs),
    }
    print("=" * 60)
    print("4종목 통합(pooled) 검정 결과 (n_dates=10, 참고용 - 표본 작아 검정력 낮음)")
    print("=" * 60)
    for k, v in pooled.items():
        print(f"  {k}: p={v['pooled_hac_t_p']:.3f} (sign test p={v['sign_test_p']:.3f}, {v['wins_model2']}/{v['n_dates']})")

    with open(OUT_DIR / "pooled_significance_appendix.json", "w") as f:
        json.dump(pooled, f, indent=2)

    summary_rows = []
    for s in STOCKS:
        summary_rows.append({
            "종목": s,
            "GARCH 원본": round(garch_results[s]["metrics_raw"]["RMSE"], 4),
            "GARCH+회귀(감성없음)": round(garch_results[s]["metrics_base"]["RMSE"], 4),
            "GARCH+회귀(감성포함,4변수)": round(garch_results[s]["metrics_sent"]["RMSE"], 4),
            "XGBoost base": round(xgb_results[s]["metrics_base"]["RMSE"], 4),
            "XGBoost +sentiment(4변수)": round(xgb_results[s]["metrics_sent"]["RMSE"], 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_appendix.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\n-> {OUT_DIR / 'summary_appendix.csv'} 저장 완료")
