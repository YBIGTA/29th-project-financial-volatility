"""
3단계: XGBoost (base / +sentiment) - Target A/B 병렬 비교 + 튜닝 + Walk-forward 홀드아웃 (v10)

[v9 -> v10 변경사항] (GARCH와 평가방식 불일치 수정)
- (v9까지) holdout 15일을 train_val(87일)로 "한 번만" 학습한 모델로 일괄 예측했음.
  GARCH는 holdout도 매일 재학습하는 walk-forward인데 XGBoost만 다른 방식이라
  공정한 비교가 아니었음.
- (v10) 하이퍼파라미터는 train_val에서 튜닝한 값을 그대로 고정(재튜닝 없음 - nested
  tuning은 연산비용이 커서 안 함)하고, holdout 15일도 GARCH와 동일하게 "그 전날까지
  데이터로 매일 재학습 -> 다음날 예측" 방식(walk-forward)으로 재평가하도록 변경.
  파라미터가 고정이라 추가 비용은 재학습 15회뿐(탐색 150회 대비 훨씬 저렴).

[v8 -> v9 변경사항] (탐색범위 적절성 재검토 요청 반영)
- 탐색범위 확장: min_child_weight, colsample_bytree, reg_lambda 추가 (GridSearchCV ->
  RandomizedSearchCV(n_iter=150)로 전환)
- CV fold 크기 고정: TimeSeriesSplit(n_splits=4, test_size=10)로 변경

[v7 -> v8 변경사항]
- Target A(target_vol_5d)/B(target_vol_pk) 각각에 대해 동일 절차로 전부 실행
- base/+sentiment 모두 01번에서 수정된 sentiment_overnight(시간정렬 버그 수정)을 사용
- 경로를 상대경로로 변경 (VS Code 등 로컬 실행 지원)

[예측 시점 정의] "t+1일 장 시작 직전" (01번 docstring 참고)
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

SCRIPT_DIR = Path(__file__).resolve().parent
VOL_DIR = SCRIPT_DIR / "outputs"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]
TARGETS = {"pk": "target_vol_pk", "5d": "target_vol_5d"}
# ↑ v9: Parkinson(pk)을 메인 target으로 확정, 5d는 부록(로버스트니스 체크)으로 유지

FEATURES_BASE = [
    "realized_vol",
    "realized_vol_lag1", "realized_vol_lag2", "realized_vol_lag3",
    "log_ret_lag1", "vkospi_change", "vkospi_change_lag1",
    "kospi_ret", "kosdaq_ret",
]
FEATURES_SENTIMENT = FEATURES_BASE + ["sentiment_intraday", "sentiment_overnight"]

HOLDOUT_DAYS = 15
MIN_TRAIN = 50
CV_SPLITS = 4
CV_TEST_SIZE = 10  # (v9) 각 CV fold의 test 크기를 고정 -> 첫 fold의 학습 표본이 너무
                   # 작아지는 문제 방지. TimeSeriesSplit(5) 기본값은 train_val(87행)
                   # 기준 첫 fold 학습표본이 17개뿐이라 하이퍼파라미터 선택이 불안정했음
                   # -> (n_splits=4, test_size=10)로 바꾸면 첫 fold부터 학습표본 47개 확보

PARAM_DISTRIBUTIONS = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],   # (v9 추가) 상관된 피처(realized_vol_lag1~3)가
                                                   # 많아 트리마다 일부 피처만 쓰게 해 과적합 억제
    "min_child_weight": [1, 3, 5],               # (v9 추가) 작은 표본에서 리프 과세분화 방지
    "reg_lambda": [1, 5, 10],                     # (v9 추가) L2 규제 - 계수 크기 직접 억제
}
N_ITER_SEARCH = 150  # RandomizedSearchCV 탐색 횟수 (전수탐색은 조합이 너무 많아짐:
                     # 3*4*5*5*4*3*3=10,800개 -> 랜덤 샘플링으로 대체)

BASE_PARAMS = dict(random_state=42)


def tune_params(train_val: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
    """train_val 구간 안에서만 TimeSeriesSplit CV로 하이퍼파라미터 탐색 (holdout 미사용).
    (v9) 전수탐색(GridSearchCV) -> RandomizedSearchCV로 변경, 탐색범위도 규제 파라미터
    (min_child_weight/colsample_bytree/reg_lambda) 포함해 확장. CV도 fold 크기 고정."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST_SIZE)
    model = xgb.XGBRegressor(**BASE_PARAMS)
    search = RandomizedSearchCV(
        model, PARAM_DISTRIBUTIONS, n_iter=N_ITER_SEARCH, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42,
    )
    search.fit(train_val[feature_cols], train_val[target_col])
    return search.best_params_, -search.best_score_


def walk_forward_predict(data: pd.DataFrame, feature_cols: list, target_col: str,
                          min_train_size: int, params: dict) -> pd.DataFrame:
    records = []
    n = len(data)
    for i in range(min_train_size, n):
        train = data.iloc[:i]
        test = data.iloc[i:i + 1]
        model = xgb.XGBRegressor(**params)
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(test[feature_cols])[0]
        records.append({"date": test["date"].values[0],
                         "actual": test[target_col].values[0], "predicted": pred})
    return pd.DataFrame(records)


def evaluate(result_df: pd.DataFrame) -> dict:
    err = result_df["actual"] - result_df["predicted"]
    rmse = np.sqrt((err ** 2).mean())
    mae = err.abs().mean()
    return {"RMSE": rmse, "MAE": mae, "n": len(result_df)}


def diebold_mariano(actual, pred1, pred2) -> dict:
    actual = np.asarray(actual)
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


def run_pipeline(features_csv, target_col: str):
    df = pd.read_csv(features_csv)
    n_trainval = len(df) - HOLDOUT_DAYS
    train_val = df.iloc[:n_trainval].reset_index(drop=True)

    # ---- 튜닝 (train_val 87일 안에서만, holdout 절대 사용 안 함) ----
    best_params_base, cv_rmse_base = tune_params(train_val, FEATURES_BASE, target_col)
    best_params_sent, cv_rmse_sent = tune_params(train_val, FEATURES_SENTIMENT, target_col)
    params_base = {**BASE_PARAMS, **best_params_base}
    params_sent = {**BASE_PARAMS, **best_params_sent}

    # ---- Walk-forward를 전체 데이터(train_val+holdout)에 대해 한 번에 수행 (v10) ----
    # (v9까지) holdout은 train_val 87일로 "한 번만" 학습한 모델로 15일을 일괄 예측했음
    # -> GARCH(매일 재학습하는 walk-forward)와 평가 방식이 달라 공정한 비교가 아니었음
    # (v10) 튜닝된 파라미터는 고정한 채(재튜닝 없음 - 비용 문제로 nested tuning은 안 함),
    # holdout 15일도 GARCH와 동일하게 "그 전날까지 데이터로 매일 재학습 -> 다음날 예측"
    # 방식으로 변경. 파라미터가 고정이라 추가 비용은 재학습 15회뿐 (탐색 대비 매우 저렴).
    full_wf_base = walk_forward_predict(df, FEATURES_BASE, target_col, MIN_TRAIN, params_base)
    full_wf_sent = walk_forward_predict(df, FEATURES_SENTIMENT, target_col, MIN_TRAIN, params_sent)

    n_diag = n_trainval - MIN_TRAIN  # train_val 안에서의 walk-forward 예측 개수 (진단용, 기존과 동일)
    wf_base = full_wf_base.iloc[:n_diag].reset_index(drop=True)          # 진단용 (기존과 동일)
    wf_sentiment = full_wf_sent.iloc[:n_diag].reset_index(drop=True)     # 진단용 (기존과 동일)
    ho_base = full_wf_base.iloc[n_diag:].reset_index(drop=True)          # 홀드아웃 (v10: walk-forward로 재평가)
    ho_sent = full_wf_sent.iloc[n_diag:].reset_index(drop=True)          # 홀드아웃 (v10: walk-forward로 재평가)

    pred_ho_base = ho_base["predicted"].values
    pred_ho_sent = ho_sent["predicted"].values
    dm_base_vs_sent = diebold_mariano(ho_base["actual"], pred_ho_base, pred_ho_sent)

    holdout_preds = pd.DataFrame({
        "date": ho_base["date"].values, "actual": ho_base["actual"].values,
        "pred_base": pred_ho_base, "pred_sent": pred_ho_sent,
    })

    return {
        "best_params_base": params_base, "best_params_sent": params_sent,
        "cv_rmse_base": cv_rmse_base, "cv_rmse_sent": cv_rmse_sent,
        "wf_base": evaluate(wf_base), "wf_sentiment": evaluate(wf_sentiment),
        "holdout_base": evaluate(ho_base), "holdout_sentiment": evaluate(ho_sent),
        "dm_base_vs_sent": dm_base_vs_sent,
        "holdout_preds": holdout_preds,
    }


if __name__ == "__main__":
    for stock_name in STOCKS:
        features_csv = VOL_DIR / f"features_{stock_name}.csv"
        for tname, tcol in TARGETS.items():
            r = run_pipeline(features_csv, tcol)

            print("=" * 60)
            print(f"[{stock_name}] Target={tname} ({tcol})")
            print("=" * 60)
            print(f"튜닝 파라미터 (base)      : {r['best_params_base']}  (CV RMSE={r['cv_rmse_base']:.4f})")
            print(f"튜닝 파라미터 (+sentiment): {r['best_params_sent']}  (CV RMSE={r['cv_rmse_sent']:.4f})")
            print("[A] walk-forward   base:", r["wf_base"], " +sentiment:", r["wf_sentiment"])
            print("[B] holdout        base:", r["holdout_base"], " +sentiment:", r["holdout_sentiment"])
            print(f"DM test (base vs +sentiment): stat={r['dm_base_vs_sent']['dm_stat']:.3f}, "
                  f"p={r['dm_base_vs_sent']['p_value']:.3f}")
            print()

            with open(VOL_DIR / f"holdout_metrics_{stock_name}_xgb_{tname}.json", "w") as f:
                json.dump({
                    "XGBoost base": r["holdout_base"],
                    "XGBoost +sentiment": r["holdout_sentiment"],
                    "dm_base_vs_sent": r["dm_base_vs_sent"],
                }, f, indent=2)

            with open(VOL_DIR / f"xgb_tuning_{stock_name}_{tname}.json", "w") as f:
                json.dump({
                    "best_params_base": r["best_params_base"], "best_params_sent": r["best_params_sent"],
                    "cv_rmse_base": r["cv_rmse_base"], "cv_rmse_sent": r["cv_rmse_sent"],
                }, f, indent=2)

            r["holdout_preds"].to_csv(VOL_DIR / f"xgb_holdout_preds_{stock_name}_{tname}.csv", index=False)

    print("전체 종목 x Target A/B XGBoost 튜닝 + 실행 완료.")
