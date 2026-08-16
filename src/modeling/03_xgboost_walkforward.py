"""
3단계: XGBoost (base / +sentiment) - Walk-forward + 홀드아웃 (v3: 4종목 전체 루프)

[v2 -> v3 변경사항]
- STOCK_NAME 하나만 처리하던 구조 -> STOCKS 리스트 전체를 순회하도록 변경
- 로직 자체(피처셋, walk-forward, 홀드아웃 분리)는 v2와 동일

[예측 시점 정의] "t+1일 장 시작 직전" (01번 docstring 참고)
"""
import json
import pandas as pd
import numpy as np
import xgboost as xgb

VOL_DIR = "/home/claude/vol_project"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]

FEATURES_BASE = [
    "realized_vol",
    "realized_vol_lag1", "realized_vol_lag2", "realized_vol_lag3",
    "log_ret_lag1", "vkospi_close", "vkospi_lag1",
    "kospi_ret", "kosdaq_ret",
]
FEATURES_SENTIMENT = FEATURES_BASE + ["sentiment_intraday", "sentiment_overnight"]

TARGET = "target_vol"
HOLDOUT_DAYS = 8
MIN_TRAIN = 30

DEFAULT_PARAMS = dict(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)


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


def run_pipeline(features_csv: str):
    df = pd.read_csv(features_csv)
    train_val = df.iloc[:-HOLDOUT_DAYS].reset_index(drop=True)
    holdout = df.iloc[-HOLDOUT_DAYS:].reset_index(drop=True)

    wf_base = walk_forward_predict(train_val, FEATURES_BASE, TARGET, MIN_TRAIN, DEFAULT_PARAMS)
    wf_sentiment = walk_forward_predict(train_val, FEATURES_SENTIMENT, TARGET, MIN_TRAIN, DEFAULT_PARAMS)

    model_base = xgb.XGBRegressor(**DEFAULT_PARAMS)
    model_base.fit(train_val[FEATURES_BASE], train_val[TARGET])
    pred_ho_base = model_base.predict(holdout[FEATURES_BASE])

    model_sent = xgb.XGBRegressor(**DEFAULT_PARAMS)
    model_sent.fit(train_val[FEATURES_SENTIMENT], train_val[TARGET])
    pred_ho_sent = model_sent.predict(holdout[FEATURES_SENTIMENT])

    ho_base = pd.DataFrame({"date": holdout["date"], "actual": holdout[TARGET], "predicted": pred_ho_base})
    ho_sent = pd.DataFrame({"date": holdout["date"], "actual": holdout[TARGET], "predicted": pred_ho_sent})

    return {
        "wf_base": evaluate(wf_base), "wf_sentiment": evaluate(wf_sentiment),
        "holdout_base": evaluate(ho_base), "holdout_sentiment": evaluate(ho_sent),
    }


if __name__ == "__main__":
    for stock_name in STOCKS:
        features_csv = f"{VOL_DIR}/features_{stock_name}.csv"
        r = run_pipeline(features_csv)

        print("=" * 60)
        print(f"[{stock_name}]")
        print("=" * 60)
        print("[A] walk-forward   base:", r["wf_base"], " +sentiment:", r["wf_sentiment"])
        print("[B] holdout        base:", r["holdout_base"], " +sentiment:", r["holdout_sentiment"])
        print()

        with open(f"{VOL_DIR}/holdout_metrics_{stock_name}_xgb.json", "w") as f:
            json.dump({
                "XGBoost base": r["holdout_base"],
                "XGBoost +sentiment": r["holdout_sentiment"],
            }, f, indent=2)

    print("전체 종목 XGBoost 실행 완료.")
