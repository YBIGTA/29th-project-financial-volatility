from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR
    / "data"
    / "analysis"
    / "analysis_kakao.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "analysis"
    / "statistics"
    / "kakao_overnight_adf_check.csv"
)


df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
)

original = (
    pd.to_numeric(
        df["overnight_sentiment_ffill"],
        errors="coerce",
    )
    .replace([np.inf, -np.inf], np.nan)
)

differenced = original.diff()

df["overnight_sentiment_diff"] = differenced


rows = []

for variable, series in {
    "overnight_sentiment_ffill": original,
    "overnight_sentiment_diff": differenced,
}.items():

    clean = series.dropna()

    auto_result = adfuller(
        clean,
        maxlag=3,
        autolag="AIC",
    )

    rows.append(
        {
            "variable": variable,
            "lag_method": "AIC_MAX_3",
            "used_lag": int(auto_result[2]),
            "nobs": int(auto_result[3]),
            "adf_stat": float(auto_result[0]),
            "p_value": float(auto_result[1]),
            "status": (
                "STATIONARY_5PCT"
                if auto_result[1] < 0.05
                else "NONSTATIONARY_5PCT"
            ),
        }
    )

    for fixed_lag in (1, 2, 3):
        fixed_result = adfuller(
            clean,
            maxlag=fixed_lag,
            autolag=None,
        )

        rows.append(
            {
                "variable": variable,
                "lag_method": (
                    f"FIXED_LAG_{fixed_lag}"
                ),
                "used_lag": fixed_lag,
                "nobs": int(fixed_result[3]),
                "adf_stat": float(
                    fixed_result[0]
                ),
                "p_value": float(
                    fixed_result[1]
                ),
                "status": (
                    "STATIONARY_5PCT"
                    if fixed_result[1] < 0.05
                    else "NONSTATIONARY_5PCT"
                ),
            }
        )


result_df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print(result_df.to_string(index=False))
print(f"\n저장 완료: {OUTPUT_PATH}")