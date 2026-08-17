from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"

FILES = {
    "005930": "analysis_samsung_electronics.csv",
    "000660": "analysis_sk_hynix.csv",
    "247540": "analysis_ecopro_bm.csv",
    "035720": "analysis_kakao.csv",
}

VARIABLES = [
    "rolling_volatility_5d",
    "cc_return",
    "overnight_sentiment_ffill",
    "parkinson_volatility",
]

rows = []

for stock_code, filename in FILES.items():
    path = ANALYSIS_DIR / filename
    df = pd.read_csv(path)

    for variable in VARIABLES:
        if variable not in df.columns:
            continue

        series = (
            pd.to_numeric(
                df[variable],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(series) < 20 or series.nunique() < 2:
            continue

        for lag in (1, 2, 3):
            result = adfuller(
                series,
                maxlag=lag,
                autolag=None,
            )

            rows.append(
                {
                    "stock_code": stock_code,
                    "variable": variable,
                    "fixed_lag": lag,
                    "nobs": int(result[3]),
                    "adf_stat": float(result[0]),
                    "p_value": float(result[1]),
                    "status": (
                        "STATIONARY_5PCT"
                        if result[1] < 0.05
                        else "NONSTATIONARY_5PCT"
                    ),
                }
            )

result_df = pd.DataFrame(rows)

output_path = (
    ANALYSIS_DIR
    / "statistics"
    / "adf_lag_sensitivity.csv"
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

rolling_result = result_df[
    result_df["variable"].eq(
        "rolling_volatility_5d"
    )
]

print(rolling_result.to_string(index=False))
print(f"\n저장 완료: {output_path}")