from pathlib import Path
import sys

import pandas as pd
from statsmodels.stats.multitest import multipletests


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from statistical_analysis.pipeline import (  # noqa: E402
    ANALYSIS_DIR,
    STATISTICS_DIR,
    STOCK_CONFIG,
    run_granger,
)


def main() -> None:
    rows = []

    reverse_specs = [
        (
            "overnight_sentiment_ffill",
            "cc_return",
            "past_return_to_overnight_sentiment",
        ),
        (
            "overnight_sentiment_ffill",
            "parkinson_volatility",
            "past_volatility_to_overnight_sentiment",
        ),
        (
            "intraday_sentiment_ffill",
            "cc_return",
            "past_return_to_intraday_sentiment",
        ),
        (
            "intraday_sentiment_ffill",
            "parkinson_volatility",
            "past_volatility_to_intraday_sentiment",
        ),
    ]

    for stock_code, stock_info in STOCK_CONFIG.items():
        stock_label = stock_info["label"]

        path = (
            ANALYSIS_DIR
            / f"analysis_{stock_label}.csv"
        )

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = (
            df.sort_values("date")
            .reset_index(drop=True)
        )

        for outcome, cause, analysis_name in reverse_specs:
            results = run_granger(
                df=df,
                outcome=outcome,
                cause=cause,
            )

            for result in results:
                rows.append(
                    {
                        "stock_code": stock_code,
                        "stock_name": stock_info["name"],
                        "analysis": analysis_name,
                        "outcome": outcome,
                        "cause": cause,
                        **result,
                    }
                )

    result_df = pd.DataFrame(rows)

    valid = (
        result_df["p_value"].notna()
        & (result_df["status"] == "OK")
    )

    result_df["p_value_bh"] = pd.NA
    result_df["significant_bh_5pct"] = False
    result_df["p_value_bonferroni"] = pd.NA
    result_df["significant_bonferroni_5pct"] = False

    if valid.any():
        p_values = result_df.loc[
            valid,
            "p_value",
        ].to_numpy()

        reject_bh, p_bh, _, _ = multipletests(
            p_values,
            alpha=0.05,
            method="fdr_bh",
        )

        reject_bonf, p_bonf, _, _ = multipletests(
            p_values,
            alpha=0.05,
            method="bonferroni",
        )

        result_df.loc[
            valid,
            "p_value_bh",
        ] = p_bh

        result_df.loc[
            valid,
            "significant_bh_5pct",
        ] = reject_bh

        result_df.loc[
            valid,
            "p_value_bonferroni",
        ] = p_bonf

        result_df.loc[
            valid,
            "significant_bonferroni_5pct",
        ] = reject_bonf

    output_path = (
        STATISTICS_DIR
        / "reverse_granger_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    significant = result_df[
        result_df["p_value"] < 0.05
    ]

    print("\n=== 보정 전 유의 결과 ===")
    if significant.empty:
        print("유의한 역방향 관계가 없습니다.")
    else:
        print(
            significant[
                [
                    "stock_code",
                    "analysis",
                    "lag",
                    "p_value",
                ]
            ].to_string(index=False)
        )

    print("\n=== BH 보정 후 유의 결과 ===")
    significant_bh = result_df[
        result_df["significant_bh_5pct"] == True
    ]

    if significant_bh.empty:
        print("BH 보정 후 유의한 결과가 없습니다.")
    else:
        print(
            significant_bh[
                [
                    "stock_code",
                    "analysis",
                    "lag",
                    "p_value",
                    "p_value_bh",
                ]
            ].to_string(index=False)
        )

    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()