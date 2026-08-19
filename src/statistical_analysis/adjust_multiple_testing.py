from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parents[2]
STATISTICS_DIR = ROOT_DIR / "data" / "analysis" / "statistics"


def adjust_file(filename: str) -> None:
    input_path = STATISTICS_DIR / filename
    output_path = STATISTICS_DIR / filename.replace(
        ".csv",
        "_adjusted.csv",
    )

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    df["p_value"] = pd.to_numeric(
        df["p_value"],
        errors="coerce",
    )

    valid = (
        df["p_value"].notna()
        & (df["status"] == "OK")
    )

    df["p_value_bh"] = pd.NA
    df["significant_bh_5pct"] = False
    df["p_value_bonferroni"] = pd.NA
    df["significant_bonferroni_5pct"] = False

    if valid.any():
        p_values = df.loc[valid, "p_value"].to_numpy()

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

        df.loc[valid, "p_value_bh"] = p_bh
        df.loc[valid, "significant_bh_5pct"] = reject_bh

        df.loc[valid, "p_value_bonferroni"] = p_bonf
        df.loc[valid, "significant_bonferroni_5pct"] = reject_bonf

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[{filename}]")
    print(f"전체 검정 수: {valid.sum()}")
    print(
        "보정 전 유의:",
        (df.loc[valid, "p_value"] < 0.05).sum(),
    )
    print(
        "BH 보정 후 유의:",
        df.loc[valid, "significant_bh_5pct"].sum(),
    )
    print(
        "Bonferroni 보정 후 유의:",
        df.loc[
            valid,
            "significant_bonferroni_5pct",
        ].sum(),
    )
    print(f"저장: {output_path}")


def main() -> None:
    adjust_file("hac_ols_results.csv")
    adjust_file("granger_results.csv")


if __name__ == "__main__":
    main()