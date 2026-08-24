"""Review how the squared sentiment score behaves on an inference CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame: pd.DataFrame, head: str) -> dict:
    positive = frame[f"model_{head}_p_positive"]
    negative = frame[f"model_{head}_p_negative"]
    neutral = frame[f"model_{head}_p_neutral"]
    difference = positive - negative
    intensity = positive + negative
    score = frame[f"{head}_sentiment_score"]
    probabilities = frame[
        [
            f"model_{head}_p_negative",
            f"model_{head}_p_neutral",
            f"model_{head}_p_positive",
        ]
    ]
    maximum_probability = probabilities.max(axis=1)

    nonzero = difference.ne(0)
    sign_mismatch = np.sign(difference[nonzero]).ne(np.sign(score[nonzero]))
    return {
        "mean_abs_difference": float(difference.abs().mean()),
        "mean_abs_score": float(score.abs().mean()),
        "median_abs_difference": float(difference.abs().median()),
        "median_abs_score": float(score.abs().median()),
        "mean_intensity": float(intensity.mean()),
        "intensity_quantiles": {
            str(q): float(intensity.quantile(q))
            for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
        },
        "pearson_difference_score": float(difference.corr(score)),
        "spearman_difference_score": float(difference.corr(score, method="spearman")),
        "sign_mismatch_count": int(sign_mismatch.sum()),
        "shrink_over_50pct_rate": float(intensity.lt(0.5).mean()),
        "difference_abs_lt_005_rate": float(difference.abs().lt(0.05).mean()),
        "score_abs_lt_005_rate": float(score.abs().lt(0.05).mean()),
        "maximum_probability_mean": float(maximum_probability.mean()),
        "maximum_probability_ge_08_rate": float(maximum_probability.ge(0.8).mean()),
        "maximum_probability_lt_05_rate": float(maximum_probability.lt(0.5).mean()),
        "score_quantiles": {
            str(q): float(score.quantile(q))
            for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
        },
        "neutral_identity_max_error": float((intensity - (1 - neutral)).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype={"target_stock": "string"},
    )
    result: dict = {"rows": len(frame)}
    for head in ("current", "future"):
        result[head] = summarize(frame, head)
        frame[f"_{head}_intensity"] = (
            frame[f"model_{head}_p_positive"] + frame[f"model_{head}_p_negative"]
        )
        grouped = frame.groupby("target_stock").agg(
            rows=("item_id", "size"),
            mean_score=(f"{head}_sentiment_score", "mean"),
            std_score=(f"{head}_sentiment_score", "std"),
            mean_intensity=(f"_{head}_intensity", "mean"),
        )
        result[f"{head}_by_stock"] = grouped.round(6).to_dict(orient="index")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
