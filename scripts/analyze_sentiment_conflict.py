"""Measure positive/negative probability conflicts from a multitask checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.finbert_dataset.select_active_learning import predict


def summarize(frame: pd.DataFrame, head: str) -> dict:
    positive = frame[f"model_{head}_p_positive"]
    negative = frame[f"model_{head}_p_negative"]
    intensity = positive + negative
    difference = (positive - negative).abs()
    conflict = 4 * positive * negative

    strict = (positive >= 0.35) & (negative >= 0.35)
    near_tie = (difference <= 0.10) & (intensity >= 0.70)
    moderate = (difference <= 0.15) & (intensity >= 0.60)
    high_conflict = conflict >= 0.75

    return {
        "strict_both_ge_035": int(strict.sum()),
        "strict_pct": float(strict.mean()),
        "near_tie_intensity_ge_070": int(near_tie.sum()),
        "near_tie_070_pct": float(near_tie.mean()),
        "moderate_tie_intensity_ge_060": int(moderate.sum()),
        "moderate_pct": float(moderate.mean()),
        "conflict_ge_075": int(high_conflict.sum()),
        "conflict_ge_075_pct": float(high_conflict.mean()),
        "conflict_quantiles": {
            str(q): float(conflict.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    predictions = predict(frame, args.checkpoint, args.batch_size, args.max_length)
    result = {
        "rows": len(predictions),
        "current": summarize(predictions, "current"),
        "future": summarize(predictions, "future"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
