"""Run a trained multitask FinBERT checkpoint and export article-level scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.finbert_dataset.select_active_learning import predict


HEADS = ("current", "future")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", default="2026-05-14")
    parser.add_argument("--end-date", default="2026-08-14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to max_length stored in the checkpoint training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype={"target_stock": "string", "original_id": "string"},
    )
    timestamps = pd.to_datetime(frame["datetime"], errors="raise")
    start = pd.Timestamp(args.start_date)
    end_exclusive = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    frame = frame.loc[(timestamps >= start) & (timestamps < end_exclusive)].copy()

    training_config = json.loads(
        (args.checkpoint / "training_config.json").read_text(encoding="utf-8")
    )
    max_length = args.max_length or int(training_config["max_length"])
    scored = predict(frame, args.checkpoint, args.batch_size, max_length)
    for head in HEADS:
        probability_columns = [
            f"model_{head}_p_{label}" for label in ("negative", "neutral", "positive")
        ]
        probability_sum = scored[probability_columns].sum(axis=1)
        scored[probability_columns] = scored[probability_columns].div(probability_sum, axis=0)
        positive = scored[f"model_{head}_p_positive"]
        negative = scored[f"model_{head}_p_negative"]
        scored[f"{head}_sentiment_score"] = positive.pow(2) - negative.pow(2)

    probability_columns = [
        f"model_{head}_p_{label}"
        for head in HEADS
        for label in ("negative", "neutral", "positive")
    ]
    result_columns = [
        column
        for column in (
            "item_id", "source", "target_stock", "datetime", "text",
            "original_index", "original_id", "headline", "body",
            "normalized_text_hash", "duplicate_group", "month", "text_length",
        )
        if column in scored.columns
    ]
    result_columns += probability_columns
    result_columns += [
        "model_current_prediction", "current_sentiment_score",
        "model_future_prediction", "future_sentiment_score",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored[result_columns].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved_rows={len(scored)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
