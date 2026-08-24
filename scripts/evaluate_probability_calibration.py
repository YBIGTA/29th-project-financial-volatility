"""Evaluate multitask classification probability calibration on a labeled CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.finbert_dataset.select_active_learning import predict
from src.train_finbert_multitask import LABELS


def metrics(probabilities: np.ndarray, truth: np.ndarray, bins: int = 10) -> dict:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == truth
    one_hot = np.eye(len(LABELS))[truth]
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    nll = -np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean()

    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "rows": len(truth),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "ece_10_bins": float(ece),
        "multiclass_brier": float(brier),
        "negative_log_loss": float(nll),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"target_stock": "string"})
    scored = predict(frame, args.checkpoint, args.batch_size, args.max_length)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    output = {}
    for head in ("current", "future"):
        probabilities = scored[
            [f"model_{head}_p_{label}" for label in LABELS]
        ].to_numpy(dtype=float)
        truth = frame[f"{head}_label"].map(label_to_id).to_numpy(dtype=int)
        output[head] = metrics(probabilities, truth)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
