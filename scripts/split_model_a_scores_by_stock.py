"""Split the unified Model A inference result into stock-specific CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STOCK_FILES = {
    "005930": "samsung_electronics",
    "000660": "sk_hynix",
    "035720": "kakao",
    "247540": "ecopro_bm",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype={"target_stock": "string", "original_id": "string"},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = 0
    for stock_code, file_name in STOCK_FILES.items():
        stock_frame = frame.loc[frame["target_stock"] == stock_code].copy()
        stock_frame = stock_frame.sort_values(["datetime", "item_id"])
        output = args.output_dir / f"sentiment_scored_{file_name}.csv"
        stock_frame.to_csv(output, index=False, encoding="utf-8-sig")
        saved_rows += len(stock_frame)
        print(f"{stock_code} {file_name}: {len(stock_frame)} rows -> {output}")

    if saved_rows != len(frame):
        unknown = sorted(set(frame["target_stock"].dropna()) - set(STOCK_FILES))
        raise RuntimeError(
            f"Row-count mismatch: input={len(frame)}, saved={saved_rows}, unknown={unknown}"
        )


if __name__ == "__main__":
    main()
