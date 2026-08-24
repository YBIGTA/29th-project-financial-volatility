"""Development 기간 Toss 원문으로 KR-FinBERT MLM 도메인 적응을 수행한다."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from src.finbert_dataset.config import ANALYSIS_START, DEVELOPMENT_START, SEED
from src.finbert_dataset.utils import load_unified_data, split_time_pools


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length,
            padding=False, return_special_tokens_mask=True,
        )


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, rows = 0.0, 0
    for batch in tqdm(loader, desc="MLM validation", leave=False):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        batch_rows = len(batch["input_ids"])
        total += float(loss.item()) * batch_rows
        rows += batch_rows
    return total / max(rows, 1)


def build_corpus(split_dir: Path) -> pd.DataFrame:
    development, _, _ = split_time_pools(load_unified_data())
    development = development[development["source"].eq("toss_community")].copy()
    held_out = pd.concat(
        [
            pd.read_csv(split_dir / "validation_seed42.csv", encoding="utf-8-sig"),
            pd.read_csv(split_dir / "test_seed42.csv", encoding="utf-8-sig"),
        ],
        ignore_index=True,
    )
    held_items = set(held_out["item_id"].astype(str))
    held_hashes = set(held_out["normalized_text_hash"].astype(str))
    corpus = development[
        ~development["item_id"].astype(str).isin(held_items)
        & ~development["normalized_text_hash"].astype(str).isin(held_hashes)
    ].copy()
    corpus = corpus[corpus["text"].fillna("").astype(str).str.strip().ne("")]
    corpus = corpus.sort_values("item_id").drop_duplicates("normalized_text_hash", keep="first")
    dates = pd.to_datetime(corpus["datetime"], errors="coerce")
    if not dates.between(DEVELOPMENT_START, ANALYSIS_START, inclusive="left").all():
        raise RuntimeError("MLM corpus에 Development 기간 밖 데이터가 있습니다.")
    if set(corpus["normalized_text_hash"].astype(str)) & held_hashes:
        raise RuntimeError("MLM corpus와 Validation/Test text가 겹칩니다.")
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="snunlp/KR-FinBert-SC")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/finbert_active_round2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finbert/model_b/mlm"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")
    seed_everything(args.seed)
    device = torch.device("cuda")
    corpus = build_corpus(args.split_dir)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(corpus))
    validation_size = max(1, round(len(corpus) * 0.05))
    validation = corpus.iloc[order[:validation_size]].copy()
    train = corpus.iloc[order[validation_size:]].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus[["item_id", "datetime", "text", "normalized_text_hash"]].to_csv(
        args.output_dir / "mlm_corpus.csv", index=False, encoding="utf-8-sig"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability, seed=args.seed
    )
    loaders = {
        name: DataLoader(
            TextDataset(frame["text"].astype(str).tolist(), tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=name == "train", collate_fn=collator,
            num_workers=0, pin_memory=True,
        )
        for name, frame in (("train", train), ("validation", validation))
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loaders["train"], desc=f"MLM epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        validation_loss = evaluate(model, loaders["validation"], device)
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        checkpoint = args.output_dir / "checkpoints" / f"epoch_{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save_pretrained(args.output_dir / "best")
            tokenizer.save_pretrained(args.output_dir / "best")
        print(f"epoch={epoch} mlm_validation_loss={validation_loss:.4f} perplexity={math.exp(min(validation_loss, 20)):.2f}")

    report = {
        "model_name": args.model_name, "seed": args.seed, "corpus_rows": len(corpus),
        "train_rows": len(train), "validation_rows": len(validation),
        "held_out_supervised_validation_test": True,
        "analysis_period_used": False, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
        "mlm_probability": args.mlm_probability, "best_validation_loss": best_loss,
        "history": history, "device": str(device), "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
