"""KR-FinBERT 기반 Current/Future 3-class 멀티태스크 fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


@dataclass
class TrainConfig:
    model_name: str
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    num_workers: int
    device: str
    gpu_name: str | None
    torch_version: str
    torch_cuda_version: str | None
    mixed_precision: str
    class_weighted_loss: bool
    current_class_weights: list[float]
    future_class_weights: list[float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SentimentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        tokens = self.tokenizer(
            str(row["text"]), truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )
        tokens["current_labels"] = LABEL_TO_ID[str(row["current_label"])]
        tokens["future_labels"] = LABEL_TO_ID[str(row["future_label"])]
        return tokens


class BatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        current = torch.tensor([row.pop("current_labels") for row in features], dtype=torch.long)
        future = torch.tensor([row.pop("future_labels") for row in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["current_labels"] = current
        batch["future_labels"] = future
        return batch


class MultiTaskFinBert(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.current_head = nn.Linear(hidden, len(LABELS))
        self.future_head = nn.Linear(hidden, len(LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = output.pooler_output if output.pooler_output is not None else output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.current_head(pooled), self.future_head(pooled)


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "current_label", "future_label"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path}: 필수 컬럼 누락 {sorted(missing)}")
    for column in ("current_label", "future_label"):
        invalid = sorted(set(frame[column].astype(str)) - set(LABELS))
        if invalid:
            raise ValueError(f"{path}: {column} 잘못된 라벨 {invalid}")
    return frame


def class_weights(frame: pd.DataFrame, column: str, device: torch.device) -> torch.Tensor:
    counts = frame[column].value_counts()
    values = [len(frame) / (len(LABELS) * max(int(counts.get(label, 0)), 1)) for label in LABELS]
    return torch.tensor(values, dtype=torch.float32, device=device)


def autocast_context(device: torch.device, precision: str):
    enabled = precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.inference_mode()
def evaluate(model, loader, device, precision: str) -> tuple[dict, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    true = {"current": [], "future": []}
    pred = {"current": [], "future": []}
    total_loss = 0.0
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, precision):
            current_logits, future_logits = model(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
            )
            loss = (loss_fn(current_logits, batch["current_labels"]) +
                    loss_fn(future_logits, batch["future_labels"])) / 2
        total_loss += float(loss.item()) * len(batch["input_ids"])
        for name, logits, labels in (
            ("current", current_logits, batch["current_labels"]),
            ("future", future_logits, batch["future_labels"]),
        ):
            true[name].extend(labels.cpu().tolist())
            pred[name].extend(logits.argmax(dim=-1).cpu().tolist())

    metrics = {"loss": total_loss / max(len(loader.dataset), 1)}
    for name in ("current", "future"):
        metrics[name] = {
            "accuracy": accuracy_score(true[name], pred[name]),
            "macro_f1": f1_score(true[name], pred[name], average="macro"),
            "classification_report": classification_report(
                true[name], pred[name], labels=range(len(LABELS)), target_names=LABELS,
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                true[name], pred[name], labels=range(len(LABELS))
            ).tolist(),
        }
    selection_score = (metrics["current"]["macro_f1"] + metrics["future"]["macro_f1"]) / 2
    metrics["mean_macro_f1"] = selection_score
    return metrics, selection_score


def save_checkpoint(model, tokenizer, output: Path, config: TrainConfig, metrics: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(output)
    torch.save(model.state_dict(), output / "multitask_model.pt")
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "training_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "label_mapping.json").write_text(
        json.dumps({"labels": LABELS, "label_to_id": LABEL_TO_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/finbert"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finbert/model_a"))
    parser.add_argument("--model-name", default="snunlp/KR-FinBert-SC")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 기본값 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="소량 데이터 1 epoch 코드 점검")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = "bf16" if bf16 else ("fp16" if device.type == "cuda" else "fp32")
    if not args.cpu and device.type != "cuda":
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. CPU 실행은 --cpu를 명시하세요.")

    epochs = 1 if args.smoke_test else args.epochs
    train = read_split(args.data_dir / "train_seed42.csv")
    validation = read_split(args.data_dir / "validation_seed42.csv")
    test = read_split(args.data_dir / "test_seed42.csv")
    if args.smoke_test:
        train, validation, test = train.head(32), validation.head(16), test.head(16)
    current_weights = class_weights(train, "current_label", device)
    future_weights = class_weights(train, "future_label", device)

    config = TrainConfig(
        model_name=args.model_name, seed=args.seed, epochs=epochs, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, max_length=args.max_length, num_workers=args.num_workers,
        device=str(device), gpu_name=gpu_name, torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda, mixed_precision=precision,
        class_weighted_loss=not args.no_class_weights,
        current_class_weights=current_weights.cpu().tolist(),
        future_class_weights=future_weights.cpu().tolist(),
    )
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = BatchCollator(tokenizer)
    loaders = {
        name: DataLoader(
            SentimentDataset(frame, tokenizer, args.max_length),
            batch_size=args.batch_size, shuffle=name == "train", collate_fn=collator,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        for name, frame in (("train", train), ("validation", validation), ("test", test))
    }
    model = MultiTaskFinBert(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    current_loss_fn = nn.CrossEntropyLoss(weight=None if args.no_class_weights else current_weights)
    future_loss_fn = nn.CrossEntropyLoss(weight=None if args.no_class_weights else future_weights)
    best_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}")
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast_context(device, precision):
                current_logits, future_logits = model(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
                )
                loss = (current_loss_fn(current_logits, batch["current_labels"]) +
                        future_loss_fn(future_logits, batch["future_labels"])) / 2
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item())
            if step % args.gradient_accumulation_steps == 0 or step == len(progress):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        validation_metrics, score = evaluate(model, loaders["validation"], device, precision)
        history.append({"epoch": epoch, "validation": validation_metrics})
        checkpoint = args.output_dir / "checkpoints" / f"epoch_{epoch}"
        save_checkpoint(model, tokenizer, checkpoint, config, validation_metrics)
        if score > best_score:
            best_score = score
            save_checkpoint(model, tokenizer, args.output_dir / "best", config, validation_metrics)
        print(f"epoch={epoch} validation_mean_macro_f1={score:.4f}")

    best_state = torch.load(args.output_dir / "best" / "multitask_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    test_metrics, _ = evaluate(model, loaders["test"], device, precision)
    final_metrics = {"best_validation_mean_macro_f1": best_score, "test": test_metrics, "history": history}
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
