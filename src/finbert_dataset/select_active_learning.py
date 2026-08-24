"""학습된 멀티태스크 모델로 추가 라벨링할 Development 표본을 선정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.train_finbert_multitask import LABELS, MultiTaskFinBert
from .config import ANALYSIS_START, DEVELOPMENT_START, OUTPUT_DIR, SEED
from .utils import load_unified_data, split_time_pools


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
            padding=False, return_tensors=None,
        )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-9, 1.0)
    return -(values * np.log(values)).sum(axis=1) / np.log(values.shape[1])


def predict(frame: pd.DataFrame, checkpoint: Path, batch_size: int, max_length: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Active Learning 추론에 CUDA GPU를 찾지 못했습니다.")
    tokenizer_path = checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config = json.loads((checkpoint / "training_config.json").read_text(encoding="utf-8"))
    model = MultiTaskFinBert(config["model_name"])
    state = torch.load(checkpoint / "multitask_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = TextDataset(frame["text"].astype(str).tolist(), tokenizer, max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda rows: tokenizer.pad(rows, padding=True, return_tensors="pt"),
        pin_memory=True, num_workers=0,
    )
    current_parts, future_parts = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Development pool inference"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current_logits, future_logits = model(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
                )
            current_parts.append(current_logits.softmax(-1).float().cpu().numpy())
            future_parts.append(future_logits.softmax(-1).float().cpu().numpy())

    current = np.concatenate(current_parts)
    future = np.concatenate(future_parts)
    result = frame.copy().reset_index(drop=True)
    for index, label in enumerate(LABELS):
        result[f"model_current_p_{label}"] = current[:, index]
        result[f"model_future_p_{label}"] = future[:, index]
    result["model_current_prediction"] = np.asarray(LABELS)[current.argmax(axis=1)]
    result["model_future_prediction"] = np.asarray(LABELS)[future.argmax(axis=1)]
    result["current_entropy"] = entropy(current)
    result["future_entropy"] = entropy(future)
    result["uncertainty_score"] = (result["current_entropy"] + result["future_entropy"]) / 2
    return result


def _round_robin(frame: pd.DataFrame, count: int, score: str) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0).copy()
    ranked = frame.sort_values(score, ascending=False).copy()
    ranked["month"] = pd.to_datetime(ranked["datetime"]).dt.strftime("%Y-%m")
    ranked["rank_in_stratum"] = ranked.groupby(
        ["target_stock", "month"], dropna=False
    ).cumcount()
    return ranked.sort_values(
        ["rank_in_stratum", score], ascending=[True, False]
    ).head(count).copy()


def balanced_take(pool: pd.DataFrame, count: int, score: str, used: set[str], group: str) -> pd.DataFrame:
    candidates = pool[~pool["item_id"].astype(str).isin(used)].copy()
    news_target = min(round(count * 0.30), int(candidates["source"].eq("edaily_news").sum()))
    news = _round_robin(candidates[candidates["source"].eq("edaily_news")], news_target, score)
    toss_target = min(count - len(news), int(candidates["source"].eq("toss_community").sum()))
    toss = _round_robin(candidates[candidates["source"].eq("toss_community")], toss_target, score)
    selected = pd.concat([news, toss], ignore_index=True)
    if len(selected) < count:
        selected_ids = set(selected["item_id"].astype(str))
        fill = _round_robin(
            candidates[~candidates["item_id"].astype(str).isin(selected_ids)],
            count - len(selected), score,
        )
        selected = pd.concat([selected, fill], ignore_index=True)
    selected["selection_group"] = group
    selected["selection_score"] = selected[score]
    used.update(selected["item_id"].astype(str))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finbert/model_a/best"))
    parser.add_argument("--labeled", type=Path, default=OUTPUT_DIR / f"labeled_combined_seed{SEED}.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / f"active_learning_2000_seed{SEED}.csv")
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--future-negative", type=int, default=1000)
    parser.add_argument("--uncertain", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.future_negative + args.uncertain > args.total:
        raise ValueError("future-negative + uncertain은 total보다 클 수 없습니다.")

    development, _, _ = split_time_pools(load_unified_data())
    labeled = pd.read_csv(args.labeled, encoding="utf-8-sig", dtype={"original_id": str})
    labeled_items = set(labeled["item_id"].astype(str))
    labeled_hashes = set(labeled["normalized_text_hash"].astype(str))
    labeled_originals = set(labeled["source"].astype(str) + ":" + labeled["original_id"].astype(str))
    original_keys = development["source"].astype(str) + ":" + development["original_id"].astype(str)
    eligible = development[
        ~development["item_id"].astype(str).isin(labeled_items)
        & ~development["normalized_text_hash"].astype(str).isin(labeled_hashes)
        & ~original_keys.isin(labeled_originals)
    ].copy()
    eligible = eligible.sort_values("item_id").drop_duplicates("normalized_text_hash", keep="first")
    predictions = predict(eligible, args.checkpoint, args.batch_size, args.max_length)

    used: set[str] = set()
    future_negative = balanced_take(
        predictions, args.future_negative, "model_future_p_negative", used, "future_negative_priority"
    )
    uncertain = balanced_take(predictions, args.uncertain, "uncertainty_score", used, "high_uncertainty")
    diversity_count = args.total - len(future_negative) - len(uncertain)
    predictions["diversity_score"] = 0.5 * predictions["uncertainty_score"] + 0.5 * (
        1.0 - predictions[[f"model_future_p_{label}" for label in LABELS]].max(axis=1)
    )
    diversity = balanced_take(predictions, diversity_count, "diversity_score", used, "diversity_fill")
    selected = pd.concat([future_negative, uncertain, diversity], ignore_index=True)
    selected.insert(0, "labeling_id", [f"active42_{index:04d}" for index in range(1, len(selected) + 1)])
    selected["current_label"] = ""
    selected["future_label"] = ""
    selected["review_status"] = "unlabeled"
    selected["label_reason"] = ""
    selected = selected.sort_values("labeling_id")

    if not pd.to_datetime(selected["datetime"]).between(DEVELOPMENT_START, ANALYSIS_START, inclusive="left").all():
        raise RuntimeError("Development 기간 밖 표본이 선택됐습니다.")
    if set(selected["normalized_text_hash"].astype(str)) & labeled_hashes:
        raise RuntimeError("기존 라벨 데이터와 normalized text 중복이 있습니다.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = {
        "seed": SEED, "eligible_rows": len(eligible), "selected_rows": len(selected),
        "eligible_source": eligible["source"].value_counts().to_dict(),
        "selection_group": selected["selection_group"].value_counts().to_dict(),
        "source": selected["source"].value_counts().to_dict(),
        "target_stock": selected["target_stock"].astype(str).value_counts().to_dict(),
        "model_future_prediction": selected["model_future_prediction"].value_counts().to_dict(),
        "development_start": str(DEVELOPMENT_START), "analysis_start_exclusive": str(ANALYSIS_START),
        "overlap_with_labeled_item_id": len(set(selected["item_id"].astype(str)) & labeled_items),
        "overlap_with_labeled_text_hash": len(set(selected["normalized_text_hash"].astype(str)) & labeled_hashes),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {args.output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
