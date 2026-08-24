"""소스·월·종목·길이 분포를 고려해 재현 가능한 500건 라벨링 표본을 만든다."""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import pandas as pd

from .config import NEWS_SAMPLE_SIZE, OUTPUT_DIR, SEED, TOSS_SAMPLE_SIZE, ensure_output_dir
from .utils import load_unified_data, split_time_pools


OUTPUT_COLUMNS = [
    "item_id",
    "source",
    "target_stock",
    "datetime",
    "text",
    "current_label",
    "future_label",
    "original_index",
    "original_id",
    "headline",
    "body",
    "normalized_text_hash",
    "duplicate_group",
]


def _group_seed(key: tuple) -> int:
    digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()
    return (SEED + int(digest[:8], 16)) % (2**32)


def _add_length_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if len(result) < 3:
        result["length_bucket"] = "medium"
        return result
    ranked = result["text_length"].rank(method="first")
    result["length_bucket"] = pd.qcut(ranked, q=3, labels=["short", "medium", "long"])
    return result


def round_robin_sample(frame: pd.DataFrame, size: int, group_columns: list[str]) -> pd.DataFrame:
    """주어진 strata queue를 순환하며 고정 seed 표본을 선택한다."""
    if len(frame) < size:
        raise ValueError(f"표본 {size}건이 필요하지만 {len(frame)}건만 있습니다.")
    frame = _add_length_bucket(frame)
    queues = []
    for key, group in frame.groupby(group_columns, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        shuffled = group.sample(frac=1, random_state=_group_seed(tuple(key)))
        queues.append((tuple(map(str, key)), deque(shuffled.index.tolist())))

    queues.sort(key=lambda item: item[0])
    rng = np.random.default_rng(SEED)
    selected: list[int] = []
    while len(selected) < size:
        active = [(key, queue) for key, queue in queues if queue]
        if not active:
            break
        order = rng.permutation(len(active))
        for position in order:
            if len(selected) == size:
                break
            selected.append(active[int(position)][1].popleft())

    if len(selected) != size:
        raise RuntimeError(f"표본 추출 실패: 요청 {size}, 추출 {len(selected)}")
    return frame.loc[selected].copy()


def balanced_sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(frame) < size:
        raise ValueError(f"표본 {size}건이 필요하지만 중복 제거 후 {len(frame)}건만 있습니다.")

    stocks = sorted(frame["target_stock"].unique())
    base, remainder = divmod(size, len(stocks))
    parts = []
    for position, stock in enumerate(stocks):
        quota = base + int(position < remainder)
        stock_frame = frame[frame["target_stock"].eq(stock)]
        if len(stock_frame) < quota:
            raise ValueError(f"{stock}: 표본 {quota}건이 필요하지만 {len(stock_frame)}건만 있습니다.")
        parts.append(round_robin_sample(stock_frame, quota, ["month", "length_bucket"]))
    return pd.concat(parts, ignore_index=False)


def build_sample() -> pd.DataFrame:
    data = load_unified_data()
    data, _, _ = split_time_pools(data)
    # 같은 종목 안의 완전 중복만 제거한다. 동일 기사가 여러 종목과 연결된 경우에는
    # 대상 종목에 따라 라벨이 달라질 수 있으므로 각 target_stock 행을 유지한다.
    unique = (
        data.sort_values(["source", "normalized_text_hash", "target_stock", "item_id"])
        .drop_duplicates(["source", "target_stock", "normalized_text_hash"], keep="first")
        .copy()
    )
    news = balanced_sample(unique[unique["source"].eq("edaily_news")], NEWS_SAMPLE_SIZE)
    toss = balanced_sample(unique[unique["source"].eq("toss_community")], TOSS_SAMPLE_SIZE)
    sample = pd.concat([news, toss], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["current_label"] = ""
    sample["future_label"] = ""
    return sample[OUTPUT_COLUMNS]


def main() -> None:
    sample = build_sample()
    ensure_output_dir()
    output = OUTPUT_DIR / f"labeling_sample_seed{SEED}.csv"
    sample.to_csv(output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    print(f"표본 저장: {output}")
    print(f"전체={len(sample)}, 소스={sample['source'].value_counts().to_dict()}")
    print("종목별:", sample.groupby(["source", "target_stock"]).size().to_dict())
    print("월별:", sample.groupby(["source", sample["datetime"].dt.to_period("M")]).size().to_dict())
    print("여러 target_stock에 걸친 정규화 text 중복:", int(sample["normalized_text_hash"].duplicated().sum()))


if __name__ == "__main__":
    main()
