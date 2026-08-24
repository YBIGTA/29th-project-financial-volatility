"""CSV 로딩, 텍스트 정규화, 중복 그룹화와 재현성 유틸리티."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import ANALYSIS_END_EXCLUSIVE, ANALYSIS_START, DEVELOPMENT_START, INPUT_DIR, SEED, STOCKS


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

CORE_COLUMNS = [
    "item_id",
    "source",
    "target_stock",
    "datetime",
    "text",
    "original_index",
    "original_id",
    "headline",
    "body",
    "normalized_text_hash",
    "duplicate_group",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(value: object) -> str:
    """원문을 바꾸지 않고 중복 판정용 문자열만 생성한다."""
    if not isinstance(value, str):
        return ""
    normalized = URL_RE.sub(" ", value).strip().lower()
    return WHITESPACE_RE.sub(" ", normalized)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _news_rows(code: str, label: str) -> pd.DataFrame:
    path = INPUT_DIR / f"edaily_news_{label}_clean.csv"
    df = _read_csv(path).reset_index(names="original_index")
    required = {"news_id", "datetime", "headline", "body", "clean_headline", "clean_body"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 누락: {sorted(missing)}")

    headline = df["clean_headline"].fillna("").astype(str).str.strip()
    body = df["clean_body"].fillna("").astype(str).str.strip()
    text = headline.where(body.eq(""), headline + ". " + body)
    original_id = df["news_id"].astype(str)
    result = pd.DataFrame(
        {
            "item_id": "edaily_news:" + original_id + ":" + code,
            "source": "edaily_news",
            "target_stock": code,
            "datetime": pd.to_datetime(df["datetime"], format="mixed", errors="coerce"),
            "text": text,
            "original_index": df["original_index"].astype(int),
            "original_id": original_id,
            "headline": df["headline"].fillna("").astype(str),
            "body": df["body"].fillna("").astype(str),
        }
    )
    return result


def _toss_rows(code: str, label: str) -> pd.DataFrame:
    path = INPUT_DIR / f"toss_community_{label}_clean.csv"
    df = _read_csv(path).reset_index(names="original_index")
    required = {"comment_id", "code", "datetime", "message", "clean_message"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 누락: {sorted(missing)}")

    original_id = df["comment_id"].astype(str)
    result = pd.DataFrame(
        {
            "item_id": "toss_community:" + original_id,
            "source": "toss_community",
            "target_stock": df["code"].astype(str).str.zfill(6),
            "datetime": pd.to_datetime(df["datetime"], format="mixed", errors="coerce"),
            "text": df["clean_message"].fillna("").astype(str),
            "original_index": df["original_index"].astype(int),
            "original_id": original_id,
            "headline": "",
            "body": df["message"].fillna("").astype(str),
        }
    )
    invalid_codes = sorted(set(result["target_stock"]) - set(STOCKS))
    if invalid_codes:
        raise ValueError(f"{path.name}에 예상하지 못한 종목코드가 있습니다: {invalid_codes}")
    return result


def load_unified_data() -> pd.DataFrame:
    """4개 종목의 이데일리 뉴스와 토스 정제 데이터를 공통 schema로 읽는다."""
    seed_everything()
    frames = []
    for code, label in STOCKS.items():
        frames.append(_news_rows(code, label))
        frames.append(_toss_rows(code, label))

    data = pd.concat(frames, ignore_index=True)
    data["text"] = data["text"].fillna("").astype(str)
    normalized = data["text"].map(normalize_text)
    data["normalized_text_hash"] = normalized.map(stable_hash)
    data["duplicate_group"] = data["source"] + ":text:" + data["normalized_text_hash"]
    data["month"] = data["datetime"].dt.to_period("M").astype("string")
    data["text_length"] = data["text"].str.len()
    return data


def split_time_pools(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """개발·최종분석 기간을 half-open interval로 완전히 분리한다."""
    datetime_values = pd.to_datetime(data["datetime"], format="mixed", errors="coerce")
    development_mask = datetime_values.ge(DEVELOPMENT_START) & datetime_values.lt(ANALYSIS_START)
    analysis_mask = datetime_values.ge(ANALYSIS_START) & datetime_values.lt(ANALYSIS_END_EXCLUSIVE)
    development = data.loc[development_mask].copy()
    analysis = data.loc[analysis_mask].copy()
    outside = data.loc[~(development_mask | analysis_mask)].copy()

    if not development.empty and development["datetime"].max() >= ANALYSIS_START:
        raise AssertionError("development_pool에 2026-05-14 이후 데이터가 포함됐습니다.")
    if not analysis.empty and analysis["datetime"].min() < ANALYSIS_START:
        raise AssertionError("analysis_pool에 2026-05-14 이전 데이터가 포함됐습니다.")
    if set(development.index) & set(analysis.index):
        raise AssertionError("development_pool과 analysis_pool 행 인덱스가 겹칩니다.")
    return development, analysis, outside


def pairwise_intersections(values: dict[str, Iterable[str]]) -> dict[str, int]:
    names = list(values)
    result: dict[str, int] = {}
    sets = {name: set(values[name]) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result[f"{left}__{right}"] = len(sets[left] & sets[right])
    return result
