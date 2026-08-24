"""데이터셋 준비 파이프라인의 공통 설정."""

from datetime import datetime
from pathlib import Path


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "6개월" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "finbert"

NEWS_SAMPLE_SIZE = 200
TOSS_SAMPLE_SIZE = 300
ADDITIONAL_RANDOM_SIZE = 1000
ADDITIONAL_FUTURE_NEGATIVE_SIZE = 500
ADDITIONAL_TOSS_SIZE = 700
ADDITIONAL_EDAILY_SIZE = 300

DEVELOPMENT_START = datetime(2026, 2, 14)
ANALYSIS_START = datetime(2026, 5, 14)
ANALYSIS_END_EXCLUSIVE = datetime(2026, 8, 15)

ANALYSIS_POOL_FILE = OUTPUT_DIR / "analysis_pool_20260514_20260814.csv"
POOL_REPORT_FILE = OUTPUT_DIR / f"time_pool_report_seed{SEED}.json"
ADDITIONAL_SAMPLE_FILE = OUTPUT_DIR / f"additional_labeling_sample_1500_seed{SEED}.csv"
ADDITIONAL_SAMPLE_REPORT_FILE = OUTPUT_DIR / f"additional_labeling_sample_1500_seed{SEED}_summary.json"

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

VALID_LABELS = {"positive", "negative", "neutral"}

STOCKS = {
    "005930": "samsung_electronics",
    "000660": "sk_hynix",
    "247540": "ecopro_bm",
    "035720": "kakao",
}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
