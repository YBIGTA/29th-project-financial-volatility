"""프로젝트 공통 설정: 대상 종목, 기간, 저장 경로."""
from datetime import datetime, timedelta
from pathlib import Path

STOCKS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "247540": "에코프로비엠",
    "035720": "카카오",
}

INDEXES = {
    "KS11": "KOSPI",
    "KQ11": "KOSDAQ",
    "VIX": "VIX",
}

STOCK_FILE_NAMES = {
    "005930": "samsung_electronics",
    "000660": "sk_hynix",
    "247540": "ecopro_bm",
    "035720": "kakao",
}

STOCK_ENGLISH_NAMES = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "247540": "EcoPro BM",
    "035720": "Kakao",
}

# FinanceDataReader 조회 코드는 유지하되, 저장 파일에는 읽기 쉬운 영문 이름을 사용합니다.
PRICE_FILE_NAMES = {
    **STOCK_FILE_NAMES,
    "KS11": "kospi",
    "KQ11": "kosdaq",
    "VIX": "vix",
}

END_DATE = datetime.today().date()
START_DATE = END_DATE - timedelta(days=180)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SIX_MONTH_DIR = ROOT_DIR / "data" / "6개월"
SIX_MONTH_RAW_DIR = SIX_MONTH_DIR / "raw"
SIX_MONTH_PROCESSED_DIR = SIX_MONTH_DIR / "processed"
SIX_MONTH_SUMMARY_DIR = SIX_MONTH_DIR / "정리"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SIX_MONTH_RAW_DIR.mkdir(parents=True, exist_ok=True)
SIX_MONTH_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SIX_MONTH_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_DELAY_SEC = 0.5

# 네이버 최신분 증분 수집에서 한 번에 확인할 안전 상한.
# 게시판 3개월 백필(run_board_backfill.py)은 이 값을 쓰지 않고 실제 다음 링크를 따라간다.
NAVER_MAX_PAGE = 100
