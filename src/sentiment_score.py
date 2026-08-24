"""NLP 감성분석: 사전학습 금융 감성분석 모델로 뉴스/게시글에 긍/부정/중립 점수를 매긴다.

킥오프 가이드의 "사전학습 모델 그대로 사용" 방침에 따라 파인튜닝 없이 그대로 추론만 한다.
모델: snunlp/KR-FinBert-SC (서울대 NLP연구실이 만든 한국어 금융 뉴스 3-class 감성분류기,
      긍정/부정/중립). Hugging Face에서 최초 실행 시 자동 다운로드.

종목별 수집 기간이 일치하는 소스만 합쳐서 스코어링한다. 삼성전자·SK하이닉스는
이데일리·토스·팍스넷을 사용하고, 카카오·에코프로비엠은 이데일리·토스를 사용한다.
네이버 게시판은 통합 지수에서 제외하고 종목별 별도 점수 파일로 저장하며,
과거 수집이 불완전한 네이버 뉴스는 모든 종목에서 제외한다.
스코어링 결과는 실제 거래일을 기준으로 장중과 장외로 나눠 집계한다.
장중(09:00 이상 15:30 미만)은 같은 거래일에, 장외(15:30 이후 및 다음 거래일 09:00 이전)는
다음 거래일에 귀속한다. 주말과 휴장일 게시물도 다음 거래일 장외 지수에 포함한다.

게시물이 없는 구간은 sentiment_index를 0으로 채우되 no_posts=1을 함께 기록한다. 따라서
실제 중립 감성(sentiment_index=0, no_posts=0)과 정보 부재를 구분할 수 있다.

이 스크립트는 Track A(크롤링)에서 Track B(감성분석/지수화, 메인 설승곤)로 넘어가는
경계의 1차 베이스라인입니다 — 본격적인 모델 비교·검증은 설승곤 담당 파트에서 이어가면 됩니다.
"""
import argparse
from bisect import bisect_right

import pandas as pd

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    SIX_MONTH_PROCESSED_DIR,
    SIX_MONTH_RAW_DIR,
    STOCKS,
    STOCK_ENGLISH_NAMES,
    STOCK_FILE_NAMES,
)

INPUT_PROCESSED_DIR = PROCESSED_DIR
OUTPUT_PROCESSED_DIR = PROCESSED_DIR
PRICE_RAW_DIR = RAW_DIR

MODEL_NAME = "snunlp/KR-FinBert-SC"
LABELS = ["negative", "neutral", "positive"]  # 모델 config의 라벨 순서

# 감성점수에 실제로 포함할 소스를 종목별로 명시한다.
# 네이버 게시판은 다른 소스와 합치지 않고 별도 sentiment_scored_board 파일로 관리한다.
# 네이버 뉴스(news_*.csv)는 과거 수집이 불완전하므로 어떤 종목에도 넣지 않는다.
SENTIMENT_SOURCES_BY_CODE = {
    "005930": ("edaily_news", "toss_community", "paxnet_board"),
    "000660": ("edaily_news", "toss_community", "paxnet_board"),
    "247540": ("edaily_news", "toss_community"),
    "035720": ("edaily_news", "toss_community"),
}


class SentimentScorer:
    def __init__(self, model_name: str = MODEL_NAME):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_batch(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        # 길이가 비슷한 문장을 같은 배치로 묶으면 짧은 게시판 제목이 긴 뉴스 본문 길이까지
        # padding되는 낭비를 줄일 수 있다. 결과는 원래 입력 순서로 되돌려 저장한다.
        indexed_texts = sorted(enumerate(texts), key=lambda item: len(item[1]) if isinstance(item[1], str) else 0)
        rows: list[dict | None] = [None] * len(texts)
        total_batches = (len(indexed_texts) + batch_size - 1) // batch_size
        with self.torch.no_grad():
            for batch_number, i in enumerate(range(0, len(indexed_texts), batch_size), start=1):
                indexed_batch = indexed_texts[i : i + batch_size]
                batch = [t if isinstance(t, str) and t.strip() else "." for _, t in indexed_batch]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                logits = self.model(**enc).logits
                probs = self.torch.softmax(logits, dim=-1).numpy()
                for (original_index, _), p in zip(indexed_batch, probs):
                    row = dict(zip(LABELS, p.tolist()))
                    row["sentiment_score"] = row["positive"] - row["negative"]  # -1(부정) ~ +1(긍정)
                    rows[original_index] = row
                if batch_number % 100 == 0 or batch_number == total_batches:
                    print(f"  감성 추론 {batch_number}/{total_batches} 배치 완료")
        return pd.DataFrame(rows)


def _load_source(code: str, filename: str, id_col: str, text_cols: list[str], source: str) -> pd.DataFrame:
    path = INPUT_PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "datetime", "text", "source"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    text = df[text_cols[0]].fillna("")
    for c in text_cols[1:]:
        text = text + ". " + df[c].fillna("")
    return pd.DataFrame(
        {
            "item_id": source + "_" + df[id_col].astype(str),
            "datetime": pd.to_datetime(df["datetime"], format="mixed"),
            "text": text.str.slice(0, 500),
            "source": source,
        }
    )


def collect_all_text(code: str) -> pd.DataFrame:
    file_name = STOCK_FILE_NAMES[code]
    loaders = {
        "edaily_news": lambda: _load_source(
            code,
            f"edaily_news_{file_name}_clean.csv",
            "news_id",
            ["clean_headline", "clean_body"],
            "edaily_news",
        ),
        "naver_board": lambda: _load_source(
            code, f"board_{file_name}_clean.csv", "nid", ["clean_title"], "naver_board"
        ),
        "toss_community": lambda: _load_source(
            code,
            f"toss_community_{file_name}_clean.csv",
            "comment_id",
            ["clean_message"],
            "toss_community",
        ),
        "paxnet_board": lambda: _load_source(
            code, f"paxnet_board_{file_name}_clean.csv", "seq", ["clean_title"], "paxnet_board"
        ),
    }
    frames = [loaders[source]() for source in SENTIMENT_SOURCES_BY_CODE[code]]

    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.strip().str.len() > 0]
    return df.drop_duplicates(subset="item_id").sort_values("datetime")


def score_all_sources_for_stock(scorer: SentimentScorer, code: str) -> pd.DataFrame:
    combined = collect_all_text(code)
    scores = scorer.score_batch(combined["text"].tolist())
    out = pd.concat([combined.reset_index(drop=True), scores], axis=1)
    out_path = OUTPUT_PROCESSED_DIR / f"sentiment_scored_{STOCK_FILE_NAMES[code]}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def score_board_for_stock(scorer: SentimentScorer, code: str) -> pd.DataFrame:
    """네이버 게시판만 별도로 점수화한다. 통합 장중·장외 지수에는 합치지 않는다."""
    file_name = STOCK_FILE_NAMES[code]
    board = _load_source(
        code, f"board_{file_name}_clean.csv", "nid", ["clean_title"], "naver_board"
    )
    board = board[board["text"].str.strip().str.len() > 0].drop_duplicates(subset="item_id")
    scores = scorer.score_batch(board["text"].tolist())
    out = pd.concat([board.reset_index(drop=True), scores], axis=1)
    out_path = OUTPUT_PROCESSED_DIR / f"sentiment_scored_board_{file_name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def _load_trading_dates(
    code: str, timestamps: pd.Series
) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    file_name = STOCK_FILE_NAMES[code]
    price_path = PRICE_RAW_DIR / f"price_daily_{file_name}.csv"
    price = pd.read_csv(price_path, encoding="utf-8-sig", usecols=["date"])
    price_dates = sorted(pd.to_datetime(price["date"], errors="raise").dt.normalize().unique().tolist())
    price_date_set = set(price_dates)

    # 가격 파일은 당일 장 마감 전에 실행하면 전 거래일까지밖에 없을 수 있다. 마지막 종가 이후
    # 게시물이 있으면 그 감성을 버리지 않고 다음 평일을 잠정 거래일로 한 칸 추가한다.
    # 추후 가격 파일이 갱신되면 실제 거래일 달력으로 다시 집계된다.
    if price_dates and (timestamps >= price_dates[-1] + pd.Timedelta(hours=15, minutes=30)).any():
        provisional_date = price_dates[-1] + pd.offsets.BDay(1)
        if provisional_date not in price_date_set:
            price_dates.append(provisional_date)

    return price_dates, price_date_set


def _assign_session(timestamp: pd.Timestamp, trading_dates: list[pd.Timestamp], trading_set: set[pd.Timestamp]):
    """게시 시각을 (영향을 받을 거래일, 장중/장외)로 변환한다."""
    date = timestamp.normalize()
    minute = timestamp.hour * 60 + timestamp.minute

    if date in trading_set and 9 * 60 <= minute < 15 * 60 + 30:
        return date, "intraday"
    if date in trading_set and minute < 9 * 60:
        return date, "overnight"

    next_index = bisect_right(trading_dates, date)
    if next_index < len(trading_dates):
        return trading_dates[next_index], "overnight"
    return pd.NaT, "overnight"


def build_session_index(code: str, scored: pd.DataFrame) -> pd.DataFrame:
    """장중은 당일, 장외는 다음 거래일에 귀속해 거래일별 감성지수를 만든다."""
    scored = scored.copy()
    scored["datetime"] = pd.to_datetime(scored["datetime"], errors="coerce")
    scored = scored.dropna(subset=["datetime", "sentiment_score"])
    trading_dates, price_date_set = _load_trading_dates(code, scored["datetime"])
    trading_set = set(trading_dates)

    assignments = scored["datetime"].map(lambda value: _assign_session(value, trading_dates, trading_set))
    scored[["session_date", "session"]] = pd.DataFrame(assignments.tolist(), index=scored.index)
    unassigned_count = int(scored["session_date"].isna().sum())
    scored = scored.dropna(subset=["session_date"])

    grouped = scored.groupby(["session_date", "session"])["sentiment_score"].agg(["mean", "size"])
    result = pd.DataFrame({"date": trading_dates})

    for session in ("intraday", "overnight"):
        if session in grouped.index.get_level_values("session"):
            session_values = grouped.xs(session, level="session")
        else:
            session_values = pd.DataFrame(columns=["mean", "size"])
        result[f"{session}_sentiment_index"] = result["date"].map(session_values["mean"]).fillna(0.0)
        result[f"{session}_item_count"] = result["date"].map(session_values["size"]).fillna(0).astype(int)
        result[f"{session}_no_posts"] = (result[f"{session}_item_count"] == 0).astype(int)

    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["price_available"] = [int(pd.Timestamp(date) in price_date_set) for date in result["date"]]
    result["code"] = STOCK_ENGLISH_NAMES[code]
    result.attrs["unassigned_item_count"] = unassigned_count

    out_path = OUTPUT_PROCESSED_DIR / f"sentiment_index_daily_{STOCK_FILE_NAMES[code]}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def rebuild_session_indexes_from_scored() -> None:
    for code, name in STOCKS.items():
        path = OUTPUT_PROCESSED_DIR / f"sentiment_scored_{STOCK_FILE_NAMES[code]}.csv"
        scored = pd.read_csv(path, encoding="utf-8-sig")
        result = build_session_index(code, scored)
        print(
            f"[session index] {code} {name}: {len(result)}거래일, "
            f"마지막 가격일 이후 미귀속 {result.attrs['unassigned_item_count']}건"
        )


def main():
    global INPUT_PROCESSED_DIR, OUTPUT_PROCESSED_DIR, PRICE_RAW_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--six-month",
        action="store_true",
        help="data/6개월의 전처리·가격 파일을 사용하고 결과도 그 아래에 저장",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="기존 sentiment_scored 파일을 사용해 장중/장외 일별 지수만 다시 생성",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        choices=STOCKS.keys(),
        help="지정한 종목코드만 스코어링 (생략하면 전체 종목)",
    )
    parser.add_argument(
        "--board-only",
        action="store_true",
        help="네이버 게시판만 별도 점수 파일로 생성하고 통합 지수에는 반영하지 않음",
    )
    args = parser.parse_args()

    if args.six_month:
        INPUT_PROCESSED_DIR = SIX_MONTH_PROCESSED_DIR
        OUTPUT_PROCESSED_DIR = SIX_MONTH_PROCESSED_DIR
        PRICE_RAW_DIR = SIX_MONTH_RAW_DIR

    if args.aggregate_only:
        rebuild_session_indexes_from_scored()
        return

    print(f"모델 로딩 중: {MODEL_NAME} (최초 실행 시 다운로드, 몇 분 걸릴 수 있음)")
    scorer = SentimentScorer()

    selected_codes = args.codes or list(STOCKS)
    for code in selected_codes:
        name = STOCKS[code]
        if args.board_only:
            scored = score_board_for_stock(scorer, code)
            print(f"[sentiment:board] {code} {name}: 네이버 게시판 {len(scored)}건 별도 저장")
            continue
        scored = score_all_sources_for_stock(scorer, code)
        daily = build_session_index(code, scored)
        no_intraday = int(daily["intraday_no_posts"].sum())
        no_overnight = int(daily["overnight_no_posts"].sum())
        by_source = scored["source"].value_counts().to_dict()
        print(
            f"[sentiment] {code} {name}: {len(scored)}건 스코어링 {by_source} -> "
            f"{len(daily)}거래일, 장중 무게시 {no_intraday}일, 장외 무게시 {no_overnight}일"
        )

    print("\n게시물이 없는 구간은 sentiment_index=0, no_posts=1로 기록했습니다.")


if __name__ == "__main__":
    main()
