"""NLP 감성분석: 사전학습 금융 감성분석 모델로 뉴스/게시글에 긍/부정/중립 점수를 매긴다.

킥오프 가이드의 "사전학습 모델 그대로 사용" 방침에 따라 파인튜닝 없이 그대로 추론만 한다.
모델: snunlp/KR-FinBert-SC (서울대 NLP연구실이 만든 한국어 금융 뉴스 3-class 감성분류기,
      긍정/부정/중립). Hugging Face에서 최초 실행 시 자동 다운로드.

카카오·에코프로비엠처럼 이데일리 뉴스만으로는 하루 기사 수가 너무 적어 날짜가 듬성듬성
빠지는 문제가 있어서, **한 소스에 의존하지 않고 지금까지 모은 모든 텍스트 소스**
(이데일리 뉴스, 네이버 뉴스, 네이버 게시판, 토스 커뮤니티, 팍스넷 게시판(삼성/SK하이닉스))를
합쳐서 스코어링한다. 그래도 기사/게시글이 하루도 없는 날은 실제로 "그 날 정보가 없다"는
뜻이므로, 3개월 전체 날짜로 채우되 그런 날은 sentiment_index를 NaN으로 남겨
(0으로 임의로 채우지 않음) Track B가 결측 처리 방식을 직접 고르게 한다.

이 스크립트는 Track A(크롤링)에서 Track B(감성분석/지수화, 메인 설승곤)로 넘어가는
경계의 1차 베이스라인입니다 — 본격적인 모델 비교·검증은 설승곤 담당 파트에서 이어가면 됩니다.
"""
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import END_DATE, PROCESSED_DIR, START_DATE, STOCKS

MODEL_NAME = "snunlp/KR-FinBert-SC"
LABELS = ["negative", "neutral", "positive"]  # 모델 config의 라벨 순서
PAXNET_VALID_CODES = ["005930", "000660"]


class SentimentScorer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        rows = []
        for i in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) and t.strip() else "." for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            for p in probs:
                row = dict(zip(LABELS, p.tolist()))
                row["sentiment_score"] = row["positive"] - row["negative"]  # -1(부정) ~ +1(긍정)
                rows.append(row)
        return pd.DataFrame(rows)


def _load_source(code: str, filename: str, id_col: str, text_cols: list[str], source: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "datetime", "text", "source"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    text = df[text_cols[0]].fillna("")
    for c in text_cols[1:]:
        text = text + ". " + df[c].fillna("")
    return pd.DataFrame(
        {
            "item_id": source + "_" + df[id_col].astype(str),
            "datetime": pd.to_datetime(df["datetime"]),
            "text": text.str.slice(0, 500),
            "source": source,
        }
    )


def collect_all_text(code: str) -> pd.DataFrame:
    frames = [
        _load_source(code, f"edaily_news_{code}_clean.csv", "news_id", ["clean_headline", "clean_body"], "edaily_news"),
        _load_source(code, f"news_{code}_clean.csv", "id", ["clean_title", "clean_body"], "naver_news"),
        _load_source(code, f"board_{code}_clean.csv", "nid", ["clean_title"], "naver_board"),
        _load_source(code, f"toss_community_{code}_clean.csv", "comment_id", ["clean_message"], "toss_community"),
    ]
    if code in PAXNET_VALID_CODES:
        frames.append(_load_source(code, f"paxnet_board_{code}_clean.csv", "seq", ["clean_title"], "paxnet_board"))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.strip().str.len() > 0]
    return df.drop_duplicates(subset="item_id").sort_values("datetime")


def score_all_sources_for_stock(scorer: SentimentScorer, code: str) -> pd.DataFrame:
    combined = collect_all_text(code)
    scores = scorer.score_batch(combined["text"].tolist())
    out = pd.concat([combined.reset_index(drop=True), scores], axis=1)
    out_path = PROCESSED_DIR / f"sentiment_scored_{code}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def build_daily_index(code: str, scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["datetime"]).dt.date
    daily = scored.groupby("date").agg(
        sentiment_index=("sentiment_score", "mean"),
        item_count=("sentiment_score", "size"),
    )

    full_range = pd.date_range(START_DATE, END_DATE, freq="D").date
    daily = daily.reindex(full_range)  # 글이 하루도 없는 날은 NaN으로 남김 (0으로 임의 대체하지 않음)
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["item_count"] = daily["item_count"].fillna(0).astype(int)
    daily["code"] = code

    out_path = PROCESSED_DIR / f"sentiment_index_daily_{code}.csv"
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    return daily


def main():
    print(f"모델 로딩 중: {MODEL_NAME} (최초 실행 시 다운로드, 몇 분 걸릴 수 있음)")
    scorer = SentimentScorer()

    for code, name in STOCKS.items():
        scored = score_all_sources_for_stock(scorer, code)
        daily = build_daily_index(code, scored)
        n_missing = daily["sentiment_index"].isna().sum()
        by_source = scored["source"].value_counts().to_dict()
        print(f"[sentiment] {code} {name}: {len(scored)}건 스코어링 {by_source} -> {len(daily)}일 중 결측 {n_missing}일")

    print("\n결측(NaN)이 남은 날은 그날 관련 텍스트가 전혀 없었다는 뜻입니다 — "
          "0으로 채우지 않았으니 Granger/GARCH 등에서 결측 처리 방식을 직접 정해서 쓰세요.")


if __name__ == "__main__":
    main()
