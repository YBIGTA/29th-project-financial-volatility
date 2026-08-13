"""NLP 감성분석: 사전학습 금융 감성분석 모델로 뉴스/게시글에 긍/부정/중립 점수를 매긴다.

킥오프 가이드의 "사전학습 모델 그대로 사용" 방침에 따라 파인튜닝 없이 그대로 추론만 한다.
모델: snunlp/KR-FinBert-SC (서울대 NLP연구실이 만든 한국어 금융 뉴스 3-class 감성분류기,
      긍정/부정/중립). Hugging Face에서 최초 실행 시 자동 다운로드.

이 스크립트는 Track A(크롤링)에서 Track B(감성분석/지수화, 메인 설승곤)로 넘어가는
경계의 1차 베이스라인입니다 — 본격적인 모델 비교·검증은 설승곤 담당 파트에서 이어가면 됩니다.
"""
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import PROCESSED_DIR, RAW_DIR, STOCKS

MODEL_NAME = "snunlp/KR-FinBert-SC"
LABELS = ["negative", "neutral", "positive"]  # 모델 config의 라벨 순서


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


def score_news_for_stock(scorer: SentimentScorer, code: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"edaily_news_{code}_clean.csv", encoding="utf-8-sig")
    texts = (df["clean_headline"].fillna("") + ". " + df["clean_body"].fillna("")).str.slice(0, 500)
    scores = scorer.score_batch(texts.tolist())
    out = pd.concat([df[["news_id", "datetime", "clean_headline"]].reset_index(drop=True), scores], axis=1)
    out_path = PROCESSED_DIR / f"sentiment_edaily_{code}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def build_daily_index(code: str, scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["datetime"]).dt.date
    daily = scored.groupby("date").agg(
        sentiment_index=("sentiment_score", "mean"),
        news_count=("sentiment_score", "size"),
    ).reset_index()
    daily["code"] = code
    out_path = PROCESSED_DIR / f"sentiment_index_daily_{code}.csv"
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    return daily


def main():
    print(f"모델 로딩 중: {MODEL_NAME} (최초 실행 시 다운로드, 몇 분 걸릴 수 있음)")
    scorer = SentimentScorer()

    for code, name in STOCKS.items():
        scored = score_news_for_stock(scorer, code)
        daily = build_daily_index(code, scored)
        print(f"[sentiment] {code} {name}: {len(scored)}건 스코어링 -> 일별 지수 {len(daily)}일치")
        print(daily.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
