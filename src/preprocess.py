"""텍스트 전처리: raw 뉴스/게시글을 감성분석 입력용으로 정제.

- HTML 잔여 태그, URL, 반복 특수문자/이모지 제거
- 공백 정리
- 중복 게시글 제거
- 너무 짧은 게시글(도배성 한두 글자) 필터링
"""
import re

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, STOCKS

URL_RE = re.compile(r"https?://\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")  # ㅋㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎㅎ 같은 4회 이상 반복
NON_TEXT_RE = re.compile(r"[^가-힣a-zA-Z0-9\s.,!?%()~\-]")
WHITESPACE_RE = re.compile(r"\s+")

MIN_BOARD_TITLE_LEN = 2


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1\1", text)
    text = NON_TEXT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_news(code: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"news_{code}.csv", encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df["clean_body"] = df["body_snippet"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "clean_body"])
    df = df[df["clean_title"].str.len() > 0]
    out_path = PROCESSED_DIR / f"news_{code}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def clean_board(code: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"board_{code}.csv", encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "author", "datetime"])
    df = df[df["clean_title"].str.len() >= MIN_BOARD_TITLE_LEN]
    out_path = PROCESSED_DIR / f"board_{code}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def main():
    for code, name in STOCKS.items():
        news_df = clean_news(code)
        board_df = clean_board(code)
        print(f"[clean] {code} {name}: news {len(news_df)}건, board {len(board_df)}건")


if __name__ == "__main__":
    main()
