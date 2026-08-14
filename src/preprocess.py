"""텍스트 전처리: raw 뉴스/게시글을 감성분석 입력용으로 정제.

- HTML 잔여 태그, URL, 반복 특수문자/이모지 제거
- 공백 정리
- 중복 게시글 제거
- 너무 짧은 게시글(도배성 한두 글자) 필터링
"""
import re

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, STOCK_FILE_NAMES, STOCKS

URL_RE = re.compile(r"https?://\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")  # ㅋㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎㅎ 같은 4회 이상 반복
NON_TEXT_RE = re.compile(r"[^가-힣a-zA-Z0-9\s.,!?%()~\-]")
WHITESPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[()\[\]{}<>《》〈〉]")

MIN_BOARD_TITLE_LEN = 2

# "(( 마감 ))"처럼 시황 마커만 있고 실제 의견이 없는 글, "외국놈들 참"처럼 짧은 욕설/감탄사만
# 있는 글은 감성분석에 노이즈만 되고 주가와도 무관해서 따로 걸러낸다.
MARKER_STOPWORDS = {
    "마감", "시작", "개장", "폐장", "장마감", "장시작", "오전장", "오후장",
    "동시호가", "단일가", "휴장", "상한가", "하한가", "체결", "완료",
}
PROFANITY_WORDS = ["놈", "새끼", "병신", "씨발", "좆", "지랄", "꺼져", "쓰레기"]
SHORT_JUNK_LEN = 15  # 이 길이 이하이면서 욕설이 섞인 글은 분석 대상에서 제외


def is_low_signal(text: str) -> bool:
    if not text:
        return True
    core = WHITESPACE_RE.sub("", BRACKET_RE.sub("", text))
    if core in MARKER_STOPWORDS:
        return True
    if len(text) <= SHORT_JUNK_LEN and any(w in text for w in PROFANITY_WORDS):
        return True
    return False


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
    name = STOCK_FILE_NAMES[code]
    df = pd.read_csv(RAW_DIR / f"news_{name}.csv", encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df["clean_body"] = df["body_snippet"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "clean_body"])
    df = df[df["clean_title"].str.len() > 0]
    out_path = PROCESSED_DIR / f"news_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def clean_naver_search_news(code: str) -> pd.DataFrame:
    """crawl_naver_search_news.py 결과 정제 — 직접 실행하지 않은 경우 파일이 없을 수 있어 건너뜀."""
    name = STOCK_FILE_NAMES[code]
    path = RAW_DIR / f"news_search_{name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["url", "code", "datetime", "clean_title", "clean_body"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df["clean_body"] = df["body"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "clean_body"])
    df = df[df["clean_title"].str.len() > 0]
    out_path = PROCESSED_DIR / f"news_search_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def clean_board(code: str) -> pd.DataFrame:
    name = STOCK_FILE_NAMES[code]
    df = pd.read_csv(RAW_DIR / f"board_{name}.csv", encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "author", "datetime"])
    df = df[df["clean_title"].str.len() >= MIN_BOARD_TITLE_LEN]
    df = df[~df["clean_title"].map(is_low_signal)]
    out_path = PROCESSED_DIR / f"board_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def clean_paxnet_board(code: str) -> pd.DataFrame:
    name = STOCK_FILE_NAMES[code]
    df = pd.read_csv(RAW_DIR / f"paxnet_board_{name}.csv", encoding="utf-8-sig")
    df["clean_title"] = df["title"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_title", "author", "datetime"])
    df = df[df["clean_title"].str.len() >= MIN_BOARD_TITLE_LEN]
    df = df[~df["clean_title"].map(is_low_signal)]
    out_path = PROCESSED_DIR / f"paxnet_board_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def clean_edaily_news(code: str) -> pd.DataFrame:
    name = STOCK_FILE_NAMES[code]
    df = pd.read_csv(RAW_DIR / f"edaily_news_{name}.csv", encoding="utf-8-sig")
    df["clean_headline"] = df["headline"].map(clean_text)
    df["clean_body"] = df["body"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_headline", "clean_body"])
    df = df[df["clean_headline"].str.len() > 0]
    out_path = PROCESSED_DIR / f"edaily_news_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


MIN_TOSS_LIKES = 1  # 추천 0인 글은 노이즈일 가능성이 높아 제외 (표본 수집분 품질 보정 겸)


def clean_toss_community(code: str) -> pd.DataFrame:
    name = STOCK_FILE_NAMES[code]
    df = pd.read_csv(RAW_DIR / f"toss_community_{name}.csv", encoding="utf-8-sig")
    df["clean_message"] = df["message"].map(clean_text)
    df = df.drop_duplicates(subset=["clean_message", "author", "datetime"])
    df = df[df["clean_message"].str.len() >= MIN_BOARD_TITLE_LEN]
    df = df[~df["clean_message"].map(is_low_signal)]
    df = df[df["likes"] >= MIN_TOSS_LIKES]
    out_path = PROCESSED_DIR / f"toss_community_{name}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


# 팍스넷 게시판은 삼성전자/SK하이닉스만 실사용 트래픽이 있고, 카카오/에코프로비엠은
# 무관한 종목 홍보성 스팸으로 도배되어 있어 정제 대상에서 제외 (README 참고).
PAXNET_VALID_CODES = ["005930", "000660"]


def main():
    for code, name in STOCKS.items():
        news_df = clean_news(code)
        board_df = clean_board(code)
        edaily_df = clean_edaily_news(code)
        toss_df = clean_toss_community(code)
        search_df = clean_naver_search_news(code)
        print(
            f"[clean] {code} {name}: naver news {len(news_df)}건, naver board {len(board_df)}건, "
            f"edaily news {len(edaily_df)}건, toss {len(toss_df)}건, naver search {len(search_df)}건"
        )

    for code in PAXNET_VALID_CODES:
        df = clean_paxnet_board(code)
        print(f"[clean] {code} {STOCKS[code]}: paxnet board {len(df)}건")


if __name__ == "__main__":
    main()
