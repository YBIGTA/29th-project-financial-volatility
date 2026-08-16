from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from config import ROOT_DIR


# =========================================================
# 1. 경로 및 파일명 설정
# =========================================================

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"
STATISTICS_DIR = ANALYSIS_DIR / "statistics"

for directory in (
    ANALYSIS_DIR,
    FIGURE_DIR,
    STATISTICS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# 종목별 merged_daily.csv 위치입니다.
# 5~6개월 데이터가 나와도 같은 위치에 파일만 교체하면 됩니다.
STOCK_CONFIG = {
    "005930": {
        "name": "Samsung Electronics",
        "label": "samsung_electronics",
        "market": "kospi",
        "path": (
            PROCESSED_DIR
            / "samsung_electronics"
            / "merged_daily.csv"
        ),
    },
    "000660": {
        "name": "SK hynix",
        "label": "sk_hynix",
        "market": "kospi",
        "path": (
            PROCESSED_DIR
            / "sk_hynix"
            / "merged_daily.csv"
        ),
    },
    "247540": {
        "name": "EcoPro BM",
        "label": "ecopro_bm",
        "market": "kosdaq",
        "path": (
            PROCESSED_DIR
            / "ecopro_bm"
            / "merged_daily.csv"
        ),
    },
    "035720": {
        "name": "Kakao",
        "label": "kakao",
        "market": "kospi",
        "path": (
            PROCESSED_DIR
            / "kakao"
            / "merged_daily.csv"
        ),
    },
}


# 요청한 정형데이터 파일을 받으면 실제 파일명만 여기서 수정합니다.
# 파일 자체의 이름을 바꿀 필요는 없습니다.
MARKET_FILE_CONFIG = {
    "kospi": {
        "filename": "kospi.csv",
        "required": False,
        "us_market": False,
    },
    "kosdaq": {
        "filename": "kosdaq.csv",
        "required": False,
        "us_market": False,
    },
    "vix": {
        "filename": "vix.csv",
        "required": False,
        "us_market": True,
    },
    "vkospi": {
        "filename": "vkospi.csv",
        "required": False,
        "us_market": False,
    },
}


MAX_GRANGER_LAG = 3


PRICE_COLUMN_ALIASES = {
    "date": "date",
    "Date": "date",
    "날짜": "date",
    "일자": "date",
    "기준일": "date",

    "open": "open",
    "Open": "open",
    "시가": "open",
    "시가지수": "open",

    "high": "high",
    "High": "high",
    "고가": "high",
    "고가지수": "high",

    "low": "low",
    "Low": "low",
    "저가": "low",
    "저가지수": "low",

    "close": "close",
    "Close": "close",
    "종가": "close",
    "종가지수": "close",

    "volume": "volume",
    "Volume": "volume",
    "거래량": "volume",
}


# =========================================================
# 2. 공통 데이터 정리 함수
# =========================================================

def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"파일 인코딩을 확인할 수 없습니다: {path}"
    )


def normalize_date(
    series: pd.Series,
    path: Path,
) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )

    result = pd.to_datetime(
        cleaned,
        errors="coerce",
    ).dt.normalize()

    if result.isna().any():
        count = int(result.isna().sum())

        raise ValueError(
            f"{path}: 날짜 변환 실패 {count}건"
        )

    return result


def normalize_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


# =========================================================
# 3. 종목별 merged_daily.csv 읽기
# =========================================================

def load_stock_merged(
    path: Path,
) -> pd.DataFrame:
    df = read_csv_flexible(path)

    required_columns = [
        "date",
        "intraday_sentiment_index",
        "intraday_item_count",
        "intraday_no_posts",
        "overnight_sentiment_index",
        "overnight_item_count",
        "overnight_no_posts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path}: 필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["date"] = normalize_date(
        df["date"],
        path,
    )

    numeric_columns = [
        column
        for column in df.columns
        if column != "date"
    ]

    for column in numeric_columns:
        df[column] = normalize_numeric(
            df[column]
        )

    if df["date"].duplicated().any():
        count = int(
            df["date"].duplicated().sum()
        )

        raise ValueError(
            f"{path}: 날짜 중복 {count}건"
        )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    for session in (
        "intraday",
        "overnight",
    ):
        score_column = (
            f"{session}_sentiment_index"
        )
        count_column = (
            f"{session}_item_count"
        )
        no_posts_column = (
            f"{session}_no_posts"
        )

        inconsistent = (
            df[count_column].eq(0)
            != df[no_posts_column].eq(1)
        )

        if inconsistent.any():
            count = int(inconsistent.sum())

            raise ValueError(
                f"{path}: {session}의 "
                f"item_count와 no_posts가 "
                f"일치하지 않는 행 {count}건"
            )

        # 크롤링팀이 전달한 원본 점수를 보존합니다.
        df[
            f"{session}_sentiment_index_raw"
        ] = df[score_column]

        # 글이 없는 경우(no_posts=1)의 0만
        # 분석용 결측치로 바꿉니다.
        df.loc[
            df[no_posts_column].eq(1),
            score_column,
        ] = np.nan

        # 팀 합의에 따른 분석용 forward-fill입니다.
        # 실제 중립 감성인 0은 그대로 유지됩니다.
        df[
            f"{session}_sentiment_ffill"
        ] = df[score_column].ffill()

    return df


# =========================================================
# 4. KOSPI·KOSDAQ·VIX·VKOSPI 파일 읽기
# =========================================================

def load_market_file(
    market_name: str,
) -> pd.DataFrame | None:
    config = MARKET_FILE_CONFIG[market_name]

    path = (
        RAW_DIR
        / config["filename"]
    )

    if not path.exists():
        print(
            f"[warning] {market_name.upper()} "
            f"파일이 없어 현재 분석에서 제외합니다: "
            f"{path}"
        )
        return None

    df = read_csv_flexible(path)

    df = df.rename(
        columns=PRICE_COLUMN_ALIASES
    )

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path}: 필수 가격 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    keep_columns = required_columns.copy()

    if "volume" in df.columns:
        keep_columns.append("volume")

    df = df[keep_columns].copy()

    df["date"] = normalize_date(
        df["date"],
        path,
    )

    for column in keep_columns:
        if column != "date":
            df[column] = normalize_numeric(
                df[column]
            )

    if df["date"].duplicated().any():
        count = int(
            df["date"].duplicated().sum()
        )

        raise ValueError(
            f"{path}: 날짜 중복 {count}건"
        )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    if market_name == "vix":
        missing_vix_rows = int(
            df[
                ["open", "high", "low", "close"]
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        if missing_vix_rows:
            print(
                f"[warning] VIX OHLC가 없는 "
                f"{missing_vix_rows}개 행을 제외합니다."
            )

        df = (
            df.dropna(
                subset=["open", "high", "low", "close"]
            )
            .reset_index(drop=True)
        )

    df["change"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    rename_columns = {
        column: f"{market_name}_{column}"
        for column in df.columns
        if column != "date"
    }

    return df.rename(
        columns=rename_columns
    )


# =========================================================
# 5. VIX 날짜 정렬
# =========================================================

def prepare_embedded_vix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    vix_columns = [
        column
        for column in [
            "vix_open",
            "vix_high",
            "vix_low",
            "vix_close",
            "vix_volume",
        ]
        if column in result.columns
    ]

    if "vix_close" not in result.columns:
        return result

    # 현재 merged_daily에 같은 날짜로 붙은
    # VIX 값을 별도 원본 컬럼으로 보존합니다.
    for column in vix_columns:
        result[
            f"{column}_reported_same_date"
        ] = result[column]

    # 한국 시장 T일 분석에는 한국장 개장 전에
    # 확인 가능한 직전 VIX를 사용합니다.
    #
    # 같은 날짜 미국 VIX 종가는 한국장이 끝난 뒤
    # 확정되므로 직전 관측값을 사용합니다.
    for column in vix_columns:
        result[column] = (
            result[column]
            .ffill()
            .shift(1)
        )

    result["vix_change"] = np.log(
        result["vix_close"]
        / result["vix_close"].shift(1)
    )

    return result


def merge_external_vix(
    stock_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> pd.DataFrame:
    result = stock_df.copy()

    # 기존 merged_daily에 들어 있던 VIX는 보존합니다.
    embedded_vix_columns = [
        column
        for column in result.columns
        if column.startswith("vix_")
    ]

    for column in embedded_vix_columns:
        result[
            f"embedded_{column}"
        ] = result[column]

    result = result.drop(
        columns=embedded_vix_columns
    )

    # VIX는 미국 시장 데이터이므로
    # 한국 거래일보다 엄격하게 이전인 가장 최근
    # 미국 거래일 값을 붙입니다.
    result = pd.merge_asof(
        result.sort_values("date"),
        vix_df.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
    )

    return result


# =========================================================
# 6. 수익률 및 변동성 변수 생성
# =========================================================

def add_price_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        df.sort_values("date")
        .copy()
    )

    result["cc_return"] = np.log(
        result["close"]
        / result["close"].shift(1)
    )

    result["overnight_return"] = np.log(
        result["open"]
        / result["close"].shift(1)
    )

    result["intraday_return"] = np.log(
        result["close"]
        / result["open"]
    )

    result["abs_cc_return"] = (
        result["cc_return"].abs()
    )

    result["squared_cc_return"] = (
        result["cc_return"] ** 2
    )

    result["parkinson_volatility"] = np.sqrt(
        (
            np.log(
                result["high"]
                / result["low"]
            ) ** 2
        )
        / (4 * math.log(2))
    )

    result["rolling_volatility_5d"] = (
        result["cc_return"]
        .rolling(5)
        .std()
    )

    return result


# =========================================================
# 7. 일별 감성지수 분포 확인
# =========================================================

def make_daily_sentiment_histogram(
    df: pd.DataFrame,
    stock_label: str,
    stock_name: str,
) -> dict[str, float | int]:
    intraday = (
        df["intraday_sentiment_index_raw"]
        .dropna()
    )

    overnight = (
        df["overnight_sentiment_index_raw"]
        .dropna()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].hist(
        intraday,
        bins=20,
        edgecolor="black",
        alpha=0.75,
        color="#2563EB",
    )

    axes[0].axvline(
        0,
        color="red",
        linestyle="--",
    )

    axes[0].set_title(
        f"{stock_name} - Intraday Daily Sentiment"
    )
    axes[0].set_xlabel("Daily sentiment index")
    axes[0].set_ylabel("Trading-day count")

    axes[1].hist(
        overnight,
        bins=20,
        edgecolor="black",
        alpha=0.75,
        color="#7C3AED",
    )

    axes[1].axvline(
        0,
        color="red",
        linestyle="--",
    )

    axes[1].set_title(
        f"{stock_name} - Overnight Daily Sentiment"
    )
    axes[1].set_xlabel("Daily sentiment index")
    axes[1].set_ylabel("Trading-day count")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / f"daily_sentiment_histogram_{stock_label}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "intraday_daily_mean": float(
            intraday.mean()
        ),
        "intraday_daily_std": float(
            intraday.std()
        ),
        "intraday_daily_min": float(
            intraday.min()
        ),
        "intraday_daily_max": float(
            intraday.max()
        ),
        "intraday_abs_lt_005_ratio": float(
            intraday.abs().lt(0.05).mean()
        ),
        "overnight_daily_mean": float(
            overnight.mean()
        ),
        "overnight_daily_std": float(
            overnight.std()
        ),
        "overnight_daily_min": float(
            overnight.min()
        ),
        "overnight_daily_max": float(
            overnight.max()
        ),
        "overnight_abs_lt_005_ratio": float(
            overnight.abs().lt(0.05).mean()
        ),
    }


# =========================================================
# 8. 분석 데이터 생성 및 병합
# =========================================================

def build_analysis_datasets() -> None:
    market_data = {
        market_name: load_market_file(
            market_name
        )
        for market_name in MARKET_FILE_CONFIG
    }

    qa_rows = []
    pooled_frames = []

    for code, stock_info in STOCK_CONFIG.items():
        stock_name = stock_info["name"]
        stock_label = stock_info["label"]
        stock_market = stock_info["market"]
        stock_path = stock_info["path"]

        print(
            f"\n[load] {code} {stock_name}: "
            f"{stock_path}"
        )

        merged = load_stock_merged(
            stock_path
        )

        total_rows_before = len(merged)

        missing_price_rows = int(
            merged[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        # 8월 14일처럼 감성은 있지만
        # 아직 가격이 없는 행은 분석에서 제외합니다.
        merged = (
            merged.dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
            .reset_index(drop=True)
        )

        if missing_price_rows:
            print(
                f"[warning] {stock_name}: "
                f"가격이 없는 {missing_price_rows}행 제외"
            )

        merged.insert(
            1,
            "stock_code",
            code,
        )

        merged.insert(
            2,
            "stock_name",
            stock_name,
        )

        merged.insert(
            3,
            "market",
            stock_market,
        )

        merged = add_price_features(
            merged
        )

        external_vix = market_data["vix"]

        if external_vix is not None:
            merged = merge_external_vix(
                merged,
                external_vix,
            )
        else:
            merged = prepare_embedded_vix(
                merged
            )

        for market_name in (
            "kospi",
            "kosdaq",
            "vkospi",
        ):
            market_df = market_data[
                market_name
            ]

            if market_df is None:
                continue

            merged = merged.merge(
                market_df,
                on="date",
                how="left",
                validate="one_to_one",
            )

        distribution_summary = (
            make_daily_sentiment_histogram(
                merged,
                stock_label,
                stock_name,
            )
        )

        output = merged.copy()

        output["date"] = (
            output["date"]
            .dt.strftime("%Y-%m-%d")
        )

        output_path = (
            ANALYSIS_DIR
            / f"analysis_{stock_label}.csv"
        )

        output.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        pooled_frames.append(output)

        qa_rows.append(
            {
                "stock_code": code,
                "stock_name": stock_name,
                "input_rows": total_rows_before,
                "analysis_rows": len(merged),
                "excluded_missing_price_rows": (
                    missing_price_rows
                ),
                "start_date": (
                    merged["date"]
                    .min()
                    .strftime("%Y-%m-%d")
                ),
                "end_date": (
                    merged["date"]
                    .max()
                    .strftime("%Y-%m-%d")
                ),
                "duplicate_dates": int(
                    merged["date"]
                    .duplicated()
                    .sum()
                ),
                "missing_ohlcv_rows": int(
                    merged[
                        [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                    ]
                    .isna()
                    .any(axis=1)
                    .sum()
                ),
                "intraday_no_posts_days": int(
                    merged[
                        "intraday_no_posts"
                    ].eq(1).sum()
                ),
                "overnight_no_posts_days": int(
                    merged[
                        "overnight_no_posts"
                    ].eq(1).sum()
                ),
                "missing_intraday_after_ffill": int(
                    merged[
                        "intraday_sentiment_ffill"
                    ]
                    .isna()
                    .sum()
                ),
                "missing_overnight_after_ffill": int(
                    merged[
                        "overnight_sentiment_ffill"
                    ]
                    .isna()
                    .sum()
                ),
                "missing_vix_close": (
                    int(
                        merged["vix_close"]
                        .isna()
                        .sum()
                    )
                    if "vix_close" in merged
                    else None
                ),
                "missing_vkospi_close": (
                    int(
                        merged["vkospi_close"]
                        .isna()
                        .sum()
                    )
                    if "vkospi_close" in merged
                    else None
                ),
                **distribution_summary,
            }
        )

        print(
            f"[analysis] {stock_name}: "
            f"{len(merged)}거래일, "
            f"{merged['date'].min().date()} ~ "
            f"{merged['date'].max().date()}"
        )

    qa = pd.DataFrame(
        qa_rows
    )

    qa.to_csv(
        ANALYSIS_DIR / "qa_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled = pd.concat(
        pooled_frames,
        ignore_index=True,
    )

    pooled.to_csv(
        ANALYSIS_DIR
        / "pooled_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "input_structure": (
            "data/processed/{stock}/merged_daily.csv"
        ),
        "overnight_definition": (
            "previous trading day 15:30 "
            "to current trading day 09:00"
        ),
        "intraday_definition": (
            "current trading day 09:00 "
            "to current trading day 15:30"
        ),
        "missing_sentiment_policy": (
            "no_posts=1 only converted to NaN; "
            "separate forward-filled column"
        ),
        "granger_max_lag": MAX_GRANGER_LAG,
        "external_kospi_loaded": (
            market_data["kospi"] is not None
        ),
        "external_kosdaq_loaded": (
            market_data["kosdaq"] is not None
        ),
        "external_vix_loaded": (
            market_data["vix"] is not None
        ),
        "external_vkospi_loaded": (
            market_data["vkospi"] is not None
        ),
    }

    (
        ANALYSIS_DIR
        / "pipeline_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =========================================================
# 9. ADF 정상성 검정
# =========================================================

def run_adf(
    series: pd.Series,
) -> dict:
    clean = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(clean) < 20:
        return {
            "status": "SKIP_TOO_FEW",
            "n": len(clean),
            "adf_stat": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
        }

    if clean.nunique() < 2:
        return {
            "status": "SKIP_CONSTANT",
            "n": len(clean),
            "adf_stat": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
        }

    try:
        result = adfuller(
            clean,
            autolag="AIC",
        )

        return {
            "status": (
                "STATIONARY_5PCT"
                if result[1] < 0.05
                else "NONSTATIONARY_5PCT"
            ),
            "n": int(result[3]),
            "adf_stat": float(result[0]),
            "p_value": float(result[1]),
            "used_lag": int(result[2]),
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "n": len(clean),
            "error": str(error),
        }


# =========================================================
# 10. HAC OLS 회귀분석
# =========================================================

def run_hac_ols(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    controls: list[str],
) -> dict:
    columns = [
        outcome,
        predictor,
        *controls,
    ]

    sample = (
        df[columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if (
        len(sample) < 20
        or sample[predictor].nunique() < 2
    ):
        return {
            "status": "SKIP",
            "n": len(sample),
        }

    try:
        x = sm.add_constant(
            sample[
                [predictor, *controls]
            ],
            has_constant="add",
        )

        model = sm.OLS(
            sample[outcome],
            x,
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": MAX_GRANGER_LAG
            },
        )

        return {
            "status": "OK",
            "n": int(model.nobs),
            "coefficient": float(
                model.params[predictor]
            ),
            "std_error_hac": float(
                model.bse[predictor]
            ),
            "t_value": float(
                model.tvalues[predictor]
            ),
            "p_value": float(
                model.pvalues[predictor]
            ),
            "r_squared": float(
                model.rsquared
            ),
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "n": len(sample),
            "error": str(error),
        }


# =========================================================
# 11. Granger 검정
# =========================================================

def run_granger(
    df: pd.DataFrame,
    outcome: str,
    cause: str,
) -> list[dict]:
    sample = (
        df[
            [outcome, cause]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if (
        len(sample) < 25
        or sample[outcome].nunique() < 2
        or sample[cause].nunique() < 2
    ):
        return [
            {
                "lag": lag,
                "status": "SKIP",
                "n": len(sample),
            }
            for lag in range(
                1,
                MAX_GRANGER_LAG + 1,
            )
        ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = grangercausalitytests(
                sample[
                    [outcome, cause]
                ],
                maxlag=MAX_GRANGER_LAG,
                verbose=False,
            )

        rows = []

        for lag in range(
            1,
            MAX_GRANGER_LAG + 1,
        ):
            f_test = (
                results[lag][0]["ssr_ftest"]
            )

            rows.append(
                {
                    "lag": lag,
                    "status": "OK",
                    "n": len(sample),
                    "f_stat": float(
                        f_test[0]
                    ),
                    "p_value": float(
                        f_test[1]
                    ),
                    "df_denom": float(
                        f_test[2]
                    ),
                    "df_num": float(
                        f_test[3]
                    ),
                }
            )

        return rows

    except Exception as error:
        return [
            {
                "lag": lag,
                "status": "ERROR",
                "n": len(sample),
                "error": str(error),
            }
            for lag in range(
                1,
                MAX_GRANGER_LAG + 1,
            )
        ]


# =========================================================
# 12. 실제 통계검정 실행
# =========================================================

def run_statistical_checks() -> None:
    adf_rows = []
    ols_rows = []
    granger_rows = []

    adf_columns = [
        "intraday_sentiment_ffill",
        "overnight_sentiment_ffill",
        "cc_return",
        "overnight_return",
        "intraday_return",
        "abs_cc_return",
        "squared_cc_return",
        "parkinson_volatility",
        "rolling_volatility_5d",
        "kospi_close",
        "kosdaq_close",
        "kospi_change",
        "kosdaq_change",
        "vix_close",
        "vix_change",
        "vkospi_close",
        "vkospi_change",
    ]

    for code, stock_info in STOCK_CONFIG.items():
        stock_label = stock_info["label"]

        path = (
            ANALYSIS_DIR
            / f"analysis_{stock_label}.csv"
        )

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = (
            df.sort_values("date")
            .reset_index(drop=True)
        )

        for column in adf_columns:
            if column not in df.columns:
                continue

            adf_rows.append(
                {
                    "stock_code": code,
                    "variable": column,
                    **run_adf(df[column]),
                }
            )

        market_control = (
            "kospi_change"
            if stock_info["market"] == "kospi"
            else "kosdaq_change"
        )

        controls = [
            column
            for column in [
                market_control,
                "vkospi_change",
                "vix_change",
            ]
            if column in df.columns
        ]

        ols_specs = [
            (
                "intraday_return",
                "intraday_sentiment_ffill",
                "same_day_intraday",
            ),
            (
                "cc_return",
                "overnight_sentiment_ffill",
                "night_to_same_day_return",
            ),
            (
                "parkinson_volatility",
                "overnight_sentiment_ffill",
                "night_to_same_day_volatility",
            ),
        ]

        for (
            outcome,
            predictor,
            analysis_name,
        ) in ols_specs:
            ols_rows.append(
                {
                    "stock_code": code,
                    "analysis": analysis_name,
                    "outcome": outcome,
                    "predictor": predictor,
                    "controls": ",".join(
                        controls
                    ),
                    **run_hac_ols(
                        df,
                        outcome,
                        predictor,
                        controls,
                    ),
                }
            )

        granger_specs = [
            (
                "cc_return",
                "overnight_sentiment_ffill",
                "past_night_to_return",
            ),
            (
                "parkinson_volatility",
                "overnight_sentiment_ffill",
                "past_night_to_volatility",
            ),
            (
                "cc_return",
                "intraday_sentiment_ffill",
                "past_day_to_next_return",
            ),
            (
                "parkinson_volatility",
                "intraday_sentiment_ffill",
                "past_day_to_next_volatility",
            ),
        ]

        for (
            outcome,
            cause,
            analysis_name,
        ) in granger_specs:
            results = run_granger(
                df,
                outcome,
                cause,
            )

            for result in results:
                granger_rows.append(
                    {
                        "stock_code": code,
                        "analysis": analysis_name,
                        "outcome": outcome,
                        "cause": cause,
                        **result,
                    }
                )

    pd.DataFrame(
        adf_rows
    ).to_csv(
        STATISTICS_DIR
        / "adf_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        ols_rows
    ).to_csv(
        STATISTICS_DIR
        / "hac_ols_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        granger_rows
    ).to_csv(
        STATISTICS_DIR
        / "granger_results.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# 13. 전체 실행
# =========================================================

def main() -> None:
    print(
        "=== Track B 분석 데이터 생성 ==="
    )

    build_analysis_datasets()

    print(
        "\n=== ADF·HAC OLS·Granger 실행 ==="
    )

    run_statistical_checks()

    print(
        f"\n완료: {ANALYSIS_DIR}"
    )


if __name__ == "__main__":
    main()