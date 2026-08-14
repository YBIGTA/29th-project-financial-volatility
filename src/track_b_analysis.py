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

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    ROOT_DIR,
    STOCKS,
    STOCK_ENGLISH_NAMES,
    STOCK_FILE_NAMES,
)

ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


MARKET_BY_STOCK = {
    "005930": "kospi",
    "000660": "kospi",
    "247540": "kosdaq",
    "035720": "kospi",
}

PRICE_COLUMN_ALIASES = {
    "date": "date",
    "Date": "date",
    "날짜": "date",
    "일자": "date",
    "open": "open",
    "Open": "open",
    "시가": "open",
    "high": "high",
    "High": "high",
    "고가": "high",
    "low": "low",
    "Low": "low",
    "저가": "low",
    "close": "close",
    "Close": "close",
    "종가": "close",
    "volume": "volume",
    "Volume": "volume",
    "거래량": "volume",
}

MAX_GRANGER_LAG = 3


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"파일 인코딩을 확인할 수 없습니다: {path}")


def normalize_date(series: pd.Series, path: Path) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )

    result = pd.to_datetime(cleaned, errors="coerce").dt.normalize()

    if result.isna().any():
        count = int(result.isna().sum())
        raise ValueError(f"{path}: 날짜 변환 실패 {count}건")

    return result


def normalize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def load_price(
    path: Path,
    require_volume: bool = True,
) -> pd.DataFrame:
    df = read_csv_flexible(path)
    df = df.rename(columns=PRICE_COLUMN_ALIASES)

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]

    if require_volume:
        required_columns.append("volume")

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path}: 필수 컬럼이 없습니다: {missing_columns}"
        )

    keep_columns = required_columns.copy()

    if not require_volume and "volume" in df.columns:
        keep_columns.append("volume")

    df = df[keep_columns].copy()
    df["date"] = normalize_date(df["date"], path)

    for column in keep_columns:
        if column != "date":
            df[column] = normalize_numeric(df[column])

    if df["date"].duplicated().any():
        raise ValueError(f"{path}: 날짜 중복이 있습니다.")

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


def load_sentiment(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)

    required_columns = [
        "date",
        "intraday_sentiment_index",
        "intraday_item_count",
        "intraday_no_posts",
        "overnight_sentiment_index",
        "overnight_item_count",
        "overnight_no_posts",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path}: 장외·장중 감성 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    optional_columns = [
        column
        for column in ["price_available", "code"]
        if column in df.columns
    ]

    df = df[required_columns + optional_columns].copy()
    df["date"] = normalize_date(df["date"], path)

    numeric_columns = [
        column
        for column in required_columns
        if column != "date"
    ]

    for column in numeric_columns:
        df[column] = normalize_numeric(df[column])

    if df["date"].duplicated().any():
        raise ValueError(f"{path}: 감성지수 날짜가 중복됩니다.")

    for session in ("intraday", "overnight"):
        score_column = f"{session}_sentiment_index"
        count_column = f"{session}_item_count"
        no_posts_column = f"{session}_no_posts"

        inconsistent = (
            (df[count_column].eq(0))
            !=
            (df[no_posts_column].eq(1))
        )

        if inconsistent.any():
            count = int(inconsistent.sum())
            raise ValueError(
                f"{path}: {session}의 item_count와 "
                f"no_posts가 일치하지 않는 행 {count}건"
            )

        # 크롤링팀이 만든 원본값을 그대로 보존합니다.
        df[f"{session}_sentiment_index_raw"] = (
            df[score_column]
        )

        # 게시글이 없는 날의 0만 NaN으로 바꿉니다.
        # 게시글이 있고 평균이 실제 0인 경우는 그대로 둡니다.
        df.loc[
            df[no_posts_column].eq(1),
            score_column,
        ] = np.nan

        # 팀 합의에 따른 분석용 forward-fill 열입니다.
        df[f"{session}_sentiment_ffill"] = (
            df[score_column].ffill()
        )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("date").copy()

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
        np.log(
            result["high"] / result["low"]
        ) ** 2
        / (4 * math.log(2))
    )

    result["rolling_volatility_5d"] = (
        result["cc_return"]
        .rolling(5)
        .std()
    )

    return result


def load_market_file(
    filename: str,
    prefix: str,
    require_volume: bool,
) -> pd.DataFrame | None:
    path = RAW_DIR / filename

    if not path.exists():
        print(f"[warning] 시장 데이터 없음: {path}")
        return None

    df = load_price(
        path,
        require_volume=require_volume,
    )

    df["change"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    rename_columns = {
        column: f"{prefix}_{column}"
        for column in df.columns
        if column != "date"
    }

    return df.rename(columns=rename_columns)


def make_score_histogram(
    code: str,
    output_directory: Path,
) -> dict[str, float | int]:
    label = STOCK_FILE_NAMES[code]
    company = STOCK_ENGLISH_NAMES[code]

    path = (
        PROCESSED_DIR
        / f"sentiment_scored_{label}.csv"
    )

    df = read_csv_flexible(path)

    if "sentiment_score" not in df.columns:
        raise ValueError(
            f"{path}: sentiment_score 컬럼이 없습니다."
        )

    scores = normalize_numeric(
        df["sentiment_score"]
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        scores,
        bins=30,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        f"{company} - Sentiment Score Distribution"
    )

    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Count")

    fig.tight_layout()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_directory
        / f"score_histogram_{label}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "score_count": int(scores.size),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_zero_count": int(
            scores.eq(0).sum()
        ),
        "score_abs_lt_001_ratio": float(
            scores.abs().lt(0.01).mean()
        ),
        "score_abs_lt_005_ratio": float(
            scores.abs().lt(0.05).mean()
        ),
        "score_abs_lt_010_ratio": float(
            scores.abs().lt(0.10).mean()
        ),
        "score_unique_count": int(
            scores.nunique()
        ),
    }


def build_analysis_datasets() -> None:
    figure_directory = (
        ANALYSIS_DIR / "figures"
    )

    kospi = load_market_file(
        "price_daily_kospi.csv",
        "kospi",
        require_volume=True,
    )

    kosdaq = load_market_file(
        "price_daily_kosdaq.csv",
        "kosdaq",
        require_volume=True,
    )

    vix = load_market_file(
        "price_daily_vix.csv",
        "vix",
        require_volume=False,
    )

    vkospi = load_market_file(
        "price_daily_vkospi.csv",
        "vkospi",
        require_volume=False,
    )

    market_data = {
        "kospi": kospi,
        "kosdaq": kosdaq,
        "vix": vix,
        "vkospi": vkospi,
    }

    qa_rows = []
    pooled_frames = []

    for code, company_korean in STOCKS.items():
        label = STOCK_FILE_NAMES[code]
        company = STOCK_ENGLISH_NAMES[code]

        price_path = (
            RAW_DIR
            / f"price_daily_{label}.csv"
        )

        sentiment_path = (
            PROCESSED_DIR
            / f"sentiment_index_daily_{label}.csv"
        )

        price = load_price(
            price_path,
            require_volume=True,
        )

        sentiment = load_sentiment(
            sentiment_path
        )

        # 가격이 존재하는 실제 거래일을 기준으로 병합합니다.
        # 감성만 있고 아직 가격이 없는 미래 행은 제외됩니다.
        merged = price.merge(
            sentiment,
            on="date",
            how="left",
            validate="one_to_one",
        )

        merged.insert(1, "stock_code", code)
        merged.insert(2, "stock_name", company)
        merged.insert(
            3,
            "market",
            MARKET_BY_STOCK[code],
        )

        merged = add_price_features(merged)

        for market_name, market_df in market_data.items():
            if market_df is None:
                continue

            merged = merged.merge(
                market_df,
                on="date",
                how="left",
                validate="one_to_one",
            )

        output = merged.copy()

        output["date"] = (
            output["date"]
            .dt.strftime("%Y-%m-%d")
        )

        output.to_csv(
            ANALYSIS_DIR
            / f"analysis_{label}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pooled_frames.append(output)

        score_summary = make_score_histogram(
            code,
            figure_directory,
        )

        qa_rows.append(
            {
                "stock_code": code,
                "stock_name": company,
                "rows": len(merged),
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
                "missing_intraday_before_ffill": int(
                    merged[
                        "intraday_sentiment_index"
                    ]
                    .isna()
                    .sum()
                ),
                "missing_intraday_after_ffill": int(
                    merged[
                        "intraday_sentiment_ffill"
                    ]
                    .isna()
                    .sum()
                ),
                "missing_overnight_before_ffill": int(
                    merged[
                        "overnight_sentiment_index"
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
                **score_summary,
            }
        )

        print(
            f"[analysis] {code} {company_korean}: "
            f"{len(merged)}거래일"
        )

    qa = pd.DataFrame(qa_rows)

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
        ANALYSIS_DIR / "pooled_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "overnight_definition": (
            "previous trading day 15:30 "
            "to current trading day 09:00"
        ),
        "intraday_definition": (
            "current trading day 09:00 "
            "to 15:30"
        ),
        "missing_sentiment_policy": (
            "raw 0 and no_posts retained; "
            "no_posts converted to null only in "
            "analysis column; separate ffill column"
        ),
        "granger_max_lag": MAX_GRANGER_LAG,
        "vkospi_loaded": vkospi is not None,
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


def run_adf(
    series: pd.Series,
) -> dict[str, float | int | str]:
    clean = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
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


def run_hac_ols(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    controls: list[str],
) -> dict[str, float | int | str]:
    columns = [
        outcome,
        predictor,
        *controls,
    ]

    sample = (
        df[columns]
        .replace([np.inf, -np.inf], np.nan)
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

    x = sm.add_constant(
        sample[[predictor, *controls]],
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


def run_granger(
    df: pd.DataFrame,
    outcome: str,
    cause: str,
) -> list[dict[str, float | int | str]]:
    sample = (
        df[[outcome, cause]]
        .replace([np.inf, -np.inf], np.nan)
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results = grangercausalitytests(
            sample[[outcome, cause]],
            maxlag=MAX_GRANGER_LAG,
            verbose=False,
        )

    rows = []

    for lag in range(
        1,
        MAX_GRANGER_LAG + 1,
    ):
        f_test = results[lag][0]["ssr_ftest"]

        rows.append(
            {
                "lag": lag,
                "status": "OK",
                "n": len(sample),
                "f_stat": float(f_test[0]),
                "p_value": float(f_test[1]),
                "df_denom": float(f_test[2]),
                "df_num": float(f_test[3]),
            }
        )

    return rows


def run_statistical_checks() -> None:
    statistics_directory = (
        ANALYSIS_DIR / "statistics"
    )

    statistics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "vix_close",
        "vkospi_close",
        "vix_change",
        "vkospi_change",
    ]

    for code in STOCKS:
        label = STOCK_FILE_NAMES[code]

        path = (
            ANALYSIS_DIR
            / f"analysis_{label}.csv"
        )

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = df.sort_values("date")

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

        controls = [
            column
            for column in [
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
                "overnight_return",
                "overnight_sentiment_ffill",
                "same_day_overnight",
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

        for outcome, predictor, analysis in ols_specs:
            ols_rows.append(
                {
                    "stock_code": code,
                    "analysis": analysis,
                    "outcome": outcome,
                    "predictor": predictor,
                    "controls": ",".join(controls),
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

        for outcome, cause, analysis in granger_specs:
            for result in run_granger(
                df,
                outcome,
                cause,
            ):
                granger_rows.append(
                    {
                        "stock_code": code,
                        "analysis": analysis,
                        "outcome": outcome,
                        "cause": cause,
                        **result,
                    }
                )

    pd.DataFrame(adf_rows).to_csv(
        statistics_directory
        / "adf_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(ols_rows).to_csv(
        statistics_directory
        / "hac_ols_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(granger_rows).to_csv(
        statistics_directory
        / "granger_results.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    print("=== Track B 분석 데이터 생성 ===")
    build_analysis_datasets()

    print("\n=== ADF·HAC OLS·Granger 실행 ===")
    run_statistical_checks()

    print(
        f"\n완료: {ANALYSIS_DIR}"
    )


if __name__ == "__main__":
    main()