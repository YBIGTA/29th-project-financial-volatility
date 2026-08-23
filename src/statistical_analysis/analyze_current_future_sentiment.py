from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


ROOT_DIR = Path(__file__).resolve().parents[2]
# 기존 6개월 merged_daily.csv를 덮어쓰지 않기 위한 별도 입력 폴더입니다.
PROCESSED_DIR = ROOT_DIR / "data" / "current_future" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "analysis" / "current_future"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = {
    "005930": ("samsung_electronics", "kospi"),
    "000660": ("sk_hynix", "kospi"),
    "247540": ("ecopro_bm", "kosdaq"),
    "035720": ("kakao", "kospi"),
}

SCORES = [
    "intraday_current_sentiment_index",
    "intraday_future_sentiment_index",
    "overnight_current_sentiment_index",
    "overnight_future_sentiment_index",
]

MAX_LAG = 3


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"인코딩을 확인할 수 없습니다: {path}")


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype("string")
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False),
        errors="coerce",
    ).dt.normalize()


def load_index(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    df = read_csv(path)
    aliases = {
        "Date": "date", "날짜": "date", "일자": "date",
        "Close": "close", "종가": "close", "종가지수": "close",
    }
    df = df.rename(columns=aliases)
    if not {"date", "close"}.issubset(df.columns):
        raise ValueError(f"{path}: date/close 컬럼이 필요합니다. 현재: {list(df.columns)}")
    out = df[["date", "close"]].copy()
    out["date"] = normalize_date(out["date"])
    out["close"] = pd.to_numeric(
        out["close"].astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )
    out = out.dropna().drop_duplicates("date").sort_values("date")
    return out.rename(columns={"close": f"{name}_close"})


def load_stock(code: str, label: str, market: str) -> pd.DataFrame:
    path = PROCESSED_DIR / label / "merged_daily.csv"
    df = read_csv(path)
    required = {
        "date", "open", "high", "low", "close",
        "intraday_item_count", "intraday_no_posts",
        "overnight_item_count", "overnight_no_posts", *SCORES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: 새 점수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["date"] = normalize_date(df["date"])
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError(f"{path}: 날짜 변환 실패 또는 중복 날짜가 있습니다.")

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # no_posts=1인 점수만 결측으로 처리한 뒤 이전 관측값으로 채웁니다.
    for session in ("intraday", "overnight"):
        no_posts = df[f"{session}_no_posts"].eq(1)
        for perspective in ("current", "future"):
            col = f"{session}_{perspective}_sentiment_index"
            df.loc[no_posts, col] = np.nan
            df[f"{col}_ffill"] = df[col].ffill()

    df["cc_return"] = np.log(df["close"]).diff()
    df["intraday_return"] = np.log(df["close"] / df["open"])
    hl = np.log(df["high"] / df["low"])
    df["parkinson_volatility"] = np.sqrt((hl ** 2) / (4 * np.log(2)))
    df["next_cc_return"] = df["cc_return"].shift(-1)
    df["next_parkinson_volatility"] = df["parkinson_volatility"].shift(-1)

    for index_name in (market, "vix", "vkospi"):
        ext = load_index(index_name)
        close_col = f"{index_name}_close"

    # merged_daily.csv에 이미 존재하는 VIX 컬럼과
    # 외부 vix.csv 컬럼이 중복되는 것을 방지합니다.
        df = df.drop(
            columns=[close_col],
            errors="ignore",
        )

        df = df.merge(
            ext,
            on="date",
            how="left",
            validate="one_to_one",
        )

    # 미국 휴장일로 비어 있는 VIX만
    # 직전 미국 거래일 값으로 채웁니다.
        if index_name == "vix":
            df[close_col] = df[close_col].ffill()

        df[f"{index_name}_change"] = np.log(
            df[close_col]
        ).diff()

    df["stock_code"] = code
    df["stock_label"] = label
    return df


def run_adf(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 20 or x.nunique() < 2:
        return {"n": len(x), "adf_stat": np.nan, "p_value": np.nan,
                "used_lag": np.nan, "status": "INSUFFICIENT"}
    try:
        stat, p, lag, nobs, *_ = adfuller(x, maxlag=MAX_LAG, autolag="AIC")
        return {"n": nobs, "adf_stat": stat, "p_value": p,
                "used_lag": lag,
                "status": "STATIONARY_5PCT" if p < 0.05 else "NONSTATIONARY_5PCT"}
    except Exception as exc:
        return {"n": len(x), "adf_stat": np.nan, "p_value": np.nan,
                "used_lag": np.nan, "status": f"ERROR: {exc}"}


def fit_hac(df: pd.DataFrame, outcome: str, predictors: list[str], controls: list[str]) -> list[dict]:
    cols = [outcome, *predictors, *controls]

    work = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if len(work) <= len(cols) + 5:
        return [{"predictor": p, "n": len(work), "coefficient": np.nan,
                 "std_error": np.nan, "p_value": np.nan, "status": "INSUFFICIENT"}
                for p in predictors]
    x = sm.add_constant(
        work[[*predictors, *controls]],
        has_constant="add",
    ).astype(float)

    y = work[outcome].astype(float)

    model = sm.OLS(
        y,
        x,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
    )
    rows = []
    for predictor in predictors:
        rows.append({
            "predictor": predictor, "n": int(model.nobs),
            "coefficient": model.params[predictor],
            "std_error": model.bse[predictor], "p_value": model.pvalues[predictor],
            "r_squared": model.rsquared, "status": "OK",
        })
    return rows


def run_granger(df: pd.DataFrame, outcome: str, cause: str) -> list[dict]:
    work = (
        df[[outcome, cause]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    rows = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tests = grangercausalitytests(work[[outcome, cause]], maxlag=MAX_LAG, verbose=False)
        for lag in range(1, MAX_LAG + 1):
            f_stat, p_value, *_ = tests[lag][0]["ssr_ftest"]
            rows.append({"lag": lag, "n": len(work), "f_stat": f_stat,
                         "p_value": p_value, "status": "OK"})
    except Exception as exc:
        for lag in range(1, MAX_LAG + 1):
            rows.append({"lag": lag, "n": len(work), "f_stat": np.nan,
                         "p_value": np.nan, "status": f"ERROR: {exc}"})
    return rows


def add_bh(
    df: pd.DataFrame,
    family_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    # 1. 분석 종류별 BH 보정
    df["p_value_bh"] = np.nan
    df["significant_bh_5pct"] = False

    for _, idx in df.groupby(
        family_cols,
        dropna=False,
    ).groups.items():
        valid_idx = [
            i
            for i in idx
            if pd.notna(df.at[i, "p_value"])
        ]

        if not valid_idx:
            continue

        reject, adjusted, _, _ = multipletests(
            df.loc[
                valid_idx,
                "p_value",
            ].astype(float),
            alpha=0.05,
            method="fdr_bh",
        )

        df.loc[
            valid_idx,
            "p_value_bh",
        ] = adjusted

        df.loc[
            valid_idx,
            "significant_bh_5pct",
        ] = reject

    # 2. 결과 파일 전체에 대한 전역 BH 보정
    df["p_value_bh_global"] = np.nan
    df["significant_bh_global_5pct"] = False

    global_valid = df["p_value"].notna()

    if global_valid.any():
        (
            global_reject,
            global_adjusted,
            _,
            _,
        ) = multipletests(
            df.loc[
                global_valid,
                "p_value",
            ].astype(float),
            alpha=0.05,
            method="fdr_bh",
        )

        df.loc[
            global_valid,
            "p_value_bh_global",
        ] = global_adjusted

        df.loc[
            global_valid,
            "significant_bh_global_5pct",
        ] = global_reject
    return df

def main() -> None:
    adf_rows, ols_rows, granger_rows, diagnostics = [], [], [], []

    for code, (label, market) in STOCKS.items():
        df = load_stock(code, label, market)
        controls = [f"{market}_change", "vix_change", "vkospi_change"]
        score_ffill = [f"{score}_ffill" for score in SCORES]

        for score in score_ffill:
            adf_rows.append({"stock_code": code, "variable": score, **run_adf(df[score])})

        # 같은 시간대 current/future의 상관과 VIF: 동시 투입 전 다중공선성 점검
        for session in ("intraday", "overnight"):
            cur = f"{session}_current_sentiment_index_ffill"
            fut = f"{session}_future_sentiment_index_ffill"
            pair = (
                df[[cur, fut]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
            )

            x = sm.add_constant(
                pair,
                has_constant="add",
            ).astype(float)
            diagnostics.append({
                "stock_code": code, "session": session, "n": len(pair),
                "current_future_correlation": pair[cur].corr(pair[fut]),
                "current_vif": variance_inflation_factor(x.values, 1),
                "future_vif": variance_inflation_factor(x.values, 2),
            })

        ols_specs = [
            # 장중-당일은 예측이 아니라 동시적 연관성입니다.
            ("intraday_return", ["intraday_current_sentiment_index_ffill"], "intraday_current_same_day_association"),
            ("intraday_return", ["intraday_future_sentiment_index_ffill"], "intraday_future_same_day_association"),
            # 장외는 당일 개장 전에 확정되므로 당일 결과의 선행변수로 해석 가능합니다.
            ("cc_return", ["overnight_current_sentiment_index_ffill"], "overnight_current_to_same_day_return"),
            ("cc_return", ["overnight_future_sentiment_index_ffill"], "overnight_future_to_same_day_return"),
            ("parkinson_volatility", ["overnight_current_sentiment_index_ffill"], "overnight_current_to_same_day_volatility"),
            ("parkinson_volatility", ["overnight_future_sentiment_index_ffill"], "overnight_future_to_same_day_volatility"),
            # 장중 점수의 진짜 예측력은 다음 거래일 결과로 별도 확인합니다.
            ("next_cc_return", ["intraday_current_sentiment_index_ffill"], "intraday_current_to_next_return"),
            ("next_cc_return", ["intraday_future_sentiment_index_ffill"], "intraday_future_to_next_return"),
            ("next_parkinson_volatility", ["intraday_current_sentiment_index_ffill"], "intraday_current_to_next_volatility"),
            ("next_parkinson_volatility", ["intraday_future_sentiment_index_ffill"], "intraday_future_to_next_volatility"),
            # future가 current를 넘어 추가 설명력을 갖는지 보는 공동 회귀(강건성).
            ("intraday_return", ["intraday_current_sentiment_index_ffill", "intraday_future_sentiment_index_ffill"], "intraday_current_future_joint_same_day"),
            ("cc_return", ["overnight_current_sentiment_index_ffill", "overnight_future_sentiment_index_ffill"], "overnight_current_future_joint_return"),
            ("parkinson_volatility", ["overnight_current_sentiment_index_ffill", "overnight_future_sentiment_index_ffill"], "overnight_current_future_joint_volatility"),
        ]
        for outcome, predictors, analysis in ols_specs:
            for result in fit_hac(df, outcome, predictors, controls):
                ols_rows.append({"stock_code": code, "analysis": analysis,
                                 "outcome": outcome, "controls": ",".join(controls), **result})

        for cause in score_ffill:
            for outcome in ("cc_return", "parkinson_volatility"):
                analysis = f"past_{cause}_to_{outcome}"
                for result in run_granger(df, outcome, cause):
                    granger_rows.append({"stock_code": code, "analysis": analysis,
                                         "outcome": outcome, "cause": cause, **result})

    adf = pd.DataFrame(adf_rows)
    ols = add_bh(pd.DataFrame(ols_rows), ["analysis"])
    granger = add_bh(pd.DataFrame(granger_rows), ["analysis"])
    diag = pd.DataFrame(diagnostics)

    adf.to_csv(OUTPUT_DIR / "current_future_adf.csv", index=False, encoding="utf-8-sig")
    ols.to_csv(OUTPUT_DIR / "current_future_hac_ols.csv", index=False, encoding="utf-8-sig")
    granger.to_csv(OUTPUT_DIR / "current_future_granger.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(OUTPUT_DIR / "current_future_diagnostics.csv", index=False, encoding="utf-8-sig")

    print("\n=== current/future 감성 분석 완료 ===")
    print(f"ADF: {len(adf)}건 / HAC OLS 계수: {len(ols)}건 / Granger: {len(granger)}건")
    print(f"결과 위치: {OUTPUT_DIR}")
    print("\nBH 보정 후 유의한 HAC OLS")
    print(ols.loc[ols["significant_bh_5pct"],
                  ["stock_code", "analysis", "predictor", "coefficient", "p_value", "p_value_bh"]].to_string(index=False))
    print("\nBH 보정 후 유의한 Granger")
    print(granger.loc[granger["significant_bh_5pct"],
                      ["stock_code", "analysis", "lag", "p_value", "p_value_bh"]].to_string(index=False))


if __name__ == "__main__":
    main()