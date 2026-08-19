from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
STATISTICS_DIR = ANALYSIS_DIR / "statistics"
FIGURE_DIR = ANALYSIS_DIR / "figures"

LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95

EVENT_WINDOW_START = -5
EVENT_WINDOW_END = 5

ESTIMATION_WINDOW = 60
ESTIMATION_GAP = 5
MIN_ESTIMATION_OBS = 30
MIN_EVENT_SPACING = 5

STOCK_CONFIG = {
    "005930": {
        "name": "Samsung Electronics",
        "label": "samsung_electronics",
        "market_return": "kospi_change",
    },
    "000660": {
        "name": "SK hynix",
        "label": "sk_hynix",
        "market_return": "kospi_change",
    },
    "247540": {
        "name": "EcoPro BM",
        "label": "ecopro_bm",
        "market_return": "kosdaq_change",
    },
    "035720": {
        "name": "Kakao",
        "label": "kakao",
        "market_return": "kospi_change",
    },
}

SESSION_CONFIG = {
    "overnight": {
        "sentiment_column": "overnight_sentiment_ffill",
        "car_windows": [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 5),
        ],
    },
    "intraday": {
        "sentiment_column": "intraday_sentiment_ffill",
        "car_windows": [
            (1, 1),
            (1, 3),
            (1, 5),
        ],
    },
}


def select_nonoverlapping_events(
    df: pd.DataFrame,
    sentiment_column: str,
) -> list[dict]:
    sentiment = pd.to_numeric(
        df[sentiment_column],
        errors="coerce",
    )

    low_threshold = sentiment.quantile(
        LOW_QUANTILE
    )
    high_threshold = sentiment.quantile(
        HIGH_QUANTILE
    )

    candidates = []

    for position, value in sentiment.items():
        if pd.isna(value):
            continue

        if value <= low_threshold:
            candidates.append(
                {
                    "position": int(position),
                    "direction": "negative",
                    "sentiment": float(value),
                    "extremeness": float(-value),
                    "threshold": float(
                        low_threshold
                    ),
                }
            )

        elif value >= high_threshold:
            candidates.append(
                {
                    "position": int(position),
                    "direction": "positive",
                    "sentiment": float(value),
                    "extremeness": float(value),
                    "threshold": float(
                        high_threshold
                    ),
                }
            )

    selected = []

    for direction in ["negative", "positive"]:
        direction_candidates = [
            row
            for row in candidates
            if row["direction"] == direction
        ]

        direction_candidates = sorted(
            direction_candidates,
            key=lambda row: row["extremeness"],
            reverse=True,
        )

        chosen_positions = []

        for candidate in direction_candidates:
            position = candidate["position"]

            overlaps = any(
                abs(position - selected_position)
                <= MIN_EVENT_SPACING
                for selected_position
                in chosen_positions
            )

            if not overlaps:
                selected.append(candidate)
                chosen_positions.append(position)

    return sorted(
        selected,
        key=lambda row: row["position"],
    )


def estimate_market_model(
    df: pd.DataFrame,
    event_position: int,
    market_return_column: str,
):
    estimation_end = (
        event_position - ESTIMATION_GAP
    )

    estimation_start = max(
        0,
        estimation_end - ESTIMATION_WINDOW,
    )

    estimation = df.iloc[
        estimation_start:estimation_end
    ][
        [
            "cc_return",
            market_return_column,
        ]
    ].copy()

    estimation = (
        estimation
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(estimation) < MIN_ESTIMATION_OBS:
        return None

    x = sm.add_constant(
        estimation[market_return_column],
        has_constant="add",
    )

    model = sm.OLS(
        estimation["cc_return"],
        x,
    ).fit()

    return {
        "alpha": float(model.params["const"]),
        "beta": float(
            model.params[market_return_column]
        ),
        "estimation_n": int(model.nobs),
        "r_squared": float(model.rsquared),
    }


def calculate_event_ar(
    df: pd.DataFrame,
    event_position: int,
    event_id: str,
    market_return_column: str,
    market_model: dict,
) -> list[dict]:
    rows = []

    for tau in range(
        EVENT_WINDOW_START,
        EVENT_WINDOW_END + 1,
    ):
        position = event_position + tau

        if position < 0 or position >= len(df):
            continue

        row = df.iloc[position]

        stock_return = pd.to_numeric(
            pd.Series([row["cc_return"]]),
            errors="coerce",
        ).iloc[0]

        market_return = pd.to_numeric(
            pd.Series(
                [row[market_return_column]]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            pd.isna(stock_return)
            or pd.isna(market_return)
        ):
            continue

        expected_return = (
            market_model["alpha"]
            + market_model["beta"]
            * market_return
        )

        abnormal_return = (
            stock_return - expected_return
        )

        rows.append(
            {
                "event_id": event_id,
                "tau": tau,
                "date": row["date"],
                "stock_return": stock_return,
                "market_return": market_return,
                "expected_return": expected_return,
                "abnormal_return": abnormal_return,
            }
        )

    return rows


def build_event_study():
    event_rows = []
    daily_ar_rows = []

    event_number = 0

    for stock_code, stock_info in (
        STOCK_CONFIG.items()
    ):
        path = (
            ANALYSIS_DIR
            / f"analysis_{stock_info['label']}.csv"
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

        for session, session_info in (
            SESSION_CONFIG.items()
        ):
            sentiment_column = session_info[
                "sentiment_column"
            ]

            events = select_nonoverlapping_events(
                df,
                sentiment_column,
            )

            for event in events:
                event_position = event["position"]

                market_model = estimate_market_model(
                    df=df,
                    event_position=event_position,
                    market_return_column=(
                        stock_info["market_return"]
                    ),
                )

                if market_model is None:
                    continue

                event_number += 1
                event_id = (
                    f"{stock_code}_"
                    f"{session}_"
                    f"{event['direction']}_"
                    f"{event_number}"
                )

                event_date = df.iloc[
                    event_position
                ]["date"]

                event_rows.append(
                    {
                        "event_id": event_id,
                        "stock_code": stock_code,
                        "stock_name": stock_info[
                            "name"
                        ],
                        "session": session,
                        "direction": event[
                            "direction"
                        ],
                        "event_date": event_date,
                        "sentiment": event[
                            "sentiment"
                        ],
                        "threshold": event[
                            "threshold"
                        ],
                        "market_return_column": (
                            stock_info[
                                "market_return"
                            ]
                        ),
                        **market_model,
                    }
                )

                ar_rows = calculate_event_ar(
                    df=df,
                    event_position=event_position,
                    event_id=event_id,
                    market_return_column=(
                        stock_info["market_return"]
                    ),
                    market_model=market_model,
                )

                for ar_row in ar_rows:
                    ar_row.update(
                        {
                            "stock_code": stock_code,
                            "stock_name": (
                                stock_info["name"]
                            ),
                            "session": session,
                            "direction": event[
                                "direction"
                            ],
                            "event_date": event_date,
                        }
                    )

                daily_ar_rows.extend(ar_rows)

    events_df = pd.DataFrame(event_rows)
    daily_ar_df = pd.DataFrame(daily_ar_rows)

    return events_df, daily_ar_df


def safe_wilcoxon(values: pd.Series):
    if len(values) < 2:
        return np.nan

    if np.allclose(values, 0):
        return 1.0

    try:
        return float(
            wilcoxon(values).pvalue
        )
    except ValueError:
        return np.nan


def summarize_car(
    events_df: pd.DataFrame,
    daily_ar_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    if events_df.empty or daily_ar_df.empty:
        return pd.DataFrame()

    group_columns = [
        "stock_code",
        "stock_name",
        "session",
        "direction",
    ]

    for group_values, events in (
        events_df.groupby(group_columns)
    ):
        (
            stock_code,
            stock_name,
            session,
            direction,
        ) = group_values

        windows = SESSION_CONFIG[
            session
        ]["car_windows"]

        event_ids = events["event_id"].tolist()

        event_daily = daily_ar_df[
            daily_ar_df["event_id"].isin(
                event_ids
            )
        ]

        for window_start, window_end in windows:
            expected_length = (
                window_end
                - window_start
                + 1
            )

            window_data = event_daily[
                event_daily["tau"].between(
                    window_start,
                    window_end,
                )
            ]

            car_by_event = (
                window_data
                .groupby("event_id")
                ["abnormal_return"]
                .agg(["sum", "count"])
                .reset_index()
            )

            car_by_event = car_by_event[
                car_by_event["count"]
                == expected_length
            ]

            car_values = car_by_event["sum"]

            if len(car_values) >= 2:
                t_result = ttest_1samp(
                    car_values,
                    popmean=0,
                    nan_policy="omit",
                )

                t_stat = float(
                    t_result.statistic
                )
                t_p_value = float(
                    t_result.pvalue
                )
            else:
                t_stat = np.nan
                t_p_value = np.nan

            summary_rows.append(
                {
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "session": session,
                    "direction": direction,
                    "window": (
                        f"[{window_start},"
                        f"{window_end}]"
                    ),
                    "window_start": window_start,
                    "window_end": window_end,
                    "event_n": len(car_values),
                    "mean_car": (
                        float(car_values.mean())
                        if len(car_values) > 0
                        else np.nan
                    ),
                    "median_car": (
                        float(car_values.median())
                        if len(car_values) > 0
                        else np.nan
                    ),
                    "std_car": (
                        float(car_values.std(ddof=1))
                        if len(car_values) > 1
                        else np.nan
                    ),
                    "t_stat": t_stat,
                    "t_p_value": t_p_value,
                    "wilcoxon_p_value": (
                        safe_wilcoxon(car_values)
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    valid = summary_df[
        "t_p_value"
    ].notna()

    summary_df["t_p_value_bh"] = pd.NA
    summary_df["significant_bh_5pct"] = False

    if valid.any():
        reject, adjusted, _, _ = multipletests(
            summary_df.loc[
                valid,
                "t_p_value",
            ],
            alpha=0.05,
            method="fdr_bh",
        )

        summary_df.loc[
            valid,
            "t_p_value_bh",
        ] = adjusted

        summary_df.loc[
            valid,
            "significant_bh_5pct",
        ] = reject

    return summary_df


def make_average_car_plot(
    daily_ar_df: pd.DataFrame,
) -> None:
    if daily_ar_df.empty:
        return

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        sharex=True,
    )

    axes = axes.flatten()

    colors = {
        ("overnight", "positive"): "#7B2CBF",
        ("overnight", "negative"): "#C77DFF",
        ("intraday", "positive"): "#0057B8",
        ("intraday", "negative"): "#64A5FF",
    }

    for axis, (
        stock_code,
        stock_info,
    ) in zip(
        axes,
        STOCK_CONFIG.items(),
    ):
        stock_data = daily_ar_df[
            daily_ar_df["stock_code"]
            .astype(str)
            .str.zfill(6)
            == stock_code
        ]

        for (
            session,
            direction,
        ), group in stock_data.groupby(
            ["session", "direction"]
        ):
            average_ar = (
                group.groupby("tau")
                ["abnormal_return"]
                .mean()
                .sort_index()
            )

            average_car = average_ar.cumsum()

            axis.plot(
                average_car.index,
                average_car.values,
                marker="o",
                label=(
                    f"{session}-"
                    f"{direction}"
                ),
                color=colors[
                    (session, direction)
                ],
            )

        axis.axhline(
            0,
            color="black",
            linewidth=0.8,
        )

        axis.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=0.8,
        )

        axis.set_title(stock_info["name"])
        axis.set_xlabel("Event time")
        axis.set_ylabel("Average CAR")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    fig.suptitle(
        "Average Cumulative Abnormal Return",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "event_study_average_car.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"그래프 저장: {output_path}")


def main() -> None:
    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events_df, daily_ar_df = (
        build_event_study()
    )

    summary_df = summarize_car(
        events_df,
        daily_ar_df,
    )

    events_path = (
        STATISTICS_DIR
        / "event_study_events.csv"
    )

    daily_path = (
        STATISTICS_DIR
        / "event_study_daily_ar.csv"
    )

    summary_path = (
        STATISTICS_DIR
        / "event_study_summary.csv"
    )

    events_df.to_csv(
        events_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_ar_df.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    make_average_car_plot(daily_ar_df)

    print("\n=== Event Study 완료 ===")
    print(f"선택된 이벤트 수: {len(events_df)}")
    print(f"일별 AR 행 수: {len(daily_ar_df)}")
    print(f"이벤트 목록: {events_path}")
    print(f"일별 AR: {daily_path}")
    print(f"CAR 요약: {summary_path}")

    if not summary_df.empty:
        print("\n=== 보정 전 유의 결과 ===")

        raw_significant = summary_df[
            summary_df["t_p_value"] < 0.05
        ]

        if raw_significant.empty:
            print("없음")
        else:
            print(
                raw_significant[
                    [
                        "stock_code",
                        "session",
                        "direction",
                        "window",
                        "event_n",
                        "mean_car",
                        "t_p_value",
                    ]
                ].to_string(index=False)
            )

        print("\n=== BH 보정 후 유의 결과 ===")

        adjusted_significant = summary_df[
            summary_df[
                "significant_bh_5pct"
            ] == True
        ]

        if adjusted_significant.empty:
            print("없음")
        else:
            print(
                adjusted_significant[
                    [
                        "stock_code",
                        "session",
                        "direction",
                        "window",
                        "event_n",
                        "mean_car",
                        "t_p_value",
                        "t_p_value_bh",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()