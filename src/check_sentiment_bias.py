from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"

FILES = {
    "005930": "analysis_samsung_electronics.csv",
    "000660": "analysis_sk_hynix.csv",
    "247540": "analysis_ecopro_bm.csv",
    "035720": "analysis_kakao.csv",
}


rows = []

for stock_code, filename in FILES.items():
    path = ANALYSIS_DIR / filename
    df = pd.read_csv(path)

    scores = pd.to_numeric(
        df["intraday_sentiment_index"],
        errors="coerce",
    ).dropna()

    negative_count = int((scores < 0).sum())
    positive_count = int((scores > 0).sum())
    zero_count = int((scores == 0).sum())

    nonzero_count = (
        negative_count + positive_count
    )

    sign_test = binomtest(
        negative_count,
        nonzero_count,
        p=0.5,
        alternative="greater",
    )

    try:
        wilcoxon_result = wilcoxon(
            scores,
            alternative="less",
            zero_method="wilcox",
        )
        wilcoxon_p_value = (
            wilcoxon_result.pvalue
        )
    except ValueError:
        wilcoxon_p_value = float("nan")

    rows.append(
        {
            "stock_code": stock_code,
            "n": len(scores),
            "negative_count": negative_count,
            "positive_count": positive_count,
            "zero_count": zero_count,
            "negative_ratio": (
                negative_count / len(scores)
                if len(scores)
                else float("nan")
            ),
            "mean": scores.mean(),
            "median": scores.median(),
            "std": scores.std(),
            "sign_test_p_value": sign_test.pvalue,
            "wilcoxon_p_value": wilcoxon_p_value,
        }
    )


result = pd.DataFrame(rows)

output_path = (
    ANALYSIS_DIR
    / "statistics"
    / "intraday_sentiment_bias_check.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print(result.to_string(index=False))
print(f"\n저장 완료: {output_path}")