"""
5단계: 통계검정 보정 - 날짜 정렬 pooled test + HAC 기반 DM test (v8)

[v7 -> v8 변경사항] (교수/선배 피드백 7,8,9번 반영)
- (기존 결함) 4종목 홀드아웃(60개)을 그냥 이어붙여 독립 표본처럼 검정했음
  -> 같은 날짜의 4종목은 같은 시장 충격을 공유하므로 진짜 독립 60개가 아님
  -> **날짜별로 4종목의 손실차이(표준화)를 먼저 평균낸 뒤, 그 15개 "날짜"를
     독립 단위로 취급해 검정**하는 방식으로 교체 (교차상관 문제 회피)
- (기존 결함) DM test가 예측오차 차이의 자기상관을 고려하지 않은 단순 t-test에
  가까웠음 -> **Newey-West(HAC, Bartlett kernel) 기반 long-run variance**로
  DM 통계량을 재계산. target_vol_5d는 5일 rolling window가 겹치는 구조라
  자기상관이 강할 수 있어 maxlag=4, target_vol_pk(1일 단위, 겹침 없음)는 maxlag=1 사용.
- Target A(5d)/B(pk) 둘 다 검정 (01/02/03번과 동일 구조)
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
VOL_DIR = SCRIPT_DIR / "outputs"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]
TARGETS = {"pk": 1, "5d": 4}  # v9: pk(메인) 먼저, 5d(부록)는 뒤에 - HAC maxlag는 그대로


def hac_long_run_var(d: np.ndarray, maxlag: int) -> float:
    """Newey-West(Bartlett kernel) long-run variance 추정."""
    n = len(d)
    d_c = d - d.mean()
    gamma0 = (d_c @ d_c) / n
    lrv = gamma0
    for lag in range(1, maxlag + 1):
        if lag >= n:
            break
        w = 1 - lag / (maxlag + 1)  # Bartlett 가중치
        gamma = (d_c[lag:] @ d_c[:-lag]) / n
        lrv += 2 * w * gamma
    return max(lrv, 1e-12)  # 수치적으로 음수/0 방지


def dm_test_hac(actual, pred1, pred2, maxlag: int) -> dict:
    actual = np.asarray(actual)
    e1 = actual - np.asarray(pred1)
    e2 = actual - np.asarray(pred2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_mean = d.mean()
    lrv = hac_long_run_var(d, maxlag)
    dm_stat = d_mean / np.sqrt(lrv / n)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n": n, "maxlag": maxlag}


def pooled_by_date_test(pairs: list, maxlag: int) -> dict:
    """pairs: [(actual, pred1, pred2), ...] 종목별 시리즈(같은 날짜 순서로 정렬됨).
    종목별로 표준화한 손실차이를 날짜별로 평균 -> 날짜를 독립 단위로 HAC t-test + 부호검정."""
    d_std_per_stock = []
    for actual, pred1, pred2 in pairs:
        actual = np.asarray(actual)
        e1 = actual - np.asarray(pred1)
        e2 = actual - np.asarray(pred2)
        d = e1 ** 2 - e2 ** 2
        std = d.std(ddof=1)
        d_std_per_stock.append(d / std if std > 0 else d * 0)

    d_by_date = np.mean(np.vstack(d_std_per_stock), axis=0)  # 날짜별 4종목 평균 (길이=15)
    n_dates = len(d_by_date)

    lrv = hac_long_run_var(d_by_date, maxlag)
    t_stat = d_by_date.mean() / np.sqrt(lrv / n_dates)
    t_p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_dates - 1))

    wins2 = int((d_by_date > 0).sum())
    sign_p = stats.binomtest(wins2, n_dates, 0.5).pvalue

    return {
        "n_dates": n_dates, "pooled_hac_t_stat": float(t_stat), "pooled_hac_t_p": float(t_p),
        "date_level_wins_for_model2": wins2, "sign_test_p": float(sign_p),
    }


if __name__ == "__main__":
    results = {}
    for tname, maxlag in TARGETS.items():
        garch_pairs, xgb_pairs = [], []
        garch_pairs_pure_sent = []  # 원본 회귀 보정 자체 vs 감성 추가분 순수 기여도 분리용
        for stock_name in STOCKS:
            g = pd.read_csv(VOL_DIR / f"garch_holdout_preds_{stock_name}_{tname}.csv")
            garch_pairs.append((g["actual"], g["pred_raw"], g["pred_ols_x"]))
            garch_pairs_pure_sent.append((g["actual"], g["pred_ols_base"], g["pred_ols_x"]))
            x = pd.read_csv(VOL_DIR / f"xgb_holdout_preds_{stock_name}_{tname}.csv")
            xgb_pairs.append((x["actual"], x["pred_base"], x["pred_sent"]))

        print("=" * 60)
        print(f"Target={tname} | [GARCH] 원본 vs OLS회귀(감성포함) - 날짜정렬 pooled (n_dates=15)")
        print("=" * 60)
        r1 = pooled_by_date_test(garch_pairs, maxlag)
        print(r1)

        print()
        print(f"Target={tname} | [GARCH] OLS회귀(감성없음) vs OLS회귀(감성포함) - 순수 감성 기여도, pooled")
        r1b = pooled_by_date_test(garch_pairs_pure_sent, maxlag)
        print(r1b)

        print()
        print(f"Target={tname} | [XGBoost] base vs +sentiment - 날짜정렬 pooled (n_dates=15)")
        r2 = pooled_by_date_test(xgb_pairs, maxlag)
        print(r2)
        print()

        # 종목별 HAC-DM test도 함께 갱신 (참고용)
        garch_dm_per_stock = {s: dm_test_hac(*p, maxlag) for s, p in zip(STOCKS, garch_pairs)}
        garch_dm_pure_sent_per_stock = {s: dm_test_hac(*p, maxlag) for s, p in zip(STOCKS, garch_pairs_pure_sent)}
        xgb_dm_per_stock = {s: dm_test_hac(*p, maxlag) for s, p in zip(STOCKS, xgb_pairs)}

        results[tname] = {
            "garch_raw_vs_sentreg_pooled": r1,
            "garch_base_vs_sentreg_pooled(순수감성기여도)": r1b,
            "xgb_base_vs_sent_pooled": r2,
            "garch_dm_per_stock_hac": garch_dm_per_stock,
            "garch_dm_pure_sent_per_stock_hac": garch_dm_pure_sent_per_stock,
            "xgb_dm_per_stock_hac": xgb_dm_per_stock,
        }

    with open(VOL_DIR / "pooled_significance_results_v8.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Target A/B 통계검정 보정 완료 ->", VOL_DIR / "pooled_significance_results_v8.json")
