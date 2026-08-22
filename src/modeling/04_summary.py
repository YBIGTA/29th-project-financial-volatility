"""
4단계: 결과 종합 요약 (v9)
outputs/ 안에 흩어진 종목별 x Target(pk/5d)별 x 모델별 holdout 결과를 한 표로 정리.
02, 03, 05번 스크립트를 먼저 실행한 뒤 이 스크립트를 돌리면 됨.
"""
from pathlib import Path
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
VOL_DIR = SCRIPT_DIR / "outputs"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]
TARGETS = ["pk", "5d"]  # pk(Parkinson, 메인) 먼저, 5d(rolling_5d, 부록)


def load_json(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    rows = []
    for tname in TARGETS:
        for stock in STOCKS:
            g = load_json(VOL_DIR / f"holdout_metrics_{stock}_garch_{tname}.json")
            x = load_json(VOL_DIR / f"holdout_metrics_{stock}_xgb_{tname}.json")
            rows.append({
                "target": tname, "종목": stock,
                "GARCH 원본": round(g["GARCH 원본"]["RMSE"], 4),
                "GARCH+회귀(감성없음)": round(g["GARCH+회귀(감성없음)"]["RMSE"], 4),
                "GARCH+회귀(감성포함)": round(g["GARCH+회귀(감성포함)"]["RMSE"], 4),
                "GARCH+Ridge(감성없음)": round(g["GARCH+Ridge(감성없음)"]["RMSE"], 4),
                "GARCH+Ridge(감성포함)": round(g["GARCH+Ridge(감성포함)"]["RMSE"], 4),
                "XGBoost base": round(x["XGBoost base"]["RMSE"], 4),
                "XGBoost +sentiment": round(x["XGBoost +sentiment"]["RMSE"], 4),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(VOL_DIR / "summary_all_models.csv", index=False)
    print("=" * 100)
    print("전체 종목 x Target x 모델 홀드아웃 RMSE 요약 (Target=pk가 메인, 5d는 부록)")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print(f"\n-> {VOL_DIR / 'summary_all_models.csv'} 저장 완료")

    # 4종목 통합 유의성 검정 결과도 같이 출력 (05번 결과 재노출)
    pooled_path = VOL_DIR / "pooled_significance_results_v8.json"
    if pooled_path.exists():
        pooled = load_json(pooled_path)
        print("\n" + "=" * 100)
        print("4종목 통합(pooled) 유의성 검정 결과 (05번 결과)")
        print("=" * 100)
        for tname, r in pooled.items():
            print(f"\n[Target={tname}]")
            print(f"  GARCH 원본 vs 회귀(감성포함)      : p={r['garch_raw_vs_sentreg_pooled']['pooled_hac_t_p']:.3f}")
            print(f"  GARCH 순수 감성 기여도(감성없음 vs 있음): p={r['garch_base_vs_sentreg_pooled(순수감성기여도)']['pooled_hac_t_p']:.3f}")
            print(f"  XGBoost base vs +sentiment       : p={r['xgb_base_vs_sent_pooled']['pooled_hac_t_p']:.3f}")
    else:
        print("\n(05_pooled_significance.py를 먼저 실행하면 통합 검정 결과도 같이 출력됨)")
