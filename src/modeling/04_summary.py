"""
4단계: 4종목 x 4모델 결과 종합 요약 (v3)
- 02, 03번을 먼저 실행해서 holdout_metrics_*.json 파일이 생성된 뒤에 실행
"""
import json

VOL_DIR = "/home/claude/vol_project"
STOCKS = ["samsung_electronics", "sk_hynix", "kakao", "ecopro_bm"]

for stock_name in STOCKS:
    with open(f"{VOL_DIR}/holdout_metrics_{stock_name}_garch.json") as f:
        garch_metrics = json.load(f)
    with open(f"{VOL_DIR}/holdout_metrics_{stock_name}_xgb.json") as f:
        xgb_metrics = json.load(f)

    all_metrics = {**garch_metrics, **xgb_metrics}

    print("=" * 60)
    print(f"[{stock_name}] 최종 홀드아웃 성능 비교 (RMSE/MAE)")
    print("=" * 60)
    print(f"{'모델':<22}{'RMSE':>10}{'MAE':>10}{'n':>6}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<22}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['n']:>6}")
    print()

print("※ 튜닝 전 골격 단계 결과. 표본(홀드아웃 8일)이 작아 절대적인 모델 우열")
print("  판단은 이르며, XGBoost 하이퍼파라미터 튜닝 이후 재평가 필요.")
