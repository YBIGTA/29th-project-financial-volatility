"""
전체 파이프라인 한 번에 실행 (v10)
로컬(VS Code 등)에서 `python run_all.py`로 메인(6개월/2변수) 01~05번을 순서대로 실행.
부록(3개월/4변수, current/future)은 별도 스크립트라 여기 포함 안 됨 - 필요하면
`python 06_appendix_current_future.py`를 따로 실행.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_build_features.py",
    "02_garch_models.py",
    "03_xgboost_walkforward.py",
    "05_pooled_significance.py",
    "04_summary.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'#' * 70}\n# 실행: {step}\n{'#' * 70}\n")
        result = subprocess.run([sys.executable, str(SCRIPT_DIR / step)])
        if result.returncode != 0:
            print(f"\n[중단] {step} 실행 중 오류 발생 (returncode={result.returncode})")
            sys.exit(result.returncode)
    print("\n전체 파이프라인 실행 완료. 결과는 outputs/ 폴더에서 확인 가능:")
    print("  - features_*.csv                 : 종목별 피처 테이블")
    print("  - holdout_metrics_*_garch_{pk,5d}.json / *_xgb_{pk,5d}.json : 종목별 홀드아웃 성능")
    print("  - xgb_tuning_*.json               : XGBoost 종목별 최적 하이퍼파라미터")
    print("  - pooled_significance_results_v8.json : 4종목 통합 유의성 검정 최종 결과")
