# 변동성 예측 모델링 - v9 최종본 (Parkinson을 메인 target으로 확정)

대상 종목: **삼성전자, SK하이닉스, 카카오, 에코프로비엠 (4종목 전체)**

## 로컬(VS Code)에서 실행하는 법

```bash
pip install -r requirements.txt
python run_all.py    # 01~05번을 순서대로 한 번에 실행 (튜닝까지 자동 포함)
```

또는 단계별로 직접 실행하고 싶으면:
```bash
python 01_build_features.py      # data/ 안의 원본 csv -> outputs/features_*.csv
python 02_garch_models.py        # GARCH 3단계 + Ridge (Target pk/5d 둘 다)
python 03_xgboost_walkforward.py # XGBoost - GridSearchCV+TimeSeriesSplit 튜닝 자동 포함 (Target pk/5d 둘 다)
python 05_pooled_significance.py # 4종목 통합 유의성 검정 (Target pk/5d 둘 다)
```

`data/` 폴더에 원본 csv 5개(`market_index_merged.csv`, `merged_daily_{종목}.csv` x4)가 이미 들어있음. 경로는 전부 스크립트 위치 기준 상대경로라 폴더째로 옮겨도 그대로 동작함. **XGBoost 튜닝은 `03_xgboost_walkforward.py`의 `PARAM_GRID`에서 `GridSearchCV`로 이미 자동 수행되므로, 스크립트를 실행하면 튜닝까지 한 번에 끝남** — 별도로 튜닝 단계를 추가할 필요 없음(그리드를 더 넓히고 싶으면 `PARAM_GRID` 값만 조정하면 됨).

## Target 최종 확정: Parkinson (`target_vol_pk`)를 메인으로

`rolling_volatility_5d`(Target A)와 `Parkinson volatility`(Target B) 두 후보를 동일 조건으로 병렬 비교한 결과(v8, 아래 5번 섹션), **Parkinson을 메인 target으로 확정**함. 근거:

1. **방법론(주된 근거)**: GARCH(1,1)은 원래 1일-앞 조건부분산을 추정하는 모델이라, 이를 채점할 target도 당일 단위(daily) 지표여야 함. `rolling_5d`는 t~t+1이 4/5일을 공유해 사실상 과거 값을 거의 그대로 베끼는 구조(자기상관 매우 강함)라 GARCH 평가 target으로는 부적합. Parkinson(당일 고가/저가 기반)은 학계·실무에서 분봉 데이터 없이 쓸 수 있는 표준적인 range-based 변동성 추정량이며, 연구질문의 "장중 변동성"과도 정의상 직접 대응됨.
2. **결과(부차적 근거)**: 실제로 Parkinson target에서 "GARCH 원본 vs 회귀보정" 차이가 통계적으로 유의하게(p=0.006) 나와, GARCH의 스케일 재보정 필요성이 명확히 확인됨(방법론적 선택이 타당했음을 뒷받침).

`rolling_5d`(Target A)는 폐기하지 않고 **부록(로버스트니스 체크)**으로 계속 산출/보고함 (`TARGETS` dict에서 `5d` 키로 유지).

## 변경사항

### 1. 감성 시간정렬 오류

원래 연구질문: **"장외 감성 → 당일(t) 장중 변동성"**, **"장중 감성 → 익일(t+1) 장중 변동성"**. 그런데 기존 코드(v7까지)는 장외/장중 감성 둘 다 shift 없이 그대로 써서, 사실상 **장외 감성도 익일(t+1)을 예측하는 구조**가 되어 있었음 — 당일 예측이어야 하는데 하루 밀려 있었던 것.

```python
sentiment_intraday = intraday_sentiment_index          # 그대로 (원래도 맞음, t일 마감에 확정 -> t+1 예측에 사용 가능)
sentiment_overnight = overnight_sentiment_index.shift(-1)  # (v8 수정) t+1일 개장 직전 확정되는 값으로 정렬
```

이렇게 하면 `target_vol[t] = vol[t+1]`이라는 통일된 target 구조를 유지하면서도, "장외 감성은 예측 대상일 당일 개장 전 정보"라는 원래 연구질문의 인과구조와 정확히 일치함.

**이 버그 수정 이후 SK하이닉스의 overnight 감성 계수가 두 target 모두 p<0.005로 유의하게 나옴** — 원래 잘못된 시간정렬 때문에 신호가 묻혀 있었을 가능성이 있음.

### 2. Target A/B 병렬 비교 구조

- **Target A (`target_vol_5d`)**: `rolling_volatility_5d` 기준, 팀 기존 합의안 (close-to-close 5일 이동표준편차)
- **Target B (`target_vol_pk`)**: Parkinson volatility 기준 (`|ln(High/Low)| / sqrt(4ln2)`, 당일 고가/저가 기반 "장중" 변동성)

두 target 모두 **동일한 피처셋**(realized_vol 계열, parkinson_vol 자체는 피처에서 제외 - Target B용으로 쓸 때 자기 정보 누수 방지)으로 GARCH/XGBoost 양쪽에서 나란히 모델링. 결과 좋은 쪽을 고르는 게 아니라, 연구질문과의 정합성 + 통계적 근거를 같이 보고 최종 target을 정하기 위함.

### 3. 통계검정 보정

- **Pooled test 재설계**: 기존엔 4종목 홀드아웃(60개)을 그냥 이어붙여 독립표본처럼 검정했는데, 같은 날짜의 4종목은 같은 시장충격을 공유해 진짜 독립이 아님. → **날짜별로 4종목 손실차이(표준화)를 먼저 평균낸 뒤, 그 15개 날짜를 독립 단위로 검정**.
- **DM test HAC 보정**: 기존 DM test는 예측오차 차이의 자기상관을 고려 안 한 단순 t-test에 가까웠음. Target A(5d)는 rolling window가 4일씩 겹치는 구조라 자기상관이 있을 수밖에 없어서 → **Newey-West(Bartlett kernel) HAC 분산으로 재계산**(5d: maxlag=4, pk: maxlag=1).
- **회귀보정 자체 효과 vs 순수 감성 기여도 분리**: "GARCH 원본 vs 회귀(감성포함)"만 보면 회귀보정 자체의 효과(스케일 재보정)와 감성지수의 순수 기여도가 섞여 보임 → "회귀(감성없음) vs 회귀(감성포함)" 비교를 추가해 감성지수만의 순수 기여도를 분리.

## 0. XGBoost 하이퍼파라미터 탐색범위 검토

train_val(87행)에 `TimeSeriesSplit(5)`를 그대로 적용하면 **첫 CV fold의 학습표본이 17개뿐**이라 하이퍼파라미터 선택이 노이즈에 가까워지는 문제가 있었음. 또한 탐색범위 자체도 `max_depth/n_estimators/learning_rate/subsample` 4개뿐이라, 87행 규모에서 더 중요할 수 있는 규제(regularization) 파라미터가 빠져 있었음.

**수정 내용**
- CV: `TimeSeriesSplit(n_splits=4, test_size=10)`로 변경 — 첫 fold부터 학습표본 47개 확보
- 탐색범위에 `min_child_weight`(리프 과세분화 방지), `colsample_bytree`(상관된 `realized_vol_lag1~3` 피처가 많아 트리마다 일부만 사용해 과적합 억제), `reg_lambda`(L2 규제) 추가
- 조합 수가 10,800개로 늘어 전수탐색(GridSearchCV) 대신 `RandomizedSearchCV(n_iter=150)`로 전환

**결과에 미친 영향**: 결론(유의성 없음) 자체는 바뀌지 않았지만, XGBoost의 base vs +sentiment 우열 방향이 미묘하게 바뀜(Target B 기준 이전엔 +sentiment 근소 우위 → 이번엔 base 근소 우위, 둘 다 비유의). 이는 애초 탐색이 좁아서 방향성 자체가 안정적이지 않았을 가능성을 보여주는 정황이기도 함 — 표본 부족 문제와 별개로 탐색범위 점검이 유의미했음.

## 1. 타겟(예측 대상) — Target A/B

```
log_ret = log(오늘종가 / 어제종가)
realized_vol = log_ret의 5일 이동표준편차
parkinson_vol = |ln(당일고가/당일저가)| / sqrt(4·ln2)

target_vol_5d[t] = realized_vol[t+1]    # Target A - close-to-close 다일간 지표
target_vol_pk[t] = parkinson_vol[t+1]   # Target B - 당일 고가/저가 기반 "장중" 지표
```

## 2. 피처(입력 변수) — Target A/B 공통

| 카테고리 | 피처 | 용도 |
|---|---|---|
| 자기 과거 정보 | `realized_vol`, `realized_vol_lag1~3`, `log_ret_lag1` | 변동성의 "관성" 반영 (GARCH는 모델 구조 자체에 내장, XGBoost만 필요) |
| 시장 배경 정보 | `vkospi_change`, `vkospi_change_lag1`, `kospi_ret`, `kosdaq_ret` (전부 로그수익률, ADF 정상성 확인됨) | 시장 전체 분위기를 통제해야 "감성지수만의 순수 기여도"를 볼 수 있음 |
| 감성지수 (핵심 검증 대상) | `sentiment_intraday`(당일, shift 없음), `sentiment_overnight`(t+1일 개장 전, shift(-1)) | 우리가 실제로 검증하려는 재료 (v8: 시간정렬 버그 수정됨) |

## 3. GARCH 3단계 + Ridge 대안 (`02_garch_models.py`)

| 단계 | 설명 |
|---|---|
| [1] GARCH 원본 | GARCH(1,1) walk-forward 예측값을 그대로 사용 (회귀 보정 없음) |
| [2] GARCH+OLS회귀(감성없음) | `target ~ garch_vol` OLS |
| [3] GARCH+OLS회귀(감성포함) | `target ~ garch_vol + sentiment_intraday + sentiment_overnight` OLS |
| [2b]/[3b] Ridge 버전 | OLS 대신 RidgeCV(TimeSeriesSplit)로 표본 부족에 따른 계수 불안정 여부 확인용 |

## 4. XGBoost base / +sentiment (`03_xgboost_walkforward.py`)

`GridSearchCV(cv=TimeSeriesSplit(5))`로 종목×피처셋×target 조합마다 독립적으로 `max_depth/n_estimators/learning_rate/subsample` 튜닝 (holdout은 튜닝에 사용 안 함). 튜닝된 파라미터로 walk-forward 재실행 + holdout 최종 평가.

## 5. 모델 평가 지표 결과 (`04_summary.py`)

4종목 × 7개 모델(GARCH 3단계 + Ridge 2종 + XGBoost 2종)의 홀드아웃(15일) RMSE 결과. Target은 최종 확정한 **Parkinson(`pk`)** 기준만 제시함 (`5d`는 부록으로 `outputs/summary_all_models.csv`에 별도 보관).

| 종목 | GARCH 원본 | GARCH+회귀<br>(감성없음) | GARCH+회귀<br>(감성포함) | GARCH+Ridge<br>(감성없음) | GARCH+Ridge<br>(감성포함) | XGBoost<br>base | XGBoost<br>+sentiment |
|---|---|---|---|---|---|---|---|
| 삼성전자 | 0.0502 | 0.0265 | **0.0249** | 0.0265 | 0.0265 | 0.0266 | 0.0276 |
| SK하이닉스 | 0.0611 | 0.0336 | 0.0306 | 0.0326 | 0.0306 | **0.0270** | 0.0275 |
| 카카오 | **0.0091** | 0.0101 | 0.0113 | 0.0098 | 0.0097 | 0.0100 | 0.0105 |
| 에코프로비엠 | 0.0265 | **0.0201** | 0.0207 | 0.0202 | 0.0202 | 0.0219 | 0.0218 |

(RMSE 기준, 굵게 표시 = 종목별 7개 모델 중 최저)

**해석**

- 어느 한 모델이 4종목 전부를 지배하지 않음 — 삼성전자/SK하이닉스는 회귀보정·XGBoost가 GARCH 원본보다 뚜렷이 낮지만, 카카오/에코프로비엠은 GARCH 원본(또는 감성 없는 회귀)이 오히려 낮음. 종목마다 최적 모델이 갈린다는 것 자체가 이 표(점추정치 비교)만으로는 "어떤 모델이 낫다"는 일반적 결론을 내리기 어렵다는 뜻이며, 그래서 통계적 유의성 검정(6번 섹션, `05_pooled_significance.py`)이 별도로 필요함.
- 삼성전자·SK하이닉스에서 GARCH 원본과 회귀보정 간 격차가 큰 것은 6번 섹션에서 확인한 "GARCH 원본이 추정하는 close-to-close 조건부분산과 target(Parkinson, 장중 range 기반)의 스케일 불일치를 회귀가 재보정해주는 효과"와 일치하는 패턴임.
- 이 표는 holdout 15일 하나의 스냅샷(점추정치)이며, 이 차이가 통계적으로 유의한지는 별도로 검정해야 함 — 해당 검정 결과는 6번 섹션 참고.

## 6. 최종 유의성 검정 결과 (`05_pooled_significance.py`)

| 비교 (4종목 통합, 날짜정렬 n=15) | Target A (5d) | Target B (Parkinson) |
|---|---|---|
| GARCH 원본 vs 회귀(감성포함) | p=0.149 | **p=0.006** (13/15일 회귀 우위) |
| GARCH 회귀(감성없음) vs 회귀(감성포함) — **순수 감성 기여도** | p=0.072 | p=0.847 (7/15일) |
| XGBoost base vs +sentiment (v10: walk-forward holdout) | p=0.664 | p=0.071 (5/15일 +sentiment 우위) |

### 핵심 해석

**Target B(Parkinson)에서 "GARCH 원본 vs 회귀(감성포함)"가 유의(p=0.006)하게 나온 건 감성지수 덕분이 아니라 회귀보정 자체의 효과다.** GARCH(1,1)은 close-to-close 조건부분산을 추정하는데, target을 Parkinson(장중 range 기반)으로 바꾸면 스케일/정의가 어긋나서 원본을 그대로 쓰면 불리해짐 — 이걸 회귀로 재보정하니 크게 개선된 것. 감성지수를 넣었는지 여부(순수 기여도 검정, p=0.847)는 이 개선과 거의 무관함.

**즉, Target A/B, GARCH/XGBoost, 순수 감성 기여도까지 전부 따져봐도 "감성지수가 변동성 예측력을 통계적으로 유의하게 개선한다"는 근거는 현재 데이터에서 발견되지 않았다.** 이건 회귀 방식이나 target 정의 같은 모델링 설계의 결함 때문이 아니라(설계는 이번 점검으로 상당 부분 검증됨), **표본 규모(6개월, 종목당 102관측치, 홀드아웃 15일)의 한계**로 보는 게 타당하다.

## 7. 현재까지 점검 완료 / 남은 한계

**점검 완료 (설계 결함 아님을 확인)**
- 감성 시간정렬 버그 수정 완료 (v8)
- Target A/B 병렬 비교로 target 정의 문제가 원인이 아님을 확인 (오히려 Target B에서는 GARCH 회귀보정의 필요성만 재확인됨, 감성지수 문제와는 별개)
- Pooled test/DM test 통계적 결함(교차상관, 자기상관 미보정) 수정 완료 — 보정 후에도 결론 동일
- Ridge 비교로 OLS 계수 불안정성 가능성 점검 (뚜렷한 개선 없음)
- 데이터 leakage 없음 확인 (holdout 분리, GARCH expanding window)
- XGBoost 하이퍼파라미터 탐색범위 확장(min_child_weight/colsample_bytree/reg_lambda 추가) + CV fold 크기 고정 (v9)
- **XGBoost holdout 평가를 GARCH와 동일한 walk-forward 방식으로 통일 (v10)** — 튜닝된 파라미터는 고정하고, holdout 15일도 매일 재학습하며 평가하도록 변경. 결론(유의성 없음)은 v9와 동일하게 유지됨

**남은 한계 (발표에 명시 권장)**
- 표본 규모(6개월, 홀드아웃 15일)가 작아 검정력이 제한적 — 실제로 작은 효과가 있어도 통계적으로 검출하기 어려울 수 있음

- XGBoost 하이퍼파라미터 튜닝이 각 walk-forward 시점마다 재탐색하는 nested/rolling tuning이 아니라, train_val 구간 전체로 한 번만 튜닝 — 완전한 온라인 재현은 아니지만 holdout 자체를 무효화할 정도는 아님
- 향후 과제: 데이터 기간 확대(1년 이상), 감성지수 구성 방식(집계 방법, 시차 구조) 재검토
