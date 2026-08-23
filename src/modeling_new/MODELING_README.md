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

## v8 변경사항 (교수/선배 피드백 반영)

### 1. (버그 수정, 최우선) 감성 시간정렬 오류

원래 연구질문: **"장외 감성 → 당일(t) 장중 변동성"**, **"장중 감성 → 익일(t+1) 장중 변동성"**. 그런데 기존 코드(v7까지)는 장외/장중 감성 둘 다 shift 없이 그대로 써서, 사실상 **장외 감성도 익일(t+1)을 예측하는 구조**가 되어 있었음 — 당일 예측이어야 하는데 하루 밀려 있었던 것.

```python
sentiment_intraday = intraday_sentiment_index          # 그대로 (원래도 맞음, t일 마감에 확정 -> t+1 예측에 사용 가능)
sentiment_overnight = overnight_sentiment_index.shift(-1)  # (v8 수정) t+1일 개장 직전 확정되는 값으로 정렬
```

이렇게 하면 `target_vol[t] = vol[t+1]`이라는 통일된 target 구조를 유지하면서도, "장외 감성은 예측 대상일 당일 개장 전 정보"라는 원래 연구질문의 인과구조와 정확히 일치함.

**이 버그 수정 이후 SK하이닉스의 overnight 감성 계수가 두 target 모두 p<0.005로 유의하게 나옴** — 원래 잘못된 시간정렬 때문에 신호가 묻혀 있었을 가능성이 있음.

### 2. Target A/B 병렬 비교 구조 (target을 교체하지 않고 둘 다 산출)

- **Target A (`target_vol_5d`)**: `rolling_volatility_5d` 기준, 팀 기존 합의안 (close-to-close 5일 이동표준편차)
- **Target B (`target_vol_pk`)**: Parkinson volatility 기준 (`|ln(High/Low)| / sqrt(4ln2)`, 당일 고가/저가 기반 "장중" 변동성)

두 target 모두 **동일한 피처셋**(realized_vol 계열, parkinson_vol 자체는 피처에서 제외 - Target B용으로 쓸 때 자기 정보 누수 방지)으로 GARCH/XGBoost 양쪽에서 나란히 모델링. 결과 좋은 쪽을 고르는 게 아니라, 연구질문과의 정합성 + 통계적 근거를 같이 보고 최종 target을 정하기 위함.

### 3. 통계검정 보정

- **Pooled test 재설계**: 기존엔 4종목 홀드아웃(60개)을 그냥 이어붙여 독립표본처럼 검정했는데, 같은 날짜의 4종목은 같은 시장충격을 공유해 진짜 독립이 아님. → **날짜별로 4종목 손실차이(표준화)를 먼저 평균낸 뒤, 그 15개 날짜를 독립 단위로 검정**.
- **DM test HAC 보정**: 기존 DM test는 예측오차 차이의 자기상관을 고려 안 한 단순 t-test에 가까웠음. Target A(5d)는 rolling window가 4일씩 겹치는 구조라 자기상관이 있을 수밖에 없어서 → **Newey-West(Bartlett kernel) HAC 분산으로 재계산**(5d: maxlag=4, pk: maxlag=1).
- **회귀보정 자체 효과 vs 순수 감성 기여도 분리**: "GARCH 원본 vs 회귀(감성포함)"만 보면 회귀보정 자체의 효과(스케일 재보정)와 감성지수의 순수 기여도가 섞여 보임 → "회귀(감성없음) vs 회귀(감성포함)" 비교를 추가해 감성지수만의 순수 기여도를 분리.

## 0. XGBoost 하이퍼파라미터 탐색범위 검토 (v9)

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

## 5. 최종 유의성 검정 결과 (`05_pooled_significance.py`)

| 비교 (4종목 통합, 날짜정렬 n=15) | Target A (5d) | Target B (Parkinson) |
|---|---|---|
| GARCH 원본 vs 회귀(감성포함) | p=0.149 | **p=0.006** (13/15일 회귀 우위) |
| GARCH 회귀(감성없음) vs 회귀(감성포함) — **순수 감성 기여도** | p=0.072 | p=0.847 (7/15일) |
| XGBoost base vs +sentiment (v10: walk-forward holdout) | p=0.664 | p=0.071 (5/15일 +sentiment 우위) |

### 핵심 해석

**Target B(Parkinson)에서 "GARCH 원본 vs 회귀(감성포함)"가 유의(p=0.006)하게 나온 건 감성지수 덕분이 아니라 회귀보정 자체의 효과다.** GARCH(1,1)은 close-to-close 조건부분산을 추정하는데, target을 Parkinson(장중 range 기반)으로 바꾸면 스케일/정의가 어긋나서 원본을 그대로 쓰면 불리해짐 — 이걸 회귀로 재보정하니 크게 개선된 것. 감성지수를 넣었는지 여부(순수 기여도 검정, p=0.847)는 이 개선과 거의 무관함.

**즉, Target A/B, GARCH/XGBoost, 순수 감성 기여도까지 전부 따져봐도 "감성지수가 변동성 예측력을 통계적으로 유의하게 개선한다"는 근거는 현재 데이터에서 발견되지 않았다.** 이건 회귀 방식이나 target 정의 같은 모델링 설계의 결함 때문이 아니라(설계는 이번 점검으로 상당 부분 검증됨), **표본 규모(6개월, 종목당 102관측치, 홀드아웃 15일)의 한계**로 보는 게 타당하다.

## 6. 현재까지 점검 완료 / 남은 한계

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

---

## 부록. Current/Future 감성지수(3개월) 간이 모델링 (`06_appendix_current_future.py`)

### 배경

감성분석팀이 기존 장중/장외 2분류 감성지수를 **"현재 상태에 대한 감성"**과 **"미래(전망)에 대한 감성"**으로 다시 나눠 4분류로 재산출함(장중-현재/장중-미래/장외-현재/장외-미래). 커뮤니티 글에 현재 상태와 미래 전망에 대한 감정이 섞여 있어 분리가 필요하다는 판단. 산출식도 개선됨:

```
sentiment_score = (positive확률 - negative확률) × (positive확률 + negative확률)
```

neutral 확률이 높을수록(즉 positive+negative 합이 작을수록) 감성지수가 0에 가깝게 나오도록 정보량을 추가한 방식.

**한계**: 이 재산출은 3개월치(종목당 원자료 64행, 피처 생성 후 usable 50행)만 진행되어 6개월 메인 분석과 표본 규모가 다름. **메인 결론은 여전히 6개월 분석(1~5번 섹션)이며, 이 부록은 4분류 감성변수의 대략적인 패턴만 참고용으로 확인.**

### 구성 (간소화 버전)

메인 파이프라인(01~05번) 대비 의도적으로 간소화함 — 표본이 작아 정교한 통계검정(Ridge, HAC-DM test, pooled test)을 돌려도 신뢰도가 낮고, 통계적 유의성 자체는 통계팀이 별도 검정 중이라 중복을 피함:
- target은 이미 확정한 **Parkinson만** 사용 (5d 병행 비교 생략)
- GARCH는 3단계(원본/회귀 감성없음/회귀 감성포함)만, Ridge·HAC-DM test·pooled test는 생략
- XGBoost는 가벼운 고정 파라미터(`max_depth=2, n_estimators=80, learning_rate=0.05, subsample=0.8, min_child_weight=3`)로 학습 — 표본이 작아 GridSearch가 오히려 과적합된 파라미터를 고를 위험이 커서 튜닝 자체를 생략
- 시간정렬 원칙은 01번과 동일 (`sentiment_overnight_current/future`는 `.shift(-1)` 적용, `intraday`는 그대로)
- 홀드아웃/최소학습 구간도 표본 크기에 맞게 재조정 (`HOLDOUT_DAYS=10`, `MIN_TRAIN=15` — 회귀 파라미터 수(5개: 상수+GARCH값+감성변수4개) 대비 학습표본이 너무 작아지지 않도록 `MIN_TRAIN`을 낮게 잡음)

### 결과 (홀드아웃 10일, RMSE)

| 종목 | GARCH 원본 | GARCH+회귀(감성없음) | GARCH+회귀(감성포함, 4변수) | XGBoost base | XGBoost +sentiment(4변수) |
|---|---|---|---|---|---|
| 삼성전자 | 0.0462 | **0.0190** | 0.0210 | 0.0164 | 0.0185 |
| SK하이닉스 | 0.0624 | 0.0222 | **0.0162** | 0.0176 | **0.0158** |
| 카카오 | 0.0076 | **0.0090** | 0.0103 | **0.0090** | 0.0100 |
| 에코프로비엠 | 0.0237 | 0.0140 | 0.0205 | **0.0134** | 0.0128 |

감성지수 4개 변수의 OLS 계수 p-value는 4종목 전부 p>0.2로 비유의함 (SK하이닉스 `overnight_future`가 p=0.265로 가장 낮았으나 여전히 비유의).

### 해석

- **6개월 메인 결과와 일관된 패턴**: GARCH 원본보다 회귀보정이 대체로 RMSE가 낮음(스케일 재보정 효과, 6번 섹션 결론과 동일) — 표본이 다른데도 같은 패턴이 재현됨
- SK하이닉스는 회귀(감성포함)가 회귀(감성없음)보다 뚜렷이 낮아(0.0162 vs 0.0222) 흥미로운 point estimate긴 하나, 개별 감성계수는 비유의(p>0.2) — 6개월 분석에서도 반복됐던 "점추정치 개선과 통계적 유의성이 별개로 움직인다"는 패턴과 동일
- 홀드아웃 10일, 회귀 학습표본 약 25행이라는 극히 작은 표본에서 나온 결과라 신뢰구간이 넓음 — 이 부록의 목적은 "확정적 결론"이 아니라 "4분류 감성지수 재산출이 명백한 이상 신호 없이 메인 분석과 정합적인 패턴을 보인다"는 정도의 확인용임

---

## 7. 부록: current/future 감성지수(4변수) - 3개월 표본 참고 결과

감성분석팀이 "현재 상태 감정"과 "미래 전망 감정"을 분리해 재산출한 4변수 버전(`intraday_current/future`, `overnight_current/future`, 계산식: `(pos-neg)*(pos+neg)`)을 검토함. 팀 합의에 따라 **6개월/2변수 결과(위 1~6번)를 메인으로 유지**하고, 이 4변수 버전은 3개월 표본(종목당 50~64행)이라 별도 부록으로 가볍게 다룸 — 메인 결론을 대체하지 않음.

**코드/실행**: `06_appendix_current_future.py` (메인 파이프라인과 별도, `data_appendix/`, `outputs_appendix/` 사용). Target은 Parkinson(pk)만 사용, `HOLDOUT_DAYS=10`, `MIN_TRAIN=30`, XGBoost 튜닝은 `RandomizedSearchCV(n_iter=60)`로 메인 대비 축소(표본이 작아 넓게 탐색할 근거가 부족함).

**시간정렬 로직**: 메인과 동일한 규칙 적용 — `intraday_*`(current/future 둘 다)는 shift 없음, `overnight_*`(current/future 둘 다)는 `shift(-1)`로 t+1일 개장 전 값 정렬.

### 결과

| 종목 | GARCH 원본 | GARCH+회귀(감성없음) | GARCH+회귀(감성포함,4변수) | XGBoost base | XGBoost +sentiment(4변수) |
|---|---|---|---|---|---|
| 삼성전자 | 0.0462 | 0.0951 | 0.1144 | 0.0151 | 0.0209 |
| SK하이닉스 | 0.0624 | 0.1322 | 0.0704 | 0.0170 | **0.0118** |
| 카카오 | **0.0076** | 0.0267 | 0.0651 | 0.0088 | 0.0089 |
| 에코프로비엠 | 0.0237 | **0.0178** | 0.0390 | 0.0129 | 0.0131 |

**4종목 통합(pooled) 검정 (n_dates=10, 참고용)**

| 비교 | p-value | 방향 |
|---|---|---|
| GARCH 원본 vs 회귀(감성포함) | **p<0.001** | GARCH 원본이 유의하게 더 정확함(10일 중 8일 승) |
| 순수 감성 기여도(감성없음 vs 감성포함) | p=0.066 | 감성 추가가 오히려 성능을 악화시키는 경향(2/10일 승) |
| XGBoost base vs +sentiment | p=0.905 | 유의한 차이 없음 |

### 해석 (표본 한계로 인해 정확도가 높지 않음을 감안)

메인(6개월)과 다르게 여기서는 **"감성지수를 GARCH 회귀에 추가하는 게 오히려 유의하게 성능을 악화시킨다"**는 결과가 나옴. 이는 감성지수 자체의 문제라기보다 **표본 부족에 따른 과적합**으로 해석하는 게 타당함:
- 학습 표본이 30~40행뿐인데 회귀 모수(`garch_vol` + 감성변수 4개 = 5개)가 상대적으로 많아 과적합 위험이 큼
- `current`/`future` 변수 간 상관관계가 0.8 이상으로 높아(다중공선성) OLS 계수 추정 자체가 불안정함
- 반면 트리 기반인 XGBoost는 이런 과적합에 상대적으로 덜 민감해 유의한 차이가 나타나지 않음(p=0.905) — OLS/GARCH 조합에서만 문제가 두드러진다는 점도 "표본 부족+다중공선성" 해석을 뒷받침함

**결론적으로 이 부록 결과는 "4변수 감성지수가 유용한지 아닌지"를 판단할 근거로 쓰기 어렵고, 오히려 3개월 표본으로는 안정적인 회귀 추정 자체가 어렵다는 걸 보여주는 사례로 발표에 활용하는 게 적절함.** 향후 감성지수 산출 기간을 늘리면(예: 6개월) 재검증이 필요함.

---

## 8. 종합 결론

**6개월 메인 분석 (1~6번)**
- 감성 시간정렬 버그(장외 감성이 익일이 아닌 당일 예측에 쓰여야 했던 문제)를 발견·수정함
- GARCH 평가 target을 방법론적으로 재검토해 Parkinson volatility로 최종 확정 (rolling_5d는 자기상관이 과도해 GARCH 1일-앞 예측 평가에 부적합)
- GARCH 3단계(원본/회귀 감성없음/회귀 감성포함) + Ridge 진단 + XGBoost(확장된 탐색범위, walk-forward holdout)까지 GARCH·XGBoost 모두 동등한 조건으로 비교
- 통계검정도 종목 간 교차상관(날짜정렬 pooled test) + 예측오차 자기상관(HAC)까지 보정
- **결론: 감성지수가 변동성 예측력을 유의하게 개선한다는 증거를 발견하지 못함.** 이는 모델링 설계의 결함이 아니라(위 점검을 통해 상당 부분 확인됨), 표본 규모(6개월, 홀드아웃 15일)의 한계로 판단하는 게 타당함

**3개월 부록 (current/future 4변수, 4~6번)**
- "현재 상태"와 "미래 전망" 감성을 분리한 새 지수를 검토, 시간정렬 로직은 메인과 동일하게 적용
- **결론: 4변수를 회귀에 추가하는 게 오히려 유의하게(p<0.001) 성능을 악화시킴** — 이는 30~40개 학습표본에 5개 회귀 모수 + current/future 간 높은 상관관계(다중공선성)로 인한 과적합으로 해석하는 게 타당함. 감성지수 자체의 유용성을 판단할 근거로 쓰기는 어려움

**최종 메시지**: 두 분석 모두 "감성지수가 변동성 예측에 기여한다"는 확증적 증거는 얻지 못했지만, 그 원인이 감성지수 자체의 무의미함이 아니라 **표본 규모 부족**이라는 점을 방법론적으로 뒷받침하는 여러 근거(통계검정 보정, 병렬 target 비교, 과적합 패턴 등)를 확보함. 향후 데이터 축적을 통한 재검증이 핵심 과제로 남음.
