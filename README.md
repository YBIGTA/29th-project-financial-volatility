# 29th-project-financial-volatility

비정형 텍스트 감성분석 기반 금융시장 변동성 예측 — Track A (데이터 크롤링 & 전처리)

## 팀 R&R

| 세부 과업 | 메인 | 서브 |
| --- | --- | --- |
| 1. 데이터 크롤링 & 전처리 | 송지훈 | 설승곤 |
| 2. NLP 감성분석 & 지수화 | 설승곤 | 송지훈 |
| 3. 통계 검증 (Granger/Event Study) | 박소영 | 윤소현 |
| 4. 변동성 모델링 & 백테스팅 | 나예린 | 박소영 |
| 5. 대시보드 & 총괄 | 윤소현 | 나예린 |

대상 종목은 삼성전자(005930), SK하이닉스(000660), 에코프로비엠(247540), 카카오(035720)입니다.

## 폴더 구조

```
src/                              크롤링·전처리·학습·추론·집계 코드
artifacts/finbert/                학습된 Model A/Model B와 평가 결과
data/raw/                         원본 크롤링 CSV
data/processed/                   전처리·라벨링·학습 데이터
data/정리/                         기존 KR-FinBERT 기본모델 결과
data/3개월_학습후_분석/processed/  Model A 기사별·일별 추론 결과
data/3개월_학습후_분석/정리/        최종 종목별·시장지수 분석 파일
```

## 데이터는 Drive로

코드는 GitHub에, csv/차트는 [Drive](https://drive.google.com/drive/folders/1QwUSq6zlkbGRhXws84SHQL-ilTBsXFUY)에 올렸습니다. `data/raw`, `data/processed`, `data/정리`는 gitignore 처리해서 push해도 올라가지 않습니다.

- 같은 파일을 동시에 건드릴 수도 있어서 파일명에 날짜·이름을 붙이는 방식으로 구분하려고 합니다: `board_samsung_electronics_0815_송지훈.csv`
- 인코딩은 UTF-8(`utf-8-sig`)로 저장했습니다. 엑셀에서 바로 열어도 한글이 깨지지 않습니다.

## 종목코드와 파일명

데이터 파일명과 CSV의 `code` 열에는 숫자 종목코드 대신 아래 영문 표기를 씁니다.

| 종목코드 | 회사명 | 파일명 표기 |
| --- | --- | --- |
| `005930` | 삼성전자 | `samsung_electronics` |
| `000660` | SK하이닉스 | `sk_hynix` |
| `035720` | 카카오 | `kakao` |
| `247540` | 에코프로비엠 | `ecopro_bm` |

시장 비교용 일봉은 `price_daily_kospi.csv`, `price_daily_kosdaq.csv`, `price_daily_vix.csv`로 저장했습니다. (`vix`는 미국 CBOE VIX이고, 한국 코스피200 변동성지수인 VKOSPI와는 다른 지표입니다.)

## 만든 코드 구성

```bash
pip install -r requirements.txt
cd src
python run_all.py             # 전체 실행 (시세 -> 사용 중인 뉴스/게시판 -> 전처리)
# 개별로 돌릴 때
python crawl_price.py         # 일봉 시세 (4종목 + KOSPI/KOSDAQ/VIX)
python crawl_board.py         # 네이버 종목토론방
python run_board_backfill.py  # 네이버 게시판 실제 다음 링크를 따라 3개월 소급
python crawl_paxnet_board.py  # 팍스넷 종목토론방
python crawl_edaily_news.py   # 이데일리 증권뉴스 (본문까지 포함)
python crawl_toss_community.py # 토스증권 커뮤니티
python preprocess.py          # 텍스트 정제
python sentiment_score.py     # 감성점수 산출 + 장중/장외 지수화 (--aggregate-only로 집계만 재실행 가능)
python build_summary.py       # 종목별 merged_daily.csv + 차트 생성 (data/정리)
```

기존 `sentiment_score.py`는 비교·레거시용 기본 KR-FinBERT 파이프라인입니다. 현재 최종 분석에는 아래의 fine-tuned **Model A** 파이프라인을 사용합니다. 상세 학습 명령은 [TRAINING.md](TRAINING.md)를 참고합니다.

네이버 뉴스는 과거 3개월을 안정적으로 열 수 없어 분석에서 제외했습니다. 네이버 게시판은 100페이지에서 끝내지 않고 하단의 실제 `다음` 링크를 세션과 Referer를 유지한 채 따라가도록 별도 백필 스크립트를 만들었습니다.

## 지금까지 모은 데이터

| 소스 | 종류 | 시간 단위 | 커버 기간 |
| --- | --- | --- | --- |
| FinanceDataReader | 일봉 시세 | 일 | 3개월 전체 |
| 이데일리 증권뉴스 | 뉴스(본문 포함) | 초 | 4종목 모두 거의 3개월 |
| 팍스넷 종목토론방 | 게시판 | 초 | 삼성전자·SK하이닉스만 3개월 전체 |
| 토스증권 커뮤니티 | 댓글(=게시글, 실보유 인증 여부 포함) | 초 | 삼성전자·SK하이닉스는 시간대 표본 방식으로 3개월, 카카오·에코프로비엠은 전량 수집 |
| 네이버 종목토론방 | 게시판 | 분 | `run_board_backfill.py`로 3개월 소급 |

팍스넷은 삼성전자·SK하이닉스만 사용했습니다. 카카오·에코프로비엠 게시판은 열어보니 실제 토론글이 거의 없고 무관한 종목(한성기업, 한주라이트, jw중외제약 등) 홍보성 스팸으로 도배되어 있어서 제외했습니다 — `preprocess.py`의 `PAXNET_VALID_CODES` 참고.

## 네이버 게시판 3개월 소급

`board.naver`는 100페이지 URL을 101로 단순 변경하는 방식이 불안정하지만, 100페이지 하단의 `다음` 링크로는 101페이지 이후에 접근할 수 있습니다. `run_board_backfill.py`는 숫자 페이지를 임의로 만들지 않고 응답 HTML에 있는 실제 다음 페이지 링크를 순서대로 따라갑니다. 25페이지마다 CSV를 중간 저장하고, 차단 또는 일시 오류가 나면 30분 뒤 같은 페이지부터 최대 6회 재시도하며, 게시물 날짜가 최근 90일 시작일보다 오래되면 종료합니다.

```bash
cd src
python run_board_backfill.py
```

네이버 뉴스와 통합검색 뉴스는 과거 3개월을 같은 방식으로 탐색할 다음 링크가 없어 분석·자동 파이프라인에서 제외했습니다. 기존에 받아둔 뉴스 CSV는 삭제하지 않지만 전처리와 감성점수 입력에는 사용하지 않습니다.

### 과거 보완 시도: 네이버 통합검색 날짜 슬라이싱

네이버 뉴스 API의 건수 상한을 피하기 위해 `crawl_naver_search_news.py`에서 검색 날짜를 하루 단위로 나누는 방식을 실험했습니다. 하루 3페이지(30건)만 표본으로 수집하고 요청 간격을 1.5~3.5초로 무작위화했으며, 일시 차단 시 20분 뒤 최대 6회 재시도하도록 구성했습니다. 이미 저장된 날짜는 건너뛰므로 중단 후 이어받을 수 있습니다.

다만 `search.naver.com`은 robots.txt에서 자동 수집을 허용하지 않고 절대 게시 시각 대신 상대 시각만 제공하므로, 현재 자동 파이프라인과 최종 분석에서는 제외합니다. 코드는 과거 실험 재현용으로만 보존합니다.

```bash
cd src
python crawl_naver_search_news.py
```

### 과거 보완 시도: 뉴스 소스 편향 완화

이데일리 단일 뉴스 소스의 성향 편중 가능성을 줄이기 위해 복수 경제지 추가를 검토했습니다.

| 후보 | 확인 결과 |
| --- | --- |
| 한국경제 | Cloudflare 봇 차단으로 자동 접근 불가 |
| 조선비즈 | 기사 목록이 클라이언트 측 JavaScript로 렌더링되어 일반 요청으로 수집 곤란 |
| 동아일보 | 서버 렌더링 페이지 접근 가능성을 확인했으나 최종 수집 소스로 채택하지 않음 |

추가 소스를 실제로 도입할 경우 `sentiment_score.py`의 입력 소스와 소스별 가중치를 함께 재검토해야 합니다. 현재 최종 분석은 아래에 설명한 확정 데이터셋과 Model A를 기준으로 합니다.

## 토스증권 커뮤니티 — 시간대 표본 수집

화면에는 글 목록이 바로 보이지 않지만, 실제로는 로그인 없이 열려있는 내부 API가 있습니다.

```
GET https://wts-cert-api.tossinvest.com/api/v4/comments
    ?subjectType=STOCK&subjectId={ISIN}&commentSortType=RECENT&lastCommentId={cursor}
```

- `subjectId`는 종목코드가 아니라 ISIN을 사용합니다 (코드↔ISIN 매핑은 `crawl_toss_community.py`의 `ISIN` 딕셔너리).
- 페이지당 11건 고정이고, 다음 페이지는 마지막 댓글의 `commentId`를 `lastCommentId`로 넘기면 됩니다.
- `commentId`는 전 종목이 공유하는 전역 순번이라 시간과 거의 선형 비례합니다 — 즉 임의의 오래된 `lastCommentId`를 넣으면 그 근처 시점 댓글로 바로 점프됩니다.

삼성전자·SK하이닉스는 댓글 속도가 너무 빨라서(3개월 전체를 순차 페이지네이션으로 모으려면 요청이 9만 건 이상 필요) 위 점프 기능으로 하루 안 장중 3곳·장외 4곳씩 고정 시각을 정해 표본을 추출합니다(`crawl_comments_sampled`). 카카오·에코프로비엠은 댓글이 적어서 기존 순차 수집(`crawl_comments`)만으로 3개월이 채워집니다. 전처리 단계에서 추천 0건인 댓글은 노이즈로 보고 제외했습니다(`preprocess.MIN_TOSS_LIKES`).

## 기존 KR-FinBERT의 토스 커뮤니티 한계 확인

초기 파이프라인은 `snunlp/KR-FinBert-SC`를 별도 fine-tuning 없이 사용해 `positive - negative`로 감성점수를 계산했습니다. 하지만 정제된 금융 뉴스에 맞춘 단일 감성 분류기라 줄임말·은어·반어·짧은 표현이 많은 토스 커뮤니티 글을 대부분 중립으로 처리했습니다.

이를 정량적으로 확인하기 위해 GPT로 작성한 3-class 라벨 데이터에서 모델 선택에 사용하지 않을 고정 Test를 분리하고, 그중 토스 커뮤니티 180건에 기존 KR-FinBERT를 적용했습니다. 기존 모델에는 Current/Future 구분이 없으므로 의미가 가장 가까운 Current 라벨을 주 평가 대상으로 삼았습니다.

| 기존 KR-FinBERT 평가 대상 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| GPT Current 라벨 | 0.3944 | 0.2534 |
| GPT Future 라벨¹ | 0.4000 | 0.2318 |

기존 모델은 180건 중 167건(92.8%)을 중립으로 예측했습니다. Current 라벨 기준 recall은 negative 0.0704, neutral 0.9275, positive 0.0500으로, 방향성이 있는 은어성 게시물의 긍정·부정을 거의 잡지 못했습니다. 이 중립 쏠림과 Current/Future 구분 부재를 확인한 뒤, 커뮤니티 표현을 학습하고 현재 상태와 미래 전망을 분리하는 멀티태스크 모델을 만들기로 했습니다.

¹ 기존 KR-FinBERT에는 Future 전용 head가 없으므로 동일한 단일 감성 출력을 Future 라벨에 대조한 값은 보조 지표입니다.

기존 구현은 `sentiment_score.py`에 비교·레거시용으로 남아 있으며 `--aggregate-only`로 모델 추론 없이 장중·장외 집계만 다시 수행할 수 있습니다.

## 최종 감성분석 모델

위 문제를 해결하기 위해 GPT 라벨을 사용해 Current/Future 멀티태스크 Model A와 도메인 MLM을 추가한 Model B를 학습했습니다. 최종 분석에는 성능이 더 높았던 fine-tuned Model A를 사용합니다.

### 라벨과 데이터 분할

한 텍스트에 다음 두 라벨을 동시에 부여한 멀티태스크 분류 모델입니다.

- `current_label`: 현재 기업·주가·시장 상황에 대한 감성
- `future_label`: 향후 기업·주가·시장 전망에 대한 감성
- 각 헤드의 클래스: `negative`, `neutral`, `positive`

초기 라벨 2,000건을 중복 그룹 단위로 Train 1,400 / Validation 300 / Test 300으로 고정 분할했습니다. 이후 Active Learning 2,000건과 2차 경계 표본 500건은 Train에만 추가했습니다.

| 데이터 | 건수 | 용도 |
| --- | ---: | --- |
| Train | 3,900 | 모델 가중치 학습 |
| Validation | 300 | 최고 epoch 선택 |
| Test | 300 | 최종 성능 비교 전용 |
| 합계 | 4,500 | 라벨 완료 데이터 |

Validation/Test 600건은 Active Learning 이후에도 그대로 고정해 성능 비교 누수를 막았습니다. 실제 분석기간인 `2026-05-14~2026-08-14` 원문은 모델 확정 전 학습·MLM·모델 선택에 사용하지 않았습니다.

### Model A와 Model B 비교

- **Model A**: `snunlp/KR-FinBert-SC` 인코더를 4,500건의 Current/Future 라벨 데이터로 fine-tuning
- **Model B**: Development Toss 원문 15,815건으로 MLM 도메인 적응 후 Model A와 같은 조건으로 fine-tuning
- 공통 설정: seed 42, max length 256, 5 epochs, learning rate `2e-5`, 유효 batch 32, BF16, Current/Future별 class-weighted loss
- GPU: NVIDIA GeForce RTX 5060 Ti

고정 Test 결과는 다음과 같습니다.

| 모델 | Current Macro-F1 | Future Macro-F1 | 평균 Macro-F1 |
| --- | ---: | ---: | ---: |
| Model A | **0.6858** | **0.6331** | **0.6594** |
| Model B | 0.6856 | 0.6189 | 0.6523 |

Model B가 Future Negative recall은 높였지만 전체 Macro-F1과 Future 성능은 낮아 최종 모델로 **Model A**를 선택했습니다.

#### 토스 커뮤니티 동일 표본에서 개선 결과

문제 확인에 사용한 것과 동일한 고정 Test의 토스 커뮤니티 180건에서 최종 Model A를 다시 평가했습니다. 예측 클래스는 각 head의 negative/neutral/positive 확률 중 최댓값으로 정했습니다.

| 동일표본 비교 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| 기존 KR-FinBERT 단일 감성 → Current 라벨 | 0.3944 | 0.2534 |
| **Model A Current head → Current 라벨** | **0.6000** | **0.5712** |
| 기존 KR-FinBERT 단일 감성 → Future 라벨 | 0.4000 | 0.2318 |
| **Model A Future head → Future 라벨** | **0.6444** | **0.6049** |

Model A Current head의 recall은 같은 표본에서 negative 0.5634, neutral 0.7536, positive 0.4000으로 개선됐습니다. 위 수치는 전체 Test 300건 성능표와 달리 토스 게시물 180건만 사용한 동일표본 비교이므로 두 표를 직접 같은 모집단의 점수로 해석하지 않습니다.

최종 체크포인트는 `artifacts/finbert/model_a_active_round2/best/`, 평가 결과는 `artifacts/finbert/model_a_active_round2/evaluation_metrics.json`에 있습니다.

### 기사별 감성점수

Model A는 Current/Future별로 세 클래스 확률을 출력합니다. 단순한 `positive-negative`보다 중립 확률을 반영하기 위해 기사별 감성점수는 다음 식을 사용합니다.

```text
D = P(positive)^2 - P(negative)^2
  = (P(positive) - P(negative)) × (1 - P(neutral))
```

따라서 방향은 유지하면서 중립 확률이 높은 기사의 크기를 줄입니다. 긍정·부정이 동시에 높고 팽팽한 사례는 전체의 약 1.5~3.3%로 확인됐습니다. 원본 확률 세 개도 기사별 파일에 보존합니다.

### 최종 추론 및 일별 집계

분석기간 19,988건을 Model A, max length 256으로 GPU 추론합니다.

```powershell
python -m src.infer_finbert_multitask `
  --input data/processed/finbert/analysis_pool_20260514_20260814.csv `
  --checkpoint artifacts/finbert/model_a_active_round2/best `
  --output data/processed/finbert/model_a_analysis_scores_20260514_20260814.csv

python -m scripts.split_model_a_scores_by_stock `
  --input data/processed/finbert/model_a_analysis_scores_20260514_20260814.csv `
  --output-dir data/3개월_학습후_분석/processed
```

일별 하나로 합치지 않고 실제 거래일 기준으로 다음과 같이 나눕니다.

- 장중: 해당 거래일 `09:00 이상~15:30 미만`
- 장외: 전 거래일 15:30 이후부터 해당 거래일 09:00 이전
- 주말·공휴일: 가장 가까운 다음 거래일 장외에 배정
- 게시물이 없는 세션: 감성지수 `0`, `no_posts=1`

세션별 Current/Future 감성지수는 **기사별 D를 먼저 계산한 다음 평균**합니다. 평균 확률에 제곱식을 다시 적용하는 방식이 아닙니다.

```powershell
python -m src.build_model_a_daily_summary `
  --price-dir data/6개월/raw `
  --market-zip "C:/Users/WF_26/Downloads/market_index-20260823T063326Z-1-001.zip" `
  --vkospi "C:/Users/WF_26/Downloads/vkospi.csv"
```

### 최종 분석 파일

종목별 `data/3개월_학습후_분석/정리/{종목}/merged_daily.csv`에는 다음 열만 남깁니다.

- 장중 Current/Future 감성지수, 기사 수, `no_posts`
- 장외 Current/Future 감성지수, 기사 수, `no_posts`
- 종목 OHLCV
- VIX OHLCV

`data/3개월_학습후_분석/정리/market_index/`에는 다음 파일이 있습니다.

- `kospi.csv`
- `kosdaq.csv`
- `vix.csv`
- `vkospi.csv`
- `market_index_merged.csv`

최종 기간은 `2026-05-14~2026-08-14`입니다. KOSPI·KOSDAQ·VKOSPI와 종목별 파일은 64거래일, VIX와 시장지수 outer-merge 파일은 미국 거래일을 포함해 67일입니다. VIX의 6월 19일과 7월 3일 결측은 미국 휴장일입니다.

## 크롤링 사이트별 상태

| 사이트 | 상태 |
| --- | --- |
| 팍스넷 (paxnet.co.kr) | 사용 중 |
| 이데일리 (edaily.co.kr) | 사용 중 |
| 토스증권 커뮤니티 | 사용 중 (비공개 API) |
| 네이버 종목토론방 | 실제 다음 링크를 순차 추적하는 3개월 백필 사용 |
| 네이버 뉴스/통합검색 뉴스 | 과거 3개월 확보가 어려워 분석에서 제외 |
| 한국경제 | Cloudflare 봇 차단으로 접근 불가 |
| 조선비즈 | 기사 목록의 JavaScript 렌더링으로 수집 보류 |
| 동아일보 | 대체 뉴스 소스로 검토했으나 최종 미채택 |
| 연합뉴스, 디시인사이드, 씽크풀 | robots.txt에 크롤러 봇 차단이 명시돼있어서 손대지 않음 |
| 다음 뉴스/금융 | 서비스 자체가 종료됨 (404) |
| investing.com | Cloudflare 봇 차단으로 접근 자체가 안 됨 |

### 네이버 게시판 최신분 증분 수집

기존 CSV 이후의 새 게시글만 누적하려면 아래 명령을 직접 실행합니다. 최신 페이지부터
확인하고 한 페이지 전체가 이미 저장된 `id/nid`이면 자동으로 멈추므로, 매번 100페이지를
다시 읽지 않습니다.

```bash
cd src
python run_naver_incremental.py
```

삼성전자·SK하이닉스 게시판은 글이 빠르게 쌓이므로 백필 완료 뒤에는 이 명령을 주기적으로 실행하면 됩니다. 과거 3개월을 처음 채울 때는 위의 `run_board_backfill.py`를 사용합니다.

## 아직 못 한 것

- 네이버·팍스넷 게시판 본문 전체는 JS 렌더링이라 가져오지 못했습니다. 지금은 제목만 있습니다.
- 네이버 오픈 API(뉴스/블로그/카페 공식 검색)는 Client ID/Secret을 발급받으면 추가할 수 있는데, 검토 결과 검색당 상한이 지금 쓰는 방식보다 낮아서(1,000건) 실익이 적다고 판단해 보류했습니다.
- 게시판·커뮤니티 데이터에는 닉네임, 자극적 표현이 섞여있습니다. 발표자료·대시보드에는 원문을 그대로 쓰지 않고 필요하면 익명 처리해서 쓸 계획입니다.
- 토스 커뮤니티 이용자의 연령 편중은 원문 데이터에 연령 정보가 없어 직접 보정하지 못했습니다. 별도 소스를 무리하게 추가하기보다 데이터 한계로 문서화합니다.
