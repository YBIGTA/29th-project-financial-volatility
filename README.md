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

대상 종목: 삼성전자(005930), SK하이닉스(000660), 에코프로비엠(247540), 카카오(035720)

## 폴더 구조

```
src/            크롤링·전처리 코드 (git 관리)
data/raw/       원본 크롤링 결과 csv (git 미포함, Drive로 관리)
data/processed/ 전처리 결과 csv (git 미포함, Drive로 관리)
```

## 데이터 저장 규칙

- **코드는 GitHub, 데이터(csv)는 [Google Drive](https://drive.google.com/drive/folders/1QwUSq6zlkbGRhXws84SHQL-ilTBsXFUY)** — `data/raw`, `data/processed`는 `.gitignore`에 포함되어 있음.
- 크롤링 후 `data/raw`, `data/processed` 안의 csv를 위 Drive 폴더에 그대로 업로드.
- 동시 편집 충돌 방지를 위해 파일명에 날짜·작성자 표기: 예) `board_005930_0815_송지훈.csv`
- 인코딩은 UTF-8(BOM 포함, `utf-8-sig`)로 통일 — 엑셀에서 바로 열어도 한글이 깨지지 않음.

## 데이터 소스 현황 (8/14 기준)

| 소스 | 종류 | 시간 정밀도 | 실제 커버 기간 | robots.txt |
| --- | --- | --- | --- | --- |
| FinanceDataReader | 일봉 시세(4종목+KOSPI/KOSDAQ/VIX) | 일 | 3개월 전체 | 라이브러리(스크래핑 아님) |
| 네이버 뉴스 (`m.stock.naver.com`) | 뉴스 | 분 | 종목별 상이(아래 표) | **Disallow** — 참고용으로만 유지 |
| 네이버 종목토론방 (`finance.naver.com`) | 게시판 | 분 | 종목별 상이(아래 표) | **Disallow** — 참고용으로만 유지 |
| 팍스넷 종목토론방 (`paxnet.co.kr`) | 게시판 | **초** | 삼성전자·SK하이닉스만 3개월 전체 | Allow |
| 이데일리 증권뉴스 (`edaily.co.kr`) | 뉴스(본문 포함) | **초** | 4종목 모두 거의 3개월 전체 | Allow |
| 토스증권 커뮤니티 | 게시판 | — | **미확보** (아래 참고) | Allow (기술적으로 미해결) |

### 종목별 실제 확보 건수 (1차 수집, 8/14)

| 종목 | 네이버 뉴스 | 네이버 게시판 | 팍스넷 게시판 | 이데일리 뉴스 |
| --- | --- | --- | --- | --- |
| 삼성전자 | 1,737건 (8/10~) | 1,989건 (8/13 15:44~) | 2,234건 (5/16~) | 773건 (5/15~) |
| SK하이닉스 | 1,710건 (8/9~) | 1,990건 (8/13 16:11~) | 875건 (5/17~) | 741건 (5/17~) |
| 에코프로비엠 | 1,588건 (6/17~) | 1,979건 (7/20~) | 사용 안 함(스팸) | 95건 (5/17~8/2) |
| 카카오 | 2,000건(상한) (7/6~) | 1,974건 (7/31~) | 사용 안 함(스팸) | 67건 (5/17~8/12) |

**팍스넷은 삼성전자·SK하이닉스만 유효합니다.** 카카오·에코프로비엠 게시판은 실토론 트래픽이
거의 없어 무관한 종목(한성기업, 한주라이트, jw중외제약 등) 홍보성 스팸으로 도배되어 있는 걸
확인해서 제외했습니다 — `preprocess.py`의 `PAXNET_VALID_CODES`도 이 두 종목만 처리합니다.

## robots.txt 확인 결과 (8/14 실측)

| 사이트 | 일반 크롤러 정책 | 비고 |
| --- | --- | --- |
| finance.naver.com | 전면 Disallow | yeti(네이버 자체봇)만 board.naver 일부 허용, `page=` 파라미터는 yeti도 금지 |
| m.stock.naver.com | 전면 Disallow | `/api/news/...` 등 API 경로는 yeti 허용 목록에도 없음 |
| news.naver.com | 전면 Disallow | ClaudeBot 등 AI봇 별도 명시 차단 |
| blog.naver.com | 시스템 경로만 제한 | **ClaudeBot 등 AI 학습/RAG 목적 봇은 명시적으로 전면 금지** |
| cafe.naver.com | 전면 Disallow | 예외 없이 모든 봇 차단 |
| 연합뉴스 (yna.co.kr) | 대체로 허용 | **ClaudeBot, Claude-Web 명시 차단** |
| 디시인사이드 (dcinside.com) | 대체로 허용 | **ClaudeBot, anthropic-ai 등 명시 차단** |
| 씽크풀 (thinkpool.com) | 대체로 허용 | **Claude 계열 5종 전부 개별 명시 차단** |
| 팍스넷 (paxnet.co.kr) | **전면 허용** | AI봇 관련 언급 없음 |
| 이데일리 (edaily.co.kr) | **전면 허용** | SEO 스크래퍼 몇 종만 차단, AI봇 언급 없음 |
| 다음 뉴스/금융 | — | 서비스 자체가 종료됨 (404) |
| tossinvest.com | **`Allow: /`** | 크롤링 정책상 허용, 실제 데이터 확보는 미해결 |
| investing.com | — | Cloudflare 봇 차단 — robots.txt조차 접근 불가, 진행 안 함 |

**네이버 뉴스/게시판, 연합뉴스, 디시인사이드, 씽크풀은 robots.txt에 Claude류 AI봇을 명시적으로
차단해뒀습니다.** 이미 수집된 네이버 news_*.csv/board_*.csv는 이 확인 이전에 만든 크롤러
결과라 그대로 남겨뒀지만(팀 판단 필요), 이후로는 AI가 이 세션에서 직접 이 사이트들에 자동
요청을 보내는 방식으로는 더 이상 수집하지 않습니다. 팍스넷·이데일리는 이런 차단이 없어서
계속 직접 실행 중입니다. 연합뉴스/디시인사이드/씽크풀용 크롤러 코드가 필요하면 작성은
해드릴 수 있으나, 실행은 직접 하시는 걸 권장합니다.

## 토스증권 커뮤니티 (미해결)

robots.txt는 허용이지만, 실제로 `/stocks/A005930/community`, `/stocks/A035720/community`를
브라우저로 열어봐도 게시글 목록이 로드되지 않았습니다 (로그인 필요 여부 불명, 데이터 API 미발견).
계속 진행하려면 로그인된 브라우저에서 F12 Network 탭으로 실제 API 호출을 확인해서 알려줘야
빠르게 붙일 수 있습니다.

## 실행 방법

```bash
pip install -r requirements.txt
cd src
python run_all.py          # 시세 -> 네이버뉴스/게시판 -> 팍스넷 -> 이데일리 -> 전처리 전체 실행
# 또는 개별 실행
python crawl_price.py
python crawl_news.py          # 네이버 뉴스
python crawl_board.py         # 네이버 게시판
python crawl_paxnet_board.py  # 팍스넷 게시판 (삼성전자/SK하이닉스 위주)
python crawl_edaily_news.py   # 이데일리 증권뉴스 (본문 포함)
python preprocess.py
```

네이버 뉴스·게시판 크롤러는 **이미 저장된 csv에 새로 수집한 데이터를 id/nid 기준으로 병합**합니다.
매일 한 번씩 실행하면 실제 보유 데이터 기간이 계속 늘어납니다 — 네이버는 페이지당 20건 ×
최대 100페이지(약 2,000건) 상한이 있어서, 게시량이 많은 삼성전자·SK하이닉스는 하루만
지나도 그날 게시글이 상한 밖으로 밀려나 영구히 못 가져오게 됩니다. 반면 팍스넷·이데일리는
페이지 상한이 없어 매번 3개월 range를 통째로 다시 수집합니다.

## 알려진 제약 / TODO

- 네이버·팍스넷 게시판 상세 페이지(본문 전체)는 JS 렌더링(SPA)이라 requests만으로는 못 가져옴 —
  현재는 제목(title) 기반 수집. 본문까지 필요하면 Selenium 등 렌더링 도구 도입 검토.
- 토스증권 커뮤니티 데이터 API 미발견 (위 참고).
- 네이버 오픈 API(뉴스/블로그/카페 공식 검색)는 Client ID/Secret 발급 필요 — 아직 미적용.
- 개인정보/민감 텍스트 주의: 게시판 데이터에는 닉네임, 자극적 표현이 포함될 수 있음 —
  발표자료·대시보드에는 원문 그대로 노출하지 말고 필요 시 익명 처리.

## 종목코드와 회사명

데이터 파일명에는 아래의 영문 표기를 사용하며, CSV의 `code` 열에도 숫자 종목코드 대신
영문 회사명이 기록되어 있습니다.

| 종목코드 | 회사명 | 영문 표기 | 파일명 표기 |
| --- | --- | --- | --- |
| `005930` | 삼성전자 | Samsung Electronics | `samsung_electronics` |
| `000660` | SK하이닉스 | SK hynix | `sk_hynix` |
| `035720` | 카카오 | Kakao | `kakao` |
| `247540` | 에코프로비엠 | EcoPro BM | `ecopro_bm` |

시장 비교용 일봉 파일은 조회 코드 대신 아래 이름을 사용합니다.

| 조회 코드 | 지표 | 파일명 |
| --- | --- | --- |
| `KS11` | 코스피(KOSPI) | `price_daily_kospi.csv` |
| `KQ11` | 코스닥(KOSDAQ) | `price_daily_kosdaq.csv` |
| `VIX` | 미국 변동성 지수 | `price_daily_vix.csv` |

## 장중·장외 감성지수 집계 기준

`sentiment_index_daily_*.csv`는 달력 날짜의 단순 일평균이 아니라 실제 주식 거래일을 기준으로
장중과 장외 감성지수를 따로 제공합니다.

- 장중(`intraday`): 거래일 09:00 이상 15:30 미만 게시물 → 같은 거래일에 귀속
- 장외(`overnight`): 전 거래일 15:30 이후부터 해당 거래일 09:00 이전 게시물 → 해당 거래일에 귀속
- 주말·공휴일 게시물 → 가장 가까운 다음 거래일의 장외 감성에 포함
- 게시물이 없는 구간 → 감성지수 `0`, 게시물 수 `0`, `no_posts=1`
- 실제 중립 감성 → 감성지수는 `0`일 수 있지만 `no_posts=0`
- 당일 가격이 아직 수집되지 않은 잠정 다음 거래일 → `price_available=0` (가격 갱신 후 재집계 필요)

기존 `sentiment_scored_*.csv`를 다시 모델 추론하지 않고 집계만 갱신하려면 다음 명령을 사용합니다.

```bash
cd src
python sentiment_score.py --aggregate-only
```
