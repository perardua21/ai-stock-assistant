# AI Stock Assistant

KOSPI 전체 종목을 대상으로 정형 데이터(가격·재무·수급·공시)와 뉴스 텍스트를 결합한 멀티모달 AI로
개별 종목의 단기 급락 위험을 예측하고, SHAP·Integrated Gradients·Ablation으로 근거를 뽑아
GPT가 해석한 설명 가능한 포트폴리오를 추천하는 서비스입니다.

## 폴더 구조

```
.
├── db/                        # 스키마 / 마이그레이션 (1순위)
│   ├── migrations/
│   │   ├── 0001_init_raw_tables.sql        # 수집 raw 테이블
│   │   ├── 0002_add_explain_tables.sql     # classification 결과 + XAI 테이블
│   │   └── 0003_add_label_crash.sql        # 급락 정답 레이블
│   ├── schema.md
│   └── seed/
├── ai/                        # 데이터 수집 + feature + label + classification + optimization + explain (2순위)
│   ├── ingestion/             # 한투 API · OpenDART · 네이버뉴스
│   │   ├── kis_price.py
│   │   ├── opendart_filing.py
│   │   ├── naver_news.py
│   │   ├── common/
│   │   └── run_ingest.py
│   ├── features/
│   ├── labeling/               # 급락 정답 레이블(y) 생성, feature(X)와 파일 분리
│   │   └── build_crash_label.py   # 5거래일 내 종가 -10% 이하 -> label_crash
│   ├── classification/
│   │   ├── structured/        # 정형데이터 baseline (Logistic Regression / RF / XGBoost)
│   │   ├── text_bert/         # 뉴스 BERT/Transformer fine-tuning
│   │   ├── fusion/            # 정형+뉴스 임베딩 결합 (멀티모달)
│   │   ├── split.py            # train/test 날짜 기준 분할 (세 branch 공통)
│   │   ├── tests/
│   │   │   └── test_split.py   # train/test 겹침·시간 역전 체크
│   │   └── requirements.txt
│   ├── optimization/
│   └── explain/                # XAI
│       ├── shap_explainer.py          # 정형 데이터 영향 -> SHAP
│       ├── integrated_gradients.py    # 뉴스 중요 표현 -> Integrated Gradients
│       ├── ablation.py                # 뉴스 1건 제거 시 변화 -> Ablation
│       ├── aggregate.py               # 위 3개 + optimization 결과 통합
│       ├── gpt_explainer.py           # 통합 근거 -> GPT 추천 사유 생성
│       ├── run_explain.py             # 엔트리포인트
│       └── requirements.txt
├── backend/                   # FastAPI (3순위, 아직 비어있음)
├── frontend/                  # React / Vercel (3순위, 아직 비어있음)
├── docs/
├── scripts/
└── conftest.py                 # pytest가 repo 루트 기준으로 import하게 해줌
```

## 개발 순서

1. **db** — 원본 데이터 raw 테이블(`0001`) → classification 결과 + XAI 테이블(`0002`).
   `db/migrations/`에 순서대로 SQL 파일을 쌓아가는 방식(Alembic 등으로 추후 전환 가능).
2. **ai/ingestion** — 한국투자증권 API → OpenDART → 네이버 뉴스 순서로 수집 모듈 작성. 지금 여기
3. **ai/features + ai/labeling → classification → explain → optimization** — 수집이 안정화된 뒤 순차 진행.
   feature(X)와 label(y)은 파일을 분리해서 만든다(`ai/features/`, `ai/labeling/`). classification은
   `ai/classification/split.py`로 train/test를 나누고 `ai/classification/tests/`로 겹침을 확인한다.
   classification(structured/text_bert/fusion)이 먼저 학습돼야 explain(SHAP/IG/Ablation)이 동작.
4. **backend, frontend** — AI 파이프라인의 입출력이 확정된 뒤 API·화면 작업.

## 시작하기

```bash
cd ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # ingestion/labeling용 (가벼움)
# classification/explain 단계에서는 각 폴더의 requirements.txt를 추가로 설치
#   pip install -r ai/classification/requirements.txt   # pytest 포함
#   pip install -r ai/explain/requirements.txt
cp ../.env.example ../.env   # 키 값 채워넣기
```

train/test 분리가 깨지지 않았는지는 아래로 확인합니다 (repo 루트에서 실행):

```bash
python -m pytest ai/classification/tests/test_split.py
```

`db/migrations/`의 SQL을 순서대로 PostgreSQL에 적용한 뒤,
`ai/ingestion/run_ingest.py`로 수집을 실행합니다.

```bash
python ai/ingestion/run_ingest.py --source kis_price --limit 10   # 우선 소수 종목으로 테스트
```

## 데이터 출처

| 데이터 | 출처 | 비고 |
| --- | --- | --- |
| 가격·수급 | 한국투자증권 Open API | OHLCV·실시간 체결 |
| 재무·공시 | OpenDART | 공시원문·재무·지분정보 |
| 뉴스 | 네이버 뉴스 검색 API | 제목·본문·발행시간 |

## XAI 구현 방법

| 대상 | 방법 | 역할 |
| --- | --- | --- |
| 정형 데이터 | SHAP | feature별 기여도를 %로 산출 |
| 뉴스 텍스트 | Integrated Gradients | 토큰(단어)별 중요도 계산 |
| 뉴스 단위 | Ablation | 뉴스 1건 제거 시 확률 변화(delta) 계산 |

자세한 테이블 설명은 `db/schema.md`, 전체 시스템 구조는 `docs/architecture.md`,
개발 참고용 통합 자료는 `docs/dev_reference.pdf` 참고.
