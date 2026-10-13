# AI Stock Assistant

KOSPI 전체 종목을 대상으로 정형 데이터(가격·재무·수급·공시)와 뉴스 텍스트를 결합한 멀티모달 AI로
개별 종목의 단기 급락 위험을 예측하고, 그 판단 근거와 포트폴리오 최적화 결과를 GPT가 함께 해석해
설명 가능한 포트폴리오를 추천하는 서비스입니다.

## 폴더 구조

```
.
├── db/                 # 스키마 / 마이그레이션 (1순위)
│   ├── migrations/
│   ├── schema.md
│   └── seed/
├── ai/                 # 데이터 수집 + feature + classification + optimization + explain (2순위)
│   ├── ingestion/      # ★ 지금 작업 중
│   │   ├── kis_price.py
│   │   ├── opendart_filing.py
│   │   ├── naver_news.py
│   │   ├── common/
│   │   └── run_ingest.py
│   ├── features/
│   ├── classification/
│   │   ├── structured/
│   │   ├── text_bert/
│   │   └── fusion/
│   ├── optimization/
│   └── explain/
├── backend/            # FastAPI (3순위, 아직 비어있음)
├── frontend/           # React / Vercel (3순위, 아직 비어있음)
├── docs/
└── scripts/
```

## 개발 순서

1. **db** — 원본 데이터를 담을 최소 테이블(`stock_master`, `price_daily`,
   `financial_statement`, `disclosure`, `news_raw`)부터 설계. `db/migrations/`에
   순서대로 SQL 파일을 쌓아가는 방식(Alembic 등으로 추후 전환 가능).
2. **ai/ingestion** — 한국투자증권 API → OpenDART → 네이버 뉴스 순서로 수집 모듈 작성.
3. **ai/features, classification, optimization, explain** — 수집이 안정화된 뒤 순차 진행.
4. **backend, frontend** — AI 파이프라인의 입출력이 확정된 뒤 API·화면 작업.

## 시작하기

```bash
cd ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env   # 키 값 채워넣기
```

`db/migrations/0001_init_raw_tables.sql`을 PostgreSQL에 적용한 뒤,
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

자세한 테이블 설명은 `db/schema.md`, 전체 시스템 구조는 `docs/architecture.md` 참고.
