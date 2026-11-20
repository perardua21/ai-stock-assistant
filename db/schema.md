# DB 스키마

migration은 `db/migrations/`에 번호 순서로 쌓는다. 처음 `docker compose up` 때 자동 적용되고,
이미 DB가 있으면 `scripts\db_migrate.bat`로 적용한다 (여러 번 실행해도 안전).

## 0001 수집 원본

| 테이블 | 설명 | 주요 컬럼 |
| --- | --- | --- |
| `stock_master` | KOSPI 종목 마스터 | `code`(PK), `name`, `market`, `is_active` |
| `price_daily` | 수정주가 일봉 · 수급 | `code`, `trade_date`, OHLCV, `trading_value`·`market_cap`(0004), `foreign_net_buy`, `inst_net_buy` |
| `financial_statement` | 재무제표 (현재 미사용) | |
| `disclosure` | 공시 목록 | `code`, `rcept_no`, `report_name`, `submitted_at` |
| `news_raw` | 에이전트가 읽은 뉴스 | `code`, `title`, `description`, `link`, `published_at` |

## 0002 분류기 결과

| 테이블 | 설명 |
| --- | --- |
| `classification_result` | 종목·날짜·모델 버전별 급락 확률 (LightGBM) |
| `feature_importance` | 위 확률의 SHAP 상위 피처 |

(0002에 있던 `news_token_importance`, `news_ablation`, `explanation_log`는 0004에서 삭제)

## 0003 라벨

| 테이블 | 설명 |
| --- | --- |
| `label_crash` | 종목·기준일별 급락 정답 (5거래일 내 종가 -10% 이하), `min_return` |

## 0004 Chronos-2 · 하네스

| 테이블 | 설명 | 만드는 코드 |
| --- | --- | --- |
| `index_daily` | KOSPI 지수 일봉 | `ai/ingestion/pykrx_market.py` |
| `forecast_feature` | Chronos-2 예측 분포 요약 (p_crash, 분위수 수익률, 불확실성). `model_name`으로 공변량 유무 구분 | `ai/forecasting/run_forecast.py` |
| `agent_decision` | 에이전트 판단 (모델 확률 → 조정 확률, judgment, 근거, 툴 호출 기록, 프롬프트·스킬 버전) | `ai/harness/memory.py` |
| `agent_score` | 5거래일 뒤 채점 (실제 급락 여부, 정답 여부, brier) | `ai/harness/scoring.py` |
| `skill_revision` | 스킬 개선안 (proposed / applied / rejected) | `ai/harness/evolve.py` |
| `portfolio_result` | 날짜별 추천 비중 | `ai/pipeline/run_daily.py` |
