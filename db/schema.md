# DB 스키마 (1차 — raw 테이블)

`migrations/0001_init_raw_tables.sql` 기준.

| 테이블 | 설명 | 주요 컬럼 |
| --- | --- | --- |
| `stock_master` | KOSPI 종목 마스터 | `code`(PK), `name`, `market`, `is_active` |
| `price_daily` | 일별 시세·수급 | `code`, `trade_date`, OHLCV, `foreign_net_buy`, `inst_net_buy` |
| `financial_statement` | 재무제표 | `code`, `fiscal_year`, `fiscal_quarter`, 매출/영업이익/순이익 등 |
| `disclosure` | 공시 원문 메타 | `code`, `rcept_no`(OpenDART 접수번호), `report_name`, `url` |
| `news_raw` | 뉴스 원문 | `code`(nullable), `title`, `description`, `link`, `published_at` |

## 추후 추가 예정 (ai 단계에서 확정되는 대로 migration 추가)

- `news_structured` — GPT로 구조화한 sentiment / impact / risk_score / event_type
- `feature_store` — feature engineering 결과 (기술지표·수급누적·재무비율 등)
- `label_crash` — 급락 라벨 (5거래일 내 종가 -10% 이하 여부)
- `classification_result` — 급락확률 예측 결과 + 판단 근거(feature importance)
- `portfolio_result` — 최적화 비중 + 사용된 제약조건
- `explanation` — GPT가 생성한 추천 사유 (XAI)

raw 테이블(이번 migration)은 "수집한 데이터를 그대로 담는" 용도이고,
위 추가 예정 테이블들은 ai 단계에서 실제 출력 형태가 확정된 뒤 설계합니다.
