# DB 스키마

## 1차 — raw 테이블 (`migrations/0001_init_raw_tables.sql`)

| 테이블 | 설명 | 주요 컬럼 |
| --- | --- | --- |
| `stock_master` | KOSPI 종목 마스터 | `code`(PK), `name`, `market`, `is_active` |
| `price_daily` | 일별 시세·수급 | `code`, `trade_date`, OHLCV, `foreign_net_buy`, `inst_net_buy` |
| `financial_statement` | 재무제표 | `code`, `fiscal_year`, `fiscal_quarter`, 매출/영업이익/순이익 등 |
| `disclosure` | 공시 원문 메타 | `code`, `rcept_no`(OpenDART 접수번호), `report_name`, `url` |
| `news_raw` | 뉴스 원문 | `code`(nullable), `title`, `description`, `link`, `published_at` |

## 2차 — classification + XAI 테이블 (`migrations/0002_add_explain_tables.sql`)

| 테이블 | 설명 | 대응하는 XAI 방법 |
| --- | --- | --- |
| `classification_result` | 종목·시점별 급락확률 예측 결과 | — |
| `feature_importance` | 정형 데이터(가격·재무·수급) feature별 기여도 | **SHAP** |
| `news_token_importance` | 뉴스 텍스트에서 단어별 중요도 | **Integrated Gradients** |
| `news_ablation` | 뉴스 1건 제거 시 급락확률 변화량 | **Ablation** |
| `explanation_log` | 위 세 근거를 종합해 GPT가 생성한 최종 추천 사유 | — |

`feature_importance` / `news_token_importance` / `news_ablation`은 모두 `classification_result.id`를
FK로 참조하고, `news_token_importance`·`news_ablation`은 `news_raw.id`도 함께 참조해 "어떤 뉴스의
어떤 단어"인지 추적 가능하게 설계했습니다.

## 3차 — 레이블 테이블 (`migrations/0003_add_label_crash.sql`)

| 테이블 | 설명 | 생성 코드 |
| --- | --- | --- |
| `label_crash` | 종목·기준일별 급락 정답 레이블 (5거래일 내 종가 -10% 이하 여부) | `ai/labeling/build_crash_label.py` |

`feature_importance` 등 classification 결과 테이블과는 따로 뒀습니다. 기준일 이후
가격으로 만드는 값이라 feature 생성 코드와 섞이지 않게 `ai/labeling/`을 따로 뒀고,
structured/text_bert/fusion이 전부 여기서 만든 레이블 하나만 씁니다.

## 추후 추가 예정 (ai 단계 진행하면서 확정되는 대로 migration 추가)

- `news_structured` — GPT로 구조화한 sentiment / impact / risk_score / event_type
- `feature_store` — feature engineering 결과 (기술지표·수급누적·재무비율 등)
- `portfolio_result` — 최적화 비중 + 사용된 제약조건

raw 테이블(1차)은 "수집한 데이터를 그대로 담는" 용도, classification+XAI 테이블(2차)은
"모델이 계산한 결과와 그 근거를 담는" 용도, 레이블 테이블(3차)은 "정답(y)을 담는" 용도입니다.
위 추후 목록은 ai 단계에서 실제 출력 형태가 확정된 뒤 설계합니다.
