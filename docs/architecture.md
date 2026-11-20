# 시스템 구조

```
[학습 영역: KOSPI 데이터로 학습]
  KOSPI 일봉 ------------------> Chronos-2 --> 가격 피처 --+
  수급·거래량·지수 --(공변량)--> Chronos-2                  +--> 급락 분류기 (LightGBM)
  수급·거래량·지수 -----------------------> 정형 피처 -----+          |
                                                            종목별 급락 확률 + SHAP
                                                                       |
[금융 특화 Harness]                                                    v
  LLM Agent
    툴: 급락 신호, 가격 요약, 뉴스 검색, 공시 검색, 과거 판단, 포트폴리오 최적화
    시스템 프롬프트, 스킬, 출력 검사
  --> 급락 판단 + 근거, 포트폴리오 추천

[Self-evolving]
  5거래일 뒤 자동 채점 --> 틀린 사례로 스킬 개선안 --> 확인 후 적용
```

## 단계별 코드

| 단계 | 코드 | 저장 |
| --- | --- | --- |
| 수집 | `ai/ingestion/pykrx_market.py` | `price_daily`, `index_daily` |
| 라벨 | `ai/labeling/build_crash_label.py` | `label_crash` |
| 정형 피처 | `ai/features/tabular.py` | 저장 안 함 (그때 계산) |
| Chronos-2 | `ai/forecasting/` windows, chronos_runner, price_features, run_forecast | `forecast_feature` |
| 분류기 | `ai/classification/train.py`, `predict.py` | `models/crash_lgbm.*`, `classification_result` |
| SHAP | `ai/explain/shap_explainer.py` | `feature_importance` |
| 하네스 | `ai/harness/` agent, tools, guards, memory | `agent_decision` |
| 포트폴리오 | `ai/optimization/portfolio.py` | `portfolio_result` |
| 채점, 개선 | `ai/harness/scoring.py`, `evolve.py` | `agent_score`, `skill_revision` |
| 매일 실행 | `ai/pipeline/run_daily.py` | `data/reports/날짜.json` |

## 미래 정보가 섞이지 않게 한 것

- 정형 피처는 기준일까지 데이터로만 계산 (`test_tabular.py`의 test_no_lookahead)
- Chronos 입력은 기준일 종가에서 끝남 (`test_forecasting.py`의 test_windows_end_at_base_date)
- train/test는 날짜로 나누고, 분할일 직전 5거래일은 train에서 뺌 (`split.py`)
- Chronos 추가 학습은 TRAIN_END_DATE 이전 데이터만 씀
- 에이전트 툴은 기준일 뒤 뉴스·공시를 걸러내고, 과거 판단의 결과도 그 시점에 나와 있던 것만 보여줌

## 검증 계획

1. 모델 단계별 비교 (`train.py`): 변동성 규칙, 로지스틱 회귀, LightGBM(정형), Chronos-2 단독,
   Chronos-2+공변량, LightGBM(정형+가격)을 같은 test 행에서 AUC, PR-AUC, 상위 k 적중률, Brier로 비교
2. 전진 테스트: 매일 판단을 저장하고 5거래일 뒤 채점 (`run_daily.py`, `scoring.py`)
3. 에이전트 비교: 분류기 확률만 쓸 때와 에이전트가 조정했을 때의 brier 비교
   (`agent_score.brier`와 `scoring.summarize`의 model_brier)

## 하네스

- `prompts/system.md` 역할, 규칙, 출력 JSON 형식
- `skills/crash_review.md` 종목 검토 절차. Self-evolving이 고치는 파일
- `skills/disclosure_risk.md` 공시 위험 등급표 (툴의 risk_tag도 같은 기준)
- `skills/portfolio_review.md` 포트폴리오 추천 사유 쓰는 절차
- `guards.py` JSON 검사, 확률 조정 ±0.3 제한, judgment와 확률이 안 맞으면 고침
- 판단마다 prompt_version, skill_version을 저장해서 버전별로 성능 비교 가능

지금은 Python으로 직접 구현했고 pi, opencode 같은 프레임워크로 옮기는 건 다음에 검토.
