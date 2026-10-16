# 시스템 구조

```
01 데이터 수집 (한투 API · OpenDART · 네이버뉴스)
        │
02 PostgreSQL 저장
        │
   ┌────┼──────────┬──────────────┐
03 Feature   04 Label(y)   05 GPT 구조화
   Engineering  (급락 여부)   (sentiment·impact·risk_score)
   (ai/features)  (ai/labeling)
        │            │              │
        └─────┬──────┴──────────────┘
              │
06 Classification(split.py로 train/test 분리) · Optimization
              │
07 GPT 설명 (XAI)
              │
08 React 화면 (Vercel)
```

Feature Engineering과 Label 생성, GPT 구조화는 저장된 데이터를 각자 병렬로 처리하며,
세 결과가 모두 Classification 단계의 입력으로 모입니다.

Label(`ai/labeling/build_crash_label.py`)은 Feature(`ai/features/`)와 파일을 분리해 뒀습니다.
기준일 이후 가격을 봐야 만들어지는 값이라 같이 두면 헷갈리기 쉬워서입니다. train/test도
`ai/classification/split.py` 한 곳에서만 날짜로 나누고, structured/text_bert/fusion이 전부
이걸 가져다 씁니다. 겹치는지는 `ai/classification/tests/test_split.py`로 확인합니다.

## 고도화 방향 — Multimodal Classification & XAI (교수님 피드백 반영)

- **Multimodal Classification**: 정형데이터(가격·재무·수급) feature + 뉴스 텍스트를
  BERT/Transformer로 fine-tuning한 임베딩을 결합(concat/attention)해 급락확률 예측.
  기존 Logistic Regression·RF·XGBoost는 정형 데이터 branch(`ai/classification/structured`)의
  baseline으로 유지.
- **End-to-End 구조 & XAI**: Classification의 판단 근거와 Optimization에서 사용한
  변수·제약조건을 GPT가 함께 참고하도록 설계. GPT는 최종 결과만 전달받는 설명
  챗봇이 아니라, 분석 과정 전체를 이해하는 XAI 인터페이스로 재설계.

## XAI 구현 방법 (확정)

Classification이 정형 데이터 branch와 뉴스 텍스트 branch로 나뉘는 멀티모달 구조이기 때문에,
각 branch 성격에 맞는 설명 기법을 따로 적용하고 그 결과를 합쳐서 GPT에게 넘깁니다.

| 대상 | 질문 | 방법 | 역할 |
| --- | --- | --- | --- |
| 정형 데이터 | 어떤 feature가 급락확률에 얼마나 영향을 줬나 | **SHAP** | 가격·재무·수급 등 각 feature의 기여도를 %로 산출 |
| 뉴스 텍스트 | 뉴스의 어떤 단어/표현이 영향을 줬나 | **Integrated Gradients** | BERT 입력 임베딩 기준 토큰별 중요도 계산 |
| 뉴스 단위 | 뉴스 하나가 없었다면 결과가 얼마나 달라졌나 | **Ablation** | 뉴스 1건 제거 후 재추론해 확률 변화(delta) 계산 |

세 방법의 결과(`ai/explain/shap_explainer.py`, `integrated_gradients.py`, `ablation.py`)는
`ai/explain/aggregate.py`에서 Optimization 제약조건과 함께 하나의 JSON 컨텍스트로 합쳐지고,
`ai/explain/gpt_explainer.py`가 이 컨텍스트를 GPT에 전달해 최종 추천 사유를 생성합니다.
(대응하는 DB 테이블: `feature_importance`, `news_token_importance`, `news_ablation`,
`explanation_log` — `db/schema.md` 참고)

자세한 발표 내용은 `AI_Stock_Assistant_발표.pptx` 및 `docs/dev_reference.pdf` 참고.
