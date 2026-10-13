# 시스템 구조

```
01 데이터 수집 (한투 API · OpenDART · 네이버뉴스)
        │
02 PostgreSQL 저장
        │
   ┌────┴─────┐
03 Feature     04 GPT 구조화
   Engineering   (sentiment·impact·risk_score)
        │          │
        └────┬─────┘
             │
05 Classification · Optimization
             │
06 React 화면 (Vercel)
```

Feature Engineering과 GPT 구조화는 저장된 데이터를 각자 병렬로 처리하며,
두 결과가 모두 Classification 단계의 입력으로 모입니다.

## 고도화 방향 (교수님 피드백 반영)

- **Multimodal Classification**: 정형데이터(가격·재무·수급) feature + 뉴스 텍스트를
  BERT/Transformer로 fine-tuning한 임베딩을 결합(concat/attention)해 급락확률 예측.
  기존 Logistic Regression·RF·XGBoost는 정형 데이터 branch의 baseline으로 유지.
- **End-to-End 구조 & XAI**: Classification의 판단 근거(feature importance 등)와
  Optimization에서 사용한 변수·제약조건을 GPT가 함께 참고하도록 설계.
  GPT는 최종 결과만 전달받는 설명 챗봇이 아니라, 분석 과정 전체를 이해하는
  XAI 인터페이스로 재설계.

자세한 내용은 발표자료(`AI_Stock_Assistant_발표.pptx`) 07쪽 참고.
