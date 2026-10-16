"""
XAI 파이프라인 엔트리포인트 (skeleton).
ai.classification(structured / text_bert / fusion)이 먼저 학습·추론돼 있어야 동작한다.

실행 순서:
  1. shap_explainer.explain_structured()        -- 정형 데이터 영향 (SHAP)
  2. integrated_gradients.explain_news_tokens() -- 뉴스 중요 표현 (Integrated Gradients)
  3. ablation.ablate_news()                     -- 뉴스 1건 제거 시 변화 (Ablation)
  4. aggregate.build_explanation_context()      -- 위 3개 + optimization 결과 통합
  5. gpt_explainer.generate_explanation()       -- 최종 추천 사유 생성
"""

# TODO: ai.classification 모델 학습이 끝난 뒤 실제 연결
if __name__ == "__main__":
    print("classification 모델이 준비되면 이 스크립트에서 1~5단계를 실행합니다.")
