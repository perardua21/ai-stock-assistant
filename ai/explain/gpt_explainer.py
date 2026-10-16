"""
aggregate.build_explanation_context() 결과를 GPT에 넣어
사용자가 이해할 수 있는 추천 사유(자연어)를 생성한다.
"""
import json
import os

from openai import OpenAI

_SYSTEM_PROMPT = (
    "당신은 주식 급락 위험 분석 결과를 설명하는 어시스턴트입니다. "
    "아래 JSON은 Classification 모델의 SHAP(정형데이터 기여도), "
    "Integrated Gradients(뉴스 단어 중요도), Ablation(뉴스 1건 제거 시 확률 변화), "
    "Portfolio Optimization에서 사용한 제약조건을 담고 있습니다. "
    "이 근거들을 바탕으로 사용자가 이해할 수 있는 한국어 추천 사유를 작성하세요. "
    "모델 내부 수치를 그대로 나열하지 말고, 어떤 정형 요인과 어떤 뉴스가 "
    "왜 중요했는지 자연스럽게 설명하세요."
)


def generate_explanation(context: dict) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    )
    return resp.choices[0].message.content
