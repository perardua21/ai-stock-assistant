"""
SHAP(정형) + Integrated Gradients(뉴스 단어) + Ablation(뉴스 단위) + Optimization
제약조건을 하나의 설명 컨텍스트(JSON)로 합쳐 GPT 프롬프트에 그대로 넣을 수 있게 만든다.
"""


def build_explanation_context(
    code: str,
    crash_probability: float,
    shap_rows: list[dict],
    ig_by_news: dict,            # {news_id: [{token, ig_score}, ...]}
    ablation_rows: list[dict],
    optimization_context: dict,  # {weight, constraints, objective 등}
    top_k: int = 5,
) -> dict:
    top_features = shap_rows[:top_k]
    top_news = ablation_rows[:top_k]

    news_detail = []
    for row in top_news:
        tokens = ig_by_news.get(row["news_id"], [])[:top_k]
        news_detail.append({**row, "key_tokens": tokens})

    return {
        "code": code,
        "crash_probability": crash_probability,
        "structured_evidence": top_features,   # SHAP
        "news_evidence": news_detail,          # Ablation + Integrated Gradients
        "optimization_context": optimization_context,
    }
