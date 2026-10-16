"""
text_bert branch(BERT/Transformer)가 뉴스에서 어떤 단어를 보고 급락확률을
높였는지, Integrated Gradients로 토큰 단위 중요도를 계산한다.
"""
import torch
from captum.attr import LayerIntegratedGradients


def explain_news_tokens(model, tokenizer, text: str, target_class: int = 1):
    """
    model: ai.classification.text_bert 에서 fine-tuning한 BERT 분류 모델
    text: 뉴스 원문 1건
    반환: [{token, ig_score}, ...] (중요도 내림차순)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    def forward(ids, attention_mask):
        return model(input_ids=ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
    baseline = torch.zeros_like(input_ids)

    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        additional_forward_args=(inputs["attention_mask"],),
        target=target_class,
        return_convergence_delta=True,
    )
    scores = attributions.sum(dim=-1).squeeze(0).detach().numpy()

    rows = [{"token": t, "ig_score": float(s)} for t, s in zip(tokens, scores)]
    rows.sort(key=lambda r: abs(r["ig_score"]), reverse=True)
    return rows
