"""
같은 종목·시점에서 뉴스를 전부 넣었을 때와 한 건 뺐을 때로 fusion 모델을
다시 돌려서 급락확률이 얼마나 바뀌는지(ablation) 비교한다.
"""


def ablate_news(predict_fn, structured_features, news_list: list[dict]):
    """
    predict_fn(structured_features, news_list) -> float  (급락확률 0~1)
    news_list: [{"id": ..., "text": ...}, ...]  해당 종목·시점에 걸린 전체 뉴스

    반환: [{news_id, prob_with_all, prob_without, delta}, ...] (|delta| 내림차순)
    """
    prob_with_all = predict_fn(structured_features, news_list)

    rows = []
    for i, news in enumerate(news_list):
        remaining = news_list[:i] + news_list[i + 1:]
        prob_without = predict_fn(structured_features, remaining)
        rows.append(
            {
                "news_id": news["id"],
                "prob_with_all": prob_with_all,
                "prob_without": prob_without,
                "delta": prob_with_all - prob_without,
            }
        )

    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows
