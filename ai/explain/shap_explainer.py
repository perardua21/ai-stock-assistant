"""
structured branch(RF/XGBoost/Logistic Regression)가 급락확률을 예측할 때
가격/재무/수급 등 각 feature가 얼마나 영향을 줬는지 SHAP으로 %를 계산한다.
"""
import numpy as np
import shap


def explain_structured(model, X, feature_names):
    """
    model: ai.classification.structured 에서 학습된 트리/선형 모델
    X: 구조화된 입력 feature 1행 (해당 종목·시점)
    반환: [{feature, shap_value, contribution_pct}, ...] (기여도 내림차순)
    """
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X)

    shap_values = explainer.shap_values(X)
    values = shap_values[0] if isinstance(shap_values, list) else shap_values
    values = np.asarray(values).reshape(-1)

    total_abs = np.abs(values).sum() or 1.0
    rows = [
        {
            "feature": name,
            "shap_value": float(v),
            "contribution_pct": round(abs(float(v)) / total_abs * 100, 2),
        }
        for name, v in zip(feature_names, values)
    ]
    rows.sort(key=lambda r: r["contribution_pct"], reverse=True)
    return rows
