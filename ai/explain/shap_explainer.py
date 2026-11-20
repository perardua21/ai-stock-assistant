"""
분류기 SHAP 값

LightGBM은 predict(pred_contrib=True)로 TreeSHAP 값을 바로 줘서 shap 패키지는 안 씀.
log-odds 단위이고 양수면 급락 확률을 올린 피처.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values(booster, X: pd.DataFrame) -> np.ndarray:
    """(n, n_features) SHAP 값. 마지막 열(기준값)은 뺀다."""
    contrib = booster.predict(X, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


def explain_rows(booster, X: pd.DataFrame, feature_names: list[str], top_k: int = 5) -> list[list[dict]]:
    """행마다 [{feature, value, shap_value, contribution_pct}, ...] (|SHAP| 내림차순 top_k)."""
    sv = shap_values(booster, X[feature_names])
    out = []
    for i in range(len(X)):
        vals = sv[i]
        total = float(np.abs(vals).sum()) or 1.0
        order = np.argsort(-np.abs(vals))[:top_k]
        out.append([
            {
                "feature": feature_names[j],
                "value": None if pd.isna(X.iloc[i][feature_names[j]]) else float(X.iloc[i][feature_names[j]]),
                "shap_value": float(vals[j]),
                "contribution_pct": round(abs(float(vals[j])) / total * 100, 2),
            }
            for j in order
        ])
    return out


def global_importance(booster, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    sv = shap_values(booster, X[feature_names])
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
