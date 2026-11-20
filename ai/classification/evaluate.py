"""
평가 지표. 모든 모델을 같은 함수로 잼

auc             ROC-AUC
pr_auc          Average Precision. 급락이 드물어서 이게 더 중요함
base_rate       test 기간 실제 급락 비율 (pr_auc 기준선)
precision_at_k  매일 위험 상위 k개 중 실제 급락 비율의 평균
lift_at_k       precision_at_k / base_rate
brier           확률 오차 (확률 내는 모델만)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_k(df: pd.DataFrame, score_col: str, k: int = 20, date_col: str = "base_date", y_col: str = "is_crash") -> float:
    tops = df.sort_values(score_col, ascending=False).groupby(date_col).head(k)
    return float(tops.groupby(date_col)[y_col].mean().mean())


def evaluate(df: pd.DataFrame, score_col: str, k: int = 20, is_probability: bool = True, y_col: str = "is_crash") -> dict:
    d = df.dropna(subset=[score_col])
    y, s = d[y_col].astype(int).to_numpy(), d[score_col].to_numpy(dtype=float)
    base = float(y.mean()) if len(y) else float("nan")
    out = {"n": int(len(d)), "base_rate": base}
    if len(np.unique(y)) < 2:
        out.update(auc=float("nan"), pr_auc=float("nan"))
    else:
        out.update(auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))
    p_at_k = precision_at_k(d, score_col, k, y_col=y_col)
    out.update({f"precision_at_{k}": p_at_k, f"lift_at_{k}": p_at_k / base if base else float("nan")})
    out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1))) if is_probability else float("nan")
    return out


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(4)
