"""
Chronos-2 예측 분위수를 가격 피처로 바꿈

p_crash      5일 중 어느 날이든 -10% 이하일 확률. 날짜별 확률을 분위수 보간으로 구하고 그중 최댓값
             (기간 최저가 기준 확률보다 조금 낮게 나옴)
q05_ret_min  5일 동안 하위 5% 분위수 수익률 중 가장 낮은 값
q10/50/90_ret  5일째 분위수 수익률
spread       q90 - q10
skew         (q90 - q50) - (q50 - q10), 음수면 아래 꼬리가 김
"""
from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
PRICE_FEATURES = ["p_crash", "q05_ret_min", "q10_ret", "q50_ret", "q90_ret", "spread", "skew"]


def _q(levels, rets, level):
    return rets[..., levels.index(level)]


def prob_below(ret_quantiles: np.ndarray, levels: list[float], threshold: float) -> float:
    """분위수 곡선 (levels → 수익률)에서 P(수익률 ≤ threshold)를 선형 보간."""
    xs = np.maximum.accumulate(np.asarray(ret_quantiles, dtype=float))  # 단조 증가 보정
    lv = np.asarray(levels, dtype=float)
    if threshold <= xs[0]:
        # 가장 낮은 분위수보다 아래: 첫 두 분위수의 기울기로 선형 외삽 (0 아래로는 안 내려감)
        slope = (lv[1] - lv[0]) / max(xs[1] - xs[0], 1e-9)
        return float(max(0.0, lv[0] - (xs[0] - threshold) * slope))
    if threshold >= xs[-1]:
        return 1.0
    return float(np.interp(threshold, xs, lv))


def quantiles_to_features(quantiles: np.ndarray, last_close: float, levels: list[float] = QUANTILE_LEVELS,
                          threshold: float = -0.10) -> dict:
    rets = np.asarray(quantiles, dtype=float) / last_close - 1.0  # (h, q)
    p_by_day = [prob_below(rets[h], levels, threshold) for h in range(rets.shape[0])]
    q10, q50, q90 = _q(levels, rets[-1], 0.1), _q(levels, rets[-1], 0.5), _q(levels, rets[-1], 0.9)
    return {
        "p_crash": float(max(p_by_day)),
        "q05_ret_min": float(_q(levels, rets, 0.05).min()),
        "q10_ret": float(q10),
        "q50_ret": float(q50),
        "q90_ret": float(q90),
        "spread": float(q90 - q10),
        "skew": float((q90 - q50) - (q50 - q10)),
    }
