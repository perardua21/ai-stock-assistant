"""
포트폴리오 최적화

    max  μ·w - λ·wᵀΣw - γ·p·w
    s.t. Σw = 1, 0 <= w <= w_max, 급락 확률 > crash_cutoff 인 종목 제외

μ  Chronos-2 5일 뒤 중앙값 수익률 (q50_ret), 없으면 0
Σ  최근 60거래일 수익률 공분산 x 5, 대각 쪽으로 20% 줄임
p  급락 확률 (에이전트가 조정했으면 조정값)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioParams:
    lam: float = 3.0            # 위험 회피 정도 λ
    gamma: float = 0.5          # 급락 위험 페널티 γ
    w_max: float = 0.10         # 종목당 최대 비중
    crash_cutoff: float = 0.30  # 이 확률을 넘으면 후보에서 제외
    shrink: float = 0.2         # 공분산 축소 비율
    lookback: int = 60          # 공분산 계산 기간 (거래일)
    horizon: int = 5


def covariance(returns: pd.DataFrame, horizon: int = 5, shrink: float = 0.2) -> np.ndarray:
    """returns: 날짜 × 종목 일간 수익률 → horizon 기준 축소 공분산."""
    s = returns.cov().to_numpy() * horizon
    s = np.nan_to_num(s, nan=0.0)
    d = np.diag(np.diag(s))
    return (1 - shrink) * s + shrink * d


def optimize(mu: np.ndarray, cov: np.ndarray, p_crash: np.ndarray, params: PortfolioParams) -> np.ndarray:
    n = len(mu)
    if n == 0:
        return np.array([])
    w_max = max(params.w_max, 1.0 / n)  # 후보가 적으면 상한을 풀어 줘야 합이 1이 된다

    def neg_obj(w):
        return -(mu @ w - params.lam * w @ cov @ w - params.gamma * p_crash @ w)

    def neg_grad(w):
        return -(mu - 2 * params.lam * cov @ w - params.gamma * p_crash)

    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_obj, w0, jac=neg_grad, method="SLSQP", bounds=[(0.0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success else w0, 0, None)
    w[w < 1e-4] = 0.0
    return w / w.sum()


def recommend(candidates: pd.DataFrame, price_panel: pd.DataFrame, params: PortfolioParams | None = None) -> pd.DataFrame:
    """
    candidates: code, crash_probability, (선택) q50_ret
    price_panel: code, trade_date, close  (기준일까지)
    반환: code, weight, expected_return, crash_probability (비중 내림차순, 0 비중 제외)
    """
    params = params or PortfolioParams()
    c = candidates[candidates["crash_probability"] <= params.crash_cutoff].copy()
    if c.empty:
        return pd.DataFrame(columns=["code", "weight", "expected_return", "crash_probability"])
    if "q50_ret" not in c:
        c["q50_ret"] = 0.0
    c["q50_ret"] = c["q50_ret"].fillna(0.0)

    px = price_panel[price_panel["code"].isin(c["code"])].pivot(index="trade_date", columns="code", values="close")
    rets = px.sort_index().pct_change().iloc[-params.lookback:]
    c = c[c["code"].isin(rets.columns)].reset_index(drop=True)
    cov = covariance(rets[c["code"]], params.horizon, params.shrink)

    w = optimize(c["q50_ret"].to_numpy(float), cov, c["crash_probability"].to_numpy(float), params)
    out = pd.DataFrame({"code": c["code"], "weight": w, "expected_return": c["q50_ret"], "crash_probability": c["crash_probability"]})
    return out[out["weight"] > 0].sort_values("weight", ascending=False).reset_index(drop=True)


def params_dict(params: PortfolioParams) -> dict:
    return asdict(params)
