import numpy as np
import pandas as pd

from ai.common.synthetic import make_market
from ai.optimization.portfolio import PortfolioParams, optimize, recommend


def test_constraints_hold():
    rng = np.random.default_rng(1)
    n = 15
    a = rng.normal(0, 0.02, (60, n))
    cov = np.cov(a.T) * 5
    w = optimize(rng.normal(0.005, 0.01, n), cov, rng.uniform(0, 0.2, n), PortfolioParams(w_max=0.15))
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all() and (w <= 0.15 + 1e-6).all()


def test_crash_penalty_lowers_weight():
    cov = np.eye(3) * 0.001
    mu = np.array([0.01, 0.01, 0.01])
    p = np.array([0.0, 0.0, 0.25])
    w = optimize(mu, cov, p, PortfolioParams(gamma=2.0, w_max=1.0))
    assert w[2] < w[0]


def test_recommend_excludes_high_risk():
    panel, _ = make_market(n_days=100, codes=tuple("ABCDEFGHIJ"))
    cand = pd.DataFrame({"code": list("ABCDEFGHIJ"), "crash_probability": [0.05] * 9 + [0.9],
                         "q50_ret": np.linspace(-0.01, 0.02, 10)})
    out = recommend(cand, panel, PortfolioParams(w_max=0.2))
    assert "J" not in set(out["code"])
    assert np.isclose(out["weight"].sum(), 1.0)
    assert (out["weight"] <= 0.2 + 1e-6).all()
