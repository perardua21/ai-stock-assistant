import numpy as np
import pandas as pd

from ai.common.synthetic import make_market
from ai.forecasting.price_features import QUANTILE_LEVELS, prob_below, quantiles_to_features
from ai.forecasting.windows import COVARIATES, make_windows


def test_prob_below_interpolates():
    rets = np.array([-0.20, -0.10, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12])
    assert np.isclose(prob_below(rets, QUANTILE_LEVELS, -0.10), 0.05)
    assert np.isclose(prob_below(rets, QUANTILE_LEVELS, -0.075), 0.075)
    assert prob_below(rets, QUANTILE_LEVELS, -0.5) == 0.0
    assert prob_below(rets, QUANTILE_LEVELS, 0.5) == 1.0


def test_quantiles_to_features():
    levels = QUANTILE_LEVELS
    last = 100.0
    # 5일 동안 점점 넓어지는 분포, 중앙값은 그대로
    q = np.stack([last + (np.array(levels) - 0.5) * 10 * (h + 1) for h in range(5)])
    f = quantiles_to_features(q, last, levels)
    assert np.isclose(f["q50_ret"], 0.0)
    assert f["q10_ret"] < 0 < f["q90_ret"]
    assert np.isclose(f["spread"], f["q90_ret"] - f["q10_ret"])
    assert 0 <= f["p_crash"] <= 1
    # 마지막 날 분포가 가장 넓으니 급락 확률도 마지막 날이 최대
    assert np.isclose(f["p_crash"], prob_below(q[-1] / last - 1, levels, -0.10))


def test_windows_end_at_base_date():
    panel, idx = make_market(n_days=200)
    ws = make_windows(panel, idx, start="2023-06-01", end="2023-07-31", context_length=128)
    assert ws, "window가 만들어져야 함"
    for w in ws[:50]:
        g = panel[(panel.code == w.code) & (panel.trade_date <= w.base_date)]
        assert w.last_close == g["close"].iloc[-1]
        assert np.allclose(w.target, g["close"].to_numpy()[-len(w.target):], rtol=1e-6)
        assert len(w.target) <= 128
        assert set(w.past_covariates) == set(COVARIATES)
        assert all(len(v) == len(w.target) for v in w.past_covariates.values())


def test_windows_stride_and_dates():
    panel, idx = make_market(n_days=200)
    d = panel["trade_date"].iloc[150]
    ws = make_windows(panel, idx, base_dates=[d])
    assert {w.base_date for w in ws} == {pd.Timestamp(d)}
    all_w = make_windows(panel, None, use_covariates=False)
    strided = make_windows(panel, None, stride=5, use_covariates=False)
    assert len(strided) * 5 >= len(all_w) - 10
    # stride를 써도 종목끼리 같은 날짜를 공유해야 함
    by_code = {}
    for w in strided:
        by_code.setdefault(w.code, set()).add(w.base_date)
    assert by_code["A"] == by_code["B"]
