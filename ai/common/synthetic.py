"""
테스트용 가짜 시장 데이터
"""
import numpy as np
import pandas as pd

from ai.common.data import normalize_panel


def make_market(n_days=300, codes=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for code in codes:
        close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        vol = rng.integers(1e4, 1e5, n_days)
        rows.append(pd.DataFrame({
            "code": code, "trade_date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": vol, "trading_value": close * vol, "market_cap": close * 1e6,
            "foreign_net_buy": rng.normal(0, 1e6, n_days), "inst_net_buy": rng.normal(0, 1e6, n_days),
        }))
    panel = normalize_panel(pd.concat(rows))
    index_df = pd.DataFrame({"trade_date": dates, "close": 2500 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))})
    return panel, index_df
