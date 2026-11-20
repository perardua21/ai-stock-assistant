import numpy as np
import pandas as pd

from ai.common.synthetic import make_market
from ai.features.tabular import TABULAR_FEATURES, build_tabular_features


def test_columns_and_ret5():
    panel, idx = make_market()
    f = build_tabular_features(panel, idx)
    assert list(f.columns) == ["code", "base_date"] + TABULAR_FEATURES
    a = panel[panel.code == "A"].reset_index(drop=True)
    fa = f[f.code == "A"].reset_index(drop=True)
    assert np.isclose(fa.loc[100, "ret_5"], a.loc[100, "close"] / a.loc[95, "close"] - 1)


def test_no_lookahead():
    """t 이후 데이터를 바꿔도 t의 피처는 그대로여야 한다."""
    panel, idx = make_market()
    base = build_tabular_features(panel, idx)
    cut = panel["trade_date"].unique()[200]
    changed = panel.copy()
    future = changed["trade_date"] > cut
    changed.loc[future, ["close", "volume", "foreign_net_buy"]] *= 3
    idx2 = idx.copy()
    idx2.loc[idx2.trade_date > cut, "close"] *= 0.5
    after = build_tabular_features(changed, idx2)
    m = base["base_date"] <= cut
    pd.testing.assert_frame_equal(base[m].reset_index(drop=True), after[m].reset_index(drop=True))


def test_sell_streak():
    panel, idx = make_market(n_days=80, codes=("A",))
    panel["foreign_net_buy"] = [1.0] * 70 + [-1.0] * 10
    f = build_tabular_features(panel, idx)
    assert f["foreign_sell_streak"].iloc[-1] == 10
    assert f["foreign_sell_streak"].iloc[69] == 0
