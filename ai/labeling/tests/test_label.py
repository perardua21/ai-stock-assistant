import numpy as np
import pandas as pd

from ai.labeling.build_crash_label import build_crash_label, build_labels_panel


def _prices(closes):
    return pd.DataFrame({"trade_date": pd.bdate_range("2025-01-01", periods=len(closes)), "close": closes})


def test_crash_detected_within_window():
    closes = [100, 101, 99, 89, 95, 100, 100, 100]  # 3일 뒤 -11%
    lab = build_crash_label(_prices(closes))
    assert bool(lab.loc[0, "is_crash"]) is True
    assert np.isclose(lab.loc[0, "min_return"], -0.11)


def test_no_crash_and_tail_dropped():
    closes = [100] * 10
    lab = build_crash_label(_prices(closes))
    assert len(lab) == 5  # 마지막 5일은 미래가 부족해 버림
    assert not lab["is_crash"].any()


def test_matches_loop_definition():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, 200)))
    lab = build_crash_label(_prices(closes))
    for i in [0, 50, 120]:
        expect = closes[i + 1 : i + 6].min() / closes[i] - 1
        assert np.isclose(lab.loc[i, "min_return"], expect)


def test_panel_keeps_codes_separate():
    a = _prices([100] * 8).assign(code="A")
    b = _prices([100, 50, 50, 50, 50, 50, 50, 50]).assign(code="B")
    lab = build_labels_panel(pd.concat([a, b]))
    assert not lab[lab.code == "A"]["is_crash"].any()
    assert lab[lab.code == "B"]["is_crash"].iloc[0]
