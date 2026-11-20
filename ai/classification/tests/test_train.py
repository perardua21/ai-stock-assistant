"""가짜 시장 데이터로 학습·평가 흐름이 끝까지 도는지 확인 (DB 없이)."""
import numpy as np
import pandas as pd
import pytest

from ai.classification.evaluate import evaluate, precision_at_k
from ai.classification.train import run_experiments
from ai.common.synthetic import make_market
from ai.features.tabular import build_tabular_features
from ai.forecasting.price_features import PRICE_FEATURES
from ai.labeling.build_crash_label import build_labels_panel


def _fake_forecast(tabular, labels, seed=0):
    """정답과 약하게 상관된 가짜 Chronos 피처 (실제 모델 대신)."""
    rng = np.random.default_rng(seed)
    f = labels[["code", "base_date"]].copy()
    signal = labels["min_return"].to_numpy()
    f["p_crash"] = np.clip(-signal * 2 + rng.normal(0, 0.1, len(f)), 0, 1)
    for c in PRICE_FEATURES[1:]:
        f[c] = rng.normal(0, 0.02, len(f))
    return f.iloc[::5]


def test_precision_at_k():
    df = pd.DataFrame({"base_date": [1, 1, 1, 2, 2, 2], "s": [3, 2, 1, 3, 2, 1], "is_crash": [1, 0, 0, 0, 0, 1]})
    assert precision_at_k(df, "s", k=1) == 0.5
    m = evaluate(df, "s", k=1, is_probability=False)
    assert m["n"] == 6 and 0 <= m["auc"] <= 1


@pytest.mark.parametrize("use_lgbm", [False, True])
def test_experiments_run(use_lgbm):
    if use_lgbm:
        pytest.importorskip("lightgbm")
    panel, idx = make_market(n_days=500, codes=tuple(f"S{i:02d}" for i in range(30)), seed=3)
    # 급락이 가끔 일어나도록 큰 하락을 섞는다
    rng = np.random.default_rng(5)
    shocks = rng.random(len(panel)) < 0.01
    panel.loc[shocks, "close"] *= 0.85
    tabular = build_tabular_features(panel, idx)
    labels = build_labels_panel(panel)
    train_end = str(panel["trade_date"].iloc[350].date())
    table, best = run_experiments(tabular, labels, {"fake": _fake_forecast(tabular, labels)}, train_end, k=5, use_lgbm=use_lgbm)
    assert "vol_rule" in table.index and "chronos[fake]" in table.index
    assert table.loc["chronos[fake]", "auc"] > 0.6  # 정답과 상관된 피처라 AUC가 높아야 함
    if use_lgbm:
        assert best["forecast_model"] == "fake"
