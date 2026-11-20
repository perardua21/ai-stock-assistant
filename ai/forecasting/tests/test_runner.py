"""ChronosForecaster가 Chronos-2 출력 모양을 제대로 풀어 쓰는지 확인 (torch 없이 가짜 파이프라인)."""
import sys
import types

import numpy as np

from ai.common.synthetic import make_market
from ai.forecasting.price_features import QUANTILE_LEVELS
from ai.forecasting.windows import make_windows


class _T:
    """torch.Tensor 흉내: float().cpu().numpy()만 있으면 됨"""

    def __init__(self, a):
        self.a = a

    def __getitem__(self, i):
        return _T(self.a[i])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakePipeline:
    seen = []

    @classmethod
    def from_pretrained(cls, model_id, device_map=None):
        cls.loaded = (model_id, device_map)
        return cls()

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size):
        FakePipeline.seen.append(len(inputs))
        out = []
        for x in inputs:
            last = float(x["target"][-1])
            # 실제 Chronos-2처럼 원소 하나가 (n_variates=1, horizon, n_quantiles)
            q = np.stack([last * (1 + (np.array(quantile_levels) - 0.5) * 0.04 * (h + 1)) for h in range(prediction_length)])
            out.append(_T(q[None]))
        return out, [None] * len(out)


def test_forecaster_shapes(monkeypatch):
    monkeypatch.setitem(sys.modules, "chronos", types.SimpleNamespace(Chronos2Pipeline=FakePipeline))
    from ai.forecasting.chronos_runner import ChronosForecaster
    from ai.forecasting.run_forecast import forecast_rows

    panel, idx = make_market(n_days=200)
    ws = make_windows(panel, idx, start="2023-08-01", stride=2)
    fc = ChronosForecaster(device="cpu")
    q = fc.predict_quantiles(ws, chunk=7)
    assert q.shape == (len(ws), 5, len(QUANTILE_LEVELS))
    assert FakePipeline.loaded == ("amazon/chronos-2", "cpu")
    assert max(FakePipeline.seen) <= 7
    rows = forecast_rows(ws, q, "chronos-2+cov", 5)
    assert abs(rows[0]["q50_ret"]) < 1e-6 and rows[0]["q10_ret"] < 0
    assert len(rows[0]["raw_quantiles"]["values"]) == 5
