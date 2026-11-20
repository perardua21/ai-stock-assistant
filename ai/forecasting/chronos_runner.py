"""
Chronos-2 실행

처음 돌릴 때 Hugging Face에서 모델을 받음 (amazon/chronos-2, 약 120M).
GPU 있으면 cuda, 없으면 cpu. cpu는 느려서 백테스트할 땐 --stride로 날짜를 줄이는 게 좋음.
추가 학습한 모델은 model_id에 저장 폴더 경로를 넣으면 됨.
"""
from __future__ import annotations

import numpy as np

from ai.forecasting.price_features import QUANTILE_LEVELS
from ai.forecasting.windows import Window


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosForecaster:
    def __init__(self, model_id: str = "amazon/chronos-2", device: str | None = None,
                 horizon: int = 5, batch_size: int = 256, quantile_levels: list[float] = QUANTILE_LEVELS):
        self.model_id = model_id
        self.device = device
        self.horizon = horizon
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self._pipe = None

    @property
    def pipeline(self):
        if self._pipe is None:
            from chronos import Chronos2Pipeline

            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device or _default_device())
        return self._pipe

    def predict_quantiles(self, windows: list[Window], chunk: int = 2048) -> np.ndarray:
        """windows → (n, horizon, n_quantiles). 공변량 유무는 window마다 같아야 한다."""
        out = []
        for i in range(0, len(windows), chunk):
            inputs = [w.to_input() for w in windows[i : i + chunk]]
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs,
                prediction_length=self.horizon,
                quantile_levels=self.quantile_levels,
                batch_size=self.batch_size,
            )
            # 각 원소: (n_variates=1, horizon, n_quantiles)
            out.extend(q[0].float().cpu().numpy() for q in quantiles)
        return np.stack(out) if out else np.empty((0, self.horizon, len(self.quantile_levels)))
