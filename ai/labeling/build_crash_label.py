"""
급락 라벨 (label_crash)

기준일 종가 대비 다음 5거래일 중 하루라도 종가가 -10% 이하면 급락.
기준일 뒤 가격을 봐야 하는 값이라 피처 코드(ai/features)랑 파일을 나눠 둠.

    python -m ai.labeling.build_crash_label
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CRASH_WINDOW_DAYS = 5
CRASH_THRESHOLD = -0.10


def build_crash_label(
    price_df: pd.DataFrame,
    window: int = CRASH_WINDOW_DAYS,
    threshold: float = CRASH_THRESHOLD,
) -> pd.DataFrame:
    """
    price_df: 종목 1개의 일별 시세 ['trade_date', 'close'], trade_date 오름차순.
    반환: 기준일별 1행 ['base_date', 'horizon_days', 'threshold', 'is_crash', 'min_return'].

    마지막 window일은 미래 데이터가 모자라서 라벨을 못 만드니 버린다.
    """
    df = price_df.sort_values("trade_date").reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    n = len(df)
    if n <= window:
        return pd.DataFrame(columns=["base_date", "horizon_days", "threshold", "is_crash", "min_return"])

    # 미래 window일 종가의 최솟값 (i+1 ~ i+window)
    future = np.lib.stride_tricks.sliding_window_view(closes[1:], window)  # (n-window, window)
    future_min = future.min(axis=1)
    min_return = future_min / closes[: n - window] - 1.0

    return pd.DataFrame({
        "base_date": df["trade_date"].iloc[: n - window].to_numpy(),
        "horizon_days": window,
        "threshold": threshold,
        "is_crash": min_return <= threshold,
        "min_return": min_return,
    })


def build_labels_panel(panel: pd.DataFrame, window: int = CRASH_WINDOW_DAYS, threshold: float = CRASH_THRESHOLD) -> pd.DataFrame:
    """전 종목 panel ['code', 'trade_date', 'close'] → 라벨 (code 컬럼 포함)."""
    out = []
    for code, g in panel.groupby("code", sort=False):
        lab = build_crash_label(g[["trade_date", "close"]], window, threshold)
        if len(lab):
            lab.insert(0, "code", code)
            out.append(lab)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    from ai.common.data import load_price_panel
    from ai.common.db import upsert_rows

    panel = load_price_panel()
    labels = build_labels_panel(panel)
    labels["base_date"] = pd.to_datetime(labels["base_date"]).dt.date
    n = upsert_rows("label_crash", labels.to_dict(orient="records"), ["code", "base_date", "horizon_days"])
    rate = labels["is_crash"].mean() if len(labels) else 0
    print(f"label_crash {n}행 저장, 급락 비율 {rate:.2%}")


if __name__ == "__main__":
    main()
