"""
급락 라벨 생성 (label_crash)

정의: 기준일 종가 대비 이후 5거래일 중 하루라도 종가가 -10% 이하면 급락(1).
structured/text_bert/fusion이 전부 이 함수로 만든 라벨을 쓴다.

feature 계산(ai/features)과는 파일을 분리해 뒀다 — 라벨은 기준일 이후
가격을 봐야 해서, 같은 파일에 있으면 미래 값이 feature로 섞이기 쉽다.
"""
from __future__ import annotations

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
    closes = df["close"].to_numpy()
    n = len(df)

    rows = []
    for i in range(n):
        future = closes[i + 1 : i + 1 + window]
        if len(future) < window:
            continue
        min_return = float(future.min() / closes[i] - 1.0)
        rows.append({
            "base_date": df.loc[i, "trade_date"],
            "horizon_days": window,
            "threshold": threshold,
            "is_crash": bool(min_return <= threshold),
            "min_return": min_return,
        })
    return pd.DataFrame(rows)
