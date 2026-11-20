"""
학습 데이터 만들기: 정형 피처 + Chronos 가격 피처 + 라벨을 (code, base_date)로 합침

Chronos 피처는 --stride 간격으로만 만들어 두는 경우가 많아서
가격 피처를 쓰면 그 날짜만 남음 (inner join)
"""
from __future__ import annotations

import pandas as pd

from ai.features.tabular import TABULAR_FEATURES
from ai.forecasting.price_features import PRICE_FEATURES

FEATURE_SETS = {
    "tabular": TABULAR_FEATURES,
    "price": PRICE_FEATURES,
    "tabular+price": TABULAR_FEATURES + PRICE_FEATURES,
}


def build_dataset(tabular: pd.DataFrame, labels: pd.DataFrame, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    """tabular: build_tabular_features 결과, labels: code/base_date/is_crash/min_return,
    forecast: forecast_feature (한 model_name만). forecast가 있으면 그 날짜만 남긴다."""
    df = tabular.merge(labels[["code", "base_date", "is_crash", "min_return"]], on=["code", "base_date"], how="inner")
    if forecast is not None:
        f = forecast[["code", "base_date"] + PRICE_FEATURES]
        df = df.merge(f, on=["code", "base_date"], how="inner")
    df["is_crash"] = df["is_crash"].astype(int)
    return df.sort_values(["base_date", "code"]).reset_index(drop=True)


def prediction_frame(tabular: pd.DataFrame, forecast: pd.DataFrame | None, date) -> pd.DataFrame:
    """운영용: 특정 날짜의 피처 행 (라벨 없음)."""
    d = pd.Timestamp(date)
    df = tabular[tabular["base_date"] == d]
    if forecast is not None:
        df = df.merge(forecast[forecast["base_date"] == d][["code", "base_date"] + PRICE_FEATURES], on=["code", "base_date"], how="left")
    return df.reset_index(drop=True)
