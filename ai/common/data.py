"""
DB에서 pandas로 읽어오는 함수들. 컬럼 이름은 여기서 맞춰 둠

price panel: code, trade_date, open, high, low, close, volume,
             trading_value, market_cap, foreign_net_buy, inst_net_buy
index: trade_date, close
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai.common.db import read_sql

KOSPI_INDEX_CODE = "1001"


def load_price_panel(start: str | None = None, end: str | None = None, codes: list[str] | None = None) -> pd.DataFrame:
    sql = """
        SELECT code, trade_date,
               open_price AS open, high_price AS high, low_price AS low, close_price AS close,
               volume, trading_value, market_cap, foreign_net_buy, inst_net_buy
        FROM price_daily
        WHERE (:start IS NULL OR trade_date >= CAST(:start AS DATE))
          AND (:end IS NULL OR trade_date <= CAST(:end AS DATE))
    """
    params = {"start": start, "end": end}
    if codes:
        sql += " AND code = ANY(:codes)"
        params["codes"] = list(codes)
    df = read_sql(sql + " ORDER BY code, trade_date", params)
    return normalize_panel(df)


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    num = ["open", "high", "low", "close", "volume", "trading_value", "market_cap", "foreign_net_buy", "inst_net_buy"]
    for c in num:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    # 거래대금이 비어 있으면 종가×거래량으로 근사
    df["trading_value"] = df["trading_value"].fillna(df["close"] * df["volume"])
    return df.sort_values(["code", "trade_date"]).reset_index(drop=True)


def load_index(start: str | None = None, end: str | None = None, index_code: str = KOSPI_INDEX_CODE) -> pd.DataFrame:
    df = read_sql(
        """
        SELECT trade_date, close_price AS close
        FROM index_daily
        WHERE index_code = :ic
          AND (:start IS NULL OR trade_date >= CAST(:start AS DATE))
          AND (:end IS NULL OR trade_date <= CAST(:end AS DATE))
        ORDER BY trade_date
        """,
        {"ic": index_code, "start": start, "end": end},
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = df["close"].astype("float64")
    return df


def load_labels(horizon: int = 5) -> pd.DataFrame:
    df = read_sql(
        "SELECT code, base_date, is_crash, min_return FROM label_crash WHERE horizon_days = :h",
        {"h": horizon},
    )
    df["base_date"] = pd.to_datetime(df["base_date"])
    return df


def load_forecast_features(model_name: str, date: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM forecast_feature WHERE model_name = :m"
    if date:
        sql += " AND base_date = CAST(:d AS DATE)"
    df = read_sql(sql, {"m": model_name, "d": date})
    df["base_date"] = pd.to_datetime(df["base_date"])
    return df.drop(columns=["created_at", "raw_quantiles"], errors="ignore")


def load_stock_names() -> dict[str, str]:
    df = read_sql("SELECT code, name FROM stock_master")
    return dict(zip(df["code"], df["name"]))
