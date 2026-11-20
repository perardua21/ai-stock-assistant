"""
정형 피처 계산

기준일 t 피처는 t까지 데이터로만 계산 (rolling은 전부 과거 방향).
종목마다 크기가 달라서 원값 대신 비율, 평균 대비 값으로 만듦.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TABULAR_FEATURES = [
    # 가격
    "ret_1", "ret_5", "ret_20", "ret_60",
    "vol_20", "vol_60", "vol_ratio",
    "drawdown_252", "ma_gap_20", "ma_gap_60",
    "range_5", "max_drop_20", "down_days_5", "limit_down_60",
    # 거래량·규모
    "volume_ratio_20", "log_trading_value_20", "log_market_cap",
    # 수급
    "foreign_flow_5", "foreign_flow_20", "inst_flow_5", "inst_flow_20", "foreign_sell_streak",
    # 시장
    "mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "rel_ret_20", "beta_60",
]


def _sell_streak(s: pd.Series) -> pd.Series:
    """연속 순매도 일수 (오늘 순매수면 0)."""
    neg = (s < 0).astype(int)
    groups = (neg == 0).cumsum()
    return neg.groupby(groups).cumsum()


def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    c = g["close"]
    r = np.log(c).diff()
    out = pd.DataFrame(index=g.index)

    for n in (1, 5, 20, 60):
        out[f"ret_{n}"] = c / c.shift(n) - 1
    out["vol_20"] = r.rolling(20, min_periods=15).std()
    out["vol_60"] = r.rolling(60, min_periods=40).std()
    out["vol_ratio"] = out["vol_20"] / out["vol_60"]
    out["drawdown_252"] = c / c.rolling(252, min_periods=60).max() - 1
    out["ma_gap_20"] = c / c.rolling(20, min_periods=15).mean() - 1
    out["ma_gap_60"] = c / c.rolling(60, min_periods=40).mean() - 1
    out["range_5"] = ((g["high"] - g["low"]) / c).rolling(5, min_periods=3).mean()
    simple = c.pct_change()
    out["max_drop_20"] = simple.rolling(20, min_periods=15).min()
    out["down_days_5"] = (simple < 0).astype(float).rolling(5, min_periods=5).sum()
    out["limit_down_60"] = (simple <= -0.25).astype(float).rolling(60, min_periods=1).max()

    vol = g["volume"]
    out["volume_ratio_20"] = np.log1p(vol) - np.log1p(vol.rolling(20, min_periods=15).mean())
    tv = g["trading_value"]
    out["log_trading_value_20"] = np.log1p(tv.rolling(20, min_periods=15).mean())
    out["log_market_cap"] = np.log1p(g["market_cap"])

    for who, col in (("foreign", "foreign_net_buy"), ("inst", "inst_net_buy")):
        for n in (5, 20):
            out[f"{who}_flow_{n}"] = g[col].rolling(n, min_periods=n).sum() / tv.rolling(n, min_periods=n).sum()
    out["foreign_sell_streak"] = _sell_streak(g["foreign_net_buy"].fillna(0))

    # 시장 (지수 수익률은 미리 붙여 둔 mkt_r 사용)
    mr = g["mkt_r"]
    out["beta_60"] = (r * mr).rolling(60, min_periods=40).mean() - r.rolling(60, min_periods=40).mean() * mr.rolling(60, min_periods=40).mean()
    out["beta_60"] = out["beta_60"] / mr.rolling(60, min_periods=40).var(ddof=0)
    return out


def build_tabular_features(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """전 종목 panel → 종목·기준일별 정형 피처."""
    idx = index_df[["trade_date", "close"]].rename(columns={"close": "mkt_close"}).sort_values("trade_date")
    idx["mkt_r"] = np.log(idx["mkt_close"]).diff()
    idx["mkt_ret_5"] = idx["mkt_close"] / idx["mkt_close"].shift(5) - 1
    idx["mkt_ret_20"] = idx["mkt_close"] / idx["mkt_close"].shift(20) - 1
    idx["mkt_vol_20"] = idx["mkt_r"].rolling(20, min_periods=15).std()

    df = panel.sort_values(["code", "trade_date"]).merge(idx, on="trade_date", how="left")
    cols = ["close", "high", "low", "volume", "trading_value", "market_cap", "foreign_net_buy", "inst_net_buy", "mkt_r"]
    feats = df.groupby("code", group_keys=False, sort=False)[cols].apply(_per_stock)
    df = pd.concat([df[["code", "trade_date", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20"]], feats], axis=1)
    df["rel_ret_20"] = df["ret_20"] - df["mkt_ret_20"]
    df = df.rename(columns={"trade_date": "base_date"})
    df[TABULAR_FEATURES] = df[TABULAR_FEATURES].replace([np.inf, -np.inf], np.nan)
    return df[["code", "base_date"] + TABULAR_FEATURES]
