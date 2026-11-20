"""
Chronos-2 입력 만들기. 종목, 기준일마다 기준일까지의 과거 구간을 잘라냄

target은 수정 종가, 공변량(past_covariates)은
  log_volume   log(1 + 거래량)
  foreign_flow 외국인 순매수 / 거래대금
  inst_flow    기관 순매수 / 거래대금
  mkt_ret      KOSPI 일간 로그수익률
미래 값을 아는 공변량은 없어서 future_covariates는 안 씀.
거래일 순서대로 자르기 때문에 주말, 공휴일은 신경 안 써도 됨.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COVARIATES = ["log_volume", "foreign_flow", "inst_flow", "mkt_ret"]


@dataclass
class Window:
    code: str
    base_date: pd.Timestamp
    last_close: float
    target: np.ndarray
    past_covariates: dict[str, np.ndarray] = field(default_factory=dict)

    def to_input(self) -> dict:
        d = {"target": self.target.astype(np.float32)}
        if self.past_covariates:
            d["past_covariates"] = {k: v.astype(np.float32) for k, v in self.past_covariates.items()}
        return d


def add_covariates(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df[["trade_date", "close"]].sort_values("trade_date").copy()
    idx["mkt_ret"] = np.log(idx["close"]).diff()
    df = panel.merge(idx[["trade_date", "mkt_ret"]], on="trade_date", how="left")
    tv = df["trading_value"].replace(0, np.nan)
    df["log_volume"] = np.log1p(df["volume"])
    df["foreign_flow"] = df["foreign_net_buy"] / tv
    df["inst_flow"] = df["inst_net_buy"] / tv
    for c in COVARIATES:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values(["code", "trade_date"]).reset_index(drop=True)


def pick_dates(all_dates, start=None, end=None, stride: int = 1) -> list[pd.Timestamp]:
    """전체 거래일 달력에서 [start, end] 구간을 stride 간격으로 고른다.
    종목마다 따로 고르면 날짜가 엇갈려서 같은 날 종목끼리 비교(상위 k 적중률)가 안 된다."""
    d = pd.Series(sorted(pd.to_datetime(pd.unique(all_dates))))
    if start:
        d = d[d >= pd.Timestamp(start)]
    if end:
        d = d[d <= pd.Timestamp(end)]
    return list(d.iloc[::stride])


def make_windows(
    panel: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    base_dates: list | None = None,
    start: str | None = None,
    end: str | None = None,
    stride: int = 1,
    context_length: int = 256,
    min_context: int = 60,
    use_covariates: bool = True,
) -> list[Window]:
    """base_dates를 주면 그 날짜만, 없으면 [start, end]를 stride 간격으로.
    window의 마지막 값이 기준일 종가이고 그 뒤 데이터는 들어가지 않는다."""
    if use_covariates:
        if index_df is None:
            raise ValueError("공변량을 쓰려면 index_df(KOSPI 지수)가 필요함")
        df = add_covariates(panel, index_df)
    else:
        df = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)

    if base_dates is None:
        calendar = index_df["trade_date"] if index_df is not None and len(index_df) else df["trade_date"]
        base_dates = pick_dates(calendar, start, end, stride)
    wanted = set(pd.to_datetime(base_dates))

    windows: list[Window] = []
    for code, g in df.groupby("code", sort=False):
        dates = pd.DatetimeIndex(g["trade_date"])
        close64 = g["close"].to_numpy(dtype=float)
        closes = close64.astype(np.float32)
        cov = {c: g[c].to_numpy(dtype=np.float32) for c in COVARIATES} if use_covariates else {}
        for i in np.flatnonzero(dates.isin(wanted)):
            if i < min_context - 1:
                continue
            s = max(0, i + 1 - context_length)
            windows.append(Window(
                code=code,
                base_date=dates[i],
                last_close=float(close64[i]),
                target=closes[s : i + 1],
                past_covariates={k: v[s : i + 1] for k, v in cov.items()},
            ))
    return windows
