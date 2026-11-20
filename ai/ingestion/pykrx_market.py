"""
pykrx로 일봉, 시가총액, 거래대금, 외국인/기관 순매수, KOSPI 지수 수집

일봉은 수정주가(네이버). 액면분할을 급락으로 잘못 잡지 않으려고.
시가총액, 순매수, 지수는 KRX라서 .env에 KRX_ID, KRX_PW 필요.

수정주가는 오늘 기준으로 과거 가격이 바뀜. 며칠치만 갱신할 때 분할이 있었던 종목은
DB 가격이랑 안 맞아서, 겹치는 날 종가가 다르면 그 종목은 처음부터 다시 받음.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from ai.common.data import KOSPI_INDEX_CODE
from ai.common.db import read_sql, upsert_rows  # .env를 여기서 먼저 읽어야 pykrx가 KRX 계정을 본다

from pykrx import stock  # noqa: E402


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _chunks(start: str, end: str, days: int = 365):
    """KRX 조회 기간 제한 때문에 1년 단위로 나눠서 부른다."""
    s = dt.datetime.strptime(start, "%Y%m%d").date()
    e = dt.datetime.strptime(end, "%Y%m%d").date()
    while s <= e:
        t = min(s + dt.timedelta(days=days - 1), e)
        yield s.strftime("%Y%m%d"), t.strftime("%Y%m%d")
        s = t + dt.timedelta(days=1)


def _krx(fn, start, end, code, cols: dict) -> pd.DataFrame:
    """KRX 함수를 기간별로 나눠 부르고 필요한 컬럼만 영어 이름으로 바꿔서 합친다.
    로그인 실패 등으로 빈 표가 오면 빈 DataFrame."""
    parts = []
    for s, e in _chunks(start, end):
        df = fn(s, e, code)
        if df is not None and len(df) and set(cols) <= set(df.columns):
            parts.append(df[list(cols)].rename(columns=cols))
    if not parts:
        return pd.DataFrame(columns=list(cols.values()))
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")]


def fetch_stock(code: str, start: str, end: str) -> list[dict]:
    """종목 하나의 [start, end] 일별 데이터. 날짜는 YYYYMMDD."""
    ohlcv = stock.get_market_ohlcv(start, end, code, adjusted=True)
    if ohlcv is None or ohlcv.empty:
        return []
    df = ohlcv.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
    cap = _krx(stock.get_market_cap, start, end, code, {"시가총액": "market_cap", "거래대금": "trading_value"})
    flow = _krx(stock.get_market_trading_value_by_date, start, end, code,
                {"외국인합계": "foreign_net_buy", "기관합계": "inst_net_buy"})
    df = df.join(cap, how="left").join(flow, how="left")

    rows = []
    for d, r in df.iterrows():
        close = _f(r.get("close"))
        if not close:
            continue  # 거래정지일은 종가 0으로 오기도 함
        vol = _f(r.get("volume"))
        rows.append({
            "code": code,
            "trade_date": pd.Timestamp(d).date(),
            "open_price": _f(r.get("open")),
            "high_price": _f(r.get("high")),
            "low_price": _f(r.get("low")),
            "close_price": close,
            "volume": int(vol) if vol is not None else None,
            "trading_value": _f(r.get("trading_value")),
            "market_cap": _f(r.get("market_cap")),
            "foreign_net_buy": _f(r.get("foreign_net_buy")),
            "inst_net_buy": _f(r.get("inst_net_buy")),
        })
    return rows


def fetch_index(start: str, end: str, index_code: str = KOSPI_INDEX_CODE) -> list[dict]:
    parts = []
    for s, e in _chunks(start, end):
        try:
            parts.append(stock.get_index_ohlcv(s, e, index_code, name_display=False))
        except Exception as ex:  # KRX 로그인이 안 되면 pykrx 안에서 KeyError 등이 남
            print(f"  [warn] 지수 조회 실패 {s}~{e}: {type(ex).__name__}: {ex}")
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return []
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")]
    rows = []
    for d, r in df.iterrows():
        vol = _f(r.get("거래량"))
        rows.append({
            "index_code": index_code,
            "trade_date": pd.Timestamp(d).date(),
            "open_price": _f(r.get("시가")),
            "high_price": _f(r.get("고가")),
            "low_price": _f(r.get("저가")),
            "close_price": _f(r.get("종가")),
            "volume": int(vol) if vol is not None else None,
            "trading_value": _f(r.get("거래대금")),
        })
    return rows


def adjusted_mismatch(rows: list[dict], stored: dict, tol: float = 0.005) -> bool:
    """새로 받은 종가와 DB 종가가 겹치는 날에 tol 이상 차이 나면 True (분할 등으로 수정주가가 바뀐 것)."""
    for r in rows:
        old = stored.get(r["trade_date"])
        if old and abs(r["close_price"] / old - 1) > tol:
            return True
    return False


def _stored_closes(codes, start, end) -> dict:
    try:
        df = read_sql(
            "SELECT code, trade_date, close_price FROM price_daily WHERE code = ANY(:c) "
            "AND trade_date BETWEEN CAST(:s AS DATE) AND CAST(:e AS DATE)",
            {"c": list(codes), "s": start, "e": end},
        )
    except Exception:
        return {}
    out: dict = {}
    for r in df.itertuples():
        out.setdefault(r.code, {})[r.trade_date] = float(r.close_price)
    return out


def _done_codes(start: str, end: str) -> set[str]:
    """--resume용: 이미 [start, end]를 거의 다 받아 둔 종목."""
    df = read_sql(
        "SELECT code, MIN(trade_date) lo, MAX(trade_date) hi FROM price_daily GROUP BY code"
    )
    s = dt.datetime.strptime(start, "%Y%m%d").date() + dt.timedelta(days=10)
    e = dt.datetime.strptime(end, "%Y%m%d").date() - dt.timedelta(days=5)
    return {r.code for r in df.itertuples() if r.lo <= s and r.hi >= e}


def fetch_all(codes: list[str], start: str, end: str, sleep: float = 0.3,
              full_start: str = "20150101", resume: bool = False) -> int:
    n = upsert_rows("index_daily", fetch_index(start, end), ["index_code", "trade_date"])
    print(f"KOSPI 지수 {n}행")
    if n == 0:
        print("  지수를 못 받음. KRX 로그인(.env의 KRX_ID, KRX_PW)을 확인하세요.")

    if resume:
        done = _done_codes(start, end)
        codes = [c for c in codes if c not in done]
        print(f"이미 받은 종목 {len(done)}개는 건너뜀")
    stored = _stored_closes(codes, start, end)

    total, failed, refetched, no_krx = 0, [], [], 0
    for i, code in enumerate(codes, 1):
        try:
            rows = fetch_stock(code, start, end)
            if rows and adjusted_mismatch(rows, stored.get(code, {})):
                refetched.append(code)
                rows = fetch_stock(code, full_start, end)
            if rows and all(r["market_cap"] is None for r in rows):
                no_krx += 1
            total += upsert_rows("price_daily", rows, ["code", "trade_date"])
        except Exception as e:  # 한 종목 실패로 전체가 멈추지 않게
            failed.append(code)
            print(f"  [error] {code}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(codes)} 종목, 누적 {total}행")
        time.sleep(sleep)

    if refetched:
        print(f"수정주가가 바뀐 종목 {len(refetched)}개 다시 받음: {', '.join(refetched[:10])}")
    if no_krx:
        print(f"시가총액/수급이 비어 있는 종목 {no_krx}개. KRX 로그인이 안 됐을 수 있음.")
    if failed:
        print(f"실패 {len(failed)}개: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    return total
