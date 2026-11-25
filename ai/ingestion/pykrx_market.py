# INGEST_FIX_V1: historical resume and KRX error logging
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


def _chunks(start: str, end: str, days: int = 90):
    """KRX 조회 기간 제한 때문에 1년 단위로 나눠서 부른다."""
    s = dt.datetime.strptime(start, "%Y%m%d").date()
    e = dt.datetime.strptime(end, "%Y%m%d").date()
    while s <= e:
        t = min(s + dt.timedelta(days=days - 1), e)
        yield s.strftime("%Y%m%d"), t.strftime("%Y%m%d")
        s = t + dt.timedelta(days=1)


def _krx(fn, start, end, code, cols: dict) -> pd.DataFrame:
    """KRX 데이터를 365일 단위로 조회하고 실패 구간만 90일로 재조회."""
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    parts = []
    failed_chunks = []

    def request(s, e):
        """주어진 기간을 최대 3회 조회."""
        for attempt in range(1, 4):
            try:
                df = fn(s, e, code)

                if (
                    df is not None
                    and not df.empty
                    and set(cols).issubset(df.columns)
                ):
                    return df[list(cols)].rename(columns=cols)

                reason = "empty response / missing columns"

            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"

            print(
                f"[warn] {code} {fn.__name__} "
                f"{s}~{e} {reason} ({attempt}/3)",
                flush=True,
            )

            if attempt < 3:
                time.sleep(2 * attempt)

        return None

    # 1차: 365일 단위로 조회
    for s, e in _chunks(start, end, days=365):
        result = request(s, e)

        if result is not None:
            parts.append(result)
        else:
            print(
                f"[info] {code} {s}~{e}: "
                "90일 단위로 재조회",
                flush=True,
            )

            # 2차: 실패한 구간만 90일 단위로 재조회
            for sub_s, sub_e in _chunks(s, e, days=90):
                sub_result = request(sub_s, sub_e)

                if sub_result is not None:
                    parts.append(sub_result)
                else:
                    failed_chunks.append(f"{sub_s}~{sub_e}")

                time.sleep(0.5)

        time.sleep(0.5)

    # 최종 실패 구간 기록
    if failed_chunks:
        log = Path("logs/krx_failed_chunks.jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)

        with log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "code": code,
                        "source": fn.__name__,
                        "chunks": failed_chunks,
                    },
                    ensure_ascii=False,
                ) + "\n"
            )

        print(
            f"[warn] {code} {fn.__name__}: "
            f"failed chunks {failed_chunks}",
            flush=True,
        )

    if not parts:
        return pd.DataFrame(columns=list(cols.values()))

    out = pd.concat(parts)

    return out[~out.index.duplicated(keep="last")]


def fetch_stock(code: str, start: str, end: str) -> list[dict]:
    """종목 하나의 [start, end] 일별 데이터. 날짜는 YYYYMMDD."""
    ohlcv = stock.get_market_ohlcv(start, end, code, adjusted=True)
    if ohlcv is None or ohlcv.empty:
        return []

    df = ohlcv.rename(columns={
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
    })

    # 실제 주가 데이터가 존재하는 첫 날짜부터 KRX 조회
    actual_start = pd.Timestamp(df.index.min()).strftime("%Y%m%d")
    actual_end = pd.Timestamp(df.index.max()).strftime("%Y%m%d")

    cap = _krx(
        stock.get_market_cap,
        actual_start,
        actual_end,
        code,
        {"시가총액": "market_cap", "거래대금": "trading_value"},
    )

    flow = _krx(
        stock.get_market_trading_value_by_date,
        actual_start,
        actual_end,
        code,
        {"외국인합계": "foreign_net_buy", "기관합계": "inst_net_buy"},
    )

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
    """Bulk historical resume: skip codes already stored at least once.

    This prevents repeatedly fetching late-listed stocks. Incomplete KRX fields
    are handled separately by repair_missing_krx.py. This is not a daily refresh.
    """
    df = read_sql("SELECT DISTINCT code FROM price_daily")
    return set(df["code"].astype(str))



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
