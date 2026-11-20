"""수집 코드 테스트. pykrx·OpenDART·네이버는 실제로 부르지 않고 응답 모양만 흉내 낸다."""
import io
import json
import sys
import types
import zipfile

import pandas as pd
import pytest


@pytest.fixture
def fake_pykrx(monkeypatch):
    calls = {"cap": [], "flow": []}

    def ohlcv(start, end, code, adjusted=True):
        idx = pd.bdate_range(start, end, name="날짜")
        df = pd.DataFrame({"시가": 100, "고가": 110, "저가": 90, "종가": 100, "거래량": 1000}, index=idx)
        df.iloc[1, df.columns.get_loc("종가")] = 0  # 거래정지일
        return df

    def cap(start, end, code):
        calls["cap"].append((start, end))
        idx = pd.bdate_range(start, end, name="날짜")
        return pd.DataFrame({"시가총액": 1e10, "거래량": 1000, "거래대금": 1e5, "상장주식수": 1e6}, index=idx)

    def flow(start, end, code):
        calls["flow"].append((start, end))
        return pd.DataFrame()  # 로그인 실패하면 빈 표가 옴

    def index_ohlcv(start, end, code, name_display=True):
        idx = pd.bdate_range(start, end, name="날짜")
        return pd.DataFrame({"시가": 2500.0, "고가": 2510.0, "저가": 2490.0, "종가": 2500.0, "거래량": 1e9, "거래대금": 1e13}, index=idx)

    stock = types.SimpleNamespace(get_market_ohlcv=ohlcv, get_market_cap=cap,
                                  get_market_trading_value_by_date=flow, get_index_ohlcv=index_ohlcv)
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=stock))
    sys.modules.pop("ai.ingestion.pykrx_market", None)
    import ai.ingestion.pykrx_market as m
    yield m, calls
    sys.modules.pop("ai.ingestion.pykrx_market", None)


def test_chunks_cover_range(fake_pykrx):
    m, _ = fake_pykrx
    parts = list(m._chunks("20230101", "20251231"))
    assert parts[0][0] == "20230101" and parts[-1][1] == "20251231"
    # 구간끼리 겹치거나 빠진 날이 없어야 함
    days = [pd.date_range(s, e) for s, e in parts]
    assert sum(len(d) for d in days) == len(pd.date_range("20230101", "20251231"))
    assert all(len(d) <= 365 for d in days)


def test_fetch_stock_merges_and_skips_halt(fake_pykrx):
    m, calls = fake_pykrx
    rows = m.fetch_stock("005930", "20240101", "20250301")
    assert len(calls["cap"]) == 2  # 1년 단위로 나눠서 호출
    assert all(r["close_price"] for r in rows)
    assert len(rows) == len(pd.bdate_range("20240101", "20250301")) - 1
    assert rows[0]["market_cap"] == 1e10 and rows[0]["trading_value"] == 1e5
    assert rows[0]["foreign_net_buy"] is None  # 수급이 비어도 에러 없이 None


def test_fetch_index(fake_pykrx):
    m, _ = fake_pykrx
    rows = m.fetch_index("20240101", "20240131")
    assert rows[0]["index_code"] == "1001" and rows[0]["close_price"] == 2500.0


def test_adjusted_mismatch(fake_pykrx):
    m, _ = fake_pykrx
    import datetime as dt
    d = dt.date(2024, 1, 2)
    rows = [{"trade_date": d, "close_price": 100.0}]
    assert not m.adjusted_mismatch(rows, {d: 100.2})
    assert m.adjusted_mismatch(rows, {d: 500.0})  # 1:5 분할 후 과거 가격이 바뀐 경우
    assert not m.adjusted_mismatch(rows, {})


def test_opendart_corp_code_and_pages(monkeypatch, tmp_path):
    from ai.ingestion import opendart_filing as od

    xml = ("<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
           "<stock_code>005930</stock_code></list><list><corp_code>00000001</corp_code>"
           "<stock_code> </stock_code></list></result>").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CORPCODE.xml", xml)

    class Resp:
        content = buf.getvalue()

        def raise_for_status(self):
            pass

    monkeypatch.setenv("OPENDART_API_KEY", "x")
    monkeypatch.setattr(od.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(od, "_CACHE", tmp_path / "corp.json")
    monkeypatch.setattr(od, "_corp_map", None)
    assert od.corp_code_of("005930") == "00126380"
    assert json.loads((tmp_path / "corp.json").read_text())["005930"] == "00126380"

    seen = []

    def fake_request(method, url, params=None, **k):
        seen.append(params)
        page = params["page_no"]
        return {"status": "000", "total_page": 2,
                "list": [{"rcept_no": f"2024000{page}", "report_nm": "유상증자결정 ", "rcept_dt": "20240105"}]}

    monkeypatch.setattr(od, "request_json", fake_request)
    rows = od.fetch_disclosures("005930", "20240101", "20240131")
    assert [r["rcept_no"] for r in rows] == ["20240001", "20240002"]
    assert seen[0]["corp_code"] == "00126380"
    assert rows[0]["report_name"] == "유상증자결정"
    assert od.fetch_disclosures("999999", "20240101", "20240131") == []


def test_naver_parsing(monkeypatch):
    from ai.ingestion import naver_news as nn

    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(nn, "request_json", lambda *a, **k: {"items": [{
        "title": "<b>삼성전자</b> &quot;실적&quot;", "description": "요약", "originallink": "http://a",
        "pubDate": "Wed, 18 Nov 2026 09:30:00 +0900"}]})
    rows = nn.fetch_news("삼성전자", display=10)
    assert rows[0]["title"] == '삼성전자 "실적"'
    assert rows[0]["published_at"].hour == 9 and rows[0]["published_at"].utcoffset().total_seconds() == 9 * 3600


def test_index_failure_does_not_crash(fake_pykrx, monkeypatch):
    m, _ = fake_pykrx

    def broken(*a, **k):
        raise KeyError("지수명")  # KRX 로그인 안 됐을 때 pykrx가 내는 에러

    monkeypatch.setattr(m.stock, "get_index_ohlcv", broken)
    assert m.fetch_index("20240101", "20240131") == []


def test_opendart_bad_key_message(monkeypatch, tmp_path):
    from ai.ingestion import opendart_filing as od

    class Resp:
        content = b'{"status":"010","message":"\ub4f1\ub85d\ub418\uc9c0 \uc54a\uc740 \ud0a4"}'
        text = content.decode()

        def raise_for_status(self):
            pass

    monkeypatch.setenv("OPENDART_API_KEY", "wrong")
    monkeypatch.setattr(od.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match="OPENDART_API_KEY"):
        od._download_corp_codes()


def test_client_does_not_retry_401(monkeypatch):
    import requests

    from ai.ingestion.common import client

    calls = []

    class R:
        status_code = 401

        def raise_for_status(self):
            raise requests.HTTPError("401", response=self)

    def fake(*a, **k):
        calls.append(1)
        return R()

    monkeypatch.setattr(client.requests, "request", fake)
    with pytest.raises(requests.HTTPError):
        client.request_json("GET", "http://x")
    assert len(calls) == 1


def test_db_url_uses_psycopg2(monkeypatch):
    from ai.common.db import db_url

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    assert db_url() == "postgresql+psycopg2://u:p@localhost:5432/db"
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:5432/db")
    assert db_url() == "postgresql+psycopg2://u:p@localhost:5432/db"
