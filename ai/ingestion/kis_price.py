"""
한국투자증권(KIS) Open API — 일별 시세·수급 수집.

TODO:
  - get_access_token(): 모의/실전 서버 전환 (KIS_IS_MOCK)
  - fetch_daily_price(): 실제 응답 필드명에 맞춰 매핑 보정 (문서 기준 응답 샘플 확인 필요)
"""
import os

from common.client import TokenCache, request_json
from common.storage import upsert_rows

_MOCK_BASE = "https://openapivts.koreainvestment.com:29443"
_REAL_BASE = "https://openapi.koreainvestment.com:9443"

_token_cache = TokenCache()


def _base_url() -> str:
    return _MOCK_BASE if os.environ.get("KIS_IS_MOCK", "true").lower() == "true" else _REAL_BASE


def _fetch_token() -> tuple[str, int]:
    resp = request_json(
        "POST",
        f"{_base_url()}/oauth2/tokenP",
        json={
            "grant_type": "client_credentials",
            "appkey": os.environ["KIS_APP_KEY"],
            "appsecret": os.environ["KIS_APP_SECRET"],
        },
    )
    return resp["access_token"], int(resp.get("expires_in", 86400))


def get_access_token() -> str:
    return _token_cache.get(_fetch_token)


def fetch_daily_price(code: str, start: str, end: str) -> list[dict]:
    """code(종목코드) 하나에 대해 [start, end] 구간의 일별 시세를 반환."""
    token = get_access_token()
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": os.environ["KIS_APP_KEY"],
        "appsecret": os.environ["KIS_APP_SECRET"],
        "tr_id": "FHKST03010100",  # 국내주식 일별시세 조회 TR ID (문서 확인 후 보정)
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }
    resp = request_json(
        "GET", f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=headers, params=params,
    )
    rows = []
    for item in resp.get("output2", []):
        rows.append({
            "code": code,
            "trade_date": item["stck_bsop_date"],
            "open_price": item["stck_oprc"],
            "high_price": item["stck_hgpr"],
            "low_price": item["stck_lwpr"],
            "close_price": item["stck_clpr"],
            "volume": item["acml_vol"],
            "foreign_net_buy": None,  # 별도 수급 TR 호출 필요 — 1차는 가격만
            "inst_net_buy": None,
        })
    return rows


def fetch_all(codes: list[str], start: str, end: str) -> int:
    total = 0
    for code in codes:
        rows = fetch_daily_price(code, start, end)
        total += upsert_rows("price_daily", rows, conflict_cols=["code", "trade_date"])
    return total
