"""
OpenDART — 공시 목록 + 재무제표 수집.

TODO:
  - fetch_financial_statement(): 계정과목 매핑(매출/영업이익/순이익 등)을
    실제 fnlttSinglAcnt 응답 구조에 맞춰 보정
"""
import os

from common.client import request_json
from common.storage import upsert_rows

_BASE = "https://opendart.fss.or.kr/api"


def fetch_disclosures(code: str, start: str, end: str) -> list[dict]:
    resp = request_json(
        "GET", f"{_BASE}/list.json",
        params={
            "crtfc_key": os.environ["OPENDART_API_KEY"],
            "corp_code": code,
            "bgn_de": start,
            "end_de": end,
            "page_count": 100,
        },
    )
    rows = []
    for item in resp.get("list", []):
        rows.append({
            "code": code,
            "rcept_no": item["rcept_no"],
            "report_name": item.get("report_nm"),
            "submitted_at": item.get("rcept_dt"),
            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}",
            "raw_payload": item,
        })
    return rows


def fetch_financial_statement(code: str, year: int, report_code: str = "11011") -> list[dict]:
    """report_code 11011=사업보고서, 11012=반기, 11013/11014=1·3분기"""
    resp = request_json(
        "GET", f"{_BASE}/fnlttSinglAcnt.json",
        params={
            "crtfc_key": os.environ["OPENDART_API_KEY"],
            "corp_code": code,
            "bsns_year": year,
            "reprt_code": report_code,
        },
    )
    # TODO: 계정과목(sj_div, account_nm)별로 매출/영업이익/순이익 등을 추출해 매핑
    return [{
        "code": code,
        "fiscal_year": year,
        "fiscal_quarter": None,
        "report_type": report_code,
        "revenue": None,
        "operating_income": None,
        "net_income": None,
        "total_assets": None,
        "total_equity": None,
        "raw_payload": resp.get("list", []),
    }]


def fetch_all_disclosures(codes: list[str], start: str, end: str) -> int:
    total = 0
    for code in codes:
        rows = fetch_disclosures(code, start, end)
        total += upsert_rows("disclosure", rows, conflict_cols=["rcept_no"])
    return total
