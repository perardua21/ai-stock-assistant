"""
OpenDART 공시 목록

corp_code는 종목코드가 아니라 8자리 고유번호라서 corpCode.xml로 매핑을 만들어서 씀
(data/dart_corp_codes.json에 저장). 예전 코드는 종목코드를 그대로 넣어서 결과가 안 나왔음.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import xml.etree.ElementTree as ET
import zipfile

import requests

from ai.common.db import ROOT, upsert_rows
from ai.ingestion.common.client import request_json

_BASE = "https://opendart.fss.or.kr/api"
_CACHE = ROOT / "data" / "dart_corp_codes.json"
_corp_map: dict[str, str] | None = None


def _download_corp_codes() -> dict[str, str]:
    resp = requests.get(f"{_BASE}/corpCode.xml", params={"crtfc_key": os.environ["OPENDART_API_KEY"]}, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):  # zip이 아니면 에러 응답 (키 오류 등)
        raise RuntimeError(f"corpCode.xml 대신 에러 응답이 옴. OPENDART_API_KEY 확인: {resp.text[:200]}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])
    mapping = {}
    for item in ET.fromstring(xml_bytes).iter("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        if stock_code:
            mapping[stock_code] = item.findtext("corp_code").strip()
    return mapping


def corp_code_of(stock_code: str, refresh: bool = False) -> str | None:
    """종목코드(6자리) → OpenDART corp_code(8자리)."""
    global _corp_map
    if _corp_map is None or refresh:
        if _CACHE.exists() and not refresh:
            _corp_map = json.loads(_CACHE.read_text(encoding="utf-8"))
        else:
            _corp_map = _download_corp_codes()
            pathlib.Path(_CACHE).parent.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(json.dumps(_corp_map), encoding="utf-8")
    return _corp_map.get(stock_code)


def fetch_disclosures(code: str, start: str, end: str, max_pages: int = 10) -> list[dict]:
    """종목 하나의 [start, end] 공시 목록. start/end는 YYYYMMDD."""
    corp_code = corp_code_of(code)
    if corp_code is None:
        return []
    rows, page = [], 1
    while page <= max_pages:
        resp = request_json(
            "GET", f"{_BASE}/list.json",
            params={
                "crtfc_key": os.environ["OPENDART_API_KEY"],
                "corp_code": corp_code,
                "bgn_de": start,
                "end_de": end,
                "page_no": page,
                "page_count": 100,
            },
        )
        if resp.get("status") != "000":  # 013 = 조회 결과 없음
            break
        for item in resp.get("list", []):
            rows.append({
                "code": code,
                "rcept_no": item["rcept_no"],
                "report_name": (item.get("report_nm") or "").strip(),
                "submitted_at": item.get("rcept_dt"),
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}",
                "raw_payload": item,
            })
        if page >= int(resp.get("total_page", 1)):
            break
        page += 1
    return rows


def fetch_all_disclosures(codes: list[str], start: str, end: str) -> int:
    total = 0
    for code in codes:
        total += upsert_rows("disclosure", fetch_disclosures(code, start, end), conflict_cols=["rcept_no"])
    return total
