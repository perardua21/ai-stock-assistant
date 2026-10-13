"""
네이버 뉴스 검색 API — 종목/기업명 기반 뉴스 수집.

TODO:
  - 종목명만으로 검색하면 동명이인/동명기업 노이즈가 섞일 수 있어
    쿼리 보정(예: "{종목명} 주가" 등) 또는 사후 필터링 필요
"""
import os

from common.client import request_json
from common.storage import upsert_rows

_URL = "https://openapi.naver.com/v1/search/news.json"


def fetch_news(query: str, display: int = 100, start: int = 1) -> list[dict]:
    headers = {
        "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
    }
    resp = request_json(
        "GET", _URL, headers=headers,
        params={"query": query, "display": display, "start": start, "sort": "date"},
    )
    rows = []
    for item in resp.get("items", []):
        rows.append({
            "code": None,  # 종목코드 매칭은 저장 후 별도 배치로 처리
            "title": item["title"],
            "description": item.get("description"),
            "link": item.get("originallink") or item.get("link"),
            "published_at": item.get("pubDate"),
            "source_query": query,
        })
    return rows


def fetch_all(queries: list[str]) -> int:
    total = 0
    for q in queries:
        rows = fetch_news(q)
        total += upsert_rows("news_raw", rows, conflict_cols=["link", "published_at"])
    return total
