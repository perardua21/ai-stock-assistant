"""
네이버 뉴스 검색 API

뉴스는 학습에 안 쓰고 에이전트가 search_news 툴로 그때그때 읽음.
제한: display 100개, start 1000까지, 날짜 지정 안 됨(최신순), 본문 없이 요약만.
"""
from __future__ import annotations

import html
import os
import re
from email.utils import parsedate_to_datetime

from ai.ingestion.common.client import request_json
from ai.common.db import upsert_rows

_URL = "https://openapi.naver.com/v1/search/news.json"
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str | None) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


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
        try:
            published = parsedate_to_datetime(item.get("pubDate"))
        except (TypeError, ValueError):
            published = None
        rows.append({
            "code": None,
            "title": _clean(item["title"]),
            "description": _clean(item.get("description")),
            "link": item.get("originallink") or item.get("link"),
            "published_at": published,
            "source_query": query,
        })
    return rows


def save_news(rows: list[dict], code: str | None = None) -> int:
    if code:
        rows = [{**r, "code": code} for r in rows]
    return upsert_rows("news_raw", rows, conflict_cols=["link", "published_at"])


def fetch_all(queries: list[str]) -> int:
    total = 0
    for q in queries:
        total += save_news(fetch_news(q))
    return total
