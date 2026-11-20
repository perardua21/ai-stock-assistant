"""
에이전트 툴

툴은 전부 기준일까지 정보만 돌려줌. 그날 계산해 둔 값은 MarketSnapshot에서 읽고,
뉴스/공시/메모리는 함수를 넘겨받아서 씀 (테스트할 때 가짜로 바꾸기 쉬움).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ai.forecasting.price_features import PRICE_FEATURES

KST = dt.timezone(dt.timedelta(hours=9))

# 공시 위험 태그. skills/disclosure_risk.md 표랑 같은 기준
DISCLOSURE_RULES = [
    ("high", ["의견거절", "부적정", "한정의견", "상장폐지", "관리종목", "매매거래정지", "거래정지", "횡령", "배임",
              "회생절차", "파산", "부도", "영업정지", "불성실공시"]),
    ("medium", ["유상증자", "전환사채", "신주인수권부사채", "교환사채", "감자", "최대주주변경", "최대주주 변경", "소송",
                "계약해지", "자기주식처분", "투자주의", "투자경고"]),
    ("positive", ["자기주식취득", "현금ㆍ현물배당", "현금·현물배당", "배당결정", "공급계약체결"]),
]


def tag_disclosure(report_name: str) -> str:
    name = (report_name or "").replace(" ", "")
    for tag, words in DISCLOSURE_RULES:
        if any(w.replace(" ", "") in name for w in words):
            return tag
    return "neutral"


# 그날 계산해 둔 값들
@dataclass
class MarketSnapshot:
    run_date: pd.Timestamp
    scores: pd.DataFrame            # code, crash_probability, top_features(list[dict])
    tabular: pd.DataFrame           # code + TABULAR_FEATURES (run_date 행)
    forecast: pd.DataFrame | None   # code + PRICE_FEATURES (run_date 행)
    names: dict[str, str] = field(default_factory=dict)
    base_rate: float | None = None  # 학습 데이터 평균 급락 비율

    def row(self, df: pd.DataFrame | None, code: str) -> dict:
        if df is None or df.empty:
            return {}
        r = df[df["code"] == code]
        return {} if r.empty else r.iloc[0].to_dict()


def _round(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


@dataclass
class Toolbox:
    snapshot: MarketSnapshot
    news_fn: Callable[[str, int], list[dict]] | None = None             # (query, limit) → 기사 목록
    disclosure_fn: Callable[[str, str, str], list[dict]] | None = None  # (code, start, end) → 공시 목록
    memory_fn: Callable[[str, int], list[dict]] | None = None           # (code, limit) → 과거 판단
    portfolio_fn: Callable[[dict], list[dict]] | None = None            # (params) → 추천 비중
    log: list[dict] = field(default_factory=list)

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.snapshot.run_date)

    # 툴
    def get_crash_signal(self, code: str) -> dict:
        s = self.snapshot
        sc = s.row(s.scores, code)
        fc = s.row(s.forecast, code)
        return {
            "code": code, "name": s.names.get(code, ""), "run_date": str(self.as_of.date()),
            "crash_probability": _round(sc.get("crash_probability")),
            "base_rate": _round(s.base_rate),
            "top_features": sc.get("top_features", []),
            "chronos": {k: _round(fc.get(k)) for k in PRICE_FEATURES} if fc else None,
        }

    def get_price_summary(self, code: str) -> dict:
        r = self.snapshot.row(self.snapshot.tabular, code)
        keys = ["ret_1", "ret_5", "ret_20", "vol_20", "drawdown_252", "volume_ratio_20",
                "foreign_flow_5", "foreign_flow_20", "inst_flow_5", "foreign_sell_streak", "rel_ret_20", "mkt_ret_5"]
        return {"code": code, **{k: _round(r.get(k)) for k in keys}} if r else {"code": code, "error": "데이터 없음"}

    def search_news(self, query: str, limit: int = 10) -> dict:
        if self.news_fn is None:
            return {"items": [], "note": "뉴스 검색 사용 안 함"}
        end = dt.datetime.combine(self.as_of.date(), dt.time(23, 59), KST)
        items = []
        for n in self.news_fn(query, max(limit * 2, 20)):
            pub = n.get("published_at")
            if pub is not None and pub > end:
                continue  # 기준일 이후 기사 차단
            items.append({"title": n.get("title"), "summary": (n.get("description") or "")[:200],
                          "published_at": str(pub) if pub else None, "link": n.get("link")})
            if len(items) >= limit:
                break
        return {"query": query, "items": items}

    def search_disclosures(self, code: str, days: int = 30) -> dict:
        if self.disclosure_fn is None:
            return {"items": [], "note": "공시 검색 사용 안 함"}
        end = self.as_of.date()
        start = end - dt.timedelta(days=days)
        rows = self.disclosure_fn(code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        items = [{"date": r.get("submitted_at"), "title": r.get("report_name"), "risk_tag": tag_disclosure(r.get("report_name")),
                  "url": r.get("url")} for r in rows if str(r.get("submitted_at") or "")[:8] <= end.strftime("%Y%m%d")]
        return {"code": code, "items": items[:20]}

    def recall_memory(self, code: str, limit: int = 5) -> dict:
        if self.memory_fn is None:
            return {"items": []}
        return {"code": code, "items": self.memory_fn(code, limit)}

    def optimize_portfolio(self, lam: float | None = None, gamma: float | None = None, w_max: float | None = None) -> dict:
        if self.portfolio_fn is None:
            return {"items": [], "note": "포트폴리오 툴 사용 안 함"}
        params = {k: v for k, v in {"lam": lam, "gamma": gamma, "w_max": w_max}.items() if v is not None}
        return {"items": self.portfolio_fn(params)}


    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            result = {"error": f"알 수 없는 툴: {name}"}
        else:
            try:
                result = fn(**args)
            except Exception as e:  # 툴 오류는 에이전트에게 그대로 알려 준다
                result = {"error": f"{type(e).__name__}: {e}"}
        self.log.append({"tool": name, "args": args, "result": result})
        return json.dumps(result, ensure_ascii=False, default=str)[:6000]


def _spec(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


_CODE = {"code": {"type": "string", "description": "6자리 종목코드"}}
TOOL_SPECS = [
    _spec("get_crash_signal", "급락 분류기 확률, SHAP 상위 피처, Chronos 예측 분포 요약", _CODE, ["code"]),
    _spec("get_price_summary", "최근 수익률·변동성·거래량·수급 지표", _CODE, ["code"]),
    _spec("search_news", "네이버 뉴스 검색 (기준일까지의 최신 기사)",
          {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, ["query"]),
    _spec("search_disclosures", "OpenDART 공시 목록과 위험 태그 (기준일 이전 days일)",
          {**_CODE, "days": {"type": "integer", "default": 30}}, ["code"]),
    _spec("recall_memory", "이 종목에 대한 과거 판단과 채점 결과", {**_CODE, "limit": {"type": "integer", "default": 5}}, ["code"]),
    _spec("optimize_portfolio", "수익률 − λ·위험 − γ·급락위험 최적화로 추천 비중 계산",
          {"lam": {"type": "number"}, "gamma": {"type": "number"}, "w_max": {"type": "number"}}, []),
]
TOOL_NAMES = {s["function"]["name"] for s in TOOL_SPECS}


def specs(names: list[str]) -> list[dict]:
    return [s for s in TOOL_SPECS if s["function"]["name"] in names]
