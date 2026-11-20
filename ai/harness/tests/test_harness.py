"""하네스 테스트. 가짜 LLM으로 툴 호출, 미래 정보 차단, 출력 검사, 채점 확인"""
import datetime as dt
import json

import pandas as pd

from ai.harness.agent import HarnessAgent
from ai.harness.context import build_context
from ai.harness.evolve import collect_cases
from ai.harness.guards import validate_decision
from ai.harness.llm import FakeLLM
from ai.harness.scoring import score_frame, summarize
from ai.harness.tools import KST, MarketSnapshot, Toolbox, tag_disclosure

RUN = pd.Timestamp("2026-11-20")


def _snapshot():
    return MarketSnapshot(
        run_date=RUN,
        scores=pd.DataFrame({"code": ["000001"], "crash_probability": [0.08],
                             "top_features": [[{"feature": "vol_20", "shap_value": 0.4, "contribution_pct": 40.0}]]}),
        tabular=pd.DataFrame({"code": ["000001"], "ret_5": [-0.06], "vol_20": [0.04]}),
        forecast=pd.DataFrame({"code": ["000001"], "p_crash": [0.05], "q05_ret_min": [-0.12], "q10_ret": [-0.07],
                               "q50_ret": [0.0], "q90_ret": [0.06], "spread": [0.13], "skew": [-0.01]}),
        names={"000001": "테스트전자"},
        base_rate=0.02,
    )


def _news(query, n):
    return [
        {"title": "미래 기사", "description": "", "published_at": dt.datetime(2026, 11, 21, 9, 0, tzinfo=KST)},
        {"title": "유상증자 결정", "description": "대규모 유상증자", "published_at": dt.datetime(2026, 11, 20, 15, 0, tzinfo=KST)},
    ]


def _disclosures(code, start, end):
    return [{"submitted_at": "20261119", "report_name": "주요사항보고서(유상증자결정)", "url": "u"}]


def test_context_has_versions_and_skills():
    ctx = build_context()
    assert ctx.prompt_version == "system@1"
    assert "cr1" in ctx.skill_version and "dr1" in ctx.skill_version
    assert "급락 위험 검토 절차" in ctx.system_prompt


def test_tools_block_future_news_and_tag_disclosures():
    tb = Toolbox(_snapshot(), news_fn=_news, disclosure_fn=_disclosures)
    titles = [i["title"] for i in tb.search_news("테스트전자")["items"]]
    assert titles == ["유상증자 결정"]
    assert tb.search_disclosures("000001")["items"][0]["risk_tag"] == "medium"
    assert tag_disclosure("감사보고서 (의견거절)") == "high"
    assert tag_disclosure("자기주식취득결정") == "positive"


def test_agent_loop_with_tools():
    final = {"judgment": "medium", "agent_probability": 0.15, "reasons": ["유상증자 공시로 희석 우려"],
             "key_evidence": [{"type": "disclosure", "text": "유상증자"}], "watch": "청약 결과"}
    llm = FakeLLM([
        {"content": None, "tool_calls": [{"id": "1", "name": "get_crash_signal", "arguments": {"code": "000001"}},
                                         {"id": "2", "name": "search_disclosures", "arguments": {"code": "000001"}}]},
        {"content": None, "tool_calls": [{"id": "3", "name": "optimize_portfolio", "arguments": {}}]},  # 이 과제에선 금지된 툴
        {"content": json.dumps(final, ensure_ascii=False), "tool_calls": []},
    ])
    tb = Toolbox(_snapshot(), news_fn=_news, disclosure_fn=_disclosures)
    dec = HarnessAgent(llm, tb).review_stock("000001")
    assert dec["judgment"] == "medium" and dec["agent_probability"] == 0.15
    assert dec["model_probability"] == 0.08
    assert [e["tool"] for e in dec["evidence"]] == ["get_crash_signal", "search_disclosures"]
    tool_msgs = [m for m in llm.calls[-1] if m["role"] == "tool"]
    assert any("쓸 수 없는 툴" in m["content"] for m in tool_msgs)


def test_guards_fix_bad_output():
    d = validate_decision({"judgment": "low", "agent_probability": 0.9, "reasons": []}, model_prob=0.1)
    assert d["agent_probability"] == 0.4          # ±0.3 제한
    assert d["judgment"] == "high"                # 확률과 안 맞는 judgment 수정
    assert d["guard_notes"]
    d2 = validate_decision(None, model_prob=0.03)
    assert d2["agent_probability"] == 0.03 and d2["judgment"] == "low"


def test_scoring_and_cases():
    dec = pd.DataFrame({"id": [1, 2, 3], "code": ["A", "B", "C"], "run_date": [RUN] * 3,
                        "judgment": ["high", "low", "low"], "agent_probability": [0.4, 0.05, 0.02],
                        "model_probability": [0.3, 0.05, 0.02], "reasons": [["r"]] * 3})
    labels = pd.DataFrame({"code": ["A", "B", "C"], "base_date": [RUN] * 3,
                           "is_crash": [True, True, False], "min_return": [-0.12, -0.15, 0.01]})
    s = score_frame(dec, labels)
    assert s["correct"].tolist() == [True, False, True]
    stats = summarize(s)
    assert stats["missed"] == 1 and stats["false_alarm"] == 0
    cases = collect_cases(s)
    assert cases[0]["type"] == "missed" and cases[0]["code"] == "B"
