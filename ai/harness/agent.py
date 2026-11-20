"""
하네스 LLM 에이전트

system(프롬프트+스킬), user(과제) → 툴 호출 반복 → 최종 JSON → guards 검사

    agent = HarnessAgent(llm, toolbox)
    decision = agent.review_stock("005930")
    summary = agent.review_portfolio(decisions)
"""
from __future__ import annotations

import json

from ai.harness.context import HarnessContext, build_context
from ai.harness.guards import parse_json, validate_decision
from ai.harness.llm import LLM, assistant_message
from ai.harness.tools import Toolbox, specs

STOCK_TOOLS = ["get_crash_signal", "get_price_summary", "search_news", "search_disclosures", "recall_memory"]
PORTFOLIO_TOOLS = ["optimize_portfolio"]


class HarnessAgent:
    def __init__(self, llm: LLM, toolbox: Toolbox, context: HarnessContext | None = None, max_steps: int = 8):
        self.llm = llm
        self.tools = toolbox
        self.ctx = context or build_context()
        self.max_steps = max_steps

    def _loop(self, user_msg: str, tool_names: list[str]) -> tuple[str | None, list[dict]]:
        start = len(self.tools.log)
        messages = [{"role": "system", "content": self.ctx.system_prompt}, {"role": "user", "content": user_msg}]
        tool_specs = specs(tool_names)
        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, tools=tool_specs)
            if not resp.get("tool_calls"):
                return resp.get("content"), self.tools.log[start:]
            messages.append(assistant_message(resp))
            for call in resp["tool_calls"]:
                if call["name"] not in tool_names:
                    out = json.dumps({"error": f"이 과제에서 쓸 수 없는 툴: {call['name']}"}, ensure_ascii=False)
                else:
                    out = self.tools.call(call["name"], call["arguments"])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
        # 툴 호출 한도를 넘으면 지금까지 본 정보로 JSON 답을 강제한다
        messages.append({"role": "user", "content": "툴 호출은 여기까지다. 지금까지 확인한 내용으로 최종 JSON만 출력해라."})
        return self.llm.chat(messages, json_mode=True).get("content"), self.tools.log[start:]

    def review_stock(self, code: str) -> dict:
        snap = self.tools.snapshot
        signal = self.tools.get_crash_signal(code)
        user = (
            f"기준일 {snap.run_date.date()} 종목 {code} {snap.names.get(code, '')}의 5거래일 급락 위험을 검토해라.\n"
            f"분류기 급락 확률 {signal['crash_probability']} (시장 평균 {signal['base_rate']}).\n"
            "스킬 절차대로 툴을 써서 확인한 뒤 최종 JSON만 출력해라."
        )
        text, log = self._loop(user, STOCK_TOOLS)
        decision = validate_decision(parse_json(text), signal["crash_probability"], snap.base_rate or 0.02)
        decision.update({
            "code": code,
            "run_date": snap.run_date.date(),
            "model_probability": signal["crash_probability"],
            "evidence": [{"tool": e["tool"], "args": e["args"], "result": e["result"]} for e in log],
            "prompt_version": self.ctx.prompt_version,
            "skill_version": self.ctx.skill_version,
            "llm_model": getattr(self.llm, "model", ""),
        })
        return decision

    def review_portfolio(self, decisions: list[dict]) -> dict:
        brief = [{"code": d["code"], "judgment": d["judgment"], "agent_probability": d["agent_probability"]} for d in decisions]
        user = (
            f"기준일 {self.tools.snapshot.run_date.date()} 포트폴리오를 추천해라. optimize_portfolio를 호출해 비중을 구하고, "
            "portfolio_review 스킬대로 검토해라. 아래는 오늘 에이전트가 검토한 종목의 판단이다.\n"
            f"{json.dumps(brief, ensure_ascii=False)}\n"
            '최종 출력은 JSON 하나: {"summary": "추천 사유 3~5문장", "warnings": ["주의할 점", ...]}'
        )
        text, log = self._loop(user, PORTFOLIO_TOOLS)
        out = parse_json(text) or {"summary": text or "", "warnings": []}
        weights = next((e["result"].get("items") for e in reversed(log) if e["tool"] == "optimize_portfolio"), None)
        out["weights"] = weights or []
        return out
