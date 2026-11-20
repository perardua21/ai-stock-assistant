"""
LLM 호출 부분. 하네스는 chat()만 씀

chat(messages, tools) -> {"content": str | None, "tool_calls": [{"id", "name", "arguments"}]}
기본은 OpenAI (.env의 OPENAI_API_KEY, LLM_MODEL). 테스트는 FakeLLM.
"""
from __future__ import annotations

import json
import os
from typing import Protocol


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> dict: ...


class OpenAILLM:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def chat(self, messages, tools=None, json_mode=False):
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # 추론형 모델 중에는 temperature를 못 바꾸는 모델이 있어서 빼고 한 번 더
            if "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content, "tool_calls": calls}


class FakeLLM:
    """테스트용: 미리 정한 응답을 순서대로 돌려준다."""

    model = "fake"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append(messages)
        return self.responses.pop(0)


def assistant_message(resp: dict) -> dict:
    """LLM 응답을 다음 호출의 messages에 넣을 assistant 메시지로 변환 (OpenAI 형식)."""
    m = {"role": "assistant", "content": resp.get("content")}
    if resp.get("tool_calls"):
        m["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            for c in resp["tool_calls"]
        ]
    return m
