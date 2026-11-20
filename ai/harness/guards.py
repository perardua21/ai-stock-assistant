"""
에이전트 출력 검사

- JSON 아니면 모델 확률로 대체
- 모델 확률에서 ±0.3 넘게 바꾸면 잘라냄
- judgment랑 확률이 크게 안 맞으면 확률 기준으로 고침
고친 내용은 guard_notes에 남김
"""
from __future__ import annotations

import json
import re

JUDGMENTS = ("high", "medium", "low")


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judgment_from_prob(p: float, base_rate: float) -> str:
    """기준 비율 대비 배수로 등급을 나눈다 (high: 5배 이상, medium: 2배 이상)."""
    if p >= max(5 * base_rate, 0.15):
        return "high"
    if p >= max(2 * base_rate, 0.06):
        return "medium"
    return "low"


def validate_decision(raw: dict | None, model_prob: float | None, base_rate: float = 0.02, max_adjust: float = 0.3) -> dict:
    notes = []
    if not isinstance(raw, dict):
        raw, notes = {}, ["출력이 JSON이 아니어서 모델 확률로 대체"]
    mp = float(model_prob) if model_prob is not None else None

    try:
        p = float(raw.get("agent_probability"))
    except (TypeError, ValueError):
        p = mp if mp is not None else base_rate
        notes.append("agent_probability 없음 → 모델 확률 사용")
    p = min(max(p, 0.0), 1.0)
    if mp is not None and abs(p - mp) > max_adjust:
        clipped = min(max(p, mp - max_adjust), mp + max_adjust)
        notes.append(f"확률 조정 폭 제한 {p:.3f} → {clipped:.3f}")
        p = clipped

    j = raw.get("judgment")
    expected = judgment_from_prob(p, base_rate)
    if j not in JUDGMENTS:
        notes.append(f"judgment 형식 오류({j}) → {expected}")
        j = expected
    elif abs(JUDGMENTS.index(j) - JUDGMENTS.index(expected)) >= 2:
        notes.append(f"judgment {j}가 확률 {p:.3f}와 맞지 않음 → {expected}")
        j = expected

    reasons = [str(r) for r in (raw.get("reasons") or []) if str(r).strip()]
    if not reasons:
        reasons = ["에이전트가 근거를 제시하지 않음 (모델 확률 기준 판단)"]
        notes.append("reasons 비어 있음")

    return {
        "judgment": j,
        "agent_probability": round(p, 5),
        "reasons": reasons[:5],
        "key_evidence": raw.get("key_evidence") or [],
        "watch": raw.get("watch") or "",
        "guard_notes": notes,
    }
