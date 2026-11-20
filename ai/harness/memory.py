"""
에이전트 메모리 (agent_decision, agent_score 테이블)
"""
from __future__ import annotations

import pandas as pd

from ai.common.db import read_sql, upsert_rows


def save_decision(d: dict) -> int:
    row = {
        "code": d["code"],
        "run_date": d["run_date"],
        "model_probability": d.get("model_probability"),
        "agent_probability": d["agent_probability"],
        "judgment": d["judgment"],
        "reasons": d["reasons"],
        "evidence": {"key_evidence": d.get("key_evidence"), "watch": d.get("watch"),
                     "guard_notes": d.get("guard_notes"), "tool_log": d.get("evidence")},
        "prompt_version": d.get("prompt_version"),
        "skill_version": d.get("skill_version"),
        "llm_model": d.get("llm_model"),
    }
    return upsert_rows("agent_decision", [row], ["code", "run_date", "prompt_version"])


def recall(code: str, limit: int = 5, before=None) -> list[dict]:
    """before 이전 판단만 가져옴. 결과(is_crash 등)는 before 시점에 이미 나왔을 판단만 보여줌.
    과거 날짜로 다시 돌릴 때 미래 결과가 보이면 안 돼서 (5거래일 + 휴일 여유 = 7영업일)"""
    b = None if before is None else pd.Timestamp(before)
    known = None if b is None else str((b - pd.offsets.BDay(7)).date())
    df = read_sql(
        """
        SELECT d.run_date, d.judgment, d.agent_probability, d.reasons,
               CASE WHEN :known IS NULL OR d.run_date <= CAST(:known AS DATE) THEN s.is_crash END AS is_crash,
               CASE WHEN :known IS NULL OR d.run_date <= CAST(:known AS DATE) THEN s.min_return END AS min_return,
               CASE WHEN :known IS NULL OR d.run_date <= CAST(:known AS DATE) THEN s.correct END AS correct
        FROM agent_decision d LEFT JOIN agent_score s ON s.decision_id = d.id
        WHERE d.code = :code AND (:before IS NULL OR d.run_date < CAST(:before AS DATE))
        ORDER BY d.run_date DESC LIMIT :n
        """,
        {"code": code, "n": limit, "before": None if b is None else str(b.date()), "known": known},
    )
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")
