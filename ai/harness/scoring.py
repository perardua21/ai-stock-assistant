"""
자동 채점. 판단일에서 5거래일 지나 label_crash가 생긴 판단을 agent_score에 저장

correct: judgment가 high면 급락 예측, 나머지는 급락 아님으로 보고 비교
brier: (agent_probability - 실제)^2. model_brier랑 비교하면 에이전트가 도움이 됐는지 보임

    python -m ai.harness.scoring
"""
from __future__ import annotations

import pandas as pd


def score_frame(decisions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """decisions: id, code, run_date, judgment, agent_probability, model_probability
    labels: code, base_date, is_crash, min_return → 채점 결과 (라벨이 있는 판단만)."""
    d = decisions.copy()
    d["run_date"] = pd.to_datetime(d["run_date"])
    lab = labels.rename(columns={"base_date": "run_date"}).copy()
    lab["run_date"] = pd.to_datetime(lab["run_date"])
    m = d.merge(lab[["code", "run_date", "is_crash", "min_return"]], on=["code", "run_date"], how="inner")
    m["is_crash"] = m["is_crash"].astype(bool)
    m["correct"] = (m["judgment"] == "high") == m["is_crash"]
    m["brier"] = (m["agent_probability"].astype(float) - m["is_crash"].astype(float)) ** 2
    m["model_brier"] = (m["model_probability"].astype(float) - m["is_crash"].astype(float)) ** 2
    return m


def summarize(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"n": 0}
    crash = scored["is_crash"]
    high = scored["judgment"] == "high"
    return {
        "n": int(len(scored)),
        "crash_rate": float(crash.mean()),
        "accuracy": float(scored["correct"].mean()),
        "recall_high": float((high & crash).sum() / max(crash.sum(), 1)),
        "precision_high": float((high & crash).sum() / max(high.sum(), 1)),
        "agent_brier": float(scored["brier"].mean()),
        "model_brier": float(scored["model_brier"].mean()),
        "missed": int((~high & crash).sum()),        # 급락했는데 high로 못 본 것
        "false_alarm": int((high & ~crash).sum()),   # high였는데 급락 안 한 것
    }


def run() -> dict:
    from ai.common.data import load_labels
    from ai.common.db import read_sql, upsert_rows

    pending = read_sql(
        """
        SELECT d.id, d.code, d.run_date, d.judgment, d.agent_probability, d.model_probability
        FROM agent_decision d LEFT JOIN agent_score s ON s.decision_id = d.id
        WHERE s.decision_id IS NULL
        """
    )
    if pending.empty:
        print("채점할 판단 없음")
        return {"n": 0}
    scored = score_frame(pending, load_labels())
    today = pd.Timestamp.today().date()
    rows = [{"decision_id": int(r.id), "scored_date": today, "min_return": float(r.min_return),
             "is_crash": bool(r.is_crash), "correct": bool(r.correct), "brier": float(r.brier)} for r in scored.itertuples()]
    upsert_rows("agent_score", rows, ["decision_id"])
    stats = summarize(scored)
    print(f"채점 {len(rows)}건 (대기 {len(pending) - len(rows)}건): {stats}")
    return stats


if __name__ == "__main__":
    run()
