"""
Self-evolving: 채점 결과로 스킬 개선안 만들기

놓친 급락, 헛경보 사례를 모아서 LLM한테 현재 스킬이랑 같이 주고 고친 스킬을 받음.
바로 덮어쓰지 않고 skills/proposed/ 에 저장 → 사람이 보고 --apply.
사례 몇 개에 스킬이 끌려가는 걸 막으려고 이렇게 함.

    python -m ai.harness.evolve --skill crash_review
    python -m ai.harness.evolve --apply 3
    python -m ai.harness.evolve --reject 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil

import pandas as pd

from ai.harness.context import SKILL_DIR, load_skills, parse_frontmatter
from ai.harness.guards import parse_json
from ai.harness.scoring import summarize

PROPOSED_DIR = SKILL_DIR / "proposed"
HISTORY_DIR = SKILL_DIR / "history"

EVOLVE_PROMPT = """너는 금융 분석 에이전트의 판단 절차(스킬)를 개선하는 역할이다.
아래는 현재 스킬, 최근 채점 통계, 틀린 사례다. 사례에 과하게 맞추지 말고, 여러 사례에 공통으로
나타나는 문제만 반영해 스킬을 고쳐라. 형식(마크다운, 번호 절차)은 유지하고 frontmatter는 쓰지 마라.
출력은 JSON 하나: {"rationale": "무엇을 왜 바꿨는지 3문장 이내", "skill_body": "고친 스킬 본문 전체"}
"""


def collect_cases(scored: pd.DataFrame, n: int = 8) -> list[dict]:
    """놓친 급락을 우선, 그다음 헛경보. scored에는 reasons 컬럼이 있어야 한다."""
    missed = scored[(scored["judgment"] != "high") & scored["is_crash"]]
    false_alarm = scored[(scored["judgment"] == "high") & ~scored["is_crash"]]
    picks = pd.concat([missed.head(n // 2 + n % 2), false_alarm.head(n // 2)])
    return [{
        "type": "missed" if r.is_crash else "false_alarm",
        "code": r.code, "run_date": str(pd.Timestamp(r.run_date).date()),
        "judgment": r.judgment, "agent_probability": float(r.agent_probability),
        "min_return": float(r.min_return), "reasons": r.reasons,
    } for r in picks.itertuples()]


def propose(llm, skill_name: str, scored: pd.DataFrame) -> dict:
    skill = next(s for s in load_skills() if s.name == skill_name)
    stats = summarize(scored)
    cases = collect_cases(scored)
    user = json.dumps({"skill": skill.body, "stats": stats, "cases": cases}, ensure_ascii=False, default=str)
    resp = llm.chat([{"role": "system", "content": EVOLVE_PROMPT}, {"role": "user", "content": user}], json_mode=True)
    out = parse_json(resp.get("content")) or {}
    if not out.get("skill_body"):
        raise RuntimeError("LLM이 스킬 개선안을 내지 못함")
    new_version = str(int(skill.version) + 1)
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROPOSED_DIR / f"{skill_name}_v{new_version}_{dt.datetime.now():%Y%m%d%H%M}.md"
    path.write_text(
        f"---\nname: {skill.name}\nversion: {new_version}\ndescription: {skill.description}\n---\n{out['skill_body'].strip()}\n",
        encoding="utf-8",
    )
    return {"skill_name": skill_name, "base_version": skill.version, "new_version": new_version,
            "proposal_path": str(path), "rationale": out.get("rationale", ""), "metrics": stats}


def apply_proposal(proposal_path: str, skill_name: str) -> None:
    target = next(s.path for s in load_skills() if s.name == skill_name)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    meta, _ = parse_frontmatter(target.read_text(encoding="utf-8"))
    shutil.copy(target, HISTORY_DIR / f"{skill_name}_v{meta.get('version', '0')}.md")
    shutil.copy(proposal_path, target)


def main():
    from ai.common.data import load_labels
    from ai.common.db import execute, read_sql
    from ai.harness.llm import OpenAILLM
    from ai.harness.scoring import score_frame

    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="crash_review")
    ap.add_argument("--days", type=int, default=60, help="최근 며칠 판단을 볼지")
    ap.add_argument("--min-cases", type=int, default=30)
    ap.add_argument("--apply", type=int)
    ap.add_argument("--reject", type=int)
    args = ap.parse_args()

    if args.apply or args.reject:
        rid = args.apply or args.reject
        rev = read_sql("SELECT * FROM skill_revision WHERE id = :i", {"i": rid}).iloc[0]
        if args.apply:
            apply_proposal(rev["proposal_path"], rev["skill_name"])
        execute("UPDATE skill_revision SET status = :s WHERE id = :i", {"s": "applied" if args.apply else "rejected", "i": rid})
        print(f"skill_revision {rid}: {'적용' if args.apply else '거절'}")
        return

    since = (pd.Timestamp.today() - pd.Timedelta(days=args.days)).date()
    dec = read_sql(
        "SELECT id, code, run_date, judgment, agent_probability, model_probability, reasons "
        "FROM agent_decision WHERE run_date >= :s", {"s": since})
    scored = score_frame(dec, load_labels())
    if len(scored) < args.min_cases:
        print(f"채점된 판단 {len(scored)}건 < {args.min_cases}건. 더 쌓인 뒤에 실행하세요.")
        return
    prop = propose(OpenAILLM(), args.skill, scored)
    execute(
        "INSERT INTO skill_revision (skill_name, base_version, new_version, proposal_path, rationale, metrics) "
        "VALUES (:skill_name, :base_version, :new_version, :proposal_path, :rationale, CAST(:metrics AS JSONB))",
        {**prop, "metrics": json.dumps(prop["metrics"])})
    print(f"개선안 저장: {prop['proposal_path']}\n이유: {prop['rationale']}")


if __name__ == "__main__":
    main()
