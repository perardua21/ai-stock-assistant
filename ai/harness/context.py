"""
prompts/system.md + skills/*.md 를 읽어서 system 메시지를 만듦

파일 맨 위 version을 판단마다 같이 저장해서 버전별 성능을 볼 수 있게 함
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

HARNESS_DIR = pathlib.Path(__file__).resolve().parent
PROMPT_PATH = HARNESS_DIR / "prompts" / "system.md"
SKILL_DIR = HARNESS_DIR / "skills"

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


@dataclass
class Skill:
    name: str
    version: str
    description: str
    body: str
    path: pathlib.Path


def load_skills(skill_dir: pathlib.Path = SKILL_DIR, names: list[str] | None = None) -> list[Skill]:
    skills = []
    for p in sorted(skill_dir.glob("*.md")):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        name = meta.get("name", p.stem)
        if names and name not in names:
            continue
        skills.append(Skill(name, meta.get("version", "0"), meta.get("description", ""), body.strip(), p))
    return skills


@dataclass
class HarnessContext:
    system_prompt: str
    prompt_version: str
    skill_version: str


def build_context(skill_names: list[str] | None = None, prompt_path: pathlib.Path = PROMPT_PATH,
                  skill_dir: pathlib.Path = SKILL_DIR) -> HarnessContext:
    meta, prompt = parse_frontmatter(prompt_path.read_text(encoding="utf-8"))
    skills = load_skills(skill_dir, skill_names)
    parts = [prompt.strip(), "\n# 스킬"]
    for s in skills:
        parts.append(f"\n## [{s.name} v{s.version}] {s.description}\n\n{s.body}")
    # 예: crash_review v2 + disclosure_risk v1 → "cr2.dr1" (DB 컬럼 길이 40자 이내)
    skill_version = ".".join("".join(w[0] for w in s.name.split("_")) + s.version for s in skills)
    return HarnessContext("\n".join(parts), f"system@{meta.get('version', '0')}", skill_version[:40])
