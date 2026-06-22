"""
core/skills.py

Skill document helpers and registry for local markdown skill files.

``persona/skills.md`` remains the short human-readable skill index loaded into
Aiko's base persona. Full repeatable workflows live under ``skills/<id>/`` as
``SKILL.md`` files and optional ``skill.yaml`` metadata. This module retrieves
skill instructions for the agent loop; it deliberately does not run tools or own
conversation state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "persona" / "skills.md"
SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class SkillDoc:
    """A discovered local skill workflow."""

    skill_id: str
    name: str
    path: Path
    summary: str
    triggers: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "path": str(self.path),
            "summary": self.summary,
            "triggers": list(self.triggers),
            "tools": list(self.tools),
        }


# ── legacy single-file helpers ────────────────────────────────────────────────

def load_skills(path: str | Path = DEFAULT_SKILLS_PATH) -> str:
    """Load the skill index/document text, returning an empty string when missing."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append_skill(text: str, path: str | Path = DEFAULT_SKILLS_PATH) -> None:
    """Append a new skill note to the skill index/document."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_skills(p).rstrip()
    addition = text.strip()
    if not addition:
        return
    p.write_text(f"{existing}\n\n{addition}\n" if existing else f"{addition}\n", encoding="utf-8")


def prune_duplicates(path: str | Path = DEFAULT_SKILLS_PATH) -> int:
    """Remove duplicate non-empty lines while preserving first occurrence order."""
    p = Path(path)
    if not p.exists():
        return 0
    lines = p.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    removed = 0
    for line in lines:
        key = line.strip()
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        output.append(line)
    p.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return removed


def search_skills(query: str, path: str | Path = DEFAULT_SKILLS_PATH, limit: int = 5) -> list[str]:
    """Simple FTS-lite search over the skill index lines."""
    terms = [t.casefold() for t in query.split() if t.strip()]
    if not terms:
        return []
    lines = [line.strip() for line in load_skills(path).splitlines() if line.strip()]
    results: list[str] = []
    for text in lines:
        folded = text.casefold()
        if all(term in folded for term in terms):
            results.append(text)
        if len(results) >= limit:
            return results

    if not results:
        for text in lines:
            folded = text.casefold()
            if any(term in folded for term in terms):
                results.append(text)
            if len(results) >= limit:
                break
    return results


# ── skill-directory registry ──────────────────────────────────────────────────

def _front_matter(markdown: str) -> tuple[dict[str, str], str]:
    """Parse a tiny YAML-like front matter block without adding dependencies."""
    if not markdown.startswith("---\n"):
        return {}, markdown
    _, _, rest = markdown.partition("---\n")
    meta_text, sep, body = rest.partition("\n---\n")
    if not sep:
        return {}, markdown
    meta: dict[str, str] = {}
    for line in meta_text.splitlines():
        key, found, value = line.partition(":")
        if found:
            meta[key.strip()] = value.strip().strip('"\'')
    return meta, body.lstrip()


def _heading_name(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line.lstrip("# ").strip() or fallback
    return fallback


def _first_paragraph(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if lines:
                break
            continue
        if stripped.startswith(("- ", "1.", "2.", "3.")) and not lines:
            continue
        lines.append(stripped)
    return " ".join(lines)[:500]


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def discover_skill_docs(root: str | Path = SKILL_ROOT) -> list[SkillDoc]:
    """Discover ``skills/<id>/SKILL.md`` workflow documents."""
    base = Path(root)
    if not base.exists():
        return []
    docs: list[SkillDoc] = []
    for skill_file in sorted(base.glob("*/SKILL.md")):
        skill_id = skill_file.parent.name
        raw = skill_file.read_text(encoding="utf-8")
        meta, body = _front_matter(raw)
        docs.append(SkillDoc(
            skill_id=meta.get("id", skill_id),
            name=meta.get("name") or _heading_name(body, skill_id.replace("_", " ").title()),
            path=skill_file,
            summary=meta.get("summary") or _first_paragraph(body),
            triggers=_split_csv(meta.get("triggers", "")),
            tools=_split_csv(meta.get("tools", "")),
        ))
    return docs


def list_skillsets() -> str:
    """Return available skill workflows as JSON text for agent tools."""
    return json.dumps({"skills": [doc.as_dict() for doc in discover_skill_docs()]}, ensure_ascii=False, indent=2)


def search_skillsets(query: str, limit: int = 3) -> list[SkillDoc]:
    """Search local skill workflows by id/name/summary/triggers/tools."""
    terms = [t.casefold() for t in query.split() if t.strip()]
    docs = discover_skill_docs()
    if not terms:
        return docs[:limit]

    scored: list[tuple[int, SkillDoc]] = []
    for doc in docs:
        haystack = " ".join([
            doc.skill_id,
            doc.name,
            doc.summary,
            " ".join(doc.triggers),
            " ".join(doc.tools),
        ]).casefold()
        score = sum(3 if term in doc.skill_id.casefold() else 1 for term in terms if term in haystack)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].skill_id))
    return [doc for _score, doc in scored[:limit]]


def search_skillsets_json(query: str, limit: int = 3) -> str:
    """Return matching skill workflows as JSON text for agent tools."""
    return json.dumps({"query": query, "matches": [doc.as_dict() for doc in search_skillsets(query, limit)]}, ensure_ascii=False, indent=2)


def load_skillset(skill_id: str, max_chars: int = 12_000) -> str:
    """Load one full skill workflow document by id."""
    cleaned = skill_id.strip().replace("/", "").replace("\\", "")
    for doc in discover_skill_docs():
        if doc.skill_id == cleaned or doc.path.parent.name == cleaned:
            text = doc.path.read_text(encoding="utf-8")[:max(1, min(max_chars, 50_000))]
            return f"<skill id=\"{doc.skill_id}\" name=\"{doc.name}\">\n{text}\n</skill>"
    return f"[skill not found: {skill_id}]"


def skill_context_for(query: str, limit: int = 2) -> str:
    """Build compact context for the most relevant skill workflows."""
    matches = search_skillsets(query, limit=limit)
    if not matches:
        return "<skill_context>\nNo matching predefined skills found. Use generic task tools.\n</skill_context>"
    loaded = [load_skillset(doc.skill_id, max_chars=6000) for doc in matches]
    return "<skill_context>\n" + "\n\n".join(loaded) + "\n</skill_context>"
