"""
core/skills.py

Small skill-document helpers. This module owns skill CRUD/search for local
markdown skill files; it deliberately does not run tools or own the agent loop.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "persona" / "skills.md"


def load_skills(path: str | Path = DEFAULT_SKILLS_PATH) -> str:
    """Load the skill document text, returning an empty string when missing."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append_skill(text: str, path: str | Path = DEFAULT_SKILLS_PATH) -> None:
    """Append a new skill note to the skill document."""
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
    """Simple FTS-lite search over skill lines until an embedding index exists."""
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

    # Fallback: tolerate one typo/extra term by matching any term when AND yields nothing.
    if not results:
        for text in lines:
            folded = text.casefold()
            if any(term in folded for term in terms):
                results.append(text)
            if len(results) >= limit:
                break
    return results
