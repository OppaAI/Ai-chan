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
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "persona" / "skills.md"
SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills"

# Same rationale as knowledge.py's _STOPWORDS: without filtering, common
# words in the query can incidentally match inside an unrelated skill's
# summary/triggers/tools text and pull the whole skill doc into context.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "who", "when", "where", "why",
    "we", "you", "i", "he", "she", "they", "this", "that", "these",
    "those", "some", "any", "all", "each", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "not", "no", "yes", "make",
    "made", "get", "got", "go", "going", "let", "lets", "want", "wants",
    "just", "so", "up", "down", "out", "about", "if", "then", "than",
})

# Minimum score before a skill counts as a genuine match — mirrors
# knowledge.py's _MIN_RELEVANCE_SCORE so both retrieval paths behave the
# same way with the same "one stray substring hit isn't enough" logic.
_MIN_RELEVANCE_SCORE = 3

# ── semantic retrieval (Harrier embeddings) ─────────────────────────────────
# Same pattern as knowledge.py: no owner/session object here, so callers pass
# in an embedder explicitly. Falls back to keyword scoring above when no
# embedder is given or embedding fails for any reason.

_SKILL_INSTRUCT = "Which predefined skill workflow is most relevant to this task?"
_SKILL_SEMANTIC_THRESHOLD = float(os.getenv("SKILL_SEMANTIC_THRESHOLD", "0.35"))


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


_skill_embed_cache: dict[tuple[str, float], np.ndarray] = {}
_skill_embed_cache_lock = threading.RLock()


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def _embed_source_text(doc: "SkillDoc") -> str:
    """Condensed representation to embed — name, summary, and triggers
    carry the actual topical signal; the full SKILL.md workflow body does
    not need to be embedded."""
    return f"{doc.name}\n{doc.summary}\n{' '.join(doc.triggers)}\n{' '.join(doc.tools)}"


def _get_skill_embedding(doc: "SkillDoc", embedder: Embedder) -> np.ndarray:
    try:
        mtime = doc.path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cache_key = (doc.skill_id, mtime)
    with _skill_embed_cache_lock:
        cached = _skill_embed_cache.get(cache_key)
        if cached is not None:
            return cached
    vector = _normalize(np.asarray(embedder.embed_query(_embed_source_text(doc)), dtype=np.float32))
    with _skill_embed_cache_lock:
        _skill_embed_cache[cache_key] = vector
    return vector


def _semantic_rank_skills(
    query: str, docs: list["SkillDoc"], embedder: Embedder, threshold: float,
) -> list["SkillDoc"] | None:
    """Return skills ranked by cosine similarity, or None if embedding
    fails (caller falls back to keyword scoring)."""
    if not docs:
        return []
    try:
        query_vector = _normalize(np.asarray(embedder.embed_query(query, instruct=_SKILL_INSTRUCT), dtype=np.float32))
        scored: list[tuple[float, "SkillDoc"]] = []
        for doc in docs:
            score = float(np.dot(query_vector, _get_skill_embedding(doc, embedder)))
            if score >= threshold:
                scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        return [doc for _score, doc in scored]
    except Exception:
        return None


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
        try:
            display_path = str(self.path.resolve().relative_to(SKILL_ROOT.resolve()))
        except ValueError:
            display_path = self.path.name
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "path": display_path,
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
        try:
            raw = skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
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


def search_skillsets(query: str, limit: int = 3, embedder: Embedder | None = None) -> list[SkillDoc]:
    """Search local skill workflows by id/name/summary/triggers/tools."""
    docs = discover_skill_docs()
    if not query.strip():
        return docs[:limit]

    if embedder is not None:
        ranked = _semantic_rank_skills(query, docs, embedder, _SKILL_SEMANTIC_THRESHOLD)
        if ranked is not None:
            return ranked[:limit]
        # embedding failed — fall through to keyword scoring

    terms = [t.casefold() for t in query.split() if t.strip() and t.casefold() not in _STOPWORDS]
    if not terms:
        # Query was entirely stopwords/junk — not a real topical query, so
        # match nothing rather than returning arbitrary skills.
        return []

    scored: list[tuple[int, SkillDoc]] = []
    for doc in docs:
        skill_id_cf = doc.skill_id.casefold()
        name_cf = doc.name.casefold()
        triggers_cf = " ".join(doc.triggers).casefold()
        summary_cf = doc.summary.casefold()
        tools_cf = " ".join(doc.tools).casefold()
        score = 0
        for term in terms:
            if term in skill_id_cf:
                score += 5
            if term in name_cf:
                score += 4
            if term in triggers_cf:
                # triggers exist specifically to catch how a user would
                # phrase this — a single trigger hit should be enough to
                # clear the relevance floor on its own.
                score += 4
            if term in summary_cf:
                score += 2
            if term in tools_cf:
                score += 1
        if score >= _MIN_RELEVANCE_SCORE:
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
            try:
                text = doc.path.read_text(encoding="utf-8", errors="replace")[:max(1, min(max_chars, 50_000))]
            except OSError as e:
                return f"[skill load failed: {e}]"
            return f"<skill id=\"{doc.skill_id}\" name=\"{doc.name}\">\n{text}\n</skill>"
    return f"[skill not found: {skill_id}]"


def skill_context_for(
    query: str, limit: int = 2, max_chars: int = 6000, embedder: Embedder | None = None,
) -> str:
    """Build compact context for the most relevant skill workflows.

    `max_chars` is a *total* budget shared across all loaded skills (split
    evenly per match), not a per-skill cap — previously each matched skill
    got its own fixed 6000-char allowance regardless of `limit`, so raising
    `limit` silently multiplied the total injected size.
    """
    matches = search_skillsets(query, limit=limit, embedder=embedder)
    if not matches:
        return "<skill_context>\nNo matching predefined skills found. Use generic task tools.\n</skill_context>"
    per_skill_chars = max(500, max_chars // max(1, len(matches)))
    loaded = [load_skillset(doc.skill_id, max_chars=per_skill_chars) for doc in matches]
    return "<skill_context>\n" + "\n\n".join(loaded) + "\n</skill_context>"