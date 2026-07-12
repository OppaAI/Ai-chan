"""
skills/skills.py

Skill document helpers and registry for local markdown skill files.

``skills/skills.md`` remains the short human-readable skill index loaded into
Aiko's base persona. Full repeatable workflows live under ``skills/skillsets/`` as
``<id>.md`` files and optional ``skill.yaml`` metadata.

skill_context_for injects RAG-style excerpts, not whole skill files:
search_skillsets ranks which skills are candidates, then skill_context_for
chunks each matched doc and keeps only the query-relevant slice. The agent
can still pull the complete workflow via the load_skillset tool when an
excerpt isn't enough.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from cognition import reason
from system.userspace import user_state_dir

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "skills" / "skills.md"
SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills"
_USER_SKILLSETS_PATH = os.getenv("USER_SKILLSETS_PATH") or str(user_state_dir() / "skillsets")

_STOPWORDS = reason.STOPWORDS

# Minimum score before a skill counts as a genuine match at the whole-doc
# level (keyword fallback path only).
_MIN_RELEVANCE_SCORE = 3

# ── whole-doc semantic ranking (which skills are candidates at all) ────────
_SKILL_INSTRUCT = "Which predefined skill workflow is most relevant to this task?"
_SKILL_SEMANTIC_THRESHOLD = float(os.getenv("SKILL_SEMANTIC_THRESHOLD", "0.35"))

# ── chunk-level RAG selection (which PART of a matched SKILL.md gets injected) ─
_SKILL_CHUNK_CHARS = int(os.getenv("SKILL_CHUNK_CHARS", "600"))
_SKILL_CHUNKS_PER_MATCH = int(os.getenv("SKILL_CHUNKS_PER_MATCH", "4"))
_SKILL_CHUNK_MIN_SCORE = float(os.getenv("SKILL_CHUNK_MIN_SCORE", "0.30"))


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


_skill_embed_cache: dict[tuple[str, float], np.ndarray] = {}
_skill_embed_cache_lock = threading.RLock()


def _embed_source_text(doc: "SkillDoc") -> str:
    """Condensed representation to embed — name, summary, triggers carry
    the topical signal; the full SKILL.md body is not embedded here."""
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
    vector = reason.normalize_vec(np.asarray(embedder.embed_query(_embed_source_text(doc)), dtype=np.float32))
    with _skill_embed_cache_lock:
        _skill_embed_cache[cache_key] = vector
    return vector


def _semantic_rank_skills(
    query: str, docs: list["SkillDoc"], embedder: Embedder, threshold: float,
) -> list["SkillDoc"] | None:
    """Rank skills by cosine similarity in one batched numpy matmul.
    Returns None if embedding fails (caller falls back to keyword scoring)."""
    if not docs:
        return []
    try:
        query_vec = np.asarray(embedder.embed_query(query, instruct=_SKILL_INSTRUCT), dtype=np.float32)
        doc_vecs = np.stack([_get_skill_embedding(doc, embedder) for doc in docs])
        scores = reason.batch_cosine_scores(query_vec, doc_vecs)
        order = np.argsort(-scores)
        return [docs[i] for i in order if scores[i] >= threshold]
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


def _discover_in(root: str | Path) -> list[SkillDoc]:
    base = Path(root)
    if not base.exists():
        return []
    docs: list[SkillDoc] = []
    for skill_file in sorted(base.glob("*.md")):
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


def discover_skill_docs() -> list[SkillDoc]:
    """Discover workflow documents from project and user skillsets."""
    docs: list[SkillDoc] = []
    for root in (SKILL_ROOT / "skillsets", Path(_USER_SKILLSETS_PATH)):
        docs.extend(_discover_in(root))
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

    terms = [t.casefold() for t in query.split() if t.strip() and t.casefold() not in _STOPWORDS]
    if not terms:
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
    """Load one full skill workflow document by id. Used for explicit
    on-demand full loads — not injected automatically into every turn."""
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
    """Build compact context from only the query-relevant excerpts of the
    most relevant skill workflows — not whole skill files. `max_chars`
    is a total budget shared across all matched skills, split per-match by
    remaining budget as it's consumed.
    """
    matches = search_skillsets(query, limit=limit, embedder=embedder)
    if not matches:
        return "<skill_context>\nNo matching predefined skills found. Use generic task tools.\n</skill_context>"

    blocks: list[str] = []
    remaining = max_chars
    for doc in matches:
        if remaining <= 0:
            break
        try:
            full_text = doc.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pieces = reason.chunk_text(full_text, _SKILL_CHUNK_CHARS)
        if not pieces:
            continue
        relevant = reason.select_relevant_chunks(
            query, pieces, embedder, top_k=_SKILL_CHUNKS_PER_MATCH,
            min_score=_SKILL_CHUNK_MIN_SCORE, instruct=_SKILL_INSTRUCT,
        )
        excerpt = "\n...\n".join(c for _score, c in relevant) if relevant else pieces[0]
        excerpt = excerpt[:remaining]
        if not excerpt:
            continue
        blocks.append(
            f'<skill id="{doc.skill_id}" name="{doc.name}">\n{excerpt}\n\n'
            f'[Excerpt only — call load_skillset("{doc.skill_id}") for the full workflow if this is insufficient.]\n'
            f'</skill>'
        )
        remaining -= len(excerpt)

    if not blocks:
        return "<skill_context>\nNo matching predefined skills found. Use generic task tools.\n</skill_context>"
    return "<skill_context>\n" + "\n\n".join(blocks) + "\n</skill_context>"
