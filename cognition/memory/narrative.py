"""Context-augmentation helpers for memory recall.

Hosts three related concerns that all run after a memory search and
produce secondary context blocks for the prompt:

  1. Supersession narratives ("Previously held / Current") — Phase 16.
  2. Scene + persona + plain fact context formatters (moved from format.py).
  3. Cross-store related-context linking (Phase 13a, moved from
     cross_store.py) — finds related knowledge + experience entries by
     shared entity overlap with the personal memory hits.
"""

from __future__ import annotations

import html
from typing import Any

from system.config import env_bool, env_int
from system.log import get_logger

log = get_logger(__name__)

# ── Cross-store tunables (also mirrored in config/memory.yaml) ────────────────
CROSS_STORE_ENABLED = env_bool("MEMORY_CROSS_STORE_ENABLED", "1")
MAX_KNOWLEDGE = max(0, env_int("MEMORY_CROSS_STORE_MAX_KNOWLEDGE", 2))
MAX_EXPERIENCE = max(0, env_int("MEMORY_CROSS_STORE_MAX_EXPERIENCE", 2))
MIN_ENTITY_OVERLAP = max(0, env_int("MEMORY_CROSS_STORE_MIN_ENTITY_OVERLAP", 1))


def _text(row: dict[str, Any] | Any) -> str:
    if isinstance(row, dict):
        return (row.get("memory") or row.get("text") or "").strip()
    return (getattr(row, "memory", None) or getattr(row, "text", None) or "").strip()


def format_supersession_narrative(
    chain: list[dict[str, Any]],
    *,
    max_chars: int = 220,
) -> str | None:
    """
    chain: oldest → newest memory rows (length >= 2).
    Returns a short "Previously held / Current" line, or None.
    """
    if not chain or len(chain) < 2:
        return None
    old = _text(chain[0])[:120]
    new = _text(chain[-1])[:120]
    if not old or not new or old == new:
        return None
    line = f'Previously held: "{old}". Current: "{new}".'
    if len(line) > max_chars:
        line = line[: max_chars - 1] + "…"
    return line


def query_wants_emotion(query: str) -> bool:
    """True when neg-avoid should be relaxed (emotional / reflective query)."""
    q = (query or "").lower()
    keys = (
        "feel",
        "felt",
        "upset",
        "happy",
        "sad",
        "angry",
        "love",
        "hate",
        "used to",
        "remember when",
        "why did i",
        "emotion",
        "mood",
        "i used to think",
        "changed my mind",
    )
    return any(k in q for k in keys)


# ── context-formatting helpers (moved from format.py) ────────────────────

def format_for_context(
    facts: list[dict[str, str]],
    *,
    max_chars: int = 1200,
    max_facts: int | None = None,
) -> str:
    """
    Format a list of memory facts into a context string injected into the
    query context window.  Each fact dict should have ``"memory"`` or
    ``"text"`` keys.

    Returns the formatted string, trimmed to ``max_chars`` characters.
    """
    parts: list[str] = []
    for i, fact in enumerate(facts):
        text = (fact.get("memory") or fact.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        if max_facts is not None and len(parts) >= max_facts:
            break
    result = " ".join(parts)
    return result[:max_chars]


def scene_context(
    limit: int = 5, user_id: str | None = None
) -> str | None:
    """
    Format a scene-level context summary for the current user, if one exists.
    Returns ``None`` when no user is set or no scene memories are available.
    """
    from cognition.memory.memorize import AikoMemorize
    memorize = AikoMemorize()
    if user_id is None:
        user_id = memorize.get_user_id()
    if not user_id:
        return None
    try:
        ctx = memorize.scene_context(limit=limit, user_id=user_id)
        return ctx
    except Exception:
        return None


def persona_context(user_id: str | None = None) -> str | None:
    """
    Format a persona-level context summary for the current user, if one exists.
    Returns ``None`` when no user is set or no persona facts are available.
    """
    from cognition.memory.memorize import AikoMemorize
    memorize = AikoMemorize()
    if user_id is None:
        user_id = memorize.get_user_id()
    if not user_id:
        return None
    try:
        ctx = memorize.persona_context(user_id=user_id)
        return ctx
    except Exception:
        return None


# ── cross-store: link personal memory hits to related KB + experience ─────
# Phase 13a. Runs after format_for_context has built the personal
# <memory_context> block and appends a secondary block of related entries
# from the knowledge and experience stores, ranked by shared-entity overlap
# with the personal hits. The two stores are NOT mixed into the personal
# RRF identity — they appear as clearly-labeled secondary context.


def _entities_from_row(row: dict | Any) -> list[str]:
    try:
        from cognition.memory.memorize import entities_from_json

        raw = None
        if hasattr(row, "keys") and "entities" in row.keys():
            raw = row["entities"]
        elif isinstance(row, dict):
            raw = row.get("entities")
        return [e.casefold() for e in entities_from_json(raw or "[]") if e]
    except Exception:
        return []


def seed_entities_from_memories(memories: list[dict], query: str = "") -> list[str]:
    """Collect casefolded entity strings from personal memory hits (+ query)."""
    seeds: list[str] = []
    seen: set[str] = set()
    for m in memories or []:
        for e in _entities_from_row(m):
            if e not in seen:
                seen.add(e)
                seeds.append(e)
    if query:
        try:
            from cognition.memory.memorize import extract_entities

            for e in extract_entities(query):
                k = e.casefold()
                if k and k not in seen:
                    seen.add(k)
                    seeds.append(k)
        except Exception as exc:
            log.debug("cross_store: extract_entities(query) skipped: %s", exc)
    return seeds


def _overlap_count(seed: set[str], row_entities: list[str]) -> int:
    if not seed or not row_entities:
        return 0
    return sum(1 for e in row_entities if e in seed)


def related_knowledge(
    query: str,
    seed_entities: list[str],
    *,
    user_id: str | None = None,
    limit: int | None = None,
    embedder=None,
) -> list[dict]:
    """Return up to `limit` knowledge chunks related to query / seed entities."""
    if not CROSS_STORE_ENABLED or MAX_KNOWLEDGE <= 0:
        return []
    limit = MAX_KNOWLEDGE if limit is None else max(0, int(limit))
    if limit <= 0:
        return []
    seed = {e.casefold() for e in (seed_entities or []) if e}
    try:
        from cognition.knowledge import search_knowledge

        hits = search_knowledge(
            query or " ",
            limit=max(limit * 4, 8),
            embedder=embedder,
            user_id=user_id,
        )
    except Exception as exc:
        log.debug("cross_store: search_knowledge failed: %s", exc)
        return []

    ranked: list[tuple[int, float, dict]] = []
    for h in hits:
        ents = _entities_from_row(h)
        ov = _overlap_count(seed, ents)
        score = float(h.get("score") or h.get("recall_score") or 0.0)
        if MIN_ENTITY_OVERLAP > 0 and seed:
            if ov < MIN_ENTITY_OVERLAP:
                continue
        ranked.append((ov, score, h))

    ranked.sort(key=lambda t: (-t[0], -t[1]))
    out: list[dict] = []
    for ov, score, h in ranked[:limit]:
        out.append({
            "id": h.get("id"),
            "store": "knowledge",
            "text": (h.get("text") or "")[:500],
            "title": h.get("title") or "",
            "score": score,
            "entity_overlap": ov,
            "entities": _entities_from_row(h),
        })
    return out


def related_experience(
    query: str,
    seed_entities: list[str],
    *,
    user_id: str | None = None,
    limit: int | None = None,
    embedder=None,
) -> list[dict]:
    """Return up to `limit` past agent experiences related to query / seeds."""
    if not CROSS_STORE_ENABLED or MAX_EXPERIENCE <= 0:
        return []
    limit = MAX_EXPERIENCE if limit is None else max(0, int(limit))
    if limit <= 0:
        return []
    seed = {e.casefold() for e in (seed_entities or []) if e}
    try:
        from agentic.experience import search_experience

        hits = search_experience(query or " ", limit=max(limit * 4, 8), embedder=embedder, user_id=user_id)
    except Exception as exc:
        log.debug("cross_store: search_experience failed: %s", exc)
        return []

    ranked: list[tuple[int, float, dict]] = []
    for h in hits:
        ents = _entities_from_row(h)
        ov = _overlap_count(seed, ents)
        score = float(h.get("recall_score") or h.get("score") or 0.0)
        if MIN_ENTITY_OVERLAP > 0 and seed:
            if ov < MIN_ENTITY_OVERLAP:
                continue
        ranked.append((ov, score, h))

    ranked.sort(key=lambda t: (-t[0], -t[1]))
    out: list[dict] = []
    for ov, score, h in ranked[:limit]:
        out.append({
            "id": h.get("id"),
            "store": "experience",
            "text": (h.get("record_text") or h.get("goal") or h.get("answer_excerpt") or "")[:500],
            "goal": h.get("goal") or "",
            "outcome": h.get("outcome") or "",
            "score": score,
            "entity_overlap": ov,
            "entities": _entities_from_row(h),
        })
    return out


def fetch_related_for_memories(
    query: str,
    memories: list[dict],
    *,
    user_id: str | None = None,
    embedder=None,
) -> dict[str, list[dict]]:
    """Convenience: seeds from memories+query → related KB + experience."""
    if not CROSS_STORE_ENABLED:
        return {"knowledge": [], "experience": []}
    seeds = seed_entities_from_memories(memories, query=query)
    return {
        "knowledge": related_knowledge(
            query, seeds, user_id=user_id, embedder=embedder
        ),
        "experience": related_experience(
            query, seeds, user_id=user_id, embedder=embedder
        ),
    }


def format_related_blocks(
    related: dict[str, list[dict]],
    *,
    max_chars: int = 800,
) -> str:
    """XML-ish secondary context for injection after <memory_context>."""
    if not related:
        return ""
    parts: list[str] = []
    budget = max_chars

    kb = related.get("knowledge") or []
    if kb and budget > 40:
        lines = [
            "<related_knowledge>",
            "Learned notes related to the same topics (secondary; not personal facts).",
        ]
        for h in kb:
            if budget <= 40:
                break
            title = html.escape((h.get("title") or "")[:80], quote=True)
            overhead = len(f'  <chunk title="{title}"></chunk>')
            room = max(0, min(220, budget - overhead))
            body = html.escape((h.get("text") or "")[:room], quote=False)
            line = f'  <chunk title="{title}">{body}</chunk>'
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line)
        lines.append("</related_knowledge>")
        parts.append("\n".join(lines))

    exp = related.get("experience") or []
    if exp and budget > 40:
        lines = [
            "<related_experience>",
            "Past agent task runs on similar topics (secondary).",
        ]
        for h in exp:
            if budget <= 0:
                break
            # Escape and truncate outcome for attribute
            outcome_raw = h.get("outcome") or ""
            outcome_escaped = html.escape(outcome_raw, quote=True)
            # Reserve space for tag structure: '  <past_task outcome="">\n</past_task>'
            # Approx 36 chars + outcome length
            tag_overhead = 36 + len(outcome_escaped)
            if budget <= tag_overhead:
                break
            # Truncate and escape body within remaining budget
            body_raw = h.get("text") or h.get("goal") or ""
            max_body_len = min(220, budget - tag_overhead)
            body_truncated = body_raw[:max_body_len]
            body_escaped = html.escape(body_truncated, quote=False)
            # Build final line and measure actual length
            line = f'  <past_task outcome="{outcome_escaped}">{body_escaped}</past_task>'
            lines.append(line)
            budget -= len(line)
        lines.append("</related_experience>")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else ""
