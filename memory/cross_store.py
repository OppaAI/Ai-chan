"""
memory/cross_store.py
Phase 13a — link personal memory hits to related knowledge + experience.

Uses existing search_knowledge / search_experience plus shared-entity overlap.
Does not mix foreign stores into personal RRF identity; callers attach results
as secondary context.
"""
from __future__ import annotations

import html
import os
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

CROSS_STORE_ENABLED = os.getenv("MEMORY_CROSS_STORE_ENABLED", "1").lower() in {
    "1", "true", "yes", "on",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MAX_KNOWLEDGE = max(0, _env_int("MEMORY_CROSS_STORE_MAX_KNOWLEDGE", 2))
MAX_EXPERIENCE = max(0, _env_int("MEMORY_CROSS_STORE_MAX_EXPERIENCE", 2))
MIN_ENTITY_OVERLAP = max(0, _env_int("MEMORY_CROSS_STORE_MIN_ENTITY_OVERLAP", 1))


def _entities_from_row(row: dict | Any) -> list[str]:
    try:
        from memory.memorize import entities_from_json

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
            from memory.memorize import extract_entities

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
        from memory.knowledge import search_knowledge

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
