"""Monthly fact extraction and merge helpers for consolidation.

Owns the LLM prompt templates plus parsing/salvage/provenance helpers used by
the consolidation lifecycle.  Mirrors :mod:`cognition.knowledge` structure.
"""
from __future__ import annotations

import re
import textwrap

from system.log import get_logger
from system.userspace import current_display_name
from cognition.consolidate.reflect import _extract_json_arrays, _salvage_truncated_facts

from .schema import _bounded_lines, _chat, HARD_SOURCE_PROVENANCE

log = get_logger(__name__)

__all__ = [
    "extract_monthly_facts_chunk",
    "hard_provenance_ok",
    "merge_monthly_facts",
    "parse_fact_array",
    "parse_fact_items",
]

_MONTHLY_FACTS_SYSTEM = textwrap.dedent("""
    You are compressing a pre-selected list of daily memory facts about {USER_ID}
    into durable long-term monthly facts for archival.

    IMPORTANT: Every source line below was already chosen by a retention gate.
    You must NOT drop source material for being "trivial." You only merge and
    rephrase.

    Rules:
    - Merge near-duplicate or repeated facts describing the same ongoing
      project, activity, or theme into ONE combined fact.
    - Do not drop a source fact unless it is an exact or near-exact duplicate
      of another source fact in this list.
    - Preserve distinct events, milestones, deadlines, decisions, and occasions.
    - CRITICAL: for date-specific occasions (birthday, anniversary, deadline,
      one-off incident, release/milestone), keep the EXACT date in the fact text.
      If the date is not in the text, it is permanently lost after this step.
    - For routine/recurring themes with no specific date significance, summarize
      at month-level without inventing a day.
    - Do not invent details, outcomes, dates, or facts not in the sources.
    - Each fact must be self-contained and short, third person, about {USER_ID}.

    Return ONLY a JSON array. Prefer objects with provenance:
      [{"fact": "...", "source_ids": ["id1", "id2"]}, ...]
    source_ids must be ids from the input lines (id=...). Many sources may map
    to one fact. Plain string arrays are accepted only as a degraded fallback.
    No markdown, no explanation.
""").strip()

_MONTHLY_FACTS_USER = textwrap.dedent("""
    Month: {month_key}
    Chunk: {idx}/{total}

    Pre-selected daily facts (id=... | text). Do not drop except exact/near duplicates.
    Map every id you use into some output source_ids:
    {facts}
""").strip()

_MONTHLY_MERGE_SYSTEM = textwrap.dedent("""
    You are merging several partial lists of monthly facts about {USER_ID} into
    ONE final deduplicated list for permanent archival.

    Rules:
    - Combine facts that describe the same underlying event/project/theme.
    - Keep every fact that includes a specific date in its text UNCHANGED
      and UNMERGED with unrelated material.
    - Drop only exact or near-exact duplicates.
    - Do not invent anything not present in the source lists.
    - Do not drop distinct non-duplicate facts.
    - One fact per line, third person, about {USER_ID}.

    Return ONLY a JSON array of short strings. No markdown, no explanation.
""").strip()

_MONTHLY_MERGE_USER = textwrap.dedent("""
    Month: {month_key}

    Partial fact lists to merge:
    {chunks}
""").strip()


def parse_fact_array(raw: str) -> list[str]:
    """Legacy: plain string facts only."""
    items = parse_fact_items(raw)
    return [it["fact"] for it in items if it.get("fact")]


def parse_fact_items(raw: str) -> list[dict]:
    """Parse monthly facts as {fact, source_ids[]} or plain strings."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    arrays = _extract_json_arrays(raw)
    out: list[dict] = []
    for candidate in reversed(arrays):
        if not candidate:
            continue
        if all(isinstance(f, str) for f in candidate):
            return [{"fact": f.strip(), "source_ids": []} for f in candidate if isinstance(f, str) and f.strip()]
        if all(isinstance(f, dict) for f in candidate):
            for f in candidate:
                fact = (f.get("fact") or f.get("text") or f.get("memory") or "").strip()
                if not fact:
                    continue
                sids = f.get("source_ids") or f.get("sources") or f.get("ids") or []
                if isinstance(sids, str):
                    sids = [sids]
                sids = [str(s).strip() for s in sids if str(s).strip()]
                out.append({"fact": fact, "source_ids": sids})
            if out:
                return out

    if re.search(r"\[\s*\{", raw):
        log.warning("Monthly-facts object array incomplete/invalid; discarding.")
        return []

    salvaged = _salvage_truncated_facts(raw)
    if salvaged:
        log.warning("Monthly-facts array truncated — salvaged %d fact(s) from partial output.", len(salvaged))
        return [{"fact": f, "source_ids": []} for f in salvaged]

    log.warning("Failed to parse monthly-facts JSON: %r", raw[:600])
    return []


def extract_monthly_facts_chunk(
    month_key: str,
    rows: list[dict],
    idx: int,
    total: int,
) -> list[dict]:
    """rows: kept day-pin dicts with id + _text (or plain text strings for legacy)."""
    lines: list[str] = []
    for r in rows:
        if isinstance(r, str):
            lines.append(f"- {r}")
            continue
        mid = str(r.get("id") or "").strip()
        txt = (r.get("_text") or r.get("memory") or "").strip()
        if not txt:
            continue
        if mid:
            lines.append(f"- id={mid} | {txt}")
        else:
            lines.append(f"- {txt}")
    user_prompt = _MONTHLY_FACTS_USER.format(
        month_key=month_key,
        idx=idx,
        total=total,
        facts=_bounded_lines(lines),
    )
    raw = _chat(_MONTHLY_FACTS_SYSTEM.format(USER_ID=current_display_name()), user_prompt, max_tokens=1100, temperature=0.1)
    return parse_fact_items(raw)


def merge_monthly_facts(month_key: str, chunk_items: list[list[dict]]) -> list[dict]:
    if len(chunk_items) == 1:
        return chunk_items[0]
    flat = [it for chunk in chunk_items for it in chunk]
    if HARD_SOURCE_PROVENANCE:
        return flat
    chunks_text = "\n\n".join(
        f"List {i+1}:\n" + "\n".join(f"- {it.get('fact','')}" for it in facts)
        for i, facts in enumerate(chunk_items)
    )
    user_prompt = _MONTHLY_MERGE_USER.format(month_key=month_key, chunks=chunks_text)
    raw = _chat(_MONTHLY_MERGE_SYSTEM.format(USER_ID=current_display_name()), user_prompt, max_tokens=1200, temperature=0.1)
    merged = parse_fact_items(raw)
    if not merged:
        return flat

    # Best-effort: reattach source_ids lost by LLM merge (match on fact text).
    by_text: dict[str, list[str]] = {}
    for it in flat:
        key = (it.get("fact") or "").strip().casefold()
        if not key:
            continue
        by_text.setdefault(key, [])
        for sid in it.get("source_ids") or []:
            s = str(sid).strip()
            if s and s not in by_text[key]:
                by_text[key].append(s)

    for it in merged:
        if it.get("source_ids"):
            continue
        key = (it.get("fact") or "").strip().casefold()
        if key in by_text:
            it["source_ids"] = list(by_text[key])
    return merged


def hard_provenance_ok(kept_rows: list[dict], fact_items: list[dict]) -> tuple[bool, set[str]]:
    """Every kept day-pin id must appear in some output source_ids."""
    kept_ids = {str(r.get("id")).strip() for r in kept_rows if r.get("id")}
    covered: set[str] = set()
    for it in fact_items:
        for sid in it.get("source_ids") or []:
            covered.add(str(sid).strip())
    if not kept_ids:
        return True, covered
    missing = kept_ids - covered
    if missing:
        log.warning(
            "Phase 11 hard provenance: %d/%d kept ids missing from source_ids (sample=%s)",
            len(missing), len(kept_ids), list(missing)[:5],
        )
        return False, covered
    return True, covered