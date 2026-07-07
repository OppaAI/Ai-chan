"""
core/consolidate.py

Monthly memory consolidation.

Runs on/after the first day of a month and consolidates the month before the
most recent full month. Example: on July 1, keep June intact and summarize May.

Scope: this ONLY touches pinned daily-granularity memory (atomic facts tagged
"[YYYY-MM-DD] ..." and "Day record for YYYY-MM-DD:" blocks written nightly by
core.reflect). Unpinned memory is entirely out of scope here — its lifecycle
is owned by core.forget's decay scoring, applied nightly via memorize.dream().
Consolidation never reads, scores, or deletes unpinned rows.

Why this exists: pinned memory has no decay mechanism by design (permanent =
immune to forget.py). Without this step, daily atomic facts would accumulate
forever with no ceiling. This step gives pinned memory the equivalent of what
dream() already gives unpinned memory — compression instead of unbounded
growth — but on a monthly cadence instead of nightly, and via merge/compress
rather than delete-if-unused (since pinned facts were deliberately chosen as
worth keeping; the compression only reduces resolution, it doesn't judge
whether the content still matters).

Date handling: like human memory, most facts lose day-level resolution once
consolidated — a fact from mid-May becomes "sometime in May," not "May 18."
But facts describing a genuinely date-significant occasion (birthdays,
anniversaries, deadlines, one-off notable events) are instructed to keep the
specific date burned into the fact text itself, since the tag alone
(month-only after consolidation) is the only remaining source of truth for
*when* — if the date isn't in the text, it's gone permanently.

Catch-up: target_month_for() anchors its month math to the 1st of the
*current* calendar month regardless of what day `now` actually is (it forces
now.replace(day=1, ...) before doing any arithmetic), so the target month is
stable across the entire month, not just on the 1st. That means if Aiko is
offline through the 1st and comes back online on, say, the 5th, running
consolidation then computes the exact same target month as if it had run on
time. The only gate that matters is whether that month's key has already been
consolidated (state["last_consolidated_month"]), not whether today happens to
be the 1st — so there is no separate now.day == 1 requirement here.

Called by ScheduleRunner.monthly_consolidate — not user-modifiable via schedule.json.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from core.log import get_logger
from core.user_context import current_user_id, user_state_path
from core.reflect import _extract_json_arrays, _salvage_truncated_facts, _DAY_RECORD_PREFIX_TMPL

log = get_logger(__name__)

CONSOLIDATION_ENABLED         = os.getenv("MONTHLY_CONSOLIDATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
CONSOLIDATION_KEEP_MONTHS     = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_KEEP_MONTHS", "1")))
CONSOLIDATION_CHUNK_MEMS      = max(5, int(os.getenv("MONTHLY_CONSOLIDATION_CHUNK_MEMS", "25")))
CONSOLIDATION_MAX_INPUT_CHARS = max(1000, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS", "6000")))
CONSOLIDATION_MIN_MEMS        = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MEMS", "5")))
def consolidation_state_path(user_id: str | None = None) -> Path:
    """Resolve monthly consolidation state path for the active user."""
    override = os.getenv("MONTHLY_CONSOLIDATION_STATE_PATH")
    if override:
        return Path(override).expanduser()
    return user_state_path("monthly_consolidation_state.jsonl", user_id or current_user_id())



LLM_BASE_URL          = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL             = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
CONSOLIDATION_LLM_TIMEOUT = float(os.getenv("MONTHLY_CONSOLIDATION_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "120")))
CONSOLIDATION_DELETE_DAILY_SUMMARIES = os.getenv("MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES", "1").lower() in {"1", "true", "yes", "on"}

# Matches the per-day tag reflect.py pins facts with, e.g. "[2026-05-18] ...".
# Used to identify which pinned rows belong to daily-granularity memory (the
# only thing this module ever compresses/deletes) as opposed to any other
# pinned content (identity facts, standing preferences, etc.) that should
# never be touched by consolidation.
_DAILY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")


# ── month math ─────────────────────────────────────────────────────────────

def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year  = dt.year + month_index // 12
    month = month_index % 12 + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def target_month_for(now: datetime) -> tuple[datetime, datetime, str]:
    """Return (start, end, key) for the month ready to consolidate.
    `now` is expected to be a local-aware or naive-local datetime; this
    function does month arithmetic in that local frame and returns local
    (not UTC-mislabeled) boundaries. Convert to UTC only at the query call
    site, same pattern as core.schedule's daily reflect job.

    Note: `now` is forced to day=1 before any month math, so the target
    month is identical no matter what day of the month `now` actually falls
    on. This is what makes catch-up correct (see module docstring) — a late
    run on, say, the 5th, targets the exact same month as an on-time run on
    the 1st would have.
    """
    local_first  = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    target_end   = _add_months(local_first, -CONSOLIDATION_KEEP_MONTHS)
    target_start = _add_months(target_end, -1)
    key          = target_start.strftime("%Y-%m")
    return target_start, target_end, key


# ── state ────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(consolidation_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    path = consolidation_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _chat(system: str, user: str, max_tokens: int = 900, temperature: float = 0.1) -> str:
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed", timeout=CONSOLIDATION_LLM_TIMEOUT)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _bounded_lines(items: list[str]) -> str:
    lines: list[str] = []
    total = 0
    for line in items:
        if total + len(line) > CONSOLIDATION_MAX_INPUT_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) or "- none"


def _memory_lines(memories: list[dict]) -> str:
    return _bounded_lines([
        f"- {(m.get('memory') or m.get('text') or '').strip()}"
        for m in memories
        if (m.get("memory") or m.get("text") or "").strip()
    ])


# ── monthly fact extraction (mirrors reflect.py's daily fact extraction,
#    applied at month scope instead of day scope) ────────────────────────────

_MONTHLY_FACTS_SYSTEM = textwrap.dedent("""
    You are compressing a month's worth of daily memory facts about Oppa into
    a smaller set of durable long-term facts, for monthly archival. This is
    how long-term memory works: routine, repeated activity fades into a
    general sense of "this was going on that month," while genuinely
    significant, date-specific occasions stay sharp and dated.

    Rules:
    - Merge near-duplicate or repeated facts describing the same ongoing
      project, activity, or theme across multiple days into ONE combined
      fact (e.g. five separate days of "iterated on webui.py port
      consolidation" become one fact summarizing that overall effort).
    - Drop trivial, one-off chatter with no lasting significance.
    - Preserve every fact describing a distinct notable event, milestone,
      deadline, decision, incident, or occasion, even if mentioned only once.
    - CRITICAL: for any fact describing a genuinely date-specific occasion —
      a birthday, anniversary, one-off event, deadline hit or missed, a
      notable incident, a release/milestone date — keep the EXACT date
      written directly in the fact's own text (e.g. "On June 3rd, Oppa
      celebrated his birthday with fruit tarts."). The specific date will
      NOT be preserved anywhere else after this — if it is not in the text,
      it is permanently lost. When in doubt about whether something counts
      as date-significant, err on the side of keeping the date.
    - For routine or recurring facts with no specific date significance, do
      NOT include a specific date — summarize at month-level only (e.g.
      "Spent much of the month refining Aiko-chan's memory retrieval
      pipeline.").
    - Do not invent details, outcomes, dates, or facts not supported by the
      source material.
    - One fact per line, third person, about Oppa.
    - Each fact must be self-contained and short, readable without needing
      the surrounding month's context.

    Return ONLY a JSON array of short strings. No markdown, no explanation.
""").strip()

_MONTHLY_FACTS_USER = textwrap.dedent("""
    Month: {month_key}
    Chunk: {idx}/{total}

    Daily facts and records from this month:
    {facts}
""").strip()

_MONTHLY_MERGE_SYSTEM = textwrap.dedent("""
    You are merging several partial lists of monthly facts about Oppa into
    ONE final deduplicated list for permanent archival.

    Rules:
    - Combine facts that describe the same underlying event, project, or
      theme, even if worded differently across the partial lists — keep
      only one merged version.
    - Keep every fact that includes a specific date in its text UNCHANGED
      and UNMERGED with anything else — these are date-significant and must
      not be diluted or combined with unrelated material.
    - Drop exact or near-exact duplicates.
    - Do not invent anything not present in the source lists.
    - One fact per line, third person, about Oppa.

    Return ONLY a JSON array of short strings. No markdown, no explanation.
""").strip()

_MONTHLY_MERGE_USER = textwrap.dedent("""
    Month: {month_key}

    Partial fact lists to merge:
    {chunks}
""").strip()


def _parse_fact_array(raw: str) -> list[str]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    arrays = _extract_json_arrays(raw)
    for candidate in reversed(arrays):
        if candidate and all(isinstance(f, str) for f in candidate):
            return [f.strip() for f in candidate if isinstance(f, str) and f.strip()]

    salvaged = _salvage_truncated_facts(raw)
    if salvaged:
        log.warning("Monthly-facts array truncated — salvaged %d fact(s) from partial output.", len(salvaged))
        return salvaged

    log.warning("Failed to parse monthly-facts JSON: %r", raw[:600])
    return []


def _extract_monthly_facts_chunk(month_key: str, facts: list[str], idx: int, total: int) -> list[str]:
    user_prompt = _MONTHLY_FACTS_USER.format(
        month_key=month_key,
        idx=idx,
        total=total,
        facts=_bounded_lines([f"- {f}" for f in facts]),
    )
    raw = _chat(_MONTHLY_FACTS_SYSTEM, user_prompt, max_tokens=900, temperature=0.1)
    return _parse_fact_array(raw)


def _merge_monthly_facts(month_key: str, chunk_facts: list[list[str]]) -> list[str]:
    if len(chunk_facts) == 1:
        return chunk_facts[0]
    chunks_text = "\n\n".join(
        f"List {i+1}:\n" + "\n".join(f"- {f}" for f in facts)
        for i, facts in enumerate(chunk_facts)
    )
    user_prompt = _MONTHLY_MERGE_USER.format(month_key=month_key, chunks=chunks_text)
    raw = _chat(_MONTHLY_MERGE_SYSTEM, user_prompt, max_tokens=1200, temperature=0.1)
    merged = _parse_fact_array(raw)
    return merged or [f for facts in chunk_facts for f in facts]  # fallback: concatenate if merge parse fails


# ── main entrypoint ───────────────────────────────────────────────────────────

def maybe_run_consolidation(memorize, now: datetime | None = None) -> dict:
    """
    Run monthly consolidation if enabled and the target month is not already
    consolidated.

    Called by ScheduleRunner._run_monthly_consolidate() on/after the 1st of
    each month. The state file guards against double-runs on reboot AND is
    the sole gate for catch-up — there is no separate "must be exactly the
    1st" requirement, since target_month_for() anchors its arithmetic to the
    1st of the current calendar month regardless of what day `now` actually
    falls on. A late run (e.g. Aiko was offline through the 1st and comes
    back on the 5th) computes the same target month an on-time run would
    have, and the state check below correctly recognizes it as not yet done.

    Compresses pinned daily-granularity memory (atomic "[YYYY-MM-DD] fact"
    rows and "Day record for YYYY-MM-DD:" blocks) for the target month into
    a smaller set of pinned "[YYYY-MM] fact" rows, then deletes the
    daily-granularity originals for that month. Unpinned memory is never
    read, scored, or deleted here — that lifecycle belongs entirely to
    core.forget / memorize.dream(), independent of this job.

    Returns a result dict with keys: ran, reason (on skip), month, count,
    facts_written, daily_deleted.
    """
    if not CONSOLIDATION_ENABLED:
        return {"ran": False, "reason": "disabled"}

    now = now or datetime.now()

    start, end, month_key = target_month_for(now)
    state = _load_state()
    if state.get("last_consolidated_month") == month_key:
        return {"ran": False, "reason": "already_done", "month": month_key}

    start_utc = start.astimezone(timezone.utc)
    end_utc   = end.astimezone(timezone.utc)
    all_memories = memorize.get_between(start_utc, end_utc)

    # Scope to pinned daily-granularity rows only — atomic fact pins and
    # day-record blocks. Anything else pinned (identity facts, standing
    # preferences, etc.) is left completely untouched, and unpinned memory
    # is never considered here at all.
    daily_rows = [
        m for m in all_memories
        if int(m.get("pinned") or 0) == 1
        and (
            _DAILY_FACT_TAG_RE.match((m.get("memory") or "").strip())
            or (m.get("memory") or "").startswith("Day record for ")
        )
    ]

    if len(daily_rows) < CONSOLIDATION_MIN_MEMS:
        state["last_consolidated_month"] = month_key
        _save_state(state)
        return {"ran": False, "reason": "too_few_memories", "month": month_key, "count": len(daily_rows)}

    source_facts = [(m.get("memory") or "").strip() for m in daily_rows if (m.get("memory") or "").strip()]
    chunks = [source_facts[i:i + CONSOLIDATION_CHUNK_MEMS] for i in range(0, len(source_facts), CONSOLIDATION_CHUNK_MEMS)]

    chunk_facts = [
        _extract_monthly_facts_chunk(month_key, chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    chunk_facts = [c for c in chunk_facts if c]  # drop empty chunks (parse failures)

    if not chunk_facts:
        return {"ran": False, "reason": "empty_extraction", "month": month_key, "count": len(daily_rows)}

    final_facts = _merge_monthly_facts(month_key, chunk_facts)
    if not final_facts:
        return {"ran": False, "reason": "empty_merge", "month": month_key, "count": len(daily_rows)}

    facts_written = 0
    written_ids: list[str] = []
    for fact in final_facts:
        try:
            mem_id = memorize.add_raw(f"[{month_key}] {fact}", pinned=True)
            if mem_id:
                facts_written += 1
                written_ids.append(mem_id)
        except Exception as e:
            log.warning("Failed to pin monthly fact %r: %s", fact, e)

    if facts_written == 0:
        return {"ran": False, "reason": "no_facts_written", "month": month_key, "count": len(daily_rows)}

    # Only now delete the daily-granularity originals for this month —
    # their content has been folded into the facts just pinned above.
    # Gated by MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES so consolidation
    # can run purely additively (archive-only) if you want a safety margin
    # before trusting deletion.
    daily_deleted = 0
    if CONSOLIDATION_DELETE_DAILY_SUMMARIES:
        for m in daily_rows:
            mem_id = m.get("id")
            if not mem_id:
                continue
            try:
                memorize.delete(mem_id)
                daily_deleted += 1
            except Exception as e:
                log.warning("Failed to delete consolidated daily row %s: %s", mem_id, e)

    state["last_consolidated_month"] = month_key
    state["last_summary_ids"]        = written_ids
    _save_state(state)

    log.info(
        "monthly_consolidate complete: month=%s source_count=%s facts_written=%s daily_deleted=%s",
        month_key, len(daily_rows), facts_written, daily_deleted,
    )
    return {
        "ran":            True,
        "month":          month_key,
        "count":          len(daily_rows),
        "facts_written":  facts_written,
        "daily_deleted":  daily_deleted,
    }