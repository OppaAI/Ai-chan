"""
memory/consolidate.py

Monthly memory consolidation.

Runs on/after the first day of a month and consolidates the month before the
most recent full month. Example: on July 1, keep June intact and summarize May.

Scope: this ONLY touches pinned daily-granularity memory (atomic facts tagged
"[YYYY-MM-DD] ..." rows in memory.db plus "Daily journal of YYYY-MM-DD:" blobs in journal.db written nightly by
memory.reflect). Unpinned memory is entirely out of scope here — its lifecycle
is owned by memory.forget's decay scoring, applied nightly via memorize.dream().
Consolidation never reads, scores, or deletes unpinned rows.

Why this exists: pinned memory has no decay mechanism by design (permanent =
immune to forget.py). Without this step, daily atomic facts would accumulate
forever with no ceiling. This step gives pinned memory the equivalent of what
dream() already gives unpinned memory — compression instead of unbounded
growth — but on a monthly cadence instead of nightly, and via merge/compress
rather than delete-if-unused (since pinned facts were deliberately chosen as
worth keeping; the compression only reduces resolution, it doesn't judge
whether the content still matters).

Retention gate (NEW):
  Previously every fact the LLM merge step returned was pinned unconditionally
  — a quiet month and a chaotic month got identical treatment, with no
  ranking, no floor/ceiling, no novelty check. That's the "coin toss" failure
  mode: boring months either wipe entirely or keep weak filler, busy months
  can explode into unbounded monthly archive growth.

  Before facts are ever sent to the LLM for extraction/merge, daily rows for
  the target month are split into:
    - must_keep: rows whose kind is 'event'/'plan' (memory.memorize.classify_kind)
      or whose text contains a date-significant keyword (deadline, birthday,
      anniversary, appointment, hackathon, interview, or an explicit loss
      event) — these always survive into the LLM step regardless of score.
    - candidates: everything else, scored 0..1 by a retention formula:
        0.30 * salience     (keyword hit, else a lower baseline)
        + 0.25 * novelty    (1 - cosine sim to nearest static theme anchor)
        + 0.20 * spacing    (access_count proxy for repeated recall)
        + 0.25 * connectivity (entity co-mention weight, from entity_relations,
                                 already written at insert time — this is the
                                 first time it's read as a numeric feature
                                 rather than only for recall-time graph fusion)

  Candidates are ranked by score; the number kept is
  max(MIN_MONTH, min(MAX_MONTH, count(score >= SOFT_THRESHOLD))) — a floor so
  a quiet month still leaves a thin trace, a ceiling so a busy month can't
  make the monthly archive grow unbounded. Only must_keep + kept candidates
  go into the existing chunk/extract/merge pipeline; the LLM still decides
  wording and how to combine survivors, never which facts live or die.

  Static anchors are theme centroids (k-means, or a single mean vector if
  scikit-learn isn't available) built from the most recent already-archived
  "[YYYY-MM] ..." monthly facts — "what themes defined recent months" — so
  novelty is measured against real prior archive, not against the current
  month's own candidates (which would be circular).

  ALL daily-granularity originals for the month (must_keep + kept + dropped
  candidates) are still deleted at the end, same as before — dropped
  candidates are folded out of the monthly archive rather than merged into
  it, must_keep + kept candidates get folded into monthly facts via the
  existing LLM merge step.

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

import numpy as np
from openai import OpenAI

from system import bioclock
from system.log import get_logger
from system.userspace import current_display_name, current_user_id, user_state_path
from memory.reflect import _extract_json_arrays, _salvage_truncated_facts
from memory.memorize import classify_kind, entities_from_json

log = get_logger(__name__)

CONSOLIDATION_ENABLED         = os.getenv("MONTHLY_CONSOLIDATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
CONSOLIDATION_KEEP_MONTHS     = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_KEEP_MONTHS", "1")))
CONSOLIDATION_CHUNK_MEMS      = max(5, int(os.getenv("MONTHLY_CONSOLIDATION_CHUNK_MEMS", "25")))
CONSOLIDATION_MAX_INPUT_CHARS = max(1000, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS", "6000")))
CONSOLIDATION_MIN_MEMS        = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MEMS", "5")))

# ── retention gate knobs (NEW) ────────────────────────────────────────────────
# Floor/ceiling on how many *candidate* (non-must-keep) facts survive into
# the monthly archive. must_keep facts are never subject to this cap.
CONSOLIDATION_MIN_MONTH        = max(0, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MONTH", "8")))
CONSOLIDATION_MAX_MONTH        = max(CONSOLIDATION_MIN_MONTH, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_MONTH", "30")))
CONSOLIDATION_SOFT_THRESHOLD   = float(os.getenv("MONTHLY_CONSOLIDATION_SOFT_THRESHOLD", "0.4"))
# How many prior "[YYYY-MM] ..." archive facts to pull for static anchor
# clustering, and how many clusters (themes) to fit against them.
CONSOLIDATION_ANCHOR_LOOKBACK  = max(2, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_LOOKBACK", "50")))
CONSOLIDATION_ANCHOR_K         = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_K", "5")))
# Retention score component weights — must sum to 1.0 to keep the score in
# a stable 0..1 range against CONSOLIDATION_SOFT_THRESHOLD.
_RETENTION_W_SALIENCE     = float(os.getenv("MONTHLY_CONSOLIDATION_W_SALIENCE", "0.30"))
_RETENTION_W_NOVELTY      = float(os.getenv("MONTHLY_CONSOLIDATION_W_NOVELTY", "0.25"))
_RETENTION_W_SPACING      = float(os.getenv("MONTHLY_CONSOLIDATION_W_SPACING", "0.20"))
_RETENTION_W_CONNECTIVITY = float(os.getenv("MONTHLY_CONSOLIDATION_W_CONNECTIVITY", "0.25"))
# access_count value treated as "fully spaced" (score saturates at 1.0 here).
_RETENTION_SPACING_SATURATION = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_SPACING_SATURATION", "5")))

def consolidation_state_path(user_id: str | None = None) -> Path:
    """Resolve monthly consolidation state path for the active user."""
    override = os.getenv("MONTHLY_CONSOLIDATION_STATE_PATH")
    if override:
        return Path(override).expanduser()
    return user_state_path("memory/monthly_consolidation_state.json", user_id or current_user_id())



LLM_BASE_URL          = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL             = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
CONSOLIDATION_LLM_TIMEOUT = float(os.getenv("MONTHLY_CONSOLIDATION_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "120")))
CONSOLIDATION_DELETE_DAILY_SUMMARIES = os.getenv("MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES", "1").lower() in {"1", "true", "yes", "on"}

# Matches the per-day tag memory/reflect.py pins facts with, e.g. "[2026-05-18] ...".
# Used to identify which pinned rows belong to daily-granularity memory (the
# only thing this module ever compresses/deletes) as opposed to any other
# pinned content (identity facts, standing preferences, etc.) that should
# never be touched by consolidation.
_DAILY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")

# Matches an already-archived monthly fact, e.g. "[2026-05] ...". Deliberately
# does NOT match the daily tag above (the day-of-month digits after the
# second "-" mean the char immediately after \d{4}-\d{2} is "-", not "]").
# Used only to source static anchor embeddings from prior archive.
_MONTHLY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}\]\s")

# Facts matching these keywords (word-agnostic substring match, casefolded)
# are treated as must-keep regardless of retention score — same rationale as
# memory/memorize.py's _SALIENCE_KEYWORDS, narrowed to genuinely
# date/occasion-significant terms rather than the broader recall-salience list.
_MUST_KEEP_KEYWORDS = (
    "deadline", "birthday", "anniversary", "appointment", "hackathon",
    "interview", "lost ", "passport", "license", "wallet",
)


def _is_must_keep(text: str) -> bool:
    """True if a daily fact should bypass retention scoring entirely.

    Uses memory.memorize.classify_kind (the same rule-based classifier
    already applied at write time) plus a narrower date/occasion keyword
    list than recall-time salience, since must-keep is a stronger bar than
    "worth boosting at dream time."
    """
    kind = classify_kind(text, default="fact")
    if kind in ("event", "plan"):
        return True
    low = (text or "").casefold()
    return any(k in low for k in _MUST_KEEP_KEYWORDS)


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
    site, same pattern as system.schedule's daily reflect job.

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

def _load_state(user_id: str | None = None) -> dict:
    try:
        return json.loads(consolidation_state_path(user_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict, user_id: str | None = None) -> None:
    path = consolidation_state_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


# ── LLM helpers ───────────────────────────────────────────────────────────────

_LLM_CLIENT: OpenAI | None = None

def _get_llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed", timeout=CONSOLIDATION_LLM_TIMEOUT)
    return _LLM_CLIENT

def _chat(system: str, user: str, max_tokens: int = 900, temperature: float = 0.1) -> str:
    client = _get_llm_client()
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


# ── retention scoring (NEW) ───────────────────────────────────────────────────
# Applied to `candidate` rows only (must_keep rows bypass scoring entirely —
# see _is_must_keep). Scores every non-must-keep daily row 0..1 so a floor/
# ceiling gate can decide how many survive into the LLM extraction/merge step,
# instead of the LLM's own judgment being the only thing standing between a
# quiet month (whole month wiped) and a busy month (unbounded archive growth).

def _entity_connectivity_weights(memorize, user_id: str) -> dict[str, float]:
    """Sum of co-mention edge weight per entity (casefolded), read from the
    entity_relations table. This table is already populated at write time
    by memory.memorize._insert_row -> upsert_co_mentions for every fact with
    2+ entities — this is the first place it's read back as a numeric
    connectivity feature rather than only for recall-time graph fusion
    (_MemoryBackend._graph_pass). Read-only; never writes here.

    Returns {} on any failure (e.g. table not yet migrated for this user) —
    callers treat that as "no connectivity signal available" rather than
    raising, since connectivity is one of four score components, not a
    hard requirement.
    """
    try:
        conn = memorize._conn
        rows = conn.execute(
            "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    except Exception as exc:
        log.debug("Connectivity weights: entity_relations read skipped: %s", exc)
        return {}

    weights: dict[str, float] = {}
    for row in rows:
        a = str(row["entity_a"] or "")
        b = str(row["entity_b"] or "")
        w = float(row["weight"] or 0.0)
        if a:
            weights[a] = weights.get(a, 0.0) + w
        if b:
            weights[b] = weights.get(b, 0.0) + w
    return weights


def _build_static_anchors(memorize, user_id: str) -> "np.ndarray | None":
    """Theme centroids from the most recently archived "[YYYY-MM] ..."
    monthly facts — "what themes defined recent months" — used as the
    novelty reference for scoring this month's candidates. Deliberately
    sourced from PRIOR archive, not this month's own candidates, to avoid
    novelty being circular (everything looks novel relative to itself).

    Returns None when there isn't enough prior archive yet (first few
    months of the system's life) or on any failure; callers treat None as
    "no novelty signal available" and fall back to a neutral 0.5, same
    as the original Grace design's "empty month" case.
    """
    try:
        all_mems = memorize.get_all(user_id=user_id)
    except Exception as exc:
        log.warning("Static anchor build: failed to fetch memories: %s", exc)
        return None

    monthly_texts = [
        (m.get("memory") or "").strip()
        for m in sorted(all_mems, key=lambda m: m.get("created_at", "") or "", reverse=True)
        if _MONTHLY_FACT_TAG_RE.match((m.get("memory") or "").strip())
    ][:CONSOLIDATION_ANCHOR_LOOKBACK]

    if len(monthly_texts) < 2:
        return None

    try:
        vectors = np.array(memorize.embed_texts(monthly_texts, query=False))
    except Exception as exc:
        log.warning("Static anchor build: embedding failed: %s", exc)
        return None

    if vectors.size == 0:
        return None

    k = max(1, min(CONSOLIDATION_ANCHOR_K, len(monthly_texts) // 3 or 1))
    if k >= 2:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(vectors)
            return km.cluster_centers_
        except Exception as exc:
            log.debug(
                "Static anchor build: kmeans unavailable/failed (%s); "
                "falling back to a single mean-vector anchor.", exc,
            )

    # Fallback (no sklearn, or k collapsed to 1): a single mean-vector
    # anchor still gives a meaningful "distance from the general theme of
    # recent months" novelty signal without requiring sklearn on Jetson.
    return np.mean(vectors, axis=0, keepdims=True)


def _score_daily_row(
    row: dict,
    *,
    static_anchors: "np.ndarray | None",
    row_vector: "np.ndarray | None",
    entity_weights: dict[str, float],
    entity_weight_cap: float,
) -> float:
    """Retention score in 0..1 for one candidate daily row (must_keep rows
    never reach this function — see _is_must_keep).

    Components:
      salience     — cheap keyword hit (memory/memorize._SALIENCE_RE would
                     also work here, but the day-fact already carries its
                     own kind/text; a plain re-check keeps this function
                     self-contained). 1.0 on hit, 0.3 baseline otherwise.
      novelty      — 1 - cosine similarity to the nearest static theme
                     anchor. Neutral (0.5) when no anchors are available yet
                     (early system life) or embedding failed for this row.
      spacing      — access_count proxy for repeated recall, saturating at
                     _RETENTION_SPACING_SATURATION accesses. access_count is
                     already decay-aware upstream (memory/forget.py), so
                     this reuses a signal that already exists rather than
                     introducing a new access-day-set column.
      connectivity — mean normalized entity_relations co-mention weight
                     across this row's own entities (already extracted and
                     stored in the `entities` column at write time).
    """
    text = row.get("_text", "") or ""
    entities = entities_from_json(row.get("entities"))

    salience = 1.0 if _SALIENCE_HIT_RE.search(text) else 0.3

    access_count = int(row.get("access_count") or 0)
    spacing = min(1.0, access_count / float(_RETENTION_SPACING_SATURATION))

    if entities and entity_weights:
        raw = [entity_weights.get(e.casefold(), 0.0) for e in entities]
        connectivity = min(1.0, (sum(raw) / len(raw)) / max(entity_weight_cap, 1e-6))
    else:
        connectivity = 0.0

    if static_anchors is not None and len(static_anchors) and row_vector is not None:
        try:
            norms = np.linalg.norm(static_anchors, axis=1) * np.linalg.norm(row_vector) + 1e-9
            sims = (static_anchors @ row_vector) / norms
            novelty = float(max(0.0, min(1.0, 1.0 - float(np.max(sims)))))
        except Exception:
            novelty = 0.5
    else:
        novelty = 0.5

    return (
        _RETENTION_W_SALIENCE * salience
        + _RETENTION_W_NOVELTY * novelty
        + _RETENTION_W_SPACING * spacing
        + _RETENTION_W_CONNECTIVITY * connectivity
    )


# Reuses the same date/occasion vocabulary as _is_must_keep's keyword check,
# but as a soft salience signal for facts that didn't already qualify as
# must_keep via kind or the narrower keyword list (e.g. "always"/"favorite"
# style facts that matter but aren't calendar-significant).
_SALIENCE_HIT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k.strip()) for k in (
        "deadline", "birthday", "anniversary", "appointment", "hackathon",
        "interview", "lost", "passport", "license", "wallet", "important",
        "breakthrough", "problem", "always", "never", "favorite", "favourite",
    )) + r")\b",
    re.IGNORECASE,
)


def _apply_retention_gate(
    memorize,
    user_id: str,
    daily_rows: list[dict],
) -> tuple[list[dict], dict]:
    """Split daily_rows into must_keep + scored/gated candidates.

    Returns (kept_rows, stats) where kept_rows = must_keep + surviving
    candidates (this is what gets chunked and sent to the LLM extraction/
    merge step), and stats is a small audit dict for the result payload —
    "why did this month keep N facts" should be answerable from the log
    line alone, matching the Grace design's auditability goal.
    """
    must_keep_ids: set[str] = set()
    must_keep_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for row in daily_rows:
        text = row.get("_text", "") or ""
        if _is_must_keep(text):
            rid = row.get("id")
            if rid:
                must_keep_ids.add(rid)
            must_keep_rows.append(row)
        else:
            candidate_rows.append(row)

    if not candidate_rows:
        return must_keep_rows, {
            "must_keep": len(must_keep_rows),
            "candidates": 0,
            "kept_candidates": 0,
            "dropped_candidates": 0,
        }

    static_anchors = _build_static_anchors(memorize, user_id)
    entity_weights = _entity_connectivity_weights(memorize, user_id)
    entity_weight_cap = max(entity_weights.values(), default=1.0) or 1.0

    candidate_texts = [row.get("_text", "") or "" for row in candidate_rows]
    candidate_vectors = None
    try:
        candidate_vectors = np.array(memorize.embed_texts(candidate_texts, query=False))
    except Exception as exc:
        log.warning(
            "Retention gate: failed to embed %d candidate row(s), "
            "novelty will default to neutral for this run: %s",
            len(candidate_texts), exc,
        )

    scored: list[tuple[float, dict]] = []
    for i, row in enumerate(candidate_rows):
        vec = candidate_vectors[i] if candidate_vectors is not None else None
        score = _score_daily_row(
            row,
            static_anchors=static_anchors,
            row_vector=vec,
            entity_weights=entity_weights,
            entity_weight_cap=entity_weight_cap,
        )
        scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    above_threshold = sum(1 for score, _ in scored if score >= CONSOLIDATION_SOFT_THRESHOLD)
    target_count = max(CONSOLIDATION_MIN_MONTH, min(CONSOLIDATION_MAX_MONTH, above_threshold))
    target_count = min(target_count, len(scored))

    kept_candidates = [row for _, row in scored[:target_count]]
    dropped_candidates = [row for _, row in scored[target_count:]]

    log.debug(
        "Retention gate: %d must_keep, %d candidates scored (threshold=%.2f "
        "above=%d), target_count=%d -> kept=%d dropped=%d",
        len(must_keep_rows), len(scored), CONSOLIDATION_SOFT_THRESHOLD,
        above_threshold, target_count, len(kept_candidates), len(dropped_candidates),
    )

    return must_keep_rows + kept_candidates, {
        "must_keep": len(must_keep_rows),
        "candidates": len(scored),
        "kept_candidates": len(kept_candidates),
        "dropped_candidates": len(dropped_candidates),
    }


# ── monthly fact extraction (mirrors memory/reflect.py's daily fact extraction,
#    applied at month scope instead of day scope) ────────────────────────────

_MONTHLY_FACTS_SYSTEM = textwrap.dedent("""
    You are compressing a month's worth of daily memory facts about {USER_ID} into
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
      written directly in the fact's own text (e.g. "On June 3rd, {USER_ID}
      celebrated a birthday with fruit tarts."). The specific date will
      NOT be preserved anywhere else after this — if it is not in the text,
      it is permanently lost. When in doubt about whether something counts
      as date-significant, err on the side of keeping the date.
    - For routine or recurring facts with no specific date significance, do
      NOT include a specific date — summarize at month-level only (e.g.
      "Spent much of the month refining Aiko-chan's memory retrieval
      pipeline.").
    - Do not invent details, outcomes, dates, or facts not supported by the
      source material.
    - One fact per line, third person, about {USER_ID}.
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
    You are merging several partial lists of monthly facts about {USER_ID} into
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
    - One fact per line, third person, about {USER_ID}.

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
    raw = _chat(_MONTHLY_FACTS_SYSTEM.format(USER_ID=current_display_name()), user_prompt, max_tokens=900, temperature=0.1)
    return _parse_fact_array(raw)


def _merge_monthly_facts(month_key: str, chunk_facts: list[list[str]]) -> list[str]:
    if len(chunk_facts) == 1:
        return chunk_facts[0]
    chunks_text = "\n\n".join(
        f"List {i+1}:\n" + "\n".join(f"- {f}" for f in facts)
        for i, facts in enumerate(chunk_facts)
    )
    user_prompt = _MONTHLY_MERGE_USER.format(month_key=month_key, chunks=chunks_text)
    raw = _chat(_MONTHLY_MERGE_SYSTEM.format(USER_ID=current_display_name()), user_prompt, max_tokens=1200, temperature=0.1)
    merged = _parse_fact_array(raw)
    return merged or [f for facts in chunk_facts for f in facts]  # fallback: concatenate if merge parse fails


# ── main entrypoint ───────────────────────────────────────────────────────────

def maybe_run_consolidation(memorize, now: datetime | None = None, user_id: str | None = None) -> dict:
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

    Compresses pinned daily-granularity records (atomic "[YYYY-MM-DD] fact"
    rows from memory.db and "Daily journal of YYYY-MM-DD:" blobs from journal.db) for the target month into
    a smaller set of pinned "[YYYY-MM] fact" rows, then deletes the
    daily-granularity originals for that month. Unpinned memory is never
    read, scored, or deleted here — that lifecycle belongs entirely to
    memory.forget / memorize.dream(), independent of this job.

    Before extraction, daily rows are passed through _apply_retention_gate:
    must-keep rows (dated/event/plan facts) always survive; the remaining
    candidates are scored and floor/ceiling-gated (see module docstring)
    so a quiet month doesn't get wiped and a busy month can't make the
    monthly archive grow unbounded. The LLM extraction/merge step still
    decides wording and how surviving facts get combined — it never decides
    which facts live or die.

    Returns a result dict with keys: ran, reason (on skip), month, count,
    facts_written, daily_deleted, plus retention-gate stats (must_keep,
    candidates, kept_candidates, dropped_candidates) when the gate ran.
    """
    if not CONSOLIDATION_ENABLED:
        return {"ran": False, "reason": "disabled"}

    now = now or bioclock.local_now()
    user_id = user_id or (memorize.get_user_id() if memorize else None) or current_user_id()

    start, end, month_key = target_month_for(now)
    state = _load_state(user_id)
    if state.get("last_consolidated_month") == month_key:
        return {"ran": False, "reason": "already_done", "month": month_key}

    start_utc = start.astimezone(timezone.utc)
    end_utc   = end.astimezone(timezone.utc)
    all_memories = memorize.get_between(start_utc, end_utc, user_id=user_id)

    # Scope memory.db to pinned daily-granularity atomic facts only. The large
    # faithful daily blobs moved to journal.db so memory recall is not polluted
    # by oversized pinned entries.
    daily_rows = [
        dict(m) | {"_store": "memory", "_text": (m.get("memory") or "").strip()}
        for m in all_memories
        if int(m.get("pinned") or 0) == 1
        and _DAILY_FACT_TAG_RE.match((m.get("memory") or "").strip())
    ]

    try:
        from memory import journal
        journal_rows = journal.get_between(start_utc, end_utc, user_id=user_id)
        daily_rows.extend(
            dict(j) | {"_store": "journal", "_text": (j.get("body") or "").strip()}
            for j in journal_rows
            if int(j.get("pinned") or 0) == 1
        )
    except Exception as exc:
        log.warning("Failed to load daily journals for monthly consolidation: %s", exc)

    if len(daily_rows) < CONSOLIDATION_MIN_MEMS:
        state["last_consolidated_month"] = month_key
        _save_state(state, user_id)
        return {"ran": False, "reason": "too_few_memories", "month": month_key, "count": len(daily_rows)}

    # ── retention gate (NEW): decide which daily rows even reach the LLM ──
    kept_rows, gate_stats = _apply_retention_gate(memorize, user_id, daily_rows)

    source_facts = [m.get("_text", "").strip() for m in kept_rows if m.get("_text", "").strip()]
    chunks = [source_facts[i:i + CONSOLIDATION_CHUNK_MEMS] for i in range(0, len(source_facts), CONSOLIDATION_CHUNK_MEMS)]

    chunk_facts = [
        _extract_monthly_facts_chunk(month_key, chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    chunk_facts = [c for c in chunk_facts if c]  # drop empty chunks (parse failures)

    if not chunk_facts:
        return {"ran": False, "reason": "empty_extraction", "month": month_key, "count": len(daily_rows), **gate_stats}

    final_facts = _merge_monthly_facts(month_key, chunk_facts)
    if not final_facts:
        return {"ran": False, "reason": "empty_merge", "month": month_key, "count": len(daily_rows), **gate_stats}

    facts_written = 0
    written_ids: list[str] = []
    for fact in final_facts:
        try:
            mem_id = memorize.add_raw(f"[{month_key}] {fact}", user_id=user_id, pinned=True)
            if mem_id:
                facts_written += 1
                written_ids.append(mem_id)
        except Exception as e:
            log.warning("Failed to pin monthly fact %r: %s", fact, e)

    if facts_written == 0:
        return {"ran": False, "reason": "no_facts_written", "month": month_key, "count": len(daily_rows), **gate_stats}

    # Only now delete the daily-granularity originals for this month — ALL
    # of them, including retention-gate-dropped candidates, since dropped
    # facts were deliberately excluded from the monthly archive rather than
    # folded into it (that's the point of the gate); their content isn't
    # preserved anywhere after this, same as any other forgotten memory.
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
                if m.get("_store") == "journal":
                    from memory import journal
                    journal.delete(mem_id, user_id=user_id)
                else:
                    memorize.delete(mem_id)
                daily_deleted += 1
            except Exception as e:
                log.warning("Failed to delete consolidated daily row %s: %s", mem_id, e)

    state["last_consolidated_month"] = month_key
    state["last_summary_ids"]        = written_ids
    _save_state(state, user_id)

    log.info(
        "monthly_consolidate complete: month=%s source_count=%s must_keep=%s "
        "candidates=%s kept_candidates=%s dropped_candidates=%s "
        "facts_written=%s daily_deleted=%s",
        month_key, len(daily_rows), gate_stats["must_keep"], gate_stats["candidates"],
        gate_stats["kept_candidates"], gate_stats["dropped_candidates"],
        facts_written, daily_deleted,
    )

    # Run monthly maintenance (archive reports, prune KB, vacuum)
    try:
        maintenance_results = _maintenance_run(user_id, memorize=memorize)
        log.info("monthly_maintenance complete: %s", maintenance_results)
    except Exception as exc:
        log.warning("monthly_maintenance failed: %s", exc)

    return {
        "ran":            True,
        "month":          month_key,
        "count":          len(daily_rows),
        "facts_written":  facts_written,
        "daily_deleted":  daily_deleted,
        **gate_stats,
    }


# ── Monthly maintenance helpers ──────────────────────────────────────────────
# Called at end of consolidation to archive reports, prune KB, vacuum DBs.

def _archive_reports(user_id: str | None = None, keep_days: int = 90) -> dict:
    """Archive report files older than keep_days to reports/archive/."""
    from pathlib import Path
    import shutil
    from datetime import datetime, timedelta
    from system.userspace import user_workspace_root

    uid = user_id or current_user_id()
    root = user_workspace_root(uid)
    reports_dir = root / "reports"
    archive_dir = reports_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now() - timedelta(days=keep_days)
    moved = 0
    errors = 0

    if not reports_dir.exists():
        return {"moved": 0, "errors": 0}

    for report_file in reports_dir.iterdir():
        if not report_file.is_file() or report_file.suffix not in {".md", ".txt", ".json"}:
            continue
        try:
            mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
            if mtime < cutoff:
                dest = archive_dir / report_file.name
                if dest.exists():
                    # Rename with timestamp to avoid collision
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest = archive_dir / f"{report_file.stem}_{ts}{report_file.suffix}"
                shutil.move(str(report_file), str(dest))
                moved += 1
        except Exception as exc:
            log.warning("failed to archive %s: %s", report_file, exc)
            errors += 1

    return {"moved": moved, "errors": errors}

def _resolve_embedder(memorize=None):
    """Best-effort embedder for maintenance (dedupe)."""
    # Try to get embedder from memorize's internal state
    if memorize is not None:
        try:
            emb = getattr(getattr(memorize, "_mem", None), "_embedder", None)
            if emb is not None and hasattr(emb, "embed_query"):
                return emb
        except Exception:
            log.warning("consolidate: failed to get embedder from memorize")
    
    # Fallback: load from reason module (your semantic caching source)
    try:
        from cognition import reason
        return reason.load_embedder()
    except Exception:
        return None

def _maintenance_run(user_id: str | None = None, memorize=None) -> dict:
    """Run monthly maintenance: archive reports, prune KB, vacuum DBs."""
    uid = user_id or current_user_id()
    results = {}

    # 1. Archive old reports (keep 90 days hot)
    try:
        results["archive_reports"] = _archive_reports(uid, keep_days=90)
    except Exception as exc:
        log.warning("archive_reports failed: %s", exc)
        results["archive_reports"] = {"error": str(exc)}

    # 2. Prune knowledge DB (archive cold, delete never-used, dedupe)
    try:
        from memory.knowledge import prune_knowledge
        emb = _resolve_embedder(memorize)  # ← GET THE EMBEDDER
        results["prune_knowledge"] = prune_knowledge(
            keep_days=30,
            min_access=2,
            archive_days=90,
            delete_days=180,
            dedupe_threshold=0.95,
            user_id=uid,
            embedder=emb,  # ← PASS IT
        )
        if emb is None:
            results["prune_knowledge_note"] = "dedupe skipped (no embedder)"
    except Exception as exc:
        log.warning("prune_knowledge failed: %s", exc)
        results["prune_knowledge"] = {"error": str(exc)}

    # 3. Vacuum both DBs to reclaim space
    try:
        from memory.knowledge import vacuum_knowledge_db
        vacuum_knowledge_db(uid)
        results["vacuum_knowledge"] = "ok"
    except Exception as exc:
        log.warning("vacuum_knowledge failed: %s", exc)
        results["vacuum_knowledge"] = {"error": str(exc)}
    
    try:
        from memory.memorize import vacuum_memory_db  # ← NEW PUBLIC IMPORT
        vacuum_memory_db(uid)
        results["vacuum_memory"] = "ok"
    except Exception as exc:
        log.warning("vacuum_memory failed: %s", exc)
        results["vacuum_memory"] = {"error": str(exc)}

    return results
