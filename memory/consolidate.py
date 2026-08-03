"""
memory/consolidate.py

Monthly memory consolidation.

Runs on/after the first day of a month and consolidates the month before the
most recent full month. Example: on July 1, keep June intact and summarize May.

Scope: this ONLY touches pinned daily-granularity memory (atomic facts tagged
"[YYYY-MM-DD] ..." rows in memory.db). Journal day blobs in journal.db are
loaded for counting/observability only in Phase 1 — they are not scored as
retention candidates and are never deleted here (no journal archival path yet).
Unpinned memory is entirely out of scope — its lifecycle is owned by
memory.forget / memorize.dream().

Retention gate (Phase 1):
  Before facts are sent to the LLM, *memory.db* day atomics for the target
  month are split into must_keep vs scored candidates (floor/ceiling).
  Static anchors come from prior "[YYYY-MM] ..." archive only.
  LLM merge/compress only decides wording among survivors.

  Full source-id provenance through the LLM (hard coverage proof before
  delete) is deferred; MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES defaults
  off so day pins are not removed until the gate is audited.

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

CONSOLIDATION_MIN_MONTH        = max(0, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MONTH", "8")))
CONSOLIDATION_MAX_MONTH        = max(CONSOLIDATION_MIN_MONTH, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_MONTH", "30")))
CONSOLIDATION_SOFT_THRESHOLD   = float(os.getenv("MONTHLY_CONSOLIDATION_SOFT_THRESHOLD", "0.4"))
CONSOLIDATION_ANCHOR_LOOKBACK  = max(2, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_LOOKBACK", "50")))
CONSOLIDATION_ANCHOR_K         = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_K", "5")))
_RETENTION_W_SALIENCE     = float(os.getenv("MONTHLY_CONSOLIDATION_W_SALIENCE", "0.30"))
_RETENTION_W_NOVELTY      = float(os.getenv("MONTHLY_CONSOLIDATION_W_NOVELTY", "0.25"))
_RETENTION_W_SPACING      = float(os.getenv("MONTHLY_CONSOLIDATION_W_SPACING", "0.20"))
_RETENTION_W_CONNECTIVITY = float(os.getenv("MONTHLY_CONSOLIDATION_W_CONNECTIVITY", "0.25"))
_RETENTION_SPACING_SATURATION = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_SPACING_SATURATION", "5")))

def consolidation_state_path(user_id: str | None = None) -> Path:
    override = os.getenv("MONTHLY_CONSOLIDATION_STATE_PATH")
    if override:
        return Path(override).expanduser()
    return user_state_path("memory/monthly_consolidation_state.json", user_id or current_user_id())


LLM_BASE_URL          = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL             = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
CONSOLIDATION_LLM_TIMEOUT = float(os.getenv("MONTHLY_CONSOLIDATION_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "120")))
# Default OFF until the retention gate has been audited on at least one real month.
CONSOLIDATION_DELETE_DAILY_SUMMARIES = os.getenv("MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES", "0").lower() in {"1", "true", "yes", "on"}

_DAILY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")
_MONTHLY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}\]\s")

_MUST_KEEP_KEYWORDS = (
    "deadline", "birthday", "anniversary", "appointment", "hackathon",
    "interview", "lost ", "passport", "license", "wallet",
)


def _is_must_keep(text: str) -> bool:
    kind = classify_kind(text, default="fact")
    if kind in ("event", "plan"):
        return True
    low = (text or "").casefold()
    return any(k in low for k in _MUST_KEEP_KEYWORDS)


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year  = dt.year + month_index // 12
    month = month_index % 12 + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def target_month_for(now: datetime) -> tuple[datetime, datetime, str]:
    local_first  = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    target_end   = _add_months(local_first, -CONSOLIDATION_KEEP_MONTHS)
    target_start = _add_months(target_end, -1)
    key          = target_start.strftime("%Y-%m")
    return target_start, target_end, key


def _load_state(user_id: str | None = None) -> dict:
    try:
        return json.loads(consolidation_state_path(user_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict, user_id: str | None = None) -> None:
    path = consolidation_state_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


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


def _entity_connectivity_weights(memorize, user_id: str) -> dict[str, float]:
    try:
        lock = getattr(getattr(memorize, "_mem", None), "_db_lock", None)
        if lock is not None:
            with lock:
                rows = memorize._conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
        else:
            rows = memorize._conn.execute(
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

    return np.mean(vectors, axis=0, keepdims=True)


_SALIENCE_HIT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k.strip()) for k in (
        "deadline", "birthday", "anniversary", "appointment", "hackathon",
        "interview", "lost", "passport", "license", "wallet", "important",
        "breakthrough", "problem", "always", "never", "favorite", "favourite",
    )) + r")\b",
    re.IGNORECASE,
)


def _score_daily_row(
    row: dict,
    *,
    static_anchors: "np.ndarray | None",
    row_vector: "np.ndarray | None",
    entity_weights: dict[str, float],
    entity_weight_cap: float,
) -> float:
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
            novelty = float(max(0.0, min(1.0, 1.0 - float(np.max(sims))))
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


def _apply_retention_gate(
    memorize,
    user_id: str,
    daily_rows: list[dict],
) -> tuple[list[dict], dict]:
    must_keep_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for row in daily_rows:
        text = row.get("_text", "") or ""
        if _is_must_keep(text):
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

    log.info(
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
    - One fact per line, third person, about {USER_ID}.
    - Each fact must be self-contained and short.

    Return ONLY a JSON array of short strings. No markdown, no explanation.
""").strip()

_MONTHLY_FACTS_USER = textwrap.dedent("""
    Month: {month_key}
    Chunk: {idx}/{total}

    Pre-selected daily facts (do not drop except exact/near duplicates):
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
    return merged or [f for facts in chunk_facts for f in facts]


def maybe_run_consolidation(memorize, now: datetime | None = None, user_id: str | None = None) -> dict:
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

    memory_day_rows = [
        dict(m) | {"_store": "memory", "_text": (m.get("memory") or "").strip()}
        for m in all_memories
        if int(m.get("pinned") or 0) == 1
        and _DAILY_FACT_TAG_RE.match((m.get("memory") or "").strip())
    ]

    # Journals: local calendar boundaries only (entry_date is YYYY-MM-DD local).
    # Not scored; not deleted in Phase 1 (no journal archival path yet).
    journal_day_rows: list[dict] = []
    try:
        from memory import journal
        journal_rows = journal.get_between(start, end, user_id=user_id)
        journal_day_rows = [
            dict(j) | {"_store": "journal", "_text": (j.get("body") or "").strip()}
            for j in journal_rows
            if int(j.get("pinned") or 0) == 1
        ]
    except Exception as exc:
        log.warning("Failed to load daily journals for monthly consolidation: %s", exc)

    source_count = len(memory_day_rows) + len(journal_day_rows)
    if len(memory_day_rows) < CONSOLIDATION_MIN_MEMS:
        state["last_consolidated_month"] = month_key
        _save_state(state, user_id)
        return {
            "ran": False,
            "reason": "too_few_memories",
            "month": month_key,
            "count": source_count,
            "memory_day_count": len(memory_day_rows),
        }

    kept_rows, gate_stats = _apply_retention_gate(memorize, user_id, memory_day_rows)

    source_facts = [m.get("_text", "").strip() for m in kept_rows if m.get("_text", "").strip()]
    chunks = [source_facts[i:i + CONSOLIDATION_CHUNK_MEMS] for i in range(0, len(source_facts), CONSOLIDATION_CHUNK_MEMS)]

    chunk_facts = [
        _extract_monthly_facts_chunk(month_key, chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    chunk_facts = [c for c in chunk_facts if c]

    if not chunk_facts:
        return {"ran": False, "reason": "empty_extraction", "month": month_key, "count": source_count, **gate_stats}

    final_facts = _merge_monthly_facts(month_key, chunk_facts)
    if not final_facts:
        return {"ran": False, "reason": "empty_merge", "month": month_key, "count": source_count, **gate_stats}

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
        return {"ran": False, "reason": "no_facts_written", "month": month_key, "count": source_count, **gate_stats}

    # Phase 1: only memory.db day pins may be deleted (when DELETE is on).
    # Journals are never deleted here — they were not LLM-archived sources.
    daily_deleted = 0
    if CONSOLIDATION_DELETE_DAILY_SUMMARIES:
        for m in memory_day_rows:
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
    _save_state(state, user_id)

    log.info(
        "monthly_consolidate complete: month=%s source_count=%s memory_days=%s "
        "journals=%s journals_deleted=0 must_keep=%s candidates=%s kept_candidates=%s "
        "dropped_candidates=%s facts_written=%s daily_deleted=%s delete_enabled=%s",
        month_key, source_count, len(memory_day_rows), len(journal_day_rows),
        gate_stats["must_keep"], gate_stats["candidates"],
        gate_stats["kept_candidates"], gate_stats["dropped_candidates"],
        facts_written, daily_deleted, CONSOLIDATION_DELETE_DAILY_SUMMARIES,
    )

    try:
        maintenance_results = _maintenance_run(user_id, memorize=memorize)
        log.info("monthly_maintenance complete: %s", maintenance_results)
    except Exception as exc:
        log.warning("monthly_maintenance failed: %s", exc)

    return {
        "ran":            True,
        "month":          month_key,
        "count":          source_count,
        "memory_day_count": len(memory_day_rows),
        "journal_count":  len(journal_day_rows),
        "facts_written":  facts_written,
        "daily_deleted":  daily_deleted,
        "journals_deleted": 0,
        **gate_stats,
    }


def _archive_reports(user_id: str | None = None, keep_days: int = 90) -> dict:
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
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest = archive_dir / f"{report_file.stem}_{ts}{report_file.suffix}"
                shutil.move(str(report_file), str(dest))
                moved += 1
        except Exception as exc:
            log.warning("failed to archive %s: %s", report_file, exc)
            errors += 1

    return {"moved": moved, "errors": errors}

def _resolve_embedder(memorize=None):
    if memorize is not None:
        try:
            emb = getattr(getattr(memorize, "_mem", None), "_embedder", None)
            if emb is not None and hasattr(emb, "embed_query"):
                return emb
        except Exception:
            log.warning("consolidate: failed to get embedder from memorize")

    try:
        from cognition import reason
        return reason.load_embedder()
    except Exception:
        return None

def _maintenance_run(user_id: str | None = None, memorize=None) -> dict:
    uid = user_id or current_user_id()
    results = {}

    try:
        results["archive_reports"] = _archive_reports(uid, keep_days=90)
    except Exception as exc:
        log.warning("archive_reports failed: %s", exc)
        results["archive_reports"] = {"error": str(exc)}

    try:
        from memory.knowledge import prune_knowledge
        emb = _resolve_embedder(memorize)
        results["prune_knowledge"] = prune_knowledge(
            keep_days=30,
            min_access=2,
            archive_days=90,
            delete_days=180,
            dedupe_threshold=0.95,
            user_id=uid,
            embedder=emb,
        )
        if emb is None:
            results["prune_knowledge_note"] = "dedupe skipped (no embedder)"
    except Exception as exc:
        log.warning("prune_knowledge failed: %s", exc)
        results["prune_knowledge"] = {"error": str(exc)}

    try:
        from memory.knowledge import vacuum_knowledge_db
        vacuum_knowledge_db(uid)
        results["vacuum_knowledge"] = "ok"
    except Exception as exc:
        log.warning("vacuum_knowledge failed: %s", exc)
        results["vacuum_knowledge"] = {"error": str(exc)}

    try:
        from memory.memorize import vacuum_memory_db
        vacuum_memory_db(uid)
        results["vacuum_memory"] = "ok"
    except Exception as exc:
        log.warning("vacuum_memory failed: %s", exc)
        results["vacuum_memory"] = {"error": str(exc)}

    return results
