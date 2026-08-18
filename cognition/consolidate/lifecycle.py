"""Consolidation lifecycle, archival, and maintenance entry points.

Owns the monthly orchestration (maybe_run_consolidation) plus the supporting
coverage gate, report archival, and knowledge/memory maintenance helpers.
Mirrors :mod:`cognition.knowledge` structure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from system import bioclock
from system.log import get_logger
from system.userspace import current_user_id

from .facts import (
    extract_monthly_facts_chunk,
    hard_provenance_ok,
    merge_monthly_facts,
)
from .promote import promote_journal_fragments
from .retention import apply_retention_gate
from .schema import (
    _DAILY_FACT_TAG_RE,
    _load_state,
    _save_state,
    CONSOLIDATION_CHUNK_MEMS,
    CONSOLIDATION_DELETE_DAILY_SUMMARIES,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_MIN_MEMS,
    DELETE_MIN_RATIO,
    DELETE_MIN_WRITTEN,
    DELETE_REQUIRE_COVERAGE,
    HARD_SOURCE_PROVENANCE,
    target_month_for,
)

log = get_logger(__name__)

__all__ = ["archive_reports", "maintenance_run", "maybe_run_consolidation"]


def _delete_coverage_ok(facts_written: int, kept_count: int) -> bool:
    """Soft archival coverage: enough monthly facts vs gated survivors."""
    if not DELETE_REQUIRE_COVERAGE:
        return True
    if facts_written < DELETE_MIN_WRITTEN:
        return False
    if kept_count <= 0:
        return facts_written >= DELETE_MIN_WRITTEN
    ratio = facts_written / float(kept_count)
    return ratio >= DELETE_MIN_RATIO


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
    end_utc = end.astimezone(timezone.utc)
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
        from cognition.consolidate import journal
        journal_rows = journal.get_between(start, end, user_id=user_id)
        journal_day_rows = [
            dict(j) | {"_store": "journal", "_text": (j.get("body") or "").strip()}
            for j in journal_rows
            if int(j.get("pinned") or 0) == 1
        ]
    except Exception as exc:
        log.warning("Failed to load daily journals for monthly consolidation: %s", exc)

    # Phase 7: selective journal promote into day pins (before gate + min-count).
    promoted_rows, journal_promoted = promote_journal_fragments(
        memorize, user_id, month_key, journal_day_rows, memory_day_rows,
    )
    if promoted_rows:
        memory_day_rows = memory_day_rows + promoted_rows

    cognitive_state = None
    cognitive_lesson_rows = []
    try:
        from cognition.memory.edge_state import for_identity
        cognitive_state = for_identity(user_id)
        cognitive_lesson_rows = [
            {"id": None, "memory": f"Interaction lesson: {lesson}", "pinned": 0,
             "access_count": 0, "access_day_count": 0, "entities": "[]",
             "status": "active", "_store": "cognitive",
             "_text": f"Interaction lesson: {lesson}", "_cognitive_lesson": True}
            for lesson in cognitive_state.snapshot().get("lessons", []) if lesson
        ]
        reflection_text = cognitive_state.reflection_summary() if cognitive_state is not None else ""
        if reflection_text and "No unresolved cognitive issue" not in reflection_text:
            reflection_row = {"id": None, "memory": f"Cognitive reflection: {reflection_text}", "pinned": 0,
                              "access_count": 0, "access_day_count": 0, "entities": "[]",
                              "status": "active", "_store": "cognitive",
                              "_text": f"Cognitive reflection: {reflection_text}", "_cognitive_reflection": True}
            memory_day_rows.append(reflection_row)
        if cognitive_lesson_rows:
            memory_day_rows = memory_day_rows + cognitive_lesson_rows
    except Exception as exc:
        log.debug("Cognitive lesson consolidation skipped: %s", exc)

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

    kept_rows, gate_stats = apply_retention_gate(memorize, user_id, memory_day_rows)

    kept_for_llm = [m for m in kept_rows if (m.get("_text") or "").strip()]
    chunks = [kept_for_llm[i:i + CONSOLIDATION_CHUNK_MEMS] for i in range(0, len(kept_for_llm), CONSOLIDATION_CHUNK_MEMS)]

    chunk_items = [
        extract_monthly_facts_chunk(month_key, chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]
    chunk_items = [c for c in chunk_items if c]

    if not chunk_items:
        return {"ran": False, "reason": "empty_extraction", "month": month_key, "count": source_count, **gate_stats}

    final_items = merge_monthly_facts(month_key, chunk_items)
    if not final_items:
        return {"ran": False, "reason": "empty_merge", "month": month_key, "count": source_count, **gate_stats}

    facts_written = 0
    written_ids: list[str] = []
    for item in final_items:
        fact = (item.get("fact") if isinstance(item, dict) else item) or ""
        fact = str(fact).strip()
        if not fact:
            continue
        try:
            mem_id = memorize.add_raw(f"[{month_key}] {fact}", user_id=user_id, pinned=True)
            if mem_id:
                facts_written += 1
                written_ids.append(mem_id)
        except Exception as e:
            log.warning("Failed to pin monthly fact %r: %s", fact, e)

    if facts_written > 0 and cognitive_state is not None and cognitive_lesson_rows:
        cognitive_state.consume_lessons()

    if facts_written == 0:
        return {"ran": False, "reason": "no_facts_written", "month": month_key, "count": source_count, **gate_stats}

    # Phase 7 soft coverage + Phase 11 hard source-id provenance before delete.
    daily_deleted = 0
    delete_skipped_reason = ""
    kept_count = len(kept_rows)
    hard_ok, covered_ids = True, set()
    if HARD_SOURCE_PROVENANCE:
        hard_ok, covered_ids = hard_provenance_ok(kept_rows, final_items)

    if CONSOLIDATION_DELETE_DAILY_SUMMARIES:
        if not _delete_coverage_ok(facts_written, kept_count):
            delete_skipped_reason = "coverage_failed"
            log.warning(
                "Phase 7: skip day-pin delete month=%s facts_written=%s kept_rows=%s "
                "(need written>=%s and ratio>=%.2f)",
                month_key, facts_written, kept_count,
                DELETE_MIN_WRITTEN, DELETE_MIN_RATIO,
            )
        elif HARD_SOURCE_PROVENANCE and not hard_ok:
            delete_skipped_reason = "hard_provenance_failed"
            log.warning(
                "Phase 11: skip day-pin delete month=%s — incomplete source_ids coverage",
                month_key,
            )
        else:
            for m in memory_day_rows:
                if m.get("_promoted_from_journal"):
                    continue
                mem_id = m.get("id")
                if not mem_id:
                    continue
                # When hard provenance is on, only delete ids that were covered.
                if HARD_SOURCE_PROVENANCE and str(mem_id) not in covered_ids:
                    continue
                try:
                    memorize.delete(mem_id)
                    daily_deleted += 1
                except Exception as e:
                    log.warning("Failed to delete consolidated daily row %s: %s", mem_id, e)

    state["last_consolidated_month"] = month_key
    state["last_summary_ids"] = written_ids
    _save_state(state, user_id)

    log.info(
        "monthly_consolidate complete: month=%s source_count=%s memory_days=%s "
        "journals=%s journals_deleted=0 must_keep=%s candidates=%s kept_candidates=%s "
        "dropped_candidates=%s facts_written=%s daily_deleted=%s delete_enabled=%s "
        "journal_promoted=%s delete_skipped=%s",
        month_key, source_count, len(memory_day_rows), len(journal_day_rows),
        gate_stats["must_keep"], gate_stats["candidates"],
        gate_stats["kept_candidates"], gate_stats["dropped_candidates"],
        facts_written, daily_deleted, CONSOLIDATION_DELETE_DAILY_SUMMARIES,
        journal_promoted, delete_skipped_reason or "none",
    )

    try:
        maintenance_results = maintenance_run(user_id, memorize=memorize)
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
        "journal_promoted": journal_promoted,
        "delete_skipped_reason": delete_skipped_reason,
        **gate_stats,
    }


def archive_reports(user_id: str | None = None, keep_days: int = 90) -> dict:
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


def maintenance_run(user_id: str | None = None, memorize=None) -> dict:
    uid = user_id or current_user_id()
    results = {}

    try:
        results["archive_reports"] = archive_reports(uid, keep_days=90)
    except Exception as exc:
        log.warning("archive_reports failed: %s", exc)
        results["archive_reports"] = {"error": str(exc)}

    try:
        from cognition.knowledge import prune_knowledge
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
        from cognition.knowledge import vacuum_knowledge_db
        vacuum_knowledge_db(uid)
        results["vacuum_knowledge"] = "ok"
    except Exception as exc:
        log.warning("vacuum_knowledge failed: %s", exc)
        results["vacuum_knowledge"] = {"error": str(exc)}

    try:
        from cognition.memory.memorize import vacuum_memory_db
        vacuum_memory_db(uid)
        results["vacuum_memory"] = "ok"
    except Exception as exc:
        log.warning("vacuum_memory failed: %s", exc)
        results["vacuum_memory"] = {"error": str(exc)}

    return results