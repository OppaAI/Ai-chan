"""Public consolidation-package façade.

The monthly consolidation implementation is grouped here (instead of under
``cognition.memory``) so retention, promotion, archival, and journal helpers can
remain together.  Implementation is split by responsibility:

- :mod:`.schema` — config, state persistence, LLM plumbing, month math
- :mod:`.retention` — retention gate, anchors, row scoring
- :mod:`.facts` — LLM fact extraction / merge / provenance
- :mod:`.promote` — journal-fragment promotion
- :mod:`.lifecycle` — monthly orchestration, archival, maintenance
- :mod:`.journal` — encrypted daily-journal store

``ConsolidationStore`` is the shared coordinator.  Module-level functions
preserve the historical import path used by schedule, think, and tests.
"""
from __future__ import annotations

from .schema import (
    CONSOLIDATION_ANCHOR_K,
    CONSOLIDATION_ANCHOR_LOOKBACK,
    CONSOLIDATION_CHUNK_MEMS,
    CONSOLIDATION_DELETE_DAILY_SUMMARIES,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_KEEP_MONTHS,
    CONSOLIDATION_MAX_INPUT_CHARS,
    CONSOLIDATION_MAX_MONTH,
    CONSOLIDATION_MIN_MEMS,
    CONSOLIDATION_MIN_MONTH,
    CONSOLIDATION_SOFT_THRESHOLD,
    DELETE_MIN_RATIO,
    DELETE_MIN_WRITTEN,
    DELETE_REQUIRE_COVERAGE,
    HARD_SOURCE_PROVENANCE,
    JOURNAL_PROMOTE,
    JOURNAL_PROMOTE_K,
    LLM_BASE_URL,
    LLM_MODEL,
    consolidation_state_path,
    target_month_for,
)
from .retention import (
    apply_retention_gate,
    build_dynamic_anchors,
    build_static_anchors,
    entity_connectivity_weights,
    is_must_keep,
    score_daily_row,
)
from .facts import (
    extract_monthly_facts_chunk,
    hard_provenance_ok,
    merge_monthly_facts,
    parse_fact_array,
    parse_fact_items,
)
from .promote import (
    journal_fragment_lines,
    promote_journal_fragments,
    score_journal_fragment,
)
from .lifecycle import (
    archive_reports,
    maintenance_run,
    maybe_run_consolidation,
)

# The journal module is imported lazily inside lifecycle to avoid a cycle with
# cognition.memory.reflect; expose it as a submodule reference for callers.
from . import journal  # noqa: F401


class ConsolidationStore:
    """Shared coordinator for the monthly consolidation subsystem."""

    def __init__(self, memorize=None):
        self.memorize = memorize

    def maybe_run(self, *, now=None, user_id: str | None = None) -> dict:
        return maybe_run_consolidation(self.memorize, now=now, user_id=user_id)

    def run_maintenance(self, *, user_id: str | None = None) -> dict:
        return maintenance_run(user_id, memorize=self.memorize)


__all__ = [
    "ConsolidationStore",
    "archive_reports",
    "apply_retention_gate",
    "build_dynamic_anchors",
    "build_static_anchors",
    "consolidation_state_path",
    "CONSOLIDATION_ANCHOR_K",
    "CONSOLIDATION_ANCHOR_LOOKBACK",
    "CONSOLIDATION_CHUNK_MEMS",
    "CONSOLIDATION_DELETE_DAILY_SUMMARIES",
    "CONSOLIDATION_ENABLED",
    "CONSOLIDATION_KEEP_MONTHS",
    "CONSOLIDATION_MAX_INPUT_CHARS",
    "CONSOLIDATION_MAX_MONTH",
    "CONSOLIDATION_MIN_MEMS",
    "CONSOLIDATION_MIN_MONTH",
    "CONSOLIDATION_SOFT_THRESHOLD",
    "DELETE_MIN_RATIO",
    "DELETE_MIN_WRITTEN",
    "DELETE_REQUIRE_COVERAGE",
    "entity_connectivity_weights",
    "extract_monthly_facts_chunk",
    "hard_provenance_ok",
    "HARD_SOURCE_PROVENANCE",
    "is_must_keep",
    "journal",
    "journal_fragment_lines",
    "JOURNAL_PROMOTE",
    "JOURNAL_PROMOTE_K",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "maintenance_run",
    "maybe_run_consolidation",
    "merge_monthly_facts",
    "parse_fact_array",
    "parse_fact_items",
    "promote_journal_fragments",
    "score_daily_row",
    "score_journal_fragment",
    "target_month_for",
]