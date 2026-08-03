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

  Phase 2 spacing: retention uses access_day_count (distinct local days a
  memory was recalled via search/_touch_memories), not raw access_count.

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
        log.debug("Connectivity weights: entity_relations read skipped: %s", exp)
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
