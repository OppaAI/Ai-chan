""":mod:`cognition.consolidate.schema`

Configuration, shared constants, state persistence, and LLM plumbing for the
monthly consolidation subsystem.  This is the shared foundation imported by the
:mod:`.retention`, :mod:`.facts`, :mod:`.promote`, and :mod:`.lifecycle`
modules — mirroring how :mod:`cognition.knowledge.schema` anchors the knowledge
package.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from system.log import get_logger
from system.userspace import current_user_id, user_state_path

log = get_logger(__name__)

# ── consolidation tuning ──────────────────────────────────────────────────────

CONSOLIDATION_ENABLED = os.getenv("MONTHLY_CONSOLIDATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
CONSOLIDATION_KEEP_MONTHS = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_KEEP_MONTHS", "1")))
CONSOLIDATION_CHUNK_MEMS = max(5, int(os.getenv("MONTHLY_CONSOLIDATION_CHUNK_MEMS", "25")))
CONSOLIDATION_MAX_INPUT_CHARS = max(1000, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS", "6000")))
CONSOLIDATION_MIN_MEMS = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MEMS", "5")))

CONSOLIDATION_MIN_MONTH = max(0, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MONTH", "8")))
CONSOLIDATION_MAX_MONTH = max(CONSOLIDATION_MIN_MONTH, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_MONTH", "30")))
CONSOLIDATION_SOFT_THRESHOLD = float(os.getenv("MONTHLY_CONSOLIDATION_SOFT_THRESHOLD", "0.44"))
CONSOLIDATION_ANCHOR_LOOKBACK = max(2, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_LOOKBACK", "50")))
CONSOLIDATION_ANCHOR_K = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_ANCHOR_K", "5")))

_RETENTION_W_SALIENCE = float(os.getenv("MONTHLY_CONSOLIDATION_W_SALIENCE", "0.30"))
_RETENTION_W_NOVELTY = float(os.getenv("MONTHLY_CONSOLIDATION_W_NOVELTY", "0.25"))
_RETENTION_W_SPACING = float(os.getenv("MONTHLY_CONSOLIDATION_W_SPACING", "0.20"))
_RETENTION_W_CONNECTIVITY = float(os.getenv("MONTHLY_CONSOLIDATION_W_CONNECTIVITY", "0.25"))
_RETENTION_W_VALENCE = float(os.getenv("MONTHLY_CONSOLIDATION_W_VALENCE", "0.10"))
_RETENTION_SPACING_SATURATION = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_SPACING_SATURATION", "5")))

# Phase 6: split novelty between static archive anchors and dynamic recent mean.
_NOVELTY_W_STATIC = float(os.getenv("MONTHLY_CONSOLIDATION_NOVELTY_W_STATIC", "0.6"))
_NOVELTY_W_DYNAMIC = float(os.getenv("MONTHLY_CONSOLIDATION_NOVELTY_W_DYNAMIC", "0.4"))
_DYNAMIC_ANCHOR_LIMIT = max(5, int(os.getenv("MONTHLY_CONSOLIDATION_DYNAMIC_ANCHOR_LIMIT", "40")))
_DYNAMIC_ANCHOR_DAYS = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_DYNAMIC_ANCHOR_DAYS", "14")))

# ── state persistence ─────────────────────────────────────────────────────────

def consolidation_state_path(user_id: str | None = None) -> Path:
    override = os.getenv("MONTHLY_CONSOLIDATION_STATE_PATH")
    if override:
        return Path(override).expanduser()
    return user_state_path("memory/monthly_consolidation_state.json", user_id or current_user_id())


def _load_state(user_id: str | None = None) -> dict:
    try:
        return json.loads(consolidation_state_path(user_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict, user_id: str | None = None) -> None:
    path = consolidation_state_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


# ── LLM plumbing ──────────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
CONSOLIDATION_LLM_TIMEOUT = float(os.getenv("MONTHLY_CONSOLIDATION_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "120")))

_LLM_CLIENT: OpenAI | None = None


def _get_llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "") or "not-needed", timeout=CONSOLIDATION_LLM_TIMEOUT)
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


# ── delete / provenance policy ────────────────────────────────────────────────

CONSOLIDATION_DELETE_DAILY_SUMMARIES = os.getenv("MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES", "0").lower() in {"1", "true", "yes", "on"}

JOURNAL_PROMOTE = os.getenv("MONTHLY_CONSOLIDATION_JOURNAL_PROMOTE", "1").lower() in {"1", "true", "yes", "on"}
JOURNAL_PROMOTE_K = max(0, int(os.getenv("MONTHLY_CONSOLIDATION_JOURNAL_PROMOTE_K", "4")))
DELETE_REQUIRE_COVERAGE = os.getenv("MONTHLY_CONSOLIDATION_DELETE_REQUIRE_COVERAGE", "1").lower() in {"1", "true", "yes", "on"}
DELETE_MIN_WRITTEN = max(0, int(os.getenv("MONTHLY_CONSOLIDATION_DELETE_MIN_WRITTEN", "1")))
DELETE_MIN_RATIO = float(os.getenv("MONTHLY_CONSOLIDATION_DELETE_MIN_RATIO", "0.15"))
HARD_SOURCE_PROVENANCE = os.getenv("MONTHLY_CONSOLIDATION_HARD_SOURCE_PROVENANCE", "1").lower() in {"1", "true", "yes", "on"}

# ── shared tag / keyword conventions ──────────────────────────────────────────

_DAILY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")
_MONTHLY_FACT_TAG_RE = re.compile(r"^\[\d{4}-\d{2}\]\s")

_MUST_KEEP_KEYWORDS = (
    "deadline", "birthday", "anniversary", "appointment", "hackathon",
    "interview", "lost ", "passport", "license", "wallet",
)

# ── month math ────────────────────────────────────────────────────────────────


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def target_month_for(now: datetime) -> tuple[datetime, datetime, str]:
    local_first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    target_end = _add_months(local_first, -CONSOLIDATION_KEEP_MONTHS)
    target_start = _add_months(target_end, -1)
    key = target_start.strftime("%Y-%m")
    return target_start, target_end, key


def _bounded_lines(items: list[str]) -> str:
    lines: list[str] = []
    total = 0
    for line in items:
        if total + len(line) > CONSOLIDATION_MAX_INPUT_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) or "- none"