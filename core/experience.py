"""
core/experience.py

Small JSONL helpers for Aiko's daily experience log.

This log is intentionally separate from persistent semantic memory: it keeps the
raw-ish turn history needed for factual daily summaries, while memory keeps
retrievable long-term facts. No LLM calls live here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EXPERIENCE_LOG = Path.home() / ".aiko" / "daily_experience.jsonl"
EXPERIENCE_LOG_PATH = Path(os.getenv("AIKO_EXPERIENCE_LOG_PATH", str(DEFAULT_EXPERIENCE_LOG)))
MAX_TURN_CHARS = int(os.getenv("AIKO_EXPERIENCE_TURN_MAX_CHARS", "4000"))


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def append_chat_turn(user_text: str, assistant_text: str, user_id: str, at: datetime | None = None) -> None:
    """Append one completed conversation turn to the daily experience log."""
    EXPERIENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = _coerce_utc(at or datetime.now(timezone.utc)).isoformat()
    record = {
        "created_at": ts,
        "user_id": user_id,
        "user": (user_text or "")[:MAX_TURN_CHARS],
        "assistant": (assistant_text or "")[:MAX_TURN_CHARS],
    }
    with EXPERIENCE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chat_turns(start: datetime, end: datetime, user_id: str | None = None) -> list[dict[str, Any]]:
    """Load chat turns whose created_at timestamp falls in [start, end)."""
    start = _coerce_utc(start)
    end = _coerce_utc(end)
    if not EXPERIENCE_LOG_PATH.exists():
        return []

    turns: list[dict[str, Any]] = []
    with EXPERIENCE_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                created_at = datetime.fromisoformat(str(record.get("created_at", "")).replace("Z", "+00:00"))
            except Exception:
                continue
            created_at = _coerce_utc(created_at)
            if created_at >= end:
                break
            if created_at < start:
                continue
            if user_id is not None and record.get("user_id") != user_id:
                continue
            turns.append(record)
    return turns
