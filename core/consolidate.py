"""
core/consolidate.py
Monthly memory consolidation.

Runs on/after the first day of a month and consolidates the month before the
most recent full month. Example: on July 1, keep June intact and summarize May.
The summary is inserted as a pinned raw memory, then unpinned detailed memories
from the consolidated month can be deleted to reduce vector DB size.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from core.experience import load_chat_turns
from core.log import get_logger

log = get_logger(__name__)

MONTHLY_CONSOLIDATION_ENABLED = os.getenv("MONTHLY_CONSOLIDATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
MONTHLY_CONSOLIDATION_KEEP_MONTHS = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_KEEP_MONTHS", "1")))
MONTHLY_CONSOLIDATION_SOURCE = os.getenv("MONTHLY_CONSOLIDATION_SOURCE", "experience").strip().lower()
MONTHLY_CONSOLIDATION_CHUNK_MEMS = max(5, int(os.getenv("MONTHLY_CONSOLIDATION_CHUNK_MEMS", "25")))
MONTHLY_CONSOLIDATION_CHUNK_TURNS = max(10, int(os.getenv("MONTHLY_CONSOLIDATION_CHUNK_TURNS", "40")))
MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS = max(1000, int(os.getenv("MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS", "6000")))
MONTHLY_CONSOLIDATION_MIN_MEMS = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_MIN_MEMS", "5")))
MONTHLY_CONSOLIDATION_DELETE_ORIGINALS = os.getenv("MONTHLY_CONSOLIDATION_DELETE_ORIGINALS", "1").lower() in {"1", "true", "yes", "on"}
MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES = os.getenv("MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES", "1").lower() in {"1", "true", "yes", "on"}
MONTHLY_CONSOLIDATION_STATE_PATH = Path(os.getenv("MONTHLY_CONSOLIDATION_STATE_PATH", str(Path.home() / ".aiko" / "monthly_consolidation_state.json")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
MONTHLY_CONSOLIDATION_LLM_TIMEOUT = float(os.getenv("MONTHLY_CONSOLIDATION_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "120")))


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def target_month_for(now: datetime) -> tuple[datetime, datetime, str]:
    """Return (start, end, key) for the month ready to consolidate."""
    local_first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    target_end = _add_months(local_first, -MONTHLY_CONSOLIDATION_KEEP_MONTHS)
    target_start = _add_months(target_end, -1)
    key = target_start.strftime("%Y-%m")
    return target_start, target_end, key


def _load_state() -> dict:
    try:
        return json.loads(MONTHLY_CONSOLIDATION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    MONTHLY_CONSOLIDATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONTHLY_CONSOLIDATION_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _chat(prompt: str, max_tokens: int = 700) -> str:
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed", timeout=MONTHLY_CONSOLIDATION_LLM_TIMEOUT)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()


def _bounded_lines(items: list[str]) -> str:
    lines: list[str] = []
    total = 0
    for line in items:
        if total + len(line) > MONTHLY_CONSOLIDATION_MAX_INPUT_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _memory_lines(memories: list[dict]) -> str:
    return _bounded_lines([
        f"- {m.get('created_at', '')}: {(m.get('memory') or m.get('text') or '').strip()}"
        for m in memories
        if (m.get("memory") or m.get("text") or "").strip()
    ])


def _turn_lines(turns: list[dict]) -> str:
    return _bounded_lines([
        f"- {t.get('created_at', '')}\n  User: {(t.get('user') or '').strip()}\n  Aiko: {(t.get('assistant') or '').strip()}"
        for t in turns
        if (t.get("user") or t.get("assistant"))
    ])


def _summarize_memory_chunk(month_key: str, memories: list[dict], idx: int, total: int) -> str:
    prompt = (
        "Summarize these long-term memory facts for Aiko's monthly memory consolidation. "
        "Keep stable preferences, identity facts, projects, deadlines, relationships, and recurring patterns. "
        "Drop trivial chatter, duplicates, and implementation details about the memory system. "
        "Use compact bullet points only. Do not invent facts.\n\n"
        f"Month: {month_key}\nChunk: {idx}/{total}\n\n"
        f"Memories:\n{_memory_lines(memories)}"
    )
    return _chat(prompt, max_tokens=500)


def _summarize_turn_chunk(month_key: str, turns: list[dict], idx: int, total: int) -> str:
    prompt = (
        "Summarize these raw Aiko/Oppa daily experience turns for monthly consolidation. "
        "Keep stable facts, preferences, projects, emotional beats, deadlines, repeated themes, and important events. "
        "Drop filler, greetings, transient wording, and duplicate details. Do not invent facts. "
        "Use compact bullet points only.\n\n"
        f"Month: {month_key}\nChunk: {idx}/{total}\n\n"
        f"Experience turns:\n{_turn_lines(turns)}"
    )
    return _chat(prompt, max_tokens=550)


def _final_summary(month_key: str, chunk_summaries: list[str]) -> str:
    prompt = (
        "Merge these chunk summaries into ONE durable pinned memory for Aiko. "
        "Write concise bullet points grouped by theme. Preserve only facts worth keeping long-term. "
        "Do not mention vectors, databases, consolidation, chunks, or internal processes. "
        "Do not invent facts.\n\n"
        f"Month: {month_key}\n\n"
        + "\n\n".join(f"Chunk summary {i+1}:\n{s}" for i, s in enumerate(chunk_summaries))
    )
    return _chat(prompt, max_tokens=900)


def maybe_run_monthly_consolidation(memorize, now: datetime | None = None) -> dict:
    """Run monthly consolidation if enabled, due, and not already done."""
    if not MONTHLY_CONSOLIDATION_ENABLED:
        return {"ran": False, "reason": "disabled"}
    now = now or datetime.now()
    if now.day != 1:
        return {"ran": False, "reason": "not_first_day"}

    start, end, month_key = target_month_for(now)
    state = _load_state()
    if state.get("last_consolidated_month") == month_key:
        return {"ran": False, "reason": "already_done", "month": month_key}

    start_utc = start.replace(tzinfo=timezone.utc)
    end_utc = end.replace(tzinfo=timezone.utc)
    memories = memorize.get_between(start_utc, end_utc)
    turns = load_chat_turns(start_utc, end_utc) if MONTHLY_CONSOLIDATION_SOURCE == "experience" else []

    if turns:
        chunks = [turns[i:i + MONTHLY_CONSOLIDATION_CHUNK_TURNS] for i in range(0, len(turns), MONTHLY_CONSOLIDATION_CHUNK_TURNS)]
        chunk_summaries = [_summarize_turn_chunk(month_key, chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
        source_count = len(turns)
        source = "experience"
    else:
        if len(memories) < MONTHLY_CONSOLIDATION_MIN_MEMS:
            state["last_consolidated_month"] = month_key
            _save_state(state)
            return {"ran": False, "reason": "too_few_memories", "month": month_key, "count": len(memories)}
        chunks = [memories[i:i + MONTHLY_CONSOLIDATION_CHUNK_MEMS] for i in range(0, len(memories), MONTHLY_CONSOLIDATION_CHUNK_MEMS)]
        chunk_summaries = [_summarize_memory_chunk(month_key, chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
        source_count = len(memories)
        source = "memory"
    summary = _final_summary(month_key, chunk_summaries)
    if not summary:
        return {"ran": False, "reason": "empty_summary", "month": month_key, "count": source_count, "source": source}

    summary_text = f"Monthly memory summary for {month_key}:\n{summary}"
    summary_id = memorize.add_raw(summary_text, pinned=True)
    if not summary_id:
        return {"ran": False, "reason": "summary_insert_failed", "month": month_key, "count": len(memories)}

    deleted = 0
    daily_deleted = 0
    if MONTHLY_CONSOLIDATION_DELETE_ORIGINALS:
        for memory in memories:
            mem_id = memory.get("id")
            text = (memory.get("memory") or "")
            is_daily_summary = text.startswith("Daily experience summary for ")
            if mem_id and not memorize._is_pinned(mem_id):
                memorize.delete(mem_id)
                deleted += 1
            elif mem_id and is_daily_summary and MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES:
                memorize.delete(mem_id)
                daily_deleted += 1

    state["last_consolidated_month"] = month_key
    state["last_summary_id"] = summary_id
    _save_state(state)
    log.info("Monthly consolidation complete: month=%s source=%s count=%s deleted=%s daily_deleted=%s summary_id=%s", month_key, source, source_count, deleted, daily_deleted, summary_id)
    return {"ran": True, "month": month_key, "source": source, "count": source_count, "deleted": deleted, "daily_deleted": daily_deleted, "summary_id": summary_id}
