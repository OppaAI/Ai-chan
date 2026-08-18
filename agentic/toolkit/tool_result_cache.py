"""Unified on-disk tool-result cache for agentic DAG workflows.

Full tool outputs land here by default (JSONL). Only cache_select() output
should be injected into LLM synthesis. Same schema for email/RSS/web/tools.
See docs/WORKFLOW_SPEC.md.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from system.log import get_logger
from system.userspace import user_state_dir

log = get_logger(__name__)

CACHE_ROOT_ENV = "AIKO_TOOL_RESULT_CACHE_ROOT"
DEFAULT_RETENTION_RUNS = int(os.getenv("AIKO_CACHE_RETENTION_RUNS", "14"))
MAX_BODY_CHARS = int(os.getenv("AIKO_CACHE_MAX_BODY_CHARS", "20000"))
SELECT_DEFAULT_LIMIT = int(os.getenv("AIKO_CACHE_SELECT_LIMIT", "12"))
SELECT_TITLE_CHARS = int(os.getenv("AIKO_CACHE_SELECT_TITLE_CHARS", "120"))
SELECT_BODY_CHARS = int(os.getenv("AIKO_CACHE_SELECT_BODY_CHARS", "400"))

_SOURCE_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_WORKFLOW_RE = re.compile(r"^[a-z0-9_][a-z0-9_\-]{0,63}$")


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _cache_root() -> Path:
    override = os.getenv(CACHE_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return user_state_dir() / "agentic" / "cache"


def _validate_workflow(workflow: str) -> str:
    w = (workflow or "").strip().lower().replace(" ", "_")
    if not _WORKFLOW_RE.match(w):
        raise ValueError(f"invalid workflow id: {workflow!r}")
    return w


def _validate_source(source: str) -> str:
    s = (source or "tool").strip().lower()
    if not _SOURCE_RE.match(s):
        raise ValueError(f"invalid source: {source!r}")
    return s


def workflow_dir(workflow: str) -> Path:
    d = _cache_root() / _validate_workflow(workflow)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _truncate(text: str, n: int) -> str:
    t = text or ""
    if len(t) <= n:
        return t
    return t[: max(0, n - 15)] + "\n\u2026[truncated]"


def normalize_record(
    item: Any,
    *,
    workflow: str,
    source: str,
    run_id: str,
    fetched_at: str | None = None,
) -> dict[str, Any]:
    now = fetched_at or datetime.now(timezone.utc).isoformat()
    if isinstance(item, str):
        return {
            "id": str(uuid.uuid4()),
            "workflow": workflow,
            "source": source,
            "run_id": run_id,
            "fetched_at": now,
            "title": _truncate(item.split("\n", 1)[0], SELECT_TITLE_CHARS),
            "body": _truncate(item, MAX_BODY_CHARS),
            "url": "",
            "score": 0.0,
            "matched": True,
            "meta": {},
        }
    if not isinstance(item, dict):
        item = {"body": str(item)}

    title = str(item.get("title") or item.get("subject") or item.get("name") or "")
    body = str(
        item.get("body")
        or item.get("content")
        or item.get("summary")
        or item.get("description")
        or item.get("text")
        or ""
    )
    url = str(item.get("url") or item.get("link") or item.get("href") or "")
    score = item.get("score")
    try:
        score_f = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        score_f = 0.0
    matched = item.get("matched")
    if matched is None:
        matched = True
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    for k in ("from", "source_feed", "published", "company", "location", "keywords"):
        if k in item and k not in meta:
            meta[k] = item[k]

    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "workflow": workflow,
        "source": source,
        "run_id": run_id,
        "fetched_at": str(item.get("fetched_at") or now),
        "title": _truncate(title, 500),
        "body": _truncate(body, MAX_BODY_CHARS),
        "url": url[:2000],
        "score": score_f,
        "matched": bool(matched),
        "meta": meta,
    }


def cache_write(
    items: Iterable[Any] | Any,
    *,
    workflow: str,
    source: str = "tool",
    run_id: str | None = None,
    state: Any = None,
    from_state: str | None = None,
) -> dict[str, Any]:
    workflow = _validate_workflow(workflow)
    source = _validate_source(source)
    run_id = run_id or _utc_run_id()

    if from_state and state is not None:
        raw = state.get(from_state) if hasattr(state, "get") else None
        if raw is None and isinstance(getattr(state, "data", None), dict):
            raw = state.data.get(from_state)
        items = raw if raw is not None else items

    if items is None:
        items = []
    if isinstance(items, dict):
        if all(isinstance(v, (dict, str)) for v in items.values()):
            seq = list(items.values())
        else:
            seq = [items]
    elif isinstance(items, (str, bytes)):
        seq = [items]
    else:
        try:
            seq = list(items)
        except TypeError:
            seq = [items]

    records = [
        normalize_record(it, workflow=workflow, source=source, run_id=run_id)
        for it in seq
    ]
    path = workflow_dir(workflow) / f"{run_id}_{source}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    index_path = workflow_dir(workflow) / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "source": source,
                    "path": str(path),
                    "count": len(records),
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    log.info(
        "cache_write workflow=%s source=%s run_id=%s n=%d path=%s",
        workflow, source, run_id, len(records), path,
    )
    out = {
        "ok": True,
        "workflow": workflow,
        "source": source,
        "run_id": run_id,
        "path": str(path),
        "count": len(records),
        "ids": [r["id"] for r in records[:50]],
    }
    if state is not None and hasattr(state, "set"):
        state.set(f"cache_{source}_path", str(path))
        state.set(f"cache_{source}_run_id", run_id)
        state.set("cache_last_run_id", run_id)
    return out


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def cache_read(
    *,
    workflow: str,
    source: str | None = None,
    run_id: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    workflow = _validate_workflow(workflow)
    d = workflow_dir(workflow)
    files = sorted(d.glob("*.jsonl"))
    files = [p for p in files if p.name != "index.jsonl"]
    if run_id:
        files = [p for p in files if p.name.startswith(run_id)]
    if source:
        source = _validate_source(source)
        files = [p for p in files if p.stem.endswith(f"_{source}")]

    records: list[dict[str, Any]] = []
    for p in files:
        records.extend(_iter_jsonl(p))
        if len(records) >= limit:
            records = records[:limit]
            break
    return {
        "ok": True,
        "workflow": workflow,
        "source": source,
        "run_id": run_id,
        "count": len(records),
        "records": records,
    }


def cache_select(
    *,
    workflow: str,
    source: str | None = None,
    run_id: str | None = None,
    keywords: list[str] | str | None = None,
    matched_only: bool = True,
    limit: int = SELECT_DEFAULT_LIMIT,
    state: Any = None,
    to_state: str = "selection",
) -> dict[str, Any]:
    workflow = _validate_workflow(workflow)
    if run_id is None and state is not None and hasattr(state, "get"):
        run_id = state.get("cache_last_run_id") or state.get("run_id")

    raw = cache_read(workflow=workflow, source=source, run_id=run_id, limit=2000)
    records = list(raw.get("records") or [])

    if matched_only:
        records = [r for r in records if r.get("matched", True)]

    kw: list[str] = []
    if isinstance(keywords, str):
        kw = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    elif keywords:
        kw = [str(k).strip().lower() for k in keywords if str(k).strip()]

    def rank(r: dict[str, Any]) -> tuple:
        text = f"{r.get('title', '')} {r.get('body', '')}".lower()
        hits = sum(1 for k in kw if k in text) if kw else 0
        return (hits, float(r.get("score") or 0.0), r.get("fetched_at") or "")

    records.sort(key=rank, reverse=True)
    selected = records[: max(1, int(limit))]

    compact = []
    for r in selected:
        compact.append(
            {
                "id": r.get("id"),
                "source": r.get("source"),
                "title": _truncate(str(r.get("title") or ""), SELECT_TITLE_CHARS),
                "body": _truncate(str(r.get("body") or ""), SELECT_BODY_CHARS),
                "url": r.get("url") or "",
                "score": r.get("score"),
                "meta": {
                    k: r.get("meta", {}).get(k)
                    for k in ("company", "location", "from", "source_feed")
                    if isinstance(r.get("meta"), dict) and r.get("meta", {}).get(k)
                },
            }
        )

    approx_chars = sum(len(x.get("title", "")) + len(x.get("body", "")) for x in compact)

    if state is not None and hasattr(state, "set"):
        state.set(to_state, compact)
        state.set(f"{to_state}_count", len(compact))
        state.set(f"{to_state}_chars", approx_chars)

    log.info(
        "cache_select workflow=%s source=%s run_id=%s n=%d chars\u2248%d",
        workflow, source, run_id, len(compact), approx_chars,
    )
    return {
        "ok": True,
        "workflow": workflow,
        "source": source,
        "run_id": run_id,
        "count": len(compact),
        "approx_chars": approx_chars,
        "selection": compact,
        "to_state": to_state,
    }


def cache_gc(
    *,
    workflow: str,
    keep_runs: int = DEFAULT_RETENTION_RUNS,
) -> dict[str, Any]:
    workflow = _validate_workflow(workflow)
    d = workflow_dir(workflow)
    files = [p for p in d.glob("*.jsonl") if p.name != "index.jsonl"]
    by_run: dict[str, list[Path]] = {}
    for p in files:
        stem = p.stem
        if "_" not in stem:
            continue
        run_id, _src = stem.rsplit("_", 1)
        by_run.setdefault(run_id, []).append(p)

    run_ids = sorted(by_run.keys(), reverse=True)
    drop = run_ids[max(0, int(keep_runs)) :]
    removed = 0
    for rid in drop:
        for p in by_run[rid]:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                log.warning("cache_gc unlink %s: %s", p, e)
    return {
        "ok": True,
        "workflow": workflow,
        "kept_runs": run_ids[:keep_runs],
        "dropped_runs": drop,
        "files_removed": removed,
    }


__all__ = [
    "cache_write",
    "cache_read",
    "cache_select",
    "cache_gc",
    "normalize_record",
    "workflow_dir",
    "SELECT_DEFAULT_LIMIT",
    "DEFAULT_RETENTION_RUNS",
]
