"""TTL-backed JSONL record store shared by workflows.

Layout:
  <user_state>/agentic/workflows/<workflow_id>/records.jsonl

Each line is one JSON object. Optional field ``stored_at`` (ISO) is used
for age-based pruning (default retain 3 days).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def workflow_data_dir(workflow_id: str) -> Path:
    """Per-user data directory for a workflow id."""
    from system.userspace import user_state_dir

    d = user_state_dir() / "agentic" / "workflows" / workflow_id
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("workflow store: cannot create %s: %s", d, e)
    return d


def _records_path(workflow_id: str) -> Path:
    return workflow_data_dir(workflow_id) / "records.jsonl"


def append_record(workflow_id: str, record: dict[str, Any], *, stored_at: str | None = None) -> tuple[Path, bool]:
    """Append one record; stamps stored_at when missing.

    Returns:
        (path, success): Path to records file and whether write succeeded.
    """
    path = _records_path(workflow_id)
    row = dict(record)
    if not row.get("stored_at"):
        if stored_at:
            row["stored_at"] = stored_at
        else:
            try:
                from system.bioclock import local_now

                row["stored_at"] = local_now().isoformat()
            except Exception:
                row["stored_at"] = datetime.now().isoformat()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path, True
    except OSError as e:
        log.warning("workflow store: append failed %s: %s", path, e)
        return path, False


def load_records(workflow_id: str, *, days: int | None = None) -> list[dict[str, Any]]:
    """Load records; optional days filter keeps only recent rows."""
    path = _records_path(workflow_id)
    if not path.is_file():
        return []
    cutoff = None
    if days is not None and days > 0:
        try:
            from system.bioclock import local_now

            cutoff = local_now() - timedelta(days=days)
        except Exception:
            cutoff = datetime.now() - timedelta(days=days)

        # Normalize cutoff to UTC for consistent comparison
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        else:
            cutoff = cutoff.astimezone(timezone.utc)

    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if cutoff is not None:
                    stamp = row.get("stored_at") or row.get("checked_at")
                    if stamp:
                        try:
                            dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                            # Normalize record timestamp to UTC for comparison
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            else:
                                dt = dt.astimezone(timezone.utc)
                            if dt < cutoff:
                                continue
                        except (TypeError, ValueError):
                            pass
                out.append(row)
    except OSError as e:
        log.warning("workflow store: read failed %s: %s", path, e)
    return out


def prune_records(workflow_id: str, *, days: int = 3) -> tuple[int, bool]:
    """Rewrite store keeping only records within ``days``.

    Returns:
        (kept_count, success): Number of records kept and whether write succeeded.
    """
    kept = load_records(workflow_id, days=days)
    path = _records_path(workflow_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(kept), True
    except OSError as e:
        log.warning("workflow store: prune failed %s: %s", path, e)
        return 0, False
