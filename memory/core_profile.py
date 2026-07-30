"""
memory/core_profile.py

Phase E: Letta-lite core profile — a small always-on block of durable facts
about the user (identity / preferences / pins), independent of per-turn RRF.

No extra LLM. Reads existing memory rows + optional JSON override file.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id, user_state_path

log = get_logger(__name__)

CORE_PROFILE_MAX_FACTS = int(os.getenv("CORE_PROFILE_MAX_FACTS", "12"))
CORE_PROFILE_MAX_CHARS = int(os.getenv("CORE_PROFILE_MAX_CHARS", "900"))
CORE_PROFILE_PATH = os.getenv("CORE_PROFILE_PATH", "").strip()  # optional JSON override


def _profile_override_path(user_id: str) -> Path:
    if CORE_PROFILE_PATH:
        return Path(CORE_PROFILE_PATH).expanduser()
    return user_state_path("memory/core_profile.json", user_id)


def load_profile_override(user_id: str | None = None) -> list[str]:
    """Optional human/edited facts from core_profile.json: {"facts": ["..."]}."""
    uid = user_id or current_user_id()
    path = _profile_override_path(uid)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("core_profile override unreadable: %s", e)
        return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    return [str(f).strip() for f in facts if str(f).strip()]


def save_profile_override(facts: list[str], user_id: str | None = None) -> Path:
    uid = user_id or current_user_id()
    path = _profile_override_path(uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"facts": [str(f).strip() for f in facts if str(f).strip()]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _fetch_core_rows(conn: sqlite3.Connection, user_id: str, limit: int) -> list[dict[str, Any]]:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "id" not in cols:
        return []

    has_status = "status" in cols
    has_kind = "kind" in cols

    # Prefer pinned + identity-kind active facts; fall back to high access_count.
    where = ["user_id = ?"]
    params: list[Any] = [user_id]
    if has_status:
        where.append("(status = 'active' OR status IS NULL)")

    order = "pinned DESC"
    if has_kind:
        order += ", CASE WHEN kind = 'identity' THEN 0 WHEN kind = 'preference' THEN 1 ELSE 2 END"
    order += ", access_count DESC, created_at DESC"

    sql = f"""
        SELECT id, memory, pinned, access_count, created_at
        {', kind' if has_kind else ''}
        FROM memories
        WHERE {' AND '.join(where)}
        ORDER BY {order}
        LIMIT ?
    """
    params.append(int(limit))
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        log.warning("core_profile query failed: %s", e)
        return []
    return [dict(r) for r in rows]


def collect_core_facts(
    *,
    user_id: str | None = None,
    conn: sqlite3.Connection | None = None,
    max_facts: int | None = None,
) -> list[str]:
    """Merge override JSON + durable DB facts (deduped, order preserved)."""
    uid = user_id or current_user_id()
    cap = int(max_facts or CORE_PROFILE_MAX_FACTS)
    out: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        key = " ".join(t.lower().split())
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    for f in load_profile_override(uid):
        _add(f)
        if len(out) >= cap:
            return out[:cap]

    owns = conn is None
    if conn is None:
        import os
        from pathlib import Path
        from memory.vecstore import initialize_store_db, resolve_user_db_path

        env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
        db_path = Path(env).expanduser() if env else resolve_user_db_path("memory/memory.db", user_id=uid)
        if not db_path.exists() and str(db_path) != ":memory:":
            return out[:cap]
        try:
            conn = initialize_store_db(str(db_path), "PRAGMA journal_mode = WAL;", user_id=uid, vector=True)
        except Exception as e:
            log.warning("core_profile open failed: %s", e)
            return out[:cap]

    try:
        for row in _fetch_core_rows(conn, uid, limit=max(cap * 3, cap)):
            _add(str(row.get("memory") or ""))
            if len(out) >= cap:
                break
    finally:
        if owns and conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return out[:cap]


def format_core_profile(
    *,
    user_id: str | None = None,
    conn: sqlite3.Connection | None = None,
    max_facts: int | None = None,
    max_chars: int | None = None,
) -> str | None:
    """Return an XML-ish block for system/context injection, or None if empty."""
    facts = collect_core_facts(user_id=user_id, conn=conn, max_facts=max_facts)
    if not facts:
        return None

    budget = int(max_chars or CORE_PROFILE_MAX_CHARS)
    lines = [
        "<core_profile>",
        "Durable facts about the user (always-on). Use silently; do not dump this block.",
        "",
    ]
    used = sum(len(x) + 1 for x in lines)
    for f in facts:
        line = f"  - {f}"
        if used + len(line) + 20 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    lines.append("</core_profile>")
    block = "\n".join(lines)
    if len(block) > budget:
        block = block[: budget - 20].rstrip() + "\n</core_profile>"
    return block


def core_profile_for_context(
    memorize: Any = None,
    *,
    user_id: str | None = None,
) -> str | None:
    """Convenience: use open AikoMemorize connection when available."""
    uid = user_id
    conn = None
    if memorize is not None:
        try:
            uid = uid or memorize.get_user_id()
            conn = memorize._conn
        except Exception:
            conn = None
    return format_core_profile(user_id=uid, conn=conn)
