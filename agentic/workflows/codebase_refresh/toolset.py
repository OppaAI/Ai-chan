"""
agentic/workflows/codebase_refresh/toolset.py

Nightly codebase refresh — rebuilds <USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db
incrementally (SHA1, prune stale) at 22:00. Jetson-optimized: batched 32, WAL, cosine.
"""
from __future__ import annotations

import json
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

def refresh_codebase(*, state=None, **kwargs) -> str:
    """Graph node: ingest entire repo into per-user codebase.db."""
    try:
        from cognition.knowledge.codebase import ingest_codebase
        from system.userspace import current_user_id
        uid = current_user_id()
        # state may carry user_id override
        if state is not None and hasattr(state, "data"):
            uid = state.data.get("user_id") or uid
        res = ingest_codebase(user_id=uid, force=False)
        log.info("codebase_refresh: %s", res)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        log.warning("codebase_refresh failed: %s", e)
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

def codebase_refresh_status(*, state=None, **kwargs) -> str:
    """Graph node: check if codebase.db exists and stats."""
    try:
        from pathlib import Path
        from system.userspace import user_state_path, current_user_id
        uid = current_user_id()
        p = user_state_path("knowledge/codebase.db", user_id=uid)
        if not p.exists():
            return json.dumps({"ok": True, "exists": False, "path": str(p)})
        import sqlite3
        conn = __import__("sqlite3").connect(str(p))
        try:
            docs = conn.execute("SELECT COUNT(*) FROM codebase_docs").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM codebase_chunks").fetchone()[0]
        finally:
            conn.close()
        return json.dumps({"ok": True, "exists": True, "path": str(p), "docs": docs, "chunks": chunks, "size_bytes": p.stat().st_size})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
