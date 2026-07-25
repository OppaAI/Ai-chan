"""SQLite-backed checkpointing for graph runs (agentic/schema.py).

A run_id identifies one invocation of execute_graph. If interrupted,
calling execute_graph again with the same run_id resumes from the last
completed node instead of restarting the whole graph.
"""
import json
import shutil
import sqlite3
import threading
from pathlib import Path

from system.log import get_logger
from system.userspace import current_user_id, user_state_dir

log = get_logger(__name__)

_DB_PATH = user_state_dir(current_user_id()) / "agentic" / "graph_checkpoints.db"
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    _ensure_migrated()
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_checkpoints (
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            tool TEXT NOT NULL,
            ok INTEGER NOT NULL,
            content TEXT NOT NULL,
            args TEXT NOT NULL,
            error_type TEXT,
            seq INTEGER NOT NULL,
            PRIMARY KEY (run_id, node_id)
        )
    """)
    return conn


def _ensure_migrated() -> None:
    """Ensure checkpoint DB is in user-space location, migrating if needed."""
    old_path = Path(__file__).parent / "graph_checkpoints.db"
    new_path = _DB_PATH
    
    if not old_path.exists():
        return
    
    if new_path.exists():
        return
    
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
        log.info("[checkpoint] Migrated checkpoint DB from %s to %s", old_path, new_path)
        
        # Remove old file after successful copy
        old_path.unlink()
        log.info("[checkpoint] Removed old checkpoint DB at %s", old_path)
    except Exception as e:
        log.error("[checkpoint] Failed to migrate checkpoint DB: %s", e)


def save_node_result(run_id: str, seq: int, result) -> None:
    """Persist one NodeResult. Called right after results[node.id] = result."""
    safe_args = {
        k: v for k, v in result.args.items()
        if k not in {"embedder", "client", "model"}
    }
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO node_checkpoints "
                "(run_id, node_id, tool, ok, content, args, error_type, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, result.node_id, result.tool, int(result.ok),
                 result.content, json.dumps(safe_args), result.error_type, seq),
            )
            conn.commit()
        finally:
            conn.close()


def load_checkpoint(run_id: str, node_result_cls) -> list:
    """Returns completed NodeResults for run_id, in original order. Empty if none."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT node_id, tool, ok, content, args, error_type FROM node_checkpoints "
                "WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        finally:
            conn.close()
    return [
        node_result_cls(node_id=r[0], tool=r[1], ok=bool(r[2]), content=r[3],
                         args=json.loads(r[4]), error_type=r[5])
        for r in rows
    ]


def clear_checkpoint(run_id: str) -> None:
    """Call after a run completes successfully, so the table doesn't grow forever."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM node_checkpoints WHERE run_id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()