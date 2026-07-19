"""SQLite-backed checkpointing for graph runs (agentic/schema.py).

A run_id identifies one invocation of execute_graph. If interrupted,
calling execute_graph again with the same run_id resumes from the last
completed node instead of restarting the whole graph.
"""
import json
import sqlite3
import threading
from pathlib import Path

_DB_PATH = Path(__file__).parent / "graph_checkpoints.db"
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
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


def save_node_result(run_id: str, seq: int, result) -> None:
    """Persist one NodeResult. Called right after results[node.id] = result."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO node_checkpoints "
                "(run_id, node_id, tool, ok, content, args, error_type, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, result.node_id, result.tool, int(result.ok),
                 result.content, json.dumps(result.args), result.error_type, seq),
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