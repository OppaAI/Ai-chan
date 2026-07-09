"""Persistent experience store for Aiko's completed agentic task runs.

Experience is not user memory and not wiki/knowledge. It is Aiko's procedural
trace of what she tried: goal, ordered tools, outcomes, verification score, and
a short result excerpt. Records do not decay or get forgotten; they are capped
only to prevent unbounded growth/noise. Because tool arguments can contain
incidental sensitive data, only argument keys and sanitized excerpts are stored,
and the SQLite DB uses the same optional SQLCipher encryption path as memory.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import sqlite_vec

from core import reason
from core.rag import fts_or_query, rrf_score
from core.log import get_logger
from core.secure import connect_sqlite
from core.userspace import current_user_id, user_state_path

log = get_logger(__name__)

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))
EXPERIENCE_DB_PATH = os.getenv("EXPERIENCE_DB_PATH", "experience/experience.db")
EXPERIENCE_QUERY_INSTRUCT = os.getenv("EXPERIENCE_QUERY_INSTRUCT", "Retrieve similar past agentic task runs").strip()
EXPERIENCE_RRF_K = int(os.getenv("EXPERIENCE_RRF_K", "60"))
EXPERIENCE_KNN_LIMIT = int(os.getenv("EXPERIENCE_KNN_LIMIT", "20"))
EXPERIENCE_FTS_LIMIT = int(os.getenv("EXPERIENCE_FTS_LIMIT", "20"))
EXPERIENCE_RECALL_SCORE_THRESHOLD = float(os.getenv("EXPERIENCE_RECALL_SCORE_THRESHOLD", "0.012"))
EXPERIENCE_MAX_ROWS = int(os.getenv("EXPERIENCE_MAX_ROWS", "5000"))
EXPERIENCE_CONTEXT_CHARS = int(os.getenv("EXPERIENCE_CONTEXT_CHARS", "2500"))

_SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)")

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiences (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    goal           TEXT NOT NULL,
    record_text    TEXT NOT NULL,
    steps_json     TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    score          REAL NOT NULL,
    answer_excerpt TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiences_user ON experiences(user_id);
CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
    record_text,
    id UNINDEXED,
    content='experiences',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS experiences_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);

CREATE TRIGGER IF NOT EXISTS experiences_ai AFTER INSERT ON experiences BEGIN
    INSERT INTO experiences_fts(rowid, record_text, id) VALUES (new.rowid, new.record_text, new.id);
END;

CREATE TRIGGER IF NOT EXISTS experiences_ad AFTER DELETE ON experiences BEGIN
    INSERT INTO experiences_fts(experiences_fts, rowid, record_text, id)
    VALUES ('delete', old.rowid, old.record_text, old.id);
    DELETE FROM experiences_vec WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS experiences_au AFTER UPDATE OF record_text ON experiences BEGIN
    INSERT INTO experiences_fts(experiences_fts, rowid, record_text, id)
    VALUES ('delete', old.rowid, old.record_text, old.id);
    INSERT INTO experiences_fts(rowid, record_text, id) VALUES (new.rowid, new.record_text, new.id);
END;
""".format(dims=EMBED_DIMS)


def _db_path(user_id: str | None = None) -> Path:
    path = Path(EXPERIENCE_DB_PATH).expanduser()
    if path.is_absolute():
        return path
    return user_state_path(str(path), user_id)


def _connect(user_id: str | None = None) -> sqlite3.Connection:
    uid = user_id or current_user_id()
    conn = connect_sqlite(_db_path(uid), user_id=uid)
    sqlite_vec.load(conn)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def _sanitize(text: str, max_chars: int = 500) -> str:
    t = _SECRET_RE.sub(r"\1\2[redacted]", text or "")
    t = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", t)
    return re.sub(r"\s+", " ", t).strip()[:max_chars]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()




@dataclass
class ExperienceStep:
    tool: str
    ok: bool
    error_type: str | None = None
    arg_keys: list[str] = field(default_factory=list)


def record_experience(owner, goal: str, steps: list[dict], final_answer: str, verified_ok: bool, score: float, embedder=None) -> str | None:
    uid = current_user_id()
    exp_steps = [
        ExperienceStep(
            tool=str(s.get("tool", "unknown")),
            ok=bool(s.get("ok")),
            error_type=s.get("error_type"),
            arg_keys=sorted((s.get("args") or {}).keys()),
        )
        for s in steps
    ]
    outcome = "ok" if verified_ok else ("partial" if any(s.ok for s in exp_steps) else "failed")
    step_text = ", ".join(f"{s.tool}({'+'.join(s.arg_keys) or '-'})[{'ok' if s.ok else s.error_type or 'fail'}]" for s in exp_steps)
    record_text = f"Goal: {_sanitize(goal, 700)}\nSteps: {step_text}\nOutcome: {outcome}\nScore: {float(score):.2f}"
    row_id = str(uuid.uuid4())
    conn = _connect(uid)
    try:
        conn.execute(
            "INSERT INTO experiences(id,user_id,goal,record_text,steps_json,outcome,score,answer_excerpt,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (row_id, uid, _sanitize(goal, 700), record_text, json.dumps([s.__dict__ for s in exp_steps], ensure_ascii=False), outcome, float(score), _sanitize(final_answer, 500), _now()),
        )
        if embedder is not None:
            vec = embedder.embed_query(record_text, instruct=EXPERIENCE_QUERY_INSTRUCT)
            conn.execute("INSERT INTO experiences_vec(id, embedding) VALUES(?, ?)", (row_id, sqlite_vec.serialize_float32(vec)))
        conn.commit()
        _prune(conn, uid)
        return row_id
    except Exception as exc:
        conn.rollback()
        log.warning("record_experience failed (non-fatal): %s", exc)
        return None
    finally:
        conn.close()


def _prune(conn: sqlite3.Connection, uid: str) -> None:
    total = conn.execute("SELECT COUNT(*) AS n FROM experiences WHERE user_id=?", (uid,)).fetchone()["n"]
    excess = max(0, int(total) - EXPERIENCE_MAX_ROWS)
    if not excess:
        return
    rows = conn.execute(
        "SELECT id FROM experiences WHERE user_id=? ORDER BY score ASC, created_at ASC LIMIT ?",
        (uid, excess),
    ).fetchall()
    for row in rows:
        conn.execute("DELETE FROM experiences WHERE id=?", (row["id"],))
    conn.commit()


def _knn(conn: sqlite3.Connection, query: str, embedder, uid: str, limit: int) -> list[sqlite3.Row]:
    if embedder is None:
        return []
    blob = sqlite_vec.serialize_float32(embedder.embed_query(query, instruct=EXPERIENCE_QUERY_INSTRUCT))
    return conn.execute(
        """
        SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
        FROM experiences_vec v
        JOIN experiences e ON e.id = v.id
        WHERE e.user_id=?
        ORDER BY dist ASC
        LIMIT ?
        """,
        (blob, uid, limit),
    ).fetchall()


def _fts(conn: sqlite3.Connection, query: str, uid: str, limit: int) -> list[sqlite3.Row]:
    fts = fts_or_query(query)
    if not fts:
        return []
    return conn.execute(
        """
        SELECT f.id
        FROM experiences_fts f
        JOIN experiences e ON e.id = f.id
        WHERE experiences_fts MATCH ? AND e.user_id=?
        ORDER BY rank
        LIMIT ?
        """,
        (fts, uid, limit),
    ).fetchall()


def search_experience(query: str, limit: int = 3, embedder=None) -> list[dict]:
    uid = current_user_id()
    conn = _connect(uid)
    try:
        rank_knn = {row["id"]: i + 1 for i, row in enumerate(_knn(conn, query, embedder, uid, EXPERIENCE_KNN_LIMIT))}
        rank_fts = {row["id"]: i + 1 for i, row in enumerate(_fts(conn, query, uid, EXPERIENCE_FTS_LIMIT))}
        ids = set(rank_knn) | set(rank_fts)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT * FROM experiences WHERE id IN ({placeholders})", list(ids)).fetchall()
        by_id = {row["id"]: row for row in rows}
        scored = []
        for eid in ids:
            score = rrf_score(eid, rank_knn, rank_fts, k=EXPERIENCE_RRF_K)
            if score >= EXPERIENCE_RECALL_SCORE_THRESHOLD and eid in by_id:
                scored.append((score, eid))
        scored.sort(key=lambda pair: (-pair[0], by_id[pair[1]]["created_at"]))
        return [dict(by_id[eid]) | {"recall_score": score} for score, eid in scored[:limit]]
    except Exception as exc:
        log.warning("Experience search failed: %s", exc)
        return []
    finally:
        conn.close()


def _attr(value: object) -> str:
    return escape(str(value or ""), quote=True)


def experience_context_for(query: str, limit: int = 3, embedder=None) -> str:
    hits = search_experience(query, limit=limit, embedder=embedder)
    if not hits:
        return "<experience_context>\nNo similar past task found.\n</experience_context>"
    remaining = EXPERIENCE_CONTEXT_CHARS
    blocks = []
    for hit in hits:
        if remaining <= 0:
            break
        steps = json.loads(hit["steps_json"] or "[]")
        step_line = ", ".join(f"{s['tool']}[{'ok' if s['ok'] else s.get('error_type') or 'fail'}]" for s in steps)
        body = f"goal: {hit['goal']}\nsteps: {step_line}\nresult: {hit['answer_excerpt']}"[:remaining]
        blocks.append(f'<past_task outcome="{_attr(hit["outcome"])}" verifier_score="{float(hit["score"]):.2f}" recall_score="{hit["recall_score"]:.4f}">\n{body}\n</past_task>')
        remaining -= len(body)
    return "<experience_context>\n" + "\n\n".join(blocks) + "\n</experience_context>"
