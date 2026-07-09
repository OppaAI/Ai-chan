"""
core/experience.py

Aiko's episodic/procedural memory of her own past agentic task runs —
distinct from core/memory.py (facts *about the user*) and core/knowledge.py
(human-authored docs/skills/wiki). Experience is machine-written: at the end
of every agentic workflow, run_agentic_chat() records what goal was
attempted, which tools were called in what order with what outcome, and
whether the final answer passed verification. Future turns can then
semantically recall "how did a similar task go last time" the same way
knowledge_context_for() recalls "what do the docs say."

Not encrypted like memory, but NOT assumed harmless either — tool args can
carry incidental personal/sensitive content (a save_note body, a search
query), so records are sanitized before persisting, same spirit as
agentic._sanitize_user_facing_tool_detail.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import reason
from core.log import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = Path(os.getenv("EXPERIENCE_DB_PATH", REPO_ROOT / "data" / "experience.db"))
_EMBED_DIM = int(os.getenv("EXPERIENCE_EMBED_DIM", "1024"))  # match owner's embedder output dim

_EXPERIENCE_INSTRUCT = "Which past task is most similar to this request?"
_CHUNK_MIN_SCORE = float(os.getenv("EXPERIENCE_MIN_SCORE", "0.35"))
_MAX_ROWS = int(os.getenv("EXPERIENCE_MAX_ROWS", "5000"))       # hard cap, oldest/lowest-score pruned
_MAX_PER_GOAL_FAMILY = int(os.getenv("EXPERIENCE_MAX_PER_GOAL", "5"))  # dedupe near-identical goals

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_vec_available = False


def _connect() -> sqlite3.Connection:
    global _conn, _vec_available
    if _conn is not None:
        return _conn
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            outcome TEXT NOT NULL,           -- 'ok' | 'failed' | 'partial'
            score REAL NOT NULL,             -- verifier score, 0..1
            answer_excerpt TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    try:
        import sqlite_vec  # type: ignore
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS experience_vectors
            USING vec0(embedding float[{_EMBED_DIM}])
        """)
        _vec_available = True
    except Exception as e:
        log.warning("sqlite-vec unavailable for experience store; falling back to numpy brute-force cosine: %s", e)
        _vec_available = False
    conn.commit()
    _conn = conn
    return conn


_ERROR_DETAIL_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)",
)


def _sanitize(text: str, max_chars: int = 300) -> str:
    """Lightweight standalone sanitizer — deliberately not imported from
    core.agentic to avoid a circular import (agentic.py will import this
    module to record experience)."""
    t = (text or "").strip()
    t = _ERROR_DETAIL_RE.sub(r"\1\2[redacted]", t)
    t = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


@dataclass
class ExperienceStep:
    tool: str
    ok: bool
    error_type: str | None = None
    arg_keys: list[str] = field(default_factory=list)  # keys only, not values — avoids persisting raw args


def record_experience(
    owner,
    goal: str,
    steps: list[dict],       # TaskState.steps entries: {"tool","ok","attempts","error_type","args"}
    final_answer: str,
    verified_ok: bool,
    score: float,
    embedder=None,
) -> str | None:
    """Persist one completed (or abandoned) agentic run. Called from
    run_agentic_chat() after final_text is decided. Returns the new row id,
    or None if embedding/storage failed (never raises — experience is a
    best-effort recall aid, not critical path)."""
    try:
        conn = _connect()
        exp_steps = [
            ExperienceStep(
                tool=s["tool"], ok=s["ok"], error_type=s.get("error_type"),
                arg_keys=sorted((s.get("args") or {}).keys()),
            )
            for s in steps
        ]
        outcome = "ok" if verified_ok else ("partial" if any(s.ok for s in exp_steps) else "failed")
        row_id = str(uuid.uuid4())
        record_text = (
            f"Goal: {goal}\n"
            f"Steps: {', '.join(f'{s.tool}({\"+\".join(s.arg_keys) or \"-\"})[{\"ok\" if s.ok else s.error_type or \"fail\"}]' for s in exp_steps)}\n"
            f"Outcome: {outcome}"
        )
        with _lock:
            conn.execute(
                "INSERT INTO experiences (id, goal, steps_json, outcome, score, answer_excerpt, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    row_id, _sanitize(goal, 500),
                    json.dumps([s.__dict__ for s in exp_steps], ensure_ascii=False),
                    outcome, float(score), _sanitize(final_answer, 400), time.time(),
                ),
            )
            if embedder is not None:
                try:
                    vec = reason.normalize_vec(np.asarray(embedder.embed_query(record_text), dtype=np.float32))
                    if _vec_available:
                        conn.execute(
                            "INSERT INTO experience_vectors (rowid, embedding) VALUES ((SELECT rowid FROM experiences WHERE id=?), ?)",
                            (row_id, vec.tobytes()),
                        )
                except Exception as e:
                    log.warning("Experience embedding failed for row %s: %s", row_id, e)
            conn.commit()
        _prune(conn, goal, embedder)
        return row_id
    except Exception as e:
        log.warning("record_experience failed (non-fatal): %s", e)
        return None


def _prune(conn: sqlite3.Connection, goal: str, embedder=None) -> None:
    """Cap total rows, and dedupe near-identical goals — keep the
    highest-scored instances per goal family rather than letting repeated
    identical tasks flood recall with redundant near-duplicates."""
    with _lock:
        total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        if total > _MAX_ROWS:
            excess = total - _MAX_ROWS
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM experiences ORDER BY score ASC, created_at ASC LIMIT ?", (excess,)
            ).fetchall()]
            for rid in ids:
                conn.execute("DELETE FROM experiences WHERE id=?", (rid,))
                if _vec_available:
                    conn.execute("DELETE FROM experience_vectors WHERE rowid=(SELECT rowid FROM experiences WHERE id=?)", (rid,))
            conn.commit()
        # simple family cap: same first ~40 chars of sanitized goal
        family = _sanitize(goal, 40)
        rows = conn.execute(
            "SELECT id, score, created_at FROM experiences WHERE goal LIKE ? ORDER BY score DESC, created_at DESC",
            (f"{family}%",),
        ).fetchall()
        for rid, _score, _created in rows[_MAX_PER_GOAL_FAMILY:]:
            conn.execute("DELETE FROM experiences WHERE id=?", (rid,))
            if _vec_available:
                conn.execute("DELETE FROM experience_vectors WHERE rowid=(SELECT rowid FROM experiences WHERE id=?)", (rid,))
        conn.commit()


def search_experience(query: str, limit: int = 3, embedder=None) -> list[dict]:
    conn = _connect()
    if embedder is None:
        rows = conn.execute(
            "SELECT id, goal, steps_json, outcome, score, answer_excerpt FROM experiences ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    elif _vec_available:
        try:
            qvec = reason.normalize_vec(np.asarray(embedder.embed_query(query, instruct=_EXPERIENCE_INSTRUCT), dtype=np.float32))
            rows = conn.execute(
                """
                SELECT e.id, e.goal, e.steps_json, e.outcome, e.score, e.answer_excerpt
                FROM experience_vectors v
                JOIN experiences e ON e.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY distance ASC
                """,
                (qvec.tobytes(), limit),
            ).fetchall()
        except Exception as e:
            log.warning("sqlite-vec search failed, falling back: %s", e)
            rows = _brute_force_search(conn, query, limit, embedder)
    else:
        rows = _brute_force_search(conn, query, limit, embedder)

    return [
        {"id": r[0], "goal": r[1], "steps": json.loads(r[2]), "outcome": r[3], "score": r[4], "answer_excerpt": r[5]}
        for r in rows
    ]


def _brute_force_search(conn, query: str, limit: int, embedder) -> list[tuple]:
    """numpy fallback when sqlite-vec extension isn't loadable — same
    pattern as knowledge.py's in-memory matmul, just against all rows."""
    all_rows = conn.execute("SELECT id, goal, steps_json, outcome, score, answer_excerpt FROM experiences").fetchall()
    if not all_rows:
        return []
    qvec = reason.normalize_vec(np.asarray(embedder.embed_query(query, instruct=_EXPERIENCE_INSTRUCT), dtype=np.float32))
    texts = [f"Goal: {r[1]}\nOutcome: {r[3]}" for r in all_rows]
    vecs = np.stack([reason.normalize_vec(np.asarray(embedder.embed_query(t), dtype=np.float32)) for t in texts])
    scores = reason.batch_cosine_scores(qvec, vecs)
    order = np.argsort(-scores)[:limit]
    return [all_rows[i] for i in order if scores[i] >= _CHUNK_MIN_SCORE]


def experience_context_for(query: str, limit: int = 3, embedder=None) -> str:
    hits = search_experience(query, limit=limit, embedder=embedder)
    if not hits:
        return "<experience_context>\nNo similar past task found.\n</experience_context>"
    blocks = []
    for h in hits:
        step_line = ", ".join(f"{s['tool']}[{'ok' if s['ok'] else s.get('error_type') or 'fail'}]" for s in h["steps"])
        blocks.append(
            f'<past_task outcome="{h["outcome"]}" score="{h["score"]:.2f}">\n'
            f"goal: {h['goal']}\nsteps: {step_line}\nresult: {h['answer_excerpt']}\n</past_task>"
        )
    return "<experience_context>\n" + "\n\n".join(blocks) + "\n</experience_context>"
