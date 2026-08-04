"""
agentic/experience.py

Persistent experience store for Aiko's completed agentic task runs.

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
import uuid
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from system.config import load_config
load_config()

try:
    from memory.vecstore import delete_by_id, initialize_store_db, insert_vector, rank_by_id, rrf_score, user_scoped_fts_search, user_scoped_vec_knn, utc_now_iso
    from memory.memorize import extract_entities, entities_to_json, entities_from_json, entity_overlap_score
except ImportError:  # lightweight practice.py/test environments may not have numpy/sqlite-vec
    from datetime import datetime, timezone
    def utc_now_iso(): return datetime.now(timezone.utc).isoformat()
    def initialize_store_db(path, ddl, user_id=None, vector=True):
        from system.userspace import user_state_dir
        db_path = Path(path)
        if not db_path.is_absolute():
            db_path = user_state_dir(user_id) / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Drop sqlite-vec-only statements for fallback mode.
        safe = re.sub(r"CREATE VIRTUAL TABLE IF NOT EXISTS experiences_vec USING vec0\([^;]+;", "", ddl, flags=re.S)
        safe = re.sub(r"DELETE FROM experiences_vec WHERE id = old.id;", "", safe)
        conn.executescript(safe)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(experiences)").fetchall()]
        if "entities" not in cols:
            conn.execute("ALTER TABLE experiences ADD COLUMN entities TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
        return conn
    def insert_vector(*args, **kwargs): return None
    def delete_by_id(conn, table, row_id): conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
    def rank_by_id(rows): return {row["id"]: i for i, row in enumerate(rows)}
    def rrf_score(eid, *rankings, k=60): return 1.0
    def user_scoped_vec_knn(*args, **kwargs): return []
    def user_scoped_fts_search(conn, fts_table, owner_table, owner_alias, query, user_id, limit):
        return conn.execute(f"SELECT * FROM {owner_table} WHERE user_id=? AND record_text LIKE ? LIMIT ?", (user_id, f"%{query}%", limit)).fetchall()
from system.log import get_logger
from system.userspace import current_user_id

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
EXPERIENCE_ENTITY_BOOST = float(os.getenv("EXPERIENCE_ENTITY_BOOST", "0.003"))
EXPERIENCE_AUTO_RELATE_THRESHOLD = float(os.getenv("EXPERIENCE_AUTO_RELATE_THRESHOLD", "0.90"))

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
    entities       TEXT NOT NULL DEFAULT '[]',
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

-- Engram relations: explicit links between experiences (continuation, contradiction, refines, synthesizes)
CREATE TABLE IF NOT EXISTS engram_relations (
    from_engram TEXT NOT NULL,
    to_engram   TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'continuation', 'contradiction', 'refines', 'synthesizes'
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (from_engram, to_engram, relation_type),
    FOREIGN KEY (from_engram) REFERENCES experiences(id) ON DELETE CASCADE,
    FOREIGN KEY (to_engram)   REFERENCES experiences(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_engram_relations_from ON engram_relations(from_engram);
CREATE INDEX IF NOT EXISTS idx_engram_relations_to   ON engram_relations(to_engram);

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


def _connect(user_id: str | None = None) -> sqlite3.Connection:
    return initialize_store_db(EXPERIENCE_DB_PATH, _DDL, user_id=user_id, vector=True)


def _sanitize(text: str, max_chars: int = 500) -> str:
    t = _SECRET_RE.sub(r"\1\2[redacted]", text or "")
    t = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", t)
    return re.sub(r"\s+", " ", t).strip()[:max_chars]


def _now() -> str:
    return utc_now_iso()




@dataclass
class ExperienceStep:
    tool: str
    ok: bool
    error_type: str | None = None
    arg_keys: list[str] = field(default_factory=list)
    args_preview: dict[str, str] = field(default_factory=dict)


def record_experience(owner, goal: str, steps: list[dict], final_answer: str, verified_ok: bool, score: float, embedder=None) -> str | None:
    uid = current_user_id()
    exp_steps = [
        ExperienceStep(
            tool=str(s.get("tool", "unknown")),
            ok=bool(s.get("ok")),
            error_type=s.get("error_type"),
            arg_keys=sorted((s.get("args") or {}).keys()),
            args_preview={k: _sanitize(str(v), 120) for k, v in (s.get("args") or {}).items()},
        )
        for s in steps
    ]
    outcome = "ok" if verified_ok else ("partial" if any(s.ok for s in exp_steps) else "failed")
    step_text = ", ".join(f"{s.tool}({'+'.join(s.arg_keys) or '-'})[{'ok' if s.ok else s.error_type or 'fail'}]" for s in exp_steps)
    args_text = "; ".join(
        f"{s.tool} args " + json.dumps(s.args_preview, ensure_ascii=False, sort_keys=True)
        for s in exp_steps if s.args_preview
    )
    record_text = (
        f"Goal: {_sanitize(goal, 700)}\n"
        f"Steps: {step_text}\n"
        f"Args: {_sanitize(args_text, 900)}\n"
        f"Outcome: {outcome}\n"
        f"Score: {float(score):.2f}\n"
        f"Result: {_sanitize(final_answer, 300)}"
    )
    row_id = str(uuid.uuid4())
    ents_json = entities_to_json(extract_entities(f"{goal} {final_answer}"))
    conn = _connect(uid)
    try:
        conn.execute(
            "INSERT INTO experiences(id,user_id,goal,record_text,steps_json,outcome,score,answer_excerpt,entities,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                row_id,
                uid,
                _sanitize(goal, 700),
                record_text,
                json.dumps([s.__dict__ for s in exp_steps], ensure_ascii=False),
                outcome,
                float(score),
                _sanitize(final_answer, 500),
                ents_json,
                _now(),
            ),
        )
        conn.commit()
        if embedder is not None:
            try:
                vec = embedder.embed_query(record_text, instruct=EXPERIENCE_QUERY_INSTRUCT)
                insert_vector(conn, "experiences_vec", row_id, vec)
                conn.commit()
            except Exception as embed_exc:
                log.warning("experience embedding failed ...: %s", embed_exc)
                vec = None
        else:
            vec = None

        # Optional Phase 8: auto engram link via embedding cosine (not RRF).
        if EXPERIENCE_AUTO_RELATE_THRESHOLD > 0 and vec is not None:
            try:
                neighbors = user_scoped_vec_knn(
                    conn,
                    vec_table="experiences_vec",
                    owner_table="experiences",
                    owner_alias="e",
                    vector=vec,
                    user_id=uid,
                    limit=5,
                    threshold=EXPERIENCE_AUTO_RELATE_THRESHOLD,
                )
                for nb in neighbors:
                    hid = str(nb["id"])
                    if hid == row_id:
                        continue
                    dist = float(nb["dist"])
                    sim = 1.0 - dist  # cosine similarity
                    if sim < EXPERIENCE_AUTO_RELATE_THRESHOLD:
                        continue
                    old = conn.execute(
                        "SELECT outcome FROM experiences WHERE id=? AND user_id=?",
                        (hid, uid),
                    ).fetchone()
                    old_outcome = (old["outcome"] if old else "") or ""
                    old_outcome = old_outcome.lower()
                    new_o = outcome.lower()
                    if (
                        old_outcome
                        and new_o
                        and old_outcome != new_o
                        and {old_outcome, new_o} & {"ok", "failed"}
                    ):
                        rel = "contradiction"
                    elif old_outcome == new_o:
                        rel = "continuation"
                    else:
                        rel = "refines"
                    record_engram_relation(
                        row_id, hid, rel, confidence=min(1.0, sim), user_id=uid
                    )
            except Exception as rel_exc:
                log.debug("auto engram relate skipped: %s", rel_exc)

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
        delete_by_id(conn, "experiences", row["id"])
    conn.commit()


def _knn(conn: sqlite3.Connection, query: str, embedder, uid: str, limit: int) -> list[sqlite3.Row]:
    if embedder is None:
        return []
    vector = embedder.embed_query(query, instruct=EXPERIENCE_QUERY_INSTRUCT)
    return user_scoped_vec_knn(
        conn,
        vec_table="experiences_vec",
        owner_table="experiences",
        owner_alias="e",
        vector=vector,
        user_id=uid,
        limit=limit,
    )


def _fts(conn: sqlite3.Connection, query: str, uid: str, limit: int) -> list[sqlite3.Row]:
    return user_scoped_fts_search(
        conn,
        fts_table="experiences_fts",
        owner_table="experiences",
        owner_alias="e",
        query=query,
        user_id=uid,
        limit=limit,
    )


def search_experience(query: str, limit: int = 3, embedder=None) -> list[dict]:
    uid = current_user_id()
    conn = _connect(uid)
    try:
        rank_knn = rank_by_id(_knn(conn, query, embedder, uid, EXPERIENCE_KNN_LIMIT))
        rank_fts = rank_by_id(_fts(conn, query, uid, EXPERIENCE_FTS_LIMIT))
        ids = set(rank_knn) | set(rank_fts)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT * FROM experiences WHERE id IN ({placeholders})", list(ids)).fetchall()
        by_id = {row["id"]: row for row in rows}
        scored = []
        for cid in ids:
            score = rrf_score(cid, rank_knn, rank_fts, k=EXPERIENCE_RRF_K)
            row = by_id.get(cid)
            if row is None:
                continue
            try:
                ents = entities_from_json(row["entities"] if "entities" in row.keys() else "[]")
            except Exception:
                ents = []
            score += EXPERIENCE_ENTITY_BOOST * entity_overlap_score(query, ents)
            if score >= EXPERIENCE_RECALL_SCORE_THRESHOLD:
                scored.append((score, cid))
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


def record_practice_experience(goal: str, steps: list[dict], final_answer: str = "practice workflow", verified_ok: bool = True, score: float = 1.0, embedder=None) -> str | None:
    """Record an operator-provided practice workflow without booting chat.

    This is used by ``practice.py`` to seed experience/playbook promotion
    while testing tiny routing/execution models such as Needle.
    """
    return record_experience(None, goal, steps, final_answer, verified_ok, score, embedder=embedder)


# ── Engram relations ───────────────────────────────────────────────────────────
# Explicit links between experiences: continuation, contradiction, refines, synthesizes

RELATION_TYPES = ("continuation", "contradiction", "refines", "synthesizes")

def record_engram_relation(from_engram: str, to_engram: str, relation_type: str, confidence: float = 1.0, user_id: str | None = None) -> bool:
    """Record an explicit relation between two experiences.

    Args:
        from_engram: Source experience ID
        to_engram: Target experience ID
        relation_type: One of 'continuation', 'contradiction', 'refines', 'synthesizes'
        confidence: 0.0-1.0 confidence score
        user_id: Optional user ID (defaults to current)

    Returns:
        True if recorded, False if invalid relation type or DB error
    """
    if relation_type not in RELATION_TYPES:
        log.warning("Invalid relation_type %r; must be one of %s", relation_type, RELATION_TYPES)
        return False
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO engram_relations
               (from_engram, to_engram, relation_type, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (from_engram, to_engram, relation_type, max(0.0, min(1.0, confidence)), utc_now_iso())
        )
        conn.commit()
        return True
    except Exception as exc:
        log.warning("record_engram_relation failed: %s", exc)
        return False
    finally:
        conn.close()


def get_engram_relations(engram_id: str, direction: str = "both", user_id: str | None = None) -> list[dict]:
    """Fetch relations for an engram.

    Args:
        engram_id: Experience ID to query
        direction: 'outgoing' (from), 'incoming' (to), or 'both'
        user_id: Optional user ID (defaults to current)

    Returns:
        List of relation dicts with keys: from_engram, to_engram, relation_type, confidence, created_at
    """
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        if direction == "outgoing":
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE from_engram=?",
                (engram_id,)
            ).fetchall()
        elif direction == "incoming":
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE to_engram=?",
                (engram_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE from_engram=? OR to_engram=?",
                (engram_id, engram_id)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
