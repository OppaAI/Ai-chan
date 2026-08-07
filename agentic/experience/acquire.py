"""Experience acquisition (record/ingest)."""
from __future__ import annotations

import json
import sqlite3
import uuid

from cognition.memory.memorize import entities_to_json, extract_entities
from cognition.memory.vecstore import delete_by_id, insert_vector, user_scoped_vec_knn
from system.log import get_logger
from system.userspace import current_user_id

from .lifecycle import record_engram_relation
from .schema import (
    ExperienceStep,
    EXPERIENCE_AUTO_RELATE_THRESHOLD,
    ExperienceSchema,
    EXPERIENCE_MAX_ROWS,
    EXPERIENCE_QUERY_INSTRUCT,
    EXPERIENCE_SUPERSEDE_ON_NEAR_DUP,
    EXPERIENCE_SUPERSEDE_THRESHOLD,
    connect,
    now,
    sanitize,
)

log = get_logger(__name__)


class ExperienceWriter:
    """Owns the experience record write path (insert, embed, auto-relate, prune)."""

    def __init__(self, schema: ExperienceSchema | None = None):
        self.schema = schema or ExperienceSchema()

    def record(
        self,
        owner,
        goal: str,
        steps: list[dict],
        final_answer: str,
        verified_ok: bool,
        score: float,
        embedder=None,
    ) -> str | None:
        return record_experience(owner, goal, steps, final_answer, verified_ok, score, embedder=embedder)

    def record_practice(
        self,
        goal: str,
        steps: list[dict],
        final_answer: str = "practice workflow",
        verified_ok: bool = True,
        score: float = 1.0,
        embedder=None,
    ) -> str | None:
        return record_practice_experience(goal, steps, final_answer, verified_ok, score, embedder=embedder)


def record_experience(owner, goal: str, steps: list[dict], final_answer: str, verified_ok: bool, score: float, embedder=None) -> str | None:
    """Legacy free-function shim: full implementation of the write path.

    :class:`ExperienceWriter.record` delegates to this."""
    uid = current_user_id()
    exp_steps = [
        ExperienceStep(
            tool=str(s.get("tool", "unknown")),
            ok=bool(s.get("ok")),
            error_type=s.get("error_type"),
            arg_keys=sorted((s.get("args") or {}).keys()),
            args_preview={k: sanitize(str(v), 120) for k, v in (s.get("args") or {}).items()},
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
        f"Goal: {sanitize(goal, 700)}\n"
        f"Steps: {step_text}\n"
        f"Args: {sanitize(args_text, 900)}\n"
        f"Outcome: {outcome}\n"
        f"Score: {float(score):.2f}\n"
        f"Result: {sanitize(final_answer, 300)}"
    )
    row_id = str(uuid.uuid4())
    ents_json = entities_to_json(extract_entities(f"{goal} {final_answer}"))
    conn = connect(uid)
    try:
        conn.execute(
            "INSERT INTO experiences(id,user_id,goal,record_text,steps_json,outcome,score,answer_excerpt,entities,created_at,status,supersedes_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row_id,
                uid,
                sanitize(goal, 700),
                record_text,
                json.dumps([s.__dict__ for s in exp_steps], ensure_ascii=False),
                outcome,
                float(score),
                sanitize(final_answer, 500),
                ents_json,
                now(),
                "active",
                None,
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
                best_sup = None  # (hid, sim)
                for nb in neighbors:
                    hid = str(nb["id"])
                    if hid == row_id:
                        continue
                    dist = float(nb["dist"])
                    sim = 1.0 - dist
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
                    if (
                        EXPERIENCE_SUPERSEDE_ON_NEAR_DUP
                        and sim >= EXPERIENCE_SUPERSEDE_THRESHOLD
                        and (best_sup is None or sim > best_sup[1])
                    ):
                        best_sup = (hid, sim)

                # after the for nb loop (same indent as `for nb`)
                if best_sup is not None:
                    hid, sim = best_sup
                    try:
                        conn.execute(
                            "UPDATE experiences SET status = 'superseded' "
                            "WHERE id = ? AND user_id = ? "
                            "AND (status = 'active' OR status IS NULL OR status = '')",
                            (hid, uid),
                        )
                        conn.execute(
                            "UPDATE experiences SET supersedes_id = ? WHERE id = ? AND user_id = ?",
                            (hid, row_id, uid),
                        )
                        conn.commit()
                        log.debug("experience supersede sim=%.3f old=%s", sim, hid[:8])
                    except Exception as sup_exc:
                        log.debug("experience supersede skipped: %s", sup_exc)
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


def record_practice_experience(goal: str, steps: list[dict], final_answer: str = "practice workflow", verified_ok: bool = True, score: float = 1.0, embedder=None) -> str | None:
    """Record an operator-provided practice workflow without booting chat.

    This is used by ``practice.py`` to seed experience/playbook promotion
    while testing tiny routing/execution models such as Needle.
    """
    return record_experience(None, goal, steps, final_answer, verified_ok, score, embedder=embedder)


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