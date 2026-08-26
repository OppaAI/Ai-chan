"""ITM Studio backend — episodic memory (EMC) timeline + pipeline view.

Shows how memory is stored in episodes: working-memory eviction stages
into emc_staging, is flushed into emc_storage, and is distilled into
semantic facts by dream() (EM→SM). Read-only.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko ITM (Episodic Memory) Studio")

from interface.webui.studio.session_binding import bind_login_session
bind_login_session(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[3] / "shared"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="itm-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

_stores_by_user: dict[str, object] = {}
_store_lock = threading.Lock()


def _get_store(user_id: str | None = None):
    """Reuse one EpisodicStore per user, never another user's database."""
    uid = user_id or "guest"
    store = _stores_by_user.get(uid)
    if store is None:
        with _store_lock:
            store = _stores_by_user.get(uid)
            if store is None:
                from cognition.memory.episode import EpisodicStore
                from cognition.memory.schema import _memory_db_path_for_user
                import os

                embed_cache = os.getenv("EMBED_CACHE_PATH") or None
                store = EpisodicStore(_memory_db_path_for_user(uid), user_id=uid, embed_cache=embed_cache)
                _stores_by_user[uid] = store
    return store


def _resolve_user_id(user_id: str | None) -> str:
    from system.userspace import current_user_id
    # Studio identity comes from the authenticated request middleware.  Do
    # not let a stale browser field or a crafted query override it.
    return current_user_id()


def _parse_entities(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except (TypeError, json.JSONDecodeError):
        pass
    return []


def _parse_distilled_into(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except (TypeError, json.JSONDecodeError):
        pass
    return []


def _episode_row(row) -> dict[str, Any]:
    def get(name: str, default: Any = None) -> Any:
        try:
            return row[name]
        except (KeyError, IndexError):
            return default
    return {
        "id": int(get("id")),
        "timestamp": get("timestamp"),
        "date": get("date"),
        "trace": get("trace"),
        "valence_tag": get("valence_tag"),
        "arousal_score": get("arousal_score"),
        "salience_score": get("salience_score"),
        "entities": _parse_entities(get("entities")),
        "source": get("source"),
        "session_id": get("session_id"),
        "recall_count": int(get("recall_count") or 0),
        "last_recalled_at": get("last_recalled_at"),
        "distilled_at": get("distilled_at"),
        "distilled_into": _parse_distilled_into(get("distilled_into")),
        "stage": "distilled" if get("distilled_at") else "storage",
    }


@app.get("/api/pipeline")
def pipeline(user_id: str | None = Query(None)):
    """Counts across the episodic pipeline stages for this user."""
    from cognition.memory.schema import _memory_db_path_for_user

    uid = _resolve_user_id(user_id)
    try:
        store = _get_store(uid)
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}

    result = {
        "user_id": uid,
        "staging": 0,
        "storage": 0,
        "distilled": 0,
        "recalled_total": 0,
    }
    try:
        result["staging"] = store.staging_count(uid)
        result["storage"] = store.storage_count(uid)
    except Exception as e:
        result["error"] = str(e)
        return {"ok": True, **result}

    try:
        conn = store._conn
        with store._lock:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  COUNT(distilled_at) AS distilled,
                  COALESCE(SUM(recall_count), 0) AS recalled
                FROM emc_storage
                WHERE user_id = ?
                """,
                (uid,),
            ).fetchone()
        if row:
            result["distilled"] = int(row["distilled"] or 0)
            result["recalled_total"] = int(row["recalled"] or 0)
    except Exception:
        pass
    return {"ok": True, **result}


@app.get("/api/episodes")
def episodes(
    user_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    stage: str | None = Query(None, description="all|storage|distilled"),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    q: str | None = Query(None, description="substring filter on trace"),
):
    """Chronological episode timeline from emc_storage (newest first)."""
    from cognition.memory.schema import _memory_db_path_for_user

    uid = _resolve_user_id(user_id)
    try:
        store = _get_store(uid)
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}

    try:
        conn = store._conn
        cols = {r[1] for r in conn.execute("PRAGMA table_info(emc_storage)").fetchall()}
        has_distilled = "distilled_at" in cols
        has_distilled_into = "distilled_into" in cols
        has_last_recall = "last_recalled_at" in cols
        sel = "id, timestamp, date, trace, valence_tag, arousal_score, salience_score, entities, source, session_id, recall_count"
        if has_distilled:
            sel += ", distilled_at"
        if has_distilled_into:
            sel += ", distilled_into"
        if has_last_recall:
            sel += ", last_recalled_at"

        sql = f"SELECT {sel} FROM emc_storage WHERE user_id = ?"
        params: list[Any] = [uid]
        if stage == "distilled" and has_distilled:
            sql += " AND distilled_at IS NOT NULL"
        elif stage == "storage" and has_distilled:
            sql += " AND distilled_at IS NULL"
        if date_from:
            sql += " AND timestamp >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND timestamp <= ?"
            params.append(date_to)
        if q and (q := str(q).strip()):
            sql += " AND trace LIKE ?"
            params.append(f"%{q}%")
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))

        with store._lock:
            rows = conn.execute(sql, params).fetchall()
        out = [_episode_row(r) for r in rows]
        return {"ok": True, "user_id": uid, "count": len(out), "episodes": out}
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}


@app.get("/api/staging")
def staging(
    user_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    """Pending staged episodes (not yet flushed to storage)."""
    from cognition.memory.schema import _memory_db_path_for_user

    uid = _resolve_user_id(user_id)
    try:
        store = _get_store(uid)
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}

    try:
        conn = store._conn
        with store._lock:
            rows = conn.execute(
                """
                SELECT id, timestamp, date, trace, valence_tag, arousal_score,
                       salience_score, entities, source, session_id
                FROM emc_staging
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (uid, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            item = _episode_row(r)
            item["stage"] = "staging"
            out.append(item)
        return {"ok": True, "user_id": uid, "count": len(out), "episodes": out}
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}


@app.get("/api/episode/{episode_id}")
def episode_detail(
    episode_id: int,
    user_id: str | None = Query(None),
):
    """Full detail for one episode, including the semantic facts it distilled into."""
    from cognition.memory.schema import _memory_db_path_for_user

    uid = _resolve_user_id(user_id)
    try:
        store = _get_store(uid)
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}

    try:
        conn = store._conn
        with store._lock:
            row = conn.execute(
                """
                SELECT id, timestamp, date, trace, valence_tag, arousal_score,
                       salience_score, entities, source, session_id,
                       recall_count, last_recalled_at, distilled_at, distilled_into
                FROM emc_storage
                WHERE id = ? AND user_id = ?
                """,
                (int(episode_id), uid),
            ).fetchone()
        if not row:
            return {"ok": False, "user_id": uid, "error": "episode not found"}
        item = _episode_row(row)
        distilled_into = item.get("distilled_into") or []
        facts = []
        if distilled_into:
            try:
                from cognition.memory.memorize import AikoMemorize

                mem = AikoMemorize(silent=True)
                for mid in distilled_into:
                    rows = conn.execute(
                        "SELECT id, memory, created_at FROM memories WHERE id = ? AND user_id = ?",
                        (mid, uid),
                    ).fetchall()
                    for fr in rows:
                        facts.append({
                            "id": fr["id"],
                            "text": fr["memory"],
                            "created_at": fr["created_at"],
                        })
            except Exception:
                pass
        item["distilled_facts"] = facts
        return {"ok": True, "user_id": uid, "episode": item}
    except Exception as e:
        return {"ok": False, "user_id": uid, "error": str(e)}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "itm-studio"}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8004)
