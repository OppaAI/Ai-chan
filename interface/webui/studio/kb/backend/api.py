"""Read-only backend for the Aiko KB storage viewer.

Serves live queries against the knowledge store (learned_docs / learned_chunks /
archive tables) so you can inspect what is actually stored, superseded,
archived, or pruned — without any write access.

Endpoints (all GET, read-only):
  /api/kb/summary          counts + breakdown by status/source
  /api/kb/docs             list documents (searchable, paginated)
  /api/kb/docs/{doc_id}    one document + its chunks
  /api/kb/chunks/{id}      one chunk detail (entities, status, supersede)
  /api/kb/search           search chunk text (FTS-ish LIKE)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko KB Storage Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve the frontend assets (style.css, script.js) so the SPA works when
# mounted at /studio/kb or run standalone. Matches the approval studio's
# convention: frontend files stay in frontend/, served under /static.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="kb-frontend")


def _db_path() -> Path:
    from cognition.knowledge.schema import KNOWLEDGE_DB_PATH
    from system.userspace import user_state_path
    return user_state_path(KNOWLEDGE_DB_PATH).resolve()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="knowledge DB not found")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _row(dict_) -> dict:
    return {k: dict_[k] for k in dict_.keys()}


@app.get("/api/kb/summary")
async def summary():
    try:
        conn = _connect()
    except HTTPException:
        return {
            "db_exists": False,
            "docs": 0,
            "chunks": 0,
            "archived": 0,
            "superseded": 0,
            "active": 0,
            "by_kind": {},
            "by_status": {},
        }
    try:
        total_chunks = conn.execute("SELECT COUNT(*) FROM learned_chunks").fetchone()[0]
        total_docs = conn.execute("SELECT COUNT(*) FROM learned_docs").fetchone()[0]
        archived = conn.execute("SELECT COUNT(*) FROM learned_chunks_archive").fetchone()[0]
        by_kind = {r["kind"]: r["n"] for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM learned_docs GROUP BY kind ORDER BY n DESC")}
        by_status = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM learned_chunks GROUP BY status ORDER BY n DESC")}
        active = conn.execute(
            "SELECT COUNT(*) FROM learned_chunks WHERE status = 'active' OR status IS NULL").fetchone()[0]
        superseded = conn.execute(
            "SELECT COUNT(*) FROM learned_chunks WHERE status = 'superseded'").fetchone()[0]
        return {
            "db_exists": True,
            "docs": total_docs,
            "chunks": total_chunks,
            "archived": archived,
            "active": active,
            "superseded": superseded,
            "by_kind": by_kind,
            "by_status": by_status,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    finally:
        conn.close()


@app.get("/api/kb/docs")
async def docs(
    q: str = "",
    kind: str = "",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    try:
        conn = _connect()
    except HTTPException:
        return {"total": 0, "docs": []}
    try:
        where, params = [], []
        if q:
            where.append("(d.title LIKE ? OR d.source LIKE ?)")
            params += [f"%{q}%", f"%{q}%"]
        if kind:
            where.append("d.kind = ?")
            params.append(kind)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = conn.execute(
            f"SELECT COUNT(*) FROM learned_docs d {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT d.id, d.title, d.source, d.kind, d.created_at,
                       COUNT(c.id) AS chunk_count,
                       SUM(CASE WHEN c.status = 'active' OR c.status IS NULL THEN 1 ELSE 0 END) AS active_chunks,
                       SUM(CASE WHEN c.status = 'superseded' THEN 1 ELSE 0 END) AS superseded_chunks
                FROM learned_docs d
                LEFT JOIN learned_chunks c ON c.doc_id = d.id
                {clause}
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        return {"total": total, "docs": [_row(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    finally:
        conn.close()


@app.get("/api/kb/docs/{doc_id}")
async def doc_detail(doc_id: str):
    try:
        conn = _connect()
    except HTTPException:
        raise HTTPException(status_code=404, detail="knowledge DB not found")
    try:
        doc = conn.execute(
            "SELECT * FROM learned_docs WHERE id = ?", (doc_id,)).fetchone()
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        chunks = conn.execute(
            """SELECT id, chunk_index, substr(text, 1, 400) AS text_preview,
                      status, supersedes_id, access_count, last_accessed, created_at, entities
               FROM learned_chunks WHERE doc_id = ?
               ORDER BY chunk_index""",
            (doc_id,),
        ).fetchall()
        return {"doc": _row(doc), "chunks": [_row(c) for c in chunks]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    finally:
        conn.close()


@app.get("/api/kb/chunks/{chunk_id}")
async def chunk_detail(chunk_id: str):
    try:
        conn = _connect()
    except HTTPException:
        raise HTTPException(status_code=404, detail="knowledge DB not found")
    try:
        chunk = conn.execute(
            """SELECT c.*, d.title AS doc_title, d.source AS doc_source, d.kind AS doc_kind
               FROM learned_chunks c JOIN learned_docs d ON d.id = c.doc_id
               WHERE c.id = ?""",
            (chunk_id,),
        ).fetchone()
        if chunk is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        row = _row(chunk)
        try:
            row["entities"] = json.loads(row.get("entities") or "[]")
        except Exception:
            row["entities"] = []
        superseded_by = conn.execute(
            "SELECT id FROM learned_chunks WHERE supersedes_id = ?", (chunk_id,)).fetchall()
        row["superseded_by"] = [r["id"] for r in superseded_by]
        return row
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    finally:
        conn.close()


@app.get("/api/kb/search")
async def search(
    q: str = "",
    status: str = "",
    limit: int = Query(50, ge=1, le=200),
):
    if not q:
        return {"total": 0, "chunks": []}
    try:
        conn = _connect()
    except HTTPException:
        return {"total": 0, "chunks": []}
    try:
        where, params = ["c.text LIKE ?"], [f"%{q}%"]
        if status:
            where.append("c.status = ?")
            params.append(status)
        clause = " AND ".join(where)
        total = conn.execute(
            f"SELECT COUNT(*) FROM learned_chunks c {_from()} WHERE {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT c.id, c.doc_id, c.chunk_index, substr(c.text, 1, 300) AS text_preview,
                       c.status, c.access_count, c.created_at, d.title AS doc_title
                {_from()}
                WHERE {clause}
                ORDER BY c.created_at DESC
                LIMIT ?""",
            params + [limit],
        ).fetchall()
        return {"total": total, "chunks": [_row(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    finally:
        conn.close()


def _from() -> str:
    return "FROM learned_chunks c LEFT JOIN learned_docs d ON d.id = c.doc_id"


@app.get("/")
async def serve_studio(request: Request):
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
