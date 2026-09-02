"""Codebase Studio backend — figure-shaped knowledge graph for Aiko's brain."""
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko Codebase Studio (Figure)")

from interface.webui.studio.session_binding import bind_login_session
bind_login_session(app)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[3] / "shared"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="codebase-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

@app.get("/api/graph")
def get_graph(limit: int = Query(400, ge=1, le=2000)):
    from interface.webui.studio.codebase.backend.graph_export import export_codebase_graph
    from system.userspace import current_user_id
    uid = current_user_id()
    return export_codebase_graph(user_id=uid, limit=limit)

@app.get("/api/search")
def search(q: str = Query(..., description="Search codebase"), limit: int = Query(8, ge=1, le=20)):
    from system.userspace import current_user_id
    uid = current_user_id()
    query = (q or "").strip()
    if not query:
        return {"query": "", "hits": [], "meta": {"user_id": uid}}
    try:
        from cognition.knowledge.codebase import search_codebase
        from cognition.memory.vecstore import HarrierEmbedder
        try:
            emb = HarrierEmbedder()
        except Exception:
            emb = None
        hits = search_codebase(query, limit=limit, embedder=emb, user_id=uid)
        return {"query": query, "hits": hits, "meta": {"user_id": uid, "count": len(hits)}}
    except Exception as e:
        return {"query": query, "hits": [], "meta": {"user_id": uid, "error": str(e)}}

@app.get("/api/ingest")
def ingest(force: bool = Query(False)):
    from cognition.knowledge.codebase import ingest_codebase
    from system.userspace import current_user_id
    uid = current_user_id()
    return ingest_codebase(user_id=uid, force=force)

@app.get("/api/health")
async def health():
    return {"ok": True, "service": "codebase-studio"}

@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
