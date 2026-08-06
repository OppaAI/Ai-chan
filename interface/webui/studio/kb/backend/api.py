"""Knowledge Graph Studio API — knowledge-only neural graph."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from system.log import get_logger

log = get_logger(__name__)

app = FastAPI(title="Aiko Knowledge Graph Studio", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/api/health")
def health():
    return {"ok": True, "studio": "knowledge-graph"}


@app.get("/api/graph")
def api_graph(
    limit: int | None = Query(200),
    user_id: str | None = Query(None),
):
    from .graph_export import export_knowledge_graph

    return export_knowledge_graph(user_id=user_id, limit=limit)


@app.get("/api/search")
def api_search(
    q: str = Query(""),
    limit: int = Query(20),
    user_id: str | None = Query(None),
):
    """Search learned knowledge (list helper for sidebar)."""
    try:
        from memory.knowledge import search_knowledge
        from system.userspace import current_user_id

        uid = user_id or current_user_id()
        hits = search_knowledge(q, limit=limit, user_id=uid) or []
        return {"query": q, "results": hits}
    except Exception as exc:
        log.warning("knowledge search failed: %s", exc)
        return {"query": q, "results": [], "error": str(exc)}


@app.get("/")
def index():
    for name in ("knowledge_graph.html", "index.html"):
        p = _FRONTEND / name
        if p.is_file():
            return FileResponse(p)
    return {"error": "frontend missing", "path": str(_FRONTEND)}


if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("KNOWLEDGE_STUDIO_PORT", "8002")))
