"""LTM Graph Studio backend — visualize personal memory nodes & links."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko LTM Graph Studio")

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

# Serve frontend assets (style.css, script.js) under /static, matching the
# other studios (approval, dag, kb).
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ltm-frontend")

# Shared studio frontend helpers (graph-bootstrap.js).
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

# AikoMemorize __init__ opens a sqlite-vec connection AND spawns a daemon
# write-worker thread. Constructing one per request leaks a connection +
# thread for the lifetime of the process, so cache a single shared instance.
_memorize = None
_memorize_lock = threading.Lock()


def _get_memorize():
    global _memorize
    if _memorize is None:
        with _memorize_lock:
            if _memorize is None:
                from cognition.memory.memorize import AikoMemorize

                _memorize = AikoMemorize(silent=True)
    return _memorize


@app.get("/api/graph")
def get_graph(
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    limit: int = Query(200, ge=1, le=2000),
    include_history: bool = Query(True, description="Include superseded memories"),
    include_entities: bool = Query(True, description="Add entity hub nodes"),
    include_knowledge: bool = Query(True, description="Phase 13: learned knowledge nodes"),
    include_experience: bool = Query(True, description="Phase 13: experience nodes"),
    include_episodes: bool | None = Query(None, description="EMC-5: episodic memory nodes"),
    date_from: str | None = Query(None, description="Only include items created at/after this date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Only include items created at/before this date (YYYY-MM-DD)"),
):
    from interface.webui.studio.memory.ltm.backend.graph_export import export_memory_graph
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    return export_memory_graph(
        user_id=uid,
        limit=limit,
        include_history=include_history,
        include_entities=include_entities,
        include_knowledge=include_knowledge,
        include_experience=include_experience,
        include_episodes=include_episodes,
        date_from=date_from,
        date_to=date_to,
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "ltm-graph-studio"}


@app.get("/api/search")
def search(
    q: str = Query(..., description="Search query across memory + knowledge"),
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    limit: int = Query(10, ge=1, le=100),
    include_history: bool = Query(False, description="Include superseded memories"),
):
    from interface.webui.studio.memory.ltm.backend.search_memory import search_memory
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    query = (q or "").strip()
    if not query:
        return {"query": "", "hits": [], "meta": {"user_id": uid}}

    try:
        memorize = _get_memorize()
    except Exception as e:
        return {"query": query, "hits": [], "meta": {"user_id": uid, "error": str(e)}}

    try:
        hits = search_memory(
            query,
            limit=limit,
            memorize=memorize,
            user_id=uid,
            include_history=include_history,
        )
    except Exception as e:
        return {"query": query, "hits": [], "meta": {"user_id": uid, "error": str(e)}}

    return {
        "query": query,
        "hits": hits,
        "meta": {
            "user_id": uid,
            "count": len(hits),
            "include_history": include_history,
            "limit": limit,
        },
    }


@app.get("/api/memory/{mem_id}/lineage")
def memory_lineage(
    mem_id: str,
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
):
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    memorize = _get_memorize()
    return memorize.get_lineage(mem_id, user_id=uid)


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    # Local only — use a TLS reverse proxy for remote access.
    uvicorn.run(app, host="127.0.0.1", port=8001)
