"""Memory Graph Studio backend — visualize personal memory nodes & links."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Aiko Memory Graph Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@app.get("/api/graph")
async def get_graph(
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    limit: int = Query(200, ge=1, le=2000),
    include_history: bool = Query(True, description="Include superseded memories"),
    include_entities: bool = Query(True, description="Add entity hub nodes"),
):
    from memory.studio.backend.graph_export import export_memory_graph
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    return export_memory_graph(
        user_id=uid,
        limit=limit,
        include_history=include_history,
        include_entities=include_entities,
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "memory-graph-studio"}


@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query across memory + knowledge"),
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    limit: int = Query(10, ge=1, le=100),
    include_history: bool = Query(False, description="Include superseded memories"),
):
    from memory.studio.backend.search_memory import search_memory
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    query = (q or "").strip()
    if not query:
        return {"query": "", "hits": [], "meta": {"user_id": uid}}

    try:
        from memory.memorize import AikoMemorize

        memorize = AikoMemorize(silent=True)
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


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
