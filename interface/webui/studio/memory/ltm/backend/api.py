"""LTM Graph Studio — FastAPI backend (local visualization only)."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Aiko LTM Graph Studio", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


# AikoMemorize opens a long-lived SQLite connection +
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
    include_episodes: bool = Query(True, description="EMC-5: episodic memory nodes"),
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
def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    user_id: str | None = Query(None),
    limit: int = Query(12, ge=1, le=50),
    include_history: bool = Query(False, description="Include superseded memories"),
):
    """Search personal memory + knowledge (studio helper)."""
    from system.userspace import current_user_id

    uid = (user_id or "").strip() or current_user_id()
    try:
        from interface.webui.studio.memory.ltm.backend.search_memory import search_memory_and_knowledge

        hits = search_memory_and_knowledge(
            q,
            user_id=uid,
            limit=limit,
            include_history=include_history,
        )
        return {
            "query": q,
            "user_id": uid,
            "hits": hits,
            "include_history": include_history,
        }
    except Exception as e:
        return {"query": q, "user_id": uid, "hits": [], "error": str(e)}
