"""Knowledge Graph Studio API — knowledge-only neural graph."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from system.log import get_logger
log = get_logger(__name__)
app = FastAPI(title="Aiko Knowledge Graph Studio", version="1.0")
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
@app.get("/api/health")
def health():
    return {"ok": True, "studio": "knowledge-graph"}
@app.get("/api/graph")
def api_graph(
    user_id: str | None = Query(None, description="User id (default: current_user_id)"),
    limit: int = Query(200, ge=1, le=1000),
    include_history: bool = Query(True, description="Include superseded/archived knowledge"),
    include_entities: bool = Query(True, description="Add entity hub nodes"),
    date_from: str | None = Query(None, description="Only include items created at/after this date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Only include items created at/before this date (YYYY-MM-DD)"),
):
    from .graph_export import export_knowledge_graph
    from system.userspace import current_user_id
    
    uid = (user_id or "").strip() or current_user_id()
    try:
        return export_knowledge_graph(
            user_id=uid,
            limit=limit,
            include_history=include_history,
            include_entities=include_entities,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as e:
        log.error("knowledge graph export failed: %s", e)
        return {
            "nodes": [],
            "edges": [],
            "meta": {
                "user_id": uid,
                "error": str(e),
                "count": 0,
            },
        }
@app.get("/api/search")
def api_search(q: str = Query(""), limit: int = Query(20, ge=1, le=100)):
    try:
        from cognition.knowledge import search_knowledge
        from system.userspace import current_user_id
        hits = search_knowledge(q, limit=limit, user_id=current_user_id()) or []
        return {"query": q, "results": hits}
    except Exception as exc:
        log.warning("knowledge search failed: %s", exc)
        return {"query": q, "results": [], "error": "search failed"}
@app.get("/")
def index():
    p = _FRONTEND / "index.html"
    if p.is_file():
        return FileResponse(p)
    log.warning("frontend missing at %s", _FRONTEND)
    raise HTTPException(status_code=404, detail="frontend missing")
if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("KNOWLEDGE_STUDIO_PORT", "8002")))
