"""Studio backend for Aiko graph visualizer.

Serves the playbooks JSON and handles API requests for the studio frontend.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Aiko Graph Studio")

# Allow connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STUDIO_DIR = BASE_DIR
STATIC_DIR = STUDIO_DIR / "static"
FRONTEND_DIR = STUDIO_DIR / "frontend"

if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ── Playbook helpers ──────────────────────────────────────────────────────────
def playbook_to_graph(playbook: dict) -> dict:
    """Convert playbook (nodes list + depends_on) to graph with edges array."""
    if not isinstance(playbook.get("nodes"), list):
        return playbook
    nodes = playbook["nodes"]
    node_map = {n.get("id"): n for n in nodes if n.get("id")}
    edges = []
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        src_node = node_map.get(nid, {})
        tool_call = {"tool": src_node.get("tool"), "args": src_node.get("args")}
        for dep in n.get("depends_on") or []:
            edges.append({"source": dep, "target": nid, "type": "depends_on", "tool_call": tool_call, "skill": src_node.get("tool")})
        if n.get("loop_to"):
            edges.append({"source": nid, "target": n["loop_to"], "type": "loop_to", "tool_call": tool_call, "skill": src_node.get("tool")})
        if n.get("fallback_to"):
            edges.append({"source": nid, "target": n["fallback_to"], "type": "fallback_to", "tool_call": tool_call, "skill": src_node.get("tool")})
    return {**playbook, "nodes": nodes, "edges": edges}


def load_playbooks_refresh() -> list:
    """Reload playbooks from graph_engine (fresh each call)."""
    from agentic.graph_engine import load_playbooks
    raw = load_playbooks()
    return [playbook_to_graph(p) for p in raw]


# Load playbooks on startup (will be refreshed on each API call)
PLAYBOOKS = load_playbooks_refresh()


@app.get("/api/playbooks")
async def get_playbooks():
    """Get all playbooks for the studio (refreshed)."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    return PLAYBOOKS


@app.get("/api/playbooks/{playbook_id}")
async def get_playbook(playbook_id: str):
    """Get a specific playbook by ID (refreshed)."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    for playbook in PLAYBOOKS:
        if playbook.get("id") == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.get("/")
async def serve_studio(request: Request):
    """Serve the studio interface (static SPA; no Jinja needed)."""
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
