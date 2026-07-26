"""Studio backend for Aiko graph visualizer.

Serves the playbooks JSON and handles API requests for the studio frontend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json

app = FastAPI(title="Aiko Graph Studio")

# Allow connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR / "frontend"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

templates = Jinja2Templates(directory=str(FRONTEND_DIR))


# ── Playbook helpers ──────────────────────────────────────────────────────────
def playbook_to_graph(playbook: dict) -> dict:
    """Convert playbook (nodes list + depends_on) to graph with edges array."""
    if not isinstance(playbook.get("nodes"), list):
        return playbook
    nodes = playbook["nodes"]
    edges = []
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        for dep in n.get("depends_on") or []:
            edges.append({"source": dep, "target": nid, "type": "depends_on"})
        if n.get("loop_to"):
            edges.append({"source": nid, "target": n["loop_to"], "type": "loop_to"})
        if n.get("fallback_to"):
            edges.append({"source": nid, "target": n["fallback_to"], "type": "fallback_to"})
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
async def serve_studio(request):
    """Serve the studio interface."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)