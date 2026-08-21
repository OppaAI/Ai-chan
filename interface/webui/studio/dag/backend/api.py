"""Studio backend for Aiko graph visualizer.

Serves the playbooks JSON and handles API requests for the studio frontend.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

BASE_DIR = Path(__file__).resolve().parent.parent
STUDIO_DIR = BASE_DIR
FRONTEND_DIR = STUDIO_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

# Serve the frontend assets (style.css, script.js) so the SPA works when
# mounted at /studio/dag or run standalone. Matches the approval studio's
# convention: frontend files stay in frontend/, served under /static.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dag-frontend")

app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")


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
    from agentic.workflows.common.graphs import get_graph
    raw = load_playbooks()
    result = []
    for p in raw:
        g = playbook_to_graph(p)
        # If playbook references a graph_id, resolve it from the registry
        graph_id = g.get("graph_id")
        if graph_id and not g.get("nodes"):
            graph = get_graph(graph_id)
            if graph:
                # Convert PlanGraph to playbook format with nodes/edges
                nodes = []
                edges = []
                for n in graph.nodes:
                    nodes.append({
                        "id": n.id,
                        "tool": n.tool,
                        "args": dict(n.args or {}),
                        "depends_on": list(n.depends_on or ()),
                        "loop_to": getattr(n, "loop_to", None),
                        "max_visits": getattr(n, "max_visits", None),
                        "fallback_to": getattr(n, "fallback_to", None),
                        "run_if": getattr(n, "run_if", None),
                    })
                    for dep in n.depends_on or ():
                        edges.append({"source": dep, "target": n.id, "type": "depends_on"})
                    loop_to = getattr(n, "loop_to", None)
                    if loop_to:
                        edges.append({"source": n.id, "target": loop_to, "type": "loop_to"})
                    fallback_to = getattr(n, "fallback_to", None)
                    if fallback_to:
                        edges.append({"source": n.id, "target": fallback_to, "type": "fallback_to"})
                g = {**g, "nodes": nodes, "edges": edges}
        result.append(g)
    return result


# Load playbooks on startup (will be refreshed on each API call)
try:
    PLAYBOOKS = load_playbooks_refresh()
except Exception as _pb_exc:
    import logging as _logging
    _logging.getLogger(__name__).warning("DAG studio: initial playbook load failed: %s", _pb_exc)
    PLAYBOOKS = []


@app.get("/api/playbooks")
def get_playbooks():
    """Get all playbooks for the studio (refreshed)."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    return PLAYBOOKS


@app.get("/api/playbooks/{playbook_id}")
def get_playbook(playbook_id: str):
    """Get a specific playbook by ID (refreshed)."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    for playbook in PLAYBOOKS:
        if playbook.get("id") == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.put("/api/playbooks/{playbook_id}")
async def save_playbook(playbook_id: str, request: Request):
    """Persist an edited playbook to the user-scoped source used by execution."""
    from agentic.graph_engine import _playbook_file, _playbook_write_guard

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("id") != playbook_id:
        raise HTTPException(status_code=400, detail="payload id must match playbook id")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise HTTPException(status_code=400, detail="playbook requires a non-empty nodes list")

    clean_nodes = []
    node_ids = set()
    for raw in nodes:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("tool"):
            raise HTTPException(status_code=400, detail="each node requires id and tool")
        node_id = str(raw["id"])
        if node_id in node_ids:
            raise HTTPException(status_code=400, detail=f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        node = {k: v for k, v in raw.items() if not k.startswith("_")}
        node["id"] = node_id
        node["tool"] = str(node["tool"])
        if not isinstance(node.get("args", {}), dict):
            raise HTTPException(status_code=400, detail=f"args must be an object for node {node_id}")
        node["depends_on"] = [str(dep) for dep in node.get("depends_on", [])]
        clean_nodes.append(node)
    for node in clean_nodes:
        unknown = [dep for dep in node["depends_on"] if dep not in node_ids]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown dependency for {node['id']}: {unknown[0]}")
        for field in ("loop_to", "fallback_to"):
            if node.get(field) and str(node[field]) not in node_ids:
                raise HTTPException(status_code=400, detail=f"unknown {field} for {node['id']}: {node[field]}")

    clean = {k: v for k, v in payload.items() if k not in {"edges", "nodes"} and not k.startswith("_")}
    clean["id"] = playbook_id
    clean["nodes"] = clean_nodes
    path = _playbook_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _playbook_write_guard(path):
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"failed to read playbook source: {exc}") from exc
        if not isinstance(existing, list):
            existing = []
        for index, item in enumerate(existing):
            if isinstance(item, dict) and item.get("id") == playbook_id:
                existing[index] = clean
                break
        else:
            existing.append(clean)
        tmp = path.with_suffix(path.suffix + ".studio.tmp")
        try:
            tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"failed to write playbook source: {exc}") from exc
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path)}


@app.get("/")
async def serve_studio(request: Request):
    """Serve the studio interface (static SPA; no Jinja needed)."""
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
