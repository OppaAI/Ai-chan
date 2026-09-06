"""Studio backend for Aiko graph visualizer (n8n-style).

Serves playbooks JSON + n8n-style CRUD:
  GET    /api/playbooks            list (refreshed)
  GET    /api/playbooks/{id}       one graph
  PUT    /api/playbooks/{id}       save edited nodes/edges
  POST   /api/playbooks            create new graph
  DELETE /api/playbooks/{id}       delete user graph (built-ins protected)
  POST   /api/playbooks/{id}/duplicate   clone with new id
  POST   /api/playbooks/{id}/validate    cycle/unknown-tool/missing-dep check
  POST   /api/playbooks/{id}/run         bounded dry-run (Jetson-safe)
  GET    /api/tools                palette: registry tools grouped by domain

Jetson notes: dry-run caps at 2 parallel workers and 60s wall-clock so a
browser click cannot OOM the Orin Nano. Approval-gated tools (code_apply,
social posts) run in dry-run only when explicitly confirmed — otherwise
they return their normal "needs approval" refusal, which is still useful
signal in the studio.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import re
import time

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


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


# Ensure the tool registry is fully populated (toolkit @tool decorators +
# workflow nodes) so validate/run see the same tools the agent runtime sees.
# Import failures must never take the studio down (Jetson boots with
# optional lanes missing), hence the broad except.
try:
    import agentic.tools  # noqa: F401
    import agentic.workflows.common.nodes  # noqa: F401
except Exception:  # pragma: no cover
    pass


def _valid_id(value: str) -> bool:
    return bool(value and _ID_RE.match(value))


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


def _builtin_ids() -> set[str]:
    """IDs shipped in code defaults — protected from DELETE."""
    try:
        from agentic.graph_engine import _default_playbooks

        return {str(p.get("id")) for p in _default_playbooks() if p.get("id")}
    except Exception:
        return set()


def _clean_nodes(nodes_raw) -> tuple[list[dict], list[str]]:
    """Validate + clean a nodes list. Returns (clean_nodes, errors)."""
    errors: list[str] = []
    if not isinstance(nodes_raw, list) or not nodes_raw:
        return [], ["playbook requires a non-empty nodes list"]
    clean_nodes: list[dict] = []
    node_ids: set[str] = set()
    for raw in nodes_raw:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("tool"):
            errors.append("each node requires id and tool")
            continue
        node_id = str(raw["id"])[:64]
        if not _valid_id(node_id):
            errors.append(f"invalid node id: {node_id}")
            continue
        if node_id in node_ids:
            errors.append(f"duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)
        node = {k: v for k, v in raw.items() if not k.startswith("_")}
        node["id"] = node_id
        node["tool"] = str(node["tool"])[:120]
        if "args" not in node or not isinstance(node.get("args"), dict):
            if "args" in node:
                errors.append(f"args must be an object for node {node_id}")
                continue
            node["args"] = {}
        node["depends_on"] = [str(dep) for dep in (node.get("depends_on") or [])]
        clean_nodes.append(node)
    for node in clean_nodes:
        for dep in node["depends_on"]:
            if dep not in node_ids:
                errors.append(f"unknown dependency for {node['id']}: {dep}")
        for field in ("loop_to", "fallback_to"):
            if node.get(field) and str(node[field]) not in node_ids:
                errors.append(f"unknown {field} for {node['id']}: {node[field]}")
    # Cycle check (depends_on only; loop_to is intentionally cyclic)
    if not errors:
        visiting: set[str] = set()
        visited: set[str] = set()
        adj = {n["id"]: list(n.get("depends_on") or []) for n in clean_nodes}

        def _visit(nid: str, stack: list[str]) -> bool:
            if nid in visiting:
                errors.append(f"cycle detected: {' -> '.join(stack + [nid])}")
                return True
            if nid in visited:
                return False
            visiting.add(nid)
            for dep in adj.get(nid, []):
                if _visit(dep, stack + [nid]):
                    return True
            visiting.discard(nid)
            visited.add(nid)
            return False

        for nid in adj:
            if _visit(nid, []):
                break
    # Unknown-tool check (warning, not fatal — validated separately too)
    try:
        from agentic.registry import registry

        known = set(registry.get_all_tool_names())
        # Also accept graph-engine built-ins registered lazily
        for node in clean_nodes:
            if node["tool"] not in known:
                errors.append(f"unknown tool for {node['id']}: {node['tool']}")
    except Exception:
        pass
    return clean_nodes, errors


def _write_playbook_entry(playbook_id: str, clean: dict) -> Path:
    from agentic.graph_engine import _playbook_file, _playbook_write_guard

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
    return path


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


@app.get("/api/tools")
def list_tools():
    """Palette source: registry tools grouped by domain (Jetson-light, cached)."""
    try:
        import agentic.tools  # noqa: F401 — ensure all toolkit modules registered
        from agentic.registry import registry

        groups: dict[str, list[dict]] = {}
        for spec in sorted(registry.all_specs(), key=lambda s: s.name):
            if not spec.graph and not spec.react:
                continue
            entry = {"name": spec.name, "description": (spec.description or "")[:160],
                     "domain": spec.domain or "general", "graph": spec.graph, "react": spec.react,
                     "needs_approval": spec.needs_approval,
                     "args": list((spec.props or {}).keys())[:8]}
            groups.setdefault(entry["domain"], []).append(entry)
        return {"groups": groups, "count": sum(len(v) for v in groups.values())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tool list failed: {exc}") from exc


@app.get("/api/playbooks/{playbook_id}")
def get_playbook(playbook_id: str):
    """Get a specific playbook by ID (refreshed)."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    for playbook in PLAYBOOKS:
        if playbook.get("id") == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.post("/api/playbooks")
async def create_playbook(request: Request):
    """Create a new user graph (n8n-style: Workflow -> New)."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="object required")
    playbook_id = str(payload.get("id") or "").strip()
    if not _valid_id(playbook_id):
        raise HTTPException(status_code=400, detail="id must match ^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
    existing_ids = {str(p.get("id")) for p in load_playbooks_refresh()}
    if playbook_id in existing_ids:
        raise HTTPException(status_code=409, detail=f"playbook already exists: {playbook_id}")
    name = str(payload.get("name") or playbook_id)[:120]
    goal = str(payload.get("goal") or payload.get("name") or playbook_id)[:500]
    nodes_raw = payload.get("nodes") or [{"id": "start", "tool": "make_plan", "args": {"goal": "$prompt"}}]
    clean_nodes, errors = _clean_nodes(nodes_raw)
    # For CREATE, tolerate unknown-tool errors (palette may lag) but not structural ones
    structural = [e for e in errors if not e.startswith("unknown tool")]
    if structural:
        raise HTTPException(status_code=400, detail="; ".join(structural[:5]))
    clean = {"id": playbook_id, "name": name, "goal": goal,
             "triggers": list(payload.get("triggers") or []),
             "requires_any": list(payload.get("requires_any") or []),
             "capabilities": list(payload.get("capabilities") or []),
             "source": "studio", "nodes": clean_nodes}
    path = _write_playbook_entry(playbook_id, clean)
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path)}


@app.put("/api/playbooks/{playbook_id}")
async def save_playbook(playbook_id: str, request: Request):
    """Persist an edited playbook to the user-scoped source used by execution."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("id") != playbook_id:
        raise HTTPException(status_code=400, detail="payload id must match playbook id")
    clean_nodes, errors = _clean_nodes(payload.get("nodes"))
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors[:5]))

    clean = {k: v for k, v in payload.items() if k not in {"edges", "nodes"} and not k.startswith("_")}
    clean["id"] = playbook_id
    clean["nodes"] = clean_nodes
    path = _write_playbook_entry(playbook_id, clean)
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path)}


@app.delete("/api/playbooks/{playbook_id}")
def delete_playbook(playbook_id: str):
    """Delete a user graph. Built-in defaults are protected (clone instead)."""
    from agentic.graph_engine import _playbook_file, _playbook_write_guard

    if playbook_id in _builtin_ids():
        raise HTTPException(status_code=400, detail="built-in playbook — duplicate it first, then edit the copy")
    path = _playbook_file()
    with _playbook_write_guard(path):
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"failed to read playbook source: {exc}") from exc
        if not isinstance(existing, list):
            existing = []
        kept = [p for p in existing if not (isinstance(p, dict) and p.get("id") == playbook_id)]
        if len(kept) == len(existing):
            raise HTTPException(status_code=404, detail="playbook not found in user store")
        tmp = path.with_suffix(path.suffix + ".studio.tmp")
        tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    return {"ok": True, "deleted": playbook_id}


@app.post("/api/playbooks/{playbook_id}/duplicate")
async def duplicate_playbook(playbook_id: str, request: Request):
    """Clone a graph (including built-ins) under a new id."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_id = str((body or {}).get("new_id") or f"{playbook_id}_copy").strip()
    if not _valid_id(new_id):
        raise HTTPException(status_code=400, detail="new_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    src = current.get(playbook_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if new_id in current:
        raise HTTPException(status_code=409, detail=f"playbook already exists: {new_id}")
    clone = {k: v for k, v in src.items() if not k.startswith("_") and k != "edges"}
    clone["id"] = new_id
    clone["name"] = f"{src.get('name') or playbook_id} (copy)"
    clone["source"] = "studio"
    if not isinstance(clone.get("nodes"), list) or not clone["nodes"]:
        raise HTTPException(status_code=400, detail="source has no editable nodes (Spec graph?)")
    clean_nodes, errors = _clean_nodes(clone["nodes"])
    structural = [e for e in errors if not e.startswith("unknown tool")]
    if structural:
        raise HTTPException(status_code=400, detail="; ".join(structural[:5]))
    clone["nodes"] = clean_nodes
    path = _write_playbook_entry(new_id, clone)
    return {"ok": True, "playbook": playbook_to_graph(clone), "path": str(path)}


@app.post("/api/playbooks/{playbook_id}/validate")
def validate_playbook(playbook_id: str):
    """Structural check without executing: cycles, deps, unknown tools."""
    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    pb = current.get(playbook_id)
    if pb is None:
        raise HTTPException(status_code=404, detail="Playbook not found")
    _, errors = _clean_nodes(pb.get("nodes"))
    entry_points = [n["id"] for n in (pb.get("nodes") or []) if isinstance(n, dict) and not (n.get("depends_on") or [])]
    warnings: list[str] = []
    if not entry_points:
        warnings.append("no entry node (every node has dependencies)")
    if len(pb.get("nodes") or []) > 20:
        warnings.append("large graph (>20 nodes) — consider splitting for Ministral-3B context")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "nodes": len(pb.get("nodes") or []), "entry_points": entry_points}


@app.post("/api/playbooks/{playbook_id}/run")
async def run_playbook_dry(request: Request, playbook_id: str):
    """Bounded dry-run: execute the saved graph with Jetson-safe limits.

    Body: {"prompt": "...", "timeout_s": 60}. Approval-gated tools keep
    their normal refusal behaviour — the run still reports per-node results.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    prompt = str((body or {}).get("prompt") or f"Studio dry-run of {playbook_id}")[:1500]
    timeout_s = max(10, min(int((body or {}).get("timeout_s") or 60), 120))
    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    pb = current.get(playbook_id)
    if pb is None:
        raise HTTPException(status_code=404, detail="Playbook not found")
    clean_nodes, errors = _clean_nodes(pb.get("nodes"))
    if [e for e in errors if not e.startswith("unknown tool")]:
        raise HTTPException(status_code=400, detail="graph has structural errors — validate first")
    try:
        from agentic.graph_engine import PlanGraph, PlanNode, execute_graph
        import concurrent.futures

        nodes = []
        for n in clean_nodes:
            nodes.append(PlanNode(id=n["id"], tool=n["tool"], args=dict(n.get("args") or {}),
                                  depends_on=tuple(n.get("depends_on") or ()),
                                  loop_to=n.get("loop_to"), max_visits=int(n.get("max_visits") or 1),
                                  fallback_to=n.get("fallback_to")))
        graph = PlanGraph(id=playbook_id, name=str(pb.get("name") or playbook_id),
                          goal=str(pb.get("goal") or prompt[:300]), nodes=tuple(nodes), source="studio-dryrun")
        # Substitute $prompt in args (same convention as the engine)
        subbed = []
        for n in graph.nodes:
            args = {k: (str(v).replace("$prompt", prompt) if isinstance(v, str) else v) for k, v in n.args.items()}
            from dataclasses import replace as _replace

            subbed.append(_replace(n, args=args))
        graph = PlanGraph(id=graph.id, name=graph.name, goal=graph.goal, nodes=tuple(subbed), source=graph.source)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(execute_graph, graph)
            result = fut.result(timeout=timeout_s)
        return {"ok": True, "graph": playbook_id,
                "final_answer": (result.final_answer or "")[:3000],
                "goal_score": result.goal_score, "goal_reasons": result.goal_reasons,
                "nodes": [{"id": r.node_id, "tool": r.tool, "ok": r.ok,
                           "error": r.error_type, "content": r.content[:800]} for r in result.results]}
    except concurrent.futures.TimeoutError:
        raise HTTPException(status_code=504, detail=f"dry-run exceeded {timeout_s}s — Jetson cap")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dry-run failed: {exc}") from exc


@app.get("/")
async def serve_studio(request: Request):
    """Serve the studio interface (static SPA; no Jinja needed)."""
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
