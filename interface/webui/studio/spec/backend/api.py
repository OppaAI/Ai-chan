"""Spec Studio backend — Layer 5 unified Graph + Spec studio.

- All playbooks (same as DAG Studio) via /api/playbooks
- Spec load/validate/preview/save for Spec-backed workflows
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

app = FastAPI(title="Aiko Spec Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8010", "http://127.0.0.1:8010"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "_shared"
WORKFLOWS_ROOT = Path(__file__).resolve().parents[5] / "agentic" / "workflows"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="spec-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

_WORKFLOW_META: dict[str, dict[str, str]] = {
    "job_hunt": {
        "graph_id": "gen_job_post",
        "name": "Job hunt (shared nodes)",
        "goal": "Fetch job listings, draft posts, save for human review",
        "workflow_id": "job_hunt",
    },
    "aurora_forecast": {
        "graph_id": "aurora_forecast",
        "name": "Aurora forecast (shared nodes)",
        "goal": "Check aurora visibility, store forecast, and notify when warranted",
        "workflow_id": "aurora_forecast",
    },
}


def _workflow_dir(workflow_key: str) -> Path:
    if workflow_key not in _WORKFLOW_META:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_key!r}")
    path = WORKFLOWS_ROOT / workflow_key
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"workflow package missing: {workflow_key}")
    return path


def _plan_graph_to_dict(graph) -> dict[str, Any]:
    nodes = []
    edges = []
    for n in graph.nodes:
        nodes.append(
            {
                "id": n.id,
                "tool": n.tool,
                "args": dict(n.args or {}),
                "depends_on": list(n.depends_on or ()),
                "loop_to": getattr(n, "loop_to", None),
                "max_visits": getattr(n, "max_visits", None),
            }
        )
        for dep in n.depends_on or ():
            edges.append({"id": f"{dep}->{n.id}", "source": dep, "target": n.id, "type": "depends_on"})
        loop_to = getattr(n, "loop_to", None)
        if loop_to:
            edges.append(
                {"id": f"{n.id}-loop->{loop_to}", "source": n.id, "target": loop_to, "type": "loop_to"}
            )
    return {
        "id": graph.id,
        "name": graph.name,
        "goal": graph.goal,
        "source": getattr(graph, "source", None),
        "nodes": nodes,
        "edges": edges,
    }


def playbook_to_graph(playbook: dict) -> dict:
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
            edges.append(
                {
                    "source": dep,
                    "target": nid,
                    "type": "depends_on",
                    "tool_call": tool_call,
                    "skill": src_node.get("tool"),
                }
            )
        if n.get("loop_to"):
            edges.append(
                {
                    "source": nid,
                    "target": n["loop_to"],
                    "type": "loop_to",
                    "tool_call": tool_call,
                    "skill": src_node.get("tool"),
                }
            )
        if n.get("fallback_to"):
            edges.append(
                {
                    "source": nid,
                    "target": n["fallback_to"],
                    "type": "fallback_to",
                    "tool_call": tool_call,
                    "skill": src_node.get("tool"),
                }
            )
    return {**playbook, "nodes": nodes, "edges": edges}


def load_playbooks_refresh() -> list:
    from agentic.graph_engine import load_playbooks

    raw = load_playbooks()
    return [playbook_to_graph(p) for p in raw]


@app.get("/api/health")
def health():
    return {"ok": True, "service": "spec-studio", "layer": 5}


@app.get("/api/playbooks")
def get_playbooks():
    try:
        return load_playbooks_refresh()
    except Exception as exc:
        log.warning("playbooks load failed: %s", exc)
        return []


@app.get("/api/playbooks/{playbook_id}")
def get_playbook(playbook_id: str):
    try:
        books = load_playbooks_refresh()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    for playbook in books:
        if playbook.get("id") == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.get("/api/workflows")
def list_workflows():
    items = []
    for key, meta in _WORKFLOW_META.items():
        d = WORKFLOWS_ROOT / key
        has_spec = (d / "spec.json").is_file() if d.is_dir() else False
        has_config = (d / "config.json").is_file() if d.is_dir() else False
        items.append(
            {
                "id": key,
                "graph_id": meta["graph_id"],
                "name": meta["name"],
                "has_spec_json": has_spec,
                "has_config_json": has_config,
                "spec_source": "spec.json" if has_spec else ("config.json" if has_config else None),
            }
        )
    registered: list[str] = []
    try:
        from agentic.workflows.common.graphs import list_graphs

        registered = list_graphs()
    except Exception as exc:
        log.debug("list_graphs failed: %s", exc)
    return {"workflows": items, "registered_graphs": registered}


@app.get("/api/workflows/{workflow_id}/spec")
def get_workflow_spec(workflow_id: str):
    from agentic.workflows.common.spec import load_spec_for_workflow

    meta = _WORKFLOW_META.get(workflow_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_id!r}")
    d = _workflow_dir(workflow_id)
    try:
        spec = load_spec_for_workflow(
            d,
            graph_id=meta["graph_id"],
            name=meta["name"],
            goal=meta["goal"],
            workflow_id=meta["workflow_id"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    source = "spec.json" if (d / "spec.json").is_file() else "config.json"
    return {"workflow_id": workflow_id, "source": source, "spec": spec.to_dict()}


@app.post("/api/validate")
async def validate_spec_body(request: Request):
    from agentic.workflows.common.spec import SpecError, validate_spec

    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    payload = raw.get("spec") if isinstance(raw.get("spec"), dict) else raw
    try:
        spec = validate_spec(payload)
    except SpecError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "spec": spec.to_dict()}


@app.post("/api/preview")
async def preview_graph(request: Request):
    from agentic.workflows.common.spec import SpecError, validate_spec
    from agentic.workflows.common.spec_graph import build_plan_graph

    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    payload = raw.get("spec") if isinstance(raw.get("spec"), dict) else raw
    goal = raw.get("goal") if isinstance(raw, dict) else None
    try:
        spec = validate_spec(payload)
        graph = build_plan_graph(spec, goal=goal if isinstance(goal, str) else None)
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "graph": _plan_graph_to_dict(graph)}


@app.put("/api/workflows/{workflow_id}/spec")
async def save_workflow_spec(workflow_id: str, request: Request):
    from agentic.workflows.common.spec import SpecError, validate_spec

    if workflow_id not in _WORKFLOW_META:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_id!r}")
    d = _workflow_dir(workflow_id)
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    payload = raw.get("spec") if isinstance(raw.get("spec"), dict) else raw
    try:
        spec = validate_spec(payload)
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    meta = _WORKFLOW_META[workflow_id]
    data = spec.to_dict()
    data["id"] = meta["graph_id"]
    data["workflow_id"] = meta["workflow_id"]
    if not data.get("name"):
        data["name"] = meta["name"]

    out = d / "spec.json"
    try:
        fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".spec.json.", suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to write {out}: {exc}") from exc
    return {"ok": True, "path": str(out.relative_to(WORKFLOWS_ROOT.parent.parent)), "spec": data}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SPEC_STUDIO_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8010)
