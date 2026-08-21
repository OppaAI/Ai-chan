#!/usr/bin/env python3
"""Apply schedule + Spec playbook fixes (PR companion for large-file edits).

Run from repo root on branch fix/schedule-spec-playbooks or after pull:
  python3 scripts/apply_schedule_spec_playbooks.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def patch_schedule() -> None:
    path = ROOT / "system/schedule.py"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    changed = False

    old = (
        '        playbook = get_playbook_by_id(graph_id)\n'
        '        if playbook is None:\n'
        '            log.warning("Schedule graph %r references unknown playbook %r — skipping", graph_def.get("id"), graph_id)\n'
        '            return\n'
    )
    new = '        playbook = get_playbook_by_id(graph_id) or {}\n'
    if old in text:
        text = text.replace(old, new, 1)
        changed = True

    old_goal = 'goal=playbook.get("goal", f"Scheduled run: {graph_id}"),'
    new_goal = 'goal=playbook.get("goal") or registered_graph.goal or f"Scheduled run: {graph_id}",'
    if old_goal in text and "registered_graph.goal" not in text:
        text = text.replace(old_goal, new_goal, 1)
        changed = True

    old_skip = 'log.warning("Playbook %r has no valid nodes — skipping", graph_id)'
    new_skip = (
        'log.warning(\n'
        '                    "Playbook %r has no valid nodes and no registered Spec graph — skipping",\n'
        '                    graph_id,\n'
        '                )'
    )
    if old_skip in text:
        text = text.replace(old_skip, new_skip, 1)
        changed = True

    if "def disable_legacy_job_post_tool_jobs" not in text:
        marker = "def ensure_weekly_social_job"
        helper = '''
def disable_legacy_job_post_tool_jobs(user_id: str | None = None) -> None:
    """Disable schedule.json tool jobs that call run_job_post_playbook."""
    path = schedule_path(user_id=user_id)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, list):
        return
    changed = False
    for job in data:
        if not isinstance(job, dict):
            continue
        tc = job.get("tool_call") or {}
        name = str(tc.get("name") or "").strip()
        if name != "run_job_post_playbook":
            continue
        if job.get("enabled", True):
            job["enabled"] = False
            changed = True
            log.info(
                "Disabled legacy schedule tool job %r (use schedule_graphs gen_job_post instead)",
                job.get("id") or job.get("title"),
            )
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")


'''
        if marker not in text:
            raise SystemExit("ensure_weekly_social_job not found")
        text = text.replace(marker, helper + marker, 1)
        changed = True

    log_line = 'log.info("Registered social handlers and seeded social jobs; Lane D uses schedule_graphs.json.")'
    if log_line in text and "disable_legacy_job_post_tool_jobs(user_id=user_id)" not in text:
        text = text.replace(log_line, "disable_legacy_job_post_tool_jobs(user_id=user_id)\n    " + log_line, 1)
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print("patched", path)
    else:
        print("schedule already patched or patterns missing")


def patch_graph_engine() -> None:
    path = ROOT / "agentic/graph_engine.py"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    changed = False

    if '"id": "aurora_forecast"' not in text:
        needle = '            "graph_id": "gen_job_post",\n            "nodes": [],\n        },\n    ]'
        insert = '''            "graph_id": "gen_job_post",
            "pipeline": "shared_5",
            "nodes": [],
        },
        {
            "id": "aurora_forecast",
            "name": "Hourly aurora visibility forecast (NOAA + Kp + clouds)",
            "triggers": [
                "aurora", "northern lights", "aurora forecast", "kp index", "aurora alert",
            ],
            "semantic_triggers": [
                "check if the aurora is visible tonight",
                "aurora forecast and cloud cover",
                "northern lights probability",
            ],
            "requires_any": ["aurora", "northern", "kp", "geomagnetic"],
            "capabilities": ["weather", "research"],
            "graph_id": "aurora_forecast",
            "pipeline": "shared_5",
            "nodes": [],
        },
    ]'''
        if needle not in text:
            raise SystemExit("gen_job_post tail not found in graph_engine")
        text = text.replace(needle, insert, 1)
        changed = True

    if "list_graphs" not in text or "Ensure every registered Spec" not in text:
        old = '''def load_playbooks() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in _default_playbooks():
        pid = p.get("id")
        if isinstance(p, dict) and pid:
            by_id[str(pid)] = dict(p)
'''
        new = '''def load_playbooks() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in _default_playbooks():
        pid = p.get("id")
        if isinstance(p, dict) and pid:
            by_id[str(pid)] = dict(p)

    # Ensure every registered Spec/shared_5 PlanGraph is visible as a playbook
    try:
        from agentic.workflows.common.graphs import list_graphs, get_graph
        for gid in list_graphs():
            if gid in by_id:
                continue
            g = get_graph(gid)
            if g is None:
                continue
            by_id[gid] = {
                "id": gid,
                "name": g.name or gid,
                "goal": g.goal or "",
                "graph_id": gid,
                "pipeline": "shared_5",
                "nodes": [],
                "triggers": [],
                "capabilities": [],
            }
    except Exception:
        pass
'''
        if old not in text:
            raise SystemExit("load_playbooks header not found")
        text = text.replace(old, new, 1)
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print("patched", path)
    else:
        print("graph_engine already patched")


def patch_social() -> None:
    path = ROOT / "agentic/toolkit/social.py"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    changed = False

    if 'get_graph("gen_job_post")' not in text:
        old = '''def _run_gen_job_post_playbook(
    *,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """Load and execute the gen_job_post playbook from the shared playbook system."""
    from agentic.graph_engine import get_playbook_by_id, PlanNode, PlanGraph, execute_graph
    playbook = get_playbook_by_id("gen_job_post")
    if playbook is None:
        return {"success": False, "error": "gen_job_post playbook not found"}
    nodes = []
    for raw in playbook.get("nodes", []):
        if isinstance(raw, dict) and raw.get("id") and raw.get("tool"):
            nodes.append(PlanNode(
                id=str(raw["id"]), tool=str(raw["tool"]),
                args=dict(raw.get("args", {})),
                depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
            ))
    graph = PlanGraph(
        id="gen_job_post", name=playbook.get("name", "Job Post"),
        goal="Draft job posts from config", nodes=tuple(nodes),
    )
    try:
        result = execute_graph(graph, llm_client=client, llm_model=model)
        return {
            "success": all(r.ok for r in result.results),
            "graph_id": result.graph.id,
            "results": [{"node": r.node_id, "tool": r.tool, "ok": r.ok, "error_type": r.error_type} for r in result.results],
            "final_answer": result.final_answer,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
'''
        new = '''def _run_gen_job_post_playbook(
    *,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """Execute Lane D via the registered Spec/shared_5 gen_job_post graph."""
    from agentic.graph_engine import get_playbook_by_id, PlanNode, PlanGraph, execute_graph

    graph = None
    try:
        from agentic.workflows.common.graphs import get_graph
        graph = get_graph("gen_job_post")
    except Exception:
        graph = None

    if graph is None:
        playbook = get_playbook_by_id("gen_job_post")
        if playbook is None:
            return {"success": False, "error": "gen_job_post playbook/graph not found"}
        nodes = []
        for raw in playbook.get("nodes", []):
            if isinstance(raw, dict) and raw.get("id") and raw.get("tool"):
                nodes.append(PlanNode(
                    id=str(raw["id"]), tool=str(raw["tool"]),
                    args=dict(raw.get("args", {})),
                    depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
                ))
        if not nodes:
            return {"success": False, "error": "gen_job_post has no registered graph and no nodes"}
        graph = PlanGraph(
            id="gen_job_post", name=playbook.get("name", "Job Post"),
            goal="Draft job posts from config", nodes=tuple(nodes),
        )
    try:
        result = execute_graph(graph, llm_client=client, llm_model=model)
        return {
            "success": all(r.ok for r in result.results),
            "graph_id": result.graph.id,
            "results": [{"node": r.node_id, "tool": r.tool, "ok": r.ok, "error_type": r.error_type} for r in result.results],
            "final_answer": result.final_answer,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
'''
        if old not in text:
            raise SystemExit("_run_gen_job_post_playbook not found")
        text = text.replace(old, new, 1)
        changed = True

    if '@tool("run_job_post_playbook")' not in text:
        anchor = '@tool(TOOLS["draft_job_post_social"])\n'
        extra = '''@tool("run_job_post_playbook")
def run_job_post_playbook(*, client=None, model: str | None = None) -> dict[str, Any]:
    """Legacy schedule tool name → Spec gen_job_post graph."""
    return _run_gen_job_post_playbook(client=client, model=model)


'''
        if anchor not in text:
            raise SystemExit("draft_job_post_social anchor not found")
        text = text.replace(anchor, extra + anchor, 1)
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print("patched", path)
    else:
        print("social already patched")


def main() -> int:
    patch_schedule()
    patch_graph_engine()
    patch_social()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
