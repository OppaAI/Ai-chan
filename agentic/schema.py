"""
agentic/schema.py

Graph-first, mostly model-free agentic executor.

This module is intentionally conservative: it only handles workflows that can be
matched to a known playbook template and whose tool arguments can be derived
from the user's prompt with deterministic heuristics. Novel/ambiguous tasks
return ``None`` so the normal ReAct loop can run once and record experience for
future promotion into the playbook.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from system.config import load_config
load_config()

from system.log import get_logger
from system.userspace import current_user_id, user_state_dir

log = get_logger(__name__)


GRAPH_AGENT_ENABLED = os.getenv("GRAPH_AGENT_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
GRAPH_AGENT_PLAYBOOK = os.getenv("GRAPH_AGENT_PLAYBOOK", "agentic/playbook.json")
GRAPH_MAX_WORKERS = int(os.getenv("GRAPH_MAX_WORKERS", "4"))

# Kept in sync with agentic.py's AGENT_NOTE_MAX_CHARS so a note saved via the
# graph executor can't end up longer than one saved via the ReAct path.
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", "5000"))

# Tools whose toolkit implementations accept an `embedder` kwarg for
# semantic scoring. dispatch_tool() in agentic.py passes the shared Harrier
# embedder for these; the graph executor previously never did, so any
# RAG-style scoring inside these tools silently degraded to keyword
# fallback when run through the graph path instead of ReAct.
_EMBEDDER_AWARE_TOOLS = {"deep_search", "deep_research"}
_TOOL_MAP_CACHE: dict[str, Callable[..., Any]] | None = None
_TOOL_MAP_LOCK = threading.Lock()
_PLAYBOOK_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PlanNode:
    id: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanGraph:
    id: str
    name: str
    goal: str
    nodes: tuple[PlanNode, ...]
    source: str = "playbook"


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    tool: str
    ok: bool
    content: str
    args: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    def summary(self, max_chars: int = 700) -> str:
        status = "ok" if self.ok else self.error_type or "failed"
        body = re.sub(r"\s+", " ", self.content or "").strip()[:max_chars]
        return f"{self.node_id}:{self.tool}[{status}] {body}".strip()


@dataclass(frozen=True)
class GraphRunResult:
    graph: PlanGraph
    results: tuple[NodeResult, ...]
    final_answer: str

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": r.tool,
                "ok": r.ok,
                "error_type": r.error_type,
                "args": r.args,
            }
            for r in self.results
        ]



@contextlib.contextmanager
def _playbook_write_guard(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _PLAYBOOK_WRITE_LOCK:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except ImportError:
                yield

def _playbook_file() -> Path:
    raw = Path(GRAPH_AGENT_PLAYBOOK)
    if raw.is_absolute():
        return raw
    return user_state_dir(current_user_id()) / raw


def _default_playbooks() -> list[dict[str, Any]]:
    """Built-in starter plans. User-promoted plans are appended on disk."""
    return [
        {
            "id": "plan_and_save_note",
            "name": "Plan and save note",
            "triggers": ["plan", "make a plan", "outline"],
            "requires_any": ["save", "note", "document", "write it down"],
            "nodes": [
                {"id": "plan", "tool": "make_plan", "args": {"goal": "$prompt", "max_steps": 8}},
                {"id": "save", "tool": "save_note", "depends_on": ["plan"], "args": {"title": "$title", "content": "$result:plan", "folder": "notes"}},
            ],
        },
        {
            "id": "checklist_and_save",
            "name": "Checklist and save note",
            "triggers": ["checklist", "todo", "to-do", "steps"],
            "requires_any": ["save", "note", "checklist"],
            "nodes": [
                {"id": "checklist", "tool": "create_checklist", "args": {"title": "$title", "items": "$heuristic_items"}},
                {"id": "save", "tool": "save_note", "depends_on": ["checklist"], "args": {"title": "$title", "content": "$result:checklist", "folder": "notes"}},
            ],
        },
        {
            "id": "research_and_save",
            "name": "Search and save brief note",
            "triggers": ["search", "research", "look up", "find"],
            "requires_any": ["save", "note", "recommendation", "report", "summary"],
            "nodes": [
                {"id": "search", "tool": "deep_search", "args": {"query": "$prompt"}},
                {"id": "save", "tool": "save_note", "depends_on": ["search"], "args": {"title": "$title", "content": "$result:search", "folder": "notes"}},
            ],
        },
        {
            "id": "simple_save_note",
            "name": "Save provided text as a note",
            "triggers": ["save note", "write note", "draft", "note"],
            "requires_any": ["save", "note", "draft"],
            "nodes": [
                {"id": "save", "tool": "save_note", "args": {"title": "$title", "content": "$prompt", "folder": "notes"}},
            ],
        },
    ]


def load_playbooks() -> list[dict[str, Any]]:
    path = _playbook_file()
    plans = _default_playbooks()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                plans.extend(p for p in data if isinstance(p, dict))
        except Exception as exc:
            log.warning("failed to load graph playbooks from %s: %s", path, exc)
    return plans


def _score_plan(plan: dict[str, Any], prompt: str, cap_ids: list[str] | None = None) -> int:
    text = prompt.casefold()
    triggers = [str(t).casefold() for t in plan.get("triggers", [])]
    required = [str(t).casefold() for t in plan.get("requires_any", [])]
    score = sum(3 for t in triggers if t and t in text)
    if required and any(t in text for t in required):
        score += 1
    domains = set(plan.get("capabilities", []))
    if cap_ids and domains.intersection(cap_ids):
        score += 3
    return score


def _title(prompt: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", prompt).strip()
    words = cleaned.split()[:8]
    return " ".join(words) or "Aiko task"


def _heuristic_items(prompt: str) -> list[str]:
    parts = re.split(r"(?:,|;|\band\b|\n)+", prompt)
    items = [p.strip(" .:-") for p in parts if len(p.strip()) > 3]
    return items[:10] or [prompt.strip()]


def _substitute(value: Any, prompt: str, results: dict[str, NodeResult]) -> Any:
    if isinstance(value, str):
        if value == "$prompt":
            return prompt
        if value == "$title":
            return _title(prompt)
        if value == "$heuristic_items":
            return _heuristic_items(prompt)
        if value.startswith("$result:"):
            node_id = value.split(":", 1)[1]
            return (results.get(node_id).content if results.get(node_id) else "")[:4000]
        return value.replace("$prompt", prompt).replace("$title", _title(prompt))
    if isinstance(value, list):
        return [_substitute(v, prompt, results) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, prompt, results) for k, v in value.items()}
    return value


def plan_from_master(user_input: str, cap_ids: list[str] | None = None) -> PlanGraph | None:
    if not GRAPH_AGENT_ENABLED:
        return None
    plans = load_playbooks()
    ranked = sorted(((_score_plan(p, user_input, cap_ids), p) for p in plans), key=lambda x: x[0], reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return None
    plan = ranked[0][1]
    nodes = []
    for raw in plan.get("nodes", []):
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("tool"):
            return None
        nodes.append(PlanNode(
            id=str(raw["id"]),
            tool=str(raw["tool"]),
            args=dict(raw.get("args") or {}),
            depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
        ))
    if not nodes:
        return None
    return PlanGraph(id=str(plan.get("id") or uuid.uuid4()), name=str(plan.get("name") or plan.get("id") or "workflow"), goal=user_input, nodes=tuple(nodes))


def _tool_map() -> dict[str, Callable[..., Any]]:
    global _TOOL_MAP_CACHE
    if _TOOL_MAP_CACHE is not None:
        return _TOOL_MAP_CACHE
    with _TOOL_MAP_LOCK:
        if _TOOL_MAP_CACHE is not None:
            return _TOOL_MAP_CACHE
        _TOOL_MAP_CACHE = _build_tool_map()
        return _TOOL_MAP_CACHE


def _build_tool_map() -> dict[str, Callable[..., Any]]:
    # Import focused toolkit modules lazily so model-free graph planning can be
    # imported/tested without loading optional heavy research dependencies.
    from agentic.toolkit.plan import make_plan, create_checklist, save_note, read_workspace_file, summarize_task_state
    mapping: dict[str, Callable[..., Any]] = {
        "make_plan": make_plan,
        "create_checklist": create_checklist,
        "save_note": save_note,
        "read_workspace_file": read_workspace_file,
        "summarize_task_state": summarize_task_state,
    }
    try:
        from agentic.toolkit.organize import schedule_job, list_schedule, cancel_schedule, schedule_reminder, list_reminders, cancel_reminder
        mapping.update({
            "schedule_job": schedule_job, "list_schedule": list_schedule, "cancel_schedule": cancel_schedule,
            "schedule_reminder": schedule_reminder, "list_reminders": list_reminders, "cancel_reminder": cancel_reminder,
        })
    except Exception as exc:
        log.debug("organize tools unavailable for graph executor: %s", exc)
    try:
        from agentic.toolkit.research import deep_search, deep_research
        mapping.update({"deep_search": deep_search, "deep_research": deep_research})
    except Exception as exc:
        log.debug("research tools unavailable for graph executor: %s", exc)
    try:
        from agentic.toolkit.photography import scan_photo_workspace, propose_photo_ingestion, write_photo_ingestion_report
        mapping.update({
            "scan_photo_workspace": scan_photo_workspace, "propose_photo_ingestion": propose_photo_ingestion,
            "write_photo_ingestion_report": write_photo_ingestion_report,
        })
    except Exception as exc:
        log.debug("photo tools unavailable for graph executor: %s", exc)
    try:
        # draft_*/post_* wrappers mirror what agentic/agentic.py already
        # registers for ReAct — see agentic/toolkit/social.py's module docstring.
        # post_photo_social/post_video_social still enforce human approval
        # internally (SocialApprovalError via _require_approved); adding
        # them here only lets a matched/promoted playbook reach the same
        # functions ReAct can already reach, it does not relax that gate.
        from agentic.toolkit.social import draft_photo_social, post_photo_social, draft_video_social, post_video_social
        mapping.update({
            "draft_photo_social": draft_photo_social, "post_photo_social": post_photo_social,
            "draft_video_social": draft_video_social, "post_video_social": post_video_social,
        })
    except Exception as exc:
        log.debug("social tools unavailable for graph executor: %s", exc)
    try:
        from agentic.toolkit.self_improve import repo_file_tree, repo_read_file, repo_search_text
        mapping.update({"repo_file_tree": repo_file_tree, "repo_read_file": repo_read_file, "repo_search_text": repo_search_text})
    except Exception as exc:
        log.debug("repo tools unavailable for graph executor: %s", exc)
    try:
        from agentic.toolkit.job_hunt import search_jobs, dedupe_postings
        mapping.update({"search_jobs": search_jobs, "dedupe_postings": dedupe_postings})
    except Exception as exc:
        log.debug("job tools unavailable for graph executor: %s", exc)
    return mapping


def _run_node(node: PlanNode, prompt: str, results: dict[str, NodeResult], embedder=None) -> NodeResult:
    tools = _tool_map()
    fn = tools.get(node.tool)
    args = _substitute(node.args, prompt, results)
    if fn is None:
        return NodeResult(node.id, node.tool, False, f"unknown graph tool: {node.tool}", args=args, error_type="unknown_tool")
    if node.tool == "save_note":
        args["content"] = str(args.get("content", ""))[:AGENT_NOTE_MAX_CHARS]
    if node.tool in _EMBEDDER_AWARE_TOOLS:
        args["embedder"] = embedder
    try:
        out = fn(**args)
        return NodeResult(node.id, node.tool, True, str(out), args=args)
    except Exception as exc:
        return NodeResult(node.id, node.tool, False, f"{type(exc).__name__}: {exc}", args=args, error_type="execution_error")


def execute_graph(graph: PlanGraph, embedder=None) -> GraphRunResult:
    pending = {node.id: node for node in graph.nodes}
    results: dict[str, NodeResult] = {}
    ordered: list[NodeResult] = []
    with ThreadPoolExecutor(max_workers=GRAPH_MAX_WORKERS) as pool:
        while pending:
            ready = [node for node in pending.values() if all(dep in results for dep in node.depends_on)]
            if not ready:
                stuck = ", ".join(sorted(pending))
                ordered.append(NodeResult("graph", "graph_executor", False, f"dependency cycle or missing dependency among: {stuck}", error_type="dependency_error"))
                break
            runnable, blocked = [], []
            for node in ready:
                if all(results[dep].ok for dep in node.depends_on):
                    runnable.append(node)
                else:
                    blocked.append(node)
            for node in blocked:
                result = NodeResult(node.id, node.tool, False, "skipped: an upstream dependency failed", error_type="dependency_failed")
                results[node.id] = result
                ordered.append(result)
                pending.pop(node.id, None)
            if not runnable:
                continue
            future_map = {pool.submit(_run_node, node, graph.goal, results, embedder): node for node in runnable}
            for fut in as_completed(future_map):
                node = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = NodeResult(node.id, node.tool, False, str(exc), error_type="execution_error")
                results[node.id] = result
                ordered.append(result)
                pending.pop(node.id, None)
    final_answer = _synthesize_without_llm(graph, tuple(ordered))
    return GraphRunResult(graph=graph, results=tuple(ordered), final_answer=final_answer)


def _synthesize_without_llm(graph: PlanGraph, results: tuple[NodeResult, ...]) -> str:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lines = [f"I ran the saved workflow '{graph.name}' without an LLM planning step."]
    if ok:
        lines.append("Completed:")
        lines.extend(f"- {r.summary()}" for r in ok)
    if failed:
        lines.append("Problems:")
        lines.extend(f"- {r.summary()}" for r in failed)
    lines.append("If this workflow was not what you intended, I can fall back to ReAct once and learn the corrected sequence.")
    return "\n".join(lines)


def run_schema_agent(user_input: str, cap_ids: list[str] | None = None, embedder=None) -> GraphRunResult | None:
    graph = plan_from_master(user_input, cap_ids=cap_ids)
    if graph is None:
        return None
    return execute_graph(graph, embedder=embedder)


def list_playbooks_json() -> str:
    """Return graph playbook metadata for tool/schema callers."""
    rows = []
    for plan in load_playbooks():
        rows.append({
            "id": plan.get("id"),
            "name": plan.get("name"),
            "triggers": plan.get("triggers", []),
            "requires_any": plan.get("requires_any", []),
            "nodes": [
                {
                    "id": n.get("id"),
                    "tool": n.get("tool"),
                    "depends_on": n.get("depends_on", []),
                    "arg_keys": sorted((n.get("args") or {}).keys()),
                }
                for n in plan.get("nodes", []) if isinstance(n, dict)
            ],
        })
    return json.dumps({"playbooks": rows}, ensure_ascii=False, indent=2)


def run_playbook_json(task: str, cap_ids: list[str] | None = None, embedder=None) -> str:
    """Run the graph executor and return a compact JSON observation."""
    result = run_schema_agent(task, cap_ids=cap_ids, embedder=embedder)
    if result is None:
        return json.dumps({
            "ok": False,
            "error_type": "no_matching_playbook",
            "task": task,
            "instruction": "Use ReAct once, then record/promote the successful workflow if it should become reusable.",
        }, ensure_ascii=False, indent=2)
    return json.dumps({
        "ok": not any(not r.ok for r in result.results),
        "graph_id": result.graph.id,
        "graph_name": result.graph.name,
        "results": [r.__dict__ for r in result.results],
        "final_answer": result.final_answer,
    }, ensure_ascii=False, indent=2)


def _promotion_args_for_step(tool: str, step: dict[str, Any]) -> dict[str, Any]:
    if tool == "make_plan":
        return {"goal": "$prompt"}
    if tool == "create_checklist":
        return {"title": "$title", "items": "$heuristic_items"}
    if tool == "save_note":
        return {"title": "$title", "content": "$prompt", "folder": "notes"}
    if tool in {"deep_search", "deep_research"}:
        return {"query": "$prompt"}
    args_preview = step.get("args_preview") or {}
    arg_keys = step.get("arg_keys") or sorted((step.get("args") or {}).keys())
    if isinstance(args_preview, dict) and args_preview:
        return {str(k): str(v) for k, v in args_preview.items()}
    return {str(k): "$prompt" for k in arg_keys}


def append_playbook_from_experience(goal: str, steps: list[dict[str, Any]], *, name: str | None = None) -> Path:
    """Promote a practiced or ReAct-discovered tool sequence into user playbooks.

    Args are stored as sanitized previews by the experience layer, so promoted
    templates intentionally use ``$prompt``/``$title`` placeholders unless the
    operator edits the JSON by hand.
    """
    nodes = []
    for idx, step in enumerate(steps, start=1):
        tool = str(step.get("tool") or "").strip()
        if not tool or tool in {"final_answer", "llm_call"}:
            continue
        node = {"id": f"step_{idx}", "tool": tool, "args": _promotion_args_for_step(tool, step)}
        if nodes:
            node["depends_on"] = [nodes[-1]["id"]]
        nodes.append(node)
    if not nodes:
        raise ValueError("no promotable tool steps found")
    path = _playbook_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_plan = {
        "id": f"practiced_{uuid.uuid4().hex[:10]}",
        "name": name or _title(goal),
        "triggers": _heuristic_items(goal)[:4],
        "requires_any": [],
        "nodes": nodes,
    }
    with _playbook_write_guard(path):
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (OSError, json.JSONDecodeError):
                existing = []
        existing.append(new_plan)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    return path
