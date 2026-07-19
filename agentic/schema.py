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
    _extras: dict[str, Any] = field(default_factory=dict)


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
    """Built-in starter plans. User-promoted plans are appended on disk.

    The graph-first research/report flow that Oppa asked for is structured
    as four reusable playbooks so the LLM-facing ReAct path doesn't have
    to invent the same sequence on every prompt:

      - "research_and_report"   (deep_research + KB + synthesize + write_report + learn_knowledge)
      - "search_kb_and_report"  (deep_search + KB + synthesize + write_report + learn_knowledge)
      - "compare_and_report"    (two parallel deep_research calls + KB + comparison synthesize + write_report + learn_knowledge)
      - "checklist_and_save"    (create_checklist + save_note; for explicit checklist asks)
      - "simple_save_note"      (just save the prompt as a note; for plain scratch saves)

    All of them use a `synthesize_report` graph tool that calls the LLM
    through the owner-supplied client+model (see ``run_schema_agent``),
    condense the combined evidence with the shared embedder when it's
    overlong, and default to a professional/formal tone unless the user
    prompt explicitly opts out. Comparisons are only produced when the
    prompt looks like a "A vs B" / "compare A and B" ask; the search
    playbook skips the comparison node entirely.
    """
    return [
        {
            "id": "research_and_report",
            "name": "Deep research, combine, synthesize, and write a report",
            "triggers": [
                "research", "deep research", "in-depth", "in depth",
                "comprehensive", "thorough", "exhaustive", "investigate",
                "study", "analyze", "analysis", "report on", "write a report",
                "give me a report", "summarize", "summary of", "overview of",
            ],
            "requires_any": [],
            "capabilities": ["research"],
            "nodes": [
                {"id": "web",    "tool": "deep_research", "args": {"query": "$prompt"}},
                {"id": "kb",     "tool": "kb_search",     "depends_on": ["web"],    "args": {"query": "$prompt"}},
                {"id": "merge",  "tool": "combine_evidence", "depends_on": ["web", "kb"],
                 "args": {"parts": ["$result:web", "$result:kb"]}},
                {"id": "draft",  "tool": "synthesize_report", "depends_on": ["merge"],
                 "args": {"evidence": "$result:merge", "prompt": "$prompt", "style": "auto"}},
                {"id": "report", "tool": "write_report", "depends_on": ["draft"],
                 "args": {"title": "$title", "content": "$result:draft", "report_dir": "reports"}},
                {"id": "learn",  "tool": "learn_report", "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:draft", "kind": "self_learned"}},
            ],
        },
        {
            "id": "search_kb_and_report",
            "name": "Quick search, combine with KB, synthesize, and write a report",
            "triggers": [
                "search", "look up", "find", "what is", "what are",
                "who is", "when did", "where is", "how do", "how to",
                "quick", "brief on", "tell me about",
            ],
            "requires_any": [],
            "capabilities": ["research"],
            "nodes": [
                {"id": "web",    "tool": "deep_search",  "args": {"query": "$prompt"}},
                {"id": "kb",     "tool": "kb_search",    "depends_on": ["web"],    "args": {"query": "$prompt"}},
                {"id": "merge",  "tool": "combine_evidence", "depends_on": ["web", "kb"],
                 "args": {"parts": ["$result:web", "$result:kb"]}},
                {"id": "draft",  "tool": "synthesize_report", "depends_on": ["merge"],
                 "args": {"evidence": "$result:merge", "prompt": "$prompt", "style": "auto"}},
                {"id": "report", "tool": "write_report", "depends_on": ["draft"],
                 "args": {"title": "$title", "content": "$result:draft", "report_dir": "reports"}},
                {"id": "learn",  "tool": "learn_report", "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:draft", "kind": "self_learned"}},
            ],
        },
        {
            "id": "compare_and_report",
            "name": "Deep research two subjects, combine with KB, synthesize a comparison, and write a report",
            "triggers": [
                "compare", "comparison", "vs", "versus", "vs.", "differences between",
                "difference between", "compared to", "compared with", "contrast",
                "A vs B", "pros and cons",
            ],
            "requires_any": [],
            "capabilities": ["research"],
            "nodes": [
                {"id": "web_a",  "tool": "deep_research", "args": {"query": "$compare_left"}},
                {"id": "web_b",  "tool": "deep_research", "args": {"query": "$compare_right"}},
                {"id": "kb",     "tool": "kb_search",     "depends_on": ["web_a", "web_b"],
                 "args": {"query": "$prompt"}},
                {"id": "merge",  "tool": "combine_evidence", "depends_on": ["web_a", "web_b", "kb"],
                 "args": {"parts": ["$result:web_a", "$result:web_b", "$result:kb"],
                          "separator": "\n\n===\n\n"}},
                {"id": "draft",  "tool": "synthesize_report", "depends_on": ["merge"],
                 "args": {"evidence": "$result:merge", "prompt": "$prompt",
                          "style": "auto", "comparison_subjects": "$compare_subjects"}},
                {"id": "report", "tool": "write_report", "depends_on": ["draft"],
                 "args": {"title": "$title", "content": "$result:draft", "report_dir": "reports"}},
                {"id": "learn",  "tool": "learn_report", "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:draft", "kind": "self_learned"}},
            ],
        },
        {
            "id": "checklist_and_save",
            "name": "Checklist and save note",
            "triggers": ["checklist", "todo", "to-do", "steps to", "how to"],
            "requires_any": ["save", "note", "checklist", "todo", "list"],
            "nodes": [
                {"id": "checklist", "tool": "create_checklist", "args": {"title": "$title", "items": "$heuristic_items"}},
                {"id": "save",      "tool": "save_note", "depends_on": ["checklist"],
                 "args": {"title": "$title", "content": "$result:checklist", "folder": "notes"}},
            ],
        },
        {
            "id": "simple_save_note",
            "name": "Save provided text as a note",
            "triggers": ["save note", "write note", "draft", "note that", "jot down", "save this"],
            "requires_any": ["save", "note", "draft"],
            "nodes": [
                {"id": "save", "tool": "save_note",
                 "args": {"title": "$title", "content": "$prompt", "folder": "notes"}},
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


def _placeholder_extras(prompt: str) -> dict[str, Any]:
    """Compute one-shot placeholder values that aren't per-node:
    compare subjects (left/right/list). Kept as a function so the same
    parsing is shared between plan_from_master and _substitute; this
    also keeps the substitution layer thin.
    """
    out: dict[str, Any] = {}
    try:
        from agentic.toolkit.synthesize import detect_compare, split_subjects
        pair = detect_compare(prompt)
        if pair is not None:
            out["compare_left"] = pair[0]
            out["compare_right"] = pair[1]
        subjects = split_subjects(prompt)
        if subjects:
            out["compare_subjects"] = subjects
    except Exception:
        pass
    return out


def _substitute(value: Any, prompt: str, results: dict[str, NodeResult],
                extras: dict[str, Any] | None = None) -> Any:
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
        if value.startswith("$") and extras and value in extras:
            return extras[value]
        return value.replace("$prompt", prompt).replace("$title", _title(prompt))
    if isinstance(value, list):
        return [_substitute(v, prompt, results, extras) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, prompt, results, extras) for k, v in value.items()}
    return value


def plan_from_master(user_input: str, cap_ids: list[str] | None = None) -> PlanGraph | None:
    if not GRAPH_AGENT_ENABLED:
        return None
    plans = load_playbooks()
    ranked = sorted(((_score_plan(p, user_input, cap_ids), p) for p in plans), key=lambda x: x[0], reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return None
    plan = ranked[0][1]
    # Stash the per-prompt placeholders on the plan for downstream use.
    # We attach to PlanGraph via a private attribute (frozen dataclass
    # doesn't allow new fields) — only this module reads it.
    extras = _placeholder_extras(user_input)
    # If the user prompt doesn't look like a comparison but the matched
    # playbook is compare_and_report, drop it so the wrong playbook
    # doesn't get selected just because "compare" appears in the
    # trigger list as a substring of unrelated text.
    if plan.get("id") == "compare_and_report" and "compare_subjects" not in extras:
        ranked = [(s, p) for s, p in ranked if p is not plan]
        if not ranked or ranked[0][0] <= 0:
            return None
        plan = ranked[0][1]
        extras = _placeholder_extras(user_input)
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
    graph = PlanGraph(
        id=str(plan.get("id") or uuid.uuid4()),
        name=str(plan.get("name") or plan.get("id") or "workflow"),
        goal=user_input,
        nodes=tuple(nodes),
        _extras=extras,
    )
    return graph


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
        # write_report is a long-form markdown writer — formerly ReAct-only
        # (see agentic/agentic.py:602). Wiring it into the graph tool map
        # lets the new research/compare playbooks produce a real report
        # file (was: a snippets dump into save_note) without falling
        # through to ReAct.
        from agentic.toolkit.reports import write_report
        mapping["write_report"] = write_report
    except Exception as exc:
        log.debug("reports tool unavailable for graph executor: %s", exc)
    try:
        # Graph-level LLM helpers (synthesize, condense, combine, polish)
        # and the KB + RAG learn wrappers live in agentic/toolkit/synthesize.py.
        # Without these, the new research/compare playbooks cannot
        # produce a real synthesized report — they would degrade back to
        # a raw evidence dump.
        from agentic.toolkit.synthesize import (
            synthesize_report, polish_text, combine_evidence,
            condense_text, kb_search, learn_report,
        )
        mapping.update({
            "synthesize_report": synthesize_report,
            "polish_text": polish_text,
            "combine_evidence": combine_evidence,
            "condense_text": condense_text,
            "kb_search": kb_search,
            "learn_report": learn_report,
        })
    except Exception as exc:
        log.debug("synthesize tools unavailable for graph executor: %s", exc)
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


def _run_node(node: PlanNode, prompt: str, results: dict[str, NodeResult],
              embedder=None, llm_client=None, llm_model: str | None = None,
              extras: dict[str, Any] | None = None) -> NodeResult:
    tools = _tool_map()
    fn = tools.get(node.tool)
    args = _substitute(node.args, prompt, results, extras)
    if fn is None:
        return NodeResult(node.id, node.tool, False, f"unknown graph tool: {node.tool}", args=args, error_type="unknown_tool")
    if node.tool == "save_note":
        args["content"] = str(args.get("content", ""))[:AGENT_NOTE_MAX_CHARS]
    # Pass embedder to tools that need it for semantic scoring/condensation.
    if node.tool in _EMBEDDER_AWARE_TOOLS:
        args["embedder"] = embedder
    # Pass LLM client/model to tools that call the model (synthesize_report,
    # polish_text, kb_search/learn_report which accept embedder). The
    # graph executor is the only place that has the owner's client+model
    # pair; the tool map functions themselves are pure so they don't reach
    # back into the owner object.
    if node.tool in {"synthesize_report", "polish_text"}:
        args["client"] = llm_client
        args["model"] = llm_model
    if node.tool in {"kb_search", "learn_report", "condense_text"}:
        args["embedder"] = embedder
    try:
        out = fn(**args)
        return NodeResult(node.id, node.tool, True, str(out), args=args)
    except Exception as exc:
        log.exception("Graph node %s (%s) raised unexpectedly", node.id, node.tool)
        return NodeResult(node.id, node.tool, False, f"{type(exc).__name__}: {exc}", args=args, error_type="execution_error")


def execute_graph(graph: PlanGraph, embedder=None, llm_client=None, llm_model: str | None = None) -> GraphRunResult:
    pending = {node.id: node for node in graph.nodes}
    results: dict[str, NodeResult] = {}
    ordered: list[NodeResult] = []
    extras = getattr(graph, "_extras", {}) or {}
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
            future_map = {pool.submit(_run_node, node, graph.goal, results, embedder, llm_client, llm_model, extras): node for node in runnable}
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


def run_schema_agent(user_input: str, cap_ids: list[str] | None = None, embedder=None,
                     llm_client=None, llm_model: str | None = None) -> GraphRunResult | None:
    graph = plan_from_master(user_input, cap_ids=cap_ids)
    if graph is None:
        return None
    return execute_graph(graph, embedder=embedder, llm_client=llm_client, llm_model=llm_model)


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


def run_playbook_json(task: str, cap_ids: list[str] | None = None, embedder=None,
                      llm_client=None, llm_model: str | None = None) -> str:
    """Run the graph executor and return a compact JSON observation."""
    result = run_schema_agent(task, cap_ids=cap_ids, embedder=embedder,
                              llm_client=llm_client, llm_model=llm_model)
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
    if tool in {"synthesize_report", "polish_text"}:
        return {"evidence": "$prompt", "prompt": "$prompt", "style": "auto"}
    if tool == "combine_evidence":
        return {"parts": ["$prompt"], "separator": "\n\n---\n\n"}
    if tool == "condense_text":
        return {"text": "$prompt", "query": "$prompt"}
    if tool == "kb_search":
        return {"query": "$prompt"}
    if tool == "learn_report":
        return {"title": "$title", "text": "$prompt"}
    if tool == "write_report":
        return {"title": "$title", "content": "$prompt"}
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
