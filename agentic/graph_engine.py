"""
agentic/graph_engine.py

Graph-first, mostly model-free agentic executor optimized for Jetson Orin Nano.

DESIGN:
  - Static DAGs (playbooks in JSON), not dynamic graphs
  - Tiered goal verification (heuristic → embedder → LLM opt-in)
  - Bounded loops (loop_to + max_visits) instead of open-ended cycles
  - Per-node timeout + interrupt for UX gates and runaway checks
  - Tool opt-in via signature inspection (no hand-curated lists)
  - Retry logic with exponential backoff to handle transient failures

LIFECYCLE:
  1. plan_from_master() — match user prompt to best playbook
  2. execute_graph() — run DAG, collect results, verify goal achievement
  3. resume_graph() — continue interrupted runs with user input
  4. append_playbook_from_experience() — promote successful runs into playbooks

PLAYBOOKS:
  Built-in: research_and_report, compare_and_report, evaluator_optimizer, etc.
  User: auto-promoted from ReAct experience or hand-edited in playbook.json

VERIFICATION TIERS (configurable):
  - GRAPH_VERIFY_HEURISTIC (on): content length, entity mentions, embedder cosine
  - GRAPH_VERIFY_EMBEDDER (on): semantic match goal ↔ output
  - GRAPH_VERIFY_LLM (off): explicit LLM check (adds 1-2s, 1-4GB)

For comparison vs LangGraph/CrewAI, see ../README.md or docs/graph_comparison.md
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import json
import os
import re
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from system.config import load_config
load_config()

from system.log import get_logger
from system.userspace import current_user_id, user_state_dir
from agentic.checkpoint import (
    save_node_result, load_checkpoint, clear_checkpoint,
    delete_node_checkpoint, save_graph_state, load_graph_state,
)

log = get_logger(__name__)

# ── Reducer strategies for GraphState ────────────────────────────────────
# Each state key can declare a reducer that controls how concurrent or
# repeated writes to that key are merged.  "replace" (default) simply
# overwrites; the others accumulate.
REDUCER_REPLACE = "replace"
REDUCER_APPEND = "append"        # list.append / list.extend
REDUCER_ADD = "add"              # numeric +=
REDUCER_SET_UNION = "set_union"  # set |= 
REDUCER_DICT_MERGE = "dict_merge"  # dict.update


def _apply_reducer(current: Any, new: Any, strategy: str | None) -> Any:
    if not strategy or strategy == REDUCER_REPLACE:
        return new
    if strategy == REDUCER_APPEND:
        if not isinstance(current, list):
            current = []
        current.extend(new if isinstance(new, list) else [new])
        return current
    if strategy == REDUCER_ADD:
        return (current if current is not None else 0) + (new if new is not None else 0)
    if strategy == REDUCER_SET_UNION:
        if not isinstance(current, set):
            current = set()
        if isinstance(new, set):
            current |= new
        else:
            current.add(new)
        return current
    if strategy == REDUCER_DICT_MERGE:
        if not isinstance(current, dict):
            current = {}
        if isinstance(new, dict):
            current.update(new)
        return current
    return new


GRAPH_AGENT_ENABLED = os.getenv("GRAPH_AGENT_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
GRAPH_AGENT_PLAYBOOK = os.getenv("GRAPH_AGENT_PLAYBOOK", "agentic/playbook.json")
GRAPH_MAX_WORKERS = int(os.getenv("GRAPH_MAX_WORKERS", "2"))

# Goal-verification tiers — each independently toggleable so users can match
# their hardware budget.  Heuristic is always cheap; embedder reuses the
# already-loaded model; LLM verification adds 1-2 s / 1-4 GB per run.
GRAPH_VERIFY_HEURISTIC = os.getenv("GRAPH_VERIFY_HEURISTIC", "1").lower() in {"1", "true", "yes", "on"}
GRAPH_VERIFY_EMBEDDER = os.getenv("GRAPH_VERIFY_EMBEDDER", "1").lower() in {"1", "true", "yes", "on"}
GRAPH_VERIFY_LLM = os.getenv("GRAPH_VERIFY_LLM", "0").lower() in {"1", "true", "yes", "on"}

# Kept in sync with agentic.py's AGENT_NOTE_MAX_CHARS so a note saved via the
# graph executor can't end up longer than one saved via the ReAct path.
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", "5000"))
GRAPH_TOOL_EXECUTION_LOG_MAX = int(os.getenv("GRAPH_TOOL_EXECUTION_LOG_MAX", "100"))
GRAPH_NODE_RESULT_MAX_CHARS = int(os.getenv("GRAPH_NODE_RESULT_MAX_CHARS", "20000"))

# Cost tracking — env-driven so Jetson local can set 0, cloud can override.
GRAPH_COST_PER_1M_INPUT = float(os.getenv("GRAPH_COST_PER_1M_INPUT", "0"))
GRAPH_COST_PER_1M_OUTPUT = float(os.getenv("GRAPH_COST_PER_1M_OUTPUT", "0"))

_TOOL_MAP_CACHE: dict[str, Callable[..., Any]] | None = None
_TOOL_MAP_LOCK = threading.Lock()
_PLAYBOOK_WRITE_LOCK = threading.Lock()
_SEMANTIC_TRIGGER_CACHE: dict[tuple[int, str], Any] = {}
_SEMANTIC_TRIGGER_LOCK = threading.Lock()


def _truncate_with_marker(text: str, max_chars: int, marker: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return marker[:max_chars]
    return text[:max_chars - len(marker)] + marker


def _trim_node_content(content: Any) -> str:
    return _truncate_with_marker(
        str(content),
        GRAPH_NODE_RESULT_MAX_CHARS,
        "\n[truncated by GRAPH_NODE_RESULT_MAX_CHARS]",
    )


def _semantic_trigger_matrix(embedder, semantic_triggers: list[Any]):
    """Return cached normalized vectors for static playbook semantic triggers."""
    if embedder is None or not semantic_triggers:
        return None
    try:
        import numpy as np
        from cognition import reason
    except Exception:
        return None
    missing: list[str] = []
    vectors: list[Any] = []
    embedder_id = id(embedder)
    with _SEMANTIC_TRIGGER_LOCK:
        for raw in semantic_triggers:
            text = str(raw)
            key = (embedder_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
            cached = _SEMANTIC_TRIGGER_CACHE.get(key)
            if cached is None:
                missing.append(text)
            else:
                vectors.append(cached)
    for text in missing:
        try:
            vec = reason.normalize_vec(np.asarray(embedder.embed_query(text), dtype=np.float32))
        except Exception:
            continue
        key = (embedder_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
        with _SEMANTIC_TRIGGER_LOCK:
            _SEMANTIC_TRIGGER_CACHE[key] = vec
        vectors.append(vec)
    if not vectors:
        return None
    try:
        return np.vstack(vectors)
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class PlanNode:
    """A single node in an execution graph with dependencies and control flow."""
    id: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    run_if: dict[str, Any] | None = None
    when: dict[str, Any] | None = None
    loop_to: str | None = None
    loop_condition: dict[str, Any] | None = None
    max_visits: int = 1
    interrupt: bool = False
    timeout_seconds: float | None = None
    max_retries: int = 0
    retry_backoff_seconds: float = 1.0
    fallback_to: str | None = None
    needs_approval: bool = False


@dataclass(frozen=True, slots=True)
class PlanGraph:
    """A directed graph of tool nodes representing an agentic workflow."""
    id: str
    name: str
    goal: str
    nodes: tuple[PlanNode, ...]
    source: str = "playbook"
    reducers: dict[str, str] = field(default_factory=dict)
    _extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Result of executing a single node in the graph."""
    node_id: str
    tool: str
    ok: bool
    content: str
    args: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    usage: dict[str, int] | None = None
    checkpoint_state: str = "{}"

    def summary(self, max_chars: int = 700) -> str:
        """Return a compact one-line summary of the node result."""
        status = "ok" if self.ok else self.error_type or "failed"
        body = re.sub(r"\s+", " ", self.content or "").strip()[:max_chars]
        return f"{self.node_id}:{self.tool}[{status}] {body}".strip()


@dataclass
class GraphState:
    """Shared mutable scratch space, threaded BY REFERENCE through every
    node in one execute_graph() run — the direct analogue of LangGraph's
    State object.

    Complements, does not replace, $result:/$prompt string substitution:
    keep using $result: for small values that benefit from being visible
    in playbook JSON, tool-call logs, and checkpoints. Use state for large
    or non-string objects (accumulated URL sets, embeddings, live handles)
    that would otherwise have to be JSON-encoded into a $result: string
    and hit _substitute's 4000-char per-arg truncation.

    A tool opts into receiving it purely by declaring a `state` parameter
    in its own signature — see _tool_params()/_run_node() below. Nothing
    elsewhere needs to know which tools use it.

    Each key may declare a reducer strategy in ``reducers`` that controls
    how writes from multiple nodes (or repeated loop cycles) merge:
    "replace" (default), "append", "add", "set_union", "dict_merge".

    Checkpointed and restored on resume (unlike the previous design where
    state was explicitly not checkpointed).
    """
    data: dict[str, Any] = field(default_factory=dict)
    reducers: dict[str, str] = field(default_factory=dict)
    # Non-serializable / runtime-only scratch (locks, live handles) that is
    # NEVER checkpointed. json.dumps(state.data) and save_graph_state both
    # serialize only `data`, so tools that need a threading.Lock or a live
    # handle must keep it here — not in `data` — to avoid a
    # "not JSON serializable" crash on the checkpoint path. See _job_index_lock
    # in job_hunt.get_next_job.
    runtime: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from graph state by key."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any, reducer: str | None = None) -> None:
        """Set a value in graph state using the specified or default reducer strategy."""
        strategy = reducer or self.reducers.get(key, REDUCER_REPLACE)
        current = self.data.get(key)
        self.data[key] = _apply_reducer(current, value, strategy)

    def inc_visit(self, node_id: str) -> None:
        """Increment the visit count for a node."""
        visits = self.get("_node_visits", {})
        visits[node_id] = visits.get(node_id, 0) + 1
        self.set("_node_visits", visits)

    def iteration(self) -> int:
        """Increment and return the current iteration count."""
        iter_count = self.get("_iteration", 0)
        self.set("_iteration", iter_count + 1)
        return iter_count + 1

    def record_tool_execution(self, tool_name: str, args: dict, result: Any) -> None:
        """Record tool execution visibility for debugging/adaptive logic"""
        exec_log = self.get("_tool_executions", [])
        exec_log.append({
            "tool": tool_name,
            "args": args,
            "result": str(result)[:500],
            "iteration": self.iteration(),
        })
        if GRAPH_TOOL_EXECUTION_LOG_MAX > 0 and len(exec_log) > GRAPH_TOOL_EXECUTION_LOG_MAX:
            exec_log = exec_log[-GRAPH_TOOL_EXECUTION_LOG_MAX:]
        self.set("_tool_executions", exec_log)

    def get_tool_executions(self, tool_name: str | None = None) -> list[dict]:
        """Get tool execution logs, optionally filtered by tool name"""
        exec_log = self.get("_tool_executions", [])
        if tool_name:
            return [e for e in exec_log if e["tool"] == tool_name]
        return exec_log


def _state_json(state: GraphState) -> str:
    """Serialize a GraphState snapshot for checkpointing.

    Uses the safe encoder so a non-serializable value that leaks into
    state.data (a threading.Lock, a live handle, numpy array, etc.) is
    coerced to its repr instead of raising TypeError and aborting the
    whole graph mid-run — resuming those keys is impossible anyway.
    """
    return json.dumps(_safe_state_dict(state))


def _safe_state_dict(state: GraphState) -> dict[str, Any]:
    """Return a checkpoint-safe copy of state.data (values coerced to str)."""
    out: dict[str, Any] = {}
    for k, v in state.data.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


@dataclass(frozen=True, slots=True)
class GraphRunResult:
    """Complete result of executing a graph workflow including all node results."""
    graph: PlanGraph
    results: tuple[NodeResult, ...]
    final_answer: str
    final_state: dict[str, Any] = field(default_factory=dict)
    interrupted: bool = False
    interrupted_at: str | None = None
    interrupted_question: str | None = None
    goal_score: float | None = None
    goal_reasons: list[str] = field(default_factory=list)

    @property
    def steps(self) -> list[dict[str, Any]]:
        """Return a simplified list of tool execution steps."""
        return [
            {
                "tool": r.tool,
                "ok": r.ok,
                "error_type": r.error_type,
                "args": r.args,
            }
            for r in self.results
        ]

    @property
    def total_tokens(self) -> int:
        """Return the sum of output tokens across all node results."""
        return sum((r.usage or {}).get("output_tokens", 0) for r in self.results)

    @property
    def total_cost(self) -> float:
        """Estimate the total cost based on input and output tokens."""
        inputs = sum((r.usage or {}).get("input_tokens", 0) for r in self.results)
        outputs = sum((r.usage or {}).get("output_tokens", 0) for r in self.results)
        return (inputs / 1e6) * GRAPH_COST_PER_1M_INPUT + (outputs / 1e6) * GRAPH_COST_PER_1M_OUTPUT



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


def _gen_job_worker_nodes(
    fetch_tool: str,
    check_tool: str,
    get_tool: str,
    draft_tool: str,
    save_tool: str,
    report_tool: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Build the gen_job_post node list for `max_workers` parallel worker chains.

    Each worker is an independent chain:
        get_job_N (LOOPS while jobs remain) -> draft_one_N -> save_one_N
    `report_tool` runs once all workers' save steps finish. Tuning the playbook's
    `max_workers` (JSON) or the JOB_HUNT_MAX_WORKERS env increases throughput at
    the cost of more concurrent LLM calls.

    The loop lives on get_next_job — the ONLY tool that advances
    job_current_index — NOT on a read-only "check" node. An earlier design put
    the "more/next" loop on check_jobs_remaining, which only *reads* the index;
    get_next_job (the index advancer) ran after the loop exits, so the condition
    could never change (current=0 < total=N forever) and every run burned its
    max_visits budget looping on "more" without ever processing a job.
    """
    max_workers = max(1, int(max_workers or 1))
    nodes: list[dict[str, Any]] = [
        {"id": "fetch_all", "tool": fetch_tool, "args": {"plan_json": "$prompt"}},
    ]
    save_deps: list[str] = []
    for idx in range(1, max_workers + 1):
        get = f"get_job_{idx}"
        draft = f"draft_one_{idx}"
        save = f"save_one_{idx}"
        nodes.extend([
            # Loop on the index-advancing node: continue pulling the next job
            # until get_next_job reports {"done": true, ...}.
            {"id": get, "tool": get_tool, "depends_on": ["fetch_all"],
             "args": {"worker_id": f"w{idx}"},
             "loop_to": get, "loop_condition": {"not": {"contains": '"done": true'}},
             "max_visits": 500},
            {"id": draft, "tool": draft_tool, "depends_on": [get],
             "args": {"job_json": f"$result:{get}", "template": ""}},
            {"id": save, "tool": save_tool, "depends_on": [draft],
             "args": {"auto_post": "false"}},
        ])
        save_deps.append(save)
    nodes.append({
        "id": "report",
        "tool": report_tool,
        "depends_on": save_deps,
        "args": {"plan": "$result:fetch_all", "search": "{}", "draft": "{}", "save": "{}"},
    })
    return nodes


def _default_playbooks() -> list[dict[str, Any]]:
    """Built-in starter plans. User-promoted plans are appended on disk.

    The graph-first research/report flow that Oppa asked for is structured
    as four reusable playbooks so the LLM-facing ReAct path doesn't have
    to invent the same sequence on every prompt:

      - "research_and_report"   (deep_research + KB + synthesize + write_report + learn_knowledge)
      - "search_kb_and_report"  (adaptive_search + KB + synthesize + write_report + learn_knowledge)
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
            "semantic_triggers": [
                "I want a thorough research report on this topic",
                "Do deep research and write up the findings",
                "Investigate this comprehensively and give me a detailed report",
                "Analyze this topic in depth with citations",
            ],
            "requires_any": ["research", "investigate", "analyze", "report", "comprehensive", "thorough", "deep", "study"],
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
            "semantic_triggers": [
                "Find information about this topic and summarize it",
                "Look up what this means and give me a clear answer",
                "Search for this and write a concise report",
                "Give me a quick overview with sources",
            ],
            "requires_any": ["search", "look up", "find", "what is", "what are", "who is", "when did", "where is", "how do", "how to", "quick", "brief", "tell me"],
            "capabilities": ["research"],
            "nodes": [
                {"id": "web",    "tool": "adaptive_search",  "args": {"query": "$prompt"}},
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
            "semantic_triggers": [
                "Compare these two things side by side",
                "What are the differences between A and B",
                "Give me a pros and cons comparison of these options",
                "Contrast these alternatives with a recommendation",
            ],
            "requires_any": ["compare", "versus", "vs", "contrast", "pros", "cons", "difference"],
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
            "semantic_triggers": [
                "Create a checklist and save it as a note",
                "Break this down into steps and save it",
            ],
            "requires_any": ["save", "note", "checklist", "todo", "list"],
            "capabilities": ["note_taking", "planning"],
            "nodes": [
                {"id": "checklist", "tool": "create_checklist", "args": {"title": "$title", "items": "$heuristic_items"}},
                {"id": "save",      "tool": "save_note", "depends_on": ["checklist"],
                 "args": {"title": "$title", "content": "$result:checklist", "folder": "notes"}},
            ],
        },
        {
            "id": "simple_save_note",
            "name": "Save provided text as a note",
            "triggers": [
                "save note", "write note", "draft", "note that", "jot down", "save this",
                "save as note", "make a note", "take a note", "write down",
            ],
            "semantic_triggers": [
                "Save this as a note for later",
                "Jot this down so I don't forget",
                "Make a note of this information",
            ],
            "requires_any": ["save", "note", "draft", "jot", "write down"],
            "capabilities": ["note_taking"],
            "nodes": [
                {"id": "save", "tool": "save_note",
                 "args": {"title": "$title", "content": "$prompt", "folder": "notes"}},
            ],
        },
        # ──────────────────────────────────────────────────────────────
        # Common agentic workflow patterns (Anthropic/Andrew Ng patterns)
        # ──────────────────────────────────────────────────────────────
        {
            "id": "prompt_chaining",
            "name": "Sequential prompt chaining — each step builds on the previous",
            "triggers": ["step by step", "in stages", "pipeline", "chain", "sequential"],
            "requires_any": ["chain", "pipeline", "stages", "steps"],
            "capabilities": ["reports"],
            "semantic_triggers": [
                "break this into sequential steps where each builds on the previous",
                "run a multi-stage pipeline where output of one feeds the next",
                "process this in a chain of dependent transformations",
            ],
            "nodes": [
                {"id": "plan",    "tool": "make_plan",          "args": {"goal": "$prompt", "max_steps": 6}},
                {"id": "step1",   "tool": "synthesize_report",  "depends_on": ["plan"],
                 "args": {"evidence": "$result:plan", "prompt": "Execute step 1 of the plan: $prompt", "style": "auto"}},
                {"id": "step2",   "tool": "synthesize_report",  "depends_on": ["step1"],
                 "args": {"evidence": "$result:step1", "prompt": "Execute step 2 using previous output: $prompt", "style": "auto"}},
                {"id": "step3",   "tool": "synthesize_report",  "depends_on": ["step2"],
                 "args": {"evidence": "$result:step2", "prompt": "Execute step 3 using previous output: $prompt", "style": "auto"}},
                {"id": "report",  "tool": "write_report",       "depends_on": ["step3"],
                 "args": {"title": "$title", "content": "$result:step3", "report_dir": "reports"}},
            ],
        },
        {
            "id": "routing_classifier",
            "name": "Route the task to a specialized handler based on intent classification",
            "triggers": ["route", "classify", "triage", "dispatch", "which team", "who should handle"],
            "requires_any": ["route", "classify", "triage", "dispatch"],
            "capabilities": ["research", "repo"],
            "semantic_triggers": [
                "classify this request and route it to the right specialist",
                "triage this issue and send it to the appropriate handler",
                "determine what type of task this is and handle it accordingly",
            ],
            "nodes": [
                {"id": "classify", "tool": "synthesize_report",
                 "args": {"evidence": "$prompt", "prompt": "Classify this user request into exactly ONE category: [coding, research, writing, analysis, planning, other]. Return only the category name.", "style": "plain"}},
                {"id": "route_coding",    "tool": "repo_file_tree",    "depends_on": ["classify"], "run_if": {"node": "classify", "equals": "coding"}, "args": {"prefix": ""}},
                {"id": "route_research",  "tool": "deep_research",     "depends_on": ["classify"], "run_if": {"node": "classify", "equals": "research"}, "args": {"query": "$prompt"}},
                {"id": "route_writing",   "tool": "synthesize_report", "depends_on": ["classify"], "run_if": {"node": "classify", "equals": "writing"},
                 "args": {"evidence": "$prompt", "prompt": "Write a polished response to: $prompt", "style": "professional"}},
                {"id": "route_analysis",  "tool": "synthesize_report", "depends_on": ["classify"], "run_if": {"node": "classify", "equals": "analysis"},
                 "args": {"evidence": "$prompt", "prompt": "Analyze this request thoroughly: $prompt", "style": "professional"}},
                {"id": "route_planning",  "tool": "make_plan",         "depends_on": ["classify"], "run_if": {"node": "classify", "equals": "planning"}, "args": {"goal": "$prompt", "max_steps": 8}},
            ],
        },
        {
            "id": "parallel_fanout_fanin",
            "name": "Parallel fan-out to multiple researchers, then fan-in synthesis",
            "triggers": ["parallel", "multiple angles", "comprehensive", "all perspectives", "exhaustive"],
            "requires_any": ["parallel", "multiple", "comprehensive", "exhaustive", "all sides"],
            "capabilities": ["research"],
            "semantic_triggers": [
                "research this from multiple angles in parallel and synthesize",
                "run parallel investigations covering all perspectives",
                "fan out to multiple researchers then combine findings",
            ],
            "nodes": [
                {"id": "plan",      "tool": "make_plan",          "args": {"goal": "$prompt", "max_steps": 5}},
                {"id": "web_1",     "tool": "deep_research",      "args": {"query": "angle 1: technical deep-dive on $prompt"}},
                {"id": "web_2",     "tool": "deep_research",      "args": {"query": "angle 2: practical applications of $prompt"}},
                {"id": "web_3",     "tool": "deep_research",      "args": {"query": "angle 3: limitations and criticisms of $prompt"}},
                {"id": "kb",        "tool": "kb_search",          "depends_on": ["web_1", "web_2", "web_3"], "args": {"query": "$prompt"}},
                {"id": "merge",     "tool": "combine_evidence",   "depends_on": ["web_1", "web_2", "web_3", "kb"],
                 "args": {"parts": ["$result:web_1", "$result:web_2", "$result:web_3", "$result:kb"]}},
                {"id": "synthesize","tool": "synthesize_report",  "depends_on": ["merge"],
                 "args": {"evidence": "$result:merge", "prompt": "$prompt", "style": "professional"}},
                {"id": "report",    "tool": "write_report",       "depends_on": ["synthesize"],
                 "args": {"title": "$title", "content": "$result:synthesize", "report_dir": "reports"}},
                {"id": "learn",     "tool": "learn_report",       "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:synthesize", "kind": "self_learned"}},
            ],
        },
        {
            "id": "orchestrator_workers",
            "name": "Orchestrator decomposes task, delegates to workers, aggregates results",
            "triggers": ["orchestrate", "delegate", "break down", "subtasks", "workers", "coordinate"],
            "requires_any": ["orchestrate", "delegate", "subtasks", "workers", "coordinate"],
            "capabilities": ["research", "repo"],
            "semantic_triggers": [
                "break this complex task into subtasks and delegate each to a specialist",
                "orchestrate multiple workers to handle different parts of this problem",
                "coordinate a team of specialized agents to solve this",
            ],
            "nodes": [
                {"id": "decompose", "tool": "make_plan",          "args": {"goal": "$prompt", "max_steps": 8}},
                {"id": "worker_1",  "tool": "deep_research",      "depends_on": ["decompose"], "args": {"query": "Subtask 1 of: $prompt"}},
                {"id": "worker_2",  "tool": "repo_search_text",   "depends_on": ["decompose"], "args": {"query": "Subtask 2 of: $prompt", "limit": 20}},
                {"id": "worker_3",  "tool": "kb_search",          "depends_on": ["decompose"], "args": {"query": "Subtask 3 of: $prompt"}},
                {"id": "worker_4",  "tool": "synthesize_report",  "depends_on": ["decompose"],
                 "args": {"evidence": "$prompt", "prompt": "Subtask 4 (synthesis): $prompt", "style": "auto"}},
                {"id": "aggregate", "tool": "combine_evidence",   "depends_on": ["worker_1", "worker_2", "worker_3", "worker_4"],
                 "args": {"parts": ["$result:worker_1", "$result:worker_2", "$result:worker_3", "$result:worker_4"]}},
                {"id": "final",     "tool": "synthesize_report",  "depends_on": ["aggregate"],
                 "args": {"evidence": "$result:aggregate", "prompt": "Final integrated answer: $prompt", "style": "professional"}},
                {"id": "report",    "tool": "write_report",       "depends_on": ["final"],
                 "args": {"title": "$title", "content": "$result:final", "report_dir": "reports"}},
                {"id": "learn",     "tool": "learn_report",       "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:final", "kind": "self_learned"}},
            ],
        },
        {
            "id": "evaluator_optimizer",
            "name": "Generate → Evaluate → Refine loop until quality threshold met",
            "triggers": ["refine", "improve", "iterate", "polish", "quality", "perfect", "best version"],
            "requires_any": ["refine", "improve", "iterate", "polish", "quality", "best"],
            "capabilities": ["reports"],
            "semantic_triggers": [
                "generate a draft, evaluate it, and keep refining until it is excellent",
                "iterate on this until the quality is as high as possible",
                "write, critique, and improve this repeatedly",
            ],
            "nodes": [
                {"id": "draft_1",   "tool": "synthesize_report",
                 "args": {"evidence": "$prompt", "prompt": "Draft 1: Write a comprehensive answer to: $prompt", "style": "professional"}},
                {"id": "eval_1",    "tool": "synthesize_report", "depends_on": ["draft_1"],
                 "args": {"evidence": "$result:draft_1", "prompt": "Critique this draft for accuracy, clarity, completeness. List specific improvements needed.", "style": "plain"}},
                {"id": "draft_2",   "tool": "synthesize_report", "depends_on": ["draft_1", "eval_1"],
                 "args": {"evidence": "$result:draft_1\n\nCritique:\n$result:eval_1", "prompt": "Revise draft 1 addressing all critique points. Output improved version.", "style": "professional"}},
                {"id": "eval_2",    "tool": "synthesize_report", "depends_on": ["draft_2"],
                 "args": {"evidence": "$result:draft_2", "prompt": "Evaluate if this version is excellent. If not, list remaining issues. If yes, say 'PASS'.", "style": "plain"}},
                {"id": "draft_3",   "tool": "synthesize_report", "depends_on": ["draft_2", "eval_2"],
                 "args": {"evidence": "$result:draft_2\n\nEvaluation:\n$result:eval_2", "prompt": "If evaluation was not PASS, do one final revision. Otherwise output the final polished version.", "style": "professional"}},
                {"id": "report",    "tool": "write_report",      "depends_on": ["draft_3"],
                 "args": {"title": "$title", "content": "$result:draft_3", "report_dir": "reports"}},
                {"id": "learn",     "tool": "learn_report",      "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:draft_3", "kind": "self_learned"}},
            ],
        },
        {
            "id": "reflection_self_critique",
            "name": "Self-reflection loop — generate, self-critique, revise",
            "triggers": ["reflect", "self-critique", "critique yourself", "review your work", "check your own"],
            "requires_any": ["reflect", "critique", "review", "self"],
            "capabilities": ["reports"],
            "semantic_triggers": [
                "generate an answer then reflect on its weaknesses and improve it",
                "self-critique your response and fix any issues you find",
                "review your own work for errors before finalizing",
            ],
            "nodes": [
                {"id": "initial",   "tool": "synthesize_report",
                 "args": {"evidence": "$prompt", "prompt": "Answer thoroughly: $prompt", "style": "professional"}},
                {"id": "reflect",   "tool": "synthesize_report", "depends_on": ["initial"],
                 "args": {"evidence": "$result:initial", "prompt": "Critically review your own answer above. Identify: factual errors, missing context, unclear reasoning, unsupported claims, style issues. Be thorough and specific.", "style": "plain"}},
                {"id": "revised",   "tool": "synthesize_report", "depends_on": ["initial", "reflect"],
                 "args": {"evidence": "$result:initial\n\nSelf-critique:\n$result:reflect", "prompt": "Produce a corrected, improved version addressing all self-critique points.", "style": "professional"}},
                {"id": "report",    "tool": "write_report",      "depends_on": ["revised"],
                 "args": {"title": "$title", "content": "$result:revised", "report_dir": "reports"}},
                {"id": "learn",     "tool": "learn_report",      "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:revised", "kind": "self_learned"}},
            ],
        },
        {
            "id": "planning_and_execution",
            "name": "Plan → Execute steps → Track progress → Final synthesis",
            "triggers": ["plan and execute", "make a plan then do it", "plan then run", "execute plan"],
            "requires_any": ["plan", "execute", "run", "do it", "implement"],
            "capabilities": ["research", "repo"],
            "semantic_triggers": [
                "create a detailed plan then execute each step in order",
                "plan this out thoroughly then carry out the plan",
                "make a step-by-step plan and follow through on it",
            ],
            "nodes": [
                {"id": "plan",        "tool": "make_plan",          "args": {"goal": "$prompt", "max_steps": 10}},
                {"id": "save_plan",   "tool": "save_note",          "depends_on": ["plan"], "args": {"title": "$title - plan", "content": "$result:plan", "folder": "notes"}},
                {"id": "step_1",      "tool": "deep_research",      "depends_on": ["plan"], "args": {"query": "Step 1 execution for: $prompt"}},
                {"id": "step_2",      "tool": "repo_search_text",   "depends_on": ["plan"], "args": {"query": "Step 2 implementation for: $prompt", "limit": 15}},
                {"id": "step_3",      "tool": "kb_search",          "depends_on": ["plan"], "args": {"query": "Step 3 knowledge for: $prompt"}},
                {"id": "step_4",      "tool": "synthesize_report",  "depends_on": ["plan"],
                 "args": {"evidence": "$prompt", "prompt": "Step 4 (synthesis/decision): $prompt", "style": "auto"}},
                {"id": "merge",       "tool": "combine_evidence",   "depends_on": ["step_1", "step_2", "step_3", "step_4"],
                 "args": {"parts": ["$result:step_1", "$result:step_2", "$result:step_3", "$result:step_4"]}},
                {"id": "final",       "tool": "synthesize_report",  "depends_on": ["merge"],
                 "args": {"evidence": "$result:merge", "prompt": "Final result of executing the plan: $prompt", "style": "professional"}},
                {"id": "report",      "tool": "write_report",       "depends_on": ["final"],
                 "args": {"title": "$title", "content": "$result:final", "report_dir": "reports"}},
                {"id": "learn",       "tool": "learn_report",       "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:final", "kind": "self_learned"}},
            ],
        },
        {
            "id": "multi_agent_collab",
            "name": "Multi-role collaboration — researcher, analyst, writer, reviewer",
            "triggers": ["collaborate", "team", "multi-agent", "roles", "researcher and writer", "analyst and reviewer"],
            "requires_any": ["collaborate", "team", "roles", "multi-agent"],
            "capabilities": ["research", "reports"],
            "semantic_triggers": [
                "have multiple agents with different roles work together on this",
                "simulate a team: researcher, analyst, writer, reviewer",
                "collaborative multi-role approach to this problem",
            ],
            "nodes": [
                {"id": "researcher",  "tool": "deep_research",      "args": {"query": "Researcher role: gather comprehensive evidence on $prompt"}},
                {"id": "analyst",     "tool": "synthesize_report",  "depends_on": ["researcher"],
                 "args": {"evidence": "$result:researcher", "prompt": "Analyst role: structure findings, identify patterns, extract key insights from: $prompt", "style": "auto"}},
                {"id": "writer",      "tool": "synthesize_report",  "depends_on": ["analyst"],
                 "args": {"evidence": "$result:analyst", "prompt": "Writer role: craft a clear, well-organized report from the analysis: $prompt", "style": "professional"}},
                {"id": "reviewer",    "tool": "synthesize_report",  "depends_on": ["writer"],
                 "args": {"evidence": "$result:writer", "prompt": "Reviewer role: check for errors, gaps, clarity issues. Suggest improvements or approve.", "style": "plain"}},
                {"id": "final",       "tool": "synthesize_report",  "depends_on": ["writer", "reviewer"],
                 "args": {"evidence": "$result:writer\n\nReview:\n$result:reviewer", "prompt": "Produce the final polished deliverable incorporating reviewer feedback.", "style": "professional"}},
                {"id": "report",      "tool": "write_report",       "depends_on": ["final"],
                 "args": {"title": "$title", "content": "$result:final", "report_dir": "reports"}},
                {"id": "learn",       "tool": "learn_report",       "depends_on": ["report"],
                 "args": {"title": "$title", "text": "$result:final", "kind": "self_learned"}},
            ],
        },
{
            "id": "gen_job_post",
            "name": "Fetch, draft, and save job listings from configured RSS feeds (parallel workers)",
            "triggers": [
                "draft job", "draft a job posting", "job listing",
                "daily job", "today's job", "today's job posts", "find jobs", "fetch jobs", "search job listings",
            ],
            "semantic_triggers": [
                "draft a job posting for social media",
                "fetch jobs from RSS feeds",
                "create draft job posts from RSS results",
                "run the daily job draft pipeline",
                "search for job listings",
                "today's job posts",
                "today's jobs",
            ],
            "requires_any": ["job", "jobs", "posting", "hiring", "career"],
            "capabilities": ["research", "job_hunt"],
            "max_workers": 2,
            "graph_id": "gen_job_post",
            "pipeline": "shared_5",
            "nodes": [],
        },
        {
            # PATCHED: aurora_forecast + Spec fallback
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
    ]


def _default_playbook_definitions() -> list[dict[str, Any]]:
    """Full default playbook definitions, used to seed playbook.json on first boot.

    Same as _default_playbooks(), but the gen_job_post RSS entry gets a
    `trigger` field added for automatic scheduling (no duplicate entry).
    """
    defaults = _default_playbooks()
    for pb in defaults:
        if pb.get("id") == "gen_job_post":
            pb["trigger"] = {"time": "23:00", "frequency": "daily"}
            break
    return defaults


def load_playbooks() -> list[dict[str, Any]]:
    """Load all available playbooks from defaults and registered graphs."""
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
        log.debug("Could not merge registered Spec graphs into playbooks", exc_info=True)

    path = _playbook_file()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for p in data:
                    if not isinstance(p, dict):
                        continue
                    pid = p.get("id")
                    if not pid:
                        by_id[f"_anon_{len(by_id)}"] = p
                        continue
                    pid = str(pid)
                    if pid in by_id:
                        merged = dict(by_id[pid])
                        merged.update(p)
                        by_id[pid] = merged
                    else:
                        by_id[pid] = p
        except Exception as exc:
            log.warning("failed to load graph playbooks from %s: %s", path, exc)

    return list(by_id.values())


def ensure_playbooks(user_id: str | None = None) -> None:
    """Seed playbook.json with full defaults if it doesn't exist yet."""
    path = _playbook_file()
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_default_playbook_definitions(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    log.info("Seeded playbook.json with %d defaults", len(_default_playbook_definitions()))


def get_playbook_by_id(playbook_id: str) -> dict[str, Any] | None:
    """Look up a playbook by ID across both code defaults and the file."""
    for pb in load_playbooks():
        if pb.get("id") == playbook_id:
            return pb
    return None


def _score_plan(plan: dict[str, Any], prompt: str, cap_ids: list[str] | None = None,
                embedder=None, prompt_vec=None) -> int:
    text = prompt.casefold()
    triggers = [str(t).casefold() for t in plan.get("triggers", [])]
    required = [str(t).casefold() for t in plan.get("requires_any", [])]

    # Hard eligibility: a playbook declaring requires_any must see at least
    # one of those tokens in the prompt. Without this, incidental semantic
    # (+1 at only 0.35 cosine) and capability (+3) bonuses could select it
    # for an unrelated request (e.g. gen_job_post winning "check internet
    # and find out what PNE is", which mentions nothing job-related).
    if required and not any(t in text for t in required):
        return 0

    # A plan is only eligible when a REAL trigger matched (keyword or semantic).
    # requires_any / capability bonuses alone ("job", "architecture", "posting"
    # appearing incidentally in an unrelated sentence) must NOT select a graph.
    trigger_pad = sum(3 for t in triggers if t and t in text)

    # Semantic scoring: if the playbook has semantic_triggers and an embedder
    # is available, score by cosine similarity against the prompt. This
    # replaces/bypasses the keyword triggers for more robust intent matching.
    sem_triggers = plan.get("semantic_triggers", [])
    if embedder is not None and sem_triggers:
        try:
            import numpy as np
            if prompt_vec is None:
                from cognition import reason
                prompt_vec = reason.normalize_vec(np.asarray(embedder.embed_query(prompt), dtype=np.float32))
            matrix = _semantic_trigger_matrix(embedder, sem_triggers)
            if matrix is not None:
                best = float(np.max(matrix @ prompt_vec))
                # Scale: 0.7+ cos = strong match (adds ~5), 0.5+ = moderate (adds ~3)
                if best >= 0.7:
                    boost = 5
                elif best >= 0.5:
                    boost = 3
                elif best >= 0.35:
                    boost = 1
                else:
                    boost = 0
                trigger_pad += boost
        except Exception:
            log.warning("graph_engine: keyword scoring embedding failed")

    # Cannot select a plan from incidental requires_any / capability bonuses
    # alone — at least one keyword or semantic trigger must have matched.
    if trigger_pad <= 0:
        return 0

    score = trigger_pad
    if required and any(t in text for t in required):
        score += 1
    domains = set(plan.get("capabilities", []))
    if cap_ids and domains.intersection(cap_ids):
        score += 3

    return score


def _title(prompt: str) -> str:
    """Extract a short title from a prompt."""
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
            out["$compare_left"] = pair[0]
            out["$compare_right"] = pair[1]
        subjects = split_subjects(prompt)
        if subjects:
            out["$compare_subjects"] = subjects
    except Exception:
        log.warning("graph_engine: failed to extract comparison subjects from prompt")
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


# ── Posting-intent guard ──────────────────────────────────────────────
# The graph-first executor auto-runs a matching playbook for every prompt.
# Requests to POST/publish an artifact that already exists (a draft, a job
# post, a note) must NOT auto-run a workflow like gen_job_post, which would
# re-draft/re-search instead of publishing. Detect that intent here so the
# prompt falls through to the ReAct loop, where direct social tools are used.
_POST_ACTION_TERMS = ("post", "publish", "submit", "share", "post now", "upload", "send")
_POST_CONTENT_TERMS = ("job", "draft", "post", "story", "article", "update",
                       "content", "note", "message", "listing")
# Explicit references to an artifact that ALREADY exists (definite/possessive/
# deictic + noun). These unambiguously publish content, even when a word like
# "draft" also appears ("post the draft", "share this report", "get the note").
_POST_EXISTING_REF = (
    "post the draft", "post the post", "post my draft", "publish the draft",
    "publish this ", "publish my ", "publish the note", "post the report",
    "publish the report", "post this ", "post that ", "post my ", "post it",
    "post them", "my post", "share the ", "share this ", "share that ",
    "share my ", "share it", "submit the ", "submit this ", "submit my ",
    "send the note", "send the post", "send it", "upload the ", "the draft",
    "the post", "the note", "the report", "the listing", "the job posts",
    "the posts", "it on", "it to", "in thread", "now", "right away",
)
_DRAFT_ACTION_TERMS = ("draft", "search", "find", "fetch", "create", "make",
                       "write", "generate", "collect", "scrape", "list",
                       "run", "schedule", "daily", "look for", "hunt", "scan",
                       "do", "today's", "today")


def _is_post_existing_content(prompt: str) -> bool:
    """True when the prompt asks to post/publish something already produced,
    rather than to draft/search for new content.

    Resolution order:
      1. Explicit existing-content reference ("post the draft", "share it",
         "the job posts", "send the note") -> publish intent.
      2. Creation verb ("do", "draft", "generate", "run", "today's") -> the
         plural noun "posts" is the OUTPUT of generation, not a publish target.
      3. Bare publish verb + content noun -> publish intent (e.g. "post a job").
    """
    t = prompt.casefold()
    if any(v in t for v in _POST_EXISTING_REF):
        return True
    if any(v in t for v in _DRAFT_ACTION_TERMS):
        return False
    has_action = any(v in t for v in _POST_ACTION_TERMS)
    has_content = any(n in t for n in _POST_CONTENT_TERMS)
    return bool(has_action and has_content)


_EMAIL_ACTION_TERMS = ("send", "email", "mail", "inbox", "reply to", "draft an email")
_EMAIL_SIGNAL_TERMS = ("email", "mail", "protonmail", "inbox", "@protonmail", "@proton.me", "subject", "attachment")


def _is_email_request(prompt: str) -> bool:
    """True when the prompt asks to send/read/search email. Email turns must
    fall through to the ReAct loop, where the direct ProtonMail tools live,
    instead of auto-running a research/plan playbook (which would burn the
    research budget and never touch the mailbox)."""
    t = prompt.casefold()
    has_action = any(v in t for v in _EMAIL_ACTION_TERMS)
    has_signal = any(v in t for v in _EMAIL_SIGNAL_TERMS)
    return has_action and has_signal


def plan_from_master(user_input: str, cap_ids: list[str] | None = None, embedder=None) -> PlanGraph | None:
    """Select the best matching playbook graph for the user input via semantic scoring."""
    if not GRAPH_AGENT_ENABLED:
        return None
    plans = load_playbooks()
    if _is_post_existing_content(user_input) or _is_email_request(user_input):
        return None
    # Capability gate: when a capability was matched, only run playbooks whose
    # declared capabilities intersect it (or that declare none). This mirrors
    # filtered_tool_schemas so a research playbook can never auto-run under a
    # social/email turn just because its semantic triggers outscore everything
    # else. No match -> keep every playbook eligible (safe default).
    if cap_ids:
        cap_set = set(cap_ids)
        plans = [
            p for p in plans
            if not p.get("capabilities") or set(p.get("capabilities", [])) & cap_set
        ]
    prompt_vec = None
    if embedder is not None:
        try:
            import numpy as np
            from cognition import reason
            prompt_vec = reason.normalize_vec(np.asarray(embedder.embed_query(user_input), dtype=np.float32))
        except Exception:
            prompt_vec = None
    ranked = sorted(((_score_plan(p, user_input, cap_ids, embedder, prompt_vec), p) for p in plans), key=lambda x: x[0], reverse=True)
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
    if plan.get("id") == "compare_and_report" and "$compare_subjects" not in extras:
        ranked = [(s, p) for s, p in ranked if p is not plan]
        if not ranked or ranked[0][0] <= 0:
            return None
    plan = ranked[0][1]

    # If the playbook references a registered graph (graph_id), use it directly
    # instead of building from inline nodes. Prefer common.graphs (all workflows)
    # then fall back to job_hunt.graph for older imports.
    graph_id = plan.get("graph_id") or plan.get("id")
    if graph_id:
        registered_graph = None
        for mod_name in (
            "agentic.workflows.common.graphs",
            "agentic.workflows.job_hunt.graph",
        ):
            try:
                mod = __import__(mod_name, fromlist=["get_graph"])
                registered_graph = mod.get_graph(graph_id)
                if registered_graph is not None:
                    break
            except Exception as exc:
                log.debug("graph_engine: get_graph via %s failed: %s", mod_name, exc)
        if registered_graph is not None:
            registered_graph = PlanGraph(
                id=registered_graph.id,
                name=registered_graph.name,
                goal=user_input,
                nodes=registered_graph.nodes,
                source=registered_graph.source,
                reducers=registered_graph.reducers,
                _extras={**extras, "max_workers": plan.get("max_workers") or 2},
            )
            return registered_graph

    # gen_job_post: if registry lookup failed, rebuild the Layer-2 shared-node
    # graph (not the legacy parallel fetch/loop worker chain).
    if plan.get("id") == "gen_job_post":
        try:
            from agentic.workflows.job_hunt.graph import build_gen_job_post_graph
            built = build_gen_job_post_graph(goal=user_input)
            return PlanGraph(
                id=built.id,
                name=built.name,
                goal=user_input,
                nodes=built.nodes,
                source=built.source,
                reducers=built.reducers,
                _extras={**extras, "max_workers": plan.get("max_workers") or 2},
            )
        except Exception as exc:
            log.warning(
                "graph_engine: gen_job_post shared graph unavailable (%s); "
                "falling back to legacy worker nodes",
                exc,
            )
            env_mw = os.getenv("JOB_HUNT_MAX_WORKERS", "").strip()
            mw = env_mw if env_mw else (plan.get("max_workers") or "2")
            try:
                mw_int = max(1, int(mw))
            except (TypeError, ValueError):
                mw_int = 2
            plan = {**plan, "max_workers": mw_int, "nodes": _gen_job_worker_nodes(
                "fetch_rss_and_email_into_state", "check_jobs_remaining", "get_next_job",
                "draft_single_job", "save_single_job_draft", "report_job_run", mw_int,
            )}
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
            run_if=dict(raw["run_if"]) if isinstance(raw.get("run_if"), dict) else None,
            when=dict(raw["when"]) if isinstance(raw.get("when"), dict) else None,
            loop_to=str(raw["loop_to"]) if raw.get("loop_to") else None,
            loop_condition=dict(raw["loop_condition"]) if isinstance(raw.get("loop_condition"), dict) else None,
            max_visits=int(raw.get("max_visits", 1) or 1),
            interrupt=bool(raw.get("interrupt", False)),
            timeout_seconds=float(raw["timeout_seconds"]) if raw.get("timeout_seconds") is not None else None,
            max_retries=int(raw.get("max_retries", 0) or 0),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 1.0) or 1.0),
            fallback_to=str(raw["fallback_to"]) if raw.get("fallback_to") else None,
            needs_approval=bool(raw.get("needs_approval", False)),
        ))        
    if not nodes:
        return None
    graph = PlanGraph(
        id=str(plan.get("id") or uuid.uuid4()),
        name=str(plan.get("name") or plan.get("id") or "workflow"),
        goal=user_input,
        nodes=tuple(nodes),
        reducers=dict(plan.get("reducers") or {}),
        _extras={**extras, "max_workers": plan.get("max_workers") or 2},
    )
    return graph


def run_subgraph(graph_json: str = "{}", goal: str = "",
                  embedder=None, client=None, model: str | None = None) -> str:
    """Graph tool: execute a PlanGraph defined inline as JSON.
    
    Accept a JSON-serializable dict of the subgraph definition (same shape as
    a playbook entry, but only ``nodes`` and optional ``reducers`` are read).
    This lets a playbook arbitrarily nest sub-computations without flattening
    them into the parent DAG.
    """
    from agentic.graph_engine import execute_graph, PlanGraph, PlanNode
    try:
        data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    except (json.JSONDecodeError, TypeError):
        return "[run_subgraph: invalid graph_json]"
    if not isinstance(data, dict):
        return "[run_subgraph: expected a dict]"
    nodes = []
    for raw in data.get("nodes", []):
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("tool"):
            return f"[run_subgraph: invalid node: {raw.get('id', '?')}]"
        nodes.append(PlanNode(
            id=str(raw["id"]),
            tool=str(raw["tool"]),
            args=dict(raw.get("args") or {}),
            depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
            run_if=dict(raw["run_if"]) if isinstance(raw.get("run_if"), dict) else None,
            when=dict(raw["when"]) if isinstance(raw.get("when"), dict) else None,
            loop_to=str(raw["loop_to"]) if raw.get("loop_to") else None,
            loop_condition=dict(raw["loop_condition"]) if isinstance(raw.get("loop_condition"), dict) else None,
            max_visits=int(raw.get("max_visits", 1) or 1),
            interrupt=bool(raw.get("interrupt", False)),
            timeout_seconds=float(raw["timeout_seconds"]) if raw.get("timeout_seconds") is not None else None,
            max_retries=int(raw.get("max_retries", 0) or 0),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 1.0) or 1.0),
            fallback_to=str(raw["fallback_to"]) if raw.get("fallback_to") else None,
            needs_approval=bool(raw.get("needs_approval", False)),
        ))
    if not nodes:
        return "[run_subgraph: empty subgraph]"
    graph = PlanGraph(
        id=data.get("id", "inline_subgraph"),
        name=data.get("name", "inline"),
        goal=goal,
        nodes=tuple(nodes),
        reducers=dict(data.get("reducers") or {}),
    )
    result = execute_graph(graph, embedder=embedder, llm_client=client, llm_model=model)
    return result.final_answer


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
    # Import graph modules to trigger @tool registration (shared nodes + lanes)
    for _mod in (
        "agentic.workflows.common.nodes",
        "agentic.workflows.job_hunt.graph",
        "agentic.workflows.aurora_forecast.graph",
    ):
        try:
            __import__(_mod)
        except Exception as exc:
            log.debug("graph_engine: failed to import %s: %s", _mod, exc)

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
        from agentic.toolkit.research import (
            adaptive_search, plan_effort, search_and_rank,
            fetch_and_condense_ranked, judge_sufficient,
            deep_research, deep_fetch_round, combine_research_rounds,
            deep_read,
        )
        mapping.update({
            "adaptive_search": adaptive_search,
            "plan_effort": plan_effort,
            "search_and_rank": search_and_rank,
            "fetch_and_condense_ranked": fetch_and_condense_ranked,
            "judge_sufficient": judge_sufficient,
            "deep_research": deep_research,
            "deep_fetch_round": deep_fetch_round,
            "combine_research_rounds": combine_research_rounds,
            "deep_read": deep_read,
        })
    except Exception as exc:
        log.debug("research graph tools unavailable for graph executor: %s", exc)
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
        from agentic.workflows.job_hunt.toolset import (
            search_jobs,
            report_job_run,
            fetch_rss_and_email_into_state, get_next_job, draft_single_job,
            save_single_job_draft, check_jobs_remaining,
        )
        mapping.update({
            "search_jobs": search_jobs,
            "report_job_run": report_job_run,
            "fetch_rss_and_email_into_state": fetch_rss_and_email_into_state,
            "get_next_job": get_next_job,
            "draft_single_job": draft_single_job,
            "save_single_job_draft": save_single_job_draft,
            "check_jobs_remaining": check_jobs_remaining,
        })
    except Exception as exc:
        log.debug("job tools unavailable for graph executor: %s", exc)
    mapping["run_subgraph"] = run_subgraph
    mapping["goal_verification"] = goal_verification
    try:
        from agentic.registry import registry
        mapping.update(registry.get_graph_tool_map())
    except Exception as exc:
        log.debug("registry graph tools unavailable for graph executor: %s", exc)
    return mapping


# ── generic per-tool context injection ───────────────────────────────────
# Instead of maintaining hand-curated tool-name sets (the old
# _EMBEDDER_AWARE_TOOLS constant plus two more inline membership checks in
# _run_node), inspect what each tool function actually declares in its own
# signature and hand it only what it asks for — the same "a tool opts in
# via its own parameter list" model LangGraph's bind_tools()/ToolNode use.
# Wiring a brand-new graph tool that needs client/model/embedder/state
# never requires touching this module again; it just declares the param.
_TOOL_PARAM_CACHE: dict[Callable[..., Any], frozenset[str]] = {}


def _tool_params(fn: Callable[..., Any]) -> frozenset[str]:
    cached = _TOOL_PARAM_CACHE.get(fn)
    if cached is not None:
        return cached
    try:
        params = frozenset(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        params = frozenset()
    _TOOL_PARAM_CACHE[fn] = params
    return params


def _run_node(node: PlanNode, prompt: str, results: dict[str, NodeResult],
               embedder=None, llm_client=None, llm_model: str | None = None,
               extras: dict[str, Any] | None = None, state: GraphState | None = None,
               run_id: str | None = None) -> NodeResult:
    tools = _tool_map()
    fn = tools.get(node.tool)
    args = _substitute(node.args, prompt, results, extras)
    try:
        from agentic.registry import registry
        spec = registry.get(node.tool)
    except Exception:
        spec = None
    if node.needs_approval or (spec is not None and getattr(spec, "needs_approval", False)):
        content = json.dumps({"status": "waiting_for_approval", "run_id": run_id, "node_id": node.id, "tool": node.tool}, ensure_ascii=False)
        return NodeResult(node.id, node.tool, False, content, args=args, error_type="needs_approval")
    if fn is None:
        return NodeResult(node.id, node.tool, False, f"unknown graph tool: {node.tool}", args=args, error_type="unknown_tool")
    if node.tool == "save_note":
        args["content"] = str(args.get("content", ""))[:AGENT_NOTE_MAX_CHARS]

    # Build the actual call kwargs separately from `args` — injected
    # objects (client/embedder/state) are NOT JSON-serializable and must
    # never end up in NodeResult.args, which gets written into
    # run_playbook_json's output and into checkpoint files.
    call_args = dict(args)
    params = _tool_params(fn)
    if "embedder" in params and "embedder" not in call_args:
        call_args["embedder"] = embedder
    if "client" in params and "client" not in call_args:
        call_args["client"] = llm_client
    if "model" in params and "model" not in call_args:
        call_args["model"] = llm_model
    if "llm_model" in params and "llm_model" not in call_args:
        call_args["llm_model"] = llm_model
    if "state" in params and "state" not in call_args:
        call_args["state"] = state
    if "user_id" in params and "user_id" not in call_args:
        call_args["user_id"] = current_user_id()

    last_exc = None
    for attempt in range(node.max_retries + 1):
        try:
            if attempt > 0:
                wait = node.retry_backoff_seconds * (2 ** (attempt - 1))
                log.info("Retry node %s, attempt %d/%d, waiting %.1fs", node.id, attempt + 1, node.max_retries + 1, wait)
                time.sleep(wait)
            out = fn(**call_args)
            usage = state.data.pop("_usage", None) if state else None
            return NodeResult(node.id, node.tool, True, _trim_node_content(out), args=args, usage=usage)
        except Exception as e:
            last_exc = e
            log.warning("Node %s retry %d/%d raised: %s", node.id, attempt + 1, node.max_retries + 1, str(e))
            if attempt < node.max_retries:
                wait = node.retry_backoff_seconds * (2 ** attempt)
                time.sleep(wait)

    if last_exc is not None:
        log.error("Node %s failed after %d attempts: %s", node.id, node.max_retries + 1, str(last_exc))
        return NodeResult(
            node.id, node.tool,
            ok=False,
            content=_trim_node_content(last_exc),
            args=args,
            error_type="exception_after_retries",
            usage=state.data.pop("_usage", None) if state else None,
        )
    return NodeResult(
        node.id, node.tool,
        ok=False,
        content="",
        args=args,
        error_type=None,
        usage=state.data.pop("_usage", None) if state else None,
    )


def _run_if_satisfied(node: PlanNode, results: dict[str, NodeResult], state: GraphState | None = None) -> bool:
    """Evaluate node gates against prior node output and/or shared graph state.

    ``run_if`` keeps the older node-output contract. ``when`` is a lightweight
    LangGraph-style conditional edge: use {"state": "key", ...condition...}
    to gate on GraphState without introducing a full workflow DSL.
    """
    for cond in (node.run_if, node.when):
        if not cond:
            continue
        actual = _condition_actual(cond, results, state)
        if actual is None or not _check_condition(cond, actual):
            return False
    return True


def _condition_actual(cond: dict[str, Any], results: dict[str, NodeResult], state: GraphState | None = None) -> str | None:
    if "state" in cond:
        if state is None:
            return None
        value = state.get(str(cond["state"]))
        if value is None:
            return None
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    if "env_truthy" in cond:
        env_var = str(cond["env_truthy"])
        env_value = os.getenv(env_var, "").strip().lower()
        return env_value
    ref = cond.get("node")
    if ref is not None:
        if ref not in results:
            return None
        return (results[ref].content or "").strip().lower()
    return ""


def _check_condition(cond: dict[str, Any], actual: str) -> bool:
    actual_lower = actual.lower()
    if "not" in cond:
        return not _check_condition(cond["not"], actual)
    if "and" in cond:
        return all(_check_condition(c, actual) for c in cond["and"])
    if "or" in cond:
        return any(_check_condition(c, actual) for c in cond["or"])
    if "env_truthy" in cond:
        # Check if environment variable is truthy
        return actual_lower in {"1", "true", "yes", "on"}
    if "equals" in cond:
        return actual_lower == str(cond["equals"]).strip().lower()
    if "contains" in cond:
        return str(cond["contains"]).strip().lower() in actual_lower
    if "matches" in cond:
        try:
            return bool(re.search(str(cond["matches"]), actual_lower))
        except re.error:
            return False
    if "gt" in cond:
        try:
            return float(actual) > float(cond["gt"])
        except (ValueError, TypeError):
            return False
    if "gte" in cond:
        try:
            return float(actual) >= float(cond["gte"])
        except (ValueError, TypeError):
            return False
    if "lt" in cond:
        try:
            return float(actual) < float(cond["lt"])
        except (ValueError, TypeError):
            return False
    if "lte" in cond:
        try:
            return float(actual) <= float(cond["lte"])
        except (ValueError, TypeError):
            return False
    if "len_gt" in cond:
        return len(actual) > int(cond["len_gt"])
    if "len_lt" in cond:
        return len(actual) < int(cond["len_lt"])
    return True


def _check_loop_condition(node: PlanNode, result: NodeResult) -> bool:
    if not node.loop_condition:
        return False
    return _check_condition(node.loop_condition, (result.content or "").strip().lower())


def _downstream_of(node_id: str, nodes_by_id: dict[str, PlanNode]) -> set[str]:
    """All node ids that transitively depend_on node_id (direct or
    indirect). Used by the loop-back handler to figure out which
    already-completed nodes need to be invalidated and re-scheduled when a
    loop_to fires — a stale downstream result computed from a PRE-loop
    upstream value must not survive into the next pass."""
    dependents: dict[str, set[str]] = {}
    for n in nodes_by_id.values():
        for dep in n.depends_on:
            dependents.setdefault(dep, set()).add(n.id)
    seen: set[str] = set()
    frontier = [node_id]
    while frontier:
        current = frontier.pop()
        for child in dependents.get(current, ()):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def _execute_graph_inner(graph: PlanGraph, embedder=None, llm_client=None,
                          llm_model: str | None = None, run_id: str | None = None,
                          _yield=None) -> GraphRunResult:
    """Shared internals for execute_graph and execute_graph_stream.

    When ``_yield`` is a callable (e.g. ``yield`` in a generator context)
    each completed NodeResult is passed to it *before* the loop-back check,
    allowing a streaming caller to observe results as they land.
    """
    nodes_by_id = {node.id: node for node in graph.nodes}
    pending = dict(nodes_by_id)
    results: dict[str, NodeResult] = {}
    ordered: list[NodeResult] = []
    extras = getattr(graph, "_extras", {}) or {}
    reducers = getattr(graph, "reducers", {}) or {}

    state = GraphState(reducers=reducers)
    visit_counts: dict[str, int] = {}

    if run_id:
        checkpoint_results = load_checkpoint(run_id, NodeResult)
        checkpoint_state = load_graph_state(run_id)
        if checkpoint_state:
            state.data.update(checkpoint_state)
        for prior in checkpoint_results:
            if prior.node_id == "__graph_state__":
                continue
            restored_state = getattr(prior, "checkpoint_state", "{}")
            if restored_state and restored_state != "{}":
                try:
                    state.data.update(json.loads(restored_state))
                except Exception:
                    log.warning("graph_engine: failed to restore checkpoint state")
            results[prior.node_id] = prior
            ordered.append(prior)
            pending.pop(prior.node_id, None)

    seq = len(ordered)

    graph_max_workers = int(extras.get("max_workers") or GRAPH_MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=graph_max_workers) as pool:
        while pending:
            ready = [node for node in pending.values() if all(dep in results for dep in node.depends_on)]
            if not ready:
                stuck = ", ".join(sorted(pending))
                nr = NodeResult("graph", "graph_executor", False, f"dependency cycle or missing dependency among: {stuck}", error_type="dependency_error")
                ordered.append(nr)
                if _yield:
                    _yield(nr)
                break
            runnable, blocked, skipped = [], [], []
            for node in ready:
                if not all(results[dep].ok for dep in node.depends_on):
                    blocked.append(node)
                elif not _run_if_satisfied(node, results, state):
                    skipped.append(node)
                else:
                    runnable.append(node)
            for node in blocked:
                nr = NodeResult(node.id, node.tool, False, "skipped: an upstream dependency failed", error_type="dependency_failed")
                results[node.id] = nr
                ordered.append(nr)
                pending.pop(node.id, None)
                if run_id:
                    save_node_result(run_id, seq, nr, state_json=_state_json(state)); seq += 1
                if _yield:
                    _yield(nr)
            for node in skipped:
                nr = NodeResult(node.id, node.tool, True, "skipped: run_if condition not met", error_type=None)
                results[node.id] = nr
                ordered.append(nr)
                pending.pop(node.id, None)
                if run_id:
                    save_node_result(run_id, seq, nr, state_json=_state_json(state)); seq += 1
                if _yield:
                    _yield(nr)
            if not runnable:
                continue
            # Propagate ContextVars (current_user_id, display_name, etc.) to worker threads.
            # ThreadPoolExecutor does not copy contextvars automatically, so without this
            # per-user state (e.g. job_hunt config path) falls back to "guest".
            ctx = contextvars.copy_context()
            future_map = {}
            for node in runnable:
                future_map[pool.submit(ctx.run, _run_node, node, graph.goal, results, embedder, llm_client, llm_model, extras, state, run_id)] = node
            for fut in as_completed(future_map):
                node = future_map[fut]
                try:
                    if node.timeout_seconds is not None:
                        result = fut.result(timeout=node.timeout_seconds)
                    else:
                        result = fut.result()
                except FutureTimeoutError:
                    result = NodeResult(node.id, node.tool, False, f"timed out after {node.timeout_seconds}s", error_type="timeout")
                    fut.cancel()
                except Exception as exc:
                    result = NodeResult(node.id, node.tool, False, str(exc), error_type="execution_error")
                results[node.id] = result
                ordered.append(result)
                pending.pop(node.id, None)
                if run_id:
                    save_node_result(run_id, seq, result, state_json=_state_json(state)); seq += 1
                    save_graph_state(run_id, _safe_state_dict(state))
                if _yield:
                    _yield(result)

                # State helper tracking
                state.inc_visit(node.id)
                state.record_tool_execution(node.tool, dict(result.args), str(result.content))

                # Handle fallback logic
                if not result.ok and node.fallback_to:
                    fallback_node = nodes_by_id.get(node.fallback_to)
                    if fallback_node:
                        if all(dep in results for dep in fallback_node.depends_on) and all(results[dep].ok for dep in fallback_node.depends_on):
                            log.info("Node %s failed (fallback to %s)", node.id, node.fallback_to)
                            
                            # Try fallback node execution with access to current state
                            fallback_tools = _tool_map()
                            fallback_fn = fallback_tools.get(fallback_node.tool)
                            if fallback_fn:
                                # Build arguments using the same logic as normal execution
                                fallback_args = _substitute(fallback_node.args, graph.goal, results, extras)
                                
                                # Build call kwargs with injected dependencies
                                call_args = dict(fallback_args)
                                params = _tool_params(fallback_fn)
                                if "embedder" in params and "embedder" not in call_args:
                                    call_args["embedder"] = embedder
                                if "client" in params and "client" not in call_args:
                                    call_args["client"] = llm_client
                                if "llm_model" in params and "llm_model" not in call_args:
                                    call_args["llm_model"] = llm_model
                                if "model" in params and "model" not in call_args:
                                    call_args["model"] = llm_model
                                if "state" in params and "state" not in call_args:
                                    call_args["state"] = state
                                
                                try:
                                    fallback_out = fallback_fn(**call_args)
                                    fallback_usage = state.data.pop("_usage", None) if state else None
                                    fallback_result = NodeResult(fallback_node.id, fallback_node.tool, True, _trim_node_content(fallback_out),
                                                              args=fallback_args, usage=fallback_usage)
                                    results[fallback_node.id] = fallback_result
                                    ordered.append(fallback_result)
                                    pending.pop(fallback_node.id, None)
                                    if run_id:
                                        save_node_result(run_id, seq, fallback_result, state_json=_state_json(state)); seq += 1
                                    save_graph_state(run_id, _safe_state_dict(state))
                                    if _yield:
                                        _yield(fallback_result)
                                    log.info("Fallback node %s succeeded", fallback_node.id)
                                except Exception as exc:
                                    log.exception("Fallback node %s failed", fallback_node.id)
                            else:
                                log.warning("Fallback node %s tool %s not found", fallback_node.id, fallback_node.tool)
                        else:
                            log.debug("Fallback node %s has unmet dependencies or failed dependencies", fallback_node.id)
                    else:
                        log.debug("Fallback target node %s not found", node.fallback_to)

                # Interrupt: stop and return partial results.
                if node.interrupt and result.ok:
                    final_answer = _synthesize_without_llm(graph, tuple(ordered))
                    if run_id:
                        clear_checkpoint(run_id)
                    return GraphRunResult(
                        graph=graph, results=tuple(ordered), final_answer=final_answer,
                        final_state=dict(state.data), interrupted=True,
                        interrupted_at=node.id,
                        interrupted_question=str(result.content[:2000]),
                    )

                # Bounded loop-back with proper checkpoint invalidation.
                if node.loop_to and node.loop_to in nodes_by_id and _check_loop_condition(node, result):
                    visit_counts[node.id] = visit_counts.get(node.id, 0) + 1
                    if visit_counts[node.id] < node.max_visits:
                        target_id = node.loop_to
                        to_reset = _downstream_of(target_id, nodes_by_id) | {target_id}
                        for reset_id in to_reset:
                            results.pop(reset_id, None)
                            pending[reset_id] = nodes_by_id[reset_id]
                            if run_id:
                                delete_node_checkpoint(run_id, reset_id)
                        ordered = [r for r in ordered if r.node_id not in to_reset]

    try:
        final_answer = _synthesize_without_llm(graph, tuple(ordered))
    finally:
        if run_id:
            clear_checkpoint(run_id)
    # Job pipeline: the fetch cache is a mid-run scratch pad. Delete it once the
    # run finishes (success OR failure) so the next run re-fetches fresh jobs.
    try:
        if graph.id == "gen_job_post":
            from agentic.workflows.job_hunt.toolset import clear_job_fetch_cache
            clear_job_fetch_cache()
    except Exception as exc:
        log.warning("graph_engine: failed to clear job fetch cache: %s", exc)
    return GraphRunResult(graph=graph, results=tuple(ordered), final_answer=final_answer, final_state=dict(state.data))


def _verify_goal(goal: str, results: tuple[NodeResult, ...],
                   embedder=None, llm_client=None, llm_model: str | None = None) -> tuple[float, list[str]]:
    """Run enabled verification tiers, return (score, reasons)."""
    score = 0.0
    reasons: list[str] = []

    if GRAPH_VERIFY_HEURISTIC:
        score, reasons = _score_goal_achievement(goal, results, embedder=None)

    if GRAPH_VERIFY_EMBEDDER and embedder is not None:
        h_score, h_reasons = _score_goal_achievement(goal, results, embedder=embedder)
        if h_score > score:
            score = h_score
        reasons.extend(r for r in h_reasons if r not in reasons)

    if GRAPH_VERIFY_LLM and llm_client is not None and llm_model:
        try:
            evidence = "\n".join(r.content for r in results if r.ok)[:3000]
            out = goal_verification(evidence=evidence, goal=goal,
                                     client=llm_client, model=llm_model)
            if "ACHIEVED" in out.upper():
                score = max(score, 0.8)
                reasons.append("llm:achieved")
            elif "PARTIAL" in out.upper():
                score = max(score, 0.4)
                reasons.append(f"llm:partial")
            else:
                score = min(score, 0.3)
                reasons.append("llm:failed")
        except Exception as exc:
            log.debug("LLM goal verification failed: %s", exc)

    return score, reasons


def execute_graph(graph: PlanGraph, embedder=None, llm_client=None,
                   llm_model: str | None = None, run_id: str | None = None) -> GraphRunResult:
    """Execute a graph workflow and return the complete result with goal verification."""
    result = _execute_graph_inner(graph, embedder=embedder, llm_client=llm_client,
                                   llm_model=llm_model, run_id=run_id)
    score, reasons = _verify_goal(graph.goal, result.results, embedder=embedder,
                                   llm_client=llm_client, llm_model=llm_model)
    return replace(result, goal_score=score, goal_reasons=reasons)


def execute_graph_stream(graph: PlanGraph, embedder=None, llm_client=None,
                          llm_model: str | None = None, run_id: str | None = None):
    """Generator variant: yields each NodeResult as it completes."""
    results: list[NodeResult] = []

    def _capture(nr: NodeResult) -> None:
        results.append(nr)

    final = _execute_graph_inner(graph, embedder=embedder, llm_client=llm_client,
                                  llm_model=llm_model, run_id=run_id, _yield=_capture)
    score, reasons = _verify_goal(graph.goal, final.results, embedder=embedder,
                                   llm_client=llm_client, llm_model=llm_model)
    final = replace(final, goal_score=score, goal_reasons=reasons)
    yield from results
    yield final


def resume_graph(graph: PlanGraph, run_id: str, resume_input: str = "",
                  embedder=None, llm_client=None, llm_model: str | None = None) -> GraphRunResult:
    """Continue an interrupted graph run.

    The ``resume_input`` is injected into GraphState as ``_resume`` so the
    interrupted node (or any downstream node that inspects state) can read
    the user's response.
    """
    state_data = dict(load_graph_state(run_id) or {})
    if resume_input:
        state_data["_resume"] = resume_input
    save_graph_state(run_id, state_data)
    return execute_graph(graph, embedder=embedder, llm_client=llm_client,
                          llm_model=llm_model, run_id=run_id)


# ── Goal achievement scoring ─────────────────────────────────────────────
# Lightweight, no-LLM heuristic: checks content length, entity mentions,
# failed nodes, and optional embedder cosine similarity.  Playbooks can also
# add an explicit ``goal_verification`` tool node for LLM-based checking.

_GOAL_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+\d+(?:\.\d+)?)?)\b")


def _score_goal_achievement(goal: str, results: tuple[NodeResult, ...],
                             embedder=None) -> tuple[float, list[str]]:
    """Rate 0.0–1.0: did the DAG output actually achieve the goal?

    Returns (score, reasons).  No LLM call — purely heuristic.
    """
    reasons: list[str] = []
    scores: list[float] = []
    final_content = " ".join(r.content for r in results if r.ok)

    # 1. Output must exist and be non-trivial
    if len(final_content.strip()) < 50:
        reasons.append("output_too_short")
        scores.append(0.0)
        return max(0.0, min(1.0, sum(scores) / max(len(scores), 1))), reasons
    reasons.append("output_present")
    scores.append(0.3)

    # 2. Entity mentions (capitalised terms like "Jetson", "RTX 3060")
    entities = _GOAL_ENTITY_RE.findall(goal)
    if entities:
        matched = sum(1 for e in entities if e.lower() in final_content.lower())
        ratio = matched / len(entities)
        reasons.append(f"entities:{matched}/{len(entities)}")
        scores.append(0.15 + 0.15 * ratio)

    # 3. Penalise failed nodes
    failed = [r for r in results if not r.ok]
    if failed:
        reasons.append(f"failed_nodes:{len(failed)}")
        scores.append(-0.2 * len(failed))

    # 4. Optional semantic check via embedder (one batched HTTP call for
    # both texts, not two serial round-trips).
    if embedder is not None:
        try:
            import numpy as np
            from cognition import reason
            batch = reason.embed_batch_or_none(embedder, [goal, final_content[:500]])
            if batch is not None and len(batch) == 2:
                gv = reason.normalize_vec(batch[0])
                cv = reason.normalize_vec(batch[1])
            else:
                gv = reason.normalize_vec(np.asarray(embedder.embed_query(goal), dtype=np.float32))
                cv = reason.normalize_vec(np.asarray(embedder.embed_query(final_content[:500]), dtype=np.float32))
            cos = float(np.dot(gv, cv))
            if cos >= 0.6:
                reasons.append(f"semantic:{cos:.2f}")
                scores.append(0.3)
            elif cos >= 0.4:
                reasons.append(f"semantic_partial:{cos:.2f}")
                scores.append(0.1)
            else:
                reasons.append(f"semantic_low:{cos:.2f}")
        except Exception:
            log.warning("graph_engine: semantic scoring failed")
    total = max(0.0, min(1.0, sum(scores) / max(len(scores), 1)))
    return total, reasons


def goal_verification(evidence: str = "", goal: str = "",
                       client=None, model: str | None = None) -> str:
    """Graph tool (optional): LLM-based check whether evidence achieves goal.

    Returns one of: ``ACHIEVED``, ``PARTIAL: <gap>``, ``FAILED: <reason>``.
    Playbooks add this as a final node when they want explicit LLM verification
    (adds one extra LLM call per run).
    """
    if not evidence or not evidence.strip():
        return "FAILED: no evidence to verify"
    if client is None or not model:
        return "ACHIEVED"  # can't verify without LLM — assume success
    from agentic.toolkit.synthesize import synthesize_report
    prompt = (
        f"Goal: {goal}\n\n"
        f"Evidence:\n{evidence[:3000]}\n\n"
        f"Does this evidence fully achieve the goal? "
        f"Reply with exactly one line: ACHIEVED, PARTIAL: <gap>, or FAILED: <reason>."
    )
    out = synthesize_report(evidence=evidence, prompt=prompt, style="plain",
                             client=client, model=model)
    for tag in ("ACHIEVED", "PARTIAL", "FAILED"):
        if tag in str(out).upper():
            return str(out)[:200].strip()
    return "ACHIEVED"  # conservative default


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


def _record_goal_engram(goal: str, score: float, reasons: list[str],
                         graph_name: str, steps: list[dict]) -> None:
    """Write a goal-achievement engram so the nightly reflection can include it."""
    try:
        from agentic.experience import record_experience
        engram_goal = f"graph_goal:{goal[:300]}"
        record_experience(None, engram_goal, steps,
                          final_answer=f"score={score:.2f} reasons={'; '.join(reasons)}",
                          verified_ok=score >= 0.6, score=score)
    except Exception:
        log.warning("graph_engine: experience recording failed")


def run_schema_agent(user_input: str, cap_ids: list[str] | None = None, embedder=None,
                     llm_client=None, llm_model: str | None = None,
                     run_id: str | None = None) -> GraphRunResult | None:
    """Plan and execute a graph workflow from user input, returning the result or None."""
    graph = plan_from_master(user_input, cap_ids=cap_ids, embedder=embedder)
    if graph is None:
        return None
    if run_id is None:
        run_id = hashlib.sha256(f"{graph.id}|{user_input}".encode()).hexdigest()[:16]
    result = execute_graph(graph, embedder=embedder, llm_client=llm_client,
                            llm_model=llm_model, run_id=run_id)
    if result.goal_score is not None:
        _record_goal_engram(graph.goal, result.goal_score, result.goal_reasons,
                            graph.name, result.steps)
        if result.goal_score < 0.4:
            log.warning("Goal '%s' score low (%.2f): %s",
                         graph.goal, result.goal_score, result.goal_reasons)
        
        # Log cost and metrics for monitoring
        log.info("Goal score: %.2f, tokens: %d, cost: $%.4f",
                 result.goal_score, result.total_tokens, result.total_cost)
    return result


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
    if tool == "deep_research":
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


from agentic.registry import TOOLS, register_tool_schema, tool


@tool(TOOLS["list_playbooks"])
def list_playbooks() -> str:
    """List all available playbooks as JSON."""
    return list_playbooks_json()


register_tool_schema(
    "run_playbook",
    "Run a saved graph/playbook workflow by matching this task prompt. This uses deterministic graph execution, not an LLM planner; if no graph matches, continue with ReAct once and learn the sequence.",
    props={
        "task": {"type": "string", "description": "The task prompt to match against graph playbooks."},
        "cap_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional matched capability ids."},
    },
    required=["task"],
    domain="graph",
    always_on=True,
)
