"""
skills/agentic.py

Aiko's task-mode loop: tool schemas, ReAct-style dispatch, and final response
handling. Pure tool implementations stay in toolkit/; chat facade, TTS,
history, and memory queue ownership stay in cognition/think.py.

Context fetch shape:
  Memory + knowledge-base (KB) are intent-agnostic — cognition.think.route()
  fetches both concurrently BEFORE intent is even resolved, since every
  path needs them. run_agentic_chat receives that fetch as `mem_kb_future`
  (or fetches directly if called standalone, e.g. a scheduled job with no
  prior route() call).

  Wiki, agentic-policy, skill, and experience context are agentic-only —
  they're only useful once intent has actually resolved to "agentic" — so
  they're fetched here, concurrently with each other, via
  _fetch_agentic_only_context(), on the same shared pool
  (cognition.CONTEXT_POOL).
"""

from __future__ import annotations

import concurrent.futures
from collections import OrderedDict
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from system.log import get_logger
from system import bioclock
from cognition import reason
from cognition import CONTEXT_POOL
from skills.skills import list_skillsets, load_skillset, load_skills, search_skillsets_json, skill_context_for
from skills.wiki import wiki_agentic_contexts_for
from skills.capability import match_capabilities, filtered_tool_schemas
from memory.knowledge import knowledge_context_for, ingest_text as ingest_knowledge_text, ingest_file as ingest_knowledge_file
from skills import experience
from skills import schema
from toolkit.tools import (
    deep_search,
    deep_research,
    make_plan,
    create_checklist,
    save_note,
    read_workspace_file,
    summarize_task_state,
    schedule_job,
    list_schedule,
    cancel_schedule,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    scan_photo_workspace,
    propose_photo_ingestion,
    write_photo_ingestion_report,
    repo_file_tree,
    repo_read_file,
    repo_search_text,
    search_jobs,
    draft_weekly_social,
    post_weekly_social,
    draft_photo_social,
    post_photo_social,
    draft_video_social,
    post_video_social,
)

log = get_logger(__name__)

MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", 8))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", 512)))
LLM_CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", 12288))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 120))
AGENT_CONTEXT_BUDGET_RATIO = float(os.getenv("AGENT_CONTEXT_BUDGET_RATIO", 0.65))
# AGENT_MEMORY_DRAIN_TIMEOUT and AGENT_MEMORY_RECALL_LIMIT removed:
#   - Draining removed: the async write's own idle-grace window
#     (MEMORY_WRITE_IDLE_GRACE in memorize.py) plus real agentic turn
#     latency (multi-iteration tool loop) meant a queued write was either
#     not started yet or already finished by the time the next turn's
#     search() ran — a short drain wait never caught anything real.
#   - Recall limit removed: memory (and KB) fetch now happens once,
#     centrally, in cognition.think._fetch_memory_and_knowledge, shared by all
#     three chat paths (see MEMORY_RECALL_LIMIT / KNOWLEDGE_RECALL_LIMIT in
#     cognition.think). Agentic no longer owns a separate limit knob for the
#     same data.
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", 1500))
AGENT_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", 3000))
AGENT_VERIFY_FINAL = os.getenv("AGENT_VERIFY_FINAL", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM = os.getenv("AGENT_VERIFY_LLM", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM_MODE = os.getenv("AGENT_VERIFY_LLM_MODE", "auto")  # "always" | "auto" | "off"
AGENT_MAX_FINAL_REPAIRS = int(os.getenv("AGENT_MAX_FINAL_REPAIRS", 2))
AGENT_VERIFY_MIN_SCORE = float(os.getenv("AGENT_VERIFY_MIN_SCORE", "0.70"))
AGENT_TOOL_RETRY_BACKOFF = float(os.getenv("AGENT_TOOL_RETRY_BACKOFF", 0.4))
AGENT_EXECUTOR_MODE = os.getenv("AGENT_EXECUTOR_MODE", "hybrid").strip().lower()  # react | graph | hybrid
AGENT_INCLUDE_EXPERIENCE_CONTEXT = os.getenv("AGENT_INCLUDE_EXPERIENCE_CONTEXT", "0").lower() in {"1", "true", "yes", "on"}

# Rolling STM window shared across all three chat paths. Mirrors
# CONTEXT_WINDOW_TURNS in cognition.think (kept as a distinct name here rather
# than importing cognition.think, which already imports skills.agentic — that
# would create a circular import).
AGENT_HISTORY_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

# Max number of times deep_search/deep_research together can be invoked in
# ONE agentic workflow. Previously hardcoded to exactly once; now tunable.
# The two tools share one budget so a single agentic workflow cannot keep
# spending web/research calls after enough evidence or snippets were gathered.
AGENT_RESEARCH_MAX_CALLS = int(os.getenv("AGENT_RESEARCH_MAX_CALLS", 1))

# TASK MODE instruction split into a small always-kept CORE (operationally
# essential rules) and a larger GUIDANCE portion that is droppable under
# context-budget pressure. Previously the whole ~950-char block was baked
# into every agentic turn and could never be shed, starving task-specific
# data (memory/wiki/skill) once the budget was exceeded.
TASK_MODE_CORE = (
    "[TASK MODE] You MUST use tools to complete tasks. Call tools first, "
    "speak after. Never describe or simulate tool results — always call the "
    "actual tool. Do not call final_answer until all needed tool calls are "
    "complete. Keep reasoning private; never write tool names or JSON in "
    "your spoken answer."
)
TASK_MODE_GUIDANCE = (
    "[TASK MODE OVERRIDE] The speech style limits in the persona do NOT apply "
    "in task mode. Do not summarize in 1-2 sentences. Output length is "
    "irrelevant until final_answer is reached.\n\n"
    "Treat agentic work as a sequence of steps, not one category: plan/decide "
    "when useful, research with deep_search for snippet-only discovery/support "
    "inside a workflow, or deep_research for fetched source reading, synthesis, "
    "and self-learning, inspect repository files for coding or architecture "
    "work, schedule with schedule_job or schedule_reminder when requested, and "
    "write or save the result when the user asks for an artifact. Research "
    "tasks should normally end in a written summary/report, even if the user "
    "only asked you to look something up, unless they explicitly ask you not "
    "to write it down. If the user asks you to save, write, schedule, or "
    "search: call the tool first, then confirm with final_answer. "
    "Tool observations are structured JSON. If ok=false, do not pretend the "
    "action succeeded: retry with corrected arguments, choose another tool or "
    "query, or clearly disclose the limitation in the final answer. "
    f"deep_search/deep_research together may be used at most {AGENT_RESEARCH_MAX_CALLS} "
    "time(s) per agentic workflow. After research returns, read its evidence "
    "and continue with the next productive step (plan, summarize, save, or "
    "answer) instead of searching again. In task mode, do not use "
    "web_search/web_fetch directly; deep_search is snippet-only and "
    "deep_research is for fetched evidence. When writing notes after research: "
    "cross-check any hardware specs, commands, or version numbers against "
    "fetched page content only — never state technical facts from memory "
    "alone. If a fact cannot be confirmed from fetched content, omit it or "
    "flag it as unverified. If a research tool result explicitly says no "
    "relevant content was found, disclose that gap plainly in the final answer "
    "instead of guessing or filling it in from memory. "
    "Social posting tools (post_weekly_social, post_photo_social, "
    "post_video_social) will refuse to run on anything not already approved "
    "by a person outside this conversation, and refuse a second post of the "
    "same draft. Only call a post_* social tool when the user explicitly "
    "asks to publish/post right now — never as an automatic follow-up to "
    "drafting, and never assume a draft is approved just because it was "
    "created. If a post_* call comes back ok=false, disclose that plainly; "
    "do not tell the user something was posted unless the tool result says so. "
    "Use <skill_context>, <knowledge_context>, and <experience_context> when "
    "they match the task. For repeatable workflows, prefer predefined skill "
    "workflow, learned knowledge, wiki operating cards, and successful similar "
    "past experience over inventing a new process. When a recalled <past_task> "
    "has outcome=\"failed\" or outcome=\"partial\", or a low verifier_score, "
    "treat its steps as a cautionary trace of what went wrong, not a template "
    "to follow — do not repeat the same tool/argument choices that led to "
    "that failure. Only reuse the tool sequence from a <past_task> with "
    "outcome=\"ok\" as a positive template. If no matching skill exists, "
    "continue with generic tools. CRITICAL: When asked to save a file, call "
    "save_note BEFORE writing any content in chat. Do not describe what you "
    "will save — just save it. Never say 'I'll now open a file' or 'I'll "
    "generate' — call the tool immediately."
)

# Placeholder bodies returned by the per-source fetchers when they have no
# real content. Injecting these XML wrappers every agentic turn wastes
# tokens for zero information — the model already receives the real blocks
# (memory / KB / TASK MODE) and the empty wrappers add no signal. We blank
# any block whose body is one of these "no match" placeholders; genuine
# "Lookup failed." placeholders (a real error worth surfacing) are kept.
_EMPTY_CONTEXT_MARKERS = (
    "No similar past task found.",
    "No matching task policy found for this request.",
    "No operational wiki pages found.",
    "No matching predefined skills found.",
    "No matching local knowledge found.",
    "No matching learned knowledge found.",
    "No relevant memories found.",
    "Lookup failed.",
)


def _blank_empty_context(block: str) -> str:
    """Return '' if `block` is an empty "No ... found." placeholder."""
    if not block:
        return ""
    for _marker in _EMPTY_CONTEXT_MARKERS:
        if _marker in block:
            return ""
    return block


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC_POLICY_PATHS = (
    _REPO_ROOT / "skills" / "SKILLS.md",
    _REPO_ROOT / "skills" / "SCHEDULE.md",
)

# Agentic policy context is now RAG-selected against the user's request,
# not injected whole. It's still bounded by _AGENTIC_POLICY_MAX_CHARS and
# is also a droppable block in _enforce_agentic_context_budget below, so a
# growing SKILLS.md/SCHEDULE.md can no longer silently blow the fixed
# "immovable" portion of the context budget.
_AGENTIC_POLICY_CHUNK_CHARS = int(os.getenv("AGENTIC_POLICY_CHUNK_CHARS", "600"))
_AGENTIC_POLICY_CHUNKS_PER_FILE = int(os.getenv("AGENTIC_POLICY_CHUNKS_PER_FILE", "4"))
_AGENTIC_POLICY_CHUNK_MIN_SCORE = float(os.getenv("AGENTIC_POLICY_CHUNK_MIN_SCORE", "0.25"))
_AGENTIC_POLICY_MAX_CHARS = int(os.getenv("AGENTIC_POLICY_MAX_CHARS", "3000"))
_AGENTIC_POLICY_INSTRUCT = "Which policy guidance applies to this task?"

# Per-file mtime-keyed cache so SKILLS.md/SCHEDULE.md are not re-read from
# disk on every agentic turn.  File unchanged → same mtime → cache hit.
# File edited → mtime changes → cache miss → re-read.
_policy_file_cache: dict[str, dict] = {}  # path -> {"content": str, "mtime": float}


def _cached_read_policy(path: Path) -> str:
    """Read a policy file, cached by mtime.  One stat() call on cache check,
    zero I/O on hit."""
    path_str = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    cached = _policy_file_cache.get(path_str)
    if cached is not None and cached["mtime"] == mtime:
        return cached["content"]
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    _policy_file_cache[path_str] = {"content": content, "mtime": mtime}
    return content


def _agentic_policy_context(user_input: str, embedder=None) -> str:
    """Load only the task policy excerpts relevant to this request, instead
    of the entire SKILLS.md/SCHEDULE.md files on every task-mode turn."""
    blocks: list[str] = []
    remaining = _AGENTIC_POLICY_MAX_CHARS
    for path in _AGENTIC_POLICY_PATHS:
        if remaining <= 0:
            break
        text = _cached_read_policy(path).strip()
        if not text:
            continue
        pieces = reason.chunk_text(text, _AGENTIC_POLICY_CHUNK_CHARS)
        if not pieces:
            continue
        relevant = reason.select_relevant_chunks(
            user_input, pieces, embedder, top_k=_AGENTIC_POLICY_CHUNKS_PER_FILE,
            min_score=_AGENTIC_POLICY_CHUNK_MIN_SCORE, instruct=_AGENTIC_POLICY_INSTRUCT,
        )
        excerpt = "\n...\n".join(c for _score, c in relevant) if relevant else pieces[0]
        excerpt = excerpt[:remaining]
        if not excerpt:
            continue
        rel = path.relative_to(_REPO_ROOT)
        blocks.append(f'<agentic_policy path="{rel}">\n{excerpt}\n</agentic_policy>')
        remaining -= len(excerpt)
    if not blocks:
        return "<agentic_policy_context>\nNo matching task policy found for this request.\n</agentic_policy_context>"
    return "<agentic_policy_context>\n" + "\n\n".join(blocks) + "\n</agentic_policy_context>"


_ERROR_PREFIX_RE = re.compile(r"^\[(?P<label>[^\]:]+)(?::\s*(?P<detail>.*))?\]$", re.DOTALL)
_DISCLOSURE_RE = re.compile(
    r"\b(couldn'?t|cannot|can't|failed|unavailable|not available|limitation|"
    r"could not|wasn'?t able|unable|unverified|not verified|partial)\b",
    re.IGNORECASE,
)
_EXTERNAL_ACTION_RE = re.compile(r"\b(send|sent|email|post|posted|buy|bought|book|booked|order|ordered|delete|deleted)\b", re.IGNORECASE)
_LOCAL_ARTIFACT_RE = re.compile(r"\b(saved|created|scheduled|cancelled|path|id|draft|note|workspace)\b", re.IGNORECASE)
# Tools that can genuinely post to a real public account. When one of these
# ran and succeeded this turn, an answer describing a real "posted" action
# is not a hallucinated external action — see _verify_final_answer.
_SOCIAL_POST_TOOLS = {"post_weekly_social", "post_photo_social", "post_video_social"}
# Any tool message over this length gets compacted to a preview once a
# later assistant message has arrived — generalized from a research-only
# rule to cover every bulky tool (repo_read_file, search_jobs, etc.), since
# any of them can accumulate across MAX_AGENT_ITER iterations otherwise.
_COMPACTABLE_MIN_CHARS = 800
_RESEARCH_TOOLS = {"deep_search", "deep_research"}



_TOOLS: dict[str, tuple[dict, object]] = {}


def _owner_embedder(owner):
    """Reuse the already-warm HarrierEmbedder — same instance think.py uses
    for memory search and intent routing — so every relevance-scoring call
    (web evidence, KB, skills, agentic policy) gets semantic scoring with
    zero extra model load. Returns None if unavailable; every scoring path
    then falls back to keyword overlap instead of failing."""
    return getattr(getattr(getattr(owner, "_memorize", None), "_mem", None), "_embedder", None)


def _fetch_agentic_only_context(user_input: str, embedder, query_vector: np.ndarray | None = None) -> dict:
    """Fetch agentic-specific context blocks concurrently: agentic policy
    (SKILLS.md/SCHEDULE.md excerpts), wiki (architecture cards + wiki's own
    knowledge RAG), predefined skill workflows, and past-task experience.

    These only matter once intent has resolved to "agentic" — unlike
    memory + KB, which cognition.think.route() fetches for every path up front,
    before intent is even known. All four reads here are independent
    (separate stores, no shared output), so they run concurrently on the
    same pool and are joined afterward; order of completion is irrelevant.

    Per-key try/except means one failed lookup surfaces a fallback block
    instead of sinking the other three.

    query_vector — pre-computed _QUERY_INSTRUCT embedding; avoids redundant
    embedding in batch_block_relevance_scores.
    """
    futures = {
        "agentic_policy": CONTEXT_POOL.submit(_agentic_policy_context, user_input, embedder=embedder),
        "wiki": CONTEXT_POOL.submit(wiki_agentic_contexts_for, user_input, embedder=embedder),
        "skill": CONTEXT_POOL.submit(skill_context_for, user_input, limit=2, max_chars=3000, embedder=embedder),
        "experience": CONTEXT_POOL.submit(experience.experience_context_for, user_input, limit=3, embedder=embedder) if AGENT_INCLUDE_EXPERIENCE_CONTEXT else None,
    }
    # "wiki" returns a (wiki_block, knowledge_block) tuple; both come
    # from a SINGLE search_wiki call (see wiki_agentic_contexts_for) instead
    # of the old two-call path that embedded the same query twice.
    fallbacks = {
        "agentic_policy": "<agentic_policy_context>\nLookup failed.\n</agentic_policy_context>",
        "wiki": ("<wiki_context>\nLookup failed.\n</wiki_context>",
                "<wiki_knowledge_context>\nLookup failed.\n</wiki_knowledge_context>"),
        "skill": "<skill_context>\nLookup failed.\n</skill_context>",
        "experience": "<experience_context>\nLookup failed.\n</experience_context>",
    }
    results = {}
    for key, future in futures.items():
        try:
            results[key] = future.result() if future is not None else ""
        except Exception as e:
            log.error("[agentic] context fetch '%s' failed: %s", key, e)
            results[key] = fallbacks[key]
    wiki_block, knowledge_block = results.pop("wiki")
    results["wiki"] = wiki_block
    results["wiki_knowledge"] = knowledge_block
    # wiki_knowledge is folded into knowledge_context downstream in
    # run_agentic_chat and scored there (combined with knowledge_block) —
    # scoring it here too is a wasted embedding call whose result
    # _enforce_agentic_context_budget never reads (it only consumes the 5
    # budget-block keys: wiki, knowledge, experience, agentic_policy, skill).
    score_keys = [k for k in results if k != "wiki_knowledge"]
    if not AGENT_INCLUDE_EXPERIENCE_CONTEXT:
        score_keys = [k for k in score_keys if k != "experience"]
    score_texts = [results[k] for k in score_keys]
    score_values = reason.batch_block_relevance_scores(embedder, user_input, score_texts, query_vector=query_vector)
    scores = dict(zip(score_keys, score_values))
    if not AGENT_INCLUDE_EXPERIENCE_CONTEXT:
        scores["experience"] = 0.0
    results["_scores"] = scores
    return results


AGENT_HISTORY_CANDIDATE_MULTIPLIER = int(os.getenv("AGENT_HISTORY_CANDIDATE_MULTIPLIER", 3))
AGENT_HISTORY_RECENCY_HALFLIFE = float(os.getenv("AGENT_HISTORY_RECENCY_HALFLIFE", 4))  # turns
AGENT_HISTORY_ALWAYS_KEEP_RECENT = int(os.getenv("AGENT_HISTORY_ALWAYS_KEEP_RECENT", 2))

# Per-message history embedding cache: each unique historical user message is
# embedded once and reused across turns. The conversation history is
# append-only, so an old message's embedding never changes — yet
# _recent_history_messages scores the full candidate window (24 pairs) every
# turn just to keep 8, re-paying the embed cost for ~16 already-dropped
# pairs on EVERY single turn. Keyed by truncated content; capped to bound
# memory across a long session (overflow just forces a one-time re-embed).
_history_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_HISTORY_EMBED_CACHE_MAX = 512

def _history_relevance_scores(embedder, user_input: str, history_texts: list[str], query_vector: np.ndarray | None) -> list[float]:
    truncated = [t[:1500] for t in history_texts]
    to_embed = [t for t in truncated if t not in _history_embed_cache]
    if to_embed:
        new_vecs = reason.embed_batch_or_none(embedder, to_embed)
        if new_vecs is not None and new_vecs.shape[0] == len(to_embed):
            for t, v in zip(to_embed, new_vecs):
                _history_embed_cache[t] = np.asarray(v, dtype=np.float32)
                _history_embed_cache.move_to_end(t)
            while len(_history_embed_cache) > _HISTORY_EMBED_CACHE_MAX:
                _history_embed_cache.popitem(last=False)  # evict oldest, not everything
        else:
            return reason.batch_block_relevance_scores(embedder, user_input, history_texts, query_vector=query_vector)
    # Mark cache hits as recently used too, so eviction stays LRU rather than
    # pure insertion-order FIFO.
    for t in truncated:
        if t in _history_embed_cache:
            _history_embed_cache.move_to_end(t)
    b_vecs = np.asarray([_history_embed_cache[t] for t in truncated], dtype=np.float32)
    if query_vector is not None:
        q_vec = np.asarray(query_vector, dtype=np.float32)
    else:
        try:
            q_vec = np.asarray(embedder.embed_query(user_input), dtype=np.float32)
        except Exception:
            return [0.0] * len(history_texts)
    scores = reason.batch_cosine_scores(q_vec, b_vecs)
    return [float(s) for s in scores]


def _recent_history_messages(owner, user_input: str = "", max_turns: int = AGENT_HISTORY_TURNS, query_vector: np.ndarray | None = None) -> list[dict]:
    with owner._history_lock:
        snapshot = list(owner._history)
    if not snapshot:
        return []
    sanitized = owner._sanitize_history(snapshot)

    pairs, i = [], 0
    while i < len(sanitized) - 1:
        if sanitized[i]["role"] == "user" and sanitized[i + 1]["role"] == "assistant":
            pairs.append((sanitized[i], sanitized[i + 1]))
            i += 2
        else:
            i += 1
    if not pairs or not user_input:
        return sanitized[-(max_turns * 2):]

    candidates = pairs[-(max_turns * AGENT_HISTORY_CANDIDATE_MULTIPLIER):]
    n = len(candidates)
    embedder = _owner_embedder(owner)

    history_texts = [u_msg["content"] for u_msg, _ in candidates]
    relevance_scores = _history_relevance_scores(embedder, user_input, history_texts, query_vector)
    scored = []
    for idx, (u_msg, a_msg) in enumerate(candidates):
        turns_ago = n - 1 - idx
        recency_weight = 0.5 ** (turns_ago / AGENT_HISTORY_RECENCY_HALFLIFE)
        relevance = relevance_scores[idx]
        scored.append((0.5 * recency_weight + 0.5 * relevance, idx))

    keep_idx = set(range(max(0, n - AGENT_HISTORY_ALWAYS_KEEP_RECENT), n))  # continuity floor
    for score, idx in sorted(scored, reverse=True):
        if len(keep_idx) >= max_turns:
            break
        keep_idx.add(idx)

    messages = []
    for idx in sorted(keep_idx):
        messages.extend(candidates[idx])
    return messages


@dataclass
class ToolResult:
    """Structured outcome for one tool call attempt."""

    ok: bool
    tool: str
    args: dict
    content: str
    error_type: str | None = None
    retryable: bool = False
    attempts: int = 1
    metadata: dict = field(default_factory=dict)

    def observation(self) -> str:
        """Render a compact machine-readable observation for the next LLM step."""
        payload = {
            "ok": self.ok,
            "tool": self.tool,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "error_type": self.error_type,
            "args": self.args,
            "content": self.content[:AGENT_TOOL_RESULT_MAX_CHARS],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class TaskState:
    """Runtime ledger of actions, evidence, and unresolved failures."""

    goal: str
    steps: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    failures: list[ToolResult] = field(default_factory=list)

    def record(self, result: ToolResult) -> None:
        self.steps.append({
            "tool": result.tool,
            "ok": result.ok,
            "attempts": result.attempts,
            "error_type": result.error_type,
            "args": result.args,
        })
        if result.ok:
            self.evidence.append(f"{result.tool}: {result.content[:500]}")
        else:
            self.failures.append(result)

    def summary(self) -> str:
        payload = {
            "goal": self.goal,
            "completed_tools": [s for s in self.steps if s["ok"]],
            "failed_tools": [s for s in self.steps if not s["ok"]],
            "evidence_count": len(self.evidence),
            "unresolved_failures": [
                {
                    "tool": f.tool,
                    "error_type": f.error_type,
                    "content": f.content[:300],
                    "args": f.args,
                }
                for f in self.failures
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class VerificationResult:
    """Final-answer verification verdict."""

    ok: bool
    feedback: str
    score: float = 1.0


def tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schemas for autonomous task mode."""
    return [schema for schema, _handler in _TOOLS.values()]


def _f(name, description, properties=None, required=None):
    """Build an OpenAI tool schema dict."""
    s = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }
    if required:
        s["function"]["parameters"]["required"] = required
    return s

# ── tool registration ────────────────────────────────────────────────────────
# Each tool is defined as (schema, handler) in a single place so that
# parameter names in the schema and the handler never drift apart.

_TOOL_DEFS: list[tuple[dict, object]] = []

def _reg(name, desc, handler, props=None, required=None):
    schema = _f(name, desc, props, required)
    _TOOL_DEFS.append((schema, handler))

def _reg_no_handler(name, desc, props=None, required=None):
    schema = _f(name, desc, props, required)
    _TOOL_DEFS.append((schema, None))

_reg_no_handler("deep_search", "Web search returning result snippets/URLs only (no full-page fetch). Use for quick discovery or as one step inside a larger workflow.",
    {"query": {"type": "string", "description": "The focused research query to search and fetch."}},
    required=["query"])

_reg_no_handler("deep_research", "Research tool that fetches and synthesizes full source pages from discovered URLs. Use when the research itself is the deliverable or for self-learning. Costs more than deep_search.",
    {"query": {"type": "string", "description": "The research question. Can be broader/less scoped than a deep_search query since the tool refines it internally."}},
    required=["query"])

_reg("make_plan", "Make plan.",
    lambda args: make_plan(args.get("goal", ""), args.get("constraints", ""), int(args.get("max_steps", 8) or 8)),
    {"goal": {"type": "string"}, "constraints": {"type": "string"}, "max_steps": {"type": "integer"}},
    required=["goal"])

_reg("create_checklist", "Make checklist.",
    lambda args: create_checklist(args.get("title", "Checklist"), args.get("items", "")),
    {"title": {"type": "string"}, "items": {"type": "string", "description": "Newline-separated checklist items."}},
    required=["title", "items"])

_reg("save_note", "Save a note to a workspace file. content MUST be plain text only, under 400 characters. No markdown tables, no bullet lists, no backticks, no quotes. Write a brief plain-text summary only.",
    lambda args: save_note(args.get("title", "Aiko note"), args.get("content", ""), args.get("folder", "notes")),
    {"title": {"type": "string", "description": "Short filename title."}, "content": {"type": "string", "description": "Plain text only. Max 400 chars. No markdown."}, "folder": {"type": "string", "description": "Subfolder, default: notes"}},
    required=["title", "content"])

_reg("read_workspace_file", "Read workspace file.",
    lambda args: read_workspace_file(args.get("relative_path", "")),
    {"relative_path": {"type": "string"}},
    required=["relative_path"])

_reg("summarize_task_state", "Summarize task state.",
    lambda args: summarize_task_state(args.get("goal", ""), args.get("done", ""), args.get("next_action", ""), args.get("risks", "")),
    {"goal": {"type": "string"}, "done": {"type": "string"}, "next_action": {"type": "string"}, "risks": {"type": "string"}},
    required=["goal"])

_reg("schedule_job", "Schedule local job/alarm. HH:MM. Frequencies: once,hourly,daily,weekdays,weekly,biweekly,monthly,custom_weekdays. Supports relative_days for today/tomorrow/day-after-tomorrow offsets.",
    lambda args: schedule_job(args.get("title", "Scheduled job"), args.get("task", "Scheduled job"), args.get("time_of_day", "06:00"), args.get("frequency", "daily"), args.get("timezone"), args.get("days_of_week"), args.get("action", "agentic"), args.get("relative_days")),
    {"title": {"type": "string"}, "task": {"type": "string"}, "time_of_day": {"type": "string", "description": "24-hour local time, e.g. 06:00"}, "frequency": {"type": "string", "enum": ["once", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"]}, "timezone": {"type": "string"}, "days_of_week": {"type": "string", "description": "Optional weekdays, e.g. Monday Wednesday Friday"}, "relative_days": {"type": "string", "description": "Optional day offset/phrase for the first due date, e.g. 0/today, 1/tomorrow, 2/day after tomorrow"}, "action": {"type": "string", "enum": ["announce", "agentic"], "description": "announce only, or agentic to let Aiko perform a local autonomous task"}},
    required=["title", "task", "time_of_day"])

_reg("list_schedule", "List schedule.",
    lambda args: list_schedule(bool(args.get("include_disabled", False))),
    {"include_disabled": {"type": "boolean"}})

_reg("cancel_schedule", "Cancel schedule item.",
    lambda args: cancel_schedule(args.get("job_id", "")),
    {"job_id": {"type": "string"}},
    required=["job_id"])

_reg("schedule_reminder", "Simple once/daily reminder.",
    lambda args: schedule_reminder(args.get("title", "Reminder"), args.get("message", "Reminder"), args.get("time_of_day", "06:00"), args.get("repeat", "daily"), args.get("timezone")),
    {"title": {"type": "string"}, "message": {"type": "string"}, "time_of_day": {"type": "string"}, "repeat": {"type": "string", "enum": ["once", "daily"]}, "timezone": {"type": "string"}},
    required=["title", "message", "time_of_day"])

_reg("list_reminders", "List reminders.",
    lambda args: list_reminders(bool(args.get("include_disabled", False))),
    {"include_disabled": {"type": "boolean"}})

_reg("cancel_reminder", "Cancel reminder by id.",
    lambda args: cancel_reminder(args.get("reminder_id", "")),
    {"reminder_id": {"type": "string"}},
    required=["reminder_id"])

_reg("list_skillsets", "List Aiko's predefined local workflow skillsets.",
    lambda args: list_skillsets(),
    {})

_reg("search_skillsets", "Search Aiko's predefined workflow skillsets by task/query.",
    lambda args: search_skillsets_json(args.get("query", ""), int(args.get("limit", 3) or 3)),
    {"query": {"type": "string"}, "limit": {"type": "integer"}},
    required=["query"])

_reg("load_skillset", "Load the full markdown instructions for one predefined skillset by id.",
    lambda args: load_skillset(args.get("skill_id", "")),
    {"skill_id": {"type": "string"}},
    required=["skill_id"])

_reg("list_master_plans", "List graph/master-plan workflows available to the model-free graph executor.",
    lambda args: schema.list_master_plans_json(),
    {})

_reg_no_handler("run_master_plan", "Run a saved graph/master-plan workflow by matching this task prompt. This uses deterministic graph execution, not an LLM planner; if no graph matches, continue with ReAct once and learn the sequence.",    {"task": {"type": "string", "description": "The task prompt to match against graph master plans."}, "cap_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional matched capability ids."}},
    required=["task"])

_reg("scan_photo_workspace", "Scan a workspace photo inbox for wildlife/nature/astro image files.",
    lambda args: scan_photo_workspace(args.get("inbox", "photos/inbox"), int(args.get("limit", 100) or 100)),
    {"inbox": {"type": "string", "description": "Workspace-relative inbox path, default photos/inbox."}, "limit": {"type": "integer"}})

_reg("propose_photo_ingestion", "Create a safe dry-run ingestion plan for photo files without moving or editing metadata.",
    lambda args: propose_photo_ingestion(args.get("inbox", "photos/inbox"), args.get("library_root", "photos/library"), args.get("rating_rule", "manual-review-first")),
    {"inbox": {"type": "string"}, "library_root": {"type": "string"}, "rating_rule": {"type": "string"}})

_reg("write_photo_ingestion_report", "Write a photo workflow report under the workspace reports folder.",
    lambda args: write_photo_ingestion_report(args.get("title", "photo-ingestion"), args.get("content", ""), args.get("report_dir", "photos/reports")),
    {"title": {"type": "string"}, "content": {"type": "string"}, "report_dir": {"type": "string"}})

_reg("repo_file_tree", "List repository text files for Aiko architecture/code navigation.",
    lambda args: repo_file_tree(args.get("prefix", ""), int(args.get("limit", 200) or 200)),
    {"prefix": {"type": "string"}, "limit": {"type": "integer"}})

_reg("repo_read_file", "Read one repository text file for architecture/code work.",
    lambda args: repo_read_file(args.get("relative_path", ""), int(args.get("max_chars", 20000) or 20000)),
    {"relative_path": {"type": "string"}, "max_chars": {"type": "integer"}},
    required=["relative_path"])

_reg("repo_search_text", "Search repository text files with simple substring matching.",
    lambda args: repo_search_text(args.get("query", ""), args.get("prefix", ""), int(args.get("limit", 50) or 50)),
    {"query": {"type": "string"}, "prefix": {"type": "string"}, "limit": {"type": "integer"}},
    required=["query"])

_reg_no_handler("learn_knowledge", "Store durable learned knowledge in Aiko's vector RAG store (encrypted when SQLite encryption is enabled). Use only when the user asks Aiko to remember/add/store knowledge, ingest pasted document text, or after explicit self-learning/research should be retained. Do not use for private personal preferences; those belong in memory. Do not use for merely saving a human-readable note; use save_note for that.",
    {"title": {"type": "string", "description": "Short title for the learned document or fact set."}, "text": {"type": "string", "description": "Knowledge text to chunk, embed, and retrieve later. Use this for pasted/extracted text."}, "relative_path": {"type": "string", "description": "Optional workspace-relative document path to ingest instead of text."}, "source": {"type": "string", "description": "Optional source URL/path/context for pasted text."}, "kind": {"type": "string", "enum": ["ingested", "self_learned", "study_note"], "description": "Where this knowledge came from."}},
    required=["title"])

_reg("search_jobs", "Search configured job boards for a role. If location is omitted, uses the job_hunt skill default location. Deduped automatically.",
    lambda args: json.dumps(search_jobs(args.get("query", ""), args.get("location", ""), int(args["max_results"]) if args.get("max_results") not in (None, "") else None, int(args["max_age_days"]) if args.get("max_age_days") not in (None, "") else None, args.get("job_type", "")), ensure_ascii=False),
    {"query": {"type": "string"}, "location": {"type": "string", "description": "Optional override. Defaults to the job_hunt skill location."}, "max_results": {"type": "integer"}, "max_age_days": {"type": "integer"}, "job_type": {"type": "string", "description": "Optional employment type filter from the user prompt, e.g. full-time, contract, remote."}},
    required=["query"])

_reg("draft_weekly_social", "Create a weekly memory-postcard draft (for X and/or Threads) from this week's pinned memories, for human review. Does NOT post anything. Use only for the weekly memory-postcard workflow, not general photo/video posting.",
    lambda args: draft_weekly_social(force=bool(args.get("force", False))),
    {"force": {"type": "boolean", "description": "Overwrite an existing draft already created for this week."}})

_reg("post_weekly_social", "Post an ALREADY HUMAN-APPROVED weekly memory-postcard draft to X and/or Threads. Will refuse (ok=false) unless a person has approved this exact draft outside this conversation, and refuses to post the same draft twice. Only call when the user explicitly asks to publish/post the draft now.",
    lambda args: post_weekly_social(args.get("draft_dir", ""), providers=args.get("providers") or None),
    {"draft_dir": {"type": "string", "description": "The draft_dir path returned by draft_weekly_social or given by the user."}, "providers": {"type": "array", "items": {"type": "string", "enum": ["x", "threads"]}, "description": "Optional subset of providers to post to; defaults to the draft's configured providers."}},
    required=["draft_dir"])

_reg("draft_photo_social", "Scan the photo inbox, caption and curate candidates, and create an Instagram photo draft bundle for human review. Does NOT post anything.",
    lambda args: draft_photo_social(inbox=args.get("inbox") or None, force=bool(args.get("force", False))),
    {"inbox": {"type": "string", "description": "Optional workspace-relative photo inbox override."}, "force": {"type": "boolean", "description": "Create a new draft even if one already exists for this run."}})

_reg("post_photo_social", "Post an ALREADY HUMAN-APPROVED photo draft to Instagram. Will refuse (ok=false) unless a person has approved this exact draft outside this conversation, and refuses to post the same draft twice. Only call when the user explicitly asks to publish/post the draft now.",
    lambda args: post_photo_social(args.get("draft_dir", ""), providers=args.get("providers") or None),
    {"draft_dir": {"type": "string", "description": "The draft_dir path returned by draft_photo_social or given by the user."}, "providers": {"type": "array", "items": {"type": "string", "enum": ["instagram"]}}},
    required=["draft_dir"])

_reg("draft_video_social", "Queue the oldest not-yet-drafted video in the video inbox that already has a matching NAME.md description file, polishing it into a YouTube title/description for human review. Does NOT post, and does NOT choose which video — dropping the file with its description IS the selection.",
    lambda args: draft_video_social(inbox=args.get("inbox") or None),
    {"inbox": {"type": "string", "description": "Optional workspace-relative video inbox override."}})

_reg("post_video_social", "Post an ALREADY HUMAN-APPROVED video draft to YouTube. Will refuse (ok=false) unless a person has approved this exact draft outside this conversation, and refuses to post the same draft twice. Only call when the user explicitly asks to publish/post the draft now.",
    lambda args: post_video_social(args.get("draft_dir", ""), providers=args.get("providers") or None),
    {"draft_dir": {"type": "string", "description": "The draft_dir path returned by draft_video_social or given by the user."}, "providers": {"type": "array", "items": {"type": "string", "enum": ["youtube"]}}},
    required=["draft_dir"])

_reg("final_answer", "Final answer.",
    lambda args: final_answer(args.get("answer", "")),
    {"answer": {"type": "string", "description": "The final answer text."}},
    required=["answer"])

# ── populate _TOOLS from _TOOL_DEFS ──────────────────────────────────────────

for _tool_schema, _tool_handler in _TOOL_DEFS:
    _tool_name = _tool_schema["function"]["name"]
    _TOOLS[_tool_name] = (_tool_schema, _tool_handler)
    

def _required_args_for(name: str) -> list[str]:
    entry = _TOOLS.get(name)
    if not entry:
        return []
    return list(entry[0].get("function", {}).get("parameters", {}).get("required", []))


def _validate_args(name: str, args: object) -> ToolResult | None:
    """Return a validation error result, or None when args are safe to dispatch."""
    if name == "final_answer":
        return None
    if not isinstance(args, dict):
        return ToolResult(
            ok=False, tool=name, args={},
            content="Tool arguments must be a JSON object. Reissue the call with valid JSON.",
            error_type="invalid_args", retryable=True,
        )
    missing = [
        key for key in _required_args_for(name)
        if args.get(key) is None or str(args.get(key)).strip() == ""
    ]
    if missing:
        return ToolResult(
            ok=False, tool=name, args=args,
            content=f"Missing required argument(s): {', '.join(missing)}. Reissue the tool call with complete arguments.",
            error_type="missing_args", retryable=True,
        )

    if name == "deep_search" and not (args.get("query") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: query must be a non-empty string. Reissue with a focused research query.",
            error_type="missing_args", retryable=True,
        )
    if name == "deep_research" and not (args.get("query") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: query must be a non-empty string. Reissue with a research question.",
            error_type="missing_args", retryable=True,
        )
    if name == "learn_knowledge" and not (
        (args.get("text") or "").strip() or (args.get("relative_path") or "").strip()
        ):
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: provide text or relative_path with knowledge to store.",
            error_type="missing_args", retryable=True,
        )
    if name in _SOCIAL_POST_TOOLS and not (args.get("draft_dir") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: draft_dir must be a non-empty path to a review bundle.",
            error_type="missing_args", retryable=True,
        )

    return None


def _classify_result(name: str, args: dict, content: str, attempts: int = 1) -> ToolResult:
    """Convert legacy string tool output into a structured result."""
    text = content or ""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        match = _ERROR_PREFIX_RE.match(stripped)
        label = (match.group("label") if match else "tool failed").lower()
        detail = match.group("detail") if match else stripped.strip("[]")
        retryable = any(marker in label for marker in ("search failed", "fetch failed"))
        retryable = retryable or any(marker in (detail or "").lower() for marker in ("timeout", "connection", "empty response"))
        return ToolResult(
            ok=False, tool=name, args=args, content=stripped,
            error_type=label.replace(" ", "_"),
            retryable=retryable,
            attempts=attempts,
            metadata={"detail": detail or label},
        )
    return ToolResult(ok=True, tool=name, args=args, content=text, attempts=attempts)


def dispatch_tool(name: str, args: dict, owner=None) -> str:
    """Run one named tool with already-decoded JSON args.

    ``owner`` is the AikoThink instance driving this agentic turn.
    deep_search and deep_research both need it for the shared embedder;
    deep_research additionally needs the already-loaded local LLM
    client/model for its adaptive continue/refine and synthesis steps.
    Every other tool is a pure function of its args and ignores it.
    """
    if name == "deep_research":
        return deep_research(
            args.get("query", ""),
            client=getattr(owner, "_client", None),
            model=getattr(owner, "_llm_model", None),
            embedder=_owner_embedder(owner),
        )
    if name == "deep_search":
        return deep_search(
            args.get("query", ""),
            embedder=_owner_embedder(owner),
        )
    if name == "run_master_plan":
        return schema.run_master_plan_json(
            args.get("task", ""),
            cap_ids=args.get("cap_ids") if isinstance(args.get("cap_ids"), list) else None,
            embedder=_owner_embedder(owner),
        )
    if name == "learn_knowledge":
        if (args.get("relative_path") or "").strip():
            doc_id = ingest_knowledge_file(
                args.get("relative_path", ""),
                title=args.get("title") or None,
                kind=args.get("kind", "ingested"),
                embedder=_owner_embedder(owner),
            )
        else:
            doc_id = ingest_knowledge_text(
                args.get("title", "Learned knowledge"),
                args.get("text", ""),
                source=args.get("source", ""),
                kind=args.get("kind", "ingested"),
                embedder=_owner_embedder(owner),
            )
        return json.dumps({"ok": bool(doc_id), "doc_id": doc_id}, ensure_ascii=False)
    entry = _TOOLS.get(name)
    if not entry or entry[1] is None:
        return f"[unknown tool: {name}]"
    if name == "save_note":
        args["content"] = args.get("content", "")[:AGENT_NOTE_MAX_CHARS]
        args["title"] = args.get("title", "aiko-note")
    return entry[1](args)


def dispatch_tool_checked(name: str, args: dict, owner=None) -> ToolResult:
    """Run a tool and return a structured result, catching unexpected exceptions."""
    try:
        content = dispatch_tool(name, args, owner=owner)
    except Exception as e:
        log.exception("Tool %s raised unexpectedly", name)
        return ToolResult(
            ok=False, tool=name, args=args,
            content=f"[tool exception: {e}]",
            error_type="tool_exception",
            retryable=False,
        )
    return _classify_result(name, args, str(content))


def _max_attempts_for(name: str) -> int:
    if name == "deep_research":
        return max(1, int(os.getenv("AGENT_DEEP_RESEARCH_ATTEMPTS", 1)))
    if name == "deep_search":
        return max(1, int(os.getenv("AGENT_WEB_TOOL_ATTEMPTS", 2)))
    if name in {"save_note", "schedule_job", "schedule_reminder"}:
        return max(1, int(os.getenv("AGENT_LOCAL_TOOL_ATTEMPTS", 1)))
    if name in _SOCIAL_POST_TOOLS:
        # Posting tools are never worth auto-retrying: a false-negative
        # retry after a real post would risk a duplicate, and the human-
        # approval / already-posted checks are deterministic, not
        # transient failures that retrying would fix.
        return 1
    return 1


def execute_tool_with_policy(name: str, args: dict, state: TaskState, owner=None) -> ToolResult:
    """Validate, run, retry, and ledger one tool call."""
    validation = _validate_args(name, args)
    if validation is not None:
        state.record(validation)
        return validation

    last = ToolResult(ok=False, tool=name, args=args, content="[tool did not run]", error_type="not_run")
    for attempt in range(1, _max_attempts_for(name) + 1):
        last = dispatch_tool_checked(name, dict(args), owner=owner)
        last.attempts = attempt
        if last.ok or not last.retryable:
            break
        if attempt < _max_attempts_for(name):
            time.sleep(AGENT_TOOL_RETRY_BACKOFF * attempt)

    state.record(last)
    return last


def _research_call_count(state: TaskState) -> int:
    """How many times deep_search/deep_research have already SUCCEEDED in
    this workflow. The two tools share one counted budget so one task cannot
    keep spending web/research calls indefinitely."""
    return sum(1 for step in state.steps if step["tool"] in _RESEARCH_TOOLS and step["ok"])


def _compact_processed_tool_context(messages: list[dict], preview_chars: int = 1500) -> None:
    """Shrink already-consumed tool observations once a later assistant
    message has arrived, instead of letting every tool call in the loop
    accumulate uncompacted for all MAX_AGENT_ITER iterations.

    Generalized from a research-tool-only rule: any tool's output can
    accumulate across iterations (repo_read_file, search_jobs, etc.), not
    just deep_search/deep_research, so this now applies uniformly
    to any tool-role message over _COMPACTABLE_MIN_CHARS.
    """
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if '"context_compacted"' in content:
            continue
        if len(content) < _COMPACTABLE_MIN_CHARS:
            continue
        try:
            parsed = json.loads(content)
            original_content = str(parsed.get("content", content))
        except (json.JSONDecodeError, AttributeError):
            original_content = content
        message["content"] = json.dumps(
            {
                "ok": True,
                "tool": message.get("name"),
                "context_compacted": True,
                "evidence_preview": original_content[:preview_chars],
            },
            ensure_ascii=False,
            indent=2,
        )


def _sanitize_user_facing_tool_detail(detail: str, max_chars: int = 300) -> str:
    """Redact sensitive/internal-looking details before surfacing blockers."""
    text = (detail or "").strip()
    if not text:
        return "unknown tool failure"
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[redacted]",
        text,
    )
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0|[^\s/]+\.local)([^\s)]*)", r"\1[internal-url-redacted]", text)
    text = re.sub(r"(?m)^\s*File \"[^\n]+", "File [internal path redacted]", text)
    text = re.sub(r"(?m)^\s*(Traceback \(most recent call last\):|During handling of the above exception.*)$", "[stack trace redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] or "unknown tool failure"

def _build_incomplete_task_answer(state: TaskState, last_content: str = "") -> str:
    """Create a useful final response when the model never emits final_answer."""
    lines: list[str] = []
    if state.evidence:
        lines.append("I completed these step(s):")
        for item in state.evidence[-5:]:
            lines.append(f"- {item[:600]}")
    if state.failures:
        lines.append("I could not fully complete the task because of these blocker(s):")
        for failure in state.failures[-3:]:
            detail = _sanitize_user_facing_tool_detail(failure.content or failure.error_type or "")
            lines.append(f"- {failure.tool}: {detail}")
    if last_content.strip():
        lines.append("Most recent model draft:")
        lines.append(last_content.strip())
    if not lines:
        lines.append(
            "I could not complete the task before the agent loop reached its step limit, "
            "and no tool results were recorded."
        )
    return "\n".join(lines)

def _coerce_verifier_bool(value) -> bool:
    """Parse verifier booleans without treating non-empty strings as True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "pass", "passed"}
    return bool(value)


def _verify_final_answer(owner, user_input: str, answer: str, state: TaskState) -> VerificationResult:
    """Check answer completeness and evidence support before Aiko speaks it."""
    issues: list[str] = []
    stripped = (answer or "").strip()
    lowered = stripped.lower()

    if not stripped:
        issues.append("The final answer is empty.")

    if state.failures and not _DISCLOSURE_RE.search(stripped):
        failed = ", ".join(f.tool for f in state.failures[-3:])
        issues.append(f"Unresolved tool failure(s) were not disclosed: {failed}.")

    if any(step["tool"] == "save_note" and step["ok"] for step in state.steps):
        if "path" not in lowered and "workspace" not in lowered and ".md" not in lowered:
            issues.append("A saved note was created, but the final answer does not mention where it was saved.")

    if any(step["tool"] in {"schedule_job", "schedule_reminder"} and step["ok"] for step in state.steps):
        if "scheduled" not in lowered and "reminder" not in lowered and "alarm" not in lowered:
            issues.append("A schedule/reminder tool succeeded, but the final answer does not confirm it.")

    # A social post tool actually running and succeeding this turn means a
    # real external action DID happen — that is not a hallucinated claim,
    # unlike every other "posted"/"sent"/"ordered" mention this heuristic
    # exists to catch. Only flag the external-action language when no real
    # post_* tool ran successfully.
    posted_for_real = any(
        step["tool"] in _SOCIAL_POST_TOOLS and step["ok"] for step in state.steps
    )
    if _EXTERNAL_ACTION_RE.search(user_input) and not _LOCAL_ARTIFACT_RE.search(stripped) and not posted_for_real:
        issues.append("The answer may imply an unsupported external action instead of a local draft/staged artifact.")

    if not issues and AGENT_VERIFY_LLM_MODE in ("off", "auto"):
        return VerificationResult(ok=True, feedback="Deterministic checks passed; LLM verify skipped.", score=1.0)
    if issues and AGENT_VERIFY_LLM_MODE == "off":
        return VerificationResult(ok=False, feedback="\n".join(issues), score=0.0)


    deterministic_note = f"\n\nDeterministic checks flagged (weigh, don't auto-fail): {'; '.join(issues)}" if issues else ""

    prompt = (
        "You are Aiko's final-answer verifier. This is NOT just a JSON schema check. "
        "Judge whether the candidate answer is accurate, complete, and supported by "
        "the task ledger/tool evidence. Do not use outside knowledge to bless facts that "
        "are missing from the ledger. Fail answers that invent unsupported details, hide "
        "tool failures, imply external actions that were not performed, omit required paths "
        "or confirmations, or do not answer the user's request. Return ONLY compact JSON "
        "with keys: pass (boolean), score (0-1), feedback (string). Do not add markdown.\n\n"
        f"User request:\n{user_input}\n\n"
        f"Task ledger/tool evidence:\n{state.summary()}\n\n"
        f"Candidate answer:\n{stripped}"
    )
    try:
        resp = owner._client.chat.completions.create(
            model=owner._llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=160,
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        ok = _coerce_verifier_bool(data.get("pass"))
        raw_score = data.get("score", 1.0 if ok else 0.0)
        feedback = str(data.get("feedback") or ("Verifier passed." if ok else "Verifier failed."))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
            ok = False
            feedback = "Verifier returned an invalid score."
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            ok = False
            feedback = f"Verifier returned an out-of-range score: {raw_score!r}."
            score = 0.0
        if score < AGENT_VERIFY_MIN_SCORE:
            ok = False
            if feedback == "Verifier passed.":
                feedback = f"Verifier score {score:.2f} below threshold {AGENT_VERIFY_MIN_SCORE:.2f}."
        return VerificationResult(ok=ok, feedback=feedback, score=score)
    except Exception as e:
        log.warning("Agent verifier failed; falling back to deterministic pass: %s", e)
        return VerificationResult(ok=True, feedback="Verifier unavailable; deterministic checks passed.", score=0.75)


def _estimate_tokens(text: str) -> int:
    """Rough chars/4 token estimate — good enough for a budget guard, not
    for billing/accounting."""
    return max(1, len(text) // 4)


def _enforce_agentic_context_budget(
    persona, agentic_policy_context, memory_context, user_input,
    wiki_context, skill_context, knowledge_context, experience_context,
    task_mode_context: str = "",
    tool_schemas: list | None = None,
    scores: dict[str, float] | None = None,
) -> tuple[str, str, str, str, str, str]:
    budget = int(LLM_CTX_SIZE * AGENT_CONTEXT_BUDGET_RATIO)
    fixed = persona + memory_context + user_input
    # Estimate from the ACTUAL filtered tool schemas sent to the LLM this
    # turn (10-12 after capability match), not the full 25-tool corpus —
    # over-reserving for every schema starves task-specific context blocks.
    tool_tokens = _estimate_tokens(json.dumps(tool_schemas or []))
    fixed_tokens = _estimate_tokens(fixed) + tool_tokens

    blocks = {
        "wiki": wiki_context, "knowledge": knowledge_context,
        "experience": experience_context, "agentic_policy": agentic_policy_context,
        "skill": skill_context, "task_mode": task_mode_context,
    }
    scores = scores or {}
    # task_mode guidance has no task-specific relevance score; treat it as
    # neutral so it sheds after clearly-irrelevant blocks (low score) but
    # before valuable task-specific data (high score).
    scores.setdefault("task_mode", 0.0)
    # fallback tie-break preserves your original weakest-first order when
    # scores are missing or tied
    fallback_rank = {"experience": 0, "wiki": 1, "knowledge": 2, "agentic_policy": 3, "skill": 4, "task_mode": 5}
    remaining = set(blocks)

    while remaining:
        total_tokens = fixed_tokens + sum(_estimate_tokens(v) for v in blocks.values())
        if total_tokens <= budget:
            break
        victim = min(remaining, key=lambda k: (scores.get(k, -1.0), fallback_rank[k]))
        log.warning(
            "[agentic] context budget exceeded (%s > %s est. tokens); dropping %s (score=%.3f)",
            total_tokens, budget, victim, scores.get(victim, -1.0),
        )
        blocks[victim] = f"<{victim}_context>\nOmitted this turn — context budget exceeded.\n</{victim}_context>"
        remaining.discard(victim)

    return blocks["wiki"], blocks["skill"], blocks["knowledge"], blocks["agentic_policy"], blocks["experience"], blocks["task_mode"]


def _stream_agent_message(owner, messages, tools, token_callback):
    """Stream an agentic LLM call, feeding text tokens to token_callback.
    Returns (SimpleNamespace, usage) matching the non-streaming shape.
    """
    stream = owner._client.chat.completions.create(
        model=owner._llm_model, messages=messages, tools=tools,
        tool_choice="auto", stream=True, max_tokens=AGENT_MAX_TOKENS,
        temperature=0.3,
    )
    content_parts = []
    tc_deltas = {}
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
            if token_callback:
                token_callback(delta.content)
        if delta.tool_calls:
            for tcd in delta.tool_calls:
                idx = tcd.index
                if idx not in tc_deltas:
                    tc_deltas[idx] = SimpleNamespace(
                        id=tcd.id or "", name="", args=[],
                    )
                if tcd.id:
                    tc_deltas[idx].id = tcd.id
                if tcd.function:
                    if tcd.function.name:
                        tc_deltas[idx].name = tcd.function.name
                    if tcd.function.arguments:
                        tc_deltas[idx].args.append(tcd.function.arguments)

    content = "".join(content_parts) if content_parts else None
    tool_calls = None
    if tc_deltas:
        tool_calls = [
            SimpleNamespace(
                id=d.id,
                function=SimpleNamespace(name=d.name, arguments="".join(d.args)),
            )
            for d in (tc_deltas[i] for i in sorted(tc_deltas))
        ]

    def _dump(exclude_none=True):
        d = {"role": "assistant", "content": content}
        if tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        return d

    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    msg.model_dump = _dump
    return msg, usage


def run_agentic_chat(owner, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None, cap_vec: np.ndarray | None = None) -> str:
    """Run task mode using the owning AikoThink instance for model/memory/output.

    mem_kb_future: a concurrent.futures.Future from
    owner._fetch_memory_and_knowledge(user_input), submitted by
    cognition.think.route() BEFORE intent was resolved to "agentic" (memory+KB
    are intent-agnostic, so route() doesn't wait for routing to finish
    before starting them). If this is None (e.g. a scheduled job calling
    agentic_chat() directly, with no prior route() call), the fetch runs
    here instead.

    query_vec — pre-computed _QUERY_INSTRUCT embedding of user_input.
    cap_vec  — pre-computed _CAPABILITY_INSTRUCT embedding of user_input.
    Both avoid redundant HTTP calls when provided.
    """
    # Reuse the same HarrierEmbedder instance already warm for memory search
    # and intent routing for every RAG-selection call below (agentic policy,
    # wiki, skill, experience, and now capability matching). Falls back to
    # keyword scoring automatically if unavailable.
    _embedder = _owner_embedder(owner)

    _query_vec = query_vec
    _cap_vec = cap_vec

    # Narrow the tool list actually sent to the LLM this turn. Previously
    # every _TOOL_SCHEMAS entry (~20 tools) was sent on every turn regardless
    # of relevance — a real cost for a 3B model's tool-selection accuracy.
    # No match -> filtered_tool_schemas returns everything unchanged, so this
    # can only shrink the list, never regress a turn.
    _matched_caps = match_capabilities(user_input, embedder=_embedder, query_vector=_cap_vec)
    tools = filtered_tool_schemas(tool_schemas(), _matched_caps)

    # Graph-first executor: known master-plan workflows can run without an LLM
    # planning loop. Novel/ambiguous tasks return None and fall back to the
    # ReAct loop once; the normal experience recorder below then captures the
    # successful sequence for later promotion into the graph master plan.
    if AGENT_EXECUTOR_MODE in {"graph", "hybrid"}:
        graph_result = schema.run_schema_agent(user_input, cap_ids=_matched_caps, embedder=_embedder)
        if graph_result is not None:
            _graph_ok = not any(not r.ok for r in graph_result.results)

            # Build a TaskState from the graph's node results so the SAME
            # final-answer verifier used by ReAct also scrutinizes graph-
            # executed answers. Previously the graph path never called
            # _verify_final_answer at all, so a node-level failure or an
            # answer that quietly omitted a required disclosure could reach
            # the user with zero scrutiny, unlike every ReAct answer.
            graph_state = TaskState(goal=user_input)
            for r in graph_result.results:
                graph_state.record(ToolResult(
                    ok=r.ok, tool=r.tool, args=r.args, content=r.content,
                    error_type=r.error_type, retryable=False, attempts=1,
                ))

            graph_verdict: VerificationResult | None = None
            if AGENT_VERIFY_FINAL:
                graph_verdict = _verify_final_answer(owner, user_input, graph_result.final_answer, graph_state)

            graph_trustworthy = _graph_ok and (graph_verdict is None or graph_verdict.ok)

            if graph_trustworthy:
                threading.Thread(
                    target=experience.record_experience,
                    args=(owner, user_input, graph_result.steps, graph_result.final_answer),
                    kwargs=dict(verified_ok=True, score=graph_verdict.score if graph_verdict else 1.0, embedder=_embedder),
                    daemon=True,
                ).start()
                graph_payload = {
                    "id": graph_result.graph.id,
                    "name": graph_result.graph.name,
                    "goal": graph_result.graph.goal,
                    "source": graph_result.graph.source,
                    "nodes": [n.__dict__ for n in graph_result.graph.nodes],
                }
                node_payload = [r.__dict__ for r in graph_result.results]
                owner.last_prompt_debug = {
                    "mode": "agentic_graph",
                    "matched_capabilities": _matched_caps,
                    "graph": graph_payload,
                    "node_results": node_payload,
                }
                owner.last_usage = {
                    "prompt_messages": [{"role": "user", "content": user_input}],
                    "completion_text": graph_result.final_answer,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                owner._emit(graph_result.final_answer, token_callback=token_callback)
                with owner._history_lock:
                    owner._history.append({"role": "user", "content": user_input})
                    owner._history.append({"role": "assistant", "content": graph_result.final_answer})
                owner._store_async(user_input, graph_result.final_answer)
                return graph_result.final_answer

            # Graph produced an untrustworthy result (a node failed, and/or
            # the verifier rejected it). Record it as a failed/partial
            # experience regardless of what happens next, so this graph
            # template stops getting reinforced by a result nobody actually
            # trusted.
            log.warning(
                "[agentic] graph result untrustworthy (nodes_ok=%s, verified=%s); "
                "executor_mode=%s",
                _graph_ok, graph_verdict.ok if graph_verdict else None, AGENT_EXECUTOR_MODE,
            )
            threading.Thread(
                target=experience.record_experience,
                args=(owner, user_input, graph_result.steps, graph_result.final_answer),
                kwargs=dict(verified_ok=False, score=graph_verdict.score if graph_verdict else 0.0, embedder=_embedder),
                daemon=True,
            ).start()

            if AGENT_EXECUTOR_MODE == "graph":
                # No ReAct fallback allowed in pure graph mode; surface the
                # graph's own (already failure-disclosing) text as before.
                final_text = graph_result.final_answer
                owner.last_usage = {
                    "prompt_messages": [{"role": "user", "content": user_input}],
                    "completion_text": final_text,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                owner._emit(final_text, token_callback=token_callback)
                with owner._history_lock:
                    owner._history.append({"role": "user", "content": user_input})
                    owner._history.append({"role": "assistant", "content": final_text})
                owner._store_async(user_input, final_text)
                return final_text
            # else: AGENT_EXECUTOR_MODE == "hybrid" — fall through to the
            # ReAct loop below instead of trusting a graph result that
            # failed a node or failed verification. This is the actual
            # "hybrid" fallback the docstring/synthesized text promised but
            # the old code never performed.

        if graph_result is None and AGENT_EXECUTOR_MODE == "graph":
            final_text = (
                "I could not match this task to a saved master-plan workflow, "
                "and AGENT_EXECUTOR_MODE=graph disables the ReAct fallback. "
                "Run practice.py or switch to AGENT_EXECUTOR_MODE=hybrid to learn it once."
            )
            owner._emit(final_text, token_callback=token_callback)
            owner.last_usage = {
                "prompt_messages": [{"role": "user", "content": user_input}],
                "completion_text": final_text,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            with owner._history_lock:
                owner._history.append({"role": "user", "content": user_input})
                owner._history.append({"role": "assistant", "content": final_text})
            owner._store_async(user_input, final_text)
            return final_text

    if mem_kb_future is not None:
        try:
            memories, knowledge_block = mem_kb_future.result()
        except Exception as e:
            log.error("Memory/KB fetch failed: %s", e)
            memories, knowledge_block = [], "<knowledge_context>\nLookup failed.\n</knowledge_context>"
    else:
        memories, knowledge_block = owner._fetch_memory_and_knowledge(user_input, query_vector=_query_vec)

    memory_block = owner._memorize.format_for_context(memories)
    memory_context = memory_block or "<memory_context>\nNo relevant memories found.\n</memory_context>"
    memory_context = _blank_empty_context(memory_context)

    # Wiki/policy/skill/experience are agentic-only — fetched now,
    # concurrently with each other, since intent has already resolved to
    # "agentic" by the time run_agentic_chat runs.
    agentic_ctx = _fetch_agentic_only_context(user_input, embedder=_embedder, query_vector=_query_vec)
    scores = agentic_ctx.pop("_scores", {})
    # Blank empty "No ... found." placeholders (P4-class fix generalized
    # from experience_context to every agentic block). The budget logic
    # below then drops these zero-information blocks instead of injecting
    # their XML wrappers on every turn.
    agentic_policy_context = _blank_empty_context(agentic_ctx["agentic_policy"])
    wiki_context = _blank_empty_context(agentic_ctx["wiki"])
    skill_context = _blank_empty_context(agentic_ctx["skill"])
    experience_context = _blank_empty_context(agentic_ctx["experience"])
    wiki_knowledge_block = _blank_empty_context(agentic_ctx["wiki_knowledge"])
    knowledge_block = _blank_empty_context(knowledge_block)
    knowledge_context = f"{wiki_knowledge_block}\n\n{knowledge_block}" if wiki_knowledge_block else knowledge_block
    # Safety net: any experience block lacking a real <past_task> element
    # (e.g. a future placeholder not covered by _EMPTY_CONTEXT_MARKERS)
    # is dropped rather than injected.
    if "<past_task" not in experience_context:
        experience_context = ""
    scores["knowledge"] = reason.batch_block_relevance_scores(_embedder, user_input, [knowledge_context], query_vector=_query_vec)[0]

    wiki_context, skill_context, knowledge_context, agentic_policy_context, experience_context, task_mode_guidance = _enforce_agentic_context_budget(
        owner._persona, agentic_policy_context, memory_context, user_input,
        wiki_context, skill_context, knowledge_context, experience_context,
        task_mode_context=TASK_MODE_GUIDANCE,
        tool_schemas=tools,
        scores=scores,
    )

    # Core task-mode rules are always kept (small, operationally essential);
    # the verbose guidance is droppable under context-budget pressure.
    agent_system = (
        f"{owner._current_system_prompt()}\n\n"
        f"{bioclock.current_datetime_block()}\n\n"        
        f"{agentic_policy_context}\n\n"
        f"{wiki_context}\n\n"
        f"{TASK_MODE_CORE}\n\n"
        f"{memory_context}\n\n"
        f"{skill_context}\n\n"
        f"{knowledge_context}\n\n"
        f"{experience_context}\n\n"
        f"{task_mode_guidance}\n\n"
    )
    messages = [
        {"role": "system", "content": agent_system},
        *_recent_history_messages(owner, user_input, query_vector=_query_vec),
        {"role": "user", "content": user_input},
    ]
    owner.last_prompt_debug = {
        "mode": "agentic",
        "system_prompt": owner._current_system_prompt(),
        "memory_prompt": memory_context,
        "web_prompt": "",
        "agentic_prompts": [
            {"label": "agentic_policy", "content": agentic_policy_context},
            {"label": "wiki_context", "content": wiki_context},
            {"label": "skill_context", "content": skill_context},
            {"label": "knowledge_context", "content": knowledge_context},
            {"label": "experience_context", "content": experience_context},
            {"label": "task_mode_core", "content": TASK_MODE_CORE},
            {"label": "task_mode_guidance", "content": task_mode_guidance},
        ],
        "matched_capabilities": _matched_caps,
        "previous_chat_messages": [dict(m) for m in messages[1:-1]],
    }
    owner.last_usage = {
        "prompt_messages": list(messages),
        "completion_text": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    final_text = ""
    last_content = ""
    seen_calls: set[tuple[str, str]] = set()
    state = TaskState(goal=user_input)
    final_repairs = 0
    last_verdict: VerificationResult | None = None
    used_incomplete_fallback = False

    for step in range(MAX_AGENT_ITER):
        if token_callback:
            token_callback("__THINKING__\n")

        try:
            msg, usage = _stream_agent_message(owner, messages, tools, token_callback)
            last_content = msg.content or ""
            owner.last_usage = {
                "prompt_messages": list(messages),
                "completion_text": last_content,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            messages.append(msg.model_dump(exclude_none=True))
            _compact_processed_tool_context(messages)
        except Exception as e:
            log.error("Agent LLM call failed: %s", e)
            state.record(ToolResult(
                ok=False, tool="llm_call", args={},
                content=f"[llm_call_failed: {e}]",
                error_type="llm_call_failed",
                retryable=False,
            ))
            break

        if not msg.tool_calls:
            candidate = msg.content or ""
            if AGENT_VERIFY_FINAL:
                verdict = _verify_final_answer(owner, user_input, candidate, state)
                last_verdict = verdict
                if not verdict.ok and final_repairs < AGENT_MAX_FINAL_REPAIRS:
                    final_repairs += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "Verifier rejected the candidate final answer. "
                            "Repair the task or answer before finalizing.\n"
                            f"Verifier score: {verdict.score}\n"
                            f"Feedback:\n{verdict.feedback}\n\n"
                            f"Task ledger:\n{state.summary()}"
                        ),
                    })
                    continue
            final_text = candidate
            break

        # Phase 1 — pre-process all tool calls (fast, sequential checks)
        batch_calls: list[tuple[str, str, dict]] = []   # (call_id, name, args)
        final_answer_data: tuple[str, dict] | None = None  # (call_id, args)
        trailing_dropped = 0

        for call_idx, call in enumerate(msg.tool_calls):
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as e:
                result = ToolResult(
                    ok=False, tool=name, args={},
                    content=f"Invalid JSON arguments: {e}. Reissue this tool call with valid JSON.",
                    error_type="invalid_json", retryable=True,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue

            log.info("[agent] step %s → %s(%s)", step, name, args)

            if name in _RESEARCH_TOOLS and _research_call_count(state) >= AGENT_RESEARCH_MAX_CALLS:
                result = ToolResult(
                    ok=False, tool=name, args=args,
                    content=(
                        f"deep_search/deep_research have already been used "
                        f"{AGENT_RESEARCH_MAX_CALLS} time(s) in this agentic workflow. "
                        "Do not search again; use the evidence already gathered to "
                        "plan, summarize, save, or answer."
                    ),
                    error_type="research_limit_reached", retryable=False,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue

            call_key = (name, json.dumps(args, sort_keys=True))
            if name != "final_answer" and call_key in seen_calls:
                result = ToolResult(
                    ok=False, tool=name, args=args,
                    content=(
                        f"Repeated tool call skipped for {name}. Choose a different "
                        "query/argument/tool, or finalize with a disclosed limitation."
                    ),
                    error_type="repeated_tool_call", retryable=True,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue

            if name != "final_answer":
                seen_calls.add(call_key)
            if token_callback:
                token_callback(f"__TOOL__:{name}({args})\n")

            if name == "final_answer":
                final_answer_data = (call.id, args)
                trailing_dropped = len(msg.tool_calls) - call_idx - 1
                break

            batch_calls.append((call.id, name, args))

        # Phase 2 — execute batch tools in parallel, collect in original order
        if batch_calls:
            submitted = [
                (call_id, name, CONTEXT_POOL.submit(
                    execute_tool_with_policy, name, args, state, owner=owner
                ))
                for call_id, name, args in batch_calls
            ]
            for call_id, name, future in submitted:
                try:
                    result = future.result()
                except Exception as e:
                    result = ToolResult(
                        ok=False, tool=name, args={},
                        content=f"[tool execution error: {e}]",
                        error_type="execution_error", retryable=False,
                    )
                    state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call_id,
                    "name": name, "content": result.observation(),
                })

        # Phase 3 — handle final_answer on main thread
        if final_answer_data:
            call_id, args = final_answer_data
            candidate = args.get("answer", "")
            if AGENT_VERIFY_FINAL:
                verdict = _verify_final_answer(owner, user_input, candidate, state)
                last_verdict = verdict
                if not verdict.ok and final_repairs < AGENT_MAX_FINAL_REPAIRS:
                    final_repairs += 1
                    messages.append({
                        "role": "tool", "tool_call_id": call_id,
                        "name": "final_answer",
                        "content": json.dumps({
                            "ok": False, "error_type": "verification_failed",
                            "score": verdict.score, "feedback": verdict.feedback,
                            "task_ledger": json.loads(state.summary()),
                            "instruction": "Repair the missing/unsupported parts, then call final_answer again.",
                        }, ensure_ascii=False, indent=2),
                    })
                    continue
            final_text = candidate
            messages.append({
                "role": "tool", "tool_call_id": call_id,
                "name": "final_answer", "content": "Answer submitted.",
            })
            if trailing_dropped:
                log.warning("[agentic] final_answer arrived mid-batch; dropping %d remaining tool call(s)", trailing_dropped)

        if final_text:
            break

    if not final_text:
        log.warning(
            "Agent loop ended without a final answer after %s iterations; tools=%s failures=%s",
            MAX_AGENT_ITER, len(state.steps), len(state.failures),
        )
        final_text = _build_incomplete_task_answer(state, last_content)
        used_incomplete_fallback = True

    if used_incomplete_fallback:
        exp_verified_ok, exp_score = False, 0.0
    elif last_verdict is not None:
        exp_verified_ok, exp_score = last_verdict.ok, last_verdict.score
    else:
        exp_verified_ok, exp_score = True, 1.0

    threading.Thread(
        target=experience.record_experience,
        args=(owner, user_input, state.steps, final_text),
        kwargs=dict(verified_ok=exp_verified_ok, score=exp_score, embedder=_embedder),
        daemon=True,
    ).start()
    owner._emit(final_text, token_callback=token_callback)

    with owner._history_lock:
        owner._history.append({"role": "user", "content": user_input})
        owner._history.append({"role": "assistant", "content": final_text})

    owner._store_async(user_input, final_text)
    return final_text
