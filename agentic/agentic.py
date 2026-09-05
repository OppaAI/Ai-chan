"""
agentic/agentic.py

Aiko's task-mode loop: tool schemas, ReAct-style dispatch, and final response
handling. Pure tool implementations stay in agentic/toolkit/; chat facade, TTS,
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
import inspect
from collections import OrderedDict
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from system.log import get_logger
from system import bioclock
from system.userspace import current_user_id, user_state_dir, user_workspace_root
from cognition import reason
from cognition import CONTEXT_POOL
from agentic.skills import list_skillsets, load_skillset, load_skills, search_skillsets_json, skill_context_for
from agentic.wiki import wiki_agentic_contexts_for
from agentic.capability import match_capabilities, filtered_tool_schemas, resolve_handoff
from agentic.guardrails import DEFAULT_POST_ANSWER_GUARDRAILS, default_pre_tool_guardrails
from cognition.knowledge import knowledge_context_for, ingest_text as ingest_knowledge_text, ingest_file as ingest_knowledge_file
from agentic import experience, skill_learning
from agentic import graph_engine as schema
from agentic.needle import NeedleClient, NeedleError, NeedleLowConfidence
from agentic.needle_orchestrator import NeedleOrchestrator, load_needle_workers
from agentic.tools import (
    adaptive_search,
    deep_research,
    deep_read,
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
    write_report,
    search_jobs,
    draft_job_post_social,
    post_job_post_social,
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
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", 5000))
AGENT_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", 8000))
AGENT_VERIFY_FINAL = os.getenv("AGENT_VERIFY_FINAL", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM = os.getenv("AGENT_VERIFY_LLM", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM_MODE = os.getenv("AGENT_VERIFY_LLM_MODE", "auto")  # "always" | "auto" | "off"
AGENT_MAX_FINAL_REPAIRS = int(os.getenv("AGENT_MAX_FINAL_REPAIRS", 2))
AGENT_VERIFY_MIN_SCORE = float(os.getenv("AGENT_VERIFY_MIN_SCORE", "0.70"))
AGENT_TOOL_RETRY_BACKOFF = float(os.getenv("AGENT_TOOL_RETRY_BACKOFF", 0.4))
AGENT_EXECUTOR_MODE = os.getenv("AGENT_EXECUTOR_MODE", "hybrid").strip().lower()  # react | graph | hybrid
AGENT_INCLUDE_EXPERIENCE_CONTEXT = os.getenv("AGENT_INCLUDE_EXPERIENCE_CONTEXT", "0").lower() in {"1", "true", "yes", "on"}
# ``needle`` uses the local Needle 2 /complete contract for the novel ReAct
# path. Graph playbooks remain deterministic; low-confidence Needle outputs
# intentionally fall back to the configured conversational LLM.
AGENT_REACT_BACKEND = os.getenv("AGENT_REACT_BACKEND", "openai").strip().lower()
NEEDLE_BASE_URL = os.getenv("NEEDLE_BASE_URL", "http://127.0.0.1:8082")
NEEDLE_TIMEOUT = float(os.getenv("NEEDLE_TIMEOUT", "15"))
NEEDLE_CONFIDENCE_THRESHOLD = float(os.getenv("NEEDLE_CONFIDENCE_THRESHOLD", "0.85"))
NEEDLE_WORKERS = os.getenv("NEEDLE_WORKERS", "")
NEEDLE_MAX_WORKERS = os.getenv("NEEDLE_MAX_WORKERS", "4")

# Rolling STM window shared across all three chat paths. Mirrors
# CONTEXT_WINDOW_TURNS in cognition.think (kept as a distinct name here rather
# than importing cognition.think, which already imports agentic.agentic — that
# would create a circular import).
AGENT_HISTORY_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

# Max number of times adaptive_search/deep_research together can be invoked in
# ONE agentic workflow. The two tools share one budget so a single agentic
# workflow cannot keep spending web/research calls after enough evidence was gathered.
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
    "when useful, use adaptive_search for discovery/support "
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
    f"adaptive_search/deep_research together may be used at most {AGENT_RESEARCH_MAX_CALLS} "
    "time(s) per agentic workflow. After research returns, read its evidence "
    "and continue with the next productive step (plan, summarize, save, or "
    "answer) instead of searching again. In task mode, do not use "
    "web_search/web_fetch directly; adaptive_search is the general-purpose "
    "research tool and deep_research is for thorough fetched evidence. "
    "When writing notes after research: "
    "cross-check any hardware specs, commands, or version numbers against "
    "fetched page content only — never state technical facts from memory "
    "alone. If a fact cannot be confirmed from fetched content, omit it or "
    "flag it as unverified. If a research tool result explicitly says no "
    "relevant content was found, disclose that gap plainly in the final answer "
    "instead of guessing or filling it in from memory. "
    "Social posting tools (post_photo_social, post_video_social)"
    "will refuse to run on anything not already approved "
    "by a person outside this conversation, and refuse a second post of the "
    "same draft. Only call a post_* social tool when the user explicitly "
    "asks to publish/post right now — never as an automatic follow-up to "
    "drafting, and never assume a draft is approved just because it was "
    "created. If a post_* call comes back ok=false, disclose that plainly; "
    "do not tell the user something was posted unless the tool result says so. "
    "For the daily job post: when the user asks to post the job post, call "
    "post_job_post_social with NO draft_dir argument — it auto-posts the most "
    "recent human-approved draft. Do NOT run the job-search/draft graph "
    "(gen_job_post) or re-draft when the user wants to post an already-"
    "approved draft."
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
    _REPO_ROOT / "agentic" / "SKILLS.md",
    _REPO_ROOT / "agentic" / "SCHEDULE.md",
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


_ERROR_PREFIX_RE = re.compile(r"^\[(?P<label>[^:\]]+)(?::\s*(?P<detail>.*))?\]$", re.DOTALL)
# Tools that can genuinely post to a real public account. When one of these
# ran and succeeded this turn, an answer describing a real "posted" action
# is not a hallucinated external action — see _verify_final_answer.
_SOCIAL_POST_TOOLS = {"post_job_post_social", "post_photo_social", "post_video_social", "post_to_social"}
# Any tool message over this length gets compacted to a preview once a
# later assistant message has arrived — generalized from a research-only
# rule to cover every bulky tool (repo_read_file, search_jobs, etc.), since
# any of them can accumulate across MAX_AGENT_ITER iterations otherwise.
_COMPACTABLE_MIN_CHARS = 800
_RESEARCH_TOOLS = {"adaptive_search", "deep_research", "deep_read"}
_PRE_TOOL_GUARDRAILS = default_pre_tool_guardrails(AGENT_RESEARCH_MAX_CALLS)
_POST_ANSWER_GUARDRAILS = DEFAULT_POST_ANSWER_GUARDRAILS

# Capability -> turn policy (tool_domains, system_overlay, max_iter,
# research_budget) is now resolved exclusively via
# agentic.capability.resolve_handoff(), which reads the single
# CAPABILITIES table in capability.py. Do not reintroduce a second,
# hand-maintained profile table here — that was the source of the
# job_hunt/reports domain mismatch bug.


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




@dataclass(frozen=True)
class AgentContext:
    """Minimal execution context passed to tools instead of the whole owner."""

    client: Any = None
    llm_model: str | None = None
    embedder: Any = None
    user_id: str | None = None
    workspace: Path | None = None
    run_id: str | None = None
    approval_bypass: frozenset[str] = frozenset()


def _agent_context(owner=None, *, run_id: str | None = None) -> AgentContext:
    uid = getattr(owner, "user_id", None) or getattr(owner, "_user_id", None) or current_user_id()
    return AgentContext(
        client=getattr(owner, "_client", None),
        llm_model=getattr(owner, "_llm_model", None),
        embedder=_owner_embedder(owner),
        user_id=uid,
        workspace=user_workspace_root(uid),
        run_id=run_id,
    )


def _context_attr(owner_or_ctx, name: str, default=None):
    if isinstance(owner_or_ctx, AgentContext):
        return getattr(owner_or_ctx, name, default)
    legacy = {"client": "_client", "llm_model": "_llm_model"}.get(name, name)
    return getattr(owner_or_ctx, legacy, default)


def _context_embedder(owner_or_ctx):
    if isinstance(owner_or_ctx, AgentContext):
        return owner_or_ctx.embedder
    return _owner_embedder(owner_or_ctx)


AGENT_TRACE_MAX_BYTES = int(os.getenv("AGENT_TRACE_MAX_BYTES", "1048576"))
AGENT_TRACE_MAX_FILES = int(os.getenv("AGENT_TRACE_MAX_FILES", "50"))

import threading
import time

# Buffer for batched agentic trace writes
_trace_buffer: list[dict] = []
_trace_buffer_lock = threading.RLock()  # Use RLock to allow reentrant flush from same thread
_TRACE_FLUSH_INTERVAL = 5.0  # flush every 5 seconds
_trace_last_flush = time.time()


def _trim_trace_dir(trace_dir: Path) -> None:
    for old in files[AGENT_TRACE_MAX_FILES:]:
        try:
            old.unlink()
        except OSError:
            log.debug("agentic: failed to unlink old trace file")


def _flush_trace_buffer() -> None:
    """Flush the trace buffer to disk."""
    global _trace_last_flush
    with _trace_buffer_lock:
        if not _trace_buffer:
            return
        # Group by (user_id, run_id) to write to correct files
        by_path: dict[Path, list[dict]] = {}
        for rec in _trace_buffer:
            uid = rec.get("user_id")
            run_id = rec.get("run_id", "default")
            trace_dir = user_state_dir(uid) / "agentic" / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            path = trace_dir / f"{run_id}.jsonl"
            by_path.setdefault(path, []).append(rec)

        for path, records in by_path.items():
            try:
                if path.exists() and path.stat().st_size >= AGENT_TRACE_MAX_BYTES:
                    path.rename(path.parent / f"{path.stem}.{int(time.time())}.jsonl")
                with path.open("a", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except Exception as exc:
                log.debug("failed to flush agentic trace batch: %s", exc)

        _trace_buffer.clear()
        _trace_last_flush = time.time()


def _append_step_trace(ctx: AgentContext | None, event: str, payload: dict[str, Any]) -> None:
    try:
        record = {"ts": time.time(), "event": event, "run_id": ctx.run_id if ctx and ctx.run_id else "default", **payload}
        if ctx and ctx.user_id:
            record["user_id"] = ctx.user_id
        flush_immediately = event in ("approval", "approval_resume", "approval_wait", "checkpoint")
        with _trace_buffer_lock:
            _trace_buffer.append(record)
            # Flush if interval exceeded or for critical events (approval, checkpoint)
            if flush_immediately or time.time() - _trace_last_flush >= _TRACE_FLUSH_INTERVAL:
                _flush_trace_buffer()
    except Exception as exc:  # pragma: no cover - tracing must never break tools
        log.debug("failed to append agentic trace: %s", exc)


def _pending_approval_path(ctx: AgentContext) -> Path:
    root = user_state_dir(ctx.user_id) / "agentic" / "pending_approvals"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{ctx.run_id}.json"


def _persist_pending_approval(ctx: AgentContext, tool: str, args: dict[str, Any], state: "TaskState") -> None:
    payload = {"run_id": ctx.run_id, "tool": tool, "args": args, "checkpoint": json.loads(state.summary()), "created_at": time.time()}
    _pending_approval_path(ctx).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


_APPROVAL_REJECT_RE = re.compile(
    r"\b(?:no|nope|nah|deny|decline|reject|cancel|stop|not\s+now|later|wait|hold\s+off)\b",
    re.I,
)
_APPROVAL_ACCEPT_RE = re.compile(
    r"\b(?:y|yes|yeah|yep|sure|ok(?:ay)?|confirm|approve|approved|resume|continue|go\s+ahead|do\s+it|proceed)\b",
    re.I,
)


def _looks_like_approval_reply(user_input: str) -> bool:
    """Conservative approval parser for pending-tool resumes.

    This runs before normal routing, so it must accept common short approvals
    ("yeah", "sure") while refusing mixed or delayed replies such as
    "yes, but not now" instead of trying to infer user intent semantically.
    """
    text = (user_input or "").strip()
    if not text or _APPROVAL_REJECT_RE.search(text):
        return False
    if not _APPROVAL_ACCEPT_RE.search(text):
        return False
    # Avoid hijacking ordinary chat that happens to contain an approval word.
    return bool(
        re.fullmatch(
            r"[\s.!?,;:-]*(?:y|yes|yeah|yep|sure|ok(?:ay)?|confirm|approve|approved|resume|continue|go\s+ahead|do\s+it|proceed)"
            r"(?:\s+(?:please|pls|now))?[\s.!?,;:-]*(?:\b(?:run-[0-9]+|r\d[0-9A-Za-z_-]*)\b)?[\s.!?,;:-]*",
            text,
            re.I,
        )
        or bool(
            re.fullmatch(
                r"[\s.!?,;:-]*(?:(?:y|yes|yeah|yep|sure|ok(?:ay)?|confirm|approve|approved|resume|continue|go\s+ahead|do\s+it|proceed)[\s.!?,;:-]+)?"
                r"(?:approve|confirm|resume|continue)\s+(?:run-[0-9]+|r\d[0-9A-Za-z_-]*)[\s.!?,;:-]*",
                text,
                re.I,
            )
        )
    )


def _maybe_resume_approval(owner, user_input: str, token_callback=None) -> str | None:
    if not _looks_like_approval_reply(user_input):
        return None
    base_ctx = _agent_context(owner)
    match = re.search(r"\b(run-[0-9]+|r\d[0-9A-Za-z_-]*)\b", user_input or "")
    if match:
        run_id = match.group(1)
    else:
        pending_root = user_state_dir(base_ctx.user_id) / "agentic" / "pending_approvals"
        pending = sorted(pending_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if pending_root.exists() else []
        if len(pending) != 1:
            return None
        run_id = pending[0].stem
    ctx = _agent_context(owner, run_id=run_id)
    path = _pending_approval_path(ctx)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    tool_name = str(data.get("tool") or "")
    state = TaskState(goal=f"resume approval {ctx.run_id}")
    resume_ctx = AgentContext(
        client=ctx.client, llm_model=ctx.llm_model, embedder=ctx.embedder,
        user_id=ctx.user_id, workspace=ctx.workspace, run_id=ctx.run_id,
        approval_bypass=frozenset({tool_name}),
    )
    _append_step_trace(resume_ctx, "approval_resume", {"tool": tool_name})
    result = execute_tool_with_policy(tool_name, dict(data.get("args") or {}), state, ctx=resume_ctx)
    if result.ok:
        try:
            path.unlink()
        except OSError:
            log.debug("agentic: failed to unlink approval pause file")
    final = result.observation()
    if token_callback:
        token_callback(final)
    return final

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
        """Record a tool execution result and update agent state."""
        try:
            from cognition.attention import for_identity
            from system.userspace import current_user_id
            state = for_identity(current_user_id())
            state.record_tool_outcome(
                result.tool,
                ok=result.ok,
                detail=result.content,
                error_type=result.error_type or "",
            )
            state.persist()
        except Exception:
            pass
        self.steps.append({
            "tool": result.tool,
            "ok": result.ok,
            "attempts": result.attempts,
            "error_type": result.error_type,
            "args": result.args,
            "content": result.content if result.ok else None,
            "observation": result.content if result.ok else None,
        })
        if result.ok:
            self.evidence.append(f"{result.tool}: {result.content[:500]}")
        else:
            self.failures.append(result)

    def summary(self) -> str:
        """Generate a JSON summary of the agentic plan execution."""
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
    """Return OpenAI-compatible tool schemas for autonomous task mode.

    Read live from the registry rather than the import-time ``_TOOLS``
    snapshot so tools registered after import (e.g. MCP bridge tools
    bootstrapped at wakeup, like the ProtonMail email tools) are visible
    to the LLM too. Order is deterministic (registry insertion order).
    """
    return [spec.to_openai_schema() for spec in registry.all_specs() if spec.react]


from agentic.registry import TOOLS, ValidationError, registry, register_tool_schema, tool
try:
    from agentic.tool_models import LearnKnowledgeArgs
except Exception:  # pragma: no cover - pydantic may be unavailable in minimal envs
    LearnKnowledgeArgs = None


def _bootstrap_tool_registry() -> None:
    """Import toolkit modules so @tool decorators populate the registry."""
    import agentic.tools  # noqa: F401
    from agentic.toolkit import synthesize  # noqa: F401


@tool(TOOLS["final_answer"])
def final_answer(answer: str) -> str:
    """Return the final answer from the agentic workflow."""
    return answer


def _register_agentic_local_tools() -> None:
    register_tool_schema(
        "learn_knowledge",
        "Store durable learned knowledge in Aiko's vector RAG store (encrypted when SQLite encryption is enabled). Use only when the user asks Aiko to remember/add/store knowledge, ingest pasted document text, or after explicit self-learning/research should be retained. Do not use for private personal preferences; those belong in memory. Do not use for merely saving a human-readable note; use save_note for that.",
        props={
            "title": {"type": "string", "description": "Short title for the learned document or fact set."},
            "text": {"type": "string", "description": "Knowledge text to chunk, embed, and retrieve later. Use this for pasted/extracted text."},
            "relative_path": {"type": "string", "description": "Optional workspace-relative document path to ingest instead of text."},
            "source": {"type": "string", "description": "Optional source URL/path/context for pasted text."},
            "kind": {"type": "string", "enum": ["ingested", "self_learned", "study_note"], "description": "Where this knowledge came from."},
        },
        required=["title"],
        domain="kb",
        args_model=LearnKnowledgeArgs,
    )


def _init_tool_tables() -> tuple[dict[str, tuple[dict, object]], list[tuple[dict, object]]]:
    _bootstrap_tool_registry()
    _register_agentic_local_tools()
    tool_defs = registry.get_react_defs()
    tools = {schema["function"]["name"]: (schema, handler) for schema, handler in tool_defs}
    return tools, tool_defs


_TOOLS, _TOOL_DEFS = _init_tool_tables()


def invoke_registered_tool(name: str, arguments: dict | None = None):
    """Invoke a registered agentic tool for a data-driven scheduled job.

    A schedule can select only an existing tool; it cannot import or execute
    arbitrary Python. Tool-specific safety gates still apply.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scheduled tool call requires a tool name")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("scheduled tool arguments must be an object")
    entry = _TOOLS.get(name)
    if entry is None or entry[1] is None:
        raise ValueError(f"unknown or non-invocable scheduled tool: {name}")
    return entry[1](arguments)
    

def _required_args_for(name: str) -> list[str]:
    entry = _TOOLS.get(name)
    if not entry:
        return []
    return list(entry[0].get("function", {}).get("parameters", {}).get("required", []))


def _json_safe(value):
    """Recursively coerce a value to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Exception):
        return str(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


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
    spec = registry.get(name)
    if spec and spec.args_model is not None:
        try:
            coerced = spec.validate_args(args)
        except Exception as exc:
            if ValidationError is not None and isinstance(exc, ValidationError):
                detail = _json_safe(exc.errors(include_url=False))
            else:
                detail = str(exc)
            return ToolResult(
                ok=False, tool=name, args=args,
                content=json.dumps({
                    "message": "Tool arguments failed schema validation. Reissue the call with corrected arguments.",
                    "errors": detail,
                }, ensure_ascii=False),
                error_type="schema_validation_failed", retryable=True,
            )
        args.clear()
        args.update(coerced)
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


def _dispatch_tool_impl(name: str, args: dict, owner=None) -> str:
    """Run one named tool with already-decoded JSON args.

    ``owner`` is the AikoThink instance driving this agentic turn.
    adaptive_search and deep_research both need it for the shared embedder;
    deep_research additionally needs the already-loaded local LLM
    client/model for its adaptive continue/refine and synthesis steps.
    Every other tool is a pure function of its args and ignores it.
    """
    # agentic/agentic.py — dispatch_tool()
    if name == "adaptive_search":
        return adaptive_search(
            args.get("query", ""),
            client=_context_attr(owner, "client"),
            model=_context_attr(owner, "llm_model"),
            embedder=_context_embedder(owner),
        )
    if name == "deep_research":
        return deep_research(
            args.get("query", ""),
            client=_context_attr(owner, "client"),
            model=_context_attr(owner, "llm_model"),
            embedder=_context_embedder(owner),
        )
    if name == "deep_read":
        return deep_read(
            args.get("url", ""),
            query=args.get("query", ""),
            embedder=_context_embedder(owner),
        )
    if name == "run_playbook":
        return schema.run_playbook_json(
            args.get("task", ""),
            cap_ids=args.get("cap_ids") if isinstance(args.get("cap_ids"), list) else None,
            embedder=_context_embedder(owner),
            llm_client=_context_attr(owner, "client"),
            llm_model=_context_attr(owner, "llm_model"),
        )
    if name == "learn_knowledge":
        if (args.get("relative_path") or "").strip():
            doc_id = ingest_knowledge_file(
                args.get("relative_path", ""),
                title=args.get("title") or None,
                kind=args.get("kind", "ingested"),
                embedder=_context_embedder(owner),
            )
        else:
            doc_id = ingest_knowledge_text(
                args.get("title", "Learned knowledge"),
                args.get("text", ""),
                source=args.get("source", ""),
                kind=args.get("kind", "ingested"),
                embedder=_context_embedder(owner),
            )
        return json.dumps({"ok": bool(doc_id), "doc_id": doc_id}, ensure_ascii=False)
    if name == "search_skillsets":
        return search_skillsets_json(
            args.get("query", ""),
            int(args.get("limit", 3) or 3),
            embedder=_context_embedder(owner),
        )
    if name == "write_report":
        return write_report(
            args.get("title", "Aiko report"), args.get("content", ""),
            args.get("report_dir", "reports"), bool(args.get("arxiv_style", False)),
            args.get("section", ""), bool(args.get("append", False)),
        )
    spec = registry.get(name)
    if not spec or spec.handler is None:
        return f"[unknown tool: {name}]"
    if name == "save_note":
        args["content"] = args.get("content", "")[:AGENT_NOTE_MAX_CHARS]
        args["title"] = args.get("title", "aiko-note")
    # Inject agent context (LLM client/model/embedder) into any registered
    # handler whose signature accepts them. Without this, generic tools like
    # draft_job_post_social receive client=None and silently skip their LLM
    # steps (enrichment, page fetching, verification). Mirrors the injection
    # graph_engine._run_node does for graph nodes.
    call_args = dict(args)
    handler = spec.handler
    try:
        _params = set(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        _params = set()
    if "client" in _params and "client" not in call_args:
        call_args["client"] = _context_attr(owner, "client")
    if "model" in _params and "model" not in call_args:
        call_args["model"] = _context_attr(owner, "llm_model")
    if "embedder" in _params and "embedder" not in call_args:
        call_args["embedder"] = _context_embedder(owner)
    try:
        return handler(**call_args)
    except TypeError:
        return handler(args)


def dispatch_tool(name: str, args: dict, owner=None) -> str:
    """Dispatch a tool and record its bounded outcome in cognitive state."""
    try:
        result = _dispatch_tool_impl(name, args, owner=owner)
        parsed = result if isinstance(result, dict) else None
        if parsed is None and isinstance(result, str) and result.lstrip().startswith("{"):
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = None
        ok = not (isinstance(result, str) and result.startswith("[unknown tool:"))
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            ok = False
        detail = str(result)[:240]
        error_type = "tool_error" if not ok else ""
    except Exception as exc:
        result = f"[tool error: {exc}]"
        ok = False
        detail = str(exc)[:240]
        error_type = type(exc).__name__
    try:
        from cognition.attention import for_identity
        from system.userspace import current_user_id
        for_identity(current_user_id()).record_tool_outcome(name, ok=ok, detail=detail, error_type=error_type)
    except Exception:
        pass
    return result

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
    if name in {"save_note", "schedule_job", "schedule_reminder"}:
        return max(1, int(os.getenv("AGENT_LOCAL_TOOL_ATTEMPTS", 1)))
    if name in _SOCIAL_POST_TOOLS:
        # Posting tools are never worth auto-retrying: a false-negative
        # retry after a real post would risk a duplicate, and the human-
        # approval / already-posted checks are deterministic, not
        # transient failures that retrying would fix.
        return 1
    return 1


_PREFERENCE_READ_ONLY_TOOLS = frozenset({
    "adaptive_search", "deep_read", "deep_research", "read_workspace_file",
    "repo_file_tree", "repo_read_file", "repo_search_text", "search_jobs",
    "list_schedule", "list_reminders", "scan_photo_workspace", "summarize_task_state",
})

def _preference_requires_approval(name: str) -> bool:
    try:
        from cognition.attention import for_identity
        preferences = for_identity(current_user_id()).snapshot().get("preferences", {})
        return preferences.get("action_confirmation") == "ask_before_acting" and name not in _PREFERENCE_READ_ONLY_TOOLS
    except Exception:
        return False

def execute_tool_with_policy(name: str, args: dict, state: TaskState, owner=None, ctx: AgentContext | None = None, guards=None) -> ToolResult:
    """Validate, guard, run, retry, and ledger one tool call."""
    ctx = ctx or (owner if isinstance(owner, AgentContext) else _agent_context(owner))
    _append_step_trace(ctx, "tool_call", {"tool": name, "args": args})
    spec = registry.get(name)
    for guard in (guards or _PRE_TOOL_GUARDRAILS):
        verdict = guard(name, args, state)
        if verdict is not None:
            result = ToolResult(
                ok=False, tool=name, args=args, content=verdict.content,
                error_type=verdict.error_type, retryable=verdict.retryable,
                metadata=verdict.metadata,
            )
            state.record(result)
            return result

    validation = _validate_args(name, args)
    if validation is not None:
        state.record(validation)
        return validation

    preference_gate = _preference_requires_approval(name)
    if ((spec and spec.needs_approval) or preference_gate) and name not in ctx.approval_bypass:
        _persist_pending_approval(ctx, name, args, state)
        draft_dir = args.get("draft_dir") if isinstance(args, dict) else None
        wait_payload = {"status": "waiting_for_approval", "run_id": ctx.run_id, "instruction": f"Reply with approve {ctx.run_id} to run {name}."}
        if preference_gate and not (spec and spec.needs_approval):
            wait_payload["reason"] = "Your saved preference requires confirmation before consequential actions."
        if draft_dir:
            wait_payload["draft_dir"] = draft_dir
            wait_payload["instruction"] = (
                f"Review the draft at {draft_dir} and approve it outside this "
                f"conversation, then reply with approve {ctx.run_id} to run {name}."
            )
        result = ToolResult(ok=False, tool=name, args=args, content=json.dumps(wait_payload, ensure_ascii=False), error_type="needs_approval", retryable=False, metadata={"run_id": ctx.run_id, "checkpoint": state.summary()})
        state.record(result)
        _append_step_trace(ctx, "approval_wait", {"tool": name, "args": args})
        _append_step_trace(ctx, "tool_result", {"tool": name, "ok": False, "error_type": "needs_approval"})
        return result

    last = ToolResult(ok=False, tool=name, args=args, content="[tool did not run]", error_type="not_run")
    for attempt in range(1, _max_attempts_for(name) + 1):
        last = dispatch_tool_checked(name, dict(args), owner=ctx)
        last.attempts = attempt
        if last.ok or not last.retryable:
            break
        if attempt < _max_attempts_for(name):
            time.sleep(AGENT_TOOL_RETRY_BACKOFF * attempt)

    state.record(last)
    _append_step_trace(ctx, "tool_result", {"tool": name, "ok": last.ok, "error_type": last.error_type, "attempts": last.attempts})
    return last


def _research_call_count(state: TaskState) -> int:
    """How many times adaptive_search/deep_research have already SUCCEEDED in
    this workflow. The two tools share one counted budget so one task cannot
    keep spending web/research calls indefinitely."""
    return sum(1 for step in state.steps if step["tool"] in _RESEARCH_TOOLS and step["ok"])


def _compact_processed_tool_context(messages: list[dict], preview_chars: int = 1500) -> None:
    """Shrink already-consumed tool observations once a later assistant
    message has arrived, instead of letting every tool call in the loop
    accumulate uncompacted for all MAX_AGENT_ITER iterations.

    Generalized from a research-tool-only rule: any tool's output can
    accumulate across iterations (repo_read_file, search_jobs, etc.), not
    just adaptive_search/deep_research, so this now applies uniformly
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
    stripped = (answer or "").strip()
    issues = [
        verdict.content
        for guard in _POST_ANSWER_GUARDRAILS
        if (verdict := guard(stripped, state, user_input)) is not None
    ]

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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "verification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "pass": {"type": "boolean"},
                            "score": {"type": "number"},
                            "feedback": {"type": "string"},
                        },
                        "required": ["pass", "score", "feedback"],
                    },
                },
            },
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
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


def _needle_prompt(messages: list[dict]) -> str:
    """Keep the Needle worker inside its small, tool-oriented context window."""
    task = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")
    observations = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"][-2:]
    if not observations:
        return task
    compact = "\n".join(observations)
    return f"Task: {task}\nLatest tool observations:\n{compact[-1800:]}"


def _needle_agent_message(messages, tools, token_callback):
    """Ask Needle for constrained calls; escalate low-confidence turns to OpenAI."""
    client = NeedleClient(
        NEEDLE_BASE_URL, timeout=NEEDLE_TIMEOUT,
        confidence_threshold=NEEDLE_CONFIDENCE_THRESHOLD,
    )
    response = client.complete(_needle_prompt(messages), tools)
    calls = [
        SimpleNamespace(
            id=f"needle-{index}",
            function=SimpleNamespace(name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False)),
        )
        for index, call in enumerate(response.calls)
    ]
    content = response.content or None
    if content and token_callback:
        token_callback(content)

    def _dump(exclude_none=True):
        data = {"role": "assistant", "content": content}
        if calls:
            data["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                for call in calls
            ]
        return data

    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    msg.model_dump = _dump
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    return msg, usage


def _needle_multi_agent_message(messages, tools, token_callback):
    """Fan a novel ReAct turn out to isolated, least-privilege Needle workers."""
    try:
        max_workers = int(NEEDLE_MAX_WORKERS)
    except ValueError as exc:
        raise NeedleError("NEEDLE_MAX_WORKERS must be an integer") from exc
    workers = load_needle_workers(
        NEEDLE_WORKERS,
        default_timeout=NEEDLE_TIMEOUT,
        default_confidence_threshold=NEEDLE_CONFIDENCE_THRESHOLD,
        max_workers=max_workers,
    )
    results = NeedleOrchestrator(workers).complete(_needle_prompt(messages), tools)
    calls = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        if result.response is None:
            log.info("[agentic] Needle worker %s unavailable: %s", result.worker_id, result.error)
            continue
        for index, call in enumerate(result.response.calls):
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            key = (call.name, arguments)
            if key in seen:
                continue
            seen.add(key)
            calls.append(SimpleNamespace(
                id=f"needle-{result.worker_id}-{index}",
                function=SimpleNamespace(name=call.name, arguments=arguments),
            ))
    content = next((result.response.content for result in results if result.response and result.response.content), None)
    if not calls and not content:
        raise NeedleError("Needle workers returned no calls or response content")
    if content and token_callback:
        token_callback(content)

    def _dump(exclude_none=True):
        data = {"role": "assistant", "content": content}
        if calls:
            data["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                for call in calls
            ]
        return data

    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    msg.model_dump = _dump
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    return msg, usage


def _stream_agent_message(owner, messages, tools, token_callback):
    """Stream an agentic LLM call, feeding text tokens to token_callback.
    Returns (SimpleNamespace, usage) matching the non-streaming shape.
    """
    if AGENT_REACT_BACKEND in {"needle", "needle_multi"}:
        try:
            if AGENT_REACT_BACKEND == "needle_multi":
                return _needle_multi_agent_message(messages, tools, token_callback)
            return _needle_agent_message(messages, tools, token_callback)
        except (NeedleLowConfidence, NeedleError) as exc:
            # Needle is a bounded action worker. Unavailable or uncertain calls
            # must not execute; the larger conversational model handles them.
            log.info("[agentic] Needle escalation to OpenAI: %s", exc)
    elif AGENT_REACT_BACKEND != "openai":
        log.warning("[agentic] unknown AGENT_REACT_BACKEND=%r; using openai", AGENT_REACT_BACKEND)

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



def _collect_file_artifacts(state: "TaskState") -> list[dict]:
    """Harvest workspace file paths from tool args/content for UI file chips."""
    files: list[dict] = []
    seen: set[str] = set()
    path_keys = ("relative_path", "path", "output_path", "file_path", "filepath", "dest", "destination")
    for step in getattr(state, "steps", []) or []:
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        for key in path_keys:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                p = val.strip()
                if p not in seen:
                    seen.add(p)
                    files.append({"label": p.rsplit("/", 1)[-1], "path": p})
        content = step.get("content") or step.get("observation") or ""
        if isinstance(content, str):
            import re as _re
            for m in _re.finditer(r"(?:saved|wrote|written to|output)\s*[:=]?\s*([\w./\\-]+\.[A-Za-z0-9]{1,8})", content, _re.I):
                p = m.group(1).strip()
                if p not in seen:
                    seen.add(p)
                    files.append({"label": p.rsplit("/", 1)[-1], "path": p})
    return files


def _emit_file_artifacts(token_callback, state: "TaskState") -> None:
    if not token_callback:
        return
    files = _collect_file_artifacts(state)
    if not files:
        return
    import json as _json
    token_callback("__FILES__:" + _json.dumps(files, ensure_ascii=False) + "\n")


def _finalize_agentic_answer(owner, user_input: str, draft: str, token_callback=None) -> str:
    """Audit and, when needed, repair the final answer before delivery."""
    finalize = getattr(owner, "_finalize_response", None)
    if callable(finalize):
        return finalize(user_input, draft, token_callback)
    owner._emit(draft, token_callback=token_callback)
    return draft

def run_agentic_chat(owner, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None, cap_vec: np.ndarray | None = None, output_model: Any | None = None) -> str:
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
    resumed = _maybe_resume_approval(owner, user_input, token_callback=token_callback)
    if resumed is not None:
        return resumed

    _embedder = _owner_embedder(owner)

    _query_vec = query_vec
    _cap_vec = cap_vec

    # Narrow the tool list actually sent to the LLM this turn. Previously
    # every _TOOL_SCHEMAS entry (~20 tools) was sent on every turn regardless
    # of relevance — a real cost for a 3B model's tool-selection accuracy.
    # No match -> filtered_tool_schemas returns everything unchanged, so this
    # can only shrink the list, never regress a turn.
    _matched_caps = match_capabilities(user_input, embedder=_embedder, query_vector=_cap_vec)
    handoff_profile = resolve_handoff(
        _matched_caps,
        default_max_iter=MAX_AGENT_ITER,
        default_research_budget=AGENT_RESEARCH_MAX_CALLS,
    )
    trace_ctx = _agent_context(owner, run_id=f"run-{int(time.time() * 1000)}")
    _append_step_trace(trace_ctx, "llm_step", {
        "step": 0,
        "matched_capabilities": _matched_caps,
        "handoff_profile": {
            "tool_domains": sorted(handoff_profile.tool_domains),
            "system_overlay": handoff_profile.system_overlay,
            "max_iter": handoff_profile.max_iter,
            "research_budget": handoff_profile.research_budget,
            "capability_ids": list(handoff_profile.capability_ids),
        },
    })
    # filtered_tool_schemas is the ONLY domain filter pass now — it derives
    # its domain set from the same CAPABILITIES table resolve_handoff just
    # used, so there is nothing left to reconcile with a second pass.
    tools = filtered_tool_schemas(tool_schemas(), list(handoff_profile.capability_ids))

    # Graph-first executor: known playbook workflows can run without an LLM
    # planning loop. Novel/ambiguous tasks return None and fall back to the
    # ReAct loop once; the normal experience recorder below then captures the
    # successful sequence for later promotion into the graph playbook.
    if AGENT_EXECUTOR_MODE in {"graph", "hybrid"}:
        graph_result = schema.run_schema_agent(
            user_input, cap_ids=_matched_caps, embedder=_embedder,
            llm_client=owner._client, llm_model=owner._llm_model,
        )
        if graph_result is not None:
            for _node in graph_result.results:
                _append_step_trace(trace_ctx, "graph_node", {"node_id": getattr(_node, "node_id", ""), "tool": getattr(_node, "tool", ""), "ok": getattr(_node, "ok", False), "error_type": getattr(_node, "error_type", None)})
            _graph_ok = not any(not r.ok for r in graph_result.results)

            # DEBUG: Log each node result
            for r in graph_result.results:
                status = "OK" if r.ok else "FAILED"
                content_preview = str(r.content)[:500] if r.content else "None"
                log.info(f"GRAPH NODE {status}: tool={r.tool} node_id={r.node_id if hasattr(r, 'node_id') else '?'} error_type={r.error_type} content={content_preview}")

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
                log.info(f"VERIFY: calling _verify_final_answer, final_answer_len={len(str(graph_result.final_answer)) if graph_result.final_answer else 0}")
                graph_verdict = _verify_final_answer(owner, user_input, graph_result.final_answer, graph_state)
                log.info(f"VERIFY: result ok={graph_verdict.ok} score={graph_verdict.score} feedback_len={len(graph_verdict.feedback) if graph_verdict.feedback else 0}")
                _append_step_trace(trace_ctx, "verify", {"ok": graph_verdict.ok, "score": graph_verdict.score, "feedback": graph_verdict.feedback[:500], "mode": "graph"})
            else:
                log.info("VERIFY: skipped (AGENT_VERIFY_FINAL=0)")

            graph_trustworthy = _graph_ok and (graph_verdict is None or graph_verdict.ok)
            log.info(f"GRAPH TRUSTWORTHY: _graph_ok={_graph_ok} verdict_ok={graph_verdict.ok if graph_verdict else None} -> {graph_trustworthy}")

            if graph_trustworthy:
                def _safe_dict(obj):
                    if is_dataclass(obj):
                        return asdict(obj)
                    if hasattr(obj, "__dict__"):
                        return obj.__dict__
                    if hasattr(obj, "__slots__"):
                        return {s: getattr(obj, s) for s in obj.__slots__}
                    return vars(obj)  # will raise clearly if truly unsupported

                CONTEXT_POOL.submit(
                    experience.record_experience,
                    owner, user_input, graph_result.steps, graph_result.final_answer,
                    verified_ok=True, score=graph_verdict.score if graph_verdict else 1.0, embedder=_embedder,
                )
                CONTEXT_POOL.submit(
                    skill_learning.propose_skill_from_run,
                    user_input, graph_result.steps, graph_result.final_answer,
                    verified_ok=True, score=graph_verdict.score if graph_verdict else 1.0, user_id=trace_ctx.user_id,
                )
                graph_payload = {
                    "id": graph_result.graph.id,
                    "name": graph_result.graph.name,
                    "goal": graph_result.graph.goal,
                    "source": graph_result.graph.source,
                    "nodes": [_safe_dict(n) for n in graph_result.graph.nodes],
                }
                node_payload = [_safe_dict(r) for r in graph_result.results]
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
                _emit_file_artifacts(token_callback, graph_state)
                final_text = _finalize_agentic_answer(owner, user_input, graph_result.final_answer, token_callback=token_callback)
                with owner._history_lock:
                    owner._history.append({"role": "user", "content": user_input})
                    owner._history.append({"role": "assistant", "content": final_text})
                    if len(owner._history) > AGENT_HISTORY_TURNS * 10:
                        owner._history = owner._history[-(AGENT_HISTORY_TURNS * 10):]
                owner._store_async(user_input, final_text)
                return final_text

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
            CONTEXT_POOL.submit(
                experience.record_experience,
                owner, user_input, graph_result.steps, graph_result.final_answer,
                verified_ok=False, score=graph_verdict.score if graph_verdict else 0.0, embedder=_embedder,
            )
            CONTEXT_POOL.submit(
                skill_learning.propose_skill_from_run,
                user_input, graph_result.steps, graph_result.final_answer,
                verified_ok=False, score=graph_verdict.score if graph_verdict else 0.0, user_id=trace_ctx.user_id,
            )

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
                _emit_file_artifacts(token_callback, graph_state)
                final_text = _finalize_agentic_answer(owner, user_input, final_text, token_callback=token_callback)
                with owner._history_lock:
                    owner._history.append({"role": "user", "content": user_input})
                    owner._history.append({"role": "assistant", "content": final_text})
                    if len(owner._history) > AGENT_HISTORY_TURNS * 10:
                        owner._history = owner._history[-(AGENT_HISTORY_TURNS * 10):]
                owner._store_async(user_input, final_text)
                return final_text
            # else: AGENT_EXECUTOR_MODE == "hybrid" — fall through to the
            # ReAct loop below instead of trusting a graph result that
            # failed a node or failed verification. This is the actual
            # "hybrid" fallback the docstring/synthesized text promised but
            # the old code never performed.

        if graph_result is None and AGENT_EXECUTOR_MODE == "graph":
            final_text = (
                "I could not match this task to a saved playbook workflow, "
                "and AGENT_EXECUTOR_MODE=graph disables the ReAct fallback. "
                "Run practice.py or switch to AGENT_EXECUTOR_MODE=hybrid to learn it once."
            )
            final_text = _finalize_agentic_answer(owner, user_input, final_text, token_callback=token_callback)
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
                if len(owner._history) > AGENT_HISTORY_TURNS * 10:
                    owner._history = owner._history[-(AGENT_HISTORY_TURNS * 10):]
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

    memory_block = owner._get_memorize().format_for_context(
        memories, query=user_input, query_vector=_query_vec
    )
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
        f"{handoff_profile.system_overlay}\n\n"
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

    turn_guards = default_pre_tool_guardrails(handoff_profile.research_budget)

    for step in range(handoff_profile.max_iter):
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
                _append_step_trace(trace_ctx, "verify", {"ok": verdict.ok, "score": verdict.score, "feedback": verdict.feedback[:500]})
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
                token_callback(f"__STATUS__:tool:{name}\n")

            if name == "final_answer":
                final_answer_data = (call.id, args)
                trailing_dropped = len(msg.tool_calls) - call_idx - 1
                break

            batch_calls.append((call.id, name, args))

        # Phase 2 — execute batch tools in parallel, collect in original order
        if batch_calls:
            submitted = [
                (call_id, name, CONTEXT_POOL.submit(
                    execute_tool_with_policy, name, args, state, owner=owner, ctx=trace_ctx, guards=turn_guards
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
                _append_step_trace(trace_ctx, "verify", {"ok": verdict.ok, "score": verdict.score, "feedback": verdict.feedback[:500]})
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
            if output_model is not None:
                try:
                    structured = output_model.model_validate_json(candidate) if isinstance(candidate, str) else output_model.model_validate(candidate)
                    candidate = json.dumps(structured.model_dump(mode="json"), ensure_ascii=False)
                except Exception as exc:
                    if final_repairs < AGENT_MAX_FINAL_REPAIRS:
                        final_repairs += 1
                        detail = exc.errors(include_url=False) if ValidationError is not None and isinstance(exc, ValidationError) else str(exc)
                        messages.append({
                            "role": "tool", "tool_call_id": call_id,
                            "name": "final_answer",
                            "content": json.dumps({
                                "ok": False, "error_type": "structured_output_validation_failed",
                                "errors": detail,
                                "instruction": "Return final_answer as valid JSON matching the requested structured output model.",
                            }, ensure_ascii=False, indent=2),
                        })
                        continue
                    final_text = json.dumps({"ok": False, "error_type": "structured_output_validation_failed", "errors": detail}, ensure_ascii=False)
                    break
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

    CONTEXT_POOL.submit(
        experience.record_experience,
        owner, user_input, state.steps, final_text,
        verified_ok=exp_verified_ok, score=exp_score, embedder=_embedder,
    )
    CONTEXT_POOL.submit(
        skill_learning.propose_skill_from_run,
        user_input, state.steps, final_text,
        verified_ok=exp_verified_ok, score=exp_score, user_id=trace_ctx.user_id,
    )
    _emit_file_artifacts(token_callback, state)
    final_text = _finalize_agentic_answer(owner, user_input, final_text, token_callback=token_callback)

    with owner._history_lock:
        owner._history.append({"role": "user", "content": user_input})
        owner._history.append({"role": "assistant", "content": final_text})
        if len(owner._history) > AGENT_HISTORY_TURNS * 10:
            owner._history = owner._history[-(AGENT_HISTORY_TURNS * 10):]

    owner._store_async(user_input, final_text)
    _flush_trace_buffer()
    from cognition.attention import flush_all_persist
    flush_all_persist()
    return final_text
