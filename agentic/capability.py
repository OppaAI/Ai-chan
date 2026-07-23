"""
agentic/capability.py

Capability routing for Aiko's agentic tool loop.

A capability holds no content of its own — it's a lookup that says, for a
given turn, which tool-schema domains should reach the LLM. Prose retrieval
(wiki/skill excerpts) is untouched and still goes through wiki_context_for /
skill_context_for exactly as before. This module only narrows the `tools=`
list passed to chat.completions.create(), which today is the full fixed
_TOOL_SCHEMAS set on every single turn regardless of what the turn needs.

Safe-by-default: if no capability matches, or embedding fails, the full
tool list is returned unchanged — this can only narrow, never break, an
existing turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import os

import numpy as np

from cognition import reason


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...


_CAPABILITY_INSTRUCT = "Which capability/tool domain applies to this task?"
_CAPABILITY_THRESHOLD = 0.35


@dataclass(frozen=True)
class Capability:
    id: str
    triggers: tuple[str, ...]        # full example phrases for semantic/keyword match
    tool_domains: tuple[str, ...] = ()


# Tool name -> domain. Any tool NOT listed here is treated as "core" and is
# always available (see ALWAYS_ON_TOOLS) — this keeps cross-cutting tools
# like make_plan/save_note from ever being accidentally filtered out.
TOOL_DOMAINS: dict[str, str] = {
    "deep_search": "research",
    "deep_research": "research",
    "schedule_job": "scheduling",
    "list_schedule": "scheduling",
    "cancel_schedule": "scheduling",
    "schedule_reminder": "scheduling",
    "list_reminders": "scheduling",
    "cancel_reminder": "scheduling",
    "list_skillsets": "skills",
    "search_skillsets": "skills",
    "load_skillset": "skills",
    "scan_photo_workspace": "photo",
    "propose_photo_ingestion": "photo",
    "write_photo_ingestion_report": "photo",
    "repo_file_tree": "repo",
    "repo_read_file": "repo",
    "repo_search_text": "repo",
    "read_paper_url": "research",
    "write_report": "reports",
    "learn_knowledge": "kb",
    "search_jobs": "jobs",
    "list_playbooks": "graph",
    "run_playbook": "graph",
    # Graph-level synthesis/KB tools (new)
    "kb_search": "kb",
    "synthesize_report": "reports",
    "polish_text": "reports",
    "combine_evidence": "reports",
    "condense_text": "reports",
    "learn_report": "kb",
    # Social posting tools — deliberately capability-gated, never ALWAYS_ON.
    # Posting is the highest-stakes tool class in the loop, so it gets the
    # same (not looser) gating as research/scheduling/photo/repo.
    #
    # Lane A (weekly postcard) is intentionally NOT exposed here — it is
    # non-agentic by design (see agentic/toolkit/social.py docstring): the scheduler
    # drives it directly via run_scheduled_weekly_social() on a Sun-Sat
    # cadence. Posting still requires draft.json["human_approved"] = true
    # regardless of path (scheduler or agent) — see _require_approved in
    # agentic/toolkit/social.py — but there is no conversational "draft/post the
    # weekly postcard" action for the agent loop to take, so it's not
    # registered as a tool. Only the inbox-driven photo/video lanes are
    # agent-callable.
    "draft_photo_social": "social",
    "post_photo_social": "social",
    "draft_video_social": "social",
    "post_video_social": "social",
}

# Always sent regardless of which capability matched — the base loop tools
# every agentic turn can plausibly need.
ALWAYS_ON_TOOLS: frozenset[str] = frozenset({
    "make_plan", "create_checklist", "save_note", "read_workspace_file",
    "summarize_task_state", "list_playbooks", "run_playbook", "final_answer",
})

# Tool domains per capability (maps capability id -> tuple of domains)
# Kept in Python since these are code-level mappings to TOOL_DOMAINS
_CAPABILITY_TOOL_DOMAINS: dict[str, tuple[str, ...]] = {
    "research": ("research", "kb", "reports"),
    "scheduling": ("scheduling",),
    "kb_proposal": ("kb", "skills"),
    "photo": ("photo",),
    "repo": ("repo", "skills", "reports"),
    "job_hunt": ("jobs",),
    "social": ("social",),
}


def _load_capability_triggers() -> dict[str, tuple[str, ...]]:
    """Load capability triggers from JSON file (full example phrases for embedding)."""
    path = Path(__file__).resolve().parent / "router" / "capability_prompts.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data["capabilities"].items()}


_TRIGGERS = _load_capability_triggers()

CAPABILITIES: dict[str, Capability] = {
    cap_id: Capability(
        id=cap_id,
        triggers=_TRIGGERS[cap_id],
        tool_domains=_CAPABILITY_TOOL_DOMAINS[cap_id],
    )
    for cap_id in _TRIGGERS
}

_trigger_embed_cache: dict[str, np.ndarray] = {}
_TRIGGER_EMBED_CACHE_MAX = 256

# On-disk tier, mirroring cognition.think._semantic_example_vectors' route
# vector cache. Same env flag so both caches turn on/off together; give it
# its own var (CAP_VECTOR_CACHE_ENABLED) instead if you want to toggle them
# independently.
_CAP_VECTOR_CACHE_DIR = Path(
    os.environ.get("CAP_VECTOR_CACHE_DIR", "cache/capability_vectors")
)


def _capability_vector_cache_path(cap: Capability, embedder: Embedder) -> Path | None:
    """Disk path for cap's trigger vector, or None if disk caching is off.

    Keyed on trigger text + embedder identity so an edit to a capability's
    triggers, or a swap of the embedding model, invalidates just that file
    instead of silently reusing a stale vector.

    NOTE: assumes `embedder` exposes stable identity attributes (e.g.
    model_name/dim). Check cognition.think's _route_vector_cache_path for
    whatever fields it actually keys on and mirror those exactly here —
    I don't have that function's body, so this is a best-effort match.
    """
    if os.environ.get("ROUTE_VECTOR_CACHE_ENABLED") != "1":
        return None
    embedder_fingerprint = getattr(embedder, "model_name", "") + ":" + str(getattr(embedder, "dim", ""))
    key_material = "|".join(cap.triggers) + f"::{embedder_fingerprint}"
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    return _CAP_VECTOR_CACHE_DIR / f"{cap.id}_{key_hash}.npz"


def _get_trigger_embedding(cap: Capability, embedder: Embedder) -> np.ndarray:
    cached = _trigger_embed_cache.get(cap.id)
    if cached is not None:
        return cached

    disk_path = _capability_vector_cache_path(cap, embedder)
    if disk_path is not None and disk_path.exists():
        try:
            with disk_path.open("rb") as f:
                data = np.load(f, allow_pickle=False)
                vec = data["vector"]
            if len(_trigger_embed_cache) >= _TRIGGER_EMBED_CACHE_MAX:
                _trigger_embed_cache.pop(next(iter(_trigger_embed_cache)))
            _trigger_embed_cache[cap.id] = vec
            return vec
        except Exception:
            pass  # fall through to recompute; corrupt/stale cache is not fatal

    text = " | ".join(cap.triggers)
    vec = reason.normalize_vec(np.asarray(embedder.embed_query(text), dtype=np.float32))

    if disk_path is not None:
        try:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            with disk_path.open("wb") as f:
                np.savez(f, vector=vec)
        except Exception:
            pass  # cache write failure shouldn't break the turn

    if len(_trigger_embed_cache) >= _TRIGGER_EMBED_CACHE_MAX:
        _trigger_embed_cache.pop(next(iter(_trigger_embed_cache)))
    _trigger_embed_cache[cap.id] = vec
    return vec


def match_capabilities(
    user_input: str, embedder: Embedder | None = None, threshold: float = _CAPABILITY_THRESHOLD,
    query_vector: np.ndarray | None = None,
) -> list[str]:
    """Return matched capability ids for this turn. Falls back to substring
    match against trigger phrases if no embedder is available or embedding
    fails — never raises.

    query_vector — pre-computed _CAPABILITY_INSTRUCT embedding; skips the
    redundant embedding HTTP call when provided.
    """
    if embedder is not None:
        try:
            if query_vector is not None:
                query_vec = reason.normalize_vec(np.asarray(query_vector, dtype=np.float32))
            else:
                query_vec = np.asarray(embedder.embed_query(user_input, instruct=_CAPABILITY_INSTRUCT), dtype=np.float32)
                query_vec = reason.normalize_vec(query_vec)
            matched = []
            for cap in CAPABILITIES.values():
                trig_vec = _get_trigger_embedding(cap, embedder)
                score = float(np.dot(query_vec, trig_vec))
                if score >= threshold:
                    matched.append(cap.id)
            return matched
        except Exception:
            pass

    folded = user_input.casefold()
    return [cap.id for cap in CAPABILITIES.values() if any(t in folded for t in cap.triggers)]


def filtered_tool_schemas(all_schemas: list[dict], cap_ids: list[str]) -> list[dict]:
    """Narrow the full tool schema list to ALWAYS_ON_TOOLS plus whatever
    domains the matched capabilities pull in. No match -> return everything
    unchanged, so this can only reduce tool-list size, never regress a turn
    that the old keyword/semantic matching would have handled fine."""
    if not cap_ids:
        return all_schemas
    # Filter out unknown capability IDs to avoid KeyError
    valid_cap_ids = [cid for cid in cap_ids if cid in CAPABILITIES]
    domains = {d for cid in valid_cap_ids for d in CAPABILITIES[cid].tool_domains}
    keep = set(ALWAYS_ON_TOOLS)
    for schema in all_schemas:
        name = schema["function"]["name"]
        if TOOL_DOMAINS.get(name) in domains:
            keep.add(name)
    filtered = [s for s in all_schemas if s["function"]["name"] in keep]
    return filtered or all_schemas
