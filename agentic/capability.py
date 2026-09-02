"""
agentic/capability.py

Capability routing for Aiko's agentic tool loop.

A capability holds no content of its own — it's a lookup that says, for a
given turn, which tool-schema domains should reach the LLM, plus the turn
policy (system overlay, max_iter, research_budget) that used to live in a
second, hand-maintained table in agentic.py. Capability is now the single
source of truth for all of that; agentic.py just calls resolve_handoff()
and applies the result. Prose retrieval (wiki/skill excerpts) is untouched
and still goes through wiki_context_for / skill_context_for exactly as
before. This module only narrows the `tools=` list passed to
chat.completions.create(), which used to be the full fixed _TOOL_SCHEMAS
set on every single turn regardless of what the turn needs.

Safe-by-default: if no capability matches, or embedding fails, the full
tool list is returned unchanged — this can only narrow, never break, an
existing turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import hashlib
import json
import os
import re

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
    system_overlay: str = ""
    max_iter: int | None = None          # None -> use global MAX_AGENT_ITER
    research_budget: int | None = None   # None -> use global AGENT_RESEARCH_MAX_CALLS


@dataclass(frozen=True)
class HandoffProfile:
    """Resolved turn policy for one or more matched capabilities."""

    tool_domains: frozenset[str]
    system_overlay: str
    max_iter: int
    research_budget: int
    capability_ids: tuple[str, ...]


# Tool domains and turn policy are derived from the central registry / this
# module. Kept in Python since these are code-level capability -> domain
# mappings, not user-editable data.
from agentic.registry import registry
from system.log import get_logger

log = get_logger(__name__)

_CAPABILITY_TOOL_DOMAINS: dict[str, tuple[str, ...]] = {
    "research": ("research", "kb", "reports"),
    "scheduling": ("scheduling",),
    "kb_proposal": ("kb", "skills"),
    "photo": ("photo", "social"),
    "repo": ("repo", "skills", "reports"),
    "job_hunt": ("jobs", "social"),
    "social": ("social",),
}

# Per-capability system overlay injected into the agentic system prompt
# when that capability is matched. Empty string -> no overlay contribution.
_CAPABILITY_SYSTEM_OVERLAYS: dict[str, str] = {
    "research": "Research thoroughly, cite evidence, then synthesize.",
    "scheduling": "Prefer schedule/reminder tools and confirm persisted ids.",
    "kb_proposal": "Write durable knowledge through learn_knowledge or proposal artifacts; do not edit trusted wiki/skills directly.",
    "photo": "Use photo workspace tools first; posting requires approval.",
    "repo": "Inspect repository files before proposing code or architecture changes.",
    "job_hunt": "Prefer job-hunt tools; social posting still requires approval.",
    "social": "Draft/post social content only under explicit request and approval gates.",
}

# Per-capability turn caps. Omitted / None -> fall back to the global
# MAX_AGENT_ITER / AGENT_RESEARCH_MAX_CALLS default passed into
# resolve_handoff() by the caller.
_CAPABILITY_MAX_ITER: dict[str, int] = {
    "scheduling": 4,
    "social": 5,
    "job_hunt": 6,
    "photo": 6,
    "kb_proposal": 5,
}

_CAPABILITY_RESEARCH_BUDGET: dict[str, int] = {
    "scheduling": 0,
    "social": 0,
    "repo": 0,
    "job_hunt": 1,
    "photo": 0,
    "kb_proposal": 1,
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
        system_overlay=_CAPABILITY_SYSTEM_OVERLAYS.get(cap_id, ""),
        max_iter=_CAPABILITY_MAX_ITER.get(cap_id),
        research_budget=_CAPABILITY_RESEARCH_BUDGET.get(cap_id),
    )
    for cap_id in _TRIGGERS
}


def resolve_handoff(
    cap_ids: list[str],
    *,
    default_max_iter: int,
    default_research_budget: int,
) -> HandoffProfile:
    """Resolve matched capability ids into one turn policy.

    Unknown ids are dropped silently (mirrors the old filtered_tool_schemas
    behavior). Multiple matched capabilities combine as: union of tool
    domains, newline-joined overlays (in CAPABILITIES iteration order via
    cap_ids order), and the MIN of any per-capability max_iter/research_budget
    override versus the running default — i.e. the most restrictive
    capability wins, never the most permissive.
    """
    valid = [cid for cid in cap_ids if cid in CAPABILITIES]
    domains: set[str] = set()
    overlays: list[str] = []
    max_iter = default_max_iter
    research_budget = default_research_budget
    for cid in valid:
        cap = CAPABILITIES[cid]
        domains.update(cap.tool_domains)
        if cap.system_overlay:
            overlays.append(cap.system_overlay)
        if cap.max_iter is not None:
            max_iter = min(max_iter, cap.max_iter)
        if cap.research_budget is not None:
            research_budget = min(research_budget, cap.research_budget)
    return HandoffProfile(
        tool_domains=frozenset(domains),
        system_overlay="\n".join(overlays),
        max_iter=max_iter,
        research_budget=research_budget,
        capability_ids=tuple(valid),
    )


# High-signal keyword fallback for environments where the embedder is
# unavailable (tests, degraded boots, or cold startup failures). The old
# fallback only matched full exemplar phrases, which often missed obvious
# asks like "remind me at 9" or "inspect this repo" and then sent every
# tool schema to ReAct. These terms keep common turns narrowed without
# blocking ambiguous turns: if nothing matches, filtered_tool_schemas() still
# receives an empty list and returns all tools.
_CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "research": ("research", "search", "look up", "latest", "current", "source", "cite", "benchmark", "investigate"),
    "scheduling": ("schedule", "remind", "reminder", "alarm", "wake me", "ping me", "recurring", "daily", "weekly"),
    "kb_proposal": ("remember this", "store this", "learn this", "knowledge base", "add to wiki", "save this insight"),
    "photo": ("photo", "photos", "image", "images", "camera", "ingest", "inbox"),
    "repo": ("repo", "repository", "source file", "codebase", "refactor", "debug", "patch", "implement", "unit test"),
    "job_hunt": ("job", "jobs", "job boards", "role", "remote", "hiring"),
    "social": ("youtube", "threads", "bluesky", "mastodon", "pixelfed", "post", "publish", "social",
               "email", "mail", "protonmail", "inbox", "send a message", "send an email", "send an email to", "send a message to"),
}


_AMBIGUOUS_SINGLE_WORD_KEYWORDS: frozenset[str] = frozenset({
    # These words frequently appear in unrelated requests (for example
    # "daily news" or "postmortem notes"). Let stronger phrase or companion
    # keyword signals route those turns instead of narrowing the tool list on
    # one broad token.
    "daily", "weekly", "post", "source", "remote",
})


def _term_matches(text: str, term: str) -> bool:
    """Return True when a keyword term matches on token/phrase boundaries."""
    folded_term = term.casefold().strip()
    if not folded_term:
        return False
    if " " not in folded_term and folded_term in _AMBIGUOUS_SINGLE_WORD_KEYWORDS:
        return False
    parts = re.findall(r"[\w']+", folded_term)
    if not parts:
        return False
    pattern = r"(?<![\w'])" + r"\W+".join(re.escape(part) for part in parts) + r"(?![\w'])"
    return re.search(pattern, text) is not None


def _capability_text_matches(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_matches(text, term) for term in terms)

_trigger_embed_cache: dict[str, np.ndarray] = {}
_TRIGGER_EMBED_CACHE_MAX = 256

# On-disk tier, mirroring cognition.think._semantic_example_vectors' route
# vector cache. Same env flag so both caches turn on/off together; give it
# its own var (CAP_VECTOR_CACHE_ENABLED) instead if you want to toggle them
# independently.
_CAP_VECTOR_CACHE_DIR = Path(
    os.environ.get("CAP_VECTOR_CACHE_DIR", "cache/capability_vectors")
)

_CAP_VECTOR_CACHE_DIR_DEFAULT = "capability_vectors"


def _capability_vector_cache_path(cap: Capability, embedder: Embedder) -> Path | None:
    payload = {
        "triggers": list(cap.triggers),
        "cap_id": cap.id,
        "embedder": {
            "class": type(embedder).__name__,
            "model": getattr(embedder, "model", None) or getattr(embedder, "model_name", None) or getattr(embedder, "name", None),
            "dims": os.getenv("EMBED_DIMS", ""),
        },
    }
    return reason.cache_vector_path(
        payload,
        cache_dir_env="CAP_VECTOR_CACHE_DIR",
        default_dir=_CAP_VECTOR_CACHE_DIR_DEFAULT,
        per_user=True,
    )


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
            log.warning("capability: failed to load cached trigger embedding")

    text = " | ".join(cap.triggers)
    vec = reason.normalize_vec(np.asarray(embedder.embed_query(text), dtype=np.float32))

    if disk_path is not None:
        try:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            with disk_path.open("wb") as f:
                np.savez(f, vector=vec)
        except Exception:
            log.warning("capability: failed to write trigger embedding cache")

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
            log.warning("capability: embedding matching failed, falling back to phrase matching")
    folded = user_input.casefold()
    phrase_matches = [cap.id for cap in CAPABILITIES.values() if _capability_text_matches(folded, cap.triggers)]
    if phrase_matches:
        return phrase_matches
    return [cap_id for cap_id, terms in _CAPABILITY_KEYWORDS.items() if _capability_text_matches(folded, terms)]


def filtered_tool_schemas(all_schemas: list[dict], cap_ids: list[str]) -> list[dict]:
    """Narrow the full tool schema list to always-on tools plus whatever
    domains the matched capabilities pull in. No match -> return everything
    unchanged, so this can only reduce tool-list size, never regress a turn
    that the old keyword/semantic matching would have handled fine.

    This is the ONLY domain filter pass — callers should not apply a second,
    separate domain filter on top of this one (see resolve_handoff's
    tool_domains, which is derived from the same CAPABILITIES table and is
    provided for callers that need the resolved domain set without
    re-deriving it, not for a second filtering pass).
    """
    if not cap_ids:
        return all_schemas
    # Filter out unknown capability IDs to avoid KeyError
    valid_cap_ids = [cid for cid in cap_ids if cid in CAPABILITIES]
    domains = {d for cid in valid_cap_ids for d in CAPABILITIES[cid].tool_domains}

    effective_domains = registry.get_tool_domains()
    effective_always_on = registry.get_always_on_tools()

    keep = set(effective_always_on)
    for schema in all_schemas:
        name = schema["function"]["name"]
        if effective_domains.get(name) in domains:
            keep.add(name)
    filtered = [s for s in all_schemas if s["function"]["name"] in keep]
    return filtered or all_schemas
