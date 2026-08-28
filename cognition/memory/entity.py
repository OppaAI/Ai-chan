"""Entity tagging, importance (I_e), and supersession-chain helpers.

I_e = (1-α)·centrality + α·recency. No LLM. Used by monthly consolidation and
mild recall boost. Also hosts rule-based entity extraction / kind / write-op
classification, valence/arousal/salience inference, and the entity_relations
co-mention graph (formerly memory/entities.py + backend blocks).
"""
from __future__ import annotations

import functools
import math
import os
import re
from datetime import datetime, timezone

import json
import sqlite3
from itertools import combinations
from typing import Any, Iterable

from .schema import KIND_FACT, _WS_RE, ensure_phase_a_schema, existing_columns
from .env import env_flag


from system.log import get_logger

log = get_logger(__name__)


ENTITY_IMPORTANCE_ALPHA = float(os.getenv("ENTITY_IMPORTANCE_ALPHA", "0.4"))
ENTITY_IMPORTANCE_BETA = float(os.getenv("ENTITY_IMPORTANCE_BETA", "0.05"))
MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT = float(os.getenv("MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT", "0.008"))
# TTL for the per-user entity-importance map cache. compute_entity_importance_map()
# full-scans all memories + entity_relations, so recall caches it briefly rather
# than recomputing on every cache-miss search. Writes invalidate it immediately.
ENTITY_IMPORTANCE_CACHE_TTL = float(os.getenv("ENTITY_IMPORTANCE_CACHE_TTL", "60.0"))
MEMORY_SUPERSESSION_CHAIN_EXPAND = os.getenv("MEMORY_SUPERSESSION_CHAIN_EXPAND", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SUPERSESSION_CHAIN_KINDS = {
    k.strip().lower()
    for k in os.getenv("MEMORY_SUPERSESSION_CHAIN_KINDS", "identity,preference,plan").split(",")
    if k.strip()
}
_REFLECTIVE_RE = re.compile(
    r"\b(used to|changed|before|previously|history|what changed|"
    r"remember when|used to be|no longer|switched from)\b",
    re.IGNORECASE,
)

MEMORY_SPREADING_ENABLED = os.getenv("MEMORY_SPREADING_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SPREADING_MAX_DEPTH = max(1, int(os.getenv("MEMORY_SPREADING_MAX_DEPTH", "2")))
MEMORY_SPREADING_DECAY = float(os.getenv("MEMORY_SPREADING_DECAY", "0.6"))
MEMORY_SPREADING_MIN_STRENGTH = float(os.getenv("MEMORY_SPREADING_MIN_STRENGTH", "0.15"))

# ── Phase 21: neural network overlay ──────────────────────────────────────
# Optional neural-net-style activation propagation over the entity co-mention
# graph.  Nodes have valence/arousal features; edges have directed weights
# that grow via Hebbian learning ("cells that fire together wire together").
NEURAL_NET_ENABLED = os.getenv("NEURAL_NET_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
NEURAL_DECAY = float(os.getenv("NEURAL_DECAY", "0.95"))      # edge weight decay per timestep
NEURAL_THRESHOLD = float(os.getenv("NEURAL_THRESHOLD", "0.5"))  # activation spill threshold
NEURAL_LEAK = float(os.getenv("NEURAL_LEAK", "0.1"))         # node activation leak per timestep
NEURAL_SPREAD_DEPTH = max(1, int(os.getenv("NEURAL_SPREAD_DEPTH", "3")))  # BFS depth
# Repairs common user/assistant name swaps in extracted facts.  No LLM /
# external services required; pure regex / template post-filter.

DEFAULT_ASSISTANT_NAME = "Aiko"
DEFAULT_USER_NAME = "Oppa"


def _user_re(name: str) -> str:
    return re.escape(name.strip())


def fix_fact_identity(
    text: str,
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> str:
    """
    Apply cheap repairs for common Oppa/Aiko swaps.
    Does not invent new facts; returns text unchanged if no rule matches.
    """
    t = (text or "").strip()
    if not t:
        return t

    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    a = (assistant_name or DEFAULT_ASSISTANT_NAME).strip() or DEFAULT_ASSISTANT_NAME
    if u.casefold() == a.casefold():
        return t

    ue = _user_re(u)

    # 1) "{User} dislikes/hates being human-like" → Aiko (persona self-talk)
    t2 = re.sub(
        rf"^({ue})\s+(dislikes|doesn't like|does not like|hates|dislike)\s+"
        rf"(to be |being )?(portrayed as )?human[- ]?like\b",
        lambda m: f"{a} {m.group(2)} being human-like",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 2) "{User} needs/must/should follow {User}'s rules" → Aiko follows user's rules
    t2 = re.sub(
        rf"^{ue}\s+(needs to|must|should|has to)\s+follow\s+{ue}'s\s+rules\b",
        lambda m: f"{a} should follow {u}'s rules",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 3) "{User} must/should obey {User}'s instructions" (same idea)
    t2 = re.sub(
        rf"^{ue}\s+(needs to|must|should|has to)\s+(obey|follow)\s+{ue}'s\s+"
        rf"(instructions|commands|orders)\b",
        lambda m: f"{a} should follow {u}'s {m.group(3)}",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 4) "{User} is an AI / assistant / language model" → Aiko
    t2 = re.sub(
        rf"^{ue}\s+(is|as)\s+(an?\s+)?(AI|assistant|language model|LLM)\b",
        lambda m: f"{a} is {m.group(2) or ''}{m.group(3)}",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    return t.strip()


def should_skip_misattributed_fact(
    text: str,
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> bool:
    """
    True if fact still looks like assistant-persona pinned on the user
    after fix_fact_identity — safer to skip than store wrong.
    """
    t = (text or "").strip()
    if not t:
        return True
    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    ue = _user_re(u)
    # User-name subject + strong assistant-only cues
    if re.match(
        rf"^{ue}.*\b(human[- ]?like|as an AI|language model|my persona)\b",
        t,
        re.IGNORECASE,
    ):
        return True
    return False


def sanitize_extracted_facts(
    facts: Iterable[str],
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> list[str]:
    """Fix identity then drop remaining misattributed persona facts."""
    out: list[str] = []
    u = user_name or DEFAULT_USER_NAME
    a = assistant_name or DEFAULT_ASSISTANT_NAME
    for raw in facts:
        if not isinstance(raw, str):
            continue
        fixed = fix_fact_identity(raw, user_name=u, assistant_name=a)
        if should_skip_misattributed_fact(fixed, user_name=u, assistant_name=a):
            continue
        if fixed:
            out.append(fixed)
    return out


def sanitize_fact_score_pairs(
    pairs: Iterable[tuple[str, int | None]],
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> list[tuple[str, int | None]]:
    """Same as sanitize_extracted_facts for (text, valence_score) pairs."""
    out: list[tuple[str, int | None]] = []
    u = user_name or DEFAULT_USER_NAME
    a = assistant_name or DEFAULT_ASSISTANT_NAME
    for item in pairs:
        if not item:
            continue
        raw, score = item[0], item[1] if len(item) > 1 else None
        if not isinstance(raw, str):
            continue
        fixed = fix_fact_identity(raw, user_name=u, assistant_name=a)
        if should_skip_misattributed_fact(fixed, user_name=u, assistant_name=a):
            continue
        if fixed:
            out.append((fixed, score))
    return out


# --- Prompt fragment (paste into extract system prompt) ---

IDENTITY_PROMPT_RULES = """
IDENTITY (critical — never get this wrong):
- The USER is named {user_name}. The ASSISTANT is {assistant_name}.
- "I/me/my" in a USER message → fact subject is {user_name}.
- "I/me/my" in an ASSISTANT message → fact subject is {assistant_name}.
- Do NOT write {assistant_name}'s preferences, personality, or self-rules as {user_name}'s.
- Do NOT write {user_name}'s preferences as {assistant_name}'s.
- Assistant talking about being human-like, persona, or "my rules" → subject is {assistant_name}.
- User commands/rules for the assistant → "{assistant_name} should follow {user_name}'s rules"
  NOT "{user_name} needs to follow {user_name}'s rules".

Examples:
Good: "{assistant_name} dislikes being portrayed as human-like."
Bad:  "{user_name} dislikes being portrayed as human-like." (when the assistant said it about herself)
Good: "{assistant_name} should follow {user_name}'s rules."
Bad:  "{user_name} needs to follow {user_name}'s rules."
Good: "{user_name} prefers dark mode."
Bad:  "{assistant_name} prefers dark mode." (when the user said it)
""".strip()


def format_identity_prompt_rules(
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> str:
    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    a = (assistant_name or DEFAULT_ASSISTANT_NAME).strip() or DEFAULT_ASSISTANT_NAME
    return IDENTITY_PROMPT_RULES.format(user_name=u, assistant_name=a)


def spread_activation(
    seed_entities: list[str],
    edges: list[tuple[str, str, float]],
    *,
    max_depth: int | None = None,
    decay: float | None = None,
    min_strength: float | None = None,
) -> dict[str, float]:
    """BFS-style activation over undirected co-mention edges.

    seed_entities: casefolded entity strings from entry-hit memories / query
    edges: list of (entity_a, entity_b, weight) already casefolded
    returns: entity -> activation strength in [0, 1+]
    """
    if not MEMORY_SPREADING_ENABLED or not seed_entities:
        return {}
    max_depth = max_depth if max_depth is not None else MEMORY_SPREADING_MAX_DEPTH
    for _ in range(max_depth):
        nxt: set[str] = set()
        for node in frontier:
            s0 = strength.get(node, 0.0)
            if s0 < min_strength:
                continue
            for nb, w in adj.get(node, []):
                # normalize soft weight: treat weight as relative, clamp
                hop = s0 * decay * min(1.0, max(0.0, w))
                if hop < min_strength:
                    continue
                if hop > strength.get(nb, 0.0):
                    strength[nb] = hop
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return {e: s for e, s in strength.items() if s >= min_strength}


def memory_max_activation(row, activation: dict[str, float]) -> float:
    if not activation:
        return 0.0
    ents = entities_from_json_safe(
        row["entities"] if hasattr(row, "keys") and "entities" in row.keys() else row.get("entities")
    )
    if not ents:
        return 0.0
    return max((activation.get(e.casefold(), 0.0) for e in ents), default=0.0)
def entities_from_json_safe(raw) -> list[str]:
    try:
        from cognition.memory.memorize import entities_from_json
        return entities_from_json(raw)
    except Exception:
        return []


def compute_entity_importance_map(memorize_or_backend, user_id: str) -> dict[str, float]:
    """I_e = (1-α)·centrality + α·recency per entity (casefolded)."""
    try:
        conn = getattr(memorize_or_backend, "_conn", None)
        if conn is None:
            mem = getattr(memorize_or_backend, "_mem", None)
            conn = getattr(mem, "_conn", None) if mem is not None else None
        if conn is None:
            return {}
        lock = getattr(getattr(memorize_or_backend, "_mem", memorize_or_backend), "_db_lock", None)

        def _read():
            try:
                rows = conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            except Exception:
                return {}, {}
            degree: dict[str, float] = {}
            for row in rows:
                a = str(row["entity_a"] or "").casefold()
                b = str(row["entity_b"] or "").casefold()
                w = float(row["weight"] or 0.0)
                if a:
                    degree[a] = degree.get(a, 0.0) + w
                if b:
                    degree[b] = degree.get(b, 0.0) + w
            last_touch: dict[str, str] = {}
            try:
                mem_rows = conn.execute(
                    """
                    SELECT entities, last_accessed_at, created_at FROM memories
                    WHERE user_id = ? AND (status = 'active' OR status IS NULL)
                    """,
                    (user_id,),
                ).fetchall()
            except Exception:
                mem_rows = []
            for mr in mem_rows:
                ents = entities_from_json_safe(mr["entities"] if "entities" in mr.keys() else "[]")
                ts = mr["last_accessed_at"] or mr["created_at"] or ""
                for e in ents:
                    key = e.casefold()
                    prev = last_touch.get(key, "")
                    if ts and (not prev or str(ts) > prev):
                        last_touch[key] = str(ts)
            return degree, last_touch

        if lock is not None:
            with lock:
                degree, last_touch = _read()
        else:
            degree, last_touch = _read()

        if not degree and not last_touch:
            return {}
        max_deg = max(degree.values(), default=1.0) or 1.0
        alpha = max(0.0, min(1.0, ENTITY_IMPORTANCE_ALPHA))
        beta = max(0.0, ENTITY_IMPORTANCE_BETA)
        now = datetime.now(timezone.utc)
        out: dict[str, float] = {}
        for e in set(degree) | set(last_touch):
            c = (degree.get(e, 0.0) / max_deg) if max_deg else 0.0
            r = 0.0
            ts = last_touch.get(e) or ""
            if ts and ts != "never":
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days = max(0.0, (now - dt).total_seconds() / 86400.0)
                    r = float(math.exp(-beta * days))
                except Exception:
                    r = 0.0
            out[e] = (1.0 - alpha) * c + alpha * r
        return out
    except Exception as exc:
        log.debug("compute_entity_importance_map failed: %s", exc)
        return {}


def memory_max_entity_importance(row, importance_map: dict[str, float]) -> float:
    if not importance_map:
        return 0.0
    try:
        raw = row["entities"] if hasattr(row, "keys") and "entities" in row.keys() else row.get("entities")
        ents = entities_from_json_safe(raw)
    except Exception:
        ents = []
    if not ents:
        return 0.0
    return max((importance_map.get(e.casefold(), 0.0) for e in ents), default=0.0)


def should_expand_supersession_chain(query: str, row) -> bool:
    if not MEMORY_SUPERSESSION_CHAIN_EXPAND:
        return False
    if _REFLECTIVE_RE.search(query or ""):
        return True
    try:
        kind = (row["kind"] if hasattr(row, "keys") else row.get("kind")) or ""
    except Exception:
        kind = ""
    return str(kind).lower() in MEMORY_SUPERSESSION_CHAIN_KINDS


def walk_supersession_chain(conn, mem_id: str, user_id: str, max_depth: int = 12) -> list[dict]:
    try:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND user_id = ?",
            (mem_id, user_id),
        ).fetchone()
        if row is None:
            return []
        chain_ids: list[str] = [mem_id]
        seen = {mem_id}
        cur = row
        for _ in range(max_depth):
            sid = cur["supersedes_id"] if "supersedes_id" in cur.keys() else None
            if not sid or sid in seen:
                break
            prev = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (sid, user_id),
            ).fetchone()
            if prev is None:
                break
            chain_ids.append(sid)
            seen.add(sid)
            cur = prev
        chain_ids.reverse()
        tip = chain_ids[-1]
        for _ in range(max_depth):
            nxt = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND supersedes_id = ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (user_id, tip),
            ).fetchone()
            if nxt is None:
                break
            nid = str(nxt["id"])
            if nid in seen:
                break
            chain_ids.append(nid)
            seen.add(nid)
            tip = nid
        out: list[dict] = []
        for mid in chain_ids:
            r = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
            if r is not None:
                out.append(dict(r))
        return out
    except Exception as exc:
        log.debug("supersession chain walk failed for %s: %s", mem_id, exc)
        return []

# ═══════════════════════════════════════════════════════════════════════════════
#  Entity extraction / valence / arousal / write-op classification  (from backend)
# ═══════════════════════════════════════════════════════════════════════════════


# Multi-word Proper Case spans: "Hugging Face", "San Francisco"
_PROPER_SPAN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b"
)
# Short ALLCAPS tokens often used as project codes: GRACE, ROS2, API
_ALLCAPS_RE = re.compile(r"\b([A-Z]{2,12}[0-9]*)\b")
# Quoted names: "Max", 'Aiko'
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,40})[\"']")
# Explicit name patterns: called X, named X, project X
_CALLED_RE = re.compile(
    r"\b(?:called|named|project|robot|dog|cat|company|team)\s+([A-Z][\w.-]{1,40})",
    re.IGNORECASE,
)

# ── domain-aware entity terms ──────────────────────────────────────────────
# Catches lowercase/casual mentions of your own stack that the generic
# Proper-Case / ALLCAPS regexes above would otherwise miss (e.g. "aiko",
# "ros2 pipeline", "the jetson"). Configurable via ENTITY_DOMAIN_TERMS so new
# projects/hardware can be added without touching code. Matched terms are
# stored using the ORIGINAL casing found in text (canonicalization, if
# desired, happens via ENTITY_ALIAS_JSON below).
_DOMAIN_ENTITY_TERMS: tuple[str, ...] = tuple(
    t.strip().lower()
    for t in os.getenv(
        "ENTITY_DOMAIN_TERMS",
        "aiko-chan,grace,aurora,eric,jetson,jetson orin,orin nano,"
        "ros2,ros,nav2,slam,lidar,oak-d,vrm,mcp,harrier,miotts,sensevoice,"
        "sherpa-onnx,eres2net,silero,qdrant,mem0,sqlite-vec,"
        "cnc,wmc,emc,mcc,msb,gce,tms,tic,toc,hrs,hrm,agi,"
        "oppaai,threads,patreon,huggingface,hugging face,cosmos,"
        "core logic,brute-force,autocomplete,piano practice,programmer laziness,"
        "editor,modules,habit,rewrites,accidental rewrites,"
        "practiced piano,40 hours,single day,frustration,irritation,programmer inefficiency,"
        "meteor shower,jokes,eating,humanoid body,amusing,sarcastic,"
        "sat properly,window,hands off lap,cloudy skies,bright meteor showers,"
        "louis,dislikes,behavior,conversations,annoyed,"
        "eager,hear a joke,wants to go outside",
    ).split(",")
    if t.strip()
)
_DOMAIN_ENTITY_RE = (
    re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(_DOMAIN_ENTITY_TERMS, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    if _DOMAIN_ENTITY_TERMS
    else None
)

# ── entity alias resolution ────────────────────────────────────────────────
# Optional canonicalization so that name variants collapse to one graph node
# instead of splitting co-mention weight across lookalike entities (e.g. if
# you decide "AuRoRA" mentions should count toward the "Grace" node). Off by
# default (empty map) — nothing is merged unless you opt in via env var:
#   ENTITY_ALIAS_JSON='{"aurora": "Grace", "agi": "Grace"}'
try:
    _ENTITY_ALIASES_RAW: dict[str, str] = json.loads(os.getenv("ENTITY_ALIAS_JSON", "{}"))
except (TypeError, ValueError, json.JSONDecodeError):
    _ENTITY_ALIASES_RAW = {}
# Built-in owner aliases: raw github id and Threads handle both map to OppaAI
# so github_205369547 / oppa.ai.bot never become separate graph hubs.
_BUILTIN_ALIASES: dict[str, str] = {
    "github_205369547": "OppaAI",
    "oppa.ai.bot": "OppaAI",
    "oppa.ai": "OppaAI",
    "@oppa.ai.bot": "OppaAI",
}
# Env aliases override builtins
_ENTITY_ALIASES: dict[str, str] = {
    **{str(k).casefold(): str(v) for k, v in _BUILTIN_ALIASES.items()},
    **{str(k).casefold(): str(v) for k, v in _ENTITY_ALIASES_RAW.items() if str(k).strip() and str(v).strip()},
}


@functools.lru_cache(maxsize=1024)
def resolve_entity_alias(entity: str) -> str:
    """Canonicalize an entity string via the configurable alias map.

    No-op unless ENTITY_ALIAS_JSON is set. Applied once inside
    extract_entities() so every caller (write path, backfill, consolidation)
    benefits automatically without needing to know about aliasing.
    """
    if not entity:
        return entity
    return _ENTITY_ALIASES.get(entity.casefold(), entity)


_STOP_ENTITIES = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "into",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "yesterday", "tomorrow", "user", "assistant", "aiko",
    "he", "she", "they", "his", "her", "their", "this", "that",
    # user identity — appears in nearly every memory, creates super-node hub
    "oppa", "oppaai",
    # generic ROS2 / dev-noise nouns — too common to be useful graph nodes
    "node", "topic", "service", "message", "callback", "function", "class",
    "file", "line", "log", "logs", "commit", "branch", "repo", "script",
})

# Kind heuristics (keyword → kind). First match wins.
_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", ("name is", "birthday", "lives in", "nationality", "age is", "is from ")),
    ("preference", ("likes", "loves", "hates", "dislikes", "prefers", "favorite", "favourite")),
    ("plan", ("deadline", "due ", "will ", "going to", "plans to", "wants to")),
    ("event", ("hackathon", "interview", "meeting", "appointment", "lost ", "joined")),
)


def _clean_entity(raw: str) -> str | None:
    s = (raw or "").strip(" .,;:!?()[]{}").strip()
    if len(s) < 2 or len(s) > 80:
        return None
    if s.casefold() in _STOP_ENTITIES:
        return None
    # Drop pure numbers
    if s.isdigit():
        return None
    return s


@functools.lru_cache(maxsize=512)
def extract_entities(text: str, *, max_entities: int = 12) -> tuple[str, ...]:
    """Extract entity-like tokens from a memory fact string.

    Deterministic and cheap. Prefer precision over recall — empty is fine.

    Order of passes (first-seen wins for dedup, casefolded):
      1. Domain terms (your own stack — catches lowercase/casual mentions)
      2. Quoted spans
      3. "called/named/project X" spans
      4. Proper-Case spans (sentence-initial no longer excluded)
      5. ALLCAPS tokens (project codes, acronyms)
    """
    if not (text or "").strip():
        return ()

    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        ent = _clean_entity(raw)
        if not ent:
            return
        ent = resolve_entity_alias(ent)
        key = ent.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(ent)

    if _DOMAIN_ENTITY_RE is not None:
        for m in _DOMAIN_ENTITY_RE.finditer(text):
            _add(m.group(1))
    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in _CALLED_RE.finditer(text):
        _add(m.group(1))
    for m in _PROPER_SPAN_RE.finditer(text):
        span = m.group(1)
        _add(span)
    for m in _ALLCAPS_RE.finditer(text):
        _add(m.group(1))

    return tuple(found[:max_entities])


def classify_kind(text: str, default: str = "fact") -> str:
    """Heuristic memory kind from fact text. No LLM."""
    low = (text or "").casefold()
    for kind, needles in _KIND_RULES:
        if any(n in low for n in needles):
            return kind
    return default


_VALENCE_POS_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F970-\U0001F973\U0001F929\U0001F60A\U0001F60D\U0001F389]|"
    r"\b(?:happy|glad|love|great|awesome|excited|relief|yay|wonderful|proud)\b",
    re.IGNORECASE,
)
_VALENCE_NEG_RE = re.compile(
    r"[\U0001F622\U0001F62D\U0001F614\U0001F61E\U0001F620\U0001F621\U0001F624]|"
    r"\b(?:sad|angry|frustrated|afraid|scared|hate|awful|terrible|worried|anxious|cry|pain)\b",
    re.IGNORECASE,
)
# Shared salience policy (write-time tags + monthly legacy fallback).
SALIENCE_POLICY_RE = re.compile(
    r"\b(?:deadline|birthday|anniversary|appointment|hackathon|interview|lost|"
    r"passport|license|wallet|important|breakthrough|problem|always|never|"
    r"favorite|favourite|remember this|never forget)\b|!{2,}",
    re.IGNORECASE,
)

_VALENCE_STRONG_RE = re.compile(
    r"\b(?:very|so|extremely|furious|devastat|ecstatic|hate this|love this|terrified|overjoyed)\b|!{2,}",
    re.IGNORECASE,
)

def infer_valence_score(text: str) -> int:
    """Return −2…+2 from emoji + lexicon + intensifiers. No LLM / no Harrier."""
    t = text or ""
    neg = bool(_VALENCE_NEG_RE.search(t))
    pos = bool(_VALENCE_POS_RE.search(t))
    strong = bool(_VALENCE_STRONG_RE.search(t))
    if neg and not pos:
        return -2 if strong else -1
    if pos and not neg:
        return 2 if strong else 1
    if neg and pos:
        return -1
    return 0

# ── config ────────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: str = "1") -> bool:
    return env_flag(name, default)


def _arousal_enabled() -> bool:
    return _env_bool("MEMORY_AROUSAL_ENABLED", "1")


def _arousal_rank_weight() -> float:
    try:
        return float(os.getenv("MEMORY_AROUSAL_RANK_WEIGHT", "0.01"))
    except ValueError:
        return 0.01


def _neg_hard_filter_enabled() -> bool:
    return _env_bool("MEMORY_NEG_HARD_FILTER", "1")


def _neg_hard_threshold() -> int:
    try:
        return int(os.getenv("MEMORY_NEG_HARD_THRESHOLD", "-1"))
    except ValueError:
        return -1


# ── arousal inference (heuristic, no LLM) ─────────────────────────────────────

_AROUSAL_HIGH_RE = re.compile(
    r"\b(?:panic|panick(?:ed|ing)?|terrified|furious|ecstatic|urgent|emergency|"
    r"scream(?:ed|ing)?|shock(?:ed|ing)?|adrenaline|frantic|hyper|"
    r"can't sleep|cant sleep|all-nighter|meltdown|breakdown)\b|!{2,}",
    re.IGNORECASE,
)
_AROUSAL_MID_RE = re.compile(
    r"\b(?:excited|anxious|nervous|stressed|worried|angry|upset|thrilled|"
    r"deadline|interview|hackathon|fight|"
    r"argument|crying|tears)\b",
    re.IGNORECASE,
)
_AROUSAL_LOW_RE = re.compile(
    r"\b(?:calm|peaceful|quiet|tired|sleepy|bored|meh|whatever|"
    r"routine|ordinary|mundane)\b",
    re.IGNORECASE,
)


def infer_arousal_score(text: str) -> int:
    """Return −1…+2 activation intensity from lexicon (no LLM).

    Convention (parallel to valence magnitude, signed only for calm):
      +2 strong high arousal, +1 moderate high, 0 neutral/unknown,
      -1 low activation (calm/flat). -2 is reserved and not yet produced
      by this heuristic.
    Ranking uses abs(score); sign is for analytics/Studio.
    """
    t = text or ""
    if _AROUSAL_HIGH_RE.search(t):
        return 2
    if _AROUSAL_MID_RE.search(t):
        return 1
    if _AROUSAL_LOW_RE.search(t):
        return -1
    return 0


def arousal_rank_bonus(arousal_score: int | None) -> float:
    """Additive score term; 0 when disabled or missing."""
    if not _arousal_enabled() or arousal_score is None:
        return 0.0
    try:
        a = int(arousal_score)
    except (TypeError, ValueError):
        return 0.0
    return _arousal_rank_weight() * (min(abs(a), 2) / 2.0)


# ── neg hard filter ───────────────────────────────────────────────────────────

_EMOTION_QUERY_RE = re.compile(
    r"\b(?:feel(?:ing)?|emotion|upset|sad|angry|anxious|stress(?:ed)?|"
    r"embarrass(?:ed|ing)?|ashamed|guilt|hate|conflict|fight|problem with|"
    r"what's wrong|what is wrong|are you ok|worried about)\b",
    re.IGNORECASE,
)


def _is_sticky_neg(mem: dict[str, Any]) -> bool:
    thr = _neg_hard_threshold()
    vs = mem.get("valence_score")
    if vs is not None:
        try:
            return int(vs) <= thr
        except (TypeError, ValueError):
            pass
    tag = (mem.get("valence_tag") or "").strip().lower()
    return tag in ("neg", "negative") and thr >= -1


def _query_engages_memory(query: str, mem: dict[str, Any]) -> bool:
    q = (query or "").casefold()
    if not q:
        return False
    if _EMOTION_QUERY_RE.search(query or ""):
        return True
    text = (mem.get("memory") or mem.get("text") or "").casefold()
    if text:
        mem_tokens = set(re.findall(r"\w+", text))
        q_tokens = {t for t in re.findall(r"\w+", q) if len(t) >= 4}
        if mem_tokens & q_tokens:
            # light token overlap; prefer entity overlap when present
            return True
    ents = mem.get("entities") or []
    if isinstance(ents, str):
        try:
            ents = json.loads(ents)
        except (ValueError, TypeError):
            ents = []
    for e in ents:
        if e and str(e).casefold() in q:
            return True
    return False


def apply_neg_hard_filter(
    memories: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Drop sticky-neg rows unless the user query engages them.

    Unsolicited = no engagement signal. Explicit reference or emotion-seeking
    query keeps the memory.
    """
    if not _neg_hard_filter_enabled():
        return memories
    out: list[dict[str, Any]] = []
    for m in memories:
        if _is_sticky_neg(m) and not _query_engages_memory(query, m):
            continue
        out.append(m)
    return out

def tag_from_score(score: int) -> str:
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "neutral"
    if s <= -1:
        return "neg"
    if s >= 1:
        return "pos"
    return "neutral"


def infer_valence_tag(text: str) -> str:
    """Backward-compatible 3-way tag derived from 5-pt score."""
    return tag_from_score(infer_valence_score(text))


def infer_salience_hit(text: str) -> int:
    return 1 if SALIENCE_POLICY_RE.search(text or "") else 0

def entity_overlap_score(query: str, entities: Iterable[str]) -> float:
    """Return 0..1 fraction of entities mentioned in the query (casefold)."""
    ents = [e for e in entities if e]
    if not ents:
        return 0.0
    q = (query or "").casefold()
    hits = sum(1 for e in ents if e.casefold() in q)
    return hits / len(ents)


def normalize_memory_text(text: str) -> str:
    """Normalize for write-op classification (lowercase, collapse whitespace)."""
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def entities_to_json(entities: list[str] | None) -> str:
    """Serialize a deduped entity list into a JSON string column value."""
    if not entities:
        return "[]"
    cleaned: list[str] = []
    seen: set[str] = set()
    for e in entities:
        s = str(e).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s[:80])
        if len(cleaned) >= 16:
            break
    return json.dumps(cleaned, ensure_ascii=False)


def entities_from_json(raw: Any) -> list[str]:
    """Parse an entities column value (JSON string or raw list) back to list."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def classify_write_op(
    *,
    similarity: float | None,
    new_text: str,
    old_text: str | None,
    dedup_threshold: float,
) -> str:
    """Return 'noop' | 'supersede' | 'add' — rule-only, no LLM."""
    if similarity is None or similarity < dedup_threshold:
        return "add"
    if normalize_memory_text(new_text) == normalize_memory_text(old_text or ""):
        return "noop"
    return "supersede"

RELATION_CO_MENTION = "co_mentions"
RELATION_RELATED = "related_to"

_ENTITY_RELATIONS_DDL = """
CREATE TABLE IF NOT EXISTS entity_relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    entity_a    TEXT NOT NULL,
    entity_b    TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'co_mentions',
    weight      REAL NOT NULL DEFAULT 1.0,
    memory_id   TEXT,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_rel_user ON entity_relations(user_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_a ON entity_relations(user_id, entity_a);
CREATE INDEX IF NOT EXISTS idx_entity_rel_b ON entity_relations(user_id, entity_b);
-- Composite indexes for common query patterns:
-- 1. Full user scan (graph_pass, neural_activate): WHERE user_id = ?
-- 2. Neural weight update: WHERE user_id = ? AND (entity_a = ? OR entity_b = ?)
-- 3. Memory cleanup: DELETE WHERE memory_id = ?
CREATE INDEX IF NOT EXISTS idx_entity_rel_user_ab ON entity_relations(user_id, entity_a, entity_b);
CREATE INDEX IF NOT EXISTS idx_entity_rel_memory_id ON entity_relations(memory_id);
-- delete() runs "DELETE FROM entity_relations WHERE memory_id = ?" on every
-- single memory deletion (cleanup's decay sweep, dream()'s merge-loser
-- deletes) — without this index that's a full table scan every time.
CREATE INDEX IF NOT EXISTS idx_entity_rel_memory_id ON entity_relations(memory_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_pair
    ON entity_relations(user_id, entity_a, entity_b, relation);
"""


def _norm_entity(e: str) -> str:
    return (e or "").strip()


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered pair key (casefold order, display preserves first-seen casing via callers)."""
    aa, bb = _norm_entity(a), _norm_entity(b)
    if aa.casefold() <= bb.casefold():
        return aa, bb
    return bb, aa


def ensure_entity_relations_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_ENTITY_RELATIONS_DDL)
    conn.commit()


def upsert_co_mentions(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    entities: Iterable[str],
    memory_id: str | None = None,
    updated_at: str | None = None,
) -> int:
    """Record co-mention pairs for entities on one memory. Returns pairs touched."""
    from cognition.memory.vecstore import utc_now_iso

    # Store casefolded identity, not display casing: entity_relations is a
    # traversal index, never shown to the user directly, and the unique
    # index on (user_id, entity_a, entity_b, relation) is a plain TEXT
    # comparison — "GRACE" and "Grace" would otherwise collide in `seen`
    # here (so we'd correctly dedup THIS memory's pairs) but still create
    # two distinct rows across different memories that used different
    # casing for the same entity, splitting one node's edges into two.
    # Alias resolution (ENTITY_ALIAS_JSON) has already run inside
    # extract_entities(), so entities arriving here are already canonical.
    ents = []
    seen: set[str] = set()
    for e in entities:
        n = _norm_entity(e)
        if not n:
            continue
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        ents.append(key)
    if len(ents) < 2:
        return 0

    ensure_entity_relations_schema(conn)
    now = updated_at or utc_now_iso()
    touched = 0
    for a, b in combinations(ents, 2):
        # No casefold-equality guard needed here: `ents` was already
        # deduped by casefold via `seen` above, so combinations() can
        # never hand us a == b.
        ea, eb = _ordered_pair(a, b)
        conn.execute(
            """
            INSERT INTO entity_relations (user_id, entity_a, entity_b, relation, weight, memory_id, updated_at)
            VALUES (?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT(user_id, entity_a, entity_b, relation) DO UPDATE SET
                weight = entity_relations.weight + 1.0,
                memory_id = COALESCE(excluded.memory_id, entity_relations.memory_id),
                updated_at = excluded.updated_at
            """,
            (user_id, ea, eb, RELATION_CO_MENTION, memory_id, now),
        )
        touched += 1
    return touched


def rebuild_entity_relations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    clear: bool = True,
) -> dict[str, int]:
    """Rebuild co-mention edges from memories.entities JSON for one user."""
    ensure_entity_relations_schema(conn)
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "entities" not in cols:
        return {"pairs": 0, "memories": 0, "note": 1}

    if clear:
        conn.execute(
            "DELETE FROM entity_relations WHERE user_id = ? AND relation = ?",
            (user_id, RELATION_CO_MENTION),
        )

    rows = conn.execute(
        """
        SELECT id, entities FROM memories
        WHERE user_id = ?
          AND (status IS NULL OR status = 'active')
        """,
        (user_id,),
    ).fetchall()

    pairs = 0
    for row in rows:
        ents = entities_from_json(row["entities"])
        pairs += upsert_co_mentions(
            conn, user_id=user_id, entities=ents, memory_id=str(row["id"])
        )
    conn.commit()
    log.info("entity_relations rebuild user=%s memories=%d pairs=%d", user_id, len(rows), pairs)
    return {"pairs": pairs, "memories": len(rows)}

def backfill_entities(
    conn: sqlite3.Connection,
    *,
    user_id: str | None = None,
    limit: int = 0,
    only_empty: bool = True,
) -> int:
    """Fill entities/kind for existing rows using rule-based extractors.

    No re-embed. Returns number of rows updated.
    """
    ensure_phase_a_schema(conn)
    cols = existing_columns(conn)
    if "entities" not in cols:
        return 0

    sql = "SELECT id, memory, entities, kind FROM memories WHERE 1=1"
    params: list[Any] = []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    if only_empty:
        sql += " AND (entities IS NULL OR entities = '' OR entities = '[]')"
    sql += " ORDER BY created_at DESC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    updated = 0
    for row in rows:
        text = row["memory"] or ""
        ents = extract_entities(text)
        kind = classify_kind(text, default=str(row["kind"] or KIND_FACT))
        conn.execute(
            "UPDATE memories SET entities = ?, kind = ? WHERE id = ?",
            (entities_to_json(ents), kind, row["id"]),
        )
        updated += 1
    if updated:
        conn.commit()
        log.info("memory Phase B backfill: updated %d rows", updated)
    return updated


def audit_entity_extraction(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    sample_n: int = 50,
) -> dict[str, Any]:
    """Diagnostic: how well is extraction covering recent memories?

    Run this after deploying extraction changes, or periodically, to check
    graph density. Not called anywhere automatically — invoke manually or
    from a maintenance script.
    """
    rows = conn.execute(
        "SELECT id, memory, entities FROM memories WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (user_id, sample_n),
    ).fetchall()

    stats = {
        "sample_size": 0,
        "zero_entity": 0,
        "single_entity": 0,
        "multi_entity": 0,
        "pairs_created": 0,
    }
    for row in rows:
        stats["sample_size"] += 1
        ents = entities_from_json(row["entities"])
        if not ents:
            # Column may not be backfilled yet — extract live for the audit.
            ents = extract_entities(row["memory"] or "")
        if len(ents) == 0:
            stats["zero_entity"] += 1
        elif len(ents) == 1:
            stats["single_entity"] += 1
        else:
            stats["multi_entity"] += 1
            stats["pairs_created"] += len(list(combinations(ents, 2)))

    log.info("entity extraction audit user=%s stats=%s", user_id, stats)
    return stats


__all__ = [
    "_ALLCAPS_RE",
    "_AROUSAL_HIGH_RE",
    "_AROUSAL_LOW_RE",
    "_AROUSAL_MID_RE",
    "_CALLED_RE",
    "_DOMAIN_ENTITY_RE",
    "_DOMAIN_ENTITY_TERMS",
    "_EMOTION_QUERY_RE",
    "_ENTITY_ALIASES",
    "_ENTITY_RELATIONS_DDL",
    "_KIND_RULES",
    "_PROPER_SPAN_RE",
    "_QUOTED_RE",
    "_STOP_ENTITIES",
    "_VALENCE_NEG_RE",
    "_VALENCE_POS_RE",
    "_VALENCE_STRONG_RE",
    "_clean_entity",
    "_env_bool",
    "_is_sticky_neg",
    "_norm_entity",
    "_ordered_pair",
    "_query_engages_memory",
    "RELATION_CO_MENTION",
    "RELATION_RELATED",
    "SALIENCE_POLICY_RE",
    "apply_neg_hard_filter",
    "arousal_rank_bonus",
    "audit_entity_extraction",
    "backfill_entities",
    "classify_kind",
    "classify_write_op",
    "ensure_entity_relations_schema",
    "entities_from_json",
    "entities_to_json",
    "entity_overlap_score",
    "extract_entities",
    "infer_arousal_score",
    "infer_salience_hit",
    "infer_valence_score",
    "infer_valence_tag",
    "normalize_memory_text",
    "rebuild_entity_relations",
    "resolve_entity_alias",
    "tag_from_score",
    "upsert_co_mentions",
    "_arousal_enabled",
    "_arousal_rank_weight",
    "_neg_hard_filter_enabled",
    "_neg_hard_threshold",
    "ENTITY_IMPORTANCE_ALPHA",
    "ENTITY_IMPORTANCE_BETA",
    "MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT",
    "MEMORY_SUPERSESSION_CHAIN_EXPAND",
    "MEMORY_SUPERSESSION_CHAIN_KINDS",
    "MEMORY_SPREADING_ENABLED",
    "MEMORY_SPREADING_MAX_DEPTH",
    "MEMORY_SPREADING_DECAY",
    "MEMORY_SPREADING_MIN_STRENGTH",
    "compute_entity_importance_map",
    "memory_max_activation",
    "memory_max_entity_importance",
    "should_expand_supersession_chain",
    "spread_activation",
    "walk_supersession_chain",
    "neural_activate",
    "neural_step",
    "neural_importance",
    "NEURAL_NET_ENABLED",
    "NEURAL_DECAY",
    "NEURAL_THRESHOLD",
    "NEURAL_LEAK",
    "NEURAL_SPREAD_DEPTH",
]
import json as _json


def neural_activate(
    memorize_or_backend: Any,
) -> dict[str, float]:
    """One timestep of neural network activation propagation over the entity graph.

    Returns a dict of entity -> activation strength after one step.
    """
    from cognition.memory.memorize import AikoMemorize

    mem = None
    if hasattr(memorize_or_backend, "_conn"):
        mem = memorize_or_backend
    elif hasattr(memorize_or_backend, "_mem"):
        mem = memorize_or_backend._mem
    if mem is None:
        log.debug("neural net: no memorize backend found")
        return {}

    user_id = getattr(mem, "_user_id", None) or getattr(memorize_or_backend, "_user_id", None)
    if not user_id:
        log.debug("neural net: no user_id found")
        return {}

    # ── load node features (valence/arousal) ──
    node_valence: dict[str, float] = {}
    node_arousal: dict[str, float] = {}
    node_count: dict[str, int] = {}

    try:
        rows = mem._conn.execute(
            """
            SELECT entity_a AS ent, valence_score, arousal_score
            FROM entity_relations
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        for row in rows:
            ent = str(row[0] or "").casefold()
            vs = row[1]
            ar = row[2]
            if vs is not None:
                node_valence[ent] = node_valence.get(ent, 0.0) + float(vs)
                node_count[ent] = node_count.get(ent, 0) + 1
            if ar is not None:
                node_arousal[ent] = node_arousal.get(ent, 0.0) + float(ar)
                node_count[ent] = node_count.get(ent, 0) + 1
        for ent in node_valence:
            node_valence[ent] = node_valence[ent] / node_count[ent] if node_count[ent] > 0 else 0.0
        for ent in node_arousal:
            node_arousal[ent] = node_arousal[ent] / node_count[ent] if node_count[ent] > 0 else 0.0
    except Exception as e:
        log.debug("neural net: valence/arousal load failed: %s", e)

    # ── load directed edges with weights ──
    edge_weight: dict[tuple[str, str], float] = {}
    try:
        rows = mem._conn.execute(
            """
            SELECT entity_a, entity_b, weight FROM entity_relations
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        for row in rows:
            a = str(row[0] or "").casefold()
            b = str(row[1] or "").casefold()
            w = float(row[1] or 0.0)
            if not a or not b or a == b:
                continue
            edge_weight[(a, b)] = edge_weight.get((a, b), 0.0) + w
            edge_weight[(b, a)] = edge_weight.get((b, a), 0.0) + w
    except Exception as e:
        log.debug("neural net: edge load failed: %s", e)

    # ── one timestep of activation propagation ──
    activation: dict[str, float] = {}
    for ent in set(list(node_valence.keys()) + list(node_arousal.keys())):
        v = node_valence.get(ent, 0.0)
        a = node_arousal.get(ent, 0.0)
        activation[ent] = abs(v) + 0.1 * abs(a) if (abs(v) > 0 or abs(a) > 0) else 0.0

    for _ in range(NEURAL_SPREAD_DEPTH):
        new_activation: dict[str, float] = {}
        for ent, act in activation.items():
            act = act * (1.0 - NEURAL_LEAK)
            if act < NEURAL_THRESHOLD:
                continue
            for (src, dst), w in edge_weight.items():
                if src != ent:
                    continue
                neighbor_act = act * w * NEURAL_DECAY
                new_activation[dst] = new_activation.get(dst, 0.0) + neighbor_act

        activation = {ent: act for ent, act in activation.items() if act >= NEURAL_THRESHOLD}

    importance: dict[str, float] = {}
    for ent, act in activation.items():
        v = node_valence.get(ent, 0.0)
        importance[ent] = act + 0.5 * abs(v)

    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    log.info("neural net: top entities: %s", top)

    # Hebbian weight update
    if NEURAL_NET_ENABLED:
        try:
            now = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
            for ent, act in activation.items():
                edges_to_bump = mem._conn.execute(
                    """
                    SELECT entity_a, entity_b, weight FROM entity_relations
                    WHERE user_id = ? AND (entity_a = ? OR entity_b = ?)
                    """,
                    (user_id, ent, ent),
                ).fetchall()
                for row in edges_to_bump:
                    a = str(row[0] or "").casefold()
                    b = str(row[1] or "").casefold()
                    idx = (a, b) if a != b else (a, a)
                    current = edge_weight.get(idx, 0.0)
                    boost = 0.01 * act * abs(node_valence.get(ent, 0.0))
                    new_w = current + boost
                    mem._conn.execute(
                        "UPDATE entity_relations SET weight = ?, updated_at = ? WHERE user_id = ? AND entity_a = ? AND entity_b = ?",
                        (new_w, now, user_id, a, b),
                    )
        except Exception as e:
            log.debug("neural net: weight update failed: %s", e)

    return importance


def neural_step(
    memorize_or_backend: Any,
) -> dict[str, float]:
    """Run one full neural network step and return entity importances.

    Convenience wrapper around neural_activate that also persists weight
    updates (Hebbian learning) when NEURAL_NET_ENABLED is true.
    """
    imp = neural_activate(memorize_or_backend)
    return imp
