"""Memory engine: SQLite/sqlite-vec personal memory backend.

Public API hub — external consumers import from here:
    from cognition.memory.memorize import AikoMemorize, ...

The engine classes (_MemoryBackend, AikoMemorize) live here; stateless
helper modules (schema, entity, imprint, search, lifecycle) are re-exported
so the stable public surface is unchanged after the backend split.
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from typing import Any
import queue
import threading
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from system import bioclock
from system import brain_trace as _brain_trace
from cognition.memory.vecstore import initialize_store_db
from system.userspace import current_display_name, current_user_id
import sqlite_vec
from openai import OpenAI

from cognition.memory.forget import ACCESS_COUNT_CAP, compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD, negative_recall_penalty, salience_score, resolve_ambient_valence
# Frozen relative-time / date-check facts (see imprint.py) are dropped from
# recalled context in format_for_context — they would otherwise masquerade
# as the current date and contaminate Aiko's sense of "now". The
# <current_datetime> block is the only authoritative clock.
from cognition.memory.imprint import _is_stale_temporal_fact
from cognition.memory.narrative import query_wants_emotion, format_supersession_narrative
from system.log import get_logger
from cognition.memory.vecstore import HarrierEmbedder

log = get_logger(__name__)

# Stateless helpers (explicit re-exports; each source module mirrors its __all__).
from .schema import (
    BOOT_LABELS,
    EMBED_DIMS,
    FTS_LIMIT,
    GRAPH_LIMIT,
    KIND_FACT,
    KIND_SCENE,
    KIND_EPISODE,
    KIND_SCHEMA,
    KNN_LIMIT,
    MEMORY_CONTEXT_FACT_CHARS,
    MEMORY_CONTEXT_TOTAL_CHARS,
    MEMORY_CROSS_STORE_CONTEXT_CHARS,
    MEMORY_CROSS_STORE_ENABLED,
    MEMORY_LIFECYCLE_BATCH_SIZE,
    MEMORY_NEG_RECALL_AVOID,
    MEMORY_NEG_RECALL_AVOID_EXCEPT,
    MEMORY_RANK_ACCESS_WEIGHT,
    MEMORY_RANK_GRAPH_WEIGHT,
    MEMORY_RANK_PINNED_WEIGHT,
    MEMORY_RANK_RECENCY_HALF_LIFE_DAYS,
    MEMORY_RANK_RECENCY_WEIGHT,
    MEMORY_RECALL_CONTEXT_MATCH_ENABLED,
    MEMORY_RECALL_CONTEXT_MATCH_WEIGHT,
    MEMORY_RECALL_SCORE_THRESHOLD,
    MEMORY_RECENCY_RERANK_ENABLED,
    MEMORY_RECENCY_RERANK_THRESHOLD,
    MEMORY_SEARCH_CACHE_SIZE,
    MEMORY_SEARCH_CACHE_TTL,
    MEMORY_SPREADING_MAX_EXTRA,
    MEMORY_SPREADING_SCORE_WEIGHT,
    MEMORY_STATE_TAGS_ENABLED,
    MEMORY_SUPERSESSION_NARRATIVE,
    MEMORY_SUPERSESSION_NARRATIVE_MAX,
    MEMORY_WRITE_IDLE_GRACE,
    MEMORY_WRITE_MAX_WAIT,
    PERSONA_CACHE_TTL,
    PERSONA_CONTEXT_CHARS,
    PERSONA_RECALL_LIMIT,
    QUICK_FTS_LIMIT,
    QUICK_GRAPH_LIMIT,
    QUICK_KNN_LIMIT,
    RRF_K,
    SCENE_CONTEXT_CHARS,
    SCENE_CONTEXT_LIMIT,
    SCENE_MEMBER_LIMIT,
    SOURCE_CHAT,
    SOURCE_DREAM,
    SOURCE_PIN,
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    _DDL,
    _PHASE_A_COLUMNS,
    _active_sql,
    _first_json_array,
    _memory_db_path_for_user,
    _sqlite_batch_get_payloads,
    _sqlite_get_vector,
    _sqlite_knn_search,
    _sqlite_pinned_ids,
    _sqlite_set_payload,
    ensure_episode_schema,
    ensure_l2_scene_schema,
    ensure_l3_schema_schema,
    ensure_phase_a_schema,
    existing_columns,
    parse_json_array,
    vacuum_memory_db,
)

from .entity import (
    MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT,
    ENTITY_IMPORTANCE_CACHE_TTL,
    MEMORY_SPREADING_ENABLED,
    SALIENCE_POLICY_RE,
    _arousal_enabled,
    apply_neg_hard_filter,
    arousal_rank_bonus,
    backfill_entities,
    classify_kind,
    classify_write_op,
    ensure_entity_relations_schema,
    entities_from_json,
    entities_to_json,
    entity_overlap_score,
    extract_entities,
    infer_arousal_score,
    infer_salience_hit,
    infer_valence_score,
    infer_valence_tag,
    normalize_memory_text,
    rebuild_entity_relations,
    should_expand_supersession_chain,
    tag_from_score,
    upsert_co_mentions,
    walk_supersession_chain,
)

from .imprint import (
    _EXTRACT_MAX_TOKENS,
    _EXTRACT_MIN_CHARS,
    _EXTRACT_PROMPT,
    _EXTRACT_TIMEOUT,
    _HEDGE_RE,
    _force_subject_name,
    _valence_from_llm,
)

from .episode import (
    _BROAD_RECALL_RE,
    _is_trivial_input,
    _normalize_memory_text,
    _sanitize_fts_query,
    DREAM_BOOST_AMOUNT,
    DREAM_MERGE_THRESHOLD,
    DREAM_SCHEMA_ENABLED,
    DREAM_SCHEMA_MAX_CLUSTERS,
    DREAM_SCHEMA_MIN_MEMBERS,
    DREAM_SCHEMA_VALENCE_MAJORITY,
    WRITE_DEDUP_THRESHOLD,
    _SALIENCE_RE,
)
from .episode import MEMORY_WM_CAPACITY  # re-exported from episode for compat


class _MemoryBackend:
    """
    sqlite-vec + FTS5 + RRF memory backend.

    Changes from original:
      - Extraction LLM runs at temperature=0.0 for deterministic fact output.
      - _extract_facts() filters hedging language via _HEDGE_RE before
        returning — uncertain facts are never persisted.
      - add() runs a dedup check per fact before insert: if a near-identical
        vector already exists (cosine >= WRITE_DEDUP_THRESHOLD), the fact is
        skipped rather than creating a redundant entry.
      - add_raw() now runs the same dedup check as add() (previously it had
        none, which allowed unbounded duplicate pinned inserts).
      - search() collapses exact-text duplicates before final ranking,
        keeping only the most recently created row per duplicate cluster;
        runs a tiered quick/wide candidate pass; applies a recency-among-
        relevant rerank. Pinned rows only get MEMORY_RANK_PINNED_WEIGHT as
        a mild score bonus (no reserved slots). Entity-graph candidates
        (via _graph_pass) are now fused into the same RRF-style scoring
        as KNN/FTS, with MEMORY_RANK_GRAPH_WEIGHT as their tiebreaker.
        See module docstring for the full stage breakdown.

    Fixes applied:
      - Phase A schema (status/supersedes_id/kind/source/entities columns)
        is now migrated once at __init__ time, not lazily on first write.
        Previously a fresh boot or un-migrated DB would hit
        `search()` before any `add()`/`add_raw()` call and crash with
        "no such column: m.status", because `_active_sql()` referenced a
        column that had never been created.
      - `search()` now holds `self._db_lock` across its entire body (quick
        pass, wide pass, and the scoring/rank step in between), not just
        the first KNN call. The connection is shared with the async write
        worker thread, so partial locking left most of the read path
        racing against concurrent writes/dream/cleanup.
      - `iter_all()` now takes the lock around each page fetch (not held
        across the yield, to avoid blocking other threads for the whole
        duration a caller spends processing a batch).
      - `add()`'s created_at now uses datetime.now(timezone.utc).isoformat(),
        matching add_raw()/_touch_memories()/pin(). It previously used
        bioclock.local_now(), which produced two incompatible clock
        formats in the same column depending on write path — skewing
        _rank_and_score's recency scoring, silently breaking _dream_boost's
        is_recent check (naive-vs-aware subtraction raised, was swallowed
        by a bare except), and making get_since()/get_between()'s raw
        string range comparisons sort inconsistently.
      - `_wait_for_write_window()`'s hard-cap check no longer requires
        `not is_active_turn()` to also be true. The deadline now overrides
        a turn state stuck "active" forever — which is the exact scenario
        MEMORY_WRITE_MAX_WAIT exists to guard against, so gating the cap on
        that same condition meant it could never fire when it mattered.
      - `_sqlite_get_vector()` now returns [] explicitly when the row is
        missing or its embedding column is empty, instead of implicitly
        returning None — callers (`_dream_merge`) treat a falsy return as
        "skip this memory", and an implicit None was accidental but is now
        an explicit, documented contract.
      - Entity-graph read path: `_graph_pass()` queries `entity_relations`
        for memories connected to entities extracted from the query text,
        and those candidates are folded into `_rank_and_score()` alongside
        KNN/FTS — a memory only the graph pass finds can now actually enter
        the result set, not just reorder it after the fact (see module
        docstring, "Entity graph fusion").
    """

    def __init__(
        self,
        db_path:         str,
        llm_base_url:    str,
        model:           str,
        embed_cache:     str | None = None,
        user_id:         str | None = None,   # NEW
        embedder:        "HarrierEmbedder | None" = None,  # shared process-wide embedder
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = db_path
        self._user_id  = user_id or current_user_id()   # CHANGED
        self._llm_base = llm_base_url.rstrip("/")
        self._model    = model
        self._client   = OpenAI(base_url=self._llm_base, api_key=os.getenv("LLM_API_KEY", "") or "not-needed")
        # Reuse the owner's shared embedder when given (keeps its TTL cache
        # warm and avoids re-reading the disk cache on every user switch);
        # standalone construction still works via embed_cache.
        self._embedder = embedder or HarrierEmbedder(cache_path=embed_cache)
        # Tri-state: None = not yet probed, True/False = known after the
        # first _extract_facts call. Lets us stop paying for a failed
        # response_format attempt every single turn once we know the
        # server in front of us doesn't support it (see _extract_facts).
        self._json_schema_supported: bool | None = None
        self._conn = self._connect()
        self._db_lock = threading.RLock()
        # Search-result cache — _search_top() reads/writes this directly, so it
        # must live on _MemoryBackend (the type that actually owns the method),
        # not just on the AikoMemorize wrapper. AikoMemorize's __init__ also
        # creates its own copy (used by switch_user paths) but the backend's
        # instance is the one the search path touches.
        from collections import OrderedDict as _OD
        self._search_cache: _OD[tuple[str, str, int, bool], tuple[float, list[dict]]] = _OD()
        self._search_cache_lock = threading.RLock()
        # Super-node cache for entity-graph fusion (see _refresh_high_freq_entities /
        # _graph_pass). Initialized here rather than lazily via getattr — this
        # class owns its own __init__, so there's no reason to defend against
        # the attribute not existing yet.
        self._high_freq_entities: set[str] = set()
        self._high_freq_computed_at: float = 0.0
        # Per-user entity-importance cache. compute_entity_importance_map()
        # full-scans all memories + entity_relations on every cache-miss
        # recall; a short TTL makes it near-free without going stale.
        self._entity_importance_cache: dict[str, tuple[float, dict[str, float]]] = {}
        self._entity_importance_cache_lock = threading.Lock()
        # FIX 1: migrate Phase A schema immediately, not lazily inside
        # add()/add_raw(). Otherwise a read-only path (search() on a fresh
        # boot or a DB that hasn't been written to yet) hits `_active_sql()`
        # referencing `m.status` before that column exists.
        with self._db_lock:
            ensure_phase_a_schema(self._conn)
            ensure_l2_scene_schema(self._conn)
            ensure_l3_schema_schema(self._conn)
            ensure_entity_relations_schema(self._conn)
            ensure_episode_schema(self._conn)

    def _connect(self) -> sqlite3.Connection:
        return initialize_store_db(self._db_path, _DDL, user_id=self._user_id, vector=True)

    def _ensure_open(self) -> None:
        """Re-open the sqlite connection if it has been closed.

        Long-running scheduled jobs (monthly_consolidate, daily_reflect_and_dream)
        can outlive a user switch — the write worker drains in-flight writes and
        the boot process may close the connection under us, leaving the backend
        holding a closed connection. Re-opening here is safe because the DB
        schema migrations are idempotent and a re-open does not change the
        schema or row visibility (sqlite-vec is file-backed and same-process).
        """
        try:
            # Cheap probe — does not allocate, just checks connection state.
            self._conn.execute("SELECT 1").fetchone()
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            log.warning(
                "Memory backend connection was closed (user switch / boot teardown?) — re-opening."
            )
            try:
                self._conn = self._connect()
            except Exception as e:
                log.error("Memory backend re-open failed: %s", e)
                raise

    def _clear_search_cache(self) -> None:
        """Drop the search-result cache. Called by AikoMemorize._clear_search_cache
        after identity / persona updates so the next recall rebuilds fresh."""
        with self._search_cache_lock:
            self._search_cache.clear()

    # ── shim for callers that hold an AikoMemorize vs _MemoryBackend interchangeably ──
    def _resolve_user_id(self, user_id: str | None = None) -> str:
        """Resolve user_id when this backend is addressed directly (e.g. from think/learn).

        AikoMemorize owns the canonical logic; this shim keeps direct _MemoryBackend
        callers from crashing with '_MemoryBackend has no attribute _resolve_user_id'.
        """
        if user_id:
            return str(user_id)
        return self._user_id or current_user_id()

    def get_user_id(self) -> str:  # type: ignore[override]
        return self._user_id or current_user_id()

    def get_display_name(self) -> str:
        # Backend has no display_name; callers should use AikoMemorize but shim avoids crash.
        try:
            from system.userspace import current_display_name
            return current_display_name() or self._user_id or ""
        except Exception:
            return self._user_id or ""

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str, *, query: bool = False) -> list[float]:
        """Embed a single string with HarrierEmbedder. Returns a plain float list."""
        if query:
            return self._embedder.embed_query(text).tolist()
        return list(self._embedder.embed([text]))[0].tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single batched GGUF call."""
        return self._embedder.embed_batch(texts).tolist()

    # ── extraction ────────────────────────────────────────────────────────────

    def _should_extract(self, messages: list[dict]) -> bool:
        """Return False for trivial turns below minimum char threshold."""
        total = sum(
            len(m.get("content") or "")
            for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        )
        return total >= _EXTRACT_MIN_CHARS

    def _extract_facts(self, messages: list[dict], display_name: str | None = None) -> list[tuple[str, int | None]]:
        """
        Send conversation to the OpenAI-compatible local LLM and parse the returned JSON fact array.

        Returns a list of (fact, valence_score) pairs in which each fact is a
        cleaned short string and valence_score is the model-provided −2..+2
        value (None when the source was a legacy string item or carried no
        usable score).

        Changes from original:
          - temperature=0.0 for deterministic output — reduces confabulation.
          - Post-parse hedge filter: facts containing uncertain language
            (_HEDGE_RE, word-boundary matched) are dropped before returning.
          - Only user/assistant turns with real content are sent.
          - response_format=json_schema is tried first (grammar-constrained,
            no fences/preamble possible); if the server rejects it, we fall
            back to an unconstrained call + _first_json_array salvage and
            remember the server doesn't support it for the rest of this
            instance's life, so we stop paying for a failing attempt every
            turn.
          - Parser accepts both legacy string items and the newer
            {"fact": ..., "valence_score": ...} object items, so an old model
            output or a mixed array never crashes the write path.
        """
        if not self._should_extract(messages):
            return []

        clean_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        ]

        while clean_messages and clean_messages[0].get("role") != "user":
            clean_messages.pop(0)

        while clean_messages and clean_messages[-1].get("role") == "assistant":
            if any(m.get("role") == "user" for m in clean_messages[:-1]):
                break
            clean_messages.pop()

        if not clean_messages:
            return []

        total = sum(len(m.get("content") or "") for m in clean_messages)
        if total < _EXTRACT_MIN_CHARS:
            return []

        user_name = (display_name or current_display_name()).strip()
        if user_name.casefold() == "aiko":
            user_name = "User"  # belt only — prefer reject at set time      
        convo = "\n".join(
            f"{user_name}: {m['content'].strip()}" if m["role"] == "user"
            else f"Aiko: {m['content'].strip()}"
            for m in clean_messages
        )

        prompt = _EXTRACT_PROMPT.format(conversation=convo, user_name=user_name)

        base_kwargs = dict(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=_EXTRACT_MAX_TOKENS,
            temperature=0.0,  # deterministic — reduces hallucinated facts
            timeout=_EXTRACT_TIMEOUT,
        )
        schema_kwargs = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "facts",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fact": {"type": "string"},
                                "subject": {"type": "string", "enum": ["user", "assistant"]},
                                "valence_score": {"type": "integer"},
                            },
                            "required": ["fact", "subject"],
                        },
                    },
                },
            }
        }

        raw = None
        if self._json_schema_supported is not False:
            # Grammar-constrained output: no markdown fences, no repeated
            # arrays, no <think> preamble possible (schema forces token 0
            # to be '['). Try this first unless we've already learned this
            # server rejects the param.
            try:
                resp = self._client.chat.completions.create(**base_kwargs, **schema_kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                self._json_schema_supported = True
            except Exception as e:
                if self._json_schema_supported is None:
                    log.warning(
                        f"Server rejected response_format=json_schema ({e}); "
                        "falling back to unconstrained output + salvage parsing "
                        "for the rest of this session."
                    )
                    self._json_schema_supported = False
                else:
                    log.warning(f"Extraction LLM call failed: {e}")
                    return []

        if raw is None:
            # Either we already know this server doesn't support
            # response_format, or the schema attempt just failed for the
            # first time — retry once, unconstrained.
            try:
                resp = self._client.chat.completions.create(**base_kwargs)
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                log.warning(f"Extraction LLM call failed: {e}")
                return []

        # With schema enforcement, parsing is a plain json.loads. Without
        # it (fallback path), the model may wrap the array in markdown
        # fences or add a preamble, or truncate it mid-response when the
        # token cap is hit — salvage the first complete array, or recover
        # the complete objects emitted so far so truncated facts aren't lost.
        parsed = parse_json_array(raw)
        if parsed is None:
            log.warning(f"Failed to parse extraction JSON: {raw[:200]!r}")
            return []

        # Normalise legacy string items and newer object items into pairs.
        pairs: list[tuple[str, int | None]] = []
        for x in parsed:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    pairs.append((t, None))
            elif isinstance(x, dict):
                raw_fact = x.get("fact") or x.get("text") or ""
                if not isinstance(raw_fact, str):
                    continue
                t = raw_fact.strip()
                sc = x.get("valence_score", x.get("valence"))
                try:
                    sc_i = max(-2, min(2, int(sc))) if sc is not None else None
                except (TypeError, ValueError):
                    sc_i = None
                subj = str(x.get("subject") or "").strip().lower()
                if subj not in ("user", "assistant"):
                    subj = "assistant" if t.casefold().startswith("aiko") else "user"
                if t:
                    t = _force_subject_name(t, subj, user_name)
                    pairs.append((t, sc_i))

        # drop facts containing hedging/uncertain language (word-boundary
        # match); drop the paired score alongside the dropped fact
        clean_pairs = []
        for fact, sc in pairs:
            if _HEDGE_RE.search(fact):
                log.debug(f"Dropped hedging fact: {fact!r}")
                continue
            clean_pairs.append((fact, sc))
              
        # Repair common user/assistant subject swaps (Oppa vs Aiko).
        from cognition.memory.entity import sanitize_fact_score_pairs
        return sanitize_fact_score_pairs(
            clean_pairs,
            user_name=user_name,
            assistant_name="Aiko",
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def _insert_row(
        self,
        *,
        mem_id: str,
        user_id: str,
        text: str,
        now: str,
        vector: list[float] | None,
        pinned: int = 0,
        source: str = SOURCE_CHAT,
        supersedes_id: str | None = None,
        kind: str | None = None,
        entities: list[str] | None = None,
        scene_id: str | None = None,
        valence_tag: str | None = None,
        llm_score: int | None = None,
        salience_hit: int | None = None,
        schema_sources: list[str] | None = None,
    ) -> None:
        """Insert one memory row (Phase A + L2 columns when present) + its
        vector, and best-effort co-mention edges for the entity graph.

        Extended columns are appended to the INSERT dynamically based on which
        additive columns actually exist on the table, so the same call works
        against a fresh DB, a Phase-A-only DB, and a fully-migrated L2 DB.

        ``vector`` may be None (embedding server unavailable): the row is
        still persisted (FTS5 keeps it searchable lexically) but no
        memories_vec row is written. backfill_missing_vectors() re-embeds
        these rows once the embedder recovers. An un-embedded row is simply
        invisible to KNN until then — never lost.

        ``schema_sources`` — Phase 21: for kind='schema' rows, the JSON list
        of source memory ids this gist was abstracted from (traceability +
        idempotency). Written only when the column exists.
        """
        cols = existing_columns(self._conn)
        kind_val = kind or classify_kind(text, default=KIND_FACT)
        ents_list = entities if entities is not None else extract_entities(text)
        ents_json = entities_to_json(ents_list)

        if llm_score is not None:
            try:
                v_score = max(-2, min(2, int(llm_score)))
            except (TypeError, ValueError):
                v_score = infer_valence_score(text)
        else:
            v_score = infer_valence_score(text)

        # Prefer explicit tag only if caller passed one; else derive from score
        if valence_tag is not None:
            v_tag = valence_tag
        else:
            v_tag = tag_from_score(v_score)
        s_hit = int(salience_hit) if salience_hit is not None else infer_salience_hit(text)
        s_hit = 1 if s_hit else 0

        base_cols = ["id", "user_id", "memory", "created_at", "access_count", "last_accessed_at", "pinned"]
        base_vals: list[Any] = [mem_id, user_id, text, now, 0, "never", pinned]
        ext_cols: list[str] = []
        ext_vals = []
        a_score = infer_arousal_score(text) if _arousal_enabled() else None
        if "status" in cols:
            ext_cols += ["status", "supersedes_id", "kind", "source", "entities"]
            ext_vals  += [STATUS_ACTIVE, supersedes_id, kind_val, source, ents_json]
        if "scene_id" in cols:
            ext_cols.append("scene_id")
            ext_vals.append(scene_id)
        if "valence_tag" in cols:
            ext_cols.append("valence_tag")
            ext_vals.append(v_tag)
        if "valence_score" in cols:
            ext_cols.append("valence_score")
            ext_vals.append(int(v_score))
        if "salience_hit" in cols:
            ext_cols.append("salience_hit")
            ext_vals.append(s_hit)
        if "schema_sources" in cols and schema_sources is not None:
            ext_cols.append("schema_sources")
            ext_vals.append(json.dumps(list(schema_sources), ensure_ascii=False))
        if MEMORY_STATE_TAGS_ENABLED and "state_json" in cols:
            try:
                import json
                hour = bioclock.local_now().hour
                state_json = json.dumps({"local_hour": int(hour)}, ensure_ascii=False)
                ext_cols.append("state_json")
                ext_vals.append(state_json)
            except Exception:
                pass
        if "arousal_score" in cols and a_score is not None:
            ext_cols.append("arousal_score")
            ext_vals.append(int(a_score))
        all_cols = base_cols + ext_cols
        placeholders = ", ".join("?" * len(all_cols))
        self._conn.execute(
            f"INSERT INTO memories ({', '.join(all_cols)}) VALUES ({placeholders})",
            base_vals + ext_vals,
        )
        if vector is not None:
            self._conn.execute(
                "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vector)),
            )

        # Phase D: live co-mention edges (best-effort; never fail the write)
        try:
            if ents_list and len([e for e in ents_list if str(e).strip()]) >= 2:
                upsert_co_mentions(
                    self._conn,
                    user_id=user_id,
                    entities=ents_list,
                    memory_id=mem_id,
                    updated_at=now if isinstance(now, str) else None,
                )
        except Exception as e:
            log.debug("entity_relations upsert skipped: %s", e)

    def _maybe_supersede_neighbor(
        self, user_id: str, vector: list[float], text: str
    ) -> tuple[str, str | None]:
        """Classify the write op against the nearest existing memory: 'add',
        'noop' (near-duplicate, skip), or 'supersede' (replace old_id)."""
        existing = _sqlite_knn_search(
            self._conn, vector, user_id,
            limit=1, threshold=WRITE_DEDUP_THRESHOLD, active_only=True,
        )
        if not existing:
            return "add", None
        sim = 1.0 - float(existing[0]["dist"])
        old_id = str(existing[0]["id"])
        row = self._conn.execute(
            "SELECT memory, pinned FROM memories WHERE id = ?", (old_id,)
        ).fetchone()
        old_text = (row["memory"] if row else "") or ""
        pinned = bool(row and row["pinned"])
        op = classify_write_op(
            similarity=sim,
            new_text=text,
            old_text=old_text,
            dedup_threshold=WRITE_DEDUP_THRESHOLD,
        )
        if op == "supersede" and pinned:
            return "add", None
        if op == "supersede":
            return "supersede", old_id
        return op, None

    def add(self, messages: list[dict], user_id: str, display_name: str | None = None) -> list[str]:
        """
        Extract facts and persist each as a row in memories + memories_vec.

        Write-path dedup/supersede: before inserting each fact, a KNN search
        checks for a near-identical vector already in the store. Near-duplicate
        text is skipped ('noop'); text that changed but is semantically the same
        supersedes the older row (status -> 'superseded').

        Embeddings for all extracted facts are computed in a single batched
        call rather than one-by-one.

        Returns list of new memory IDs. Empty list if nothing extracted.
        """
        pairs = self._extract_facts(messages, display_name=display_name)
        if not pairs:
            return []
        facts = [f for f, _ in pairs]

        # created_at is UTC everywhere (matches add_raw()/_touch_memories()/
        # pin()) — see the class docstring's "Fixes applied" note. Mixing
        # local_now() here and UTC elsewhere broke every downstream
        # comparison: _rank_and_score's recency scoring, _dream_boost's
        # is_recent check, and get_since()/get_between()'s string range
        # comparisons.
        now = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []

        try:
            vectors = self._embed_batch(facts)
        except Exception as e:
            log.warning("Batch embedding failed; persisting %d facts without vectors: %s", len(pairs), e)
            vectors = [None] * len(pairs)
        if len(vectors) != len(pairs):
            log.warning(
                "Batch embedding count mismatch: %d facts, %d vectors; persisting without vectors",
                len(pairs), len(vectors),
            )
            vectors = [None] * len(pairs)

        with self._db_lock:
            try:
                for (fact, llm_sc), vector in zip(pairs, vectors):
                    if vector is not None:
                        op, supersedes_id = self._maybe_supersede_neighbor(user_id, vector, fact)
                    else:
                        # No embedding → can't dedup/supersede; persist as-is.
                        # A later backfill makes it searchable; dream() merge
                        # collapses any near-duplicate this could create.
                        op, supersedes_id = "add", None
                    if op == "noop":
                        log.debug("Skipping near-duplicate fact: %r", fact)
                        continue
                    if op == "supersede" and supersedes_id:
                        cols = existing_columns(self._conn)
                        if "status" in cols:
                            self._conn.execute(
                                "UPDATE memories SET status = ? WHERE id = ?",
                                (STATUS_SUPERSEDED, supersedes_id),
                            )
                            log.info("Superseded memory %s with new fact", supersedes_id)
                    mem_id = str(uuid.uuid4())

                    # Phase 4: tag from fact text (and soft signal from last assistant msg).
                    assist_blob = next(
                        (
                            (m.get("content") or "")
                            for m in reversed(messages)
                            if m.get("role") == "assistant"
                        ),
                        "",
                    )[-400:]
                    tag_src = f"{fact}\n{assist_blob}"
                    v_score = infer_valence_score(tag_src)
                    s_hit = max(infer_salience_hit(fact), infer_salience_hit(assist_blob))

                    self._insert_row(
                        mem_id=mem_id,
                        user_id=user_id,
                        text=fact,
                        now=now,
                        vector=vector,
                        pinned=0,
                        source=SOURCE_CHAT,
                        supersedes_id=supersedes_id,
                        llm_score=llm_sc if (_valence_from_llm() and llm_sc is not None) else v_score,
                        valence_tag=None,
                        salience_hit=s_hit,
                    )
                    ids.append(mem_id)
                self._conn.commit()
            except Exception as e:
                log.warning("Failed to upsert fact batch: %s", e)
                self._conn.rollback()
                return []
        # The embedder is healthy right now (we embedded successfully above),
        # so opportunistically repair rows persisted during a previous outage.
        if ids and all(v is not None for v in vectors):
            try:
                self._maybe_backfill_missing_vectors(user_id)
            except Exception as e:
                log.debug("post-write vector backfill skipped: %s", e)
        return ids

    def add_raw(self, memory: str, user_id: str, *, pinned: bool = False) -> str | None:
        """
        Persist one already-curated memory string without LLM extraction.

        Runs the same write-time dedup/supersede check as add(): near-duplicates
        are skipped; semantically-equal-but-changed text supersedes the older
        row. This closes the gap that previously let repeated calls (e.g. a
        daily-record pin job re-running for the same day) accumulate unbounded
        duplicate rows — especially dangerous for pinned=True inserts, since
        dream()'s merge pass can never delete a pinned memory even as a
        duplicate loser.
        """
        text = (memory or "").strip()
        if not text:
            return None
        try:
            vector = self._embed(text)
        except Exception as e:
            log.warning("Failed to embed raw memory; persisting without vector: %s", e)
            vector = None
        mem_id: str | None = None
        with self._db_lock:
            self._ensure_open()
            try:
                if vector is not None:
                    op, supersedes_id = self._maybe_supersede_neighbor(user_id, vector, text)
                else:
                    op, supersedes_id = "add", None
                if op == "noop":
                    log.debug("Skipping near-duplicate raw memory: %r", text[:80])
                    return None
                if op == "supersede" and supersedes_id:
                    cols = existing_columns(self._conn)
                    if "status" in cols:
                        self._conn.execute(
                            "UPDATE memories SET status = ? WHERE id = ?",
                            (STATUS_SUPERSEDED, supersedes_id),
                        )
                mem_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                self._insert_row(
                    mem_id=mem_id,
                    user_id=user_id,
                    text=text,
                    now=now,
                    vector=vector,
                    pinned=1 if pinned else 0,
                    source=SOURCE_PIN if pinned else SOURCE_CHAT,
                    supersedes_id=supersedes_id,
                )
                self._conn.commit()
            except Exception as e:
                log.warning("Failed to insert raw memory: %s", e)
                self._conn.rollback()
                return None
        # Backfill outside the lock so a recovery embed never blocks recall.
        if mem_id is not None and vector is not None:
            try:
                self._maybe_backfill_missing_vectors(user_id)
            except Exception as e:
                log.debug("post-write vector backfill skipped: %s", e)
        return mem_id

    def supersede_exact(self, memory_id: str, replacement: str, user_id: str) -> str | None:
        """Write a confirmed replacement linked to one exact memory row."""
        text = (replacement or "").strip()
        if not memory_id or not text or not user_id:
            return None
        try:
            vector = self._embed(text)
        except Exception as exc:
            log.warning("Confirmed supersession embedding failed: %s", exc)
            vector = None
        with self._db_lock:
            try:
                row = self._conn.execute("SELECT id, pinned, user_id, status FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id)).fetchone()
                if not row or str(row[3] or "active") == STATUS_SUPERSEDED:
                    return None
                cols = existing_columns(self._conn)
                if "status" in cols:
                    self._conn.execute("UPDATE memories SET status = ? WHERE id = ? AND user_id = ?", (STATUS_SUPERSEDED, memory_id, user_id))
                new_id = str(uuid.uuid4())
                self._insert_row(mem_id=new_id, user_id=user_id, text=text, now=datetime.now(timezone.utc).isoformat(), vector=vector, pinned=int(row[1] or 0), source=SOURCE_CHAT, supersedes_id=memory_id)
                self._conn.commit()
                return new_id
            except Exception as exc:
                log.warning("Confirmed memory supersession failed: %s", exc)
                self._conn.rollback()
                return None

    # ── vector backfill ────────────────────────────────────────────────────────
    # Rows persisted while the embedder was down have no memories_vec entry
    # (see _insert_row's optional vector). These are FTS-searchable but
    # invisible to KNN until re-embedded. backfill_missing_vectors() repairs
    # them in bounded batches, triggered after the next write that succeeds
    # with a live embedder (so we never hammer a down server).

    def _missing_vector_rows(self, user_id: str, limit: int) -> list[sqlite3.Row]:
        with self._db_lock:
            return self._conn.execute(
                """
                SELECT m.id, m.memory
                FROM memories m
                LEFT JOIN memories_vec v ON v.id = m.id
                WHERE m.user_id = ?
                  AND (m.status = 'active' OR m.status IS NULL)
                  AND v.id IS NULL
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

    def backfill_missing_vectors(self, user_id: str, limit: int | None = None) -> int:
        """Re-embed memories rows that have no memories_vec row (persisted
        during an embedder outage). Returns count backfilled. Best-effort;
        a still-down embedder returns 0."""
        batch = int(os.getenv("MEMORY_VECTOR_BACKFILL_LIMIT", "200")) if limit is None else max(1, int(limit))
        try:
            rows = self._missing_vector_rows(user_id, batch)
            if not rows:
                return 0
            texts = [r["memory"] for r in rows]
            vectors = self._embed_batch(texts)
            backfilled = 0
            with self._db_lock:
                for row, vec in zip(rows, vectors):
                    # sqlite-vec 0.1.9 has no INSERT OR REPLACE/IGNORE on vec0
                    # tables, so guard against a concurrent write inserting a
                    # vector for this id between our SELECT and INSERT. Skip —
                    # never fail the whole batch.
                    present = self._conn.execute(
                        "SELECT 1 FROM memories_vec WHERE id = ?", (row["id"],)
                    ).fetchone()
                    if present:
                        continue
                    self._conn.execute(
                        "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                        (row["id"], sqlite_vec.serialize_float32(vec)),
                    )
                    backfilled += 1
                self._conn.commit()
            log.info("Backfilled vectors for %d memories (user=%s)", backfilled, user_id)
            return backfilled
        except Exception as e:
            try:
                self._conn.rollback()
            except Exception:
                pass
            log.debug("vector backfill skipped: %s", e)
            return 0

    def _maybe_backfill_missing_vectors(self, user_id: str) -> int:
        """Backfill rows persisted without a vector. Called only after a write
        that embedded successfully, so it never retries a down embedder per
        write. backfill_missing_vectors() cheaply no-ops when nothing is
        missing, so no separate gate query is needed."""
        try:
            return self.backfill_missing_vectors(user_id)
        except Exception as e:
            log.debug("vector backfill skipped: %s", e)
            return 0

    # ── read ──────────────────────────────────────────────────────────────────

    # ── L2 scene blocks (backend) ─────────────────────────────────────────────
    # A scene is a normal memories row with kind=KIND_SCENE whose text is a
    # mid-grain episode summary. The atomic facts it was built from carry
    # scene_id back to it, so recall can surface the scene (via its own
    # vector match) and expand to its members, or re-link a recalled member to
    # its episode.

    def build_scene(
        self,
        user_id: str,
        *,
        summary: str,
        member_ids: list[str],
        pinned: bool = False,
        created_at: str | None = None,
    ) -> str | None:
        """Persist one scene row (kind='scene') and tag each member fact with
        scene_id pointing back to it. Returns the new scene id, or None if the
        summary is empty / embedding fails. Caller must hold nothing; this
        takes the lock around the creates + link update.

        Re-runs write-time dedup/supersede on the summary itself so re-building
        the same scene replaces (supersedes) rather than duplicates.
        """
        summary = (summary or "").strip()
        if not summary:
            return None
        try:
            vector = self._embed(summary)
        except Exception as e:
            # Same resilience as add()/add_raw(): persist without a vector so
            # the scene (and its member relinks) survive the outage. It stays
            # FTS-searchable and is re-embedded by vector backfill on recovery.
            log.warning("Failed to embed scene summary; persisting without vector: %s", e)
            vector = None
        nows = created_at or datetime.now(timezone.utc).isoformat()
        self._invalidate_entity_importance(user_id)
        with self._db_lock:
            try:
                op, supersedes_id = (None, None)
                if vector is not None:
                    op, supersedes_id = self._maybe_supersede_neighbor(user_id, vector, summary)
                if op == "supersede" and supersedes_id:
                    if "status" in existing_columns(self._conn):
                        self._conn.execute(
                            "UPDATE memories SET status = ? WHERE id = ?",
                            (STATUS_SUPERSEDED, supersedes_id),
                        )
                    # Re-use the superseded row's id so dangling member links
                    # keep pointing at a live scene.
                    mem_id = supersedes_id
                    self._update_scene_row(mem_id, summary, vector, nows, pinned)
                else:
                    mem_id = str(uuid.uuid4())
                    self._insert_row(
                        mem_id=mem_id,
                        user_id=user_id,
                        text=summary,
                        now=nows,
                        vector=vector,
                        pinned=1 if pinned else 0,
                        source=SOURCE_CHAT,
                        kind=KIND_SCENE,
                    )
                member_ids = [m for m in (member_ids or []) if m]
                if member_ids:
                    placeholders = ", ".join("?" * len(member_ids))
                    self._conn.execute(
                        f"UPDATE memories SET scene_id = ? "
                        f"WHERE id IN ({placeholders}) AND user_id = ?",
                        [mem_id, *member_ids, user_id],
                    )
                self._conn.commit()
                return mem_id
            except Exception as e:
                log.warning("Failed to build scene: %s", e)
                self._conn.rollback()
                return None

    def _update_scene_row(
        self, scene_id: str, summary: str, vector: list[float], now: str, pinned: bool,
    ) -> None:
        """Replace a scene row's summary (superseded id reuse) in place."""
        self._conn.execute(
            "UPDATE memories SET memory = ?, created_at = ?, pinned = ?, status = ? WHERE id = ?",
            (summary, now, 1 if pinned else 0, STATUS_ACTIVE, scene_id),
        )
        self._conn.execute(
            "UPDATE memories_vec SET embedding = ? WHERE id = ?",
            (sqlite_vec.serialize_float32(vector), scene_id),
        )

    def list_scenes(
        self,
        user_id: str,
        limit: int = SCENE_CONTEXT_LIMIT,
        active_only: bool = True,
    ) -> list[dict]:
        """Return the most recent scene rows for a user, newest first."""
        if "kind" not in existing_columns(self._conn):
            return []
        status_sql = _active_sql(active_only)
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM memories m
                WHERE m.user_id = ? AND m.kind = ?
                  {status_sql}
                ORDER BY m.created_at DESC
                LIMIT ?
                """.format(status_sql=status_sql),
                (user_id, KIND_SCENE, int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def scene_members(self, scene_id: str, user_id: str, limit: int = SCENE_MEMBER_LIMIT) -> list[dict]:
        """Return the atomic-fact rows linked to a scene, oldest first."""
        if "scene_id" not in existing_columns(self._conn) or not scene_id:
            return []
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM memories m
                WHERE m.scene_id = ? AND m.user_id = ?
                ORDER BY m.created_at ASC
                LIMIT ?
                """,
                (scene_id, user_id, int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def _fts_pass(self, fts_query: str | None, user_id: str, fts_limit: int, active_only: bool = True) -> list[sqlite3.Row]:
        """Run one FTS5 BM25 pass. Returns [] if fts_query is None (nothing usable to match).
        Caller must hold self._db_lock."""
        if fts_query is None:
            return []
        status_sql = _active_sql(active_only)
        return self._conn.execute(
            """
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            AND m.user_id = ?
            {status_sql}
            ORDER BY rank
            LIMIT ?
            """.format(status_sql=status_sql),
            (fts_query, user_id, fts_limit),
        ).fetchall()

    # An entity connected to more than this fraction of a user's active
    # memories is a "super-node" (e.g. the assistant's own name, or a
    # nickname used in most facts) — matching on it returns a near-random
    # slice of the whole memory store rather than a meaningful signal, so
    # _graph_pass drops it from the query entity list before searching.
    _GRAPH_SUPER_NODE_FRACTION = float(os.getenv("MEMORY_GRAPH_SUPER_NODE_FRACTION", "0.3"))
    # How long a cached high-frequency-entity set is trusted before recompute.
    _GRAPH_SUPER_NODE_TTL = float(os.getenv("MEMORY_GRAPH_SUPER_NODE_TTL", "3600"))

    def _refresh_high_freq_entities(self, user_id: str) -> None:
        """
        Recompute the set of super-node entities for this user, if the
        cached set is missing or stale. Caller must hold self._db_lock.

        An entity counts as high-frequency if it co-mentions on more than
        _GRAPH_SUPER_NODE_FRACTION of the user's active memories. Cheap
        enough to run per-call under a TTL guard (two aggregate queries),
        so no separate scheduled job is required — it self-heals as the
        user's memory store grows.

        FIX: this used to count COUNT(DISTINCT memory_id) from
        entity_relations, which undercounts — see _graph_pass for why
        entity_relations.memory_id only ever holds the last memory per
        entity pair. An entity mentioned in 50 memories but always
        alongside a rotating cast of different co-mentioned partners could
        have its true frequency badly undercounted, letting a real
        super-node slip through the filter undetected. Counting distinct
        memory ids directly from memories.entities (ground truth, one row
        per memory) fixes this the same way _graph_pass's read path was
        fixed.
        """
        now = time.monotonic()
        if now - self._high_freq_computed_at < self._GRAPH_SUPER_NODE_TTL and self._high_freq_entities is not None:
            return
        try:
            total_row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id = ? AND (status = 'active' OR status IS NULL)",
                (user_id,),
            ).fetchone()
            total = int(total_row["n"] or 0) if total_row else 0
            if total <= 0:
                self._high_freq_entities = set()
                self._high_freq_computed_at = now
                return
            rows = self._conn.execute(
                """
                SELECT LOWER(je.value) AS entity, COUNT(DISTINCT mm.id) AS cnt
                FROM memories mm, json_each(mm.entities) je
                WHERE mm.user_id = ? AND (mm.status = 'active' OR mm.status IS NULL)
                GROUP BY LOWER(je.value)
                """,
                (user_id,),
            ).fetchall()
            self._high_freq_entities = {
                row["entity"] for row in rows
                if total > 0 and (row["cnt"] or 0) / total > self._GRAPH_SUPER_NODE_FRACTION
            }
            self._high_freq_computed_at = now
        except Exception as e:
            log.debug("high-freq entity refresh skipped: %s", e)
            if self._high_freq_entities is None:
                self._high_freq_entities = set()
            self._high_freq_computed_at = now

    def _graph_pass(
        self,
        query_entities: list[str],
        user_id: str,
        limit: int,
        active_only: bool = True,
    ) -> list[sqlite3.Row]:
        """
        Fetch memories whose stored entity list (memories.entities) overlaps
        the query entities, ranked by overlap count. Returns [] if the query
        has no extractable entities, or if every extracted entity is a
        super-node (see _refresh_high_freq_entities). Caller must hold
        self._db_lock.

        Query entities are casefolded before matching, since memories.entities
        is compared case-insensitively here (LOWER()) against already-
        casefolded query entities.

        FIX: this used to join through entity_relations.memory_id ordered by
        SUM(weight). That's broken: upsert_co_mentions does
        `memory_id = COALESCE(excluded.memory_id, entity_relations.memory_id)`
        on conflict, which means each (entity_a, entity_b) edge only ever
        remembers the LAST memory that mentioned that pair — every earlier
        memory that also mentioned both entities becomes permanently
        unreachable via this path, even though it's still active and still
        genuinely connected. In practice this biased graph recall toward
        whichever memory most recently touched an entity pair ("graph only
        finds recent stuff"), not all memories connected to the query.

        memories.entities is the ground-truth per-memory entity list —
        written once at insert time (_insert_row) and never overwritten —
        so matching against it directly returns every connected memory, not
        just the newest one per pair. entity_relations (co-mention edges)
        is still written at write time and still used by
        _refresh_high_freq_entities' super-node detection is now also
        sourced from memories.entities for the same reason; see below.
        entity_relations itself is retained for rebuild_entity_relations
        tooling and potential future edge-weight features, but is no longer
        read on this hot path.

        This is the entity-graph read path: candidates from here are folded
        into _rank_and_score() alongside KNN/FTS, so a memory that only the
        graph connects to the query can actually enter the result set (not
        just reorder results that were already found some other way — see
        module docstring, "Entity graph fusion").
        """
        if not query_entities:
            return []
        self._refresh_high_freq_entities(user_id)
        super_nodes = self._high_freq_entities
        filtered = [e.casefold() for e in query_entities if e.casefold() not in super_nodes]
        if not filtered:
            return []
        status_sql = _active_sql(active_only, alias="mm")
        placeholders = ",".join("?" * len(filtered))
        return self._conn.execute(
            f"""
            SELECT mm.id AS id, COUNT(*) AS w
            FROM memories mm, json_each(mm.entities) je
            WHERE mm.user_id = ?
              AND LOWER(je.value) IN ({placeholders})
              {status_sql}
            GROUP BY mm.id
            ORDER BY w DESC
            LIMIT ?
            """,
            [user_id] + filtered + [limit],
        ).fetchall()

    def _spreading_extra_ids(
        self,
        user_id: str,
        seed_rows: list,
        *,
        exclude_ids: set[str],
        query: str = "",
    ) -> tuple[dict[str, float], list[str]]:
        """Return (entity_activation, extra_memory_ids) via entity_relations walk."""
        if not MEMORY_SPREADING_ENABLED or MEMORY_SPREADING_MAX_EXTRA <= 0:
            return {}, []
        try:
            from cognition.memory.entity import (
                entities_from_json_safe,
                spread_activation,
            )
        except Exception:
            return {}, []

        seeds: list[str] = []
        for row in seed_rows:
            try:
                raw = row["entities"] if hasattr(row, "keys") else row.get("entities")
                seeds.extend(entities_from_json_safe(raw))
            except Exception:
                pass

        if query:
            try:
                seeds.extend(extract_entities(query))
            except Exception:
                pass

        if not seeds:
            return {}, []

        try:
            edge_rows = self._conn.execute(
                "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            edges = [
                (str(r["entity_a"] or ""), str(r["entity_b"] or ""), float(r["weight"] or 0.0))
                for r in edge_rows
            ]
        except Exception:
            return {}, []

        activation = spread_activation(seeds, edges)
        if not activation:
            return {}, []

        # Strongest entities first (skip pure seeds at 1.0 if you want only neighbors —
        # usually keep all activated)
        ranked_ents = sorted(activation.keys(), key=lambda e: -activation[e])
        extra: list[str] = []
        seen = set(exclude_ids)
        if not ranked_ents:
            return activation, extra

        # Batch fetch: single query for all entities instead of N+1
        # Build LIKE conditions for each entity
        ent_conditions = " OR ".join(["entities LIKE ?"] * len(ranked_ents))
        params = [user_id] + [f"%{ent}%" for ent in ranked_ents]
        rows = self._conn.execute(
            f"""
            SELECT id, entities FROM memories
            WHERE user_id = ?
              AND (status = 'active' OR status IS NULL)
              AND ({ent_conditions})
            ORDER BY last_accessed_at DESC
            LIMIT ?
            """,
            params + [MEMORY_SPREADING_MAX_EXTRA * 3],  # fetch extra to account for filtering
        ).fetchall()

        for r in rows:
            if len(extra) >= MEMORY_SPREADING_MAX_EXTRA:
                break
            mid = str(r["id"])
            if mid in seen:
                continue
            # tighter check: entity really in JSON list
            ents = entities_from_json_safe(r["entities"])
            # Check if any of our ranked entities is in this memory's entities
            if not any(ent.casefold() in {e.casefold() for e in ents} for ent in ranked_ents):
                continue
            seen.add(mid)
            extra.append(mid)
        return activation, extra

    def _fetch_full_rows(self, ids: set[str]) -> dict[str, Any]:
        """Fetch full memory rows for the given IDs. Caller must hold lock."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        # Select only columns needed for ranking to reduce I/O and memory
        cols = "id, memory, created_at, pinned, access_count, last_accessed_at, valence_tag, valence_score, salience_hit, entities, status, kind, supersedes_id"
        rows = self._conn.execute(
            f"SELECT {cols} FROM memories WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _rank_and_score(
        self,
        rank_knn: dict,
        rank_fts: dict,
        rank_graph: dict | None = None,
        entity_importance_map: dict | None = None,
        query_context: tuple[int, int] | None = None,
        row_by_id: dict | None = None,
    ) -> tuple[list[str], dict, dict]:
        """
        Dedup + score one candidate pool (from either the quick or wide pass).
        
        If row_by_id is provided, it must contain full rows for all candidate IDs.
        Otherwise, caller must hold self._db_lock and rows will be fetched internally.

        1. (Optional) Fetch full rows for the union of KNN/FTS/graph candidate ids.
        2. Collapse exact-text duplicates, keeping the most recently created
           row per duplicate cluster.
        3. Score every surviving id: RRF fusion (KNN + FTS + graph) plus
           recency/access/pinned bonuses.

        query_context — (query_valence, query_arousal) inferred from the query
        text. When provided, a small encoding-specificity boost is added to
        memories whose stored valence matches the query valence sign (mood-
        congruent recall; see MEMORY_RECALL_CONTEXT_MATCH_*).

        Returns (ids sorted best-first by score, {id: score}, {id: row}).
        Recency-among-relevant reranking is applied afterward by the
        caller (search()), not here — this method only produces the base
        score-ordered list (RRF + recency + access + pinned + graph weight +
        entity importance + context match).
        """
        rank_graph = rank_graph or {}
        entity_importance_map = entity_importance_map or {}
        q_valence, q_arousal = query_context or (0, 0)

        all_ids = set(rank_knn) | set(rank_fts) | set(rank_graph)
        if not all_ids:
            return [], {}, {}

        if row_by_id is None:
            placeholders = ",".join("?" * len(all_ids))
            # Select only columns needed for ranking to reduce I/O and memory
            cols = "id, memory, created_at, pinned, access_count, last_accessed_at, valence_tag, valence_score, salience_hit, entities, status, kind, supersedes_id"
            rows = self._conn.execute(
                f"SELECT {cols} FROM memories WHERE id IN ({placeholders})",
                list(all_ids),
            ).fetchall()
            row_by_id = {row["id"]: row for row in rows}

        # ── recall-time dedup: collapse exact-text duplicates, keep newest ──
        # Handles the case dream() structurally can't: pinned duplicate rows
        # (dream's merge never deletes a pinned memory, even as the loser),
        # and any duplicate created between dream() runs.
        best_by_text: dict[str, str] = {}
        for mid in all_ids:
            row = row_by_id.get(mid)
            if row is None:
                continue
            norm = _normalize_memory_text(row["memory"])
            current_best = best_by_text.get(norm)
            if current_best is None:
                best_by_text[norm] = mid
                continue
            if row["created_at"] > row_by_id[current_best]["created_at"]:
                best_by_text[norm] = mid
        deduped_ids = set(best_by_text.values())

        def _recency_score(created_at: str) -> float:
            try:
                created = datetime.fromisoformat((created_at or "").replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
                return 0.5 ** (age_days / max(MEMORY_RANK_RECENCY_HALF_LIFE_DAYS, 1e-6))
            except Exception:
                return 0.0

        # Phase 17: session dynamic anchor (mean of last-K query vectors)
        _session_mean = None
        _session_vecs: dict = {}
        try:
            from cognition.memory.entity import (
                MEMORY_SESSION_ANCHOR_ENABLED,
                MEMORY_SESSION_ANCHOR_WEIGHT,
                load_memory_vectors,
                session_boost_for,
                session_mean,
            )
            if MEMORY_SESSION_ANCHOR_ENABLED and MEMORY_SESSION_ANCHOR_WEIGHT > 0:
                # user_id is on rows; pick any
                _uid = None
                for _mid in deduped_ids:
                    _r = row_by_id.get(_mid)
                    if _r is not None and _r["user_id"]:
                        _uid = str(_r["user_id"])
                        break
                if _uid:
                    _session_mean = session_mean(_uid)
                    if _session_mean is not None:
                        _session_vecs = load_memory_vectors(self._conn, list(deduped_ids))
        except Exception as _sess_exc:
            log.debug("session anchor prep skipped: %s", _sess_exc)

        def final_score(mem_id: str) -> float:
            """Compute reciprocal rank fusion score with recency and access boosts."""
            knn = rank_knn.get(mem_id, 0)
            fts = rank_fts.get(mem_id, 0)
            graph = rank_graph.get(mem_id, 0)
            score = 0.0
            if knn:
                score += 1.0 / (RRF_K + knn)
            if fts:
                score += 1.0 / (RRF_K + fts)
            if graph:
                score += MEMORY_RANK_GRAPH_WEIGHT / (RRF_K + graph)

            row = row_by_id.get(mem_id)
            if row is not None:
                score += MEMORY_RANK_RECENCY_WEIGHT * _recency_score(row["created_at"])
                score += MEMORY_RANK_ACCESS_WEIGHT * min(int(row["access_count"] or 0), ACCESS_COUNT_CAP) / max(ACCESS_COUNT_CAP, 1)
                if int(row["pinned"] or 0):
                    score += MEMORY_RANK_PINNED_WEIGHT

                # Phase 19: arousal intensity (small tiebreaker)
                try:
                    score += arousal_rank_bonus(row["arousal_score"] if row.get("arousal_score") is not None else None)
                except Exception:
                    pass

                # Phase 21: encoding-specificity / context-match recall.
                # Mood-congruent boost when the query valence sign matches the
                # stored valence sign (both non-neutral), plus a smaller
                # arousal-intensity match.
                if (
                    MEMORY_RECALL_CONTEXT_MATCH_ENABLED
                    and MEMORY_RECALL_CONTEXT_MATCH_WEIGHT > 0
                    and (q_valence or q_arousal)
                ):
                    try:
                        mem_valence = row.get("valence_score")
                        mem_valence = int(mem_valence) if mem_valence is not None else None
                        if q_valence and mem_valence is not None and mem_valence != 0:
                            if (q_valence > 0) == (mem_valence > 0):
                                score += MEMORY_RECALL_CONTEXT_MATCH_WEIGHT
                        mem_arousal = row.get("arousal_score")
                        mem_arousal = int(mem_arousal) if mem_arousal is not None else None
                        if q_arousal and mem_arousal is not None and mem_arousal != 0:
                            if (abs(q_arousal) >= 1) == (abs(mem_arousal) >= 1):
                                score += MEMORY_RECALL_CONTEXT_MATCH_WEIGHT * 0.5
                    except Exception:
                        pass

                # Phase 3: entity importance boost
                if MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT > 0 and entity_importance_map:
                    try:
                        from cognition.memory.entity import memory_max_entity_importance
                        score += MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT * memory_max_entity_importance(
                            row, entity_importance_map
                        )
                    except Exception as exc:
                        log.debug("entity importance boost skipped: %s", exc)

                # Phase 17: align with this chat's query topic mean
                if _session_mean is not None:
                    try:
                        score += session_boost_for(mem_id, _session_mean, _session_vecs)
                    except Exception:
                        pass

            return score

        scored_ids = sorted(deduped_ids, key=final_score, reverse=True)
        scores = {mid: final_score(mid) for mid in scored_ids}
        return scored_ids, scores, row_by_id

    def _apply_recency_rerank(
        self,
        scored_ids: list[str],
        scores: dict,
        row_by_id: dict,
    ) -> list[str]:
        """
        Stage 3 — recency-among-relevant reorder (see module docstring).

        Candidates whose score clears MEMORY_RECENCY_RERANK_THRESHOLD are
        pulled to the front, sorted by created_at descending among
        themselves (most recent first). Candidates below the threshold keep
        their original score-descending relative order and follow behind.

        This is a genuine reorder, not another additive weight: two
        similarly-relevant memories can swap places here even if their RRF
        scores differ, as long as both clear the bar.
        """
        if not MEMORY_RECENCY_RERANK_ENABLED or not scored_ids:
            return scored_ids

        relevant = [mid for mid in scored_ids if scores.get(mid, 0.0) >= MEMORY_RECENCY_RERANK_THRESHOLD]
        if not relevant:
            return scored_ids

        relevant_sorted = sorted(
            relevant,
            key=lambda mid: row_by_id[mid]["created_at"] if mid in row_by_id else "",
            reverse=True,
        )
        relevant_set = set(relevant)
        rest = [mid for mid in scored_ids if mid not in relevant_set]
        return relevant_sorted + rest

    # ── entity-importance cache ──────────────────────────────────────────────
    # compute_entity_importance_map() full-scans all active memories plus
    # entity_relations (entity.py) — a couple of table scans on every cache-miss
    # recall just to feed a 0.008-weight boost. A short per-user TTL makes it
    # near-free. Any write invalidates the touched user's entry immediately.

    def _get_entity_importance_map(self, user_id: str) -> dict[str, float]:
        now = time.monotonic()
        with self._entity_importance_cache_lock:
            cached = self._entity_importance_cache.get(user_id)
            if cached is not None and now - cached[0] <= ENTITY_IMPORTANCE_CACHE_TTL:
                return cached[1]
        try:
            from cognition.memory.entity import compute_entity_importance_map
            val = compute_entity_importance_map(self, user_id) or {}
        except Exception as exc:
            # Transient failure: do NOT cache an empty map for the full TTL —
            # that would silently kill the (small) importance boost for the
            # whole window. Return empty now; the next recall retries.
            log.debug("entity importance map skipped: %s", exc)
            return {}
        with self._entity_importance_cache_lock:
            self._entity_importance_cache[user_id] = (now, val)
            if len(self._entity_importance_cache) > 64:
                # drop the oldest entry under the lock (simple eviction)
                oldest = min(self._entity_importance_cache, key=lambda k: self._entity_importance_cache[k][0])
                self._entity_importance_cache.pop(oldest, None)
        return val

    def _invalidate_entity_importance(self, user_id: str) -> None:
        with self._entity_importance_cache_lock:
            self._entity_importance_cache.pop(user_id, None)

    def search(self, query: str, user_id: str, limit: int = 5, vector: list[float] | None = None, include_history: bool = False) -> list[dict]:
        """
        KNN + FTS5 + entity-graph -> RRF fusion search, with a tiered
        quick/wide candidate pass and recency-among-relevant reranking.
        Pinned status and graph connectivity are both only mild RRF-style
        score tiebreakers (no reserved slots, no post-hoc override).
        See module docstring for the full stage-by-stage description.

        1. Embed the query once (_embed) — this is the dominant cost
            regardless of which pass runs below, so it is never repeated.
        2. Extract entities from the query text (same rule-based extractor
            used at write time) for the graph pass.
        3. Quick pass: pull QUICK_KNN_LIMIT / QUICK_FTS_LIMIT / QUICK_GRAPH_LIMIT
            candidates, dedup + score them. If that already fills `limit`
            results and the weakest of them clears MEMORY_RECALL_SCORE_THRESHOLD,
            use it as-is — most turns stop here and never pay for the wider
            SQL scan.
        4. Otherwise widen to KNN_LIMIT / FTS_LIMIT / GRAPH_LIMIT and re-rank
            the larger pool from scratch (rank positions shift when the pool
            grows, so this is a fresh scoring pass, not a merge with the quick
            pass).
        5. Reorder the resulting candidates by recency-among-relevant.
        6. Truncate to `limit` and return as payload dicts.

        vector — pre-computed query embedding; skips the _embed HTTP call.
        include_history — when False (default), superseded memories are excluded.

        FIX 2: the entire DB-touching portion of this method (all three
        candidate passes and the scoring step) now runs under a single
        `self._db_lock` acquisition. Previously only the first quick KNN
        call was locked, leaving the FTS pass, the scoring pass (which
        reads the full `memories` rows), and the wide-pass fallback racing
        against the async write-worker thread on the same connection.
        """
        with _brain_trace.step("_MemoryBackend.search", layer="recall",
                               inputs={"query": query, "limit": limit,
                                       "vector_supplied": vector is not None,
                                       "include_history": include_history,
                                       "user_id": user_id}) as ctx:
            return self._search_inner(query, user_id, limit, vector, include_history, ctx)
        # Phase 17: record query embedding for session dynamic anchor
        try:
            from cognition.memory.entity import push_query_vector
            push_query_vector(user_id, vector)
        except Exception:
            pass
        fts_query = _sanitize_fts_query(query)
        query_entities = extract_entities(query, max_entities=5)
        active_only = not include_history

        # Phase 21: encoding-specificity context. Cheap heuristic valence/
        # arousal of the query itself, matched against stored memory tags.
        query_context: tuple[int, int] | None = None
        if MEMORY_RECALL_CONTEXT_MATCH_ENABLED and MEMORY_RECALL_CONTEXT_MATCH_WEIGHT > 0:
            try:
                q_val = int(infer_valence_score(query))
                q_aro = int(infer_arousal_score(query))
                if q_val or q_aro:
                    query_context = (q_val, q_aro)
            except Exception:
                query_context = None

        entity_importance_map = {}
        if MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT > 0:
            entity_importance_map = self._get_entity_importance_map(user_id) or {}

        # ── Quick pass: fetch candidate IDs only (short lock scope) ──
        with self._db_lock:
            self._ensure_open()
            quick_knn_rows = _sqlite_knn_search(
                self._conn, vector, user_id, QUICK_KNN_LIMIT, active_only=active_only
            )
            rank_knn_q = {row["id"]: i + 1 for i, row in enumerate(quick_knn_rows)}
            quick_fts_rows = self._fts_pass(fts_query, user_id, QUICK_FTS_LIMIT, active_only=active_only)
            rank_fts_q = {row["id"]: i + 1 for i, row in enumerate(quick_fts_rows)}
            quick_graph_rows = self._graph_pass(query_entities, user_id, QUICK_GRAPH_LIMIT, active_only=active_only)
            rank_graph_q = {row["id"]: i + 1 for i, row in enumerate(quick_graph_rows)}

        # Fetch full rows for quick pass candidates (separate lock scope)
        quick_all_ids = set(rank_knn_q) | set(rank_fts_q) | set(rank_graph_q)
        with self._db_lock:
            quick_row_by_id = self._fetch_full_rows(quick_all_ids)

        # Score quick pass (no lock needed)
        scored_ids, scores, row_by_id = self._rank_and_score(
            rank_knn_q, rank_fts_q, rank_graph_q,
            entity_importance_map=entity_importance_map,
            query_context=query_context,
            row_by_id=quick_row_by_id,
        )

        confident = (
            len(scored_ids) >= limit
            and scores.get(scored_ids[limit - 1], 0.0) >= MEMORY_RECALL_SCORE_THRESHOLD
        )

        # ── Widen only if the quick pass was under-filled or under-confident ──
        if not confident:
            with self._db_lock:
                wide_knn_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, KNN_LIMIT, active_only=active_only
                )
                rank_knn_w = {row["id"]: i + 1 for i, row in enumerate(wide_knn_rows)}
                wide_fts_rows = self._fts_pass(fts_query, user_id, FTS_LIMIT, active_only=active_only)
                rank_fts_w = {row["id"]: i + 1 for i, row in enumerate(wide_fts_rows)}
                wide_graph_rows = self._graph_pass(query_entities, user_id, GRAPH_LIMIT, active_only=active_only)
                rank_graph_w = {row["id"]: i + 1 for i, row in enumerate(wide_graph_rows)}

            wide_all_ids = set(rank_knn_w) | set(rank_fts_w) | set(rank_graph_w)
            with self._db_lock:
                wide_row_by_id = self._fetch_full_rows(wide_all_ids)

            scored_ids, scores, row_by_id = self._rank_and_score(
                rank_knn_w, rank_fts_w, rank_graph_w,
                entity_importance_map=entity_importance_map,
                query_context=query_context,
                row_by_id=wide_row_by_id,
            )

        ordered_ids = self._apply_recency_rerank(scored_ids, scores, row_by_id)
        top_ids = ordered_ids[:limit]

        results = []
        for mid in top_ids:
            if mid not in row_by_id:
                continue
            d = dict(row_by_id[mid])
            d["_recall_score"] = scores.get(mid, 0.0)
            results.append(d)

        activation: dict[str, float] = {}
        if MEMORY_SPREADING_ENABLED and results:
            try:
                with self._db_lock:
                    exclude = {str(r.get("id")) for r in results}
                    activation, extra_ids = self._spreading_extra_ids(
                        user_id, results, exclude_ids=exclude, query=query,
                    )
                    if extra_ids:
                        placeholders = ",".join("?" * len(extra_ids))
                        rows = self._conn.execute(
                            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                            list(extra_ids) + [user_id],
                        ).fetchall()
                        for row in rows:
                            d = dict(row)
                            d["_recall_score"] = 0.0
                            d["_from_spreading"] = True
                            results.append(d)
            except Exception as exc:
                log.debug("spreading activation skipped: %s", exc)

        if MEMORY_SPREADING_SCORE_WEIGHT > 0 and activation and results:
            try:
                from cognition.memory.entity import memory_max_activation
                for r in results:
                    boost = MEMORY_SPREADING_SCORE_WEIGHT * memory_max_activation(r, activation)
                    r["_recall_score"] = float(r.get("_recall_score") or 0.0) + boost
                results.sort(key=lambda x: float(x.get("_recall_score") or 0.0), reverse=True)
            except Exception as exc:
                log.debug("spreading activation score boost skipped: %s", exc)

        if MEMORY_NEG_RECALL_AVOID and results:
            relax = MEMORY_NEG_RECALL_AVOID_EXCEPT and query_wants_emotion(query or "")
            if not relax:
                from cognition.memory.forget import negative_recall_penalty
                for r in results:
                    if r.get("pinned"):
                        continue
                    pen = negative_recall_penalty(
                        valence_tag=r.get("valence_tag"),
                        valence_score=r.get("valence_score"),
                    )
                    if pen > 0:
                        r["_recall_score"] = float(r.get("_recall_score") or 0.0) - pen
                results.sort(key=lambda x: float(x.get("_recall_score") or x.get("score") or 0.0), reverse=True)

        # Phase 19: sticky-neg not volunteered unless query engages them
        results = apply_neg_hard_filter(results, query)
        return results[:limit]

    def _expand_supersession_chains(self, query: str, user_id: str, results: list[dict], limit: int = 5) -> list[dict]:
        """Expand supersession chains — older → newer for hits that need lineage."""
        if not results:
            return results
        expanded: list[dict] = []
        seen: set[str] = set()
        with self._db_lock:
            for hit in results:
                mid = str(hit.get("id") or "")
                if not mid or mid in seen:
                    continue
                if not should_expand_supersession_chain(query, hit):
                    expanded.append(hit)
                    seen.add(mid)
                    continue
                chain = walk_supersession_chain(self._conn, mid, user_id)
                if len(chain) <= 1:
                    expanded.append(hit)
                    seen.add(mid)
                    continue
                chain_rows = [dict(n) for n in chain]
                for node in chain_rows:
                    nid = str(node.get("id") or "")
                    if not nid or nid in seen:
                        continue
                    node = dict(node)
                    node["_recall_score"] = hit.get("_recall_score", 0.0)
                    node["_supersession_chain"] = chain_rows
                    expanded.append(node)
                    seen.add(nid)
        return expanded[: max(limit, min(len(expanded), limit * 2))]

    def _search_inner(self, query, user_id, limit, vector, include_history, ctx):
        """Heavy lift of search(), split out so brain_trace.step can wrap it."""
        if vector is None:
            vector = self._embed(query, query=True)
            ctx.add_extra(vector_embedded_here=True, dims=len(vector))
        else:
            ctx.add_extra(vector_embedded_here=False, dims=len(vector))
        # Phase 17: record query embedding for session dynamic anchor
        try:
            from cognition.memory.entity import push_query_vector
            push_query_vector(user_id, vector)
        except Exception:
            pass
        fts_query = _sanitize_fts_query(query)
        query_entities = extract_entities(query, max_entities=5)
        active_only = not include_history

        # Phase 21: encoding-specificity context. Cheap heuristic valence/
        # arousal of the query itself, matched against stored memory tags.
        query_context: tuple[int, int] | None = None
        if MEMORY_RECALL_CONTEXT_MATCH_ENABLED and MEMORY_RECALL_CONTEXT_MATCH_WEIGHT > 0:
            try:
                q_val = int(infer_valence_score(query))
                q_aro = int(infer_arousal_score(query))
                if q_val or q_aro:
                    query_context = (q_val, q_aro)
            except Exception:
                query_context = None

        entity_importance_map = {}
        if MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT > 0:
            entity_importance_map = self._get_entity_importance_map(user_id) or {}

        # ── Quick pass: fetch candidate IDs only (short lock scope) ──
        with self._db_lock:
            self._ensure_open()
            quick_knn_rows = _sqlite_knn_search(
                self._conn, vector, user_id, QUICK_KNN_LIMIT, active_only=active_only
            )
            rank_knn_q = {row["id"]: i + 1 for i, row in enumerate(quick_knn_rows)}
            quick_fts_rows = self._fts_pass(fts_query, user_id, QUICK_FTS_LIMIT, active_only=active_only)
            rank_fts_q = {row["id"]: i + 1 for i, row in enumerate(quick_fts_rows)}
            quick_graph_rows = self._graph_pass(query_entities, user_id, QUICK_GRAPH_LIMIT, active_only=active_only)
            rank_graph_q = {row["id"]: i + 1 for i, row in enumerate(quick_graph_rows)}

        # Fetch full rows for quick pass candidates (separate lock scope)
        quick_all_ids = set(rank_knn_q) | set(rank_fts_q) | set(rank_graph_q)
        with self._db_lock:
            quick_row_by_id = self._fetch_full_rows(quick_all_ids)

        # Score quick pass (no lock needed)
        scored_ids, scores, row_by_id = self._rank_and_score(
            rank_knn_q, rank_fts_q, rank_graph_q,
            entity_importance_map=entity_importance_map,
            query_context=query_context,
            row_by_id=quick_row_by_id,
        )

        confident = (
            len(scored_ids) >= limit
            and scores.get(scored_ids[limit - 1], 0.0) >= MEMORY_RECALL_SCORE_THRESHOLD
        )
        ctx.add_extra(
            quick_pass={"knn": len(rank_knn_q), "fts": len(rank_fts_q),
                        "graph": len(rank_graph_q),
                        "scored": len(scored_ids), "confident": confident}
        )

        # ── Widen only if the quick pass was under-filled or under-confident ──
        if not confident:
            with self._db_lock:
                wide_knn_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, KNN_LIMIT, active_only=active_only
                )
                rank_knn_w = {row["id"]: i + 1 for i, row in enumerate(wide_knn_rows)}
                wide_fts_rows = self._fts_pass(fts_query, user_id, FTS_LIMIT, active_only=active_only)
                rank_fts_w = {row["id"]: i + 1 for i, row in enumerate(wide_fts_rows)}
                wide_graph_rows = self._graph_pass(query_entities, user_id, GRAPH_LIMIT, active_only=active_only)
                rank_graph_w = {row["id"]: i + 1 for i, row in enumerate(wide_graph_rows)}

            wide_all_ids = set(rank_knn_w) | set(rank_fts_w) | set(rank_graph_w)
            with self._db_lock:
                wide_row_by_id = self._fetch_full_rows(wide_all_ids)

            scored_ids, scores, row_by_id = self._rank_and_score(
                rank_knn_w, rank_fts_w, rank_graph_w,
                entity_importance_map=entity_importance_map,
                query_context=query_context,
                row_by_id=wide_row_by_id,
            )
            ctx.add_extra(wide_pass={"knn": len(rank_knn_w), "fts": len(rank_fts_w),
                                    "graph": len(rank_graph_w), "scored": len(scored_ids)})

        ordered_ids = self._apply_recency_rerank(scored_ids, scores, row_by_id)
        top_ids = ordered_ids[:limit]

        results = []
        for mid in top_ids:
            if mid not in row_by_id:
                continue
            d = dict(row_by_id[mid])
            d["_recall_score"] = scores.get(mid, 0.0)
            results.append(d)

        activation: dict[str, float] = {}
        if MEMORY_SPREADING_ENABLED and results:
            try:
                with self._db_lock:
                    exclude = {str(r.get("id")) for r in results}
                    activation, extra_ids = self._spreading_extra_ids(
                        user_id, results, exclude_ids=exclude, query=query,
                    )
                    if extra_ids:
                        placeholders = ",".join("?" * len(extra_ids))
                        rows = self._conn.execute(
                            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                            list(extra_ids) + [user_id],
                        ).fetchall()
                        for row in rows:
                            d = dict(row)
                            d["_recall_score"] = 0.0
                            d["_from_spreading"] = True
                            results.append(d)
            except Exception as exc:
                log.debug("spreading activation skipped: %s", exc)

        if MEMORY_SPREADING_SCORE_WEIGHT > 0 and activation and results:
            try:
                from cognition.memory.entity import memory_max_activation
                for r in results:
                    boost = MEMORY_SPREADING_SCORE_WEIGHT * memory_max_activation(r, activation)
                    r["_recall_score"] = float(r.get("_recall_score") or 0.0) + boost
                results.sort(key=lambda x: float(x.get("_recall_score") or 0.0), reverse=True)
            except Exception as exc:
                log.debug("spreading activation score boost skipped: %s", exc)

        if MEMORY_NEG_RECALL_AVOID and results:
            relax = MEMORY_NEG_RECALL_AVOID_EXCEPT and query_wants_emotion(query or "")
            if not relax:
                from cognition.memory.forget import negative_recall_penalty
                for r in results:
                    if r.get("pinned"):
                        continue
                    pen = negative_recall_penalty(
                        valence_tag=r.get("valence_tag"),
                        valence_score=r.get("valence_score"),
                    )
                    if pen > 0:
                        r["_recall_score"] = float(r.get("_recall_score") or 0.0) - pen
                results.sort(key=lambda x: float(x.get("_recall_score") or x.get("score") or 0.0), reverse=True)

        # Phase 19: sticky-neg not volunteered unless query engages them
        results = apply_neg_hard_filter(results, query)
        results = results[:limit]

        ctx.set(
            outputs={
                "returned": len(results),
                "top_score": round(float(results[0]["_recall_score"]), 4) if results else None,
                "top_hit_preview": (results[0].get("memory") or "")[:200] if results else None,
            },
            factors=[
                f"KNN+FTS+entity-graph fused via RRF (k={RRF_K})",
                f"recency rerank applied: {MEMORY_RECENCY_RERANK_ENABLED}",
                f"super-node entities filtered: {len(self._high_freq_entities)}",
                f"neg-recall-avoid: {MEMORY_NEG_RECALL_AVOID}",
                f"spreading: {MEMORY_SPREADING_ENABLED}",
            ],
        )
        return results

    def _search_top(self, query, user_id, limit, query_vector, include_history, ctx):
        """Heavy lift of AikoMemorize.search — extracted so the brain tracer
        can wrap the whole call without forcing a return through `with`."""
        user_id = self._resolve_user_id(user_id)
        if _is_trivial_input(query or ""):
            ctx.set(outputs={"skipped": True, "reason": "trivial_input"},
                    factors=["input matches trivial pattern (hi/ok/thanks/...) — no KNN/FTS cost"])
            log.debug(f"Skipping search for trivial input: {query!r}")
            return []

        if _BROAD_RECALL_RE.search(query or ""):
            results = self._recent_or_important_memories(
                user_id=user_id, limit=limit, include_history=include_history
            )
            results = self._expand_scenes(user_id, results[:int(limit)])
            self._touch_memories(results)
            ctx.set(outputs={"short_circuit": "broad_recall", "returned": len(results)},
                    factors=[f"query matches _BROAD_RECALL_RE → recent_or_important path (no KNN)"])
            return results

        cache_key = (user_id, " ".join((query or "").lower().split()), int(limit), bool(include_history))
        now_s = time.monotonic()

        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and now_s - cached[0] <= MEMORY_SEARCH_CACHE_TTL:
                self._search_cache.move_to_end(cache_key)
                results = [dict(r) for r in cached[1]]
                try:
                    from cognition.memory.entity import MEMORY_SUPERSESSION_CHAIN_EXPAND
                    if MEMORY_SUPERSESSION_CHAIN_EXPAND and results:
                        results = self._mem._expand_supersession_chains(
                            query, user_id, results, limit=limit
                        )
                except Exception as exc:
                    log.debug("supersession chain expand skipped: %s", exc)
                self._touch_memories(results)
                ctx.set(outputs={"cache": "hit", "returned": len(results)},
                        factors=[f"cache TTL {MEMORY_SEARCH_CACHE_TTL}s"])
                return results
            if cached:
                self._search_cache.pop(cache_key, None)

        # Run the core RRF search (KNN + FTS + entity graph, fused)
        results = self._mem.search(
            query,
            user_id=user_id,
            limit=limit,
            vector=query_vector,
            include_history=include_history,
        )
        # Fold L2 scene parents/members into the recall set (see _expand_scenes).
        results = self._expand_scenes(user_id, results)
        log.debug("[memory] search miss, scores=%s", [r.get("_recall_score") for r in results])

        # Search replay logging (optional, feature-gated)
        if os.getenv("AIKO_REPLAY_SEARCHES"):
            self._write_search_replay(query, results, user_id)

        # Cache and return
        with self._search_cache_lock:
            self._search_cache[cache_key] = (now_s, [dict(r) for r in results])
            while len(self._search_cache) > MEMORY_SEARCH_CACHE_SIZE:
                self._search_cache.popitem(last=False)

        try:
            from cognition.memory.entity import MEMORY_SUPERSESSION_CHAIN_EXPAND
            if MEMORY_SUPERSESSION_CHAIN_EXPAND and results:
                results = self._mem._expand_supersession_chains(
                    query, user_id, results, limit=limit
                )
        except Exception as exc:
            log.debug("supersession chain expand skipped: %s", exc)

        self._touch_memories(results)
        ctx.set(outputs={"cache": "miss", "returned": len(results)},
                factors=[f"scene_expansion applied", "touch_memories incremented access_count"])
        return results

    # ── L2 scene expansion ─────────────────────────────────────────────────────
    # After RRF returns a set, re-link episode structure so yes the scene row
    # itself is searchable, but also: a recalled-member pulls in its parent
    # scene, and a recalled scene carries its members. Keeps "what happened
    # during X" recoverable without rearchitecting search scoring.

    def iter_all(self, user_id: str, batch_size: int = MEMORY_LIFECYCLE_BATCH_SIZE):
        """Yield memory records for a user in rowid order without one giant list.

        FIX: each page fetch is now taken under self._db_lock. The lock is
        NOT held across the yield, so a slow consumer (e.g. dream()/cleanup()
        processing a batch) doesn't block the write-worker thread for the
        whole duration — only the actual SQL scan is protected.
        """
        last_rowid = 0
        while True:
            with self._db_lock:
                rows = self._conn.execute(
                    """
                    SELECT rowid, id, memory, created_at, status,
                           valence_tag, valence_score, salience_hit
                    FROM memories
                    WHERE user_id = ? AND rowid > ?
                    ORDER BY rowid ASC
                    LIMIT ?
                    """,
                    (user_id, last_rowid, batch_size),
                ).fetchall()
            if not rows:
                break
            for row in rows:
                last_rowid = row["rowid"]
                yield {
                    "id": row["id"],
                    "memory": row["memory"],
                    "created_at": row["created_at"],
                    "status": row["status"],
                    "valence_tag": row["valence_tag"],
                    "valence_score": row["valence_score"],
                    "salience_hit": row["salience_hit"],
                }

    def _scene_cols_available(self) -> bool:
        try:
            cols = existing_columns(self._conn)
        except Exception:
            return False
        return "scene_id" in cols and "kind" in cols

    def get_all(self, user_id: str) -> list[dict]:
        """Return all memories for a user as a list."""
        return list(self.iter_all(user_id=user_id))

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        """Return all memories created after the given datetime."""
        user_id = user_id or self._user_id
        with self._db_lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (user_id, since.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None, limit: int = 0) -> list[dict]:
        """Return memories created within a datetime range, optionally limited."""
        user_id = user_id or self._user_id
        sql = """
            SELECT * FROM memories
            WHERE user_id = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
        """
        params: list[Any] = [user_id, start.isoformat(), end.isoformat()]
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        with self._db_lock:
            self._ensure_open()
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
      
    def get_by_id(self, mem_id: str, user_id: str | None = None) -> dict | None:
        """Return a single memory by ID, or None if not found."""
        uid = user_id  # require user_id from caller for safety
        with self._db_lock:
            row = self._conn.execute(
                """
                SELECT id, memory, created_at, status, supersedes_id, pinned,
                       kind, source, valence_score, arousal_score, entities, access_count
                FROM memories
                WHERE id = ? AND (? IS NULL OR user_id = ?)
                """,
                (mem_id, uid, uid),
            ).fetchone()
        return dict(row) if row else None

    def find_by_supersedes(self, mem_id: str, user_id: str | None = None) -> list[dict]:
        """Return all memories that supersede the given memory ID."""
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT id, memory, created_at, status, supersedes_id, pinned,
                       kind, source, valence_score, arousal_score, entities, access_count
                FROM memories
                WHERE supersedes_id = ? AND (? IS NULL OR user_id = ?)
                ORDER BY created_at ASC
                LIMIT 16
                """,
                (mem_id, user_id, user_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        """
        Delete one memory and its vector. Also cascades to entity_relations —
        without this, a deleted memory's co-mention edges stay behind
        forever (dream()'s merge-loser deletes and cleanup()'s decay
        deletes both route through this method), so _graph_pass would keep
        joining to memory ids that no longer exist, silently degrading
        results over time instead of erroring.
        """
        user_id = None
        with self._db_lock:
            self._ensure_open()
            user_id = self._conn.execute(
                "SELECT user_id FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM entity_relations WHERE memory_id = ?", (memory_id,))
            self._conn.commit()
        if user_id:
            self._invalidate_entity_importance(user_id["user_id"])

    def delete_all(self, user_id: str) -> None:
        """Delete all memories and vectors for a user."""
        with self._db_lock:
            self._conn.execute(
                "DELETE FROM memories_vec WHERE id IN (SELECT id FROM memories WHERE user_id = ?)",
                (user_id,),
            )
            self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM entity_relations WHERE user_id = ?", (user_id,))
            self._conn.commit()
        self._invalidate_entity_importance(user_id)


class AikoMemorize:
    """
    Persistent memory with Ebbinghaus decay lifecycle and nightly dream() pass.

    Boot sequence (called by wakeup.py in order):
        memorize = AikoMemorize()
        memorize.cleanup()

    Access tracking:
        Every search() call updates the memories table (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Pinned memories:
        Created via pin() — the pinned=1 column flag makes them
        immune to cleanup(), dream prune, and dream merge (as the loser).
        At recall they compete on the same blended score as everything
        else, with only MEMORY_RANK_PINNED_WEIGHT as a mild tiebreaker
        (the old stage-4 reserved-slot path was removed).
        Recall-time dedup (in _MemoryBackend.search) still collapses
        multiple pinned rows with identical text down to the most recent
        one, since dream() structurally cannot do this for pinned rows.

    Entity graph:
        entity_relations co-mention edges (written in _insert_row via
        upsert_co_mentions) are now read at recall time by
        _MemoryBackend._graph_pass() and fused into the same RRF-style
        scoring as KNN/FTS. This class no longer does its own separate
        entity extraction / rerank pass after the fact — that logic
        (previously _extract_query_entities / _boost_by_entity_relations,
        gated by AIKO_ENTITY_BOOST) has moved into _MemoryBackend, where
        it can influence which candidates enter the pool, not just their
        order within it.

    Async write queue:
        queue_write() lets a caller enqueue a fire-and-forget memory write
        (LLM-based fact extraction + persist) that runs on a dedicated
        background thread, without blocking the caller's turn. The caller
        expresses when it's safe to run via two callables (is_active_turn,
        idle_since) rather than this class inspecting the caller's state
        directly — see queue_write() below.

    Dream pass (call nightly at 00:00):
        1. Boost salient memories' access_count so they survive decay.
        2. Merge near-duplicate vectors — keeps higher-access copy.
        3. Prune decayed memories via cleanup().
    """

    def __init__(self, silent: bool = False) -> None:
        self._user_id_override = None
        self._silent = silent
        self._display_name: str | None = None
        self._search_cache: OrderedDict[tuple[str, str, int, bool], tuple[float, list[dict]]] = OrderedDict()
        self._search_cache_lock = threading.RLock()
        self._llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
        self._model = os.getenv("EXTRACT_MODEL") or os.getenv("LLM_MODEL", "ministral")
        self._embed_cache = os.getenv("EMBED_CACHE_PATH") or os.getenv("FASTEMBED_CACHE_PATH")

        # One shared embedder for the process lifetime. It's a cheap HTTP
        # client (the model lives in llama-server) and holds the disk-cache
        # contents — rebuilding it on every switch_user would re-read the
        # whole EMBED_CACHE_PATH JSONL for nothing.
        self._embedder = HarrierEmbedder(cache_path=self._embed_cache)

        # Pre-auth boot opens NO sqlite connection at all: guest has no
        # per-user store, so constructing a backend would just build schema
        # on a throwaway tempfile DB. The backend is materialized lazily on
        # first touch (see the _mem property) or explicitly by switch_user()
        # when a real identity connects.
        uid = current_user_id()
        if not silent:
            log.info("Memory identity at boot: %s%s", uid, " (lazy — no DB until login)" if uid == "guest" else "")
        if uid == "guest":
            self._mem_backend: _MemoryBackend | None = None
        else:
            self._mem_backend = _MemoryBackend(
                db_path=_memory_db_path_for_user(uid),
                llm_base_url=self._llm_base_url,
                model=self._model,
                user_id=uid,
                embedder=self._embedder,
            )
        self._write_queue: "queue.Queue[tuple]" = queue.Queue()
        self._user_switch_lock = threading.RLock()
        self._write_worker = threading.Thread(target=self._write_loop, daemon=True)
        self._write_worker.start()
        self._last_cache_clear_time: float = 0.0
        # L3 persona cache — always-hydrated stable identity blob, TTL-cached so
        # the cheap SQL only reruns occasionally even though it's injected every
        # turn. Invalidated on any write (via _clear_search_cache).
        self._persona_lock = threading.RLock()
        self._persona_cache_at: float = 0.0
        self._persona_cached: str | None = None
        # Episodic-memory (EMC) facade. Owns the per-user EpisodicStore cache
        # and exposes queue_episode / episodic-recall helpers as proper methods
        # on this class (previously monkey-patched on at boot).
        try:
            from cognition.memory.episode import EpisodicMemory
            self.episodic: EpisodicMemory | None = EpisodicMemory(self)
        except Exception as e:
            log.debug("EpisodicMemory init failed: %s", e)
            self.episodic = None
        if not silent:
            log.info("Ready.")

    @property
    def _mem(self) -> _MemoryBackend:
        """Lazily-materialized memory backend.

        All 150+ internal references to self._mem keep working unchanged:
        touching it as a real user opens (or returns) that user's store.
        Touching it while still guest materializes the tempfile-backed
        guest DB — exactly the old pre-lazy behaviour — so nothing can
        crash before login; normal flows simply never touch it until
        switch_user() binds a real identity.
        """
        if self._mem_backend is None:
            self._open()
        assert self._mem_backend is not None  # _open always assigns
        return self._mem_backend

    @_mem.setter
    def _mem(self, backend: _MemoryBackend | None) -> None:
        self._mem_backend = backend

    def embedder(self) -> HarrierEmbedder:
        """Shared embedder instance — safe to use before any DB is open."""
        return self._embedder

    def is_open(self) -> bool:
        """True once a sqlite backend exists (never true for untouched guest)."""
        return self._mem_backend is not None

    def _open(self, uid: str | None = None) -> None:
        """Open (or reopen) the sqlite-vec store for a given user_id."""
        uid = uid or self._user_id_override or current_user_id()
        db_path = _memory_db_path_for_user(uid)
        if not self._silent:
            log.info("Opening sqlite-vec memory store for %s ...", uid)
        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=self._llm_base_url,
            model=self._model,
            user_id=uid,
            embedder=self._embedder,
        )
        if not self._silent:
            log.info("Memory store ready for %s.", uid)

    @property
    def _db_lock(self):  # type: ignore[override]
        """Expose backend lock so 'AikoMemorize has no attribute _db_lock' never fires."""
        if self._mem_backend is not None:
            return self._mem_backend._db_lock
        return threading.RLock()

    def _search_top(self, *args, **kwargs):  # type: ignore[override]
        """Proxy so direct AikoMemorize._search_top calls don't crash (owner is _MemoryBackend)."""
        return self._mem._search_top(*args, **kwargs)

    @property
    def _conn(self) -> sqlite3.Connection:
        """Single reach-through point into the backend's connection.

        Was previously copied into self._conn separately in __init__ and
        _open(); collapsing both into one property means a future change
        to _MemoryBackend's connection ownership (e.g. a pool, or a
        different sqlite-vec binding) only needs updating here, not at
        every call site that currently reads self._conn directly.
        """
        # Touch the lazy materializer so callers that grab _conn before
        # _open() has been called (e.g. early boot, scheduled jobs that
        # ran without an authenticated session) get a real backend, not
        # an AttributeError on _mem.
        _ = self._mem
        return self._mem._conn

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Thin wrapper for one-off statements against the backend's
        connection. Prefer this for new code over reaching into
        self._conn directly.

        Does NOT take the lock itself — callers doing a read-modify-write
        sequence (or anything that must be atomic with respect to the
        async write-worker thread) must still wrap the call(s) in
        `with self._mem._db_lock:`, same as existing direct self._conn use
        elsewhere in this class."""
        return self._conn.execute(sql, params)

    def _commit(self) -> None:
        """Thin wrapper mirroring _exec — commit through the same seam.
        Same caveat: does not take self._mem._db_lock itself."""
        self._conn.commit()

    def switch_user(self, user_id: str) -> None:
        """Switch to a different user's memory store after draining pending writes."""
        with self._user_switch_lock:
            # Drain any pending writes first. If a write is still in flight when
            # we close the connection and reassign self._mem below, the
            # write-worker thread's in-progress self.add() call can end up
            # operating on a closed connection (or, worse, silently switch to
            # the *new* user's connection mid-write). Bail out rather than risk
            # cross-user data corruption; the caller can retry the switch once
            # the write actually finishes.
            # Use MEMORY_WRITE_MAX_WAIT + buffer since the write worker may wait
            # for an idle window up to that long before processing queued writes.
            switch_timeout = MEMORY_WRITE_MAX_WAIT + 10.0
            if not self.wait_for_writes(timeout=switch_timeout):
                log.error(
                    f"switch_user({user_id!r}): pending write(s) did not finish "
                    f"within {switch_timeout:.0f}s — aborting user switch to avoid writing to the "
                    "wrong connection. Try again shortly."
                )
                return

            # Flush + close any per-user episode stores so no in-flight
            # episode DB write bleeds into the new user's connection. The
            # facade is the single owner of that store cache now.
            if self.episodic is not None:
                self.episodic.close_all()

            self._user_id_override = user_id
            self._display_name = None
            if self._mem_backend is not None:
                # Read _mem_backend directly — going through the self._mem
                # property here would lazily materialize a backend just to
                # close it (defeating lazy guest boot).
                with self._mem._db_lock:
                    try:
                        self._conn.execute("PRAGMA optimize")
                        self._conn.commit()
                    except Exception:
                        log.warning("memorize: PRAGMA optimize failed")
                    try:
                        self._conn.close()
                    except Exception:
                        log.warning("memorize: closing old connection failed")
            self._open(user_id)

    def get_user_id(self) -> str:
        """Return the user_id this instance is currently opened for."""
        if self._user_id_override:
            return self._user_id_override
        if self._mem_backend is not None:
            return self._mem_backend._user_id
        return current_user_id()

    def set_display_name(self, name: str) -> None:
        """Set the display name for this user (e.g. GitHub login)."""
        stripped = name.strip() if name else None
        if stripped and stripped.casefold() == "aiko":
            raise ValueError("Display name cannot be 'Aiko' (reserved for assistant)")
        self._display_name = stripped

    def get_display_name(self) -> str:
        """Return the display name for this user, or fall back to user_id."""
        return self._display_name or self.get_user_id()

    # ── episodic memory (EMC) integration ─────────────────────────────────────
    # Episodic ingest and recall used to be monkey-patched onto AikoMemorize
    # at boot from a separate wire-up module. With the EpisodicMemory facade
    # they are proper methods: queue_episode is called from
    # AikoThink._store_async and _format_episodes_for_context is called from
    # format_for_context below.

    def queue_episode(
        self,
        user_input: str,
        response_text: str,
        cognitive_state: dict | None = None,
        user_id: str | None = None,
    ) -> None:
        """Stage one conversation turn into the episodic buffer (EMC-2)."""
        if self.episodic is not None:
            self.episodic.queue_episode(
                user_input, response_text,
                cognitive_state=cognitive_state, user_id=user_id,
            )

    def _format_episodes_for_context(self, query: str, query_vector=None) -> str | None:
        """Render the <episodic_context> block, or None if disabled / no hits."""
        if self.episodic is None:
            return None
        return self.episodic.format_for_context(query, query_vector)

    def _resolve_user_id(self, user_id: str | None = None) -> str:
        """Resolve the effective user_id for this call.

        An explicit argument always wins. Otherwise, falls back to THIS
        instance's own bound identity (get_user_id()) — never the ambient
        contextvar. An AikoMemorize instance is constructed for (or
        switch_user()'d to) a specific user; calls issued against it from
        another thread — the scheduler's daemon thread, the async write
        worker, a standalone script — must resolve against that bound
        identity, not whatever current_user_id() happens to return in
        the calling thread's own context (which is usually unset/"guest").
        This is the fix for the ambient-user-id bug class tracked across
        memory/ and sensory/.
        """
        return user_id or self.get_user_id()

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str | None = None, display_name: str | None = None) -> bool:
        """
        Store a conversation turn into long-term memory.
        Returns True on success, False on failure.
        """
        try:
            user_id = self._resolve_user_id(user_id)
            t       = time.perf_counter()
            ids     = self._mem.add(messages, user_id=user_id, display_name=display_name)
            elapsed = time.perf_counter() - t
            if ids:
                self._maybe_clear_search_cache()
                self._mem._invalidate_entity_importance(user_id)
                log.info(f"Saved {len(ids)} memories in {elapsed:.2f}s")
                if _brain_trace and _brain_trace.TRACE_ENABLED:
                    _brain_trace.record_step(
                        "semantic.add",
                        layer="write",
                        outputs={"memories_saved": len(ids),
                                 "elapsed_s": round(elapsed, 3)},
                        factors=[
                            f"LLM extracted {len(ids)} fact(s) → stored in memories table",
                            "search cache invalidated; entity importance map invalidated",
                        ],
                    )
            else:
                log.debug(f"No facts extracted ({elapsed:.2f}s) — nothing saved.")
                if _brain_trace and _brain_trace.TRACE_ENABLED:
                    _brain_trace.record_step(
                        "semantic.add",
                        layer="write",
                        outputs={"memories_saved": 0, "elapsed_s": round(elapsed, 3)},
                        factors=["no durable facts in this turn (greeting/no-op/dedup)"],
                    )
            return True
        except Exception as e:
            log.error(f"Save failed: {e}")
            if _brain_trace and _brain_trace.TRACE_ENABLED:
                _brain_trace.record_step(
                    "semantic.add",
                    layer="write",
                    outputs={"error": str(e)},
                )
            return False

    def pin(self, messages: list[dict], user_id: str | None = None, display_name: str | None = None) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.
        Pinned memories are immune to cleanup, dream pruning, and merge losses.
        Returns True on success, False on any failure.
        """
        try:
            user_id = self._resolve_user_id(user_id)
            ids = self._mem.add(messages, user_id=user_id, display_name=display_name)
            self._mem._invalidate_entity_importance(user_id)

            if not ids:
                # add() returns [] almost exclusively because every extracted
                # fact was a no-op/supersede against something already in the
                # store (see _maybe_supersede_neighbor) — not because nothing
                # relevant exists. Re-searching on the same text and pinning
                # the top hits recovers the (already-present) memory the
                # caller meant to pin, rather than silently failing. This is
                # not a substitute for write-time dedup skipping a near-
                # duplicate write; it's recovering the row that write-time
                # dedup correctly decided not to duplicate.
                query = "\n".join(
                    (m.get("content") or "").strip()
                    for m in messages
                    if (m.get("content") or "").strip()
                )
                ids = [
                    str(m.get("id"))
                    for m in self.search(query, user_id=user_id, limit=3)
                    if m.get("id")
                ]

            if not ids:
                # Two genuinely different situations land here, and we can't
                # tell them apart from this side: (a) add() no-op'd because
                # every fact was a dedup/supersede (the re-search above
                # should have recovered something, so reaching this branch
                # at all is the unusual case), or (b) fact extraction itself
                # failed/returned nothing, so there was never a row to find.
                log.warning(
                    "pin(): no memory IDs found to pin after add()+re-search. "
                    "Either extraction produced nothing to save, or the "
                    "re-search missed the deduped row — check upstream logs "
                    "for an extraction failure before assuming the latter."
                )
                return False

            for mem_id in ids:
                with self._mem._db_lock:
                    _sqlite_set_payload(self._conn, mem_id, {"pinned": 1})

            self._clear_search_cache()
            log.info(f"Pinned {len(ids)} memories: {ids}")
            return True
        except Exception as e:
            log.error(f"Pin failed: {e}")
            return False

    # ── async write queue ────────────────────────────────────────────────────

    def queue_write(
        self,
        user_input: str,
        response_text: str,
        *,
        user_id: str | None = None,
        display_name: str | None = None,
        is_active_turn=None,
        idle_since=None,
    ) -> None:
        """Queue an async memory write for a conversation turn.

        Runs on this instance's dedicated write-worker thread — the caller's
        turn is never blocked on LLM-based fact extraction. `is_active_turn`
        (callable[[], bool]) and `idle_since` (callable[[], float], a
        time.time()-style timestamp of the caller's last chat activity) let
        the write wait for an idle window before using the shared LLM,
        without this module needing to know how the caller tracks turn
        state. If either is omitted, the write runs as soon as it's
        dequeued with no idle wait.
        """
        with self._user_switch_lock:
            user_id = self._resolve_user_id(user_id)  # resolved here, on the caller's thread — not in _write_loop
            display_name = display_name or current_display_name()
            if _brain_trace and _brain_trace.TRACE_ENABLED:
                _brain_trace.record_step(
                    "semantic.queue_write",
                    layer="write",
                    inputs={"user_chars": len(user_input or ""),
                            "response_chars": len(response_text or ""),
                            "user_id": user_id,
                            "queue_depth": self._write_queue.qsize()},
                    factors=[
                        "semantic memory (durable facts) — written async on write-worker thread",
                        "will wait for idle window before LLM fact extraction (when is_active_turn given)",
                    ],
                )
            self._write_queue.put((user_input, response_text, user_id, display_name, is_active_turn, idle_since))

    def _write_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    self._write_queue.task_done()
                    return
                user_input, response_text, user_id, display_name, is_active_turn, idle_since = item
                self._wait_for_write_window(is_active_turn, idle_since)
                if _brain_trace and _brain_trace.TRACE_ENABLED:
                    _brain_trace.record_step(
                        "semantic.write_loop",
                        layer="write",
                        inputs={"user_id": user_id,
                                "user_chars": len(user_input or ""),
                                "response_chars": len(response_text or "")},
                        factors=[
                            "write worker dequeued item; calling _MemoryBackend.add() for LLM fact extraction",
                        ],
                    )
                self.add([
                    {"role": "user", "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ], user_id=user_id, display_name=display_name)
            except Exception as e:
                log.error(f"Async memory write failed: {e}")
                try:
                    from system.notice import get_notice_bus
                    get_notice_bus(user_id).push("memory", "background memory save failed — chat unaffected")
                except Exception:
                    pass
            finally:
                self._write_queue.task_done()
                # Periodic WAL checkpoint to prevent unbounded WAL growth
                try:
                    if self._conn:
                        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass

    def _wait_for_write_window(self, is_active_turn, idle_since) -> None:
        """Wait until the caller reports idle before running an extraction
        write on the shared LLM. No-ops immediately if the caller didn't
        supply idle-tracking callables."""
        if is_active_turn is None or idle_since is None:
            return
        deadline = time.monotonic() + max(0.0, MEMORY_WRITE_MAX_WAIT)
        while True:
            idle_for = time.time() - idle_since()
            if not is_active_turn() and idle_for >= MEMORY_WRITE_IDLE_GRACE:
                return
            # FIX: was `if MEMORY_WRITE_MAX_WAIT > 0 and time.monotonic() >= deadline`.
            # Gating the cap on MAX_WAIT > 0 meant setting MEMORY_WRITE_MAX_WAIT=0
            # to mean "don't wait at all" instead disabled the cap entirely —
            # the exact scenario it exists for (is_active_turn() stuck True
            # forever) would then spin indefinitely. `deadline` already bakes
            # in `max(0.0, MEMORY_WRITE_MAX_WAIT)`, so comparing against it
            # directly does the right thing for 0 (fires immediately) and
            # for any positive value, with no separate gate needed.
            if time.monotonic() >= deadline:
                return
            sleep_for = min(0.5, max(0.05, MEMORY_WRITE_IDLE_GRACE - idle_for))
            time.sleep(sleep_for)

    def wait_for_writes(self, timeout: float | None = None) -> bool:
        """Block until all queued async writes complete, or `timeout`
        elapses. Returns True if the queue drained, False on timeout."""
        if timeout is None:
            self._write_queue.join()
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        with self._write_queue.all_tasks_done:
            while self._write_queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._write_queue.all_tasks_done.wait(remaining)
        return True

    def close(self) -> None:
        """Stop the write-worker thread and close the sqlite connection."""
        try:
            if not self.wait_for_writes(timeout=5.0):
                log.warning("AikoMemorize.close(): pending writes did not finish within 5s")
        except Exception as e:
            log.debug("AikoMemorize.close(): wait_for_writes: %s", e)
        # Stop the write-worker thread if it's running
        if self._write_worker is not None and self._write_worker.is_alive():
            try:
                self._write_queue.put(None)
                self._write_worker.join(timeout=2.0)
            except Exception as e:
                log.debug("AikoMemorize.close(): write worker shutdown: %s", e)
        if self._mem_backend is not None:
            with self._mem._db_lock:
                try:
                    self._conn.close()
                except Exception:
                    pass

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str | None = None, limit: int = 5, query_vector: list[float] | None = None, include_history: bool = False) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.
        Side-effect: increments access_count and updates last_accessed_at
        for all returned memories in a single batched UPDATE.

        When MEMORY_SUPERSESSION_CHAIN_EXPAND is on, reflective queries
        (and hits with kind in MEMORY_SUPERSESSION_CHAIN_KINDS) may expand
        supersedes_id lineages. In that case the returned list can grow up
        to about 2×limit (oldest → newest along each expanded chain).
        Callers that need a hard ceiling should truncate themselves.

        Entity-graph fusion happens inside self._mem.search() now (see
        _MemoryBackend._graph_pass / _rank_and_score) — this method no
        longer does a separate post-hoc entity rerank pass.
        """
        with _brain_trace.step("AikoMemorize.search", layer="recall",
                               inputs={"query": query, "limit": limit,
                                       "user_id": user_id,
                                       "vector_supplied": query_vector is not None}) as ctx:
            return self._mem._search_top(query, user_id, limit, query_vector, include_history, ctx)

    # ── L2 scene expansion ─────────────────────────────────────────────────────
    # After RRF returns a set, re-link episode structure so yes the scene row
    # itself is searchable, but also: a recalled-member pulls in its parent
    # scene, and a recalled scene carries its members. Keeps "what happened
    # during X" recoverable without rearchitecting search scoring.

    # ── L2 scene blocks (backend) ─────────────────────────────────────────────
    # A scene is a normal memories row with kind=KIND_SCENE whose text is a
    # mid-grain episode summary. The atomic facts it was built from carry
    # scene_id back to it, so recall can surface the scene (via its own
    # vector match) and expand to its members, or re-link a recalled member to
    # its episode.

    def _scene_cols_available(self) -> bool:
        try:
            cols = existing_columns(self._conn)
        except Exception:
            return False
        return "scene_id" in cols and "kind" in cols

    def _expand_scenes(self, user_id: str, results: list[dict]) -> list[dict]:
        if not results or not self._scene_cols_available():
            return results
        out: list[dict] = []
        seen = {str(r.get("id")) for r in results if r.get("id")}
        scene_cache: dict[str, dict | None] = {}
        with self._mem._db_lock:
            for r in results:
                rid = str(r.get("id") or "")
                # A scene row itself: attach a compact member list.
                if r.get("kind") == KIND_SCENE:
                    entry = dict(r)
                    entry["_scene"] = True
                    members = self._mem.scene_members(rid, user_id, limit=SCENE_MEMBER_LIMIT)
                    if members:
                        entry["_scene_members"] = [(m.get("memory") or "")[:160] for m in members]
                    out.append(entry)
                    continue
                out.append(r)
                sid = r.get("scene_id")
                if not sid or str(sid) in seen:
                    continue
                if sid not in scene_cache:
                    row = self._conn.execute(
                        """
                        SELECT * FROM memories m
                        WHERE m.id = ? AND m.user_id = ? AND m.kind = ?
                          AND (m.status = 'active' OR m.status IS NULL)
                        """,
                        (sid, user_id, KIND_SCENE),
                    ).fetchone()
                    scene_cache[sid] = dict(row) if row else None
                srow = scene_cache[sid]
                if srow:
                    sd = dict(srow)
                    sd["_recall_score"] = r.get("_recall_score", 0.0)
                    sd["_scene"] = True  # ADD THIS LINE
                    out.append(sd)
                    seen.add(str(sid))
        return out

    # ── L3 persona cache ───────────────────────────────────────────────────────
    # An always-hydrated, cheap (no embeddings, no KNN/FTS/graph) blob of the
    # most stable identity facts. Refreshed on write (cache invalidation) and
    # TTL-cached within a session. Injected every turn via persona_context().

    def persona_context(self) -> str | None:
        """L3 stable-identity blob for context injection. TTL-cached."""
        with self._persona_lock:
            nows = time.monotonic()
            if self._persona_cached is not None and nows - self._persona_cache_at < PERSONA_CACHE_TTL:
                return self._persona_cached
            block = self._build_persona_context()
            self._persona_cached = block
            self._persona_cache_at = nows
            return block

    def _build_persona_context(self) -> str | None:
        user_id = self._resolve_user_id(self._user_id_override)
        # Persona only needs Phase A 'kind' (not the L2 scene_id column).
        try:
            if "kind" not in existing_columns(self._conn):
                return None
        except Exception:
            return None
        with self._mem._db_lock:
            rows = self._conn.execute(
                """
                SELECT id, memory
                FROM memories m
                WHERE m.user_id = ? AND m.kind = ?
                  AND (m.status = 'active' OR m.status IS NULL)
                ORDER BY m.pinned DESC, m.access_count DESC, m.created_at ASC
                LIMIT ?
                """,
                (user_id, "identity", PERSONA_RECALL_LIMIT),
            ).fetchall()
        texts = [str(r["memory"]).strip() for r in rows if (r["memory"] or "").strip()]
        if not texts:
            return None
        lines = [
            "<persona>",
            "Stable facts about the person — always available. Use silently.",
            "",
        ]
        for t in texts:
            clip = t if len(t) <= MEMORY_CONTEXT_FACT_CHARS else t[:MEMORY_CONTEXT_FACT_CHARS].rstrip() + "..."
            lines.append(f"  - {clip}")
        lines.append("</persona>")
        block = "\n".join(lines)
        if len(block) > PERSONA_CONTEXT_CHARS:
            block = block[:PERSONA_CONTEXT_CHARS].rstrip() + "\n</persona>"
        return block

    # ── L2 scene public API ────────────────────────────────────────────────────

    def build_scene(
        self,
        summary: str,
        member_ids: list[str],
        user_id: str | None = None,
        *,
        pinned: bool = False,
    ) -> str | None:
        """Build an L2 scene linking member fact ids, and clear caches."""
        user_id = self._resolve_user_id(user_id)
        scene_id = self._mem.build_scene(
            user_id, summary=summary, member_ids=member_ids, pinned=pinned
        )
        if scene_id:
            self._clear_search_cache()
        return scene_id

    def list_scenes(self, user_id: str | None = None, limit: int = SCENE_CONTEXT_LIMIT) -> list[dict]:
        """Return recent scenes (long-running episodes) for a user."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.list_scenes(user_id, limit=limit)

    def scene_members(self, scene_id: str, user_id: str | None = None, limit: int = SCENE_MEMBER_LIMIT) -> list[dict]:
        """Return memories that are members of a given scene."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.scene_members(scene_id, user_id, limit=limit)

    def scene_context(self, limit: int = SCENE_CONTEXT_LIMIT, user_id: str | None = None) -> str | None:
        """Compact recent-scenes bootstrap block (L2), independent of RRF."""
        user_id = self._resolve_user_id(user_id)
        scenes = self._mem.list_scenes(user_id, limit=limit)
        if not scenes:
            return None
        lines = ["<scenes>", "Recent long-running episodes — use only if relevant.", ""]
        for s in scenes:
            text = (s.get("memory") or "").strip()
            if not text:
                continue
            if len(text) > MEMORY_CONTEXT_FACT_CHARS:
                text = text[:MEMORY_CONTEXT_FACT_CHARS].rstrip() + "..."
            lines.append(f"  - {text}")
        lines.append("</scenes>")
        block = "\n".join(lines)
        if len(block) > SCENE_CONTEXT_CHARS:
            block = block[:SCENE_CONTEXT_CHARS].rstrip() + "\n</scenes>"
        return block

    def _write_search_replay(self, query: str, results: list[dict], user_id: str) -> None:
        """Append search to replay log (debug/tuning, env-gated)."""
        try:
            db_path = Path(_memory_db_path_for_user(user_id))
            replay_path = db_path.parent / "search_replay.jsonl"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "query": query[:200],
                "result_count": len(results),
                "results": [
                    {
                        "id": r["id"],
                        "score": round(r.get("_recall_score", 0.0), 6),
                        "text": r["memory"][:100],
                    }
                    for r in results
                ],
            }
            with open(replay_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.debug(f"search replay write failed: {e}")

    def _recent_or_important_memories(self, user_id: str, limit: int, include_history: bool = False) -> list[dict]:
        """
        FIX 3: status filtering (active-only, unless include_history) is now
        applied in SQL, before the dedup+truncate step below — not by the
        caller after truncation. Previously the caller filtered out
        superseded rows AFTER this method had already cut the candidate
        pool down to `limit`, which could return fewer than `limit` results
        even when the wider fetch window had enough active memories to
        fill it.
        """
        fetch_n = max(int(limit) * 4, int(limit) + 10)
        status_sql = _active_sql(not include_history)
        with self._mem._db_lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM memories m
                WHERE m.user_id = ?
                  {status_sql}
                ORDER BY m.pinned DESC, m.created_at DESC, m.access_count DESC
                LIMIT ?
                """.format(status_sql=status_sql),
                (user_id, fetch_n),
            ).fetchall()

        best_by_text: dict[str, sqlite3.Row] = {}
        order: list[str] = []
        for row in rows:
            norm = _normalize_memory_text(row["memory"])
            existing = best_by_text.get(norm)
            if existing is None:
                best_by_text[norm] = row
                order.append(norm)
            elif row["created_at"] > existing["created_at"]:
                best_by_text[norm] = row

        deduped = [best_by_text[norm] for norm in order][:int(limit)]
        out = []
        for r in deduped:
            d = dict(r)
            d["_recall_score"] = 1.0  # broad recall is explicit — never filtered
            out.append(d)
        return out

    def _touch_memories(self, results: list[dict]) -> None:
        """Bump access_count on every recall; bump access_day_count only on a
        new local calendar day (Phase 2 spacing effect).

        access_count stays the total touch counter for decay/ranking.
        access_day_count counts distinct local days a memory was recalled —
        monthly consolidation uses this for the spacing term of R.
        """
        if not results:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            today_local = bioclock.local_now().strftime("%Y-%m-%d")
        except Exception:
            today_local = datetime.now().astimezone().strftime("%Y-%m-%d")
        mem_ids = [str(r.get("id", "")) for r in results if r.get("id")]
        if not mem_ids:
            return
        with self._mem._db_lock:
            try:
                cols = existing_columns(self._conn)
                has_day_count = "access_day_count" in cols
                placeholders = ",".join("?" * len(mem_ids))
                if has_day_count:
                    rows = self._conn.execute(
                        f"SELECT id, last_accessed_at FROM memories WHERE id IN ({placeholders})",
                        mem_ids,
                    ).fetchall()
                    day_bump_ids: list[str] = []
                    for row in rows:
                        la = row["last_accessed_at"] or "never"
                        if la == "never" or not la:
                            day_bump_ids.append(str(row["id"]))
                            continue
                        try:
                            ts = datetime.fromisoformat(str(la).replace("Z", "+00:00"))
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            last_local = ts.astimezone().strftime("%Y-%m-%d")
                        except Exception:
                            last_local = ""
                        if last_local != today_local:
                            day_bump_ids.append(str(row["id"]))

                    self._conn.execute(
                        f"""
                        UPDATE memories
                        SET access_count = MIN(access_count + 1, 255),
                            last_accessed_at = ?
                        WHERE id IN ({placeholders})
                        """,
                        [now_iso] + mem_ids,
                    )
                    if day_bump_ids:
                        ph2 = ",".join("?" * len(day_bump_ids))
                        self._conn.execute(
                            f"""
                            UPDATE memories
                            SET access_day_count = MIN(access_day_count + 1, 255)
                            WHERE id IN ({ph2})
                            """,
                            day_bump_ids,
                        )
                else:
                    self._conn.execute(
                        f"""
                        UPDATE memories
                        SET access_count = MIN(access_count + 1, 255),
                            last_accessed_at = ?
                        WHERE id IN ({placeholders})
                        """,
                        [now_iso] + mem_ids,
                    )
                self._conn.commit()
            except Exception as e:
                log.warning(f"Access tracking failed for {mem_ids}: {e}")

    MIN_CLEAR_INTERVAL: float = 0.5  # seconds — debounce window for cache invalidation

    def _clear_search_cache(self) -> None:
        # The search-result cache lives on _MemoryBackend now (see _MemoryBackend.__init__).
        # Delegate so we don't keep a separate copy on AikoMemorize that would
        # silently desync from the one the search path actually reads/writes.
        if self._mem_backend is not None:
            self._mem._clear_search_cache()
        # Persona blob is identity-derived; a new/changed identity fact must
        # be reflected on the next turn.
        with self._persona_lock:
            self._persona_cached = None
            self._persona_cache_at = 0.0

    def _maybe_clear_search_cache(self) -> None:
        """Time-debounced cache clearing — invalidate on write, but only if
        at least MIN_CLEAR_INTERVAL has elapsed since the last clear.

        Normal-paced conversation (one write per turn, seconds between them)
        always sees fresh data.  Rapid writes within the same debounce window
        (bulk import, batch writes) keep the cache warm instead of cold-starting
        on every single write — the only acceptable staleness window.
        """
        now = time.monotonic()
        if now - self._last_cache_clear_time >= self.MIN_CLEAR_INTERVAL:
            self._clear_search_cache()
            self._last_cache_clear_time = now

    def format_for_context(
        self,
        memories: list[dict],
        *,
        query: str = "",
        related: dict | None = None,
        user_id: str | None = None,
        embedder=None,
        query_vector=None,
    ) -> str | None:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.

        created_at is always stored in UTC (see _MemoryBackend.add()/
        add_raw()), but the age labels here ("today", "yesterday", "N days
        ago") should reflect the person's local calendar day, not UTC's.
        So each row's UTC created_at is converted into local time before
        diffing against bioclock.local_now() — diffing the raw UTC
        timestamp against a local "now" would misplace the day boundary
        by whatever the local UTC offset is (e.g. a memory from 11pm local
        last night could read as "today" or vice versa near midnight).
        """
        if not memories:
            return None

        now = bioclock.local_now()
        # Local tz offset used to convert stored UTC timestamps into the
        # same local frame as `now`, regardless of whether bioclock returns
        # a naive or tz-aware datetime.
        local_tz = datetime.now().astimezone().tzinfo
        now_is_aware = isinstance(now, datetime) and now.tzinfo is not None

        lines = [
            "<memory_context>",
            "Facts about the person you are speaking with — not a separate person. Use silently. Never quote or reference this block directly.",
            "IMPORTANT: dates and 'today'/'yesterday' inside these memories refer to when the event happened, never to the current date. The only authoritative 'now' is the <current_datetime> block. Never treat a date, month, or time inside a memory as today's date.",
            "",
        ]
        kept = False
        for m in memories:
            text       = m.get("memory") or m.get("text")
            if not text:
                continue
            # Frozen relative-time / date-check facts masquerade as the
            # current date (e.g. "Oppa today is Monday, August 10, 2026",
            # "Aiko checks the date of July 3"). Drop them from context —
            # stale ephemeral junk, and the main source of Aiko's "what
            # month is it?" confusion. They stay in the DB (recoverable)
            # until util/scrub_stale_memory_dates.py removes them.
            if _is_stale_temporal_fact(text):
                log.debug("memory recall dropped stale temporal fact: %r", text[:120])
                continue
            kept = True
            if len(text) > MEMORY_CONTEXT_FACT_CHARS:
                text = text[:MEMORY_CONTEXT_FACT_CHARS].rstrip() + "..."
            created_at = m.get("created_at")
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        # legacy rows written before the UTC-everywhere fix —
                        # treat as UTC rather than silently mismatching.
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts_local = ts.astimezone(local_tz)
                    if not now_is_aware:
                        ts_local = ts_local.replace(tzinfo=None)
                    delta = now - ts_local
                    days  = delta.days
                    if days == 0:
                        age = "today"
                    elif days == 1:
                        age = "yesterday"
                    else:
                        age = f"{days} days ago"
                    lines.append(f"  - [{age}] {text}")
                except Exception:
                    lines.append(f"  - {text}")
            else:
                lines.append(f"  - {text}")

        if not kept:
            return None

        lines.append("</memory_context>")
        block = "\n".join(lines)
        if len(block) > MEMORY_CONTEXT_TOTAL_CHARS:
            block = block[:MEMORY_CONTEXT_TOTAL_CHARS].rstrip() + "\n</memory_context>"

        if MEMORY_SUPERSESSION_NARRATIVE and MEMORY_SUPERSESSION_NARRATIVE_MAX > 0:
            narr_lines = []
            seen_keys: set[str] = set()
            for r in memories or []:
                if len(seen_keys) >= MEMORY_SUPERSESSION_NARRATIVE_MAX:
                    break
                chain = r.get("_supersession_chain")
                if not chain or len(chain) < 2:
                    continue
                key = str(chain[-1].get("id") or "")  # tip id
                if not key or key in seen_keys:
                    continue
                line = format_supersession_narrative(chain)
                if line:
                    narr_lines.append(f"  {line}")
                    seen_keys.add(key)
            if narr_lines:
                block += (
                    "\n\n<memory_update>\n"
                    + "\n".join(narr_lines)
                    + "\n</memory_update>"
                )

        # Phase 13a: secondary related knowledge / experience
        if MEMORY_CROSS_STORE_ENABLED and MEMORY_CROSS_STORE_CONTEXT_CHARS > 0:
            try:
                from cognition.memory.narrative import fetch_related_for_memories, format_related_blocks

                rel = related
                if rel is None:
                    rel = fetch_related_for_memories(
                        query or "",
                        memories,
                        user_id=user_id or self._resolve_user_id(None),
                        embedder=embedder,
                    )
                extra = format_related_blocks(
                    rel or {},
                    max_chars=MEMORY_CROSS_STORE_CONTEXT_CHARS,
                )
                if extra:
                    block = block + "\n\n" + extra
            except Exception as exc:
                log.debug("cross_store context skipped: %s", exc)

        # Brain trace: capture exactly what gets injected into the prompt.
        _brain_trace.record_step(
            "AikoMemorize.format_for_context",
            layer="context",
            inputs={"memories_count": len(memories),
                    "cross_store_enabled": bool(MEMORY_CROSS_STORE_ENABLED)},
            outputs={"block_chars": len(block or ""), "block_preview": (block or "")[:1200]},
            factors=[
                f"age labels computed in local tz: today/yesterday/N days ago",
                f"stale temporal facts dropped via _is_stale_temporal_fact",
                f"supersession_narrative cap={MEMORY_SUPERSESSION_NARRATIVE_MAX}",
            ],
        )

        # Append the episodic-memory block (EMC-3) if available. Joint
        # budget: when EMC_JOINT_BUDGET is on, both blocks share the SM
        # total-char budget and the SM block is trimmed to make room.
        try:
            from cognition.memory.episode import EMC_JOINT_BUDGET
        except Exception:
            EMC_JOINT_BUDGET = False
        try:
            em_block = self._format_episodes_for_context(query or "", query_vector)
        except Exception as e:
            log.debug("EMC format_episodes skipped: %s", e)
            em_block = None
        if not em_block:
            return block
        if not block:
            return em_block
        if not EMC_JOINT_BUDGET:
            return f"{block}\n\n{em_block}"

        shared = int(MEMORY_CONTEXT_TOTAL_CHARS)
        # If episodic block alone exceeds the shared budget, re-render
        # with a tighter cap so the SM block still has room.
        if len(em_block) > shared:
            try:
                from cognition.memory.episode import EMC_RECALL_LIMIT
                store = self.episodic.get_store() if self.episodic is not None else None
                if store is not None:
                    hits = store.search(
                        query or "",
                        limit=EMC_RECALL_LIMIT,
                        user_id=self.get_user_id(),
                        query_vector=query_vector,
                    )
                    em_block = store.format_for_context(hits, max_chars=shared) or em_block
            except Exception as exc:
                log.debug("EMC joint-budget trim failed: %s", exc)
        em_len = len(em_block)
        sm_budget = shared - em_len - 2
        if len(block) > sm_budget:
            sm_closing = "\n</memory_context>"
            cut = sm_budget - len(sm_closing) if "</memory_context>" in block else sm_budget
            sm_trim = block[:cut]
            if "</memory_context>" in block and "</memory_context>" not in sm_trim:
                sm_trim = sm_trim.rstrip() + sm_closing
            block = sm_trim
        return f"{block}\n\n{em_block}"

    # ── dream pass ────────────────────────────────────────────────────────────

    def dream(
        self,
        user_id:   str | None = None,
        dry_run:   bool  = False,
        threshold: float = DREAM_MERGE_THRESHOLD,
    ) -> dict:
        """
        Nightly memory consolidation pass.

        Stages (in order):
          1. Boost  — salient memories get +DREAM_BOOST_AMOUNT access_count.
          2. Merge  — near-duplicate pairs (cosine >= threshold) are collapsed.
          3. Schema — recurring entity+valence clusters are abstracted into a
             single generalized kind='schema' gist memory (Phase 21).
          4. Prune  — standard decay cleanup runs last.

        all_mems is fetched once and passed through to cleanup() so the
        prune stage doesn't re-scan the table from scratch.

        Returns dict: {boosted, merged, schemas, pruned, duration_s}
        """
        user_id = self._resolve_user_id(user_id)
        t_start = time.perf_counter()
        log.info(f"{'(dry-run) ' if dry_run else ''}Starting consolidation pass...")

        mem_ids: list[str] = []
        all_batch_mems: list[dict] = []
        boosted = 0

        for batch in self._iter_memory_batches(user_id):
            batch_ids = [str(m.get("id", "")) for m in batch if m.get("id")]
            if not batch_ids:
                continue
            all_batch_mems.extend(batch)
            # FIX: only active memories are eligible as a merge *source*.
            # _dream_merge's KNN neighbor search is already active_only=True,
            # but the outer mem_id being probed wasn't filtered — so a
            # superseded memory could get compared against an active
            # near-duplicate, and _resolve_duplicate's tie-break (higher
            # access_count wins, with no status check) could pick the
            # *active* memory as the loser if the superseded one had
            # accumulated more accesses before being superseded. Superseded
            # rows are inert history; they should age out via cleanup()'s
            # normal decay path, not compete with active memories for
            # survival in the merge pass.
            merge_source_ids = [
                str(m.get("id", "")) for m in batch
                if m.get("id") and m.get("status") != STATUS_SUPERSEDED
            ]
            mem_ids.extend(merge_source_ids)
            payload_map = self._batch_get_payloads(batch_ids)
            with self._mem._db_lock:
                pinned_ids = _sqlite_pinned_ids(self._conn, batch_ids)
            boosted += self._dream_boost(batch, payload_map, pinned_ids=pinned_ids, dry_run=dry_run)

        if not mem_ids:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "schemas": 0, "pruned": 0, "duration_s": 0.0}

        with self._mem._db_lock:
            pinned_ids = _sqlite_pinned_ids(self._conn, mem_ids)
        merged = self._dream_merge(mem_ids, user_id=user_id, threshold=threshold, pinned_ids=pinned_ids, dry_run=dry_run)
        schemas = self._dream_schema(user_id=user_id, dry_run=dry_run)
        try:
            pin_age = max(1, int(os.getenv("MEMORY_PIN_REBALANCE_AGE_DAYS", "45")))
        except (TypeError, ValueError):
            pin_age = 45
        try:
            pin_access = max(0, int(os.getenv("MEMORY_PIN_REBALANCE_MIN_ACCESS", "2")))
        except (TypeError, ValueError):
            pin_access = 2
        pin_result = self.rebalance_pins(user_id, max_age_days=pin_age, min_access_count=pin_access, dry_run=dry_run)
        # FIX: dream() previously called cleanup() with no _all_mems, so the
        # "already-fetched memory list is passed through here to avoid a
        # redundant get_all() scan" claim in cleanup()'s docstring was dead —
        # cleanup() always re-scanned the whole table from scratch, right
        # after dream() had just scanned it for the boost stage. Passing the
        # batches we already fetched above wires up that optimization for
        # real.
        prune_result = self.cleanup(user_id=user_id, dry_run=dry_run, _all_mems=all_batch_mems)
        pruned = prune_result.get("deleted", 0)

        duration = round(time.perf_counter() - t_start, 2)
        log.info(
            f"{'(dry-run) ' if dry_run else ''}"
            f"Done — boosted={boosted}, merged={merged}, schemas={schemas}, pruned={pruned}, "
            f"duration={duration}s"
        )
        return {"boosted": boosted, "merged": merged, "schemas": schemas, "pruned": pruned, "unpinned": pin_result.get("unpinned", 0), "duration_s": duration}

    def _dream_boost(
        self,
        all_mems:    list[dict],
        payload_map: dict,
        pinned_ids:  set[str] | None = None,
        dry_run:     bool = False,
    ) -> int:
        """
        Increment access_count on memories matching salience heuristics.
        Pinned memories pass through unchanged.
        Returns count of memories boosted.

        Phase 5: prefers stored salience_hit / non-neutral valence_tag when set.
        Uses composite salience_score (incl. access_day_count spacing) so
        dream-boost is no longer a blunt boolean OR of heuristics.
        """
        now     = datetime.now(timezone.utc)
        boost_ids: list[str] = []
        pinned_ids = pinned_ids or set()

        # Optional day-count map so spaced-repetition signal reaches salience_score.
        mem_ids = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        day_map: dict[str, int] = {}
        try:
            with self._mem._db_lock:
                cols = existing_columns(self._conn)
                if "access_day_count" in cols and mem_ids:
                    ph = ",".join("?" * len(mem_ids))
                    for row in self._conn.execute(
                        f"SELECT id, access_day_count FROM memories WHERE id IN ({ph})",
                        mem_ids,
                    ).fetchall():
                        day_map[str(row["id"])] = int(row["access_day_count"] or 0)
        except Exception:
            day_map = {}

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            if not mem_id:
                continue
            if mem_id in pinned_ids:
                continue

            text     = m.get("memory") or ""
            ac, _la  = payload_map.get(mem_id, (0, "never"))

            is_recent  = False
            created_at = m.get("created_at", "")
            if created_at:
                try:
                    ts        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    is_recent = (now - ts).days <= 7
                except Exception:
                    log.warning("memorize: failed to parse created_at")

            # Phase 5: prefer stored turn tags when present.
            stored_salient = False
            emotional = False
            if os.getenv("DREAM_BOOST_USE_STORED_TAGS", "1").lower() in {"1", "true", "yes", "on"}:
                try:
                    sh = m.get("salience_hit")
                    if sh is not None and str(sh) != "":
                        stored_salient = bool(int(sh))
                    vt = m.get("valence_tag") or "neutral"
                    if isinstance(vt, str) and vt.strip().lower() in ("neg", "pos"):
                        emotional = True
                except Exception:
                    pass

            # Composite salience (0..1) instead of pure boolean OR of heuristics.
            age_days = None
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                except Exception:
                    age_days = 7.0 if is_recent else None

            day_n = day_map.get(mem_id)
            if day_n is None:
                try:
                    day_n = int(m.get("access_day_count") or 0)
                except (TypeError, ValueError):
                    day_n = 0

            try:
                from cognition.memory.forget import salience_score
                s_score = salience_score(
                    text=text,
                    access_count=ac,
                    access_day_count=day_n,
                    valence_tag=m.get("valence_tag"),
                    valence_score=m.get("valence_score"),
                    salience_hit=1 if stored_salient else (m.get("salience_hit") or 0),
                    age_days=age_days,
                )
            except Exception:
                s_score = 1.0 if (stored_salient or emotional or bool(_SALIENCE_RE.search(text)) or ac >= 3 or is_recent) else 0.0

            if s_score < 0.35:
                continue

            boost_ids.append(mem_id)

        if boost_ids and not dry_run:
            with self._mem._db_lock:
                try:
                    placeholders = ",".join("?" * len(boost_ids))
                    self._conn.execute(
                        f"""
                        UPDATE memories
                        SET access_count = MIN(access_count + ?, 255)
                        WHERE id IN ({placeholders})
                        """,
                        [DREAM_BOOST_AMOUNT] + boost_ids,
                    )
                    self._conn.commit()
                except Exception as e:
                    log.warning(f"Batch boost failed for {len(boost_ids)} memories: {e}")
                    self._conn.rollback()
                    return 0

        boosted = len(boost_ids)
        if boosted:
            log.info(f"{'(dry-run) ' if dry_run else ''}Boosted {boosted} memories.")
        return boosted

    def _dream_merge(
        self,
        mem_ids:   list[str],
        user_id:   str,
        threshold: float = DREAM_MERGE_THRESHOLD,
        pinned_ids: set[str] | None = None,
        dry_run:   bool  = False,
    ) -> int:
        """
        Detect and collapse near-duplicate memory vectors.
        Pinned memories are never chosen as the loser.
        Returns count of memories deleted as duplicates.
        """
        deleted_ids: set[str] = set()
        pinned_ids = pinned_ids or set()
        merged = 0

        for mem_id in mem_ids:
            if mem_id in deleted_ids:
                continue
            if mem_id in pinned_ids:
                continue

            with self._mem._db_lock:
                vector = _sqlite_get_vector(self._conn, mem_id)
            if not vector:
                continue

            with self._mem._db_lock:
                try:
                    neighbor_rows = _sqlite_knn_search(
                        self._conn, vector, user_id, limit=4, threshold=threshold
                    )
                except Exception as e:
                    log.warning(f"Similarity search failed for {mem_id}: {e}")
                    continue

            for row in neighbor_rows:
                neighbor_id = row["id"]
                if neighbor_id == mem_id:
                    continue
                if neighbor_id in deleted_ids:
                    continue

                similarity = 1.0 - row["dist"]
                n_merged = self._resolve_duplicate(
                    mem_id, neighbor_id, similarity, pinned_ids=pinned_ids, dry_run=dry_run
                )
                if n_merged:
                    deleted_ids.add(neighbor_id)
                    merged += 1

        if merged:
            log.info(f"{'(dry-run) ' if dry_run else ''}Merged {merged} duplicate memories.")
        return merged

    def _dream_schema(self, user_id: str | None = None, dry_run: bool = False) -> int:
        """Phase 21: schema abstraction in the dream pass.

        Cluster active memories that share entities, and for each entity with
        enough members (>= DREAM_SCHEMA_MIN_MEMBERS) and a consistent valence
        sign (>= DREAM_SCHEMA_VALENCE_MAJORITY fraction), synthesize ONE
        generalized "schema" memory (kind='schema') that captures the recurring
        theme. The schema memory's `schema_sources` column records the source
        memory ids it was abstracted from, enabling idempotency — subsequent
        dream runs skip clusters already abstracted.

        Returns the count of schema memories written (0 in dry_run).
        """
        user_id = self._resolve_user_id(user_id)
        if not DREAM_SCHEMA_ENABLED:
            return 0

        # Ensure the schema_sources column exists
        try:
            ensure_l3_schema_schema(self._conn)
        except Exception as e:
            log.debug("schema_sources migration: %s", e)

        # Idempotency: load existing schema fact entities and their source
        # ids, so we don't re-abstract the same clusters every night.
        schema_covered_entities: set[str] = set()
        schema_sources_union: set[str] = set()
        try:
            existing = self._conn.execute(
                "SELECT entities, schema_sources FROM memories "
                "WHERE user_id = ? AND kind = ? AND (status = 'active' OR status IS NULL)",
                (user_id, KIND_SCHEMA),
            ).fetchall()
            for r in existing:
                for e in entities_from_json(r[0] or "[]"):
                    schema_covered_entities.add(e)
                try:
                    srcs = json.loads(r[1] or "[]")
                    schema_sources_union.update(srcs)
                except Exception:
                    pass
        except Exception as e:
            log.debug("load existing schema facts: %s", e)

        # Fetch active memories with entities and valence_score
        # (iter_all does NOT include entities, so we query directly)
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT id, entities, valence_score, pinned
                FROM memories
                WHERE user_id = ? AND (status = 'active' OR status IS NULL)
                """,
                (user_id,),
            ).fetchall()

        # Build entity -> list of (mem_id, valence_score) clusters
        clusters: dict[str, list[tuple[str, int | None]]] = {}
        for row in rows:
            mid = str(row[0])
            pinned = bool(row[3])
            if pinned:
                continue  # don't abstract pinned memories
            ents = entities_from_json(row[1] or "")
            if not ents:
                # fall back to on-the-fly extraction from memory text
                try:
                    ents = extract_entities(row[2] or "", max_entities=5)
                    if not ents:
                        continue
                except Exception:
                    continue
            vs = row[2]  # valence_score, may be None
            for e in ents:
                e = str(e).strip()
                if not e:
                    continue
                clusters.setdefault(e, []).append((mid, vs))

        # Sort clusters by size descending, cap at MAX_CLUSTERS
        sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        if len(sorted_clusters) > DREAM_SCHEMA_MAX_CLUSTERS:
            sorted_clusters = sorted_clusters[:DREAM_SCHEMA_MAX_CLUSTERS]

        written = 0
        for entity, members in sorted_clusters:
            if len(members) < DREAM_SCHEMA_MIN_MEMBERS:
                continue

            # Idempotency: skip if this entity already has a schema that
            # covers many of the same members (schema_sources overlap).
            skip = False
            for sc_e in schema_covered_entities:
                if sc_e == entity:
                    # a schema for this entity already exists → don't re-abstract
                    skip = True
                    break
            if skip:
                continue

            # Compute dominant valence sign among members with stored scores
            valences = [v for _, v in members if v is not None]
            if not valences:
                continue  # no valence data — nothing to abstract on

            pos = sum(1 for v in valences if v > 0)
            neg = sum(1 for v in valences if v < 0)
            total_with_val = len(valences)
            dominant = None
            if pos > neg and pos / total_with_val >= DREAM_SCHEMA_VALENCE_MAJORITY:
                dominant = 2  # positive
            elif neg > pos and neg / total_with_val >= DREAM_SCHEMA_VALENCE_MAJORITY:
                dominant = -2  # negative
            else:
                dominant = 0  # neutral / mixed

            n = len(members)
            # Synthesize gist text via template (deterministic, offline-safe)
            name = current_display_name() or "User"
            if dominant == 2:  # positive
                gist = f"{name} often has positive experiences involving {entity}."
            elif dominant == -2:  # negative
                gist = f"{name} often has negative/difficult experiences involving {entity}."
            else:  # neutral / mixed
                gist = f"{entity} is a recurring topic for {name} ({n} related memories)."

            # For better gist, include the count of related memories
            # (we can refine later — template stays grounded)
            if dominant in (2, -2):
                gist = f"{entity} is a recurring {'positive' if dominant == 2 else 'negative'} theme for {name} ({n} memories)."

            if dry_run:
                written += 1
                continue

            # Embed the gist (best-effort; None if embedder down)
            try:
                vector = self._embed(gist)
            except Exception as e:
                log.debug("embed schema gist: %s", e)
                vector = None

            # Determine schema_tag/valence based on dominant
            from cognition.memory.entity import tag_from_score
            vs_tag = tag_from_score(dominant) if dominant is not None else "neutral"

            # Insert schema memory
            # Collect member ids for traceability / future idempotency
            member_ids = [mid for mid, _ in members]
            # Use _insert_row with kind=KIND_SCHEMA, source=SOURCE_DREAM,
            # entities=[entity], and schema_sources=member_ids JSON
            now = datetime.now(timezone.utc).isoformat()
            mem_id = str(uuid.uuid4())
            self._insert_row(
                mem_id=mem_id,
                user_id=user_id,
                text=gist,
                now=now,
                vector=vector,
                pinned=0,
                source=SOURCE_DREAM,
                kind=KIND_SCHEMA,
                entities=[entity],
                valence_tag=vs_tag,
                llm_score=dominant if dominant is not None else None,
                salience_hit=1,
                schema_sources=member_ids,
            )
            self._conn.commit()
            written += 1

        return written

    def _resolve_duplicate(
        self,
        id_a:    str,
        id_b:    str,
        score:   float,
        pinned_ids: set[str] | None = None,
        dry_run: bool = False,
    ) -> bool:
        pinned_ids = pinned_ids or set()
        if id_a in pinned_ids or id_b in pinned_ids:
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _     = payload_map.get(id_a, (0, "never"))
        ac_b, _     = payload_map.get(id_b, (0, "never"))
        with self._mem._db_lock:
            row_map = {
                row["id"]: row["created_at"]
                for row in self._conn.execute(
                    "SELECT id, created_at FROM memories WHERE id IN (?, ?)", (id_a, id_b)
                ).fetchall()
            }
        if ac_a == ac_b:
            loser = id_b if row_map.get(id_a, "") >= row_map.get(id_b, "") else id_a
        else:
            loser = id_b if ac_a > ac_b else id_a

        if dry_run:
            log.info(
                f"(dry-run) Would merge: score={score:.3f} "
                f"ac_a={ac_a} ac_b={ac_b} → delete {loser}"
            )
            return True

        try:
            self._mem.delete(memory_id=loser)
            log.info(
                f"Merged duplicate (score={score:.3f}, "
                f"ac_a={ac_a}, ac_b={ac_b}) → deleted {loser}"
            )
            return True
        except Exception as e:
            log.warning(f"Merge delete failed for {loser}: {e}")
            return False

    # ── working memory (Grasp live hub) ───────────────────────────────────────
    # Thin delegates to cognition.memory.grasp's per-identity buffer. Memorize
    # owns the WM surface so think.py never needs grasp-specific imports or
    # monkeypatched methods — see the old install_into_think() history.

    def wm_record_turn(self, user_input: str, response_text: str) -> None:
        """Fill the current identity's working-memory buffer after a completed turn."""
        try:
            from cognition.memory.grasp import record_turn
            record_turn(user_input, response_text)
        except Exception:
            pass

    def wm_context_block(self) -> str:
        """Scored <grasp> block for system-prompt injection ('' when empty/disabled)."""
        try:
            from cognition.memory.grasp import get_context_block
            return get_context_block()
        except Exception:
            return ""

    def wm_reset(self) -> None:
        """Clear the current identity's working-memory buffer (/reset)."""
        try:
            from cognition.memory.grasp import clear_live
            clear_live()
        except Exception:
            pass

    def wm_studio_state(self) -> dict:
        """Live WM state for Grasp Studio (current identity)."""
        from cognition.memory.grasp import live_studio_state
        return live_studio_state()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id:   str | None = None,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run:   bool  = False,
        _all_mems: list[dict] | None = None,
        _pinned_ids: set[str] | None = None,
    ) -> dict:
        """
        Prune decayed memories below threshold score.
        Grace period (default 35 days) protects newly created memories.
        Pinned memories are unconditionally kept.

        _all_mems: internal - when called from dream(), the already-fetched
        memory list is passed through here to avoid a redundant get_all() scan.

        Returns dict: {deleted, kept, failed, candidates (dry_run only)}.
        """
        # Lazy guest boot: pruning a throwaway tempfile DB that was never
        # written to is pure waste — and touching self._mem here would
        # materialize it. No backend open yet + guest identity → no-op.
        if self._mem_backend is None and (user_id or self.get_user_id()) == "guest":
            return {"pruned": 0, "skipped": "guest (no store open)"}
        user_id = self._resolve_user_id(user_id)
        source = [_all_mems] if _all_mems is not None else self._iter_memory_batches(user_id)

        kept = 0
        deleted: list[str] = []
        failed: list[dict] = []
        dry_candidates: list[dict] = []
        saw_any = False

        for batch in source:
            if not batch:
                continue
            saw_any = True
            batch_kept, candidates = self._cleanup_candidates(
                batch,
                _pinned_ids=_pinned_ids,
            )
            kept += batch_kept

            if dry_run:
                dry_candidates.extend(candidates)
                continue

            for c in candidates:
                try:
                    self._mem.delete(memory_id=c["id"])
                    deleted.append(c["id"])
                except Exception as e:
                    failed.append({"id": c["id"], "error": str(e)})

        if not saw_any:
            return {"deleted": 0, "kept": 0, "failed": 0}

        if dry_run:
            dry_candidates.sort(key=lambda x: x["weighted_score"])
            log.info(f"Dry run: {len(dry_candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": dry_candidates}

        if deleted:
            self._clear_search_cache()
            self.optimize()

        log.info(f"Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    def _iter_memory_batches(self, user_id: str, batch_size: int = MEMORY_LIFECYCLE_BATCH_SIZE):
        """Yield lifecycle scan batches without retaining the full table."""
        batch: list[dict] = []
        for mem in self._mem.iter_all(user_id=user_id, batch_size=batch_size):
            batch.append(mem)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _cleanup_candidates(
        self,
        all_mems: list[dict],
        _pinned_ids: set[str] | None = None,
    ) -> tuple[int, list[dict]]:
        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)
        with self._mem._db_lock:
            pinned_ids  = _pinned_ids if _pinned_ids is not None else _sqlite_pinned_ids(self._conn, mem_ids)

        candidates = []
        kept       = 0
        lineage_ids: set[str] = set()

        try:
            with self._mem._db_lock:
                rows = self._conn.execute(
                    "SELECT supersedes_id FROM memories WHERE supersedes_id IS NOT NULL AND supersedes_id != ''"
                ).fetchall()
            lineage_ids = {str(r["supersedes_id"]) for r in rows if r["supersedes_id"]}
        except Exception:
            lineage_ids = set()

        # Ambient mood for offline cleanup so mood-dependent forgetting is live.
        ambient_valence = None
        try:
            ambient_valence = resolve_ambient_valence(self.get_user_id())
        except Exception:
            ambient_valence = None

        # Optional day-count map when column exists.
        day_map: dict[str, int] = {}
        try:
            with self._mem._db_lock:
                cols = existing_columns(self._conn)
                if "access_day_count" in cols and mem_ids:
                    ph = ",".join("?" * len(mem_ids))
                    for row in self._conn.execute(
                        f"SELECT id, access_day_count FROM memories WHERE id IN ({ph})",
                        mem_ids,
                    ).fetchall():
                        day_map[str(row["id"])] = int(row["access_day_count"] or 0)
        except Exception:
            day_map = {}

        for m in all_mems:
            mem_id     = str(m.get("id", ""))
            ac, la     = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            if mem_id in pinned_ids:
                kept += 1
                continue

            if mem_id in lineage_ids:
                kept += 1
                continue

            v_tag = m.get("valence_tag")
            v_score = m.get("valence_score")
            day_n = day_map.get(mem_id)
            if day_n is None:
                try:
                    day_n = int(m.get("access_day_count") or 0)
                except (TypeError, ValueError):
                    day_n = 0
            if should_cleanup(
                ac, la, created_at,
                valence_tag=v_tag, valence_score=v_score,
                query_valence=ambient_valence,
                access_day_count=day_n,
            ):
                w = compute_weighted_score(
                    ac, la,
                    valence_tag=v_tag, valence_score=v_score,
                    query_valence=ambient_valence,
                    access_day_count=day_n,
                )
                candidates.append({
                    "id":               mem_id,
                    "memory":           m.get("memory", "")[:120],
                    "access_count":     ac,
                    "weighted_score":   round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        candidates.sort(key=lambda x: x["weighted_score"])
        return kept, candidates

    def rebalance_pins(
        self,
        user_id: str,
        *,
        max_age_days: int = 45,
        min_access_count: int = 2,
        dry_run: bool = False,
    ) -> dict:
        """Make ordinary pinned memories forgettable without deleting them.

        Pins are a preservation hint, not a guarantee of eternal storage.
        Identity, safety, and explicit durable rules stay pinned; old rows
        that have never (or barely) been recalled are simply unpinned so the
        normal decay/cleanup path can evaluate them later.
        """
        uid = str(user_id or "")
        try:
            age_days = max(1, int(max_age_days))
        except (TypeError, ValueError):
            age_days = 45
        try:
            access_floor = max(0, int(min_access_count))
        except (TypeError, ValueError):
            access_floor = 2
        now = datetime.now(timezone.utc)
        protected = re.compile(
            r"\b(my name|i am|i\x27m|birthday|allerg|medical|safety|emergency|"
            r"from now on|always|never|do not forget|don\x27t forget|remember this)\b",
            re.IGNORECASE,
        )
        candidates: list[dict] = []
        for row in self.get_all(user_id=uid):
            if not int(row.get("pinned") or 0):
                continue
            text = str(row.get("memory") or "")
            if protected.search(text):
                continue
            if str(row.get("status") or "active").casefold() not in {"", "active"}:
                continue
            try:
                accesses = int(row.get("access_count") or 0)
            except (TypeError, ValueError):
                accesses = 0
            if accesses >= access_floor:
                continue
            stamp = row.get("last_accessed_at") or row.get("created_at") or ""
            if str(stamp).casefold() == "never":
                stamp = row.get("created_at") or ""
            try:
                touched = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                if touched.tzinfo is None:
                    touched = touched.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            if (now - touched).total_seconds() < age_days * 86400:
                continue
            candidates.append({"id": str(row.get("id") or ""), "memory": text[:120], "age_days": round((now - touched).total_seconds() / 86400, 1), "access_count": accesses})

        if not dry_run and candidates:
            ids = [item["id"] for item in candidates if item["id"]]
            placeholders = ",".join("?" for _ in ids)
            with self._mem._db_lock:
                self._conn.execute(
                    f"UPDATE memories SET pinned = 0 WHERE user_id = ? AND id IN ({placeholders})",
                    [uid, *ids],
                )
                self._conn.commit()
        return {"unpinned": 0 if dry_run else len(candidates), "candidates": candidates, "protected": True}

    def memory_health(self, user_id: str | None = None, *, max_age_days: int = 45, min_access_count: int = 2) -> dict:
        """Return retention diagnostics without modifying memory rows."""
        uid = self._resolve_user_id(user_id)
        rows = self.get_all(user_id=uid)
        pinned = sum(1 for row in rows if int(row.get("pinned") or 0))
        candidates = self.rebalance_pins(uid, max_age_days=max_age_days, min_access_count=min_access_count, dry_run=True)
        total = len(rows)
        return {
            "total": total,
            "pinned": pinned,
            "pinned_ratio": round(pinned / total, 3) if total else 0.0,
            "eligible_for_unpin": len(candidates.get("candidates", [])),
            "healthy_pin_ratio": pinned == 0 or pinned / max(total, 1) <= 0.25,
        }

    def optimize(self) -> None:
        """Run SQLite PRAGMA optimize to rebuild query planner statistics."""
        with self._mem._db_lock:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.commit()
            except Exception as e:
                log.debug(f"SQLite optimize skipped: {e}")

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str | None = None) -> list[dict]:
        """Return all stored memories for a user."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_all(user_id=user_id)

    def add_raw(self, memory: str, user_id: str | None = None, *, pinned: bool = False, metadata: dict | None = None) -> str | None:
        """Persist one already-curated memory string without LLM extraction."""
        # metadata is accepted for call-site clarity; the current schema stores
        # only the curated text plus pinned flag.
        user_id = self._resolve_user_id(user_id)
        mem_id = self._mem.add_raw(memory, user_id=user_id, pinned=pinned)
        if mem_id:
            self._maybe_clear_search_cache()
            self._mem._invalidate_entity_importance(user_id)
        return mem_id

    def supersede_exact(self, memory_id: str, replacement: str, user_id: str | None = None) -> str | None:
        """Supersede exactly one confirmed memory row and preserve lineage."""
        uid = self._resolve_user_id(user_id)
        new_id = self._mem.supersede_exact(memory_id, replacement, uid)
        if new_id:
            self._clear_search_cache()
            self._mem._invalidate_entity_importance(uid)
        return new_id

    def backfill_missing_vectors(self, user_id: str | None = None, *, limit: int | None = None) -> int:
        """Re-embed any memories rows persisted without a vector (embedder
        outage). Returns count repaired. Useful as a manual repair or from a
        scheduled job; write-path calls trigger it automatically too."""
        user_id = self._resolve_user_id(user_id)
        n = self._mem.backfill_missing_vectors(user_id, limit=limit)
        if n:
            self._clear_search_cache()
        return n

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_since(since, user_id=user_id)

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created in [start, end), oldest first."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_between(start, end, user_id=user_id)
      
    def get_lineage(self, mem_id: str, user_id: str | None = None) -> dict:
        """Return the full supersession lineage chain for a memory."""
        from cognition.memory.lineage import walk_supersession_lineage
        uid = user_id or self.get_user_id()
        store = self._mem
        return walk_supersession_lineage(store, mem_id, user_id=uid)
      
    def delete(self, memory_id: str) -> None:
        """Delete one memory from the store and clear search cache."""
        self._mem.delete(memory_id)
        self._clear_search_cache()


    def clear(self, user_id: str | None = None) -> None:
        """Wipe all memories for a user. Use carefully."""
        user_id = self._resolve_user_id(user_id)
        self._mem.delete_all(user_id=user_id)
        self._clear_search_cache()
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """Batch retrieve access_count + last_accessed_at in a single query."""
        return _sqlite_batch_get_payloads(self._conn, mem_ids)

    def embed_text(self, text: str, *, query: bool = False) -> list[float]:
        """Embed one text string with the configured memory embedding model."""
        return self._mem._embed(text, query=query)

    def embed_texts(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed multiple strings with the configured memory embedding model."""
        if query:
            return self._mem._embedder.embed_queries(texts).tolist()   # applies instruct prefix
        return self._mem._embed_batch(texts)                           # document side — no prefix


format_for_context = AikoMemorize.format_for_context

# Back-compat aliases — tests import public names but engine uses underscore variants
MemoryBackend = _MemoryBackend  # type: ignore
is_trivial_input = _is_trivial_input  # type: ignore
sanitize_fts_query = _sanitize_fts_query  # type: ignore
memory_normalize_text = _normalize_memory_text  # type: ignore
first_json_array = _first_json_array  # type: ignore

__all__ = [
    "AikoMemorize",
    "BOOT_LABELS",
    "EMBED_DIMS",
    "MEMORY_LIFECYCLE_BATCH_SIZE",
    "MEMORY_RECALL_SCORE_THRESHOLD",
    "MEMORY_RECENCY_RERANK_THRESHOLD",
    "MEMORY_SEARCH_CACHE_TTL",
    "MEMORY_WRITE_IDLE_GRACE",
    "MEMORY_WRITE_MAX_WAIT",
    "PERSONA_CACHE_TTL",
    "WRITE_DEDUP_THRESHOLD",
    "_MemoryBackend",
    "MemoryBackend",
    "is_trivial_input",
    "sanitize_fts_query",
    "memory_normalize_text",
    "first_json_array",
    "_PHASE_A_COLUMNS",
    "_first_json_array",
    "_is_trivial_input",
    "_normalize_memory_text",
    "_sanitize_fts_query",
    "backfill_entities",
    "classify_kind",
    "classify_write_op",
    "entities_from_json",
    "entities_to_json",
    "entity_overlap_score",
    "ensure_entity_relations_schema",
    "ensure_episode_schema",
    "ensure_l2_scene_schema",
    "ensure_phase_a_schema",
    "existing_columns",
    "extract_entities",
    "format_for_context",
    "infer_salience_hit",
    "infer_valence_score",
    "infer_valence_tag",
    "normalize_memory_text",
    "rebuild_entity_relations",
    "SALIENCE_POLICY_RE",
    "tag_from_score",
    "upsert_co_mentions",
    "vacuum_memory_db",
]

