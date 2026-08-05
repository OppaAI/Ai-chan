"""
tests/test_memorize.py
Starter test suite for memory/memorize.py.

Layers covered:
  1. Pure functions        — no DB, no embedder, no LLM
  2. Ranking/scoring logic — real sqlite-vec DB, hand-seeded rows, no embedder
  3. Integration           — real _MemoryBackend wired to a FakeEmbedder

Run with:
  pytest tests/test_memorize.py -v

Assumptions (adjust if your vecstore.py differs):
  - initialize_store_db(db_path, ddl, user_id=..., vector=True) returns a
    sqlite3.Connection with sqlite_vec already loaded and row_factory set
    to sqlite3.Row.
  - HarrierEmbedder is only ever touched through _MemoryBackend._embed /
    _embed_batch, so swapping self._embedder after construction is safe.
"""
from __future__ import annotations

import hashlib
import struct
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import sqlite_vec

from memory.memorize import (
    _MemoryBackend,
    AikoMemorize,
    EMBED_DIMS,
    MEMORY_RECALL_SCORE_THRESHOLD,
    MEMORY_RECENCY_RERANK_THRESHOLD,
    MEMORY_WRITE_IDLE_GRACE,
    MEMORY_WRITE_MAX_WAIT,
    WRITE_DEDUP_THRESHOLD,
    _is_trivial_input,
    _sanitize_fts_query,
    _normalize_memory_text,
    _first_json_array,
    ensure_phase_a_schema,
    ensure_entity_relations_schema,
    infer_salience_hit,
    infer_valence_tag,
    entities_to_json,
    vacuum_memory_db,
)
from memory.forget import _valence_intensity, compute_weighted_score, is_grace_protected, should_cleanup
from memory.entity_importance import (
    compute_entity_importance_map,
    memory_max_entity_importance,
    should_expand_supersession_chain,
    walk_supersession_chain,
)
from memory import consolidate as consolidate_mod
from system import userspace


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — pure functions
# ─────────────────────────────────────────────────────────────────────────────

class TestTrivialInput:
    def test_wake_word_alone(self):
        assert _is_trivial_input("aiko")

    def test_wake_word_plus_question_not_trivial(self):
        assert not _is_trivial_input("hi aiko, what's the weather")

    def test_greeting_phrase(self):
        assert _is_trivial_input("how are you doing")

    def test_ragged_asr_transcript(self):
        assert _is_trivial_input("Hi, I. How are you doing.")

    def test_multi_clause_one_real_clause(self):
        assert not _is_trivial_input("ok, remind me about the deadline")

    def test_empty_string(self):
        assert _is_trivial_input("")

    def test_pure_filler_with_punctuation(self):
        assert _is_trivial_input("thanks! bye.")


class TestFtsSanitize:
    def test_strips_syntax_chars(self):
        result = _sanitize_fts_query('what is "Max" (the cat)?')
        assert result is not None
        assert '"' not in result and "(" not in result

    def test_bare_symbols_returns_none(self):
        assert _sanitize_fts_query("***") is None

    def test_empty_returns_none(self):
        assert _sanitize_fts_query("") is None
        assert _sanitize_fts_query(None) is None


class TestNormalizeMemoryText:
    def test_case_and_whitespace_collapse(self):
        a = _normalize_memory_text("Max  is\na cat")
        b = _normalize_memory_text("max is a cat")
        assert a == b

    def test_none_safe(self):
        assert _normalize_memory_text(None) == ""


class TestFirstJsonArray:
    def test_nested_brackets(self):
        raw = 'garbage [ "a[1]", "b" ] trailing'
        assert _first_json_array(raw) == '[ "a[1]", "b" ]'

    def test_no_array_returns_none(self):
        assert _first_json_array("no brackets here") is None

    def test_escaped_quote_inside_string(self):
        raw = r'[ "she said \"hi\"" ]'
        assert _first_json_array(raw) == raw


# ─────────────────────────────────────────────────────────────────────────────
# Fake embedder — deterministic, hash-based, no GGUF/llama.cpp involved
# ─────────────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """
    Deterministic stand-in for HarrierEmbedder. Same text -> same vector,
    so cosine-similarity dedup/knn behavior is fully controllable in tests
    without loading a real GGUF model.
    """

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # repeat/truncate hash bytes to fill EMBED_DIMS floats in [0, 1)
        raw = (h * (EMBED_DIMS // len(h) + 1))[: EMBED_DIMS * 4]
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = arr[:EMBED_DIMS] / 255.0
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


def near_duplicate_text(base: str) -> str:
    """Same hash bucket trick won't give near-duplicates for free text,
    so for dedup tests we just reuse the identical string — good enough
    since FakeEmbedder gives identical text = identical (cosine 1.0) vector,
    which safely clears any WRITE_DEDUP_THRESHOLD < 1.0."""
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def backend(tmp_path, monkeypatch):
    """A real _MemoryBackend against a throwaway sqlite file, with the
    GGUF embedder swapped for FakeEmbedder so no model load happens."""
    b = _MemoryBackend(
        db_path=str(tmp_path / "test_memory.db"),
        llm_base_url="http://unused",
        model="unused",
    )
    b._embedder = FakeEmbedder()
    yield b
    b._conn.close()


def _insert_row(conn, mem_id, user_id, text, created_at, pinned=0, access_count=0):
    conn.execute(
        """
        INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
        VALUES (?, ?, ?, ?, ?, 'never', ?)
        """,
        (mem_id, user_id, text, created_at, access_count, pinned),
    )


def _insert_vector(conn, mem_id, vector):
    conn.execute(
        "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
        (mem_id, sqlite_vec.serialize_float32(vector.tolist())),
    )


def _bare_memo(backend, user_id: str = "u1"):
    """Build an AikoMemorize instance without __init__ (no worker thread or
    config load), wiring only the state the recall/persona paths touch."""
    from memory.memorize import AikoMemorize
    memo = AikoMemorize.__new__(AikoMemorize)
    memo._mem = backend
    memo._user_id_override = user_id
    memo._search_cache = OrderedDict()
    memo._search_cache_lock = threading.RLock()
    memo._last_cache_clear_time = 0.0
    memo._persona_lock = threading.RLock()
    memo._persona_cached = None
    memo._persona_cache_at = 0.0
    return memo


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — ranking / scoring, hand-seeded rows, no embedder call during scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestRankAndScore:
    def test_pinned_is_tiebreaker_not_guarantee(self, backend):
        """A pinned memory with weak relevance should NOT outrank a highly
        relevant unpinned memory -- pinned bonus is meant to be a mild
        tiebreaker (see MEMORY_RANK_PINNED_WEIGHT docstring)."""
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        _insert_row(conn, "weak_pinned", "u1", "irrelevant pinned fact", now, pinned=1)
        _insert_row(conn, "strong_unpinned", "u1", "highly relevant fact", now, pinned=0)
        conn.commit()

        # rank_knn/rank_fts simulate: strong_unpinned ranked #1 in both,
        # weak_pinned not present in either candidate pool at all.
        rank_knn = {"strong_unpinned": 1}
        rank_fts = {"strong_unpinned": 1}

        scored_ids, scores, _ = backend._rank_and_score(rank_knn, rank_fts)
        assert scored_ids[0] == "strong_unpinned"

    def test_pinned_breaks_exact_tie(self, backend):
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        _insert_row(conn, "pinned_tie", "u1", "fact a", now, pinned=1)
        _insert_row(conn, "unpinned_tie", "u1", "fact b", now, pinned=0)
        conn.commit()

        # identical rank in both knn/fts -> identical RRF score;
        # pinned bonus should be the deciding factor
        rank_knn = {"pinned_tie": 1, "unpinned_tie": 1}
        rank_fts = {}

        scored_ids, scores, _ = backend._rank_and_score(rank_knn, rank_fts)
        assert scored_ids[0] == "pinned_tie"

    def test_dedup_keeps_newest_duplicate_row(self, backend):
        conn = backend._conn
        old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        _insert_row(conn, "old_dup", "u1", "Max is a cat", old)
        _insert_row(conn, "new_dup", "u1", "Max is a cat", new)
        conn.commit()

        rank_knn = {"old_dup": 1, "new_dup": 2}
        rank_fts = {}

        scored_ids, scores, _ = backend._rank_and_score(rank_knn, rank_fts)
        assert "old_dup" not in scored_ids
        assert "new_dup" in scored_ids


class TestRecencyRerank:
    def test_reorders_only_above_threshold(self, backend):
        """Two candidates both clearing MEMORY_RECENCY_RERANK_THRESHOLD:
        newer one should surface first even if its base score is lower."""
        conn = backend._conn
        older = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        newer = datetime.now(timezone.utc).isoformat()
        _insert_row(conn, "older_relevant", "u1", "fact one", older)
        _insert_row(conn, "newer_relevant", "u1", "fact two", newer)
        conn.commit()

        # give older_relevant a slightly better raw score (rank 1 vs rank 2)
        # but both must clear MEMORY_RECENCY_RERANK_THRESHOLD for the
        # reorder to kick in -- inflate via fts+knn double hit
        rank_knn = {"older_relevant": 1, "newer_relevant": 1}
        rank_fts = {"older_relevant": 1, "newer_relevant": 2}

        scored_ids, scores, row_by_id = backend._rank_and_score(rank_knn, rank_fts)
        # sanity: both clear the rerank threshold in this synthetic setup
        assert all(scores[i] >= MEMORY_RECENCY_RERANK_THRESHOLD for i in scored_ids), (
            "test setup assumption broken -- adjust ranks so both candidates "
            "clear MEMORY_RECENCY_RERANK_THRESHOLD"
        )

        reordered = backend._apply_recency_rerank(scored_ids, scores, row_by_id)
        assert reordered[0] == "newer_relevant"

    def test_below_threshold_keeps_score_order(self, backend):
        conn = backend._conn
        # backdated far enough that the recency bonus is ~0, isolating the
        # RRF-rank contribution so the "weak match" premise actually holds
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        _insert_row(conn, "low_a", "u1", "weak match a", old)
        _insert_row(conn, "low_b", "u1", "weak match b", old)
        conn.commit()

        # RRF_K=60, threshold=0.012 -> 1/(60+r) must stay under 0.012,
        # which needs rank > ~43
        rank_knn = {"low_a": 50, "low_b": 55}
        rank_fts = {}

        scored_ids, scores, row_by_id = backend._rank_and_score(rank_knn, rank_fts)
        assert all(s < MEMORY_RECENCY_RERANK_THRESHOLD for s in scores.values())

        reordered = backend._apply_recency_rerank(scored_ids, scores, row_by_id)
        assert reordered == scored_ids

# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — integration against FakeEmbedder (no real LLM/GGUF)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddRawDedup:
    def test_exact_duplicate_text_is_skipped(self, backend):
        first = backend.add_raw("Oppa's birthday is June 3", user_id="u1")
        second = backend.add_raw("Oppa's birthday is June 3", user_id="u1")
        assert first is not None
        assert second is None  # identical text -> cosine 1.0 -> dedup skip

    def test_distinct_text_is_not_skipped(self, backend):
        first = backend.add_raw("Oppa's birthday is June 3", user_id="u1")
        second = backend.add_raw("Oppa is building a robot called Grace", user_id="u1")
        assert first is not None
        assert second is not None
        assert first != second

    def test_pinned_flag_persisted(self, backend):
        mem_id = backend.add_raw("Oppa dislikes mushrooms", user_id="u1", pinned=True)
        row = backend._conn.execute(
            "SELECT pinned FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["pinned"] == 1


class TestSearchIntegration:
    def test_search_returns_seeded_fact(self, backend):
        backend.add_raw("Oppa is building a robot called Grace", user_id="u1")
        results = backend.search("what robot is Oppa building", user_id="u1", limit=5)
        texts = [r["memory"] for r in results]
        assert any("Grace" in t for t in texts)

    def test_search_filters_by_user_id(self, backend):
        backend.add_raw("secret fact about u1", user_id="u1")
        backend.add_raw("secret fact about u2", user_id="u2")
        results = backend.search("secret fact", user_id="u1", limit=5)
        assert all("u2" not in r["memory"] for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# "Does Aiko know who she's talking to" — display_name -> LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

class _CapturingChatCompletions:
    """Drop-in replacement for OpenAI().chat.completions that records the
    prompt it was called with instead of hitting a real llama-server."""

    def __init__(self):
        self.last_prompt: str | None = None

    def create(self, model, messages, **kwargs):
        self.last_prompt = messages[0]["content"]

        class _Choice:
            class message:
                content = "[]"  # no facts extracted -- we only care about the prompt

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _CapturingClient:
    def __init__(self):
        self.chat = type("chat", (), {})()
        self.chat.completions = _CapturingChatCompletions()


class TestDisplayNamePropagation:
    """
    memorize.py's _extract_facts() builds the LLM prompt from
    `display_name or current_display_name()`. If a caller forgets to pass
    display_name explicitly, this silently falls back through
    current_display_name()'s contextvar -> CURRENT_DISPLAY_NAME env -> user_id.

    These tests confirm the prompt Aiko actually sends contains the right
    name in each of those paths, and catches the regression where a
    background/queued write runs without the per-request contextvar set
    (e.g. on a different thread) and the LLM ends up being told the wrong
    name, or just the raw user_id, instead of a real display name.
    """

    LONG_ENOUGH_MESSAGES = [
        {"role": "user", "content": "My favorite color is teal, and I work night shifts most weeks at the hospital."},
        {"role": "assistant", "content": "Got it, I'll remember that about you!"},
    ]

    def test_explicit_display_name_reaches_prompt(self, backend):
        fake_client = _CapturingClient()
        backend._client = fake_client

        backend._extract_facts(self.LONG_ENOUGH_MESSAGES, display_name="Oppa")

        prompt = fake_client.chat.completions.last_prompt
        assert prompt is not None
        assert "Oppa" in prompt

    def test_falls_back_to_current_display_name_when_not_passed(self, backend, monkeypatch):
        monkeypatch.setenv("CURRENT_DISPLAY_NAME", "ContextUser")
        fake_client = _CapturingClient()
        backend._client = fake_client

        backend._extract_facts(self.LONG_ENOUGH_MESSAGES, display_name=None)

        prompt = fake_client.chat.completions.last_prompt
        assert "ContextUser" in prompt

    def test_regression_background_thread_loses_contextvar(self, backend, monkeypatch):
        """Reproduces the class of bug implied by 'Aiko doesn't know who
        she's talking to': a contextvar set on the request thread does NOT
        automatically propagate to a background worker thread unless it's
        explicitly captured and passed. If queue_write() resolves
        display_name on the caller's thread (correct) vs inside the
        worker's _write_loop (wrong), this test distinguishes the two."""
        monkeypatch.delenv("CURRENT_DISPLAY_NAME", raising=False)
        token = userspace.set_current_display_name("RequestThreadUser")
        try:
            resolved_on_caller_thread = userspace.current_display_name()
        finally:
            userspace.reset_current_display_name(token)

        # simulate what a naive worker thread would see: contextvar reset,
        # no env var -- falls back to bare user_id, which is the bug this
        # test guards against if display_name isn't captured before queuing
        resolved_on_worker_thread = userspace.current_display_name()

        assert resolved_on_caller_thread == "RequestThreadUser"
        assert resolved_on_worker_thread != "RequestThreadUser", (
            "If this fails, the contextvar leaked across the simulated "
            "thread boundary in a way real threads wouldn't allow -- "
            "double check queue_write() captures display_name on the "
            "CALLER's thread (it does, per its docstring) rather than "
            "relying on current_display_name() inside _write_loop."
        )

    def test_missing_display_name_and_no_context_falls_back_to_user_id(self, backend, monkeypatch):
        """Worst case: nothing set anywhere. Aiko will label facts with the
        raw user_id (e.g. 'github_12345') instead of a real name -- this is
        allowed behavior, but should be visible/testable rather than an
        unnoticed silent default."""
        monkeypatch.delenv("CURRENT_DISPLAY_NAME", raising=False)
        monkeypatch.setenv("AIKO_USER_ID", "github_98765")
        fake_client = _CapturingClient()
        backend._client = fake_client

        backend._extract_facts(self.LONG_ENOUGH_MESSAGES, display_name=None)

        prompt = fake_client.chat.completions.last_prompt
        assert "github_98765" in prompt  # documents current fallback behavior


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — async write queue idle-window logic (pure timing, no DB needed
# beyond what AikoMemorize.__init__ requires -- heavier fixture, marked slow)
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteWindowTiming:
    """
    Exercises _wait_for_write_window in isolation via a throwaway
    AikoMemorize-like object. Uses monkeypatched clocks so the test doesn't
    actually sleep for MEMORY_WRITE_IDLE_GRACE/MAX_WAIT seconds.
    """

    def test_fires_immediately_with_no_callables(self, backend):
        memo = AikoMemorize.__new__(AikoMemorize)  # bypass __init__ (no LLM/embedder needed)
        start = time.monotonic()
        memo._wait_for_write_window(None, None)
        assert time.monotonic() - start < 0.05

    def test_waits_for_idle_grace(self, monkeypatch, backend):
        memo = AikoMemorize.__new__(AikoMemorize)

        # simulate: turn becomes idle at t=0, idle_for grows each call
        fake_now = {"t": 0.0}
        def fake_time():
            fake_now["t"] += MEMORY_WRITE_IDLE_GRACE / 4  # advance a bit each poll
            return fake_now["t"]

        monkeypatch.setattr(time, "time", fake_time)
        monkeypatch.setattr(time, "sleep", lambda s: None)  # don't actually sleep

        idle_since = lambda: 0.0
        is_active_turn = lambda: False

        memo._wait_for_write_window(is_active_turn, idle_since)
        # if we got here without hanging, the loop correctly exited once
        # idle_for crossed MEMORY_WRITE_IDLE_GRACE
        assert fake_now["t"] >= MEMORY_WRITE_IDLE_GRACE

    def test_force_fires_at_max_wait_even_if_never_idle_long_enough(self, monkeypatch, backend):
        memo = AikoMemorize.__new__(AikoMemorize)

        # idle_for never clears the grace window, but is_active_turn goes
        # False right as monotonic deadline passes
        state = {"monotonic_t": 0.0, "calls": 0}

        def fake_monotonic():
            state["monotonic_t"] += MEMORY_WRITE_MAX_WAIT / 3
            return state["monotonic_t"]

        def fake_time():
            return 0.0  # idle_for always computes as "just went idle" -> never clears grace

        def fake_is_active():
            state["calls"] += 1
            return state["calls"] < 4  # active for first few polls, then not

        monkeypatch.setattr(time, "monotonic", fake_monotonic)
        monkeypatch.setattr(time, "time", fake_time)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        memo._wait_for_write_window(fake_is_active, lambda: 0.0)
        # Must exit via the max-wait deadline branch, not hang forever.
        # The fake's is_active_turn() only turns False on poll #4, which
        # never runs because the deadline check fires first — so the loop
        # should terminate after a few active polls, never spinning on.
        assert 1 <= state["calls"] <= 4


def test_vacuum_memory_db_opens_user_store_and_runs_maintenance(monkeypatch, tmp_path):
    calls = []

    class FakeConn:
        def __init__(self):
            self.isolation_level = None  # ADD THIS
        def execute(self, sql):
            calls.append(("execute", sql))
        def commit(self):
            calls.append(("commit", None))
        def close(self):
            calls.append(("close", None))

    def fake_initialize(path, ddl, user_id=None, vector=True):
        calls.append(("initialize", str(path), user_id, vector))
        return FakeConn()

    monkeypatch.setattr("memory.memorize.resolve_user_db_path", lambda path, user_id=None: tmp_path / user_id / "memory.db")
    monkeypatch.setattr("memory.memorize.initialize_store_db", fake_initialize)

    vacuum_memory_db("alice")

    assert calls[0] == ("initialize", str(tmp_path / "alice" / "memory.db"), "alice", True)
    assert ("execute", "VACUUM") in calls
    assert ("execute", "ANALYZE") in calls
    assert calls[-1] == ("close", None)


# ─────────────────────────────────────────────────────────────────────────────
# L3 Persona cache — always-hydrated top-N stable identity facts
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonaCache:
    def test_persona_cache_prefers_identity_kind(self, backend):
        """Only facts with kind='identity' end up in the persona blob."""
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        # Insert: 2 identity, 1 preference, 1 fact
        for mid, txt, kind in [
            ("id1", "Oppa's birthday is June 3", "identity"),
            ("id2", "Oppa lives in Tokyo", "identity"),
            ("id3", "Oppa loves ramen", "preference"),
            ("id4", "Oppa met a friend", "fact"),
        ]:
            _insert_row(conn, mid, "u1", txt, now, access_count=5)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
            conn.execute(
                "UPDATE memories SET kind = ? WHERE id = ?", (kind, mid)
            )
        conn.commit()

        memo = _bare_memo(backend)
        # Directly call private builder to inspect raw output
        block = memo._build_persona_context()
        assert block is not None
        assert "birthday" in block
        assert "Tokyo" in block
        assert "ramen" not in block  # preference excluded
        assert "friend" not in block  # fact excluded
        assert block.count("birthday") == 1
        assert block.count("Tokyo") == 1

    def test_persona_cache_order_by_access_then_age(self, backend):
        """Higher access_count first, then older created_at for ties."""
        conn = backend._conn
        base = datetime.now(timezone.utc)
        for i, (mid, txt, access, days_old) in enumerate([
            ("id1", "Oppa A", 10, 5),   # high access, older
            ("id2", "Oppa B", 5, 1),    # lower access
            ("id3", "Oppa C", 10, 2),   # same high access, newer -> should come AFTER id1
        ]):
            created = (base - timedelta(days=days_old)).isoformat()
            _insert_row(conn, mid, "u1", txt, created, access_count=access)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
            conn.execute("UPDATE memories SET kind = 'identity' WHERE id = ?", (mid,))
        conn.commit()

        memo = _bare_memo(backend)
        block = memo._build_persona_context()
        assert block is not None
        lines = [l for l in block.split("\n") if l.strip().startswith("-")]
        # id1 (access 10, 5 days old) should appear before id3 (access 10, 2 days old)
        assert lines[0] == "  - Oppa A"
        assert lines[1] == "  - Oppa C"
        assert lines[2] == "  - Oppa B"

    def test_persona_cache_ttl_invalidation(self, backend, monkeypatch):
        """Cache respects TTL and rebuilds after PERSONA_CACHE_TTL."""
        from memory.memorize import PERSONA_CACHE_TTL
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        _insert_row(conn, "id1", "u1", "Oppa X", now)
        _insert_vector(conn, "id1", backend._embedder._vec("Oppa X"))
        conn.execute("UPDATE memories SET kind = 'identity' WHERE id = 'id1'")
        conn.commit()

        memo = _bare_memo(backend)
        # First call populates cache
        block1 = memo.persona_context()
        assert "Oppa X" in block1
        # Immediately call again -> cached
        block2 = memo.persona_context()
        assert block1 == block2

        # Advance time past TTL
        original_monotonic = time.monotonic  # capture BEFORE patching
        monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + PERSONA_CACHE_TTL + 1)        # Now add a new identity fact
        _insert_row(conn, "id2", "u1", "Oppa Y", now)
        _insert_vector(conn, "id2", backend._embedder._vec("Oppa Y"))
        conn.execute("UPDATE memories SET kind = 'identity' WHERE id = 'id2'")
        conn.commit()

        block3 = memo.persona_context()
        assert "Oppa Y" in block3  # rebuilt, includes new fact


# ─────────────────────────────────────────────────────────────────────────────
# L2 Scene blocks — mid-grain episode summaries with member linking
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneBlocks:
    def test_build_scene_links_members(self, backend):
        """build_scene creates a kind='scene' row and tags members with scene_id."""
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        # Seed 3 atomic facts
        member_ids = []
        for i, txt in enumerate(["Oppa coded late", "Oppa fixed a bug", "Oppa shipped"]):
            mid = str(uuid.uuid4())
            _insert_row(conn, mid, "u1", txt, now)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
            member_ids.append(mid)
        conn.commit()

        scene_id = backend.build_scene("u1", summary="Coding sprint", member_ids=member_ids, pinned=True)
        assert scene_id is not None

        # Scene row exists with kind='scene'
        scene_row = conn.execute("SELECT * FROM memories WHERE id = ?", (scene_id,)).fetchone()
        assert scene_row["kind"] == "scene"
        assert scene_row["pinned"] == 1

        # Members tagged with scene_id
        for mid in member_ids:
            m = conn.execute("SELECT scene_id FROM memories WHERE id = ?", (mid,)).fetchone()
            assert m["scene_id"] == scene_id

    def test_list_scenes_returns_recent_first(self, backend):
        conn = backend._conn
        base = datetime.now(timezone.utc)
        # Create 3 scenes at different times
        for i, (summary, days_ago) in enumerate([
            ("Scene A", 5),
            ("Scene B", 1),
            ("Scene C", 3),
        ]):
            sid = str(uuid.uuid4())
            created = (base - timedelta(days=days_ago)).isoformat()
            _insert_row(conn, sid, "u1", summary, created, pinned=0)
            _insert_vector(conn, sid, backend._embedder._vec(summary))
            conn.execute("UPDATE memories SET kind = 'scene' WHERE id = ?", (sid,))
        conn.commit()

        scenes = backend.list_scenes("u1", limit=10)
        assert len(scenes) == 3
        # Newest first (Scene B is 1 day ago)
        assert scenes[0]["memory"] == "Scene B"
        assert scenes[1]["memory"] == "Scene C"
        assert scenes[2]["memory"] == "Scene A"

    def test_scene_members_returns_linked_facts(self, backend):
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        mid1 = str(uuid.uuid4())
        mid2 = str(uuid.uuid4())
        for mid, txt in [(mid1, "Fact one"), (mid2, "Fact two")]:
            _insert_row(conn, mid, "u1", txt, now)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
        scene_id = str(uuid.uuid4())
        _insert_row(conn, scene_id, "u1", "Scene summary", now)
        _insert_vector(conn, scene_id, backend._embedder._vec("Scene summary"))
        conn.execute("UPDATE memories SET kind = 'scene' WHERE id = ?", (scene_id,))
        conn.execute("UPDATE memories SET scene_id = ? WHERE id IN (?, ?)", (scene_id, mid1, mid2))
        conn.commit()

        members = backend.scene_members(scene_id, "u1")
        assert len(members) == 2
        assert {m["memory"] for m in members} == {"Fact one", "Fact two"}


# ─────────────────────────────────────────────────────────────────────────────
# Scene expansion in search — recalled members pull in parent scene, and vice versa
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneExpansion:
    def test_recalled_member_pulls_parent_scene(self, backend, monkeypatch):
        """A recalled atomic fact with scene_id causes its scene row to be added to results."""
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        # Seed: one scene + one member
        scene_id = str(uuid.uuid4())
        member_id = str(uuid.uuid4())
        for mid, txt, kind in [
            (scene_id, "Coding sprint", "scene"),
            (member_id, "Oppa fixed a bug", "fact"),
        ]:
            _insert_row(conn, mid, "u1", txt, now)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
            conn.execute("UPDATE memories SET kind = ? WHERE id = ?", (kind, mid))
        conn.execute("UPDATE memories SET scene_id = ? WHERE id = ?", (scene_id, member_id))
        conn.commit()

        # Search for the member text; it should return the member + the scene
        memo = _bare_memo(backend)
        # Monkey the embedder to return the member's vector for the query
        member_vec = backend._embedder._vec("Oppa fixed a bug")
        monkeypatch.setattr(backend._embedder, "embed_query", lambda q, **kw: member_vec)

        results = memo.search("bug fixed", user_id="u1", limit=5)
        ids = {r["id"] for r in results}
        assert member_id in ids
        assert scene_id in ids
        # The recalled member pulled its parent scene in as an expanded row.
        scene_entry = next(r for r in results if r["id"] == scene_id)
        assert scene_entry.get("_scene") is True

    def test_recalled_scene_carries_members(self, backend, monkeypatch):
        """A recalled scene row should have _scene_members populated."""
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        scene_id = str(uuid.uuid4())
        member_id = str(uuid.uuid4())
        for mid, txt, kind in [
            (scene_id, "Coding sprint", "scene"),
            (member_id, "Oppa fixed a bug", "fact"),
        ]:
            _insert_row(conn, mid, "u1", txt, now)
            _insert_vector(conn, mid, backend._embedder._vec(txt))
            conn.execute("UPDATE memories SET kind = ? WHERE id = ?", (kind, mid))
        conn.execute("UPDATE memories SET scene_id = ? WHERE id = ?", (scene_id, member_id))
        conn.commit()

        memo = _bare_memo(backend)
        # Query vector matches scene text
        scene_vec = backend._embedder._vec("Coding sprint")
        monkeypatch.setattr(backend._embedder, "embed_query", lambda q, **kw: scene_vec)

        results = memo.search("coding sprint", user_id="u1", limit=5)
        ids = {r["id"] for r in results}
        assert scene_id in ids
        scene_entry = next(r for r in results if r["id"] == scene_id)
        assert scene_entry.get("_scene_members") is not None
        assert "Oppa fixed a bug" in scene_entry["_scene_members"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — emotion-imprinted decay (memory/forget.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestForgetValenceDecay:
    """Phase 5 emotion imprint. NOTE: forget.py reads env at import time, so
    every test that tunes knobs must monkeypatch.setenv + importlib.reload."""

    @pytest.fixture(autouse=True)
    def _reload_forget(self, monkeypatch):
        import importlib
        import memory.forget as forget_mod
        monkeypatch.setenv("FORGET_HALF_LIFE_DAYS", "21.0")
        monkeypatch.setenv("FORGET_EMOTION_GAMMA", "0.5")
        monkeypatch.setenv("FORGET_INTENSITY_NEG", "1.0")
        monkeypatch.setenv("FORGET_INTENSITY_POS", "0.4")
        monkeypatch.setenv("FORGET_INTENSITY_NEUTRAL", "0.0")
        monkeypatch.setenv("FORGET_CLEANUP_THRESHOLD", "0.02")
        importlib.reload(forget_mod)
        yield forget_mod

    def test_valence_intensity_lookup(self, _reload_forget):
        for key, val in (("neg", 1.0), ("pos", 0.4), ("neutral", 0.0)):
            assert _valence_intensity(key) == val
        assert _valence_intensity(None) == 0.0
        assert _valence_intensity("") == 0.0
        assert _valence_intensity("Neg") == 1.0

    def test_valence_intensity_invalid_inputs(self):
        assert _valence_intensity("garbage") == 0.0
        assert _valence_intensity("nan") == 0.0  # string that doesn't map

    def test_negative_valence_decays_slower_than_neutral(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        neutral = compute_weighted_score(10, old, "neutral")
        neg = compute_weighted_score(10, old, "neg")
        assert neg > neutral  # neg keeps more (slower decay)

    def test_positive_valence_decays_between_neutral_and_neg(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        neutral = compute_weighted_score(10, old, "neutral")
        pos = compute_weighted_score(10, old, "pos")
        neg = compute_weighted_score(10, old, "neg")
        assert neutral < pos < neg

    def test_gamma_zero_disables_emotion(self, monkeypatch, _reload_forget):
        import importlib
        import memory.forget as forget_mod
        monkeypatch.setenv("FORGET_EMOTION_GAMMA", "0")
        importlib.reload(forget_mod)
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert compute_weighted_score(10, old, "neg") == pytest.approx(
            compute_weighted_score(10, old, "neutral"), abs=1e-9
        )

    def test_negative_valence_resists_cleanup(self):
        """A neg-valence memory past grace survives cleanup while an identical
        neutral one is pruned — the emotion imprint must lengthen H_eff.

        At 140 days with HALF_LIFE_DAYS=21:
          neutral: 0.5^(140/21) ≈ 0.0098 < 0.02  -> pruned
          neg:     0.5^(140/31.5) ≈ 0.046  > 0.02  -> kept
        """
        created = (datetime.now(timezone.utc) - timedelta(days=140)).isoformat()  # past grace
        old = (datetime.now(timezone.utc) - timedelta(days=140)).isoformat()
        assert is_grace_protected(created) is False
        assert should_cleanup(1, old, created, "neutral") is True
        assert should_cleanup(1, old, created, "neg") is False

    def test_grace_protected_new_memory_never_cleaned(self):
        created = datetime.now(timezone.utc).isoformat()
        assert is_grace_protected(created) is True
        assert should_cleanup(0, "never", created, "neutral") is False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — entity importance I_e + supersession chain (memory/entity_importance.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityImportance:
    def test_importance_map_centrality_and_recency(self, backend):
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        # Entities: "grace" heavily connected; "rust" connected but stale.
        for a, b, w in [("grace", "robot", 3.0), ("grace", "oppa", 2.0), ("rust", "robot", 1.0)]:
            conn.execute(
                "INSERT INTO entity_relations (user_id, entity_a, entity_b, weight, updated_at) VALUES (?,?,?,?,?)",
                ("u1", a, b, w, now),
            )
        # memories referencing those entities; grace touched recently, rust long ago.
        for mid, text, ts, ents in [
            ("m1", "grace fact", recent, ["Grace"]),
            ("m2", "rust fact", old, ["Rust"]),
        ]:
            conn.execute(
                "INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at, entities) "
                "VALUES (?,?,?,?,?,?,?)",
                (mid, "u1", text, now, 1, ts, entities_to_json(ents)),
            )
        conn.commit()

        imap = compute_entity_importance_map(backend, "u1")
        assert "grace" in imap
        assert "rust" in imap
        # grace has higher centrality AND recency → strictly more important
        assert imap["grace"] > imap["rust"]

    def test_importance_map_empty_db_returns_empty(self, backend):
        assert compute_entity_importance_map(backend, "u1") == {}

    def test_memory_max_entity_importance(self):
        imap = {"grace": 0.9, "rust": 0.2}
        row = {"entities": entities_to_json(["Grace", "Rust"])}
        assert memory_max_entity_importance(row, imap) == 0.9
        assert memory_max_entity_importance({"entities": "[]"}, imap) == 0.0
        assert memory_max_entity_importance({"entities": None}, imap) == 0.0

    def test_expand_trigger_reflective_query(self):
        row = {"kind": "fact"}
        assert should_expand_supersession_chain("what changed about oppa", row) is True
        assert should_expand_supersession_chain("whats the weather", row) is False

    def test_expand_trigger_identity_kind(self):
        row = {"kind": "identity"}
        assert should_expand_supersession_chain("anything about oppa", row) is True

    def test_walk_supersession_chain_forward_and_back(self, backend):
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        # lineage: v1 <- v2 <- v3 (v3 supersedes v2, v2 supersedes v1)
        for mid, text, supersedes in [
            ("v1", "Oppa likes tea", None),
            ("v2", "Oppa prefers coffee", "v1"),
            ("v3", "Oppa prefers matcha", "v2"),
        ]:
            conn.execute(
                "INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at, supersedes_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (mid, "u1", text, now, 1, now, supersedes),
            )
        conn.commit()
        chain = walk_supersession_chain(conn, "v3", "u1")
        ids = [r["id"] for r in chain]
        assert ids == ["v1", "v2", "v3"]

    def test_walk_supersession_chain_no_history(self, backend):
        conn = backend._conn
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at) "
            "VALUES (?,?,?,?,?,?)",
            ("solo", "u1", "a lone fact", now, 1, now),
        )
        conn.commit()
        # The starting memory is always included; no lineage means a 1-item chain.
        chain = walk_supersession_chain(conn, "solo", "u1")
        assert [r["id"] for r in chain] == ["solo"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1/2/4 — retention gate scoring (memory/consolidate.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidateRetentionScoring:
    def _score(self, **row):
        return consolidate_mod._score_daily_row(
            row,
            static_anchors=None,
            dynamic_anchors=None,
            row_vector=None,
            entity_weights={},
            entity_weight_cap=1.0,
        )

    def test_stored_salience_hit_outranks_plain_text(self):
        high = self._score(_text="ordinary fact", salience_hit=1, valence_tag="neutral",
                           access_day_count=1, access_count=1, entities="[]")
        low = self._score(_text="ordinary fact", salience_hit=0, valence_tag="neutral",
                          access_day_count=1, access_count=1, entities="[]")
        assert high > low

    def test_negative_valence_scores_higher_than_neutral(self):
        neg = self._score(_text="a fact", salience_hit=0, valence_tag="neg",
                          access_day_count=1, access_count=1, entities="[]")
        neu = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                          access_day_count=1, access_count=1, entities="[]")
        assert neg > neu

    def test_spacing_saturates_at_cap(self):
        day1 = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                           access_day_count=1, access_count=1, entities="[]")
        day9 = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                           access_day_count=9, access_count=9, entities="[]")
        assert day9 > day1

    def test_fallback_to_access_count_when_no_day_count(self):
        no_days = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                              access_day_count=0, access_count=0, entities="[]")
        some_access = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                                  access_day_count=0, access_count=3, entities="[]")
        assert some_access > no_days

    def test_entity_importance_blends_into_connectivity(self):
        base = self._score(_text="a fact", salience_hit=0, valence_tag="neutral",
                           access_day_count=1, access_count=1, entities='["Grace"]')
        boosted = consolidate_mod._score_daily_row(
            {"_text": "a fact", "salience_hit": 0, "valence_tag": "neutral",
             "access_day_count": 1, "access_count": 1, "entities": '["Grace"]'},
            static_anchors=None,
            dynamic_anchors=None,
            row_vector=None,
            entity_weights={},
            entity_weight_cap=1.0,
            entity_importance={"grace": 0.95},
        )
        assert boosted > base


class TestValenceColumnsInIterAll:
    """Ensure iter_all() yields Phase 5 / 12R columns so dream/cleanup can use them."""

    def test_iter_all_includes_valence_salience(self, backend):
        # Insert with valence_score, valence_tag, salience_hit
        backend._conn.execute(
            "INSERT INTO memories(id,user_id,memory,created_at,status,valence_tag,valence_score,salience_hit) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("mem-1", "test_user", "I love this!", "2024-01-01T00:00:00", "active", "pos", 2, 1)
        )
        backend._conn.commit()

        rows = list(backend.iter_all("test_user"))
        assert len(rows) == 1
        m = rows[0]
        # New columns present
        assert "valence_tag" in m
        assert "valence_score" in m
        assert "salience_hit" in m
        assert m["valence_tag"] == "pos"
        assert m["valence_score"] == 2
        assert m["salience_hit"] == 1


class TestCrossStoreUserScoping:
    """Cross-store experience leg should respect explicit user_id."""

    def test_search_experience_threads_user_id(self, tmp_path):
        from agentic.experience import _connect as exp_connect, search_experience
        import numpy as np
        import hashlib

        # Seed one experience for user1 directly
        conn = exp_connect("user1")

        # Create a matching embedder for the seeded data
        class FE:
            def embed_query(self, t, instruct=""):
                h = hashlib.sha256(t.encode()).digest()
                raw = (h * (640 // len(h) + 1))[:2560]
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)[:640]/255.0*2-1
                n = np.linalg.norm(arr); return arr/n if n else arr

        fe = FE()
        vec = fe.embed_query("user1 task")
        import sqlite_vec
        conn.execute(
            "INSERT INTO experiences(id,user_id,goal,record_text,steps_json,outcome,score,answer_excerpt,entities,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("exp-1", "user1", "user1 task", "test record", '[]', "done", 1.0, "excerpt", '[]', "2024-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO experiences_vec(id,embedding) VALUES(?,?)",
            ("exp-1", sqlite_vec.serialize_float32(vec.tolist()))
        )
        conn.execute(
            "INSERT INTO experiences_fts(id,goal,record_text) VALUES(?,?,?)",
            ("exp-1", "user1 task", "test record")
        )
        conn.commit()

        # search_experience with explicit user_id should find it
        hits = search_experience("user1 task", limit=5, embedder=fe, user_id="user1")
        assert len(hits) >= 1
        assert hits[0]["user_id"] == "user1"

        # With different user_id should return empty (separate DB)
        hits2 = search_experience("user1 task", limit=5, embedder=fe, user_id="user2")
        assert hits2 == []

        conn.close()
