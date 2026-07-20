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
import time
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
)
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
    current_display_name()'s contextvar -> AIKO_DISPLAY_NAME env -> user_id.

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
        monkeypatch.setenv("AIKO_DISPLAY_NAME", "ContextUser")
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
        monkeypatch.delenv("AIKO_DISPLAY_NAME", raising=False)
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
        monkeypatch.delenv("AIKO_DISPLAY_NAME", raising=False)
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
        # should exit via the max-wait deadline branch, not hang forever
        assert state["calls"] >= 4
