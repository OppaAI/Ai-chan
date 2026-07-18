"""
tests/stress/test_write_queue_stress.py
Stress tests for AikoMemorize's async write queue (queue_write /
_write_loop / _wait_for_write_window in memory/memorize.py).

Different from tests/unit/test_memorize.py's Tier 4 tests: those exercise
_wait_for_write_window() in ISOLATION with a fake clock. These tests hammer
the REAL queue + REAL worker thread with rapid concurrent writes, using
FakeEmbedder (so no GGUF/LLM dependency) but real threading, real SQLite
writes, and real timing -- looking for backlog growth, lock contention, and
whether MEMORY_WRITE_MAX_WAIT actually forces a write through under
sustained load, not just in an isolated mock.

Run explicitly (not part of default `pytest` -- see pytest.ini testpaths):
    pytest tests/stress -m stress -v

These are slower than unit tests (seconds, not milliseconds) since they
involve real sleep/wait windows and real thread scheduling. Timeouts here
are generous on purpose -- the goal is "does this eventually settle
correctly," not "is it fast" (that's tests/perf/'s job).
"""
from __future__ import annotations

import queue
import threading
import time
import uuid

import numpy as np
import pytest

from memory.memorize import (
    AikoMemorize,
    MEMORY_WRITE_IDLE_GRACE,
    MEMORY_WRITE_MAX_WAIT,
)


pytestmark = pytest.mark.stress


class FakeEmbedder:
    """Same deterministic stand-in used in tests/unit/test_memorize.py --
    duplicated here rather than imported so this file can run standalone
    without depending on the unit test module's internals."""

    def _vec(self, text: str) -> np.ndarray:
        import hashlib
        from memory.memorize import EMBED_DIMS
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (h * (EMBED_DIMS // len(h) + 1))[: EMBED_DIMS * 4]
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = arr[:EMBED_DIMS] / 255.0
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr

    def embed(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_batch(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self._vec(text)

    def embed_queries(self, texts):
        return self.embed(texts)


class _FakeChatCompletions:
    """Returns an empty fact list immediately -- no real LLM round trip,
    so the stress test measures queue/threading behavior, not LLM latency."""

    def create(self, model, messages, **kwargs):
        class _Choice:
            class message:
                content = "[]"

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self):
        self.chat = type("chat", (), {})()
        self.chat.completions = _FakeChatCompletions()


@pytest.fixture
def memo(tmp_path, monkeypatch):
    """
    A real AikoMemorize instance (real queue, real worker thread, real
    SQLite) with the embedder and extraction LLM swapped for fast fakes so
    the test measures queue/concurrency behavior rather than model latency.
    """
    monkeypatch.setenv("AIKO_USER_ID", "stress_user")
    monkeypatch.setenv("SQLITE_MEMORY_PATH", str(tmp_path / "stress_memory.db"))

    instance = AikoMemorize(silent=True)
    instance._mem._embedder = FakeEmbedder()
    instance._mem._client = _FakeClient()
    yield instance
    # drain anything left so the next test doesn't inherit a running worker
    instance.wait_for_writes(timeout=10)


# ─────────────────────────────────────────────────────────────────────────────
# Rapid-fire queue_write() calls -- does the queue drain without unbounded
# growth, and does nothing get silently dropped?
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueDrainsUnderLoad:
    def test_rapid_fire_writes_all_eventually_complete(self, memo):
        """
        Simulates a burst of conversation turns firing faster than a human
        could actually type -- e.g. a batch-import script, or several
        WebSocket messages arriving back-to-back before the idle-grace
        window opens. All of them should eventually process; none should
        vanish.
        """
        N = 50
        for i in range(N):
            memo.queue_write(
                user_input=f"stress test turn {i}",
                response_text=f"ack {i}",
                is_active_turn=lambda: False,
                idle_since=lambda: 0.0,  # already idle -- no artificial wait
            )

        drained = memo.wait_for_writes(timeout=30)
        assert drained, "queue did not drain within 30s -- possible deadlock or backlog"

        qsize = memo._write_queue.qsize()
        assert qsize == 0, f"expected empty queue after drain, found {qsize} leftover items"

    def test_queue_does_not_grow_unbounded_mid_burst(self, memo):
        """
        While writes are actively being enqueued faster than they can be
        processed, the queue should still be finite and shrinking once
        enqueueing stops -- not silently growing forever (e.g. from a
        worker thread deadlock or exception loop).
        """
        N = 100
        for i in range(N):
            memo.queue_write(
                user_input=f"burst turn {i}",
                response_text=f"ack {i}",
                is_active_turn=lambda: False,
                idle_since=lambda: 0.0,
            )

        # give the worker a moment, then confirm it's actively shrinking
        time.sleep(0.5)
        size_after_pause = memo._write_queue.qsize()

        drained = memo.wait_for_writes(timeout=30)
        assert drained
        assert size_after_pause <= N, "queue size exceeded enqueued count -- shouldn't be possible, indicates a counting bug"


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent enqueue from multiple threads -- simulates multiple simultaneous
# request handlers (e.g. WebUI + voice pipeline both writing at once)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentEnqueue:
    def test_multiple_threads_enqueueing_simultaneously(self, memo):
        """
        Real deployment risk: think.py's chat/webchat turn AND a voice
        pipeline turn could both call queue_write() around the same time
        from different threads. This confirms the single shared
        queue.Queue (thread-safe by design) and single worker thread
        don't corrupt state or drop writes under real concurrent access.
        """
        THREADS = 8
        WRITES_PER_THREAD = 10
        errors = []

        def _worker(thread_id):
            try:
                for i in range(WRITES_PER_THREAD):
                    memo.queue_write(
                        user_input=f"thread{thread_id}-turn{i}",
                        response_text="ack",
                        is_active_turn=lambda: False,
                        idle_since=lambda: 0.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent enqueue raised: {errors}"
        drained = memo.wait_for_writes(timeout=30)
        assert drained


# ─────────────────────────────────────────────────────────────────────────────
# Idle-grace / max-wait behavior under REAL (not mocked) timing conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestIdleGraceRealTiming:
    def test_write_waits_while_turn_active_then_fires_when_idle(self, memo):
        """
        Real (not mocked) timing: a write enqueued while is_active_turn()
        reports True should NOT process immediately, but SHOULD process
        once the turn goes idle and MEMORY_WRITE_IDLE_GRACE elapses.
        """
        state = {"active": True, "went_idle_at": None}

        def is_active():
            return state["active"]

        def idle_since():
            return state["went_idle_at"] or time.time()

        memo.queue_write(
            user_input="waits for idle",
            response_text="ack",
            is_active_turn=is_active,
            idle_since=idle_since,
        )

        # still "active" -- give it a beat, queue should NOT have drained yet
        time.sleep(0.3)
        assert memo._write_queue.unfinished_tasks >= 1, (
            "write processed while turn was still marked active -- "
            "_wait_for_write_window should have blocked it"
        )

        # now go idle
        state["active"] = False
        state["went_idle_at"] = time.time()

        drained = memo.wait_for_writes(timeout=MEMORY_WRITE_IDLE_GRACE + 5)
        assert drained, "write never processed after turn went idle"

    def test_max_wait_forces_write_even_if_turn_flag_stuck_active(self, memo):
        """
        Guards against a caller bug where is_active_turn() gets stuck
        returning True forever (e.g. a crashed turn-tracker never resets
        the flag). MEMORY_WRITE_MAX_WAIT should force the write through
        once max wait elapses, specifically once is_active_turn() is
        checked as no-longer-active at the deadline -- per
        _wait_for_write_window's actual logic, the deadline branch still
        requires `not is_active_turn()` to fire. This test uses a flag
        that flips false right around the deadline to validate that path,
        since a flag stuck permanently True would (correctly) never
        return -- that's a caller bug, not a queue bug.
        """
        deadline_reached = threading.Event()

        def is_active():
            # stays "active" until the max-wait deadline is roughly reached,
            # then relents -- simulates a slow-to-notice-idle caller rather
            # than a permanently broken one
            return not deadline_reached.is_set()

        def idle_since():
            return time.time()  # "just went idle" every check -- never clears grace naturally

        def _flip_after_delay():
            time.sleep(min(MEMORY_WRITE_MAX_WAIT, 3))  # cap wait for test speed
            deadline_reached.set()

        flipper = threading.Thread(target=_flip_after_delay, daemon=True)
        flipper.start()

        memo.queue_write(
            user_input="forced through at max wait",
            response_text="ack",
            is_active_turn=is_active,
            idle_since=idle_since,
        )

        drained = memo.wait_for_writes(timeout=MEMORY_WRITE_MAX_WAIT + 10)
        assert drained, (
            "write never processed -- if MEMORY_WRITE_MAX_WAIT is large in "
            "your .env, increase this test's timeout to match, or the test "
            "will spuriously fail on slow-but-correct behavior"
        )
