"""
tests/integration/test_multiuser_isolation.py
Integration tests confirming two users' memories, contexts, and paths never
cross-contaminate -- exercised through REAL components wired together
(system.userspace contextvars + AikoMemorize + real SQLite files), not
mocked at the unit level.

Different from tests/unit/test_userspace.py and test_memorize.py: those
test each module in isolation. These tests simulate two concurrent "users"
(e.g. two OAuth sessions, or a WebUI request + a background voice-pipeline
turn) and confirm the INTEGRATION between userspace's per-request context
and memorize's per-user SQLite store actually holds under realistic
concurrent access -- since a bug here would mean user A sees user B's
memories, which is the worst possible failure mode for a personal-memory
system.

Run explicitly (not part of default `pytest` -- see pytest.ini testpaths):
    pytest tests/integration -m integration -v
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from memory.memorize import AikoMemorize, EMBED_DIMS
from system import userspace


pytestmark = pytest.mark.integration


class FakeEmbedder:
    """Deterministic stand-in -- isolation testing doesn't need real
    embedding quality, just real plumbing (contextvars + SQLite files)."""

    def _vec(self, text: str) -> np.ndarray:
        import hashlib
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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AIKO_USER_ID", "AIKO_DISPLAY_NAME", "USER_STATE_ROOT", "SQLITE_MEMORY_PATH"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_STATE_ROOT", str(tmp_path))
    return tmp_path


def _make_memo_for_user(user_id: str) -> AikoMemorize:
    """
    Build a real AikoMemorize scoped to a specific user, the way it would
    actually be constructed per-request/session (uid resolved from
    userspace at __init__ time), with fakes swapped in for embedder/LLM
    only -- everything else (path resolution, SQLite file, contextvar
    reads) is real.
    """
    token = userspace.set_current_user_id(user_id)
    try:
        instance = AikoMemorize(silent=True)
    finally:
        userspace.reset_current_user_id(token)
    instance._mem._embedder = FakeEmbedder()
    instance._mem._client = _FakeClient()
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# Two real users, two real SQLite files, real path resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoUsersFullyIsolated:
    def test_users_get_different_db_files_on_disk(self, state_root):
        memo_a = _make_memo_for_user("user_alice")
        memo_b = _make_memo_for_user("user_bob")

        assert memo_a._mem._db_path != memo_b._mem._db_path
        assert userspace.user_state_dir("user_alice") != userspace.user_state_dir("user_bob")

    def test_writes_by_one_user_never_appear_in_others_search(self, state_root):
        memo_a = _make_memo_for_user("user_alice")
        memo_b = _make_memo_for_user("user_bob")

        memo_a.add_raw("Alice's secret project is called Nightingale", user_id="user_alice")
        memo_b.add_raw("Bob's secret project is called Falcon", user_id="user_bob")

        alice_results = memo_a.search("secret project", user_id="user_alice", limit=5)
        bob_results = memo_b.search("secret project", user_id="user_bob", limit=5)

        alice_texts = " ".join(r["memory"] for r in alice_results)
        bob_texts = " ".join(r["memory"] for r in bob_results)

        assert "Nightingale" in alice_texts
        assert "Falcon" not in alice_texts
        assert "Falcon" in bob_texts
        assert "Nightingale" not in bob_texts

    def test_shared_backend_instance_still_filters_by_user_id_param(self, state_root):
        """
        Even if a single _MemoryBackend/connection were shared across users
        (e.g. a bug where switch_user() wasn't called correctly), search()
        and add() take user_id as an explicit SQL filter parameter -- this
        confirms that filter actually holds when both users' rows live in
        the SAME physical table, which is the scenario a contextvar mixup
        would produce.
        """
        memo = _make_memo_for_user("shared_backend_user")
        memo.add_raw("fact belonging to user_x", user_id="user_x")
        memo.add_raw("fact belonging to user_y", user_id="user_y")

        results_x = memo.search("fact belonging", user_id="user_x", limit=5)
        results_y = memo.search("fact belonging", user_id="user_y", limit=5)

        assert all("user_x" in r["memory"] for r in results_x)
        assert all("user_y" in r["memory"] for r in results_y)
        assert not any("user_y" in r["memory"] for r in results_x)
        assert not any("user_x" in r["memory"] for r in results_y)


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent access -- two "sessions" active at the same time on different
# threads, contextvars must not bleed across them
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentSessionIsolation:
    def test_contextvar_does_not_leak_across_threads(self, state_root):
        """
        Simulates two concurrent request-handler threads (e.g. two WebUI
        tabs, or WebUI + voice pipeline) each setting their own user
        context. Confirms system.userspace's contextvar is properly
        thread-local -- a real risk if any code path accidentally reads a
        module-level global instead of the contextvar.
        """
        results = {}

        def _handler(user_id, display_name):
            token_u = userspace.set_current_user_id(user_id)
            token_d = userspace.set_current_display_name(display_name)
            try:
                time.sleep(0.05)  # give the other thread a chance to interleave
                results[user_id] = {
                    "seen_user_id": userspace.current_user_id(),
                    "seen_display_name": userspace.current_display_name(),
                }
            finally:
                userspace.reset_current_display_name(token_d)
                userspace.reset_current_user_id(token_u)

        t1 = threading.Thread(target=_handler, args=("user_alpha", "Alpha"))
        t2 = threading.Thread(target=_handler, args=("user_beta", "Beta"))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["user_alpha"]["seen_user_id"] == "user_alpha"
        assert results["user_alpha"]["seen_display_name"] == "Alpha"
        assert results["user_beta"]["seen_user_id"] == "user_beta"
        assert results["user_beta"]["seen_display_name"] == "Beta"

    def test_concurrent_writes_from_two_users_both_land_correctly(self, state_root):
        """
        Two users' AikoMemorize.add_raw() calls firing at roughly the same
        time from different threads -- confirms no row ends up attributed
        to the wrong user_id under real thread interleaving (not just
        sequential calls, which the earlier tests already cover).
        """
        memo_a = _make_memo_for_user("concurrent_alice")
        memo_b = _make_memo_for_user("concurrent_bob")
        errors = []

        def _write_a():
            try:
                for i in range(20):
                    memo_a.add_raw(f"alice fact {i}", user_id="concurrent_alice")
            except Exception as e:
                errors.append(("alice", e))

        def _write_b():
            try:
                for i in range(20):
                    memo_b.add_raw(f"bob fact {i}", user_id="concurrent_bob")
            except Exception as e:
                errors.append(("bob", e))

        t1 = threading.Thread(target=_write_a)
        t2 = threading.Thread(target=_write_b)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert not errors, f"concurrent writes raised: {errors}"

        alice_all = memo_a.get_all(user_id="concurrent_alice")
        bob_all = memo_b.get_all(user_id="concurrent_bob")

        assert all("alice fact" in m["memory"] for m in alice_all)
        assert all("bob fact" in m["memory"] for m in bob_all)
        assert len(alice_all) == 20
        assert len(bob_all) == 20


# ─────────────────────────────────────────────────────────────────────────────
# switch_user() -- the explicit re-scoping path (e.g. admin/debug tooling
# that reopens AikoMemorize against a different user's store)
# ─────────────────────────────────────────────────────────────────────────────

class TestSwitchUser:
    def test_switch_user_fully_reopens_to_new_store(self, state_root):
        memo = _make_memo_for_user("initial_user")
        memo.add_raw("belongs to initial user", user_id="initial_user")

        memo.switch_user("second_user")
        memo._mem._embedder = FakeEmbedder()  # re-apply fake after reopen
        memo._mem._client = _FakeClient()
        memo.add_raw("belongs to second user", user_id="second_user")

        assert memo.get_user_id() == "second_user"

        second_user_memories = memo.get_all(user_id="second_user")
        assert any("second user" in m["memory"] for m in second_user_memories)
        assert not any("initial user" in m["memory"] for m in second_user_memories)

    def test_switch_user_does_not_affect_other_open_instance(self, state_root):
        """Guards against a shared-connection-pool style bug where
        switching one AikoMemorize instance's user context somehow affects
        a separate instance still scoped to the original user."""
        memo_a = _make_memo_for_user("user_stays_put")
        memo_a.add_raw("original data", user_id="user_stays_put")

        memo_b = _make_memo_for_user("user_stays_put")
        memo_b.switch_user("someone_else")
        memo_b._mem._embedder = FakeEmbedder()
        memo_b._mem._client = _FakeClient()

        # memo_a should be completely unaffected by memo_b's switch_user call
        assert memo_a.get_user_id() == "user_stays_put"
        still_there = memo_a.get_all(user_id="user_stays_put")
        assert any("original data" in m["memory"] for m in still_there)
