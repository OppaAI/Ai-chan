"""
tests/perf/benchmark_memory.py
Latency/performance benchmarks for memory/memorize.py, intended to run ON
THE JETSON (or whatever the actual target device is) against the REAL
HarrierEmbedder and a REAL (or realistically-sized) sqlite-vec store --
NOT against FakeEmbedder. These are fundamentally different from
test_memorize.py's logic tests: they measure wall-clock time, which is
only meaningful on the real device under real load.

Everything here is marked @pytest.mark.perf and excluded from normal CI --
run it explicitly, only on the device you care about:

    pytest tests/perf/benchmark_memory.py -m perf --benchmark-only

FIRST RUN (no baseline yet) -- just capture numbers and save them:

    pytest tests/perf/benchmark_memory.py -m perf --benchmark-only \\
        --benchmark-save=jetson_orin_nano_baseline

Every run after that -- compare against the saved baseline and fail if a
change regresses the mean by more than 25% (tune this % once you have a
feel for normal run-to-run noise on your hardware):

    pytest tests/perf/benchmark_memory.py -m perf --benchmark-only \\
        --benchmark-compare=jetson_orin_nano_baseline \\
        --benchmark-compare-fail=mean:25%

Baselines are stored under .benchmarks/ by pytest-benchmark itself --
commit that directory (or at least the specific baseline file) to git so
regressions show up in PR review, same idea as your bench_intent_routing.py
results.

Requires: pip install pytest-benchmark --break-system-packages

Things intentionally NOT covered here (see eval/ instead):
  - extraction/recall ACCURACY (this file only measures speed)
  - anything requiring FakeEmbedder (that belongs in test_memorize.py,
    where determinism matters more than realism)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

try:
    from memory.memorize import (
        AikoMemorize,
        MEMORY_LIFECYCLE_BATCH_SIZE,
        MEMORY_SEARCH_CACHE_TTL,
        _MemoryBackend,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip(f"memory.memorize not importable in this environment: {e}", allow_module_level=True)


pytestmark = pytest.mark.perf


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures -- REAL backend, real embedder. No FakeEmbedder in this file.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_backend(tmp_path_factory):
    """
    A real _MemoryBackend with the actual configured HarrierEmbedder and
    (if LLM_BASE_URL is reachable) the real extraction LLM. Skips the whole
    module if the embedder can't load -- e.g. running this on a dev laptop
    without the GGUF model present, or in CI where it doesn't belong.
    """
    db_path = tmp_path_factory.mktemp("perf") / "perf_memory.db"
    try:
        backend = _MemoryBackend(
            db_path=str(db_path),
            llm_base_url="http://localhost:8080/v1",
            model="ministral",
        )
        # touch the embedder once so a load failure surfaces here, not
        # mid-benchmark
        backend._embed("warmup", query=True)
    except Exception as e:
        pytest.skip(
            f"Real embedder/model not available in this environment "
            f"(expected when not running on the actual Jetson): {e}"
        )
    yield backend
    backend._conn.close()


@pytest.fixture(scope="module")
def seeded_backend(real_backend):
    """
    Seed a realistic number of memories so search/dream benchmarks reflect
    actual production scale rather than an empty-table best case. Adjust
    SEED_COUNT to match your real memory.db row count (check with
    `SELECT COUNT(*) FROM memories` on the live Jetson DB) so this stays
    representative as Aiko's memory store grows.
    """
    SEED_COUNT = 500
    now = datetime.now(timezone.utc)
    texts = [
        f"Oppa fact number {i}: this is a synthetic memory used only for "
        f"perf benchmarking, not representative content."
        for i in range(SEED_COUNT)
    ]
    vectors = real_backend._embed_batch(texts)
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        created = (now - timedelta(days=i % 90)).isoformat()
        import uuid
        mem_id = str(uuid.uuid4())
        real_backend._conn.execute(
            """
            INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
            VALUES (?, ?, ?, ?, ?, 'never', ?)
            """,
            (mem_id, "perf_user", text, created, i % 20, 1 if i % 50 == 0 else 0),
        )
        import sqlite_vec
        real_backend._conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (mem_id, sqlite_vec.serialize_float32(vector.tolist())),
        )
    real_backend._conn.commit()
    return real_backend


# ─────────────────────────────────────────────────────────────────────────────
# Embedding latency -- gates every single turn's memory search
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingLatency:
    def test_single_query_embed_latency(self, benchmark, real_backend):
        """Single query embedding -- this runs on EVERY user turn that isn't
        trivial-input-skipped, so it's the most turn-latency-critical number
        in this whole file."""
        benchmark(real_backend._embed, "what did I say about my deadline", query=True)

    def test_batch_embed_latency_ten_facts(self, benchmark, real_backend):
        """Batch embedding during add() -- runs once per extraction, not
        per-turn, so a slower budget is acceptable here than for query embed."""
        facts = [f"synthetic fact {i} for batch embed timing" for i in range(10)]
        benchmark(real_backend._embed_batch, facts)


# ─────────────────────────────────────────────────────────────────────────────
# Search latency at realistic memory-store scale
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchLatency:
    def test_quick_pass_search_latency(self, benchmark, seeded_backend):
        """Typical turn: query embeds once, quick pass (QUICK_KNN_LIMIT/
        QUICK_FTS_LIMIT candidates) is confident, wide pass never runs.
        This should be the common case in normal conversation."""
        benchmark(seeded_backend.search, "tell me about Oppa", "perf_user", 5)

    def test_wide_pass_search_latency(self, benchmark, seeded_backend, monkeypatch):
        """Force the wide-pass path by setting the confidence threshold
        impossibly high, so every search widens. This is the worst-case
        per-turn search latency -- worth knowing explicitly since it's a
        real (if less common) path in production, not just a hypothetical."""
        import memory.memorize as memorize_module
        monkeypatch.setattr(memorize_module, "MEMORY_RECALL_SCORE_THRESHOLD", 999.0)
        benchmark(seeded_backend.search, "tell me about Oppa", "perf_user", 5)


class TestCacheEffectiveness:
    """
    Not a hard latency assertion (which would be device-specific) -- this
    checks the cache actually buys you something, as a RATIO, which holds
    regardless of which device it runs on.
    """

    def test_cache_hit_meaningfully_faster_than_miss(self, seeded_backend):
        memo = AikoMemorize.__new__(AikoMemorize)
        memo._mem = seeded_backend
        memo._conn = seeded_backend._conn
        from collections import OrderedDict
        import threading
        memo._search_cache = OrderedDict()
        memo._search_cache_lock = threading.RLock()
        memo._last_cache_clear_time = 0.0

        query = "unique cache timing query about Oppa's deadline"

        t0 = time.perf_counter()
        memo.search(query, user_id="perf_user", limit=5)
        miss_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        memo.search(query, user_id="perf_user", limit=5)
        hit_time = time.perf_counter() - t0

        assert hit_time < miss_time / 3, (
            f"Cache hit ({hit_time:.4f}s) should be meaningfully faster than "
            f"a cold miss ({miss_time:.4f}s) -- if this fails, check whether "
            f"MEMORY_SEARCH_CACHE_TTL={MEMORY_SEARCH_CACHE_TTL} expired "
            f"mid-test or the cache key isn't matching as expected."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dream pass duration at scale -- this is a maintenance-window operation,
# but on constrained hardware a slow/OOM-prone dream pass is a real risk
# ─────────────────────────────────────────────────────────────────────────────

class TestDreamPassDuration:
    def test_full_dream_pass_duration(self, benchmark, seeded_backend):
        """
        Runs the full boost -> merge -> prune pipeline against the seeded
        store. Uses dry_run where possible isn't an option here since
        dream() doesn't expose a global dry_run flag at the top level in
        the same call signature as cleanup() -- if you add one, prefer
        benchmarking the dry_run path so this doesn't mutate seeded_backend
        state between repeated benchmark iterations.

        NOTE: pytest-benchmark calls the target function multiple times by
        default to get a stable mean. Since dream() mutates the DB (merges/
        prunes), repeated calls will see a shrinking store each iteration --
        this benchmark is really only valid for `--benchmark-min-rounds=1`.
        Run it as:
            pytest tests/perf/benchmark_memory.py::TestDreamPassDuration \\
                -m perf --benchmark-only --benchmark-min-rounds=1
        """
        memo = AikoMemorize.__new__(AikoMemorize)
        memo._mem = seeded_backend
        memo._conn = seeded_backend._conn

        def _run_dream_dry():
            return memo.dream(user_id="perf_user", dry_run=True)

        benchmark(_run_dream_dry)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction LLM round-trip -- shared resource contention matters here,
# since think.py's chat turn and memory extraction may compete for the
# same llama-server instance
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionLatency:
    def test_fact_extraction_round_trip(self, benchmark, real_backend):
        """
        Full extraction call: prompt build + LLM round trip + JSON parse +
        hedge filter. This is the number that matters for whether
        queue_write()'s idle-grace window (MEMORY_WRITE_IDLE_GRACE) is
        actually long enough in practice -- if extraction regularly takes
        longer than the idle grace, writes will queue up under load.
        """
        messages = [
            {"role": "user", "content": "My favorite color is teal and I work night shifts at the warehouse."},
            {"role": "assistant", "content": "Got it, noted!"},
        ]
        benchmark(real_backend._extract_facts, messages, display_name="PerfTestUser")


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted connection open overhead (system/secure.py) -- a one-time-per-
# session cost, not per-turn like everything above. Doesn't need baseline
# regression tracking the way hot-path memory benchmarks do; this is a
# curiosity check for whether SQLCipher key derivation + cipher init is
# milliseconds or seconds on constrained hardware, since a WebUI that opens
# a fresh connection per request (rather than one long-lived connection per
# session) would feel this on every request.
# ─────────────────────────────────────────────────────────────────────────────

class TestEncryptedConnectionOverhead:
    def test_encrypted_connection_open_close(self, benchmark, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_ENCRYPTION", "1")
        monkeypatch.setenv("DATA_KEY_SECRET", "benchmark-secret")

        try:
            import pysqlcipher3  # noqa: F401
        except ImportError:
            pytest.skip("pysqlcipher3 not installed in this environment -- "
                        "run this benchmark on the Jetson where it's actually installed")

        from system.secure import connect_sqlite
        import uuid

        def _open_and_close():
            conn = connect_sqlite(tmp_path / f"bench_{uuid.uuid4().hex}.db", user_id="perf_user")
            conn.close()

        benchmark(_open_and_close)

    def test_plaintext_connection_open_close_for_comparison(self, benchmark, tmp_path, monkeypatch):
        """Same operation with encryption OFF, as a baseline to compare the
        encrypted variant against -- the interesting number is the DELTA
        between these two, not either one in isolation."""
        monkeypatch.delenv("SQLITE_ENCRYPTION", raising=False)
        from system.secure import connect_sqlite
        import uuid

        def _open_and_close():
            conn = connect_sqlite(tmp_path / f"bench_{uuid.uuid4().hex}.db", user_id="perf_user")
            conn.close()

        benchmark(_open_and_close)
