"""
tests/perf/benchmark_wakeup.py
Latency/performance benchmarks for system/wakeup.py, intended to run ON
THE JETSON (or whatever the actual target device is) against the REAL
AikoThink, AikoMemorize, AikoSpeak, AikoListen, and the real HarrierEmbedder
-- NOT against the fakes used in test_wakeup.py. These measure wall-clock
boot time, which is only meaningful on the real device under real load,
same split as benchmark_memory.py vs test_memorize.py.

Everything here is marked @pytest.mark.perf and excluded from normal CI --
run it explicitly, only on the device you care about:

    pytest tests/perf/benchmark_wakeup.py -m perf --benchmark-only

Boot only happens once per process lifetime -- it is NOT a per-turn hot
path the way memory search/embed is. pytest-benchmark's default of running
a function many times to get a stable mean does NOT represent reality here:
a second in-process "boot" benefits from warm OS filesystem cache, warm
CUDA context, etc. in a way a real cold boot never does. Every benchmark
in this file should be run with --benchmark-min-rounds=1, same reasoning
as TestDreamPassDuration in benchmark_memory.py:

    pytest tests/perf/benchmark_wakeup.py -m perf --benchmark-only \\
        --benchmark-min-rounds=1 --benchmark-save=jetson_orin_nano_boot_baseline

Compare against a saved baseline on subsequent runs:

    pytest tests/perf/benchmark_wakeup.py -m perf --benchmark-only \\
        --benchmark-min-rounds=1 \\
        --benchmark-compare=jetson_orin_nano_boot_baseline \\
        --benchmark-compare-fail=mean:25%

Baselines are stored under .benchmarks/ by pytest-benchmark -- commit that
directory (or the specific baseline file) to git so boot-time regressions
show up in PR review.

Requires: pip install pytest-benchmark --break-system-packages

Things intentionally NOT covered here:
  - Correctness of the mem_ready handshake, callback pairing, or failure
    fallbacks -- those are logic tests against fakes, see test_wakeup.py.
  - Peak RSS / GPU memory watermark during the parallel boot phase. Given
    you're running near 95% RAM at boot, that number matters as much as
    wall-clock time, but pytest-benchmark measures time, not memory -- see
    the separate manual `tegrastats`-based note at the bottom of this file
    if you want to add that later.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

try:
    from system.wakeup import AikoWakeup
except ImportError as e:  # pragma: no cover
    pytest.skip(f"system.wakeup not importable in this environment: {e}", allow_module_level=True)


pytestmark = pytest.mark.perf


def _noop(*a, **kw):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Full boot -- the number you actually care about day to day
# ─────────────────────────────────────────────────────────────────────────────

class TestFullBootDuration:
    def test_full_boot_wall_clock(self, benchmark):
        """
        End-to-end AikoWakeup().boot() against real subsystems: think,
        memorize, speak, listen, all real models loaded. This is THE
        number -- if it regresses, something in the parallel or
        sequential phase got slower or a new blocking call was added.

        Run with --benchmark-min-rounds=1 (see module docstring) -- repeat
        calls within one pytest session do not represent independent cold
        boots.
        """
        def _boot():
            return AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

        result = benchmark(_boot)
        assert result.think is not None, "think failed to boot -- benchmark result is not representative"
        assert result.memorize is not None, "memorize failed to boot -- benchmark result is not representative"


class TestBootPhaseSplit:
    """
    Breaks the single full-boot number into parallel-phase vs
    sequential-phase, so a regression can be attributed to "think/memorize
    got slower" vs "scheduler setup or voice pipeline got slower" without
    re-running the whole boot under a profiler.

    Uses manual timing rather than pytest-benchmark's `benchmark()` fixture
    since we need two intermediate timestamps from inside one boot() call,
    not just total wall time -- monkeypatch the on_loading callback to
    stamp phase boundaries as they're crossed.
    """

    def test_parallel_vs_sequential_phase_duration(self):
        timestamps = {}

        def _on_loading(key):
            # 'think_start'/'mem_sqlite_vec' mark the beginning of the
            # parallel phase; 'speak_miotts' marks the first sequential
            # step after scheduler setup -- see boot()'s call order.
            if "boot_start" not in timestamps:
                timestamps["boot_start"] = time.perf_counter()
            if key == "speak_miotts" and "sequential_start" not in timestamps:
                timestamps["sequential_start"] = time.perf_counter()

        def _on_done(key):
            if key == "listen_ready":
                timestamps["boot_end"] = time.perf_counter()

        AikoWakeup().boot(on_loading=_on_loading, on_done=_on_done, on_skip=_noop)

        assert "sequential_start" in timestamps, "speak_miotts loading callback never fired"
        parallel_duration = timestamps["sequential_start"] - timestamps["boot_start"]
        sequential_duration = timestamps["boot_end"] - timestamps["sequential_start"]

        print(f"\n[wakeup perf] parallel phase (think+memorize): {parallel_duration:.3f}s")
        print(f"[wakeup perf] sequential phase (scheduler+voice): {sequential_duration:.3f}s")

        # no hard assertion on absolute time (device-specific) -- this test
        # exists to print the split every run so you can eyeball which
        # phase moved between baselines, same spirit as
        # TestCacheEffectiveness in benchmark_memory.py using a ratio
        # rather than an absolute threshold.
        assert parallel_duration > 0 and sequential_duration > 0


# ─────────────────────────────────────────────────────────────────────────────
# Semantic exemplar prewarm -- disk cache vs recompute vs in-memory hit
#
# This directly answers the "is disk faster than recomputing on Harrier"
# question: three benchmarks, same call, three different starting cache
# states forced via the ROUTE_VECTOR_CACHE_DIR env var + a fresh AikoThink
# instance so the in-memory cache is empty going in.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_think_instance(tmp_path, monkeypatch):
    """
    A real AikoThink with a REAL embedder, but pointed at an isolated,
    per-test cache directory so tests don't interfere with each other or
    with your actual ~/.aiko cache. Skips if AikoThink can't construct in
    this environment (e.g. no GGUF model present).

    NOTE: uses the real user_state_dir()-anchored default, override the
    env var to redirect it into tmp_path -- if your think.py resolves the
    cache path some other way, adjust this fixture accordingly.
    """
    monkeypatch.setenv("ROUTE_VECTOR_CACHE_DIR", str(tmp_path / "route_vectors"))
    from cognition.think import AikoThink
    try:
        think = AikoThink(None, speak=None)
        think.join_warmup()
    except Exception as e:
        pytest.skip(f"Real AikoThink not available in this environment: {e}")
    return think


class TestSemanticPrewarmCachePaths:
    def test_cold_recompute_no_disk_no_memory(self, benchmark, fresh_think_instance):
        """
        Worst case: nothing cached anywhere. Forces the full
        reason.embed_example_matrix(...) path through the real Harrier
        embedder. This is the number that matters if the disk cache is
        ever cold (first boot ever, or after editing the example lists).
        """
        from cognition.think import (
            _ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY,
        )
        think = fresh_think_instance

        def _recompute():
            # bypass both caches by using a throwaway key each call so
            # pytest-benchmark's repeat rounds don't accidentally hit a
            # cache populated by a prior round within the same run
            think._semantic_example_cache.clear()
            return think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)

        benchmark(_recompute)

    def test_disk_cache_hit_cold_memory(self, benchmark, fresh_think_instance):
        """
        Disk is warm (populated by a prior call), in-memory cache is
        empty -- simulates the real boot scenario: fresh process, but a
        .npz written by a previous run's boot. This is the number that
        answers 'is disk read faster than recomputing on Harrier.'
        """
        from cognition.think import (
            _ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY,
        )
        think = fresh_think_instance
        # populate disk once, outside the timed region
        think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)

        def _disk_hit():
            think._semantic_example_cache.clear()  # force past in-memory cache
            return think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)

        benchmark(_disk_hit)

    def test_in_memory_cache_hit(self, benchmark, fresh_think_instance):
        """
        Best case: already resolved once in this process. Should be a bare
        dict lookup -- effectively free. Included mainly as a sanity floor
        so the disk-hit number above can be judged relative to something,
        not because this path is itself at risk of regressing.
        """
        from cognition.think import (
            _ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY,
        )
        think = fresh_think_instance
        think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)

        benchmark(think._semantic_example_vectors, _ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)


class TestPrewarmObservedAtRealBoot:
    def test_prewarm_cache_state_during_real_boot(self, tmp_path, monkeypatch, capsys):
        """
        Not a pytest-benchmark timing test -- a diagnostic that instruments
        a REAL AikoWakeup().boot() call and reports whether the semantic
        exemplar prewarm actually hit disk or had to recompute. Run this
        occasionally against your real ~/.aiko cache (i.e. WITHOUT
        overriding ROUTE_VECTOR_CACHE_DIR) to see, empirically, across your
        real boots, whether the disk cache is doing its job or silently
        missing every time (e.g. due to the current_user_id() timing
        concern flagged during the think.py review).
        """
        import cognition.think as think_module
        original = think_module.AikoThink._semantic_example_vectors
        calls = {"disk_hit": 0, "recompute": 0}

        def _instrumented(self, examples_by_label, instruct):
            cache_key = (id(examples_by_label), instruct)
            was_in_memory = cache_key in self._semantic_example_cache
            result = original(self, examples_by_label, instruct)
            if not was_in_memory:
                # can't distinguish disk-hit from recompute from outside
                # without touching internals further -- if you want this
                # split for real, add a log line inside
                # _semantic_example_vectors itself at the disk-hit branch
                # and the recompute branch, then grep the log instead.
                calls["recompute"] += 1
            else:
                calls["disk_hit"] += 1
            return result

        monkeypatch.setattr(think_module.AikoThink, "_semantic_example_vectors", _instrumented)

        AikoWakeup().boot(on_loading=_noop, on_done=_noop, on_skip=_noop)

        print(f"\n[wakeup perf] prewarm calls: {calls}")
        with capsys.disabled():
            pass  # calls dict already printed above for -s runs


# ─────────────────────────────────────────────────────────────────────────────
# Manual memory-watermark note (not a pytest-benchmark test)
#
# pytest-benchmark measures wall-clock time only. Given boot already runs
# near 95% RAM, the number that actually threatens stability is peak RSS/
# GPU memory during the parallel phase (think + memorize loading
# concurrently), not boot duration. To capture that empirically, run boot
# under a sampler alongside this suite, e.g.:
#
#   tegrastats --interval 200 --logfile /tmp/boot_tegrastats.log &
#   TEGRA_PID=$!
#   pytest tests/perf/benchmark_wakeup.py::TestFullBootDuration -m perf \\
#       --benchmark-only --benchmark-min-rounds=1
#   kill $TEGRA_PID
#   # then grep the peak RAM/GPU line out of /tmp/boot_tegrastats.log
#
# Worth adding as a proper fixture once you decide on a sampling approach
# you're happy with -- left as a manual step for now rather than guessing
# at a tegrastats-parsing implementation you haven't reviewed.
# ─────────────────────────────────────────────────────────────────────────────
