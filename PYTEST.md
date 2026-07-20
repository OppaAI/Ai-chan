# Test / eval / benchmark layout

```
tests/
  unit/            # pytest, deterministic, mirrors source modules 1:1
    test_memorize.py
    test_userspace.py
    test_secure.py
  integration/     # pytest, real components wired together, marked @pytest.mark.integration
    test_multiuser_isolation.py
  stress/          # pytest, concurrency/load, marked @pytest.mark.stress
    test_write_queue_stress.py
  perf/            # pytest-benchmark, marked @pytest.mark.perf
    benchmark_memory.py   # includes memory hot-path AND secure.py connection overhead
pytest.ini          # registers perf/integration/stress markers, scopes default runs to tests/unit/
eval/
  eval_memory_extraction.py   # standalone script, not pytest -- run directly
  eval_memory_recall.py       # standalone script, not pytest -- run directly
```

## Why these are split this way

**`tests/unit/`** — one file per source module (`memorize.py` -> `test_memorize.py`,
etc). Pure logic, deterministic, uses `FakeEmbedder` where an embedding model
would otherwise be needed. Fast, runs anywhere (dev laptop, CI), no real
hardware or model required.

Run with:
```
pytest
```
(picks up `tests/unit/` only, per `pytest.ini`)

**`tests/integration/`** — real components wired together (real userspace
contextvars, real SQLite files, `FakeEmbedder` where a real model isn't
needed for plumbing correctness). Currently covers multi-user isolation --
the highest-consequence failure mode for a personal-memory system, since a
bug here means user A sees user B's memories.

Run with:
```
pytest tests/integration -m integration -v
```

**`tests/stress/`** — concurrency and load, using the real
`queue.Queue`/worker thread and real SQLite writes, but `FakeEmbedder`/fake
LLM client so it measures queue/threading behavior rather than model
latency. Currently covers the async write queue under rapid-fire and
concurrent access.

Run with:
```
pytest tests/stress -m stress -v
```

 only meaningful against the real
HarrierEmbedder/Ministral on the actual target device (Jetson Orin Nano).
Excluded from default `pytest` runs entirely.

First time, with no baseline yet:
```
pytest tests/perf -m perf --benchmark-only --benchmark-save=jetson_orin_nano_baseline
```

Every run after, to catch regressions:
```
pytest tests/perf -m perf --benchmark-only \
  --benchmark-compare=jetson_orin_nano_baseline \
  --benchmark-compare-fail=mean:25%
```

Commit the `.benchmarks/` baseline directory (or at least the specific
baseline file) so regressions are visible in PR review.

**`eval/`** — accuracy, not pass/fail. Extraction and recall quality depend
on non-deterministic LLM output, so these are scored (precision/recall/F1,
recall@k/MRR) and reported, not asserted. Not pytest tests at all -- run
them directly and diff the saved JSON against the previous run:

```
python eval/eval_memory_extraction.py --verbose --out results/extraction_$(date +%F).json
python eval/eval_memory_recall.py --verbose --out results/recall_$(date +%F).json
```

## Adding a new module's tests

Follow the same pattern: one new `tests/unit/test_<module>.py` per source
file. Only add to `tests/perf/` if the module sits on a per-turn hot path
where wall-clock latency matters on constrained hardware. Only add an
`eval/` script if the module's output is non-deterministic and needs
accuracy scoring rather than a binary assertion.
