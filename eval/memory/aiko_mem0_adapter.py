"""
Aiko adapter for mem0ai/memory-benchmarks
==========================================

Drop-in stand-in for ``benchmarks.common.mem0_client.Mem0Client``, wired
directly to ``cognition.memory.memorize.AikoMemorize`` (confirmed against
backend.py — not the earlier guess-based version of this file).

Implements the same async surface the runners call:

    await client.add(messages, user_id, timestamp=...)
    await client.search(query, user_id, top_k=200)
    await client.delete_user(user_id)
    await client.close()

How to use
----------
1. Copy this file into the benchmark repo, e.g.::

       cp aiko_mem0_adapter.py /path/to/memory-benchmarks/benchmarks/common/

2. Point PYTHONPATH at your Aiko repo root (so ``cognition`` / ``system`` import).

3. Patch the runner to construct this client instead of Mem0Client, e.g. in
   ``benchmarks/locomo/run.py`` (and longmemeval / beam)::

       from benchmarks.common.aiko_mem0_adapter import AikoMemClient as Mem0Client

4. Run as usual (OSS flags that only apply to Mem0 are ignored)::

       python -m benchmarks.locomo.run --project-name aiko-locomo

IMPORTANT — isolate the benchmark DB from your real Aiko memory
-----------------------------------------------------------------
AikoMemorize opens ONE sqlite file per process, chosen at construction time
via ``system.userspace.current_user_id()`` (see ``_memory_db_path_for_user``
in backend.py). Every bench user_id you pass to add()/search() is just a
row-level filter (`memories.user_id`) *within that one file* — it does NOT
give each bench user_id its own database.

That means:
  - If you run this against your normal Aiko user context, benchmark rows
    land in the SAME db as Aiko's real memories.
  - Set SQLITE_MEMORY_PATH to a scratch path before importing anything, so
    the whole benchmark run — regardless of how many synthetic user_ids the
    runner iterates over — writes to one throwaway file, isolated from
    production memory:

        export SQLITE_MEMORY_PATH=/tmp/aiko_bench_memory.db

  - delete_user() clears rows for one user_id (AikoMemorize.clear()); it
    does not touch other bench users' rows in the same file, but you still
    want SQLITE_MEMORY_PATH pointed at scratch so a bug or a skipped
    delete_user() can never bleed into your real memory store.

What is / isn't measured
------------------------
Covered: ingest turns -> store -> retrieve -> answer/judge (same as Mem0
suite) — exercises the real RRF fusion (KNN+FTS+entity-graph), write-time
dedup/supersede, Ebbinghaus access tracking, and recency-among-relevant
rerank on the actual recall path.
Not covered: dream() consolidation, valence-avoid recall shaping, monthly
gate, L2 scenes, L3 persona cache — these run on a schedule / on separate
call paths mem0-style single-turn benchmarks don't exercise. A benchmark
number here validates retrieval + fusion, not the whole memory system.

Environment
-----------
AIKO_ROOT            Path to Aiko repo (prepended to sys.path if set)
SQLITE_MEMORY_PATH    Scratch DB path — set this, see warning above
AIKO_BENCH_USER       Force a fixed user id for all rows (optional)
AIKO_BENCH_DRY_RUN    If "1", skip real writes/searches (smoke only)

Full run procedure
-------------------
1. Clone the benchmark repo and install deps::

       git clone https://github.com/mem0ai/memory-benchmarks.git
       cd memory-benchmarks
       pip install -r requirements.txt

2. Copy this adapter in and point it at your Aiko checkout::

       cp /path/to/aiko_mem0_adapter.py benchmarks/common/
       export AIKO_ROOT=/path/to/Aiko-chan   # repo root containing cognition/, system/

3. Isolate the DB (see warning above — do this every time, not just once)::

       export SQLITE_MEMORY_PATH=/tmp/aiko_bench_memory.db
       rm -f /tmp/aiko_bench_memory.db*     # start clean between runs

4. Point Aiko's runtime deps at your running local server (same LLM_BASE_URL
   / EXTRACT_MODEL / EMBED_MODEL your normal Aiko boot uses — the adapter
   goes through the real AikoMemorize extraction path, so whatever model
   serves that endpoint does the fact extraction for every ingested turn)::

       export LLM_BASE_URL=http://localhost:8080/v1
       export EXTRACT_MODEL=ministral        # or whatever you're benchmarking

5. Swap the runner's client import. Each benchmark script constructs
   Mem0Client directly, so either edit the one-line import in
   benchmarks/locomo/run.py (and longmemeval/run.py, beam/run.py) from::

       from benchmarks.common.mem0_client import Mem0Client

   to::

       from benchmarks.common.aiko_mem0_adapter import AikoMemClient as Mem0Client

   or monkeypatch it at the top of a wrapper script instead of editing the
   runner files directly::

       import benchmarks.common.mem0_client as mc
       from benchmarks.common.aiko_mem0_adapter import AikoMemClient
       mc.Mem0Client = AikoMemClient

6. Smoke-test the adapter alone before a full run (catches import/DB path
   problems in seconds instead of mid-benchmark)::

       python -m benchmarks.common.aiko_mem0_adapter

7. Set the judge/answerer LLM (see "Judge LLM" below), then run. The
   ``--backend``/``--mem0-api-key``/``--mem0-host`` flags in the README are
   Mem0-cloud/Mem0-server-specific and don't apply here — the adapter talks
   to AikoMemorize in-process, not over HTTP — so omit them::

       export OPENAI_API_KEY=sk-...     # judge + answerer LLM, see below
       python -m benchmarks.locomo.run \\
           --project-name aiko-locomo \\
           --judge-model gpt-4o \\
           --answerer-model gpt-4o \\
           --top-k 50

       # LongMemEval (500 questions, larger/slower)
       python -m benchmarks.longmemeval.run \\
           --project-name aiko-longmemeval \\
           --all-questions

8. Results land under results/ in the benchmark repo (per-question JSON +
   aggregate scores). The bundled Next.js UI (``npm run dev`` in the repo
   root) can browse them if you want more than the CLI summary.

Settings worth sweeping deliberately
-------------------------------------
- ``--top-k`` / ``--top-k-cutoffs``: this adapter passes top_k straight
  through as AikoMemorize.search()'s ``limit``, which drives the
  quick-vs-wide tiered candidate pass (QUICK_KNN_LIMIT/KNN_LIMIT etc. in
  backend.py) — a very large top_k will always trigger the wide pass.
- ``EXTRACT_MODEL``: the local LLM doing fact extraction on ingest. This is
  the single biggest lever on recall quality and is exactly the axis you're
  already comparing (Nanbeige4.2-3B vs Ministral, per your recent
  benchmarking) — run the benchmark once per candidate extraction model,
  same SQLITE_MEMORY_PATH wiped between runs, everything else held constant.
- ``MEMORY_RANK_GRAPH_WEIGHT`` / ``MEMORY_RANK_PINNED_WEIGHT`` / RRF_K /
  decay half-life: all read from os.environ at backend.py import time (see
  the constants block there) — export overrides before importing Aiko to
  A/B your ranking knobs without touching code.
- Judge and answerer model: see below — hold these fixed across your own
  comparison runs (extraction-model sweeps etc.); only change them if
  you're deliberately testing judge sensitivity.

Judge LLM — is one required, and does it have to be a specific model?
------------------------------------------------------------------------
Yes to "required" (for anything past ``--predict-only``), no to "specific."
The benchmark's answer-generation and judging stage is a separate LLM call
from Aiko's own extraction/search — it's the harness reading back what got
retrieved and scoring it against ground truth, not something AikoMemorize
does internally. The mem0ai/memory-benchmarks CLI exposes it directly::

    --answerer-model MODEL   LLM that generates the answer from retrieved memories (default: gpt-4o)
    --judge-model MODEL      LLM that scores that answer against ground truth (default: gpt-4o)
    --provider PROVIDER      openai | anthropic | azure (default: openai)

Defaults to OpenAI gpt-4o for both roles, needing OPENAI_API_KEY set. You
can swap in Anthropic or Azure via --provider plus the matching model name
and API key env var. mem0's own published LOCOMO baselines mostly use
gpt-4o-mini as judge (cheaper, still standard in the literature); the repo
defaults to full gpt-4o. Two things matter more than which specific model
you pick:
  - Consistency: if you're comparing Aiko across multiple runs (e.g. per
    EXTRACT_MODEL sweep above), keep --judge-model and --answerer-model
    identical across all of them — the scores aren't comparable otherwise.
  - Only use ``--evaluate-only`` runs against another provider's public
    published numbers with real caution: those numbers used whatever judge
    that team picked, which may not match yours.
Use ``--predict-only`` to run ingest+search without touching the judge/
answerer at all — useful for iterating on Aiko's retrieval quality alone
without burning API credits every pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make Aiko importable
# ---------------------------------------------------------------------------

_AIKO_ROOT = os.environ.get("AIKO_ROOT", "").strip()
if _AIKO_ROOT and _AIKO_ROOT not in sys.path:
    sys.path.insert(0, _AIKO_ROOT)


def _resolve_user_id(user_id: str) -> str:
    forced = os.environ.get("AIKO_BENCH_USER", "").strip()
    return forced or (user_id or "bench")


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize_search_hits(raw: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Map AikoMemorize.search() rows -> Mem0-shaped list[{memory, score, id}].

    AikoMemorize.search() returns dicts shaped like the `memories` table row
    plus `_recall_score` (see backend.py `search()` / `_rank_and_score()`):
    keys include memory, id, created_at, access_count, pinned, valence_tag,
    valence_score, kind, _recall_score, and occasionally _scene / _scene_members
    or _supersession_chain when scene/lineage expansion kicked in.
    """
    out: list[dict[str, Any]] = []
    for i, r in enumerate(raw or []):
        if not isinstance(r, dict):
            out.append({"memory": str(r), "score": 0.0, "id": str(i)})
            continue
        text = r.get("memory") or r.get("text") or ""
        score = r.get("_recall_score")
        if score is None:
            score = r.get("score", 0.0)
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        entry: dict[str, Any] = {
            "memory": str(text),
            "score": score_f,
            "id": str(r.get("id") or i),
        }
        for key in ("created_at", "kind", "pinned", "valence_tag", "valence_score"):
            if r.get(key) is not None:
                entry[key] = r[key]
        out.append(entry)

    out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return out[: max(1, top_k)]


class AikoMemClient:
    """Mem0Client-compatible async facade over AikoMemorize.

    One AikoMemorize instance is created and reused for the client's whole
    lifetime — it owns a HarrierEmbedder (model load) and a background
    write-queue thread, both expensive to spin up per-call. All calls are
    routed through asyncio.to_thread() because AikoMemorize's own methods
    are synchronous/blocking (LLM extraction calls, sqlite I/O) — this is
    not a workaround for a threading mismatch, it's the correct way to call
    a blocking API from an async runner.
    """

    def __init__(
        self,
        mode: str = "oss",  # ignored — kept for signature parity with Mem0Client
        host: str | None = None,
        api_key: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rpm: int = 120,
        timeout: float = 300.0,
        **_kwargs: Any,
    ) -> None:
        self.mode = "aiko"
        self.host = host or "aiko-local"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._dry = os.environ.get("AIKO_BENCH_DRY_RUN", "").strip() in ("1", "true", "yes")
        self._aiko: Any = None
        if not self._dry:
            from cognition.memory.memorize import AikoMemorize

            self._aiko = AikoMemorize(silent=True)
            logger.info("AikoMemClient ready (AikoMemorize instance constructed)")
        else:
            logger.warning("AIKO_BENCH_DRY_RUN=1 — no real memory I/O")

    async def __aenter__(self) -> "AikoMemClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        # AikoMemorize has no explicit close(); the write-queue thread is a
        # daemon thread and the sqlite connection closes with the process.
        # If a run needs a hard drain point, wait on pending writes instead:
        if self._aiko is not None:
            await asyncio.to_thread(self._aiko.wait_for_writes, 30.0)

    # ------------------------------------------------------------------ add
    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Ingest a message chunk into Aiko memory via AikoMemorize.add().

        NOTE: AikoMemorize.add() returns bool, not the ids/text of facts it
        extracted (that detail lives one layer down, in the private
        _MemoryBackend.add()). For benchmark purposes this is almost always
        fine — LOCOMO/longmemeval score the *search* step, not what add()
        echoes back — so on success we synthesize a single placeholder
        result rather than reach into the private API. If you need the real
        per-fact records (e.g. to audit exactly what got extracted per
        turn), call self._aiko._mem.add(...) directly instead and adapt the
        return value; that method does return real ids, just isn't part of
        the public class.
        """
        uid = _resolve_user_id(user_id)
        text = _messages_to_text(messages)
        if not text.strip():
            return {"results": []}

        if self._dry:
            return {"results": [{"memory": text[:200], "event": "ADD", "id": "dry"}]}

        assert self._aiko is not None

        def _write() -> bool:
            return self._aiko.add(messages, user_id=uid)

        try:
            ok = await asyncio.to_thread(_write)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Aiko add failed for user=%s: %s", uid, exc)
            return None

        if not ok:
            return None
        return {
            "results": [
                {"memory": text[:500], "event": "ADD", "id": str(time.time())}
            ]
        }

    # --------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Search Aiko memory via AikoMemorize.search(); return Mem0-shaped hits.

        include_history=False (default) — superseded/inactive rows are
        excluded, matching normal recall behavior.
        """
        uid = _resolve_user_id(user_id)
        if self._dry:
            return [{"memory": f"[dry] {query}", "score": 1.0, "id": "dry"}]

        assert self._aiko is not None

        def _search() -> list[dict]:
            return self._aiko.search(query, user_id=uid, limit=top_k, include_history=False)

        try:
            raw = await asyncio.to_thread(_search)
            hits = _normalize_search_hits(raw, top_k)
            if score_debug:
                for h in hits:
                    h.setdefault("score_debug", {"combined_score": h.get("score", 0)})
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.exception("Aiko search failed: %s", exc)
            return []

    # ---------------------------------------------------------- delete_user
    async def delete_user(self, user_id: str) -> bool:
        """Wipe one bench user's rows via AikoMemorize.clear(user_id=...)."""
        uid = _resolve_user_id(user_id)
        if self._dry:
            return True
        assert self._aiko is not None

        def _del() -> bool:
            self._aiko.clear(user_id=uid)
            return True

        try:
            return bool(await asyncio.to_thread(_del))
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_user failed: %s", exc)
            return False


# Alias matching Mem0 naming so ``as Mem0Client`` imports read cleanly
Mem0Client = AikoMemClient


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _smoke() -> None:
        # Sanity check before a real run — make sure you set SQLITE_MEMORY_PATH
        # to a scratch file, not your real Aiko db, before running this.
        client = AikoMemClient()
        uid = "aiko-bench-smoke"
        await client.delete_user(uid)
        resp = await client.add(
            [
                {"role": "user", "content": "My name is Ada and I live in Vancouver."},
                {"role": "assistant", "content": "Nice to meet you, Ada!"},
            ],
            user_id=uid,
            timestamp=int(datetime.now(tz=timezone.utc).timestamp()),
        )
        print("add ->", resp)
        hits = await client.search("Where does Ada live?", user_id=uid, top_k=5)
        print("search ->", hits)
        await client.close()

    asyncio.run(_smoke())
