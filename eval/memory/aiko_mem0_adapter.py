"""
Aiko adapter for mem0ai/memory-benchmarks
==========================================

Drop-in stand-in for ``benchmarks.common.mem0_client.Mem0Client``.

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

       # from benchmarks.common.mem0_client import Mem0Client
       from benchmarks.common.aiko_mem0_adapter import AikoMemClient as Mem0Client

   Or monkeypatch before ``main()``::

       import benchmarks.common.mem0_client as mc
       from benchmarks.common.aiko_mem0_adapter import AikoMemClient
       mc.Mem0Client = AikoMemClient

4. Run as usual (OSS flags that only apply to Mem0 are ignored)::

       python -m benchmarks.locomo.run --project-name aiko-locomo
       python -m benchmarks.longmemeval.run --project-name aiko-lme --all-questions

What is / isn't measured
------------------------
Covered: ingest turns → store → retrieve → answer/judge (same as Mem0 suite).
Not covered: monthly gate, day-pins, dream, valence freeze, session novelty —
those need a separate Aiko harness.

Environment
-----------
AIKO_ROOT          Path to Aiko repo (prepended to sys.path if set)
AIKO_BENCH_USER    Force a fixed user id for all rows (optional)
AIKO_BENCH_DRY_RUN If "1", skip real writes/searches (smoke only)
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


# ---------------------------------------------------------------------------
# Best-effort imports of Aiko public APIs (cognition package)
# ---------------------------------------------------------------------------

def _import_aiko() -> dict[str, Any]:
    """Load write/search helpers. Raises ImportError with a clear message."""
    apis: dict[str, Any] = {}
    errors: list[str] = []

    # Search (Phase B unified facade — preferred)
    for path in (
        "cognition.memory.search",
        "cognition.memory",
        "memory.search",
        "memory.memorize",
    ):
        try:
            mod = __import__(path, fromlist=["*"])
            for name in ("search_memory", "search", "recall"):
                fn = getattr(mod, name, None)
                if callable(fn):
                    apis["search_memory"] = fn
                    apis["search_mod"] = path
                    break
            if "search_memory" in apis:
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")

    # Write / memorize
    for path in (
        "cognition.memory.memorize",
        "memory.memorize",
    ):
        try:
            mod = __import__(path, fromlist=["*"])
            for name in (
                "memorize_turn",
                "memorize_messages",
                "memorize_text",
                "add_memory",
                "write_memory",
                "ingest_messages",
            ):
                fn = getattr(mod, name, None)
                if callable(fn):
                    apis["memorize"] = fn
                    apis["memorize_mod"] = path
                    apis["memorize_name"] = name
                    break
            # Some codebases expose a class
            if "memorize" not in apis:
                for name in ("Memorize", "MemoryWriter", "PersonalMemory"):
                    cls = getattr(mod, name, None)
                    if cls is not None:
                        apis["memorize_cls"] = cls
                        apis["memorize_mod"] = path
                        break
            if "memorize" in apis or "memorize_cls" in apis:
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")

    # Optional: clear user memories for delete_user
    for path in (
        "cognition.memory.memorize",
        "cognition.memory.lifecycle",
        "memory.memorize",
    ):
        try:
            mod = __import__(path, fromlist=["*"])
            for name in ("delete_user_memories", "clear_user_memory", "purge_user"):
                fn = getattr(mod, name, None)
                if callable(fn):
                    apis["delete_user"] = fn
                    break
            if "delete_user" in apis:
                break
        except Exception:
            pass

    if "search_memory" not in apis and "memorize" not in apis and "memorize_cls" not in apis:
        raise ImportError(
            "Could not import Aiko memory APIs. Set AIKO_ROOT to the repo root.\n"
            + "\n".join(errors[:8])
        )
    return apis


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize_search_hits(raw: Any, top_k: int) -> list[dict[str, Any]]:
    """Map Aiko search results → Mem0-shaped list[{memory, score, id}]."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = raw.get("results") or raw.get("memories") or raw.get("hits") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    out: list[dict[str, Any]] = []
    for i, r in enumerate(items):
        if not isinstance(r, dict):
            text = str(r)
            out.append({"memory": text, "score": 0.0, "id": str(i)})
            continue
        text = (
            r.get("memory")
            or r.get("text")
            or r.get("content")
            or r.get("fact")
            or r.get("summary")
            or ""
        )
        score = r.get("score")
        if score is None:
            score = r.get("rrf") or r.get("similarity") or r.get("rank") or 0.0
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        mid = str(r.get("id") or r.get("memory_id") or i)
        entry: dict[str, Any] = {"memory": str(text), "score": score_f, "id": mid}
        for key in ("created_at", "updated_at"):
            if r.get(key):
                entry[key] = r[key]
        out.append(entry)

    out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return out[: max(1, top_k)]


class AikoMemClient:
    """Mem0Client-compatible async facade over Aiko personal memory."""

    def __init__(
        self,
        mode: str = "oss",  # ignored — kept for signature parity
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
        self._apis: dict[str, Any] | None = None
        if not self._dry:
            self._apis = _import_aiko()
            logger.info(
                "AikoMemClient ready (search=%s memorize=%s)",
                self._apis.get("search_mod"),
                self._apis.get("memorize_mod") or self._apis.get("memorize_name"),
            )
        else:
            logger.warning("AIKO_BENCH_DRY_RUN=1 — no real memory I/O")

    async def __aenter__(self) -> "AikoMemClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        return None

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
        """Ingest a message chunk into Aiko memory.

        Returns ``{"results": [...]}`` on success (Mem0 shape), or None on failure.
        """
        uid = _resolve_user_id(user_id)
        text = _messages_to_text(messages)
        if not text.strip():
            return {"results": []}

        if self._dry:
            return {"results": [{"memory": text[:200], "event": "ADD", "id": "dry"}]}

        assert self._apis is not None

        def _write() -> list[dict[str, Any]]:
            apis = self._apis
            # Prefer a dedicated messages/turn API if present
            mem_fn = apis.get("memorize")
            if mem_fn is not None:
                try:
                    # Try rich signature first
                    result = mem_fn(
                        messages,
                        user_id=uid,
                        timestamp=timestamp,
                        observation_date=observation_date,
                        metadata=metadata or {},
                    )
                except TypeError:
                    try:
                        result = mem_fn(text, user_id=uid)
                    except TypeError:
                        result = mem_fn(text)
                return _coerce_write_results(result, text)

            cls = apis.get("memorize_cls")
            if cls is not None:
                inst = cls(user_id=uid) if _takes_user_id(cls) else cls()
                for method in ("memorize_messages", "add", "write", "memorize"):
                    fn = getattr(inst, method, None)
                    if not callable(fn):
                        continue
                    try:
                        result = fn(messages, user_id=uid)
                    except TypeError:
                        try:
                            result = fn(text, user_id=uid)
                        except TypeError:
                            result = fn(text)
                    return _coerce_write_results(result, text)

            raise RuntimeError(
                "No usable Aiko memorize API found. Wire memorize_turn / "
                "memorize_text in _import_aiko()."
            )

        try:
            results = await asyncio.to_thread(_write)
            return {"results": results}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Aiko add failed for user=%s: %s", uid, exc)
            return None

    # --------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Search Aiko memory; return Mem0-shaped hits."""
        uid = _resolve_user_id(user_id)
        if self._dry:
            return [{"memory": f"[dry] {query}", "score": 1.0, "id": "dry"}]

        assert self._apis is not None
        search_fn = self._apis.get("search_memory")
        if search_fn is None:
            logger.error("No search_memory API bound")
            return []

        def _search() -> Any:
            try:
                return search_fn(query, user_id=uid, limit=top_k)
            except TypeError:
                try:
                    return search_fn(query, limit=top_k, user_id=uid)
                except TypeError:
                    try:
                        return search_fn(query, top_k)
                    except TypeError:
                        return search_fn(query)

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
        """Best-effort wipe of benchmark user data."""
        uid = _resolve_user_id(user_id)
        if self._dry:
            return True
        if not self._apis:
            return False
        fn = self._apis.get("delete_user")
        if fn is None:
            logger.warning(
                "No delete_user API in Aiko — benchmark isolation may leak across runs "
                "(user_id=%s). Use unique --project-name / user ids.",
                uid,
            )
            return False

        def _del() -> bool:
            try:
                fn(user_id=uid)
                return True
            except TypeError:
                fn(uid)
                return True

        try:
            return bool(await asyncio.to_thread(_del))
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_user failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _takes_user_id(cls: type) -> bool:
    try:
        import inspect

        return "user_id" in inspect.signature(cls).parameters
    except Exception:
        return False


def _coerce_write_results(result: Any, fallback_text: str) -> list[dict[str, Any]]:
    if result is None:
        return [{"memory": fallback_text[:500], "event": "ADD", "id": str(time.time())}]
    if isinstance(result, dict):
        if "results" in result and isinstance(result["results"], list):
            return list(result["results"])
        text = result.get("memory") or result.get("text") or fallback_text[:500]
        return [{"memory": text, "event": result.get("event", "ADD"), "id": str(result.get("id", ""))}]
    if isinstance(result, list):
        return [_coerce_write_results(x, fallback_text)[0] for x in result]
    return [{"memory": str(result)[:500], "event": "ADD", "id": ""}]


# Alias matching Mem0 naming so ``as Mem0Client`` imports read cleanly
Mem0Client = AikoMemClient


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _smoke() -> None:
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
        print("add →", resp)
        hits = await client.search("Where does Ada live?", user_id=uid, top_k=5)
        print("search →", hits)
        await client.close()

    asyncio.run(_smoke())
