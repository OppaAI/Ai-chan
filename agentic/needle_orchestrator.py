"""Deterministic, bounded orchestration for independent Needle 2 workers.

Workers are configured explicitly and must point at separate Needle servers (or
at server sessions that the Needle deployment documents as isolated).  This
module only aggregates constrained tool proposals; Aiko's normal ReAct loop
continues to validate, approve, and execute every proposed call.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from agentic.needle import NeedleClient, NeedleError, NeedleResponse


@dataclass(frozen=True)
class NeedleWorkerSpec:
    """One explicitly configured Needle worker with a least-privilege tool set."""

    id: str
    role: str
    base_url: str
    allowed_tools: tuple[str, ...]
    confidence_threshold: float = 0.85
    timeout: float = 15.0


@dataclass(frozen=True)
class NeedleWorkerResult:
    """A worker response or a safe, serializable failure record."""

    worker_id: str
    role: str
    response: NeedleResponse | None = None
    error: str = ""


def load_needle_workers(
    raw: str,
    *,
    default_timeout: float,
    default_confidence_threshold: float,
    max_workers: int = 4,
) -> tuple[NeedleWorkerSpec, ...]:
    """Parse ``NEEDLE_WORKERS`` JSON into validated, deterministic worker specs.

    Empty configuration deliberately means no workers.  Each worker must name
    its permitted tools, so a configuration typo cannot silently create an
    unrestricted action worker.
    """
    if not raw.strip():
        return ()
    if max_workers < 1:
        raise NeedleError("NEEDLE_MAX_WORKERS must be at least 1")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NeedleError(f"NEEDLE_WORKERS must be a JSON array: {exc}") from exc
    if not isinstance(items, list):
        raise NeedleError("NEEDLE_WORKERS must be a JSON array")
    if len(items) > max_workers:
        raise NeedleError(f"NEEDLE_WORKERS exceeds configured worker limit ({max_workers})")

    workers: list[NeedleWorkerSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise NeedleError(f"NEEDLE_WORKERS[{index}] must be an object")
        worker_id = str(item.get("id") or "").strip()
        role = str(item.get("role") or worker_id).strip()
        base_url = str(item.get("base_url") or "").strip()
        allowed_raw = item.get("allowed_tools")
        if not worker_id or not role or not base_url:
            raise NeedleError(f"NEEDLE_WORKERS[{index}] requires id, role, and base_url")
        if worker_id in seen_ids:
            raise NeedleError(f"NEEDLE_WORKERS contains duplicate worker id: {worker_id}")
        if not isinstance(allowed_raw, list) or not allowed_raw or not all(isinstance(name, str) and name for name in allowed_raw):
            raise NeedleError(f"NEEDLE_WORKERS[{index}].allowed_tools must be a non-empty string array")
        try:
            confidence_threshold = float(item.get("confidence_threshold", default_confidence_threshold))
            timeout = float(item.get("timeout", default_timeout))
        except (TypeError, ValueError) as exc:
            raise NeedleError(f"NEEDLE_WORKERS[{index}] has an invalid timeout or confidence_threshold") from exc
        if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
            raise NeedleError(f"NEEDLE_WORKERS[{index}].confidence_threshold must be between 0 and 1")
        if not math.isfinite(timeout) or timeout <= 0:
            raise NeedleError(f"NEEDLE_WORKERS[{index}].timeout must be a positive finite number")
        seen_ids.add(worker_id)
        workers.append(NeedleWorkerSpec(
            id=worker_id,
            role=role,
            base_url=base_url,
            allowed_tools=tuple(dict.fromkeys(allowed_raw)),
            confidence_threshold=confidence_threshold,
            timeout=timeout,
        ))
    return tuple(workers)


class NeedleOrchestrator:
    """Fan out one task to configured Needle workers and merge valid calls.

    The caller supplies the already capability-filtered schemas for the current
    Aiko turn.  A worker gets the intersection of that set and its static
    allow-list; the response is checked against that exact intersection again
    by :class:`NeedleClient`.
    """

    def __init__(
        self,
        workers: tuple[NeedleWorkerSpec, ...],
        *,
        client_factory: Callable[..., NeedleClient] = NeedleClient,
    ) -> None:
        if not workers:
            raise NeedleError("Needle multi-worker backend requires at least one configured worker")
        self.workers = workers
        self._client_factory = client_factory

    @staticmethod
    def _tools_for_worker(worker: NeedleWorkerSpec, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = set(worker.allowed_tools)
        return [tool for tool in tools if str(tool.get("function", {}).get("name") or "") in allowed]

    @staticmethod
    def _worker_prompt(worker: NeedleWorkerSpec, task: str) -> str:
        return f"Role: {worker.role}.\nTask: {task}"

    def _run_one(self, worker: NeedleWorkerSpec, task: str, tools: list[dict[str, Any]]) -> NeedleWorkerResult:
        worker_tools = self._tools_for_worker(worker, tools)
        if not worker_tools:
            return NeedleWorkerResult(worker.id, worker.role, error="no allowed tools are available for this turn")
        try:
            client = self._client_factory(
                worker.base_url,
                timeout=worker.timeout,
                confidence_threshold=worker.confidence_threshold,
            )
            return NeedleWorkerResult(
                worker.id,
                worker.role,
                response=client.complete(self._worker_prompt(worker, task), worker_tools),
            )
        except NeedleError as exc:
            return NeedleWorkerResult(worker.id, worker.role, error=str(exc))

    def complete(self, task: str, tools: list[dict[str, Any]]) -> tuple[NeedleWorkerResult, ...]:
        """Run all workers concurrently, preserving configured worker order."""
        results: dict[str, NeedleWorkerResult] = {}
        with ThreadPoolExecutor(max_workers=len(self.workers), thread_name_prefix="needle-worker") as pool:
            futures = {pool.submit(self._run_one, worker, task, tools): worker for worker in self.workers}
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    results[worker.id] = future.result()
                except Exception as exc:  # defensive: one worker cannot abort the team
                    results[worker.id] = NeedleWorkerResult(worker.id, worker.role, error=str(exc))
        ordered = tuple(results[worker.id] for worker in self.workers)
        if not any(result.response is not None for result in ordered):
            details = "; ".join(f"{result.worker_id}: {result.error}" for result in ordered)
            raise NeedleError(f"all Needle workers failed or were ineligible: {details}")
        return ordered
