"""
EMC-4: dream-time distillation of episodic memory → semantic facts.

During AikoMemorize.dream(), undistrilled episodes are summarized into
durable facts (via the same LLM extract path style as turn writes) and
written with add_raw. Episodes are then marked distilled_at so they are
not re-processed.

Human analogy: sleep consolidates episodic traces into semantic knowledge.
WM→EM already happened in EMC-2; this is EM→SM.
"""
from __future__ import annotations

import json
import re
from typing import Any

from system.log import get_logger
from cognition.memory.env import env_bool, env_int

log = get_logger(__name__)

EMC_DREAM_ENABLED = env_bool("EMC_DREAM_ENABLED", "1")
EMC_DREAM_LIMIT = max(0, env_int("EMC_DREAM_LIMIT", 12))
EMC_DREAM_BATCH = max(1, env_int("EMC_DREAM_BATCH", 4))
EMC_DREAM_MIN_CHARS = max(20, env_int("EMC_DREAM_MIN_CHARS", 60))
EMC_DREAM_MAX_TOKENS = max(64, env_int("EMC_DREAM_MAX_TOKENS", 256))

_DISTILL_PROMPT = """\
You extract durable long-term facts from past conversation moments.
Only keep stable facts about the user (preferences, identity, plans, relationships).
Skip greetings, one-off logistics, and anything already ephemeral.
Output a JSON array of strings. Empty array if nothing durable.
Each fact must be a single short sentence in third person about the user.

Moments:
{moments}

JSON array:
"""


def ensure_distilled_column(conn) -> None:
    """Add distilled_at + distilled_into to emc_storage if missing (idempotent).

    distilled_into is a JSON array of semantic-memory ids the episode
    consolidated into (EM→SM link used by the LTM/ITM studios).
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(emc_storage)").fetchall()}
        if "distilled_at" not in cols:
            conn.execute("ALTER TABLE emc_storage ADD COLUMN distilled_at TEXT")
            conn.commit()
            log.info("EMC-4: added emc_storage.distilled_at")
        if "distilled_into" not in cols:
            conn.execute("ALTER TABLE emc_storage ADD COLUMN distilled_into TEXT")
            conn.commit()
            log.info("EMC-4: added emc_storage.distilled_into")
    except Exception as e:
        log.debug("EMC-4 distilled_at migration: %s", e)


def _parse_facts(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                fact = (item.get("fact") or item.get("text") or "").strip()
                if fact:
                    out.append(fact)
    return out


def _candidate_episodes(store, user_id: str, limit: int) -> list[dict]:
    ensure_distilled_column(store._conn)
    with store._lock:
        rows = store._conn.execute(
            """
            SELECT id, timestamp, date, trace, salience_score, recall_count
            FROM emc_storage
            WHERE user_id = ?
              AND (superseded_by IS NULL)
              AND (distilled_at IS NULL)
              AND length(trace) >= ?
            ORDER BY
              COALESCE(salience_score, 0) DESC,
              COALESCE(recall_count, 0) DESC,
              timestamp DESC
            LIMIT ?
            """,
            (user_id, EMC_DREAM_MIN_CHARS, limit),
        ).fetchall()
    return [
        {
            "id": int(r[0]),
            "timestamp": r[1],
            "date": r[2],
            "trace": r[3],
            "salience_score": r[4],
            "recall_count": int(r[5] or 0),
        }
        for r in rows
    ]


def _mark_distilled(store, ids: list[int], *, distilled_into: list[str] | None = None) -> None:
    if not ids:
        return
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    into_json = json.dumps(list(distilled_into or []), ensure_ascii=False)
    with store._lock:
        for eid in ids:
            store._conn.execute(
                "UPDATE emc_storage SET distilled_at = ?, distilled_into = ? WHERE id = ?",
                (now, into_json, eid),
            )
        store._conn.commit()


def _llm_distill(client, model: str, moments: list[str]) -> list[str]:
    if not moments or client is None:
        return []
    block = "\n\n".join(f"[{i+1}]\n{m}" for i, m in enumerate(moments))
    prompt = _DISTILL_PROMPT.format(moments=block[:6000])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=EMC_DREAM_MAX_TOKENS,
            temperature=0.0,
            timeout=45.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_facts(raw)
    except Exception as e:
        log.debug("EMC-4 LLM distill failed: %s", e)
        return []


def distill_episodes(
    memorize,
    *,
    user_id: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Distill undistrilled episodes into semantic facts via memorize.add_raw."""
    from cognition.memory.episode import EMC_ENABLED

    result = {
        "candidates": 0,
        "distilled_episodes": 0,
        "facts_written": 0,
        "dry_run": dry_run,
        "enabled": bool(EMC_DREAM_ENABLED and EMC_ENABLED),
    }
    if not EMC_ENABLED or not EMC_DREAM_ENABLED:
        return result

    top = EMC_DREAM_LIMIT if limit is None else max(0, int(limit))
    if top <= 0:
        return result

    uid = user_id or memorize.get_user_id()
    store = None
    try:
        store = memorize._get_episode_store(uid)
    except Exception as e:
        log.debug("EMC-4 no episode store: %s", e)
        return result
    if store is None:
        return result

    try:
        store.flush_all()
    except Exception as e:
        log.debug("EMC-4 flush_all failed: %s", e)

    candidates = _candidate_episodes(store, uid, top)
    result["candidates"] = len(candidates)
    if not candidates:
        return result

    backend = getattr(memorize, "_mem", None)
    client = getattr(backend, "_client", None) if backend else None
    model = getattr(backend, "_model", None) or "ministral"

    facts_written = 0
    distilled_ids: list[int] = []

    for i in range(0, len(candidates), EMC_DREAM_BATCH):
        batch = candidates[i : i + EMC_DREAM_BATCH]
        moments = [c["trace"] for c in batch if (c.get("trace") or "").strip()]
        facts = _llm_distill(client, model, moments)
        if dry_run:
            log.info(
                "EMC-4 dry-run batch episodes=%d facts=%d sample=%r",
                len(batch),
                len(facts),
                (facts[:2] if facts else []),
            )
            # dry-run: never mark; only count batches that would produce facts
            if facts:
                distilled_ids.extend(c["id"] for c in batch)
            continue

        # Only mark distilled when the LLM returned durable facts.
        # Empty extract → leave distilled_at NULL so the episodes can retry next dream.
        if not facts:
            log.debug(
                "EMC-4 empty extract; not marking episodes=%s",
                [c["id"] for c in batch],
            )
            continue

        batch_success = True
        batch_mem_ids: list[str] = []
        for fact in facts:
            try:
                mid = memorize.add_raw(fact, user_id=uid, pinned=False)
                if mid:
                    facts_written += 1
                    batch_mem_ids.append(str(mid))
            except Exception as e:
                log.debug("EMC-4 add_raw failed: %s", e)
                batch_success = False

        # Only mark distilled if all facts were written successfully
        if batch_success:
            ids = [c["id"] for c in batch]
            _mark_distilled(store, ids, distilled_into=batch_mem_ids)
            distilled_ids.extend(ids)

    result["distilled_episodes"] = len(distilled_ids)
    result["facts_written"] = facts_written
    log.info(
        "EMC-4 distill candidates=%d episodes=%d facts=%d dry_run=%s",
        result["candidates"],
        result["distilled_episodes"],
        facts_written,
        dry_run,
    )
    return result


def attach_dream_hook() -> None:
    """Wrap AikoMemorize.dream to run EMC distill after SM consolidation."""
    from cognition.memory.memorize import AikoMemorize

    if getattr(AikoMemorize, "_emc4_dream_patched", False):
        return

    _orig = AikoMemorize.dream

    def dream(self, user_id=None, dry_run=False, threshold=None, **kwargs):
        if threshold is None:
            from cognition.memory.lifecycle import DREAM_MERGE_THRESHOLD
            threshold = DREAM_MERGE_THRESHOLD
        result = _orig(self, user_id=user_id, dry_run=dry_run, threshold=threshold, **kwargs)
        try:
            emc = distill_episodes(self, user_id=user_id, dry_run=dry_run)
            if isinstance(result, dict):
                result["emc_distill"] = emc
        except Exception as e:
            log.warning("EMC-4 distill after dream failed: %s", e)
            if isinstance(result, dict):
                result["emc_distill_error"] = str(e)
        return result

    AikoMemorize.dream = dream  # type: ignore[method-assign]
    AikoMemorize._emc4_dream_patched = True  # type: ignore[attr-defined]
    log.debug("EMC-4 dream hook attached")
