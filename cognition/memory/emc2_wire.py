"""
EMC-2 runtime wire-up.

Provides AikoMemorize.queue_episode + lazy EpisodicStore when not yet
native on the class. Think-side ingest is native in AikoThink._store_async
(calls queue_episode after queue_write) — this module does NOT wrap
_store_async, so each turn is staged once.

Called from ensure_episode_schema (memory boot path).
"""
from __future__ import annotations

from system.log import get_logger

log = get_logger(__name__)

_WIRED = False


def apply_emc2_hooks() -> None:
    """Idempotent: ensure AikoMemorize has queue_episode / episode store."""
    global _WIRED
    if _WIRED:
        return
    try:
        _patch_memorize()
        _WIRED = True
        log.info("EMC-2 hooks applied (queue_episode on AikoMemorize)")
    except Exception as e:
        log.warning("EMC-2 hooks failed: %s", e)


def _patch_memorize() -> None:
    from cognition.memory.memorize import AikoMemorize

    if getattr(AikoMemorize, "_emc2_native", False):
        return  # already native in class body

    if not hasattr(AikoMemorize, "queue_episode"):

        def queue_episode(self, user_input: str, response_text: str) -> None:
            """EMC-2: stage one turn into episodic memory (best-effort)."""
            try:
                store = self._get_episode_store()
                if store is None:
                    return
                store.ingest_turn(user_input, response_text, user_id=self.get_user_id())
            except Exception as e:
                log.debug("queue_episode skipped: %s", e)

        AikoMemorize.queue_episode = queue_episode  # type: ignore[method-assign]

    if not hasattr(AikoMemorize, "_get_episode_store"):

        def _get_episode_store(self):
            if getattr(self, "_episode_store", None) is not None:
                return self._episode_store
            try:
                from cognition.memory.episode import EpisodicStore, EMC_ENABLED
                if not EMC_ENABLED:
                    return None
                from cognition.memory.schema import _memory_db_path_for_user
                uid = self.get_user_id()
                self._episode_store = EpisodicStore(
                    _memory_db_path_for_user(uid),
                    user_id=uid,
                    embedder=self._mem._embedder,
                )
                return self._episode_store
            except Exception as e:
                log.debug("episode store init failed: %s", e)
                return None

        AikoMemorize._get_episode_store = _get_episode_store  # type: ignore[method-assign]

    # Flush on switch_user (only wrap once)
    if not getattr(AikoMemorize, "_emc2_switch_patched", False):
        _orig_switch = AikoMemorize.switch_user

        def switch_user(self, user_id: str) -> None:
            if getattr(self, "_episode_store", None) is not None:
                try:
                    self._episode_store.flush_all()
                    self._episode_store.close()
                except Exception:
                    log.debug("episode store flush on switch_user failed")
                self._episode_store = None
            return _orig_switch(self, user_id)

        AikoMemorize.switch_user = switch_user  # type: ignore[method-assign]
        AikoMemorize._emc2_switch_patched = True  # type: ignore[attr-defined]
