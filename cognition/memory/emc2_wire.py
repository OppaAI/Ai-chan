"""
EMC-2 runtime wire-up.

Applies best-effort hooks so conversation turns enter the episodic buffer
without requiring large-file rewrites of memorize.py / think.py.

Called once from ensure_episode_schema (already on the memory boot path)
and also safe to call from wakeup.
"""
from __future__ import annotations

from system.log import get_logger

log = get_logger(__name__)

_WIRED = False


def apply_emc2_hooks() -> None:
    """Idempotent: patch AikoMemorize + AikoThink for EMC-2 ingest."""
    global _WIRED
    if _WIRED:
        return
    try:
        _patch_memorize()
        _patch_think()
        _WIRED = True
        log.info("EMC-2 hooks applied (queue_episode + _store_async)")
    except Exception as e:
        log.warning("EMC-2 hooks failed: %s", e)


def _patch_memorize() -> None:
    from cognition.memory.memorize import AikoMemorize

    if getattr(AikoMemorize, "queue_episode", None) is not None and getattr(
        AikoMemorize, "_emc2_native", False
    ):
        return  # already native in class body

    def queue_episode(self, user_input: str, response_text: str) -> None:
        """EMC-2: stage one turn into episodic memory (best-effort)."""
        try:
            store = self._get_episode_store()
            if store is None:
                return
            store.ingest_turn(user_input, response_text, user_id=self.get_user_id())
        except Exception as e:
            log.debug("queue_episode skipped: %s", e)

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

    AikoMemorize.queue_episode = queue_episode  # type: ignore[method-assign]
    AikoMemorize._get_episode_store = _get_episode_store  # type: ignore[method-assign]

    # Flush on switch_user
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


def _patch_think() -> None:
    try:
        from cognition.think import AikoThink
    except Exception as e:
        log.debug("EMC-2 think patch deferred: %s", e)
        return

    if getattr(AikoThink, "_emc2_store_patched", False):
        return

    _orig = AikoThink._store_async

    def _store_async(self, user_input: str, response_text: str) -> None:
        _orig(self, user_input, response_text)
        try:
            mem = self._get_memorize()
            if mem is not None and hasattr(mem, "queue_episode"):
                mem.queue_episode(user_input, response_text)
        except Exception:
            pass

    AikoThink._store_async = _store_async  # type: ignore[method-assign]
    AikoThink._emc2_store_patched = True  # type: ignore[attr-defined]
