"""
EMC-2/3 runtime wire-up.

EMC-2: AikoMemorize.queue_episode + lazy EpisodicStore + flush on switch_user.
EMC-3: wrap format_for_context to append episodic recall (joint budget).

Think-side ingest is native in AikoThink._store_async (queue_episode once).
This module does NOT wrap _store_async.

Called from ensure_episode_schema (memory boot path).
"""
from __future__ import annotations

from system.log import get_logger

log = get_logger(__name__)

_WIRED = False


def apply_emc2_hooks() -> None:
    """Idempotent: memorize queue_episode + format_for_context EM append."""
    global _WIRED
    if _WIRED:
        return
    try:
        from cognition.memory.episode_recall import attach_recall_to_store
        attach_recall_to_store()
        _patch_memorize()
        _WIRED = True
        log.info("EMC hooks applied (queue_episode + episodic recall format)")
    except Exception as e:
        log.warning("EMC hooks failed: %s", e)


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

    # EMC-3: episodic recall on format_for_context (once)
    if not getattr(AikoMemorize, "_emc3_format_patched", False):
        _orig_fmt = AikoMemorize.format_for_context

        def format_for_context(
            self,
            memories,
            *,
            query: str = "",
            related=None,
            user_id=None,
            embedder=None,
        ):
            sm_block = _orig_fmt(
                self,
                memories,
                query=query,
                related=related,
                user_id=user_id,
                embedder=embedder,
            )
            try:
                em_block = self._format_episodes_for_context(query or "")
            except Exception as e:
                log.debug("EMC format_episodes skipped: %s", e)
                em_block = None

            if not em_block:
                return sm_block
            if not sm_block:
                return em_block

            try:
                from cognition.memory.episode_recall import EMC_JOINT_BUDGET
                from cognition.memory.schema import MEMORY_CONTEXT_TOTAL_CHARS
            except Exception:
                return f"{sm_block}\n\n{em_block}"

            if not EMC_JOINT_BUDGET:
                return f"{sm_block}\n\n{em_block}"

            shared = int(MEMORY_CONTEXT_TOTAL_CHARS)
            if len(em_block) > shared:
                from cognition.memory.episode_recall import EMC_RECALL_LIMIT
                store = self._get_episode_store()
                if store is not None:
                    hits = store.search(query or "", limit=EMC_RECALL_LIMIT, user_id=self.get_user_id())
                    em_block = store.format_for_context(hits, max_chars=shared) or em_block
            em_len = len(em_block)
            sm_budget = shared - em_len - 2
            if len(sm_block) > sm_budget:
                sm_closing = "\n</memory_context>"
                cut = sm_budget - len(sm_closing) if "</memory_context>" in sm_block else sm_budget
                sm_trim = sm_block[:cut]
                if "</memory_context>" in sm_block and "</memory_context>" not in sm_trim:
                    sm_trim = sm_trim.rstrip() + sm_closing
                sm_block = sm_trim
            return f"{sm_block}\n\n{em_block}"

        def _format_episodes_for_context(self, query: str) -> str | None:
            from cognition.memory.episode import EMC_ENABLED
            from cognition.memory.episode_recall import (
                EMC_RECALL_ENABLED,
                EMC_RECALL_LIMIT,
            )
            if not EMC_ENABLED or not EMC_RECALL_ENABLED or EMC_RECALL_LIMIT <= 0:
                return None
            if not (query or "").strip():
                return None
            store = self._get_episode_store()
            if store is None:
                return None
            hits = store.search(query, limit=EMC_RECALL_LIMIT, user_id=self.get_user_id())
            if not hits:
                return None
            return store.format_for_context(hits)

        AikoMemorize.format_for_context = format_for_context  # type: ignore[method-assign]
        AikoMemorize._format_episodes_for_context = _format_episodes_for_context  # type: ignore[method-assign]
        AikoMemorize._emc3_format_patched = True  # type: ignore[attr-defined]


apply_emc3_hooks = apply_emc2_hooks
