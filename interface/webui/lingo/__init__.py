"""
Lingo: Japanese conversation learning with spaced repetition.

Shared vocab: interface/webui/lingo/vocab_pool.db
Per-user progress: USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db
"""

from .router import router
from .srs import LingoSRS, ReviewGrade, LingoVocabCard, ReviewLog, init_srs_db
from .vocab import VocabExtractor, MemoryBridge

try:
    from .learn_api import attach_learn_routes
    from .router import get_lingo_session, _award_xp
    attach_learn_routes(router, get_lingo_session, award_xp=_award_xp)
except Exception:
    import logging
    logging.getLogger(__name__).warning("learn_api routes not attached", exc_info=True)

try:
    from .spawn import register_lingo_spawn_handler, warm_pools_on_startup, ensure_lingo_spawn_job
    register_lingo_spawn_handler(seed_jobs=False)
    warm_pools_on_startup()

    # Post-login schedule bootstrap does not yet call ensure_lingo_spawn_job
    # directly; wrap it so the hourly job is seeded with other user jobs.
    def _wrap_bootstrap() -> None:
        try:
            from system import schedule as sched
            if getattr(sched.bootstrap_non_system_jobs, "_lingo_spawn_wrapped", False):
                return
            _orig = sched.bootstrap_non_system_jobs

            def _wrapped(*, think=None, memorize=None, timezone=None):
                _orig(think=think, memorize=memorize, timezone=timezone)
                try:
                    uid = None
                    if memorize is not None and hasattr(memorize, "get_user_id"):
                        uid = memorize.get_user_id()
                    ensure_lingo_spawn_job(timezone=timezone, user_id=uid)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "Failed to seed lingo vocab spawn job"
                    )

            _wrapped._lingo_spawn_wrapped = True  # type: ignore[attr-defined]
            sched.bootstrap_non_system_jobs = _wrapped  # type: ignore[assignment]
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Could not wrap schedule bootstrap for lingo spawn", exc_info=True
            )

    _wrap_bootstrap()
except Exception:
    import logging
    logging.getLogger(__name__).warning("lingo spawn handler not registered", exc_info=True)

__all__ = [
    "router",
    "LingoSRS",
    "ReviewGrade",
    "LingoVocabCard",
    "ReviewLog",
    "init_srs_db",
    "VocabExtractor",
    "MemoryBridge",
]
