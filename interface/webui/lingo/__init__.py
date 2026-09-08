"""
Lingo: Japanese conversation learning with spaced repetition.

Main exports:
- router: FastAPI router (include in your app)
- LingoSRS: Spaced repetition scheduler (for testing/extensions)
- VocabExtractor: Japanese -> vocabulary parser
- MemoryBridge: vocab reviews -> Aiko long-term memory
- ReviewGrade: Enum for SM-2 grades (0-4)

Shared vocab content: interface/webui/lingo/vocab_pool.db (spawn.py)
Per-user progress: USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db (srs.py)
"""

from .router import router
from .srs import LingoSRS, ReviewGrade, LingoVocabCard, ReviewLog, init_srs_db
from .vocab import VocabExtractor, MemoryBridge

# Attach Learn API (shared pool → per-user progress) without rewriting router.py
try:
    from .learn_api import attach_learn_routes
    from .router import get_lingo_session, _award_xp
    attach_learn_routes(router, get_lingo_session, award_xp=_award_xp)
except Exception:
    import logging
    logging.getLogger(__name__).warning("learn_api routes not attached", exc_info=True)

# Hourly spawn handler + optional A+B startup warm (non-blocking)
try:
    from .spawn import register_lingo_spawn_handler, warm_pools_on_startup, ensure_lingo_spawn_job
    register_lingo_spawn_handler(seed_jobs=False)
    warm_pools_on_startup()
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
    "ensure_lingo_spawn_job",
]
