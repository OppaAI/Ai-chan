"""
Lingo: Japanese conversation learning with spaced repetition.

Shared vocab: interface/webui/lingo/vocab_pool.db (JLPT-tagged)
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
    from .spawn import register_lingo_spawn_handler, warm_pools_on_startup
    from .levels import ensure_owner_n1
    register_lingo_spawn_handler(seed_jobs=False)
    ensure_owner_n1()  # OppaAI / AIKO_USER_ID → N1 in data/levels/
    warm_pools_on_startup()
except Exception:
    import logging
    logging.getLogger(__name__).warning("lingo spawn/level boot failed", exc_info=True)

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
