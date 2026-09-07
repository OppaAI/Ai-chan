"""
Lingo: Japanese conversation learning with spaced repetition.

Main exports:
- router: FastAPI router (include in your app)
- LingoSRS: Spaced repetition scheduler (for testing/extensions)
- VocabExtractor: Japanese -> vocabulary parser
- MemoryBridge: vocab reviews -> Aiko long-term memory
- ReviewGrade: Enum for SM-2 grades (0-4)
"""

from .router import router
from .srs import LingoSRS, ReviewGrade, LingoVocabCard, ReviewLog, init_srs_db
from .vocab import VocabExtractor, MemoryBridge

__version__ = "1.1.0"
__all__ = [
    "router",
    "LingoSRS",
    "VocabExtractor",
    "MemoryBridge",
    "ReviewGrade",
    "LingoVocabCard",
    "ReviewLog",
    "init_srs_db",
]
