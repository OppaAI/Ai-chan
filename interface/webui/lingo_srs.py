"""
Lingo SRS Module: Spaced Repetition Scheduling + Vocabulary Tracking

This module integrates Anki-style (SM-2) spaced repetition with Japanese vocabulary
tracking. Vocabulary is extracted from tutor responses, stored in SQLite with review
schedules, and fed back into Aiko's memory system for retention modeling.

Architecture:
- LingoVocabCard: atomic vocab unit (kanji, reading, meaning, context)
- LingoSRS: SM-2 scheduler (interval, ease factor, review history)
- VocabExtractor: parse Japanese text → kanji/word/reading triples
- MemoryBridge: vocab reviews → Aiko LTM with valence (correct=+1, mistake=-1)
"""

import sqlite3
import logging
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import unicodedata

log = logging.getLogger(__name__)

# ============================================================================
# Schema & Constants
# ============================================================================

DB_PATH = Path(__file__).parent / "lingo_vocab.db"

# SM-2 algorithm constants (Anki defaults)
EASY_BONUS = 1.3
HARD_PENALTY = 0.6
AGAIN_INTERVAL = 1  # minutes

class ReviewGrade(Enum):
    """Quality of response (0-5, SM-2 scale)"""
    AGAIN = 0      # Complete blackout; correct response with serious errors or inefficient
    HARD = 1       # Incorrect response, but upon seeing correct response, it's remembered
    GOOD = 2       # Correct response after some hesitation
    EASY = 3       # Correct response without any hesitation
    PERFECT = 4    # Correct response, optimal recall speed


@dataclass
class LingoVocabCard:
    """Atomic vocabulary unit for SRS tracking"""
    id: Optional[int] = None
    user_id: str = ""
    # Core linguistic data
    kanji: str = ""              # Kanji form (if applicable)
    hiragana: str = ""           # Hiragana reading
    romaji: str = ""             # Romaji (for display/matching)
    meaning: str = ""            # English meaning(s)
    pos: str = ""                # Part of speech (noun, verb, adjective, etc)
    context: str = ""            # Example sentence where vocab appeared
    # SRS tracking
    interval: int = 0            # Days until next review
    ease_factor: float = 2.5     # SM-2 ease (default 2.5)
    review_count: int = 0        # Total reviews
    consecutive_correct: int = 0 # Streak of correct answers
    last_review: Optional[str] = None  # ISO datetime of last review
    next_review: Optional[str] = None  # ISO datetime of scheduled review
    created_at: str = ""         # ISO datetime of card creation
    source_context: str = ""     # Lingo conversation turn ID (for traceability)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewLog:
    """Single review attempt for analytics"""
    id: Optional[int] = None
    card_id: int = 0
    user_id: str = ""
    grade: int = 0               # ReviewGrade enum value
    review_date: str = ""        # ISO datetime
    before_interval: int = 0
    after_interval: int = 0
    before_ease: float = 0.0
    after_ease: float = 0.0
    response_time_ms: int = 0    # How long user took to respond

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Database Initialization
# ============================================================================

def init_srs_db():
    """Create SRS tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Vocab card table
    c.execute("""
        CREATE TABLE IF NOT EXISTS lingo_vocab_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kanji TEXT,
            hiragana TEXT NOT NULL,
            romaji TEXT,
            meaning TEXT NOT NULL,
            pos TEXT,
            context TEXT,
            interval INTEGER DEFAULT 0,
            ease_factor REAL DEFAULT 2.5,
            review_count INTEGER DEFAULT 0,
            consecutive_correct INTEGER DEFAULT 0,
            last_review TEXT,
            next_review TEXT,
            created_at TEXT NOT NULL,
            source_context TEXT,
            UNIQUE(user_id, hiragana, meaning)
        )
    """)

    # Review history log
    c.execute("""
        CREATE TABLE IF NOT EXISTS lingo_review_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            grade INTEGER NOT NULL,
            review_date TEXT NOT NULL,
            before_interval INTEGER,
            after_interval INTEGER,
            before_ease REAL,
            after_ease REAL,
            response_time_ms INTEGER,
            FOREIGN KEY(card_id) REFERENCES lingo_vocab_cards(id)
        )
    """)

    # Learning stats (aggregate per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS lingo_learning_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            total_cards INTEGER DEFAULT 0,
            learned_today INTEGER DEFAULT 0,
            reviews_today INTEGER DEFAULT 0,
            avg_ease REAL DEFAULT 2.5,
            last_update TEXT
        )
    """)

    conn.commit()
    conn.close()
    log.info("SRS database initialized")


# ============================================================================
# Vocabulary Card Management
# ============================================================================

class LingoSRS:
    """Interface for SRS operations (CRUD, scheduling, review)"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        init_srs_db()

    def add_card(self, card: LingoVocabCard) -> LingoVocabCard:
        """Insert a new vocab card or retrieve existing."""
        card.user_id = self.user_id
        card.created_at = datetime.utcnow().isoformat()
        card.next_review = datetime.utcnow().isoformat()  # Review immediately

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO lingo_vocab_cards
                (user_id, kanji, hiragana, romaji, meaning, pos, context, 
                 interval, ease_factor, created_at, source_context, next_review)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id, card.kanji, card.hiragana, card.romaji,
                card.meaning, card.pos, card.context,
                card.interval, card.ease_factor,
                card.created_at, card.source_context, card.next_review
            ))
            conn.commit()
            card.id = c.lastrowid
            log.debug(f"Added vocab card: {card.hiragana} ({card.meaning})")
            return card
        except sqlite3.IntegrityError:
            # Card already exists; retrieve it
            log.debug(f"Card already exists: {card.hiragana}")
            c.execute("""
                SELECT * FROM lingo_vocab_cards
                WHERE user_id = ? AND hiragana = ? AND meaning = ?
            """, (self.user_id, card.hiragana, card.meaning))
            row = c.fetchone()
            conn.close()
            return self._row_to_card(row) if row else card
        finally:
            conn.close()

    def get_due_cards(self, limit: int = 10) -> List[LingoVocabCard]:
        """Fetch cards due for review (next_review <= now)"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        now = datetime.utcnow().isoformat()
        c.execute("""
            SELECT * FROM lingo_vocab_cards
            WHERE user_id = ? AND next_review <= ?
            ORDER BY next_review ASC
            LIMIT ?
        """, (self.user_id, now, limit))

        rows = c.fetchall()
        conn.close()
        return [self._row_to_card(row) for row in rows]

    def record_review(self, card_id: int, grade: ReviewGrade, response_time_ms: int = 0) -> LingoVocabCard:
        """
        Record a review attempt and update card schedule using SM-2 algorithm.

        SM-2 formula:
            I(1) = 1
            I(2) = 6
            I(n) = I(n-1) × EF

            EF' = EF + (0.1 − (5 − q) × 0.08 − (5 − q) × 0.02)
                where q ∈ [0, 5]

        Intervals are in days; next_review = now + I(n).
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Fetch current card state
        c.execute("SELECT * FROM lingo_vocab_cards WHERE id = ? AND user_id = ?", (card_id, self.user_id))
        row = c.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Card {card_id} not found for user {self.user_id}")

        card = self._row_to_card(row)
        q = grade.value  # Quality (0-5)

        # SM-2 algorithm
        if q < 3:
            # Incorrect or hard: reset interval
            new_interval = AGAIN_INTERVAL
            new_ease = card.ease_factor
            card.consecutive_correct = 0
        else:
            # Correct: advance interval
            if card.review_count == 0:
                new_interval = 1
            elif card.review_count == 1:
                new_interval = 6
            else:
                new_interval = int(card.interval * card.ease_factor)

            # Update ease factor
            new_ease = max(1.3, card.ease_factor + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            card.consecutive_correct += 1

        # Update card
        now = datetime.utcnow()
        next_review = (now + timedelta(days=new_interval)).isoformat()

        c.execute("""
            UPDATE lingo_vocab_cards
            SET interval = ?, ease_factor = ?, review_count = review_count + 1,
                consecutive_correct = ?, last_review = ?, next_review = ?
            WHERE id = ?
        """, (
            new_interval, new_ease, card.consecutive_correct,
            now.isoformat(), next_review, card.id
        ))

        # Log review
        c.execute("""
            INSERT INTO lingo_review_logs
            (card_id, user_id, grade, review_date, before_interval, after_interval,
             before_ease, after_ease, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, self.user_id, q, now.isoformat(),
            card.interval, new_interval,
            card.ease_factor, new_ease, response_time_ms
        ))

        conn.commit()
        conn.close()

        card.interval = new_interval
        card.ease_factor = new_ease
        card.review_count += 1
        card.last_review = now.isoformat()
        card.next_review = next_review

        log.info(f"Review recorded: {card.hiragana} (grade={q}, interval={new_interval}d)")
        return card

    def get_stats(self) -> Dict:
        """Fetch learning statistics for user"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total cards
        c.execute("SELECT COUNT(*) FROM lingo_vocab_cards WHERE user_id = ?", (self.user_id,))
        total = c.fetchone()[0]

        # Learned today (reviewed today)
        today = datetime.utcnow().date().isoformat()
        c.execute("""
            SELECT COUNT(DISTINCT card_id) FROM lingo_review_logs
            WHERE user_id = ? AND DATE(review_date) = ?
        """, (self.user_id, today))
        learned_today = c.fetchone()[0]

        # Reviews today
        c.execute("""
            SELECT COUNT(*) FROM lingo_review_logs
            WHERE user_id = ? AND DATE(review_date) = ?
        """, (self.user_id, today))
        reviews_today = c.fetchone()[0]

        # Avg ease
        c.execute("""
            SELECT AVG(after_ease) FROM lingo_review_logs
            WHERE user_id = ?
        """, (self.user_id,))
        avg_ease = c.fetchone()[0] or 2.5

        conn.close()

        return {
            "total_cards": total,
            "learned_today": learned_today,
            "reviews_today": reviews_today,
            "avg_ease": round(avg_ease, 2)
        }

    @staticmethod
    def _row_to_card(row: tuple) -> LingoVocabCard:
        """Convert SQLite row to LingoVocabCard"""
        return LingoVocabCard(
            id=row[0], user_id=row[1], kanji=row[2], hiragana=row[3],
            romaji=row[4], meaning=row[5], pos=row[6], context=row[7],
            interval=row[8], ease_factor=row[9], review_count=row[10],
            consecutive_correct=row[11], last_review=row[12], next_review=row[13],
            created_at=row[14], source_context=row[15]
        )


# ============================================================================
# Vocabulary Extraction (Kanji → Word → Reading → Meaning)
# ============================================================================

class VocabExtractor:
    """
    Parse Japanese text and extract vocabulary for SRS cards.

    Strategy:
    1. Segment Japanese text into words (use MeCab-like heuristic or regex)
    2. Classify each token: kanji word, hiragana verb/adj, etc.
    3. Match against known dictionary (fallback: LLM lookup)
    4. Create card with reading + meaning
    """

    # Common kanji readings cache (subset for performance; full dict loaded on demand)
    COMMON_READINGS = {
        "日": {"hiragana": ["ひ", "か"], "meaning": "day, sun"},
        "本": {"hiragana": ["ほん"], "meaning": "book, origin"},
        "人": {"hiragana": ["ひと", "にん"], "meaning": "person"},
        "学": {"hiragana": ["がく"], "meaning": "study, learning"},
        "校": {"hiragana": ["こう"], "meaning": "school"},
        "生": {"hiragana": ["せい", "い"], "meaning": "life, birth"},
        "先": {"hiragana": ["せん"], "meaning": "before, ahead"},
        "生": {"hiragana": ["い"], "meaning": "life, birth (in 先生)"},
        "中": {"hiragana": ["なか", "ちゅう"], "meaning": "middle, inside"},
        "大": {"hiragana": ["だい", "おお"], "meaning": "large, big"},
        "小": {"hiragana": ["しょう", "ちいさ"], "meaning": "small"},
        "朝": {"hiragana": ["あさ"], "meaning": "morning"},
        "子": {"hiragana": ["こ", "し"], "meaning": "child, offspring"},
        "食": {"hiragana": ["しょく", "た"], "meaning": "eat, food"},
        "べ": {"hiragana": ["べ"], "meaning": "eat (verb suffix)"},
        "水": {"hiragana": ["みず"], "meaning": "water"},
    }

    def __init__(self, llm_client=None):
        """llm_client: optional Anthropic client for vocabulary lookup fallback"""
        self.llm_client = llm_client

    def extract_vocab(self, japanese_text: str, context: str = "") -> List[LingoVocabCard]:
        """
        Parse Japanese text and extract vocabulary cards.
        Returns list of LingoVocabCard (ready to add to SRS).
        """
        if not japanese_text:
            return []

        # Remove furigana markup (e.g., "漢字{かんじ}" → "漢字")
        clean_text = re.sub(r'\{[^}]+\}', '', japanese_text)
        # Remove actions/emoji (already done in sanitize, but double-check)
        clean_text = re.sub(r'\*[^*]*\*', '', clean_text)

        cards = []
        seen = set()  # Dedup by (hiragana, meaning) tuple

        # Segment by kanji clusters and hiragana words
        tokens = self._segment_tokens(clean_text)

        for token in tokens:
            if len(token) < 1 or not self._is_japanese(token):
                continue

            # Try to extract vocab from token
            vocab = self._extract_token(token, context)
            if vocab:
                key = (vocab.hiragana, vocab.meaning)
                if key not in seen:
                    cards.append(vocab)
                    seen.add(key)

        log.debug(f"Extracted {len(cards)} vocab cards from: {japanese_text[:50]}")
        return cards

    def _segment_tokens(self, text: str) -> List[str]:
        """
        Segment Japanese text into tokens (rough morphological split).
        Simple heuristic: split on particle-like hiragana; keep kanji clusters.
        """
        # Patterns: kanji clusters, hiragana words, special chars
        pattern = r'[\u4e00-\u9fff]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+'
        return re.findall(pattern, text)

    def _extract_token(self, token: str, context: str = "") -> Optional[LingoVocabCard]:
        """
        Attempt to extract vocab from single token.
        Tries kanji lookup, then hiragana lookup, then LLM fallback.
        """
        # All hiragana: likely verb/adjective
        if self._is_hiragana(token):
            # Known common verb/adjective
            if token in ["です", "ます", "ある", "いる", "する", "なる"]:
                return self._hiragana_card(token, context)
            # Generic hiragana word
            return LingoVocabCard(
                hiragana=token,
                meaning=f"[hiragana: {token}]",  # Fallback
                context=context,
                pos="unknown"
            )

        # Kanji present: try lookup
        if self._has_kanji(token):
            card = self._kanji_lookup(token)
            if card:
                card.context = context
                return card

        return None

    def _kanji_lookup(self, token: str) -> Optional[LingoVocabCard]:
        """Look up kanji token in cache; fallback to LLM if needed"""
        # Exact match in cache
        if token in self.COMMON_READINGS:
            reading_info = self.COMMON_READINGS[token]
            hiragana = reading_info["hiragana"][0]  # Primary reading
            return LingoVocabCard(
                kanji=token,
                hiragana=hiragana,
                romaji=self._hiragana_to_romaji(hiragana),
                meaning=reading_info["meaning"],
                pos="noun"
            )

        # Fallback: assume first hiragana reading is primary (crude but works for common words)
        # Full implementation would use MeCab or similar
        return None

    def _hiragana_card(self, token: str, context: str = "") -> LingoVocabCard:
        """Create card for hiragana token (verb/adjective)"""
        meanings = {
            "です": "is, am, are (polite)",
            "ます": "does, go (polite ending)",
            "ある": "to exist, to have",
            "いる": "to be, to exist (animate)",
            "する": "to do, to make",
            "なる": "to become",
        }
        return LingoVocabCard(
            hiragana=token,
            romaji=self._hiragana_to_romaji(token),
            meaning=meanings.get(token, f"[verb/adj: {token}]"),
            context=context,
            pos="verb" if token.endswith("う") or token in ["する", "ある", "いる"] else "auxiliary"
        )

    @staticmethod
    def _is_japanese(text: str) -> bool:
        """Check if text contains any Japanese characters"""
        return bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))

    @staticmethod
    def _is_hiragana(text: str) -> bool:
        """Check if text is all hiragana"""
        return bool(re.match(r'^[\u3040-\u309f]+$', text))

    @staticmethod
    def _has_kanji(text: str) -> bool:
        """Check if text contains kanji"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    @staticmethod
    def _hiragana_to_romaji(hiragana: str) -> str:
        """Convert hiragana to romaji (simple mapping)"""
        mapping = {
            "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
            "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
            "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
            "さ": "sa", "し": "si", "す": "su", "せ": "se", "そ": "so",
            "ざ": "za", "じ": "zi", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
            "た": "ta", "ち": "ti", "つ": "tu", "て": "te", "と": "to",
            "だ": "da", "ぢ": "di", "づ": "du", "で": "de", "ど": "do",
            "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
            "は": "ha", "ひ": "hi", "ふ": "hu", "へ": "he", "ほ": "ho",
            "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
            "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
            "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
            "や": "ya", "ゆ": "yu", "よ": "yo",
            "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
            "わ": "wa", "を": "wo", "ん": "n",
        }
        return "".join(mapping.get(char, char) for char in hiragana)


# ============================================================================
# Memory Bridge: Vocab Reviews → Aiko LTM
# ============================================================================

class MemoryBridge:
    """
    Integration layer: feed vocabulary reviews into Aiko's long-term memory system.

    Valence mapping:
    - Correct (grade >= GOOD): +1 valence (learning success)
    - Mistake (grade < GOOD): -1 valence (reinforces correction)
    """

    def __init__(self, aiko_instance=None):
        """aiko_instance: Aiko's main class (for access to _memory)"""
        self.aiko = aiko_instance

    def record_vocab_event(
        self,
        user_id: str,
        card: LingoVocabCard,
        grade: ReviewGrade,
        is_correct: bool
    ):
        """Log vocabulary review to Aiko's episodic memory."""
        if not self.aiko or not hasattr(self.aiko, '_memory'):
            log.warning("Aiko memory not available; skipping memory bridge")
            return

        valence = 1 if is_correct else -1
        event_text = (
            f"JJ reviewed '{card.hiragana}' ({card.meaning}) in Lingo. "
            f"Grade: {grade.name}. "
            f"{'Correct—building retention.' if is_correct else 'Corrected—reinforcing pattern.'}"
        )

        try:
            # Insert episodic memory (scene-based)
            scene_id = f"lingo_vocab_{card.id}_{datetime.utcnow().timestamp()}"
            self.aiko._memory.insert_episodic(
                user_id=user_id,
                scene_id=scene_id,
                content=event_text,
                valence=valence,
                tags=["lingo", "vocabulary", "srs", card.hiragana],
                source="lingo_srs"
            )
            log.info(f"Vocab event logged to LTM: {event_text[:60]}")
        except Exception as e:
            log.error(f"Failed to log vocab event to memory: {e}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "LingoVocabCard",
    "ReviewLog",
    "ReviewGrade",
    "LingoSRS",
    "VocabExtractor",
    "MemoryBridge",
    "init_srs_db",
]
