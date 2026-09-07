"""
Lingo SRS core: SM-2 spaced-repetition scheduling over a SQLite-backed
vocabulary table.

Split out of the old lingo_srs.py per lingo_architecture.md: this module
owns only the data model (LingoVocabCard, ReviewLog), the DB schema, and
the LingoSRS scheduler class. Japanese text parsing lives in vocab.py so
each module can be tested independently (you can exercise SM-2 math
without ever touching a real Japanese sentence).

=====================================================================
BUGFIX PASS:
  1. record_review() called `grade.value` assuming its caller always
     passed a valid ReviewGrade enum member. A bad int (e.g. a stray 99
     from a client bug) would raise deep inside SM-2 math with a
     confusing traceback. Added explicit validation that produces a
     clear ValueError up front.
  2. Added get_weak_vocab() -- there was no way to identify cards the
     user is actually struggling with (high review failure rate); it
     just returned whatever was due next, in due-date order. This
     backs the "Practice These" dashboard widget.
  3. Added count_due_cards() -- callers were doing
     `len(srs.get_due_cards(limit=1000))` just to get a count, which
     fetches and deserializes up to 1000 full rows for something a
     single COUNT(*) query answers directly (lingo_audit.md #4).
=====================================================================
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

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

    def count_due_cards(self) -> int:
        """
        Count cards due for review without fetching them (lingo_audit.md #4).
        Callers that only need the number (progress bars, "cards remaining"
        badges) should use this instead of `len(get_due_cards(limit=1000))`,
        which pulls and deserializes up to 1000 full rows just to count them.
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("""
            SELECT COUNT(*) FROM lingo_vocab_cards
            WHERE user_id = ? AND next_review <= ?
        """, (self.user_id, now))
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_weak_vocab(self, limit: int = 10, min_reviews: int = 2, error_threshold: float = 0.4) -> List[LingoVocabCard]:
        """
        Fetch cards the user is actually struggling with, ranked by review
        failure rate, rather than just whatever's next in the due queue. A
        card needs at least `min_reviews` attempts before it's eligible
        (otherwise a single bad first guess dominates the list), and needs
        a failure rate (grade < GOOD) above `error_threshold`.
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT lvc.*,
                   CAST(SUM(CASE WHEN lrl.grade < ? THEN 1 ELSE 0 END) AS FLOAT)
                       / COUNT(lrl.id) AS error_rate,
                   COUNT(lrl.id) AS review_count
            FROM lingo_vocab_cards lvc
            JOIN lingo_review_logs lrl ON lvc.id = lrl.card_id
            WHERE lvc.user_id = ?
            GROUP BY lvc.id
            HAVING review_count >= ? AND error_rate >= ?
            ORDER BY error_rate DESC, review_count DESC
            LIMIT ?
        """, (ReviewGrade.GOOD.value, self.user_id, min_reviews, error_threshold, limit))

        rows = c.fetchall()
        conn.close()
        # Each row has two extra trailing columns (error_rate, review_count)
        # beyond the normal card row shape -- strip them before mapping.
        return [self._row_to_card(row[:16]) for row in rows]

    def record_review(self, card_id: int, grade, response_time_ms: int = 0) -> LingoVocabCard:
        """
        Record a review attempt and update card schedule using SM-2 algorithm.

        SM-2 formula:
            I(1) = 1
            I(2) = 6
            I(n) = I(n-1) x EF

            EF' = EF + (0.1 - (5 - q) * 0.08 - (5 - q) * 0.02)
                where q in [0, 5]

        Intervals are in days; next_review = now + I(n).

        `grade` accepts either a ReviewGrade enum member or a raw int
        (0-4). A raw out-of-range int is validated up front with a clear
        message instead of blowing up deep inside the SM-2 math.
        """
        if isinstance(grade, ReviewGrade):
            grade_enum = grade
        else:
            try:
                grade_enum = ReviewGrade(int(grade))
            except (ValueError, TypeError):
                valid = ", ".join(f"{g.value}={g.name}" for g in ReviewGrade)
                raise ValueError(f"Invalid review grade {grade!r}; must be one of: {valid}")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Fetch current card state
        c.execute("SELECT * FROM lingo_vocab_cards WHERE id = ? AND user_id = ?", (card_id, self.user_id))
        row = c.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Card {card_id} not found for user {self.user_id}")

        card = self._row_to_card(row)
        q = grade_enum.value  # Quality (0-5)

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


__all__ = [
    "LingoVocabCard",
    "ReviewLog",
    "ReviewGrade",
    "LingoSRS",
    "init_srs_db",
    "DB_PATH",
]
