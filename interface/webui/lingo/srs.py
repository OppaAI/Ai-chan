"""
Lingo SRS core: SM-2 over per-user SQLite at USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

log = logging.getLogger(__name__)

_LEGACY_DB_PATH = Path(__file__).parent / "lingo_vocab.db"
DB_PATH = _LEGACY_DB_PATH
EASY_BONUS = 1.3
HARD_PENALTY = 0.6
AGAIN_INTERVAL = 1


def db_path_for(user_id: str) -> Path:
    try:
        from system.userspace import user_state_dir
        root = user_state_dir(user_id) / "agentic" / "lingo"
    except Exception:
        root = Path(__file__).parent / "user_data" / (user_id or "guest") / "agentic" / "lingo"
    root.mkdir(parents=True, exist_ok=True)
    return root / "vocab.db"


class ReviewGrade(Enum):
    AGAIN = 0
    HARD = 1
    GOOD = 2
    EASY = 3
    PERFECT = 4


@dataclass
class LingoVocabCard:
    id: Optional[int] = None
    user_id: str = ""
    kanji: str = ""
    hiragana: str = ""
    romaji: str = ""
    meaning: str = ""
    pos: str = ""
    context: str = ""
    interval: int = 0
    ease_factor: float = 2.5
    review_count: int = 0
    consecutive_correct: int = 0
    last_review: Optional[str] = None
    next_review: Optional[str] = None
    created_at: str = ""
    source_context: str = ""
    level: str = "N5"  # JLPT track at learn time; sessions filter level<=user

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewLog:
    id: Optional[int] = None
    card_id: int = 0
    user_id: str = ""
    grade: int = 0
    review_date: str = ""
    before_interval: int = 0
    after_interval: int = 0
    before_ease: float = 0.0
    after_ease: float = 0.0
    response_time_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def init_srs_db(db_path: Path | None = None):
    path = Path(db_path) if db_path is not None else _LEGACY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS lingo_vocab_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, kanji TEXT,
        hiragana TEXT NOT NULL, romaji TEXT, meaning TEXT NOT NULL, pos TEXT, context TEXT,
        interval INTEGER DEFAULT 0, ease_factor REAL DEFAULT 2.5, review_count INTEGER DEFAULT 0,
        consecutive_correct INTEGER DEFAULT 0, last_review TEXT, next_review TEXT,
        created_at TEXT NOT NULL, source_context TEXT, level TEXT DEFAULT 'N5',
        UNIQUE(user_id, hiragana, meaning))""")
    # Migrate pre-level DBs (per-user vocab.db files created before this).
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(lingo_vocab_cards)").fetchall()]
        if "level" not in cols:
            c.execute("ALTER TABLE lingo_vocab_cards ADD COLUMN level TEXT DEFAULT 'N5'")
    except sqlite3.Error:
        pass
    c.execute("""CREATE TABLE IF NOT EXISTS lingo_review_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, card_id INTEGER NOT NULL, user_id TEXT NOT NULL,
        grade INTEGER NOT NULL, review_date TEXT NOT NULL, before_interval INTEGER,
        after_interval INTEGER, before_ease REAL, after_ease REAL, response_time_ms INTEGER,
        FOREIGN KEY(card_id) REFERENCES lingo_vocab_cards(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS lingo_learning_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT UNIQUE NOT NULL,
        total_cards INTEGER DEFAULT 0, learned_today INTEGER DEFAULT 0,
        reviews_today INTEGER DEFAULT 0, avg_ease REAL DEFAULT 2.5, last_update TEXT)""")
    conn.commit()
    conn.close()


def _migrate_legacy_user_rows(user_id: str, dest: Path) -> None:
    if not _LEGACY_DB_PATH.is_file() or not user_id:
        return
    flag = dest.with_suffix(".migrated")
    if flag.is_file():
        return
    try:
        src = sqlite3.connect(_LEGACY_DB_PATH)
        dst = sqlite3.connect(dest)
        try:
            src.row_factory = sqlite3.Row
            init_srs_db(dest)
            for table in ("lingo_vocab_cards", "lingo_review_logs", "lingo_learning_stats"):
                try:
                    rows = src.execute(f"SELECT * FROM {table} WHERE user_id = ?", (user_id,)).fetchall()
                except sqlite3.OperationalError:
                    continue
                if not rows:
                    continue
                cols = rows[0].keys()
                ph = ",".join("?" * len(cols))
                cl = ",".join(cols)
                for row in rows:
                    try:
                        dst.execute(f"INSERT OR IGNORE INTO {table} ({cl}) VALUES ({ph})", tuple(row[c] for c in cols))
                    except sqlite3.Error:
                        continue
            dst.commit()
            flag.write_text("ok\n", encoding="utf-8")
        finally:
            src.close()
            dst.close()
    except Exception:
        log.warning("Legacy SRS migration failed for %s", user_id, exc_info=True)


class LingoSRS:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_path = db_path_for(user_id)
        init_srs_db(self.db_path)
        _migrate_legacy_user_rows(user_id, self.db_path)

    def add_card(self, card: LingoVocabCard) -> LingoVocabCard:
        card.user_id = self.user_id
        card.created_at = datetime.utcnow().isoformat()
        card.next_review = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("""INSERT INTO lingo_vocab_cards
                (user_id, kanji, hiragana, romaji, meaning, pos, context,
                 interval, ease_factor, created_at, source_context, next_review, level)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.user_id, card.kanji, card.hiragana, card.romaji, card.meaning, card.pos, card.context,
                 card.interval, card.ease_factor, card.created_at, card.source_context, card.next_review,
                 card.level or "N5"))
            conn.commit()
            card.id = c.lastrowid
            return card
        except sqlite3.IntegrityError:
            c.execute("SELECT * FROM lingo_vocab_cards WHERE user_id=? AND hiragana=? AND meaning=?",
                      (self.user_id, card.hiragana, card.meaning))
            row = c.fetchone()
            return self._row_to_card(row) if row else card
        finally:
            conn.close()

    def get_due_cards(self, limit: int = 10) -> List[LingoVocabCard]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("SELECT * FROM lingo_vocab_cards WHERE user_id=? AND next_review<=? ORDER BY next_review ASC LIMIT ?",
                  (self.user_id, now, limit))
        rows = c.fetchall()
        conn.close()
        return [self._row_to_card(r) for r in rows]

    def count_due_cards(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("SELECT COUNT(*) FROM lingo_vocab_cards WHERE user_id=? AND next_review<=?", (self.user_id, now))
        n = c.fetchone()[0]
        conn.close()
        return n

    def get_weak_vocab(self, limit: int = 10, min_reviews: int = 2, error_threshold: float = 0.4) -> List[LingoVocabCard]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""SELECT lvc.*,
            CAST(SUM(CASE WHEN lrl.grade < ? THEN 1 ELSE 0 END) AS FLOAT)/COUNT(lrl.id) AS error_rate,
            COUNT(lrl.id) AS review_count
            FROM lingo_vocab_cards lvc JOIN lingo_review_logs lrl ON lvc.id = lrl.card_id
            WHERE lvc.user_id = ? GROUP BY lvc.id
            HAVING review_count >= ? AND error_rate >= ?
            ORDER BY error_rate DESC, review_count DESC LIMIT ?""",
            (ReviewGrade.GOOD.value, self.user_id, min_reviews, error_threshold, limit))
        rows = c.fetchall()
        conn.close()
        return [self._row_to_card(r[:17]) for r in rows]

    def get_random_cards(self, limit: int = 10,
                         allowed_levels: Optional[List[str]] = None) -> List[LingoVocabCard]:
        """Random learnt cards for Review/Practice sessions (repeats across
        sessions allowed; level-gated to equal-or-lower JLPT)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        rows = c.execute("SELECT * FROM lingo_vocab_cards WHERE user_id=?", (self.user_id,)).fetchall()
        conn.close()
        cards = [self._row_to_card(r) for r in rows]
        if allowed_levels is not None:
            allowed = set(allowed_levels)
            cards = [cd for cd in cards if (cd.level or "N5") in allowed]
        import random
        random.shuffle(cards)
        return cards[:limit]

    def record_review(self, card_id: int, grade, response_time_ms: int = 0) -> LingoVocabCard:
        if isinstance(grade, ReviewGrade):
            ge = grade
        else:
            try:
                ge = ReviewGrade(int(grade))
            except (ValueError, TypeError):
                valid = ", ".join(f"{g.value}={g.name}" for g in ReviewGrade)
                raise ValueError(f"Invalid review grade {grade!r}; must be one of: {valid}")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM lingo_vocab_cards WHERE id=? AND user_id=?", (card_id, self.user_id))
        row = c.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Card {card_id} not found for user {self.user_id}")
        card = self._row_to_card(row)
        q = ge.value
        if q < 3:
            new_interval, new_ease = AGAIN_INTERVAL, card.ease_factor
            card.consecutive_correct = 0
        else:
            if card.review_count == 0:
                new_interval = 1
            elif card.review_count == 1:
                new_interval = 6
            else:
                new_interval = int(card.interval * card.ease_factor)
            new_ease = max(1.3, card.ease_factor + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            card.consecutive_correct += 1
        now = datetime.utcnow()
        next_review = (now + timedelta(days=new_interval)).isoformat()
        c.execute("""UPDATE lingo_vocab_cards SET interval=?, ease_factor=?, review_count=review_count+1,
            consecutive_correct=?, last_review=?, next_review=? WHERE id=?""",
            (new_interval, new_ease, card.consecutive_correct, now.isoformat(), next_review, card.id))
        c.execute("""INSERT INTO lingo_review_logs
            (card_id, user_id, grade, review_date, before_interval, after_interval, before_ease, after_ease, response_time_ms)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (card_id, self.user_id, q, now.isoformat(), card.interval, new_interval, card.ease_factor, new_ease, response_time_ms))
        conn.commit()
        conn.close()
        card.interval, card.ease_factor = new_interval, new_ease
        card.review_count += 1
        card.last_review, card.next_review = now.isoformat(), next_review
        return card

    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lingo_vocab_cards WHERE user_id=?", (self.user_id,))
        total = c.fetchone()[0]
        today = datetime.utcnow().date().isoformat()
        c.execute("SELECT COUNT(DISTINCT card_id) FROM lingo_review_logs WHERE user_id=? AND DATE(review_date)=?",
                  (self.user_id, today))
        learned_today = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM lingo_review_logs WHERE user_id=? AND DATE(review_date)=?",
                  (self.user_id, today))
        reviews_today = c.fetchone()[0]
        c.execute("SELECT AVG(after_ease) FROM lingo_review_logs WHERE user_id=?", (self.user_id,))
        avg_ease = c.fetchone()[0] or 2.5
        conn.close()
        return {"total_cards": total, "learned_today": learned_today,
                "reviews_today": reviews_today, "avg_ease": round(avg_ease, 2)}

    @staticmethod
    def _row_to_card(row: tuple) -> LingoVocabCard:
        # 16-col legacy rows (no level) vs 17-col current rows. Weak-vocab
        # rows carry 2 trailing aggregate cols -- callers already strip them.
        raw_level = row[16] if len(row) > 16 else "N5"
        level = raw_level if isinstance(raw_level, str) and raw_level else "N5"
        return LingoVocabCard(
            id=row[0], user_id=row[1], kanji=row[2], hiragana=row[3], romaji=row[4],
            meaning=row[5], pos=row[6], context=row[7], interval=row[8], ease_factor=row[9],
            review_count=row[10], consecutive_correct=row[11], last_review=row[12],
            next_review=row[13], created_at=row[14], source_context=row[15], level=level)


__all__ = ["LingoVocabCard", "ReviewLog", "ReviewGrade", "LingoSRS", "init_srs_db", "DB_PATH", "db_path_for"]
