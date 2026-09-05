"""cognition/memory/grasp.py - Grasp (temporary working memory) framework for Aiko.

Fast, capacity-limited active buffer for the *current conversational focus*.
Miller 7±2 soft slots + token-budget dual guard. All scoring is pure Python
(no embedder, no LLM on the hot path).

Lifecycle: Induction -> Filling -> Sustaining -> Receding -> Evicting

Daily journal: single date-stamped JSONL, append-on-fill (faithful trail).
Day boundary = local calendar date of the *user* turn timestamp.
Writes are async (daemon thread) so fill() stays near-zero latency.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from system.config import env_flag, env_float, env_int, env_str

log = logging.getLogger("aiko.grasp")

# brain_trace is the optional per-step tracer (--trace). Imported lazily to
# avoid a hard dep when --trace is off; the import resolves at first use.
try:
    from system import brain_trace as _brain_trace
    _BRAIN_TRACE_OK = True
except Exception:
    _BRAIN_TRACE_OK = False
    _brain_trace = None

GRASP_ENABLED = env_flag("GRASP_ENABLED", "1")
GRASP_MILLER_MIN = max(1, env_int("GRASP_MILLER_MIN", 5))
GRASP_MILLER_MAX = max(GRASP_MILLER_MIN, env_int("GRASP_MILLER_MAX", 9))
GRASP_MILLER_CENTER = max(GRASP_MILLER_MIN, min(GRASP_MILLER_MAX, env_int("GRASP_MILLER_CENTER", 7)))
GRASP_TOKEN_BUDGET = max(0, env_int("GRASP_TOKEN_BUDGET", 0))
GRASP_W_EMOTION = env_float("GRASP_W_EMOTION", 0.14)
GRASP_W_IMPORTANCE = env_float("GRASP_W_IMPORTANCE", 0.17)
GRASP_W_RECENCY = env_float("GRASP_W_RECENCY", 0.14)
GRASP_W_RELEVANCE = env_float("GRASP_W_RELEVANCE", 0.11)
GRASP_W_NOVELTY = env_float("GRASP_W_NOVELTY", 0.11)
GRASP_W_QUESTION = env_float("GRASP_W_QUESTION", 0.09)
GRASP_W_ENTITY = env_float("GRASP_W_ENTITY", 0.08)
GRASP_W_RECALL_FREQ = env_float("GRASP_W_RECALL_FREQ", 0.09)
GRASP_W_PRIMACY = env_float("GRASP_W_PRIMACY", 0.07)
GRASP_RECENCY_HALF_LIFE = max(1.0, env_float("GRASP_RECENCY_HALF_LIFE", 4.0))
GRASP_RECALL_FREQ_CAP = max(1, env_int("GRASP_RECALL_FREQ_CAP", 6))
GRASP_PRIMACY_SPAN = max(1.0, env_float("GRASP_PRIMACY_SPAN", 6.0))
GRASP_JOURNAL_ENABLED = env_flag("GRASP_JOURNAL_ENABLED", "1")
GRASP_JOURNAL_DIR = env_str("GRASP_JOURNAL_DIR", str(Path.home() / ".local" / "share" / "aiko" / "journal"))
_CHARS_PER_TOKEN = 4.0

# tiktoken is not a dependency — probe once at import so estimate_tokens()
# doesn't pay a failed import + exception on every call in the hot path.
try:
    import tiktoken as _tiktoken
    _tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken = None

_POS_WORDS = re.compile(r"\b(love|great|awesome|amazing|happy|glad|thanks|thank you|yay|nice|good|wonderful|excellent|perfect)\b", re.I)
_NEG_WORDS = re.compile(r"\b(hate|awful|terrible|sad|angry|mad|upset|annoyed|frustrated|bad|worst|sucks|furious)\b", re.I)
_IMPORTANCE_RE = re.compile(
    r"\b(remember|pin|important|always|never|prefer|favorite|favourite|my name|i am|i'm|birthday|allergy|"
    r"allergic|don't forget|do not forget|please note|key point|from now on)\b",
    re.I,
)
_QUESTION_RE = re.compile(r"\?|\b(what|why|how|when|where|who|which|can you|could you|would you|do you|is there)\b", re.I)
_NAME_LIKE_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.:]\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_POS_EMOJI = set("😀😃😄😁🙂😊😍🥰❤️👍🎉✨🔥")
_NEG_EMOJI = set("😞😢😭😠😡💔👎😱")

def _local_tz():
    try:
        return datetime.now().astimezone().tzinfo or timezone.utc
    except Exception:
        return timezone.utc

def local_now() -> datetime:
    return datetime.now(tz=_local_tz())

def ensure_aware(dt: datetime | None = None) -> datetime:
    if dt is None:
        return local_now()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_local_tz())
    return dt.astimezone(_local_tz())

def day_key_from_user_ts(user_ts: datetime | float | None) -> str:
    if isinstance(user_ts, (int, float)):
        dt = datetime.fromtimestamp(user_ts, tz=_local_tz())
    elif isinstance(user_ts, datetime):
        dt = ensure_aware(user_ts)
    else:
        dt = local_now()
    return dt.strftime("%Y-%m-%d")

def journal_path_for_day(day: str | None = None, journal_dir: str | Path | None = None) -> Path:
    d = day or day_key_from_user_ts(None)
    return Path(journal_dir or GRASP_JOURNAL_DIR) / f"{d}.jsonl"

@dataclass
class GraspTurn:
    user: str
    assistant: str
    tokens: int
    emotion: float = 0.0
    importance: float = 0.0
    relevance: float = 0.0
    novelty: float = 0.0
    question: float = 0.0
    entity: float = 0.0
    recall_count: int = 0
    created_turn: int = 0
    user_ts: float = field(default_factory=time.time)
    assistant_ts: float | None = None
    created_at: float = field(default_factory=time.time)
    score: float = 0.0
    _token_set: set[str] = field(default_factory=set, repr=False)

    def text(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"

    def factor_breakdown(self) -> dict[str, float]:
        return {
            "emotion": (self.emotion + 1.0) * 0.5,
            "importance": self.importance,
            "recency": 0.0,
            "relevance": self.relevance,
            "novelty": self.novelty,
            "question": self.question,
            "entity": self.entity,
            "recall_freq": min(1.0, self.recall_count / max(1, GRASP_RECALL_FREQ_CAP)),
            "primacy": 0.0,
            "score": self.score,
        }

    def to_journal_record(self, *, event: str = "fill") -> dict:
        ts = datetime.fromtimestamp(self.user_ts, tz=_local_tz())
        return {
            "ts": ts.isoformat(timespec="seconds"),
            "ts_unix": self.user_ts,
            "day": day_key_from_user_ts(self.user_ts),
            "event": event,
            "turn": self.created_turn,
            "user": self.user,
            "assistant": self.assistant,
            "tokens": self.tokens,
            "score": round(self.score, 4),
            "recall_count": self.recall_count,
            "factors": {
                "emotion": round((self.emotion + 1.0) * 0.5, 4),
                "importance": round(self.importance, 4),
                "relevance": round(self.relevance, 4),
                "novelty": round(self.novelty, 4),
                "question": round(self.question, 4),
                "entity": round(self.entity, 4),
                "recall_freq": round(min(1.0, self.recall_count / max(1, GRASP_RECALL_FREQ_CAP)), 4),
            },
            "assistant_ts": self.assistant_ts,
        }

class DailyJournal:
    def __init__(self, journal_dir: str | Path | None = None, enabled: bool | None = None):
        self.journal_dir = Path(journal_dir or GRASP_JOURNAL_DIR)
        self.enabled = GRASP_JOURNAL_ENABLED if enabled is None else bool(enabled)
        self._lock = threading.Lock()

    def path_for(self, user_ts: datetime | float | None = None) -> Path:
        return journal_path_for_day(day_key_from_user_ts(user_ts), self.journal_dir)

    def append_turn(self, turn: GraspTurn, *, event: str = "fill") -> None:
        if not self.enabled:
            return
        record = turn.to_journal_record(event=event)
        path = self.path_for(turn.user_ts)
        def _write() -> None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with self._lock:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(line)
            except Exception:
                pass
        threading.Thread(target=_write, daemon=True).start()

    def read_day(self, day: str | None = None) -> list[dict]:
        if day is None:
            day = (local_now() - timedelta(days=1)).strftime("%Y-%m-%d")
        path = journal_path_for_day(day, self.journal_dir)
        if not path.is_file():
            return []
        out: list[dict] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        out.sort(key=lambda r: (r.get("ts_unix") or 0, r.get("turn") or 0))
        return out

    def yesterday(self) -> list[dict]:
        return self.read_day(None)

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _tiktoken is not None:
        try:
            return len(_tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / _CHARS_PER_TOKEN))

def _content_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))

def score_emotion(user: str, assistant: str) -> float:
    text = f"{user} {assistant}"
    pos = 1 if _POS_WORDS.search(text) else 0
    neg = 1 if _NEG_WORDS.search(text) else 0
    for ch in text:
        if ch in _POS_EMOJI:
            pos += 1
        elif ch in _NEG_EMOJI:
            neg += 1
    if pos == neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))

def score_importance(user: str, assistant: str) -> float:
    text = f"{user}\n{assistant}"
    score = 0.0
    if _IMPORTANCE_RE.search(text):
        score += 0.45
    length = len(text)
    if length > 400:
        score += 0.15
    elif length > 150:
        score += 0.08
    return max(0.0, min(1.0, score))

def score_question(user: str) -> float:
    if not user:
        return 0.0
    return max(0.0, min(1.0, len(_QUESTION_RE.findall(user)) * 0.35))

def score_entity(user: str, assistant: str) -> float:
    text = f"{user} {assistant}"
    return max(0.0, min(1.0, len(_NAME_LIKE_RE.findall(text)) * 0.12 + len(_NUMBER_RE.findall(text)) * 0.08))

def score_recency(created_turn: int, current_turn: int, half_life: float = GRASP_RECENCY_HALF_LIFE) -> float:
    age = max(0, current_turn - created_turn)
    if half_life <= 0:
        return 1.0 if age == 0 else 0.0
    return 0.5 ** (age / half_life)

def score_primacy(created_turn: int, span: float = GRASP_PRIMACY_SPAN) -> float:
    if created_turn <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (created_turn - 1) / span))

def score_relevance_to_anchor(token_set: set[str], anchor_tokens: set[str] | None) -> float:
    if not anchor_tokens or not token_set:
        return 0.0
    return max(0.0, min(1.0, len(token_set & anchor_tokens) / max(3, len(token_set) ** 0.5)))

def score_novelty(token_set: set[str], others: list[set[str]]) -> float:
    if not token_set or not others:
        return 1.0
    max_j = 0.0
    for o in others:
        if not o:
            continue
        union = len(token_set | o)
        if union:
            max_j = max(max_j, len(token_set & o) / union)
    return max(0.0, min(1.0, 1.0 - max_j))

def score_recall_freq(recall_count: int, cap: int = GRASP_RECALL_FREQ_CAP) -> float:
    return max(0.0, min(1.0, recall_count / max(1, cap)))

def compute_score(turn: "GraspTurn", current_turn: int) -> float:
    emo = (turn.emotion + 1.0) * 0.5
    rec = score_recency(turn.created_turn, current_turn)
    rf = score_recall_freq(turn.recall_count)
    pri = score_primacy(turn.created_turn)
    weights = {
        "emotion": GRASP_W_EMOTION, "importance": GRASP_W_IMPORTANCE, "recency": GRASP_W_RECENCY,
        "relevance": GRASP_W_RELEVANCE, "novelty": GRASP_W_NOVELTY, "question": GRASP_W_QUESTION,
        "entity": GRASP_W_ENTITY, "recall_freq": GRASP_W_RECALL_FREQ, "primacy": GRASP_W_PRIMACY,
    }
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    return (
        weights["emotion"] * emo + weights["importance"] * turn.importance + weights["recency"] * rec
        + weights["relevance"] * turn.relevance + weights["novelty"] * turn.novelty
        + weights["question"] * turn.question + weights["entity"] * turn.entity
        + weights["recall_freq"] * rf + weights["primacy"] * pri
    ) / total_w

class GraspBuffer:
    def __init__(self, *, miller_min: int = GRASP_MILLER_MIN, miller_max: int = GRASP_MILLER_MAX,
                 miller_center: int = GRASP_MILLER_CENTER, token_budget: int = GRASP_TOKEN_BUDGET,
                 static_anchor_tokens: set[str] | None = None,
                 on_evict: Callable[[GraspTurn], None] | None = None,
                 journal: DailyJournal | None = None, journal_enabled: bool | None = None) -> None:
        self.miller_min = max(1, miller_min)
        self.miller_max = max(self.miller_min, miller_max)
        self.miller_center = max(self.miller_min, min(self.miller_max, miller_center))
        self.token_budget = max(0, token_budget)
        self.static_anchor_tokens = static_anchor_tokens or set()
        self.on_evict = on_evict
        if journal is not None:
            self.journal = journal
        elif journal_enabled is False:
            self.journal = DailyJournal(enabled=False)
        else:
            self.journal = DailyJournal(enabled=journal_enabled)
        self._slots: list[GraspTurn] = []
        self._turn_counter = 0

    @property
    def size(self) -> int:
        return len(self._slots)

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._turn_counter = 0

    def set_static_anchor(self, tokens: Iterable[str]) -> None:
        self.static_anchor_tokens = {t.lower() for t in tokens if t}

    def fill(self, user: str, assistant: str, *, token_budget: int | None = None,
             user_ts: datetime | float | None = None, assistant_ts: datetime | float | None = None) -> list[GraspTurn]:
        if not GRASP_ENABLED:
            return []
        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user and not assistant:
            return []
        if isinstance(user_ts, datetime):
            u_unix = ensure_aware(user_ts).timestamp()
        elif isinstance(user_ts, (int, float)):
            u_unix = float(user_ts)
        else:
            u_unix = time.time()
        if isinstance(assistant_ts, datetime):
            a_unix = ensure_aware(assistant_ts).timestamp()
        elif isinstance(assistant_ts, (int, float)):
            a_unix = float(assistant_ts)
        else:
            a_unix = time.time()
        self._turn_counter += 1
        text = f"User: {user}\nAssistant: {assistant}"
        tokens = estimate_tokens(text)
        token_set = _content_tokens(text)
        turn = GraspTurn(
            user=user, assistant=assistant, tokens=tokens,
            emotion=score_emotion(user, assistant), importance=score_importance(user, assistant),
            question=score_question(user), entity=score_entity(user, assistant),
            relevance=score_relevance_to_anchor(token_set, self.static_anchor_tokens),
            created_turn=self._turn_counter, user_ts=u_unix, assistant_ts=a_unix, _token_set=token_set,
        )
        turn.novelty = score_novelty(token_set, [s._token_set for s in self._slots])
        turn.score = compute_score(turn, self._turn_counter)
        self._slots.append(turn)
        self.journal.append_turn(turn, event="fill")
        self._rescore()
        budget = self.token_budget if token_budget is None else max(0, token_budget)
        return self._enforce_capacity(budget)

    def touch(self, n: int | None = None) -> None:
        targets = self._slots if n is None else self._slots[: max(0, n)]
        for t in targets:
            t.recall_count += 1

    def get_context_block(self, *, max_tokens: int | None = None, touch: bool = True) -> str:
        if not self._slots:
            return ""
        lines: list[str] = []
        used = 0
        limit = max_tokens if max_tokens is not None else 0
        included: list[GraspTurn] = []
        for t in self._slots:
            if limit and used + t.tokens > limit:
                break
            lines.append(t.text())
            used += t.tokens
            included.append(t)
        if touch:
            for t in included:
                t.recall_count += 1
        if not lines:
            return ""
        return "<grasp>\nCurrent conversational focus (most salient first):\n\n" + "\n\n".join(lines) + "\n</grasp>"

    def snapshot(self) -> list[GraspTurn]:
        return list(self._slots)

    def flush_resident_to_journal(self, *, event: str = "day_close") -> int:
        n = 0
        for t in self._slots:
            self.journal.append_turn(t, event=event)
            n += 1
        return n

    def studio_state(self) -> dict:
        return {
            "turn_counter": self._turn_counter, "size": self.size, "total_tokens": self.total_tokens,
            "miller": {"min": self.miller_min, "center": self.miller_center, "max": self.miller_max},
            "token_budget": self.token_budget, "anchor_size": len(self.static_anchor_tokens),
            "journal_dir": str(self.journal.journal_dir), "journal_enabled": self.journal.enabled,
            "slots": [{
                "user": t.user[:120], "assistant": t.assistant[:120], "tokens": t.tokens,
                "created_turn": t.created_turn, "user_ts": t.user_ts, "day": day_key_from_user_ts(t.user_ts),
                "recall_count": t.recall_count,
                "factors": {**t.factor_breakdown(), "recency": score_recency(t.created_turn, self._turn_counter),
                            "primacy": score_primacy(t.created_turn)},
                "score": t.score,
            } for t in self._slots],
        }

    def _rescore(self) -> None:
        for i, t in enumerate(self._slots):
            t.novelty = score_novelty(t._token_set, [s._token_set for j, s in enumerate(self._slots) if j != i])
            t.score = compute_score(t, self._turn_counter)
        self._slots.sort(key=lambda t: t.score, reverse=True)

    def _enforce_capacity(self, token_budget: int) -> list[GraspTurn]:
        evicted: list[GraspTurn] = []
        while len(self._slots) > self.miller_max:
            victim = self._slots.pop()
            evicted.append(victim)
            self._fire_evict(victim)
        while len(self._slots) > self.miller_center and self._should_trim_to_center():
            victim = self._slots.pop()
            evicted.append(victim)
            self._fire_evict(victim)
        if token_budget > 0:
            while self._slots and self.total_tokens > token_budget:
                victim = self._slots.pop()
                evicted.append(victim)
                self._fire_evict(victim)
        return evicted

    def _should_trim_to_center(self) -> bool:
        return bool(self._slots) and self._slots[-1].score < 0.35

    def _fire_evict(self, turn: GraspTurn) -> None:
        try:
            self.journal.append_turn(turn, event="evict")
        except Exception:
            pass
        if self.on_evict:
            try:
                self.on_evict(turn)
            except Exception:
                pass

def build_grasp(*, static_anchor_tokens: set[str] | None = None,
                on_evict: Callable[[GraspTurn], None] | None = None,
                journal_dir: str | Path | None = None,
                journal_enabled: bool | None = None) -> GraspBuffer:
    journal = None
    if journal_dir is not None or journal_enabled is not None:
        journal = DailyJournal(journal_dir=journal_dir, enabled=journal_enabled)
    return GraspBuffer(
        miller_min=GRASP_MILLER_MIN, miller_max=GRASP_MILLER_MAX, miller_center=GRASP_MILLER_CENTER,
        token_budget=GRASP_TOKEN_BUDGET, static_anchor_tokens=static_anchor_tokens,
        on_evict=on_evict, journal=journal, journal_enabled=journal_enabled,
    )

def load_journal_day(day: str | None = None, journal_dir: str | Path | None = None) -> list[dict]:
    """Helper for nightly reflect/dream: load a day's faithful journal."""
    return DailyJournal(journal_dir=journal_dir).read_day(day)


# ═════════════════════════════════════════════════════════════════════════════
# Live hub — per-identity working-memory registry + studio snapshot publishing
#
# Aiko's think process owns the in-memory GraspBuffer per user identity. After
# each turn it publishes an atomic JSON snapshot so Grasp Studio (separate
# process) can poll live state without sharing an address space.
#
# Snapshot path (override with GRASP_LIVE_STATE_PATH):
#   ~/.local/share/aiko/grasp/live_state.<identity>.json
#
# Identity is resolved lazily PER CALL (argument → current_user_id() →
# "default"), so nothing here needs a user at import or install time — the
# buffer for a real user simply starts filling on their first turn.
# ═════════════════════════════════════════════════════════════════════════════

GRASP_LIVE_ENABLED = env_flag("GRASP_LIVE_ENABLED", "1")
GRASP_LIVE_STATE_PATH = env_str(
    "GRASP_LIVE_STATE_PATH",
    str(Path.home() / ".local" / "share" / "aiko" / "grasp" / "live_state.json"),
)

_lock = threading.RLock()
_buffers: dict[str, GraspBuffer] = {}
_evictions: dict[str, list[dict[str, Any]]] = {}
_MAX_EVICT = 40
_last_publish: dict[str, float] = {}
_last_publish_error: dict[str, str | None] = {}

_DEFAULT_IDENTITY = "default"


def _resolve_identity(identity: str | None) -> str:
    """Resolve identity from argument or current_user_id() context or fallback."""
    if identity:
        return identity
    try:
        from system.userspace import current_user_id
        return current_user_id() or _DEFAULT_IDENTITY
    except Exception:
        return _DEFAULT_IDENTITY


def _live_state_path(identity: str) -> Path:
    """Return per-identity snapshot path."""
    base = Path(GRASP_LIVE_STATE_PATH)
    return base.with_name(f"{base.stem}.{identity}{base.suffix}")


def get_live_buffer(identity: str | None = None) -> GraspBuffer:
    """Lazy per-identity buffer used by the Aiko process."""
    ident = _resolve_identity(identity)
    with _lock:
        if ident not in _buffers:
            _buffers[ident] = build_grasp(on_evict=lambda turn, _id=ident: _on_evict(_id, turn))
            # LRU-cap past 8 identities; evicted buffers are simply dropped
            # (their turns already published via _on_evict on overflow).
            while len(_buffers) > 8:
                oldest = next(iter(_buffers))
                if oldest == ident:
                    break
                _buffers.pop(oldest, None)
        return _buffers[ident]


def set_static_anchor_tokens(tokens: set[str] | list[str] | None, identity: str | None = None) -> None:
    if not tokens:
        return
    buf = get_live_buffer(identity=identity)
    buf.set_static_anchor(tokens)


# Small stopword set for anchor building — content words only want to anchor
# relevance (grasp._TOKEN_RE already enforces [a-z0-9_]{3+}; CJK never matches,
# so Japanese persona keywords are a known v1 limitation).
_GRASP_ANCHOR_STOP = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "man", "new", "now", "old", "see", "two", "way", "who", "its",
    "did", "that", "this", "with", "from", "they", "have", "will", "your",
    "what", "when", "where", "which", "their", "there", "these", "those",
    "about", "would", "could", "should", "please", "always", "never",
    "aiko", "chan", "user", "assistant", "http", "https", "www", "com",
})


def build_anchor_tokens(*texts: str, limit: int = 64) -> list[str]:
    """Frequency-ranked content tokens from the given texts, capped at
    ``limit`` (~30-80 discriminates well; more flattens relevance scoring).

    Sources should be identity-stable text — persona/SOUL content, pinned
    memories — so ``score_relevance_to_anchor`` keeps turns that touch who
    Aiko is and what persistently matters resident longer in the buffer.
    """
    counts: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok in _GRASP_ANCHOR_STOP:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[: max(0, limit)]]


def record_turn(
    user: str,
    assistant: str,
    *,
    user_ts: float | None = None,
    assistant_ts: float | None = None,
    identity: str | None = None,
) -> list[GraspTurn]:
    """Fill live buffer after a completed conversation turn. No-op if disabled."""
    if not GRASP_LIVE_ENABLED:
        return []
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user and not assistant:
        return []
    ident = _resolve_identity(identity)
    if _BRAIN_TRACE_OK and _brain_trace and _brain_trace.TRACE_ENABLED:
        _brain_trace.record_step(
            "grasp.record_turn",
            layer="write",
            inputs={"identity": ident, "user_chars": len(user),
                    "assistant_chars": len(assistant)},
            factors=[f"working-memory live buffer (capacity={GRASP_LIVE_CAPACITY})"],
        )
    buf = get_live_buffer(identity=ident)
    with _lock:
        evicted = buf.fill(
            user,
            assistant,
            user_ts=user_ts if user_ts is not None else time.time(),
            assistant_ts=assistant_ts if assistant_ts is not None else time.time(),
        )
        try:
            buf.get_context_block(touch=True)
        except Exception:
            pass
        _publish_unlocked(identity=ident)
        if _BRAIN_TRACE_OK and _brain_trace and _brain_trace.TRACE_ENABLED:
            _brain_trace.record_step(
                "grasp.record_turn",
                layer="write",
                outputs={"buffer_filled": True, "evicted_count": len(evicted)},
                factors=[f"eviction pushes oldest turn(s) out of working memory"],
            )
        return list(evicted)


def clear_live(identity: str | None = None) -> None:
    ident = _resolve_identity(identity)
    with _lock:
        if ident in _buffers:
            _buffers[ident].clear()
        _evictions.pop(ident, None)
        _publish_unlocked(identity=ident)


def live_studio_state(identity: str | None = None) -> dict[str, Any]:
    ident = _resolve_identity(identity)
    buf = get_live_buffer(identity=ident)
    with _lock:
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions.get(ident, []))
        state["updated_at"] = time.time()
        return state


def _publish_unlocked(identity: str) -> None:
    """Publish per-identity snapshot. Caller must resolve identity and hold _lock."""
    if not GRASP_LIVE_ENABLED:
        return
    try:
        path = _live_state_path(identity)
        buf = get_live_buffer(identity=identity)
        state = _enrich_valence(buf.studio_state())
        state["mode"] = "live"
        state["evictions"] = list(_evictions.get(identity, []))
        state["updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Drop stale temp from a prior crashed publish, then create 0600
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, json.dumps(state, ensure_ascii=False).encode("utf-8"))
        finally:
            os.close(fd)
        # Ensure temp file has 0600 before replace
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        # Ensure final file has 0600 after replace
        os.chmod(path, 0o600)
        _last_publish[identity] = time.time()
        _last_publish_error[identity] = None
    except Exception as e:
        _last_publish_error[identity] = f"{type(e).__name__}: {e}"
        log.warning(
            "grasp live snapshot publish failed identity=%s path=%s err=%s",
            identity,
            path,
            e,
            exc_info=True,
        )


def read_live_snapshot(identity: str | None = None) -> dict[str, Any] | None:
    ident = _resolve_identity(identity)
    path = _live_state_path(ident)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        data.setdefault("mode", "live")
        return data
    except Exception:
        return None


def snapshot_age_seconds(identity: str | None = None) -> float | None:
    ident = _resolve_identity(identity)
    path = _live_state_path(ident)
    if not path.is_file():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def publish_health(identity: str | None = None) -> dict[str, Any]:
    """Snapshot publication status for Studio /api/health."""
    ident = _resolve_identity(identity)
    return {
        "last_publish_at": _last_publish.get(ident) or None,
        "last_publish_error": _last_publish_error.get(ident),
        "path": str(_live_state_path(ident)),
    }


def get_context_block(*, max_tokens: int | None = 1200, touch: bool = True, identity: str | None = None) -> str:
    """Return the scored WM block for injection into the LLM system prompt.

    Empty string when disabled or buffer is empty. touch=True bumps recall_count
    on included slots so frequently-used focus items stick longer.
    """
    if not GRASP_LIVE_ENABLED:
        return ""
    try:
        ident = _resolve_identity(identity)
        buf = get_live_buffer(identity=ident)
        with _lock:
            block = buf.get_context_block(max_tokens=max_tokens, touch=touch)
            if block:
                _publish_unlocked(identity=ident)
            return block or ""
    except Exception:
        return ""


def _valence_tag(emotion: float) -> str:
    if emotion >= 0.25:
        return "positive"
    if emotion <= -0.25:
        return "negative"
    return "neutral"


def _enrich_valence(state: dict[str, Any]) -> dict[str, Any]:
    """Attach valence_tag to studio slots from emotion factor or raw emotion."""
    slots = state.get("slots") or []
    for s in slots:
        if s.get("valence_tag"):
            continue
        emo = s.get("emotion")
        if emo is None:
            fe = (s.get("factors") or {}).get("emotion")
            if fe is not None:
                emo = float(fe) * 2.0 - 1.0
            else:
                emo = 0.0
        s["emotion"] = round(float(emo), 4)
        s["valence_score"] = s["emotion"]
        s["valence_tag"] = _valence_tag(float(emo))
    return state


def _on_evict(identity: str, turn: GraspTurn) -> None:
    """Eviction callback bound to a specific identity."""
    with _lock:
        emo = float(turn.emotion)
        vtag = _valence_tag(emo)
        bucket = _evictions.setdefault(identity, [])
        bucket.insert(
            0,
            {
                "user": (turn.user or "")[:160],
                "assistant": (turn.assistant or "")[:160],
                "score": round(float(turn.score), 4),
                "emotion": round(emo, 4),
                "valence_tag": vtag,
                "recall_count": int(turn.recall_count),
                "tokens": int(turn.tokens),
                "created_turn": int(turn.created_turn),
            },
        )
        del bucket[_MAX_EVICT:]
