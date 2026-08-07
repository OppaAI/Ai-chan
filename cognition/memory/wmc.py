"""cognition/memory/wmc.py - Working Memory Cortex (WMC) framework for Aiko.

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
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

# -- config helpers -----------------------------------------------------------

def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


# -- public knobs -------------------------------------------------------------

WMC_ENABLED = _env_flag("WMC_ENABLED", "1")

WMC_MILLER_MIN = max(1, _env_int("WMC_MILLER_MIN", 5))
WMC_MILLER_MAX = max(WMC_MILLER_MIN, _env_int("WMC_MILLER_MAX", 9))
WMC_MILLER_CENTER = max(WMC_MILLER_MIN, min(WMC_MILLER_MAX, _env_int("WMC_MILLER_CENTER", 7)))
WMC_TOKEN_BUDGET = max(0, _env_int("WMC_TOKEN_BUDGET", 0))

WMC_W_EMOTION = _env_float("WMC_W_EMOTION", 0.14)
WMC_W_IMPORTANCE = _env_float("WMC_W_IMPORTANCE", 0.17)
WMC_W_RECENCY = _env_float("WMC_W_RECENCY", 0.14)
WMC_W_RELEVANCE = _env_float("WMC_W_RELEVANCE", 0.11)
WMC_W_NOVELTY = _env_float("WMC_W_NOVELTY", 0.11)
WMC_W_QUESTION = _env_float("WMC_W_QUESTION", 0.09)
WMC_W_ENTITY = _env_float("WMC_W_ENTITY", 0.08)
WMC_W_RECALL_FREQ = _env_float("WMC_W_RECALL_FREQ", 0.09)
WMC_W_PRIMACY = _env_float("WMC_W_PRIMACY", 0.07)

WMC_RECENCY_HALF_LIFE = max(1.0, _env_float("WMC_RECENCY_HALF_LIFE", 4.0))
WMC_RECALL_FREQ_CAP = max(1, _env_int("WMC_RECALL_FREQ_CAP", 6))
WMC_PRIMACY_SPAN = max(1.0, _env_float("WMC_PRIMACY_SPAN", 6.0))

WMC_JOURNAL_ENABLED = _env_flag("WMC_JOURNAL_ENABLED", "1")
WMC_JOURNAL_DIR = _env_str(
    "WMC_JOURNAL_DIR",
    str(Path.home() / ".local" / "share" / "aiko" / "journal"),
)

_CHARS_PER_TOKEN = 4.0

_POS_WORDS = re.compile(
    r"\b(love|great|awesome|amazing|happy|glad|thanks|thank you|yay|nice|good|wonderful|excellent|perfect)\b",
    re.I,
)
_NEG_WORDS = re.compile(
    r"\b(hate|awful|terrible|sad|angry|mad|upset|annoyed|frustrated|bad|worst|sucks|furious)\b",
    re.I,
)
_IMPORTANCE_RE = re.compile(
    r"\b(remember|pin|important|always|never|prefer|favorite|favourite|my name|i am|i'm|birthday|"
    r"allergy|allergic|don't forget|do not forget|please note|key point|from now on)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(what|why|how|when|where|who|which|can you|could you|would you|do you|is there)\b",
    re.I,
)
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
    """Local calendar date of the *user* turn — authoritative day boundary."""
    if isinstance(user_ts, (int, float)):
        dt = datetime.fromtimestamp(user_ts, tz=_local_tz())
    elif isinstance(user_ts, datetime):
        dt = ensure_aware(user_ts)
    else:
        dt = local_now()
    return dt.strftime("%Y-%m-%d")


def journal_path_for_day(day: str | None = None, journal_dir: str | Path | None = None) -> Path:
    d = day or day_key_from_user_ts(None)
    root = Path(journal_dir or WMC_JOURNAL_DIR)
    return root / f"{d}.jsonl"


@dataclass
class WMTurn:
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
            "recall_freq": min(1.0, self.recall_count / max(1, WMC_RECALL_FREQ_CAP)),
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
                "recall_freq": round(
                    min(1.0, self.recall_count / max(1, WMC_RECALL_FREQ_CAP)), 4
                ),
            },
            "assistant_ts": self.assistant_ts,
        }


class DailyJournal:
    """Append-only daily JSONL. Day key = local date of *user* turn."""

    def __init__(self, journal_dir: str | Path | None = None, enabled: bool | None = None):
        self.journal_dir = Path(journal_dir or WMC_JOURNAL_DIR)
        self.enabled = WMC_JOURNAL_ENABLED if enabled is None else bool(enabled)
        self._lock = threading.Lock()

    def path_for(self, user_ts: datetime | float | None = None) -> Path:
        return journal_path_for_day(day_key_from_user_ts(user_ts), self.journal_dir)

    def append_turn(self, turn: WMTurn, *, event: str = "fill") -> None:
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
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
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
    hits = len(_QUESTION_RE.findall(user))
    return max(0.0, min(1.0, hits * 0.35))


def score_entity(user: str, assistant: str) -> float:
    text = f"{user} {assistant}"
    names = len(_NAME_LIKE_RE.findall(text))
    nums = len(_NUMBER_RE.findall(text))
    return max(0.0, min(1.0, names * 0.12 + nums * 0.08))


def score_recency(created_turn: int, current_turn: int, half_life: float = WMC_RECENCY_HALF_LIFE) -> float:
    age = max(0, current_turn - created_turn)
    if half_life <= 0:
        return 1.0 if age == 0 else 0.0
    return 0.5 ** (age / half_life)


def score_primacy(created_turn: int, span: float = WMC_PRIMACY_SPAN) -> float:
    if created_turn <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (created_turn - 1) / span))


def score_relevance_to_anchor(token_set: set[str], anchor_tokens: set[str] | None) -> float:
    if not anchor_tokens or not token_set:
        return 0.0
    overlap = len(token_set & anchor_tokens)
    return max(0.0, min(1.0, overlap / max(3, len(token_set) ** 0.5)))


def score_novelty(token_set: set[str], others: list[set[str]]) -> float:
    if not token_set or not others:
        return 1.0
    max_j = 0.0
    for o in others:
        if not o:
            continue
        inter = len(token_set & o)
        union = len(token_set | o)
        if union:
            max_j = max(max_j, inter / union)
    return max(0.0, min(1.0, 1.0 - max_j))


def score_recall_freq(recall_count: int, cap: int = WMC_RECALL_FREQ_CAP) -> float:
    return max(0.0, min(1.0, recall_count / max(1, cap)))


def compute_score(turn: "WMTurn", current_turn: int) -> float:
    emo = (turn.emotion + 1.0) * 0.5
    rec = score_recency(turn.created_turn, current_turn)
    rf = score_recall_freq(turn.recall_count)
    pri = score_primacy(turn.created_turn)
    weights = {
        "emotion": WMC_W_EMOTION,
        "importance": WMC_W_IMPORTANCE,
        "recency": WMC_W_RECENCY,
        "relevance": WMC_W_RELEVANCE,
        "novelty": WMC_W_NOVELTY,
        "question": WMC_W_QUESTION,
        "entity": WMC_W_ENTITY,
        "recall_freq": WMC_W_RECALL_FREQ,
        "primacy": WMC_W_PRIMACY,
    }
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    return (
        weights["emotion"] * emo
        + weights["importance"] * turn.importance
        + weights["recency"] * rec
        + weights["relevance"] * turn.relevance
        + weights["novelty"] * turn.novelty
        + weights["question"] * turn.question
        + weights["entity"] * turn.entity
        + weights["recall_freq"] * rf
        + weights["primacy"] * pri
    ) / total_w


class WorkingMemoryCortex:
    def __init__(
        self,
        *,
        miller_min: int = WMC_MILLER_MIN,
        miller_max: int = WMC_MILLER_MAX,
        miller_center: int = WMC_MILLER_CENTER,
        token_budget: int = WMC_TOKEN_BUDGET,
        static_anchor_tokens: set[str] | None = None,
        on_evict: Callable[[WMTurn], None] | None = None,
        journal: DailyJournal | None = None,
        journal_enabled: bool | None = None,
    ) -> None:
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
        self._slots: list[WMTurn] = []
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

    def fill(
        self,
        user: str,
        assistant: str,
        *,
        token_budget: int | None = None,
        user_ts: datetime | float | None = None,
        assistant_ts: datetime | float | None = None,
    ) -> list[WMTurn]:
        if not WMC_ENABLED:
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

        turn = WMTurn(
            user=user,
            assistant=assistant,
            tokens=tokens,
            emotion=score_emotion(user, assistant),
            importance=score_importance(user, assistant),
            question=score_question(user),
            entity=score_entity(user, assistant),
            relevance=score_relevance_to_anchor(token_set, self.static_anchor_tokens),
            created_turn=self._turn_counter,
            user_ts=u_unix,
            assistant_ts=a_unix,
            _token_set=token_set,
        )
        others = [s._token_set for s in self._slots]
        turn.novelty = score_novelty(token_set, others)
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
        included: list[WMTurn] = []
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
        return (
            "<working_memory>\n"
            "Current conversational focus (most salient first):\n\n"
            + "\n\n".join(lines)
            + "\n</working_memory>"
        )

    def snapshot(self) -> list[WMTurn]:
        return list(self._slots)

    def flush_resident_to_journal(self, *, event: str = "day_close") -> int:
        n = 0
        for t in self._slots:
            self.journal.append_turn(t, event=event)
            n += 1
        return n

    def studio_state(self) -> dict:
        return {
            "turn_counter": self._turn_counter,
            "size": self.size,
            "total_tokens": self.total_tokens,
            "miller": {
                "min": self.miller_min,
                "center": self.miller_center,
                "max": self.miller_max,
            },
            "token_budget": self.token_budget,
            "anchor_size": len(self.static_anchor_tokens),
            "journal_dir": str(self.journal.journal_dir),
            "journal_enabled": self.journal.enabled,
            "slots": [
                {
                    "user": t.user[:120],
                    "assistant": t.assistant[:120],
                    "tokens": t.tokens,
                    "created_turn": t.created_turn,
                    "user_ts": t.user_ts,
                    "day": day_key_from_user_ts(t.user_ts),
                    "recall_count": t.recall_count,
                    "factors": {
                        **t.factor_breakdown(),
                        "recency": score_recency(t.created_turn, self._turn_counter),
                        "primacy": score_primacy(t.created_turn),
                    },
                    "score": t.score,
                }
                for t in self._slots
            ],
        }

    def _rescore(self) -> None:
        for i, t in enumerate(self._slots):
            others = [s._token_set for j, s in enumerate(self._slots) if j != i]
            t.novelty = score_novelty(t._token_set, others)
            t.score = compute_score(t, self._turn_counter)
        self._slots.sort(key=lambda t: t.score, reverse=True)

    def _enforce_capacity(self, token_budget: int) -> list[WMTurn]:
        evicted: list[WMTurn] = []
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
        if not self._slots:
            return False
        return self._slots[-1].score < 0.35

    def _fire_evict(self, turn: WMTurn) -> None:
        try:
            self.journal.append_turn(turn, event="evict")
        except Exception:
            pass
        if self.on_evict:
            try:
                self.on_evict(turn)
            except Exception:
                pass


def build_wmc(
    *,
    static_anchor_tokens: set[str] | None = None,
    on_evict: Callable[[WMTurn], None] | None = None,
    journal_dir: str | Path | None = None,
    journal_enabled: bool | None = None,
) -> WorkingMemoryCortex:
    journal = None
    if journal_dir is not None or journal_enabled is not None:
        journal = DailyJournal(journal_dir=journal_dir, enabled=journal_enabled)
    return WorkingMemoryCortex(
        miller_min=WMC_MILLER_MIN,
        miller_max=WMC_MILLER_MAX,
        miller_center=WMC_MILLER_CENTER,
        token_budget=WMC_TOKEN_BUDGET,
        static_anchor_tokens=static_anchor_tokens,
        on_evict=on_evict,
        journal=journal,
        journal_enabled=journal_enabled,
    )


def load_journal_day(day: str | None = None, journal_dir: str | Path | None = None) -> list[dict]:
    """Helper for nightly reflect/dream: load a day's faithful journal."""
    return DailyJournal(journal_dir=journal_dir).read_day(day)
