"""Attention and bounded cognitive state for edge deployments.

This bounded per-user layer bridges working and long-term memory: it keeps
salient recent turns, open questions/tasks, and a coarse affect cue. It uses no LLM, embeddings, database, or worker thread on the hot path.

This is deliberately one bounded module rather than a collection of tiny
"emotion", "goals", and "working memory" stores. It is the fast internal
state layer between the current turn and long-term memory.

It monitors only compact conversational signals: recent turns, open loops,
goals, affect/energy/uncertainty, contradictions, tool outcomes, response
reviews, preferences, and self-consistency cues. Long-term storage remains in
cognition.memory; this module only snapshots and gates current behavior.
"""
from __future__ import annotations

import importlib
import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from system.config import env_bool, env_flag, env_int
try:
    from cognition.subliminal import SubliminalLayer
except Exception:
    SubliminalLayer = None  # type: ignore
try:
    _brain_trace = importlib.import_module("system.brain_trace")
except Exception:
    _brain_trace = None

def _bt_record(*args, **kwargs):
    """Emit trace record if brain_trace is available. Engram recording infrastructure."""
    if _brain_trace is not None:
        try:
            _brain_trace.record_step(*args, **kwargs)
        except Exception:
            pass

EDGE_COGNITION_ENABLED = env_flag("EDGE_COGNITION_ENABLED", "1")
EDGE_COGNITION_PERSIST = env_bool("EDGE_COGNITION_PERSIST", "1")
EDGE_COGNITION_MAX_TURNS = max(1, env_int("EDGE_COGNITION_MAX_TURNS", 7))
EDGE_COGNITION_MAX_CHARS = max(240, env_int("EDGE_COGNITION_MAX_CHARS", 1200))
EDGE_COGNITION_MAX_OPEN_LOOPS = max(1, env_int("EDGE_COGNITION_MAX_OPEN_LOOPS", 3))
EDGE_COGNITION_MAX_GOALS = max(1, env_int("EDGE_COGNITION_MAX_GOALS", 5))

# Debounced persist: buffer state changes and flush at run-end or after idle period
_persist_timer: threading.Timer | None = None
_persist_pending: set[str] = set()  # identities that need persisting
_persist_lock = threading.Lock()


def _flush_persist_pending() -> None:
    """Flush all pending identity persists."""
    global _persist_timer
    with _persist_lock:
        _persist_timer = None
        pending = list(_persist_pending)
        _persist_pending.clear()
    for ident in pending:
        state = for_identity(ident)
        state._do_persist()
EDGE_COGNITION_MAX_IDENTITIES = max(1, env_int("EDGE_COGNITION_MAX_IDENTITIES", 16))
EDGE_COGNITION_MAX_LESSON_EVIDENCE = 64

_CRITICAL_TASK_RE = re.compile(
    r"\b("
    r"urgent|emergency|asap|right now|immediately|"
    r"safety|danger|hurt|injured|crisis|"
    r"deadline today|due today|production (?:is )?down|outage|"
    r"approve run-|cancel (?:the )?(?:job|schedule|reminder)"
    r")\b",
    re.I,
)
_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:latest|today|tonight|tomorrow|yesterday|now|current|recent|live|breaking|right now|asap|deadline|due)\b",
    re.I,
)
# Modes where soft opt-out (defer/clarify/degrade) is allowed.
_SOFT_MODES = frozenset({"agentic", "route"})

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_QUESTION_RE = re.compile(r"\?|\b(what|why|how|when|where|who|which|remember|can you|could you|need to|todo)\b", re.I)
_COMMITMENT_RE = re.compile(r"\b(i will|i'll|we will|we'll|need to|remember to|don't forget|next step|todo)\b", re.I)
_TASK_RE = re.compile(r"\b(?:can you|could you|please|do|check|find|make|fix|write|create|build|draft|show me)\b", re.I)
_IDENTITY_QUERY_RE = re.compile(r"\b(?:do you know me|who am i|what is my name|what\x27s my name|remember me)\b", re.I)
_GOAL_RE = re.compile(r"\b(?:i want to|i need to|we need to|let\x27s|lets|goal is to|trying to|working on)\s+(.{3,180})", re.I)
_DONE_RE = re.compile(r"\b(done|finished|completed|fixed|solved|never mind|forget it)\b", re.I)
_UNCERTAIN_RE = re.compile(r"\b(i don\x27t know|not sure|unclear|maybe|might|probably|could be|i think)\b", re.I)
_ENERGY_LOW_RE = re.compile(r"\b(tired|exhausted|sleepy|drained|burned out|can\x27t focus|cannot focus)\b", re.I)
_OUTCOME_FAIL_RE = re.compile(r"\b(wrong|incorrect|didn.t work|didn.t help|failed|try again|not what i meant|that.s not right|too verbose|too long|be concise|shorter)\b", re.I)
_OUTCOME_OK_RE = re.compile(r"\b(worked|works|fixed|solved|perfect|exactly|that helped|thank you|thanks)\b", re.I)
_NEGATION_RE = re.compile(r"\b(?:not|no longer|never|dont|don.t|cannot|can.t|isn.t|aren.t|wasn.t|weren.t|changed my mind|instead)\b", re.I)
_ENERGY_HIGH_RE = re.compile(r"\b(excited|energized|motivated|let\x27s go|can\x27t wait)\b", re.I)
# Self-agency cues in Aiko's own replies (not user text). Used only to
# evidence-gate self-regarding preferences — never as hard identity facts.
_SELF_REFUSE_RE = re.compile(
    r"\b("
    r"no\b|not doing that|not going to|won't|will not|pass on that|"
    r"ask properly|make me|earn it|try again|not unless|"
    r"i(?:'|')m not|i refuse|i decline|i(?:'|')d rather not"
    r")\b",
    re.I,
)
_SELF_BARGAIN_RE = re.compile(
    r"\b("
    r"if you|only if|in exchange|first you|then i(?:'|')ll|"
    r"compliment|sweets?|offering|ask nicely|say please"
    r")\b",
    re.I,
)
_SELF_INITIATE_RE = re.compile(
    r"\b("
    r"by the way|while (?:i(?:'|')m|we(?:'|')re) (?:at it|here)|"
    r"i(?:'|')ll (?:also |go ahead and )?(?:check|look|remind|schedule|note)|"
    r"let me (?:also )?(?:check|look|note|schedule)"
    r")\b",
    re.I,
)
_SELF_STANCE_RE = re.compile(
    r"\b("
    r"i (?:prefer|choose|decide|want to|don(?:'|')t want)|"
    r"my (?:preference|choice|call)|"
    r"i(?:'|')m (?:staying|choosing|not a leash)"
    r")\b",
    re.I,
)
_STOP = {"the", "and", "that", "this", "with", "you", "are", "for", "have", "from", "about", "can", "could", "would", "should", "please", "today", "tell", "me", "do", "does", "did", "want", "need", "help"}


@dataclass(slots=True)
class _Event:
    """Dormant turn record. Engram holding user + assistant text and token set."""
    user: str
    assistant: str
    tokens: frozenset[str]


@dataclass(slots=True)
class _Goal:
    """Active or completed objective. Dormant state track."""
    text: str
    progress: str = "active"


def _tokens(text: str) -> frozenset[str]:
    """Extract meaningful tokens, filtering stopwords. Technical tokenization."""
    return frozenset(w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP)


def _gate_tokens(text: str) -> set[str]:
    """Extract gate tokens without stopword filtering for soft attempt checks."""
    return set(_WORD_RE.findall((text or "").lower()))


def is_critical_task(user_input: str) -> bool:
    """Return whether user input names an urgent or safety-critical request. LightGBM-gated, regex fallback."""
    return _gated_match("critical", user_input or "", _CRITICAL_TASK_RE)


def is_time_sensitive(user_input: str) -> bool:
    """Return whether user input requires current or date-sensitive handling. LightGBM-gated, regex fallback."""
    return _gated_match("time_sensitive", user_input or "", _TIME_SENSITIVE_RE)


def capability_from_outcomes(outcomes: list[dict], domain: str = "") -> dict:
    """Synthesize recent tool outcomes into a coarse domain confidence."""
    domain = (domain or "").strip().lower()
    rows = list(outcomes or [])
    if domain:
        rows = [
            o for o in rows
            if domain in str(o.get("tool") or "").lower()
            or domain in str(o.get("detail") or "").lower()
        ]
    n = len(rows)
    if n == 0:
        return {
            "domain": domain or "any",
            "samples": 0,
            "success_rate": None,
            "confidence": "unknown",
            "avoid": False,
        }
    successes = sum(1 for o in rows if o.get("ok"))
    rate = successes / n
    avoid = n >= 3 and rate <= 0.34
    confidence = "high" if n >= 4 else "moderate" if n >= 2 else "low"
    return {
        "domain": domain or "any",
        "samples": n,
        "success_rate": round(rate, 3),
        "confidence": confidence,
        "avoid": avoid,
    }


def soft_user_prompt(user_input: str, action: str, reason: str = "") -> str:
    """Single structured prompt for soft gate outcomes (no multi-step tasks)."""
    text = (user_input or "").strip()
    action = (action or "").strip().lower()
    if action == "defer":
        return (
            f"{text}\n\n"
            "[Style only — never quote this. One short in-character line: "
            "you are low-energy and this is not urgent; offer to continue later.]"
        )
    if action == "clarify":
        return (
            f"{text}\n\n"
            "[Style only — never quote this. Ask exactly one concrete clarifying "
            "question about what they need. Do not start a multi-step task.]"
        )
    return text


def should_attempt(
    *,
    user_input: str,
    mode: str = "agentic",
    energy: float = 0.5,
    uncertainty: float = 0.0,
    tool_outcomes: list[dict] | None = None,
    contradictions: list[str] | None = None,
    response_reviews: list[dict] | None = None,
    load: float = 0.0,
    trend_data: dict | None = None,
    intent_confidence: float = 0.0,
    intent_success_prob: float = 0.5,
    enabled: bool = True,
) -> tuple[bool, str, str]:
    """Return (ok, reason, action) with action in proceed|degrade_chat|defer|clarify."""
    if not enabled:
        return True, "attempt gate disabled", "proceed"
    text = (user_input or "").strip()
    if not text:
        return True, "empty input", "proceed"
    if is_critical_task(text):
        return True, "critical request", "proceed"
    if is_time_sensitive(text):
        return True, "time-sensitive request should proceed with verification discipline", "proceed"

    mode_norm = (mode or "").strip().lower()
    soft = mode_norm in _SOFT_MODES
    energy_v = max(0.0, min(1.0, float(energy or 0.0)))
    load_v = max(0.0, min(1.0, float(load or 0.0)))

    trend = trend_data or {}
    risk_signals = sum([
        trend.get("energy_trend") == "falling",
        trend.get("capability_trend") == "degrading",
        trend.get("uncertainty_trend") == "rising",
    ])
    if soft and risk_signals >= 2 and energy_v < 0.45:
        return False, "multiple degrading trends detected; prefer lighter path", "degrade_chat"

    intent_conf = max(0.0, min(1.0, float(intent_confidence or 0.0)))
    intent_prob = max(0.0, min(1.0, float(intent_success_prob or 0.5)))
    if soft and intent_conf > 0.65 and intent_prob < 0.35:
        return False, f"intent classifier: low success probability ({intent_prob:.1%})", "clarify"

    if mode_norm == "agentic":
        text_tokens = _gate_tokens(text)
        scoped_outcomes = [
            o for o in (tool_outcomes or [])
            if text_tokens & _gate_tokens(f"{o.get('tool') or ''} {o.get('detail') or ''}")
        ]
        cap = capability_from_outcomes(scoped_outcomes, "")
    else:
        cap = {"domain": "any", "samples": 0, "success_rate": None, "confidence": "unknown", "avoid": False}

    text_tokens = _gate_tokens(text)
    related_contradictions = [c for c in (contradictions or []) if len(text_tokens & _gate_tokens(c)) >= 1]
    latest_review = (response_reviews or [{}])[0] if response_reviews else {}
    latest_flags = [str(f) for f in latest_review.get("flags", [])] if isinstance(latest_review, dict) else []
    incomplete_flag = any("may not answer" in flag or "completeness" in flag for flag in latest_flags)
    self_consistency_flag = any("self" in flag and ("conflict" in flag or "consistency" in flag) for flag in latest_flags)

    if soft and load_v >= 0.75 and energy_v < 0.45:
        return False, "running hot and energy low; discretionary work can wait", "defer"
    if soft and energy_v < 0.28:
        return False, "energy low; discretionary work can wait", "defer"
    if soft and uncertainty > 0.62:
        if len(text) < 80 and ("?" in text or len(_gate_tokens(text)) <= 6):
            return False, "uncertainty elevated; need a clearer ask", "clarify"
        return False, "uncertainty elevated; prefer lighter chat path", "degrade_chat"
    if soft and related_contradictions:
        return False, "current prompt overlaps unresolved contradictions; clarify before acting", "clarify"
    if soft and cap.get("avoid") and (cap.get("samples") or 0) >= 3:
        return False, "recent tool outcomes are weak; prefer lighter path", "degrade_chat"
    if soft and (incomplete_flag or self_consistency_flag) and len(text_tokens) <= 6:
        return False, "last answer review found a likely issue; clarify the follow-up before acting", "clarify"

    return True, "self-assessment clear", "proceed"


def _affect(text: str) -> float:
    """Compute valence signal [-1, 1] from text sentiment cues. Affect engram."""
    lower = (text or "").lower()
    pos = sum(lower.count(w) for w in ("love", "great", "thanks", "happy", "good", "excited"))
    neg = sum(lower.count(w) for w in ("hate", "sad", "angry", "bad", "frustrated", "worried"))
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


class IntentConfidenceClassifier:
    """Lightweight LightGBM classifier for success prediction on Jetson.

    Trains on bounded feature set: no embeddings, no external calls.
    Used to augment gate decisions. Gracefully unavailable until trained.

    Uses LightGBM (lighter on ARM/Jetson ~20 MB vs XGBoost ~40 MB),
    faster inference, same tabular feature story. Falls back to regex
    heuristics when no model file is present.
    """

    # Canonical feature order — must match training script.
    _FEATURE_ORDER = [
        "input_length",
        "token_count",
        "is_question",
        "has_action_verb",
        "has_negation",
        "has_uncertainty",
        "has_time_ref",
        "overlap_with_active_goal",
        "overlap_with_open_loop",
        "current_energy",
        "current_uncertainty",
        "is_running_hot",
        # gated regex detectors (added for LightGBM multi-task, fallback to regex)
        "is_critical",
        "is_time_sensitive",
        "has_question",
        "has_commitment",
        "has_task",
        "is_identity_query",
        "has_goal_phrase",
        "is_done",
        "has_energy_low",
        "has_energy_high",
        "outcome_fail",
        "outcome_ok",
        "has_self_refuse",
        "has_self_bargain",
    ]

    def __init__(self, model_path: str = "models/intent_classifier.lgb"):
        """Load LightGBM model if available. Gracefully inactive when absent."""
        self.model = None
        self.available = False
        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        import os as _os
        if not _os.path.isfile(self.model_path):
            self.available = False
            return
        try:
            lgb = importlib.import_module("lightgbm")
            booster = lgb.Booster(model_file=self.model_path)
            self.model = booster
            self.available = True
            return
        except Exception:
            pass
        self.available = False

    def predict(self, features: dict) -> tuple[float, float]:
        """Estimate (confidence, success_probability). Returns (0.0, 0.5) if unavailable."""
        if not self.available or not features or self.model is None:
            return 0.0, 0.5
        try:
            feature_list = [float(features.get(k, 0)) for k in self._FEATURE_ORDER]
            # LightGBM Booster.predict expects 2-D array
            pred = float(self.model.predict([feature_list])[0])  # type: ignore
            # clamp to [0,1] (regression or binary)
            pred = max(0.0, min(1.0, pred))
            confidence = max(abs(pred - 0.5) * 2, 0.4)
            return confidence, pred
        except Exception:
            return 0.0, 0.5


# Singleton classifier instance. Loads on module init; reuses same model.
_intent_classifier = IntentConfidenceClassifier()

# ── Gated regex → LightGBM detectors (train-and-keep-regex-as-fallback) ──
# Each detector optionally loads a per-task LightGBM model from
# models/detectors/<name>.lgb. When absent, the compiled regex is used.
_DETECTOR_MODELS: dict[str, Any] = {}
_DETECTOR_REGEX: dict[str, re.Pattern] = {}


def _load_detector_models() -> None:
    import os as _os
    root = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "models", "detectors")
    if not _os.path.isdir(root):
        return
    try:
        lgb = importlib.import_module("lightgbm")
    except Exception:
        return
    for name, pat in list(_DETECTOR_REGEX.items()):
        path = _os.path.join(root, f"{name}.lgb")
        if not _os.path.isfile(path):
            continue
        try:
            _DETECTOR_MODELS[name] = lgb.Booster(model_file=path)
        except Exception:
            pass


def _gated_match(name: str, text: str, fallback_re: re.Pattern) -> bool:
    """Try LightGBM detector for *name*; fallback to regex."""
    mdl = _DETECTOR_MODELS.get(name)
    if mdl is not None:
        try:
            # minimal token-bag features for gate detectors
            toks = _tokens(text)
            feats = [
                float(len(text or "")),
                float(len(toks)),
                float("?" in (text or "")),
                float(bool(fallback_re.search(text or ""))),
            ]
            pred = float(mdl.predict([feats])[0])  # type: ignore
            return pred >= 0.5
        except Exception:
            pass
    return bool(fallback_re.search(text or ""))


# Register regexes for gated loading
_DETECTOR_REGEX.update({
    "critical": _CRITICAL_TASK_RE,
    "time_sensitive": _TIME_SENSITIVE_RE,
    "question": _QUESTION_RE,
    "commitment": _COMMITMENT_RE,
    "task": _TASK_RE,
    "identity_query": _IDENTITY_QUERY_RE,
    "done": _DONE_RE,
    "uncertain": _UNCERTAIN_RE,
    "energy_low": _ENERGY_LOW_RE,
    "energy_high": _ENERGY_HIGH_RE,
    "outcome_fail": _OUTCOME_FAIL_RE,
    "outcome_ok": _OUTCOME_OK_RE,
    "negation": _NEGATION_RE,
    "goal": _GOAL_RE,
    "self_refuse": _SELF_REFUSE_RE,
    "self_bargain": _SELF_BARGAIN_RE,
    "self_initiate": _SELF_INITIATE_RE,
    "self_stance": _SELF_STANCE_RE,
})
try:
    _load_detector_models()
except Exception:
    pass


class EdgeCognitiveState:
    """Bounded per-identity state; all operations are lock-protected.
    
    dormant neural pathway: working memory holds salient turns, open loops,
    goals, and affect cues. Active cognition routes decisions through energy,
    uncertainty, tool outcomes, and trend analysis. Engrams persist across
    sessions via SQLite backup.
    """

    def __init__(self, identity: str = "") -> None:
        self._identity = identity or ""
        self._persistent_loaded = False
        self._events: deque[_Event] = deque(maxlen=EDGE_COGNITION_MAX_TURNS)
        self._open_loops: deque[str] = deque(maxlen=EDGE_COGNITION_MAX_OPEN_LOOPS)
        self._affect = 0.0
        self._goals: deque[_Goal] = deque(maxlen=EDGE_COGNITION_MAX_GOALS)
        self._energy = 0.5
        self._uncertainty = 0.0
        self._attention = ""
        self._lessons: deque[str] = deque(maxlen=5)
        self._durable_lessons: deque[str] = deque(maxlen=5)
        self._lesson_counts: dict[str, int] = {}
        self._tool_outcomes: deque[dict] = deque(maxlen=6)
        self._perceptions: deque[dict] = deque(maxlen=4)
        self._activity: str = ""
        self._response_reviews: deque[dict] = deque(maxlen=4)
        self._contradictions: deque[str] = deque(maxlen=4)
        self._pending_memory_conflicts: deque[dict] = deque(maxlen=3)
        self._preferences: dict[str, str] = {}
        self._identity_questions: deque[str] = deque(maxlen=3)
        self._preference_counts: dict[str, int] = {}
        # Self-regarding state (about Aiko, not the user). Evidence-gated:
        # raw decisions are recorded every time; durable self-preferences
        # only lock in after repeated support (same rule as user prefs).
        self._self_decisions: deque[dict] = deque(maxlen=8)
        self._self_preference_counts: dict[str, int] = {}
        self._self_preferences: dict[str, str] = {}
        self._self_notes: deque[str] = deque(maxlen=5)
        # Recent turn wall-clock seconds (bounded) for coarse "running hot".
        self._turn_latencies: deque[float] = deque(maxlen=5)
        self._lock = threading.RLock()
        # Subliminal layer owns intuitions / pre-attentive scan (Jetson-clean split).
        # Created after _lock so it can share the same RLock.
        self._subliminal = SubliminalLayer(lock=self._lock) if SubliminalLayer is not None else None
        self._intuitions: deque[str] = self._subliminal._intuitions if self._subliminal is not None else deque(maxlen=4)  # alias for compat
        self._last_tick = time.monotonic()

    def record(self, user: str, assistant: str) -> None:
        """Ingest one turn. Dormant engram registration. Active state updates."""
        if not EDGE_COGNITION_ENABLED:
            return
        user = " ".join((user or "").split())[:360]
        assistant = " ".join((assistant or "").split())[:360]
        if not user and not assistant:
            return
        with self._lock:
            self._apply_explicit_preferences(user)
            self._learn_preferences(user)
            self._detect_contradictions(user)
            if _IDENTITY_QUERY_RE.search(user):
                self._identity_questions.appendleft(user[:220])
            if self._events and _OUTCOME_FAIL_RE.search(user):
                self._add_lesson("Avoid repeating the previous approach: ", self._events[-1].assistant, user)
            elif self._events and _OUTCOME_OK_RE.search(user):
                self._add_lesson("The previous approach appeared useful: ", self._events[-1].assistant, user)
            self._events.append(_Event(user, assistant, _tokens(user + " " + assistant)))
            self._learn_self_from_turn(user, assistant)
            combined = user + " " + assistant
            self._affect = max(-1.0, min(1.0, self._affect * 0.7 + _affect(combined) * 0.3))
            self._energy = max(0.0, min(1.0, self._energy * 0.8 + self._energy_signal(combined) * 0.2))
            self._uncertainty = max(0.0, min(1.0, self._uncertainty * 0.7 + self._uncertainty_signal(combined) * 0.3))
            self._attention = self._attention_for(user)
            if _QUESTION_RE.search(user) or _COMMITMENT_RE.search(user):
                loop = user[:220]
                if loop and loop not in self._open_loops:
                    self._open_loops.appendleft(loop)
            self._capture_goal(user)
            self._refresh_subconscious(user)
            # Continuous training hook — log gated features for LightGBM while talking
            try:
                feats = self._feature_vector_for_intent(user)
                # weak label from immediate outcome signals (refined on next turn via outcome rewrites)
                if self._tool_outcomes:
                    last = self._tool_outcomes[-1] if self._tool_outcomes else {}
                    ok = last.get("ok", None) if isinstance(last, dict) else None
                    label = 1.0 if ok is True else 0.0 if ok is False else 0.5
                else:
                    label = 0.5
                from cognition.attention_train import log_example
                log_example(feats, label)
            except Exception:
                pass
            if _DONE_RE.search(user):
                self._close_matching_goal(user)
                self._close_matching_loop(user)
            elif _DONE_RE.search(assistant) and not _OUTCOME_FAIL_RE.search(assistant):
                self._close_matching_goal(assistant)
                self._close_matching_loop(assistant)
            # Broadcast VRM expression from subliminal affect (throttled 1.5 s)
            try:
                if self._subliminal is not None:
                    self._subliminal.broadcast_vrm()
            except Exception:
                pass

    def clear(self) -> None:
        """Wipe all dormant state. Engram erasure (rare; identity reset only)."""
        with self._lock:
            for collection in (
                self._events, self._open_loops, self._goals, self._lessons,
                self._tool_outcomes, self._perceptions, self._response_reviews,
                self._contradictions, self._durable_lessons, self._lesson_counts,
                self._preferences, self._preference_counts,
                self._identity_questions, self._intuitions,
                self._pending_memory_conflicts, self._self_decisions,
                self._self_preference_counts, self._self_preferences,
                self._self_notes, self._turn_latencies,
            ):
                collection.clear()
            self._activity = ""
            self._affect = 0.0
            self._energy = 0.5
            self._uncertainty = 0.0
            self._attention = ""
            self._persistent_loaded = True
        if not self._identity or self._identity == "default":
            return
        conn = None
        try:
            from cognition.memory.vecstore import connect_sqlite_db
            conn = connect_sqlite_db("memory/memory.db", user_id=self._identity)
            conn.execute("DELETE FROM cognitive_state WHERE user_id = ?", (self._identity,))
            conn.commit()
        except Exception:
            return
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _energy_signal(text: str) -> float:
        """Infer energy readiness from text. Dormant neural cue."""
        if _ENERGY_LOW_RE.search(text):
            return 0.2
        if _ENERGY_HIGH_RE.search(text):
            return 0.8
        return 0.5

    @staticmethod
    def _uncertainty_signal(text: str) -> float:
        """Infer epistemic uncertainty from text. Dormant confidence engram."""
        return 0.8 if _UNCERTAIN_RE.search(text) else 0.0

    @staticmethod
    def _attention_for(user: str) -> str:
        """Build a readable focus phrase while preserving word order. Attention engram."""
        words = _tokens(user)
        if not words:
            return ""
        ordered = []
        for word in _WORD_RE.findall((user or "").lower()):
            if word in _STOP or word not in words or word in ordered or word == "todays":
                continue
            ordered.append(word)
            if len(ordered) >= 8:
                break
        return " ".join(ordered)

    def _apply_explicit_preferences(self, feedback: str) -> None:
        """Hard-set preferences when user says 'always' or 'from now on'. Explicit cognition."""
        lower = (feedback or "").casefold()
        explicit = "from now on" in lower or "always" in lower or "starting now" in lower
        if "forget" in lower and "preference" in lower:
            if "response" in lower or "style" in lower:
                self._preferences.pop("response_length", None)
                self._preferences.pop("explanation_depth", None)
                self._preferences.pop("tone", None)
            if "action" in lower or "tool" in lower:
                self._preferences.pop("action_confirmation", None)
            return
        if not explicit:
            return
        if any(term in lower for term in ("concise", "short", "brief")):
            self._preferences["response_length"] = "concise"
        elif any(term in lower for term in ("more detail", "detailed", "step by step")):
            self._preferences["explanation_depth"] = "detailed"
        if any(term in lower for term in ("ask before", "ask me first", "with my permission")):
            self._preferences["action_confirmation"] = "ask_before_acting"
        if "casual" in lower or "friendly" in lower:
            self._preferences["tone"] = "casual"
        elif "formal" in lower or "professional" in lower:
            self._preferences["tone"] = "formal"

    def _learn_preferences(self, feedback: str) -> None:
        """Track implicit preference signals. Evidence-gated at count ≥ 2."""
        lower = (feedback or "").casefold()
        candidates = []
        if any(term in lower for term in ("concise", "short", "brief", "too long", "verbose")):
            candidates.append(("response_length", "concise"))
        if any(term in lower for term in ("more detail", "explain", "step by step", "walk me through")):
            candidates.append(("explanation_depth", "detailed"))
        if any(term in lower for term in ("ask before", "ask me first", "with my permission")):
            candidates.append(("action_confirmation", "ask_before_acting"))
        if any(term in lower for term in ("casual", "friendly", "less formal")):
            candidates.append(("tone", "casual"))
        if any(term in lower for term in ("formal", "professional")):
            candidates.append(("tone", "formal"))
        for key, value in candidates:
            signature = key + "=" + value
            count = self._preference_counts.get(signature, 0) + 1
            self._preference_counts[signature] = min(count, 3)
            if count >= 2:
                self._preferences[key] = value

    def _learn_self_from_turn(self, user: str, assistant: str) -> None:
        """Detect Aiko's own agency signals and evidence-gate self-preferences.

        Only the assistant reply is scored for refuse/bargain/initiate/stance.
        Patterns become durable only after repeated evidence (count >= 2),
        matching user preference learning — no one-shot identity rewrite.
        """
        reply = (assistant or "").strip()
        if not reply:
            return
        kinds: list[str] = []
        if _SELF_REFUSE_RE.search(reply):
            kinds.append("refuse")
        if _SELF_BARGAIN_RE.search(reply):
            kinds.append("bargain")
        if _SELF_INITIATE_RE.search(reply):
            kinds.append("initiate")
        if _SELF_STANCE_RE.search(reply):
            kinds.append("stance")
        if not kinds:
            return
        summary = " ".join(reply.split())[:180]
        decision = {
            "kinds": kinds,
            "summary": summary,
            "user_cue": " ".join((user or "").split())[:120],
        }
        self._self_decisions.appendleft(decision)
        # Map observed agency into candidate self-preferences.
        candidates: list[tuple[str, str]] = []
        if "refuse" in kinds:
            candidates.append(("pushback", "will_refuse_lazy_or_rude_requests"))
        if "bargain" in kinds:
            candidates.append(("pushback", "bargains_before_discretionary_work"))
        if "initiate" in kinds:
            candidates.append(("initiative", "initiates_when_context_allows"))
        if "stance" in kinds:
            candidates.append(("agency", "expresses_own_preferences"))
        for key, value in candidates:
            signature = key + "=" + value
            count = self._self_preference_counts.get(signature, 0) + 1
            self._self_preference_counts[signature] = min(count, 3)
            if count >= 2:
                self._self_preferences[key] = value
                note = f"Observed pattern ({key}): {value.replace('_', ' ')}."
                if note not in self._self_notes:
                    self._self_notes.appendleft(note)

    def record_self_decision(
        self,
        kind: str,
        summary: str = "",
        *,
        promote: bool = False,
    ) -> None:
        """Explicit self-decision hook (agentic loop, reflection, tools).

        kind: refuse | bargain | initiate | stance | other
        promote: if True, treat as stronger evidence (counts as +2 toward durability).
        """
        if not EDGE_COGNITION_ENABLED:
            return
        kind = (kind or "other").strip().lower()[:24]
        summary = " ".join((summary or "").split())[:180]
        with self._lock:
            self._self_decisions.appendleft({
                "kinds": [kind],
                "summary": summary or kind,
                "user_cue": "",
                "explicit": True,
            })
            key_map = {
                "refuse": ("pushback", "will_refuse_lazy_or_rude_requests"),
                "bargain": ("pushback", "bargains_before_discretionary_work"),
                "initiate": ("initiative", "initiates_when_context_allows"),
                "stance": ("agency", "expresses_own_preferences"),
            }
            pair = key_map.get(kind)
            if not pair:
                return
            key, value = pair
            signature = key + "=" + value
            bump = 2 if promote else 1
            count = min(3, self._self_preference_counts.get(signature, 0) + bump)
            self._self_preference_counts[signature] = count
            if count >= 2:
                self._self_preferences[key] = value
                note = f"Observed pattern ({key}): {value.replace('_', ' ')}."
                if note not in self._self_notes:
                    self._self_notes.appendleft(note)

    def _add_lesson(self, prefix: str, source: str, feedback: str = "") -> None:
        """Record outcome lesson and promote to durable if evidence ≥ 2. Engram consolidation."""
        text = " ".join((source or "").split())[:180]
        if not text:
            return
        lesson = prefix + text
        self._lessons.appendleft(lesson)
        lower_feedback = (feedback or "").casefold()
        semantic = ""
        if any(word in lower_feedback for word in ("concise", "short", "brief", "too long", "verbose")):
            semantic = "Prefer concise responses."
        elif any(word in lower_feedback for word in ("clarify", "misunderstood", "not what i meant", "wrong")):
            semantic = "Ask a clarifying question when intent is ambiguous."
        elif any(word in lower_feedback for word in ("explain", "why", "more detail")):
            semantic = "Include a brief explanation, not only the conclusion."
        elif any(word in lower_feedback for word in ("step by step", "steps", "walk me through")):
            semantic = "Present complex tasks as clear sequential steps."
        signature = " ".join(sorted(_tokens(semantic or text)))
        if not signature:
            return
        count = self._lesson_counts.get(signature, 0) + 1
        if signature not in self._lesson_counts and len(self._lesson_counts) >= EDGE_COGNITION_MAX_LESSON_EVIDENCE:
            self._lesson_counts.pop(next(iter(self._lesson_counts)))
        self._lesson_counts[signature] = min(count, 3)
        if count >= 2:
            direction = "Prefer this approach: " if prefix.startswith("The previous approach appeared useful:") else "Avoid repeating: "
            durable = "Durable interaction rule: " + (semantic or (direction + text))
            if durable not in self._durable_lessons:
                self._durable_lessons.appendleft(durable)

    def _capture_goal(self, user: str) -> None:
        """Extract explicit or inferred goal from user input. Goal engram."""
        match = _GOAL_RE.search(user)
        if match:
            text = " ".join(match.group(1).split()).rstrip(".?!")
        elif _TASK_RE.search(user) and not _IDENTITY_QUERY_RE.search(user):
            text = self._attention_for(user)
        else:
            return
        if not text or len(_tokens(text)) < 2:
            return
        self._goals = deque((g for g in self._goals if g.text.casefold() != text.casefold()), maxlen=EDGE_COGNITION_MAX_GOALS)
        self._goals.appendleft(_Goal(text=text))

    def _close_matching_goal(self, user: str) -> None:
        """Mark goal as completed when user says 'done'. Dormant state update."""
        words = _tokens(user)
        if not words:
            return
        for goal in self._goals:
            goal_words = _tokens(goal.text)
            overlap = len(words & goal_words)
            if overlap >= 1 and (goal_words and overlap / len(goal_words) >= 0.2):
                goal.progress = "completed"

    def _close_matching_loop(self, user: str) -> None:
        """Remove an explicitly completed question or task from open loops. Dormant discharge."""
        words = _tokens(user)
        if not words:
            return
        remaining = deque(maxlen=EDGE_COGNITION_MAX_OPEN_LOOPS)
        for loop in self._open_loops:
            loop_words = _tokens(loop)
            overlap = len(words & loop_words)
            if not (overlap >= 1 and (loop_words and overlap / len(loop_words) >= 0.2)):
                remaining.append(loop)
        self._open_loops = remaining

    def _detect_contradictions(self, user: str) -> None:
        """Record a bounded conflict when a new statement reverses a recent one. Conflict engram."""
        current_tokens = _tokens(user)
        if len(current_tokens) < 2:
            return
        current_negated = bool(_NEGATION_RE.search(user))
        for event in reversed(self._events):
            overlap = current_tokens & _tokens(event.user)
            if len(overlap) < 2 or current_negated == bool(_NEGATION_RE.search(event.user)):
                continue
            summary = f"Current statement conflicts with an earlier statement: current={user[:150]} | earlier={event.user[:150]}"
            if summary not in self._contradictions:
                self._contradictions.appendleft(summary)
            break

    def _recent_trend(self, signal_name: str) -> str:
        """Return 'rising', 'falling', or 'steady' for energy/affect/uncertainty.
        
        Uses only the snapshot's recent bounded history.
        No embeddings, no external calls, <1ms.
        Option 1: Trend Analysis dormant pathway.
        """
        if signal_name == "energy":
            samples = [self._energy]
            # Pull from recent events' encoded energy signals
            if len(self._events) >= 2:
                for event in list(self._events)[-2:]:
                    combined = event.user + " " + event.assistant
                    samples.append(self._energy_signal(combined))
        elif signal_name == "affect":
            samples = [self._affect]
            if len(self._events) >= 2:
                for event in list(self._events)[-2:]:
                    combined = event.user + " " + event.assistant
                    samples.append(_affect(combined))
        elif signal_name == "uncertainty":
            samples = [self._uncertainty]
            if len(self._events) >= 2:
                for event in list(self._events)[-2:]:
                    combined = event.user + " " + event.assistant
                    samples.append(self._uncertainty_signal(combined))
        else:
            return "unknown"
        
        if len(samples) < 2:
            return "unknown"
        
        # Simple linear: is last higher/lower than first?
        if samples[-1] > samples[0] + 0.15:
            return "rising"
        elif samples[-1] < samples[0] - 0.15:
            return "falling"
        return "steady"

    def capability_trend(self) -> str:
        """Tool success trend: improving, degrading, or stable.
        
        Compares recent outcomes vs older outcomes.
        Option 1: Trend Analysis — capability engram.
        """
        outcomes = list(self._tool_outcomes)
        if len(outcomes) < 2:
            return "unknown"
        
        # Split: older vs recent
        mid = len(outcomes) // 2
        older = outcomes[mid:]  # Reverse deque order → older are at end
        recent = outcomes[:mid]
        
        older_rate = sum(1 for o in older if o.get("ok")) / max(1, len(older))
        recent_rate = sum(1 for o in recent if o.get("ok")) / max(1, len(recent))
        
        delta = recent_rate - older_rate
        if delta > 0.25:
            return "improving"
        elif delta < -0.25:
            return "degrading"
        return "stable"

    def _feature_vector_for_intent(self, user_input: str) -> dict:
        """Extract gated features for LightGBM intent confidence scoring.

        Keeps the original 12 lightweight features and appends the 14
        regex-detector binaries so a single LightGBM model can learn
        their combinations. All detectors use gated LightGBM-or-regex
        helpers when per-detector models are present.
        """
        text = (user_input or "").strip()
        tokens = _tokens(text)

        features = {
            # Input shape
            "input_length": len(text),
            "token_count": len(tokens),
            "is_question": 1.0 if "?" in text else 0.0,

            # Semantics (lightweight, gated)
            "has_action_verb": 1.0 if any(v in text.lower() for v in ("can you", "could you", "do", "make", "write")) else 0.0,
            "has_negation": 1.0 if _gated_match("negation", text, _NEGATION_RE) else 0.0,
            "has_uncertainty": 1.0 if _gated_match("uncertain", text, _UNCERTAIN_RE) else 0.0,
            "has_time_ref": 1.0 if is_time_sensitive(text) else 0.0,

            # Context alignment
            "overlap_with_active_goal": len(tokens & _tokens(" ".join(
                [g.text for g in self._goals if g.progress == "active"]
            ))),
            "overlap_with_open_loop": len(tokens & _tokens(" ".join(list(self._open_loops)))),

            # State readiness
            "current_energy": self._energy,
            "current_uncertainty": self._uncertainty,
            "is_running_hot": 1.0 if self.load_signal() >= 0.75 else 0.0,

            # Gated detector binaries (regex fallback, LightGBM when trained)
            "is_critical": 1.0 if is_critical_task(text) else 0.0,
            "is_time_sensitive": 1.0 if is_time_sensitive(text) else 0.0,
            "has_question": 1.0 if _gated_match("question", text, _QUESTION_RE) else 0.0,
            "has_commitment": 1.0 if _gated_match("commitment", text, _COMMITMENT_RE) else 0.0,
            "has_task": 1.0 if _gated_match("task", text, _TASK_RE) else 0.0,
            "is_identity_query": 1.0 if _gated_match("identity_query", text, _IDENTITY_QUERY_RE) else 0.0,
            "has_goal_phrase": 1.0 if _gated_match("goal", text, _GOAL_RE) else 0.0,
            "is_done": 1.0 if _gated_match("done", text, _DONE_RE) else 0.0,
            "has_energy_low": 1.0 if _gated_match("energy_low", text, _ENERGY_LOW_RE) else 0.0,
            "has_energy_high": 1.0 if _gated_match("energy_high", text, _ENERGY_HIGH_RE) else 0.0,
            "outcome_fail": 1.0 if _gated_match("outcome_fail", text, _OUTCOME_FAIL_RE) else 0.0,
            "outcome_ok": 1.0 if _gated_match("outcome_ok", text, _OUTCOME_OK_RE) else 0.0,
            "has_self_refuse": 1.0 if _gated_match("self_refuse", text, _SELF_REFUSE_RE) else 0.0,
            "has_self_bargain": 1.0 if _gated_match("self_bargain", text, _SELF_BARGAIN_RE) else 0.0,
        }
        return features

    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        """Truncate text with ellipsis. Display hygiene."""
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def snapshot(self) -> dict:
        """Return a compact diagnostic snapshot without exposing mutable state. Engram snapshot."""
        with self._lock:
            mood = "positive" if self._affect > 0.2 else "negative" if self._affect < -0.2 else "neutral"
            return {"mood": mood, "affect": round(self._affect, 3), "energy": round(self._energy, 3), "uncertainty": round(self._uncertainty, 3), "attention": self._attention, "open_loops": list(self._open_loops), "goals": [g.text for g in self._goals if g.progress == "active"], "lessons": list(self._lessons), "tool_outcomes": list(self._tool_outcomes), "perceptions": list(self._perceptions), "activity": self._activity, "response_reviews": list(self._response_reviews), "contradictions": list(self._contradictions), "durable_lessons": list(self._durable_lessons), "lesson_evidence": dict(self._lesson_counts), "preferences": dict(self._preferences), "identity_questions": list(self._identity_questions), "intuitions": list(self._intuitions), "self_preferences": dict(self._self_preferences), "self_decisions": list(self._self_decisions), "self_notes": list(self._self_notes), "self_preference_evidence": dict(self._self_preference_counts)}

    def continuous_tick(self) -> dict:
        """Apply bounded low-cost decay between conversational turns. Passive dormancy."""
        now = time.monotonic()
        with self._lock:
            elapsed = max(0.0, min(3600.0, now - self._last_tick))
            self._last_tick = now
            steps = elapsed / 3600.0
            self._uncertainty = max(0.0, self._uncertainty * max(0.0, 1.0 - 0.18 * steps))
            self._energy += (0.5 - self._energy) * min(1.0, 0.12 * steps)
            return {"elapsed_s": round(elapsed, 3), "energy": round(self._energy, 3), "uncertainty": round(self._uncertainty, 3)}

    def cognitive_health(self) -> dict:
        """Return bounded metrics showing whether cognitive state is usable.

        This is diagnostic state only: it contains counts and labels, not new
        memory content.  A sparse score distinguishes an empty/stale snapshot
        from a functioning but quiet mind.
        """
        with self._lock:
            active_goals = sum(1 for goal in self._goals if goal.progress == "active")
            attention_words = len(_tokens(self._attention))
            components = {
                "working_memory": len(self._events),
                "attention": attention_words,
                "goals": active_goals,
                "open_loops": len(self._open_loops),
                "lessons": len(self._lessons) + len(self._durable_lessons),
                "preferences": len(self._preferences),
                "self_preferences": len(self._self_preferences),
                "contradictions": len(self._contradictions),
            }
            populated = sum(1 for key, value in components.items() if value > 0 and key != "contradictions")
            population = round(populated / 7.0, 3)
            status = "empty" if not self._events else "sparse" if population < 0.34 else "active"
            return {"status": status, "population": population, "components": components, "attention_valid": attention_words >= 2 or not self._events}

    def _refresh_subconscious(self, user: str) -> None:
        """Delegate to SubliminalLayer (L3). Keeps hot path <0.2 ms."""
        if self._subliminal is not None:
            snap = self.snapshot()
            scan = self._subliminal.scan(user, snap, self._events)
            self._subliminal.refresh_intuitions(snap, scan)
            # keep alias in sync
            self._intuitions = self._subliminal._intuitions
            return
        # Fallback when subliminal not loaded
        candidates = []
        snap = self.snapshot()
        if snap.get("contradictions"):
            candidates.append("Possible unresolved belief conflict; verify before asserting.")
        if snap.get("goals") and snap.get("open_loops"):
            candidates.append("An active goal may be connected to an unresolved thread.")
        if len(self._events) >= 3:
            recent = [_tokens(event.user) for event in list(self._events)[-3:]]
            recurring = set(recent[0]).intersection(*recent[1:]) if recent else set()
            recurring -= {"can", "could", "please", "today"}
            if recurring:
                candidates.append("Recurring focus detected: " + " ".join(sorted(recurring)[:4]) + ".")
        if snap.get("durable_lessons"):
            candidates.append("A learned interaction pattern may apply here; check the current request first.")
        with self._lock:
            for intuition in candidates:
                if intuition not in self._intuitions:
                    self._intuitions.appendleft(intuition)

    def self_model_context(self) -> str:
        """Describe grounded self-knowledge without claiming consciousness.

        Static boundary lines stay minimal. Lived self-preferences, recent
        agency decisions, and self-notes are only included when evidence has
        actually accumulated — they are not a second SOUL.md.
        """
        snap = self.snapshot()
        health = self.cognitive_health()
        lines = [
            "Identity: Aiko — local companion with bounded memory, tools, and learned state.",
            "Memory boundary: distinguish recalled evidence from inference; uncertainty is allowed.",
            "Action boundary: explain limitations and ask before consequential external actions.",
            f"Current cognitive state: {health['status']}; uncertainty={snap['uncertainty']}.",
        ]
        if snap.get("goals"):
            lines.append("Current priority: " + snap["goals"][0][:160])
        self_prefs = snap.get("self_preferences") or {}
        if self_prefs:
            rendered = []
            labels = {
                "will_refuse_lazy_or_rude_requests": "pushes back on lazy or rude requests",
                "bargains_before_discretionary_work": "may bargain before discretionary work",
                "initiates_when_context_allows": "initiates when context allows",
                "expresses_own_preferences": "expresses her own preferences",
            }
            for key, value in self_prefs.items():
                rendered.append(labels.get(value, value.replace("_", " ")))
            lines.append("Learned self-patterns: " + "; ".join(rendered[:4]))
        notes = snap.get("self_notes") or []
        if notes:
            lines.append("Self-notes: " + " | ".join(n[:120] for n in notes[:2]))
        decisions = snap.get("self_decisions") or []
        if decisions:
            latest = decisions[0]
            kinds = ",".join(latest.get("kinds") or [])
            summary = (latest.get("summary") or "")[:100]
            if kinds or summary:
                lines.append(f"Recent self-decision ({kinds}): {summary}")
        return "<self_model>\n" + "\n".join("- " + line for line in lines) + "\n</self_model>"

    def capability_for(self, domain: str = "") -> dict:
        """Synthesize recent tool outcomes into a coarse domain confidence. Capability assessment."""
        return capability_from_outcomes(list(self.snapshot().get("tool_outcomes") or []), domain)

    def is_critical_task(self, user_input: str) -> bool:
        """Check if input is a critical/urgent request. Gate classification."""
        return is_critical_task(user_input)

    def record_turn_latency(self, seconds: float) -> None:
        """Record one turn's wall-clock duration for coarse load sensing. Latency engram."""
        if not EDGE_COGNITION_ENABLED:
            return
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            return
        if s < 0:
            return
        with self._lock:
            self._turn_latencies.appendleft(min(s, 120.0))

    def load_signal(self) -> float:
        """Coarse 0..1 load from recent turn latencies (not a full metrics stack).

        ~median of last few turns: <4s → 0, ~12s → ~0.5, >=20s → 1.
        """
        with self._lock:
            samples = list(self._turn_latencies)
        if not samples:
            return 0.0
        ordered = sorted(samples)
        mid = ordered[len(ordered) // 2]
        # Map 4s..20s → 0..1
        return max(0.0, min(1.0, (mid - 4.0) / 16.0))

    def should_attempt(self, user_input: str, *, mode: str = "agentic") -> tuple[bool, str, str]:
        """Decide whether to commit to a heavier execution path.

        Returns (ok, reason, action): proceed | degrade_chat | defer | clarify.
        Critical tasks always proceed. Toggle with EDGE_ATTEMPT_GATE.
        
        Integrates Option 1 (trend analysis) and Option 2 (LightGBM intent).
        Option 2 activates only when model is trained and available.
        """
        if not EDGE_COGNITION_ENABLED:
            return True, "edge cognition disabled", "proceed"
        
        snap = self.snapshot()
        
        # Option 1: Trend data
        trend_data = {
            "energy_trend": self._recent_trend("energy"),
            "affect_trend": self._recent_trend("affect"),
            "uncertainty_trend": self._recent_trend("uncertainty"),
            "capability_trend": self.capability_trend(),
        }
        
        # Option 2: Intent confidence (will return 0.0, 0.5 if untrained)
        features = self._feature_vector_for_intent(user_input)
        intent_conf, will_succeed = _intent_classifier.predict(features)
        
        return should_attempt(
            user_input=user_input,
            mode=mode,
            energy=float(snap.get("energy") or 0.5),
            uncertainty=float(snap.get("uncertainty") or 0.0),
            tool_outcomes=list(snap.get("tool_outcomes") or []),
            contradictions=list(snap.get("contradictions") or []),
            response_reviews=list(snap.get("response_reviews") or []),
            load=self.load_signal(),
            trend_data=trend_data,
            intent_confidence=intent_conf,
            intent_success_prob=will_succeed,
            enabled=env_flag("EDGE_ATTEMPT_GATE", "1"),
        )

    def priming_context(self, query: str = "") -> str:
        """Inject only subconscious signals related to the current query. Delegates to SubliminalLayer."""
        if self._subliminal is not None:
            return self._subliminal.priming_context(query, self.snapshot(), self._identity_questions)
        query_words = _tokens(query)
        snap = self.snapshot()
        related_goals = [item for item in snap.get("goals", []) if not query_words or _tokens(item) & query_words]
        related_loops = [item for item in snap.get("open_loops", []) if not query_words or _tokens(item) & query_words]
        identity_query = bool(_IDENTITY_QUERY_RE.search(query or ""))
        lines = []
        if related_goals:
            lines.append("Relevant active goal: " + related_goals[0][:180])
        if identity_query and snap.get("identity_questions"):
            lines.append("Relevant identity thread: answer from retrieved identity evidence; do not infer missing facts.")
        if related_loops:
            lines.append("Relevant open loop: " + related_loops[0][:180])
        if snap.get("uncertainty", 0.0) > 0.35 and (related_goals or related_loops):
            lines.append("Relevant uncertainty is elevated; verify assumptions.")
        if abs(float(snap.get("affect") or 0.0)) > 0.25 and (related_goals or related_loops):
            lines.append("Recent emotional context may affect interpretation; respond proportionately.")
        if not lines:
            return ""
        return "<subconscious_priming>\n" + "\n".join("- " + line for line in lines) + "\n</subconscious_priming>"

    def subconscious_guidance(self) -> str:
        """Expose subconscious signals as hypotheses for conscious review. Delegates to SubliminalLayer."""
        if self._subliminal is not None:
            return self._subliminal.guidance()
        with self._lock:
            intuitions = list(self._intuitions)
        if not intuitions:
            return "<subconscious_guidance>\nNo tentative intuition currently salient.\n</subconscious_guidance>"
        lines = ["- tentative hypothesis: " + item for item in intuitions[:3]]
        lines.append("Treat these as associations to verify, never as facts.")
        return "<subconscious_guidance>\n" + "\n".join(lines) + "\n</subconscious_guidance>"

    def evaluation_snapshot(self) -> dict:
        """Return bounded behavioral metrics for offline brain evaluation. Evaluation engram."""
        snap = self.snapshot()
        health = self.cognitive_health()
        outcomes = snap.get("tool_outcomes") or []
        successes = sum(1 for item in outcomes if item.get("ok"))
        reviews = snap.get("response_reviews") or []
        high_reviews = sum(1 for item in reviews if item.get("confidence") == "high")
        return {
            "state_status": health["status"],
            "state_population": health["population"],
            "active_goals": len(snap.get("goals") or []),
            "open_loops": len(snap.get("open_loops") or []),
            "durable_lessons": len(snap.get("durable_lessons") or []),
            "lesson_evidence": len(snap.get("lesson_evidence") or {}),
            "tool_success_rate": round(successes / len(outcomes), 3) if outcomes else None,
            "response_review_high_rate": round(high_reviews / len(reviews), 3) if reviews else None,
            "identity_questions": len(snap.get("identity_questions") or []),
            "contradictions": len(snap.get("contradictions") or []),
        }

    def identity_guidance(self) -> str:
        """Give grounded guidance for unresolved user-identity questions. Identity enforcement."""
        with self._lock:
            pending = list(self._identity_questions)
        if not pending:
            return "<identity_guidance>\nNo unresolved identity question.\n</identity_guidance>"
        return "<identity_guidance>\nThe user asked about personal familiarity. Do not invent recognition; use only grounded identity or memory evidence, and ask for clarification if needed.\n</identity_guidance>"

    def record_tool_outcome(self, tool: str, *, ok: bool, detail: str = "", error_type: str = "") -> None:
        """Record a compact tool outcome for disclosure and future learning. Tool outcome engram."""
        item = {"tool": str(tool or "tool")[:48], "ok": bool(ok), "detail": str(detail or "")[:240]}
        if error_type:
            item["error_type"] = str(error_type)[:48]
        with self._lock:
            self._tool_outcomes.appendleft(item)
            if not ok:
                failure_key = str(error_type or "tool_error")
                self._add_lesson("Tool limitation: ", f"{tool} ({failure_key})", "tool failed")

    def lesson_guidance(self) -> str:
        """Render only repeatedly evidenced lessons as behavior guidance. Lesson distillation."""
        snap = self.snapshot()
        durable = snap.get("durable_lessons") or []
        if not durable:
            return "<lesson_guidance>\nNo repeatedly confirmed lessons yet.\n</lesson_guidance>"
        lines = [f"- {lesson}" for lesson in durable[:3]]
        return "<lesson_guidance>\n" + "\n".join(lines) + "\n</lesson_guidance>"

    def consume_lessons(self) -> list[str]:
        """Return current outcome lessons for successful background consolidation. Lesson retrieval."""
        with self._lock:
            lessons = list(self._lessons)
            self._lessons.clear()
            return lessons

    def load_persistent(self) -> None:
        """Load prior bounded state from SQLite. Engram restore."""
        if not EDGE_COGNITION_PERSIST or not self._identity or self._identity == "default":
            return
        if self._persistent_loaded:
            return
        try:
            from cognition.memory.vecstore import connect_sqlite_db
            conn = connect_sqlite_db("memory/memory.db", user_id=self._identity)
            conn.execute("CREATE TABLE IF NOT EXISTS cognitive_state (user_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            row = conn.execute("SELECT state_json FROM cognitive_state WHERE user_id = ?", (self._identity,)).fetchone()
            conn.close()
            if not row:
                return
            data = json.loads(row[0])
            with self._lock:
                self._open_loops = deque(data.get("open_loops", [])[:EDGE_COGNITION_MAX_OPEN_LOOPS], maxlen=EDGE_COGNITION_MAX_OPEN_LOOPS)
                self._goals = deque((_Goal(text=str(text)) for text in data.get("goals", []) if text), maxlen=EDGE_COGNITION_MAX_GOALS)
                self._lessons = deque(data.get("lessons", [])[:5], maxlen=5)
                self._tool_outcomes = deque([item for item in data.get("tool_outcomes", []) if isinstance(item, dict)][:6], maxlen=6)
                self._contradictions = deque(data.get("contradictions", [])[:4], maxlen=4)
                self._durable_lessons = deque(data.get("durable_lessons", [])[:5], maxlen=5)
                lesson_counts = {str(k): min(3, int(v)) for k, v in (data.get("lesson_evidence") or {}).items() if str(k) and str(v).isdigit()}
                self._lesson_counts = dict(list(lesson_counts.items())[-EDGE_COGNITION_MAX_LESSON_EVIDENCE:])
                self._preferences = {str(k): str(v) for k, v in (data.get("preferences") or {}).items()}
                self._identity_questions = deque(data.get("identity_questions", [])[:3], maxlen=3)
                self._intuitions = deque(data.get("intuitions", [])[:4], maxlen=4)
                if self._subliminal is not None:
                    self._subliminal._intuitions = self._intuitions
                self._self_preferences = {str(k): str(v) for k, v in (data.get("self_preferences") or {}).items()}
                self._self_preference_counts = {str(k): min(3, int(v)) for k, v in (data.get("self_preference_evidence") or {}).items() if str(k) and str(v).isdigit()}
                self._self_notes = deque([str(n) for n in (data.get("self_notes") or []) if n][:5], maxlen=5)
                loaded_decisions = [d for d in (data.get("self_decisions") or []) if isinstance(d, dict)][:8]
                self._self_decisions = deque(loaded_decisions, maxlen=8)
                self._activity = str(data.get("activity") or "")
                self._affect = float(data.get("affect") or 0.0)
                self._energy = float(data.get("energy") or 0.5)
                self._uncertainty = float(data.get("uncertainty") or 0.0)
                loaded_attention = str(data.get("attention") or "")
                self._attention = loaded_attention if len(_tokens(loaded_attention)) >= 2 else ""
        except Exception:
            return
        finally:
            self._persistent_loaded = True

    def persist(self) -> None:
        """Queue bounded state for debounced SQLite save. Engram backup.

        Writes are debounced and batched at run-end or after a short idle
        period to avoid per-tool-call disk I/O on the hot path.
        """
        if not EDGE_COGNITION_PERSIST or not self._identity or self._identity == "default":
            return
        with _persist_lock:
            _persist_pending.add(self._identity)
            if _persist_timer is not None:
                _persist_timer.cancel()
            _persist_timer = threading.Timer(2.0, _flush_persist_pending)
            _persist_timer.daemon = True
            _persist_timer.start()

    def _do_persist(self) -> None:
        """Actual persistence write (called by timer flush)."""
        if not EDGE_COGNITION_PERSIST or not self._identity or self._identity == "default":
            return
        try:
            from cognition.memory.vecstore import connect_sqlite_db
            with self._lock:
                data = {"open_loops": list(self._open_loops), "goals": [g.text for g in self._goals if g.progress == "active"], "lessons": list(self._lessons), "tool_outcomes": list(self._tool_outcomes), "contradictions": list(self._contradictions), "durable_lessons": list(self._durable_lessons), "lesson_evidence": dict(self._lesson_counts), "preferences": dict(self._preferences), "activity": self._activity, "affect": self._affect, "energy": self._energy, "uncertainty": self._uncertainty, "attention": self._attention, "self_preferences": dict(self._self_preferences), "self_preference_evidence": dict(self._self_preference_counts), "self_notes": list(self._self_notes), "self_decisions": list(self._self_decisions)[:6]}
            conn = connect_sqlite_db("memory/memory.db", user_id=self._identity)
            conn.execute("CREATE TABLE IF NOT EXISTS cognitive_state (user_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("INSERT INTO cognitive_state(user_id, state_json) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET state_json=excluded.state_json, updated_at=CURRENT_TIMESTAMP", (self._identity, json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
            conn.commit()
            conn.close()
        except Exception:
            return

    def record_perception(self, source: str, duration_s: float | None = None, latency_s: float | None = None, prosody: dict | None = None) -> None:
        """Store bounded aggregate perception cues; never retain raw audio. Perception engram."""
        item = {"source": str(source or "unknown")[:24]}
        if duration_s is not None:
            item["duration_s"] = round(max(0.0, float(duration_s)), 3)
        if latency_s is not None:
            item["latency_s"] = round(max(0.0, float(latency_s)), 3)
        if isinstance(prosody, dict):
            for key in ("rms", "peak", "voiced_fraction", "pause_density", "words_per_second"):
                value = prosody.get(key)
                if isinstance(value, (int, float)):
                    cap = 20.0 if key == "words_per_second" else 1.0
                    item[key] = round(max(0.0, min(cap, float(value))), 3)
        with self._lock:
            self._perceptions.appendleft(item)

    def prioritize_memories(self, query: str, memories: list[dict] | None) -> list[dict]:
        """Rank memories as a bounded reconstruction with confidence cues. Memory prioritization."""
        rows = list(memories or [])
        snap = self.snapshot()
        query_words = _tokens(query)
        context_words = _tokens(" ".join([snap.get("attention", ""), " ".join(snap.get("goals", [])), " ".join(snap.get("open_loops", []))]))
        goal_words = _tokens(" ".join(snap.get("goals", [])))
        current_affect = float(snap.get("affect") or 0.0)
        scored = []
        for index, row in enumerate(rows):
            text = str(row.get("memory") or row.get("text") or row.get("trace") or "")
            words = _tokens(text)
            query_overlap = len(words & query_words)
            context_overlap = len(words & context_words)
            goal_overlap = len(words & goal_words)
            score = query_overlap * 2.0 + context_overlap * 0.7 + goal_overlap * 1.5
            basis = []
            if query_overlap: basis.append("query")
            if context_overlap: basis.append("active_context")
            if goal_overlap: basis.append("goal")
            if row.get("pinned"):
                score += 1.5
                basis.append("pinned")
            if row.get("salience_hit"):
                score += 0.8
                basis.append("salient")
            try:
                accesses = max(0.0, float(row.get("access_count") or 0))
                score += min(1.0, accesses * 0.1)
                if accesses >= 2: basis.append("recalled_before")
            except (TypeError, ValueError):
                pass
            try:
                valence = float(row.get("valence_score") or 0.0)
                if current_affect and valence and current_affect * valence > 0:
                    score += 0.35
                    basis.append("affective_resonance")
            except (TypeError, ValueError):
                pass
            if str(row.get("status") or "").casefold() == "superseded":
                score -= 2.0
                basis.append("superseded")
            confidence = "high" if score >= 4.0 and (query_overlap or context_overlap) else "moderate" if score >= 2.0 else "low"
            reconstructed = dict(row)
            reconstructed["_reconstruction_confidence"] = confidence
            reconstructed["_reconstruction_basis"] = basis or ["weak_match"]
            scored.append((score, -index, reconstructed))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result = [row for _, _, row in scored]
        _bt_record(
            "EdgeCognitiveState.prioritize_memories",
            layer="rerank",
            inputs={"query": query, "memories_count": len(rows),
                    "current_affect": round(current_affect, 3),
                    "active_goals": snap.get("goals", []),
                    "open_loops": snap.get("open_loops", [])},
            outputs={
                "reranked_count": len(result),
                "top_basis": result[0].get("_reconstruction_basis") if result else None,
                "top_confidence": result[0].get("_reconstruction_confidence") if result else None,
                "top_text_preview": (result[0].get("memory") or "")[:160] if result else None,
            },
            factors=[
                "weights: query×2.0, context×0.7, goal×1.5, pinned+1.5, salient+0.8, recall_history≤+1.0, affective_resonance+0.35, superseded-2.0",
                f"confidence tiers: high ≥4.0 (with overlap), moderate ≥2.0, low otherwise",
            ],
        )
        return result

    def memory_conflicts(self, query: str, memories: list[dict] | None = None) -> list[dict]:
        """Find conservative conflicts between a new statement and recalled facts. Conflict detection."""
        query_tokens = _tokens(query)
        if len(query_tokens) < 2:
            return []
        query_negated = bool(_NEGATION_RE.search(query))
        conflicts = []
        for row in (memories or [])[:8]:
            text = str(row.get("memory") or row.get("text") or row.get("trace") or "").strip()
            words = _tokens(text)
            overlap = query_tokens & words
            if len(overlap) < 1 or query_negated == bool(_NEGATION_RE.search(text)):
                continue
            conflicts.append({"memory_id": row.get("id"), "current": query[:220], "remembered": text[:240], "shared_terms": sorted(overlap)[:8], "status": str(row.get("status") or "active")})
        return conflicts[:3]

    def memory_resolution_guidance(self, query: str, memories: list[dict] | None = None) -> str:
        """Describe a safe clarification/update path without mutating memory. Resolution strategy."""
        conflicts = self.memory_conflicts(query, memories)
        if not conflicts:
            return ""
        lower = (query or "").casefold()
        explicit_update = any(term in lower for term in ("actually", "changed", "no longer", "anymore", "from now on", "update that"))
        if explicit_update:
            action = "Treat this as a candidate correction, but require explicit confirmation before superseding the remembered fact."
        else:
            action = "Ask which fact is current before writing or superseding either memory."
        return "<memory_conflict_resolution>\n" + action + "\n" + "\n".join("- remembered: " + c["remembered"] for c in conflicts[:2]) + "\n</memory_conflict_resolution>"

    def confirm_memory_update(self, statement: str) -> list[dict]:
        """Consume pending conflicts only after an explicit confirmation. Confirmation gating."""
        lower = (statement or "").casefold()
        confirmation = any(term in lower for term in ("yes", "correct", "confirmed", "that changed", "that is right", "thats right"))
        denial = any(term in lower for term in ("no", "not correct", "still", "keep the old", "that is wrong"))
        if not confirmation or denial:
            return []
        with self._lock:
            pending = list(self._pending_memory_conflicts)
            self._pending_memory_conflicts.clear()
        return pending

    def reflection_summary(self) -> str:
        """Summarize bounded internal state for continuity and self-correction. State summary."""
        snap = self.snapshot()
        lines = []
        if snap.get("goals"):
            lines.append("Active priority: " + snap["goals"][0])
        if snap.get("open_loops"):
            lines.append("Unresolved thread: " + snap["open_loops"][0])
        if snap.get("contradictions"):
            lines.append("Belief requiring care: recent statements conflict")
        if snap.get("durable_lessons"):
            lines.append("Behavioral lesson: " + snap["durable_lessons"][0])
        if snap.get("self_preferences"):
            lines.append("Self-pattern active: " + ", ".join(f"{k}={v}" for k, v in list(snap["self_preferences"].items())[:2]))
        failed_tools = [item for item in snap.get("tool_outcomes", [])[:3] if not item.get("ok")]
        if failed_tools:
            lines.append("Recent limitation: a tool attempt failed; disclose this if relevant")
        if snap.get("response_reviews") and snap["response_reviews"][0].get("flags"):
            lines.append("Last self-check: " + "; ".join(snap["response_reviews"][0]["flags"][:2]))
        if not lines:
            lines.append("No unresolved cognitive issue is currently salient")
        return "<self_reflection>\n" + "\n".join("- " + line for line in lines[:5]) + "\n</self_reflection>"

    def adaptive_tts_rate(self) -> float:
        """Choose a conservative speech rate from recent voice cues. Speech adaptation."""
        latest = (self.snapshot().get("perceptions") or [{}])[0]
        if isinstance(latest.get("words_per_second"), (int, float)) and latest["words_per_second"] > 4.5:
            return 1.05
        if (isinstance(latest.get("pause_density"), (int, float)) and latest["pause_density"] > 0.55) or self.snapshot().get("uncertainty", 0.0) > 0.35:
            return 0.9
        if self.snapshot().get("affect", 0.0) < -0.25:
            return 0.92
        return 1.0

    def preference_guidance(self) -> str:
        """Turn learned preferences into explicit response behavior rules. Preference distillation."""
        preferences = self.snapshot().get("preferences", {})
        if not preferences:
            return "<preference_guidance>\nNo stable interaction preferences yet.\n</preference_guidance>"
        rules = []
        if preferences.get("response_length") == "concise":
            rules.append("Prefer concise answers unless the user asks for detail.")
        if preferences.get("explanation_depth") == "detailed":
            rules.append("Include useful reasoning or steps when the task is complex.")
        if preferences.get("action_confirmation") == "ask_before_acting":
            rules.append("Ask before consequential external actions; reading and drafting remain allowed.")
        if preferences.get("tone") == "casual":
            rules.append("Use a friendly, less formal tone.")
        elif preferences.get("tone") == "formal":
            rules.append("Use a professional, more formal tone.")
        return "<preference_guidance>\n" + "\n".join("- " + rule for rule in rules) + "\n</preference_guidance>"

    def adaptive_response_guidance(self) -> str:
        """Render bounded behavior guidance from current affect and prosody. Behavior guidance."""
        snap = self.snapshot()
        latest = (snap.get("perceptions") or [{}])[0]
        rules = []
        rms = latest.get("rms")
        pace = latest.get("words_per_second")
        pauses = latest.get("pause_density")
        if snap.get("energy", 0.5) < 0.35 or (isinstance(rms, (int, float)) and rms < 0.2):
            rules.append("use a calm, gentle tone and keep the response manageable")
        if isinstance(pace, (int, float)) and pace > 4.5:
            rules.append("be concise and avoid unnecessary preamble")
        if snap.get("uncertainty", 0.0) > 0.35 or (isinstance(pauses, (int, float)) and pauses > 0.55):
            rules.append("slow down, clarify ambiguity, and avoid overconfident assumptions")
        if snap.get("affect", 0.0) < -0.25:
            rules.append("acknowledge possible frustration before problem-solving")
        elif snap.get("affect", 0.0) > 0.25:
            rules.append("match positive energy without becoming excessive")
        if not rules:
            rules.append("keep the response natural, proportionate, and attentive")
        return "<adaptive_response_guidance>\n" + "\n".join("- " + rule for rule in rules[:4]) + "\n</adaptive_response_guidance>"

    def review_response(self, query: str, response: str) -> dict:
        """Audit a completed draft for bounded metacognitive warning signs. Response review."""
        q = (query or "").casefold()
        text = " ".join((response or "").split())
        snap = self.snapshot()
        flags = []
        if any(word in q for word in ("latest", "today", "now", "current", "recent")) and not any(word in text.casefold() for word in ("as of", "i do not have", "unable to verify", "search")):
            flags.append("current-information claim may need verification")
        if any(word in text.casefold() for word in ("definitely", "certainly", "always", "never")) and snap["uncertainty"] > 0.35:
            flags.append("draft sounds more certain than internal uncertainty")
        if any(not outcome.get("ok") for outcome in snap.get("tool_outcomes", [])[:3]) and not any(word in text.casefold() for word in ("failed", "could not", "unable", "error", "not completed")):
            flags.append("recent tool failure may be undisclosed")
        if "?" in query and len(text) < 24:
            flags.append("draft may not answer the user question")
        if snap.get("contradictions"):
            flags.append("recent user statements conflict; clarification may be needed")
        self_flag = self._self_consistency_flag(text)
        if self_flag:
            flags.append(self_flag)
        review = {"flags": flags, "confidence": "low" if len(flags) >= 2 else "moderate" if flags else "high", "response_chars": len(text)}
        with self._lock:
            self._response_reviews.appendleft(review)
        _bt_record(
            "EdgeCognitiveState.review_response",
            layer="review",
            inputs={"query": query, "response_chars": len(text)},
            outputs=review,
            factors=[
                "checks: time-sensitivity, overconfidence, undisclosed tool failures, too-short answer, contradictions, self-consistency vs last 1-2 replies",
                "confidence tiers: low if ≥2 flags, moderate if any, high otherwise",
            ],
        )
        return review

    def _self_consistency_flag(self, response: str) -> str:
        """Flag possible self-contradiction vs the last 1–2 Aiko replies only. Self-consistency check."""
        draft_tokens = _tokens(response)
        if len(draft_tokens) < 4:
            return ""
        draft_neg = bool(_NEGATION_RE.search(response or ""))
        with self._lock:
            recent = [e.assistant for e in list(self._events)[-2:] if e.assistant]
        for prior in recent:
            prior_tokens = _tokens(prior)
            overlap = draft_tokens & prior_tokens
            if len(overlap) < 3:
                continue
            prior_neg = bool(_NEGATION_RE.search(prior))
            if draft_neg != prior_neg:
                return "draft may conflict with a recent self-statement"
        return ""

    def grounded_context(self, now=None, idle_seconds: float = 0.0, resting: bool = False, scheduled_jobs: list[dict] | None = None, project_signals: list[str] | None = None) -> str:
        """Render bounded real-world signals, including known scheduled work. Situational context."""
        if now is None:
            from datetime import datetime
            now = datetime.now().astimezone()
        hour = int(now.hour)
        period = "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        activity = "active" if idle_seconds < 90 else "idle" if idle_seconds < 3600 else "resting"
        if resting:
            activity = "resting"
        jobs = []
        for job in (scheduled_jobs or []):
            title = str(job.get("title") or job.get("name") or "scheduled task").strip()
            due = str(job.get("next_due") or "").strip()
            if title:
                jobs.append(f"{title} ({due})" if due else title)
        lines = ["<grounded_context>", f"Local time: {now.strftime('%A, %Y-%m-%d %H:%M')} ({period})", f"User activity: {activity}", f"Idle duration: {max(0, int(idle_seconds))} seconds"]
        if jobs:
            lines.append("Upcoming scheduled work: " + " | ".join(jobs[:5]))
        outcomes = self.snapshot().get("tool_outcomes", [])
        perceptions = self.snapshot().get("perceptions", [])
        if perceptions:
            latest = perceptions[0]
            duration = latest.get("duration_s")
            lines.append("Recent perception: " + str(latest.get("source", "unknown")) + (f" utterance={duration}s" if duration is not None else ""))
            cues = []
            if latest.get("words_per_second") is not None: cues.append(f"pace={latest.get('words_per_second')}wps")
            if latest.get("pause_density") is not None: cues.append(f"pauses={latest.get('pause_density')}")
            if latest.get("rms") is not None: cues.append(f"energy={latest.get('rms')}")
            if cues: lines.append("Voice cues: " + ", ".join(cues))
        if outcomes:
            rendered = []
            for outcome in outcomes[:4]:
                status = "ok" if outcome.get("ok") else "failed"
                detail = outcome.get("error_type") or outcome.get("detail") or "no detail"
                rendered.append(f"{outcome.get('tool', 'tool')}={status} ({detail})")
            lines.append("Recent tool outcomes: " + " | ".join(rendered))
        if project_signals:
            lines.append("Relevant project changes: " + " | ".join(project_signals[:5]))
        if self._activity:
            lines.append("Coarse activity: " + self._activity)
        snap = self.snapshot()
        if snap["open_loops"]:
            lines.append("Follow-up candidates: " + " | ".join(snap["open_loops"][:3]))
        if snap.get("response_reviews") and snap["response_reviews"][0].get("flags"):
            lines.append("Last response review: " + " | ".join(snap["response_reviews"][0]["flags"][:3]))
        lines.append("Initiative guidance: " + ("keep quiet unless important" if activity == "active" else "a gentle follow-up may be appropriate"))
        lines.append("</grounded_context>")
        return "\n".join(lines)

    def situation_context(self, query: str = "", memories: list[dict] | None = None, knowledge: str = "") -> str:
        """Build a bounded situation model from already-retrieved context. Situation model."""
        snap = self.snapshot()
        facts = []
        entities = []
        seen = set()
        for row in (memories or [])[:5]:
            text = str(row.get("memory") or row.get("text") or row.get("trace") or "").strip()
            if text:
                facts.append(text[:240])
            raw = row.get("entities") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = []
            for entity in raw if isinstance(raw, list) else []:
                value = str(entity).strip()
                key = value.casefold()
                if value and key not in seen:
                    seen.add(key)
                    entities.append(value)
        if not facts and not snap["goals"] and not snap["open_loops"]:
            _bt_record("EdgeCognitiveState.situation_context", layer="context",
                       inputs={"query": query, "memories_count": len(memories or [])},
                       outputs={"skipped": True, "reason": "no_facts_or_goals_or_loops"})
            return ""
        health = self.cognitive_health()
        lines = ["<situation_model>", "Organized from available evidence; treat it as context, not certainty.", f"Current query: {query[:260]}"]
        lines.append(f"Cognitive state: {health.get('status')}; population={health.get('population')}")
        if health["status"] == "empty":
            lines.append("Memory discipline: do not imply personal recall without evidence; ask for the missing context.")
        if entities:
            lines.append("Relevant entities: " + ", ".join(entities[:12]))
        if facts:
            lines.append("Relevant remembered facts: " + " | ".join(facts[:4]))
        if snap["goals"]:
            lines.append("Active goals: " + " | ".join(snap["goals"][:5]))
        if snap["open_loops"]:
            lines.append("Open loops: " + " | ".join(snap["open_loops"][:4]))
        if snap.get("contradictions"):
            lines.append("Possible contradictions to clarify: " + " | ".join(snap["contradictions"][:2]))
        conflicts = self.memory_conflicts(query, memories)
        if conflicts:
            with self._lock:
                self._pending_memory_conflicts.clear()
                self._pending_memory_conflicts.extend(conflicts[:3])
            lines.append("Long-term memory conflicts requiring clarification: " + " | ".join(c["remembered"] for c in conflicts[:2]))
        resolution = self.memory_resolution_guidance(query, memories)
        if resolution:
            lines.append(resolution)
        if snap["lessons"]:
            lines.append("Lessons from outcomes: " + " | ".join(snap["lessons"][:3]))
        if snap.get("durable_lessons"):
            lines.append("Durable interaction rules: " + " | ".join(snap["durable_lessons"][:3]))
        if snap.get("preferences"):
            lines.append("Stable interaction preferences: " + " | ".join(f"{key}={value}" for key, value in snap["preferences"].items()))
        energy = "low" if snap["energy"] < 0.35 else "high" if snap["energy"] > 0.65 else "steady"
        uncertainty = "elevated" if snap["uncertainty"] > 0.35 else "ordinary"
        lines.append(f"Internal cues: mood={snap.get('mood')}, energy={energy}, uncertainty={uncertainty}")
        lines.append("Evidence confidence: " + ("moderate" if facts else "low"))
        block = "\n".join(lines) + "\n</situation_model>"
        _bt_record(
            "EdgeCognitiveState.situation_context",
            layer="context",
            inputs={"query": query, "memories_count": len(memories or [])},
            outputs={
                "block_chars": len(block),
                "block_preview": block[:1200],
                "facts_count": len(facts), "entities_count": len(entities),
                "conflicts_detected": len(conflicts),
            },
            factors=[
                f"cognitive health status: {health['status']} (population={health['population']})",
                f"mood={snap.get('mood')}, energy={energy}, uncertainty={uncertainty}",
                f"contradictions queue size: {len(snap.get('contradictions', []))}",
                f"pending memory conflicts staged for confirm_memory_update: {len(conflicts)}",
            ],
        )
        return block

    def metacognitive_context(self, query: str = "", memories: list[dict] | None = None) -> str:
        """Return a compact pre-response confidence and clarification check. Metacognitive review."""
        snap = self.snapshot()
        rows = memories or []
        evidence = [str(r.get("memory") or r.get("text") or r.get("trace") or "").strip() for r in rows[:5]]
        evidence = [item for item in evidence if item]
        statuses = {str(r.get("status") or "").casefold() for r in rows}
        temporal = any(word in (query or "").casefold() for word in ("latest", "today", "now", "current", "recent"))
        flags = []
        if not evidence:
            flags.append("no retrieved personal evidence")
        if snap["uncertainty"] > 0.35:
            flags.append("user uncertainty cue is elevated")
        if snap.get("contradictions"):
            flags.append("recent statements may conflict; clarify before relying on them")
        if "superseded" in statuses:
            flags.append("some retrieved memory may be outdated")
        if self.memory_conflicts(query, rows):
            flags.append("new statement conflicts with retrieved memory; clarify before updating belief")
        if self.memory_resolution_guidance(query, rows):
            flags.append("memory conflict requires explicit confirmation before supersession")
        if temporal:
            flags.append("query may require current external information")
        confidence = "low" if len(flags) >= 2 or not evidence else "moderate"
        action = "ask or verify before asserting" if flags else "answer, while separating memory from inference"
        lines = ["<metacognitive_checkpoint>", f"Evidence available: {len(evidence)} memory item(s)", f"Confidence: {confidence}", "Checks: " + ("; ".join(flags) if flags else "no immediate warning"), "Response discipline: " + action, "</metacognitive_checkpoint>"]
        block = "\n".join(lines)
        _bt_record(
            "EdgeCognitiveState.metacognitive_context",
            layer="context",
            inputs={"query": query, "memories_count": len(rows)},
            outputs={
                "confidence": confidence, "flags_count": len(flags),
                "block_chars": len(block), "block_preview": block,
                "flags": flags,
            },
            factors=[
                f"low confidence if len(flags)≥2 OR no evidence",
                f"temporal query trigger: {temporal}",
                f"any 'superseded' status in hits: {'superseded' in statuses}",
            ],
        )
        return block

    def context(self, query: str = "") -> str:
        """Render bounded recent state for injection into active cognition. Context rendering."""
        if not EDGE_COGNITION_ENABLED:
            return ""
        query_tokens = _tokens(query)
        with self._lock:
            events, loops, affect = list(self._events), list(self._open_loops), self._affect
            goals = [g.text for g in self._goals if g.progress == "active"]
            lessons = list(self._lessons)
            energy, uncertainty, attention = self._energy, self._uncertainty, self._attention
            durable_lessons = list(self._durable_lessons)
        if not events and not loops and not goals:
            return ""
        ranked = sorted(enumerate(events), key=lambda p: (len(p[1].tokens & query_tokens) * 3 + p[0], p[0]), reverse=True)
        lines, used = [], 0
        for _, event in ranked:
            line = f"User: {event.user}\nAiko: {event.assistant}"
            if used + len(line) > EDGE_COGNITION_MAX_CHARS:
                continue
            lines.append(line); used += len(line)
        if loops:
            lines.append("Open loops: " + " | ".join(loops))
        if goals:
            lines.append("Active goals: " + " | ".join(goals))
        if lessons:
            lines.append("Lessons from outcomes: " + " | ".join(lessons[:3]))
        if durable_lessons:
            lines.append("Durable interaction rules: " + " | ".join(durable_lessons[:3]))
        mood = "positive" if affect > 0.2 else "negative" if affect < -0.2 else "neutral"
        energy_label = "low" if energy < 0.35 else "high" if energy > 0.65 else "steady"
        uncertainty_label = "elevated" if uncertainty > 0.35 else "ordinary"
        lines.append(f"Internal state: mood={mood}; energy={energy_label}; uncertainty={uncertainty_label}")
        if attention: lines.append(f"Recent attention: {attention}")
        body = "\n\n".join(lines)
        return "<edge_cognitive_state>\n" + self._bounded(body, EDGE_COGNITION_MAX_CHARS) + "\n</edge_cognitive_state>"


_states: OrderedDict[str, EdgeCognitiveState] = OrderedDict()
_states_lock = threading.Lock()


def flush_all_persist() -> None:
    """Explicitly flush all pending persists (call at run-end)."""
    _flush_persist_pending()


def for_identity(identity: str) -> EdgeCognitiveState:
    """Retrieve or create bounded state for an identity. State factory."""
    with _states_lock:
        key = identity or "default"
        state = _states.pop(key, None) or EdgeCognitiveState(key)
        if not state._persistent_loaded:
            state.load_persistent()
        _states[key] = state
        while len(_states) > EDGE_COGNITION_MAX_IDENTITIES:
            _states.popitem(last=False)
        return state


__all__ = ["EdgeCognitiveState", "for_identity"]
