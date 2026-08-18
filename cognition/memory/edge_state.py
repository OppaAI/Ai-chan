"""Small, zero-I/O cognitive state for edge deployments.

This bounded per-user layer bridges working and long-term memory: it keeps
salient recent turns, open questions/tasks, and a coarse affect cue. It uses no LLM, embeddings, database, or worker thread on the hot path.

This is deliberately one bounded module rather than a collection of tiny
"emotion", "goals", and "working memory" stores. It is the fast internal
state layer between the current turn and long-term memory.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass

from .env import env_bool, env_flag, env_int

EDGE_COGNITION_ENABLED = env_flag("EDGE_COGNITION_ENABLED", "1")
EDGE_COGNITION_PERSIST = env_bool("EDGE_COGNITION_PERSIST", "1")
EDGE_COGNITION_MAX_TURNS = max(1, env_int("EDGE_COGNITION_MAX_TURNS", 7))
EDGE_COGNITION_MAX_CHARS = max(240, env_int("EDGE_COGNITION_MAX_CHARS", 1200))
EDGE_COGNITION_MAX_OPEN_LOOPS = max(1, env_int("EDGE_COGNITION_MAX_OPEN_LOOPS", 3))
EDGE_COGNITION_MAX_GOALS = max(1, env_int("EDGE_COGNITION_MAX_GOALS", 5))
EDGE_COGNITION_MAX_IDENTITIES = max(1, env_int("EDGE_COGNITION_MAX_IDENTITIES", 16))
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
_STOP = {"the", "and", "that", "this", "with", "you", "are", "for", "have", "from", "about", "can", "could", "would", "should", "please", "today", "tell", "me", "do", "does", "did", "want", "need", "help"}


@dataclass(slots=True)
class _Event:
    user: str
    assistant: str
    tokens: frozenset[str]


@dataclass(slots=True)
class _Goal:
    text: str
    progress: str = "active"


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP)


def _affect(text: str) -> float:
    lower = (text or "").lower()
    pos = sum(lower.count(w) for w in ("love", "great", "thanks", "happy", "good", "excited"))
    neg = sum(lower.count(w) for w in ("hate", "sad", "angry", "bad", "frustrated", "worried"))
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


class EdgeCognitiveState:
    """Bounded per-identity state; all operations are lock-protected."""

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
        self._lock = threading.RLock()

    def record(self, user: str, assistant: str) -> None:
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
            if _DONE_RE.search(user):
                self._close_matching_goal(user)
                self._close_matching_loop(user)
            elif _DONE_RE.search(assistant) and not _OUTCOME_FAIL_RE.search(assistant):
                self._close_matching_goal(assistant)
                self._close_matching_loop(assistant)

    def clear(self) -> None:
        with self._lock:
            self._events.clear(); self._open_loops.clear(); self._goals.clear(); self._lessons.clear(); self._tool_outcomes.clear(); self._perceptions.clear(); self._activity = ""; self._response_reviews.clear(); self._contradictions.clear(); self._durable_lessons.clear(); self._lesson_counts.clear(); self._affect = 0.0; self._energy = 0.5; self._uncertainty = 0.0; self._attention = ""


    @staticmethod
    def _energy_signal(text: str) -> float:
        if _ENERGY_LOW_RE.search(text):
            return 0.2
        if _ENERGY_HIGH_RE.search(text):
            return 0.8
        return 0.5


    @staticmethod
    def _uncertainty_signal(text: str) -> float:
        return 0.8 if _UNCERTAIN_RE.search(text) else 0.0


    @staticmethod
    def _attention_for(user: str) -> str:
        """Build a readable focus phrase while preserving word order."""
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

    def _add_lesson(self, prefix: str, source: str, feedback: str = "") -> None:
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
        self._lesson_counts[signature] = min(count, 3)
        if count >= 2:
            direction = "Prefer this approach: " if prefix.startswith("The previous approach appeared useful:") else "Avoid repeating: "
            durable = "Durable interaction rule: " + (semantic or (direction + text))
            if durable not in self._durable_lessons:
                self._durable_lessons.appendleft(durable)

    def _capture_goal(self, user: str) -> None:
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
        words = _tokens(user)
        if not words:
            return
        for goal in self._goals:
            goal_words = _tokens(goal.text)
            overlap = len(words & goal_words)
            if overlap >= 1 and (goal_words and overlap / len(goal_words) >= 0.2):
                goal.progress = "completed"


    def _close_matching_loop(self, user: str) -> None:
        """Remove an explicitly completed question or task from open loops."""
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
        """Record a bounded conflict when a new statement reverses a recent one."""
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



    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def snapshot(self) -> dict:
        """Return a compact diagnostic snapshot without exposing mutable state."""
        with self._lock:
            mood = "positive" if self._affect > 0.2 else "negative" if self._affect < -0.2 else "neutral"
            return {"mood": mood, "affect": round(self._affect, 3), "energy": round(self._energy, 3), "uncertainty": round(self._uncertainty, 3), "attention": self._attention, "open_loops": list(self._open_loops), "goals": [g.text for g in self._goals if g.progress == "active"], "lessons": list(self._lessons), "tool_outcomes": list(self._tool_outcomes), "perceptions": list(self._perceptions), "activity": self._activity, "response_reviews": list(self._response_reviews), "contradictions": list(self._contradictions), "durable_lessons": list(self._durable_lessons), "lesson_evidence": dict(self._lesson_counts), "preferences": dict(self._preferences), "identity_questions": list(self._identity_questions)}

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
                "contradictions": len(self._contradictions),
            }
            populated = sum(1 for key, value in components.items() if value > 0 and key != "contradictions")
            population = round(populated / 6.0, 3)
            status = "empty" if not self._events else "sparse" if population < 0.34 else "active"
            return {"status": status, "population": population, "components": components, "attention_valid": attention_words >= 2 or not self._events}

    def identity_guidance(self) -> str:
        """Give grounded guidance for unresolved user-identity questions."""
        with self._lock:
            pending = list(self._identity_questions)
        if not pending:
            return "<identity_guidance>\nNo unresolved identity question.\n</identity_guidance>"
        return "<identity_guidance>\nThe user asked about personal familiarity. Do not invent recognition; use only grounded identity or memory evidence, and ask for clarification if needed.\n</identity_guidance>"

    def record_tool_outcome(self, tool: str, *, ok: bool, detail: str = "", error_type: str = "") -> None:
        """Record a compact tool outcome for disclosure and future learning."""
        item = {"tool": str(tool or "tool")[:48], "ok": bool(ok), "detail": str(detail or "")[:240]}
        if error_type:
            item["error_type"] = str(error_type)[:48]
        with self._lock:
            self._tool_outcomes.appendleft(item)
            if not ok:
                failure_key = str(error_type or "tool_error")
                self._add_lesson("Tool limitation: ", f"{tool} ({failure_key})", "tool failed")

    def lesson_guidance(self) -> str:
        """Render only repeatedly evidenced lessons as behavior guidance."""
        snap = self.snapshot()
        durable = snap.get("durable_lessons") or []
        if not durable:
            return "<lesson_guidance>\nNo repeatedly confirmed lessons yet.\n</lesson_guidance>"
        lines = [f"- {lesson}" for lesson in durable[:3]]
        return "<lesson_guidance>\n" + "\n".join(lines) + "\n</lesson_guidance>"

    def consume_lessons(self) -> list[str]:
        """Return current outcome lessons for successful background consolidation."""
        with self._lock:
            lessons = list(self._lessons)
            self._lessons.clear()
            return lessons

    def load_persistent(self) -> None:
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
                self._lesson_counts = {str(k): min(3, int(v)) for k, v in (data.get("lesson_evidence") or {}).items() if str(k) and str(v).isdigit()}
                self._preferences = {str(k): str(v) for k, v in (data.get("preferences") or {}).items()}
                self._identity_questions = deque(data.get("identity_questions", [])[:3], maxlen=3)
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
        if not EDGE_COGNITION_PERSIST or not self._identity or self._identity == "default":
            return
        try:
            from cognition.memory.vecstore import connect_sqlite_db
            with self._lock:
                data = {"open_loops": list(self._open_loops), "goals": [g.text for g in self._goals if g.progress == "active"], "lessons": list(self._lessons), "tool_outcomes": list(self._tool_outcomes), "contradictions": list(self._contradictions), "durable_lessons": list(self._durable_lessons), "lesson_evidence": dict(self._lesson_counts), "preferences": dict(self._preferences), "activity": self._activity, "affect": self._affect, "energy": self._energy, "uncertainty": self._uncertainty, "attention": self._attention}
            conn = connect_sqlite_db("memory/memory.db", user_id=self._identity)
            conn.execute("CREATE TABLE IF NOT EXISTS cognitive_state (user_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("INSERT INTO cognitive_state(user_id, state_json) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET state_json=excluded.state_json, updated_at=CURRENT_TIMESTAMP", (self._identity, json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
            conn.commit()
            conn.close()
        except Exception:
            return

    def record_perception(self, source: str, duration_s: float | None = None, latency_s: float | None = None, prosody: dict | None = None) -> None:
        """Store bounded aggregate perception cues; never retain raw audio."""
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
        """Rank memories as a bounded reconstruction with confidence cues."""
        rows = list(memories or [])
        if not rows:
            return []
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
        return [row for _, _, row in scored]

    def memory_conflicts(self, query: str, memories: list[dict] | None = None) -> list[dict]:
        """Find conservative conflicts between a new statement and recalled facts."""
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
        """Describe a safe clarification/update path without mutating memory."""
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
        """Consume pending conflicts only after an explicit confirmation."""
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
        """Summarize bounded internal state for continuity and self-correction."""
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
        failed_tools = [item for item in snap.get("tool_outcomes", [])[:3] if not item.get("ok")]
        if failed_tools:
            lines.append("Recent limitation: a tool attempt failed; disclose this if relevant")
        if snap.get("response_reviews") and snap["response_reviews"][0].get("flags"):
            lines.append("Last self-check: " + "; ".join(snap["response_reviews"][0]["flags"][:2]))
        if not lines:
            lines.append("No unresolved cognitive issue is currently salient")
        return "<self_reflection>\n" + "\n".join("- " + line for line in lines[:5]) + "\n</self_reflection>"

    def adaptive_tts_rate(self) -> float:
        """Choose a conservative speech rate from recent voice cues."""
        latest = (self.snapshot().get("perceptions") or [{}])[0]
        if isinstance(latest.get("words_per_second"), (int, float)) and latest["words_per_second"] > 4.5:
            return 1.05
        if (isinstance(latest.get("pause_density"), (int, float)) and latest["pause_density"] > 0.55) or self.snapshot().get("uncertainty", 0.0) > 0.35:
            return 0.9
        if self.snapshot().get("affect", 0.0) < -0.25:
            return 0.92
        return 1.0

    def preference_guidance(self) -> str:
        """Turn learned preferences into explicit response behavior rules."""
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
        """Render bounded behavior guidance from current affect and prosody."""
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
        """Audit a completed draft for bounded metacognitive warning signs."""
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
        review = {"flags": flags, "confidence": "low" if len(flags) >= 2 else "moderate" if flags else "high", "response_chars": len(text)}
        with self._lock:
            self._response_reviews.appendleft(review)
        return review

    def grounded_context(self, now=None, idle_seconds: float = 0.0, resting: bool = False, scheduled_jobs: list[dict] | None = None, project_signals: list[str] | None = None) -> str:
        """Render bounded real-world signals, including known scheduled work."""
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
            if latest.get("words_per_second") is not None: cues.append(f"pace={latest["words_per_second"]}wps")
            if latest.get("pause_density") is not None: cues.append(f"pauses={latest["pause_density"]}")
            if latest.get("rms") is not None: cues.append(f"energy={latest["rms"]}")
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
        """Build a bounded situation model from already-retrieved context."""
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
            return ""
        health = self.cognitive_health()
        lines = ["<situation_model>", "Organized from available evidence; treat it as context, not certainty.", f"Current query: {query[:260]}"]
        lines.append(f"Cognitive state: {health["status"]}; population={health["population"]}")
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
        lines.append(f"Internal cues: mood={snap["mood"]}, energy={energy}, uncertainty={uncertainty}")
        lines.append("Evidence confidence: " + ("moderate" if facts else "low"))
        return "\n".join(lines) + "\n</situation_model>"

    def metacognitive_context(self, query: str = "", memories: list[dict] | None = None) -> str:
        """Return a compact pre-response confidence and clarification check."""
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
        return "\n".join(lines)

    def context(self, query: str = "") -> str:
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


def for_identity(identity: str) -> EdgeCognitiveState:
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
