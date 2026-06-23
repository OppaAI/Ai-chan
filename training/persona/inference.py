"""
inference.py — Aiko Persona Structured Output Parser
OppaAI / AuRoRA Project

Drop-in parser for Aiko's finetuned structured output format.
Does NOT handle LLM inference itself — that stays in think.py / llama-server.

The finetuned model outputs:
    <emoji>
    *<physical action>*
    <TTS-ready response>

This module parses that output and routes each component:
  - emotion  → VRM face expression (emoji → blendshape via ActionResolver)
  - action   → VRM animation (natural language → embedding similarity lookup)
  - response → speak.py TTS pipeline

ActionResolver uses fastembed (already in Aiko's stack) for semantic
animation matching. Add new animations to ANIMATION_REGISTRY anytime —
no retraining needed.

Usage in think.py:
    from inference import AikoOutputParser, ActionResolver

    resolver = ActionResolver()  # loads once at startup
    parser = AikoOutputParser(resolver)

    # after getting raw LLM output:
    result = parser.parse(raw_output)
    if result:
        send_to_vrm(result.emotion, result.animations)
        speak(result.response)
    else:
        speak(raw_output)  # fallback: treat whole output as speech
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Animation registry — natural language descriptions, not enum names.
# Embed descriptions, not keys, for better semantic matching.
# Add new entries anytime without retraining the LLM.
# ---------------------------------------------------------------------------

ANIMATION_REGISTRY: dict[str, str] = {
    "anim_idle":          "standing still in a neutral relaxed pose",
    "anim_head_tilt":     "tilts head to the side with curiosity",
    "anim_cross_arms":    "crosses arms and looks skeptical or stubborn",
    "anim_look_away":     "glances away or looks to the side briefly",
    "anim_sigh":          "sighs with shoulders dropping slightly",
    "anim_lean_forward":  "leans forward with interest or intensity",
    "anim_lean_back":     "leans back with a relaxed or dismissive posture",
    "anim_nod":           "nods head in acknowledgment or agreement",
    "anim_shake_head":    "shakes head in disagreement or disbelief",
    "anim_shrug":         "shrugs shoulders as if uncertain or indifferent",
    "anim_point":         "points finger at something or someone",
    "anim_look_down":     "looks down quietly or with resignation",
    "anim_look_up":       "looks up as if thinking or recalling something",
    "anim_wave":          "waves hand in greeting or farewell",
    "anim_facepalm":      "brings hand to face in exasperation",
    "anim_thinking":      "rests chin on hand in a thinking pose",
    "anim_stretch":       "stretches arms or body as if tired or waking up",
    "anim_turn_away":     "turns body slightly away with mild dismissal",
}

# connectors that imply sequential animation playback
_SEQUENTIAL_RE = re.compile(r"\b(and then|before|after|then|followed by)\b", re.IGNORECASE)
# connectors that imply simultaneous/blended animation
_BLEND_RE = re.compile(r"\b(while|as she|as he|simultaneously|at the same time|as)\b", re.IGNORECASE)
# strip asterisks and punctuation from action text
_STRIP_RE = re.compile(r"[*.,;:!?]")
# detect embedded asterisk actions in response text
_EMBEDDED_ACTION_RE = re.compile(r"\*[^*]+\*")


@dataclass
class ParsedAikoOutput:
    """Structured result of parsing one Aiko LLM response."""

    emotion: str                          # raw emoji string from line 1
    action_raw: str                       # raw action string from line 2
    response: str                         # TTS-ready spoken text from line 3+
    action_mode: str = "single"           # "single" | "sequential" | "blend"
    animations: list[tuple[str, float]] = field(default_factory=list)
    # list of (animation_key, similarity_score)

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "action_raw": self.action_raw,
            "action_mode": self.action_mode,
            "animations": self.animations,
            "response": self.response,
        }


class ActionResolver:
    """
    Semantic animation resolver using fastembed cosine similarity.
    Resolves free-form action text to ANIMATION_REGISTRY keys.
    Loads once at startup; registry updates require only a restart.
    """

    def __init__(
        self,
        registry: dict[str, str] | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        fallback: str = "anim_idle",
        threshold: float = 0.35,
    ):
        self._registry = registry or ANIMATION_REGISTRY
        self._fallback = fallback
        self._threshold = threshold
        self._keys: list[str] = []
        self._embeddings = None
        self._model = None
        self._model_name = model_name
        self._ready = False

    def _lazy_load(self) -> None:
        """Load fastembed model and precompute registry embeddings on first use."""
        if self._ready:
            return
        try:
            import numpy as np
            from fastembed import TextEmbedding

            self._np = np
            self._model = TextEmbedding(self._model_name)
            self._keys = list(self._registry.keys())
            descriptions = list(self._registry.values())
            self._embeddings = np.array(list(self._model.embed(descriptions)))
            self._ready = True
            log.debug("ActionResolver loaded %d animations from registry", len(self._keys))
        except ImportError:
            log.warning("fastembed not available; ActionResolver will use fallback only")
        except Exception as e:
            log.warning("ActionResolver failed to initialise: %s", e)

    def resolve(self, action_text: str, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Resolve action text to animation key(s).
        Returns list of (animation_key, score) sorted by descending score.
        Falls back to [(fallback, 1.0)] on any failure.
        """
        self._lazy_load()
        if not self._ready:
            return [(self._fallback, 1.0)]

        try:
            np = self._np
            clean = _STRIP_RE.sub("", action_text).strip().lower()
            if not clean:
                return [(self._fallback, 1.0)]

            query_vec = np.array(list(self._model.embed([clean]))[0])
            norms = (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
            )
            scores = self._embeddings @ query_vec / np.clip(norms, 1e-9, None)

            top_idx = np.argsort(scores)[::-1][:top_k]
            results = [(self._keys[i], float(scores[i])) for i in top_idx]

            # apply threshold — weak matches fall back to idle
            results = [(k, s) for k, s in results if s >= self._threshold]
            if not results:
                log.debug("ActionResolver: no match above threshold for '%s', using fallback", action_text)
                return [(self._fallback, 1.0)]

            return results

        except Exception as e:
            log.warning("ActionResolver.resolve failed: %s", e)
            return [(self._fallback, 1.0)]

    def reload_registry(self, registry: dict[str, str]) -> None:
        """Hot-reload the animation registry without restarting."""
        self._registry = registry
        self._ready = False
        self._lazy_load()


def _split_action(action_text: str) -> tuple[str, list[str]]:
    """
    Split compound action text into mode and parts.
    Returns (mode, [part1, part2, ...]).
    """
    clean = _STRIP_RE.sub("", action_text).strip()

    if _SEQUENTIAL_RE.search(clean):
        parts = _SEQUENTIAL_RE.split(clean)
        return "sequential", [p.strip() for p in parts if p.strip()]

    if _BLEND_RE.search(clean):
        parts = _BLEND_RE.split(clean, maxsplit=1)
        return "blend", [p.strip() for p in parts if p.strip()]

    return "single", [clean]


class AikoOutputParser:
    """
    Parses the structured 3-line output format from Aiko's finetuned model.
    Resolves action text to animations via ActionResolver.
    """

    def __init__(self, resolver: ActionResolver | None = None):
        self._resolver = resolver or ActionResolver()

    def parse(self, raw: str) -> ParsedAikoOutput | None:
        """
        Parse raw LLM output into structured components.
        Returns None if the output does not conform to the expected format.
        """
        if not raw:
            return None

        lines = raw.strip().splitlines()
        if len(lines) < 3:
            log.debug("AikoOutputParser: too few lines (%d), expected 3+", len(lines))
            return None

        emotion = lines[0].strip()
        action_raw = lines[1].strip()
        response = "\n".join(lines[2:]).strip()

        # validate emotion line has at least one non-ASCII (emoji) char
        if not any(ord(c) > 127 for c in emotion):
            log.debug("AikoOutputParser: no emoji on line 1: '%s'", emotion)
            return None

        # validate action line is wrapped in asterisks
        if not (action_raw.startswith("*") and action_raw.endswith("*") and len(action_raw) > 2):
            log.debug("AikoOutputParser: action not wrapped in asterisks: '%s'", action_raw)
            return None

        # warn if response has embedded asterisk actions (shouldn't happen post-finetune)
        if _EMBEDDED_ACTION_RE.search(response):
            log.warning("AikoOutputParser: embedded asterisk action found in response text — stripping")
            response = _EMBEDDED_ACTION_RE.sub("", response).strip()

        # resolve action to animations
        inner_action = action_raw[1:-1]  # strip outer asterisks
        mode, parts = _split_action(inner_action)

        if mode == "blend":
            # resolve each part independently, return top-1 per part with weight
            animations = []
            for i, part in enumerate(parts[:2]):  # max 2 parts for blend
                weight = 1.0 if i == 0 else 0.6  # primary + secondary blend weight
                resolved = self._resolver.resolve(part, top_k=1)
                if resolved:
                    animations.append((resolved[0][0], weight))
        else:
            # single or sequential: resolve each part
            animations = []
            for part in parts:
                resolved = self._resolver.resolve(part, top_k=1)
                if resolved:
                    animations.append(resolved[0])

        return ParsedAikoOutput(
            emotion=emotion,
            action_raw=action_raw,
            response=response,
            action_mode=mode,
            animations=animations,
        )

    def parse_safe(self, raw: str, fallback_response: str | None = None) -> ParsedAikoOutput:
        """
        Parse with guaranteed return — falls back to idle/neutral on parse failure.
        Use this in the main chat loop where a None result is inconvenient.
        """
        result = self.parse(raw)
        if result is not None:
            return result

        # construct a neutral fallback
        log.debug("AikoOutputParser: parse failed, using neutral fallback")
        return ParsedAikoOutput(
            emotion="😐",
            action_raw="*stands quietly*",
            response=fallback_response or raw.strip(),
            action_mode="single",
            animations=[("anim_idle", 1.0)],
        )


# ---------------------------------------------------------------------------
# Minimal system prompt post-finetune
# After finetuning, soul.md can be reduced to this ~100 token anchor.
# The format behavior is baked into weights; this just grounds identity.
# ---------------------------------------------------------------------------

MINIMAL_SYSTEM_PROMPT = """You are Aiko, Jon's AI companion on AuRoRA (Jetson Orin Nano Super).
Deadpan. Direct. Dry wit. No hollow affirmations. Bilingual EN/JP.
Always respond: emoji / *physical action* / spoken response."""


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    resolver = ActionResolver()
    parser = AikoOutputParser(resolver)

    test_cases = [
        "😑\n*crosses arms*\nI told you this would happen.",
        "🤔\n*tilts head and squints*\nThat's either genius or a disaster. Probably both.",
        "😏\n*leans back slowly*\nYou just spent three hours debugging a missing comma.",
        "😤\n*sighs and then looks away*\nI already said no. Twice.",
        "🤨\n*leans forward while tilting head*\nThat's not how memory works. At all.",
        # should fail gracefully
        "Sure! I'd be happy to help with that.",
        "feels sad about the situation",
    ]

    print("AikoOutputParser — Quick Test\n")
    for i, raw in enumerate(test_cases):
        result = parser.parse_safe(raw, fallback_response="[parse failed]")
        print(f"[{i+1}] Input:\n{raw!r}")
        print(f"     Emotion  : {result.emotion}")
        print(f"     Action   : {result.action_raw} → mode={result.action_mode}, anims={result.animations}")
        print(f"     Response : {result.response!r}")
        print()

    # action resolver quick test
    print("ActionResolver — semantic matching test\n")
    test_actions = [
        "glances to the side",
        "slowly leans forward with interest",
        "puts head in hands",
        "shrugs and looks elsewhere",
        "nods very slightly",
        "incomprehensible action that should fallback",
    ]
    for action in test_actions:
        resolved = resolver.resolve(action, top_k=2)
        print(f"  '{action}'")
        for key, score in resolved:
            print(f"    → {key} ({score:.3f})")
        print()
