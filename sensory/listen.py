"""
sensory/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio; Silero VAD (neural) is the single authoritative
    speech/silence gate for ALL audio sources, local mic or WebUI.
  - For the WebUI path, the browser only runs a lightweight energy-RMS gate
    client-side (see static/vad.js) to decide "loud enough to bother sending" —
    it is NOT a speech/silence judgment. Silero here is what actually decides
    what is speech, on every chunk, regardless of source.
  - Transcribes via SenseVoice (sherpa-onnx, int8 ONNX) in a background thread,
    then applies post-ASR name/phrase corrections (see correct_asr_text below)
  - Optionally verifies the speaker against one enrolled voice embedding
    (sherpa-onnx SpeakerEmbeddingExtractor) on the same buffered audio, run
    in parallel with transcription — see SPEAKER_VERIFY_ENABLED below
  - Optionally gates responses behind a wake word ("Hey Aiko") and/or a
    trigger phrase said alongside speaker verification ("Here is Oppa") —
    see WAKE_WORD below
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_asr() → load_vad() → load_speaker_id() → join_warmup()
    for granular boot progress reporting via wakeup.py
  - Always-on barge-in VAD monitor: start_barge_in_monitor() runs a
    lightweight Silero-only daemon that sets _barge_in_event when speech is
    detected during TTS playback, enabling speak.wait_or_barge_in()

Barge-in, the speaker-verify drop-gate, and post-ASR corrections are native
to this module (see barge_in_enabled() / speaker_verify_gate() /
correct_asr_text() below) — there is no external monkeypatch layer and no
separate bind-at-boot step (formerly sensory/voice_gates.py, then
sensory/listen_native.py). BARGE_IN_ENABLED=0 (the default) is enforced
directly inside trigger_barge_in() / _barge_in_loop() / listen(), so the
switch is load-bearing by construction rather than depending on some other
module getting imported and run first.

Dependencies:
    pip install sherpa-onnx numpy silero-vad scipy huggingface_hub rapidfuzz
    Model: auto-downloaded to HF cache on first use (see ASR_MODEL in .env)
    parec (PulseAudio) required for mic capture — no PortAudio/sounddevice
    rapidfuzz is optional — falls back to stdlib difflib if not installed,
    just slower.

Speaker verification (optional — see SPEAKER_VERIFY_ENABLED in .env):
    1. Download a speaker embedding model (.onnx) from
       https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
       e.g. 3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx (~28MB)
    2. Set SPEAKER_MODEL_PATH in .env to point at it
    3. Enroll your voice: python -m util.enroll_speak
    4. Set SPEAKER_VERIFY_ENABLED=1 in .env
    SPEAKER_VERIFY_GATE=1 additionally drops utterances that fail the cosine
    match (see speaker_verify_gate() / listen() below); default 0 keeps the
    score as metadata only (legacy behavior).

Wake word / trigger phrase (optional — see config/sensory.yaml):
    WAKE_WORD ("" by default, disabled): SenseVoice mangles
    "Aiko" unpredictably since it's not a normal English word, so matching is
    fuzzy (rapidfuzz ratio) against the leading words of the transcript, not
    an exact substring check. WAKE_WORD_ALIASES lets you hardcode observed
    mishearings ("hey iko|hey eco|hey ecko") as extra candidates.

    Once woken/triggered, Aiko stays "active" (no phrase required) until
    ACTIVATION_TIMEOUT_S seconds pass with no further utterance, at which
    point the session goes back to sleep and the configured phrase(s) are
    required again. Use AikoListen.is_active() to check this from other
    subsystems (e.g. suppress proactive/unsolicited behavior while asleep),
    and AikoListen.sleep_now() to force it inactive (e.g. an explicit
    "go to sleep" command).

Post-ASR name / phrase corrections (S2, no finetune):
    Lightweight ordered phrase replacements applied after every transcript
    (see correct_asr_text() below), for names ASR reliably mangles (e.g.
    "Aiko", "OppaAI"). Configure extra pairs via ASR_CORRECTIONS in .env:
      "op ai->OppaAI|hey iko->hey Aiko|my project x->Project X"
    Built-in defaults already cover Aiko / OppaAI; user map is applied on
    top and wins on same key. Longest phrase matches first.

Known architectural limitation (audit item #7, not fixed here):
    _transcribe() runs SenseVoice as a single full-utterance batch decode
    after Silero declares silence — there are no interim/partial
    transcripts while the user is still speaking, and end-of-turn latency
    is bounded below by SILENCE_CHUNKS * ~32ms (~2.1s default) regardless of
    how clearly the sentence ended. SenseVoice itself is non-streaming, so
    fixing this means adding a second, cheaper streaming ASR model in front
    (fast draft for interim UI display, SenseVoice still used for the final
    transcript) — a real feature addition with its own model-loading and
    latency-budget tradeoffs on the Jetson, not something to fold into a
    bug-fix pass blind. Left as a scoped follow-up.
"""
from __future__ import annotations

import onnxruntime as _ort
if hasattr(_ort, "set_default_logger_severity"):
    _ort.set_default_logger_severity(3)

from functools import lru_cache
from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad
from system.userspace import user_state_path
import json
import logging as _logging
import numpy as np
import os
import re

log = _logging.getLogger(__name__)

from scipy.signal import resample_poly
import select
import sherpa_onnx
import subprocess
import threading
import time
import torch
import warnings

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None
    import difflib as _difflib

warnings.filterwarnings("ignore")
_logging.getLogger("sherpa_onnx").setLevel(_logging.ERROR)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'listen_asr':     'Loading SenseVoice ASR model...',
    'listen_silero':  'Loading Silero VAD...',
    'listen_speaker': 'Loading speaker verification...',
    'listen_warmup':  'Warming up ASR pipeline...',
    'listen_ready':   'Microphone ready',
    'listen_skip':    'ASR skipped (text mode)',
}

# ── config ────────────────────────────────────────────────────────────────────

ASR_DEVICE      = os.getenv("ASR_DEVICE", "cpu")       # resolved from config/sensory.yaml via load_config()
ASR_LANGUAGE    = os.getenv("ASR_LANGUAGE", "auto")    # auto, zh, en, ja, ko, yue, nospeech
ASR_NUM_THREADS = int(os.getenv("ASR_NUM_THREADS", "4"))

# HuggingFace repo — model.int8.onnx + tokens.txt downloaded on first use
ASR_MODEL = os.getenv(
    "ASR_MODEL",
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)

# LISTEN_VAD_SILENCE_MS / LISTEN_VAD_PAD_MS were removed from here — they were
# read from sensory.yaml but never wired into any actual behavior (the real
# silence timeout is SILENCE_CHUNKS * _CHUNK_MS_ACTUAL below, ~2.1s by
# default, not the 300ms the old yaml comment implied). Rather than silently
# retargeting SILENCE_CHUNKS to match the misleading 300ms default — a
# behavior change that needs testing on hardware, not a blind patch — the
# dead keys have been dropped from sensory.yaml. If you want silence timeout
# configurable in milliseconds again, derive SILENCE_CHUNKS from a real ms
# value explicitly: SILENCE_CHUNKS = round(VAD_SILENCE_MS / _CHUNK_MS_ACTUAL).
# LISTEN_VAD_PAD_MS was never implemented for the local-mic path at all —
# only the WebUI gets pre-speech padding, via vad.js's PRE_SPEECH_BUFS
# (client-side, hardcoded ~700ms, independent of this config).

SAMPLE_RATE         = 16000                                          # ASR + Silero target
LISTEN_DEVICE       = os.getenv("LISTEN_DEVICE", None)              # None = default

CHUNK_DURATION_MS   = int(os.getenv("LISTEN_CHUNK_MS",         30))  # Silero minimum
VAD_THRESHOLD       = float(os.getenv("LISTEN_VAD_THRESHOLD", 0.5))  # Silero speech prob cutoff
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS",   66))  # matches config/sensory.yaml default
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",       10))
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",      30))

BARGE_IN_THRESHOLD     = float(os.getenv("BARGE_IN_THRESHOLD",     "0.95"))  # matches config/sensory.yaml default
BARGE_IN_CONFIRM       = int(os.getenv("BARGE_IN_CONFIRM_CHUNKS",  "4"))     # matches config/sensory.yaml default
BARGE_IN_COOLDOWN_MS   = int(os.getenv("BARGE_IN_COOLDOWN_MS",     "800"))
# BARGE_IN_ALWAYS_ON is intentionally NOT cached here — see barge_in_always_on()
# below. It must be read live, like BARGE_IN_ENABLED and SPEAKER_VERIFY_GATE,
# so it can be toggled at runtime without a process restart.

# ── speaker verification config ──────────────────────────────────────────────
# Single-enrollment 1:1 verification (not multi-speaker identification) —
# Aiko has exactly one "owner" voice to check against.

SPEAKER_VERIFY_ENABLED   = os.getenv("SPEAKER_VERIFY_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
SPEAKER_MODEL_PATH       = os.path.expanduser(os.getenv("SPEAKER_MODEL_PATH", ""))            # path to embedding .onnx
SPEAKER_VERIFY_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.5"))  # cosine sim cutoff
SPEAKER_NUM_THREADS      = int(os.getenv("SPEAKER_NUM_THREADS", "1"))

# ── wake word / trigger phrase config ────────────────────────────────────────
# WAKE_WORD: "" disables wake-word gating entirely (Aiko responds to every
#   utterance, as before). When set, ASR is unreliable on "Aiko" (not a
#   normal English word) so matching is fuzzy, not exact-substring.
WAKE_WORD             = os.getenv("WAKE_WORD", "").strip().lower()
WAKE_WORD_ALIASES     = [w.strip().lower() for w in os.getenv("WAKE_WORD_ALIASES", "").split("|") if w.strip()]
WAKE_FUZZY_THRESHOLD  = float(os.getenv("WAKE_FUZZY_THRESHOLD", "70"))

ACTIVATION_TIMEOUT_S = float(os.getenv("ACTIVATION_TIMEOUT_S", "3600"))  # matches config/sensory.yaml default

_CHUNK_SAMPLES_VAD = 512                                             # at 16 kHz, ~32 ms
_CHUNK_MS_ACTUAL   = (_CHUNK_SAMPLES_VAD / SAMPLE_RATE) * 1000.0      # 32.0 ms — the real, non-configurable chunk size
_MAX_CHUNKS        = int(MAX_RECORD_SECONDS * 1000 / _CHUNK_MS_ACTUAL)
# NOTE: CHUNK_DURATION_MS (LISTEN_CHUNK_MS in sensory.yaml) is NOT used to
# compute _MAX_CHUNKS anymore — Silero's chunk size is fixed at 512 samples
# (32ms @ 16kHz) regardless of that config value, so using it here silently
# drifted MAX_RECORD_SECONDS off its configured value. CHUNK_DURATION_MS is
# kept only as a documented constant; see sensory.yaml comment.

# parec command — captures at 16kHz mono float32, uses default PulseAudio source
_PAREC_CMD = [
    "parec",
    "--rate=16000",
    "--channels=1",
    "--format=float32le",
    "--latency-msec=30",
]


# ── native gate flags (formerly sensory/voice_gates.py, then listen_native.py) ─
# Master switches, read live from env (not cached at import) and enforced
# directly inside AikoListen methods below — no external bind step.

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def barge_in_enabled() -> bool:
    """Master barge-in switch (browser + Jetson)."""
    return _env_bool("BARGE_IN_ENABLED", "0")


def barge_in_always_on() -> bool:
    """Local Silero monitor outside TTS wait — only meaningful if
    barge_in_enabled() is also True."""
    return barge_in_enabled() and _env_bool("BARGE_IN_ALWAYS_ON", "0")


def speaker_verify_gate() -> bool:
    """When True (and verification is active), a failed cosine match drops
    the utterance in listen(). When False, the score is metadata only."""
    return _env_bool("SPEAKER_VERIFY_GATE", "0")


# ── post-ASR name / phrase corrections (S2, no finetune) ────────────────────
# Pipe-separated "heard->fixed" pairs via ASR_CORRECTIONS. Applied after
# SenseVoice (longest match first). Built-in defaults cover Aiko / OppaAI
# mangling; user map is applied on top and wins on same key.

_DEFAULT_ASR_PAIRS: tuple[tuple[str, str], ...] = (
    ("hey aiko", "hey Aiko"),
    ("hey iko", "hey Aiko"),
    ("hey eco", "hey Aiko"),
    ("hey ecko", "hey Aiko"),
    ("hey echo", "hey Aiko"),
    ("hey ico", "hey Aiko"),
    ("hey aico", "hey Aiko"),
    ("hi aiko", "hi Aiko"),
    ("hi iko", "hi Aiko"),
    ("aiko", "Aiko"),
    ("oppaai", "OppaAI"),
    ("oppa ai", "OppaAI"),
    ("op ai", "OppaAI"),
    ("oppa a i", "OppaAI"),
    ("opper ai", "OppaAI"),
    ("opa ai", "OppaAI"),
)


def _parse_asr_user_map(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in (raw or "").split("|"):
        part = part.strip()
        if not part or "->" not in part:
            continue
        src, dst = part.split("->", 1)
        src, dst = src.strip(), dst.strip()
        if src and dst:
            out.append((src.lower(), dst))
    return out


@lru_cache(maxsize=4)
def _pairs_cached(user_raw: str) -> tuple[tuple[str, str], ...]:
    user = _parse_asr_user_map(user_raw)
    # User entries first so they override defaults when sources collide
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    for src, dst in list(user) + list(_DEFAULT_ASR_PAIRS):
        key = src.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append((src, dst))
    # Longest source first so "hey aiko" wins over "aiko"
    merged.sort(key=lambda p: len(p[0]), reverse=True)
    return tuple(merged)


def correction_pairs() -> tuple[tuple[str, str], ...]:
    return _pairs_cached(os.getenv("ASR_CORRECTIONS", "").strip())


def correct_asr_text(text: str) -> str:
    """Apply name/phrase corrections; preserves non-matched regions."""
    if not text or not text.strip():
        return text
    out = text
    for src, dst in correction_pairs():
        # Word-boundary-ish match, case-insensitive
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
        out = pattern.sub(dst, out)
    return out


def _resolve_sense_voice_files() -> tuple[str, str]:
    """
    Resolve SenseVoice model + tokens from HF cache.
    Downloads on first use; idempotent thereafter.
    Set HF_HUB_OFFLINE=1 to prevent network access and serve from cache only.
    Override the repo with ASR_MODEL in .env to swap models without code changes.
    """
    model_path  = hf_hub_download(repo_id=ASR_MODEL, filename="model.int8.onnx")
    tokens_path = hf_hub_download(repo_id=ASR_MODEL, filename="tokens.txt")
    return model_path, tokens_path


def _load_sense_voice_recognizer() -> sherpa_onnx.OfflineRecognizer:
    """Load SenseVoice as a sherpa-onnx OfflineRecognizer via factory method."""
    model_path, tokens_path = _resolve_sense_voice_files()

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        language=ASR_LANGUAGE,
        use_itn=True,
        num_threads=ASR_NUM_THREADS,
        provider=ASR_DEVICE,
        debug=False,
    )


# ── wake word / trigger phrase helpers ───────────────────────────────────────

def _ratio(a: str, b: str) -> float:
    """Fuzzy string similarity, 0-100. rapidfuzz if available, else difflib."""
    if _fuzz is not None:
        return _fuzz.ratio(a, b)
    return _difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _strip_prefix_phrase(text: str, candidates: list[str], threshold: float) -> tuple[bool, str]:
    """
    Fuzzy-match any of `candidates` against the leading words of `text` and,
    on a hit, strip the matched prefix off. ASR is unreliable on words like
    "Aiko" (not standard English), so this checks a small window of
    word-counts around each candidate's length rather than requiring an
    exact substring. Only the front of the utterance is checked — wake
    words / trigger phrases are always said first, never buried mid-sentence.

    Returns (matched, remainder_text). remainder_text == text unchanged if
    no match was found.
    """
    words = text.split()
    if not words or not candidates:
        return False, text

    best_score, best_end, best_phrase = 0.0, 0, ""
    for phrase in candidates:
        phrase_words = phrase.split()
        n = len(phrase_words)
        if n == 0:
            continue
        for span in range(max(1, n - 1), min(len(words), n + 2) + 1):
            window = " ".join(words[:span])
            score = _ratio(window, phrase)
            if score > best_score:
                best_score, best_end, best_phrase = score, span, phrase

    log.info("[wake] text=%r  best_phrase=%r  best_score=%.1f  threshold=%.1f  matched=%s",
             text, best_phrase, best_score, threshold, best_score >= threshold)
    if best_score >= threshold:
        return True, " ".join(words[best_end:]).strip()
    return False, text


# ── listen ────────────────────────────────────────────────────────────────────

class AikoListen:
    """
    Microphone capture + SenseVoice ASR transcription (+ optional speaker
    verification against one enrolled voice, + optional wake word / trigger
    phrase gating).
    Uses parec (PulseAudio) for mic capture — no PortAudio/sounddevice.
    Silero VAD gates recording for robust, noise-resilient speech detection,
    for every audio source (local mic and WebUI alike).

    Barge-in, the speaker-verify drop-gate, and post-ASR corrections are
    native (see module docstring) — trigger_barge_in(), _barge_in_loop(),
    listen(), and _transcribe() all enforce their respective switches
    directly, with no external bind step required.

    When chunk_source is provided (WebUI path), the browser has already
    applied a lightweight energy-RMS gate client-side (see static/vad.js) —
    this is only a "loud enough to send" filter, not a speech/silence
    decision. Silero scores every chunk that arrives via chunk_source
    exactly as it does for the local-mic path — no separate flag needed.

    Staged init:
        listen = AikoListen()    # no heavy loading
        listen.load_asr()        # loads the SenseVoice model
        listen.load_vad()        # loads Silero VAD + kicks off warmup thread
        listen.load_speaker_id() # loads embedding model + enrolled vector (no-op if disabled)
        listen.join_warmup()     # blocks until warmup completes

    Barge-in monitor (call after join_warmup):
        listen.start_barge_in_monitor()
        Pauses automatically while _record() is active to avoid mic conflicts.
        No-ops (idles until stopped) when BARGE_IN_ENABLED=0.

    Wake word / trigger phrase gating (see module docstring for config):
        listen.is_active()   — True if currently awake/triggered
        listen.sleep_now()   — force back to asleep (e.g. explicit command)
    """

    def __init__(self) -> None:
        self._model:      sherpa_onnx.OfflineRecognizer | None = None
        self._vad_model:  object | None       = None
        self._lock        = threading.Lock()
        self._warmup_done = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        self._barge_in_event:  threading.Event = threading.Event()
        self._barge_in_armed:  threading.Event = threading.Event()
        self._barge_in_active: bool             = False
        self._barge_in_thread: threading.Thread | None = None

        # set while _record() is running — pauses barge-in to avoid mic conflict
        self._recording = threading.Event()

        # speaker verification — None if disabled or model missing
        self._speaker_extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None
        self._enrolled_embedding: np.ndarray | None = None
        self._speaker_lock = threading.Lock()

        # wake word / trigger phrase activation session — 0 / expired means
        # "asleep", i.e. the configured phrase(s) must be said again.
        self._activation_lock = threading.Lock()
        self._active_until: float = 0.0

    # ── staged init ───────────────────────────────────────────────────────────

    def load_asr(self) -> None:
        self._model = _load_sense_voice_recognizer()

    def load_vad(self) -> None:
        self._vad_model = load_silero_vad(onnx=True)
        # self._vad_model.eval()  # PyTorch-only, not needed for OnnxWrapper
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    @staticmethod
    def speaker_enroll_path() -> str:
        """Resolve fresh at call time — NOT cached at import, since import
        happens at boot before any user is authenticated (current_user_id()
        would return 'guest' at that point)."""
        return str(user_state_path("profile/speaker_enrollment.json"))

    def load_speaker_id(self) -> None:
        """
        Load the speaker embedding model + enrolled embedding, if speaker
        verification is enabled. Silently no-ops (verification stays off)
        if disabled, the model path is missing, or no enrollment exists yet
        — listen() always falls back to speaker=None in that case, it never
        raises, so a missing enrollment can't break normal listening.
        """
        if not SPEAKER_VERIFY_ENABLED:
            return
        if not SPEAKER_MODEL_PATH or not os.path.isfile(SPEAKER_MODEL_PATH):
            log.warning(
                f"[listen] SPEAKER_VERIFY_ENABLED=1 but SPEAKER_MODEL_PATH "
                f"is missing or invalid ({SPEAKER_MODEL_PATH!r}); verification disabled."
            )
            return
        enroll_path = self.speaker_enroll_path()
        if not os.path.isfile(enroll_path):
            log.warning(
                f"[listen] SPEAKER_VERIFY_ENABLED=1 but no enrollment found at "
                f"{enroll_path!r}; run enroll_speaker.py first. Verification disabled."
            )
            return

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SPEAKER_MODEL_PATH,
            num_threads=SPEAKER_NUM_THREADS,
            debug=False,
            provider=ASR_DEVICE,
        )
        self._speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

        with open(enroll_path) as f:
            data = json.load(f)
        self._enrolled_embedding = np.asarray(data["embedding"], dtype=np.float32)

    def join_warmup(self) -> None:
        self._warmup_done.wait()

    # ── speaker verification ──────────────────────────────────────────────────

    def speaker_verify_active(self) -> bool:
        """True if speaker verification is loaded and ready to run."""
        return self._speaker_extractor is not None and self._enrolled_embedding is not None

    def _compute_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Compute a speaker embedding for a float32 16kHz audio buffer."""
        stream = self._speaker_extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        embedding = self._speaker_extractor.compute(stream)
        return np.asarray(embedding, dtype=np.float32)

    def _verify_speaker(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Compare audio against the enrolled embedding via cosine similarity.
        Returns (is_match, score). Thread-safe — extractor sessions aren't
        guaranteed reentrant, so this is serialized alongside _transcribe().
        """
        with self._speaker_lock:
            embedding = self._compute_embedding(audio)
        a, b = embedding, self._enrolled_embedding
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        score = float(np.dot(a, b) / denom)
        return score >= SPEAKER_VERIFY_THRESHOLD, score

    # ── wake word / trigger phrase activation gate ───────────────────────────

    def gate_enabled(self) -> bool:
        """True if wake word gating is configured. Useful for UI to show
        a sleep/wake indicator only when the feature is actually in use."""
        return bool(WAKE_WORD)

    def is_active(self) -> bool:
        """
        True if Aiko is currently awake/triggered (no phrase currently
        required). Always True if gating isn't configured. Other subsystems
        — e.g. proactive/unsolicited engagement — should check this before
        acting, since once the session idles out, proactive behavior should
        stay quiet until the wake word / trigger phrase is said again.
        """
        if not self.gate_enabled():
            return True
        with self._activation_lock:
            return self._active_until > time.monotonic()

    def sleep_now(self) -> None:
        """Force the activation session inactive immediately — e.g. for an
        explicit 'go to sleep' voice command."""
        with self._activation_lock:
            self._active_until = 0.0

    def extend_activation(self) -> None:
        """Extend the wake-word-free session window."""
        with self._activation_lock:
            self._active_until = time.monotonic() + ACTIVATION_TIMEOUT_S

    _extend_activation = extend_activation  # compat alias

    def _apply_activation_gate(self, text: str, verified: bool | None) -> tuple[str | None, dict]:
        """
        Enforce wake-word gating on a freshly transcribed utterance.

        Returns (command_text, gate_info):
          - command_text is None  → gate failed; caller must
            silently drop the utterance (no response, no side effects)
          - command_text is text  w/ any matched wake word prefix stripped off
            (unchanged if gating isn't configured, or the session was
            already active so no phrase check was needed)

        gate_info = {"woke": bool|None}:
          None means gate wasn't configured / not evaluated this call
          (e.g. session was already active). Useful for logging / UI state.
        """
        if not bool(WAKE_WORD):
            return text, {"woke": None}

        if self.is_active():
            self._extend_activation()
            return text, {"woke": None}

        matched, remainder = _strip_prefix_phrase(
            text, [WAKE_WORD, *WAKE_WORD_ALIASES], WAKE_FUZZY_THRESHOLD
        )
        if not matched:
            log.debug("[gate] wake word %r NOT matched in %r — dropping", WAKE_WORD, text)
            return None, {"woke": False}

        self._extend_activation()
        return remainder.strip(), {"woke": True}

    # ── barge-in monitor ──────────────────────────────────────────────────────

    def start_barge_in_monitor(self) -> None:
        if self._barge_in_active:
            return
        self._barge_in_active = True
        self._barge_in_thread = threading.Thread(
            target=self._barge_in_loop, daemon=True,
        )
        self._barge_in_thread.start()

    def stop_barge_in_monitor(self) -> None:
        self._barge_in_active = False

    def trigger_barge_in(self) -> None:
        """
        Externally signal a barge-in, bypassing the local-mic Silero monitor.
        Used by the WebUI path: the browser's own energy-VAD detects speech
        during TTS playback and reports it over the websocket as a 'barge_in'
        message — this lets that message interrupt speak.wait_or_barge_in()
        exactly as if the physical Jetson mic had detected it.

        No-op when BARGE_IN_ENABLED=0 (master switch — see barge_in_enabled()).
        """
        if not barge_in_enabled():
            return
        self._barge_in_event.set()

    def _barge_in_loop(self) -> None:
        """
        Always-on VAD monitor via parec. Pauses while _record() is active.

        No-ops entirely (idles until stop_barge_in_monitor()) when
        BARGE_IN_ENABLED=0 — this is the master switch; no parec process is
        even spawned in that case.
        """
        if not barge_in_enabled():
            while self._barge_in_active:
                time.sleep(0.5)
            return

        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4

        try:
            proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)
            consecutive = 0
            paused = False
            while self._barge_in_active:
                if self._recording.is_set() or (not barge_in_always_on() and not self._barge_in_armed.is_set()):
                    time.sleep(0.05)
                    consecutive = 0
                    paused = True
                    continue

                if paused:
                    # parec kept writing into the pipe the whole time we
                    # weren't reading — discard the backlog so the next
                    # score is against live audio, not ~1s of stale buffer.
                    _drain_stale_audio(proc.stdout, bytes_per_chunk)
                    paused = False

                raw = proc.stdout.read(bytes_per_chunk)
                if len(raw) < bytes_per_chunk:
                    break

                if self._barge_in_event.is_set():
                    consecutive = 0
                    continue

                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                score = self._score_chunk(chunk)

                if score >= BARGE_IN_THRESHOLD:
                    consecutive += 1
                    if consecutive >= BARGE_IN_CONFIRM:
                        self._barge_in_event.set()
                        consecutive = 0
                        threading.Timer(
                            BARGE_IN_COOLDOWN_MS / 1000.0,
                            self._barge_in_event.clear,
                        ).start()
                else:
                    consecutive = 0
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning(f"Barge-in monitor died: {exc}")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                log.warning("listen: failed to terminate barge-in monitor process")

    # ── public api ────────────────────────────────────────────────────────────

    def listen(
        self,
        status_callback=None,
        wait_fn=None,
        speak=None,
        chunk_source=None,
    ) -> tuple[str, dict]:
        """
        Returns (text, info). info always has a "verified" key:
          - None  if speaker verification is disabled / not loaded
          - True  if the buffered audio matched the enrolled voice
          - False if it didn't match
        info also carries "speaker_score" (float or None) for logging/tuning.
        Verification never blocks or fails transcription — it's metadata
        attached alongside the text. If SPEAKER_VERIFY_GATE=1 (see
        speaker_verify_gate()) and the match failed, the utterance is
        dropped and ("", info) is returned instead — same shape as "no
        speech detected".

        info additionally carries "woke" (bool|None):
          - None means wake word gate wasn't configured / not evaluated
            this call (e.g. the session was already active, so no phrase
            check ran)
          - If wake word gating IS configured and the required phrase was
            not detected, this method returns ("", info) — same shape as
            "no speech detected" — so callers that already treat empty text
            as "nothing to do" handle this for free. Any matched wake word
            prefix is stripped from the returned text.

        If speak is playing and BARGE_IN_ENABLED=0, this blocks with a plain
        poll loop until playback finishes (no barge interrupt is possible —
        trigger_barge_in()/_barge_in_loop() are no-ops in that state anyway).
        If BARGE_IN_ENABLED=1, it waits via speak.wait_or_barge_in() so a
        detected barge can interrupt the wait early.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None,
            forwarded to _record(). See _record() docstring. None (default)
            preserves the existing local-mic (parec) behavior. Silero scores
            every chunk regardless of source — including chunks the browser
            has already pre-filtered client-side (static/vad.js) — so there
            is no separate flag needed for that path.
        """
        if speak is not None and speak.is_playing():
            if not barge_in_enabled():
                _cb(status_callback, "__WAITING__")
                while speak.is_playing():
                    time.sleep(0.05)
                speak = None
                wait_fn = None  # already waited out TTS; avoid double wait_fn
            else:
                _cb(status_callback, "__WAITING__")
                self._barge_in_armed.set()
                try:
                    interrupted = speak.wait_or_barge_in(self._barge_in_event)
                finally:
                    self._barge_in_armed.clear()
                if interrupted:
                    self._barge_in_event.clear()
        elif wait_fn is not None:
            wait_fn()

        _cb(status_callback, "__LISTENING__")
        listen_started_at = time.monotonic()
        audio = self._record(
            status_callback,
            chunk_source=chunk_source,
        )
        recording_stopped_at = time.monotonic()
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return "", {
                "verified": None,
                "speaker_score": None,
                "woke": None,
                "listen_started_at": listen_started_at,
                "recording_stopped_at": recording_stopped_at,
            }

        _cb(status_callback, "__TRANSCRIBING__")

        info = {
            "verified": None,
            "speaker_score": None,
            "woke": None,
            "listen_started_at": listen_started_at,
            "recording_stopped_at": recording_stopped_at,
        }
        if self.speaker_verify_active():
            result_box: dict = {}

            def _run_verify():
                result_box["verified"], result_box["speaker_score"] = self._verify_speaker(audio)

            verify_thread = threading.Thread(target=_run_verify, daemon=True)
            verify_thread.start()
            text = self._transcribe(audio)
            verify_thread.join()
            info["verified"]      = result_box.get("verified")
            info["speaker_score"] = result_box.get("speaker_score")
        else:
            text = self._transcribe(audio)

        gated_text, gate_info = self._apply_activation_gate(text, info.get("verified"))
        info["woke"] = gate_info["woke"]

        _cb(status_callback, "__IDLE__")

        if (
            speaker_verify_gate()
            and self.speaker_verify_active()
            and info.get("verified") is False
        ):
            log.info(
                "[gate] speaker verify failed (score=%s) — dropping utterance",
                info.get("speaker_score"),
            )
            return "", info

        if gated_text is None:
            return "", info
        return gated_text, info

    def listen_async(self, on_result, status_callback=None) -> threading.Thread:
        """on_result receives (text, info) — same shape as listen()'s return."""
        def _run():
            text, info = self.listen(status_callback=status_callback)
            on_result(text, info)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    # ── recording ─────────────────────────────────────────────────────────────

    def _score_chunk(self, chunk: np.ndarray) -> float:
        """
        Run Silero VAD on a 512-sample float32 chunk at 16kHz.

        NOTE (audit item #6): load_silero_vad(onnx=True) returns an
        ONNX-backed wrapper — inference itself runs through onnxruntime,
        not torch — but the `silero-vad` pip package's wrapper still expects
        a torch.Tensor as input and manages its own recurrent state as torch
        tensors internally, so `torch` stays a hard dependency here (the
        `no_grad()` context below is a genuine no-op for an onnxruntime call,
        kept only because removing torch as a dependency requires it).
        Fully dropping torch means moving this VAD onto sherpa_onnx's own
        VoiceActivityDetector (same Silero ONNX weights, pure onnxruntime,
        no torch) — see BOOT_LABELS module docstring discussion. That's a
        state-machine-level change: sherpa_onnx's VAD is segment-oriented
        (accept_waveform + is_speech_detected()/front()), not a per-chunk
        probability score, so _record()'s and _barge_in_loop()'s hangover /
        confirm-frame counters would need to be rewritten against that API
        rather than swapped in place. Left as a follow-up requiring hardware
        testing on AuRoRA rather than folded into this bug-fix pass.
        """
        if len(chunk) < _CHUNK_SAMPLES_VAD:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES_VAD - len(chunk)))
        else:
            chunk = chunk[:_CHUNK_SAMPLES_VAD]

        tensor = torch.from_numpy(chunk.copy()).unsqueeze(0)
        with torch.no_grad():
            prob = self._vad_model(tensor, SAMPLE_RATE).item()
        return prob

    def _record(
        self,
        status_callback=None,
        chunk_source=None,
    ) -> np.ndarray | None:
        """
        Capture audio until silence after speech detected. Silero VAD scores
        every chunk here, regardless of source — it is the single
        authoritative speech/silence gate.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None.
            If None (default), audio is captured locally via parec — this is
            the path used by the robot/TUI, unchanged.
            If provided, that callable is polled instead of parec — used by
            the WebUI to feed mic audio streamed in from the browser over the
            WebSocket. Must return exactly `bytes_per_chunk` bytes of
            float32LE PCM, or None to signal end-of-stream (e.g. the browser
            energy-VAD sentinel b"" was received, or client disconnected).
            Chunks arriving this way have already passed the browser's
            client-side energy-RMS gate (static/vad.js) — a coarse "loud
            enough to send" filter, not a speech/silence decision — so
            Silero still scores every chunk exactly as it does for the
            local-mic path below.
        """
        audio_chunks   = []
        silence_count  = 0
        speech_count   = 0
        hearing_speech = False
        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4

        _cb(status_callback, "__LISTENING__")
        self._recording.set()

        proc = None
        use_external = chunk_source is not None

        try:
            if not use_external:
                proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)

            for _ in range(_MAX_CHUNKS):
                if use_external:
                    raw = chunk_source(bytes_per_chunk)
                else:
                    raw = proc.stdout.read(bytes_per_chunk)

                if raw is None or len(raw) < bytes_per_chunk:
                    # None  → browser end-of-utterance sentinel (b"") or timeout
                    # short → parec pipe closed / underrun
                    break

                chunk = np.frombuffer(raw, dtype=np.float32).copy()

                # Silero scores every chunk from every source — the browser's
                # energy gate (for the WebUI chunk_source path) only decided
                # whether to forward the chunk at all, not whether it's speech.
                is_speech = self._score_chunk(chunk) >= VAD_THRESHOLD

                if is_speech:
                    hearing_speech = True
                    silence_count  = 0
                    speech_count  += 1
                    audio_chunks.append(chunk)
                else:
                    if hearing_speech:
                        silence_count += 1
                        audio_chunks.append(chunk)
                        if silence_count >= SILENCE_CHUNKS:
                            break

        except Exception:
            _cb(status_callback, "__IDLE__")
            return None
        finally:
            self._recording.clear()
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    log.warning("listen: failed to terminate record process")

        if not audio_chunks:
            return None

        # ── utterance length gate ─────────────────────────────────────────────
        # Silero has genuinely scored every chunk regardless of source, so
        # speech_count reflects real detected speech — no reason to bypass
        # this for the WebUI path anymore.
        if speech_count < MIN_SPEECH_CHUNKS:
            return None

        return np.concatenate(audio_chunks).astype(np.float32)

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe float32 16kHz audio using SenseVoice via sherpa-onnx,
        then apply post-ASR name/phrase corrections (see correct_asr_text).
        """
        with self._lock:
            stream = self._model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3
            result = stream.result
            text   = result.text.strip()
            # SenseVoice prepends language/emotion tags like <|en|><|NEUTRAL|><|Speech|><|withitn|>
            # Strip them for clean output
            text = re.sub(r'<\|[^|]+\|>', '', text).strip()

        try:
            return correct_asr_text(text)
        except Exception:
            return text

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            stream  = self._model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, silence)
            self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3
            tensor = torch.zeros(1, _CHUNK_SAMPLES_VAD)
            with torch.no_grad():
                self._vad_model(tensor, SAMPLE_RATE)
        except Exception:
            log.warning("listen: warmup failed")
        finally:
            self._warmup_done.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _drain_stale_audio(stream, chunk_size: int, max_iters: int = 64) -> None:
    """
    Non-blocking discard of any backlog sitting in a pipe's OS buffer.

    parec keeps writing continuously even while _barge_in_loop stops reading
    (e.g. while paused for _record() or while disarmed) — on typical Linux
    pipe buffers (~64KB) that's roughly a second of stale audio. Call this
    right before resuming reads so the first score after a pause is against
    live audio, not backlog.
    """
    fd = stream.fileno()
    for _ in range(max_iters):
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            break
        try:
            chunk = os.read(fd, chunk_size)
        except OSError:
            break
        if not chunk:
            break


def _cb(callback, msg: str) -> None:
    if callback:
        try:
            callback(msg)
        except Exception:
            log.warning("listen: callback raised")
