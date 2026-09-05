"""
sensory/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio; Silero VAD (neural, pure onnxruntime via
    sherpa_onnx — no torch, see "VAD backend" below) is the single
    authoritative speech/silence gate for ALL audio sources, local mic or
    WebUI.
  - For the WebUI path, the browser only runs a lightweight energy-RMS gate
    client-side (see static/vad.js) to decide "loud enough to bother sending" —
    it is NOT a speech/silence judgment. Silero here is what actually decides
    what is speech, on every chunk, regardless of source.
  - Transcribes via SenseVoice (sherpa-onnx, int8 ONNX) in a background thread,
    then applies post-ASR name/phrase corrections (see correct_asr_text below)
  - Optionally verifies the speaker against one enrolled voice embedding
    (sherpa-onnx SpeakerEmbeddingExtractor) on the same buffered audio, run
    in parallel with transcription — see SPEAKER_VERIFY_ENABLED below
  - Optionally gates responses behind a wake word ("Hey Aiko"), either
    acoustically (livekit-wakeword, real-time, no torch) or via fuzzy
    post-ASR text match (legacy fallback) — see "Wake word" below
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_asr() → load_vad() → load_wakeword() → load_speaker_id()
    → join_warmup() for granular boot progress reporting via wakeup.py
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

VAD backend (torch removed):
    Previously load_vad() used the `silero-vad` pip package
    (load_silero_vad(onnx=True)) — its inference ran through onnxruntime,
    but its Python wrapper still required `torch.Tensor` in/out and managed
    the model's recurrent state as torch tensors, so `torch` (~1GB+ install)
    stayed a hard dependency purely for tensor plumbing, not compute.
    VAD now runs through sherpa_onnx.VoiceActivityDetector instead — same
    Silero ONNX weights, pure onnxruntime, zero torch. That API is
    segment-oriented (accept_waveform() + is_speech_detected()) rather than
    a bare per-chunk probability, so _score_chunk() has been replaced with
    _is_speech() (see there for the two-instance-vs-two-threshold note) and
    _record()/_barge_in_loop() were adjusted to call it. Torch is no longer
    imported anywhere in this module.

Wake word (see config/sensory.yaml):
    WAKE_WORD ("" by default, disabled) turns wake-word gating on. Two
    engines are supported, chosen automatically:

    1. Acoustic (preferred) — set WAKE_WORD_MODEL_PATH to a trained
       livekit-wakeword ONNX classifier (see util/train_wakeword.py or the
       livekit-wakeword CLI: https://docs.livekit.io/agents/multimodality/audio/wakeword/).
       Detection runs frame-by-frame on raw audio *during* _record(), before
       any ASR — so unlike the old approach, SenseVoice never has to run
       just to check whether the wake word was said, and "asleep" utterances
       never get transcribed at all. Requires `pip install livekit-wakeword`
       (numpy + onnxruntime only, no torch) and a trained model — there is
       no pre-trained "hey Aiko" model, you must train one.
    2. Fuzzy ASR-text (legacy fallback) — used automatically if
       WAKE_WORD_MODEL_PATH is unset, the file is missing, or livekit-wakeword
       isn't installed. SenseVoice mangles "Aiko" unpredictably since it's
       not a normal English word, so matching is fuzzy (rapidfuzz ratio)
       against the leading words of the *transcribed* text, not an exact
       substring check — see _strip_prefix_phrase() / _apply_activation_gate().
       WAKE_WORD_ALIASES lets you hardcode observed mishearings
       ("hey iko|hey eco|hey ecko") as extra candidates. This engine still
       runs full ASR on every utterance while asleep, same as before.

    Once woken/triggered (either engine), Aiko stays "active" (no phrase
    required) until ACTIVATION_TIMEOUT_S seconds pass with no further
    utterance, at which point the session goes back to sleep and the
    configured phrase(s) are required again. Use AikoListen.is_active() to
    check this from other subsystems (e.g. suppress proactive/unsolicited
    behavior while asleep), and AikoListen.sleep_now() to force it inactive
    (e.g. an explicit "go to sleep" command).

Dependencies:
    pip install sherpa-onnx numpy huggingface_hub rapidfuzz
    pip install livekit-wakeword   # optional — acoustic wake word engine
    pip install sounddevice        # microphone capture (PortAudio backend)
    Models: SenseVoice + Silero VAD auto-downloaded to HF cache on first use
    (see ASR_MODEL in .env; the VAD weights come from csukuangfj/vad on HF).
    sounddevice (PortAudio) for mic capture — no parec/PulseAudio required.
    rapidfuzz is optional — falls back to stdlib difflib if not installed,
    just slower (only exercised by the fuzzy wake-word fallback now, not
    the VAD path).

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

from functools import lru_cache
from system.config import env_float, env_int
from system.userspace import current_user_id, user_state_path
import json
import logging as _logging
import numpy as np
import os
import re

log = _logging.getLogger(__name__)

# Heavy voice runtime (onnxruntime CUDA libs ~160MB, sherpa_onnx, HF hub)
# loads lazily via _ensure_runtime() on first model load — importing this
# module (e.g. --text mode, boot labels) must stay cheap.
_ort = None               # onnxruntime module, once loaded
sherpa_onnx = None        # sherpa_onnx module, once loaded
hf_hub_download = None    # huggingface_hub downloader, once loaded


def _ensure_runtime() -> None:
    """Import the voice runtime on first model load. Idempotent."""
    global _ort, sherpa_onnx, hf_hub_download
    if sherpa_onnx is not None and hf_hub_download is not None:
        return
    import onnxruntime as _ort_mod
    if hasattr(_ort_mod, "set_default_logger_severity"):
        _ort_mod.set_default_logger_severity(3)
    from huggingface_hub import hf_hub_download as _hfd
    import sherpa_onnx as _sh
    _logging.getLogger("sherpa_onnx").setLevel(_logging.ERROR)
    _ort, sherpa_onnx, hf_hub_download = _ort_mod, _sh, _hfd


import threading
import time
import warnings

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None
    import difflib as _difflib

try:
    from livekit.wakeword import WakeWordModel as _WakeWordModel
except ImportError:
    _WakeWordModel = None  # acoustic wake engine unavailable — falls back to fuzzy ASR-text matching

warnings.filterwarnings("ignore")

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'listen_asr':     'Loading SenseVoice ASR model...',
    'listen_silero':  'Loading Silero VAD...',
    'listen_wake':    'Loading wake word model...',
    'listen_speaker': 'Loading speaker verification...',
    'listen_warmup':  'Warming up ASR pipeline...',
    'listen_ready':   'Microphone ready',
    'listen_skip':    'ASR skipped (text mode)',
}

# ── config ────────────────────────────────────────────────────────────────────

ASR_DEVICE      = os.getenv("ASR_DEVICE", "cpu")       # resolved from config/sensory.yaml via load_config()
ASR_LANGUAGE    = os.getenv("ASR_LANGUAGE", "auto")    # auto, zh, en, ja, ko, yue, nospeech
ASR_NUM_THREADS = env_int("ASR_NUM_THREADS", 4)
LISTEN_DEVICE   = env_int("LISTEN_DEVICE", -1)  # sounddevice input index; -1 = default

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

CHUNK_DURATION_MS   = env_int("LISTEN_CHUNK_MS",         30)  # Silero minimum
VAD_THRESHOLD       = env_float("LISTEN_VAD_THRESHOLD", 0.5)  # Silero speech prob cutoff
SILENCE_CHUNKS      = env_int("LISTEN_SILENCE_CHUNKS",   66)  # matches config/sensory.yaml default
MIN_SPEECH_CHUNKS   = env_int("LISTEN_MIN_CHUNKS",       10)
MAX_RECORD_SECONDS  = env_int("LISTEN_MAX_SECONDS",      30)

BARGE_IN_THRESHOLD     = env_float("BARGE_IN_THRESHOLD",     0.95)  # matches config/sensory.yaml default
BARGE_IN_CONFIRM       = env_int("BARGE_IN_CONFIRM_CHUNKS",  4)     # matches config/sensory.yaml default
BARGE_IN_COOLDOWN_MS   = env_int("BARGE_IN_COOLDOWN_MS",     800)
# BARGE_IN_ALWAYS_ON is intentionally NOT cached here — see barge_in_always_on()
# below. It must be read live, like BARGE_IN_ENABLED and SPEAKER_VERIFY_GATE,
# so it can be toggled at runtime without a process restart.

# ── speaker verification config ──────────────────────────────────────────────
# Single-enrollment 1:1 verification (not multi-speaker identification) —
# Aiko has exactly one "owner" voice to check against.

SPEAKER_VERIFY_ENABLED   = os.getenv("SPEAKER_VERIFY_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
SPEAKER_MODEL_PATH       = os.path.expanduser(os.getenv("SPEAKER_MODEL_PATH", ""))            # path to embedding .onnx
SPEAKER_VERIFY_THRESHOLD = env_float("SPEAKER_VERIFY_THRESHOLD", 0.5)  # cosine sim cutoff
SPEAKER_NUM_THREADS      = env_int("SPEAKER_NUM_THREADS", 1)

# ── wake word / trigger phrase config ────────────────────────────────────────
# WAKE_WORD: "" disables wake-word gating entirely (Aiko responds to every
#   utterance, as before). When set, ASR is unreliable on "Aiko" (not a
#   normal English word) so matching is fuzzy, not exact-substring.
WAKE_WORD             = os.getenv("WAKE_WORD", "").strip().lower()
WAKE_WORD_ALIASES     = [w.strip().lower() for w in os.getenv("WAKE_WORD_ALIASES", "").split("|") if w.strip()]
WAKE_FUZZY_THRESHOLD  = env_float("WAKE_FUZZY_THRESHOLD", 70)

# Acoustic wake engine (livekit-wakeword) — used instead of the fuzzy
# ASR-text engine above when a trained model is configured and available.
# See module docstring "Wake word" section.
WAKE_WORD_MODEL_PATH     = os.path.expanduser(os.getenv("WAKE_WORD_MODEL_PATH", ""))
WAKE_WORD_THRESHOLD      = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
WAKE_WORD_CONFIRM_FRAMES = int(os.getenv("WAKE_WORD_CONFIRM_FRAMES", "3"))

ACTIVATION_TIMEOUT_S = float(os.getenv("ACTIVATION_TIMEOUT_S", "3600"))  # matches config/sensory.yaml default

_CHUNK_SAMPLES_VAD = 512                                             # at 16 kHz, ~32 ms
_CHUNK_MS_ACTUAL   = (_CHUNK_SAMPLES_VAD / SAMPLE_RATE) * 1000.0      # 32.0 ms — the real, non-configurable chunk size
_MAX_CHUNKS        = int(MAX_RECORD_SECONDS * 1000 / _CHUNK_MS_ACTUAL)
# NOTE: CHUNK_DURATION_MS (LISTEN_CHUNK_MS in sensory.yaml) is NOT used to
# compute _MAX_CHUNKS anymore — Silero's chunk size is fixed at 512 samples
# (32ms @ 16kHz) regardless of that config value, so using it here silently
# drifted MAX_RECORD_SECONDS off its configured value. CHUNK_DURATION_MS is
# kept only as a documented constant; see sensory.yaml comment.


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
        # dst is passed through a replacer function, not directly to
        # pattern.sub() — re.sub() interprets backslash sequences (\1,
        # \g<name>) in a raw replacement string, so a user-configured
        # ASR_CORRECTIONS pair like "foo->\1bar" would silently be treated
        # as a backreference instead of literal text. The lambda makes dst
        # a plain literal replacement regardless of its contents.
        out = pattern.sub(lambda _m, _dst=dst: _dst, out)
    return out


def _resolve_sense_voice_files() -> tuple[str, str]:
    """
    Resolve SenseVoice model + tokens from HF cache.
    Downloads on first use; idempotent thereafter.
    Set HF_HUB_OFFLINE=1 to prevent network access and serve from cache only.
    Override the repo with ASR_MODEL in .env to swap models without code changes.
    """
    _ensure_runtime()
    model_path  = hf_hub_download(repo_id=ASR_MODEL, filename="model.int8.onnx")
    tokens_path = hf_hub_download(repo_id=ASR_MODEL, filename="tokens.txt")
    return model_path, tokens_path


def _resolve_silero_vad_file() -> str:
    """
    Resolve the standalone Silero VAD ONNX weights from HF cache (downloads
    on first use, idempotent thereafter — same pattern as
    _resolve_sense_voice_files()). This is the raw onnxruntime-only weight
    file (no torch involved anywhere in obtaining or running it), mirrored
    by the sherpa-onnx maintainer at csukuangfj/vad on the HF hub — the same
    file the sherpa-onnx project's own README points at via a GitHub release
    asset (github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx).
    """
    _ensure_runtime()
    return hf_hub_download(repo_id="csukuangfj/vad", filename="silero_vad.onnx")


def _build_vad(threshold: float) -> sherpa_onnx.VoiceActivityDetector:
    """
    Build a sherpa_onnx.VoiceActivityDetector (pure onnxruntime, no torch)
    at a given speech-probability threshold. See the "VAD backend" section
    of the module docstring and the comment on AikoListen._vad_model /
    _barge_vad_model for why threshold is baked in per-instance rather than
    passed per-call.

    min_silence_duration / min_speech_duration are set to one chunk
    (_CHUNK_MS_ACTUAL, ~32ms) rather than sherpa_onnx's own defaults
    (0.5s / 0.25s) — the old torch-based _score_chunk() returned a raw
    per-chunk probability with no built-in hangover/smoothing at all, and
    _record()'s SILENCE_CHUNKS / _barge_in_loop()'s BARGE_IN_CONFIRM
    counters are what implement hangover in this codebase. Keeping the
    detector's own internal smoothing near-zero preserves that division of
    responsibility instead of layering two independent hangover mechanisms
    on top of each other.
    """
    _ensure_runtime()
    min_dur_s = _CHUNK_MS_ACTUAL / 1000.0
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model               = _resolve_silero_vad_file()
    config.silero_vad.threshold           = threshold
    config.silero_vad.min_silence_duration = min_dur_s
    config.silero_vad.min_speech_duration  = min_dur_s
    config.silero_vad.max_speech_duration  = float(MAX_RECORD_SECONDS)
    config.silero_vad.window_size          = _CHUNK_SAMPLES_VAD
    config.sample_rate = SAMPLE_RATE
    config.num_threads  = 1  # VAD is tiny; not worth multithreading
    config.provider     = "cpu"  # tiny model — CUDA EP dispatch overhead isn't worth it
    config.debug        = False
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)


def _load_sense_voice_recognizer():
    """Load SenseVoice as a sherpa-onnx OfflineRecognizer via factory method.

    Returns None if the full `sherpa-onnx` Python package is not installed
    (only the C++ runtime `sherpa-onnx-core` is on the path). Callers must
    check for None and skip ASR-dependent features gracefully — voice listen
    will not work without the full Python bindings, but text chat continues.
    """
    model_path, tokens_path = _resolve_sense_voice_files()
    if not hasattr(sherpa_onnx, "OfflineRecognizer"):
        log.warning(
            "sherpa_onnx.OfflineRecognizer not available — the installed package "
            "is sherpa-onnx-core (C++ runtime only). To enable voice listen, install "
            "the full 'sherpa-onnx' Python package on the active arch."
        )
        return None
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

def _prosody_features(audio: np.ndarray, text: str, sample_rate: int = SAMPLE_RATE) -> dict:
    """Extract transient aggregate speech cues; callers must discard audio."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return {}
    magnitude = np.abs(samples)
    duration = samples.size / float(sample_rate)
    voiced = magnitude >= 0.015
    words = len((text or "").split())
    return {
        "rms": float(min(1.0, np.sqrt(np.mean(samples * samples)) * 8.0)),
        "peak": float(min(1.0, np.max(magnitude))),
        "voiced_fraction": float(np.mean(voiced)),
        "pause_density": float(1.0 - np.mean(voiced)),
        "words_per_second": float(min(20.0, words / duration)) if duration > 0 and words else 0.0,
    }


class AikoListen:
    """
    Microphone capture + SenseVoice ASR transcription (+ optional speaker
    verification against one enrolled voice, + optional wake word / trigger
    phrase gating).
    Uses sounddevice (PortAudio) for mic capture — unified with TTS playback backend.
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
        listen.load_vad()        # loads Silero VAD + kicks off warmup thread (barge detector builds lazily iff enabled)
        listen.load_wakeword()   # loads acoustic wake model if configured (no-op otherwise — see module docstring)
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
        # Two separate VoiceActivityDetector instances: sherpa_onnx bakes
        # the speech-probability threshold into the detector at construction
        # time (unlike the old torch-wrapper callable, which took a raw
        # threshold-free probability out and let callers each apply their
        # own cutoff). _record() and _barge_in_loop() want different
        # thresholds (VAD_THRESHOLD vs the much stricter BARGE_IN_THRESHOLD),
        # so they get their own instance rather than sharing one. They never
        # run concurrently anyway (_barge_in_loop() pauses while
        # self._recording is set), so this isn't about thread-safety, just
        # keeping the two threshold configs independent.
        self._vad_model:       sherpa_onnx.VoiceActivityDetector | None = None
        self._barge_vad_model: sherpa_onnx.VoiceActivityDetector | None = None
        # Acoustic wake word (livekit-wakeword) — None if not configured/
        # installed, in which case listen() falls back to the fuzzy
        # post-ASR text engine (_apply_activation_gate / _strip_prefix_phrase).
        self._wake_model:      object | None = None  # a livekit.wakeword.WakeWordModel, if loaded
        self._wake_model_name: str | None = None
        self._lock        = threading.Lock()
        # Lazy voice boot (see ensure_ready): models load on first mic arm,
        # not at app startup — keeps boot fast and RAM low in text mode.
        self._ensure_lock = threading.Lock()
        self._ready       = False
        self._warmup_done = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        self._barge_in_event:  threading.Event = threading.Event()
        self._barge_in_armed:  threading.Event = threading.Event()
        self._barge_in_active: bool             = False
        self._barge_in_thread: threading.Thread | None = None

        # set while _record() is running — pauses barge-in to avoid mic conflict
        self._recording = threading.Event()

        # sounddevice (lazy-loaded, silencing ALSA noise)
        self._sd = None

        # speaker verification — None if disabled or model missing
        self._speaker_extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None
        self._enrolled_embedding: np.ndarray | None = None
        self._speaker_lock = threading.Lock()

        # wake word / trigger phrase activation session — 0 / expired means
        # "asleep", i.e. the configured phrase(s) must be said again.
        self._activation_lock = threading.Lock()
        self._active_until: float = 0.0

    def _load_sd(self):
        """Lazy-load sounddevice, silencing ALSA noise."""
        if self._sd is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                import sounddevice as sd
                self._sd = sd
        return self._sd

    # ── staged init ───────────────────────────────────────────────────────────

    def ensure_ready(self) -> None:
        """Lazy voice boot: load ASR → VAD → warmup → barge-in monitor once.

        Idempotent and thread-safe. wakeup.boot() no longer runs these steps
        at startup — the first mic arm (WebUI broadcast or CLI listen call)
        triggers this instead, overlapping model load with the browser's
        permission/arming UX. Subsequent calls are no-ops.
        """
        if self._ready:
            return
        with self._ensure_lock:
            if self._ready:
                return
            self.load_asr()
            self.load_vad()
            self.join_warmup()
            self.start_barge_in_monitor()
            self._ready = True

    def load_asr(self) -> None:
        self._model = _load_sense_voice_recognizer()

    def load_vad(self) -> None:
        self._vad_model       = _build_vad(VAD_THRESHOLD)
        # Barge detector is built lazily by _ensure_barge_vad() on first
        # enabled barge-in use — BARGE_IN_ENABLED=0 (default) never pays
        # for the second 30s-buffer instance.
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    def _ensure_barge_vad(self) -> bool:
        """Build the barge-in VAD on first enabled use. False on failure."""
        if self._barge_vad_model is not None:
            return True
        try:
            self._barge_vad_model = _build_vad(BARGE_IN_THRESHOLD)
            return True
        except Exception as e:
            log.warning("[listen] barge VAD build failed: %s", e)
            return False

    def load_wakeword(self) -> None:
        """
        Load the acoustic wake-word model, if configured. No-ops (leaving
        self._wake_model = None) if WAKE_WORD isn't set, WAKE_WORD_MODEL_PATH
        is unset/missing, or livekit-wakeword isn't installed — listen()
        falls back to the fuzzy post-ASR text engine in every one of those
        cases, so this is always safe to call and never raises.
        """
        if not self.gate_enabled():
            return
        if not WAKE_WORD_MODEL_PATH:
            log.info("[listen] WAKE_WORD is set but WAKE_WORD_MODEL_PATH is empty — "
                     "using fuzzy ASR-text wake matching.")
            return
        if not os.path.isfile(WAKE_WORD_MODEL_PATH):
            log.warning("[listen] WAKE_WORD_MODEL_PATH=%r not found — "
                        "falling back to fuzzy ASR-text wake matching.",
                        WAKE_WORD_MODEL_PATH)
            return
        if _WakeWordModel is None:
            log.warning("[listen] WAKE_WORD_MODEL_PATH is set but livekit-wakeword isn't "
                        "installed (`pip install livekit-wakeword`) — falling back to "
                        "fuzzy ASR-text wake matching.")
            return
        try:
            self._wake_model = _WakeWordModel(models=[WAKE_WORD_MODEL_PATH])
            self._wake_model_name = os.path.splitext(os.path.basename(WAKE_WORD_MODEL_PATH))[0]
        except Exception:
            log.exception("[listen] failed to load acoustic wake word model — "
                          "falling back to fuzzy ASR-text wake matching.")
            self._wake_model = None
            self._wake_model_name = None

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
        _ensure_runtime()
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
            self.extend_activation()
            return text, {"woke": None}

        matched, remainder = _strip_prefix_phrase(
            text, [WAKE_WORD, *WAKE_WORD_ALIASES], WAKE_FUZZY_THRESHOLD
        )
        if not matched:
            log.debug("[gate] wake word %r NOT matched in %r — dropping", WAKE_WORD, text)
            return None, {"woke": False}

        self.extend_activation()
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
        Always-on VAD monitor via sounddevice. Pauses while _record() is active.

        Idles (checking BARGE_IN_ENABLED roughly every 0.5s) while disabled —
        this is the master switch. The check happens on every idle tick and
        again on every scoring iteration.
        """
        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4
        samples_per_chunk = _CHUNK_SAMPLES_VAD

        while self._barge_in_active:
            if not barge_in_enabled():
                time.sleep(0.5)
                continue
            if not self._ensure_barge_vad():
                time.sleep(0.5)
                continue

            stream = None
            try:
                sd = self._load_sd()
                device = LISTEN_DEVICE if LISTEN_DEVICE >= 0 else None
                stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=samples_per_chunk,
                    device=device,
                )
                stream.start()

                consecutive = 0
                while self._barge_in_active and barge_in_enabled():
                    if self._recording.is_set() or (not barge_in_always_on() and not self._barge_in_armed.is_set()):
                        time.sleep(0.05)
                        consecutive = 0
                        continue

                    if self._barge_in_event.is_set():
                        # Still consume data to keep the stream buffer from filling
                        try:
                            stream.read(samples_per_chunk)
                        except Exception:
                            pass
                        consecutive = 0
                        continue

                    data, overflowed = stream.read(samples_per_chunk)
                    if overflowed:
                        log.warning("[listen] barge-in input overflow")
                    chunk = data.flatten().copy()

                    is_speech = self._is_speech(self._barge_vad_model, chunk)

                    if is_speech:
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
                log.warning("listen: barge-in monitor died: %s", exc)
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        log.warning("listen: failed to close barge-in input stream")

    # ── public api ────────────────────────────────────────────────────────────

    def listen(self, status_callback=None, speak=None, wait_fn=None, chunk_source=None):
        """
        Record and transcribe a single utterance (local mic or WebUI chunk stream).

        Performs full audio capture via Silero VAD scoring, then passes audio to SenseVoice
        for transcription. Handles speaker verification, wake-word gating, and post-ASR
        corrections based on configuration.

        Lazy voice boot: models are loaded here on first use if ensure_ready()
        wasn't already triggered by an earlier mic arm.

        Args:
            status_callback (callable, optional): Callback(msg: str) invoked with status tokens
                (__LISTENING__, __TRANSCRIBING__, __IDLE__, etc.) for UI updates.
            speak (Speak, optional): Speak instance for barge-in coordination. If provided,
                enables speaker-interrupt detection during TTS playback.
            wait_fn (callable, optional): Optional synchronization function; passed to
                internal wait logic (rarely used).
            chunk_source (callable, optional): For WebUI integration only. Callable(n_bytes)
                that returns raw audio bytes from browser pcm-worklet. If None, uses local
                PulseAudio (parec) for microphone input.

        Returns:
            tuple[str, bool]: (transcript, woke_this_call)
                - transcript: Recognized text, or empty string if no speech detected / failed
                - woke_this_call: True if a wake-word session was triggered this call
                  (only relevant when WAKE_WORD gating is enabled)

        Notes:
            - Silero VAD is authoritative for speech/silence detection on ALL audio,
              regardless of source (local mic or WebUI chunk_source path).
            - WEBUI_BROWSER_VAD_GATE (browser energy pre-filter in static/vad.js) only
              controls pre-forwarding, not server-side gating.
            - Breaking change: vad_presegmented parameter was removed. Silero now scores
              every chunk, making pre-segmentation obsolete.
            - If speaker_verify_gate() is enabled, utterances from unrecognized speakers
              are dropped silently (see SPEAKER_VERIFY_GATE in config).
            - If a wake-word gate is configured, utterances are held until wake phrase
              detected (either acoustic model or fuzzy ASR-text match).
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

        # Lazy voice boot: first call pays the model-load cost here (or it
        # already ran earlier via a mic-arm kick from the WebUI).
        self.ensure_ready()
        _cb(status_callback, "__LISTENING__")
        listen_started_at = time.monotonic()
        audio, woke_acoustic = self._record(
            status_callback,
            chunk_source=chunk_source,
        )
        recording_stopped_at = time.monotonic()
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return "", {
                "verified": None,
                "speaker_score": None,
                # Acoustic engine may have woken the session even though no
                # (or too-short) speech followed within this same call —
                # extend_activation() already ran inside _record() in that
                # case, so reflect it here rather than reporting None.
                "woke": True if woke_acoustic else None,
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

        if self._wake_model is not None:
            # Acoustic engine owns wake gating end-to-end: if we got this far
            # with audio at all, either the session was already active or
            # _record() just confirmed the wake word acoustically (both
            # cases mean this utterance is a real command) — no post-ASR
            # text stripping needed, so skip the fuzzy engine entirely.
            gated_text = text
            info["woke"] = True if woke_acoustic else None
        else:
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
        info["duration_s"] = round(len(audio) / float(SAMPLE_RATE), 3)
        info["prosody"] = _prosody_features(audio, gated_text, SAMPLE_RATE)
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

    def _is_speech(self, vad: sherpa_onnx.VoiceActivityDetector, chunk: np.ndarray) -> bool:
        """
        Feed a 512-sample float32 chunk at 16kHz into a sherpa_onnx
        VoiceActivityDetector and report whether it currently considers
        this position speech. Pure onnxruntime under the hood — no torch
        (see "VAD backend" in the module docstring). Replaces the old
        torch-based _score_chunk(), which returned a raw probability;
        threshold is now baked into `vad` at construction (_build_vad()),
        so this returns a bool directly instead of a float to compare.

        `vad` is whichever detector the caller owns (self._vad_model for
        _record(), self._barge_vad_model for _barge_in_loop()) — see the
        comment on those two attributes in __init__ for why there are two.
        """
        if len(chunk) < _CHUNK_SAMPLES_VAD:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES_VAD - len(chunk)))
        else:
            chunk = chunk[:_CHUNK_SAMPLES_VAD]

        vad.accept_waveform(chunk)
        is_speech = vad.is_speech_detected()

        # VoiceActivityDetector buffers completed speech segments internally
        # (front()/pop()) for callers that want it to do their audio
        # buffering for them. We do our own buffering in _record() instead
        # (audio_chunks list, built from the raw chunks we already have), so
        # drain and discard its internal queue here — otherwise it grows
        # unbounded over a long session (e.g. the always-on barge-in
        # monitor) up to buffer_size_in_seconds before wrapping.
        while not vad.empty():
            vad.pop()

        return is_speech

    def _score_wake(self, chunk: np.ndarray) -> float:
        """
        Score a raw audio chunk against the loaded acoustic wake-word model.
        Only called when self._wake_model is not None (see load_wakeword()).
        Feeds our existing 512-sample (32ms) chunks directly — livekit-
        wakeword's predict() takes 16kHz int16/float32 frames and buffers
        internally, per its docs; if this turns out to need a specific
        frame size on hardware (openWakeWord-family models traditionally
        window in 80ms/1280-sample steps), adjust by accumulating chunks
        into a small ring buffer before calling predict(). Not verified on
        AuRoRA hardware yet — flag if wake accuracy looks off.
        """
        scores = self._wake_model.predict(chunk)
        return float(scores.get(self._wake_model_name, 0.0))

    def _record(
        self,
        status_callback=None,
        chunk_source=None,
    ) -> tuple[np.ndarray | None, bool]:
        """
        Capture audio until silence after speech detected. Silero VAD scores
        every chunk here, regardless of source — it is the single
        authoritative speech/silence gate.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None.
            If None (default), audio is captured locally via sounddevice — this is
            the path used by the robot/TUI.
            If provided, that callable is polled instead — used by
            the WebUI to feed mic audio streamed in from the browser over the
            WebSocket. Must return exactly `bytes_per_chunk` bytes of
            float32LE PCM, or None to signal end-of-stream (e.g. the browser
            energy-VAD sentinel b"" was received, or client disconnected).
            Chunks arriving this way have already passed the browser's
            client-side energy-RMS gate (static/vad.js) — a coarse "loud
            enough to send" filter, not a speech/silence decision — so
            Silero still scores every chunk exactly as it does for the
            local-mic path below.

        Returns (audio, woke). `woke` is True iff the acoustic wake engine
        (self._wake_model) fired during this call — see listen() for how
        that's threaded into the returned "woke" info key. Always False
        when the acoustic engine isn't loaded (fuzzy-text engine, or gating
        disabled, or session already active).
        """
        audio_chunks   = []
        silence_count  = 0
        speech_count   = 0
        hearing_speech = False
        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4
        samples_per_chunk = _CHUNK_SAMPLES_VAD

        # Acoustic wake gate: only relevant if a wake model is loaded AND
        # the session isn't already active. If either is false, wake_needed
        # is False and every chunk goes straight into VAD scoring below,
        # same as the original unconditional behavior — the fuzzy post-ASR
        # engine (if configured instead) still runs afterwards in
        # listen()/_apply_activation_gate() exactly as before.
        wake_needed = self._wake_model is not None and self.gate_enabled() and not self.is_active()
        woke_this_call = False
        wake_hits = 0

        _cb(status_callback, "__LISTENING__")
        self._recording.set()

        use_external = chunk_source is not None
        stream = None

        try:
            if not use_external:
                sd = self._load_sd()
                device = LISTEN_DEVICE if LISTEN_DEVICE >= 0 else None
                stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=samples_per_chunk,
                    device=device,
                )
                stream.start()

            self._vad_model.reset()

            for _ in range(_MAX_CHUNKS):
                if use_external:
                    raw = chunk_source(bytes_per_chunk)
                    if raw is None or len(raw) < bytes_per_chunk:
                        break
                    chunk = np.frombuffer(raw, dtype=np.float32).copy()
                else:
                    # Read from sounddevice (blocking read with timeout via blocksize)
                    data, overflowed = stream.read(samples_per_chunk)
                    if overflowed:
                        log.warning("[listen] input overflow")
                    chunk = data.flatten().copy()

                if wake_needed:
                    # Asleep and using the acoustic engine: score for the
                    # wake word instead of accumulating into the transcript
                    # buffer. Pre-wake audio is never buffered or
                    # transcribed — this is the point of the acoustic
                    # engine over the fuzzy-text one, which had to run full
                    # SenseVoice on every utterance just to check.
                    if self._score_wake(chunk) >= WAKE_WORD_THRESHOLD:
                        wake_hits += 1
                    else:
                        wake_hits = 0
                    if wake_hits >= WAKE_WORD_CONFIRM_FRAMES:
                        wake_needed = False
                        woke_this_call = True
                        self.extend_activation()
                        self._vad_model.reset()  # start VAD fresh from the wake point
                    continue

                # Silero scores every chunk from every source — the browser's
                # energy gate (for the WebUI chunk_source path) only decided
                # whether to forward the chunk at all, not whether it's speech.
                is_speech = self._is_speech(self._vad_model, chunk)

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
            return None, False
        finally:
            self._recording.clear()
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    log.warning("listen: failed to close input stream")

        if not audio_chunks:
            return None, woke_this_call

        # ── utterance length gate ─────────────────────────────────────────────
        # Silero has genuinely scored every chunk regardless of source, so
        # speech_count reflects real detected speech — no reason to bypass
        # this for the WebUI path anymore.
        if speech_count < MIN_SPEECH_CHUNKS:
            return None, woke_this_call

        return np.concatenate(audio_chunks).astype(np.float32), woke_this_call

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe float32 16kHz audio using SenseVoice via sherpa-onnx,
        then apply post-ASR name/phrase corrections (see correct_asr_text).

        Returns empty string if ASR is unavailable (sherpa-onnx Python package
        not installed; only the C++ runtime sherpa-onnx-core is on the path).
        """
        if self._model is None:
            log.warning("listen: _transcribe called but ASR model is not loaded — returning empty text")
            return ""
        try:
            with self._lock:
                stream = self._model.create_stream()
                stream.accept_waveform(SAMPLE_RATE, audio)
                self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3
                result = stream.result
                text   = result.text.strip()
                # SenseVoice prepends language/emotion tags like <|en|><|NEUTRAL|><|Speech|><|withitn|>
                # Strip them for clean output
                text = re.sub(r'<\|[^|]+\|>', '', text).strip()
        except Exception:
            try:
                from system.notice import get_notice_bus
                get_notice_bus(current_user_id()).push("ASR", "voice transcription failed — please repeat or type instead")
            except Exception:
                pass
            raise

        try:
            return correct_asr_text(text)
        except Exception:
            return text

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        if self._model is None:
            log.debug("listen: _warmup skipped — ASR model not loaded")
            return
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            stream  = self._model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, silence)
            self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3

            warm_chunk = np.zeros(_CHUNK_SAMPLES_VAD, dtype=np.float32)
            self._is_speech(self._vad_model, warm_chunk)
            self._vad_model.reset()
            # Warm the barge detector only when barge-in is enabled —
            # otherwise it stays unbuilt (see _ensure_barge_vad).
            if barge_in_enabled() and self._ensure_barge_vad():
                self._is_speech(self._barge_vad_model, warm_chunk)
                self._barge_vad_model.reset()

            if self._wake_model is not None:
                self._score_wake(warm_chunk)
        except Exception:
            log.warning("listen: warmup failed")
        finally:
            self._warmup_done.set()


def _cb(callback, msg: str) -> None:
    if callback:
        try:
            callback(msg)
        except Exception:
            log.warning("listen: callback raised")
