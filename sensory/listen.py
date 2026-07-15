"""
sensory/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio; Silero VAD (neural) is the single authoritative
    speech/silence gate for ALL audio sources, local mic or WebUI.
  - For the WebUI path, the browser only runs a lightweight energy-RMS gate
    client-side (see static/vad.js) to decide "loud enough to bother sending" —
    it is NOT a speech/silence judgment. Silero here is what actually decides
    what is speech, on every chunk, regardless of source.
  - Transcribes via SenseVoice (sherpa-onnx, int8 ONNX) in a background thread
  - Optionally verifies the speaker against one enrolled voice embedding
    (sherpa-onnx SpeakerEmbeddingExtractor) on the same buffered audio, run
    in parallel with transcription — see SPEAKER_VERIFY_ENABLED below
  - Optionally gates responses behind a wake word ("Hey Aiko") and/or a
    trigger phrase said alongside speaker verification ("Here is Oppa") —
    see WAKE_WORD / TRIGGER_PHRASE below
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_asr() → load_vad() → load_speaker_id() → join_warmup()
    for granular boot progress reporting via wakeup.py
  - Always-on barge-in VAD monitor: start_barge_in_monitor() runs a
    lightweight Silero-only daemon that sets _barge_in_event when speech is
    detected during TTS playback, enabling speak.wait_or_barge_in()

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

Wake word / trigger phrase (optional — see config/listen.yaml):
    WAKE_WORD ("hey aiko" by default off / "" disables): SenseVoice mangles
    "Aiko" unpredictably since it's not a normal English word, so matching is
    fuzzy (rapidfuzz ratio) against the leading words of the transcript, not
    an exact substring check. WAKE_WORD_ALIASES lets you hardcode observed
    mishearings ("hey iko|hey eco|hey ecko") as extra candidates.

    TRIGGER_PHRASE ("here is oppa" by default off / "" disables) only takes
    effect when SPEAKER_VERIFY_ENABLED=1. It can follow the wake word
    ("Hey Aiko here is Oppa ...") or stand alone if WAKE_WORD is unset. When
    TRIGGER_REQUIRE_SPEAKER_MATCH=1 (default), the phrase alone isn't
    enough — the cosine-similarity speaker check on that same utterance
    must also pass.

    Once woken/triggered, Aiko stays "active" (no phrase required) until
    ACTIVATION_TIMEOUT_S seconds pass with no further utterance, at which
    point the session goes back to sleep and the configured phrase(s) are
    required again. Use AikoListen.is_active() to check this from other
    subsystems (e.g. suppress proactive/unsolicited behavior while asleep),
    and AikoListen.sleep_now() to force it inactive (e.g. an explicit
    "go to sleep" command).
"""
from __future__ import annotations

import onnxruntime as _ort
if hasattr(_ort, "set_default_logger_severity"):
    _ort.set_default_logger_severity(3)

from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad
from system.userspace import user_state_path
import json
import logging
import numpy as np
import os

from scipy.signal import resample_poly
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
logging.getLogger("sherpa_onnx").setLevel(logging.ERROR)

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

ASR_DEVICE      = os.getenv("ASR_DEVICE", "cpu")       # resolved from config/listen.yaml via load_config()
ASR_LANGUAGE    = os.getenv("ASR_LANGUAGE", "auto")    # auto, zh, en, ja, ko, yue, nospeech
ASR_NUM_THREADS = int(os.getenv("ASR_NUM_THREADS", "4"))

# HuggingFace repo — model.int8.onnx + tokens.txt downloaded on first use
ASR_MODEL = os.getenv(
    "ASR_MODEL",
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)

VAD_SILENCE_MS      = int(os.getenv("LISTEN_VAD_SILENCE_MS", 300))
VAD_PAD_MS          = int(os.getenv("LISTEN_VAD_PAD_MS",     100))

SAMPLE_RATE         = 16000                                          # ASR + Silero target
LISTEN_DEVICE       = os.getenv("LISTEN_DEVICE", None)              # None = default

CHUNK_DURATION_MS   = int(os.getenv("LISTEN_CHUNK_MS",         30))  # Silero minimum
VAD_THRESHOLD       = float(os.getenv("LISTEN_VAD_THRESHOLD", 0.5))  # Silero speech prob cutoff
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS",   20))
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",       10))
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",      30))

BARGE_IN_THRESHOLD     = float(os.getenv("BARGE_IN_THRESHOLD",     "0.65"))
BARGE_IN_CONFIRM       = int(os.getenv("BARGE_IN_CONFIRM_CHUNKS",  "2"))
BARGE_IN_COOLDOWN_MS   = int(os.getenv("BARGE_IN_COOLDOWN_MS",     "800"))
BARGE_IN_ALWAYS_ON     = os.getenv("BARGE_IN_ALWAYS_ON", "0").lower() in {"1", "true", "yes", "on"}

# ── speaker verification config ──────────────────────────────────────────────
# Single-enrollment 1:1 verification (not multi-speaker identification) —
# Aiko has exactly one "owner" voice to check against.

SPEAKER_VERIFY_ENABLED   = os.getenv("SPEAKER_VERIFY_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
SPEAKER_MODEL_PATH       = os.getenv("SPEAKER_MODEL_PATH", "")            # path to embedding .onnx
SPEAKER_VERIFY_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.5"))  # cosine sim cutoff
SPEAKER_NUM_THREADS      = int(os.getenv("SPEAKER_NUM_THREADS", "1"))

# ── wake word / trigger phrase config ────────────────────────────────────────
# WAKE_WORD: "" disables wake-word gating entirely (Aiko responds to every
#   utterance, as before). When set, ASR is unreliable on "Aiko" (not a
#   normal English word) so matching is fuzzy, not exact-substring.
# TRIGGER_PHRASE: only takes effect when SPEAKER_VERIFY_ENABLED is also on.
#   "" disables it. Can be said standalone or right after the wake word.

WAKE_WORD             = os.getenv("WAKE_WORD", "").strip().lower()
WAKE_WORD_ALIASES     = [w.strip().lower() for w in os.getenv("WAKE_WORD_ALIASES", "").split("|") if w.strip()]
WAKE_FUZZY_THRESHOLD  = float(os.getenv("WAKE_FUZZY_THRESHOLD", "70"))

TRIGGER_PHRASE                = os.getenv("TRIGGER_PHRASE", "").strip().lower()
TRIGGER_FUZZY_THRESHOLD       = float(os.getenv("TRIGGER_FUZZY_THRESHOLD", "70"))
TRIGGER_REQUIRE_SPEAKER_MATCH = os.getenv("TRIGGER_REQUIRE_SPEAKER_MATCH", "1").lower() in {"1", "true", "yes", "on"}

ACTIVATION_TIMEOUT_S = float(os.getenv("ACTIVATION_TIMEOUT_S", "30"))

_CHUNK_SAMPLES_VAD = 512                                             # at 16 kHz, ~32 ms
_MAX_CHUNKS        = int(MAX_RECORD_SECONDS * 1000 / CHUNK_DURATION_MS)

# parec command — captures at 16kHz mono float32, uses default PulseAudio source
_PAREC_CMD = [
    "parec",
    "--rate=16000",
    "--channels=1",
    "--format=float32le",
    "--latency-msec=30",
]


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

    best_score, best_end = 0.0, 0
    for phrase in candidates:
        phrase_words = phrase.split()
        n = len(phrase_words)
        if n == 0:
            continue
        for span in range(max(1, n - 1), min(len(words), n + 2) + 1):
            window = " ".join(words[:span])
            score = _ratio(window, phrase)
            if score > best_score:
                best_score, best_end = score, span

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

    When chunk_source is provided (WebUI path), the caller may set
    vad_presegmented=True to indicate that the browser has already applied a
    lightweight energy-RMS gate client-side (see static/vad.js) — this is
    only a "loud enough to send" filter, not a speech/silence decision. Silero
    still scores every chunk that arrives via chunk_source; vad_presegmented
    only changes how the *minimum utterance length* gate is interpreted (see
    _record() docstring).

    Staged init:
        listen = AikoListen()    # no heavy loading
        listen.load_asr()        # loads the SenseVoice model
        listen.load_vad()        # loads Silero VAD + kicks off warmup thread
        listen.load_speaker_id() # loads embedding model + enrolled vector (no-op if disabled)
        listen.join_warmup()     # blocks until warmup completes

    Barge-in monitor (call after join_warmup):
        listen.start_barge_in_monitor()
        Pauses automatically while _record() is active to avoid mic conflicts.

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
            logging.getLogger(__name__).warning(
                f"[listen] SPEAKER_VERIFY_ENABLED=1 but SPEAKER_MODEL_PATH "
                f"is missing or invalid ({SPEAKER_MODEL_PATH!r}); verification disabled."
            )
            return
        enroll_path = self.speaker_enroll_path()
        if not os.path.isfile(enroll_path):
            logging.getLogger(__name__).warning(
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
        """True if either wake word or (speaker-verified) trigger phrase
        gating is configured. Useful for UI to show a sleep/wake indicator
        only when the feature is actually in use."""
        return bool(WAKE_WORD) or (bool(TRIGGER_PHRASE) and SPEAKER_VERIFY_ENABLED)

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

    def _extend_activation(self) -> None:
        with self._activation_lock:
            self._active_until = time.monotonic() + ACTIVATION_TIMEOUT_S

    def _apply_activation_gate(self, text: str, verified: bool | None) -> tuple[str | None, dict]:
        """
        Enforce wake-word / trigger-phrase gating on a freshly transcribed
        utterance.

        Returns (command_text, gate_info):
          - command_text is None            → gate failed; caller must
            silently drop the utterance (no response, no side effects)
          - command_text is text w/ any matched wake word / trigger phrase
            prefix stripped off (unchanged if gating isn't configured, or
            the session was already active so no phrase check was needed)

        gate_info = {"woke": bool|None, "triggered": bool|None}:
          None means that particular gate wasn't configured / not evaluated
          this call (e.g. session was already active, or that feature is
          off). Useful for logging / UI state.
        """
        require_wake    = bool(WAKE_WORD)
        require_trigger = bool(TRIGGER_PHRASE) and SPEAKER_VERIFY_ENABLED
        gate_info = {"woke": None, "triggered": None}

        if not require_wake and not require_trigger:
            return text, gate_info

        if self.is_active():
            # already awake/triggered — no phrase needed, just keep the
            # idle clock from expiring
            self._extend_activation()
            return text, gate_info

        remainder = text

        if require_wake:
            matched, remainder = _strip_prefix_phrase(
                remainder, [WAKE_WORD, *WAKE_WORD_ALIASES], WAKE_FUZZY_THRESHOLD
            )
            gate_info["woke"] = matched
            if not matched:
                return None, gate_info

        if require_trigger:
            matched, remainder = _strip_prefix_phrase(
                remainder, [TRIGGER_PHRASE], TRIGGER_FUZZY_THRESHOLD
            )
            gate_info["triggered"] = matched
            if not matched:
                return None, gate_info
            if TRIGGER_REQUIRE_SPEAKER_MATCH and not verified:
                # phrase alone isn't enough — the voiceprint on this same
                # utterance must also match the enrolled speaker
                return None, gate_info

        self._extend_activation()
        return remainder.strip(), gate_info

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
        """
        self._barge_in_event.set()
  
    def _barge_in_loop(self) -> None:
        """Always-on VAD monitor via parec. Pauses while _record() is active."""
        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4

        try:
            proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)
            consecutive = 0
            while self._barge_in_active:
                if self._recording.is_set() or (not BARGE_IN_ALWAYS_ON and not self._barge_in_armed.is_set()):
                    time.sleep(0.05)
                    consecutive = 0
                    continue

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
                pass

    # ── public api ────────────────────────────────────────────────────────────

    def listen(
        self,
        status_callback=None,
        wait_fn=None,
        speak=None,
        chunk_source=None,
        vad_presegmented: bool = False,
    ) -> tuple[str, dict]:
        """
        Returns (text, info). info always has a "verified" key:
          - None  if speaker verification is disabled / not loaded
          - True  if the buffered audio matched the enrolled voice
          - False if it didn't match
        info also carries "speaker_score" (float or None) for logging/tuning.
        Verification never blocks or fails transcription — it's metadata
        attached alongside the text, not a gate in front of it.

        info additionally carries "woke" and "triggered" (bool|None each):
          - None means that gate wasn't configured / not evaluated this call
            (e.g. the session was already active, so no phrase check ran)
          - If wake word and/or trigger phrase gating IS configured and the
            required phrase(s) were not detected, this method returns
            ("", info) — same shape as "no speech detected" — so callers
            that already treat empty text as "nothing to do" handle this
            for free. Any matched wake word / trigger phrase prefix is
            stripped from the returned text.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None,
            forwarded to _record(). See _record() docstring. None (default)
            preserves the existing local-mic (parec) behavior.

        vad_presegmented: when True, the browser has already applied a
            lightweight energy-RMS gate (static/vad.js) before forwarding
            chunks — a "loud enough to send" filter, not a speech decision.
            Silero still scores every chunk in _record() regardless; this
            flag only affects how the minimum-utterance-length check is
            applied. See _record() for details.
        """
        if speak is not None and speak.is_playing():
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
            vad_presegmented=vad_presegmented,
        )
        recording_stopped_at = time.monotonic()
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return "", {
                "verified": None,
                "speaker_score": None,
                "woke": None,
                "triggered": None,
                "listen_started_at": listen_started_at,
                "recording_stopped_at": recording_stopped_at,
            }

        _cb(status_callback, "__TRANSCRIBING__")

        info = {
            "verified": None,
            "speaker_score": None,
            "woke": None,
            "triggered": None,
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
        info["woke"]      = gate_info["woke"]
        info["triggered"] = gate_info["triggered"]

        _cb(status_callback, "__IDLE__")
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
        """Run Silero VAD on a 512-sample float32 chunk at 16kHz."""
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
        vad_presegmented: bool = False,
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

        vad_presegmented: when True, chunks arrived after the browser's
            client-side energy-RMS gate (static/vad.js) — a coarse "loud
            enough to send" filter, not a speech/silence decision, so Silero
            still scores every chunk exactly as it does for the local-mic
            path. The only thing vad_presegmented changes is downstream
            bookkeeping is identical to the non-presegmented path now that
            Silero is genuinely gating — kept as a named parameter for
            clarity and in case source-specific tuning is needed later.
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
                # energy gate (when vad_presegmented) only decided whether to
                # forward the chunk at all, not whether it's speech.
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
                    pass

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
        """Transcribe float32 16kHz audio using SenseVoice via sherpa-onnx."""
        import re
        with self._lock:
            stream = self._model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3
            result = stream.result
            text   = result.text.strip()
            # SenseVoice prepends language/emotion tags like <|en|><|NEUTRAL|><|Speech|><|withitn|>
            # Strip them for clean output
            text = re.sub(r'<\|[^|]+\|>', '', text).strip()
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
            pass
        finally:
            self._warmup_done.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _cb(callback, msg: str) -> None:
    if callback:
        try:
            callback(msg)
        except Exception:
            pass
