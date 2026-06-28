"""
core/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio with Silero VAD (neural, energy-independent)
  - Transcribes via SenseVoice (sherpa-onnx, int8 ONNX) in a background thread
  - Optionally verifies the speaker against one enrolled voice embedding
    (sherpa-onnx SpeakerEmbeddingExtractor) on the same buffered audio, run
    in parallel with transcription — see SPEAKER_VERIFY_ENABLED below
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_asr() → load_vad() → load_speaker_id() → join_warmup()
    for granular boot progress reporting via wakeup.py
  - Always-on barge-in VAD monitor: start_barge_in_monitor() runs a
    lightweight Silero-only daemon that sets _barge_in_event when speech is
    detected during TTS playback, enabling speak.wait_or_barge_in()

Dependencies:
    pip install sherpa-onnx numpy silero-vad scipy huggingface_hub
    Model: auto-downloaded to HF cache on first use (see ASR_MODEL in .env)
    parec (PulseAudio) required for mic capture — no PortAudio/sounddevice

Speaker verification (optional — see SPEAKER_VERIFY_ENABLED in .env):
    1. Download a speaker embedding model (.onnx) from
       https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
       e.g. 3dspeaker_speech_eres2net_base_sv_en_voxceleb_16k.onnx (~28MB)
    2. Set SPEAKER_MODEL_PATH in .env to point at it
    3. Enroll your voice: python enroll_speaker.py
    4. Set SPEAKER_VERIFY_ENABLED=1 in .env
"""
import onnxruntime as _ort
# listen.py line 31
if hasattr(_ort, "set_default_logger_severity"):
    _ort.set_default_logger_severity(3)
    
from huggingface_hub import hf_hub_download
from silero_vad import load_silero_vad
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

ASR_DEVICE      = os.getenv("ASR_DEVICE", "cpu")       # cpu only for now (no CUDA EP on JP7.2)
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
BARGE_IN_ALWAYS_ON      = os.getenv("BARGE_IN_ALWAYS_ON", "0").lower() in {"1", "true", "yes", "on"}

# ── speaker verification config ──────────────────────────────────────────────
# Single-enrollment 1:1 verification (not multi-speaker identification) —
# Aiko has exactly one "owner" voice to check against.

SPEAKER_VERIFY_ENABLED = os.getenv("SPEAKER_VERIFY_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
SPEAKER_MODEL_PATH     = os.getenv("SPEAKER_MODEL_PATH", "")            # path to embedding .onnx
USER_ID                = os.getenv("USER_ID", "owner")
SPEAKER_ENROLL_PATH    = os.path.join("user", f"{USER_ID.lower()}.json")
SPEAKER_VERIFY_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.5"))  # cosine sim cutoff
SPEAKER_NUM_THREADS       = int(os.getenv("SPEAKER_NUM_THREADS", "1"))

_CHUNK_SAMPLES_VAD = 512                                            # at 16 kHz, ~32 ms
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


# ── listen ────────────────────────────────────────────────────────────────────

class AikoListen:
    """
    Microphone capture + SenseVoice ASR transcription (+ optional speaker
    verification against one enrolled voice).
    Uses parec (PulseAudio) for mic capture — no PortAudio/sounddevice.
    Silero VAD gates recording for robust, noise-resilient speech detection.

    Staged init:
        listen = AikoListen()    # no heavy loading
        listen.load_asr()        # loads the SenseVoice model
        listen.load_vad()        # loads Silero VAD + kicks off warmup thread
        listen.load_speaker_id() # loads embedding model + enrolled vector (no-op if disabled)
        listen.join_warmup()     # blocks until warmup completes

    Barge-in monitor (call after join_warmup):
        listen.start_barge_in_monitor()
        Pauses automatically while _record() is active to avoid mic conflicts.
    """

    def __init__(self) -> None:
        self._model:      sherpa_onnx.OfflineRecognizer | None = None
        self._vad_model:  object | None       = None
        self._lock        = threading.Lock()
        self._warmup_done = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        self._barge_in_event:  threading.Event = threading.Event()
        self._barge_in_armed:   threading.Event = threading.Event()
        self._barge_in_active: bool             = False
        self._barge_in_thread: threading.Thread | None = None

        # set while _record() is running — pauses barge-in to avoid mic conflict
        self._recording = threading.Event()

        # speaker verification — None if disabled or model missing
        self._speaker_extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None
        self._enrolled_embedding: np.ndarray | None = None
        self._speaker_lock = threading.Lock()

    # ── staged init ───────────────────────────────────────────────────────────

    def load_asr(self) -> None:
        self._model = _load_sense_voice_recognizer()

    def load_vad(self) -> None:
        self._vad_model = load_silero_vad(onnx=True)
        # self._vad_model.eval()  # PyTorch-only, not needed for OnnxWrapper
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

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
        if not os.path.isfile(SPEAKER_ENROLL_PATH):
            logging.getLogger(__name__).warning(
                f"[listen] SPEAKER_VERIFY_ENABLED=1 but no enrollment found at "
                f"{SPEAKER_ENROLL_PATH!r}; run enroll_speaker.py first. Verification disabled."
            )
            return

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=SPEAKER_MODEL_PATH,
            num_threads=SPEAKER_NUM_THREADS,
            debug=False,
            provider="cpu",
        )
        self._speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

        with open(SPEAKER_ENROLL_PATH) as f:
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
            except Exception:
                pass

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
        attached alongside the text, not a gate in front of it.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None,
            forwarded to _record(). See _record() docstring. None (default)
            preserves the existing local-mic (parec) behavior.
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
        audio = self._record(status_callback, chunk_source=chunk_source)
        recording_stopped_at = time.monotonic()
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return "", {
                "verified": None,
                "speaker_score": None,
                "listen_started_at": listen_started_at,
                "recording_stopped_at": recording_stopped_at,
            }

        _cb(status_callback, "__TRANSCRIBING__")

        info = {
            "verified": None,
            "speaker_score": None,
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

        _cb(status_callback, "__IDLE__")
        return text, info

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

    def _record(self, status_callback=None, chunk_source=None) -> np.ndarray | None:
        """
        Capture audio until silence after speech detected.

        chunk_source: optional callable(bytes_per_chunk) -> bytes | None.
            If None (default), audio is captured locally via parec — this is
            the path used by the robot/TUI, unchanged.
            If provided, that callable is polled instead of parec — used by
            the WebUI to feed mic audio streamed in from the browser over the
            WebSocket. Must return exactly `bytes_per_chunk` bytes of
            float32LE PCM, or None to signal end-of-stream (e.g. client
            disconnected).
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
                    break

                chunk = np.frombuffer(raw, dtype=np.float32).copy()
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
                except Exception:
                    pass

        if speech_count < MIN_SPEECH_CHUNKS:
            return None

        return np.concatenate(audio_chunks).astype(np.float32)

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz audio using SenseVoice via sherpa-onnx."""
        with self._lock:
            stream = self._model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._model.decode_stream(stream)  # decode_stream in sherpa-onnx >= 1.13.3
            result = stream.result
            text = result.text.strip()
            # SenseVoice prepends language/emotion tags like <|en|><|NEUTRAL|><|Speech|><|withitn|>
            # Strip them for clean output
            import re
            text = re.sub(r'<\|[^|]+\|>', '', text).strip()
            return text

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            stream = self._model.create_stream()
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