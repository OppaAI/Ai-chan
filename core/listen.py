"""
core/listen.py

Aiko's speech-to-text input layer.
  - Captures microphone audio with Silero VAD (neural, energy-independent)
  - Transcribes via ReazonSpeech K2 ASR (sherpa-onnx, Zipformer RNN-T) in a
    background thread
  - Exposes listen() (blocking) and listen_async() (callback) for UI
  - Staged init: load_asr() → load_vad() → join_warmup() for granular
    boot progress reporting via wakeup.py
    (load_whisper() is kept as an alias of load_asr() so existing
    wakeup.py call sites don't need to change)
  - Always-on barge-in VAD monitor: start_barge_in_monitor() runs a
    lightweight Silero-only daemon that sets _barge_in_event when speech is
    detected during TTS playback, enabling speak.wait_or_barge_in()

Dependencies:
    git clone https://github.com/reazon-research/ReazonSpeech
    pip install ReazonSpeech/pkg/k2-asr   # pulls in sherpa-onnx
    pip install numpy silero-vad scipy
    (CUDA optional via sherpa-onnx's CUDA build — falls back to CPU
    automatically. NOTE: on AuRoRA the Jetson GPU stack is currently down
    — missing /dev/nvhost-gpu — so this will resolve to cpu until that's
    fixed; no code change will be needed once it is)
    parec (PulseAudio) required for mic capture — no PortAudio/sounddevice
"""

from reazonspeech.k2.asr import load_model, transcribe, audio_from_numpy, TranscribeConfig
from silero_vad import load_silero_vad
import logging
from math import gcd
import numpy as np
import os
from scipy.signal import resample_poly
import subprocess
import threading
import time
import torch
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("sherpa_onnx").setLevel(logging.ERROR)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'listen_whisper': 'Loading ReazonSpeech ASR model...',   # key kept for wakeup.py compat
    'listen_asr':     'Loading ReazonSpeech ASR model...',
    'listen_silero':  'Loading Silero VAD...',
    'listen_warmup':  'Warming up ASR pipeline...',
    'listen_ready':   'Microphone ready',
    'listen_skip':    'ASR skipped (text mode)',
}

# ── config ────────────────────────────────────────────────────────────────────

ASR_DEVICE          = os.getenv("ASR_DEVICE",            "auto")    # cpu, cuda, coreml, auto
ASR_PRECISION        = os.getenv("ASR_PRECISION",        "int8")    # fp32, int8, int8-fp32
ASR_LANGUAGE         = os.getenv("ASR_LANGUAGE",          "ja-en")  # ja, ja-en (bilingual JA/EN)
VAD_SILENCE_MS      = int(os.getenv("LISTEN_VAD_SILENCE_MS", 300))
VAD_PAD_MS          = int(os.getenv("LISTEN_VAD_PAD_MS",     100))

SAMPLE_RATE         = 16000                                          # ASR + Silero target
LISTEN_DEVICE       = os.getenv("LISTEN_DEVICE", None)              # None = default

CHUNK_DURATION_MS   = int(os.getenv("LISTEN_CHUNK_MS",         30))  # Silero minimum
VAD_THRESHOLD       = float(os.getenv("LISTEN_VAD_THRESHOLD", 0.5))  # Silero speech prob cutoff
SILENCE_CHUNKS      = int(os.getenv("LISTEN_SILENCE_CHUNKS",   20))
MIN_SPEECH_CHUNKS   = int(os.getenv("LISTEN_MIN_CHUNKS",       10))
MAX_RECORD_SECONDS  = int(os.getenv("LISTEN_MAX_SECONDS",      30))  # K2 model caps ~30s/clip

BARGE_IN_THRESHOLD     = float(os.getenv("BARGE_IN_THRESHOLD",     "0.65"))
BARGE_IN_CONFIRM       = int(os.getenv("BARGE_IN_CONFIRM_CHUNKS",  "2"))
BARGE_IN_COOLDOWN_MS   = int(os.getenv("BARGE_IN_COOLDOWN_MS",     "800"))

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


def _resolve_asr_device(device_hint: str) -> tuple[str, str]:
    """Return (device, precision) resolving 'auto' to cuda if available.

    'auto' currently lands on cpu on AuRoRA because the Jetson GPU stack
    (missing /dev/nvhost-gpu) is unresolved — once that's fixed this will
    pick cuda back up with no code change.
    """
    if device_hint != "auto":
        return device_hint, ASR_PRECISION
    try:
        if torch.cuda.is_available():
            return "cuda", ASR_PRECISION
    except Exception:
        pass
    return "cpu", ASR_PRECISION


# ── listen ────────────────────────────────────────────────────────────────────

class AikoListen:
    """
    Microphone capture + ReazonSpeech K2 ASR transcription.
    Uses parec (PulseAudio) for mic capture — no PortAudio/sounddevice.
    Silero VAD gates recording for robust, noise-resilient speech detection.

    Staged init:
        listen = AikoListen()   # no heavy loading
        listen.load_asr()       # loads the ReazonSpeech K2 model (alias: load_whisper())
        listen.load_vad()       # loads Silero VAD + kicks off warmup thread
        listen.join_warmup()    # blocks until warmup completes

    Barge-in monitor (call after join_warmup):
        listen.start_barge_in_monitor()
        Pauses automatically while _record() is active to avoid mic conflicts.
    """

    def __init__(self) -> None:
        self._device, self._precision = _resolve_asr_device(ASR_DEVICE)
        self._model:      object | None       = None  # sherpa_onnx.OfflineRecognizer
        self._vad_model:  object | None       = None
        self._lock        = threading.Lock()
        self._warmup_done = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        self._barge_in_event:  threading.Event = threading.Event()
        self._barge_in_active: bool             = False
        self._barge_in_thread: threading.Thread | None = None

        # set while _record() is running — pauses barge-in to avoid mic conflict
        self._recording = threading.Event()

    # ── staged init ───────────────────────────────────────────────────────────

    def load_asr(self) -> None:
        self._model = load_model(
            device=self._device,
            precision=self._precision,
            language=ASR_LANGUAGE,
        )

    load_whisper = load_asr  # backward-compat alias for existing wakeup.py call sites

    def load_vad(self) -> None:
        self._vad_model = load_silero_vad()
        self._vad_model.eval()
        self._warmup_thread = threading.Thread(target=self._warmup, daemon=True)
        self._warmup_thread.start()

    def join_warmup(self) -> None:
        self._warmup_done.wait()

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
                # pause while main recording is active — avoid mic conflict
                if self._recording.is_set():
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
    ) -> str:
        if speak is not None and speak.is_playing():
            _cb(status_callback, "__WAITING__")
            interrupted = speak.wait_or_barge_in(self._barge_in_event)
            if interrupted:
                self._barge_in_event.clear()
        elif wait_fn is not None:
            wait_fn()

        _cb(status_callback, "__LISTENING__")
        audio = self._record(status_callback)
        if audio is None:
            _cb(status_callback, "__IDLE__")
            return ""
        _cb(status_callback, "__TRANSCRIBING__")
        text = self._transcribe(audio)
        _cb(status_callback, "__IDLE__")
        return text

    def listen_async(self, on_result, status_callback=None) -> threading.Thread:
        def _run():
            text = self.listen(status_callback=status_callback)
            on_result(text)

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

    def _record(self, status_callback=None) -> np.ndarray | None:
        """Capture mic via parec until silence after speech detected."""
        audio_chunks   = []
        silence_count  = 0
        speech_count   = 0
        hearing_speech = False
        bytes_per_chunk = _CHUNK_SAMPLES_VAD * 4

        _cb(status_callback, "__LISTENING__")
        self._recording.set()

        try:
            proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)
            for _ in range(_MAX_CHUNKS):
                raw = proc.stdout.read(bytes_per_chunk)
                if len(raw) < bytes_per_chunk:
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
            try:
                proc.terminate()
            except Exception:
                pass

        if speech_count < MIN_SPEECH_CHUNKS:
            return None

        return np.concatenate(audio_chunks).astype(np.float32)

    # ── transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            audio_data = audio_from_numpy(audio, SAMPLE_RATE)
            ret = transcribe(self._model, audio_data, TranscribeConfig(verbose=False))
            return ret.text.strip()

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        try:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            audio_data = audio_from_numpy(silence, SAMPLE_RATE)
            transcribe(self._model, audio_data, TranscribeConfig(verbose=False))
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