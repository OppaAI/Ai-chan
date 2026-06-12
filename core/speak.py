"""
core/speak.py

Aiko's voice output via MioTTS inference server.
Preset-based voice reference: "jp_female", "en_female", or a custom registered preset.

Server setup (run separately):
    # 1. Start LLM backend (Ollama example):
    ollama serve
    ollama run hf.co/Aratako/MioTTS-GGUF:MioTTS-0.6B-Q4_K_M.gguf
    # 2. Start MioTTS synthesis API:
    python run_server.py --llm-base-url http://localhost:11434/v1 \
        --llm-model "hf.co/Aratako/MioTTS-GGUF:MioTTS-0.6B-Q4_K_M.gguf"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --wait "Block until done."
"""

import base64
import io
import os
import re
import sys
import time
import threading
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.silence import silent_stderr

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'speak_miotts': 'Connecting to MioTTS server...',
    'speak_ready':  'TTS ready',
    'speak_skip':   'TTS skipped (text mode)',
}

# ── config ────────────────────────────────────────────────────────────────────

MIOTTS_API_URL = os.getenv("MIOTTS_API_URL",  "http://localhost:8001")
MIOTTS_PRESET  = os.getenv("MIOTTS_PRESET",   "jp_female")
MIOTTS_DEVICE  = int(os.getenv("MIOTTS_DEVICE", "-1"))

# ── text sanitization ─────────────────────────────────────────────────────────

_REPLACEMENTS = [
    (r'\*+',   ''),
    (r'—',     ', '),
    (r'–',     ', '),
    (r'`',     ''),
    (r'#+ ',   ''),
    (r'\[|\]', ''),
    (r'\(|\)', ''),
    (r'~',     ''),
    (r'_',     ' '),
    (r'/',     ' '),
    (r'\\',    ''),
    (r'[<>{}|@#$%^&+=]', ''),
]

_RE_REPLACEMENTS = [(re.compile(p), r) for p, r in _REPLACEMENTS]

def sanitize_for_tts(text: str) -> str:
    """Strip/replace symbols the MioTTS phonemizer cannot handle."""
    # Strip emoji and non-BMP unicode (emoticons, symbols, pictographs)
    text = re.sub(
        r'[\U00010000-\U0010ffff'       # non-BMP (most emoji)
        r'\U0001F000-\U0001FFFF'        # extra emoji blocks
        r'\u2600-\u27BF'               # misc symbols, dingbats
        r'\u2300-\u23FF'               # misc technical
        r'\u25A0-\u25FF'               # geometric shapes
        r'\u2700-\u27BF'               # dingbats
        r']',
        '', text, flags=re.UNICODE
    )
    for pattern, replacement in _RE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    MioTTS inference server client.
    Synthesis is a single HTTP round-trip; playback uses sounddevice.
    Printing to console is the caller's responsibility — speak.py is silent.
    """

    def __init__(self, silent: bool = False) -> None:
        self._lock      = threading.Lock()
        self._playing   = threading.Event()
        self._stop_flag = threading.Event()
        self._silent    = silent
        with silent_stderr():
            import sounddevice as _sd
            self._sd = _sd                 # eagerly loaded to avoid curses fd conflict
        self._token_buf: list[str] = []        # accumulate feed() tokens
        if not silent:
            print(f"[speak] MioTTS ready | url: {MIOTTS_API_URL} | preset: {MIOTTS_PRESET}")

    def warmup(self) -> bool:
        """Health-check the MioTTS server — called from wakeup.py during boot."""
        return self._health_check()

    def _health_check(self) -> bool:
        """Ping /health to confirm the server is up."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{MIOTTS_API_URL}/health", timeout=5) as r:
                return r.status == 200
        except Exception as e:
            print(f"[speak] MioTTS server not reachable: {e}")
            return False

    def _load_sd(self):
        """Lazy-load sounddevice, silencing ALSA noise."""
        if self._sd is None:
            with silent_stderr():
                import sounddevice as sd
                self._sd = sd
        return self._sd

    # ── synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> bytes | None:
        """
        POST to MioTTS /v1/tts and return raw WAV bytes.
        Returns None on failure.
        """
        import json
        import urllib.request
        text = text[:300]  # MioTTS hard limit
        payload = json.dumps({
            "text": text,
            "reference": {"type": "preset", "preset_id": MIOTTS_PRESET},
            "output":    {"format": "base64"},
        }).encode()
        req = urllib.request.Request(
            f"{MIOTTS_API_URL}/v1/tts",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.loads(r.read())
                return base64.b64decode(body["audio"])
        except Exception as e:
            print(f"[speak] synthesis error: {e}")
            return None

    # ── playback ──────────────────────────────────────────────────────────────

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        """Play WAV bytes via sounddevice."""
        import scipy.io.wavfile as wav_io
        self._playing.set()
        self._stop_flag.clear()
        try:
            sd = self._load_sd()
            rate, data = wav_io.read(io.BytesIO(wav_bytes))
            device = MIOTTS_DEVICE if MIOTTS_DEVICE >= 0 else None

            # Resample if output device doesn't natively support the source rate
            # (e.g. cheap USB DACs locked to 48000 Hz while MioTTS outputs 44100)
            if device is not None:
                dev_info = sd.query_devices(device)
                target_rate = int(dev_info["default_samplerate"])
                if target_rate != rate:
                    from scipy.signal import resample
                    num_samples = int(len(data) * target_rate / rate)
                    data = resample(data, num_samples).astype(data.dtype)
                    rate = target_rate

            sd.play(data, rate, device=device)
            while sd.get_stream().active:
                if self._stop_flag.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)
        except Exception as e:
            print(f"[speak] playback error: {e}")
        finally:
            try:
                sd = self._load_sd()
                sd.stop()
            except Exception:
                pass
            self._playing.clear()

    def _speak_thread(self, text: str) -> None:
        """Split into sentence chunks ≤300 chars, synthesize and play each."""
        self._playing.set()
        self._stop_flag.clear()
        try:
            for chunk in self._chunk_text(text):
                if self._stop_flag.is_set():
                    break
                wav = self._synthesize(chunk)
                if wav:
                    self._play_wav_bytes(wav)
        finally:
            self._playing.clear()

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 280) -> list[str]:
        """Split text at sentence boundaries into chunks under max_chars."""
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        chunks = []
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                while sentence:
                    chunks.append(sentence[:max_chars])
                    sentence = sentence[max_chars:]
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    # ── public api ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """Synthesize a complete string, non-blocking. Caller prints to console."""
        clean = sanitize_for_tts(text)
        if not clean:
            return False
        self.stop()
        t = threading.Thread(target=self._speak_thread, args=(clean,), daemon=True)
        t.start()
        return True

    def feed(self, token: str) -> None:
        """Accumulate a token for deferred synthesis."""
        if token:
            self._token_buf.append(token)

    def play_async(self) -> None:
        """Synthesize and play all buffered tokens, then clear the buffer."""
        text = sanitize_for_tts("".join(self._token_buf))
        self._token_buf.clear()
        if not text:
            return
        self.stop()
        self._playing.set()
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()

    def feed_and_play(self, token_iterator) -> None:
        """Consume a token iterator, then synthesize and play. Non-blocking."""
        tokens = []
        for token in token_iterator:
            tokens.append(token)
        text = sanitize_for_tts("".join(tokens))
        if not text:
            return
        self.stop()
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def wait(self) -> None:
        """Block until playback finishes naturally."""
        while self.is_playing():
            time.sleep(0.05)

    def wait_or_barge_in(self, barge_in_event: threading.Event) -> bool:
        """
        Block until TTS finishes naturally OR barge_in_event is set.
        Returns True if interrupted, False if finished naturally.
        """
        while self.is_playing():
            if barge_in_event.is_set():
                self.stop()
                return True
            time.sleep(0.02)
        return False

    def stop(self) -> None:
        if self.is_playing():
            self._stop_flag.set()
            while self.is_playing():
                time.sleep(0.02)


# ── list audio devices ────────────────────────────────────────────────────────

def list_devices() -> None:
    import sounddevice as sd
    print("[speak] Available audio output devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            print(f"  {i:2d}: {dev['name']}")


# ── standalone test ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko speak test (MioTTS)")
    parser.add_argument("text", nargs="?",
        default="Hello! I'm Aiko. Nice to meet you! I run locally on your machine, so everything stays private.")
    parser.add_argument("--devices", action="store_true")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--wait",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.preset:
        os.environ["MIOTTS_PRESET"] = args.preset
    MIOTTS_PRESET = os.getenv("MIOTTS_PRESET", "jp_female")

    voice = AikoSpeak()
    ok    = voice.speak(args.text)
    if args.wait:
        voice.wait()
    sys.exit(0 if ok else 1)