"""
core/speak.py

Aiko's voice output via RealtimeTTS + Kokoro.
Voice blending supported via formula strings: "0.2*af_nicole + 0.8*jf_alpha"

Install:
    uv add "realtimetts[kokoro]"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --wait "Block until done."
"""

import os
import re
import sys
import time
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.silence import silent_stderr

# ── config ────────────────────────────────────────────────────────────────────

KOKORO_VOICE  = os.getenv("KOKORO_VOICE",  "0.2*af_nicole + 0.8*jf_alpha")
KOKORO_SPEED  = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG   = os.getenv("KOKORO_LANG",   "en-us")
KOKORO_DEVICE = int(os.getenv("KOKORO_DEVICE", "-1"))

# ── text sanitization ─────────────────────────────────────────────────────────

_REPLACEMENTS = [
    (r'\*',    ''),
    (r'—',     ', '),
    (r'–',     ', '),
    (r'`',     ''),
    (r'#+ ',   ''),
    (r'\[|\]', ''),
]

_RE_REPLACEMENTS = [(re.compile(p), r) for p, r in _REPLACEMENTS]


def sanitize_for_tts(text: str) -> str:
    """Strip/replace symbols the Kokoro phonemizer cannot handle."""
    for pattern, replacement in _RE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    RealtimeTTS wrapper using Kokoro engine.
    ALSA/PyAudio C-level noise suppressed via fd-level stderr redirect.
    Printing to console is the caller's responsibility — speak.py is silent.
    """

    def __init__(self, silent: bool = False) -> None:
        self._stream = None
        self._ready  = False
        self._silent = silent
        if not silent:
            print(f"[speak] Kokoro ready (lazy) | voice: {KOKORO_VOICE} | speed: {KOKORO_SPEED}")

    def warmup(self) -> bool:
        """Pre-load the engine — called from background thread in cli.py."""
        return self._init_engine()

    def _init_engine(self) -> bool:
        """Initialise Kokoro engine. ALSA device probe silenced via fd redirect."""
        if self._ready:
            return True
        try:
            from RealtimeTTS import TextToAudioStream, KokoroEngine
            engine = KokoroEngine(
                voice=KOKORO_VOICE,
                default_speed=KOKORO_SPEED,
            )
            with silent_stderr():
                self._stream = TextToAudioStream(
                    engine,
                    output_device_index=KOKORO_DEVICE if KOKORO_DEVICE >= 0 else None,
                    frames_per_buffer=4096,   # prevents PCM underruns on Jetson
                )
            self._ready = True
            if not getattr(self, '_silent', False):
                print(f"[speak] Kokoro engine loaded | voice: {KOKORO_VOICE!r}")
            return True
        except Exception as e:
            print(f"[speak] failed to load Kokoro engine: {e}")
            return False

    # ── internal ───────────────────────────────────────────────────────────────

    def _play_async(self) -> None:
        """
        Shared play_async with tuned latency params.
        fast_sentence_fragment=False  — wait for complete sentences, no missing words
        minimum_sentence_length=15    — don't fire on very short fragments
        minimum_first_fragment_length=20 — hold first chunk until solid
        buffer_threshold_seconds=0.3  — enough buffer to prevent underruns
        """
        with silent_stderr():
            self._stream.play_async(
                fast_sentence_fragment=False,
                minimum_sentence_length=15,
                minimum_first_fragment_length=20,
                buffer_threshold_seconds=0.3,
                language=KOKORO_LANG,
            )

    def _sanitized_iterator(self, iterator):
        """Sanitize tokens silently — caller handles console printing."""
        for token in iterator:
            clean = sanitize_for_tts(token)
            if clean:
                yield clean

    # ── public api ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """Synthesize a complete string, non-blocking. Caller prints to console."""
        if not self._init_engine():
            return False
        try:
            self.stop()
            self._stream.feed(sanitize_for_tts(text))
            self._play_async()
            return True
        except Exception as e:
            print(f"[speak] speak error: {e}")
            return False

    def feed(self, token: str) -> None:
        """Feed a single token to TTS. Silent — caller prints to console."""
        if not self._init_engine():
            return
        if not token:
            return
        clean = sanitize_for_tts(token)
        if clean:
            try:
                self._stream.feed(clean)
            except Exception as e:
                print(f"\n[speak] feed error: {e}")

    def play_async(self) -> None:
        """Begin async playback after a manual feed() loop."""
        if not self._ready or not self._stream:
            return
        try:
            self._play_async()
        except Exception as e:
            print(f"[speak] play_async error: {e}")

    def feed_and_play(self, token_iterator) -> None:
        """Feed a token iterator and start async playback. Non-blocking."""
        if not self._init_engine():
            return
        try:
            self.stop()
            self._stream.feed(self._sanitized_iterator(token_iterator))
            self._play_async()
        except Exception as e:
            print(f"[speak] feed_and_play error: {e}")

    def is_playing(self) -> bool:
        if not self._stream:
            return False
        return self._stream.is_playing()

    def wait(self) -> None:
        while self.is_playing():
            time.sleep(0.05)

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass


# ── list audio devices ────────────────────────────────────────────────────────

def list_devices() -> None:
    import sounddevice as sd
    print("[speak] Available audio output devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            print(f"  {i:2d}: {dev['name']}")


# ── standalone test ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko speak test")
    parser.add_argument("text", nargs="?",
        default="Hello! I'm Aiko. Nice to meet you! I run locally on your machine, so everything stays private.")
    parser.add_argument("--devices", action="store_true")
    parser.add_argument("--voice",   default=None)
    parser.add_argument("--speed",   default=None, type=float)
    parser.add_argument("--wait",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    args = _parse_args()

    if args.devices:
        list_devices()
        sys.exit(0)

    if args.voice: os.environ["KOKORO_VOICE"] = args.voice
    if args.speed: os.environ["KOKORO_SPEED"] = str(args.speed)
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "0.2*af_nicole + 0.8*jf_alpha")

    voice = AikoSpeak()
    ok    = voice.speak(args.text)
    if args.wait:
        voice.wait()
    sys.exit(0 if ok else 1)
