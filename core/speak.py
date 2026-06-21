"""
core/speak.py

Aiko's voice output via MioTTS inference server.
Preset-based voice reference: "jp_female", "en_female", or a custom registered preset.

Server setup (run separately):
    # 1. Start the local OpenAI-compatible model server with the miotts model.
    # 2. Start MioTTS synthesis API:
    python run_server.py --llm-base-url http://localhost:8080/v1 \
        --llm-model "miotts"

Standalone test:
    python core/speak.py
    python core/speak.py "Hello, I'm Aiko!"
    python core/speak.py --devices
    python core/speak.py --wait "Block until done."
    python core/speak.py --synced --wait "Watch the words land with the voice."
"""

import base64
import io
import os
import queue
import re
import sys
import time
import threading
import argparse
import unicodedata

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.silence import silent_stderr
from core.log import get_logger

log = get_logger(__name__)

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
    (r'-{2,}', ', '),
    (r'`',     ''),
    (r'#+ ',   ''),
    (r'"',      ' '),
    (r'\[|\]', ' '),
    (r'\(|\)', ' '),
    (r'~',     ''),
    (r'_',     ' '),
    (r'/',     ' '),
    (r'\\',    ''),
    (r'[<>{}|@#$%^&+=]', ' '),
]

_RE_REPLACEMENTS = [(re.compile(p), r) for p, r in _REPLACEMENTS]

_TTS_PUNCTUATION = set(".,!?;:'-")
_UNICODE_PUNCTUATION = {
    '…': '...',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '。': '.',
    '、': ',',
    '？': '?',
    '！': '!',
    '：': ':',
    '；': ';',
    '「': '"',
    '」': '"',
    '『': '"',
    '』': '"',
}

def sanitize_for_tts(text: str) -> str:
    """Keep only text and common punctuation the MioTTS phonemizer handles."""
    # Strip any leading emoji (and optional colon/whitespace) at the very start of the text
    text = text.lstrip()
    text = re.sub(
        r'^(?:[\U00010000-\U0010FFFF'
        r'\u2600-\u27BF'
        r'\u2300-\u23FF'
        r'\u2B00-\u2BFF'
        r'\u25A0-\u25FF'
        r'\u2700-\u27BF'
        r'\U0001FA00-\U0001FFFF'
        r'\U0001F000-\U0001FFFF'
        r']\s*)+:?\s*',
        '', text, flags=re.UNICODE
    )

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

    filtered = []
    for char in text:
        char = _UNICODE_PUNCTUATION.get(char, char)
        if char in _TTS_PUNCTUATION:
            filtered.append(char)
        elif char.isspace():
            filtered.append(' ')
        elif unicodedata.category(char)[0] in {'L', 'N'}:
            filtered.append(char)
        elif unicodedata.category(char)[0] == 'P':
            filtered.append(' ')
    text = ''.join(filtered)

    text = re.sub(r"(?<!\w)'|'(?!\w)", ' ', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,;:])\1+', r'\1', text)
    text = re.sub(r'([!?]){3,}', r'\1\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── speak ─────────────────────────────────────────────────────────────────────

class AikoSpeak:
    """
    MioTTS inference server client.
    Synthesis is a single HTTP round-trip; playback uses sounddevice.
    Printing to console is the caller's responsibility — speak.py is silent.

    Two playback modes:
      - speak()        fire-and-forget, no on-screen pacing.
      - speak_synced()  same playback, but also calls an on_word callback
                         paced to roughly track each chunk's real audio
                         duration (karaoke-style), instead of a fixed
                         artificial per-word delay decoupled from the voice.
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
        self._stream_queue = None
        self._stream_thread = None
        self._streaming_active = False
        if not silent:
            log.info(f"[speak] MioTTS ready | url: {MIOTTS_API_URL} | preset: {MIOTTS_PRESET}")

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
            log.warning(f"[speak] MioTTS server not reachable: {e}")
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
        if len(text) > 300:
            log.warning(f"[speak] truncating oversized TTS chunk: {len(text)} chars")
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
            log.error(f"[speak] synthesis error: {e}")
            return None

    @staticmethod
    def _wav_duration(wav_bytes: bytes) -> float:
        """Return the duration (seconds) of a WAV blob, or 0.0 if unreadable."""
        import wave
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                frames = w.getnframes()
                rate   = w.getframerate()
                return frames / float(rate) if rate else 0.0
        except Exception:
            return 0.0

    # ── playback ──────────────────────────────────────────────────────────────

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        """
        Play WAV bytes via sounddevice.
        NOTE: this does not set/clear self._playing or self._stop_flag —
        the calling entry point (_speak_thread / _speak_thread_synced) owns
        those flags for the whole utterance. Touching them per-chunk here
        used to cause is_playing() to flicker false between chunks, and
        could wipe a stop() request that landed between chunks.
        """
        import scipy.io.wavfile as wav_io
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
            log.error(f"[speak] playback error: {e}")
        finally:
            try:
                sd = self._load_sd()
                sd.stop()
            except Exception:
                pass

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

    def _emit_words_timed(self, text: str, duration: float, on_word) -> None:
        """
        Call on_word() for each word in `text`, paced so the words land
        roughly across `duration` seconds — the real audio length of this
        chunk — instead of a fixed artificial delay. Weighted by word length
        (longer words ≈ longer to say) rather than splitting time evenly.

        This is an estimate, not forced phoneme alignment (MioTTS doesn't
        expose word/phoneme timestamps), but it tracks the actual pace of
        the chunk instead of an arbitrary one.
        """
        words = text.split()
        if not words:
            return
        if duration <= 0:
            # couldn't determine audio length — just emit immediately
            for i, word in enumerate(words):
                on_word(word if i == 0 else " " + word)
            return

        # Keep a small lead-in so the first word appears when audio begins,
        # then distribute later words by a speech-ish duration estimate.
        # Punctuation receives extra time because TTS usually pauses there.
        weights = []
        for word in words:
            weight = max(1.0, len(re.sub(r"[^\w]", "", word)) * 0.75) + 0.8
            if re.search(r"[.!?。！？]$", word):
                weight += 3.0
            elif re.search(r"[,;:、]$", word):
                weight += 1.5
            weights.append(weight)

        total = sum(weights) or 1.0
        usable_duration = max(0.05, duration - 0.08)
        start = time.monotonic() + 0.02
        elapsed = 0.0
        for i, (word, weight) in enumerate(zip(words, weights, strict=True)):
            if self._stop_flag.is_set():
                break
            sleep_time = (start + elapsed) - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            on_word(word if i == 0 else " " + word)
            elapsed += usable_duration * (weight / total)

    def _speak_thread_synced(self, text: str, on_word=None) -> None:
        """
        Like _speak_thread, but for each chunk: synthesize first (so the
        real audio duration is known), then play the audio and pace
        on-screen word emission to that chunk's duration in parallel —
        karaoke-style — instead of typing the whole response out at a fixed
        pace and only starting audio afterward.
        """
        self._playing.set()
        self._stop_flag.clear()
        try:
            for chunk in self._chunk_text(text):
                if self._stop_flag.is_set():
                    break
                wav = self._synthesize(chunk)
                if not wav:
                    continue
                duration = self._wav_duration(wav)

                play_thread = threading.Thread(
                    target=self._play_wav_bytes, args=(wav,), daemon=True
                )
                play_thread.start()

                if on_word:
                    self._emit_words_timed(chunk, duration, on_word)

                play_thread.join()
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

    def speak_synced(self, text: str, on_word=None) -> bool:
        """
        Synthesize and play `text`, calling on_word(word_chunk) timed to
        track each chunk's real TTS audio duration (karaoke-style) instead
        of printing the whole response immediately and starting audio
        afterward. Non-blocking — runs in a background thread, same as
        speak(). on_word receives each word pre-padded with a leading space
        except the first, e.g. "Hello", " I'm", " Aiko".
        """
        clean = sanitize_for_tts(text)
        if not clean:
            return False
        self.stop()
        t = threading.Thread(
            target=self._speak_thread_synced, args=(clean, on_word), daemon=True
        )
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

    def start_speech_stream(self) -> None:
        """Initialize and start the background streaming synthesis/playback loop."""
        self.stop()
        self._stream_queue = queue.Queue()
        self._streaming_active = True
        self._stop_flag.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_worker, daemon=True
        )
        self._stream_thread.start()

    def feed_speech_stream(self, text: str) -> None:
        """Feed a text sentence/chunk to the speech stream queue."""
        if self._streaming_active and text:
            self._stream_queue.put(text)

    def stop_speech_stream(self) -> None:
        """Signal that no more text will be fed to the speech stream."""
        if self._streaming_active:
            self._streaming_active = False
            try:
                self._stream_queue.put(None)
            except Exception:
                pass

    def _stream_worker(self) -> None:
        self._playing.set()
        try:
            while self._streaming_active or not self._stream_queue.empty():
                try:
                    chunk = self._stream_queue.get(timeout=0.1)
                except queue.Empty:
                    if not self._streaming_active:
                        break
                    continue
                
                if chunk is None:
                    self._stream_queue.task_done()
                    break
                    
                if self._stop_flag.is_set():
                    self._stream_queue.task_done()
                    continue
                    
                clean = sanitize_for_tts(chunk)
                if clean:
                    for tts_chunk in self._chunk_text(clean):
                        if self._stop_flag.is_set():
                            break
                        wav = self._synthesize(tts_chunk)
                        if wav and not self._stop_flag.is_set():
                            self._play_wav_bytes(wav)
                
                self._stream_queue.task_done()
        except Exception as e:
            log.error(f"[speak] Stream worker exception: {e}")
        finally:
            self._playing.clear()
            self._streaming_active = False

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
        self._stop_flag.set()
        self._streaming_active = False
        if self._stream_queue is not None:
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                    self._stream_queue.task_done()
                except Exception:
                    break
        
        try:
            sd = self._load_sd()
            sd.stop()
        except Exception:
            pass

        if self._stream_thread is not None:
            if self._stream_thread.is_alive():
                try:
                    self._stream_queue.put_nowait(None)
                except Exception:
                    pass
                self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

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
    parser.add_argument("--synced", action="store_true",
        help="demo karaoke-style synced typing instead of plain speak()")
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

    if args.synced:
        def _print_word(w: str) -> None:
            print(w, end="", flush=True)
        print("Aiko-chan: ", end="", flush=True)
        ok = voice.speak_synced(args.text, on_word=_print_word)
        if args.wait:
            voice.wait()
        print()
    else:
        ok = voice.speak(args.text)
        if args.wait:
            voice.wait()

    sys.exit(0 if ok else 1)
