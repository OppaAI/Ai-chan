"""
sensory/speak.py

Aiko's voice output via MioTTS inference server.
Preset-based voice reference: "jp_female", "en_female", or a custom registered preset.

Server setup (run separately):
    # 1. Start the local OpenAI-compatible model server with the miotts model.
    # 2. Start MioTTS synthesis API:
    python run_server.py --llm-base-url http://localhost:8080/v1 \
        --llm-model "miotts"

Standalone test:
    python sensory/speak.py
    python sensory/speak.py "Hello, I'm Aiko!"
    python sensory/speak.py --devices
    python sensory/speak.py --wait "Block until done."
    python sensory/speak.py --synced --wait "Watch the words land with the voice."
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import time
import threading
import argparse
import unicodedata
import queue
from system.log import get_logger, silent_stderr

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

MIOTTS_MAX_TOKENS         = int(os.getenv("MIOTTS_MAX_TOKENS", "300"))
MIOTTS_TEMPERATURE        = float(os.getenv("MIOTTS_TEMPERATURE", "0.8"))
MIOTTS_TOP_P              = float(os.getenv("MIOTTS_TOP_P", "1.0"))
MIOTTS_REPETITION_PENALTY = float(os.getenv("MIOTTS_REPETITION_PENALTY", "1.15"))
MIOTTS_PRESENCE_PENALTY   = float(os.getenv("MIOTTS_PRESENCE_PENALTY", "0.0"))
MIOTTS_FREQUENCY_PENALTY  = float(os.getenv("MIOTTS_FREQUENCY_PENALTY", "0.0"))
MIOTTS_BEST_OF_N_ENABLED  = os.getenv("MIOTTS_BEST_OF_N_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MIOTTS_BEST_OF_N          = int(os.getenv("MIOTTS_BEST_OF_N", "2"))

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

_EMOJI_SEQUENCE_CHARS = {
    "\u200d",  # zero-width joiner
    "\ufe0e",  # text presentation selector
    "\ufe0f",  # emoji presentation selector
    "\u20e3",  # combining enclosing keycap
}


def _is_tts_noise(char: str) -> bool:
    """Return True for emoji fragments and symbols that confuse MioTTS."""
    codepoint = ord(char)
    if char in _EMOJI_SEQUENCE_CHARS:
        return True
    if 0x1F000 <= codepoint <= 0x1FFFF:
        return True
    if 0x2600 <= codepoint <= 0x27BF:
        return True
    if 0x2300 <= codepoint <= 0x23FF:
        return True
    if 0x2B00 <= codepoint <= 0x2BFF:
        return True
    if 0xFE00 <= codepoint <= 0xFE0F:
        return True
    if 0xE0100 <= codepoint <= 0xE01EF:
        return True

    category = unicodedata.category(char)
    return category[0] == "S" or category in {"Cf", "Cc", "Cs", "Co"}


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Split a long run on natural boundaries without discarding remainder."""
    parts = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars + 1]
        split_at = -1
        for pattern in (r"[\s,;:]\S*$", r"\S+$"):
            match = re.search(pattern, window)
            if match and match.start() > 0:
                split_at = match.start()
                break
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def sanitize_for_tts(text: str) -> str:
    """Keep only text and common punctuation the MioTTS phonemizer handles."""
    text = text.lstrip()
    for pattern, replacement in _RE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    filtered = []
    for char in text:
        substituted = _UNICODE_PUNCTUATION.get(char, char)
        if len(substituted) > 1:
            # Multi-char substitution (e.g. '…' → '...') — append directly
            filtered.append(substituted)
            continue
        char = substituted
        if char in _TTS_PUNCTUATION:
            filtered.append(char)
        elif char.isspace():
            filtered.append(' ')
        elif unicodedata.category(char)[0] in {'L', 'N'}:
            filtered.append(char)
        elif _is_tts_noise(char):
            filtered.append(' ')
        elif unicodedata.category(char)[0] == 'P':
            filtered.append(' ')
    text = ''.join(filtered)

    text = re.sub(r'^\s*[:;,.!?-]+\s*', '', text)
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
        self._stream_chunks: list[str] = []
        self._stream_queue = None
        self._stream_thread = None
        self._stream_on_word = None
        self._streaming_active = False
        self._first_audio_callback = None
        self._first_audio_fired = threading.Event()

        # When enabled, streamed TTS drives the UI token callback word-by-word
        # against real WAV duration, instead of letting LLM tokens paint early.
        self.karaoke_text = os.getenv("KARAOKE_TEXT", "0").lower() in {
            "1", "true", "yes", "on",
        }

        # ── remote audio sink (WebUI) ────────────────────────────────────
        # If set, _play_wav_bytes() also hands each synthesized WAV chunk to
        # this callback (e.g. webui.py's broadcast_audio_bytes) so a
        # connected browser can play it — needed for remote/WAN use where
        # nobody's in the room to hear the Jetson's own speaker.
        self._audio_sink = None
        self._viseme_sink = None
        # When True (default), local sounddevice playback is allowed. If a
        # WebUI audio sink is registered and a browser is actively connected,
        # _play_wav_bytes() temporarily suppresses local playback to avoid
        # doubled/phased audio. With no connected browser, playback remains
        # local as usual. Set False to always silence the local speaker.
        self.local_playback = True

        if not silent:
            log.info(f"[speak] MioTTS ready | url: {MIOTTS_API_URL} | preset: {MIOTTS_PRESET}")

    def set_first_audio_callback(self, callback) -> None:
        """Register a callback invoked when the next utterance starts playback."""
        self._first_audio_callback = callback
        self._first_audio_fired.clear()

    def _notify_first_audio_start(self) -> None:
        if self._first_audio_fired.is_set():
            return
        self._first_audio_fired.set()
        callback = self._first_audio_callback
        if callback is not None:
            try:
                callback()
            except Exception as e:
                log.warning("[speak] first-audio callback failed: %s", e)

    def set_audio_sink(self, callback) -> None:
        """
        Register a callback(wav_bytes: bytes) -> None invoked for every
        synthesized chunk, in addition to (or instead of, see
        `local_playback`) local sounddevice playback. Pass None to remove.
        Typical wiring in your boot script:
            voice.set_audio_sink(web.broadcast_audio_bytes)
        """
        self._audio_sink = callback

    def _has_remote_listener(self) -> bool:
        """Return True when the registered WebUI audio sink has clients.

        The WebUI passes a bound method (AikoWeb.broadcast_audio_bytes) as the
        audio sink. Looking through the bound method lets speak.py avoid local
        playback only when a browser is actually connected, while keeping normal
        Jetson/TUI playback unchanged when no remote listener exists.
        """
        sink_owner = getattr(self._audio_sink, "__self__", None)
        checker = getattr(sink_owner, "has_remote_listener", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception as e:
            log.warning("[speak] remote listener check failed: %s", e)
            return False

    def set_viseme_sink(self, callback) -> None:
        """
        Register a callback(viseme: str, weight: float) -> None invoked during
        TTS playback so a remote avatar can lip-sync to the synthesized voice.
        """
        self._viseme_sink = callback

    def _emit_viseme(self, viseme: str, weight: float = 1.0) -> None:
        if self._viseme_sink is None:
            return
        try:
            self._viseme_sink(viseme, weight)
        except Exception as e:
            log.error("[speak] viseme sink error: %s", e)

    def _viseme_for_word(self, word: str) -> str:
        lowered = word.lower()
        for char in lowered:
            if char in "aあかがさざただなはばぱまやゃらわ":
                return "A"
            if char in "iいきぎしじちぢにひびぴみり":
                return "I"
            if char in "uうくぐすずつづぬふぶぷむゆゅる":
                return "U"
            if char in "eえけげせぜてでねへべぺめれ":
                return "E"
            if char in "oおこごそぞとどのほぼぽもよょろを":
                return "O"
        return "A"

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
            "llm": {
                "temperature": MIOTTS_TEMPERATURE,
                "top_p": MIOTTS_TOP_P,
                "max_tokens": MIOTTS_MAX_TOKENS,
                "repetition_penalty": MIOTTS_REPETITION_PENALTY,
                "presence_penalty": MIOTTS_PRESENCE_PENALTY,
                "frequency_penalty": MIOTTS_FREQUENCY_PENALTY,
            },
            "best_of_n": {
                "enabled": MIOTTS_BEST_OF_N_ENABLED,
                "n": MIOTTS_BEST_OF_N,
            },
            "output": {"format": "base64"},
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
        Play WAV bytes via sounddevice (if local_playback is enabled) and/or
        hand them to the registered remote audio sink (browser playback).
        NOTE: this does not set/clear self._playing or self._stop_flag —
        the calling entry point (_speak_thread / _speak_thread_synced) owns
        those flags for the whole utterance. Touching them per-chunk here
        used to cause is_playing() to flicker false between chunks, and
        could wipe a stop() request that landed between chunks.
        """
        self._notify_first_audio_start()

        if self._audio_sink is not None:
            try:
                self._audio_sink(wav_bytes)
            except Exception as e:
                log.error(f"[speak] audio sink error: {e}")

        # If a browser is connected to the WebUI audio sink, do not also play
        # through the Jetson/local sounddevice. Hearing both endpoints at once
        # (or a remotely forwarded local speaker plus browser playback) can sound
        # like stereo/phasing/doubling. Keep the call blocking for the WAV
        # duration so callers preserve normal turn timing.
        remote_listener_active = self._audio_sink is not None and self._has_remote_listener()
        if not self.local_playback or remote_listener_active:
            duration = self._wav_duration(wav_bytes)
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if self._stop_flag.is_set():
                    break
                time.sleep(0.05)
            return

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

    def _speech_stream_worker(self, chunk_queue, on_word=None) -> None:
        """
        Synthesize and play streamed sentence chunks as soon as they arrive.

        The LLM/UI stream still owns text display; this worker only handles
        sentence-level TTS so voice can start before the full answer is done.

        Pipelined synthesis: the next chunk's HTTP call starts in a background
        thread while the current chunk plays, hiding the HTTP round-trip behind
        audio playback.
        """
        self._playing.set()
        try:
            while not self._stop_flag.is_set():
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                clean = sanitize_for_tts(chunk)
                if not clean:
                    if on_word:
                        self._emit_words_timed(chunk, 0.0, on_word)
                    continue
                pieces = list(self._chunk_text(clean))
                next_synth = None  # (thread, result_container)
                for i, piece in enumerate(pieces):
                    if self._stop_flag.is_set():
                        break

                    if next_synth is not None:
                        synth_thread, synth_result = next_synth
                        synth_thread.join()
                        wav = synth_result[0] if synth_result else None
                    else:
                        wav = self._synthesize(piece)

                    if not wav:
                        if on_word:
                            self._emit_words_timed(piece, 0.0, on_word)
                        next_synth = None
                        continue

                    # Pre-synthesize the next piece while this one plays
                    has_next = i + 1 < len(pieces)
                    if has_next:
                        nr: list = []
                        nt = threading.Thread(
                            target=lambda p=pieces[i+1], r=nr: r.append(self._synthesize(p)),
                            daemon=True,
                        )
                        nt.start()
                        next_synth = (nt, nr)
                    else:
                        next_synth = None

                    if on_word or self._viseme_sink is not None:
                        duration = self._wav_duration(wav)
                        play_thread = threading.Thread(
                            target=self._play_wav_bytes, args=(wav,), daemon=True
                        )
                        play_thread.start()
                        self._emit_words_timed(piece, duration, on_word)
                        play_thread.join()
                    else:
                        self._play_wav_bytes(wav)
        finally:
            self._playing.clear()

    def _emit_words_timed(self, text: str, duration: float, on_word=None) -> None:
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
                self._emit_viseme(self._viseme_for_word(word), 0.85)
                if on_word:
                    on_word(word if i == 0 else " " + word)
            self._emit_viseme("A", 0.0)
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
            self._emit_viseme(self._viseme_for_word(word), 0.85)
            if on_word:
                on_word(word if i == 0 else " " + word)
            elapsed += usable_duration * (weight / total)
        remaining = (start + usable_duration) - time.monotonic()
        while remaining > 0 and not self._stop_flag.is_set():
            time.sleep(min(0.05, remaining))
            remaining = (start + usable_duration) - time.monotonic()
        self._emit_viseme("A", 0.0)

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
        sentences = [
            match.group(0).strip()
            for match in re.finditer(r'[^.!?。！？\n\r]+[.!?。！？]+|[^.!?。！？\n\r]+', text.strip())
            if match.group(0).strip()
        ]
        chunks = []
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_split_oversized_text(sentence, max_chars))
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
        self._first_audio_fired.clear()
        self._playing.set()
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
        self._first_audio_fired.clear()
        self._playing.set()
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
        self._first_audio_fired.clear()
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
        self._first_audio_fired.clear()
        self._playing.set()
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()

    def start_speech_stream(self, on_word=None) -> None:
        """Start sentence-level TTS playback for one streamed response."""
        self.stop()
        self._first_audio_fired.clear()
        with self._lock:
            self._stream_chunks = []
            self._stream_queue = queue.Queue()
            self._stream_on_word = on_word
            self._streaming_active = True
            self._stop_flag.clear()
            self._stream_thread = threading.Thread(
                target=self._speech_stream_worker,
                args=(self._stream_queue, self._stream_on_word),
                daemon=True,
            )
            self._stream_thread.start()

    def feed_speech_stream(self, text: str) -> None:
        """Queue a completed streamed sentence/chunk for immediate TTS."""
        if not text:
            return
        with self._lock:
            if self._streaming_active:
                self._stream_chunks.append(text)
                if self._stream_queue is not None:
                    self._stream_queue.put(text)

    def stop_speech_stream(self) -> None:
        """Finish the current sentence-level TTS stream."""
        with self._lock:
            if not self._streaming_active:
                return
            self._streaming_active = False
            self._stream_chunks = []
            self._stream_on_word = None
            stream_queue = self._stream_queue
            self._stream_queue = None
        if stream_queue is not None:
            stream_queue.put(None)

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
        with self._lock:
            self._streaming_active = False
            self._stream_chunks = []
            self._stream_on_word = None
            stream_queue = self._stream_queue
            self._stream_queue = None
        if stream_queue is not None:
            stream_queue.put(None)

        try:
            sd = self._load_sd()
            sd.stop()
        except Exception:
            pass

        if self._stream_thread is not None:
            if self._stream_thread.is_alive():
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
    from system.config import load_config
    load_config()
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
