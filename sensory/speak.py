"""
sensory/speak.py

Aiko's voice output via MioTTS synthesis.
Preset-based voice reference: "jp_female", "aiko_flat", or a custom registered preset.

Server setup (run separately):
    # Using mmnga/mio-tts-cpp (C++ implementation):
    ./build/mio-tts-server \
      -m ~/Aiko-chan/models/miotts/MioTTS-0.4B-Q4_K_M.gguf \
      -mv ~/Aiko-chan/models/miotts/miocodec.gguf \
      --tts-wavlm-model ~/Aiko-chan/models/miotts/wavlm_base_plus_2l_f32.gguf \
      --reference-file-json '[{"key":"jp_female","path":"~/Aiko-chan/models/miotts/jp_female.emb.gguf"},{"key":"aiko_flat","path":"~/Aiko-chan/models/miotts/aiko_flat.emb.gguf"}]'
    
    # Or via systemd:
    sudo systemctl start miotts

Standalone test:
    python sensory/speak.py
    python sensory/speak.py "Hello, I'm Aiko!"
    python sensory/speak.py --devices
    python sensory/speak.py --wait "Block until done."
    python sensory/speak.py --synced --wait "Watch the words land with the voice."
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import threading
import argparse
import unicodedata
import queue
from system.config import env_float, env_int
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
MIOTTS_DEVICE  = env_int("MIOTTS_DEVICE", -1)

MIOTTS_MAX_TOKENS         = env_int("MIOTTS_MAX_TOKENS", 300)
MIOTTS_TEMPERATURE        = env_float("MIOTTS_TEMPERATURE", 0.8)
MIOTTS_TOP_P              = env_float("MIOTTS_TOP_P", 1.0)
MIOTTS_REPETITION_PENALTY = env_float("MIOTTS_REPETITION_PENALTY", 1.15)
MIOTTS_PRESENCE_PENALTY   = env_float("MIOTTS_PRESENCE_PENALTY", 0.0)
MIOTTS_FREQUENCY_PENALTY  = env_float("MIOTTS_FREQUENCY_PENALTY", 0.0)
MIOTTS_BEST_OF_N_ENABLED  = os.getenv("MIOTTS_BEST_OF_N_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MIOTTS_BEST_OF_N          = env_int("MIOTTS_BEST_OF_N", 2)

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


# Numbered lists should still be spoken ("one", "two") but MioTTS often
# garbles "1." / "2)" into repeats or letter-soup. Normalize to "N, ".
_LIST_MARKER_RE = re.compile(
    r"(?m)^\s*(?:(\d{1,2})[.)]\s+|[-*•]\s+)"
)
_INLINE_ENUM_RE = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[.!?]))(\d{1,2})[.)]\s+"
)


def _normalize_list_markers(text: str) -> str:
    """Keep ordinal numbers speakable; drop only bare bullet symbols."""
    def _num(m: re.Match) -> str:
        n = m.group(1)
        return f"{n}, " if n else ""
    text = _LIST_MARKER_RE.sub(_num, text)
    text = _INLINE_ENUM_RE.sub(_num, text)
    return text


def sanitize_for_tts(text: str) -> str:
    """Keep only text and common punctuation the MioTTS phonemizer handles."""
    text = text.lstrip()
    text = _normalize_list_markers(text)
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


_EMOJI_LEADING_RE = re.compile(
    r"^\s*([\U0001F300-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]|\:[a-zA-Z0-9_-]+\:)\s*:?\s*",
    re.UNICODE
)
_EMOJI_TO_EMOTION = {
    "😊": "happy", "😄": "happy", "😁": "happy", "😆": "happy", "🥰": "happy", "😍": "happy", "🙂": "happy", "😋": "happy", "🌸": "happy", "✨": "happy", "❤️": "happy", "💖": "happy", "☺️": "happy",
    "😒": "angry", "😡": "angry", "😠": "angry", "😤": "angry", "🤬": "angry", "💢": "angry",
    "😭": "sorrow", "😢": "sorrow", "🥺": "sorrow", "☹️": "sorrow", "🙁": "sorrow", "😔": "sorrow", "😞": "sorrow", "💧": "sorrow",
    "😮": "surprised", "😯": "surprised", "😲": "surprised", "😳": "surprised", "🤯": "surprised", "😱": "surprised", "⁉️": "surprised", "❓": "surprised",
    "😜": "fun", "🤪": "fun", "😏": "fun", "😈": "fun", "🙃": "fun", "😉": "fun",
    "😐": "neutral", "😑": "neutral", "😶": "neutral", "🤖": "neutral", "😴": "neutral", "🤔": "thinking", "💭": "thinking",
}
_EMOJI_HEADER_RE = re.compile(
    r"^\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]|\:[a-zA-Z0-9_-]+\:)?\s*:\s*",
    re.UNICODE
)
_ACTION_ASTERISK_RE = re.compile(r"\*[^*]+\*")
_THOUGHT_PAREN_RE = re.compile(r"\([^)]+\)")
_FEELING_BRACKET_RE = re.compile(r"\[[^\]]+\]")
_STRUCTURED_SEP_RE = re.compile(r"\n\s*---\s*\n")
_EMOTION_LINE_RE = re.compile(r"(?im)^\s*EMOTION\s*:\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*$")
_ACTION_LINE_RE = re.compile(r"(?im)^\s*ACTION\s*:\s*(.+?)\s*$")
_ALLOWED_EMOTIONS = {
    "neutral", "happy", "shy", "sad", "annoyed", "surprised", "thinking",
}


def parse_aiko_response(text: str) -> dict:
    """Split a model reply into emotion / action / dialogue channels."""
    if isinstance(text, (list, tuple)):
        raw = text[0] if text else ""
    else:
        raw = str(text or "")
    emotion = "neutral"
    action = "none"
    body = raw.strip()

    body = re.sub(r"(?m)^\s*---+\s*$", "", body)

    em_match = _EMOTION_LINE_RE.search(body)
    if em_match:
        cand = em_match.group(1).strip().lower()
        emotion = cand if cand in _ALLOWED_EMOTIONS else _EMOJI_TO_EMOTION.get(cand, cand)
        body = _EMOTION_LINE_RE.sub("", body)
    else:
        emoji_match = _EMOJI_LEADING_RE.match(body)
        if emoji_match:
            symbol = emoji_match.group(1).strip()
            emotion = _EMOJI_TO_EMOTION.get(symbol, symbol)
            body = _EMOJI_LEADING_RE.sub("", body)

    ac_match = _ACTION_LINE_RE.search(body)
    if ac_match:
        action = (ac_match.group(1) or "none").strip() or "none"
        body = _ACTION_LINE_RE.sub("", body)

    body = re.sub(r"(?m)^\s*---+\s*$", "", body)
    body = _EMOJI_HEADER_RE.sub("", body)
    body = _ACTION_ASTERISK_RE.sub("", body)
    body = _THOUGHT_PAREN_RE.sub("", body)
    body = _FEELING_BRACKET_RE.sub("", body)
    body = re.sub(r"\*+", " ", body)
    dialogue = re.sub(r"\s{2,}", " ", body).strip()
    return {
        "emotion": emotion,
        "action": action,
        "dialogue": dialogue,
        "raw": raw,
    }


def extract_dialogue_for_tts(text: str) -> str:
    """Extract pure dialogue for TTS from structured or legacy mixed replies."""
    if not text:
        return ""
    parsed = parse_aiko_response(text)
    return sanitize_for_tts(parsed["dialogue"])


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
        # Lazy PortAudio init via _load_sd() (text mode never pays for it).
        # NOTE: an older revision loaded sounddevice eagerly here to dodge a
        # curses fd conflict; current boot has no curses UI, so lazy is safe.
        self._sd = None
        self._token_buf: list[str] = []        # accumulate feed() tokens
        self._stream_chunks: list[str] = []
        self._stream_queue = None
        self._stream_thread = None
        self._stream_on_word = None
        self._streaming_active = False
        self._first_audio_callback = None
        self._first_audio_fired = threading.Event()
        self._speech_rate = 1.0
        self._speech_volume = 1.0
        self._speech_pitch = 0.0

        # When enabled, streamed TTS drives the UI token callback word-by-word
        # against real WAV duration, instead of letting LLM tokens paint early.
        self.karaoke_text = os.getenv("KARAOKE_TEXT", "0").lower() in {
            "1", "true", "yes", "on",
        }
        # Owner uid captured on the caller's thread for notice-bus pushes.
        # Worker threads don't inherit contextvars, so _synthesize failures
        # inside _speech_stream_worker would otherwise key by guest/env.
        self._notice_uid: str | None = None
        # Persistent HTTP session for MioTTS (keep-alive: one TCP handshake
        # per process instead of per 280-char chunk).
        self._http_session = None

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

    def set_karaoke_enabled(self, enabled: bool) -> None:
        """Runtime toggle for karaoke text pacing (e.g. /karaoke command)."""
        self.karaoke_text = bool(enabled)

    def set_speech_rate(self, rate: float) -> None:
        """Set a bounded optional synthesis speed for the next utterances."""
        try:
            self._speech_rate = max(0.85, min(1.15, float(rate)))
        except (TypeError, ValueError):
            self._speech_rate = 1.0

    def set_expression(self, rate: float = 1.0, volume: float = 1.0, pitch: float = 0.0) -> None:
        """Set bounded optional per-utterance expressive controls."""
        try:
            self._speech_rate = max(0.85, min(1.15, float(rate)))
            self._speech_volume = max(0.75, min(1.15, float(volume)))
            self._speech_pitch = max(-0.15, min(0.15, float(pitch)))
        except (TypeError, ValueError):
            self._speech_rate, self._speech_volume, self._speech_pitch = 1.0, 1.0, 0.0

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

    def _http(self):
        """Lazy persistent session (HTTP keep-alive) for MioTTS calls."""
        if self._http_session is None:
            import requests
            from requests.adapters import HTTPAdapter
            session = requests.Session()
            session.mount("http://", HTTPAdapter(pool_connections=2, pool_maxsize=4))
            session.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=4))
            self._http_session = session
        return self._http_session

    def _synthesize(self, text: str) -> bytes | None:
        """
        POST to MioTTS /mio/tts and return raw WAV bytes.
        Returns None on failure.

        NOTE on the temp file: the server replies with a server-side
        output_file path on the shared local disk, so the read+unlink
        round-trip is inherent to this API (no inline-audio mode) — the
        per-chunk saving here is TCP keep-alive via _http(), not disk I/O.
        """
        import json
        if len(text) > 300:
            log.warning(f"[speak] truncating oversized TTS chunk: {len(text)} chars")
            text = text[:300]

        payload_data = {
            "text": text,
            "reference_key": MIOTTS_PRESET,
            "temp": MIOTTS_TEMPERATURE,
            "top_p": MIOTTS_TOP_P,
            "n_predict": MIOTTS_MAX_TOKENS,
            "repeat_penalty": MIOTTS_REPETITION_PENALTY,
        }
        if abs(self._speech_rate - 1.0) > 0.01:
            payload_data["speed"] = round(self._speech_rate, 3)
        if os.getenv("MIOTTS_EXPRESSIVE_CONTROLS", "0").lower() in {"1", "true", "yes", "on"}:
            if abs(self._speech_volume - 1.0) > 0.01:
                payload_data["volume"] = round(self._speech_volume, 3)
            if abs(self._speech_pitch) > 0.01:
                payload_data["pitch"] = round(self._speech_pitch, 3)

        timeout = 60 if MIOTTS_BEST_OF_N_ENABLED else 30
        try:
            r = self._http().post(f"{MIOTTS_API_URL}/mio/tts", json=payload_data, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            if "output_file" not in body:
                log.error(f"[speak] unexpected TTS response keys: {list(body.keys())}")
                self._push_notice("TTS", "voice synthesis failed — continuing in text")
                return None
            # Read the WAV file that was written
            wav_path = body["output_file"]
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
            # Clean up temp file
            try:
                os.remove(wav_path)
            except Exception as e:
                log.warning(f"[speak] failed to delete temp file {wav_path}: {e}")
            return wav_bytes
        except Exception as e:
            log.error(f"[speak] synthesis error: {e}")
            self._push_notice("TTS", "voice synthesis failed — continuing in text")
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

        # Prefer soundfile (present on Jetson) — scipy is broken on aarch64
        # (missing _ccallback_c). Fall back to scipy only if soundfile unavailable.
        def _read_wav_sf(b: bytes):
            try:
                import soundfile as sf
                data, rate = sf.read(io.BytesIO(b), always_2d=False)
                return rate, data
            except Exception:
                import scipy.io.wavfile as wav_io  # last resort; may be broken on Jetson
                return wav_io.read(io.BytesIO(b))

        def _resample_fallback(data, orig_rate: int, target: int = 48000):
            if orig_rate == target:
                return data, target
            try:
                from scipy.signal import resample
                num_samples = int(len(data) * target / orig_rate)
                return resample(data, num_samples).astype(data.dtype), target
            except Exception:
                # linear interpolation fallback — no scipy needed
                import numpy as np
                xp = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, int(len(data) * target / orig_rate))
                if data.ndim == 1:
                    return np.interp(x_new, xp, data).astype(data.dtype), target
                # stereo: interpolate per channel
                out = np.stack([np.interp(x_new, xp, data[:, c]) for c in range(data.shape[1])], axis=1)
                return out.astype(data.dtype), target

        try:
            sd = self._load_sd()
            rate, data = _read_wav_sf(wav_bytes)
            device = MIOTTS_DEVICE if MIOTTS_DEVICE >= 0 else None

            # Always resample to 48000 Hz (device requirement)
            if rate != 48000:
                data, rate = _resample_fallback(data, rate, 48000)

            sd.play(data, rate, device=device)
            sd.wait()  # Wait until playback finishes
            if self._stop_flag.is_set():
                sd.stop()
        except Exception as e:
            log.error(f"[speak] playback error: {e}")
        finally:
            try:
                sd = self._load_sd()
                sd.stop()
            except Exception:
                log.warning("speak: sd.stop() failed in playback")

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
                clean = extract_dialogue_for_tts(chunk)
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
            # TTS failed or WAV unreadable — fall back to estimated pacing
            # (instead of instant burst) so karaoke text still types at a
            # readable rate while silent. Keeps UI usable during MioTTS OOM.
            try:
                fallback_wps = float(os.getenv("KARAOKE_FALLBACK_WPS", "2.6"))
            except (TypeError, ValueError):
                fallback_wps = 2.6
            fallback_wps = max(0.5, min(10.0, fallback_wps))
            log.warning("[speak] karaoke duration=0, fallback pacing %.1f wps for %d words",
                        fallback_wps, len(words))
            start = time.monotonic()
            for i, word in enumerate(words):
                if self._stop_flag.is_set():
                    break
                self._emit_viseme(self._viseme_for_word(word), 0.85)
                if on_word:
                    on_word(word if i == 0 else " " + word)
                if i + 1 < len(words):
                    time.sleep(1.0 / fallback_wps)
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

    def _capture_notice_uid(self) -> None:
        try:
            from system.userspace import current_user_id
            self._notice_uid = current_user_id()
        except Exception:
            pass

    def _push_notice(self, area: str, brief: str) -> None:
        """Push a one-line notice; never raises (bus must not break audio)."""
        try:
            from system.notice import get_notice_bus
            from system.userspace import current_user_id
            uid = self._notice_uid or current_user_id()
            get_notice_bus(uid).push(area, brief)
        except Exception:
            pass

    def speak(self, text: str) -> bool:
        """Synthesize a complete string, non-blocking. Caller prints to console."""
        clean = extract_dialogue_for_tts(text)
        if not clean:
            return False
        self._capture_notice_uid()
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
        clean = extract_dialogue_for_tts(text)
        if not clean:
            return False
        self._capture_notice_uid()
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
            # Drop-oldest cap: a producer outpacing play_async can't grow
            # this without bound on long sessions.
            total = sum(len(t) for t in self._token_buf)
            while total > 4000 and len(self._token_buf) > 1:
                total -= len(self._token_buf.pop(0))

    def play_async(self) -> None:
        """Synthesize and play all buffered tokens, then clear the buffer."""
        text = extract_dialogue_for_tts("".join(self._token_buf))
        self._token_buf.clear()
        if not text:
            return
        self._capture_notice_uid()
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
        text = extract_dialogue_for_tts("".join(tokens))
        if not text:
            return
        self._capture_notice_uid()
        self.stop()
        self._first_audio_fired.clear()
        self._playing.set()
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()

    def start_speech_stream(self, on_word=None) -> None:
        """Start sentence-level TTS playback for one streamed response."""
        self._capture_notice_uid()
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
                total = sum(len(c) for c in self._stream_chunks)
                while total > 4000 and len(self._stream_chunks) > 1:
                    total -= len(self._stream_chunks.pop(0))
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

        # Skip ALSA probe entirely when audio was never initialized
        # (e.g. text-mode sessions): nothing is playing, so nothing to stop.
        if self._sd is not None:
            try:
                sd = self._load_sd()
                sd.stop()
            except Exception:
                log.warning("speak: sd.stop() failed in stop_stream")

        if self._stream_thread is not None:
            if self._stream_thread.is_alive():
                self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        deadline = time.monotonic() + 2.0
        while self.is_playing() and time.monotonic() < deadline:
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
