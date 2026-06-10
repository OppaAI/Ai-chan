"""
tts_test.py

Chat loop: faster-whisper STT → Ollama LLM → MioTTS TTS.
Listens for speech, transcribes, replies, speaks.

Usage:
    uv run tts_test.py
    uv run tts_test.py --no-tts     # text only
    uv run tts_test.py --no-stt     # keyboard input only

    # Override mic at runtime:
    MIC_DEVICE=29 MIC_CHANNELS=1 SAMPLE_RATE=16000 uv run tts_test.py
    # List available input devices:
    python3 -c "import sounddevice as sd; [print(i, d['name'], 'in:', d['max_input_channels']) for i, d in enumerate(sd.query_devices())]"
"""

import argparse
import base64
import io
import json
import math
import os
import re
import urllib.request

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav_io
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF:UD-Q4_K_XL")
MIOTTS_URL    = os.getenv("MIOTTS_URL",    "http://localhost:8001/v1/tts")
MIOTTS_PRESET = os.getenv("MIOTTS_PRESET", "jp_female")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "250"))

MIC_DEVICE    = int(os.getenv("MIC_DEVICE",   "29"))
MIC_CHANNELS  = int(os.getenv("MIC_CHANNELS", "1"))
SAMPLE_RATE   = int(os.getenv("SAMPLE_RATE",  "16000"))

WHISPER_MODEL    = "turbo"
WHISPER_DEVICE   = "cuda"
WHISPER_LANG     = None
WHISPER_RATE     = 16000
RECORD_SECONDS   = 5
SILENCE_THRESH   = 0.01
SILENCE_DURATION = 1.2

SYSTEM_PROMPT = (
    "You are Aiko, a quiet and deadpan AI companion. "
    "Reply in plain text only — no markdown, no asterisks, no bold, no bullets, no hashtags. "
    "Keep every reply to 1-2 short sentences. "
    "Speak naturally, you may mix English and Japanese."
)

# ── sanitize ──────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub(r'[_`]', '', text)
    text = re.sub(r'—', ', ', text)
    text = re.sub(r'–', ', ', text)
    for ch in ('"', "'", '\u201c', '\u201d', '\u2018', '\u2019'):
        text = text.replace(ch, '')
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# ── stt ───────────────────────────────────────────────────────────────────────

def load_whisper() -> WhisperModel:
    print(f"[STT] Loading whisper-{WHISPER_MODEL} on {WHISPER_DEVICE}...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="float16")
    print("[STT] Ready.")
    return model


def record_until_silence() -> np.ndarray:
    """Record audio until silence is detected or max duration reached."""
    print("[STT] Listening... (speak now)")
    chunk = int(SAMPLE_RATE * 0.1)
    max_chunks = int(RECORD_SECONDS / 0.1)
    silent_chunks_needed = int(SILENCE_DURATION / 0.1)

    frames = []
    silent_count = 0
    speaking = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="float32",
        device=MIC_DEVICE,
    ) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            mono = data.mean(axis=1) if MIC_CHANNELS > 1 else data.flatten()
            frames.append(mono)
            rms = float(np.sqrt(np.mean(mono ** 2)))
            if rms > SILENCE_THRESH:
                speaking = True
                silent_count = 0
            elif speaking:
                silent_count += 1
                if silent_count >= silent_chunks_needed:
                    break

    audio = np.concatenate(frames)

    # resample to 16kHz for Whisper if needed
    if SAMPLE_RATE != WHISPER_RATE:
        g = math.gcd(WHISPER_RATE, SAMPLE_RATE)
        audio = resample_poly(audio, WHISPER_RATE // g, SAMPLE_RATE // g)
    return audio


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANG,
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(s.text for s in segments).strip()
    if text:
        print(f"[STT] ({info.language}) {text}")
    return text

# ── llm ───────────────────────────────────────────────────────────────────────

def chat(history: list[dict]) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": history,
        "stream": False,
        "max_tokens": 60,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip()

# ── tts ───────────────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    text = sanitize(text)[:MAX_TTS_CHARS]
    if not text:
        return
    payload = json.dumps({
        "text": text,
        "reference": {"type": "preset", "preset_id": MIOTTS_PRESET},
        "output":    {"format": "base64"},
    }).encode()
    req = urllib.request.Request(
        MIOTTS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
        wav_bytes = base64.b64decode(body["audio"])
        rate, data = wav_io.read(io.BytesIO(wav_bytes))
        sd.play(data, rate)
        sd.wait()
    except Exception as e:
        print(f"[TTS error] {e}")

# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--no-stt", action="store_true", help="Keyboard input only")
    args = parser.parse_args()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    whisper = None
    if not args.no_stt:
        whisper = load_whisper()

    print("\n=== Aiko Chat Test ===")
    print(f"Model  : {OLLAMA_MODEL}")
    print(f"STT    : {'disabled (keyboard)' if args.no_stt else f'whisper-{WHISPER_MODEL}'}")
    print(f"TTS    : {'disabled' if args.no_tts else MIOTTS_PRESET}")
    print(f"Mic    : sounddevice device {MIC_DEVICE} ({MIC_CHANNELS}ch @ {SAMPLE_RATE}Hz)")
    print("Ctrl+C to exit.\n")

    while True:
        try:
            if args.no_stt:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Bye.")
                    break
            else:
                audio = record_until_silence()
                user_input = transcribe(whisper, audio)
                if not user_input:
                    print("[STT] No speech detected, try again.")
                    continue
                print(f"You: {user_input}")

        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        history.append({"role": "user", "content": user_input})

        try:
            print("Aiko: ", end="", flush=True)
            reply = chat(history)
            print(reply)
            history.append({"role": "assistant", "content": reply})

            if not args.no_tts:
                speak(reply)

        except Exception as e:
            print(f"[LLM error] {e}")
            history.pop()

if __name__ == "__main__":
    main()