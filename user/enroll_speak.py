"""
enroll_speaker.py

One-time enrollment for Aiko's speaker verification (listen.py).
Records a few seconds of your voice via parec, computes a speaker
embedding with the same sherpa-onnx model listen.py uses at runtime,
and saves it to SPEAKER_ENROLL_PATH (default: ./speaker_enrollment.json).

Usage:
    python enroll_speaker.py
    python enroll_speaker.py --seconds 8
    python enroll_speaker.py --model /path/to/embedding_model.onnx

Requires the same SPEAKER_MODEL_PATH env var (or --model) as listen.py,
pointing at a sherpa-onnx speaker embedding .onnx file:
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
e.g. 3dspeaker_speech_eres2net_base_sv_en_voxceleb_16k.onnx (~28MB)

Saves to user/<USER_ID lowercased>.json (matches listen.py's lookup) —
no separate env var needed, just set USER_ID like the rest of Aiko's config.

Re-running this script overwrites any existing enrollment.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import sherpa_onnx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SAMPLE_RATE = 16000
_PAREC_CMD = [
    "parec",
    "--rate=16000",
    "--channels=1",
    "--format=float32le",
    "--latency-msec=30",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a voice for Aiko speaker verification")
    parser.add_argument("--model", default=os.getenv("SPEAKER_MODEL_PATH", ""),
        help="Path to sherpa-onnx speaker embedding .onnx model")
    parser.add_argument("--out", default=os.path.join("user", f"{os.getenv('USER_ID', 'owner').lower()}.json"),
        help="Where to save the enrollment JSON (default: user/<USER_ID>.json)")
    parser.add_argument("--seconds", type=float, default=6.0,
        help="How long to record (seconds). 5-8s of normal speech works well.")
    parser.add_argument("--name", default=os.getenv("USER_ID", "owner"),
        help="Label stored alongside the embedding (for your reference only)")
    return parser.parse_args()


def _record_seconds(seconds: float) -> np.ndarray:
    """Capture raw mic audio for a fixed duration via parec."""
    n_samples = int(seconds * SAMPLE_RATE)
    n_bytes   = n_samples * 4  # float32

    print(f"Recording for {seconds:.1f}s — speak naturally, a sentence or two is plenty.")
    for i in (3, 2, 1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  recording now!          ")

    proc = subprocess.Popen(_PAREC_CMD, stdout=subprocess.PIPE)
    try:
        raw = b""
        while len(raw) < n_bytes:
            chunk = proc.stdout.read(n_bytes - len(raw))
            if not chunk:
                break
            raw += chunk
    finally:
        proc.terminate()

    print("Done recording.")
    return np.frombuffer(raw, dtype=np.float32).copy()


def main() -> int:
    args = _parse_args()

    if not args.model or not os.path.isfile(args.model):
        print(f"error: --model / SPEAKER_MODEL_PATH not set or file not found: {args.model!r}")
        print("Download a model from:")
        print("  https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models")
        return 1

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.model,
        num_threads=1,
        debug=False,
        provider="cpu",
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    audio = _record_seconds(args.seconds)
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    if rms < 1e-4:
        print("warning: recorded audio looks silent — check your mic / PulseAudio default source.")

    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    stream.input_finished()
    embedding = np.asarray(extractor.compute(stream), dtype=np.float32)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "name": args.name,
            "embedding": embedding.tolist(),
            "dim": int(embedding.shape[0]),
            "model": os.path.basename(args.model),
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f)

    print(f"Enrolled '{args.name}' ({embedding.shape[0]}-dim embedding) → {out_path}")
    print("Set SPEAKER_VERIFY_ENABLED=1 and SPEAKER_MODEL_PATH in .env to activate verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
