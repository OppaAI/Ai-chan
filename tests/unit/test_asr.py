#!/usr/bin/env python3
"""
test_listen_init.py — staged diagnostic harness for core/listen.py's AikoListen init.

AikoListen's real init has no top-level try/except, and _warmup() swallows its
own exceptions, so a bare "error during init" from the app doesn't tell you
which stage failed. This script runs each stage separately and prints a full
traceback at whichever one breaks.

Run from the project root (same directory as main.py) so `import sensory.listen`
resolves:

    uv run python test_listen_init.py                # full staged init, no mic
    uv run python test_listen_init.py --mic           # also do a real listen() pass
    uv run python test_listen_init.py --stage env     # just import + device resolution
    uv run python test_listen_init.py --stage mirror  # just the HF mirror download + file check
    uv run python test_listen_init.py --stage asr     # mirror + sherpa_onnx recognizer build

Env vars are read the same way listen.py reads them (ASR_DEVICE, ASR_PRECISION,
ASR_LANGUAGE, LISTEN_* etc.) — set them before running if you want to test a
specific combination.
"""

import argparse
import os
import sys
import time
import traceback


def banner(label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")


def run_stage(label: str, fn):
    """Run one init stage, report pass/fail with full traceback, return result or None."""
    banner(label)
    t0 = time.time()
    try:
        result = fn()
        print(f"[OK]   {label} ({time.time() - t0:.2f}s)")
        return result
    except Exception:
        print(f"[FAIL] {label} ({time.time() - t0:.2f}s)")
        traceback.print_exc()
        return None


def check_env():
    """Import core.listen and resolve device/precision exactly as AikoListen.__init__ does."""
    import sensory.listen as L

    print(f"ASR_DEVICE      = {L.ASR_DEVICE}")
    print(f"ASR_PRECISION   = {L.ASR_PRECISION}")
    print(f"ASR_LANGUAGE    = {L.ASR_LANGUAGE}")

    device, precision = L._resolve_asr_device(L.ASR_DEVICE)
    print(f"resolved device    = {device}")
    print(f"resolved precision = {precision}")

    try:
        import torch
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    except Exception as e:
        print(f"torch import/probe failed: {type(e).__name__}: {e}")

    # Note: torch seeing CUDA does NOT guarantee sherpa_onnx's bundled ONNX
    # Runtime was built with a CUDA execution provider for this Jetson —
    # those are independent stacks. If device resolves to "cuda" here but
    # the ASR stage below fails inside from_transducer(), that mismatch is
    # the first thing to suspect.

    return device, precision


def check_mirror_download(precision: str):
    """Isolate the HF snapshot_download step from the sherpa_onnx load step."""
    import sensory.listen as L
    import huggingface_hub as hf

    repo = L._JA_EN_MIRROR_REPO
    print(f"repo = {repo}")

    try:
        basedir = hf.snapshot_download(repo, local_files_only=True)
        print(f"found cached snapshot (offline): {basedir}")
    except Exception as e:
        print(f"no offline cache ({type(e).__name__}: {e}), trying network download...")
        basedir = hf.snapshot_download(repo)
        print(f"downloaded snapshot: {basedir}")

    epochs = L._JA_EN_MIRROR_EPOCHS
    suffix_by_precision = {
        "fp32":      ["onnx", "onnx", "onnx"],
        "fp16":      ["fp16.onnx", "fp16.onnx", "fp16.onnx"],
        "int8":      ["int8.onnx", "int8.onnx", "int8.onnx"],
        "int8-fp32": ["int8.onnx", "onnx", "int8.onnx"],
    }
    if precision not in suffix_by_precision:
        raise ValueError(f"Unknown precision '{precision}'")
    enc_sfx, dec_sfx, join_sfx = suffix_by_precision[precision]
    wanted = [
        "tokens.txt",
        f"encoder-epoch-{epochs}-avg-1.{enc_sfx}",
        f"decoder-epoch-{epochs}-avg-1.{dec_sfx}",
        f"joiner-epoch-{epochs}-avg-1.{join_sfx}",
    ]

    print(f"checking {precision} files in {basedir}:")
    missing = []
    for fname in wanted:
        path = os.path.join(basedir, fname)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {'OK     ' if exists else 'MISSING'} {fname} ({size} bytes)")
        if not exists:
            missing.append(fname)

    if missing:
        raise FileNotFoundError(f"missing files in mirror snapshot: {missing}")
    return basedir


def check_asr_full(device: str, precision: str):
    """Exercise the real load_asr() path exactly as AikoListen calls it."""
    import sensory.listen as L

    listener = L.AikoListen()
    listener.load_asr()
    print(f"listener._model = {listener._model}")
    return listener


def check_vad(listener):
    listener.load_vad()
    listener.join_warmup()
    print("VAD loaded and warmup finished")
    return listener


def check_mic(listener):
    print("Speak now (recording stops after a pause)...")
    text = listener.listen()
    print(f"transcribed: {text!r}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mic", action="store_true", help="also run a real listen() pass on the mic")
    parser.add_argument(
        "--stage",
        choices=["env", "mirror", "asr", "full"],
        default="full",
        help="stop after this stage (default: full = asr+vad, plus mic if --mic)",
    )
    args = parser.parse_args()

    env_result = run_stage("ENV / DEVICE RESOLUTION", check_env)
    if env_result is None:
        print("\nStopping: can't resolve device/precision without a clean import — see traceback above.")
        sys.exit(1)
    device, precision = env_result
    if args.stage == "env":
        return

    if args.stage == "mirror":
        run_stage("HF MIRROR DOWNLOAD + FILE CHECK", lambda: check_mirror_download(precision))
        return

    listener = run_stage("ASR LOAD (load_asr)", lambda: check_asr_full(device, precision))
    if listener is None or args.stage == "asr":
        if listener is None:
            print("\nASR load failed. Try `--stage mirror` to check whether this is a download/")
            print("missing-file issue versus a sherpa_onnx.OfflineRecognizer.from_transducer issue.")
        return

    listener = run_stage("VAD LOAD + WARMUP", lambda: check_vad(listener))
    if listener is None:
        return

    if args.mic:
        run_stage("MIC TEST (listen())", lambda: check_mic(listener))

    banner("ALL STAGES PASSED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")qemu-user-static
