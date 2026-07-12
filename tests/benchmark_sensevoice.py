import time
import sherpa_onnx
import soundfile as sf

BASE = "/home/oppa-ai/.cache/huggingface/hub/models--csukuangfj--sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/snapshots/2365baeacb507f821a0c8120fcee3d484dba7a07"
WAV = "/home/oppa-ai/Aiko-chan/assets/Aiko_trim.wav"

samples, sample_rate = sf.read(WAV, dtype="float32")
print(f"Loaded wav: {samples.shape}, sample_rate={sample_rate}\n")

PROVIDERS = ["cpu", "cuda", "tensorrt"]
RUNS_PER_PROVIDER = 5  # first run absorbs cold-start/engine-build cost

for provider in PROVIDERS:
    print(f"=== provider: {provider} ===")
    try:
        t0 = time.perf_counter()
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{BASE}/model.int8.onnx",
            tokens=f"{BASE}/tokens.txt",
            use_itn=True,
            provider=provider,
        )
        init_time = time.perf_counter() - t0
        print(f"  init time: {init_time:.3f}s")
    except Exception as e:
        print(f"  FAILED to init: {e}\n")
        continue

    times = []
    transcript = None
    for i in range(RUNS_PER_PROVIDER):
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        t0 = time.perf_counter()
        recognizer.decode_stream(stream)
        dt = time.perf_counter() - t0
        times.append(dt)
        transcript = stream.result.text
        print(f"  run {i+1}: {dt*1000:.1f} ms")

    print(f"  transcript: {transcript}")
    print(f"  first run:  {times[0]*1000:.1f} ms")
    print(f"  best of rest: {min(times[1:])*1000:.1f} ms")
    print(f"  avg of rest:  {sum(times[1:])/len(times[1:])*1000:.1f} ms\n")