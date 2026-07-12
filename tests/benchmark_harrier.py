import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX_DIR = "/home/oppa-ai/scratch/harrier-onnx-test"
MODEL_PATH = f"{ONNX_DIR}/onnx/model_fp16.onnx"
TOKENIZER_PATH = f"{ONNX_DIR}/tokenizer.json"

TEST_TEXTS = [
    "Jon is refactoring Aiko-chan's core chat architecture for parallel context fetches.",
    "AuRoRA uses a biologically-inspired cognitive architecture with working memory and episodic memory.",
    "みんなにとって、とやくんはものすごく特別な人だったのかもしれない",
    "The Jetson Orin Nano runs JetPack 7.2 with CUDA 13 and Python 3.12.",
    "Harrier embeddings use last-token pooling with L2 normalization.",
]

RUNS = 10

# ── ONNX (fp16, CUDA) ──────────────────────────────────────────────────────
print("=== Loading ONNX fp16 model (CUDA EP) ===")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("Active providers:", sess.get_providers())

def embed_onnx(texts):
    encodings = [tokenizer.encode(t) for t in texts]
    max_len = max(len(e.ids) for e in encodings)
    input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, e in enumerate(encodings):
        ids = e.ids
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    return outputs[0]

print("\n=== ONNX fp16/CUDA timing ===")
times = []
for i in range(RUNS):
    t0 = time.perf_counter()
    vecs = embed_onnx(TEST_TEXTS)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  run {i+1}: {dt*1000:.1f} ms  (shape={vecs.shape})")
print(f"  first run:  {times[0]*1000:.1f} ms")
print(f"  best of rest: {min(times[1:])*1000:.1f} ms")
print(f"  avg of rest:  {sum(times[1:])/len(times[1:])*1000:.1f} ms")

# ── llama-server GGUF (HarrierEmbedder) ────────────────────────────────────
print("\n=== llama-server GGUF timing ===")
import sys
sys.path.insert(0, "/home/oppa-ai/Aiko-chan")
from core.embed import HarrierEmbedder

embedder = HarrierEmbedder()

times = []
for i in range(RUNS):
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(TEST_TEXTS)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  run {i+1}: {dt*1000:.1f} ms  (shape={vecs.shape})")
print(f"  first run:  {times[0]*1000:.1f} ms")
print(f"  best of rest: {min(times[1:])*1000:.1f} ms")
print(f"  avg of rest:  {sum(times[1:])/len(times[1:])*1000:.1f} ms")