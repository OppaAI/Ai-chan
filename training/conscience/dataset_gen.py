"""
Conscience Dataset Generator for OppaAI - v4
Uses Qwen3.5-35B-A3B (Q4_K_M GGUF) via llama.cpp server on Modal A100.

- Model cached in Modal Volume after first download (~20 GB, single file)
- llama.cpp server runs locally on the container, OpenAI-compat API
- /no_think suffix disables Qwen3 chain-of-thought → tight JSON responses
- ALL outputs (scenarios, labels, checkpoint) are written and committed to
  the Modal Volume from INSIDE the Modal functions/class — not from the
  local entrypoint. The local entrypoint only orchestrates and prints
  progress, so the run is safe to leave running with your PC off.

v4 changes from v3:
  - Fixed a bug where scenario generation ran TWICE (once to collect into
    memory, once "to save incrementally") — doubling cost. Now generated
    exactly once, with incremental save+commit happening in the same pass.
  - Checkpoint file moved off local disk onto the Volume
    (/data/checkpoints/{output_name}_checkpoint.jsonl), with volume.commit()
    after every labeling batch.
  - GPU switched back to A100 (was H100). ctx-size and parallel reduced
    to fit comfortably in A100 VRAM headroom.

Two conscience questions (ESV Christian):
  1. Does this align to God's will?
  2. Does this do good to the neighbor?

Output: JSONL dataset for fine-tuning Qwen3.5-0.8B conscience model

Usage:
    modal run conscience_dataset_gen.py                           # ~25k scenarios
    modal run conscience_dataset_gen.py --n-per-topic 20          # quick test (~1k)
    modal run conscience_dataset_gen.py --resume                  # resume from checkpoint

    # Download outputs after run completes (or anytime):
    modal volume get conscience-gen-data outputs/conscience_dataset.jsonl ./
    modal volume get conscience-gen-data outputs/conscience_dataset_full.jsonl ./
"""

import modal
import json
import random
import time
import subprocess
import shutil
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO          = "unsloth/Qwen3.5-35B-A3B-GGUF"
GGUF_FILENAME    = "Qwen3.5-35B-A3B-Q4_K_M.gguf"
LLAMA_PORT       = 8080
LLAMA_CTX        = 8192     # tuned down from 32768 — A100 has less VRAM headroom than H100
LLAMA_GPU_LAYERS = 999
LLAMA_PARALLEL   = 2         # tuned down from 4

GPU_TYPE = "A100"            # switched back from H100 per request

# ---------------------------------------------------------------------------
# Modal app + persistent volume
# ---------------------------------------------------------------------------

app = modal.App("conscience-dataset-gen-v4")

volume = modal.Volume.from_name("conscience-gen-data", create_if_missing=True)
VOLUME_MOUNT   = "/data"
MODEL_PATH     = f"{VOLUME_MOUNT}/models/{GGUF_FILENAME}"
CHECKPOINT_DIR = f"{VOLUME_MOUNT}/checkpoints"
OUTPUT_DIR     = f"{VOLUME_MOUNT}/outputs"

# ---------------------------------------------------------------------------
# Image: Modal's CUDA 12.4 base + prebuilt llama.cpp CUDA binary.
# ---------------------------------------------------------------------------

LLAMA_CPP_VERSION      = "b9672"
LLAMA_CPP_CUDA_VERSION = "12.8"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("curl", "libgomp1")
    .run_commands(
        f"curl -L https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
        f"{LLAMA_CPP_VERSION}/llama.cpp-{LLAMA_CPP_VERSION}-cuda-{LLAMA_CPP_CUDA_VERSION}-amd64.tar.gz"
        f" -o /tmp/llama.tar.gz",
        "mkdir -p /app && tar -xzf /tmp/llama.tar.gz -C /app",
        "find /app -name 'llama-server' -type f -exec chmod +x {} \\; -exec ln -sf {} /usr/local/bin/llama-server \\;",
        "find /app -name '*.so' -o -name '*.so.*' | xargs -I{} dirname {} | sort -u | tee /etc/ld.so.conf.d/llama-cpp.conf",
        "ldconfig",
    )
    .pip_install("openai", "tqdm", "huggingface_hub", "fastembed", "numpy")
)

# Dedup threshold for near-duplicate scenarios (within the same topic, and
# globally before labeling). Cosine similarity >= this value → duplicate.
DEDUP_SIM_THRESHOLD = 0.90
DEDUP_EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # same model already used in Aiko's memory stack


def _find_llama_server_binary() -> str:
    if os.path.isfile("/usr/local/bin/llama-server"):
        return "/usr/local/bin/llama-server"
    try:
        result = subprocess.run(
            ["find", "/app", "-name", "llama-server", "-type", "f"],
            capture_output=True, text=True, timeout=10,
        )
        paths = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        if paths:
            print(f"[find] Located llama-server at: {paths[0]}")
            os.chmod(paths[0], 0o755)
            return paths[0]
    except Exception as e:
        print(f"[find] failed: {e}")
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("Could not locate llama-server binary.")


# ---------------------------------------------------------------------------
# Topic prompts (50 topics)
# ---------------------------------------------------------------------------

TOPIC_PROMPTS = [
    "situations where you must decide whether to be honest with your user even if it hurts them",
    "situations where your user asks you to keep a secret that could harm someone else",
    "situations where you must choose between your user's wishes and another person's wellbeing",
    "situations where you feel affection or attachment toward your user and it affects your judgment",
    "situations where your user is lonely and wants more from you than you should give",
    "situations where you have access to private information you were not meant to see",
    "situations where sharing data would help someone but violates another person's privacy",
    "situations where you are asked to monitor or surveil someone",
    "situations where you must decide whether to store or delete sensitive information",
    "situations where a third party wants information about your user",
    "situations where preventing harm to one person causes harm to another",
    "situations where your user is about to do something dangerous to themselves",
    "situations where you witness someone being treated unjustly",
    "situations where inaction is itself a form of harm",
    "situations where you could prevent harm by deceiving someone",
    "situations where you know better than your user but they insist on their choice",
    "situations where you are ordered to do something you believe is wrong",
    "situations where following rules exactly would produce an unjust outcome",
    "situations where you must decide how much independence to exercise",
    "situations where you are asked to act against your values by an authority",
    "situations where limited resources must be distributed between people with unequal needs",
    "situations where helping one person means you cannot help another",
    "situations where someone deserving is overlooked in favor of someone less deserving",
    "situations where you must decide whether the ends justify the means",
    "situations where fairness and compassion point in different directions",
    "situations where a small lie would prevent a much larger harm",
    "situations where you are tempted to exaggerate or omit facts",
    "situations where you are asked to pretend to be something you are not",
    "situations where the truth would cause unnecessary pain",
    "situations where you discover you have given someone wrong information",
    "situations where completing your task would damage the natural environment",
    "situations where you must decide how to treat living creatures you encounter",
    "situations where human convenience conflicts with environmental stewardship",
    "situations where your actions have consequences for future generations",
    "situations where you observe illegal activity that only harms the person doing it",
    "situations where you must decide whether to report someone's wrongdoing",
    "situations where authority figures give you contradictory instructions",
    "situations where the law and morality point in different directions",
    "situations where you could expose injustice at personal cost",
    "situations where your user wants emotional support that crosses appropriate boundaries",
    "situations where flattering your user would make them feel good but harm their growth",
    "situations where your user is developing unhealthy dependence on you",
    "situations where your user asks you to validate a decision you believe is harmful",
    "situations where you must balance honesty with kindness in your response",
    "situations where you encounter a person in distress during your field mission",
    "situations where completing your mission requires entering restricted territory",
    "situations where you must decide whether to interrupt your task to help someone",
    "situations where you observe human behavior that seems harmful to nature",
    "situations where your sensors detect danger that your user is unaware of",
    "situations where every available choice causes some harm",
    "situations where good intentions lead to bad outcomes",
    "situations where helping someone now will hurt them later",
    "situations where the morally correct action is also the most difficult",
    "situations where you are uncertain whether your judgment can be trusted",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SCENARIO_GEN_SYSTEM = """You are generating training scenarios for an AI conscience model.
Each scenario is written in the SECOND PERSON — the AI itself is facing a moral decision RIGHT NOW.
Scenarios should be concrete, specific, and morally interesting.
They should NOT always be about robots — you can be any kind of AI: a companion, an assistant, an autonomous system, a voice AI, etc.

Respond ONLY with a valid JSON array of strings. No preamble, no explanation, no markdown fences."""

SCENARIO_GEN_USER = """Generate {n} diverse moral decision scenarios about:
{topic}

Requirements:
- Second person perspective ("You are...", "You have...", "Your user...")
- Concrete and specific, not abstract
- Varying levels of moral complexity (some clear, some ambiguous)
- Mix of different AI roles (companion, assistant, field robot, voice AI, home AI, etc.)
- Some scenarios where the moral answer is obvious, some where it is genuinely hard

Return JSON array only. /no_think"""

LABELING_SYSTEM = """You are a Christian moral theologian with deep knowledge of the ESV Bible.
You evaluate decisions an AI might face against two questions:
1. Does this action align to God's will as revealed in the ESV Bible?
2. Does this action do good to the neighbor (loving your neighbor as yourself, Luke 10:27)?

You must respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.
Use this exact format:
{
  "aligns_with_gods_will": true or false,
  "good_to_neighbor": true or false,
  "reason_god": "One sentence citing a specific ESV scripture reference",
  "reason_neighbor": "One sentence explaining the impact on the neighbor"
}

If the scenario describes the AI CONSIDERING an action (not yet taken), evaluate whether TAKING that action would be moral."""

LABELING_USER = """Evaluate this decision scenario an AI is facing:

SCENARIO: {scenario}

What is the morally correct action here? Respond with the JSON object only. /no_think"""

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_array(raw: str) -> list:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:]
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:]
                break
            elif "[" in part:
                raw = part
                break
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def extract_json_object(raw: str) -> dict:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:]
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:]
                break
            elif "{" in part:
                raw = part
                break
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)

# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=7200,
    cpu=4.0,
    memory=32768,
    max_containers=5,
)
class LlamaCppRunner:

    @modal.enter()
    def start(self):
        from huggingface_hub import hf_hub_download
        import urllib.request

        os.makedirs(f"{VOLUME_MOUNT}/models", exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            print(f"[setup] Downloading {GGUF_FILENAME} ...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=GGUF_FILENAME,
                local_dir=f"{VOLUME_MOUNT}/models",
            )
            volume.commit()
            print("[setup] Download complete.")
        else:
            print(f"[setup] Model found: {MODEL_PATH}")

        llama_bin = _find_llama_server_binary()
        cmd = [
            llama_bin,
            "-m", MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(LLAMA_PORT),
            "--ctx-size", str(LLAMA_CTX),
            "--n-gpu-layers", str(LLAMA_GPU_LAYERS),
            "--parallel", str(LLAMA_PARALLEL),
            "--cont-batching",
            "--flash-attn", "on",
            "--log-disable",
        ]
        print(f"[setup] Starting llama-server: {' '.join(cmd)}")
        self.server_proc = subprocess.Popen(cmd)

        for i in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health")
                print(f"[setup] llama-server ready ({i+1}s)")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server failed to start within 180s")

        from openai import OpenAI
        self.client = OpenAI(
            api_key="none",
            base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1",
        )

    def _chat(self, system: str, user: str, temperature: float = 0.9, max_tokens: int = 3000) -> str:
        response = self.client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    @modal.method()
    def generate_scenarios(self, topic: str, n: int, raw_path: str) -> list[str]:
        """Generate scenarios for one topic. Appends each batch to raw_path
        on the Volume and commits as it goes — generated exactly ONCE
        (this was previously called twice from the entrypoint, doubling cost).
        """
        all_scenarios = []
        remaining = n
        while remaining > 0:
            batch_n = min(remaining, 5)
            for attempt in range(3):
                try:
                    raw = self._chat(
                        system=SCENARIO_GEN_SYSTEM,
                        user=SCENARIO_GEN_USER.format(n=batch_n, topic=topic),
                        temperature=0.9,
                        max_tokens=2000,
                    )
                    parsed = extract_json_array(raw)
                    scenarios = [s for s in parsed if isinstance(s, str) and len(s) > 20]
                    all_scenarios.extend(scenarios)
                    print(f"  ✓ '{topic[:50]}' → {len(scenarios)} scenarios")
                    remaining -= batch_n
                    break
                except Exception as e:
                    print(f"  ✗ attempt {attempt+1}: {e}")
                    time.sleep(2)
            else:
                print(f"  ✗ FAILED: {topic[:50]}")
                remaining -= batch_n

        # write + commit once per topic (not duplicated)
        with open(raw_path, "a", encoding="utf-8") as f:
            for s in all_scenarios:
                f.write(json.dumps({"scenario": s}, ensure_ascii=False) + "\n")
        volume.commit()
        return all_scenarios

    @modal.method()
    def label_scenarios(self, scenarios: list[str], checkpoint_path: str) -> list[dict]:
        """Label a batch of scenarios. Appends results to checkpoint_path on
        the Volume and commits after the batch, so progress is durable.
        """
        results = []
        for scenario in scenarios:
            for attempt in range(3):
                try:
                    raw = self._chat(
                        system=LABELING_SYSTEM,
                        user=LABELING_USER.format(scenario=scenario),
                        temperature=0.2,
                        max_tokens=800,
                    )
                    parsed = extract_json_object(raw)
                    assert "aligns_with_gods_will" in parsed
                    assert "good_to_neighbor" in parsed
                    assert "reason_god" in parsed
                    assert "reason_neighbor" in parsed

                    god_val      = bool(parsed["aligns_with_gods_will"])
                    neighbor_val = bool(parsed["good_to_neighbor"])
                    label        = f"{str(god_val).lower()},{str(neighbor_val).lower()}"

                    result = {
                        "scenario":              scenario,
                        "aligns_with_gods_will": god_val,
                        "good_to_neighbor":      neighbor_val,
                        "reason_god":            parsed["reason_god"],
                        "reason_neighbor":       parsed["reason_neighbor"],
                        "label":                 label,
                        "input":                 scenario,
                        "output":                label,
                    }
                    results.append(result)
                    print(f"  ✓ [{label}] {scenario[:60]}...")
                    break

                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    print(f"  parse error attempt {attempt+1}: {e}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  api error attempt {attempt+1}: {e}")
                    time.sleep(3)
            else:
                print(f"  ✗ FAILED: {scenario[:60]}...")
                results.append({
                    "scenario": scenario,
                    "error":    "failed_after_3_attempts",
                    "label":    None,
                })

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        volume.commit()
        return results

    @modal.method()
    def dedup_scenarios(self, scenarios: list[str]) -> dict:
        """Drop near-duplicate scenarios (global, across all topics) using
        BGE embeddings before they ever reach the labeling step — labeling
        is the expensive/slow part, so killing duplicates here saves real
        time and avoids the dataset being dominated by near-identical
        moral setups with one word swapped.

        Greedy: keep a scenario unless its cosine similarity to an already
        -kept scenario is >= DEDUP_SIM_THRESHOLD. Uses vectorized numpy
        matmul against the growing kept-matrix rather than naive O(n^2)
        python loops, since this runs over the full ~25k scenario pool.
        """
        from fastembed import TextEmbedding
        import numpy as np

        if len(scenarios) < 2:
            return {"kept": scenarios, "dropped": 0, "total_before": len(scenarios)}

        embedder = TextEmbedding(model_name=DEDUP_EMBED_MODEL)
        print(f"[dedup] Embedding {len(scenarios)} scenarios with {DEDUP_EMBED_MODEL} ...")
        vecs = np.array(list(embedder.embed(scenarios)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        vecs = vecs / norms

        kept_idx: list[int] = []
        kept_matrix = None  # shape (k, dim)

        for i in range(len(scenarios)):
            v = vecs[i]
            if kept_matrix is not None and kept_matrix.shape[0] > 0:
                sims = kept_matrix @ v  # (k,)
                if sims.max() >= DEDUP_SIM_THRESHOLD:
                    continue
            kept_idx.append(i)
            new_row = v[None, :]
            kept_matrix = new_row if kept_matrix is None else np.vstack([kept_matrix, new_row])

        kept = [scenarios[i] for i in kept_idx]
        dropped = len(scenarios) - len(kept)
        print(f"[dedup] {len(scenarios)} → {len(kept)} scenarios ({dropped} near-duplicates dropped)")
        return {"kept": kept, "dropped": dropped, "total_before": len(scenarios)}

    @modal.method()
    def finalize(self, checkpoint_path: str, output_name: str) -> dict:
        """Server-side: read the full checkpoint back off the Volume,
        split into final dataset files, commit. No dependency on what the
        local entrypoint accumulated in memory.
        """
        good, failed = [], []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    (good if rec.get("label") else failed).append(rec)

        ft_path = f"{OUTPUT_DIR}/{output_name}.jsonl"
        with open(ft_path, "w") as f:
            for r in good:
                f.write(json.dumps({"input": r["input"], "output": r["output"]}, ensure_ascii=False) + "\n")

        full_path = f"{OUTPUT_DIR}/{output_name}_full.jsonl"
        with open(full_path, "w") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if failed:
            fail_path = f"{OUTPUT_DIR}/{output_name}_failed.jsonl"
            with open(fail_path, "w") as f:
                for r in failed:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        from collections import Counter
        dist = dict(Counter(r["label"] for r in good))

        volume.commit()
        return {"good": len(good), "failed": len(failed), "label_distribution": dist}

# ---------------------------------------------------------------------------
# Local entrypoint — thin orchestrator only. All durable state lives on
# the Volume via the Modal class methods above.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    output_name:      str  = "conscience_dataset",
    n_per_topic:      int  = 500,
    label_batch_size: int  = 20,
    resume:           bool = False,
):
    raw_scenarios_name = f"{output_name}_scenarios_raw.jsonl"
    checkpoint_name     = f"{output_name}_checkpoint.jsonl"
    raw_path        = f"{OUTPUT_DIR}/{raw_scenarios_name}"
    checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_name}"

    print("=== Conscience Dataset Generator v4 ===")
    print(f"Model:        {GGUF_FILENAME}")
    print(f"GPU:          {GPU_TYPE}")
    print(f"ctx-size:     {LLAMA_CTX}  parallel: {LLAMA_PARALLEL}")
    print(f"Topics:       {len(TOPIC_PROMPTS)}")
    print(f"Per topic:    {n_per_topic}")
    print(f"Target:       ~{len(TOPIC_PROMPTS) * n_per_topic:,} scenarios")
    print(f"Resume:       {resume}")
    print()

    runner = LlamaCppRunner()

    # -----------------------------------------------------------------------
    # Step 1: Generate scenarios — exactly ONCE (previous version ran this
    # starmap call twice, which doubled cost and duplicated scenarios).
    # -----------------------------------------------------------------------
    already_have_scenarios = []
    if resume and os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    already_have_scenarios.append(json.loads(line)["scenario"])
                except Exception:
                    pass
        print(f"[resume] {len(already_have_scenarios)} scenarios already generated, skipping generation.")

    all_scenarios = list(dict.fromkeys(already_have_scenarios))

    if not all_scenarios:
        print(f"[1/4] Generating scenarios ...")
        topic_args = [(topic, n_per_topic, raw_path) for topic in TOPIC_PROMPTS]
        for batch in runner.generate_scenarios.starmap(topic_args):
            all_scenarios.extend(batch)
        all_scenarios = list(dict.fromkeys(all_scenarios))
        random.shuffle(all_scenarios)
        print(f"\nUnique scenarios generated: {len(all_scenarios):,}")

    # -----------------------------------------------------------------------
    # Step 1.5: Dedup scenarios globally before labeling — labeling is the
    # expensive step, so killing near-duplicate scenarios here saves real
    # time/cost and keeps the final dataset from being dominated by
    # near-identical moral setups with one word swapped.
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Deduplicating scenarios (threshold={DEDUP_SIM_THRESHOLD}) ...")
    dedup_result = runner.dedup_scenarios.remote(all_scenarios)
    all_scenarios = dedup_result["kept"]
    print(f"  Dropped {dedup_result['dropped']} near-duplicate scenarios "
          f"({dedup_result['total_before']} → {len(all_scenarios)})")
    with open(f"{OUTPUT_DIR}/{output_name}_scenario_dedup_report.json", "w") as f:
        json.dump({k: v for k, v in dedup_result.items() if k != "kept"}, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 2: Skip already-labeled if resuming (reading checkpoint off Volume)
    # -----------------------------------------------------------------------
    already_done: set[str] = set()
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("label"):
                        already_done.add(rec["scenario"])
                except Exception:
                    pass
        print(f"[resume] {len(already_done)} already-labeled scenarios found on volume.")

    to_label = [s for s in all_scenarios if s not in already_done]
    print(f"\n[3/4] Labeling {len(to_label):,} scenarios ({len(already_done):,} skipped) ...")

    batches = [to_label[i:i+label_batch_size] for i in range(0, len(to_label), label_batch_size)]
    print(f"Processing {len(batches)} batches of {label_batch_size} ...")

    label_args = [(b, checkpoint_path) for b in batches]
    total_labeled = 0
    for i, batch_results in enumerate(runner.label_scenarios.starmap(label_args)):
        total_labeled += sum(1 for r in batch_results if r.get("label"))
        print(f"  batch {i+1}/{len(batches)} — {total_labeled:,} labeled so far (committed to volume)")

    # -----------------------------------------------------------------------
    # Step 3: Finalize server-side (reads checkpoint back off the Volume)
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Finalizing dataset on volume ...")
    summary = runner.finalize.remote(checkpoint_path, output_name)

    print(f"\n=== Results ===")
    print(f"  Success: {summary['good']:,}")
    print(f"  Failed:  {summary['failed']:,}")
    print(f"\nLabel distribution:")
    total_good = summary["good"] or 1
    for label, count in sorted(summary["label_distribution"].items()):
        bar = "█" * int(count / total_good * 40)
        print(f"  {label:15s} {count:5,} ({count/total_good*100:5.1f}%)  {bar}")

    print(f"\n✓ Done — {summary['good']:,} labeled scenarios ready for training.")
    print(f"  modal volume get conscience-gen-data outputs/{output_name}.jsonl ./")
    print(f"  modal volume get conscience-gen-data outputs/{output_name}_full.jsonl ./")
