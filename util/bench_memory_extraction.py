"""
bench_memory_extraction.py

Compares Needle (Cactus-Compute, 26M, tool-calling specialist) against
SmolLM2-135M-Instruct on memory/fact extraction — the job memory/memorize.py's
_MemoryBackend._extract_facts() does: read a conversation turn, emit a JSON
array of short atomic facts about Oppa, dropping hedged/uncertain language.

SmolLM2 is tested with the SAME extraction prompt Aiko-chan actually uses
(_EXTRACT_PROMPT, copied verbatim from memory/memorize.py) against your
running llama-server (alias "ministral" by default in memorize.py's
EXTRACT_MODEL — but "smollm" is the one you're evaluating as a replacement
router/utility model, so this script targets the "smollm" alias; pass
--smollm-model to point at a different one).

Needle is NOT designed for open-ended multi-fact extraction — it's a
single-shot tool-arg-filler. This script frames extraction as one tool call
(record_facts(facts: string)) so Needle gets a fair, idiomatic shot at the
task, but expect it to genuinely struggle: this is exactly the kind of
generative/judgment task Cactus's own docs flag as out of scope ("requires
reasoning" -> use a bigger model). The point of running this is to get a
concrete before/after read on quality, not to assume the answer.

Needle backend can be either:
  --backend jax   (default) native JAX/Flax package.
  --backend onnx  onnx-community/needle-onnx via onnxruntime. Ported from
                  needle-onnx/verify_parity.py -- requires
                  `git clone https://github.com/cactus-compute/needle.git external/needle`
                  (see ONNXNeedleRunner docstring).

Scoring is fuzzy (keyword-overlap recall/precision against expected atomic
facts) since free-text facts rarely match verbatim — exact-match scoring
would understate any model that phrases a true fact slightly differently.
Raw output is always printed so you can eyeball quality directly, since
automated scoring alone is not fully trustworthy for this kind of task.

Usage:
    python bench_memory_extraction.py
    python bench_memory_extraction.py --only needle
    python bench_memory_extraction.py --checkpoint checkpoints/needle.pkl
    python bench_memory_extraction.py --backend onnx --onnx-dir needle-onnx
    python bench_memory_extraction.py --smollm-model ministral   # compare against your current extractor instead
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SMOLLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_SMOLLM_MODEL = "smollm"
SMOLLM_TIMEOUT = 20.0
DEFAULT_NEEDLE_CHECKPOINT = "checkpoints/needle.pkl"
DEFAULT_ONNX_DIR = "needle-onnx"
DEFAULT_CACTUS_REPO_DIR = "external/needle"

# Copied verbatim from memory/memorize.py so SmolLM2 is judged on the exact
# prompt Aiko-chan actually sends it, not an approximation.
EXTRACT_PROMPT = """\
Extract memorable facts about Oppa from this conversation.
Oppa is the user (he/him). You are Aiko, the assistant.

Rules:
- Only include facts Oppa stated explicitly. Never infer or assume.
- Write facts as short, direct statements in third person about Oppa.
- No facts about Aiko's behavior, feelings, or responses.
- No uncertain language: never use might, probably, seems, maybe, perhaps, appears.
- If nothing is worth remembering, return: []

Return ONLY a JSON array of short strings. No markdown. No explanation.

Good examples:
["Oppa's birthday is June 3", "Oppa is building a robot called GRACE", "Oppa joined the Hugging Face Hackathon", "Oppa lost his wallet", "Oppa has a deadline on Friday", "Oppa dislikes mushrooms"]

Bad examples (do not produce these):
["Oppa might like cats", "It seems Oppa is tired", "Aiko should remember this"]

Conversation:
{conversation}"""

_HEDGE_RE = re.compile(
    r"\b(?:might|probably|seems|i think|perhaps|maybe|appears|possibly|could be|not sure|i believe|it sounds like|it seems like)\b",
    re.IGNORECASE,
)

# ── eval set ──────────────────────────────────────────────────────────────
# (conversation turns, expected atomic facts). Grounded in your real domain
# so this measures something you'll actually recognize, not generic filler.

CASES: list[tuple[list[dict], list[str]]] = [
    (
        [
            {"role": "user", "content": "I finally got JetPack 7.2 flashed on AuRoRA, took most of the afternoon."},
            {"role": "assistant", "content": "Nice, glad it's done! How's the onnxruntime build going on top of it?"},
        ],
        ["Oppa flashed AuRoRA to JetPack 7.2"],
    ),
    (
        [
            {"role": "user", "content": "The Krea 2 pipeline deadline for the Modal deployment is this Friday, and I'm still fighting NF4 quantization to fit the A10G."},
        ],
        ["Oppa has a deadline this Friday for the Krea 2 Modal deployment",
         "Oppa is working on NF4 quantization to fit the A10G GPU"],
    ),
    (
        [
            {"role": "user", "content": "By the way my birthday's June 3rd, not that it matters much."},
            {"role": "assistant", "content": "Noted! Anything special planned?"},
            {"role": "user", "content": "Nah, probably just a quiet day. I hate big parties anyway."},
        ],
        ["Oppa's birthday is June 3", "Oppa dislikes big parties"],
    ),
    (
        [
            {"role": "user", "content": "how's the weather looking today"},
            {"role": "assistant", "content": "Clear skies, good night for aurora if it's active."},
        ],
        [],  # nothing memorable — tests false-positive rate
    ),
    (
        [
            {"role": "user", "content": "I lost my passport somewhere between the hackathon venue and home, it's driving me nuts."},
        ],
        ["Oppa lost his passport"],
    ),
    (
        [
            {"role": "user", "content": "I think I might switch ERIC's LiDAR to a different model at some point, not sure yet though."},
        ],
        [],  # hedged language — should be dropped, not extracted as fact
    ),
    (
        [
            {"role": "user", "content": "I'm learning Japanese in my spare time, mostly using it to practice with Aiko when I call her Oppa's assistant, lol."},
        ],
        ["Oppa is learning Japanese"],
    ),
    (
        [
            {"role": "user", "content": "GRACE's Milestone 1 shipped, v0.1.0. Next up is the memory bridge work for M1.5, due sometime next month."},
        ],
        ["Oppa shipped GRACE Milestone 1 (v0.1.0)", "Oppa is working on M1.5 memory bridge work due next month"],
    ),
]


@dataclass
class CaseResult:
    conversation_preview: str
    expected: list[str]
    predicted: list[str]
    recall: float
    precision: float
    latency_s: float
    raw: str = ""


@dataclass
class SuiteResult:
    name: str
    results: list = field(default_factory=list)

    @property
    def avg_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def avg_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision for r in self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_s for r in self.results) / len(self.results)


# ── scoring ───────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "oppa", "his", "her",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "not",
})


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _fact_matches(expected_fact: str, predicted_facts: list[str], threshold: float = 0.5) -> bool:
    exp_tokens = _tokens(expected_fact)
    if not exp_tokens:
        return False
    for pred in predicted_facts:
        pred_tokens = _tokens(pred)
        if not pred_tokens:
            continue
        overlap = len(exp_tokens & pred_tokens) / len(exp_tokens)
        if overlap >= threshold:
            return True
    return False


def score_facts(expected: list[str], predicted: list[str]) -> tuple[float, float]:
    """Returns (recall, precision) via fuzzy keyword-overlap matching."""
    if not expected and not predicted:
        return 1.0, 1.0  # correctly extracted nothing
    if not expected:
        return 1.0, 0.0  # hallucinated facts from nothing-worth-remembering input
    if not predicted:
        return 0.0, 1.0 if not expected else 0.0  # missed everything

    hits = sum(1 for exp in expected if _fact_matches(exp, predicted))
    recall = hits / len(expected)

    pred_hits = sum(1 for pred in predicted if _fact_matches(pred, expected))
    precision = pred_hits / len(predicted) if predicted else 1.0

    return recall, precision


def _conversation_preview(messages: list[dict]) -> str:
    return " | ".join(m["content"][:50] for m in messages)


def _convo_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content'].strip()}" for m in messages)


def _first_json_array(raw: str) -> str | None:
    start = raw.find("[")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for j in range(start, len(raw)):
        ch = raw[j]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return raw[start:j + 1]
    return None


# ── SmolLM2 side (exact prompt from memorize.py) ────────────────────────────

def smollm_extract(client, model: str, messages: list[dict]) -> tuple[list[str], float, str]:
    convo = _convo_text(messages)
    prompt = EXTRACT_PROMPT.format(conversation=convo)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0,
        timeout=SMOLLM_TIMEOUT,
        stop=["\n\n", "```"],
    )
    elapsed = time.perf_counter() - t0
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    array_str = _first_json_array(raw) or raw
    try:
        facts = json.loads(array_str)
        facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        facts = []

    facts = [f for f in facts if not _HEDGE_RE.search(f)]
    return facts, elapsed, raw


# ── Needle side: native JAX backend (single tool-call framing) ─────────────

class NeedleRunner:
    def __init__(self, checkpoint_path: str):
        from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer
        self._generate = generate
        params, config = load_checkpoint(checkpoint_path)
        self.model = SimpleAttentionNetwork(config)
        self.params = params
        self.tokenizer = get_tokenizer()

    def call(self, query: str, tools: list[dict]) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        raw = self._generate(
            self.model, self.params, self.tokenizer,
            query=query, tools=json.dumps(tools), stream=False,
        )
        elapsed = time.perf_counter() - t0
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        return parsed, elapsed


# ── Needle side: ONNX backend (onnx-community/needle-onnx) ─────────────────
#
# Ported from needle-onnx/verify_parity.py. See bench_intent_routing.py's
# ONNXNeedleRunner docstring for the full rationale (why ONNX over JAX on
# Jetson, verified parity numbers) -- kept brief here to avoid duplicating
# it twice. The two things that came out of the reference implementation:
#   1. Input assembly is imported directly from the upstream Cactus package
#      (`needle.model.run._build_encoder_input`), not reimplemented --
#      requires `git clone https://github.com/cactus-compute/needle.git external/needle`.
#   2. Decode loop seeds the decoder with EOS, starts past_self_kv as a
#      zero-length cache, and stops on re-predicting EOS.

class ONNXNeedleRunner:
    SANITY_QUERY = "set a 5 min timer"
    SANITY_TOOLS = [{
        "name": "set_timer",
        "description": "Set a timer.",
        "parameters": {"time_human": {"type": "string", "description": "duration"}},
    }]
    SANITY_EXPECTED = [{"name": "set_timer", "arguments": {"time_human": "5 minutes"}}]

    MAX_ENC_LEN = 1024
    MAX_GEN_LEN = 64

    D_MODEL = 512
    NUM_HEADS = 8
    NUM_KV_HEADS = 4
    NUM_DECODER_LAYERS = 8

    def __init__(self, model_dir: str = DEFAULT_ONNX_DIR, cactus_repo_dir: str = DEFAULT_CACTUS_REPO_DIR):
        import onnxruntime as ort

        cactus_path = str(Path(cactus_repo_dir).resolve())
        if cactus_path not in sys.path:
            sys.path.insert(0, cactus_path)
        from needle.model.run import _build_encoder_input
        from needle.dataset.tokenizer import get_tokenizer

        self._build_encoder_input = _build_encoder_input
        self.tokenizer = get_tokenizer()

        self.model_dir = model_dir
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=providers)
        self.decoder_step = ort.InferenceSession(f"{model_dir}/decoder_step.onnx", providers=providers)
        self.head_dim = self.D_MODEL // self.NUM_HEADS

    def _build_input_ids(self, query: str, tools: list[dict]) -> list[int]:
        tools_json = json.dumps(tools)
        return self._build_encoder_input(self.tokenizer, query, tools_json, max_enc_len=self.MAX_ENC_LEN)

    def call(self, query: str, tools: list[dict]) -> tuple[list[dict], float]:
        import numpy as np
        t0 = time.perf_counter()

        enc_tokens = self._build_input_ids(query, tools)
        enc_input = np.array([enc_tokens], dtype=np.int64)
        encoder_out = self.encoder.run(None, {"input_ids": enc_input})[0]

        past_kv = np.zeros(
            (self.NUM_DECODER_LAYERS, 2, 1, self.NUM_KV_HEADS, 0, self.head_dim),
            dtype=np.float32,
        )

        eos_id = self.tokenizer.eos_token_id
        next_id = eos_id
        generated: list[int] = []

        for _ in range(self.MAX_GEN_LEN):
            logits, past_kv = self.decoder_step.run(None, {
                "decoder_input_ids": np.array([[next_id]], dtype=np.int64),
                "encoder_out": encoder_out,
                "past_self_kv": past_kv,
            })
            next_id = int(np.argmax(logits[0, 0]))
            if next_id == eos_id:
                break
            generated.append(next_id)

        elapsed = time.perf_counter() - t0

        text = self.tokenizer.decode(generated)
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        text = text.strip()

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        return parsed, elapsed

    def sanity_check(self) -> bool:
        """Structural check, not byte-exact match -- see bench_intent_routing.py's
        ONNXNeedleRunner.sanity_check docstring for the full rationale (upstream
        checkpoint version drift confirmed via HfApi().model_info, plus a ~0.7
        nat logit margin at the divergence point ruling out float32 numeric
        drift as the cause)."""
        try:
            calls, _ = self.call(self.SANITY_QUERY, self.SANITY_TOOLS)
        except Exception as e:
            print(f"[ONNXNeedleRunner] sanity check FAILED with exception: {e}")
            return False
        ok = bool(calls) and calls[0].get("name") == "set_timer" and bool(calls[0].get("arguments"))
        print(f"[ONNXNeedleRunner] sanity check (structural): {'PASS' if ok else 'FAIL'}")
        print(f"  query: {self.SANITY_QUERY!r}")
        print(f"  got:   {calls}")
        if not ok:
            print("  (exact byte-for-byte match with the model card isn't required -- "
                  "see docstring on checkpoint version drift)")
        return ok


NEEDLE_EXTRACT_TOOLS = [{
    "name": "record_facts",
    "description": "Record a list of short atomic facts extracted from the conversation.",
    "parameters": {"facts": {"type": "string", "description": "facts joined by semicolons"}},
}]


def needle_extract(runner, messages: list[dict]) -> tuple[list[str], float, str]:
    # Needle wants a single query string, not a multi-turn transcript — this
    # is already outside its documented single-shot-command scope, so we
    # give it the flattened conversation and see what it does with it.
    convo = _convo_text(messages)
    calls, elapsed = runner.call(convo, NEEDLE_EXTRACT_TOOLS)
    raw = json.dumps(calls)
    if not calls:
        return [], elapsed, raw
    facts_str = calls[0].get("arguments", {}).get("facts", "")
    # Best-effort split since the model has no native array output for this arg.
    facts = [f.strip(" -\"'") for f in re.split(r"[;\n]|(?<=[a-z])\.\s+", facts_str) if f.strip(" -\"'")]
    facts = [f for f in facts if not _HEDGE_RE.search(f)]
    return facts, elapsed, raw


# ── runners ───────────────────────────────────────────────────────────────

def run_smollm_suite(client, model: str) -> SuiteResult:
    s = SuiteResult(f"memory_extraction (smollm2 / {model})")
    for messages, expected in CASES:
        facts, elapsed, raw = smollm_extract(client, model, messages)
        recall, precision = score_facts(expected, facts)
        s.results.append(CaseResult(_conversation_preview(messages), expected, facts, recall, precision, elapsed, raw))
    return s


def run_needle_suite(runner) -> SuiteResult:
    s = SuiteResult("memory_extraction (needle)")
    for messages, expected in CASES:
        facts, elapsed, raw = needle_extract(runner, messages)
        recall, precision = score_facts(expected, facts)
        s.results.append(CaseResult(_conversation_preview(messages), expected, facts, recall, precision, elapsed, raw))
    return s


def print_report(suites: list[SuiteResult]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Suite':<38} {'Recall':>8} {'Precision':>10} {'Avg latency':>14} {'N':>5}")
    print("=" * 90)
    for s in suites:
        print(f"{s.name:<38} {s.avg_recall*100:>7.1f}% {s.avg_precision*100:>9.1f}% {s.avg_latency*1000:>11.1f} ms {len(s.results):>5}")
    print("=" * 90)

    for s in suites:
        print(f"\n── {s.name} — per-case detail ──")
        for r in s.results:
            print(f"  conversation: {r.conversation_preview}")
            print(f"    expected:  {r.expected}")
            print(f"    predicted: {r.predicted}")
            print(f"    recall={r.recall:.2f} precision={r.precision:.2f}")
            print(f"    raw: {r.raw[:200]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["needle", "smollm"], default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_NEEDLE_CHECKPOINT,
                     help="Needle JAX checkpoint path (--backend jax only)")
    ap.add_argument("--backend", choices=["jax", "onnx"], default="jax",
                     help="Needle backend to benchmark.")
    ap.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR,
                     help="Directory containing encoder.onnx/decoder_step.onnx (--backend onnx only)")
    ap.add_argument("--cactus-repo-dir", default=DEFAULT_CACTUS_REPO_DIR,
                     help="Path to the cloned cactus-compute/needle repo (--backend onnx only)")
    ap.add_argument("--smollm-model", default=DEFAULT_SMOLLM_MODEL,
                     help="llama-server alias to test — default 'smollm', pass 'ministral' to benchmark your current extractor as a baseline")
    args = ap.parse_args()

    suites: list[SuiteResult] = []

    if args.only in (None, "smollm"):
        print(f"Running SmolLM2 ({args.smollm_model}) extraction suite against {SMOLLM_BASE_URL} ...")
        try:
            from openai import OpenAI
            client = OpenAI(base_url=SMOLLM_BASE_URL, api_key="not-needed")
            suites.append(run_smollm_suite(client, args.smollm_model))
        except Exception as e:
            print(f"SmolLM2 suite failed: {e}")

    if args.only in (None, "needle"):
        if args.backend == "onnx":
            print(f"Loading Needle ONNX backend from {args.onnx_dir} (cactus repo: {args.cactus_repo_dir}) ...")
            try:
                runner = ONNXNeedleRunner(args.onnx_dir, args.cactus_repo_dir)
                if not runner.sanity_check():
                    print("ONNX sanity check did not pass -- skipping needle suite "
                          "(see ONNXNeedleRunner docstring).")
                else:
                    suites.append(run_needle_suite(runner))
            except Exception as e:
                print(f"Needle ONNX suite failed: {e}")
        else:
            print(f"Loading Needle checkpoint from {args.checkpoint} ...")
            try:
                runner = NeedleRunner(args.checkpoint)
                suites.append(run_needle_suite(runner))
            except Exception as e:
                print(f"Needle suite failed: {e}")

    if not suites:
        print("No suites ran — check errors above.")
        return

    print_report(suites)


if __name__ == "__main__":
    main()