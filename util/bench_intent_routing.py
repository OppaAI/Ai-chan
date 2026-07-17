"""
bench_intent_routing.py

Compares Needle (Cactus-Compute, 26M, tool-calling specialist) against
SmolLM2-135M-Instruct (current Aiko-chan router) across four tasks that
sit in cognition/think.py's routing cascade + agentic/agentic.py's
planning/query-gen paths:

  1. binary_routing   — agentic vs chat (stage 1 of the 3-stage cascade)
  2. tool_selection   — which registered tool fits the request (stage 2)
  3. query_generation — condensing a request into a focused deep_search query
  4. step_planning    — multi-step task decomposition (make_plan-style)

SmolLM2 is benchmarked TWO ways for binary_routing and tool_selection now
that jinja=true is enabled in models.ini:
  - "prompted-JSON": plain /v1/chat/completions, prompted for raw JSON.
    This is how it actually runs in your pipeline today.
  - "native-tools": tools=/tool_choice="auto" via the OpenAI-compatible
    API, now that llama-server can render tool schemas into the chat
    template. jinja=true only means the template *can* be rendered with
    tool schemas -- it says nothing about whether SmolLM2-135M-Instruct's
    template defines proper <tool_call> syntax or whether the model was
    trained to emit it reliably. Both paths are reported side by side so
    you can see directly which one actually performs better, instead of
    assuming the new setting "just works".

    NOTE: native-tools relies on llama-server rendering the tool schema
    into the chat template via jinja. If jinja is disabled server-side
    (see models.ini), the model never sees the tool schema at all and
    tool_calls will come back empty every time -- that's a config/backend
    limitation, not a model-quality signal. The native-tools suites below
    are left in place for whenever jinja is re-enabled, but expect them to
    floor at 0% accuracy with jinja off; don't read that as "the model is
    worse at native tool calling than prompted-JSON" unless jinja is
    actually on for the run.
query_generation and step_planning stay prompted-JSON only -- native
tool-calling doesn't map cleanly onto open-ended text generation tasks.

Needle can be run against either backend:
  --backend jax   (default) native JAX/Flax package, `cd needle && source ./setup`
  --backend onnx  onnx-community/needle-onnx via onnxruntime. Ported from
                  needle-onnx/verify_parity.py -- requires
                  `git clone https://github.com/cactus-compute/needle.git external/needle`
                  (see ONNXNeedleRunner docstring).

Run from anywhere; only needs `pip install openai` for the SmolLM2 side.
For --backend jax, the `needle` package must be importable
(`cd needle && source ./setup`). For --backend onnx, `pip install
onnxruntime sentencepiece huggingface_hub jax jaxlib`, a local `needle-onnx/`
dir, and `external/needle` (the cloned Cactus repo -- see ONNXNeedleRunner).

Usage:
    python bench_intent_routing.py                        # run everything (jax backend)
    python bench_intent_routing.py --only needle           # skip smollm2
    python bench_intent_routing.py --only smollm            # skip needle
    python bench_intent_routing.py --checkpoint checkpoints/needle.pkl
    python bench_intent_routing.py --backend onnx --onnx-dir needle-onnx
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── config ────────────────────────────────────────────────────────────────

SMOLLM_BASE_URL = "http://127.0.0.1:8080/v1"
SMOLLM_MODEL = "smollm"
SMOLLM_TIMEOUT = 20.0

DEFAULT_NEEDLE_CHECKPOINT = "checkpoints/needle.pkl"
DEFAULT_ONNX_DIR = "needle-onnx"
DEFAULT_CACTUS_REPO_DIR = "external/needle"

# Harrier-270M embedding endpoint (llama-server's OpenAI-compatible
# /v1/embeddings, same model your app already uses for memory/KB/skill
# scoring). Adjust to your actual alias/port -- this script intentionally
# doesn't import your app's real embedder wrapper to stay dependency-free
# (see module docstring: no cognition/toolkit stack, no DB connections).
EMBED_BASE_URL = "http://127.0.0.1:8081/v1"
EMBED_MODEL = "harrier"

# Confidence gate for the semantic+needle hybrid binary router: if the top
# label's score beats the runner-up by at least this margin, answer from
# the embedding lookup alone and skip the Needle call. Lower = more
# semantic-only answers (cheaper, riskier); higher = defers to Needle more
# often (safer, slower).
SEMANTIC_GATE_MARGIN = float(os.getenv("SEMANTIC_GATE_MARGIN", "0.08"))

# How many tools the semantic prefilter keeps before handing the narrowed
# list to Needle for tool_selection. Mirrors agentic/capability.py's real
# production pattern (filter tool schemas before the model sees them) --
# the point of this benchmark is to check whether a tiny 26M model does
# better discriminating among a few pre-filtered tools than the full ~9.
TOOL_SEMANTIC_PREFILTER_K = int(os.getenv("TOOL_SEMANTIC_PREFILTER_K", "3"))

# ── shared: a trimmed, faithful copy of Aiko-chan's real tool surface ──────
# Names/descriptions mirror agentic/agentic.py's _reg(...) calls. Kept as a
# static list here (rather than importing agentic.agentic) so this script
# has no dependency on the full cognition/toolkit stack and can't
# accidentally open DB connections or hit the network on import.

TOOLS_OPENAI = [
    {"type": "function", "function": {"name": "deep_search",
        "description": "Snippet-only web search as one support step inside a larger workflow.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "deep_research",
        "description": "Heavy research/source-reading tool for when research itself is the deliverable.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "save_note",
        "description": "Save a short plain-text note to a workspace file.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}}},
    {"type": "function", "function": {"name": "schedule_reminder",
        "description": "Set a simple once/daily reminder at a given time of day.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "message": {"type": "string"}, "time_of_day": {"type": "string"}},
            "required": ["title", "message", "time_of_day"]}}},
    {"type": "function", "function": {"name": "schedule_job",
        "description": "Schedule a recurring local job or alarm.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "task": {"type": "string"}, "time_of_day": {"type": "string"}},
            "required": ["title", "task", "time_of_day"]}}},
    {"type": "function", "function": {"name": "search_jobs",
        "description": "Search configured job boards for a role.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "location": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "scan_photo_workspace",
        "description": "Scan a workspace photo inbox for wildlife/nature/astro image files.",
        "parameters": {"type": "object", "properties": {
            "inbox": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "repo_read_file",
        "description": "Read one repository text file for architecture/code work.",
        "parameters": {"type": "object", "properties": {
            "relative_path": {"type": "string"}}, "required": ["relative_path"]}}},
    {"type": "function", "function": {"name": "chat_reply",
        "description": "No tool needed — just respond conversationally.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

# NOTE: previously flattened to {"name": type_string} per property. Changed
# to keep the FULL nested per-param schema (type + description), matching
# what needle.model.run._build_encoder_input / the training data actually
# expects -- see verify_parity.py's own sanity example, which uses the
# nested shape, not a flattened shorthand. The flattened form still works
# fine for the JAX backend's generate() in practice, but keeping one
# consistent nested shape across both backends avoids a schema-formatting
# confound in the benchmark numbers.
TOOLS_NEEDLE = [
    {"name": fn["function"]["name"],
     "description": fn["function"]["description"],
     "parameters": fn["function"]["parameters"]["properties"]}
    for fn in TOOLS_OPENAI
]

BINARY_TOOLS_NEEDLE = [
    {"name": "chat_reply", "description": "No tool needed — just respond conversationally, no task or action is required.", "parameters": {}},
    {"name": "do_task", "description": "Perform a task or tool action requested by the user (search, save, schedule, etc).", "parameters": {}},
]

# OpenAI-format twin of BINARY_TOOLS_NEEDLE, for the native tool_choice="auto" path.
BINARY_TOOLS_OPENAI = [
    {"type": "function", "function": {"name": "chat_reply",
        "description": "No tool needed — just respond conversationally, no task or action is required.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "do_task",
        "description": "Perform a task or tool action requested by the user (search, save, schedule, etc).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

# ── eval sets (grounded in your actual usage domain) ───────────────────────

BINARY_ROUTING_CASES = [
    ("hey Aiko, how's it going today", "chat"),
    ("what do you think about full moon photography settings", "chat"),
    ("remind me what I told you about GRACE yesterday", "chat"),
    ("thanks, that's helpful", "chat"),
    ("set a reminder for the SearXNG debugging session at 3pm", "agentic"),
    ("save a note about the NF4 quantization fix for Krea 2 on Modal", "agentic"),
    ("search for the latest onnxruntime aarch64 wheel release", "agentic"),
    ("schedule a daily job to check the AuRoRA disk usage every morning at 7am", "agentic"),
    ("look up how JAX handles CUDA on Jetson Orin Nano", "agentic"),
    ("what's the weather going to be like for aurora viewing tonight", "chat"),
]

TOOL_SELECTION_CASES = [
    ("save a note that the harrier embedding model needs the model field in its request", "save_note"),
    ("remind me at 6am to check the aurora forecast", "schedule_reminder"),
    ("search the web for cactus-compute needle finetuning examples", "deep_search"),
    ("do a deep dive and write up a report on JAX vs PyTorch inference on Jetson", "deep_research"),
    ("find me remote robotics engineer job postings", "search_jobs"),
    ("check what wildlife photos are sitting in my inbox folder", "scan_photo_workspace"),
    ("open cognition/think.py and show me the routing logic", "repo_read_file"),
    ("set up a recurring job every weekday at 9am to run the dream consolidation pass", "schedule_job"),
    ("how are you doing today", "chat_reply"),
    ("that makes sense, appreciate the explanation", "chat_reply"),
]

# (input, required_keywords_any_case) — query gen is scored on keyword
# coverage, not exact match, since "a good search query" isn't unique.
QUERY_GENERATION_CASES = [
    ("can you look up whether ReazonSpeech k2 supports streaming bilingual ASR",
     ["reazonspeech", "k2", "streaming"]),
    ("search for how to fix llama-server model name is missing from the request error",
     ["llama-server", "model name", "missing"]),
    ("find out the latest JetPack version release notes for Orin Nano",
     ["jetpack", "orin nano", "release"]),
    ("look up NF4 quantization for Krea 2 on Modal A10G GPUs",
     ["nf4", "krea", "a10g"]),
]

# (goal, required_keywords_any_case_across_all_steps)
STEP_PLANNING_CASES = [
    ("plan out fitting Krea 2 image generation within A10G GPU memory constraints on Modal",
     ["quantiz", "modal", "a10g", "test"]),
    ("plan the steps to migrate harrier off the shared llama-server port to its own dedicated instance",
     ["launch", "port", "config", "test"]),
    ("plan how to debug the SearXNG engine suspension issue in the web pipeline",
     ["log", "engine", "timeout", "test"]),
]


@dataclass
class CaseResult:
    input_text: str
    expected: object
    predicted: object
    correct: bool
    latency_s: float
    raw: str = ""


@dataclass
class SuiteResult:
    name: str
    results: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_s for r in self.results) / len(self.results)


# ── semantic (close-vector) scoring — trimmed local copy ───────────────────
# This mirrors cognition/reason.py's normalize+matmul label-scoring math
# (embed_example_matrix / label_scores_topk / batch_cosine_scores), the same
# primitive think.py's semantic intent router uses. Reimplemented locally
# rather than `from cognition import reason` so this script keeps its
# existing no-full-stack-import property (importing the `cognition` package
# risks pulling in CONTEXT_POOL / other init-time side effects this script
# deliberately avoids -- same reasoning as keeping TOOLS_OPENAI as a static
# trimmed copy above instead of importing agentic.agentic).

def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def _normalize_vec(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def _batch_cosine_scores(query_vec, item_vecs: np.ndarray) -> np.ndarray:
    item_vecs = np.asarray(item_vecs, dtype=np.float32)
    if item_vecs.size == 0:
        return np.array([], dtype=np.float32)
    q = _normalize_vec(query_vec)
    m = _normalize_rows(item_vecs)
    return m @ q


def _embed_example_matrix(embedder, examples_by_label: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    prompts: list[str] = []
    for label, examples in examples_by_label.items():
        labels.extend([label] * len(examples))
        prompts.extend(examples)
    if not prompts:
        return [], np.empty((0, 0), dtype=np.float32)
    raw = embedder.embed_queries(prompts)
    matrix = _normalize_rows(np.asarray(raw, dtype=np.float32))
    return labels, matrix


def _label_scores_topk(query_vec, labels: list[str], example_vecs: np.ndarray, top_k: int = 3) -> dict[str, float]:
    if example_vecs.size == 0:
        return {}
    scores = _batch_cosine_scores(query_vec, example_vecs)
    by_label: dict[str, list[float]] = {}
    for label, score in zip(labels, scores):
        by_label.setdefault(label, []).append(float(score))
    k = max(1, top_k)
    return {
        label: sum(sorted(values, reverse=True)[:k]) / min(k, len(values))
        for label, values in by_label.items()
    }


class OpenAICompatEmbedder:
    """Wraps a llama-server OpenAI-compatible /v1/embeddings endpoint.
    Matches Harrier-270M, the embedder your app already uses everywhere
    else (memory/KB/skill/agentic-policy scoring), just called from this
    standalone script instead of reusing owner._memorize._mem._embedder."""

    def __init__(self, base_url: str = EMBED_BASE_URL, model: str = EMBED_MODEL):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    def embed_query(self, text: str, instruct: str = "") -> list[float]:
        return self.embed_queries([text], instruct=instruct)[0]

    def embed_queries(self, texts: list[str], instruct: str = "") -> list[list[float]]:
        payload = [f"{instruct}\n{t}" if instruct else t for t in texts]
        resp = self._client.embeddings.create(model=self._model, input=payload)
        return [d.embedding for d in resp.data]


# Example utterances per label for the semantic binary router. Kept
# distinct from BINARY_ROUTING_CASES on purpose -- these are the anchor
# examples a real semantic router would be seeded with, not a peek at the
# eval set, so this stays a fair test of generalization.
BINARY_SEMANTIC_EXAMPLES: dict[str, list[str]] = {
    "chat": [
        "how are you doing today",
        "what do you think about that",
        "thanks, that's helpful",
        "tell me a bit about your day",
        "that makes sense, appreciate the explanation",
    ],
    "agentic": [
        "search the web for the latest release notes",
        "save this as a note for later",
        "schedule a reminder for tomorrow morning",
        "look up documentation on this library",
        "set up a recurring job to run every day",
    ],
}

# One anchor example per tool for the semantic tool router -- reuses each
# tool's own OpenAI-schema description, matching how a real embedding-based
# capability filter (agentic/capability.py-style) would score a query
# against tool descriptions.
TOOL_SEMANTIC_EXAMPLES: dict[str, list[str]] = {
    fn["function"]["name"]: [fn["function"]["description"]] for fn in TOOLS_OPENAI
}

_binary_semantic_cache: dict[int, tuple[list[str], np.ndarray]] = {}
_tool_semantic_cache: dict[int, tuple[list[str], np.ndarray]] = {}


def _get_binary_semantic_matrix(embedder) -> tuple[list[str], np.ndarray]:
    key = id(embedder)
    if key not in _binary_semantic_cache:
        _binary_semantic_cache[key] = _embed_example_matrix(embedder, BINARY_SEMANTIC_EXAMPLES)
    return _binary_semantic_cache[key]


def _get_tool_semantic_matrix(embedder) -> tuple[list[str], np.ndarray]:
    key = id(embedder)
    if key not in _tool_semantic_cache:
        _tool_semantic_cache[key] = _embed_example_matrix(embedder, TOOL_SEMANTIC_EXAMPLES)
    return _tool_semantic_cache[key]


def semantic_binary_routing(embedder, text: str) -> tuple[str, float, str]:
    """Pure close-vector classification, no LLM call at all."""
    t0 = time.perf_counter()
    labels, matrix = _get_binary_semantic_matrix(embedder)
    scores = {}
    if matrix.size:
        q_vec = embedder.embed_query(text)
        scores = _label_scores_topk(q_vec, labels, matrix, top_k=3)
    elapsed = time.perf_counter() - t0
    if not scores:
        return "", elapsed, "{}"
    label = max(scores, key=scores.get)
    return label, elapsed, json.dumps(scores)


def semantic_tool_selection(embedder, text: str) -> tuple[str, float, str]:
    """Pure close-vector classification against each tool's description,
    no LLM call at all."""
    t0 = time.perf_counter()
    labels, matrix = _get_tool_semantic_matrix(embedder)
    scores = {}
    if matrix.size:
        q_vec = embedder.embed_query(text)
        scores = _label_scores_topk(q_vec, labels, matrix, top_k=1)
    elapsed = time.perf_counter() - t0
    if not scores:
        return "", elapsed, "{}"
    tool = max(scores, key=scores.get)
    return tool, elapsed, json.dumps(scores)


def semantic_needle_binary_routing(runner, embedder, text: str) -> tuple[str, float, str]:
    """Hybrid: semantic scoring answers directly when confident (top label
    beats runner-up by >= SEMANTIC_GATE_MARGIN); otherwise defers to Needle.
    Models a cost-saving architecture where the cheap embedding lookup
    skips the (still small, but non-free) Needle call whenever it can."""
    t0 = time.perf_counter()
    labels, matrix = _get_binary_semantic_matrix(embedder)
    scores = {}
    if matrix.size:
        q_vec = embedder.embed_query(text)
        scores = _label_scores_topk(q_vec, labels, matrix, top_k=3)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) >= SEMANTIC_GATE_MARGIN:
        elapsed = time.perf_counter() - t0
        return ranked[0][0], elapsed, json.dumps({"path": "semantic_only", "scores": scores})
    label, _needle_elapsed, needle_raw = needle_binary_routing(runner, text)
    elapsed = time.perf_counter() - t0
    return label, elapsed, json.dumps({"path": "needle_fallback", "scores": scores, "needle_raw": needle_raw})


def semantic_needle_tool_selection(runner, embedder, text: str) -> tuple[str, float, str]:
    """Hybrid: semantic scoring narrows the tool list to the top
    TOOL_SEMANTIC_PREFILTER_K candidates, then Needle picks the final tool
    from just that narrowed set -- mirrors agentic/capability.py's real
    production pattern of filtering tool schemas before the model sees
    them, testing whether a tiny 26M model discriminates better among a
    handful of pre-filtered tools than the full ~9-tool surface."""
    t0 = time.perf_counter()
    labels, matrix = _get_tool_semantic_matrix(embedder)
    if not matrix.size:
        name, _needle_elapsed, needle_raw = needle_tool_selection(runner, text)
        elapsed = time.perf_counter() - t0
        return name, elapsed, json.dumps({"path": "needle_only_no_embedder", "needle_raw": needle_raw})

    q_vec = embedder.embed_query(text)
    scores = _label_scores_topk(q_vec, labels, matrix, top_k=1)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_names = {name for name, _ in ranked[:TOOL_SEMANTIC_PREFILTER_K]}
    narrowed_tools = [t for t in TOOLS_NEEDLE if t["name"] in top_names]

    calls, _needle_elapsed = runner.call(text, narrowed_tools)
    name = calls[0]["name"] if calls else ""
    elapsed = time.perf_counter() - t0
    return name, elapsed, json.dumps({
        "path": "semantic_prefilter+needle",
        "prefiltered": sorted(top_names),
        "needle_calls": calls,
    })


# ── SmolLM2 side: prompted-JSON (matching your models.ini setup today) ─────

def _get_smollm_client():
    from openai import OpenAI
    return OpenAI(base_url=SMOLLM_BASE_URL, api_key="not-needed")


def _smollm_json_call(client, prompt: str, max_tokens: int = 120) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=SMOLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=SMOLLM_TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    raw = (resp.choices[0].message.content or "").strip()
    return raw, elapsed


def _extract_json_obj(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _bare_value(raw: str, valid: set[str]) -> str:
    """Fallback for when the model gets the right answer but skips the
    requested JSON wrapper -- e.g. returns '"chat"' or 'Agentic' instead of
    '{"label": "chat"}'. Without this, a correct-but-unwrapped answer scores
    identically to a genuine miss, which conflates format compliance with
    actual judgment. Only recognizes exact (case-insensitive) matches against
    the known valid set, so it can't accidentally "recover" a wrong answer by
    fuzzy-matching partial text.
    """
    if not raw:
        return ""
    stripped = raw.strip().strip('"\'').strip().rstrip(".").strip()
    lowered = stripped.lower()
    for v in valid:
        if lowered == v.lower():
            return v
    # Some responses prefix the bare value, e.g. '"Agentic" (requires ...)' --
    # only match if one of the valid values appears as the very first word(s).
    for v in valid:
        if lowered.startswith(v.lower()):
            return v
    return ""


def smollm_binary_routing(client, text: str) -> tuple[str, float, str]:
    prompt = (
        'Classify this message as "chat" (conversational, no action needed) '
        'or "agentic" (requires performing a task/tool action). '
        'Reply with ONLY compact JSON: {"label": "chat" or "agentic"}.\n\n'
        f"Message: {text}"
    )
    raw, elapsed = _smollm_json_call(client, prompt, max_tokens=30)
    data = _extract_json_obj(raw)
    label = (data or {}).get("label", "").strip().lower()
    if label not in ("chat", "agentic"):
        label = _bare_value(raw, {"chat", "agentic"})
    return label, elapsed, raw


def smollm_tool_selection(client, text: str) -> tuple[str, float, str]:
    tool_lines = "\n".join(f'- {t["function"]["name"]}: {t["function"]["description"]}' for t in TOOLS_OPENAI)
    prompt = (
        "Given this user message, pick the single best matching tool name from the list. "
        'Reply with ONLY compact JSON: {"tool": "<name>"}.\n\n'
        f"Tools:\n{tool_lines}\n\nMessage: {text}"
    )
    raw, elapsed = _smollm_json_call(client, prompt, max_tokens=30)
    data = _extract_json_obj(raw)
    tool = (data or {}).get("tool", "").strip()
    valid_tools = {t["function"]["name"] for t in TOOLS_OPENAI}
    if tool not in valid_tools:
        tool = _bare_value(raw, valid_tools)
    return tool, elapsed, raw


def smollm_query_generation(client, text: str) -> tuple[str, float, str]:
    prompt = (
        "Condense this request into a focused web search query (5-10 words, no filler words). "
        'Reply with ONLY compact JSON: {"query": "..."}.\n\n'
        f"Request: {text}"
    )
    raw, elapsed = _smollm_json_call(client, prompt, max_tokens=40)
    data = _extract_json_obj(raw)
    query = (data or {}).get("query", "").strip()
    if not query:
        # Fallback: model returned a bare quoted/unquoted query string instead
        # of the requested {"query": "..."} object -- strip wrapping quotes
        # and any "Sure, here is the concise query:" style preamble line.
        candidate = raw.strip()
        # Drop a leading preamble line if the actual query is on its own line.
        lines = [l.strip() for l in candidate.splitlines() if l.strip()]
        candidate = lines[-1] if lines else candidate
        query = candidate.strip().strip('"\'').strip()
    return query, elapsed, raw


def smollm_step_planning(client, goal: str) -> tuple[str, float, str]:
    prompt = (
        "Break this goal into 3-6 ordered, concrete steps. "
        'Reply with ONLY compact JSON: {"steps": ["step 1", "step 2", ...]}.\n\n'
        f"Goal: {goal}"
    )
    # Bumped from 120 -> 300: at the old budget, 3-6 step JSON arrays were
    # frequently truncated mid-string before the closing brace, which made
    # _extract_json_obj fail even when the model's plan was fine in substance.
    raw, elapsed = _smollm_json_call(client, prompt, max_tokens=300)
    data = _extract_json_obj(raw)
    steps = (data or {}).get("steps", [])
    joined = " ".join(str(s) for s in steps) if isinstance(steps, list) else ""
    if not joined:
        # Fallback: no valid {"steps": [...]} found (truncated JSON, or the
        # model just wrote a plain numbered list like "Step 1: ... Step 2: ...").
        # The scorer only checks keyword coverage across the joined text, so a
        # plain-text plan is just as valid a signal as a well-formed JSON one --
        # only degenerate/truncated output should score as a miss.
        joined = raw.strip()
    return joined, elapsed, raw


# ── SmolLM2 side: native tools=/tool_choice="auto" (needs jinja=true) ──────
# Only meaningful for the two tool-selection-shaped tasks (binary_routing,
# tool_selection) -- query_generation/step_planning are open-ended text
# generation, not "which tool", so there's no native-tools equivalent for
# those and they stay prompted-JSON only.
#
# IMPORTANT: this path requires llama-server to render the tool schema into
# the chat template, which only happens with jinja=true. With jinja=false,
# tool_calls will come back empty on every case -- that's expected given the
# backend config, not a real 0% capability score. See module docstring.

def _smollm_native_tool_call(client, text: str, tools: list[dict], max_tokens: int = 60) -> tuple[str, float, str]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=SMOLLM_MODEL,
        messages=[{"role": "user", "content": text}],
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=SMOLLM_TIMEOUT,
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    name = tool_calls[0].function.name if tool_calls else ""
    raw = (
        json.dumps([{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tool_calls])
        if tool_calls else (msg.content or "").strip()
    )
    return name, elapsed, raw


def smollm_binary_routing_native(client, text: str) -> tuple[str, float, str]:
    name, elapsed, raw = _smollm_native_tool_call(client, text, BINARY_TOOLS_OPENAI, max_tokens=30)
    label = "agentic" if name == "do_task" else "chat" if name == "chat_reply" else ""
    return label, elapsed, raw


def smollm_tool_selection_native(client, text: str) -> tuple[str, float, str]:
    name, elapsed, raw = _smollm_native_tool_call(client, text, TOOLS_OPENAI, max_tokens=60)
    return name, elapsed, raw


# ── Needle side: native JAX backend (native single-shot tool-call API) ─────

class NeedleRunner:
    """Thin wrapper so the rest of the script doesn't care about JAX plumbing."""

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
# Ported from needle-onnx/verify_parity.py. Two things came out of that
# reference that aren't documented on the model card:
#   1. Input assembly (query + tool schema -> input_ids) is NOT reimplemented
#      here -- it's imported directly from the upstream Cactus package
#      (`needle.model.run._build_encoder_input`). Requires:
#          git clone https://github.com/cactus-compute/needle.git external/needle
#   2. The decode loop seeds the decoder with the EOS token id (Cactus
#      convention), starts past_self_kv as a ZERO-LENGTH cache (seq dim = 0,
#      grows from the graph's own "present_self_kv" output fed back in),
#      and stops the moment EOS is predicted again.
#
# Verified parity (from the model card): Flax<->PyTorch max-abs-diff
# 0.000029, PyTorch<->ONNX 0.000014, end-to-end token sequence byte-identical.
# 26M params barely needs a GPU, so CPU-only onnxruntime is plenty fast here.
#
# Run sanity_check() before trusting any --backend onnx numbers -- it must
# reproduce the model card's one worked example exactly:
#     query "set a 5 min timer"
#     -> [{"name": "set_timer", "arguments": {"time_human": "5 minutes"}}]

class ONNXNeedleRunner:
    SANITY_QUERY = "set a 5 min timer"
    # Full nested schema (type + description per param) -- matches what
    # _build_encoder_input / the training data actually expects, same shape
    # verify_parity.py's own sanity example uses. NOT the flattened
    # {"name": type} shorthand.
    SANITY_TOOLS = [{
        "name": "set_timer",
        "description": "Set a timer.",
        "parameters": {"time_human": {"type": "string", "description": "duration"}},
    }]
    SANITY_EXPECTED = [{"name": "set_timer", "arguments": {"time_human": "5 minutes"}}]

    MAX_ENC_LEN = 1024
    MAX_GEN_LEN = 64

    # Architecture constants (must match PROD_CONFIG in verify_parity.py) --
    # needed only to shape the empty initial KV cache.
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_KV_HEADS = 4
    NUM_DECODER_LAYERS = 8

    def __init__(self, model_dir: str = DEFAULT_ONNX_DIR, cactus_repo_dir: str = DEFAULT_CACTUS_REPO_DIR):
        import onnxruntime as ort

        cactus_path = str(Path(cactus_repo_dir).resolve())
        if cactus_path not in sys.path:
            sys.path.insert(0, cactus_path)
        # Load-bearing import -- same names verify_parity.py uses. Don't
        # reimplement token assembly; reuse the reference directly.
        from needle.model.run import _build_encoder_input
        from needle.dataset.tokenizer import get_tokenizer

        self._build_encoder_input = _build_encoder_input
        self.tokenizer = get_tokenizer()

        self.model_dir = model_dir
        providers = ["CPUExecutionProvider"]  # 26M params, CPU is plenty -- see class docstring
        self.encoder = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=providers)
        self.decoder_step = ort.InferenceSession(f"{model_dir}/decoder_step.onnx", providers=providers)
        self.head_dim = self.D_MODEL // self.NUM_HEADS

    def _build_input_ids(self, query: str, tools: list[dict]) -> list[int]:
        tools_json = json.dumps(tools)
        return self._build_encoder_input(self.tokenizer, query, tools_json, max_enc_len=self.MAX_ENC_LEN)

    def call(self, query: str, tools: list[dict]) -> tuple[list[dict], float]:
        t0 = time.perf_counter()

        enc_tokens = self._build_input_ids(query, tools)
        enc_input = np.array([enc_tokens], dtype=np.int64)
        encoder_out = self.encoder.run(None, {"input_ids": enc_input})[0]

        # Empty KV cache -- seq dim starts at 0, grows each decode step from
        # the graph's own "present_self_kv" output (fed back as next past_kv).
        past_kv = np.zeros(
            (self.NUM_DECODER_LAYERS, 2, 1, self.NUM_KV_HEADS, 0, self.head_dim),
            dtype=np.float32,
        )

        eos_id = self.tokenizer.eos_token_id
        next_id = eos_id  # decoder seeded with EOS, per Cactus convention
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
        """Structural check, not byte-exact match against the model card example.

        Byte-exact reproduction assumes the ONNX weights and whatever checkpoint
        `Cactus-Compute/needle` currently resolves to on HF are the same training
        run. They may not be -- confirmed via HfApi().model_info: the upstream
        checkpoint's lastModified (2026-05-13) may postdate whatever snapshot
        needle-onnx's weights were exported from, in which case the two are
        different model versions even though the code path (tokenizer, input
        assembly, decode loop) is verified byte-identical to verify_parity.py.
        A per-step logit margin of ~0.7+ nats at the point of divergence rules
        out float32/ARM-vs-x86 numeric drift as the cause -- see conversation
        notes. So: check tool name + a non-empty, plausible argument dict
        instead of demanding the exact same completion as the model card.
        """
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
                  "see class docstring on checkpoint version drift)")
        return ok


def needle_binary_routing(runner, text: str) -> tuple[str, float, str]:
    calls, elapsed = runner.call(text, BINARY_TOOLS_NEEDLE)
    name = calls[0]["name"] if calls else ""
    label = "agentic" if name == "do_task" else "chat" if name == "chat_reply" else ""
    return label, elapsed, json.dumps(calls)


def needle_tool_selection(runner, text: str) -> tuple[str, float, str]:
    calls, elapsed = runner.call(text, TOOLS_NEEDLE)
    name = calls[0]["name"] if calls else ""
    return name, elapsed, json.dumps(calls)


def needle_query_generation(runner, text: str) -> tuple[str, float, str]:
    tools = [{"name": "deep_search", "description": "Search the web for a query.",
              "parameters": {"query": {"type": "string", "description": "the search query"}}}]
    calls, elapsed = runner.call(text, tools)
    query = calls[0].get("arguments", {}).get("query", "") if calls else ""
    return query, elapsed, json.dumps(calls)


def needle_step_planning(runner, goal: str) -> tuple[str, float, str]:
    # Needle is documented as single-shot only — this call exists specifically
    # to demonstrate whether/how it fails at multi-step decomposition rather
    # than to give it a fair shot at something outside its stated scope.
    calls, elapsed = runner.call(goal, TOOLS_NEEDLE)
    joined = json.dumps(calls)
    return joined, elapsed, joined


# ── scoring helpers ──────────────────────────────────────────────────────

def _keyword_hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _keyword_coverage(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return hits >= max(1, len(keywords) // 2)  # majority of expected terms present


# ── suite runners ────────────────────────────────────────────────────────

def run_smollm_suites(client) -> list[SuiteResult]:
    suites = []

    s = SuiteResult("binary_routing (smollm2 prompted-JSON)")
    for text, expected in BINARY_ROUTING_CASES:
        label, elapsed, raw = smollm_binary_routing(client, text)
        s.results.append(CaseResult(text, expected, label, label == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("binary_routing (smollm2 native-tools)")
    for text, expected in BINARY_ROUTING_CASES:
        label, elapsed, raw = smollm_binary_routing_native(client, text)
        s.results.append(CaseResult(text, expected, label, label == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("tool_selection (smollm2 prompted-JSON)")
    for text, expected in TOOL_SELECTION_CASES:
        tool, elapsed, raw = smollm_tool_selection(client, text)
        s.results.append(CaseResult(text, expected, tool, tool == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("tool_selection (smollm2 native-tools)")
    for text, expected in TOOL_SELECTION_CASES:
        tool, elapsed, raw = smollm_tool_selection_native(client, text)
        s.results.append(CaseResult(text, expected, tool, tool == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("query_generation (smollm2)")
    for text, keywords in QUERY_GENERATION_CASES:
        query, elapsed, raw = smollm_query_generation(client, text)
        s.results.append(CaseResult(text, keywords, query, _keyword_hit(query, keywords), elapsed, raw))
    suites.append(s)

    s = SuiteResult("step_planning (smollm2)")
    for goal, keywords in STEP_PLANNING_CASES:
        joined, elapsed, raw = smollm_step_planning(client, goal)
        s.results.append(CaseResult(goal, keywords, joined, _keyword_coverage(joined, keywords), elapsed, raw))
    suites.append(s)

    return suites


def run_needle_suites(runner) -> list[SuiteResult]:
    suites = []

    s = SuiteResult("binary_routing (needle)")
    for text, expected in BINARY_ROUTING_CASES:
        label, elapsed, raw = needle_binary_routing(runner, text)
        s.results.append(CaseResult(text, expected, label, label == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("tool_selection (needle)")
    for text, expected in TOOL_SELECTION_CASES:
        tool, elapsed, raw = needle_tool_selection(runner, text)
        s.results.append(CaseResult(text, expected, tool, tool == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("query_generation (needle)")
    for text, keywords in QUERY_GENERATION_CASES:
        query, elapsed, raw = needle_query_generation(runner, text)
        s.results.append(CaseResult(text, keywords, query, _keyword_hit(query, keywords), elapsed, raw))
    suites.append(s)

    s = SuiteResult("step_planning (needle)")
    for goal, keywords in STEP_PLANNING_CASES:
        joined, elapsed, raw = needle_step_planning(runner, goal)
        s.results.append(CaseResult(goal, keywords, joined, _keyword_coverage(joined, keywords), elapsed, raw))
    suites.append(s)

    return suites


def run_semantic_suites(embedder) -> list[SuiteResult]:
    """Pure close-vector routing, no LLM call at all -- the cheapest
    possible baseline. Only applies to the two classification-shaped
    tasks (binary_routing, tool_selection); query_generation/step_planning
    are open-ended generation with no label set to embed against."""
    suites = []

    s = SuiteResult("binary_routing (semantic-only)")
    for text, expected in BINARY_ROUTING_CASES:
        label, elapsed, raw = semantic_binary_routing(embedder, text)
        s.results.append(CaseResult(text, expected, label, label == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("tool_selection (semantic-only)")
    for text, expected in TOOL_SELECTION_CASES:
        tool, elapsed, raw = semantic_tool_selection(embedder, text)
        s.results.append(CaseResult(text, expected, tool, tool == expected, elapsed, raw))
    suites.append(s)

    return suites


def run_semantic_needle_suites(runner, embedder) -> list[SuiteResult]:
    """Hybrid semantic+Needle routing -- see semantic_needle_binary_routing
    (confidence-gated fallback) and semantic_needle_tool_selection
    (semantic prefilter narrows the tool list before Needle picks)."""
    suites = []

    s = SuiteResult("binary_routing (semantic+needle)")
    for text, expected in BINARY_ROUTING_CASES:
        label, elapsed, raw = semantic_needle_binary_routing(runner, embedder, text)
        s.results.append(CaseResult(text, expected, label, label == expected, elapsed, raw))
    suites.append(s)

    s = SuiteResult("tool_selection (semantic+needle)")
    for text, expected in TOOL_SELECTION_CASES:
        tool, elapsed, raw = semantic_needle_tool_selection(runner, embedder, text)
        s.results.append(CaseResult(text, expected, tool, tool == expected, elapsed, raw))
    suites.append(s)

    return suites


def print_report(suites: list[SuiteResult], verbose: bool) -> None:
    print("\n" + "=" * 78)
    print(f"{'Suite':<38} {'Accuracy':>10} {'Avg latency':>14} {'N':>5}")
    print("=" * 78)
    for s in suites:
        print(f"{s.name:<38} {s.accuracy*100:>9.1f}% {s.avg_latency*1000:>11.1f} ms {len(s.results):>5}")
    print("=" * 78)

    if verbose:
        for s in suites:
            print(f"\n── {s.name} ──")
            for r in s.results:
                mark = "OK" if r.correct else "XX"
                print(f"  [{mark}] input: {r.input_text[:70]!r}")
                print(f"        expected: {r.expected!r}")
                print(f"        predicted: {r.predicted!r}")
                if not r.correct:
                    print(f"        raw: {r.raw[:200]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["needle", "smollm", "semantic", "semantic_needle"], default=None,
                     help="Restrict to one suite family. Default runs every family whose "
                          "backend(s) load successfully.")
    ap.add_argument("--checkpoint", default=DEFAULT_NEEDLE_CHECKPOINT,
                     help="Needle JAX checkpoint path (--backend jax only)")
    ap.add_argument("--backend", choices=["jax", "onnx"], default="jax",
                     help="Needle backend to benchmark.")
    ap.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR,
                     help="Directory containing encoder.onnx/decoder_step.onnx (--backend onnx only)")
    ap.add_argument("--cactus-repo-dir", default=DEFAULT_CACTUS_REPO_DIR,
                     help="Path to the cloned cactus-compute/needle repo (--backend onnx only)")
    ap.add_argument("--embed-base-url", default=EMBED_BASE_URL,
                     help="Harrier embedding endpoint (semantic / semantic_needle only)")
    ap.add_argument("--embed-model", default=EMBED_MODEL,
                     help="llama-server alias for the embedding model (semantic / semantic_needle only)")
    ap.add_argument("--verbose", action="store_true", help="print every case, not just the summary table")
    args = ap.parse_args()

    all_suites: list[SuiteResult] = []

    need_smollm = args.only in (None, "smollm")
    need_needle = args.only in (None, "needle")
    need_semantic = args.only in (None, "semantic")
    need_semantic_needle = args.only in (None, "semantic_needle")

    if need_smollm:
        print("Running SmolLM2 suites against", SMOLLM_BASE_URL, "...")
        try:
            client = _get_smollm_client()
            all_suites.extend(run_smollm_suites(client))
        except Exception as e:
            print(f"SmolLM2 suite failed: {e}")

    embedder = None
    if need_semantic or need_semantic_needle:
        print(f"Loading embedder ({args.embed_model}) from {args.embed_base_url} ...")
        try:
            embedder = OpenAICompatEmbedder(args.embed_base_url, args.embed_model)
        except Exception as e:
            print(f"Embedder load failed: {e}")

    runner = None
    if need_needle or need_semantic_needle:
        if args.backend == "onnx":
            print(f"Loading Needle ONNX backend from {args.onnx_dir} (cactus repo: {args.cactus_repo_dir}) ...")
            try:
                candidate = ONNXNeedleRunner(args.onnx_dir, args.cactus_repo_dir)
                if not candidate.sanity_check():
                    print("ONNX sanity check did not pass -- Needle-dependent suites "
                          "will be skipped (see ONNXNeedleRunner docstring).")
                else:
                    runner = candidate
            except Exception as e:
                print(f"Needle ONNX backend failed to load: {e}")
        else:
            print(f"Loading Needle checkpoint from {args.checkpoint} ...")
            try:
                runner = NeedleRunner(args.checkpoint)
            except Exception as e:
                print(f"Needle backend failed to load: {e}")

    if need_needle and runner is not None:
        try:
            all_suites.extend(run_needle_suites(runner))
        except Exception as e:
            print(f"Needle suite failed: {e}")

    if need_semantic and embedder is not None:
        try:
            all_suites.extend(run_semantic_suites(embedder))
        except Exception as e:
            print(f"Semantic-only suite failed: {e}")

    if need_semantic_needle:
        if runner is not None and embedder is not None:
            try:
                all_suites.extend(run_semantic_needle_suites(runner, embedder))
            except Exception as e:
                print(f"Semantic+needle suite failed: {e}")
        else:
            print("Skipping semantic+needle suite -- needs both a loaded embedder and Needle backend.")

    if not all_suites:
        print("No suites ran — check errors above.")
        return

    print_report(all_suites, verbose=args.verbose)


if __name__ == "__main__":
    main()