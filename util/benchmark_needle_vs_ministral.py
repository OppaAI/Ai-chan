"""
Benchmark Cactus Needle 26M (ONNX) vs Ministral3-3b (llama.cpp server) on
intent/tool-call routing: same queries + tool schemas, compare tool-name
accuracy, argument accuracy, and latency.

Setup:
    pip install onnxruntime sentencepiece huggingface_hub numpy requests

Assumes:
    - needle_onnx_infer.py is in the same directory (imports its helpers)
    - llama.cpp server is already running with Ministral3-3b loaded,
      exposing OpenAI-compatible /v1/chat/completions with `tools` param

Adjust MINISTRAL_URL / MINISTRAL_MODEL_NAME below to match your setup.
"""
import json
import time

import requests

from needle_onnx_infer import (
    download_artifacts,
    load_sessions,
    get_tokenizer,
    run_needle,
)

MINISTRAL_URL = "http://localhost:8080/v1/chat/completions"
MINISTRAL_MODEL_NAME = "ministral3-3b"  # adjust to whatever your server expects, if anything

# --- Canonical test cases -------------------------------------------------
# Each case: query, tool schema(s), and the expected tool name / key args.
# Extend this list with cases from your real routing traffic for a
# meaningful benchmark rather than relying on these samples alone.
TEST_CASES = [
    {
        "query": "set a 5 min timer",
        "tools": [
            {
                "name": "set_timer",
                "description": "Set a timer.",
                "parameters": {
                    "time_human": {"type": "string", "description": "duration", "required": True}
                },
            }
        ],
        "expected_tool": "set_timer",
    },
    {
        "query": "what's the weather like in Chilliwack today",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "location": {"type": "string", "description": "City name.", "required": True}
                },
            }
        ],
        "expected_tool": "get_weather",
    },
    {
        "query": "remind me to check the rover battery in an hour",
        "tools": [
            {
                "name": "set_reminder",
                "description": "Create a reminder for a future time.",
                "parameters": {
                    "text": {"type": "string", "description": "Reminder content.", "required": True},
                    "when": {"type": "string", "description": "When to remind, relative or absolute.", "required": True},
                },
            }
        ],
        "expected_tool": "set_reminder",
    },
    {
        "query": "how's it going",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {"location": {"type": "string", "description": "City name.", "required": True}},
            }
        ],
        # No tool call expected -- tests false-positive rate on chit-chat.
        "expected_tool": None,
    },
]


def needle_tools_to_openai_format(tools):
    """Convert Needle's flat tool schema into OpenAI-style `tools` param."""
    out = []
    for t in tools:
        props = {}
        required = []
        for pname, pinfo in t["parameters"].items():
            props[pname] = {"type": pinfo["type"], "description": pinfo.get("description", "")}
            if pinfo.get("required"):
                required.append(pname)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        )
    return out


def run_ministral(query, tools, url=MINISTRAL_URL, model=MINISTRAL_MODEL_NAME):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "tools": needle_tools_to_openai_format(tools),
        "temperature": 0,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=30)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    tool_calls = data["choices"][0]["message"].get("tool_calls")
    if not tool_calls:
        return None, {}, latency

    call = tool_calls[0]["function"]
    name = call["name"]
    try:
        args = json.loads(call["arguments"])
    except (json.JSONDecodeError, TypeError):
        args = {}
    return name, args, latency


def run_needle_case(query, tools, enc_sess, dec_sess, tokenizer):
    tools_json = json.dumps(tools)
    t0 = time.perf_counter()
    raw = run_needle(query, tools_json, enc_sess, dec_sess, tokenizer)
    latency = time.perf_counter() - t0

    try:
        parsed = json.loads(raw)
        if parsed:
            return parsed[0]["name"], parsed[0].get("arguments", {}), latency
    except (json.JSONDecodeError, IndexError, TypeError, KeyError):
        pass
    return None, {}, latency


def main():
    paths = download_artifacts()
    enc_sess, dec_sess = load_sessions(paths)
    tokenizer = get_tokenizer(paths["needle.model"])

    results = []
    for case in TEST_CASES:
        query, tools, expected = case["query"], case["tools"], case["expected_tool"]

        n_name, n_args, n_lat = run_needle_case(query, tools, enc_sess, dec_sess, tokenizer)
        m_name, m_args, m_lat = run_ministral(query, tools)

        results.append(
            {
                "query": query,
                "expected": expected,
                "needle": {"tool": n_name, "args": n_args, "latency_ms": round(n_lat * 1000, 1), "correct": n_name == expected},
                "ministral": {"tool": m_name, "args": m_args, "latency_ms": round(m_lat * 1000, 1), "correct": m_name == expected},
            }
        )

    # --- report ---
    needle_correct = sum(r["needle"]["correct"] for r in results)
    ministral_correct = sum(r["ministral"]["correct"] for r in results)
    n = len(results)

    print(f"\n{'Query':<45} {'Expected':<15} {'Needle':<25} {'Ministral':<25}")
    print("-" * 115)
    for r in results:
        needle_str = f"{r['needle']['tool']} ({r['needle']['latency_ms']}ms)"
        ministral_str = f"{r['ministral']['tool']} ({r['ministral']['latency_ms']}ms)"
        print(f"{r['query']:<45} {str(r['expected']):<15} {needle_str:<25} {ministral_str:<25}")

    print(f"\nNeedle tool-name accuracy:    {needle_correct}/{n}")
    print(f"Ministral tool-name accuracy: {ministral_correct}/{n}")

    avg_needle_lat = sum(r["needle"]["latency_ms"] for r in results) / n
    avg_ministral_lat = sum(r["ministral"]["latency_ms"] for r in results) / n
    print(f"\nAvg Needle latency:    {avg_needle_lat:.1f} ms")
    print(f"Avg Ministral latency: {avg_ministral_lat:.1f} ms")


if __name__ == "__main__":
    main()
