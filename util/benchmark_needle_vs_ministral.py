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

LLM_BASE_URL = "http://localhost:8080/v1"  # OpenAI-compatible chat endpoint, usually llama.cpp /v1 URL.
MINISTRAL_URL = f"{LLM_BASE_URL}/chat/completions"
MINISTRAL_MODEL_NAME = "ministral"  # matches the model alias/name your llama.cpp server expects

# --- Canonical test cases -------------------------------------------------
# Each case: query, tool schema(s), and the expected tool name / key args.
# Mix of: clean single-arg calls, multi-arg calls, paraphrases of the same
# intent (robustness check), ambiguous phrasing, and pure chit-chat (no
# tool call expected -- tests false-positive rate).
#
# Swap these for real (query, expected_tool) pairs from Aiko's routing logs
# once you have a batch -- these are still synthetic stand-ins.

TIMER_TOOL = {
    "name": "set_timer",
    "description": "Set a timer.",
    "parameters": {"time_human": {"type": "string", "description": "duration", "required": True}},
}
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {"location": {"type": "string", "description": "City name.", "required": True}},
}
REMINDER_TOOL = {
    "name": "set_reminder",
    "description": "Create a reminder for a future time.",
    "parameters": {
        "text": {"type": "string", "description": "Reminder content.", "required": True},
        "when": {"type": "string", "description": "When to remind, relative or absolute.", "required": True},
    },
}
NOTE_TOOL = {
    "name": "save_note",
    "description": "Save a short note for later.",
    "parameters": {
        "title": {"type": "string", "description": "Short title for the note.", "required": True},
        "body": {"type": "string", "description": "Note content.", "required": True},
    },
}
SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information.",
    "parameters": {"query": {"type": "string", "description": "Search query.", "required": True}},
}
MUSIC_TOOL = {
    "name": "play_music",
    "description": "Play music, optionally by artist or genre.",
    "parameters": {
        "query": {"type": "string", "description": "Song, artist, or genre.", "required": True},
    },
}
NO_ACTION_TOOL = {
    "name": "no_action",
    "description": "Use this when the user is making conversation, chit-chatting, or no other tool applies. No parameters needed.",
    "parameters": {},
}

_RAW_TEST_CASES = [
    # --- clean single-arg tool calls ---
    {"query": "set a 5 min timer", "tools": [TIMER_TOOL], "expected_tool": "set_timer"},
    {"query": "timer for 10 minutes please", "tools": [TIMER_TOOL], "expected_tool": "set_timer"},
    {"query": "can you start a 20 second timer", "tools": [TIMER_TOOL], "expected_tool": "set_timer"},
    {"query": "what's the weather like in Chilliwack today", "tools": [WEATHER_TOOL], "expected_tool": "get_weather"},
    {"query": "is it raining in Vancouver", "tools": [WEATHER_TOOL], "expected_tool": "get_weather"},
    {"query": "weather forecast for tomorrow in Abbotsford", "tools": [WEATHER_TOOL], "expected_tool": "get_weather"},
    {"query": "search for the latest jetson orin nano firmware release notes", "tools": [SEARCH_TOOL], "expected_tool": "web_search"},
    {"query": "look up the current price of RTX 3060", "tools": [SEARCH_TOOL], "expected_tool": "web_search"},
    {"query": "play some lofi music", "tools": [MUSIC_TOOL], "expected_tool": "play_music"},
    {"query": "put on some Radiohead", "tools": [MUSIC_TOOL], "expected_tool": "play_music"},

    # --- multi-arg tool calls (the suspected weak point) ---
    {"query": "remind me to check the rover battery in an hour", "tools": [REMINDER_TOOL], "expected_tool": "set_reminder"},
    {"query": "remind me tomorrow at 9am to submit the hackathon writeup", "tools": [REMINDER_TOOL], "expected_tool": "set_reminder"},
    {"query": "set a reminder for the vet appointment next Tuesday", "tools": [REMINDER_TOOL], "expected_tool": "set_reminder"},
    {"query": "save a note titled grocery list with milk eggs and bread", "tools": [NOTE_TOOL], "expected_tool": "save_note"},
    {"query": "jot down a note about the harrier embedding memory bug", "tools": [NOTE_TOOL], "expected_tool": "save_note"},

    # --- ambiguous / requires disambiguation between multiple tools ---
    {
        "query": "remind me about the weather tomorrow",
        "tools": [WEATHER_TOOL, REMINDER_TOOL],
        "expected_tool": "set_reminder",
    },
    {
        "query": "what's Chilliwack like this weekend",
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
        "expected_tool": "get_weather",
    },
    {
        "query": "find me some new music to listen to",
        "tools": [MUSIC_TOOL, SEARCH_TOOL],
        "expected_tool": "web_search",
    },

    # --- pure chit-chat / small talk (no tool call expected) ---
    {"query": "how's it going", "tools": [WEATHER_TOOL], "expected_tool": None},
    {"query": "you're pretty smart, aren't you", "tools": [WEATHER_TOOL], "expected_tool": None},
    {"query": "tell me a joke", "tools": [TIMER_TOOL], "expected_tool": None},
    {"query": "what do you think about robots taking over the world", "tools": [SEARCH_TOOL], "expected_tool": None},
    {"query": "good morning", "tools": [TIMER_TOOL, WEATHER_TOOL], "expected_tool": None},
    {"query": "thanks for the help earlier", "tools": [REMINDER_TOOL], "expected_tool": None},
    {"query": "I'm bored", "tools": [MUSIC_TOOL], "expected_tool": None},
    {"query": "what's your favorite animal", "tools": [SEARCH_TOOL], "expected_tool": None},

    # --- near-miss chit-chat (mentions a tool-adjacent word without asking for the tool) ---
    {"query": "I hate when timers go off too early", "tools": [TIMER_TOOL], "expected_tool": None},
    {"query": "the weather has been so weird lately don't you think", "tools": [WEATHER_TOOL], "expected_tool": None},
    {"query": "I used to play music professionally", "tools": [MUSIC_TOOL], "expected_tool": None},
    {"query": "reminders always stress me out honestly", "tools": [REMINDER_TOOL], "expected_tool": None},
]

DEBUG_RAW_NEEDLE_OUTPUT = True  # print raw Needle output whenever parsing fails or misfires

# Give every case an explicit "no tool needed" option, so a model that
# genuinely thinks no tool applies has a formal way to say so instead of
# being forced to pick from only the tools that happen to be relevant.
TEST_CASES = [
    {**case, "tools": case["tools"] + [NO_ACTION_TOOL]} for case in _RAW_TEST_CASES
]


def normalize_tool_name(name):
    """Treat explicit no_action selection the same as declining to call any tool."""
    return None if name == "no_action" else name


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


MINISTRAL_TIMEOUT_S = 120  # bumped from 30 -- Jetson under memory pressure can be slow per-request
MINISTRAL_MAX_TOKENS = 64  # cap generation -- we only care about the routing decision, not full chit-chat replies


def run_ministral(query, tools, url=MINISTRAL_URL, model=MINISTRAL_MODEL_NAME, timeout=MINISTRAL_TIMEOUT_S):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "tools": needle_tools_to_openai_format(tools),
        "temperature": 0,
        "max_tokens": MINISTRAL_MAX_TOKENS,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        latency = time.perf_counter() - t0
        print(f"[WARN] Ministral request failed for query={query!r}: {e}")
        return None, {}, latency

    latency = time.perf_counter() - t0
    if not resp.ok:
        print(f"\n--- Ministral request failed ({resp.status_code}) ---")
        print("Payload sent:", json.dumps(payload, indent=2))
        print("Response body:", resp.text)
        print("---\n")
        return None, {}, latency

    data = resp.json()

    tool_calls = data["choices"][0]["message"].get("tool_calls")
    if not tool_calls:
        return None, {}, latency

    call = tool_calls[0]["function"]
    name = normalize_tool_name(call["name"])
    try:
        args = json.loads(call["arguments"])
    except (json.JSONDecodeError, TypeError):
        args = {}
    return name, args, latency


import re


def _extract_name_fallback(raw):
    """Best-effort tool-name extraction when full JSON parsing fails --
    distinguishes 'picked the wrong tool' from 'picked the right tool but
    generated malformed/degenerate arguments'."""
    match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    return match.group(1) if match else None


def run_needle_case(query, tools, enc_sess, dec_sess, tokenizer, expected=None):
    tools_json = json.dumps(tools)
    t0 = time.perf_counter()
    raw = run_needle(query, tools_json, enc_sess, dec_sess, tokenizer)
    latency = time.perf_counter() - t0

    name, args = None, {}
    parse_failed = False
    try:
        parsed = json.loads(raw)
        if parsed:
            name = parsed[0]["name"]
            args = parsed[0].get("arguments", {})
    except (json.JSONDecodeError, IndexError, TypeError, KeyError):
        parse_failed = True
        name = _extract_name_fallback(raw)  # tool name may still be salvageable

    name = normalize_tool_name(name)

    if DEBUG_RAW_NEEDLE_OUTPUT and (parse_failed or name != expected):
        print(f"\n[DEBUG] query={query!r} expected={expected!r}")
        print(f"[DEBUG] raw Needle output: {raw!r}")
        if parse_failed:
            print(f"[DEBUG] -> JSON parse failed, name salvaged via regex: {name!r}")
        print()

    return name, args, latency, parse_failed


def main():
    paths = download_artifacts()
    enc_sess, dec_sess = load_sessions(paths)
    tokenizer = get_tokenizer(paths["needle.model"])

    results = []
    total = len(TEST_CASES)
    for i, case in enumerate(TEST_CASES, 1):
        query, tools, expected = case["query"], case["tools"], case["expected_tool"]
        print(f"[{i}/{total}] {query!r} ...", flush=True)

        n_name, n_args, n_lat, n_parse_failed = run_needle_case(query, tools, enc_sess, dec_sess, tokenizer, expected)
        print(f"    needle:    {n_name!r} ({n_lat*1000:.0f}ms)", flush=True)

        m_name, m_args, m_lat = run_ministral(query, tools)
        print(f"    ministral: {m_name!r} ({m_lat*1000:.0f}ms)", flush=True)

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