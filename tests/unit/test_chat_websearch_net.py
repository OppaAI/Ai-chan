"""Unit tests for the last-resort websearch net in AikoThink.chat()."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from cognition import think as think_module
from cognition.think import AikoThink


class FakeMemorize:
    def format_for_context(self, *a, **kw):
        return ""

    def persona_context(self):
        return ""


class FakeMemoryInner:
    def __init__(self):
        self._embedder = None


def _net_think() -> AikoThink:
    think = object.__new__(AikoThink)
    think._memorize = FakeMemorize()
    think._memorize_lock = threading.Lock()
    think._active_user_ids = set()
    think._active_users_lock = threading.Lock()
    think._last_chat_time = 0.0
    think._proactive_lock = threading.Lock()
    think._proactive_resting = False
    think._history = []
    think._history_lock = threading.RLock()
    think._reasoning = False
    think._get_speak = lambda: None
    think._current_system_prompt = lambda *a, **kw: "SYS"
    return think


def test_websearch_net_block_formats_results(monkeypatch):
    think = _net_think()
    captured = {}

    def fake_search(query, n):
        captured["query"] = query
        captured["n"] = n
        return ([{"title": "PNE", "url": "https://pne.example", "content": "The Fair is on."}], None)

    from agentic.toolkit import websearch
    monkeypatch.setattr(websearch, "web_search", fake_search)

    block = think._websearch_net_block("check internet what PNE is")
    assert captured == {"query": "check internet what PNE is", "n": 3}
    assert "https://pne.example" in block
    assert block.startswith("1. PNE")


def test_websearch_net_block_empty_on_error_or_no_results(monkeypatch):
    think = _net_think()
    from agentic.toolkit import websearch

    monkeypatch.setattr(websearch, "web_search", lambda q, n: ([], "searxng down"))
    assert think._websearch_net_block("anything") == ""

    monkeypatch.setattr(websearch, "web_search", lambda q, n: ([], None))
    assert think._websearch_net_block("anything") == ""


@pytest.fixture()
def chat_env(monkeypatch):
    """Stub out everything chat() touches besides the net under test."""
    think = _net_think()

    class EdgeState:
        def prioritize_memories(self, *a, **kw):
            return []

        def situation_context(self, *a, **kw):
            return ""

        def metacognitive_context(self, *a, **kw):
            return ""

    from cognition.memory import edge_state
    monkeypatch.setattr(edge_state, "for_identity", lambda uid: EdgeState())
    monkeypatch.setattr(think_module.bioclock, "current_datetime_block", lambda: "")
    monkeypatch.setattr(think, "_resolve_mem_kb", lambda *a, **kw: ([], ""))
    monkeypatch.setattr(think, "_sanitize_history", lambda history: history)
    monkeypatch.setattr(think, "_store_async", lambda *a, **kw: None)
    monkeypatch.setattr(think, "_finalize_response", lambda user_input, resp, cb=None, already_emitted=None: resp)
    return think


def test_chat_runs_websearch_net_for_internet_asks(chat_env, monkeypatch):
    think = chat_env
    net_calls = []

    def fake_net(query, token_callback=None):
        net_calls.append(query)
        return "1. PNE\n   https://pne.example\n   The Fair."

    monkeypatch.setattr(think, "_websearch_net_block", fake_net)
    seen_systems = []

    def fake_stream(trimmed, system=None, token_callback=None, emit=None):
        seen_systems.append(system)
        return "ok"

    monkeypatch.setattr(think, "_stream_response", fake_stream)

    result = think.chat("check internet to see what PNE is")
    assert result == "ok"
    assert net_calls == ["check internet to see what PNE is"]
    assert "<search_results query='check internet to see what PNE is'>" in seen_systems[0]
    assert "https://pne.example" in seen_systems[0]


def test_chat_skips_websearch_net_without_hint_words(chat_env, monkeypatch):
    think = chat_env
    net_calls = []
    monkeypatch.setattr(think, "_websearch_net_block", lambda q, token_callback=None: net_calls.append(q) or "x")

    def fail_stream(*a, **kw):
        raise AssertionError("stream should still be reached")

    monkeypatch.setattr(think, "_stream_response", fail_stream)
    try:
        think.chat("what do you think about minimalism")
    except AssertionError as exc:
        assert "stream" in str(exc)
    assert net_calls == []


def test_chat_net_failure_degrades_to_plain_chat(chat_env, monkeypatch):
    think = chat_env
    monkeypatch.setattr(think, "_websearch_net_block", lambda q, token_callback=None: "")

    def fake_stream(trimmed, system=None, token_callback=None, emit=None):
        assert "<search_results" not in system
        return "plain reply"

    monkeypatch.setattr(think, "_stream_response", fake_stream)
    assert think.chat("check internet to see what PNE is") == "plain reply"
