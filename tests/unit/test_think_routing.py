"""Unit tests for cognition.think intent routing shortcuts."""
from __future__ import annotations

import threading

import numpy as np

from cognition import think as think_module
from cognition.think import AikoThink, _load_route_examples


class FakeEmbedder:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def embed_query(self, text: str, instruct: str = ""):
        self.calls.append((text, instruct))
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeMemoryInner:
    def __init__(self, embedder):
        self._embedder = embedder


class FakeMemorize:
    def __init__(self, embedder):
        self._mem = FakeMemoryInner(embedder)


def _bare_think(embedder: FakeEmbedder | None = None) -> AikoThink:
    think = object.__new__(AikoThink)
    think._memorize = FakeMemorize(embedder or FakeEmbedder())
    think._memorize_lock = threading.Lock()
    think._active_user_ids = set()
    think._active_users_lock = threading.Lock()
    think._last_chat_time = 0.0
    think._proactive_lock = threading.Lock()
    think._proactive_resting = False
    return think


def test_route_examples_include_greeting_label():
    examples = _load_route_examples()

    assert "greeting" in examples
    assert any("hello" in example.lower() for example in examples["greeting"])


def test_route_intent_can_return_greeting_from_semantic_scores(monkeypatch):
    embedder = FakeEmbedder()
    think = _bare_think(embedder)
    monkeypatch.setattr(think, "_semantic_example_vectors", lambda examples, instruct: (["greeting"], np.array([[1.0, 0.0]], dtype=np.float32)))
    monkeypatch.setattr(think_module.reason, "label_scores_topk", lambda *a, **kw: {"greeting": 0.95, "localchat": 0.10})

    assert think._route_intent("hello there") == "greeting"
    assert embedder.calls == [("hello there", think_module._ROUTE_INSTRUCT_QUATERNARY)]


def test_greeting_route_skips_memory_recall_and_writeback(monkeypatch):
    think = _bare_think()
    calls = {}

    monkeypatch.setattr(think, "_route_intent", lambda user_input: "greeting")
    monkeypatch.setattr(think, "_fetch_memory_and_knowledge", lambda *a, **kw: calls.setdefault("fetch", True))

    def fake_chat(user_input, **kwargs):
        calls["chat_kwargs"] = kwargs
        return "hi!"

    monkeypatch.setattr(think, "chat", fake_chat)

    assert think.route("hi") == "hi!"
    assert "fetch" not in calls
    assert calls["chat_kwargs"]["skip_memory"] is True
    assert calls["chat_kwargs"]["store_turn"] is False


def test_non_greeting_route_starts_memory_after_intent(monkeypatch):
    think = _bare_think()
    events: list[str] = []

    monkeypatch.setattr(think, "_route_intent", lambda user_input: events.append("intent") or "localchat")

    class ImmediatePool:
        def submit(self, fn, *args):
            events.append("submit_mem_kb")

            class Future:
                def result(self_inner):
                    return fn(*args)

            return Future()

    monkeypatch.setattr(think_module, "CONTEXT_POOL", ImmediatePool())
    monkeypatch.setattr(think, "_fetch_memory_and_knowledge", lambda *a, **kw: ([], "<knowledge_context />"))
    monkeypatch.setattr(think, "chat", lambda *a, **kw: "ok")

    assert think.route("tell me about routers") == "ok"
    assert events == ["intent", "submit_mem_kb"]
