"""Unit tests for cognition.think intent routing shortcuts."""
from __future__ import annotations

import threading
from types import SimpleNamespace

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


def test_ternary_route_examples_exclude_greeting_label():
    examples = _load_route_examples(include_greeting=False)

    assert set(examples) == {"agentic", "webchat", "localchat"}


def test_route_intent_can_return_greeting_from_semantic_scores(monkeypatch):
    embedder = FakeEmbedder()
    think = _bare_think(embedder)
    monkeypatch.setattr(think, "_semantic_example_vectors", lambda examples, instruct: (["greeting"], np.array([[1.0, 0.0]], dtype=np.float32)))
    monkeypatch.setattr(think_module.reason, "label_scores_topk", lambda *a, **kw: {"greeting": 0.95, "localchat": 0.10})

    intent, query_vec = think._route_intent("hello there")
    assert intent == "greeting"
    assert query_vec is not None
    assert embedder.calls == [("hello there", think_module._ROUTE_INSTRUCT_QUATERNARY)]


def test_greeting_route_skips_memory_recall_and_writeback(monkeypatch):
    think = _bare_think()
    calls = {}

    monkeypatch.setattr(think, "_route_intent", lambda user_input: ("greeting", None))
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

    monkeypatch.setattr(think, "_route_intent", lambda user_input: events.append("intent") or ("localchat", None))

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


def test_degrade_chat_is_recorded_without_promoting_agency(monkeypatch):
    from cognition import attention

    think = _bare_think()
    state = attention.EdgeCognitiveState()
    monkeypatch.setattr(attention, "for_identity", lambda _identity: state)
    monkeypatch.setattr(think, "chat", lambda *_args, **_kwargs: "soft reply")

    assert think._soft_gate_reply("draft this", "degrade_chat", "low confidence") == "soft reply"

    assert state.snapshot()["self_decisions"][0]["kinds"] == ["degrade_chat"]
    assert not state.snapshot()["self_preference_evidence"]


class FakeCompletions:
    def __init__(self, label: str):
        self.label = label
        self.last_messages = None

    def create(self, **kwargs):
        self.last_messages = kwargs["messages"]
        message = SimpleNamespace(content=self.label)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, label: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(label))


def test_ternary_llm_classifier_preserves_three_way_labels():
    think = _bare_think()
    think._client = FakeClient("greeting")
    think._router_model = "router"

    assert think._classify_ternary_intent_llm("hello") == "localchat"
    prompt = think._client.chat.completions.last_messages[0]["content"]
    assert "Labels: [agentic, webchat, chat]" in prompt
    assert "Label: greeting" not in prompt


def test_quaternary_llm_classifier_allows_greeting_label():
    think = _bare_think()
    think._client = FakeClient("greeting")
    think._router_model = "router"

    assert think._classify_quaternary_intent_llm("hello") == "greeting"
    prompt = think._client.chat.completions.last_messages[0]["content"]
    assert "Labels: [greeting, agentic, webchat, chat]" in prompt
    assert "Label: greeting" in prompt


def test_ambiguous_semantic_scores_fall_back_to_quaternary_tiebreak(monkeypatch):
    """Webchat above threshold but inside the gap must be resolved by the
    quaternary LLM classifier — webchat stays reachable (the old binary
    agentic-or-chat tie-break collapsed it to localchat)."""
    think = _bare_think(FakeEmbedder())
    monkeypatch.setenv("ROUTE_WEBCHAT_THRESHOLD", "0.60")
    monkeypatch.setattr(think_module, "_SEMANTIC_ROUTE_MIN_GAP", 0.05)
    monkeypatch.setattr(
        think, "_semantic_example_vectors",
        lambda examples, instruct: (["webchat", "agentic"], np.zeros((2, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(
        think_module.reason, "label_scores_topk",
        lambda *a, **kw: {"webchat": 0.65, "agentic": 0.63, "localchat": 0.10},
    )
    tiebreak_calls = []
    monkeypatch.setattr(
        think, "_classify_quaternary_intent_llm",
        lambda text, allow_agentic=True: tiebreak_calls.append(text) or "webchat",
    )

    intent, _ = think._route_intent("what does PNE stand for")
    assert tiebreak_calls == ["what does PNE stand for"]
    assert intent == "webchat"


def test_recall_query_resolves_pronouns_from_recent_history():
    think = _bare_think()
    think._history_lock = threading.Lock()
    think._history = [
        {"role": "user", "content": "Do you know what PNE is?"},
        {"role": "assistant", "content": "PNE? You mean Playland?"},
    ]
    q = think._recall_query("We went there yesterday together. Did you enjoy?")
    assert "We went there yesterday" in q
    assert "Playland" in q  # antecedent context folded in


def test_recall_query_is_plain_input_with_empty_history():
    think = _bare_think()
    assert think._recall_query("hello there") == "hello there"


def test_fetch_memory_uses_enriched_query(monkeypatch):
    think = _bare_think()
    think._history_lock = threading.Lock()
    think._history = [
        {"role": "user", "content": "Do you know what PNE is?"},
        {"role": "assistant", "content": "PNE? Playland."},
    ]
    captured = {}

    class FakeMemorizeWrapper:
        _mem = SimpleNamespace(_embedder=None)

        def search(self, query, limit=3, query_vector=None, **kw):
            captured["query"] = query
            return []

    think._memorize = FakeMemorizeWrapper()
    monkeypatch.setattr(think_module, "knowledge_context_for", lambda *a, **kw: "")

    memories, kb = think._fetch_memory_and_knowledge("We went there yesterday. Did you enjoy?")
    assert (memories, kb) == ([], "")
    assert "PNE? Playland." in captured["query"]
