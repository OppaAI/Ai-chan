"""Unit tests for memory recall injection in the Threads reply monitor."""

import importlib

threads = importlib.import_module("interface.mcp_server.social.services.threads")


class FakeMemorize:
    def __init__(self, hits):
        self.hits = hits
        self.searched = []

    def get_user_id(self):
        return "github_205369547"

    def search(self, query, user_id=None, limit=3):
        self.searched.append((query, user_id, limit))
        return self.hits


def test_recall_block_formats_top_hits():
    memorize = FakeMemorize([
        {"memory": "OppaAI purchased an ESP32-based robotic dog", "score": 0.8},
        {"memory": "[Threads 2026-08-23] Exchange about PNE opening hours", "score": 0.7},
        {"memory": "", "score": 0.6},  # empty fact filtered
    ])
    block = threads._threads_memory_context("should I bring the robot dog to PNE?", memorize)
    assert "ESP32-based robotic dog" in block
    assert "PNE opening hours" in block
    assert block.startswith("Long-term memories")
    assert "never mention" in block
    assert (block, )  # sanity: non-empty


def test_recall_searches_with_comment_text_and_limit_3():
    memorize = FakeMemorize([])
    threads._threads_memory_context("what time does the fair open?", memorize)
    assert memorize.searched == [("what time does the fair open?", "github_205369547", 3)]


def test_recall_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("THREADS_RECALL_ENABLED", "0")
    memorize = FakeMemorize([{"memory": "fact"}])
    assert threads._threads_memory_context("anything", memorize) == ""
    assert memorize.searched == []


def test_recall_empty_on_no_hits_or_no_memorize():
    assert threads._threads_memory_context("hi", FakeMemorize([])) == ""
    assert threads._threads_memory_context("hi", None) == ""
    assert threads._threads_memory_context("", FakeMemorize([{"memory": "x"}])) == ""


def test_recall_survives_search_failure():
    class Broken:
        def get_user_id(self):
            return "u"

        def search(self, *a, **kw):
            raise RuntimeError("db locked")

    assert threads._threads_memory_context("hello", Broken()) == ""


def test_infer_reply_includes_memory_section(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SOCIAL_PERSONA_PATH", "")
    seen = {}

    def fake_create(**kwargs):
        seen["prompt"] = kwargs["messages"][-1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="🙂 Here you go."))])

    class C:
        chat = SimpleNamespace(completions=SimpleNamespace(create=staticmethod(fake_create)))

    monkeypatch.setattr(threads, "_get_llm_client", lambda: C())
    reply = {"id": "1", "username": "oppa.ai.bot", "text": "what did we do at PNE?"}
    out = threads._infer_reply(reply, [], memory_context="Long-term memories that may be relevant\n- OppaAI visited PNE with Aiko")
    assert out == "🙂 Here you go."
    assert "<memory_context>" in seen["prompt"]
    assert "visited PNE" in seen["prompt"]
