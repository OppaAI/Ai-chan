"""Unit tests for owner-account interaction memory in the Threads monitor."""

import importlib

threads = importlib.import_module("interface.mcp_server.social.services.threads")


class FakeMemorize:
    def __init__(self, display_name="OppaAI", user_id="github_205369547"):
        self.display_name = display_name
        self.user_id = user_id
        self.calls = []

    def get_display_name(self):
        return self.display_name

    def get_user_id(self):
        return self.user_id

    def add(self, messages, user_id=None, display_name=None):
        self.calls.append({"messages": messages, "user_id": user_id, "display_name": display_name})
        return True


def test_interaction_memory_saved_for_owner_account(monkeypatch):
    memorize = FakeMemorize()
    reply = {
        "username": "oppa.ai.bot",
        "timestamp": "2026-08-23T14:32:11+0000",
        "text": "Hi Aiko, we went to PNE today and it rained all afternoon.",
    }
    assert threads._save_interaction_memory(reply, "Sounds like a soggy but fun day!", memorize) is True
    assert len(memorize.calls) == 1
    call = memorize.calls[0]
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["user", "assistant"]
    assert "PNE today" in call["messages"][0]["content"]
    assert "[Threads 2026-08-23]" in call["messages"][0]["content"]
    assert "soggy" in call["messages"][1]["content"]


def test_interaction_memory_ignores_other_accounts(monkeypatch):
    memorize = FakeMemorize()
    reply = {"username": "random_visitor", "text": "Hi Aiko what is up"}
    assert threads._save_interaction_memory(reply, "Hello!", memorize) is False
    assert memorize.calls == []


def test_interaction_memory_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("THREADS_INTERACTION_MEMORY_ENABLED", "0")
    memorize = FakeMemorize()
    reply = {"username": "oppa.ai.bot", "text": "Hi Aiko from PNE"}
    assert threads._save_interaction_memory(reply, "Have fun!", memorize) is False
    assert memorize.calls == []


def test_interaction_memory_skips_empty_or_sensitive(monkeypatch):
    memorize = FakeMemorize()
    assert threads._save_interaction_memory({"username": "oppa.ai.bot", "text": ""}, "reply", memorize) is False
    sensitive = {"username": "oppa.ai.bot", "text": "my api_key=supersecretvalue123"}
    assert threads._save_interaction_memory(sensitive, "noted", memorize) is False
    assert memorize.calls == []


def test_interaction_memory_survives_memorize_failure():
    class BrokenMemorize(FakeMemorize):
        def add(self, messages, user_id=None, display_name=None):
            raise RuntimeError("memory store down")

    reply = {"username": "oppa.ai.bot", "text": "Hi Aiko"}
    assert threads._save_interaction_memory(reply, "hi!", BrokenMemorize()) is False
