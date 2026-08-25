from __future__ import annotations

import os
import queue

import pytest

from interface.webui.webui import AikoWeb
from system.userspace import current_display_name, current_user_id, reset_current_display_name, reset_current_user_id


class DummyMemorize:
    def __init__(self) -> None:
        self.switched: list[str] = []
        self.display_names: list[str] = []

    def switch_user(self, uid: str) -> None:
        self.switched.append(uid)

    def set_display_name(self, name: str) -> None:
        self.display_names.append(name)


@pytest.fixture(autouse=True)
def clean_user_env(monkeypatch):
    monkeypatch.delenv("AIKO_USER_ID", raising=False)
    monkeypatch.delenv("CURRENT_DISPLAY_NAME", raising=False)


@pytest.mark.asyncio
async def test_bind_user_context_is_request_local(monkeypatch):
    web = AikoWeb.__new__(AikoWeb)
    web._memorize = DummyMemorize()

    user_token, display_token, display_name = await web._bind_user_context(
        "alice", {"username": "Alice"}
    )
    try:
        assert current_user_id() == "alice"
        assert current_display_name() == "Alice"
        assert display_name == "Alice"
        assert os.getenv("AIKO_USER_ID") is None
        assert web._memorize.switched == []
        assert web._memorize.display_names == []
        assert not hasattr(web, "_current_user_id")
        assert not hasattr(web, "_current_display_name")
    finally:
        reset_current_display_name(display_token)
        reset_current_user_id(user_token)

    assert current_user_id() == "guest"


def test_get_input_uses_queued_identity_not_shared_state(monkeypatch):
    web = AikoWeb.__new__(AikoWeb)
    web._input_q = queue.Queue()
    web._input_q.put(("hello", "bob", "Bobby"))
    web._broadcast = lambda payload: None
    web._push_vitals = lambda: None

    assert web.get_input() == "hello"
    assert current_user_id() == "bob"
    assert current_display_name() == "Bobby"
