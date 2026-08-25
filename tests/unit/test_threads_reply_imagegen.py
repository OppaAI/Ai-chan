"""Unit tests for on-demand image generation in the Threads reply monitor.

Covers the intent gate, prompt extraction, Modal imagegen call, and the
unified text/image reply poster. No network access; all clients are faked.
"""

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

threads = importlib.import_module("interface.mcp_server.social.services.threads")


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = str(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, **kwargs):
        self.calls.append((url, data, kwargs))
        return self.responses.pop(0)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class FakeLLMClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


@pytest.mark.parametrize(
    "text",
    [
        "Hi Aiko can you draw a cat?",
        "@oppa.ai.bot generate an image of a sunset over Tokyo",
        "gen image of a robot dog",
        "make me a picture of a rainy train station",
        "paint me something calm",
    ],
)
def test_image_request_gate_matches_explicit_requests(text):
    assert threads._IMAGE_REQUEST_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "nice picture!",
        "remember this: image hosting tips",
        "what do you think about modern art?",
        "hello there!",
    ],
)
def test_image_request_gate_ignores_non_requests(text):
    assert not threads._IMAGE_REQUEST_RE.search(text)


def test_threads_image_request_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("THREADS_IMAGEGEN_ENABLED", "0")
    assert threads._threads_image_request("please draw a cat") == ""


def test_threads_image_request_no_match_skips_llm(monkeypatch):
    monkeypatch.setenv("THREADS_IMAGEGEN_ENABLED", "1")
    called = []
    monkeypatch.setattr(
        threads, "_get_llm_client", lambda: called.append(1) or None
    )
    assert threads._threads_image_request("hello there!") == ""
    assert called == []


def test_extracted_scene_prompt_is_cleaned(monkeypatch):
    monkeypatch.setenv("THREADS_IMAGEGEN_ENABLED", "1")
    monkeypatch.setattr(
        threads,
        "_get_llm_client",
        lambda: FakeLLMClient("A shiba inu astronaut drifting over Tokyo, neon signs below"),
    )
    scene = threads._threads_image_request("@oppa.ai.bot please draw a shiba astronaut")
    assert scene.startswith("A shiba inu astronaut")


def test_extraction_rejects_non_requests(monkeypatch):
    monkeypatch.setenv("THREADS_IMAGEGEN_ENABLED", "1")
    monkeypatch.setattr(threads, "_get_llm_client", lambda: FakeLLMClient("NONE"))
    # Gate matches ("image of") but the LLM confirms it is not a drawing request.
    assert threads._threads_image_request("show me an image of your lab") == ""


def test_generate_reply_image_requires_endpoint(monkeypatch):
    monkeypatch.setenv("IMAGEGEN_URL", "")
    assert threads._generate_reply_image("a cat") is None


def test_generate_reply_image_sends_reference_images(monkeypatch):
    png_b64 = base64.b64encode(b"\x89PNG-fake-bytes").decode()
    monkeypatch.setenv("IMAGEGEN_URL", "https://imggen.example")
    monkeypatch.setenv("THREADS_IMAGEGEN_TIMEOUT", "5")
    session = FakeSession([FakeResponse({"image_b64": png_b64})])
    monkeypatch.setattr(threads, "get_session", lambda: session)
    monkeypatch.setattr(threads, "_load_reference_images", lambda: ["ref-aiko", "ref-user"])

    path = threads._generate_reply_image("me and Aiko watching the stars")

    assert path is not None
    _, _, kwargs = session.calls[0]
    assert kwargs["json"]["reference_images"] == ["ref-aiko", "ref-user"]
    threads._cleanup_temp_image(path)


def test_generate_reply_image_without_reference_files_skips_refs(monkeypatch):
    png_b64 = base64.b64encode(b"\x89PNG-fake-bytes").decode()
    monkeypatch.setenv("IMAGEGEN_URL", "https://imggen.example")
    session = FakeSession([FakeResponse({"image_b64": png_b64})])
    monkeypatch.setattr(threads, "get_session", lambda: session)
    monkeypatch.setattr(threads, "_load_reference_images", lambda: [])

    path = threads._generate_reply_image("a cat")

    assert path is not None
    _, _, kwargs = session.calls[0]
    assert "reference_images" not in kwargs["json"]
    threads._cleanup_temp_image(path)


def test_load_reference_images_degrades_to_empty_on_import_failure(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fail_cognition(name, *args, **kwargs):
        if name.startswith("cognition"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_cognition)
    assert threads._load_reference_images() == []


def test_generate_reply_image_writes_and_cleans_temp_png(monkeypatch):
    png_b64 = base64.b64encode(b"\x89PNG-fake-bytes").decode()
    monkeypatch.setenv("IMAGEGEN_URL", "https://imggen.example")
    monkeypatch.setenv("THREADS_IMAGEGEN_TIMEOUT", "5")
    session = FakeSession([FakeResponse({"image_b64": png_b64})])
    monkeypatch.setattr(threads, "get_session", lambda: session)

    path = threads._generate_reply_image("a cat astronaut")

    assert path is not None
    assert Path(path).read_bytes() == b"\x89PNG-fake-bytes"
    url, payload, _ = session.calls[0]
    assert url == "https://imggen.example/generate"
    assert payload is None
    assert _["json"]["prompt"].startswith("a cat astronaut")
    threads._cleanup_temp_image(path)
    assert not Path(path).exists()


def test_generate_reply_image_api_failure_returns_none(monkeypatch):
    monkeypatch.setenv("IMAGEGEN_URL", "https://imggen.example")
    session = FakeSession([FakeResponse({}, status_code=500)])
    monkeypatch.setattr(threads, "get_session", lambda: session)
    assert threads._generate_reply_image("a cat") is None


def test_post_reply_text_fast_path(monkeypatch):
    monkeypatch.setenv("THREADS_API_BASE", "https://graph.threads.net/v1.0")
    session = FakeSession([FakeResponse({"id": "resp-1"})])
    monkeypatch.setattr(threads, "get_session", lambda: session)

    result = threads._post_threads_reply("tok", "user-1", "hi!", "comment-1")

    assert result["ok"] is True
    assert result["response_id"] == "resp-1"
    url, data, _ = session.calls[0]
    assert url.endswith("/me/threads")
    assert data["media_type"] == "TEXT"
    assert data["auto_publish_text"] == "true"
    assert data["reply_to_id"] == "comment-1"


def test_post_reply_with_image_uses_two_step_publish(monkeypatch):
    monkeypatch.setenv("THREADS_API_BASE", "https://graph.threads.net/v1.0")
    monkeypatch.setenv("THREADS_PUBLISH_DELAY_SECONDS", "0")
    monkeypatch.setattr(
        threads, "_upload_to_imgbb", lambda p: {"ok": True, "url": "https://i.ibb.co/gen.png"}
    )
    session = FakeSession(
        [FakeResponse({"id": "container-9"}), FakeResponse({"id": "published-9"})]
    )
    monkeypatch.setattr(threads, "get_session", lambda: session)

    result = threads._post_threads_reply(
        "tok", "user-1", "here you go", "comment-7", image_path="/tmp/gen.png"
    )

    assert result["ok"] is True
    assert result["response_id"] == "published-9"
    (create_url, create_data, _), (publish_url, publish_data, _) = session.calls
    assert create_url.endswith("/user-1/threads")
    assert create_data["media_type"] == "IMAGE"
    assert create_data["image_url"] == "https://i.ibb.co/gen.png"
    assert create_data["text"] == "here you go"
    assert create_data["reply_to_id"] == "comment-7"
    assert publish_url.endswith("/user-1/threads_publish")
    assert publish_data["creation_id"] == "container-9"


def test_post_reply_image_upload_failure_reports_stage(monkeypatch):
    monkeypatch.setattr(
        threads, "_upload_to_imgbb", lambda p: {"ok": False, "error": "IMGBB_API_KEY not set"}
    )
    result = threads._post_threads_reply(
        "tok", "user-1", "hi", "comment-1", image_path="/tmp/gen.png"
    )
    assert result["ok"] is False
    assert result["stage"] == "image_upload"
