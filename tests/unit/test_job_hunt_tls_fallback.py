"""Unit tests for Lane D's TLS-resilient HTTP GET helper.

When the venv-backed certifi bundle transiently disappears (removable-storage
hiccup), _http_get_with_tls_fallback must retry once against a CA store that
exists instead of failing the whole RSS/email fetch node.
"""

import importlib

import pytest

toolset = importlib.import_module("agentic.workflows.job_hunt.toolset")
tls = importlib.import_module("system.tls")

CA_ERROR = OSError(
    "Could not find a suitable TLS CA certificate bundle, "
    "invalid path: /gone/.venv/site-packages/certifi/cacert.pem"
)


def test_ca_failure_retries_with_healed_bundle(monkeypatch):
    calls = []

    def fake_get(url, timeout=None, headers=None, verify=None):
        calls.append({"url": url, "timeout": timeout, "verify": verify})
        if len(calls) == 1:
            raise CA_ERROR
        return "resp"

    monkeypatch.setattr(toolset.requests, "get", fake_get)
    monkeypatch.setattr(tls, "heal_verify", lambda failed: "/etc/ssl/ca-healed.pem")

    resp = toolset._http_get_with_tls_fallback(
        "https://feeds.example/rss", timeout=30, headers={"User-Agent": "t"}
    )

    assert resp == "resp"
    assert [c["verify"] for c in calls] == [None, "/etc/ssl/ca-healed.pem"]
    assert calls[0]["url"] == calls[1]["url"] == "https://feeds.example/rss"


def test_non_ca_oserror_propagates_without_retry(monkeypatch):
    calls = []

    def fake_get(url, timeout=None, headers=None, verify=None):
        calls.append(1)
        raise OSError("name resolution failed")

    monkeypatch.setattr(toolset.requests, "get", fake_get)
    with pytest.raises(OSError, match="name resolution failed"):
        toolset._http_get_with_tls_fallback("https://feeds.example/rss", timeout=5)
    assert len(calls) == 1


def test_no_heal_available_reraises_original(monkeypatch):
    def fake_get(url, timeout=None, headers=None, verify=None):
        raise CA_ERROR

    monkeypatch.setattr(toolset.requests, "get", fake_get)
    monkeypatch.setattr(tls, "heal_verify", lambda failed: None)
    with pytest.raises(OSError, match="TLS CA certificate bundle"):
        toolset._http_get_with_tls_fallback("https://feeds.example/rss", timeout=5)
