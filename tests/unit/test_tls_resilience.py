"""Unit tests for resilient TLS CA-bundle handling (system/tls + users).

The runtime checkout lives on removable-media-backed storage; when it hiccups,
certifi.where() transiently stops existing and every HTTPS call dies with
requests' "Could not find a suitable TLS CA certificate bundle" OSError.
These tests pin the heal-and-retry behavior: verification is re-pointed at a
bundle that exists, never disabled.
"""

import importlib
import os
import sys
import types

import pytest
import requests

tls = importlib.import_module("system.tls")
social_services = importlib.import_module("interface.mcp_server.social.services")


CA_ERROR = OSError(
    "Could not find a suitable TLS CA certificate bundle, "
    "invalid path: /gone/.venv/site-packages/certifi/cacert.pem"
)


@pytest.fixture
def clean_tls_env(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.delitem(sys.modules, "certifi", raising=False)
    yield


def test_resolve_prefers_existing_env_override(monkeypatch, tmp_path):
    bundle = tmp_path / "ca.pem"
    bundle.touch()
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    assert tls.resolve_ca_bundle() == str(bundle)


def test_resolve_skips_missing_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "missing.pem"))
    fake = types.ModuleType("certifi")
    fake.where = lambda: str(tmp_path / "certifi.pem")
    (tmp_path / "certifi.pem").touch()
    monkeypatch.setitem(sys.modules, "certifi", fake)
    assert tls.resolve_ca_bundle() == str(tmp_path / "certifi.pem")


def test_resolve_skips_excluded_certifi_path(monkeypatch, tmp_path):
    certifi_bundle = tmp_path / "cacert.pem"
    certifi_bundle.touch()
    fake = types.ModuleType("certifi")
    fake.where = lambda: str(certifi_bundle)
    monkeypatch.setitem(sys.modules, "certifi", fake)
    monkeypatch.setattr(tls, "_SYSTEM_BUNDLES", (), raising=False)
    # Exists but just failed mid-flight -> excluded -> nothing else available.
    assert tls.resolve_ca_bundle(exclude=(str(certifi_bundle),)) is None
    # Without exclusion it is returned.
    assert tls.resolve_ca_bundle() == str(certifi_bundle)


def test_resolve_falls_through_to_system_bundle(monkeypatch, tmp_path):
    fake = types.ModuleType("certifi")
    fake.where = lambda: "/nonexistent/cacert.pem"
    monkeypatch.setitem(sys.modules, "certifi", fake)
    system = tmp_path / "ca-certificates.crt"
    system.touch()
    monkeypatch.setattr(tls, "_SYSTEM_BUNDLES", (str(system),), raising=False)
    assert tls.resolve_ca_bundle(exclude=("/nonexistent",)) == str(system)


def test_resolve_returns_none_when_all_missing(monkeypatch):
    fake = types.ModuleType("certifi")
    fake.where = lambda: "/nonexistent/cacert.pem"
    monkeypatch.setitem(sys.modules, "certifi", fake)
    monkeypatch.setattr(tls, "_SYSTEM_BUNDLES", (), raising=False)
    assert tls.resolve_ca_bundle() is None


def test_is_ca_bundle_error_matches_only_ca_failures():
    assert tls.is_ca_bundle_error(CA_ERROR)
    assert not tls.is_ca_bundle_error(OSError("connection refused"))
    assert not tls.is_ca_bundle_error(ValueError("nope"))


def test_heal_verify_excludes_only_string_verifies(monkeypatch):
    monkeypatch.setattr(tls, "resolve_ca_bundle", lambda exclude=(): f"healed:{sorted(exclude)}")
    assert tls.heal_verify("/gone/cacert.pem") == "healed:['/gone/cacert.pem']"
    assert tls.heal_verify(True) == "healed:[]"
    assert tls.heal_verify(None) == "healed:[]"


def test_get_session_mounts_self_healing_adapter():
    session = social_services.get_session()
    for adapter in session.adapters.values():
        assert isinstance(adapter, social_services._CASelfHealingAdapter)


def test_self_healing_adapter_retries_with_healed_bundle(monkeypatch):
    calls = []

    class StubSuper:
        def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
            calls.append(verify)
            if len(calls) == 1:
                raise CA_ERROR
            return "ok"

    monkeypatch.setattr(
        social_services.HTTPAdapter, "send", lambda self, request, **kw: StubSuper().send(request, **kw)
    )
    healed_path = "/etc/ssl/certs/ca-certificates.crt"
    monkeypatch.setattr(tls, "heal_verify", lambda failed: healed_path)
    adapter = social_services._CASelfHealingAdapter()
    assert adapter.send(requests.Request("GET", "https://x.example").prepare()) == "ok"
    assert calls == [True, healed_path]
    assert len(calls) == 2


def test_self_healing_adapter_reraises_non_ca_errors(monkeypatch):
    def boom(self, request, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(social_services.HTTPAdapter, "send", boom)
    adapter = social_services._CASelfHealingAdapter()
    with pytest.raises(OSError, match="connection reset"):
        adapter.send(requests.Request("GET", "https://x.example").prepare())


def test_self_healing_adapter_reraises_when_no_heal_possible(monkeypatch):
    calls = []

    def one_shot(self, request, **kwargs):
        calls.append(1)
        raise CA_ERROR

    monkeypatch.setattr(social_services.HTTPAdapter, "send", one_shot)
    monkeypatch.setattr(tls, "heal_verify", lambda failed: None)
    adapter = social_services._CASelfHealingAdapter()
    with pytest.raises(OSError, match="TLS CA certificate bundle"):
        adapter.send(requests.Request("GET", "https://x.example").prepare())
    assert len(calls) == 1
