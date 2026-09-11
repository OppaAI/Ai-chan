"""Regression tests for Go owner fallback and KataGo process cleanup."""

import subprocess

import pytest
from fastapi import Request

from interface.webui.go import games_go, katago


def _request(host: str, headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "client": (host, 1234),
            "headers": [
                (name.lower().encode(), value.encode())
                for name, value in (headers or {}).items()
            ],
        }
    )


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.255.255.254", "::1", "100.64.0.1", "100.127.255.254"],
)
def test_owner_fallback_allows_direct_loopback_and_tailscale(host, monkeypatch):
    monkeypatch.delenv("GAMES_APP_SECRET", raising=False)

    assert games_go._allow_owner_fallback(_request(host))


@pytest.mark.parametrize(
    "host",
    ["localhost", "100.63.255.255", "100.128.0.1", "100.200.1.1", "not-an-ip"],
)
def test_owner_fallback_rejects_non_trusted_addresses(host, monkeypatch):
    monkeypatch.delenv("GAMES_APP_SECRET", raising=False)

    assert not games_go._allow_owner_fallback(_request(host))


@pytest.mark.parametrize("header", ["Forwarded", "X-Forwarded-For", "X-Real-IP"])
def test_owner_fallback_does_not_trust_proxy_supplied_loopback(header, monkeypatch):
    monkeypatch.delenv("GAMES_APP_SECRET", raising=False)

    request = _request("127.0.0.1", {header: "for=203.0.113.10"})

    assert not games_go._allow_owner_fallback(request)


class _Stream:
    def __init__(self):
        self.closed = False

    def write(self, _data):
        pass

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _SlowToReapProcess:
    def __init__(self, reap_error=None):
        self.stdin = _Stream()
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.wait_timeouts = []
        self.killed = False
        self.reap_error = reap_error

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        assert katago._proc is self
        self.wait_timeouts.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("katago", timeout)
        if self.reap_error is not None:
            raise self.reap_error
        return 0


def test_shutdown_retains_process_until_blocking_reap():
    proc = _SlowToReapProcess()
    katago._proc = proc
    katago._ready = True
    katago._board_size = 9

    katago._shutdown()

    assert proc.killed
    assert proc.wait_timeouts == [5, None]
    assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))
    assert katago._proc is None


def test_shutdown_retains_process_when_blocking_reap_fails():
    proc = _SlowToReapProcess(RuntimeError("reap failed"))
    katago._proc = proc

    try:
        with pytest.raises(RuntimeError, match="reap failed"):
            katago._shutdown()

        assert katago._proc is proc
        assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))
    finally:
        katago._proc = None
