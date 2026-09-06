"""
agentic/toolkit/it_ops.py

IT + everyday helper tools for n8n-style DAGs on Jetson Orin Nano 8GB.

Design constraints (Ministral-3B + 8GB unified RAM):
  - No heavy deps: stdlib + psutil (already a core dep) only.
  - Every tool is bounded: timeouts, output caps, no unbounded scans.
  - Read-only by default. The only "mutating" helper (disk_large_files)
    is a preview — it never deletes.
  - All tools are graph=True + react=True so they appear in the DAG
    palette and are callable from ReAct.
  - JSON-block output via agentic.toolkit.common.json_block so the
    graph engine's $result: substitution stays small.

Tool list (IT):
  sys_health        CPU/RAM/disk + loadavg, 1s sample
  process_top       Top-N processes by CPU/RAM (psutil, no kill)
  disk_large_files  Preview largest files under a dir (no delete)
  log_triage        Tail a log file + extract ERROR/WARN lines
  port_check        TCP connect check with 2s timeout
  http_check        Lightweight HTTP GET status/timing (urllib, no JS)
  dns_lookup        Resolve a hostname to IPs
  ssl_expiry        Days until TLS cert expires for host:port
  service_status    systemctl is-active check (fails gracefully)
  docker_ps         `docker ps` preview (timeout 5s, fails gracefully)
  file_find         Bounded glob search under workspace/repo
  net_info          Hostname, IPs, basic interface summary

Tool list (everyday):
  text_summarize    LLM condense (needs client/model; heuristic fallback)
  text_translate    LLM translate (needs client/model; passthrough fallback)
  unit_convert      Offline length/mass/temp/data conversions
  pass_gen          Cryptographically random password (secrets)
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
from pathlib import Path

from agentic.registry import TOOLS, tool
from agentic.toolkit.common import json_block
from system.log import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_FIND_RESULTS = 50
MAX_LOG_LINES = 500
MAX_FILE_PREVIEW = 20000


def _spec(name: str, description: str):
    return TOOLS[name] if name in TOOLS else name


# ── IT: system ──────────────────────────────────────────────────────────

@tool(
    _spec("sys_health", "Check CPU/RAM/disk health (Jetson-safe, 1s sample)."),
    description="Check CPU/RAM/disk health (Jetson-safe, 1s sample).",
    graph=True,
    react=True,
    domain="it_ops",
)
def sys_health() -> str:
    """Sample CPU (1s), RAM, disk, loadavg. Never blocks longer than ~1.2s."""
    try:
        import psutil

        cpu = float(psutil.cpu_percent(interval=1.0))
        ram = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        try:
            load = os.getloadavg()
        except OSError:
            load = (0.0, 0.0, 0.0)
        status = "healthy"
        if cpu >= 90 or ram.percent >= 90 or (disk.used / disk.total) >= 0.9:
            status = "critical"
        elif cpu >= 75 or ram.percent >= 75 or (disk.used / disk.total) >= 0.8:
            status = "warning"
        return json_block("sys_health", {
            "ok": True,
            "cpu_percent": round(cpu, 1),
            "ram_percent": round(float(ram.percent), 1),
            "ram_used_gb": round(ram.used / (1024 ** 3), 2),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "disk_percent": round(disk.used / disk.total * 100, 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "load_1_5_15": [round(float(x), 2) for x in load],
            "status": status,
        })
    except Exception as e:
        log.warning("sys_health failed: %s", e)
        return json_block("sys_health", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("process_top", "List top processes by CPU/RAM (read-only, no kill)."),
    description="List top processes by CPU/RAM (read-only, no kill).",
    graph=True,
    react=True,
    domain="it_ops",
)
def process_top(limit: int = 8, sort_by: str = "cpu") -> str:
    """Top-N processes. Bounded, read-only — no kill signal is ever sent."""
    try:
        import psutil

        limit = max(1, min(int(limit or 8), 20))
        sort_by = (sort_by or "cpu").strip().lower()
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                procs.append({
                    "pid": int(info.get("pid") or 0),
                    "name": str(info.get("name") or "?")[:60],
                    "cpu": round(float(info.get("cpu_percent") or 0.0), 1),
                    "mem": round(float(info.get("memory_percent") or 0.0), 1),
                })
            except Exception:
                continue
        key = "mem" if sort_by in {"mem", "memory", "ram"} else "cpu"
        procs.sort(key=lambda r: r[key], reverse=True)
        return json_block("process_top", {"ok": True, "sort_by": key, "count": len(procs[:limit]), "top": procs[:limit]})
    except Exception as e:
        return json_block("process_top", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("disk_large_files", "Preview largest files under a directory (read-only, never deletes)."),
    description="Preview largest files under a directory (read-only, never deletes).",
    graph=True,
    react=True,
    domain="it_ops",
)
def disk_large_files(directory: str = "logs", limit: int = 10, max_depth: int = 3) -> str:
    """List largest files. Preview only — callers must delete by hand."""
    try:
        limit = max(1, min(int(limit or 10), 25))
        max_depth = max(1, min(int(max_depth or 3), 5))
        base = (REPO_ROOT / directory.strip().lstrip("/\\")).resolve() if (directory or "").strip() else REPO_ROOT
        if base != REPO_ROOT and REPO_ROOT not in base.parents:
            # Allow absolute workspace paths too, but never escape to /etc etc.
            # Confine to repo + user workspace.
            from system.userspace import user_workspace_root, current_user_id

            try:
                ws = Path(str(user_workspace_root(current_user_id()))).resolve()
                if base != ws and ws not in base.parents:
                    return json_block("disk_large_files", {"ok": False, "error": "directory escapes repo/workspace"})
            except Exception:
                return json_block("disk_large_files", {"ok": False, "error": "directory escapes repo"})
        if not base.exists():
            return json_block("disk_large_files", {"ok": False, "error": f"not found: {directory}"})
        found: list[dict] = []
        base_depth = len(base.parts)
        for path in base.rglob("*"):
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                if len(path.parts) - base_depth > max_depth:
                    continue
                size = path.stat().st_size
                found.append({"path": str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else str(path), "bytes": size})
            except OSError:
                continue
            if len(found) >= 2000:
                break
        found.sort(key=lambda r: r["bytes"], reverse=True)
        top = found[:limit]
        for row in top:
            row["mb"] = round(row["bytes"] / (1024 * 1024), 2)
        return json_block("disk_large_files", {"ok": True, "directory": str(directory), "scanned": len(found), "top": top, "note": "preview only — nothing was deleted"})
    except Exception as e:
        return json_block("disk_large_files", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("log_triage", "Tail a log file and extract ERROR/WARNING lines (bounded)."),
    description="Tail a log file and extract ERROR/WARNING lines (bounded).",
    graph=True,
    react=True,
    domain="it_ops",
)
def log_triage(log_file: str = "logs/aiko.log", lines: int = 120) -> str:
    """Read last N lines, return anomaly subset. Defaults to repo logs/aiko.log."""
    try:
        limit = max(1, min(int(lines or 120), MAX_LOG_LINES))
        candidate = (REPO_ROOT / log_file.strip().lstrip("/\\")).resolve() if (log_file or "").strip() else (REPO_ROOT / "logs" / "aiko.log")
        if candidate != REPO_ROOT and REPO_ROOT not in candidate.parents:
            return json_block("log_triage", {"ok": False, "error": "log_file escapes repository"})
        if not candidate.exists() or not candidate.is_file():
            return json_block("log_triage", {"ok": False, "error": f"log not found: {log_file}", "hint": "run Aiko once to create logs/aiko.log"})
        with open(candidate, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        recent = content[-limit:]
        anomalies = [ln.strip()[:300] for ln in recent if ("ERROR" in ln or "WARNING" in ln or "Exception" in ln or "Traceback" in ln)]
        return json_block("log_triage", {
            "ok": True, "file": str(candidate.relative_to(REPO_ROOT)),
            "parsed_lines": len(recent), "anomalies_found": len(anomalies),
            "anomalies": anomalies[:20],
            "tail": "".join(recent[-5:])[-1000:],
        })
    except Exception as e:
        return json_block("log_triage", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("service_status", "Check systemd service state (read-only systemctl is-active)."),
    description="Check systemd service state (read-only systemctl is-active).",
    graph=True,
    react=True,
    domain="it_ops",
)
def service_status(service: str = "docker") -> str:
    """`systemctl is-active <service>` with 5s timeout. Read-only."""
    name = (service or "").strip().split()[0][:64] if service else ""
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@-_:. " for c in name):
        return json_block("service_status", {"ok": False, "error": "invalid service name"})
    try:
        proc = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=5)
        state = (proc.stdout or "").strip() or (proc.stderr or "").strip() or "unknown"
        return json_block("service_status", {"ok": True, "service": name, "state": state[:60], "active": state.strip() == "active"})
    except FileNotFoundError:
        return json_block("service_status", {"ok": False, "service": name, "error": "systemctl not available"})
    except subprocess.TimeoutExpired:
        return json_block("service_status", {"ok": False, "service": name, "error": "timeout"})
    except Exception as e:
        return json_block("service_status", {"ok": False, "service": name, "error": str(e)[:200]})


@tool(
    _spec("docker_ps", "Preview running docker containers (read-only, 5s timeout)."),
    description="Preview running docker containers (read-only, 5s timeout).",
    graph=True,
    react=True,
    domain="it_ops",
)
def docker_ps() -> str:
    """`docker ps --format` preview. Fails gracefully when docker is absent."""
    try:
        proc = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                              capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return json_block("docker_ps", {"ok": False, "error": (proc.stderr or "docker ps failed")[:300]})
        rows = []
        for line in (proc.stdout or "").splitlines():
            parts = line.split("\t")
            rows.append({"name": (parts[0] if len(parts) > 0 else "")[:80],
                         "status": (parts[1] if len(parts) > 1 else "")[:120],
                         "ports": (parts[2] if len(parts) > 2 else "")[:120]})
            if len(rows) >= 20:
                break
        return json_block("docker_ps", {"ok": True, "count": len(rows), "containers": rows})
    except FileNotFoundError:
        return json_block("docker_ps", {"ok": False, "error": "docker not installed"})
    except subprocess.TimeoutExpired:
        return json_block("docker_ps", {"ok": False, "error": "timeout"})
    except Exception as e:
        return json_block("docker_ps", {"ok": False, "error": str(e)[:200]})


# ── IT: network ─────────────────────────────────────────────────────────

@tool(
    _spec("port_check", "TCP port check with 2s timeout (read-only)."),
    description="TCP port check with 2s timeout (read-only).",
    graph=True,
    react=True,
    domain="it_ops",
)
def port_check(host: str = "127.0.0.1", port: int = 8787, timeout: float = 2.0) -> str:
    """Single TCP connect. No banner grab, no scan — one host:port only."""
    try:
        host = (host or "").strip()[:253]
        port = max(1, min(int(port), 65535))
        timeout = max(0.5, min(float(timeout or 2.0), 5.0))
        if not host:
            return json_block("port_check", {"ok": False, "error": "host required"})
        start = __import__("time").monotonic()
        with socket.create_connection((host, port), timeout=timeout):
            pass
        ms = round((__import__("time").monotonic() - start) * 1000, 1)
        return json_block("port_check", {"ok": True, "host": host, "port": port, "open": True, "latency_ms": ms})
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return json_block("port_check", {"ok": True, "host": host, "port": port, "open": False, "error": str(e)[:150]})
    except Exception as e:
        return json_block("port_check", {"ok": False, "error": str(e)[:200]})


@tool(
    _spec("http_check", "Lightweight HTTP status check (no JS, 8s timeout)."),
    description="Lightweight HTTP status check (no JS, 8s timeout).",
    graph=True,
    react=True,
    domain="it_ops",
)
def http_check(url: str = "http://localhost:8787/", timeout: float = 8.0) -> str:
    """GET with urllib, max 64KB body. No JS rendering (see deep_read for that)."""
    import time
    import urllib.request

    url = (url or "").strip()[:500]
    if not url.startswith(("http://", "https://")):
        return json_block("http_check", {"ok": False, "error": "url must start with http:// or https://"})
    try:
        timeout = max(1.0, min(float(timeout or 8.0), 15.0))
        req = urllib.request.Request(url, headers={"User-Agent": "Aiko-itops/1.0"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(65536)
            ms = round((time.monotonic() - start) * 1000, 1)
            return json_block("http_check", {"ok": True, "url": url, "status": int(resp.status),
                                             "latency_ms": ms, "bytes": len(body)})
    except Exception as e:
        return json_block("http_check", {"ok": True, "url": url, "open": False, "error": str(e)[:250]})


@tool(
    _spec("dns_lookup", "Resolve hostname to IP addresses (read-only)."),
    description="Resolve hostname to IP addresses (read-only).",
    graph=True,
    react=True,
    domain="it_ops",
)
def dns_lookup(hostname: str = "localhost") -> str:
    """getaddrinfo wrapper, deduped, max 10 addresses."""
    try:
        hostname = (hostname or "").strip()[:253]
        if not hostname:
            return json_block("dns_lookup", {"ok": False, "error": "hostname required"})
        infos = socket.getaddrinfo(hostname, None)
        addrs = sorted({info[4][0] for info in infos})[:10]
        return json_block("dns_lookup", {"ok": True, "hostname": hostname, "addresses": addrs})
    except socket.gaierror as e:
        return json_block("dns_lookup", {"ok": False, "hostname": hostname, "error": str(e)[:200]})
    except Exception as e:
        return json_block("dns_lookup", {"ok": False, "error": str(e)[:200]})


@tool(
    _spec("ssl_expiry", "Days until TLS cert expires for host:port (read-only)."),
    description="Days until TLS cert expires for host:port (read-only).",
    graph=True,
    react=True,
    domain="it_ops",
)
def ssl_expiry(host: str = "example.com", port: int = 443, timeout: float = 5.0) -> str:
    """Fetch peer cert, parse notAfter. No verification bypass beyond default ctx."""
    try:
        from datetime import datetime, timezone

        host = (host or "").strip()[:253]
        port = max(1, min(int(port or 443), 65535))
        timeout = max(1.0, min(float(timeout or 5.0), 10.0))
        if not host:
            return json_block("ssl_expiry", {"ok": False, "error": "host required"})
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                cert = tls.getpeercert()
        not_after = str((cert or {}).get("notAfter") or "")
        if not not_after:
            return json_block("ssl_expiry", {"ok": False, "host": host, "error": "no expiry in cert"})
        expiry = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        days = (expiry - datetime.now(timezone.utc)).days
        status = "ok" if days > 30 else ("warning" if days > 7 else "critical")
        return json_block("ssl_expiry", {"ok": True, "host": host, "port": port,
                                         "not_after": not_after, "days_left": days, "status": status})
    except Exception as e:
        return json_block("ssl_expiry", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("file_find", "Bounded filename search under repo/workspace (glob, max 50)."),
    description="Bounded filename search under repo/workspace (glob, max 50).",
    graph=True,
    react=True,
    domain="it_ops",
)
def file_find(pattern: str = "*.log", directory: str = "logs", limit: int = 20) -> str:
    """fnmatch over one directory tree, depth-safe, no content scan."""
    try:
        limit = max(1, min(int(limit or 20), MAX_FIND_RESULTS))
        pattern = (pattern or "*").strip()[:120] or "*"
        base = (REPO_ROOT / (directory or "").strip().lstrip("/\\")).resolve() if (directory or "").strip() else REPO_ROOT
        if base != REPO_ROOT and REPO_ROOT not in base.parents:
            return json_block("file_find", {"ok": False, "error": "directory escapes repository"})
        if not base.exists():
            return json_block("file_find", {"ok": False, "error": f"not found: {directory}"})
        matches = []
        skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
        for path in base.rglob("*"):
            if any(part in skip for part in path.relative_to(REPO_ROOT).parts if path != REPO_ROOT and REPO_ROOT in path.parents):
                continue
            if path.is_file() and fnmatch.fnmatch(path.name, pattern):
                try:
                    rel = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(path)
                matches.append(rel)
                if len(matches) >= limit:
                    break
        return json_block("file_find", {"ok": True, "pattern": pattern, "directory": directory, "count": len(matches), "files": matches})
    except Exception as e:
        return json_block("file_find", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("net_info", "Hostname + local IPs summary (read-only)."),
    description="Hostname + local IPs summary (read-only).",
    graph=True,
    react=True,
    domain="it_ops",
)
def net_info() -> str:
    """No external calls — gethostname + getaddrinfo only."""
    try:
        hostname = socket.gethostname()[:120]
        try:
            infos = socket.getaddrinfo(hostname, None)
            addrs = sorted({info[4][0] for info in infos})[:10]
        except socket.gaierror:
            addrs = []
        return json_block("net_info", {"ok": True, "hostname": hostname, "addresses": addrs})
    except Exception as e:
        return json_block("net_info", {"ok": False, "error": str(e)[:200]})


# ── Everyday ────────────────────────────────────────────────────────────

@tool(
    _spec("text_summarize", "Condense long text via LLM (heuristic fallback, Jetson-light)."),
    description="Condense long text via LLM (heuristic fallback, Jetson-light).",
    graph=True,
    react=True,
    domain="everyday",
)
def text_summarize(text: str = "", max_sentences: int = 5, *, client=None, model: str | None = None) -> str:
    """LLM summarize when client/model present, else first-N-sentences fallback.

    Keeps Ministral-3B prompts small: evidence capped at 3000 chars.
    """
    try:
        import re

        max_sentences = max(1, min(int(max_sentences or 5), 10))
        text = (text or "").strip()
        if not text:
            return json_block("text_summarize", {"ok": False, "error": "text required"})
        evidence = text[:3000]
        if client is not None and model:
            try:
                from agentic.toolkit.synthesize import synthesize_report

                out = synthesize_report(evidence=evidence,
                                        prompt=f"Summarize in at most {max_sentences} sentences.",
                                        style="plain", client=client, model=model)
                return json_block("text_summarize", {"ok": True, "mode": "llm", "summary": str(out)[:2000]})
            except Exception as e:
                log.debug("text_summarize llm fallback: %s", e)
        parts = re.split(r"(?<=[.!?])\s+", evidence)
        summary = " ".join(p for p in parts[:max_sentences if False else max_sentences])[:1500]
        return json_block("text_summarize", {"ok": True, "mode": "heuristic", "summary": summary})
    except Exception as e:
        return json_block("text_summarize", {"ok": False, "error": str(e)[:200]})


@tool(
    _spec("text_translate", "Translate short text via LLM (passthrough fallback)."),
    description="Translate short text via LLM (passthrough fallback).",
    graph=True,
    react=True,
    domain="everyday",
)
def text_translate(text: str = "", target: str = "Japanese", *, client=None, model: str | None = None) -> str:
    """Capped at 1500 chars so a 3B model stays coherent. No LLM = passthrough."""
    try:
        text = (text or "").strip()[:1500]
        target = (target or "Japanese").strip()[:40] or "Japanese"
        if not text:
            return json_block("text_translate", {"ok": False, "error": "text required"})
        if client is not None and model:
            try:
                from agentic.toolkit.synthesize import synthesize_report

                out = synthesize_report(evidence=text, prompt=f"Translate to {target}. Output translation only.",
                                        style="plain", client=client, model=model)
                return json_block("text_translate", {"ok": True, "mode": "llm", "target": target, "translation": str(out)[:2000]})
            except Exception as e:
                log.debug("text_translate llm fallback: %s", e)
        return json_block("text_translate", {"ok": True, "mode": "passthrough",
                                             "target": target, "translation": text,
                                             "note": "no LLM available — returned original"})
    except Exception as e:
        return json_block("text_translate", {"ok": False, "error": str(e)[:200]})


_UNIT_TABLE: dict[str, dict[str, float]] = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "mi": 1609.344},
    "mass": {"g": 0.001, "kg": 1.0, "lb": 0.45359237, "oz": 0.028349523125},
    "data": {"b": 1 / 8, "kb": 1000 / 8, "mb": 1e6 / 8, "gb": 1e9 / 8, "kib": 1024 / 8, "mib": 1024 ** 2 / 8, "gib": 1024 ** 3 / 8},
}


@tool(
    _spec("unit_convert", "Offline unit conversion (length/mass/temp/data)."),
    description="Offline unit conversion (length/mass/temp/data).",
    graph=True,
    react=True,
    domain="everyday",
)
def unit_convert(value: float = 1.0, from_unit: str = "km", to_unit: str = "mi") -> str:
    """Pure arithmetic, no network. Temp handled separately (C/F/K)."""
    try:
        src = (from_unit or "").strip().lower()
        dst = (to_unit or "").strip().lower()
        val = float(value)
        # Temperature
        if src in {"c", "f", "k"} and dst in {"c", "f", "k"}:
            if src == "c":
                c = val
            elif src == "f":
                c = (val - 32) * 5 / 9
            else:
                c = val - 273.15
            out = {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[dst]
            return json_block("unit_convert", {"ok": True, "value": val, "from": src, "to": dst, "result": round(out, 4)})
        for _cat, table in _UNIT_TABLE.items():
            if src in table and dst in table:
                base = val * table[src]
                out = base / table[dst]
                return json_block("unit_convert", {"ok": True, "value": val, "from": src, "to": dst, "result": round(out, 6)})
        return json_block("unit_convert", {"ok": False, "error": f"unsupported conversion: {from_unit} -> {to_unit}"})
    except Exception as e:
        return json_block("unit_convert", {"ok": False, "error": str(e)[:200]})


@tool(
    _spec("pass_gen", "Generate a random password (secrets, never logged)."),
    description="Generate a random password (secrets, never logged).",
    graph=True,
    react=True,
    domain="everyday",
)
def pass_gen(length: int = 20, symbols: bool = True) -> str:
    """secrets-based. Length 8-64. Caller must store it — we don't."""
    try:
        length = max(8, min(int(length or 20), 64))
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if bool(symbols):
            alphabet += "!@#$%^&*-_=+"
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        return json_block("pass_gen", {"ok": True, "length": length, "password": pwd})
    except Exception as e:
        return json_block("pass_gen", {"ok": False, "error": str(e)[:200]})


__all__ = [
    "sys_health", "process_top", "disk_large_files", "log_triage",
    "service_status", "docker_ps", "port_check", "http_check",
    "dns_lookup", "ssl_expiry", "file_find", "net_info",
    "text_summarize", "text_translate", "unit_convert", "pass_gen",
]
