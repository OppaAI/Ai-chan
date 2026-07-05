"""
core/health.py
Live hardware and OS telemetry for Aiko-chan's vitals bar.

Provides:
    _read_sys_info()  — one-shot startup snapshot (CPU, RAM, storage, OS)
    _ram_used_str()   — live RSS string, excludes reclaimable page cache
    _db_size_str()    — live Qdrant point count
    _fmt_uptime()     — HH:MM:SS formatter
"""

import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import urllib.request

from core.config import load_config
load_config()

_DB_SIZE_CACHE: tuple[float, str] = (0.0, "? mem")
_DB_SIZE_TTL = float(os.getenv("DB_SIZE_TTL", "1.0"))


def _read_sys_info() -> dict:
    """
    Sample the host environment at startup, reading raw hardware and OS signals
    from the kernel's exposed interfaces.

    Returns a dict containing:
        cpu          — model name string from /proc/cpuinfo or platform fallback
        ram_total_kb — total physical memory in kilobytes
        ram          — human-readable RAM string (e.g. '7.4 GB')
        storage      — root partition size from df
        os           — JetPack revision string, or PRETTY_NAME from /etc/os-release
    """
    info = {}

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key = line.split(":")[0].strip().lower()
                if key in ("model name", "hardware", "model"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if not info.get("cpu"):
        info["cpu"] = platform.processor() or platform.machine() or "unknown"

    # Total RAM (stored as KB for vitals calculations)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_total_kb"] = int(line.split()[1])
                    break
    except Exception:
        info["ram_total_kb"] = 0

    ram_kb = info.get("ram_total_kb", 0)
    ram_gb = ram_kb / 1024 / 1024
    info["ram"] = f"{ram_gb:.1f} GB" if ram_gb >= 1 else f"{ram_kb // 1024} MB"

    # Root partition size
    try:
        out = subprocess.check_output(["df", "-h", "/"], text=True).splitlines()[1]
        info["storage"] = out.split()[1]
    except Exception:
        info["storage"] = "unknown"

    # OS / runtime — JetPack first, then /etc/os-release
    try:
        with open("/etc/nv_tegra_release") as f:
            raw = f.readline().strip()
            m   = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', raw)
            info["os"] = f"JetPack R{m.group(1)}.{m.group(2)}" if m else raw[:40]
    except FileNotFoundError:
        try:
            with open("/etc/os-release") as f:
                d = {}
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        d[k] = v.strip('"')
            info["os"] = d.get("PRETTY_NAME", platform.version()[:40])
        except Exception:
            info["os"] = platform.version()[:40] or "unknown"

    return info


def _ram_used_str() -> str:
    """
    Read current memory pressure from the kernel and return a live usage string.

    Subtracts Cached and Buffers from the used total so the display reflects
    true process RSS rather than inflated kernel page-cache numbers.

    Returns a string of the form 'X.X/Y.Y GB', or '? GB' on read failure.
    """
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable", "Cached:", "Buffers")):
                    k, v = line.split(":")
                    vals[k.strip()] = int(v.split()[0])  # KB
                if len(vals) == 4:
                    break
        total     = vals.get("MemTotal",     0)
        available = vals.get("MemAvailable", 0)
        cached    = vals.get("Cached",       0)
        buffers   = vals.get("Buffers",      0)
        # True process RSS — excludes reclaimable page cache
        real_used = (total - available - cached - buffers) / 1024 / 1024
        total_gb  = total / 1024 / 1024
        return f"{real_used:.1f}/{total_gb:.1f} GB"
    except Exception:
        return "? GB"


def _db_size_str() -> str:
    """
    Probe the sqlite-vec memory store and return the number of living engrams.
    Cached briefly so the vitals refresh loop does not open SQLite every draw.
    Returns a string of the form 'N entries', or '? mem' on any error.
    """
    global _DB_SIZE_CACHE
    now = time.monotonic()
    cached_at, cached_value = _DB_SIZE_CACHE
    if now - cached_at < _DB_SIZE_TTL:
        return cached_value

    conn = None
    try:
        import sqlite3, sqlite_vec
        db_path = os.getenv("SQLITE_MEMORY_PATH", str(Path.home() / ".aiko" / "memory.db"))
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        value = f"{count} entries"
    except Exception:
        value = "? mem"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    _DB_SIZE_CACHE = (now, value)
    return value


def _fmt_uptime(seconds: float) -> str:
    """
    Format a raw elapsed-seconds value into a human-readable session age string.

    Args:
        seconds: Elapsed time in seconds since the session awakened.

    Returns:
        A string of the form 'HH:MM:SS'.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"
