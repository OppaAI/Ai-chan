"""Event-driven watcher for the knowledge drop folder.

Replaces the periodic interval scan of the workspace knowledge folder with
an inotify-based watcher: zero idle CPU (blocks in the kernel on select),
wakes the instant a file lands, and triggers the knowledge ingest handler.

Linux-only (inotify). On platforms or environments without inotify, falls
back to nothing (the interval scan remains available as a safety net).
"""
from __future__ import annotations

import ctypes
import os
import select
import struct
import threading
import time
from pathlib import Path

from system.log import get_logger
from system.userspace import user_workspace_root

log = get_logger(__name__)

# ── inotify constants (Linux ABI) ─────────────────────────────────────────────
_IN_NONBLOCK = 0o4000  # O_NONBLOCK
_IN_CLOEXEC = 0o2000000
_IN_CREATE = 0x00000100
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_WATCH_MASK = _IN_CREATE | _IN_CLOSE_WRITE | _IN_MOVED_TO

_INOTIFY_EVENT = struct.Struct("iIII")  # wd, mask, cookie, len


class KnowledgeFolderWatcher:
    """Watches <workspace>/<knowledge_dir> and triggers ingest on new files."""

    def __init__(
        self,
        *,
        knowledge_dir: str = "library",
        debounce_seconds: float = 1.5,
        on_files=None,
    ):
        self._knowledge_dir = knowledge_dir
        self._debounce = debounce_seconds
        self._on_files = on_files
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._wd: int | None = None

    @property
    def folder(self) -> Path:
        return (user_workspace_root() / self._knowledge_dir).resolve()

    def start(self) -> bool:
        """Start the watcher thread. Returns False when inotify is unavailable."""
        if self._thread is not None and self._thread.is_alive():
            return True
        try:
            self._fd = _inotify_init()
        except Exception:
            log.warning("knowledge watcher: inotify unavailable — using interval scan only")
            return False
        try:
            folder = self.folder
            folder.mkdir(parents=True, exist_ok=True)
            self._wd = _inotify_add_watch(self._fd, str(folder), _WATCH_MASK)
            log.info("knowledge watcher: watching %s", folder)
        except Exception as exc:
            _inotify_close(self._fd)
            self._fd = None
            log.warning("knowledge watcher: failed to add watch: %s", exc)
            return False
        self._thread = threading.Thread(target=self._run, name="aiko-knowledge-watch", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._fd is not None:
            _inotify_close(self._fd)
            self._fd = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Block in kernel until an inotify event or stop-wakeup window.
                r, _, _ = select.select([self._fd], [], [], 0.5)
            except OSError:
                continue
            if not r:
                continue

            pending: list[Path] = []
            while True:
                try:
                    data = os.read(self._fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                pending.extend(_parse_events(data, self.folder))

            # Debounce: wait for the copy to settle, then fire once.
            time.sleep(self._debounce)
            if not pending:
                continue

            new_files = [p for p in pending if p.is_file() and not p.name.startswith(".")]
            if not new_files:
                continue
            log.info("knowledge watcher: %d new file(s) detected", len(new_files))
            if self._on_files is not None:
                try:
                    self._on_files(new_files)
                except Exception:
                    log.exception("knowledge watcher: ingest callback failed")


def _inotify_init() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    return fd


def _inotify_add_watch(fd: int, path: str, mask: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    wd = libc.inotify_add_watch(fd, path.encode(), ctypes.c_uint32(mask))
    if wd < 0:
        raise OSError(ctypes.get_errno(), f"inotify_add_watch failed for {path}")
    return wd


def _inotify_close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _parse_events(data: bytes, folder: Path) -> list[Path]:
    paths: list[Path] = []
    offset = 0
    while offset < len(data):
        wd, mask, _cookie, name_len = _INOTIFY_EVENT.unpack_from(data, offset)
        offset += _INOTIFY_EVENT.size
        name = data[offset : offset + name_len].rstrip(b"\x00").decode("utf-8", errors="replace")
        offset += name_len
        if mask & _WATCH_MASK and name:
            paths.append(folder / name)
    return paths
