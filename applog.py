"""
In-memory application log for the Admin panel.

Everything the app prints (model loading, generations, downloads, errors) is kept in a ring
buffer of recent lines so the Admin > Logs tab can show it without reading files. Install once
at start-up with install(); it tees sys.stdout / sys.stderr, so the console still sees everything.

    from applog import log, tail
    log("Loading model ...")          # timestamped line, also printed to the console
    tail(200, contains="LoRA")        # last lines, newest last
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

MAX_LINES = 3000
_lines: deque[tuple[float, str, str]] = deque(maxlen=MAX_LINES)   # (unix time, level, text)
_lock = threading.Lock()
_installed = False


def _add(level: str, text: str) -> None:
    text = text.rstrip()
    if not text:
        return
    with _lock:
        _lines.append((time.time(), level, text))


def log(message: str, level: str = "info") -> None:
    """Record a line and print it to the console."""
    print(message, flush=True)       # the tee records it with a timestamp when installed
    if not _installed:
        _add(level, message)


_ERROR_WORDS = re.compile(r"error|exception|traceback|failed|critical", re.IGNORECASE)


def _classify(line: str, stream_level: str) -> str:
    """stderr carries progress bars and warnings too, so only lines that look like errors count as errors."""
    if _ERROR_WORDS.search(line):
        return "error"
    return "info" if stream_level in ("info", "stderr") else stream_level


class _Tee:
    """File-like object that forwards writes to the real stream and the ring buffer."""

    def __init__(self, stream, level: str):
        self._stream = stream
        self._level = level
        self._buffer = ""

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.split("\r")[-1]          # progress bars rewrite the line; keep the final state only
            if line.strip():
                _add(_classify(line, self._level), line)
        if len(self._buffer) > 20000:            # a progress bar without newlines must not grow unbounded
            self._buffer = self._buffer[-2000:]
        return len(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Tee stdout and stderr into the buffer (idempotent)."""
    global _installed
    if _installed:
        return
    sys.stdout = _Tee(sys.stdout, "info")
    sys.stderr = _Tee(sys.stderr, "stderr")
    _installed = True
    _add("info", f"log started {datetime.now().isoformat(timespec='seconds')}")


def tail(n: int = 200, contains: str = "", level: str | None = None) -> list[tuple[float, str, str]]:
    needle = contains.lower().strip()
    with _lock:
        rows = list(_lines)
    if level:
        rows = [r for r in rows if r[1] == level]
    if needle:
        rows = [r for r in rows if needle in r[2].lower()]
    return rows[-n:]


def tail_text(n: int = 200, contains: str = "", level: str | None = None) -> str:
    out = []
    for ts, lvl, text in tail(n, contains, level):
        stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        mark = "!" if lvl == "error" else " "
        out.append(f"{stamp} {mark} {text}")
    return "\n".join(out) if out else "(no log lines yet)"


def counts() -> dict[str, int]:
    with _lock:
        rows = list(_lines)
    return {"lines": len(rows), "errors": sum(1 for r in rows if r[1] == "error")}
