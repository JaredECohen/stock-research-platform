"""Process memory instrumentation + allocator trim.

Added after a production OOM-kill on Render (2026-08-12). The failure
mode that motivated this: `vector_store.search` loaded an entire
ticker's — and, on a missing-ticker path, the entire corpus's —
`doc_chunks` rows into Python, each embedding costing ~48 KB as a list
of 1536 boxed floats. The instance hit its memory ceiling and was
SIGKILL'd, which bypasses Python entirely: no traceback, no log line,
nothing in the `try/except` around the call. Diagnosing it took a code
audit because the process left no evidence behind.

Two jobs here:

1. **`log_rss`** — cheap RSS breadcrumbs at expensive boundaries (memo
   runs, vector searches). The next time an instance dies we want the
   last log line to say how much memory was resident and what was
   running, rather than nothing at all.

2. **`trim_memory`** — `gc.collect()` plus glibc `malloc_trim(0)`.
   CPython returning objects to its own freelists does not return pages
   to the OS, and with ~15 threads (uvicorn + regen worker + the
   APScheduler pool) glibc spreads allocations across many arenas that
   each retain their high-water mark. Render bills and kills on RSS, so
   after a memo regen — the single largest allocator in the process — we
   explicitly hand the pages back.

Both are best-effort and never raise: this module is diagnostics, and
diagnostics must not be able to take down a memo run.
"""
from __future__ import annotations

import gc
import logging
import os
import platform
import resource
from typing import Optional

log = logging.getLogger(__name__)

# `ru_maxrss` is bytes on macOS/BSD and kilobytes on Linux. Resolve once
# rather than guessing from the magnitude of a sample (a 9.5 MB
# interpreter and a 9.5 GB one are indistinguishable that way).
_RU_MAXRSS_DIVISOR = 1024 * 1024 if platform.system() == "Darwin" else 1024


def rss_mb() -> Optional[float]:
    """Current process RSS in MB, or None when unavailable.

    Prefers `/proc/self/statm` on Linux — that is the *live* RSS, which
    is what Render's limit is enforced against. `ru_maxrss` is only ever
    the high-water mark, so it never falls after a trim and would make
    `trim_memory` look like a no-op.
    """
    try:
        with open("/proc/self/statm", "rb") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        pass
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RU_MAXRSS_DIVISOR
    except Exception:  # pragma: no cover — platform without getrusage
        return None


def peak_rss_mb() -> Optional[float]:
    """High-water-mark RSS in MB, or None when unavailable."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RU_MAXRSS_DIVISOR
    except Exception:  # pragma: no cover
        return None


def log_rss(label: str, **context: object) -> Optional[float]:
    """Log current + peak RSS against `label`. Returns current RSS in MB.

    Emitted at INFO so it survives production log levels — these lines
    are the evidence trail for the next OOM, and are useless if they are
    filtered out by default.
    """
    cur = rss_mb()
    if cur is None:  # pragma: no cover — platform without getrusage
        return None
    peak = peak_rss_mb()
    extra = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
    log.info(
        "rss %s: %.1f MB (peak %.1f MB)%s",
        label, cur, peak if peak is not None else cur,
        f" {extra}" if extra else "",
    )
    return cur


def trim_memory(label: str = "") -> Optional[float]:
    """`gc.collect()` + glibc `malloc_trim(0)`. Returns MB released.

    No-ops safely off glibc (macOS, musl/Alpine): `malloc_trim` simply
    will not resolve and we keep the `gc.collect()` benefit. Returns
    None when RSS can't be read.
    """
    before = rss_mb()
    try:
        gc.collect()
    except Exception:  # pragma: no cover
        pass
    try:
        import ctypes
        # Only glibc exposes malloc_trim. `ctypes.CDLL(None)` looks up the
        # symbol in the already-loaded process image, which avoids
        # hardcoding a soname ("libc.so.6" is absent on musl and macOS).
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
    except Exception as exc:  # pragma: no cover — non-glibc platform
        log.debug("malloc_trim unavailable: %s", exc)
    after = rss_mb()
    if before is None or after is None:
        return None
    released = before - after
    log.info(
        "trim_memory%s: %.1f MB -> %.1f MB (released %.1f MB)",
        f" [{label}]" if label else "", before, after, released,
    )
    return released
