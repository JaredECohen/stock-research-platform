"""Outbound-network guard for the test session.

The suite is network-free by construction: providers are gated off under
the demo-only flag pair and the LLM keys are blank in CI. This guard makes
that a property rather than a habit — any socket connect or DNS lookup to
a non-loopback host raises inside the caller, so a test that reaches for a
provider degrades exactly as it would on a CI runner with no network, and
the attempt is recorded and reported at the end of the session.

`RUN_LIVE_TESTS=1` (the opt-in for `pytest.mark.live`) leaves the network
alone; so does `MM_ALLOW_NETWORK=1` for ad-hoc local debugging.
"""
from __future__ import annotations

import os
import socket
from collections import defaultdict

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "", None}

_hits: dict[str, set[str]] = defaultdict(set)
_current = {"id": "<session>"}
_installed = False
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_getaddrinfo = socket.getaddrinfo


def enabled() -> bool:
    return os.environ.get("RUN_LIVE_TESTS") != "1" and os.environ.get("MM_ALLOW_NETWORK") != "1"


def _host(addr) -> str:
    return str(addr[0]) if isinstance(addr, tuple) else str(addr)


def _local(host) -> bool:
    return host in LOOPBACK or str(host).startswith("127.")


def _connect(self, addr):
    host = _host(addr)
    if not _local(host):
        _hits[_current["id"]].add(f"connect:{host}")
        raise OSError(f"netguard: outbound connection to {host} blocked in tests")
    return _orig_connect(self, addr)


def _connect_ex(self, addr):
    host = _host(addr)
    if not _local(host):
        _hits[_current["id"]].add(f"connect_ex:{host}")
        return 111
    return _orig_connect_ex(self, addr)


def _getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        _hits[_current["id"]].add(f"dns:{host}")
        raise socket.gaierror(f"netguard: DNS lookup of {host!r} blocked in tests")
    return _orig_getaddrinfo(host, *args, **kwargs)


def install() -> None:
    global _installed
    if _installed or not enabled():
        return
    socket.socket.connect = _connect          # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex    # type: ignore[method-assign]
    socket.getaddrinfo = _getaddrinfo         # type: ignore[assignment]
    _installed = True


def set_current(test_id: str) -> None:
    _current["id"] = test_id


def hits() -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in _hits.items() if v}
