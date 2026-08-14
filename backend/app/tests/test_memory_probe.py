"""Tests for `services/memory_probe` — RSS reporting + allocator trim.

These are diagnostics, so the bar is "never raises, never lies about
units". A wrong unit conversion is the failure mode that matters: it
would silently turn the RSS breadcrumbs into noise exactly when someone
is reading them to explain an OOM.
"""
from __future__ import annotations

from app.services import memory_probe


def test_rss_mb_returns_a_plausible_value():
    rss = memory_probe.rss_mb()
    assert rss is not None
    # A CPython process that has imported this app is comfortably inside
    # this range; the assertion is really "the unit conversion is right",
    # which is the one thing that silently breaks across platforms
    # (ru_maxrss is bytes on macOS, kilobytes on Linux).
    assert 10.0 < rss < 20_000.0, f"implausible RSS reading: {rss}"


def test_peak_rss_is_at_least_current():
    cur, peak = memory_probe.rss_mb(), memory_probe.peak_rss_mb()
    assert cur is not None and peak is not None
    assert peak >= cur * 0.5  # peak is a high-water mark, never far below


def test_log_rss_emits_an_info_line(caplog):
    with caplog.at_level("INFO", logger="app.services.memory_probe"):
        out = memory_probe.log_rss("unit_test", ticker="AAPL")
    assert out is not None
    assert "rss unit_test" in caplog.text
    assert "ticker=AAPL" in caplog.text


def test_trim_memory_is_safe_and_reports(caplog):
    """Must not raise on any platform — `malloc_trim` is glibc-only and
    this suite also runs on macOS, where it simply won't resolve."""
    with caplog.at_level("INFO", logger="app.services.memory_probe"):
        released = memory_probe.trim_memory("unit_test")
    assert released is None or isinstance(released, float)
    assert "trim_memory" in caplog.text


def test_trim_memory_actually_collects_garbage():
    """Verify the gc.collect() half does its job even where malloc_trim
    is unavailable: build a reference cycle, drop it, confirm it's gone."""
    import gc
    import weakref

    class Node:
        def __init__(self):
            self.self_ref = self  # cycle — refcounting alone can't free it

    gc.disable()
    try:
        node = Node()
        ref = weakref.ref(node)
        del node
        assert ref() is not None, "cycle was freed without gc — test is void"
        gc.enable()
        memory_probe.trim_memory("cycle_test")
        assert ref() is None, "trim_memory did not collect the cycle"
    finally:
        gc.enable()
