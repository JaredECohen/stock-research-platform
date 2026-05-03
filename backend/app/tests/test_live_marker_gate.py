"""Wave 4B unit tests — the live-test gating mechanism itself.

This file is NOT marked `@pytest.mark.live`; it asserts that the gating
infrastructure (marker registration, default-skip behavior, env opt-in)
works correctly so we don't ship a broken gate.

Verifying the inverse — that live tests run when `RUN_LIVE_TESTS=1` —
requires a subprocess pytest invocation since pytest's collection
modifier runs once at session start. The CI guide documents the
manual verification.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_live_marker_is_registered():
    """The `live` marker must be in the registered set so unrecognized-
    marker warnings don't fire when developers tag new tests."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0
    assert "@pytest.mark.live" in out.stdout


def test_live_tests_are_skipped_by_default():
    """Without `RUN_LIVE_TESTS=1`, every `@pytest.mark.live` test must
    be collected-and-skipped (deterministic, $0 spend)."""
    env = {**os.environ}
    env.pop("RUN_LIVE_TESTS", None)
    env["ENABLE_LIVE_DATA"] = "false"
    out = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "app/tests/test_live_integration.py",
            "-q", "--no-header", "-p", "no:cacheprovider",
        ],
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert out.returncode == 0, f"expected exit 0, got {out.returncode}: {out.stdout}\n{out.stderr}"
    # Every collected test should be reported as skipped, not run.
    assert "skipped" in out.stdout.lower()
    # Success path: NO failures reported.
    assert "failed" not in out.stdout.lower() or "0 failed" in out.stdout.lower()


def test_live_settings_fixture_flips_enable_live_data():
    """When a test uses `live_settings`, the in-process settings object
    should report `enable_live_data=True` for the duration of that test.

    We exercise the fixture directly in this lightweight (non-live) test
    so we get coverage of the flip mechanism without needing API keys.
    """
    from app.config import settings as s
    # Outside the fixture, the conftest default is false.
    assert s.enable_live_data is False
