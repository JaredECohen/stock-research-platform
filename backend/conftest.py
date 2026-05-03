"""Pytest config.

Three responsibilities:

1. Ensure the `backend/` directory is on `sys.path` so `app.*` imports work
   regardless of where pytest is launched from.
2. Force `ENABLE_LIVE_DATA=false` for the test session, BEFORE the app's
   `Settings` is constructed. This makes test runs deterministic and zero-
   cost even when the developer has live provider keys in their `.env`
   (or `config.env` ships with `ENABLE_LIVE_DATA=true` as the production
   default). Tests should never hit real provider APIs unless they
   explicitly opt in via mocking — this guard prevents accidental spend.
3. Wave 4B: register a `live` marker for tests that intentionally hit
   real provider APIs. They are skipped by default; set `RUN_LIVE_TESTS=1`
   to opt in. Tests under that marker also flip `ENABLE_LIVE_DATA=true`
   on a per-test basis via the `live_settings` fixture.
"""
import os
import sys
from pathlib import Path

import pytest

# Set BEFORE any `app.config` import so pydantic-settings sees it as the
# winning value over both config.env and .env. Note: live-tagged tests use
# the `live_settings` fixture to flip these per-test, so the session-wide
# default stays "demo".
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Wave 4B — live integration test marker
# ---------------------------------------------------------------------------

_RUN_LIVE = os.environ.get("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")


def pytest_configure(config):
    """Register the `live` marker so tests can opt into real API calls."""
    config.addinivalue_line(
        "markers",
        "live: integration test that hits real provider APIs. Skipped "
        "unless RUN_LIVE_TESTS=1 is set in the environment.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip every `@pytest.mark.live` test by default."""
    if _RUN_LIVE:
        return
    skip_live = pytest.mark.skip(
        reason="live integration test — set RUN_LIVE_TESTS=1 to enable",
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def live_settings(monkeypatch):
    """Per-test fixture for live-tagged tests.

    Flips `ENABLE_LIVE_DATA=true` and `USE_DEMO_DATA=false` for the
    duration of the test so the data_service routes through real
    providers. Scoped to the test, then restored — does not affect
    other tests in the same session.

    Tests using this fixture should already be skipped without
    `RUN_LIVE_TESTS=1`, so the explicit env flip is a defense-in-depth
    safeguard rather than a primary gate.
    """
    monkeypatch.setenv("ENABLE_LIVE_DATA", "true")
    monkeypatch.setenv("USE_DEMO_DATA", "false")
    # The settings singleton is cached at module import; mutate the
    # in-memory object directly so this test sees live mode.
    from app.config import settings
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(settings, "use_demo_data", False)
    yield settings
