"""Pytest harness — register the DemoProvider fixture for the test session.

Wave 9b: production no longer falls back to demo data when a live
provider misses; tests that hit `data_service` would otherwise return
empty results. The autouse fixture here injects `DemoProvider` at the
head of every capability chain so tests get deterministic, network-free
responses without any per-test wiring.

To exercise the live provider chain in a specific test, opt out:

    @pytest.mark.live_only
    def test_real_fmp_call(): ...

(See `test_live_marker_gate.py` for the marker-handling logic.)
"""
from __future__ import annotations

import pytest

from app.services.data_service import get_data_service
from app.tests.fixtures.demo_provider import DemoProvider


@pytest.fixture(autouse=True, scope="session")
def _register_demo_provider():
    """Wire the in-memory demo provider into `data_service` for the entire
    test session. Cleared on teardown so subprocess pytests don't leak
    state."""
    ds = get_data_service()
    provider = DemoProvider()
    ds.register_test_provider(provider)
    yield provider
    ds.register_test_provider(None)
