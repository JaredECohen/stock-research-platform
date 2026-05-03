"""Wave 6D tests — schema migration registry.

Covers:
- Empty registry → `latest_version` returns 1; no-op upgrade.
- A single registered upgrader bumps a v1 payload to v2.
- A v1 → v3 chain walks both v1→v2 and v2→v3 in order.
- Missing intermediate upgrader is logged and skipped (payload passes
  through with the version bumped).
- Upgrader exception aborts the chain at the failing step but doesn't
  raise out to the caller.
- `cache_get` applies migrations on read — a payload stored at v1 with
  a registered v1→v2 upgrader comes out at v2.
"""
from __future__ import annotations

import pytest

from app.cache import migrations as M
from app.cache import snapshots as cache_snapshots


def _isolate_registry():
    """Clear the global registry before/after each test."""
    M._REGISTRY.clear()


def test_latest_version_default_is_1():
    _isolate_registry()
    assert M.latest_version("anything") == 1


def test_register_and_upgrade_single_step():
    _isolate_registry()
    M.register(
        "test_kind", from_version=1,
        upgrader=lambda p: {**p, "added_in_v2": True},
    )
    assert M.latest_version("test_kind") == 2
    out = M.upgrade_payload("test_kind", {"schema_version": 1, "x": "y"})
    assert out["schema_version"] == 2
    assert out["added_in_v2"] is True
    assert out["x"] == "y"


def test_chain_v1_to_v3_walks_both_steps():
    _isolate_registry()
    M.register("ck", 1, lambda p: {**p, "step": "v2"})
    M.register("ck", 2, lambda p: {**p, "step2": "v3"})
    assert M.latest_version("ck") == 3
    out = M.upgrade_payload("ck", {"schema_version": 1})
    assert out["schema_version"] == 3
    assert out["step"] == "v2"
    assert out["step2"] == "v3"


def test_payload_already_at_target_returns_unchanged():
    _isolate_registry()
    M.register("kind", 1, lambda p: {**p, "added": True})
    payload = {"schema_version": 2, "x": 1}
    out = M.upgrade_payload("kind", payload)
    assert out == payload


def test_target_version_explicit_caps_chain():
    _isolate_registry()
    M.register("kind", 1, lambda p: {**p, "v2": True})
    M.register("kind", 2, lambda p: {**p, "v3": True})
    out = M.upgrade_payload("kind", {"schema_version": 1}, target_version=2)
    assert out["schema_version"] == 2
    assert "v3" not in out


def test_missing_intermediate_upgrader_is_skipped(caplog):
    """When a step has no registered upgrader, the version bumps and the
    payload passes through unchanged — surfaced as a warning."""
    _isolate_registry()
    M.register("kind", 1, lambda p: {**p, "from_v1": True})
    # Skip v2→v3 by registering only v3→v4. latest_version=4 in this case.
    M.register("kind", 3, lambda p: {**p, "from_v3": True})
    with caplog.at_level("WARNING"):
        out = M.upgrade_payload("kind", {"schema_version": 1})
    assert "No migration registered" in caplog.text
    assert out["from_v1"] is True
    # v2→v3 was skipped silently; v3→v4 still ran.
    assert out["from_v3"] is True
    assert out["schema_version"] == 4


def test_upgrader_exception_does_not_raise():
    _isolate_registry()

    def boom(payload):
        raise RuntimeError("intentional")

    M.register("kind", 1, boom)
    out = M.upgrade_payload("kind", {"schema_version": 1, "x": 1})
    # Exception caught, chain aborted; payload returns at the version it failed at.
    assert out["x"] == 1
    assert out["schema_version"] == 1


def test_register_rejects_zero_or_negative_version():
    with pytest.raises(ValueError):
        M.register("kind", 0, lambda p: p)


def test_unregister_removes_entry():
    _isolate_registry()
    M.register("kind", 1, lambda p: p)
    assert ("kind", 1) in M.registered()
    M.unregister("kind", 1)
    assert ("kind", 1) not in M.registered()


def test_upgrade_payload_handles_non_dict_input():
    _isolate_registry()
    assert M.upgrade_payload("kind", "string") == "string"


# ---------------------------------------------------------------------------
# cache_get integration — read-time upgrade
# ---------------------------------------------------------------------------

def test_cache_get_applies_migrations_on_read():
    _isolate_registry()
    M.register(
        "wave6d_test", 1,
        lambda p: {**p, "fresh_field_added": True},
    )
    # Write a payload at v1 (the cache_put default).
    cache_snapshots.cache_put(
        "TST_WAVE6D", "wave6d_test",
        payload={"value": 42},
        schema_version=1,
    )
    # Read it back — schema_version=1 stored, but the upgrader runs.
    got = cache_snapshots.cache_get("TST_WAVE6D", "wave6d_test")
    assert got is not None
    assert isinstance(got.payload, dict)
    assert got.payload.get("fresh_field_added") is True
    assert got.payload.get("value") == 42
    assert got.payload.get("schema_version") == 2
