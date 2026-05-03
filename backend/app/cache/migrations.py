"""Wave 6D — schema migration registry for cached snapshots.

Each `ResearchSnapshot` carries a `schema_version` int. When code evolves
(new fields, renamed keys), older snapshots already in the DB still need
to deserialize cleanly. This module maps `(kind, from_version) -> upgrader`
so a read-time call can transform an old payload into the current shape
without the application layer caring which version it came from.

Design points:
- Upgraders are registered explicitly via `register(kind, from_version, fn)`.
  No magic global registry — keeping registration explicit makes diffs
  easy to review and prevents accidental cross-kind contamination.
- A version chain (e.g., v1 → v2 → v3) is walked one step at a time so
  any intermediate upgrade is exercised. This means each migration only
  needs to know how to go up by one.
- Upgraders are pure functions: `(payload: dict) -> dict`. No I/O, no
  side effects. Easy to unit-test in isolation.
- Reads that miss an upgrader for a step return the payload unchanged
  with a logged warning (defense — if someone bumps the version without
  a migration, we'd rather pass through the old shape than blow up at
  runtime).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


Upgrader = Callable[[Dict[str, Any]], Dict[str, Any]]


# Registry: `(kind, from_version) -> upgrader_fn`.
# Each entry upgrades a payload from `from_version` to `from_version + 1`.
_REGISTRY: Dict[Tuple[str, int], Upgrader] = {}


def register(kind: str, from_version: int, upgrader: Upgrader) -> None:
    """Register an upgrader for `(kind, from_version) -> from_version + 1`.

    Re-registration overwrites silently — useful for tests that want to
    swap an upgrader without unregistering first.
    """
    if from_version < 1:
        raise ValueError("from_version must be >= 1")
    _REGISTRY[(kind, from_version)] = upgrader


def unregister(kind: str, from_version: int) -> None:
    _REGISTRY.pop((kind, from_version), None)


def registered() -> List[Tuple[str, int]]:
    """Return all registered (kind, from_version) keys — for testing/audit."""
    return sorted(_REGISTRY.keys())


def latest_version(kind: str) -> int:
    """The highest known target version for `kind`.

    Computed as `max(from_version for (k, _) in registry if k == kind) + 1`,
    defaulting to 1 if `kind` has no registered upgraders.
    """
    versions = [fv for k, fv in _REGISTRY if k == kind]
    return (max(versions) + 1) if versions else 1


def upgrade_payload(
    kind: str, payload: Dict[str, Any], target_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk `payload` from its embedded `schema_version` up to `target_version`.

    `target_version` defaults to the registry's latest known target for
    `kind`. Each step calls the upgrader registered for the current
    `(kind, from_version)` pair; missing upgraders are skipped with a
    warning (the payload passes through unchanged).

    Returns a *new* dict (the upgraders may mutate input in place, but
    the caller-facing contract is value-stable).
    """
    if not isinstance(payload, dict):
        return payload  # nothing to upgrade
    current = int(payload.get("schema_version") or 1)
    target = target_version if target_version is not None else latest_version(kind)
    if current >= target:
        return payload
    out = dict(payload)
    while current < target:
        fn = _REGISTRY.get((kind, current))
        if fn is None:
            log.warning(
                "No migration registered for kind=%s v%d → v%d; passing through",
                kind, current, current + 1,
            )
            current += 1
            continue
        try:
            out = fn(out) or out
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "Migration kind=%s v%d → v%d raised %s; aborting upgrade chain",
                kind, current, current + 1, exc,
            )
            return out
        current += 1
        out["schema_version"] = current
    return out


# ---------------------------------------------------------------------------
# Cache-read integration
# ---------------------------------------------------------------------------

def upgrade_snapshot_payload(snap: Any) -> Dict[str, Any]:
    """Apply migrations to a `ResearchSnapshot` row's payload, returning the
    upgraded dict. Convenience wrapper for read-side callers.

    Doesn't write the upgraded payload back to the DB — read-time upgrade
    keeps the migration cost low (each row touched once per read, then
    cached upstream by whoever the consumer is).
    """
    if snap is None or not isinstance(getattr(snap, "payload", None), dict):
        return getattr(snap, "payload", {}) or {}
    payload = snap.payload
    return upgrade_payload(snap.kind, payload)
