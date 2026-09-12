"""`auth/usage.py` — atomic, idempotent monthly meters (FEAT-002, S1).

The race test is deterministic on purpose: a thread race on a file
SQLite DB is a `database is locked` flake, not a test. Instead the
competing reservation is injected at the exact point between "counter
row exists" and "conditional increment", which is the interleaving the
conditional UPDATE has to survive.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from app.auth import usage
from app.database import SessionLocal


def _uid() -> int:
    # Distinct per test so the shared DB never couples tests.
    return int(uuid.uuid4().int % 1_000_000_000)


def _key() -> str:
    return "k-" + uuid.uuid4().hex


NOW = datetime(2026, 9, 8, 12, 0, 0)


def test_period_key_and_bounds():
    assert usage.period_key(NOW) == "2026-09"
    assert usage.period_bounds("2026-09") == (datetime(2026, 9, 1), datetime(2026, 10, 1))
    assert usage.period_bounds("2026-12") == (datetime(2026, 12, 1), datetime(2027, 1, 1))
    assert usage.resets_at(NOW) == datetime(2026, 10, 1)


def test_reserve_commit_lifecycle():
    uid = _uid()
    with SessionLocal() as db:
        r = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(),
                          resource_ref="NVDA", plan_at_charge="free", now=NOW)
        assert r.allowed and r.event is not None and r.event.status == "reserved"
        assert r.used == 1 and r.limit == 1 and r.remaining == 0
        assert usage.commit(db, r.event.id) is True
        assert usage.commit(db, r.event.id) is False, "commit is idempotent"
        assert usage.release(db, r.event.id) is False, "a committed charge cannot be released"
        assert usage.used(db, uid, "research_run", "2026-09") == 1


def test_quota_refuses_and_leaves_no_event():
    uid = _uid()
    with SessionLocal() as db:
        assert usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW).allowed
        second = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW)
        assert not second.allowed and second.event is None
        assert second.used == 1 and second.limit == 1
        assert len(usage.history(db, uid, "2026-09")) == 1


def test_release_gives_the_unit_back():
    uid = _uid()
    with SessionLocal() as db:
        r = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW)
        assert usage.release(db, r.event.id) is True
        assert usage.release(db, r.event.id) is False, "release is idempotent"
        assert usage.used(db, uid, "research_run", "2026-09") == 0
        again = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW)
        assert again.allowed


def test_retry_with_the_same_idempotency_key_never_double_charges():
    uid = _uid()
    key = _key()
    with SessionLocal() as db:
        first = usage.reserve(db, user_id=uid, feature="pm_chat", limit=10, idempotency_key=key, now=NOW)
        retry = usage.reserve(db, user_id=uid, feature="pm_chat", limit=10, idempotency_key=key, now=NOW)
        assert retry.replayed and retry.event.id == first.event.id
        assert usage.used(db, uid, "pm_chat", "2026-09") == 1
        # A replay never releases the original's charge on its own
        # (`Grant.release` skips replayed grants); the original may.
        assert usage.used(db, uid, "pm_chat", "2026-09") == 1
        assert usage.commit(db, first.event.id) is True
        again = usage.reserve(db, user_id=uid, feature="pm_chat", limit=10, idempotency_key=key, now=NOW)
        assert again.replayed and again.allowed and again.event.status == "committed"
        assert usage.used(db, uid, "pm_chat", "2026-09") == 1


def test_released_key_is_reserved_afresh_not_refused():
    """Regression: a released event under a key used to replay as
    `allowed=False`, so a memo open that failed after reserving (the key
    for distinct-resource features is user:feature:month:TICKER by
    design) refused that ticker for the rest of the month with a
    "quota exceeded" the counter contradicted. A released charge was
    given back; the key must be reservable again, on the same row."""
    uid = _uid()
    key = _key()
    with SessionLocal() as db:
        first = usage.reserve(db, user_id=uid, feature="memo_view", limit=3, idempotency_key=key,
                              resource_ref="NVDA", plan_at_charge="free", now=NOW)
        assert first.allowed
        assert usage.release(db, first.event.id) is True
        assert usage.used(db, uid, "memo_view", "2026-09") == 0
        assert usage.distinct_resources(db, uid, "memo_view", "2026-09") == set()

        again = usage.reserve(db, user_id=uid, feature="memo_view", limit=3, idempotency_key=key,
                              resource_ref="NVDA", plan_at_charge="pro", now=NOW)
        assert again.allowed and not again.replayed, "the re-reservation is this caller's own"
        assert again.event.id == first.event.id, "same row, flipped back to reserved"
        assert again.event.status == "reserved" and again.event.finalized_at is None
        assert again.event.plan_at_charge == "pro" and again.event.created_at == NOW
        assert again.used == 1 and again.remaining == 2
        assert usage.distinct_resources(db, uid, "memo_view", "2026-09") == {"NVDA"}
        # And it can be committed like any reservation; one event row, not two.
        assert usage.commit(db, again.event.id) is True
        assert len(usage.history(db, uid, "2026-09")) == 1


def test_released_key_still_respects_the_limit():
    """Re-reserving a released key goes through the same guarded counter
    UPDATE: with the allowance spent by other keys it is refused and the
    released row stays released (nothing charged, nothing flipped)."""
    uid = _uid()
    key = _key()
    with SessionLocal() as db:
        first = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=key, now=NOW)
        usage.release(db, first.event.id)
        other = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW)
        assert other.allowed
        refused = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=key, now=NOW)
        assert not refused.allowed and refused.event is None
        assert refused.used == 1 and refused.limit == 1
        assert usage.find_event(db, key).status == "released"
        assert usage.used(db, uid, "research_run", "2026-09") == 1


def test_two_retries_racing_for_a_released_key_charge_once(monkeypatch):
    """Both find the released row; the guarded `status='released'` flip
    lets exactly one through, and the loser replays the winner's
    reservation without a second counter increment."""
    uid = _uid()
    key = _key()
    with SessionLocal() as db:
        first = usage.reserve(db, user_id=uid, feature="pm_chat", limit=5, idempotency_key=key, now=NOW)
        first_id = first.event.id
        usage.release(db, first_id)
    original = usage._insert_counter_if_missing
    state = {"injected": False}

    def interleave(db, user_id, feature, pk, now):
        if not state["injected"]:
            state["injected"] = True
            with SessionLocal() as other:
                b = usage.reserve(other, user_id=user_id, feature=feature, limit=5, idempotency_key=key, now=now)
                state["b"] = (b.allowed, b.replayed)
        original(db, user_id, feature, pk, now)

    monkeypatch.setattr(usage, "_insert_counter_if_missing", interleave)
    with SessionLocal() as db:
        a = usage.reserve(db, user_id=uid, feature="pm_chat", limit=5, idempotency_key=key, now=NOW)
        assert state["injected"]
        assert state["b"] == (True, False), "the injected retry made the reservation"
        assert a.allowed and a.replayed, "the outer retry replays it"
        assert a.event.id == first_id
        assert usage.used(db, uid, "pm_chat", "2026-09") == 1
        assert len(usage.history(db, uid, "2026-09")) == 1


def test_unlimited_still_counts():
    uid = _uid()
    with SessionLocal() as db:
        for _ in range(3):
            r = usage.reserve(db, user_id=uid, feature="memo_view", limit=None, idempotency_key=_key(), now=NOW)
            assert r.allowed and r.remaining is None
        assert usage.used(db, uid, "memo_view", "2026-09") == 3


def test_period_rollover_starts_a_fresh_counter():
    uid = _uid()
    with SessionLocal() as db:
        assert usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW).allowed
        assert not usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW).allowed
        october = datetime(2026, 10, 1, 0, 0, 1)
        assert usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=october).allowed
        assert usage.used(db, uid, "research_run", "2026-09") == 1
        assert usage.used(db, uid, "research_run", "2026-10") == 1


def test_distinct_resources_tracks_tickers_not_calls():
    uid = _uid()
    with SessionLocal() as db:
        for t in ("NVDA", "COST", "NVDA"):
            usage.reserve(db, user_id=uid, feature="memo_view", limit=None, idempotency_key=_key(),
                          resource_ref=t, now=NOW)
        assert usage.distinct_resources(db, uid, "memo_view", "2026-09") == {"NVDA", "COST"}


def test_interleaved_reservations_cannot_both_win(monkeypatch):
    """Two sessions race for the last unit. Session B's whole reservation
    runs after A has checked for an existing event and before A touches
    the counter — the check-then-increment gap the conditional UPDATE
    exists to close. Exactly one may succeed and the counter must never
    exceed the limit. (B runs before A's first write, so this is a real
    interleaving on SQLite too, which allows one writer at a time.)"""
    uid = _uid()
    original = usage._insert_counter_if_missing
    state = {"injected": False}

    def interleave(db, user_id, feature, key, now):
        if not state["injected"]:
            state["injected"] = True
            with SessionLocal() as other:
                b = usage.reserve(other, user_id=user_id, feature=feature, limit=1,
                                  idempotency_key=_key(), now=now)
                state["b_allowed"] = b.allowed
        original(db, user_id, feature, key, now)

    monkeypatch.setattr(usage, "_insert_counter_if_missing", interleave)
    with SessionLocal() as db:
        a = usage.reserve(db, user_id=uid, feature="research_run", limit=1, idempotency_key=_key(), now=NOW)
    assert state["injected"]
    assert [a.allowed, state["b_allowed"]].count(True) == 1
    with SessionLocal() as db:
        assert usage.used(db, uid, "research_run", "2026-09") == 1
        assert len(usage.history(db, uid, "2026-09")) == 1


def test_lost_race_on_the_same_idempotency_key_replays_the_winner(monkeypatch):
    """A retried request whose original is still being written must find
    the original, not a second charge."""
    uid = _uid()
    key = _key()
    original = usage._insert_counter_if_missing
    state = {"injected": False}

    def interleave(db, user_id, feature, pk, now):
        if not state["injected"]:
            state["injected"] = True
            with SessionLocal() as other:
                usage.reserve(other, user_id=user_id, feature=feature, limit=5, idempotency_key=key, now=now)
        original(db, user_id, feature, pk, now)

    monkeypatch.setattr(usage, "_insert_counter_if_missing", interleave)
    with SessionLocal() as db:
        r = usage.reserve(db, user_id=uid, feature="pm_chat", limit=5, idempotency_key=key, now=NOW)
        assert r.allowed and r.replayed
        assert usage.used(db, uid, "pm_chat", "2026-09") == 1
        assert len(usage.history(db, uid, "2026-09")) == 1


def test_history_is_newest_first_and_bounded():
    uid = _uid()
    with SessionLocal() as db:
        for i in range(5):
            usage.reserve(db, user_id=uid, feature="pm_chat", limit=None, idempotency_key=_key(),
                          resource_ref=f"T{i}", now=datetime(2026, 9, 1, i))
        rows = usage.history(db, uid, "2026-09", limit=3)
        assert [r.resource_ref for r in rows] == ["T4", "T3", "T2"]
