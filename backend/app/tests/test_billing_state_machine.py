"""`services/billing_service.apply_subscription_object` — every §5.2 row
and the transitions between them (FEAT-002, S4).

DB-level: a `users` row, a Stripe-shaped object, and `resolve_plan` on
the result. No HTTP, no signatures — `test_webhooks.py` covers those.
Times are pinned to the fixture era (2025-09) and passed as `now`, so
the assertions do not drift with the calendar.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.auth.entitlements import current_subscription, resolve_for_user
from app.auth.plans import resolve_plan
from app.config import settings
from app.database import SessionLocal
from app.models import Subscription, User
from app.services import billing_service as bs
from app.tests.billing_helpers import make_user, subscription_object, uid

NOW = datetime(2025, 9, 10, 12, 0, 0)
PERIOD_END = datetime(2025, 10, 9, 6, 40)      # 1759992000
CREATED = 1757400000                            # the fixtures' `created`
GRACE = timedelta(days=settings.grace_days)


def _apply(user_id: int, obj: dict, *, created: int = CREATED, now: datetime = NOW, source: str = "webhook"):
    with SessionLocal() as db:
        user = db.get(User, user_id)
        row, outcome = bs.apply_subscription_object(db, user, obj, event_created=created, now=now, source=source)
        db.commit()
        db.refresh(row)
        state = resolve_for_user(db, user, now)
        db.expunge_all()
        return row, outcome, state


def _obj(customer: str, sub: str, **over) -> dict:
    return subscription_object(customer=customer, id=sub, **over)


@pytest.fixture()
def user():
    return make_user(customer=f"cus_{uid()}", now=NOW)


# ---------------------------------------------------------------------------
# §5.2 rows
# ---------------------------------------------------------------------------

def test_trialing_is_an_expected_pro_state_until_period_end(user):
    """Checkout during a local trial carries the remaining trial time to
    Stripe, so `trialing` is the normal post-checkout state — Pro until
    the Stripe trial (= the local trial end) runs out."""
    sub = f"sub_{uid()}"
    row, outcome, state = _apply(user.id, _obj(user.stripe_customer_id, sub, status="trialing"))
    assert outcome == "applied" and row.stripe_status == "trialing"
    assert state.plan == "pro" and state.source == "subscription" and state.ends_at == PERIOD_END
    assert state.warning is None
    _, _, later = _apply(user.id, _obj(user.stripe_customer_id, sub, status="trialing"),
                         created=CREATED + 1, now=PERIOD_END + timedelta(seconds=1))
    assert later.plan == "free"


def test_active_rolling(user):
    _, _, state = _apply(user.id, _obj(user.stripe_customer_id, f"sub_{uid()}", status="active"))
    assert state.plan == "pro" and state.ends_at is None and state.warning is None
    assert state.period_end == PERIOD_END and state.cancel_at_period_end is False


def test_active_cancel_at_period_end_keeps_pro_until_the_end_then_free(user):
    sub = f"sub_{uid()}"
    obj = _obj(user.stripe_customer_id, sub, status="active", cancel_at_period_end=True, canceled_at=CREATED + 100)
    row, _, state = _apply(user.id, obj)
    assert row.cancel_at_period_end is True and row.canceled_at is not None
    assert state.plan == "pro" and state.ends_at == PERIOD_END
    assert state.warning and "Pro ends on 2025-10-09" in state.warning
    with SessionLocal() as db:
        after = resolve_plan(db.get(User, user.id), db.get(Subscription, row.id), [], PERIOD_END)
        assert after.plan == "free"


def test_past_due_sets_grace_once_and_active_clears_it(user):
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"))
    row, _, state = _apply(user.id, _obj(user.stripe_customer_id, sub, status="past_due"), created=CREATED + 10)
    assert row.past_due_since == datetime(1970, 1, 1) + timedelta(seconds=CREATED + 10)
    assert row.grace_until == row.past_due_since + GRACE
    assert state.plan == "pro" and state.source == "grace" and state.ends_at == row.grace_until
    assert "update your card" in (state.warning or "")

    # A second past_due delivery does not push the grace window out.
    row2, _, _ = _apply(user.id, _obj(user.stripe_customer_id, sub, status="past_due"), created=CREATED + 20)
    assert row2.past_due_since == row.past_due_since and row2.grace_until == row.grace_until

    # After grace: Free, with the warning that says why.
    with SessionLocal() as db:
        lapsed = resolve_plan(db.get(User, user.id), db.get(Subscription, row.id), [], row.grace_until)
        assert lapsed.plan == "free" and "paused" in (lapsed.warning or "")

    row3, _, state3 = _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"), created=CREATED + 30)
    assert row3.past_due_since is None and row3.grace_until is None
    assert state3.plan == "pro" and state3.source == "subscription" and state3.warning is None


def test_unpaid_is_free_after_grace(user):
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="past_due"))
    row, _, state = _apply(user.id, _obj(user.stripe_customer_id, sub, status="unpaid"), created=CREATED + 5)
    assert state.plan == "pro" and state.source == "grace", "still inside the grace window"
    with SessionLocal() as db:
        after = resolve_plan(db.get(User, user.id), db.get(Subscription, row.id), [], row.grace_until)
        assert after.plan == "free" and after.warning == "Subscription unpaid"


def test_canceled_with_ended_at_is_free_from_ended_at(user):
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"))
    ended = CREATED + 3600
    row, _, state_before = _apply(
        user.id, _obj(user.stripe_customer_id, sub, status="canceled", canceled_at=CREATED + 10, ended_at=ended),
        created=CREATED + 10, now=NOW,
    )
    assert row.ended_at == datetime(1970, 1, 1) + timedelta(seconds=ended)
    # `ended` is an hour after the fixture's `created` (2025-09-09), a
    # day before NOW — so Free now, Pro a minute before it ended.
    assert state_before.plan == "free"
    with SessionLocal() as db:
        early = resolve_plan(db.get(User, user.id), db.get(Subscription, row.id), [], row.ended_at - timedelta(minutes=1))
        assert early.plan == "pro" and early.ends_at == row.ended_at


def test_canceled_without_ended_at_gets_the_event_time(user):
    row, _, state = _apply(user.id, _obj(user.stripe_customer_id, f"sub_{uid()}", status="canceled",
                                         canceled_at=None, ended_at=None))
    assert row.ended_at == datetime(1970, 1, 1) + timedelta(seconds=CREATED)
    assert state.plan == "free"


@pytest.mark.parametrize("status,warning", [
    ("incomplete", "Checkout not completed"),
    ("incomplete_expired", "Checkout not completed"),
    ("paused", "Subscription paused"),
    ("some_future_status", None),
])
def test_non_granting_statuses_are_free(user, status, warning):
    _, _, state = _apply(user.id, _obj(user.stripe_customer_id, f"sub_{uid()}", status=status))
    assert state.plan == "free"
    assert state.warning == warning


def test_no_subscription_trial_ahead_is_pro_then_free():
    u = make_user(trial_ends_in=timedelta(days=3), now=NOW)
    with SessionLocal() as db:
        user = db.get(User, u.id)
        assert resolve_for_user(db, user, NOW).source == "trial"
        assert resolve_for_user(db, user, NOW + timedelta(days=3)).plan == "free"


def test_subscription_beats_trial_when_it_lasts_longer():
    u = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=3), now=NOW)
    _, _, state = _apply(u.id, _obj(u.stripe_customer_id, f"sub_{uid()}", status="active"))
    assert state.source == "subscription" and state.ends_at is None
    assert state.trial_ends_at == u.trial_ends_at, "the trial fact stays visible"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_older_event_is_ignored_and_never_resurrects_access(user):
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="canceled", ended_at=CREATED + 50), created=CREATED + 50)
    row, outcome, state = _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"), created=CREATED + 10)
    assert outcome == "ignored_stale"
    assert row.stripe_status == "canceled" and row.latest_event_created == CREATED + 50
    assert state.plan == "free"


def test_equal_timestamp_is_applied(user):
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"), created=CREATED)
    row, outcome, _ = _apply(user.id, _obj(user.stripe_customer_id, sub, status="past_due"), created=CREATED)
    assert outcome == "applied" and row.stripe_status == "past_due"


def test_reconcile_applies_as_of_now_so_a_late_webhook_cannot_undo_it(user):
    sub = f"sub_{uid()}"
    obj = _obj(user.stripe_customer_id, sub, status="canceled", ended_at=CREATED + 50)
    row, _, _ = _apply(user.id, obj, created=bs.unix(NOW), now=NOW, source="reconcile")
    assert row.latest_event_created == bs.unix(NOW)
    _, outcome, state = _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"), created=CREATED + 100)
    assert outcome == "ignored_stale" and state.plan == "free"


def test_wrong_customer_is_refused(user):
    with SessionLocal() as db:
        with pytest.raises(bs.OwnershipMismatch):
            bs.apply_subscription_object(db, db.get(User, user.id), _obj("cus_somebody_else", f"sub_{uid()}"),
                                         event_created=CREATED, now=NOW)
        db.rollback()
    assert current_subscription_id(user.id) is None


def test_subscription_already_owned_by_another_user_is_refused():
    a = make_user(customer=f"cus_{uid()}", now=NOW)
    b = make_user(customer=f"cus_{uid()}", now=NOW)
    sub = f"sub_{uid()}"
    _apply(a.id, _obj(a.stripe_customer_id, sub))
    with SessionLocal() as db:
        with pytest.raises(bs.OwnershipMismatch):
            bs.apply_subscription_object(db, db.get(User, b.id), _obj(b.stripe_customer_id, sub),
                                         event_created=CREATED + 1, now=NOW)
        db.rollback()


def test_object_without_id_is_refused(user):
    with SessionLocal() as db:
        with pytest.raises(bs.OwnershipMismatch):
            bs.apply_subscription_object(db, db.get(User, user.id), {"status": "active"}, event_created=CREATED)
        db.rollback()


def current_subscription_id(user_id: int):
    with SessionLocal() as db:
        row = current_subscription(db, user_id)
        return row.id if row else None


# ---------------------------------------------------------------------------
# What is (not) written
# ---------------------------------------------------------------------------

def test_never_writes_trial_columns():
    u = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=2), now=NOW)
    before = (u.trial_started_at, u.trial_ends_at, u.trial_source)
    obj = _obj(u.stripe_customer_id, f"sub_{uid()}", status="trialing", trial_start=CREATED, trial_end=CREATED + 10 * 86400)
    _apply(u.id, obj)
    with SessionLocal() as db:
        after = db.get(User, u.id)
        assert (after.trial_started_at, after.trial_ends_at, after.trial_source) == before


def test_snapshot_is_whitelisted_and_price_interval_recorded(user):
    row, _, _ = _apply(user.id, _obj(user.stripe_customer_id, f"sub_{uid()}"))
    assert row.stripe_price_id == "price_TEST_monthly" and row.billing_interval == "month"
    assert row.current_period_end == PERIOD_END and row.plan == "pro"
    assert set(row.snapshot) == {
        "id", "status", "customer", "price_id", "interval", "current_period_start", "current_period_end",
        "cancel_at_period_end", "canceled_at", "ended_at", "trial_end", "livemode", "created",
    }
    assert "default_payment_method" not in str(row.snapshot)


def test_period_is_read_from_items_on_the_newer_api_shape(user):
    obj = _obj(user.stripe_customer_id, f"sub_{uid()}")
    obj.pop("current_period_start")
    obj.pop("current_period_end")
    obj["items"]["data"][0]["current_period_start"] = CREATED
    obj["items"]["data"][0]["current_period_end"] = CREATED + 30 * 86400
    row, _, _ = _apply(user.id, obj)
    assert row.current_period_end == datetime(1970, 1, 1) + timedelta(seconds=CREATED + 30 * 86400)


def test_fills_an_empty_customer_id_but_never_overwrites_one():
    u = make_user(customer=None, now=NOW)
    cust = f"cus_{uid()}"
    _apply(u.id, _obj(cust, f"sub_{uid()}"))
    with SessionLocal() as db:
        assert db.get(User, u.id).stripe_customer_id == cust


def test_transition_analytics(user):
    from app.tests.billing_helpers import events_for
    sub = f"sub_{uid()}"
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active"))
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active", cancel_at_period_end=True), created=CREATED + 1)
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="active", cancel_at_period_end=True), created=CREATED + 2)
    _apply(user.id, _obj(user.stripe_customer_id, sub, status="canceled", ended_at=CREATED + 3), created=CREATED + 3)
    reasons = [e.props.get("reason") for e in events_for(user.id, "subscription_canceled")]
    assert reasons == ["cancel_at_period_end", "canceled"], "once per transition, not per delivery"


def test_trial_converted_is_emitted_only_while_the_trial_is_ahead():
    from app.tests.billing_helpers import events_for
    ahead = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=4), now=NOW)
    _apply(ahead.id, _obj(ahead.stripe_customer_id, f"sub_{uid()}", status="trialing"))
    assert len(events_for(ahead.id, "trial_converted")) == 1
    over = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=-1), now=NOW)
    _apply(over.id, _obj(over.stripe_customer_id, f"sub_{uid()}", status="active"))
    assert events_for(over.id, "trial_converted") == []
