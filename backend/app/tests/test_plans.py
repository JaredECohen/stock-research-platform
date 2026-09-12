"""`auth/plans.resolve_plan` — every row of the state table (plan §5.2),
including the boundary instants. Pure function, no database."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.auth.plans import PLAN_FREE, PLAN_NONE, PLAN_PRO, resolve_plan

NOW = datetime(2026, 9, 8, 12, 0, 0)
DAY = timedelta(days=1)


def user(**kw):
    base = dict(account_state="active", trial_ends_at=None)
    base.update(kw)
    return SimpleNamespace(**base)


def sub(**kw):
    base = dict(
        stripe_status="active", current_period_end=NOW + 20 * DAY, cancel_at_period_end=False,
        canceled_at=None, ended_at=None, past_due_since=None, grace_until=None, updated_at=NOW,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def override(kind, value=None, *, expires_at=None, starts_at=None, revoked_at=None):
    return SimpleNamespace(kind=kind, value=value, expires_at=expires_at,
                           starts_at=starts_at or NOW - DAY, revoked_at=revoked_at)


@pytest.mark.parametrize("subscription,plan,source,ends_at,warning_fragment", [
    # trialing — EXPECTED after a checkout that carried the local trial end
    (sub(stripe_status="trialing", current_period_end=NOW + 3 * DAY), PLAN_PRO, "subscription", NOW + 3 * DAY, None),
    (sub(stripe_status="trialing", current_period_end=NOW - DAY), PLAN_FREE, "default", None, None),
    # active, rolling
    (sub(stripe_status="active"), PLAN_PRO, "subscription", None, None),
    # active, cancels at period end
    (sub(stripe_status="active", cancel_at_period_end=True), PLAN_PRO, "subscription", NOW + 20 * DAY, "Pro ends on"),
    (sub(stripe_status="active", cancel_at_period_end=True, current_period_end=NOW - DAY), PLAN_FREE, "default", None, None),
    # past_due inside grace
    (sub(stripe_status="past_due", past_due_since=NOW - 2 * DAY, grace_until=NOW + 5 * DAY), PLAN_PRO, "grace", NOW + 5 * DAY, "Payment failed"),
    # past_due, grace computed from past_due_since when grace_until unset (7d default)
    (sub(stripe_status="past_due", past_due_since=NOW - 2 * DAY), PLAN_PRO, "grace", NOW + 5 * DAY, "Payment failed"),
    # past_due after grace
    (sub(stripe_status="past_due", past_due_since=NOW - 9 * DAY, grace_until=NOW - 2 * DAY), PLAN_FREE, "default", None, "Payment failed"),
    # unpaid
    (sub(stripe_status="unpaid", grace_until=NOW + DAY), PLAN_PRO, "grace", NOW + DAY, "unpaid"),
    (sub(stripe_status="unpaid", grace_until=NOW - DAY), PLAN_FREE, "default", None, "unpaid"),
    (sub(stripe_status="unpaid"), PLAN_FREE, "default", None, "unpaid"),
    # canceled
    (sub(stripe_status="canceled", ended_at=NOW - DAY), PLAN_FREE, "default", None, None),
    (sub(stripe_status="canceled", ended_at=NOW + 2 * DAY), PLAN_PRO, "subscription", NOW + 2 * DAY, "Pro ends on"),
    # incomplete / expired / paused
    (sub(stripe_status="incomplete"), PLAN_FREE, "default", None, "Checkout not completed"),
    (sub(stripe_status="incomplete_expired"), PLAN_FREE, "default", None, "Checkout not completed"),
    (sub(stripe_status="paused"), PLAN_FREE, "default", None, "paused"),
    # something Stripe invents next year
    (sub(stripe_status="mystery"), PLAN_FREE, "default", None, None),
])
def test_subscription_rows(subscription, plan, source, ends_at, warning_fragment):
    state = resolve_plan(user(), subscription, [], NOW)
    assert state.plan == plan
    assert state.source == source
    assert state.ends_at == ends_at
    if warning_fragment is None:
        assert state.warning is None
    else:
        assert warning_fragment in (state.warning or "")


def test_no_subscription_no_trial_is_free_default():
    state = resolve_plan(user(), None, [], NOW)
    assert state.plan == PLAN_FREE and state.source == "default" and state.warning is None


def test_trial_ahead_is_pro_with_the_exact_end_date():
    ends = NOW + 4 * DAY
    state = resolve_plan(user(trial_ends_at=ends), None, [], NOW)
    assert state.plan == PLAN_PRO and state.source == "trial"
    assert state.ends_at == ends and state.trial_ends_at == ends
    assert state.warning == f"Trial ends {ends:%Y-%m-%d}"


def test_trial_boundary_is_exclusive():
    """`trial_ends_at == now` means the trial is over."""
    state = resolve_plan(user(trial_ends_at=NOW), None, [], NOW)
    assert state.plan == PLAN_FREE
    assert state.trial_ends_at == NOW, "the date stays visible after expiry"
    just_before = resolve_plan(user(trial_ends_at=NOW), None, [], NOW - timedelta(seconds=1))
    assert just_before.plan == PLAN_PRO


def test_period_end_boundary_is_exclusive():
    s = sub(stripe_status="active", cancel_at_period_end=True, current_period_end=NOW)
    assert resolve_plan(user(), s, [], NOW).plan == PLAN_FREE
    assert resolve_plan(user(), s, [], NOW - timedelta(seconds=1)).plan == PLAN_PRO


def test_grace_uses_the_configured_days(monkeypatch):
    s = sub(stripe_status="past_due", past_due_since=NOW - 2 * DAY)
    assert resolve_plan(user(), s, [], NOW, grace_days=1).plan == PLAN_FREE
    assert resolve_plan(user(), s, [], NOW, grace_days=3).plan == PLAN_PRO


def test_plan_override_grants_pro_until_expiry():
    o = override("plan", "pro", expires_at=NOW + 10 * DAY)
    state = resolve_plan(user(), None, [o], NOW)
    assert state.plan == PLAN_PRO and state.source == "override" and state.ends_at == NOW + 10 * DAY
    assert resolve_plan(user(), None, [override("plan", "pro", expires_at=NOW)], NOW).plan == PLAN_FREE
    assert resolve_plan(user(), None, [override("plan", "pro", revoked_at=NOW)], NOW).plan == PLAN_FREE
    assert resolve_plan(user(), None, [override("plan", "pro", starts_at=NOW + DAY)], NOW).plan == PLAN_FREE


def test_suspension_beats_a_paid_subscription():
    state = resolve_plan(user(), sub(stripe_status="active"), [override("suspend")], NOW)
    assert state.plan == PLAN_NONE and state.suspended and state.source == "suspended"
    state = resolve_plan(user(account_state="suspended"), sub(stripe_status="active"), [], NOW)
    assert state.suspended


def test_max_resolution_prefers_the_longest_lasting_pro():
    trial = user(trial_ends_at=NOW + 3 * DAY)
    # rolling subscription beats a dated trial
    state = resolve_plan(trial, sub(stripe_status="active"), [], NOW)
    assert state.source == "subscription" and state.ends_at is None
    assert state.trial_ends_at == NOW + 3 * DAY
    # a longer override beats a shorter trial
    state = resolve_plan(trial, None, [override("plan", "pro", expires_at=NOW + 30 * DAY)], NOW)
    assert state.source == "override"
    # a longer trial beats a canceling subscription that ends sooner
    s = sub(stripe_status="active", cancel_at_period_end=True, current_period_end=NOW + DAY)
    state = resolve_plan(user(trial_ends_at=NOW + 5 * DAY), s, [], NOW)
    assert state.source == "trial"
    assert state.cancel_at_period_end is True, "billing facts stay visible"


def test_free_after_trial_keeps_subscription_warning():
    s = sub(stripe_status="unpaid")
    state = resolve_plan(user(trial_ends_at=NOW - DAY), s, [], NOW)
    assert state.plan == PLAN_FREE and "unpaid" in state.warning
