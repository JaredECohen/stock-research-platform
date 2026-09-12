"""`monitoring/billing_loop` (FEAT-002, S4).

The loop changes no plan — `resolve_plan` is read-time — so the tests
are about what it emits once, what it deletes, and what it settles.
Rows are created with unique ids and unmistakable markers so the shared
test database never couples two tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.auth import analytics, usage
from app.config import settings
from app.database import SessionLocal
from app.models import (
    ActiveAction,
    AnalyticsEvent,
    CronLoopRun,
    RateLimitWindow,
    RegenJob,
    Subscription,
    UsageEvent,
    User,
)
from app.monitoring import KNOWN_LOOPS, billing_loop, register_all
from app.services import billing_service as bs
from app.tests.billing_helpers import FakeStripe, enable_billing, events_for, load_event, make_user, uid

NOW = datetime.utcnow().replace(microsecond=0)


@pytest.fixture(autouse=True)
def _quiet_stripe(monkeypatch):
    """No key → the reconcile step is a no-op; tests that want it set one."""
    monkeypatch.setattr(settings, "stripe_secret_key", "")


def _subscribe(user: User, *, status: str = "active", **over) -> Subscription:
    obj = load_event("customer_subscription_created", customer=user.stripe_customer_id, sub=f"sub_{uid()}",
                     user_id=user.id)["data"]["object"]
    obj["status"] = status
    obj.update(over)
    with SessionLocal() as db:
        row, _ = bs.apply_subscription_object(db, db.get(User, user.id), obj, event_created=1, now=NOW)
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_billing_loop_is_registered_and_known():
    class FakeScheduler:
        def __init__(self):
            self.jobs = {}

        def add_job(self, fn, trigger, **kw):
            self.jobs[kw["id"]] = (fn, trigger, kw)

    sched = FakeScheduler()
    register_all(sched)
    assert "billing_loop" in KNOWN_LOOPS and "billing_loop" in sched.jobs
    fn, trigger, kw = sched.jobs["billing_loop"]
    assert trigger == "interval" and kw.get("hours") == 1 and fn is billing_loop.tick


# ---------------------------------------------------------------------------
# trial_expired
# ---------------------------------------------------------------------------

def test_trial_expired_is_emitted_once_per_user():
    expired = make_user(trial_ends_in=timedelta(days=-1), now=NOW)
    ahead = make_user(trial_ends_in=timedelta(days=2), now=NOW)
    converted = make_user(customer=f"cus_{uid()}", trial_ends_in=timedelta(days=-1), now=NOW)
    _subscribe(converted)
    long_ago = make_user(trial_ends_in=-timedelta(days=billing_loop.LOOKBACK_DAYS + 1), now=NOW)

    first = billing_loop.run_once(now=NOW)
    second = billing_loop.run_once(now=NOW)
    assert first["expired"] >= 1 and second["expired"] == 0, "idempotent across runs"
    assert len(events_for(expired.id, "trial_expired")) == 1
    assert events_for(expired.id, "trial_expired")[0].plan == "free"
    assert events_for(ahead.id, "trial_expired") == []
    assert events_for(converted.id, "trial_expired") == [], "a converted user did not expire"
    assert events_for(long_ago.id, "trial_expired") == [], "outside the candidate window"
    # A run after the lookback window does not re-emit for this user
    # either (other tests' users may expire by then — not asserted on).
    billing_loop.run_once(now=NOW + timedelta(days=billing_loop.LOOKBACK_DAYS + 1))
    assert len(events_for(expired.id, "trial_expired")) == 1


def test_lookback_is_shorter_than_analytics_retention():
    assert billing_loop.LOOKBACK_DAYS < analytics.RETENTION_DAYS


# ---------------------------------------------------------------------------
# downgraded
# ---------------------------------------------------------------------------

def test_downgraded_is_emitted_once_for_each_lapse_kind():
    ended = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(ended, status="canceled", ended_at=bs.unix(NOW - timedelta(hours=2)))
    period_end = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(period_end, status="active", cancel_at_period_end=True,
               current_period_end=bs.unix(NOW - timedelta(hours=1)))
    grace = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(grace, status="past_due")
    with SessionLocal() as db:
        row = db.query(Subscription).filter(Subscription.user_id == grace.id).one()
        row.grace_until = NOW - timedelta(minutes=5)
        db.commit()
    still_pro = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(still_pro, status="active")
    re_upgraded = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(re_upgraded, status="canceled", ended_at=bs.unix(NOW - timedelta(hours=3)))
    _subscribe(re_upgraded, status="active")

    billing_loop.run_once(now=NOW)
    billing_loop.run_once(now=NOW)
    for u in (ended, period_end, grace):
        assert len(events_for(u.id, "downgraded")) == 1, u.id
        assert events_for(u.id, "downgraded")[0].plan == "free"
    assert events_for(still_pro.id, "downgraded") == []
    assert events_for(re_upgraded.id, "downgraded") == [], "a newer subscription keeps them Pro"


# ---------------------------------------------------------------------------
# GC
# ---------------------------------------------------------------------------

def test_gc_deletes_expired_windows_leases_and_old_analytics():
    marker = f"zz_gc_{uid()}"
    with SessionLocal() as db:
        db.add(RateLimitWindow(key=f"{marker}:old", count=3, expires_at=NOW - timedelta(minutes=1)))
        db.add(RateLimitWindow(key=f"{marker}:live", count=3, expires_at=NOW + timedelta(minutes=1)))
        db.add(ActiveAction(user_id=1, feature="pm_chat", lease_token=f"{marker}-old", started_at=NOW,
                            expires_at=NOW - timedelta(seconds=1)))
        db.add(ActiveAction(user_id=1, feature="pm_chat", lease_token=f"{marker}-live", started_at=NOW,
                            expires_at=NOW + timedelta(seconds=60)))
        db.add(AnalyticsEvent(ts=NOW - timedelta(days=91), event_name="landing_view", anon_id=marker))
        db.add(AnalyticsEvent(ts=NOW - timedelta(days=89), event_name="landing_view", anon_id=marker))
        db.commit()
    result = billing_loop.run_once(now=NOW)
    assert result["gc_limits"] >= 2 and result["gc_analytics"] >= 1
    with SessionLocal() as db:
        assert {r.key for r in db.query(RateLimitWindow).filter(RateLimitWindow.key.like(f"{marker}%"))} == {f"{marker}:live"}
        assert {r.lease_token for r in db.query(ActiveAction).filter(ActiveAction.lease_token.like(f"{marker}%"))} == {f"{marker}-live"}
        kept = db.query(AnalyticsEvent).filter(AnalyticsEvent.anon_id == marker).all()
        assert len(kept) == 1 and kept[0].ts == NOW - timedelta(days=89)
        db.query(AnalyticsEvent).filter(AnalyticsEvent.anon_id == marker).delete()
        db.query(RateLimitWindow).filter(RateLimitWindow.key.like(f"{marker}%")).delete(synchronize_session=False)
        db.query(ActiveAction).filter(ActiveAction.lease_token.like(f"{marker}%")).delete(synchronize_session=False)
        db.commit()


def test_usage_and_webhook_rows_are_never_gcd():
    """Audit tables: the loop has no code path that deletes them."""
    import inspect
    src = inspect.getsource(billing_loop)
    assert "UsageEvent" not in src and "BillingWebhookEvent" not in src


# ---------------------------------------------------------------------------
# Stale reservations
# ---------------------------------------------------------------------------

def _reserve(user_id: int, *, age: timedelta, feature: str = "research_run") -> int:
    with SessionLocal() as db:
        r = usage.reserve(db, user_id=user_id, feature=feature, limit=None, idempotency_key=f"k-{uid()}",
                          resource_ref="NVDA", now=NOW - age)
        return r.event.id


def _job(event_id: int, status: str) -> None:
    with SessionLocal() as db:
        db.add(RegenJob(ticker="NVDA", run_id=f"run-{uid()}", status=status, usage_event_id=event_id,
                        enqueued_at=NOW - timedelta(hours=8), finished_at=NOW - timedelta(hours=7)))
        db.commit()


def test_stale_reservations_are_settled_against_their_jobs():
    u = make_user(now=NOW)
    ok = _reserve(u.id, age=timedelta(hours=7))
    _job(ok, "succeeded")
    bad = _reserve(u.id, age=timedelta(hours=7))
    _job(bad, "failed")
    orphan = _reserve(u.id, age=timedelta(hours=7))
    running = _reserve(u.id, age=timedelta(hours=7))
    _job(running, "running")
    fresh = _reserve(u.id, age=timedelta(hours=1))
    in_request = _reserve(u.id, age=timedelta(hours=7), feature="memo_view")

    result = billing_loop.run_once(now=NOW)
    res = result["reservations"]
    assert res["committed"] >= 1 and res["released"] >= 3 and res["left"] >= 1
    with SessionLocal() as db:
        status = {eid: db.get(UsageEvent, eid).status for eid in (ok, bad, orphan, running, fresh, in_request)}
    assert status == {ok: "committed", bad: "released", orphan: "released", running: "reserved",
                      fresh: "reserved", in_request: "released"}
    with SessionLocal() as db:
        db.query(RegenJob).filter(RegenJob.usage_event_id.in_((ok, bad, running))).delete(synchronize_session=False)
        db.commit()


# ---------------------------------------------------------------------------
# Stale subscriptions (only with a key)
# ---------------------------------------------------------------------------

def test_reconcile_step_is_a_noop_without_a_key():
    u = make_user(customer=f"cus_{uid()}", now=NOW)
    _subscribe(u)
    with SessionLocal() as db:
        db.query(Subscription).filter(Subscription.user_id == u.id).update({"updated_at": NOW - timedelta(days=2)})
        db.commit()
    result = billing_loop.run_once(now=NOW)
    assert result["reconciled"] == {"checked": 0, "applied": 0, "errors": 0}


def test_reconcile_step_refetches_stale_rows_and_never_downgrades_on_error(monkeypatch):
    enable_billing(monkeypatch)
    fake = FakeStripe().install(monkeypatch)
    u = make_user(customer=f"cus_{uid()}", now=NOW)
    row = _subscribe(u)
    with SessionLocal() as db:
        db.query(Subscription).filter(Subscription.id == row.id).update({"updated_at": NOW - timedelta(days=2)})
        db.commit()

    fake.fail = bs.StripeError(500, code="api_error")
    result = billing_loop.run_once(now=NOW)
    assert result["reconciled"]["errors"] >= 1
    with SessionLocal() as db:
        after = db.get(Subscription, row.id)
        assert after.stripe_status == "active" and after.updated_at == NOW - timedelta(days=2)

    fake.fail = None
    fake.subscription = load_event("customer_subscription_deleted", customer=u.stripe_customer_id,
                                   sub=row.stripe_subscription_id, user_id=u.id)["data"]["object"]
    result = billing_loop.run_once(now=NOW)
    assert result["reconciled"]["applied"] >= 1
    with SessionLocal() as db:
        after = db.get(Subscription, row.id)
        assert after.stripe_status == "canceled" and after.ended_at is not None
        assert after.latest_event_created == bs.unix(NOW)
    assert fake.calls_to("GET", f"/subscriptions/{row.stripe_subscription_id}")


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

def test_run_once_records_the_run_with_counts_in_the_note():
    billing_loop.run_once(now=NOW)
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "billing_loop").one()
    assert row.success is True
    for key in ("expired=", "downgraded=", "gc=", "reservations=", "reconciled="):
        assert key in row.note


def test_a_failing_step_is_recorded_and_does_not_stop_the_others(monkeypatch):
    def boom(db, **kw):
        raise RuntimeError("analytics table missing")
    monkeypatch.setattr(billing_loop, "emit_trial_expired", boom)
    result = billing_loop.run_once(now=NOW)
    assert result["step_errors"] and "expired: RuntimeError" in result["step_errors"][0]
    assert isinstance(result["gc_limits"], int), "later steps still ran"
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "billing_loop").one()
    assert row.success is False and "failed=expired" in row.note


def test_tick_records_a_crash_before_reraising(monkeypatch):
    calls = []

    def crash(**kw):
        raise RuntimeError("db gone")

    monkeypatch.setattr(billing_loop, "run_once", crash)
    monkeypatch.setattr(billing_loop, "record_run", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(RuntimeError):
        billing_loop.tick()
    assert calls[0][0] == ("billing_loop",) and calls[0][1]["success"] is False
    assert "RuntimeError" in calls[0][1]["note"]
