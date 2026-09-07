"""provider_cache hardening — stale-fallback caps, ledger rows, put() races.

The clock is `provider_cache._now`, monkeypatched to a controllable
fake so row ages are exact; the DB is whatever isolated sqlite the run
was given. Keys are unique per test so leftover rows from an earlier
run can't leak into assertions.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.cache import CacheCostLog
from app.database import SessionLocal
from app.models import ProviderCache
from app.services import provider_cache as pc


class FakeClock:
    def __init__(self) -> None:
        self.base = datetime.utcnow().replace(microsecond=0)
        self.current = self.base

    def now(self) -> datetime:
        return self.current

    def rewind(self, seconds: int) -> None:
        self.current = self.base - timedelta(seconds=seconds)

    def reset(self) -> None:
        self.current = self.base


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(pc, "_now", fake.now)
    return fake


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Retries must not cost wall-clock time in the suite; record the
    calls so the backoff test can assert it happened."""
    calls: list = []
    monkeypatch.setattr(pc, "_sleep", lambda s: calls.append(s))
    return calls


def _key(label: str) -> str:
    return f"{label}-{time.perf_counter_ns()}"


def _seed(clock: FakeClock, capability: str, key: str, payload, age_seconds: int) -> None:
    """Write a row whose fetched_at is `age_seconds` before the clock's base."""
    clock.rewind(age_seconds)
    pc.put(capability, key, payload)
    clock.reset()


def _ledger_rows(key: str) -> list:
    with SessionLocal() as db:
        rows = db.execute(
            select(CacheCostLog).where(CacheCostLog.subject == pc.STALE_LOG_SUBJECT)
        ).scalars().all()
        return [
            (r.kind, json.loads(r.note)) for r in rows
            if json.loads(r.note).get("key") == key
        ]


def _row_count(capability: str, key: str) -> int:
    with SessionLocal() as db:
        return len(db.execute(
            select(ProviderCache).where(
                ProviderCache.capability == capability, ProviderCache.key == key,
            )
        ).scalars().all())


# ---------------------------------------------------------------------------
# cached_call
# ---------------------------------------------------------------------------

def test_fresh_hit_skips_fetcher(clock):
    key = _key("fresh")
    _seed(clock, "profile", key, {"name": "cached"}, age_seconds=60)

    def fetcher():
        raise AssertionError("fetcher must not run on a fresh hit")

    assert pc.cached_call("profile", key, fetcher) == {"name": "cached"}
    assert _ledger_rows(key) == []


def test_expired_row_is_refetched_and_rewritten(clock):
    key = _key("expired")
    _seed(clock, "quote", key, {"price": 1.0}, age_seconds=600)  # quote TTL is 60s
    calls = []

    def fetcher():
        calls.append(1)
        return {"price": 2.0}

    assert pc.cached_call("quote", key, fetcher) == {"price": 2.0}
    assert calls == [1]
    with SessionLocal() as db:
        row = db.execute(select(ProviderCache).where(ProviderCache.key == key)).scalar_one()
        assert row.payload_json == {"price": 2.0}
        assert row.fetched_at == clock.base
    assert _row_count("quote", key) == 1


def test_provider_miss_serves_stale_within_cap_and_logs_age(clock, caplog):
    key = _key("stale-ok")
    age = 2 * 86400  # prices cap is 7d
    _seed(clock, "prices", key, [{"date": "2026-01-02", "close": 1}], age_seconds=age)

    with caplog.at_level(logging.WARNING, logger=pc.__name__):
        got = pc.cached_call("prices", key, lambda: None)

    assert got == [{"date": "2026-01-02", "close": 1}]
    assert any(
        "serving stale" in r.getMessage() and f"age_seconds={age}" in r.getMessage()
        for r in caplog.records
    )
    assert _ledger_rows(key) == [
        ("stale_served", {"capability": "prices", "key": key, "age_seconds": age}),
    ]


def test_provider_miss_refuses_row_beyond_cap(clock, caplog):
    key = _key("stale-refused")
    age = 8 * 86400  # past the 7d prices cap
    _seed(clock, "prices", key, [{"date": "2026-01-02", "close": 1}], age_seconds=age)

    with caplog.at_level(logging.WARNING, logger=pc.__name__):
        got = pc.cached_call("prices", key, lambda: None)

    assert got is None
    assert any(
        "provider miss and cached row too stale" in r.getMessage()
        and f"capability=prices key={key} age_seconds={age}" in r.getMessage()
        for r in caplog.records
    )
    assert _ledger_rows(key) == [
        ("stale_refused", {"capability": "prices", "key": key, "age_seconds": age}),
    ]
    # The row itself is kept — only the fallback is refused.
    assert _row_count("prices", key) == 1


def test_provider_miss_with_no_row_returns_none_without_ledger(clock):
    key = _key("absent")
    assert pc.cached_call("profile", key, lambda: {}) is None
    assert _ledger_rows(key) == []


def test_unknown_capability_uses_seven_day_default(clock):
    assert pc.max_stale_seconds("factor_returns") == 7 * 86400
    key = _key("unknown-cap")
    _seed(clock, "factor_returns", key, {"v": 1}, age_seconds=6 * 86400)
    assert pc.cached_call("factor_returns", key, lambda: None, ttl_seconds=1) == {"v": 1}
    _seed(clock, "factor_returns", key, {"v": 1}, age_seconds=8 * 86400)
    assert pc.cached_call("factor_returns", key, lambda: None, ttl_seconds=1) is None


def test_default_caps_match_spec():
    assert pc.MAX_STALE_BY_CAPABILITY == {
        "profile": 30 * 86400, "prices": 7 * 86400, "quote": 3600,
        "ratios": 7 * 86400, "estimates": 14 * 86400, "earnings": 14 * 86400,
        "news": 86400, "macro": 14 * 86400,
    }


# ---------------------------------------------------------------------------
# settings override
# ---------------------------------------------------------------------------

def test_env_override_changes_cap(clock, monkeypatch):
    monkeypatch.setattr(pc.settings, "provider_cache_max_stale", "quote=600, news=12h;macro:2d")
    assert pc.max_stale_seconds("quote") == 600
    assert pc.max_stale_seconds("news") == 12 * 3600
    assert pc.max_stale_seconds("macro") == 2 * 86400
    assert pc.max_stale_seconds("profile") == 30 * 86400  # untouched default

    key = _key("override")
    _seed(clock, "quote", key, {"price": 9.0}, age_seconds=900)  # default cap 3600 would serve
    assert pc.cached_call("quote", key, lambda: None) is None
    assert _ledger_rows(key)[0][0] == "stale_refused"


def test_invalid_override_entry_is_ignored_with_warning(monkeypatch, caplog):
    raw = f"quote=abc,=5,ratios=-1,profile=120,garbage,{_key('unique')}"
    monkeypatch.setattr(pc.settings, "provider_cache_max_stale", raw)
    with caplog.at_level(logging.WARNING, logger=pc.__name__):
        assert pc.max_stale_seconds("profile") == 120
        assert pc.max_stale_seconds("quote") == 3600      # bad entry → default
        assert pc.max_stale_seconds("ratios") == 7 * 86400
    bad = [r.getMessage() for r in caplog.records if "invalid provider_cache_max_stale" in r.getMessage()]
    assert len(bad) == 5
    assert any("'quote=abc'" in m for m in bad)
    assert any("'ratios=-1'" in m for m in bad)


def test_blank_override_keeps_defaults(monkeypatch):
    monkeypatch.setattr(pc.settings, "provider_cache_max_stale", "")
    assert pc.max_stale_seconds("earnings") == 14 * 86400


# ---------------------------------------------------------------------------
# get(serve_stale=True, max_age_seconds=...)
# ---------------------------------------------------------------------------

def test_get_serve_stale_honours_max_age(clock):
    key = _key("get-max-age")
    _seed(clock, "profile", key, {"n": 1}, age_seconds=3600)
    assert pc.get("profile", key, ttl_seconds=60) is None
    assert pc.get("profile", key, ttl_seconds=60, serve_stale=True) == {"n": 1}
    assert pc.get("profile", key, ttl_seconds=60, serve_stale=True, max_age_seconds=7200) == {"n": 1}
    assert pc.get("profile", key, ttl_seconds=60, serve_stale=True, max_age_seconds=1800) is None


def test_cached_call_fallback_uses_get_stale_path(clock, monkeypatch):
    """Regression: the cap check used to be an inline copy in
    cached_call rather than the `get(serve_stale=True, max_age_seconds)`
    path, so the two could drift. The fallback must go through the same
    lookup with the capability's cap."""
    key = _key("wiring")
    _seed(clock, "news", key, {"h": 1}, age_seconds=7200)  # past 1h TTL, inside 24h cap
    seen: list = []
    real_lookup = pc._lookup

    def recording_lookup(*args, **kwargs):
        seen.append(kwargs)
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(pc, "_lookup", recording_lookup)
    assert pc.cached_call("news", key, lambda: None) == {"h": 1}
    stale_calls = [k for k in seen if k.get("serve_stale")]
    assert len(stale_calls) == 1
    assert stale_calls[0]["max_age_seconds"] == pc.max_stale_seconds("news")


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_get_and_cached_call_agree_at_the_cap_boundary(clock, offset):
    """`get(serve_stale=True, max_age_seconds=cap)` and the provider-miss
    fallback are the same decision; they must flip at the same second."""
    cap = pc.max_stale_seconds("quote")
    key = _key(f"boundary{offset}")
    _seed(clock, "quote", key, {"p": 1}, age_seconds=cap + offset)
    via_get = pc.get("quote", key, ttl_seconds=0, serve_stale=True, max_age_seconds=cap)
    via_call = pc.cached_call("quote", key, lambda: None)
    assert via_get == via_call
    assert (via_call is None) == (offset >= 0)
    kinds = [kind for kind, _ in _ledger_rows(key)]
    assert kinds == (["stale_refused"] if offset >= 0 else ["stale_served"])


# ---------------------------------------------------------------------------
# put() race handling
# ---------------------------------------------------------------------------

class _FailingCommitSession:
    """Real session whose first `fail_times` commits raise IntegrityError."""

    def __init__(self, real, budget: dict):
        self._real = real
        self._budget = budget

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def commit(self):
        if self._budget["remaining"] > 0:
            self._budget["remaining"] -= 1
            self._real.rollback()
            raise IntegrityError("INSERT INTO provider_cache", {}, Exception("UNIQUE constraint failed"))
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_failing_commits(monkeypatch, times: int) -> dict:
    budget = {"remaining": times}
    real_factory = pc.SessionLocal
    monkeypatch.setattr(pc, "SessionLocal", lambda: _FailingCommitSession(real_factory(), budget))
    return budget


def test_put_survives_two_integrity_errors(clock, monkeypatch, _no_backoff):
    key = _key("race-2")
    budget = _install_failing_commits(monkeypatch, times=2)
    pc.put("profile", key, {"ok": True})
    assert budget["remaining"] == 0
    assert _row_count("profile", key) == 1
    assert len(_no_backoff) == 2  # backed off between attempts 1→2 and 2→3


def test_put_raises_after_three_integrity_errors(clock, monkeypatch, caplog):
    key = _key("race-3")
    _install_failing_commits(monkeypatch, times=3)
    with caplog.at_level(logging.WARNING, logger=pc.__name__):
        with pytest.raises(IntegrityError):
            pc.put("profile", key, {"secret": "payload"})
    msgs = [r.getMessage() for r in caplog.records if "gave up after 3 IntegrityErrors" in r.getMessage()]
    assert len(msgs) == 1
    assert f"capability=profile key={key}" in msgs[0]
    assert "secret" not in msgs[0] and "payload" not in msgs[0]


def test_put_real_double_insert_converges_on_one_row(clock, monkeypatch):
    """Genuine race: a competitor inserts the same (capability, key)
    between our SELECT and our INSERT, so the unique index fires for
    real; the retry must find the winner and update it in place."""
    key = _key("race-real")
    real_factory = pc.SessionLocal
    state = {"raced": False}

    class RacingSession(_FailingCommitSession):
        def commit(self):
            if not state["raced"]:
                state["raced"] = True
                with real_factory() as other:
                    other.add(ProviderCache(
                        capability="profile", key=key,
                        payload_json={"who": "competitor"}, fetched_at=pc._now(),
                    ))
                    other.commit()
            return self._real.commit()

    monkeypatch.setattr(pc, "SessionLocal", lambda: RacingSession(real_factory(), {"remaining": 0}))
    pc.put("profile", key, {"who": "us"})
    assert state["raced"] is True
    assert _row_count("profile", key) == 1
    assert pc.get("profile", key) == {"who": "us"}


# ---------------------------------------------------------------------------
# stale_stats
# ---------------------------------------------------------------------------

def test_stale_stats_aggregates_ledger_rows(clock):
    cap = f"cap{time.perf_counter_ns()}"  # private bucket, immune to other tests' rows
    k1, k2, k3 = _key("s1"), _key("s2"), _key("s3")
    _seed(clock, cap, k1, {"v": 1}, age_seconds=100)
    _seed(clock, cap, k2, {"v": 2}, age_seconds=3000)
    _seed(clock, cap, k3, {"v": 3}, age_seconds=10 * 86400)  # beyond 7d default cap
    assert pc.cached_call(cap, k1, lambda: None, ttl_seconds=1) == {"v": 1}
    assert pc.cached_call(cap, k2, lambda: None, ttl_seconds=1) == {"v": 2}
    assert pc.cached_call(cap, k3, lambda: None, ttl_seconds=1) is None

    stats = pc.stale_stats(window_hours=24)
    assert "error" not in stats
    assert stats["window_hours"] == 24
    assert stats["truncated"] is False
    assert stats["stale_served"] >= 2 and stats["stale_refused"] >= 1
    assert stats["by_capability"][cap] == {
        "served": 2, "refused": 1, "max_age_seconds": 10 * 86400,
    }
    assert stats["oldest_served_age_seconds"] >= 3000


def test_stale_stats_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(pc, "SessionLocal", boom)
    stats = pc.stale_stats(window_hours=6)
    assert stats["stale_served"] == 0 and stats["stale_refused"] == 0
    assert stats["by_capability"] == {}
    assert stats["window_hours"] == 6
    assert "RuntimeError: db down" in stats["error"]


def test_ledger_write_failure_does_not_break_data_path(clock, monkeypatch, caplog):
    key = _key("ledger-down")
    _seed(clock, "profile", key, {"n": 1}, age_seconds=3600)

    def broken(*a, **k):
        raise RuntimeError("ledger unavailable")
    monkeypatch.setattr(pc, "log_cost", broken)
    with caplog.at_level(logging.WARNING, logger=pc.__name__):
        assert pc.cached_call("profile", key, lambda: None, ttl_seconds=1) == {"n": 1}
    assert any("could not record stale_served" in r.getMessage() for r in caplog.records)
