"""A capability missing from `TTL_BY_CAPABILITY` must expire, not freeze.

This is the guard for the defect that made the filings pipeline dead on
arrival, and the shape of it is worth stating because the whole failure lived
in one `dict.get` with no default.

`cached_call` did `ttl_seconds = TTL_BY_CAPABILITY.get(capability)`, and
`_is_fresh(fetched_at, None)` reads None as "never expires". So a capability
nobody remembered to put in the table was not cached badly — it was cached
*permanently*. `filings`, `transcripts` and `key_metrics` were all in that
state. The consequence was not a stale number on a page: SEC EDGAR was read
exactly once per ticker for the life of the deployment, `edgar_poller` diffed
every pass against that first frozen accession list, and `on_filing_event`
never fired once. NVDA filed four times — a 10-Q and three 8-Ks — and the
platform did not notice any of them.

Nothing in the suite caught it, and a configuration-level test would not have.
`assert TTL_BY_CAPABILITY["filings"] == 3600` passes against a table and says
nothing about whether a row that is past its TTL is actually refetched, which
is the only property anyone cares about. So the tests here age real rows and
assert on whether the fetcher runs.

Two halves:

- Behavioural: an unlisted capability expires at `DEFAULT_TTL_SECONDS`; an
  explicit `NEVER_EXPIRES` still never expires; a `filings_index` row goes
  stale on the poller's timescale and the provider is re-consulted, through
  `DataService` rather than the cache module alone.
- Structural: every capability `DataService` actually passes to `_cached` has
  a TTL stated somewhere a reader can see it. The list is read out of the
  source with `ast`, not typed out here, so a capability added tomorrow fails
  this test instead of quietly inheriting a default.
"""
from __future__ import annotations

import ast
import inspect
import time
from datetime import datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import Company
from app.services import data_service as ds_mod
from app.services import provider_cache as pc
from app.services.data_service import get_data_service


class FakeClock:
    """Controllable `provider_cache._now`, so row ages are exact."""

    def __init__(self) -> None:
        self.current = datetime.utcnow().replace(microsecond=0)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current = self.current + timedelta(seconds=seconds)


@pytest.fixture()
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(pc, "_now", fake.now)
    return fake


def _key(label: str) -> str:
    """Unique per call, so a leftover row can never answer for a live one."""
    return f"{label}-{time.perf_counter_ns()}"


class CountingFetcher:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.payload


# ---------------------------------------------------------------------------
# Behaviour: expiry
# ---------------------------------------------------------------------------

def test_an_unlisted_capability_expires_instead_of_caching_forever(clock):
    """The class fix. A capability with no table entry is not immortal."""
    capability = "zz_capability_nobody_listed"
    assert capability not in pc.TTL_BY_CAPABILITY
    key = _key("unlisted")
    fetcher = CountingFetcher([{"v": 1}])

    assert pc.cached_call(capability, key, fetcher) == [{"v": 1}]
    assert fetcher.calls == 1

    # Inside the default TTL: served from the row, provider untouched.
    clock.advance(pc.DEFAULT_TTL_SECONDS - 60)
    assert pc.cached_call(capability, key, fetcher) == [{"v": 1}]
    assert fetcher.calls == 1, "a fresh row must not re-consult the provider"

    # Past it: refetched. Under the old code this row was still "fresh" a
    # year later, which is exactly how EDGAR came to be read once per ticker
    # and never again.
    clock.advance(120)
    fetcher.payload = [{"v": 2}]
    assert pc.cached_call(capability, key, fetcher) == [{"v": 2}]
    assert fetcher.calls == 2


def test_a_year_old_unlisted_row_is_not_served_as_fresh(clock):
    """The reproduction, kept as a test: 365 days old, still cached."""
    key = _key("ancient")
    fetcher = CountingFetcher({"filed": "once"})
    pc.cached_call("zz_capability_nobody_listed", key, fetcher)

    clock.advance(365 * 86400)
    fetcher.payload = {"filed": "again"}
    assert pc.cached_call("zz_capability_nobody_listed", key, fetcher) == {"filed": "again"}
    assert fetcher.calls == 2


def test_an_explicit_never_expires_still_never_expires(clock):
    """Never-expire is still available — it just has to be asked for.

    The point of the fix is not to remove the capability, it is to make it a
    thing a call site says out loud rather than something a missing table row
    does silently.
    """
    key = _key("sentinel")
    fetcher = CountingFetcher({"immutable": True})

    pc.cached_call("zz_capability_nobody_listed", key, fetcher, ttl_seconds=pc.NEVER_EXPIRES)
    clock.advance(365 * 86400)
    assert pc.cached_call(
        "zz_capability_nobody_listed", key, fetcher, ttl_seconds=pc.NEVER_EXPIRES,
    ) == {"immutable": True}
    assert fetcher.calls == 1, "NEVER_EXPIRES must still mean never"


def test_ttl_seconds_for_falls_back_without_returning_none():
    """`None` is the value that used to leak through as never-expire."""
    assert pc.ttl_seconds_for("zz_capability_nobody_listed") == pc.DEFAULT_TTL_SECONDS
    assert pc.ttl_seconds_for("news") == pc.TTL_BY_CAPABILITY["news"]


# ---------------------------------------------------------------------------
# Behaviour: the capabilities that were immortal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("capability", ["filings", "filings_index", "transcripts", "key_metrics"])
def test_the_formerly_immortal_capabilities_expire(clock, capability):
    key = _key(capability)
    fetcher = CountingFetcher([{"n": 1}])
    pc.cached_call(capability, key, fetcher)

    clock.advance(pc.TTL_BY_CAPABILITY[capability] + 1)
    fetcher.payload = [{"n": 2}]
    assert pc.cached_call(capability, key, fetcher) == [{"n": 2}]
    assert fetcher.calls == 2


@pytest.fixture()
def cache_backed_data_service(monkeypatch):
    """A `DataService` that actually reads through `provider_cache`.

    `_cached` short-circuits to the fetcher whenever a test provider is
    registered — conftest registers `DemoProvider` for the whole session — so
    a test that wants to exercise the cache has to take that off first. The
    provider chain is replaced with a counter at `_try_chain`, which is the
    seam below the cache and above the network.
    """
    ds = get_data_service()
    previous = ds._test_provider
    ds.register_test_provider(None)
    try:
        yield ds
    finally:
        ds.register_test_provider(previous)


def test_a_filings_index_row_goes_stale_and_the_provider_is_reconsulted(
    clock, cache_backed_data_service, monkeypatch,
):
    """End to end through `DataService`: capability, key and TTL line up.

    Asserted here rather than on the table because the failure mode was never
    a wrong number — it was a capability string that had no row at all, which
    a table assertion cannot see.
    """
    ds = cache_backed_data_service
    ticker = f"ZZIDX{int(time.perf_counter_ns() % 100000)}"
    with SessionLocal() as db:
        db.merge(Company(
            ticker=ticker, company_name="Index Test Co",
            sector="Test", industry="Test", universe_tier="data_only",
            cik="0000001234",
        ))
        db.commit()

    calls: list[dict] = []
    payload = [{"type": "10-K", "accession_number": "0000001234-26-000001"}]

    def fake_chain(capability, fn_name, *args, **kwargs):
        calls.append({"capability": capability, "fn": fn_name, "kwargs": kwargs})
        return payload

    monkeypatch.setattr(ds, "_try_chain", fake_chain)
    try:
        assert ds.get_filings_index(ticker) == payload
        assert len(calls) == 1
        assert calls[0]["capability"] == "filings_index"
        assert calls[0]["kwargs"]["fetch_text"] is False, (
            "the index read must ask the provider to skip document bodies — "
            "that is the entire reason this capability exists"
        )

        # Within the TTL the poll is free.
        clock.advance(pc.TTL_BY_CAPABILITY["filings_index"] - 10)
        ds.get_filings_index(ticker)
        assert len(calls) == 1

        # Past it — and well inside the poller's 30-minute interval, which is
        # the property that makes change detection possible at all.
        clock.advance(20)
        payload = [
            {"type": "10-K", "accession_number": "0000001234-26-000001"},
            {"type": "8-K", "accession_number": "0000001234-26-000002"},
        ]
        assert len(ds.get_filings_index(ticker)) == 2
        assert len(calls) == 2
    finally:
        pc.invalidate("filings_index", ticker.upper())
        with SessionLocal() as db:
            db.query(Company).filter(Company.ticker == ticker).delete()
            db.commit()


def test_the_index_ttl_is_shorter_than_the_poll_interval():
    """A cache that outlives the poll would make the poller skip cycles.

    Not a taste question: with a TTL equal to the 30-minute interval, every
    other pass reads a row written by the previous one and observes nothing.
    """
    from app.monitoring import edgar_poller  # noqa: F401  (documents the pairing)

    assert pc.TTL_BY_CAPABILITY["filings_index"] <= 30 * 60 / 2


# ---------------------------------------------------------------------------
# Structure: every cached capability states its TTL
# ---------------------------------------------------------------------------

def _cached_capabilities() -> list[tuple[str, bool, int]]:
    """`(capability, has_ttl_override, lineno)` for every `self._cached` call.

    Read out of `data_service.py` with `ast` rather than listed here, because
    a hardcoded list is exactly as good as the memory of whoever adds the
    next capability — and the bug being guarded against is a capability
    someone forgot about.
    """
    source_path = inspect.getsourcefile(ds_mod)
    assert source_path
    with open(source_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=source_path)

    found: list[tuple[str, bool, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_cached"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            raise AssertionError(
                f"{source_path}:{node.lineno}: _cached called with a non-literal "
                "capability — this guard can no longer tell what is being cached"
            )
        capability = node.args[0].value
        override = any(
            kw.arg == "ttl_override" and _always_states_a_ttl(kw.value)
            for kw in node.keywords
        )
        found.append((str(capability), override, node.lineno))
    return found


def _always_states_a_ttl(value: ast.AST) -> bool:
    """Does this `ttl_override=` expression supply a TTL on every branch?

    `ttl_override=None` means "use the table", so it excuses nothing. Nor
    does a conditional with None on one side: `get_filings` passes
    `NEVER_EXPIRES if prefer_cached else None`, which falls back to the
    table for every caller that did not opt in, and so still owes an entry.
    Checking only for a literal None would read that call site as fully
    overridden and let `filings` drop out of the table unnoticed — the
    capability at the centre of the bug this file guards.
    """
    if isinstance(value, ast.Constant):
        return value.value is not None
    if isinstance(value, ast.IfExp):
        return _always_states_a_ttl(value.body) and _always_states_a_ttl(value.orelse)
    return True


def test_the_capability_walk_finds_the_call_sites():
    """If the walk breaks, the guard below passes vacuously. Say so first."""
    capabilities = {c for c, _, _ in _cached_capabilities()}
    assert len(capabilities) >= 12, f"only found {sorted(capabilities)}"
    assert {"profile", "filings", "filings_index", "transcripts", "key_metrics"} <= capabilities


def test_every_cached_capability_has_an_explicit_ttl():
    """The property the old code violated three times over.

    A missing entry is no longer catastrophic — it gets `DEFAULT_TTL_SECONDS`
    now instead of eternity — but it is still unstated intent. `filings` wants
    a week because a filed document is immutable; `filings_index` wants
    minutes because it drives a poll. A default cannot express either, and
    the failure of the previous design was precisely that nobody had to say.
    """
    offenders = [
        f"{capability} (data_service.py:{lineno})"
        for capability, override, lineno in _cached_capabilities()
        if capability not in pc.TTL_BY_CAPABILITY and not override
    ]
    assert not offenders, (
        "these capabilities are cached with no TTL stated anywhere — add an entry "
        "to provider_cache.TTL_BY_CAPABILITY, or pass ttl_override at the call "
        "site if the value is specific to that one read:\n  " + "\n  ".join(offenders)
    )


def test_listed_ttls_are_positive_or_the_explicit_sentinel():
    """`None` must never reappear in the table as a way of saying forever."""
    for capability, ttl in pc.TTL_BY_CAPABILITY.items():
        assert isinstance(ttl, int), f"{capability} has a non-integer TTL"
        assert ttl > 0 or ttl == pc.NEVER_EXPIRES, (
            f"{capability}={ttl!r}: use the NEVER_EXPIRES sentinel to mean forever"
        )
