"""Filing-driven fundamentals refresh (FIX-005, W5a S6).

Owner decision 2026-09-24 (decision 4, verbatim): "fundamental data should
load as often as it changes, i.e. quarterly results queried quarterly." So
the trigger is the filing, not a timer: `edgar_poller` already reads every
in-tier issuer's EDGAR index every 30 minutes, and that index carries the
period of report of each 10-Q/10-K/20-F/40-F. `observe_many` records it; a
refresh is scheduled only when a filed period is missing from durable
storage, so deploying this against a fully imported database schedules
nothing (no herd).

State lives in Postgres (`fundamental_refresh_state`, one row per ticker)
because its writers run in different threads and processes: the poller and
the nightly drain (worker), the regen pull-through (worker, a job thread),
admin syncs and first-contact requests (web). Every write of the scheduling
fields is a compare-and-set on `row_version` that re-reads and recomputes
on conflict, and a run's outcome is derived from the stored state after
the run, never from what the run believed when it started. A filing
observed while its refresh is being recorded is therefore never lost.

Schedule, stated in nights (the drain runs once a night inside
`history_backfill` at 03:15 UTC; a due time within an hour after the
03:00 UTC slot is rounded down to it, so a check planned for 03:20 is not
pushed a whole day):

- night 1: first check, at least `PUBLICATION_LAG` after the filing is seen;
- nights 2, 3, 5 and 9: retries while FMP has not published the period
  (+12 h, +24 h, +48 h, +96 h); the night-9 attempt may let the fallback
  provider fill exactly the filed period FMP still lacks;
- then weekly, until `ABANDON_AFTER` (45 days) from the first check.

A ticker still missing its filed period after the night-9 attempt is
"stuck": the nightly loop fails on the night it becomes stuck and names it
without failing on later nights. Companies with no filing signal (foreign
filers, out-of-tier names, detection gaps) are covered by a weekly
calendar check once a period is overdue, and by a 120-day sweep.

Every run goes through S5's FMP-primary `backfill_fundamentals` with
`mode="unattended"`, so a scheduled refresh quarantines only small, safe
sets and records a planned repair for anything else.

ETFs are never scheduled: they file no statements (`request` refuses them).
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, Table, func, inspect, or_, select, update
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import Company, FinancialPeriod, FundamentalRefreshState
from . import fundamental_history_service as fhs

log = logging.getLogger(__name__)

# First check after a filing is observed. FMP's publication lag is
# UNMEASURED: `last_result` receipts (attempt counts per filing) measure
# it; tune after one earnings season (late Oct 2026).
PUBLICATION_LAG = timedelta(hours=6)
# Retries while FMP has not published the filed period. Short at first
# (FMP usually publishes within a day), then backing off.
RETRY_BACKOFF = (timedelta(hours=12), timedelta(hours=24), timedelta(hours=48),
                 timedelta(hours=96), timedelta(days=7))
# The fifth attempt (night 9) concludes "FMP lacks it" and lets the
# fallback provider fill that period (owner decision: others fill only
# periods FMP lacks).
SECONDARY_AFTER_ATTEMPTS = 5
# Stop filing retries. The coverage issue stays blocking and the ticker
# stays named in the nightly note.
ABANDON_AFTER = timedelta(days=45)
# Calendar fallback: an overdue period is re-checked at most weekly.
CALENDAR_RECHECK = timedelta(days=7)
# A US filer is re-confirmed every quarter by its own filing; the sweep only
# catches detection gaps (CIK lookup failure, a long poller outage).
SWEEP_AFTER = timedelta(days=120)
# Operational guards, not a regulatory filing calendar.
QUARTERLY_DEADLINE_DAYS = 45
ANNUAL_DEADLINE_DAYS = 90
FOREIGN_INTERIM_DEADLINE_DAYS = 60
FOREIGN_ANNUAL_DEADLINE_DAYS = 120
CALENDAR_GRACE_DAYS = 7
# Peak earnings day in a ~173-name universe is ~25 filers; each refresh is
# six FMP JSON calls, no document bodies, no LLM.
MAX_FUNDAMENTAL_REFRESHES_PER_PASS = 30
MAX_DRAIN_SECONDS = 15 * 60
PACE_SECONDS = 1.0
LEASE = timedelta(minutes=15)
# The drain runs inside `history_backfill` (cron 03:15 UTC).
DRAIN_SLOT_HOUR = 3
DRAIN_SLOT_TOLERANCE = timedelta(hours=1)

QUARTERLY_FORMS = frozenset({"10-Q", "10-K"})
ANNUAL_FORMS = frozenset({"10-K", "20-F", "40-F"})
FOREIGN_ANNUAL_FORMS = frozenset({"20-F", "40-F"})
AMENDMENT_FORMS = frozenset({"10-K/A", "10-Q/A", "20-F/A", "40-F/A"})
# Evidence only; never ends an expectation (see `KNOWN_REPORTING_ENDED`).
DEREGISTRATION_FORMS = frozenset({"15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G", "15F-15D", "25", "25-NSE"})
# Triggers whose run is an attempt at a period that must arrive.
RETRY_TRIGGERS = frozenset({"filing", "first_import", "first_contact"})
TRIGGER_PRIORITY = {"filing": 0, "first_import": 1, "first_contact": 1, "amendment": 2,
                    "admin": 2, "calendar": 3, "sweep": 4}

_TABLE = cast(Table, FundamentalRefreshState.__table__)


def _now() -> datetime:
    """Naive UTC (every stored timestamp is). A test seam."""
    return datetime.utcnow()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()


def drain_slot(when: datetime) -> datetime:
    """Round a due time within an hour after 03:00 UTC down to 03:00.

    A retry computed from a check that ran at 03:20 would otherwise be due
    at 03:20 the next night and miss that night's 03:15 drain, costing a
    whole extra day per retry.
    """
    slot = when.replace(hour=DRAIN_SLOT_HOUR, minute=0, second=0, microsecond=0)
    if slot > when:
        slot -= timedelta(days=1)
    return slot if when - slot <= DRAIN_SLOT_TOLERANCE else when


def _ensure_table(db: Session) -> None:
    _TABLE.create(bind=db.get_bind(), checkfirst=True)


def _has_table(db: Session) -> bool:
    return inspect(db.connection()).has_table(_TABLE.name)


def _chunks(items: list[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Compare-and-set
# ---------------------------------------------------------------------------

def _insert_ignore(db: Session, values: dict[str, Any]) -> bool:
    from sqlalchemy.dialects import postgresql, sqlite
    insert: Any = postgresql.insert if db.get_bind().dialect.name == "postgresql" else sqlite.insert
    result = _rows(db.execute(insert(_TABLE).values(**values).on_conflict_do_nothing(index_elements=["ticker"])))
    return bool(result)


_DEFAULTS: dict[str, Any] = {"status": "idle", "trigger": "", "attempts": 0, "last_result": {}}


def _rows(result: Any) -> int:
    """Rows an UPDATE/INSERT matched (the compare-and-set verdict)."""
    return int(cast(CursorResult, result).rowcount)


def _read(db: Session, ticker: str) -> dict[str, Any] | None:
    row = db.execute(select(_TABLE).where(_TABLE.c.ticker == ticker)).mappings().first()
    return dict(row) if row is not None else None


def _cas(db: Session, ticker: str, compute: Callable[[dict[str, Any] | None], dict[str, Any] | None], *,
         current: dict[str, Any] | None | bool = False, tries: int = 5) -> dict[str, Any] | None:
    """Apply `compute(row) -> changes` with a compare-and-set on `row_version`.

    On conflict the row is re-read and `compute` runs again on the newer
    state, so a concurrent writer's facts are merged, never overwritten.
    `current` may pass a row already read in bulk (`False` = read it here).
    The caller commits.
    """
    row = _read(db, ticker) if current is False else current
    for _ in range(tries):
        assert row is None or isinstance(row, dict)
        changes = compute(row)
        if not changes:
            return row
        stamp = _now()
        if row is None:
            values = {**_DEFAULTS, **changes, "ticker": ticker, "row_version": 1, "updated_at": stamp}
            if _insert_ignore(db, values):
                return values
        else:
            version = row.get("row_version")
            guard = _TABLE.c.row_version.is_(None) if version is None else _TABLE.c.row_version == version
            result = db.execute(update(_TABLE).where(_TABLE.c.ticker == ticker, guard)
                                .values(**changes, row_version=(version or 0) + 1, updated_at=stamp))
            if _rows(result) == 1:
                return {**row, **changes, "row_version": (version or 0) + 1, "updated_at": stamp}
        log.info("fundamental refresh state %s changed concurrently; re-reading", ticker)
        row = _read(db, ticker)
    raise RuntimeError(f"fundamental refresh state {ticker}: compare-and-set kept conflicting")


# ---------------------------------------------------------------------------
# What is stored, what is filed, what is missing
# ---------------------------------------------------------------------------

def _stored_primary(db: Session, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Newest usable named-provider period per statement and cadence.

    Usable the way `fhs._coverage` judges it: each statement's PRIMARY line
    (income revenue, balance total assets, cash from operations), a named
    source, a valid currency and a finite value. Per ticker:
    `{"ends": {statement: {cadence: (period_end, fiscal_quarter)}},
    "fetched": newest fetched_at}`; absent = no usable named rows at all.
    """
    out: dict[str, dict[str, Any]] = {}
    if not tickers or not inspect(db.connection()).has_table(FinancialPeriod.__tablename__):
        return out
    primary_lines = or_(*[(FinancialPeriod.statement == statement) & (FinancialPeriod.line_item == line)
                          for statement, line in fhs.PRIMARY.items()])
    for chunk in _chunks(tickers):
        rows = db.execute(select(FinancialPeriod.ticker, FinancialPeriod.statement, FinancialPeriod.fiscal_quarter,
                                 FinancialPeriod.period_end, FinancialPeriod.value, FinancialPeriod.currency,
                                 FinancialPeriod.fetched_at)
                          .where(FinancialPeriod.ticker.in_(chunk), primary_lines,
                                 FinancialPeriod.source.not_in(fhs.LEGACY_SOURCES),
                                 FinancialPeriod.period_end.is_not(None), FinancialPeriod.value.is_not(None))).all()
        for ticker, statement, quarter, end, value, currency, fetched in rows:
            if not fhs._valid_currency(currency) or not math.isfinite(value):
                continue
            fact = out.setdefault(ticker, {"ends": {}, "fetched": None})
            per = fact["ends"].setdefault(statement, {})
            cadence = "annual" if quarter is None else "quarterly"
            if cadence not in per or end > per[cadence][0]:
                per[cadence] = (end, quarter)
            if fetched is not None and (fact["fetched"] is None or fetched > fact["fetched"]):
                fact["fetched"] = fetched
    return out


def _lagging_end(fact: dict[str, Any], cadence: str, *, strict: bool) -> tuple[date, int | None] | None:
    """The period every statement holds for `cadence`: the minimum newest end.

    `strict`: a statement with no row for the cadence makes it None (nothing
    is stored for all three). Otherwise the statements that have the cadence
    decide (the calendar's overdue anchor).
    """
    ends = [per.get(cadence) for per in (fact["ends"].get(s, {}) for s in fhs.PRIMARY)]
    if strict and any(e is None for e in ends):
        return None
    present = [e for e in ends if e is not None]
    return min(present, key=lambda e: e[0]) if present else None


def _stored_newest(db: Session, tickers: list[str]) -> dict[str, dict[str, date | None]]:
    """Newest period stored in ALL three statements per cadence; absent = no rows.

    Coverage judges a filed period current only when income, balance sheet
    and cash flow all hold it (`fhs._coverage` per statement). Deciding
    "stored" from revenue alone let a ticker whose FMP balance sheet or cash
    flow lagged its income statement go idle with coverage still blocking,
    never retried or named until the next filing.
    """
    out: dict[str, dict[str, date | None]] = {}
    for ticker, fact in _stored_primary(db, tickers).items():
        newest: dict[str, date | None] = {}
        for cadence in ("quarterly", "annual"):
            end = _lagging_end(fact, cadence, strict=True)
            newest[cadence] = end[0] if end else None
        out[ticker] = newest
    return out


def _missing(ticker: str, state: dict[str, Any], newest: dict[str, date | None] | None) -> dict[str, Any]:
    """Filed periods not stored (by more than the 52/53-week tolerance).

    `{"first_import": True}` when nothing named is stored at all. Nothing is
    ever missing for an issuer in the evidence-backed reporting-ended
    registry.
    """
    if fhs.reporting_ended_on(ticker) is not None:
        return {}
    if newest is None:
        return {"first_import": True}
    tolerance = timedelta(days=fhs.PERIOD_END_TOLERANCE_DAYS)
    out: dict[str, Any] = {}
    for cadence, filed in (("quarterly", state.get("filed_quarter_end")), ("annual", state.get("filed_annual_end"))):
        stored = newest[cadence]
        if filed is not None and (stored is None or filed > stored + tolerance):
            out[cadence] = filed
    return out


def _summarize_index(filings: list[dict]) -> dict[str, Any]:
    """Newest periodic filing per cadence, newest amendment, newest deregistration."""
    def newest(forms: frozenset[str]) -> tuple | None:
        best = None
        for f in filings:
            end = _date(f.get("period_end"))
            if f.get("type") in forms and end is not None:
                key = (end, _date(f.get("filing_date")) or date.min, f.get("accession_number") or "")
                if best is None or key > best[0]:
                    best = (key, f)
        if best is None:
            return None
        (end, filed_on, accession), f = best
        return end, f.get("type"), accession or None, filed_on if filed_on != date.min else None

    amendment = max(((_date(f.get("filing_date")) or date.min, f.get("accession_number") or "")
                     for f in filings if f.get("type") in AMENDMENT_FORMS and f.get("accession_number")),
                    default=None)
    dereg = max(((_date(f.get("filing_date")) or date.min, f.get("type"))
                 for f in filings if f.get("type") in DEREGISTRATION_FORMS), default=None)
    return {"quarter": newest(QUARTERLY_FORMS), "annual": newest(ANNUAL_FORMS),
            "amendment": amendment[1] if amendment else None,
            "dereg": (dereg[1], dereg[0] if dereg[0] != date.min else None) if dereg else None}


def _is_pending(state: dict[str, Any] | None) -> bool:
    return state is not None and state.get("status") == "pending"


def _priority(trigger: str | None) -> int:
    return TRIGGER_PRIORITY.get(trigger or "", 9)


def _observe_changes(ticker: str, cur: dict[str, Any] | None, summary: dict[str, Any],
                     newest_for: Callable[[], dict[str, date | None] | None], now: datetime) -> dict[str, Any] | None:
    state = cur or {}
    changes: dict[str, Any] = {}
    filed_changed = False
    for prefix, observed_key, fact in (("filed_quarter", "quarter_observed_at", summary["quarter"]),
                                       ("filed_annual", "annual_observed_at", summary["annual"])):
        if fact is None:
            continue
        end, form, accession, filed_on = fact
        if state.get(f"{prefix}_end") is None or end > state[f"{prefix}_end"]:
            changes.update({f"{prefix}_end": end, f"{prefix}_form": form, f"{prefix}_accession": accession,
                            f"{prefix}_on": filed_on, observed_key: now})
            filed_changed = True
    amendment_trigger = False
    if summary["amendment"] and summary["amendment"] != state.get("last_amendment_accession"):
        # A first observation only records the accession: no deploy-day herd.
        amendment_trigger = state.get("last_amendment_accession") is not None
        changes["last_amendment_accession"] = summary["amendment"]
    if summary["dereg"] and summary["dereg"] != (state.get("last_deregistration_form"), state.get("last_deregistration_on")):
        # Evidence only: never reporting-ended, never a schedule change.
        changes.update(last_deregistration_form=summary["dereg"][0], last_deregistration_on=summary["dereg"][1])
    if not changes:
        return None
    if filed_changed or cur is None:
        missing = _missing(ticker, {**state, **changes}, newest_for())
        if missing:
            trigger = "first_import" if missing.get("first_import") else "filing"
            due = drain_slot(now + PUBLICATION_LAG)
            restart = dict(trigger=trigger, attempts=0, first_due_at=due, requested_at=now, stuck_since=None,
                           stuck_reported_at=None)
            if not _is_pending(cur):
                changes.update(status="pending", due_at=due, **restart)
            elif _priority(trigger) < _priority(state.get("trigger")) or (filed_changed and state.get("trigger") == trigger):
                # Upgrade a calendar/sweep/amendment/admin request to a filing
                # (so it gets retries), or restart the retry clock for a
                # newer filed period. Never delay a check already due.
                changes.update(due_at=min(state.get("due_at") or due, due), **restart)
            return changes
    if amendment_trigger and _priority("amendment") < _priority(state.get("trigger") if _is_pending(cur) else None):
        due = drain_slot(now + fhs.PUBLICATION_GRACE)
        if _is_pending(cur):
            changes.update(trigger="amendment", due_at=min(state.get("due_at") or due, due), requested_at=now)
        else:
            # `first_due_at` and `attempts` are left alone: an abandoned row
            # keeps its abandonment (see `request`).
            changes.update(status="pending", trigger="amendment", due_at=due, requested_at=now)
    return changes


def observe_many(observations: list[tuple[str, list[dict]]], *, now: datetime | None = None) -> list[str]:
    """Record what each issuer has reported; schedule only missing periods.

    `observations` are `(ticker, filings_index_rows)` from one poller pass.
    DB-only, one session. Tickers whose own write failed are returned by
    name; the others are committed.
    """
    now = now or _now()
    summaries = {t.strip().upper(): _summarize_index(filings or []) for t, filings in observations}
    failed: list[str] = []
    with SessionLocal() as db:
        _ensure_table(db)
        existing: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(list(summaries)):
            for row in db.execute(select(_TABLE).where(_TABLE.c.ticker.in_(chunk))).mappings():
                existing[row["ticker"]] = dict(row)
        # Stored period ends are read only where a filed end changed or the
        # row is new: most passes observe nothing new and cost one query.
        need = []
        for ticker, s in summaries.items():
            state = existing.get(ticker)
            if state is None and any(s.values()):
                need.append(ticker)
            elif state is not None and any(
                    fact is not None and (state.get(f"{prefix}_end") is None or fact[0] > state[f"{prefix}_end"])
                    for prefix, fact in (("filed_quarter", s["quarter"]), ("filed_annual", s["annual"]))):
                need.append(ticker)
        newest = _stored_newest(db, need)
        db.commit()
        for ticker, summary in summaries.items():
            if ticker not in existing and not any(summary.values()):
                continue

            def newest_for(ticker: str = ticker) -> dict[str, date | None] | None:
                if ticker not in need:
                    return _stored_newest(db, [ticker]).get(ticker)
                return newest.get(ticker)

            def compute(cur: dict[str, Any] | None, ticker: str = ticker, summary: dict[str, Any] = summary,
                        newest_for: Callable[[], dict[str, date | None] | None] = newest_for) -> dict[str, Any] | None:
                return _observe_changes(ticker, cur, summary, newest_for, now)

            try:
                _cas(db, ticker, compute, current=existing.get(ticker))
                db.commit()
            except Exception as exc:
                db.rollback()
                failed.append(ticker)
                log.warning("fundamentals refresh-state observe failed for %s: %s", ticker, type(exc).__name__)
    return failed


def _is_etf(db: Session, ticker: str) -> bool:
    return bool(db.execute(select(Company.is_etf).where(Company.ticker == ticker)).scalar_one_or_none())


def request(ticker: str, trigger: str, *, due_at: datetime | None = None, now: datetime | None = None,
            db: Session | None = None) -> dict[str, Any] | None:
    """Ask for a refresh. Keeps an existing request that is due sooner or ranks higher.

    An ETF is refused (returns None, nothing written): it files no income
    statement, so a fundamentals import can never succeed and would only
    spend ~10 FMP runs over 45 days and fail the nightly loop once.
    """
    if trigger not in TRIGGER_PRIORITY:
        raise ValueError(f"unknown fundamentals refresh trigger {trigger!r}")
    ticker = ticker.strip().upper()
    now = now or _now()
    due = drain_slot(due_at or now)

    def compute(cur: dict[str, Any] | None) -> dict[str, Any] | None:
        if _is_pending(cur):
            assert cur is not None
            if _priority(trigger) < _priority(cur.get("trigger")):
                changes: dict[str, Any] = {"trigger": trigger, "due_at": min(cur.get("due_at") or due, due),
                                           "requested_at": now}
                if trigger in RETRY_TRIGGERS and cur.get("trigger") not in RETRY_TRIGGERS:
                    changes.update(attempts=0, first_due_at=due)
                return changes
            if cur.get("due_at") is None or due < cur["due_at"]:
                return {"due_at": due, "requested_at": now}
            return None
        changes = {"status": "pending", "trigger": trigger, "due_at": due, "requested_at": now}
        if trigger in RETRY_TRIGGERS or cur is None:
            changes.update(attempts=0, first_due_at=due if trigger in RETRY_TRIGGERS else None)
        # Otherwise `first_due_at` and `attempts` are kept. On an idle row
        # they are non-zero only when filing retries were abandoned, and
        # `record_result` measures abandonment from `first_due_at`: clearing
        # it here let the next weekly calendar check restart 45 days of
        # retries, forever.
        return changes

    own = db is None
    session = db or SessionLocal()
    try:
        _ensure_table(session)
        if _is_etf(session, ticker):
            log.info("fundamentals refresh not requested for %s: an ETF has no statements to import", ticker)
            return None
        state = _cas(session, ticker, compute)
        session.commit()
        return state
    finally:
        if own:
            session.close()


# ---------------------------------------------------------------------------
# Lease, run, result
# ---------------------------------------------------------------------------

def claim(ticker: str, *, now: datetime | None = None, db: Session | None = None) -> bool:
    """Take the per-ticker refresh lease (drain, pull-through and admin sync share it)."""
    ticker = ticker.strip().upper()
    now = now or _now()
    own = db is None
    session = db or SessionLocal()
    try:
        _ensure_table(session)
        _insert_ignore(session, {**_DEFAULTS, "ticker": ticker, "row_version": 0, "updated_at": now})
        result = session.execute(update(_TABLE).where(
            _TABLE.c.ticker == ticker, or_(_TABLE.c.lease_until.is_(None), _TABLE.c.lease_until < now),
        ).values(lease_until=now + LEASE))
        session.commit()
        return _rows(result) == 1
    finally:
        if own:
            session.close()


def release(ticker: str, *, db: Session | None = None) -> None:
    own = db is None
    session = db or SessionLocal()
    try:
        session.execute(update(_TABLE).where(_TABLE.c.ticker == ticker.strip().upper()).values(lease_until=None))
        session.commit()
    finally:
        if own:
            session.close()


def history_capable() -> bool:
    """True when the financials chain has a named provider with a history adapter.

    Under CI (`use_demo_data_only`) the chain is the demo provider, so
    scheduling and draining are deterministic no-ops.
    """
    try:
        chain = fhs.get_data_service()._live_chain("financials")
    except Exception:
        return False
    return any(str(getattr(p, "name", "")) not in fhs.LEGACY_SOURCES
               and callable(getattr(p, "get_financial_history", None)) for p in chain)


def _start_for(ticker: str) -> date:
    from .market_data_backfill import requested_start
    return requested_start(ticker, today=fhs._today())


def _run(ticker: str, state: dict[str, Any]) -> dict[str, Any]:
    allow_secondary = (state.get("trigger") in RETRY_TRIGGERS
                       and (state.get("attempts") or 0) + 1 >= SECONDARY_AFTER_ATTEMPTS)
    return fhs.backfill_fundamentals(ticker, _start_for(ticker), force_refresh=True, mode="unattended",
                                     allow_secondary_for_expected=allow_secondary)


def _failed_report(exc: BaseException) -> dict[str, Any]:
    return {"success": False, "committed": False, "attempts": [], "coverage": {}, "rows_written": 0,
            "rows_refreshed": 0, "rows_quarantined": 0, "issues": [], "error_type": type(exc).__name__}


def _invalidate_caches(ticker: str) -> dict[str, Any]:
    """This is the moment FMP has actually published (design §1.4): drop the
    7-day `financials` provider-cache row and the 90-day `company_cold`
    snapshot that were built from pre-filing statements."""
    out: dict[str, Any] = {}
    try:
        from .provider_cache import invalidate
        out["financials"] = {"success": True, "rows_removed": invalidate("financials", ticker)}
    except Exception as exc:
        out["financials"] = {"success": False, "error_type": type(exc).__name__}
    try:
        from ..cache.snapshots import invalidate as invalidate_snapshots
        out["company_cold"] = {"success": True, "rows_invalidated": invalidate_snapshots(ticker, kind="company_cold")}
    except Exception as exc:
        out["company_cold"] = {"success": False, "error_type": type(exc).__name__}
    return out


def _issue_kinds(report: dict[str, Any]) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for issue in report.get("issues") or []:
        if not issue.get("resolved"):
            kinds[issue.get("kind", "unknown")] = kinds.get(issue.get("kind", "unknown"), 0) + 1
    return kinds


def entitlement_denials(ticker: str, report: dict[str, Any]) -> list[str]:
    """Unresolved FMP 401/402/403 (a real plan gap, not a symbol spelling)."""
    return [f"{ticker}:{i.get('endpoint')}:{i.get('cadence')}:{i.get('status')}"
            for i in report.get("issues") or []
            if i.get("kind") == "provider_entitlement_denied" and not i.get("resolved")]


def record_result(ticker: str, report: dict[str, Any], *, trigger: str, now: datetime | None = None,
                  started_at: datetime | None = None, invalidate_caches: bool = True,
                  db: Session | None = None) -> dict[str, Any]:
    """Record one run and derive the next state from what is now STORED.

    Whatever the run's trigger, if EDGAR shows a filed period that is still
    not stored the row stays (or goes back to) pending as a filing retry,
    keeping its first due time and attempt count; only a run that is itself
    a retry (`RETRY_TRIGGERS`) counts as an attempt. A request that arrived
    while the run was in flight (`requested_at > started_at`) is kept.
    Compare-and-set on `row_version`: on conflict the row is re-read and
    the outcome recomputed, so nothing a concurrent observer wrote is lost.
    """
    ticker = ticker.strip().upper()
    now = now or _now()
    changed = (report.get("committed") and ((report.get("rows_written") or 0) - (report.get("rows_refreshed") or 0) > 0
                                            or (report.get("rows_quarantined") or 0) > 0))
    caches = _invalidate_caches(ticker) if changed and invalidate_caches else None
    fmp_answered = any(a.get("provider") == fhs.PRIMARY_PROVIDER and a.get("received")
                       for a in report.get("attempts") or [])
    counted = trigger in RETRY_TRIGGERS
    outcome: dict[str, Any] = {}
    own = db is None
    session = db or SessionLocal()

    def compute(cur: dict[str, Any] | None) -> dict[str, Any]:
        state = cur or {}
        newest = _stored_newest(session, [ticker]).get(ticker)
        missing = _missing(ticker, state, newest)
        attempts = (state.get("attempts") or 0) + (1 if counted else 0)
        interrupted = (_is_pending(cur) and started_at is not None and state.get("requested_at") is not None
                       and state["requested_at"] > started_at)
        changes: dict[str, Any] = {"last_checked_at": now}
        if fmp_answered:
            changes["last_success_at"] = now
        if trigger == "calendar":
            changes["last_calendar_check_at"] = now
        abandoned = newly_stuck = False
        if missing:
            first_due = state.get("first_due_at") or now
            abandoned = now - first_due >= ABANDON_AFTER
            stuck = abandoned or attempts >= SECONDARY_AFTER_ATTEMPTS
            newly_stuck = stuck and state.get("stuck_since") is None
            if newly_stuck:
                changes.update(stuck_since=now, stuck_reported_at=None)
            if not interrupted:
                if abandoned:
                    # first_due_at is kept: a later calendar check that finds
                    # the same period missing must not restart 45 days of
                    # retries. A newer filed period restarts it (observe).
                    changes.update(status="idle", due_at=None, attempts=attempts, first_due_at=first_due)
                else:
                    keep = state.get("trigger") if _is_pending(cur) and state.get("trigger") in RETRY_TRIGGERS else None
                    next_trigger = keep or ("first_import" if missing.get("first_import") else "filing")
                    if counted:
                        due = drain_slot(now + RETRY_BACKOFF[min(attempts - 1, len(RETRY_BACKOFF) - 1)])
                    elif _is_pending(cur) and state.get("due_at") is not None:
                        due = state["due_at"]
                    else:
                        due = drain_slot(now + RETRY_BACKOFF[0])
                    changes.update(status="pending", trigger=next_trigger, attempts=attempts, due_at=due,
                                   first_due_at=first_due)
        elif not interrupted:
            changes.update(status="idle", trigger="", due_at=None, first_due_at=None, attempts=0, stuck_since=None,
                           stuck_reported_at=None)
        final = {**state, **changes}
        newest_iso = {c: _iso(v) for c, v in (newest or {}).items()}
        changes["last_result"] = {
            "checked_at": now.isoformat(), "trigger": trigger, "success": bool(report.get("success")),
            "satisfied": not missing, "abandoned": abandoned, "attempt": attempts if counted else state.get("attempts") or 0,
            "providers": report.get("provider") or [], "fmp_answered": fmp_answered,
            "rows_written": report.get("rows_written") or 0, "rows_refreshed": report.get("rows_refreshed") or 0,
            "rows_quarantined": report.get("rows_quarantined") or 0,
            "quarantine_repair_id": report.get("quarantine_repair_id"),
            "planned_repair_id": report.get("planned_repair_id"),
            "new_period_ends": report.get("new_period_ends") or {},
            "expected": {"quarterly": _iso(state.get("filed_quarter_end")), "annual": _iso(state.get("filed_annual_end"))},
            "newest": newest_iso, "missing": {c: _iso(v) for c, v in missing.items() if c != "first_import"},
            "issue_kinds": _issue_kinds(report), "entitlement_denied": entitlement_denials(ticker, report),
            "error_type": report.get("error_type"), "cache_invalidation": caches,
        }
        outcome.clear()
        outcome.update(ticker=ticker, satisfied=not missing, missing=missing, abandoned=abandoned,
                       newly_stuck=newly_stuck, status=final.get("status"), trigger=final.get("trigger"),
                       attempts=final.get("attempts"), due_at=final.get("due_at"),
                       filed={"quarterly": (state.get("filed_quarter_form"), state.get("filed_quarter_on")),
                              "annual": (state.get("filed_annual_form"), state.get("filed_annual_on"))},
                       cache_invalidation=caches)
        return changes

    try:
        _ensure_table(session)
        _cas(session, ticker, compute)
        session.commit()
    finally:
        if own:
            session.close()
    return outcome


def _missing_note(outcome: dict[str, Any]) -> list[str]:
    notes = []
    for cadence, end in outcome["missing"].items():
        if cadence == "first_import":
            notes.append(f"{outcome['ticker']}:first_import(attempt {outcome['attempts']}, "
                         f"next {_slot_text(outcome['due_at'])})")
            continue
        form, filed_on = outcome["filed"].get(cadence, (None, None))
        notes.append(f"{outcome['ticker']}:{cadence}:{end.isoformat()}({form} filed {_iso(filed_on)}, "
                     f"attempt {outcome['attempts']}, next {_slot_text(outcome['due_at'])})")
    return notes


def _slot_text(when: datetime | None) -> str:
    return when.strftime("%Y-%m-%dT%H:%M") if when else "none"


# ---------------------------------------------------------------------------
# Nightly: calendar/sweep scheduling and the bounded drain
# ---------------------------------------------------------------------------

def schedule_calendar_and_sweep(tickers: list[str], *, now: datetime | None = None) -> dict[str, list[str]]:
    """Request checks for companies the filing signal does not cover. DB-only.

    Runs over every `fundamentals_required` company (every `companies`
    row), not just the polled tier (integration critique): an out-of-tier
    or foreign issuer has no filing trigger, so an overdue period is
    re-checked weekly and a long-unconfirmed ticker every 120 days. These
    deadlines are operational guards, not a regulatory calendar.
    """
    now = now or _now()
    today = now.date()
    tickers = sorted({t.strip().upper() for t in tickers})
    out: dict[str, list[str]] = {"calendar": [], "sweep": [], "first_import": []}
    with SessionLocal() as db:
        _ensure_table(db)
        states: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(tickers):
            for row in db.execute(select(_TABLE).where(_TABLE.c.ticker.in_(chunk))).mappings():
                states[row["ticker"]] = dict(row)
        stored = _stored_primary(db, tickers)
        db.commit()
        for ticker in tickers:
            state = states.get(ticker)
            if _is_pending(state) or fhs.reporting_ended_on(ticker) is not None:
                continue
            stored_fact = stored.get(ticker)
            if stored_fact is None:
                # Nothing named is stored. Request the durable import unless
                # an earlier one was abandoned (it keeps `first_due_at`). A
                # bare idle row, left by an admin sync that took the lease
                # and then failed or found its sync claim held, is not an
                # answer: without this the company was never scheduled again.
                if state is None or state.get("first_due_at") is None:
                    request(ticker, "first_import", now=now, db=db)
                    out["first_import"].append(ticker)
                continue
            # The overdue anchor is the statement that lags: a balance sheet
            # FMP has not published yet is as missing as the income line.
            quarter = _lagging_end(stored_fact, "quarterly", strict=False)
            annual = _lagging_end(stored_fact, "annual", strict=False)
            fact = {"q": quarter[0] if quarter else None, "fq": quarter[1] if quarter else None,
                    "a": annual[0] if annual else None, "fetched": stored_fact["fetched"]}
            foreign = (state or {}).get("filed_annual_form") in FOREIGN_ANNUAL_FORMS
            annual_deadline = FOREIGN_ANNUAL_DEADLINE_DAYS if foreign else ANNUAL_DEADLINE_DAYS
            interim_deadline = FOREIGN_INTERIM_DEADLINE_DAYS if foreign else QUARTERLY_DEADLINE_DAYS
            overdue = False
            if fact["q"] is not None:
                # After Q3 the next quarter arrives with the annual report.
                deadline = annual_deadline if fact["fq"] == 3 else interim_deadline
                overdue = today > fact["q"] + timedelta(days=91 + deadline + CALENDAR_GRACE_DAYS)
            if fact["a"] is not None:
                overdue = overdue or today > fact["a"] + timedelta(days=365 + annual_deadline + CALENDAR_GRACE_DAYS)
            last_calendar = (state or {}).get("last_calendar_check_at")
            last_checked = (state or {}).get("last_checked_at")
            if overdue and (last_calendar is None or now - last_calendar >= CALENDAR_RECHECK):
                request(ticker, "calendar", now=now, db=db)
                out["calendar"].append(ticker)
            elif (fact["fetched"] is None or now - fact["fetched"] >= SWEEP_AFTER) and (
                    last_checked is None or now - last_checked >= CALENDAR_RECHECK):
                request(ticker, "sweep", now=now, db=db)
                out["sweep"].append(ticker)
    return out


def _empty_result() -> dict[str, Any]:
    return {"refreshed": 0, "satisfied": 0, "pending": 0, "rows_written": 0, "rows_quarantined": 0,
            "repair_ids": [], "over_cap": [], "over_budget": [], "leased": [], "missing": [], "stuck": [],
            "still_missing": [], "errors": [], "entitlement_denied": [], "skipped_reason": None}


def _refresh_one(ticker: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Claimed-run-recorded-released for one due ticker; None if not run."""
    with SessionLocal() as db:
        if not claim(ticker, now=_now(), db=db):
            result["leased"].append(ticker)
            return None
        state = _read(db, ticker)
    try:
        # Re-check under the lease: a pull-through may have just run it.
        if not _is_pending(state) or state is None or state.get("due_at") is None or state["due_at"] > _now():
            return None
        started = _now()
        try:
            report = _run(ticker, state)
        except Exception as exc:
            result["errors"].append(f"{ticker}:{type(exc).__name__}")
            log.warning("fundamentals refresh failed for %s: %s", ticker, type(exc).__name__)
            report = _failed_report(exc)
        outcome = record_result(ticker, report, trigger=state.get("trigger") or "filing", started_at=started)
        outcome["report"] = report
        return outcome
    finally:
        try:
            release(ticker)
        except Exception as exc:  # the lease expires on its own
            log.warning("fundamentals refresh lease release failed for %s: %s", ticker, type(exc).__name__)


def drain(*, now: datetime | None = None, limit: int = MAX_FUNDAMENTAL_REFRESHES_PER_PASS) -> dict[str, Any]:
    """Run due refreshes in `(trigger priority, due time)` order, bounded.

    At most `limit` refreshes and `MAX_DRAIN_SECONDS`; everything over
    either bound is named, never dropped (its row stays pending and due).
    """
    now = now or _now()
    started = _monotonic()
    result = _empty_result()
    with SessionLocal() as db:
        _ensure_table(db)
        due = [dict(r) for r in db.execute(select(_TABLE).where(
            _TABLE.c.status == "pending", _TABLE.c.due_at.is_not(None), _TABLE.c.due_at <= now)).mappings()]
        db.commit()
    due.sort(key=lambda r: (_priority(r.get("trigger")), r["due_at"], r["ticker"]))
    count = 0
    for index, row in enumerate(due):
        ticker = row["ticker"]
        if count >= limit:
            result["over_cap"] = [r["ticker"] for r in due[index:]]
            break
        if _monotonic() - started >= MAX_DRAIN_SECONDS:
            result["over_budget"] = [r["ticker"] for r in due[index:]]
            break
        try:
            outcome = _refresh_one(ticker, result)
        except Exception as exc:
            result["errors"].append(f"{ticker}:{type(exc).__name__}")
            log.warning("fundamentals refresh bookkeeping failed for %s: %s", ticker, type(exc).__name__)
            continue
        if outcome is None:
            continue
        count += 1
        report = outcome.pop("report")
        result["refreshed"] += 1
        result["satisfied"] += int(outcome["satisfied"])
        result["pending"] += int(outcome["status"] == "pending")
        result["rows_written"] += report.get("rows_written") or 0
        result["rows_quarantined"] += report.get("rows_quarantined") or 0
        for key in ("quarantine_repair_id", "planned_repair_id"):
            if report.get(key):
                result["repair_ids"].append(f"{ticker}:{report[key]}")
        result["entitlement_denied"].extend(entitlement_denials(ticker, report))
        if outcome["missing"] and outcome["status"] == "pending" and not outcome["newly_stuck"]:
            result["missing"].extend(_missing_note(outcome))
        _sleep(PACE_SECONDS)
    _report_stuck(result, now)
    return result


def _report_stuck(result: dict[str, Any], now: datetime) -> None:
    """A stuck ticker fails the nightly loop once, then is only named.

    `stuck` = became stuck since the last report (by the drain or a memo
    pull-through): the loop fails. `still_missing` = reported before and
    still missing its filed period: named without failing, so one issuer
    FMP never publishes cannot keep the loop red for weeks.
    """
    with SessionLocal() as db:
        rows = db.execute(select(_TABLE.c.ticker, _TABLE.c.stuck_since, _TABLE.c.stuck_reported_at)
                          .where(_TABLE.c.stuck_since.is_not(None)).order_by(_TABLE.c.ticker)).all()
        for ticker, since, reported in rows:
            if reported is None:
                result["stuck"].append(ticker)
                db.execute(update(_TABLE).where(_TABLE.c.ticker == ticker, _TABLE.c.stuck_since == since)
                           .values(stuck_reported_at=now))
            else:
                result["still_missing"].append(f"{ticker}(since {since.date().isoformat()})")
        db.commit()


def _company_tickers() -> list[str]:
    """Every company that can have fundamentals: ETFs file no statements."""
    with SessionLocal() as db:
        return [t for (t,) in db.execute(select(Company.ticker).where(func.coalesce(Company.is_etf, False).is_(False))
                                         .order_by(Company.ticker)).all()]


def nightly(*, now: datetime | None = None) -> dict[str, Any]:
    """Schedule calendar/sweep checks, then drain. Never raises."""
    result = _empty_result()
    try:
        if not history_capable():
            result["skipped_reason"] = "no named history provider in the financials chain"
            return result
        scheduled = schedule_calendar_and_sweep(_company_tickers(), now=now)
        result = drain(now=now)
        result["scheduled"] = scheduled
    except Exception as exc:
        log.warning("fundamentals nightly refresh failed: %s", type(exc).__name__)
        result["errors"].append(f"nightly:{type(exc).__name__}")
    return result


def refresh_if_due(ticker: str, *, now: datetime | None = None) -> dict[str, Any] | None:
    """Pull one pending, due refresh forward (memo start). None when nothing ran.

    WHY: a memo requested after FMP has published but before the 03:15
    drain should get the new period. One ticker, only when already pending
    and due; it never triggers a memo.
    """
    ticker = ticker.strip().upper()
    now = now or _now()
    if not history_capable():
        return None
    with SessionLocal() as db:
        if not _has_table(db):
            return None
        state = _read(db, ticker)
    if not _is_pending(state) or state is None or state.get("due_at") is None or state["due_at"] > now:
        return None
    result = _empty_result()
    outcome = _refresh_one(ticker, result)
    if outcome is None:
        return None
    report = outcome.pop("report")
    return {"ticker": ticker, "trigger": state.get("trigger"), "satisfied": outcome["satisfied"],
            "status": outcome["status"], "rows_written": report.get("rows_written") or 0,
            "errors": result["errors"]}


def state_summary(tickers: list[str], *, db: Session | None = None) -> dict[str, dict[str, Any]]:
    """Compact per-ticker refresh state (admin coverage, Explorer)."""
    own = db is None
    session = db or SessionLocal()
    out: dict[str, dict[str, Any]] = {}
    try:
        if not _has_table(session):
            return out
        wanted = sorted({t.strip().upper() for t in tickers})
        for chunk in _chunks(wanted):
            for row in session.execute(select(_TABLE).where(_TABLE.c.ticker.in_(chunk))).mappings():
                out[row["ticker"]] = {
                    "status": row["status"], "trigger": row["trigger"], "due_at": _iso(row["due_at"]),
                    "first_due_at": _iso(row["first_due_at"]), "attempts": row["attempts"],
                    "stuck_since": _iso(row["stuck_since"]),
                    "filed_quarter_end": _iso(row["filed_quarter_end"]), "filed_quarter_form": row["filed_quarter_form"],
                    "filed_quarter_on": _iso(row["filed_quarter_on"]),
                    "quarter_observed_at": _iso(row["quarter_observed_at"]),
                    "filed_annual_end": _iso(row["filed_annual_end"]), "filed_annual_form": row["filed_annual_form"],
                    "filed_annual_on": _iso(row["filed_annual_on"]),
                    "annual_observed_at": _iso(row["annual_observed_at"]),
                    "last_deregistration_form": row["last_deregistration_form"],
                    "last_deregistration_on": _iso(row["last_deregistration_on"]),
                    "last_checked_at": _iso(row["last_checked_at"]), "last_success_at": _iso(row["last_success_at"]),
                    "last_result": row["last_result"] or {},
                }
        return out
    finally:
        if own:
            session.close()
