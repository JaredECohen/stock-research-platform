"""One-time deep backfill of `price_month_ends`, so the scorecard
evaluation can stop saying `insufficient_data`.

WHY THIS EXISTS
---------------
`scorecard_evaluation` draws no verdict until its panel holds
`DEFAULT_MIN_OBS` (2000) usable observations spread over
`DEFAULT_MIN_MONTHS` (24) distinct month ends. Its month-end price store
is fed from the app's rolling 252-day window
(`scorecard_service._SERIES_WINDOW_DAYS`), and `pit_prepare` adds exactly
ONE month per run going forward — so the evaluation is mute for well over
a year after launch (that is what `scorecard_evaluation.PRICE_DEPTH_NOTE`
says, and it is honest, not broken). One deep fetch per name converts a
large amount of finished work from silent to useful.

`data_service.get_price_history(ticker, days=N)` takes `days` as a
PARAMETER, not a cap: `FMPProvider.get_price_history` passes it straight
through as the API's `limit`, and `DataService._cached` keys the entry on
`f"{ticker}:{days}"`. A 1260-day fetch therefore gets its own provider-cache
entry and does NOT evict or pollute the 252-day one every other consumer
reads. That is the whole reason a deep backfill is possible without
disturbing the live app.

WHAT IT COSTS
-------------
One PAID provider call per ticker per `--days` value, per the `prices`
cache TTL (1 day fresh / 7 days stale-usable, `services/provider_cache`).
That is why `--max-tickers` defaults to a deliberately small number: an
accidental run is cheap, and the report always names what the budget cut
so the operator knows how many runs remain.

RUNNING IT IN PRODUCTION
------------------------
From the Render **worker** shell (it already has the DB and the provider
keys; the web service must not do provider work), in `backend/`::

    # 1. Look before you spend — fetches nothing.
    python -m app.scripts.backfill_month_end_prices --dry-run

    # 2. Spend a small budget, confirm the coverage delta, repeat.
    python -m app.scripts.backfill_month_end_prices --max-tickers 25

    # 3. Keep re-running the SAME command until the report's
    #    `names still to fetch` line reads 0. It is `budget.needs_work`
    #    in `--json`, and it counts the names this run selected plus the
    #    names the budget cut — the work a further run would still do.

Each run advances because a name leaves the candidate list for one of two
reasons, both durable:

  satisfied  it reached the per-name price-depth target, so another fetch
             would buy nothing.
  exhausted  the provider was ALREADY ASKED at this depth and could not
             fill it (a recent listing, a delisted name, an outage). The
             attempt is recorded in `price_backfill_attempts`, so the name
             is not re-bought on every subsequent run.

Without that second rule the loop stalls: a name that can never reach the
target is re-selected forever, eats the whole budget and blocks the rest
of the universe while still costing money every run. `--retry-exhausted`
re-opens them (use it after a provider outage is resolved), and so does a
larger `--days`, since the ledger records the depth that was asked for.

Safe to interrupt: every ticker commits on its own (`sync_price_month_ends`
opens and closes a session per ticker, and the attempt row is committed
right after it), a Ctrl-C during a fetch is caught and the run stops
cleanly, and the partial report — including the coverage the finished
tickers did buy — is still printed, with exit status 130. Resume is just
the same command again.

Idempotent by construction. The row-level upsert in
`scorecard_pit.sync_price_month_ends` leaves an unchanged `(ticker,
month_end)` row alone, so nothing is duplicated and an already-stored
month end is never rewritten; on top of that, this script skips a whole
ticker that is satisfied or exhausted, so the paid call is not made
either.

THIS IS AN OPERATOR TOOL. It is never called from a monitoring loop, a
route, or application code — `test_backfill_month_end_prices` asserts
that nothing outside `app/scripts` and `app/tests` imports it. The
recurring, one-month-at-a-time equivalent is the worker's
`pit_prepare`; the shallow CLI equivalent is `app.scripts.scorecard_backfill
--prices`, which exists for the 252-day window and does not budget, plan
or report coverage.

WHAT THE NUMBERS MEAN
---------------------
Two different things are reported, and conflating them is how this script
would lie to an operator.

1. PRICE-STORE DEPTH — derived here from the stored `price_month_ends`
   rows. `build_panel` turns a month-end score row into a usable
   observation only when four month-end closes are stored for that
   ticker: `M-12` and `M-1` (momentum_12_1 and reversal_1m, both LASSO
   controls), `M` itself, and `M+1` (the forward return). So a contiguous
   run of `L` stored month ends yields `L - 13` price-evaluable months,
   and 37 contiguous month ends give 24 — roughly 780 trading days, which
   is why `--days` defaults to 1260. This is the CEILING the backfill
   raises, and the per-name `--depth-target-months` is a BUDGET rule
   ("stop paying for this name"), not the evaluation's gate.

2. THE EVALUATION READ-OUT — obtained by asking
   `scorecard_evaluation` itself: `build_panel` → `_quintile` → `_lasso`,
   read-only, no run row, no provider call. This is what the evaluation
   will actually see, and it is the number the exercise exists to move.

They must not be conflated, because the evaluation applies filters the
price store knows nothing about: one succeeded run per month wins,
`overall_z` must be present, `coverage` must clear
`settings.scorecard_min_coverage`, the month must be quintile-eligible,
and every LASSO control (`log_mktcap`, `momentum_12_1`, `reversal_1m`,
`beta`, `roa`) must be non-null. A universe can be at full price depth
and still yield ZERO usable observations. Re-deriving those filters here
would drift from the producer, so this script does not re-derive them —
it runs them and prints `n_obs`, `n_months` and the dropped-row reasons.

Note also that the evaluation's floors are PANEL-WIDE, not per name:
`double_selection` counts `len(unique(month_ids))` across every name, and
`min_obs` is a total. A per-name month count is nowhere in the gate. The
per-name target here exists only to decide when to stop paying for a
ticker.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    select,
)

from ..agents.log_safety import safe_exc
from ..config import settings
from ..database import SessionLocal
from ..models import PriceMonthEnd
from ..services import scorecard_evaluation, scorecard_pit, scorecard_service

log = logging.getLogger(__name__)

# The month arithmetic MUST be the evaluation's own, not a re-derivation:
# a one-month disagreement here would report coverage the panel does not
# actually have.
_next_month_end = scorecard_evaluation._next_month_end
_prev_month_end = scorecard_evaluation._prev_month_end

MIN_MONTHS = scorecard_evaluation.DEFAULT_MIN_MONTHS
MIN_OBS = scorecard_evaluation.DEFAULT_MIN_OBS

# Offsets `build_panel` needs stored, relative to an observation month M,
# for that month to survive into the LASSO: M-12 and M-1 (momentum_12_1,
# reversal_1m), M (the formation close), M+1 (the forward return).
_REQUIRED_OFFSETS: tuple[int, ...] = (-12, -1, 0, 1)
# Contiguous month ends needed for one evaluable month; see the docstring.
LEAD_MONTHS = max(_REQUIRED_OFFSETS) - min(_REQUIRED_OFFSETS)  # 13

# ~5 calendar years of trading days. 37 contiguous month ends (the depth
# that gives 24 evaluable months) is about 780; 1260 leaves room for
# holidays, halts and a name that listed mid-window.
DEFAULT_DAYS = 1260
# Deliberately small. An accidental run costs a handful of provider calls;
# the report names every ticker the budget cut.
DEFAULT_MAX_TICKERS = 5
# Per-name price depth at which this script stops paying for a ticker.
# A BUDGET rule, not the evaluation's gate — the gate is panel-wide (see
# the module docstring). It defaults to the evaluation's month floor only
# because a name carrying fewer months than the whole panel needs is
# obviously still worth a fetch.
DEFAULT_DEPTH_TARGET_MONTHS = MIN_MONTHS


# ---------------------------------------------------------------------------
# Attempt ledger
# ---------------------------------------------------------------------------
#
# Script-private table, deliberately NOT on `Base.metadata`: it is operator
# bookkeeping, not application schema, so it must not appear in `init_db`,
# in the schema-drift reconciler or in the frozen model/table name sets.
# It is created lazily on first use, the same way
# `scorecard_pit._ensure_tables` does, and it lives in the DATABASE rather
# than a module dict because the answer has to survive the process — the
# whole point is that the NEXT run knows what this one asked for.

_ATTEMPTS_METADATA = MetaData()

price_backfill_attempts = Table(
    "price_backfill_attempts",
    _ATTEMPTS_METADATA,
    Column("ticker", String(16), primary_key=True),
    Column("attempted_at", DateTime, nullable=False),
    # The depth that was ASKED FOR. A later run with a bigger --days is a
    # genuinely new question, so it re-opens the name.
    Column("requested_days", Integer, nullable=False),
    # Complete months the provider returned. None (with a reason) when the
    # provider returned nothing at all — never 0, which would read as
    # "a listing too young for one complete month".
    Column("months_returned", Integer, nullable=True),
    Column("outcome", String(24), nullable=False),
    Column("reason", Text, nullable=False),
)

OUTCOME_FETCHED = "fetched"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_ERROR = "error"


def _utcnow() -> datetime:
    """Clock seam — tests set this instead of freezing time."""
    return datetime.utcnow()


def _ensure_attempts_table(db: Any) -> None:
    price_backfill_attempts.create(db.get_bind(), checkfirst=True)


@dataclass(frozen=True)
class Attempt:
    """One recorded provider question for one ticker."""
    ticker: str
    attempted_at: datetime | None
    requested_days: int
    months_returned: int | None
    outcome: str
    reason: str

    def covers(self, days: int) -> bool:
        """True when this attempt already asked for at least `days`, so
        asking again at `days` would buy the same answer."""
        return self.requested_days >= days

    def retirement_reason(self) -> str:
        when = self.attempted_at.date().isoformat() if self.attempted_at else "an earlier run"
        got = (
            f"{self.months_returned} complete months"
            if self.months_returned is not None
            else "no series at all"
        )
        return (
            f"asked on {when} at {self.requested_days} trading days and got {got} "
            f"({self.outcome}: {self.reason}). Re-asking at this depth buys the same answer; "
            f"a larger --days re-opens it, and --retry-exhausted forces a retry."
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attempted_at"] = self.attempted_at.isoformat() if self.attempted_at else None
        return d


def load_attempts(db: Any, tickers: list[str]) -> dict[str, Attempt]:
    """`{ticker: Attempt}` for the tickers in scope, one query."""
    out: dict[str, Attempt] = {}
    if not tickers:
        return out
    _ensure_attempts_table(db)
    rows = db.execute(
        select(price_backfill_attempts).where(price_backfill_attempts.c.ticker.in_(tickers))
    ).all()
    for r in rows:
        out[r.ticker] = Attempt(
            ticker=r.ticker, attempted_at=r.attempted_at, requested_days=int(r.requested_days),
            months_returned=None if r.months_returned is None else int(r.months_returned),
            outcome=r.outcome or "", reason=r.reason or "",
        )
    return out


def record_attempt(
    ticker: str, *, days: int, months_returned: int | None, outcome: str, reason: str,
    db: Any | None = None,
) -> None:
    """Write (replacing) this ticker's attempt row and COMMIT.

    Committed per ticker, right after its fetch, so a Ctrl-C or a crash
    leaves the ledger agreeing with what was actually bought.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_attempts_table(db)
        db.execute(delete(price_backfill_attempts).where(price_backfill_attempts.c.ticker == ticker))
        db.execute(price_backfill_attempts.insert().values(
            ticker=ticker, attempted_at=_utcnow(), requested_days=int(days),
            months_returned=months_returned, outcome=outcome, reason=reason,
        ))
        db.commit()
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------------------
# Price-store coverage (the CEILING, not the gate)
# ---------------------------------------------------------------------------

def _iter_month_ends(first: date, last: date) -> Iterator[date]:
    cur = first
    while cur <= last:
        yield cur
        cur = _next_month_end(cur)


@dataclass(frozen=True)
class TickerCoverage:
    """One ticker's month-end depth. Every count here is derived from
    stored rows; nothing is estimated, and `None` always carries a
    `reason` rather than standing in as 0.

    `price_evaluable` is a CEILING: months whose four required closes are
    stored. Whether the evaluation can use them depends on filters the
    price store cannot see — see `EvaluationReadout`.
    """
    ticker: str
    months_stored: int
    first_month: date | None
    last_month: date | None
    gap_months: int
    price_evaluable: int
    depth_target_met: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["first_month"] = self.first_month.isoformat() if self.first_month else None
        d["last_month"] = self.last_month.isoformat() if self.last_month else None
        return d


@dataclass(frozen=True)
class Coverage:
    """Universe-level price-store depth.

    `price_evaluable` and `price_evaluable_months` are the price-side
    analogues of the evaluation's two panel-wide floors (`min_obs` total
    observations, `min_months` DISTINCT month ends) — upper bounds on
    them, never a claim about them.
    """
    depth_target_months: int
    min_months: int
    min_obs: int
    tickers: tuple[TickerCoverage, ...]
    months_stored: int
    price_evaluable: int
    price_evaluable_months: int
    names_at_depth_target: int

    @property
    def price_could_meet_min_obs(self) -> bool:
        return self.price_evaluable >= self.min_obs

    @property
    def price_could_meet_min_months(self) -> bool:
        return self.price_evaluable_months >= self.min_months

    def by_ticker(self) -> dict[str, TickerCoverage]:
        return {c.ticker: c for c in self.tickers}

    def to_dict(self) -> dict[str, Any]:
        stored = [c.months_stored for c in self.tickers]
        return {
            "depth_target_months": self.depth_target_months,
            "min_months": self.min_months,
            "min_obs": self.min_obs,
            "n_tickers": len(self.tickers),
            "months_stored": self.months_stored,
            "months_stored_min": min(stored) if stored else None,
            "months_stored_max": max(stored) if stored else None,
            "months_stored_reason": None if stored else "no tickers in scope",
            "price_evaluable": self.price_evaluable,
            "price_evaluable_months": self.price_evaluable_months,
            "price_could_meet_min_obs": self.price_could_meet_min_obs,
            "price_could_meet_min_months": self.price_could_meet_min_months,
            "names_at_depth_target": self.names_at_depth_target,
            "tickers": [c.to_dict() for c in self.tickers],
        }


def evaluable_months(stored: Iterable[date]) -> set[date]:
    """Month ends that `build_panel` could turn into a LASSO observation:
    every offset in `_REQUIRED_OFFSETS` is also stored. Set membership,
    not a contiguity assumption — a store with holes is counted for what
    it is."""
    have = set(stored)
    out: set[date] = set()
    for m in have:
        if all(_shift(m, k) in have for k in _REQUIRED_OFFSETS if k):
            out.add(m)
    return out


def _shift(m: date, k: int) -> date:
    if k > 0:
        cur = m
        for _ in range(k):
            cur = _next_month_end(cur)
        return cur
    return _prev_month_end(m, -k)


def load_price_month_ends(db: Any, tickers: list[str]) -> dict[str, set[date]]:
    """`{ticker: {month_end, ...}}` for the tickers in scope, one query."""
    out: dict[str, set[date]] = {t: set() for t in tickers}
    if not tickers:
        return out
    rows = db.execute(
        select(PriceMonthEnd.ticker, PriceMonthEnd.month_end)
        .where(PriceMonthEnd.ticker.in_(tickers))
    ).all()
    for t, me in rows:
        if t in out and me is not None:
            out[t].add(me)
    return out


def build_coverage(
    tickers: list[str],
    price_months: dict[str, set[date]],
    *,
    depth_target_months: int = DEFAULT_DEPTH_TARGET_MONTHS,
    min_months: int = MIN_MONTHS,
    min_obs: int = MIN_OBS,
) -> Coverage:
    """Pure: turn stored month ends into the price-store report. No IO, so
    the arithmetic is testable without a provider or a database."""
    per: list[TickerCoverage] = []
    all_evaluable: set[date] = set()
    for t in tickers:
        stored = price_months.get(t) or set()
        ordered = sorted(stored)
        ok = evaluable_months(ordered)
        all_evaluable |= ok
        gaps = 0
        reason: str | None = None
        if ordered:
            gaps = sum(1 for m in _iter_month_ends(ordered[0], ordered[-1]) if m not in stored)
        else:
            reason = "no month ends stored for this ticker"
        if reason is None and len(ok) < depth_target_months:
            reason = (
                f"{len(ordered)} month ends stored ({gaps} missing inside the span) "
                f"yield {len(ok)} price-evaluable months; {depth_target_months} is this "
                f"script's per-name target. {depth_target_months + LEAD_MONTHS} contiguous "
                f"month ends reach it."
            )
        per.append(TickerCoverage(
            ticker=t, months_stored=len(ordered),
            first_month=ordered[0] if ordered else None,
            last_month=ordered[-1] if ordered else None,
            gap_months=gaps, price_evaluable=len(ok),
            depth_target_met=len(ok) >= depth_target_months,
            reason=reason,
        ))
    return Coverage(
        depth_target_months=depth_target_months, min_months=min_months, min_obs=min_obs,
        tickers=tuple(per),
        months_stored=sum(c.months_stored for c in per),
        price_evaluable=sum(c.price_evaluable for c in per),
        price_evaluable_months=len(all_evaluable),
        names_at_depth_target=sum(1 for c in per if c.depth_target_met),
    )


def measure_coverage(
    tickers: list[str], *, depth_target_months: int = DEFAULT_DEPTH_TARGET_MONTHS,
    min_months: int = MIN_MONTHS, min_obs: int = MIN_OBS,
) -> Coverage:
    with SessionLocal() as db:
        scorecard_pit._ensure_tables(db)
        prices = load_price_month_ends(db, tickers)
    return build_coverage(
        tickers, prices, depth_target_months=depth_target_months,
        min_months=min_months, min_obs=min_obs,
    )


# ---------------------------------------------------------------------------
# The evaluation read-out (the GATE, asked of its own producer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationReadout:
    """What `scorecard_evaluation` actually yields right now.

    Not a re-derivation: `build_panel`, `_quintile` and `_lasso` are the
    production code paths, called read-only (no run row is created, no row
    is written, no provider is touched). `n_obs` and `n_months` are the
    two quantities the verdict gate compares against `min_obs` and
    `min_months`, and both are PANEL-WIDE totals.
    """
    verdict: str
    n_obs: int
    n_months: int
    min_obs: int
    min_months: int
    panel_rows: int
    n_skipped_rows: int
    n_skipped_by_reason: dict[str, int]
    reasons: tuple[str, ...]

    @property
    def meets_min_obs(self) -> bool:
        return self.n_obs >= self.min_obs

    @property
    def meets_min_months(self) -> bool:
        return self.n_months >= self.min_months

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "n_obs": self.n_obs,
            "n_months": self.n_months,
            "min_obs": self.min_obs,
            "min_months": self.min_months,
            "meets_min_obs": self.meets_min_obs,
            "meets_min_months": self.meets_min_months,
            "panel_rows": self.panel_rows,
            "n_skipped_rows": self.n_skipped_rows,
            "n_skipped_by_reason": dict(self.n_skipped_by_reason),
            "reasons": list(self.reasons),
        }


def read_evaluation(
    version_key: str, *, min_months: int = MIN_MONTHS, min_obs: int = MIN_OBS,
) -> tuple[EvaluationReadout | None, str | None]:
    """Run the evaluation's own panel + LASSO and report what it sees.

    Returns `(readout, None)` or `(None, reason)` — a failure is named,
    never rendered as a zero. READ-ONLY: unlike `run_evaluation` this
    creates no `scorecard_runs` row and persists no result.

    The read-out describes the WHOLE panel for `version_key`, not the
    `--tickers` subset, because the gate is panel-wide.
    """
    try:
        with SessionLocal() as db:
            scorecard_evaluation._ensure_tables(db)
            panel = scorecard_evaluation.build_panel(version_key, db=db)
        min_leg = int(settings.scorecard_min_leg_n)
        min_coverage = float(settings.scorecard_min_coverage)
        quintile = scorecard_evaluation._quintile(panel, min_leg=min_leg, min_coverage=min_coverage)
        lasso = scorecard_evaluation._lasso(
            panel, quintile, min_months=min_months, min_obs=min_obs, min_coverage=min_coverage,
        )
    except Exception as exc:  # a failed read-out must not lose the fetch report
        reason = f"{type(exc).__name__}: {safe_exc(exc)}"
        log.warning("backfill_month_end_prices: evaluation read-out failed: %s", reason)
        return None, reason
    return EvaluationReadout(
        verdict=str(lasso.get("verdict") or ""),
        n_obs=int(lasso.get("n_obs") or 0),
        n_months=int(lasso.get("n_months") or 0),
        min_obs=min_obs, min_months=min_months,
        panel_rows=int(panel.get("n_rows") or 0),
        n_skipped_rows=int(lasso.get("n_skipped_rows") or 0),
        n_skipped_by_reason=dict(lasso.get("n_skipped_by_reason") or {}),
        reasons=tuple(lasso.get("reasons") or ()),
    ), None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """What a run would do, before any provider call is made."""
    fetch: tuple[str, ...]
    satisfied: tuple[str, ...]
    exhausted: tuple[str, ...]
    over_budget: tuple[str, ...]
    exhausted_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def needs_work(self) -> int:
        """Names a further run would still pay for: this run's selection
        plus what the budget cut. Zero is the operator's stop signal."""
        return len(self.fetch) + len(self.over_budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetch": list(self.fetch), "n_fetch": len(self.fetch),
            "satisfied": list(self.satisfied), "n_satisfied": len(self.satisfied),
            "exhausted": list(self.exhausted), "n_exhausted": len(self.exhausted),
            "exhausted_reasons": dict(self.exhausted_reasons),
            "over_budget": list(self.over_budget), "n_over_budget": len(self.over_budget),
            "needs_work": self.needs_work,
        }


def plan_backfill(
    tickers: list[str],
    coverage: Coverage,
    *,
    max_tickers: int,
    days: int = DEFAULT_DAYS,
    attempts: dict[str, Attempt] | None = None,
    include_satisfied: bool = False,
    retry_exhausted: bool = False,
) -> Plan:
    """Split the universe into fetch / satisfied / exhausted / cut-by-budget.

    Two rules retire a name, and BOTH are needed for a re-run to advance:

    `satisfied`  it reached the per-name price-depth target, so a fetch
                 would buy nothing.
    `exhausted`  the provider was already asked at >= `days` and could not
                 fill it. Without this rule a name that CANNOT reach the
                 target (a recent listing, a delisted name, a dead provider
                 chain) is re-selected on every run, consumes the budget
                 forever and blocks the rest of the universe — a paid
                 infinite loop, not a slow one.

    Ticker order is preserved (the caller sorts), so a re-run with the same
    budget deterministically picks up where the last one stopped.
    `over_budget` and `exhausted` are both COUNTED and named, never
    silently dropped.
    """
    if max_tickers < 0:
        raise ValueError("max_tickers cannot be negative")
    attempts = attempts or {}
    candidates: list[str] = []
    satisfied: list[str] = []
    exhausted: list[str] = []
    reasons: dict[str, str] = {}
    by_ticker = coverage.by_ticker()
    for t in tickers:
        cov = by_ticker.get(t)
        if cov is not None and cov.depth_target_met and not include_satisfied:
            satisfied.append(t)
            continue
        prior = attempts.get(t)
        if prior is not None and prior.covers(days) and not include_satisfied and not retry_exhausted:
            exhausted.append(t)
            reasons[t] = prior.retirement_reason()
            continue
        candidates.append(t)
    return Plan(
        fetch=tuple(candidates[:max_tickers]),
        satisfied=tuple(satisfied),
        exhausted=tuple(exhausted),
        over_budget=tuple(candidates[max_tickers:]),
        exhausted_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# The backfill
# ---------------------------------------------------------------------------

@dataclass
class FetchTotals:
    months: int = 0
    written: int = 0
    incomplete_trailing: int = 0
    errors: int = 0
    unavailable: int = 0
    completed: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "months": self.months, "written": self.written,
            "incomplete_trailing_months": self.incomplete_trailing,
            "errors": self.errors, "unavailable": self.unavailable,
            "completed": self.completed,
            "failed": list(self.failed),
        }


def fetch_tickers(tickers: Iterable[str], *, days: int) -> tuple[FetchTotals, bool]:
    """Deep-sync each ticker, committing per ticker. Returns
    `(totals, interrupted)`.

    One bad ticker never stops the rest; a provider outage
    (`PriceSeriesUnavailable`) is recorded as an error with its reason and
    NOT as a ticker with zero months — `months: 0` must only ever mean a
    listing too young for one complete month.

    Every ticker asked for is written to the attempt ledger, whatever the
    outcome, and committed immediately. That record is what stops the next
    run re-buying a name the provider already declined.
    """
    totals = FetchTotals()
    done = 0
    for t in tickers:
        try:
            res = scorecard_pit.sync_price_month_ends(t, days=days)
        except KeyboardInterrupt:
            totals.completed = done
            return totals, True
        except scorecard_pit.PriceSeriesUnavailable as exc:
            totals.errors += 1
            totals.unavailable += 1
            reason = f"price series unavailable: {safe_exc(exc)}"
            totals.failed.append({"ticker": t, "reason": reason})
            record_attempt(t, days=days, months_returned=None,
                           outcome=OUTCOME_UNAVAILABLE, reason=reason)
            log.warning("backfill_month_end_prices: no price series for %s: %s", t, safe_exc(exc))
            continue
        except Exception as exc:  # one ticker must not abort the run
            totals.errors += 1
            reason = f"{type(exc).__name__}: {safe_exc(exc)}"
            totals.failed.append({"ticker": t, "reason": reason})
            record_attempt(t, days=days, months_returned=None,
                           outcome=OUTCOME_ERROR, reason=reason)
            log.warning("backfill_month_end_prices: sync failed for %s: %s", t, safe_exc(exc))
            continue
        totals.months += res["months"]
        totals.written += res["written"]
        totals.incomplete_trailing += res["skipped"]
        record_attempt(
            t, days=days, months_returned=res["months"], outcome=OUTCOME_FETCHED,
            reason=f"{res['months']} complete month ends returned, {res['written']} rows written",
        )
        done += 1
    totals.completed = done
    return totals, False


def _delta(
    before: Coverage, after: Coverage,
    ev_before: EvaluationReadout | None, ev_after: EvaluationReadout | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "months_stored": after.months_stored - before.months_stored,
        "price_evaluable": after.price_evaluable - before.price_evaluable,
        "price_evaluable_months": after.price_evaluable_months - before.price_evaluable_months,
        "names_at_depth_target": after.names_at_depth_target - before.names_at_depth_target,
    }
    if ev_before is None or ev_after is None:
        out["evaluation_n_obs"] = None
        out["evaluation_n_months"] = None
        out["evaluation_reason"] = "an evaluation read-out is missing, so its delta is unknown"
    else:
        out["evaluation_n_obs"] = ev_after.n_obs - ev_before.n_obs
        out["evaluation_n_months"] = ev_after.n_months - ev_before.n_months
    return out


def run(
    *,
    tickers: list[str] | None = None,
    days: int = DEFAULT_DAYS,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    dry_run: bool = False,
    include_satisfied: bool = False,
    retry_exhausted: bool = False,
    depth_target_months: int = DEFAULT_DEPTH_TARGET_MONTHS,
    min_months: int = MIN_MONTHS,
    min_obs: int = MIN_OBS,
) -> dict[str, Any]:
    """Measure, plan, (optionally) fetch, measure again. The return value
    is the whole report — `main` only formats it."""
    version_key = scorecard_service.VERSION_KEY
    universe = sorted({m.ticker for m in scorecard_service.universe_members(tickers=tickers)})
    before = measure_coverage(
        universe, depth_target_months=depth_target_months, min_months=min_months, min_obs=min_obs,
    )
    with SessionLocal() as db:
        attempts = load_attempts(db, universe)
    plan = plan_backfill(
        universe, before, max_tickers=max_tickers, days=days, attempts=attempts,
        include_satisfied=include_satisfied, retry_exhausted=retry_exhausted,
    )
    ev_before, ev_before_reason = read_evaluation(version_key, min_months=min_months, min_obs=min_obs)

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "days": days,
        "universe": {"n_tickers": len(universe), "source": "explicit --tickers" if tickers else "auto_analysis tier"},
        "budget": {"max_tickers": max_tickers, **plan.to_dict()},
        "coverage_before": before.to_dict(),
        "evaluation_before": ev_before.to_dict() if ev_before else None,
        "evaluation_before_reason": ev_before_reason,
    }
    if dry_run:
        # Nothing is fetched, so "after" would be a copy of "before" and
        # saying otherwise would be a lie. The delta is reported as None
        # with the reason instead of 0.
        report["fetched"] = None
        report["fetched_reason"] = "dry run: no provider call was made and no row was written"
        report["coverage_after"] = None
        report["coverage_after_reason"] = "dry run: coverage is unchanged from coverage_before"
        report["evaluation_after"] = None
        report["evaluation_after_reason"] = "dry run: the evaluation read-out is unchanged from evaluation_before"
        report["delta"] = None
        report["delta_reason"] = "dry run: nothing changed"
        report["partial"] = []
        report["interrupted"] = False
        return report

    totals, interrupted = fetch_tickers(plan.fetch, days=days)
    after = measure_coverage(
        universe, depth_target_months=depth_target_months, min_months=min_months, min_obs=min_obs,
    )
    ev_after, ev_after_reason = read_evaluation(version_key, min_months=min_months, min_obs=min_obs)
    touched = set(plan.fetch)
    after_by_ticker = after.by_ticker()
    partial = [
        c.to_dict() for t in sorted(touched)
        if (c := after_by_ticker.get(t)) is not None and not c.depth_target_met
    ]
    report["fetched"] = totals.to_dict()
    report["interrupted"] = interrupted
    report["partial"] = partial
    report["coverage_after"] = after.to_dict()
    report["evaluation_after"] = ev_after.to_dict() if ev_after else None
    report["evaluation_after_reason"] = ev_after_reason
    report["delta"] = _delta(before, after, ev_before, ev_after)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tickers(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return sorted({t.strip().upper() for t in raw.split(",") if t.strip()})


def _fmt_coverage(label: str, cov: dict[str, Any] | None, reason: str | None) -> list[str]:
    if cov is None:
        return [f"{label}: not measured ({reason or 'no reason recorded'})"]
    lo, hi = cov["months_stored_min"], cov["months_stored_max"]
    span = f"min {lo}, max {hi}" if lo is not None else (cov["months_stored_reason"] or "no reason recorded")
    return [
        f"{label} (price store — a CEILING, not the evaluation's gate):",
        f"  month ends stored     {cov['months_stored']} across {cov['n_tickers']} tickers ({span})",
        f"  price-evaluable       {cov['price_evaluable']} observations over "
        f"{cov['price_evaluable_months']} distinct months "
        f"(panel floors min_obs {cov['min_obs']}: "
        f"{'reachable' if cov['price_could_meet_min_obs'] else 'NOT reachable'}; "
        f"min_months {cov['min_months']}: "
        f"{'reachable' if cov['price_could_meet_min_months'] else 'NOT reachable'})",
        f"  at per-name target    {cov['names_at_depth_target']}/{cov['n_tickers']} names have "
        f"{cov['depth_target_months']}+ price-evaluable months (this script's budget rule)",
    ]


def _fmt_evaluation(label: str, ev: dict[str, Any] | None, reason: str | None) -> list[str]:
    if ev is None:
        return [f"{label}: not measured ({reason or 'no reason recorded'})"]
    dropped = ev["n_skipped_by_reason"]
    drop_text = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())) if dropped else "none"
    return [
        f"{label} (scorecard_evaluation's OWN panel — this is the gate):",
        f"  verdict               {ev['verdict']}",
        f"  usable observations   {ev['n_obs']} of {ev['min_obs']} required "
        f"({'met' if ev['meets_min_obs'] else 'not met'}) — panel-wide total, not per name",
        f"  distinct months       {ev['n_months']} of {ev['min_months']} required "
        f"({'met' if ev['meets_min_months'] else 'not met'}) — panel-wide distinct month ends",
        f"  panel rows            {ev['panel_rows']} built, {ev['n_skipped_rows']} dropped before the LASSO",
        f"  dropped because       {drop_text}",
    ]


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    head = "month-end price backfill" + (" — DRY RUN (nothing fetched, nothing written)" if report["dry_run"] else "")
    lines.append(head)
    lines.append(f"  universe              {report['universe']['n_tickers']} tickers "
                 f"({report['universe']['source']})")
    lines.append(f"  window                {report['days']} trading days requested per ticker")
    budget = report["budget"]
    lines.append(f"  budget                {budget['max_tickers']} tickers max — "
                 f"{budget['n_fetch']} selected, {budget['n_satisfied']} already at the per-name target "
                 f"(no call), {budget['n_exhausted']} exhausted (no call), "
                 f"{budget['n_over_budget']} left for a later run")
    lines.append(f"  names still to fetch  {budget['needs_work']} "
                 "(re-run the same command until this is 0)")
    if budget["over_budget"]:
        shown = ", ".join(budget["over_budget"][:20])
        more = "" if len(budget["over_budget"]) <= 20 else f", … (+{len(budget['over_budget']) - 20} more)"
        lines.append(f"  skipped for budget    {shown}{more}")
    for t in budget["exhausted"][:20]:
        lines.append(f"  EXHAUSTED {t}: {budget['exhausted_reasons'].get(t) or 'no reason recorded'}")
    if len(budget["exhausted"]) > 20:
        lines.append(f"  … (+{len(budget['exhausted']) - 20} more exhausted names, see --json)")
    lines.append("")
    lines.extend(_fmt_coverage("coverage before", report["coverage_before"], None))
    lines.append("")
    fetched = report.get("fetched")
    if fetched is None:
        lines.append(f"fetch: skipped ({report.get('fetched_reason') or 'no reason recorded'})")
        if budget["fetch"]:
            lines.append("  would fetch           " + ", ".join(budget["fetch"]))
    else:
        lines.append(f"fetch: {fetched['months']} month ends seen, {fetched['written']} rows written, "
                     f"{fetched['incomplete_trailing_months']} incomplete trailing months skipped, "
                     f"{fetched['errors']} errors ({fetched['unavailable']} provider outages)")
        for f in fetched["failed"]:
            lines.append(f"  FAILED {f['ticker']}: {f['reason']}")
        if report.get("interrupted"):
            lines.append(f"  INTERRUPTED after {fetched['completed']} tickers — "
                         "completed tickers are committed; re-run the same command to resume")
    for p in report.get("partial") or []:
        lines.append(f"  PARTIAL {p['ticker']}: {p['reason'] or 'no reason recorded'}")
    lines.append("")
    lines.extend(_fmt_coverage("coverage after", report.get("coverage_after"), report.get("coverage_after_reason")))
    delta = report.get("delta")
    lines.append("")
    if delta is None:
        lines.append(f"delta: none ({report.get('delta_reason') or 'no reason recorded'})")
    else:
        # Signed, not "+"-prefixed: a delta is arithmetic, and a shrinking
        # count (a corrected close, a purge) must not print as a gain.
        lines.append(f"delta: {delta['months_stored']:+d} month ends, "
                     f"{delta['price_evaluable']:+d} price-evaluable observations, "
                     f"{delta['price_evaluable_months']:+d} price-evaluable months, "
                     f"{delta['names_at_depth_target']:+d} names at the per-name target")
        if delta.get("evaluation_n_obs") is None:
            lines.append(f"  evaluation delta    unknown ({delta.get('evaluation_reason') or 'no reason recorded'})")
        else:
            lines.append(f"  evaluation delta    {delta['evaluation_n_obs']:+d} usable observations, "
                         f"{delta['evaluation_n_months']:+d} distinct months")
    lines.append("")
    ev = report.get("evaluation_after")
    ev_reason = report.get("evaluation_after_reason")
    if ev is None and report["dry_run"]:
        ev, ev_reason = report.get("evaluation_before"), report.get("evaluation_before_reason")
    lines.extend(_fmt_evaluation("EVALUATION READ-OUT", ev, ev_reason))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.backfill_month_end_prices",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers. Default: the auto_analysis scorecard universe.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Trading days of history to request per ticker (default {DEFAULT_DAYS}; its own "
                             "provider-cache entry, so the app's 252-day series is untouched). Raising it "
                             "re-opens names retired as exhausted at a shallower depth.")
    parser.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
                        help=f"Budget: at most this many PAID fetches (default {DEFAULT_MAX_TICKERS}). "
                             "Everything cut is counted and named.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch nothing, write nothing; report current coverage and what would be fetched.")
    parser.add_argument("--include-satisfied", action="store_true",
                        help="Also re-sync names already at the per-name target or retired as exhausted "
                             "(they are skipped by default, which is what makes a re-run advance).")
    parser.add_argument("--retry-exhausted", action="store_true",
                        help="Re-ask for names retired as exhausted at this depth — use after a provider "
                             "outage is resolved. Names already at the target are still skipped.")
    parser.add_argument("--depth-target-months", type=int, default=DEFAULT_DEPTH_TARGET_MONTHS,
                        help=f"Price-evaluable months at which this script stops paying for a name "
                             f"(default {DEFAULT_DEPTH_TARGET_MONTHS}). A BUDGET rule: the evaluation's own "
                             "floors are panel-wide and are reported separately.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of text.")
    args = parser.parse_args(argv)

    if args.max_tickers < 0:
        parser.error("--max-tickers cannot be negative")
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.depth_target_months < 1:
        parser.error("--depth-target-months must be at least 1")

    report = run(
        tickers=_parse_tickers(args.tickers), days=args.days, max_tickers=args.max_tickers,
        dry_run=args.dry_run, include_satisfied=args.include_satisfied,
        retry_exhausted=args.retry_exhausted, depth_target_months=args.depth_target_months,
    )
    if args.json:
        print(json.dumps(report, default=str, sort_keys=True))
    else:
        print(format_report(report))
    if report.get("interrupted"):
        return 130
    return 1 if (report.get("fetched") or {}).get("errors") else 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
