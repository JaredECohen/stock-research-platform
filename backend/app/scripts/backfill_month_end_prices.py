"""One-time deep backfill of `price_month_ends`, so the scorecard
evaluation can stop saying `insufficient_data`.

WHY THIS EXISTS
---------------
`scorecard_evaluation` draws no verdict until the panel has
`DEFAULT_MIN_MONTHS` (24) months per name and `DEFAULT_MIN_OBS` (2000)
observations in total. Its month-end price store is fed from the app's
rolling 252-day window (`scorecard_service._SERIES_WINDOW_DAYS`), and
`pit_prepare` adds exactly ONE month per run going forward — so the
evaluation is mute for well over a year after launch (that is what
`scorecard_evaluation.PRICE_DEPTH_NOTE` says, and it is honest, not
broken). One deep fetch per name converts a large amount of finished work
from silent to useful.

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

    # 3. Keep re-running the SAME command until `needs_work` is 0.
    #    Names that already clear the minimum are skipped without a fetch,
    #    so each run advances to the next slice of the universe.

Safe to interrupt: every ticker commits on its own (`sync_price_month_ends`
opens and closes a session per ticker), a Ctrl-C during a fetch is caught
and the run stops cleanly, and the partial report — including the coverage
the finished tickers did buy — is still printed, with exit status 130.
Resume is just the same command again.

Idempotent by construction. The row-level upsert in
`scorecard_pit.sync_price_month_ends` leaves an unchanged `(ticker,
month_end)` row alone, so nothing is duplicated and an already-stored
month end is never rewritten; on top of that, this script skips a whole
ticker that already clears the minimum, so the paid call is not made
either.

THIS IS AN OPERATOR TOOL. It is never called from a monitoring loop, a
route, or application code — `test_backfill_month_end_prices` asserts
that nothing outside `app/scripts` and `app/tests` imports it. The
recurring, one-month-at-a-time equivalent is the worker's
`pit_prepare`; the shallow CLI equivalent is `app.scripts.scorecard_backfill
--prices`, which exists for the 252-day window and does not budget, plan
or report coverage.

WHAT THE COVERAGE NUMBERS MEAN
------------------------------
`build_panel` turns a month-end score row into a usable observation only
when four month-end closes are stored for that ticker: `M-12` and `M-1`
(momentum_12_1 and reversal_1m, both LASSO controls), `M` itself, and
`M+1` (the forward return). So a contiguous run of `L` stored month ends
yields `L - 13` evaluable months, and clearing the 24-month floor needs
37 contiguous month ends — roughly 780 trading days, which is why
`--days` defaults to 1260.

Two counts are reported, and they are NOT the same number:

  price_evaluable   what the PRICE store can support — the ceiling this
                    script raises.
  panel             the intersection with the month-end `scorecard_scores`
                    rows that actually exist, from succeeded runs. This is
                    what the evaluation will really see.

Prices alone do not produce a verdict: a month also needs a scored row
for that month end (`scorecard_queue.KIND_BACKFILL` runs). Reporting only
the ceiling would overstate what this backfill achieved, so both are
printed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select

from ..agents.log_safety import safe_exc
from ..database import SessionLocal
from ..models import PriceMonthEnd, ScorecardRun, ScorecardScore
from ..services import scorecard_evaluation, scorecard_pit, scorecard_queue, scorecard_service

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

# ~5 calendar years of trading days. 37 contiguous month ends (the floor
# for 24 evaluable months) is about 780; 1260 leaves room for holidays,
# halts and a name that listed mid-window.
DEFAULT_DAYS = 1260
# Deliberately small. An accidental run costs a handful of provider calls;
# the report names every ticker the budget cut.
DEFAULT_MAX_TICKERS = 5


# ---------------------------------------------------------------------------
# Coverage
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
    `reason` rather than standing in as 0."""
    ticker: str
    months_stored: int
    first_month: date | None
    last_month: date | None
    gap_months: int
    price_evaluable: int
    panel_observations: int
    clears_minimum: bool
    panel_clears_minimum: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["first_month"] = self.first_month.isoformat() if self.first_month else None
        d["last_month"] = self.last_month.isoformat() if self.last_month else None
        return d


@dataclass(frozen=True)
class Coverage:
    """Universe-level coverage. `price_*` is what the price store can
    support; `panel_*` is the intersection with the month-end score rows
    that exist, which is what the evaluation actually reads."""
    min_months: int
    min_obs: int
    tickers: tuple[TickerCoverage, ...]
    months_stored: int
    price_evaluable: int
    price_names_clearing: int
    panel_observations: int
    panel_names_clearing: int
    scored_months: int
    scored_rows: int

    @property
    def price_meets_min_obs(self) -> bool:
        return self.price_evaluable >= self.min_obs

    @property
    def panel_meets_min_obs(self) -> bool:
        return self.panel_observations >= self.min_obs

    def by_ticker(self) -> dict[str, TickerCoverage]:
        return {c.ticker: c for c in self.tickers}

    def to_dict(self) -> dict[str, Any]:
        stored = [c.months_stored for c in self.tickers]
        return {
            "min_months": self.min_months,
            "min_obs": self.min_obs,
            "n_tickers": len(self.tickers),
            "months_stored": self.months_stored,
            "months_stored_min": min(stored) if stored else None,
            "months_stored_max": max(stored) if stored else None,
            "months_stored_reason": None if stored else "no tickers in scope",
            "price_evaluable": self.price_evaluable,
            "price_names_clearing": self.price_names_clearing,
            "price_meets_min_obs": self.price_meets_min_obs,
            "panel_observations": self.panel_observations,
            "panel_names_clearing": self.panel_names_clearing,
            "panel_meets_min_obs": self.panel_meets_min_obs,
            "scored_month_ends": self.scored_months,
            "scored_rows": self.scored_rows,
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


def load_scored_month_ends(db: Any, tickers: list[str], *, version_key: str) -> dict[str, set[date]]:
    """`{ticker: {as_of, ...}}` for month-end score rows from SUCCEEDED
    runs — the same filter `build_panel` applies. These are the months the
    evaluation can pair a price with."""
    out: dict[str, set[date]] = {t: set() for t in tickers}
    if not tickers:
        return out
    rows = db.execute(
        select(ScorecardScore.ticker, ScorecardScore.as_of)
        .join(ScorecardRun, ScorecardRun.id == ScorecardScore.run_id)
        .where(
            ScorecardScore.version_key == version_key,
            ScorecardScore.ticker.in_(tickers),
            ScorecardScore.is_month_end.is_(True),
            ScorecardRun.status == scorecard_queue.STATUS_SUCCEEDED,
        )
    ).all()
    for t, as_of in rows:
        if t in out and as_of is not None:
            out[t].add(as_of)
    return out


def build_coverage(
    tickers: list[str],
    price_months: dict[str, set[date]],
    scored_months: dict[str, set[date]],
    *,
    min_months: int = MIN_MONTHS,
    min_obs: int = MIN_OBS,
) -> Coverage:
    """Pure: turn stored month ends into the coverage report. No IO, so the
    arithmetic is testable without a provider or a database."""
    per: list[TickerCoverage] = []
    total_panel = 0
    for t in tickers:
        stored = price_months.get(t) or set()
        scored = scored_months.get(t) or set()
        ordered = sorted(stored)
        ok = evaluable_months(ordered)
        panel = ok & scored
        gaps = 0
        reason: str | None = None
        if ordered:
            gaps = sum(1 for m in _iter_month_ends(ordered[0], ordered[-1]) if m not in stored)
        else:
            reason = "no month ends stored for this ticker"
        if reason is None and len(ok) < min_months:
            reason = (
                f"{len(ordered)} month ends stored ({gaps} missing inside the span) "
                f"yield {len(ok)} evaluable months; {min_months} required. "
                f"{min_months + LEAD_MONTHS} contiguous month ends clear it."
            )
        total_panel += len(panel)
        per.append(TickerCoverage(
            ticker=t, months_stored=len(ordered),
            first_month=ordered[0] if ordered else None,
            last_month=ordered[-1] if ordered else None,
            gap_months=gaps, price_evaluable=len(ok), panel_observations=len(panel),
            clears_minimum=len(ok) >= min_months, panel_clears_minimum=len(panel) >= min_months,
            reason=reason,
        ))
    scored_all = {m for t in tickers for m in (scored_months.get(t) or set())}
    return Coverage(
        min_months=min_months, min_obs=min_obs, tickers=tuple(per),
        months_stored=sum(c.months_stored for c in per),
        price_evaluable=sum(c.price_evaluable for c in per),
        price_names_clearing=sum(1 for c in per if c.clears_minimum),
        panel_observations=total_panel,
        panel_names_clearing=sum(1 for c in per if c.panel_clears_minimum),
        scored_months=len(scored_all),
        scored_rows=sum(len(scored_months.get(t) or set()) for t in tickers),
    )


def measure_coverage(
    tickers: list[str], *, version_key: str, min_months: int = MIN_MONTHS, min_obs: int = MIN_OBS,
) -> Coverage:
    with SessionLocal() as db:
        scorecard_pit._ensure_tables(db)
        scorecard_evaluation._ensure_tables(db)
        prices = load_price_month_ends(db, tickers)
        scored = load_scored_month_ends(db, tickers, version_key=version_key)
    return build_coverage(tickers, prices, scored, min_months=min_months, min_obs=min_obs)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """What a run would do, before any provider call is made."""
    fetch: tuple[str, ...]
    satisfied: tuple[str, ...]
    over_budget: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetch": list(self.fetch), "n_fetch": len(self.fetch),
            "satisfied": list(self.satisfied), "n_satisfied": len(self.satisfied),
            "over_budget": list(self.over_budget), "n_over_budget": len(self.over_budget),
        }


def plan_backfill(
    tickers: list[str], coverage: Coverage, *, max_tickers: int, include_satisfied: bool = False,
) -> Plan:
    """Split the universe into fetch / already-satisfied / cut-by-budget.

    Ticker order is preserved (the caller sorts), so a re-run with the same
    budget deterministically picks up where the last one stopped: the names
    it finished now fall into `satisfied` and drop out of the candidate
    list. `over_budget` is COUNTED and named, never silently dropped.
    """
    if max_tickers < 0:
        raise ValueError("max_tickers cannot be negative")
    candidates: list[str] = []
    satisfied: list[str] = []
    by_ticker = coverage.by_ticker()
    for t in tickers:
        cov = by_ticker.get(t)
        if cov is not None and cov.clears_minimum and not include_satisfied:
            satisfied.append(t)
        else:
            candidates.append(t)
    return Plan(
        fetch=tuple(candidates[:max_tickers]),
        satisfied=tuple(satisfied),
        over_budget=tuple(candidates[max_tickers:]),
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
            totals.failed.append({"ticker": t, "reason": f"price series unavailable: {safe_exc(exc)}"})
            log.warning("backfill_month_end_prices: no price series for %s: %s", t, safe_exc(exc))
            continue
        except Exception as exc:  # one ticker must not abort the run
            totals.errors += 1
            totals.failed.append({"ticker": t, "reason": f"{type(exc).__name__}: {safe_exc(exc)}"})
            log.warning("backfill_month_end_prices: sync failed for %s: %s", t, safe_exc(exc))
            continue
        totals.months += res["months"]
        totals.written += res["written"]
        totals.incomplete_trailing += res["skipped"]
        done += 1
    totals.completed = done
    return totals, False


def run(
    *,
    tickers: list[str] | None = None,
    days: int = DEFAULT_DAYS,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    dry_run: bool = False,
    include_satisfied: bool = False,
    min_months: int = MIN_MONTHS,
    min_obs: int = MIN_OBS,
) -> dict[str, Any]:
    """Measure, plan, (optionally) fetch, measure again. The return value
    is the whole report — `main` only formats it."""
    version_key = scorecard_service.VERSION_KEY
    universe = sorted({m.ticker for m in scorecard_service.universe_members(tickers=tickers)})
    before = measure_coverage(universe, version_key=version_key, min_months=min_months, min_obs=min_obs)
    plan = plan_backfill(universe, before, max_tickers=max_tickers, include_satisfied=include_satisfied)

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "days": days,
        "universe": {"n_tickers": len(universe), "source": "explicit --tickers" if tickers else "auto_analysis tier"},
        "budget": {"max_tickers": max_tickers, **plan.to_dict()},
        "coverage_before": before.to_dict(),
    }
    if dry_run:
        # Nothing is fetched, so "after" would be a copy of "before" and
        # saying otherwise would be a lie. The delta is reported as None
        # with the reason instead of 0.
        report["fetched"] = None
        report["fetched_reason"] = "dry run: no provider call was made and no row was written"
        report["coverage_after"] = None
        report["coverage_after_reason"] = "dry run: coverage is unchanged from coverage_before"
        report["delta"] = None
        report["delta_reason"] = "dry run: nothing changed"
        report["partial"] = []
        report["interrupted"] = False
        return report

    totals, interrupted = fetch_tickers(plan.fetch, days=days)
    after = measure_coverage(universe, version_key=version_key, min_months=min_months, min_obs=min_obs)
    touched = set(plan.fetch)
    after_by_ticker = after.by_ticker()
    partial = [
        c.to_dict() for t in sorted(touched)
        if (c := after_by_ticker.get(t)) is not None and not c.clears_minimum
    ]
    report["fetched"] = totals.to_dict()
    report["interrupted"] = interrupted
    report["partial"] = partial
    report["coverage_after"] = after.to_dict()
    report["delta"] = {
        "months_stored": after.months_stored - before.months_stored,
        "price_evaluable": after.price_evaluable - before.price_evaluable,
        "price_names_clearing": after.price_names_clearing - before.price_names_clearing,
        "panel_observations": after.panel_observations - before.panel_observations,
        "panel_names_clearing": after.panel_names_clearing - before.panel_names_clearing,
    }
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
        f"{label}:",
        f"  month ends stored     {cov['months_stored']} across {cov['n_tickers']} tickers ({span})",
        f"  price-evaluable       {cov['price_evaluable']} observations; "
        f"{cov['price_names_clearing']}/{cov['n_tickers']} names clear {cov['min_months']} months "
        f"(min_obs {cov['min_obs']}: {'met' if cov['price_meets_min_obs'] else 'not met'})",
        f"  panel (scored+priced) {cov['panel_observations']} observations; "
        f"{cov['panel_names_clearing']}/{cov['n_tickers']} names clear {cov['min_months']} months "
        f"(min_obs {cov['min_obs']}: {'met' if cov['panel_meets_min_obs'] else 'not met'})",
        f"  scored month ends     {cov['scored_month_ends']} months / {cov['scored_rows']} rows "
        f"(the other half of an observation; prices alone draw no verdict)",
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
                 f"{budget['n_fetch']} selected, {budget['n_satisfied']} already clear the minimum "
                 f"(no call), {budget['n_over_budget']} left for a later run")
    if budget["over_budget"]:
        shown = ", ".join(budget["over_budget"][:20])
        more = "" if len(budget["over_budget"]) <= 20 else f", … (+{len(budget['over_budget']) - 20} more)"
        lines.append(f"  skipped for budget    {shown}{more}")
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
                     f"{delta['price_names_clearing']:+d} names clearing on price depth, "
                     f"{delta['panel_observations']:+d} panel observations, "
                     f"{delta['panel_names_clearing']:+d} names clearing in the panel")
    after = report.get("coverage_after") or report["coverage_before"]
    lines.append(f"NAMES CLEARING THE EVALUATION MINIMUMS: {after['panel_names_clearing']} of "
                 f"{after['n_tickers']} (price depth alone would allow {after['price_names_clearing']})")
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
                             "provider-cache entry, so the app's 252-day series is untouched).")
    parser.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
                        help=f"Budget: at most this many PAID fetches (default {DEFAULT_MAX_TICKERS}). "
                             "Everything cut is counted and named.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch nothing, write nothing; report current coverage and what would be fetched.")
    parser.add_argument("--include-satisfied", action="store_true",
                        help="Also re-sync names that already clear the minimum (they are skipped by default, "
                             "which is what makes a re-run advance instead of repeating).")
    parser.add_argument("--min-months", type=int, default=MIN_MONTHS,
                        help=f"Evaluable months a name needs to count as done (default {MIN_MONTHS}, the "
                             "evaluation's own floor).")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of text.")
    args = parser.parse_args(argv)

    if args.max_tickers < 0:
        parser.error("--max-tickers cannot be negative")
    if args.days < 1:
        parser.error("--days must be at least 1")

    report = run(
        tickers=_parse_tickers(args.tickers), days=args.days, max_tickers=args.max_tickers,
        dry_run=args.dry_run, include_satisfied=args.include_satisfied, min_months=args.min_months,
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
