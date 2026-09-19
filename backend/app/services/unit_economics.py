"""What a metered operation actually costs, read out of the logs we already keep.

The FEAT-002 allowances and the $29.99 Pro price were chosen before anything
measured the marginal model spend of the operations they meter. This module
answers the question from data that already exists — `llm_call_logs` (one row
per real provider call, tagged with `feature` and `run_id` by
`agents.llm.llm_call_context`) priced with `services.llm_metrics.estimate_cost_usd`
— and expresses the answer against `auth/features.py`, so the output reads
"a Free allowance costs X a month, a Pro allowance costs Y".

**It never calls a provider or a model.** `scripts/audit_unit_costs.py` is the
tool that spends money to produce a sample (nightly-live only, see
`docs/economics/unit-costs-2026-09.md`); this module is the opposite — it
observes the traffic that already happened. The two are complementary: the
script measures a controlled action, this measures the fleet.

Three rules govern every number that leaves here:

1. **Median and p90, never a lone mean.** A memo run's cost is long-tailed
   (a sector with a chatty analyst roster, a retry after a refusal); a mean
   hides exactly the tail a plan limit has to survive.
2. **A thin sample is reported as insufficient, with the count.** Below
   `MIN_UNITS_FOR_MEDIAN` units there is no median; below
   `MIN_UNITS_FOR_P90` there is no p90. The field is `None` and the reason
   names the count and the threshold. A made-up unit cost is worse than none:
   it would be multiplied by an allowance and priced against.
3. **Nothing silently vanishes.** Calls that cannot be attributed to a unit,
   units whose model is missing from the price table, calls that came back
   without token counts, units whose calls reach back past the oldest row the
   scan read, and rows dropped by the scan cap are each counted and named. A
   zero never stands in for an unknown.
4. **A total says what it does not cover.** `dcf`, `comps` and `portfolio`
   spend real money on every call and have no monthly allowance to multiply,
   so no plan total here can be complete. They ride into every plan block as
   `unmeasured_terms`, the total goes null, and what *is* measured is
   reported next to it as the lower bound it is.

What one *unit* is differs per operation and is stated in the output
(`basis`, `basis_note`): a `research_run` is a `run_id` (~26 calls), a
`chart_commentary` is one call because `auth/features.py` documents exactly
one cheap-route call per request, and a `pm_chat` *turn* has no identifier in
the log at all — so its figure is per call and flagged as a floor for a turn,
never quietly presented as the per-turn cost.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import features
from ..database import SessionLocal
from ..models import LLMCallLog
from .llm_metrics import (
    estimate_cost_usd,
    price_source,
)

# --- knobs ------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 30
# Hard ceiling on rows read per request. The route is read-only and must not
# turn into a table scan as the log grows; `generated_at` is indexed, so the
# newest `MAX_ROWS_SCANNED` rows in the window come off the index. Whatever
# the cap drops is counted and reported (`scan.rows_dropped_by_cap`) together
# with the window the kept rows actually cover.
MAX_ROWS_SCANNED = 20_000
# How far below the oldest scanned row to look for calls of a run the sample
# already has. A run-grouped unit is only a unit if every one of its calls was
# read; these two bound the check that finds the ones that were not (see
# `_straddling_units`). A day is orders of magnitude longer than a memo run,
# and `boundary_check.exhaustive` reports whether it was enough rather than
# assuming it.
BOUNDARY_LOOKBACK = timedelta(days=1)
BOUNDARY_LOOKBACK_ROWS = 5_000
# A median over fewer than this many units is noise wearing a decimal point.
MIN_UNITS_FOR_MEDIAN = 20
# A p90 is a tail estimate and needs more than a median does: with 20 points
# the "p90" is the 18th, one bad sample from meaningless.
MIN_UNITS_FOR_P90 = 30

# Pro list price. Mirrors `scripts/audit_unit_costs.PRO_PRICE_USD` and the
# Stripe price set up in `docs/ops/feat-002-setup.md`; the go/no-go threshold
# (variable cost under half the price) is `docs/economics/unit-costs-2026-09.md` §3.
PRO_PRICE_USD = 29.99
PLAN_THRESHOLD_USD = {"pro": round(PRO_PRICE_USD * 0.5, 3), "free": 1.50}
PLANS = ("free", "pro")

BASIS_RUN_ID = "run_id"
BASIS_CALL = "call"


@dataclass(frozen=True)
class Operation:
    """One metered operation and how its calls group into billable units."""

    key: str
    log_feature: str          # `llm_call_logs.feature` written by llm_call_context
    unit: str                 # what one unit is, in words
    basis: str                # BASIS_RUN_ID | BASIS_CALL
    basis_note: str           # why that grouping is (or is not) the billable unit
    plan_feature: str | None  # the `auth/features.py` allowance this is charged against
    # True when `basis` cannot see the whole billable unit, so every figure
    # derived from it is a floor. Never silently: it rides into the output.
    understates_unit: bool = False


OPERATIONS: tuple[Operation, ...] = (
    Operation(
        key="research_run",
        log_feature="research_run",
        unit="one memo run (the full agent committee on a ticker)",
        basis=BASIS_RUN_ID,
        basis_note=(
            "regen_worker tags every call of a run with the job's run_id, so a unit is "
            "exact: all ~26 specialist/PM/risk calls of one memo, summed"
        ),
        plan_feature="research_run",
    ),
    Operation(
        key="pm_chat",
        log_feature="pm_chat",
        unit="one LLM call inside an Ask-the-PM turn",
        basis=BASIS_CALL,
        basis_note=(
            "no per-turn identifier is written to llm_call_logs: /api/chat sets only "
            "user_id and feature, and a turn issues classify_intent plus at least one "
            "answer call. The figure below is therefore PER CALL, and a turn costs at "
            "least that much — it is a floor, not the per-turn cost. Logging a turn id "
            "in llm_call_context is what would close the gap"
        ),
        plan_feature="pm_chat",
        understates_unit=True,
    ),
    Operation(
        key="chart_commentary",
        log_feature="chart_commentary",
        unit="one chart commentary",
        basis=BASIS_CALL,
        basis_note=(
            "auth/features.py and services/chart_commentary.py agree that a request is "
            "exactly one cheap-route call, so one call is one billable commentary"
        ),
        plan_feature="chart_commentary",
    ),
    Operation(
        key="industry_report",
        log_feature="industry_report",
        unit="one industry report edition",
        basis=BASIS_RUN_ID,
        basis_note=(
            "industry_report_worker tags each analyst call with the job run_id, so a unit "
            "is one generated edition"
        ),
        # `industry_analysis` gates *reading* history/changes; nothing meters
        # generation per user. The cost is fleet-fixed, not per subscriber.
        plan_feature=None,
    ),
)


# --- percentiles ------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest observed value at or above rank
    `ceil(q·n)`. No interpolation, deliberately — every figure this module
    reports is then a cost that an actual unit actually incurred, not an
    average of two of them. `q` is a fraction in [0, 1].

    Hand-checkable: for n=20 sorted values, q=0.5 → index 9 (the 10th value)
    and q=0.9 → index 17 (the 18th).
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


# --- scanning ---------------------------------------------------------------

@dataclass
class _Unit:
    """One billable unit under construction."""

    cost_usd: float = 0.0
    n_calls: int = 0
    problems: set[str] = field(default_factory=set)


# Why a unit is unusable. Keys are stable (they appear in the output), values
# explain them to whoever reads the report.
EXCLUSION_REASONS = {
    "unpriced_model": (
        "the model is in neither llm_metrics.MODEL_PRICES_PER_MTOK nor "
        "PROVIDER_PRICE_FALLBACK, so its calls would price at $0 — add the model to the "
        "price table and re-read"
    ),
    "missing_token_counts": (
        "a successful call reported 0 input and 0 output tokens, so its cost is unknown "
        "rather than zero (the provider response carried no usage block)"
    ),
    "partial_unit": (
        "at least one of the unit's calls is older than the oldest row read — the scan "
        "cap cut the sample short, or the window start did — so the calls that were read "
        "price only part of it. Excluded rather than reported as a cheap whole unit"
    ),
}


def _is_priced(provider: str, model: str) -> bool:
    return price_source(provider, model) != "unpriced"


# Why a figure that includes a provider-default rate is reported but not
# trusted. Distinct from `EXCLUSION_REASONS`: the unit stays in the sample
# (the default keeps it off $0), the report just names the model.
FALLBACK_PRICED_REASON = (
    "priced at the provider default because the model has no row in "
    "llm_metrics.MODEL_PRICES_PER_MTOK — every figure that includes it is a guess; "
    "add the model's list price and re-read"
)


def _scan(db: Session, *, since: datetime, until: datetime, max_rows: int,
          log_features: Iterable[str]) -> tuple[list[Any], dict[str, Any]]:
    """The newest `max_rows` tagged rows in the window, plus what was dropped.

    Two statements. The read is bounded — newest-first off the `generated_at`
    index, LIMIT `max_rows`. The COUNT is **not**: it is bounded by the
    window, not by the cap, so its cost grows with the rows in the window
    (and `llm_call_logs.feature` carries no index, so the feature filter is a
    scan of them). That is a deliberate trade: the COUNT is the only way the
    cap can report the rows it left behind instead of quietly under-reporting
    the sample, and a report whose sample size is a guess is worth nothing.
    `window_days` is capped at 90 by the route, which bounds it in practice.
    """
    wanted = sorted(set(log_features))
    where = (
        LLMCallLog.feature.in_(wanted),
        LLMCallLog.generated_at >= since,
        LLMCallLog.generated_at < until,
    )
    total = int(db.execute(
        select(func.count()).select_from(LLMCallLog).where(*where)
    ).scalar() or 0)
    rows = list(db.execute(
        select(
            LLMCallLog.feature, LLMCallLog.run_id, LLMCallLog.provider, LLMCallLog.model,
            LLMCallLog.tokens_in, LLMCallLog.tokens_out, LLMCallLog.success,
            LLMCallLog.generated_at,
            LLMCallLog.cache_read_tokens, LLMCallLog.cache_write_tokens,
        )
        .where(*where)
        .order_by(LLMCallLog.generated_at.desc())
        .limit(max_rows)
    ).all())
    dropped = max(0, total - len(rows))
    covered_from = min((r.generated_at for r in rows), default=None)
    scan: dict[str, Any] = {
        "rows_in_window": total,
        "rows_scanned": len(rows),
        "rows_dropped_by_cap": dropped,
        "max_rows_scanned": max_rows,
        "dropped_reason": None,
        # The oldest row actually read. Equal to the window start unless the
        # cap bit, in which case the sample covers a shorter period than the
        # window says and the reader must know.
        "scanned_from": covered_from.isoformat() if covered_from else None,
    }
    if dropped:
        scan["dropped_reason"] = (
            f"{dropped} row(s) in the window were not read: the per-request cap is "
            f"{max_rows} rows, newest first. Every figure below covers "
            f"{scan['scanned_from']} onwards, not the whole window"
        )
    return rows, scan


def _straddling_units(db: Session, rows: Sequence[Any], *, floor: datetime,
                      run_features: Sequence[str]) -> tuple[set[str], dict[str, Any]]:
    """Run-grouped units of the sample that reach back past its oldest row.

    A `run_id` unit is a unit only if every one of its calls was read. A run
    that started before `floor` — because the cap cut the read short, or
    because the window did — contributes its tail alone, and pricing that tail
    as a whole run invents a cheap unit no run ever incurred. So the calls
    immediately below `floor` are read (bounded by `BOUNDARY_LOOKBACK` in time
    and `BOUNDARY_LOOKBACK_ROWS` in rows, newest first, on the `generated_at`
    index) and every run of the sample found among them is named and returned
    for exclusion.

    The check is exhaustive when the period it looked at is at least as long
    as the longest unit the sample itself shows. When it is not, the returned
    block says so with the numbers instead of implying a guarantee.
    """
    first: dict[str, datetime] = {}
    last: dict[str, datetime] = {}
    for r in rows:
        if r.feature in run_features and r.run_id:
            first[r.run_id] = min(first.get(r.run_id, r.generated_at), r.generated_at)
            last[r.run_id] = max(last.get(r.run_id, r.generated_at), r.generated_at)
    longest = max((last[k] - first[k] for k in first), default=timedelta(0))

    zone_start = floor - BOUNDARY_LOOKBACK
    below = list(db.execute(
        select(LLMCallLog.run_id, LLMCallLog.generated_at)
        .where(
            LLMCallLog.feature.in_(sorted(set(run_features))),
            LLMCallLog.run_id.is_not(None),
            LLMCallLog.generated_at < floor,
            LLMCallLog.generated_at >= zone_start,
        )
        .order_by(LLMCallLog.generated_at.desc())
        .limit(BOUNDARY_LOOKBACK_ROWS)
    ).all())
    # The lookback hit its own row cap: it saw back only as far as its oldest row.
    capped = len(below) >= BOUNDARY_LOOKBACK_ROWS
    checked_from = below[-1].generated_at if capped else zone_start
    straddlers = {r.run_id for r in below if r.run_id} & set(first)
    info: dict[str, Any] = {
        "floor": floor.isoformat(),
        "checked_from": checked_from.isoformat(),
        "rows_checked": len(below),
        "straddling_units": len(straddlers),
        "longest_unit_span_seconds": round(longest.total_seconds(), 3),
        "exhaustive": (floor - checked_from) >= longest,
        "reason": None,
    }
    if not info["exhaustive"]:
        info["reason"] = (
            f"the check reached back only to {info['checked_from']} ({len(below)} rows, "
            f"its own cap), which is less than the longest unit the sample shows "
            f"({longest}). A unit that began before that point and was cut by the scan "
            "cap would not have been caught — widen BOUNDARY_LOOKBACK_ROWS or narrow "
            "the window"
        )
    return straddlers, info


def _units_for(op: Operation, rows: Sequence[Any],
               partial_runs: set[str]) -> dict[str, Any]:
    """Group one operation's rows into units and price each one.

    `partial_runs` are `run_id`s known to reach back past the oldest row read
    (`_straddling_units`); their units are excluded rather than priced from
    the fragment that was read.
    """
    mine = [r for r in rows if r.feature == op.log_feature]
    units: dict[str, _Unit] = {}
    unattributed = 0
    failed_calls = 0
    models: set[str] = set()
    unpriced_models: set[str] = set()
    fallback_priced_models: set[str] = set()
    n_calls = 0

    for i, r in enumerate(mine):
        model_label = f"{r.provider or '?'}/{r.model or '?'}"
        models.add(model_label)
        source = price_source(r.provider, r.model)
        priced = source != "unpriced"
        if not priced:
            unpriced_models.add(model_label)
        elif source == "provider_default":
            fallback_priced_models.add(model_label)
        if op.basis == BASIS_RUN_ID:
            if not r.run_id:
                # A run-grouped call with no run_id belongs to no unit. It is
                # real spend, so it is counted and named rather than dropped.
                unattributed += 1
                continue
            key = r.run_id
        else:
            if not r.success:
                # One call is one delivered unit here, and a failed call
                # delivered nothing. Counted separately so the failure rate
                # stays visible.
                failed_calls += 1
                continue
            key = f"{op.key}:{i}"
        n_calls += 1
        unit = units.setdefault(key, _Unit())
        unit.n_calls += 1
        if op.basis == BASIS_RUN_ID and key in partial_runs:
            unit.problems.add("partial_unit")
        unit.cost_usd += estimate_cost_usd(
            r.provider, r.model, r.tokens_in, r.tokens_out,
            cache_read_tokens=r.cache_read_tokens, cache_write_tokens=r.cache_write_tokens,
        )
        if not priced:
            unit.problems.add("unpriced_model")
        if r.success and not (r.tokens_in or r.tokens_out):
            unit.problems.add("missing_token_counts")
        if op.basis == BASIS_RUN_ID and not r.success:
            failed_calls += 1

    excluded: dict[str, int] = {}
    n_excluded = 0
    usable: list[float] = []
    for unit in units.values():
        if unit.problems:
            # A unit with two problems is counted under both, so these
            # per-reason counts can sum past `n_units_excluded` — which is the
            # number of units actually removed.
            n_excluded += 1
            for p in sorted(unit.problems):
                excluded[p] = excluded.get(p, 0) + 1
            continue
        usable.append(round(unit.cost_usd, 6))
    return {
        "units": usable,
        "n_units_seen": len(units),
        "n_units_excluded": n_excluded,
        "n_calls_in_units": n_calls,
        "unattributed_calls": unattributed,
        "failed_calls": failed_calls,
        "models": sorted(models),
        "unpriced_models": sorted(unpriced_models),
        "fallback_priced_models": sorted(fallback_priced_models),
        "excluded_units": excluded,
    }


def _figures(op: Operation, grouped: dict[str, Any], *, window_days: int,
             rows_dropped: int) -> dict[str, Any]:
    """Turn one operation's usable units into the reported block.

    `rows_dropped` is `scan.rows_dropped_by_cap`: an unread row cannot be
    attributed to an operation, so once the cap has bitten, the *volume* of
    any one operation in the window is unknown — not the fraction of it that
    was read.
    """
    units: list[float] = grouped["units"]
    n = len(units)
    reasons: dict[str, str] = {}
    cost: dict[str, float | None] = {"median": None, "p90": None, "mean": None,
                                     "min": None, "max": None}
    n_excluded = grouped["n_units_excluded"]
    if n >= MIN_UNITS_FOR_MEDIAN:
        cost["median"] = round(percentile(units, 0.5), 6)
        cost["mean"] = round(sum(units) / n, 6)
        cost["min"] = round(min(units), 6)
        cost["max"] = round(max(units), 6)
        status = "observed"
    else:
        status = "insufficient_sample"
        reasons["median"] = (
            f"{n} usable unit(s) of {op.unit} in the window — fewer than the "
            f"{MIN_UNITS_FOR_MEDIAN} this module requires before it will state a cost"
            + (f"; {n_excluded} further unit(s) were excluded, see excluded_units"
               if n_excluded else "")
        )
        reasons["mean"] = reasons["median"]
    if n >= MIN_UNITS_FOR_P90:
        cost["p90"] = round(percentile(units, 0.9), 6)
    else:
        reasons["p90"] = (
            f"{n} usable unit(s) — a p90 needs at least {MIN_UNITS_FOR_P90}; with fewer "
            "the tail is one sample wide"
        )

    # Volume, not cost: every unit the window actually shows, including the
    # ones no cost could be put on. Unknown — never a smaller number stated
    # as fact — once the cap has left rows unread.
    if rows_dropped:
        units_per_30d: float | None = None
        reasons["units_per_30d"] = (
            f"the scan cap left {rows_dropped} row(s) of the window unread, and an "
            f"unread row cannot be attributed to an operation, so the number of "
            f"{op.unit} in the window is unknown. Re-read with a larger max_rows or a "
            "narrower window"
        )
    else:
        units_per_30d = round(grouped["n_units_seen"] * 30.0 / window_days, 2)

    excluded = grouped["excluded_units"]
    out: dict[str, Any] = {
        "unit": op.unit,
        "basis": op.basis,
        "basis_note": op.basis_note,
        "understates_unit": op.understates_unit,
        "plan_feature": op.plan_feature,
        "status": status,
        "n_units": n,
        "n_units_seen": grouped["n_units_seen"],
        "n_units_excluded": n_excluded,
        "n_calls_in_units": grouped["n_calls_in_units"],
        "cost_usd_per_unit": cost,
        "reasons": reasons,
        "units_per_30d": units_per_30d,
        "models": grouped["models"],
        "failed_calls": grouped["failed_calls"],
        "thresholds": {"median": MIN_UNITS_FOR_MEDIAN, "p90": MIN_UNITS_FOR_P90},
    }
    if excluded:
        out["excluded_units"] = {
            kind: {"n": count, "reason": EXCLUSION_REASONS[kind]}
            for kind, count in sorted(excluded.items())
        }
    if grouped["unpriced_models"]:
        out["unpriced_models"] = grouped["unpriced_models"]
    if grouped.get("fallback_priced_models"):
        out["fallback_priced_models"] = {
            "models": grouped["fallback_priced_models"],
            "reason": FALLBACK_PRICED_REASON,
        }
    if grouped["unattributed_calls"]:
        out["unattributed_calls"] = {
            "n": grouped["unattributed_calls"],
            "reason": (
                "tagged with this feature but carrying no run_id, so the call belongs to "
                "no unit. Real spend, excluded from the per-unit figures"
            ),
        }
    return out


# --- plan projection --------------------------------------------------------

def _allowance_term(op: Operation, block: dict[str, Any], plan: str) -> dict[str, Any]:
    """One plan × operation line: allowance × observed unit cost."""
    assert op.plan_feature is not None
    resolved = features.allowance(op.plan_feature, plan)
    term: dict[str, Any] = {
        "feature": op.plan_feature,
        "limit": resolved.limit,
        "allowed": resolved.allowed,
        "monthly_usd_median": None,
        "monthly_usd_p90": None,
        "reason": None,
    }
    if not resolved.allowed:
        term["reason"] = f"{op.plan_feature} is not available on the {plan} plan"
        return term
    if resolved.limit is None:
        term["reason"] = (
            f"{op.plan_feature} is unlimited on {plan}: there is no allowance to multiply, "
            "so a monthly ceiling cannot be stated — multiply the unit cost by expected "
            "usage instead"
        )
        return term
    median = block["cost_usd_per_unit"]["median"]
    p90 = block["cost_usd_per_unit"]["p90"]
    if median is None:
        term["reason"] = block["reasons"].get("median", "no observed unit cost")
        return term
    term["monthly_usd_median"] = round(resolved.limit * median, 4)
    term["monthly_usd_p90"] = None if p90 is None else round(resolved.limit * p90, 4)
    if p90 is None:
        term["reason"] = block["reasons"].get("p90")
    if op.understates_unit:
        term["floor_only"] = True
        term["floor_reason"] = op.basis_note
    return term


def _plan_block(plan: str, operations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    terms: dict[str, dict[str, Any]] = {}
    total_median = 0.0
    total_p90 = 0.0
    priced: list[str] = []
    unpriced: list[dict[str, str]] = []
    floors: list[str] = []
    p90_complete = True

    for op in OPERATIONS:
        if op.plan_feature is None:
            continue
        term = _allowance_term(op, operations[op.key], plan)
        terms[op.key] = term
        if not term["allowed"]:
            continue
        if term["monthly_usd_median"] is None:
            unpriced.append({"operation": op.key, "reason": term["reason"] or "unknown"})
            continue
        priced.append(op.key)
        total_median += term["monthly_usd_median"]
        if term["monthly_usd_p90"] is None:
            p90_complete = False
        else:
            total_p90 += term["monthly_usd_p90"]
        if term.get("floor_only"):
            floors.append(op.key)

    # Cost-bearing features the plan allows that this report never measures
    # at all (dcf, comps, portfolio: money leaves on every call and there is
    # no monthly allowance to multiply). They are not "unpriced terms" — they
    # are not terms here at all — but the total is not a total without them.
    unmeasured = _unmeasured_terms(plan)
    complete = not unpriced and not unmeasured
    threshold = PLAN_THRESHOLD_USD[plan]
    # What IS measured, even when the total cannot be — and `None`, not
    # `0.0`, when nothing was: a subtotal of nothing is not free spend.
    subtotal = round(total_median, 4) if priced else None
    block: dict[str, Any] = {
        "price_usd_per_month": PRO_PRICE_USD if plan == "pro" else 0.0,
        "terms": terms,
        "monthly_variable_cost_usd": {
            "median": round(total_median, 4) if complete else None,
            "p90": round(total_p90, 4) if complete and p90_complete else None,
            "covers": priced,
            "is_complete": complete,
        },
        "measured_subtotal_usd_median": subtotal,
        "measured_terms": priced,
        "unpriced_terms": unpriced,
        "unmeasured_terms": unmeasured,
        "threshold_usd": threshold,
        "threshold_source": (
            "docs/economics/unit-costs-2026-09.md §3 — Pro variable cost under half the "
            "list price, Free under $1.50/user/month"
        ),
        "verdict": None,
        "notes": [],
    }
    if unpriced:
        block["notes"].append(
            "the monthly total is null because "
            + "; ".join(f"{u['operation']}: {u['reason']}" for u in unpriced)
            + ". The measured subtotal covers only "
            + (", ".join(priced) if priced else "nothing — no term has a measured cost")
        )
    if unmeasured:
        block["notes"].append(
            "the monthly total is null because the plan allows cost-bearing features "
            "this report does not measure — "
            + "; ".join(f"{u['feature']}: {u['reason']}" for u in unmeasured)
            + ". The measured subtotal is the metered allowances only, so it is a lower "
            "bound on the plan's variable cost, not the cost"
        )
    if floors:
        block["notes"].append(
            "a lower bound, not a ceiling: "
            + ", ".join(floors)
            + " is priced per LLM call because the log carries no per-unit identifier"
        )

    # The go/no-go answer, and — as importantly — when there is not one.
    # A measured subtotal already over the threshold settles it whatever is
    # missing, because a missing term can only add cost; a subtotal under it
    # settles nothing while anything is missing.
    missing = [u["operation"] for u in unpriced] + [u["feature"] for u in unmeasured]
    verdict: dict[str, Any] = {
        "under_threshold_at_median": None,
        "decided": False,
        "basis": (
            "median unit costs × the plan allowance, every allowance exhausted"
            if complete else
            "the median unit cost × the plan allowance of the measured terms only — "
            "not every cost-bearing feature the plan allows"
        ),
        "is_a_floor": bool(floors),
        "reason": None,
    }
    if subtotal is None:
        verdict["reason"] = (
            "no term has a measured unit cost, so there is nothing to compare with the "
            f"${threshold} threshold"
        )
    elif subtotal >= threshold:
        verdict.update(under_threshold_at_median=False, decided=True)
        if not complete:
            verdict["reason"] = (
                f"decided on the measured terms alone: they already total ${subtotal:,.2f} "
                f"against a ${threshold} threshold, and the terms this report cannot "
                "measure (" + ", ".join(missing) + ") can only add to that"
            )
    elif complete:
        verdict.update(under_threshold_at_median=True, decided=True)
    else:
        verdict["reason"] = (
            f"undecided: the measured terms total ${subtotal:,.2f}, under the ${threshold} "
            "threshold, but " + ", ".join(missing) + " are not measured here, so the "
            "plan's variable cost could still be over it"
        )
    block["verdict"] = verdict

    if plan == "pro":
        block["gross_margin_pct_at_median"] = round(
            100.0 * (PRO_PRICE_USD - total_median) / PRO_PRICE_USD, 2,
        ) if complete else None
        if not complete:
            block["notes"].append(
                "gross margin is null while any cost-bearing term is unmeasured — an "
                "unmeasured term is not a free one"
            )
        if subtotal is not None:
            block["gross_margin_pct_at_median_ceiling"] = round(
                100.0 * (PRO_PRICE_USD - subtotal) / PRO_PRICE_USD, 2,
            )
            if not complete:
                block["notes"].append(
                    "gross_margin_pct_at_median_ceiling is an upper bound and nothing "
                    f"more: the measured terms alone leave that share of the "
                    f"${PRO_PRICE_USD} price, and every term missing from the subtotal "
                    "can only cut it further"
                )
    return block


def _uncovered() -> dict[str, features.Feature]:
    """Every entitlement-matrix feature no `Operation` measures."""
    covered = {op.plan_feature for op in OPERATIONS if op.plan_feature}
    return {name: feat for name, feat in sorted(features.FEATURES.items())
            if name not in covered}


def _uncovered_reason(feat: features.Feature) -> str:
    """Why this report does not price `feat`. One branch per combination of
    the two flags, so no shape can fall through unmentioned — least of all
    the dangerous one (metered *and* cost-bearing), which is a feature that
    both spends money and has an allowance to multiply, i.e. one this module
    is supposed to be measuring."""
    if feat.cost_bearing and feat.metered:
        return (
            "cost-bearing AND metered, but OPERATIONS has no entry for it: its "
            "llm_call_logs rows are never read and its allowance is never priced, so no "
            "plan total here includes it. Add an Operation for it — this is the drift "
            "this list exists to catch"
        )
    if feat.cost_bearing:
        return (
            "cost-bearing but not metered per month, so there is no allowance to "
            "multiply — see the U_* usage assumptions in "
            "docs/economics/unit-costs-2026-09.md §3 and ASSUMED_UNMETERED in "
            "scripts/audit_unit_costs.py"
        )
    if feat.metered:
        return (
            "metered but not cost-bearing: it reads what a research_run already paid "
            "for, and writes no llm_call_logs rows of its own"
        )
    return (
        "neither metered nor cost-bearing: no LLM spend to observe and no allowance to "
        "multiply, so there is nothing here to price"
    )


def _terms_not_covered() -> list[dict[str, str]]:
    """Plan features this report does not price, each with why.

    Derived from `auth/features.py` rather than listed by hand, and covering
    *every* uncovered feature rather than two of the four flag combinations,
    so a new cost-bearing feature cannot quietly fall out of the picture.
    """
    return [{"feature": name, "reason": _uncovered_reason(feat)}
            for name, feat in _uncovered().items()]


def _unmeasured_terms(plan: str) -> list[dict[str, str]]:
    """Cost-bearing features `plan` allows that this report never measures.

    Real spend the plan totals cannot see: `dcf` and `comps` on every plan
    (they follow a memo on Free), `portfolio` on Pro. Derived from the
    entitlement matrix, so a feature added there arrives here rather than
    silently widening the gap between the total and the invoice.
    """
    return [
        {"feature": name, "reason": _uncovered_reason(feat)}
        for name, feat in _uncovered().items()
        if feat.cost_bearing and features.allowance(name, plan).allowed
    ]


# --- entry point ------------------------------------------------------------

def build_report(*, window_days: int = DEFAULT_WINDOW_DAYS, now: datetime | None = None,
                 max_rows: int = MAX_ROWS_SCANNED,
                 db: Session | None = None) -> dict[str, Any]:
    """Observed cost per metered operation, and what each plan's allowance costs.

    `now` is injectable so tests can pin the clock; `max_rows` bounds the read
    (see `MAX_ROWS_SCANNED`). Pure reads — no provider call, no model call,
    nothing written.
    """
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    until = now or datetime.utcnow()
    since = until - timedelta(days=window_days)

    own = db is None
    session = db if db is not None else SessionLocal()
    try:
        # Mirrors llm_metrics: a direct caller should not need init_db().
        LLMCallLog.__table__.create(bind=session.get_bind(), checkfirst=True)
        rows, scan = _scan(session, since=since, until=until, max_rows=max_rows,
                           log_features=[op.log_feature for op in OPERATIONS])
        # The oldest point the sample can see: the cap's cut when it bit,
        # otherwise the window start. A run-grouped unit reaching back past it
        # is a fragment, not a unit.
        floor = min((r.generated_at for r in rows), default=since) \
            if scan["rows_dropped_by_cap"] else since
        partial_runs, scan["boundary_check"] = _straddling_units(
            session, rows, floor=floor,
            run_features=[op.log_feature for op in OPERATIONS
                          if op.basis == BASIS_RUN_ID],
        )
    finally:
        if own:
            session.close()

    operations = {
        op.key: _figures(op, _units_for(op, rows, partial_runs),
                         window_days=window_days,
                         rows_dropped=scan["rows_dropped_by_cap"])
        for op in OPERATIONS
    }
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "window": {
            "days": window_days,
            "since": since.isoformat(),
            "until": until.isoformat(),
        },
        "scan": scan,
        "source": {
            "rows": "llm_call_logs (one row per real provider call)",
            "pricing": (
                "llm_metrics.estimate_cost_usd — list-price arithmetic over logged token "
                "counts, not an invoice. Reconcile against the provider dashboards before "
                "a pricing decision"
            ),
            "allowances": "auth/features.py (with ENTITLEMENT_OVERRIDES_JSON applied)",
        },
        "operations": operations,
        "plans": {plan: _plan_block(plan, operations) for plan in PLANS},
        "terms_not_covered": _terms_not_covered(),
    }
