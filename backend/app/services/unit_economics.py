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
   without token counts, and rows dropped by the scan cap are each counted
   and named. A zero never stands in for an unknown.

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
    MODEL_PRICES_PER_MTOK,
    PROVIDER_PRICE_FALLBACK,
    estimate_cost_usd,
)

# --- knobs ------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 30
# Hard ceiling on rows read per request. The route is read-only and must not
# turn into a table scan as the log grows; `generated_at` is indexed, so the
# newest `MAX_ROWS_SCANNED` rows in the window come off the index. Whatever
# the cap drops is counted and reported (`scan.rows_dropped_by_cap`) together
# with the window the kept rows actually cover.
MAX_ROWS_SCANNED = 20_000
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
}


def _is_priced(provider: str, model: str) -> bool:
    return ((model or "").lower() in MODEL_PRICES_PER_MTOK
            or (provider or "").lower() in PROVIDER_PRICE_FALLBACK)


def _scan(db: Session, *, since: datetime, until: datetime, max_rows: int,
          log_features: Iterable[str]) -> tuple[list[Any], dict[str, Any]]:
    """The newest `max_rows` tagged rows in the window, plus what was dropped.

    Two bounded, index-served statements: a COUNT over the window and a
    LIMITed newest-first read. The COUNT is what lets the cap report the rows
    it left behind instead of quietly under-reporting the sample.
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


def _units_for(op: Operation, rows: Sequence[Any]) -> dict[str, Any]:
    """Group one operation's rows into units and price each one."""
    mine = [r for r in rows if r.feature == op.log_feature]
    units: dict[str, _Unit] = {}
    unattributed = 0
    failed_calls = 0
    models: set[str] = set()
    unpriced_models: set[str] = set()
    n_calls = 0

    for i, r in enumerate(mine):
        model_label = f"{r.provider or '?'}/{r.model or '?'}"
        models.add(model_label)
        priced = _is_priced(r.provider, r.model)
        if not priced:
            unpriced_models.add(model_label)
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
        unit.cost_usd += estimate_cost_usd(r.provider, r.model, r.tokens_in, r.tokens_out)
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
        "excluded_units": excluded,
    }


def _figures(op: Operation, grouped: dict[str, Any], *, window_days: int) -> dict[str, Any]:
    """Turn one operation's usable units into the reported block."""
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
        "units_per_30d": (round(n * 30.0 / window_days, 2) if window_days else None),
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

    complete = not unpriced
    block: dict[str, Any] = {
        "price_usd_per_month": PRO_PRICE_USD if plan == "pro" else 0.0,
        "terms": terms,
        "monthly_variable_cost_usd": {
            "median": round(total_median, 4) if complete else None,
            "p90": round(total_p90, 4) if complete and p90_complete else None,
        },
        # What IS measured, even when the total cannot be — and `None`, not
        # `0.0`, when nothing was: a subtotal of nothing is not free spend.
        "measured_subtotal_usd_median": round(total_median, 4) if priced else None,
        "measured_terms": priced,
        "unpriced_terms": unpriced,
        "threshold_usd": PLAN_THRESHOLD_USD[plan],
        "threshold_source": (
            "docs/economics/unit-costs-2026-09.md §3 — Pro variable cost under half the "
            "list price, Free under $1.50/user/month"
        ),
        "verdict": None,
        "notes": [],
    }
    if not complete:
        block["notes"].append(
            "the monthly total is null because "
            + "; ".join(f"{u['operation']}: {u['reason']}" for u in unpriced)
            + ". The measured subtotal covers only "
            + (", ".join(priced) if priced else "nothing — no term has a measured cost")
        )
    if floors:
        block["notes"].append(
            "a lower bound, not a ceiling: "
            + ", ".join(floors)
            + " is priced per LLM call because the log carries no per-unit identifier"
        )
    if complete:
        under = total_median < PLAN_THRESHOLD_USD[plan]
        block["verdict"] = {
            "under_threshold_at_median": under,
            "basis": "median unit costs × the plan allowance, every allowance exhausted",
            "is_a_floor": bool(floors),
        }
    if plan == "pro" and complete:
        block["gross_margin_pct_at_median"] = round(
            100.0 * (PRO_PRICE_USD - total_median) / PRO_PRICE_USD, 2,
        )
    elif plan == "pro":
        block["gross_margin_pct_at_median"] = None
        block["notes"].append(
            "gross margin is null while any metered term is unmeasured — an unmeasured "
            "term is not a free one"
        )
    return block


def _terms_not_covered() -> list[dict[str, str]]:
    """Plan features this report deliberately does not price, each with why.

    Derived from `auth/features.py` rather than listed by hand, so a new
    cost-bearing feature cannot quietly fall out of the picture.
    """
    covered = {op.plan_feature for op in OPERATIONS if op.plan_feature}
    out: list[dict[str, str]] = []
    for name, feat in sorted(features.FEATURES.items()):
        if name in covered:
            continue
        if feat.cost_bearing and not feat.metered:
            out.append({"feature": name, "reason": (
                "cost-bearing but not metered per month, so there is no allowance to "
                "multiply — see the U_* usage assumptions in "
                "docs/economics/unit-costs-2026-09.md §3"
            )})
        elif feat.metered and not feat.cost_bearing:
            out.append({"feature": name, "reason": (
                "metered but not cost-bearing: it reads what a research_run already paid "
                "for, and writes no llm_call_logs rows of its own"
            )})
    return out


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
    finally:
        if own:
            session.close()

    operations = {
        op.key: _figures(op, _units_for(op, rows), window_days=window_days)
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
