"""Fundamental Factor Scorecard — pure feature engine (``fs-v1``).

No DB, no I/O, no numpy. Two steps:

1. ``pit_snapshot(rows, as_of)`` turns long-format ``financial_periods``
   rows into a point-in-time view: only rows whose ``available_at`` is on
   or before ``as_of`` survive, with a known completed period and credible
   availability no earlier than period end, grouped into annual points. ``fs-v1``
   ingests annual statements only, so "TTM" here is the latest annual row
   that was knowable at ``as_of``; quarterly rows are ignored and counted.
2. ``compute_features(snapshot, price_ctx, sector)`` evaluates every
   formula in ``scorecard_spec.FEATURE_SPEC`` and returns the raw (unsigned,
   unnormalised) values. Anything that cannot be computed is ``None`` with
   a reason — never a zero, never a neutral placeholder. The reasons are
   the product's "n/a because …" text, so they are stable strings.

Missing inputs are never zero-filled. A line that is absent from the
snapshot is *unknown*: the persistence slice stores a provider null as a
null row and ``pit_snapshot`` drops it, so by the time a formula runs
"absent" and "unknown" are indistinguishable. A 0 invented for a missing
partner line (R&D next to SG&A, buybacks next to dividends, short-term
investments next to cash, one half of the debt split) would be
standardised and ranked as if it had been observed, so the feature is
null with ``missing:<line>`` instead. A provider that reports 0 stores 0
and scores as 0. The rule is frozen in ``scorecard_spec.RULES`` and
therefore in the spec hash.

Row contract (owned by the persistence slice, mirrored here so this
module can be tested without a database)::

    (statement, line_item, period, period_end, fiscal_year, fiscal_quarter, value, available_at)

The persistence reader appends an optional ninth metadata mapping containing
the row ID, ticker, provider and availability provenance for exclusion audits.

``price_ctx`` is ``{"price": float | None, "price_date": date | None,
"shares_fallback": float | None}``; ``shares_fallback`` is
``Company.shares_outstanding`` and is only used when the income statement
has no diluted share count (the source is recorded in the result context).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from statistics import stdev
from typing import Any

from app.finance import ratios
from app.finance.pit_eligibility import as_date, date_exclusion_reasons, exclusion_record
from app.finance.scorecard_spec import (
    FEATURE_SPEC,
    FeatureSpec,
    applicable_features,
    normalize_sector,
)

STATEMENT_INCOME = "income"
STATEMENT_BALANCE = "balance"
STATEMENT_CASH = "cash"
_STATEMENTS = (STATEMENT_INCOME, STATEMENT_BALANCE, STATEMENT_CASH)
# Owner decision 2026-09-24: FMP is the primary fundamentals provider. Kept
# as a literal (not imported from services) so the engine stays pure; it must
# equal `fundamental_history_service.PRIMARY_PROVIDER`.
_PRIMARY_PROVIDER = "fmp"

# Cap on interest coverage: a name paying almost no interest would
# otherwise post a ratio in the thousands and own the whole z-scale even
# after winsorization.
INTEREST_COVERAGE_CAP = 50.0
# Cash-conversion style ratios are clipped so a near-zero net income does
# not turn a normal cash flow into a 40x "conversion".
CONVERSION_CLIP = 3.0
# Gross-margin stability needs enough annual points to be a dispersion
# rather than a difference; five is all the FMP annual history gives us.
STABILITY_MIN_POINTS = 3
STABILITY_MAX_POINTS = 5

# Reason strings for null features. Kept as constants so the API/UI can
# match on them; free text goes after the colon.
REASON_EXCLUDED = "excluded"          # excluded:sector=Financials
REASON_MISSING = "missing"            # missing:revenue
REASON_NO_PRIOR = "no_prior_period"   # no_prior_period:fy=2023
REASON_DENOMINATOR = "denominator"    # denominator:market_cap<=0
REASON_NO_SNAPSHOT = "no_snapshot"    # no annual point available at as_of
REASON_HISTORY = "insufficient_history"  # insufficient_history:n=2<3


# ---------------------------------------------------------------------------
# Point-in-time snapshot
# ---------------------------------------------------------------------------

def _to_date(x: Any) -> date | None:
    """Availability stamps arrive as date, datetime or ISO string; compare on
    the calendar date so a filing accepted at 16:05 on the as-of day counts
    as available that day."""
    return as_date(x)


@dataclass(frozen=True)
class AnnualPoint:
    """One fiscal year's statements as knowable at the snapshot's as-of."""
    fiscal_year: int
    period_end: date | None
    available_at: date | None   # latest availability among the rows included
    income: dict[str, float] = field(default_factory=dict)
    balance: dict[str, float] = field(default_factory=dict)
    cash: dict[str, float] = field(default_factory=dict)

    def statement(self, name: str) -> dict[str, float]:
        if name == STATEMENT_INCOME:
            return self.income
        if name == STATEMENT_BALANCE:
            return self.balance
        if name == STATEMENT_CASH:
            return self.cash
        raise KeyError(name)


@dataclass(frozen=True)
class PitSnapshot:
    """Annual points visible at ``as_of``, newest first.

    ``notes`` counts what was dropped and why, so the run can report
    ``pit_excluded`` etc. instead of silently scoring on a thinner panel.
    """
    as_of: date
    points: tuple[AnnualPoint, ...]
    notes: dict[str, int]
    excluded_rows: tuple[dict[str, Any], ...] = ()

    @property
    def latest(self) -> AnnualPoint | None:
        return self.points[0] if self.points else None

    def point_for_year(self, fiscal_year: int) -> AnnualPoint | None:
        for p in self.points:
            if p.fiscal_year == fiscal_year:
                return p
        return None

    def years_back(self, n: int) -> AnnualPoint | None:
        """The point exactly ``n`` fiscal years before the latest one. Growth
        features need the adjacent year, not "the next older row we happen
        to have" — a two-year gap labelled as one-year growth is wrong."""
        latest = self.latest
        if latest is None:
            return None
        return self.point_for_year(latest.fiscal_year - n)

    @property
    def prior(self) -> AnnualPoint | None:
        return self.years_back(1)


def pit_snapshot(rows: Iterable[Sequence[Any]], as_of: date) -> PitSnapshot:
    """Build the point-in-time annual view of one ticker's rows.

    Rules (each dropped row is counted in ``notes``):

    * ``available_at`` missing → ``missing_available_at`` (excluded: we
      cannot prove it was knowable);
    * ``available_at`` after ``as_of`` → ``pit_excluded``;
    * unknown period end → ``missing_period_end`` / ``rows_unusable``;
    * future period end or availability before period end → ``pit_excluded``;
      all date exclusion reasons and supplied identities are retained, uncapped;
    * ``fiscal_quarter`` set (1-4) → ``quarterly_ignored`` (fs-v1 is annual);
    * ``value`` None / non-finite → ``null_values_dropped``;
    * no fiscal year and no period_end to derive one from → ``rows_unusable``;
    * unknown statement name → ``rows_unusable``;
    * a second row for the same (year, statement, line) → counted in
      ``duplicate_rows``; the winner is chosen by the data, not by input
      order (see below).

    Duplicates are real: two period labels for one fiscal year (``FY2024``
    from FMP, ``2024`` from Alpha Vantage) both land in ``financial_periods``
    across provider fallbacks. The winner is a primary-provider (FMP) row
    when one is visible (owner decision 2026-09-24; rows without source
    metadata are non-primary), then the row with the latest
    ``available_at`` (a later fill or restatement supersedes an earlier
    one), then the greater period label, then the greater value. Adding the
    primary rank changes ``inputs_hash`` once, only for tickers where a
    non-primary duplicate used to win, so their next scoring run is not
    skipped as unchanged. Because
    the choice depends only on the rows, the snapshot — and therefore
    ``inputs_hash`` — is identical for any ordering of the same rows,
    which the persistence slice's ``(version_key, as_of, inputs_hash)``
    skip rule depends on; an unordered SELECT must not change the score.
    """
    notes: dict[str, int] = {
        "rows_seen": 0,
        "missing_available_at": 0,
        "pit_excluded": 0,
        "quarterly_ignored": 0,
        "null_values_dropped": 0,
        "rows_unusable": 0,
        "duplicate_rows": 0,
        "missing_period_end": 0,
        "available_before_period_end": 0,
        "period_end_after_as_of": 0,
    }
    # (fiscal_year, statement, line_item) -> (is primary, available_at, period label, value)
    chosen: dict[tuple[int, str, str], tuple[bool, date, str, float]] = {}
    period_end_by_year: dict[int, date | None] = {}
    excluded_rows: list[dict[str, Any]] = []

    for row in rows:
        notes["rows_seen"] += 1
        statement, line_item, _period, period_end, fiscal_year, fiscal_quarter, value, available_at = row[:8]
        metadata = dict(row[8]) if len(row) > 8 and isinstance(row[8], Mapping) else {}
        if statement not in _STATEMENTS or not line_item:
            notes["rows_unusable"] += 1
            continue
        avail = _to_date(available_at)
        reasons = date_exclusion_reasons(period_end, available_at, as_of)
        if reasons:
            for reason in ("missing_available_at", "missing_period_end", "available_before_period_end", "period_end_after_as_of"):
                if reason in reasons:
                    notes[reason] += 1
            if "missing_period_end" in reasons:
                notes["rows_unusable"] += 1
            if any(reason in reasons for reason in ("available_after_as_of", "available_before_period_end", "period_end_after_as_of")):
                notes["pit_excluded"] += 1
            excluded_rows.append(exclusion_record({**metadata, "statement": statement, "line_item": line_item,
                "period": _period, "period_end": period_end, "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter, "value": value, "available_at": available_at}, reasons, as_of))
            continue
        # `date_exclusion_reasons` derives `missing_available_at` from the same
        # `as_date(available_at)` that produced `avail`, so empty reasons mean it
        # parsed. Asserted rather than assumed: if those two ever stop agreeing,
        # this fails here instead of storing a None availability that only breaks
        # later, in a comparison in another function.
        assert avail is not None
        if fiscal_quarter not in (None, 0):
            notes["quarterly_ignored"] += 1
            continue
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            notes["null_values_dropped"] += 1
            continue
        pe = _to_date(period_end)
        fy = fiscal_year if fiscal_year is not None else (pe.year if pe is not None else None)
        if fy is None:
            notes["rows_unusable"] += 1
            continue
        fy = int(fy)
        key = (fy, statement, line_item)
        candidate = (metadata.get("source") == _PRIMARY_PROVIDER, avail,
                     "" if _period is None else str(_period), float(value))
        incumbent = chosen.get(key)
        if incumbent is not None:
            notes["duplicate_rows"] += 1
        if incumbent is None or candidate > incumbent:
            chosen[key] = candidate
        prev_pe = period_end_by_year.get(fy)
        if pe is not None and (prev_pe is None or pe > prev_pe):
            period_end_by_year[fy] = pe
        elif fy not in period_end_by_year:
            period_end_by_year[fy] = pe

    # Sorted so the statement dicts (and everything hashed from them) have
    # an order that does not depend on how the rows arrived.
    by_year: dict[int, dict[str, dict[str, float]]] = {}
    available_by_year: dict[int, date] = {}
    for key in sorted(chosen):
        fy, statement, line_item = key
        _primary, avail, _label, val = chosen[key]
        stmts = by_year.setdefault(fy, {s: {} for s in _STATEMENTS})
        stmts[statement][line_item] = val
        prev_avail = available_by_year.get(fy)
        if prev_avail is None or avail > prev_avail:
            available_by_year[fy] = avail

    points = tuple(
        AnnualPoint(
            fiscal_year=fy,
            period_end=period_end_by_year.get(fy),
            available_at=available_by_year.get(fy),
            income=stmts[STATEMENT_INCOME],
            balance=stmts[STATEMENT_BALANCE],
            cash=stmts[STATEMENT_CASH],
        )
        for fy, stmts in sorted(by_year.items(), key=lambda kv: kv[0], reverse=True)
    )
    notes["annual_points"] = len(points)
    excluded_rows.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return PitSnapshot(as_of=as_of, points=points, notes=notes, excluded_rows=tuple(excluded_rows))


def inputs_hash(snapshot: PitSnapshot, price_ctx: Mapping[str, Any] | None) -> str:
    """Fingerprint of everything the features were computed from, so a
    stored row can prove which inputs produced it (lineage) and a re-run
    on identical inputs can be recognised as such."""
    ctx = price_ctx or {}
    payload = {
        "as_of": snapshot.as_of.isoformat(),
        "points": [
            {
                "fiscal_year": p.fiscal_year,
                "period_end": p.period_end.isoformat() if p.period_end else None,
                "available_at": p.available_at.isoformat() if p.available_at else None,
                "income": p.income,
                "balance": p.balance,
                "cash": p.cash,
            }
            for p in snapshot.points
        ],
        "price": ctx.get("price"),
        "price_date": _iso(ctx.get("price_date")),
        "shares_fallback": ctx.get("shares_fallback"),
    }
    if snapshot.excluded_rows:
        payload["pit_exclusions"] = snapshot.excluded_rows
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iso(x: Any) -> str | None:
    d = _to_date(x)
    return d.isoformat() if d is not None else (str(x) if x is not None else None)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    """Raw feature values plus the audit trail the UI needs.

    ``values`` has every spec feature as a key; ``reasons`` has an entry
    for every None in ``values``; ``context`` carries the derived scalars
    (market cap, EV, share source, tax-rate provenance, …) that the
    persisted row keeps under ``feature_raw`` for explainability.
    """
    values: dict[str, float | None]
    reasons: dict[str, str]
    context: dict[str, Any]
    applicable: frozenset[str]


_Value = tuple[float | None, str | None]


class _Calc:
    """Evaluation context for one ticker at one as-of. Holds the snapshot
    points and the derived scalars every formula shares (market cap, EV,
    shares) so each feature function is a few honest lines."""

    def __init__(self, snapshot: PitSnapshot, price_ctx: Mapping[str, Any] | None) -> None:
        self.snapshot = snapshot
        self.latest = snapshot.latest
        self.prior = snapshot.years_back(1)
        self.prior2 = snapshot.years_back(2)
        self.three_back = snapshot.years_back(3)
        ctx = price_ctx or {}
        self.price = _finite(ctx.get("price"))
        self.price_date = _to_date(ctx.get("price_date"))
        self.shares_fallback = _finite(ctx.get("shares_fallback"))
        self.flags: list[str] = []
        self.context: dict[str, Any] = {}
        self.shares: float | None = None
        self.shares_source: str = "none"
        self.mktcap: float | None = None
        self.total_debt: float | None = None
        self.total_debt_reason: str | None = None   # why total_debt is None
        self.net_debt: float | None = None
        self.net_debt_reason: str | None = None     # why net_debt is None
        self.ev: float | None = None
        self.ebitda: float | None = None
        self.avg_assets: float | None = None
        self.avg_assets_basis: str = "none"
        self._derive()

    # -- derived scalars ---------------------------------------------------

    def _derive(self) -> None:
        latest = self.latest
        income = latest.income if latest else {}
        balance = latest.balance if latest else {}

        shares = _get(income, "weighted_avg_shares_diluted")
        shares_source = "income_statement"
        if shares is None or shares <= 0:
            shares = self.shares_fallback if (self.shares_fallback or 0) > 0 else None
            shares_source = "company_shares_outstanding" if shares is not None else "none"
        self.shares = shares
        self.shares_source = shares_source

        self.mktcap = self.price * shares if (self.price is not None and self.price > 0 and shares) else None

        debt, debt_reason, debt_flag = _total_debt(balance)
        if debt_flag:
            self.flags.append(debt_flag)
        self.total_debt, self.total_debt_reason = debt, debt_reason
        cash = _get(balance, "cash_and_equivalents")
        sti = _get(balance, "short_term_investments")
        # Every term of net debt must be reported. An absent short-term
        # investments line is unknown, not zero — `ratios.net_debt` and the
        # screener fill it with 0, but here that 0 would be standardised and
        # ranked as if observed (fs-v1 missing-input rule).
        if debt is None:
            self.net_debt, self.net_debt_reason = None, debt_reason
        elif cash is None:
            self.net_debt, self.net_debt_reason = None, f"{REASON_MISSING}:cash_and_equivalents"
        elif sti is None:
            self.net_debt, self.net_debt_reason = None, f"{REASON_MISSING}:short_term_investments"
        else:
            self.net_debt, self.net_debt_reason = debt - cash - sti, None
        self.ev = (self.mktcap + self.net_debt) if (self.mktcap is not None and self.net_debt is not None) else None

        self.ebitda, ebitda_flag = _ebitda(income, latest.cash if latest else {})
        if ebitda_flag:
            self.flags.append(ebitda_flag)
        ta_latest = _get(balance, "total_assets")
        ta_prior = _get(self.prior.balance, "total_assets") if self.prior else None
        if ta_latest is not None and ta_prior is not None:
            self.avg_assets = (ta_latest + ta_prior) / 2.0
            self.avg_assets_basis = "average_two_years"
        elif ta_latest is not None:
            # A single balance sheet is a level, not an average; scoring on
            # it is a documented approximation, not a fabricated number.
            self.avg_assets = ta_latest
            self.avg_assets_basis = "latest_only"
            self.flags.append("avg_assets_latest_only")
        else:
            self.avg_assets = None
            self.avg_assets_basis = "none"

        self.context.update({
            "fiscal_year": latest.fiscal_year if latest else None,
            "period_end": latest.period_end.isoformat() if latest and latest.period_end else None,
            "available_at": latest.available_at.isoformat() if latest and latest.available_at else None,
            "prior_fiscal_year": self.prior.fiscal_year if self.prior else None,
            "annual_points": len(self.snapshot.points),
            "price": self.price,
            "price_date": self.price_date.isoformat() if self.price_date else None,
            "shares": shares,
            "shares_source": shares_source,
            "market_cap": self.mktcap,
            "total_debt": debt,
            "net_debt": self.net_debt,
            "enterprise_value": self.ev,
            "ebitda": self.ebitda,
            "avg_assets_basis": self.avg_assets_basis,
        })

    # -- accessors ---------------------------------------------------------

    def inc(self, key: str) -> float | None:
        return _get(self.latest.income, key) if self.latest else None

    def bal(self, key: str) -> float | None:
        return _get(self.latest.balance, key) if self.latest else None

    def cf(self, key: str) -> float | None:
        return _get(self.latest.cash, key) if self.latest else None

    def prior_inc(self, key: str) -> float | None:
        return _get(self.prior.income, key) if self.prior else None

    def _need_prior(self) -> str | None:
        if self.prior is None:
            fy = self.latest.fiscal_year - 1 if self.latest else None
            return f"{REASON_NO_PRIOR}:fy={fy}"
        return None

    # -- valuation ---------------------------------------------------------

    def earnings_yield(self) -> _Value:
        return _over_mktcap(self, self.inc("net_income"), "net_income")

    def fcf_yield(self) -> _Value:
        return _over_mktcap(self, self.cf("free_cash_flow"), "free_cash_flow")

    def ebitda_ev_yield(self) -> _Value:
        if self.ebitda is None:
            return None, f"{REASON_MISSING}:ebitda"
        return _over_ev(self, self.ebitda)

    def sales_ev_yield(self) -> _Value:
        rev = self.inc("revenue")
        if rev is None:
            return None, f"{REASON_MISSING}:revenue"
        return _over_ev(self, rev)

    # -- quality -----------------------------------------------------------

    def roic(self) -> _Value:
        if self.latest is None:
            return None, REASON_NO_SNAPSHOT
        # Shared definition: the effective rate when credible, the statutory
        # fallback only for profitable names, None for a loss-maker with no
        # credible rate. The provenance label is kept in the context so a
        # null ROIC can be told apart by cause.
        value, provenance = ratios.roic_with_provenance(self.latest.income, self.latest.balance, tax_rate=None)
        self.context["roic_provenance"] = provenance
        if value is None:
            return None, f"{REASON_MISSING}:{provenance}"
        return _finite_or_reason(value, "roic")

    def roa(self) -> _Value:
        ni = self.inc("net_income")
        if ni is None:
            return None, f"{REASON_MISSING}:net_income"
        return _over_avg_assets(self, ni)

    def roe(self) -> _Value:
        ni = self.inc("net_income")
        eq = self.bal("shareholders_equity")
        if ni is None:
            return None, f"{REASON_MISSING}:net_income"
        if eq is None:
            return None, f"{REASON_MISSING}:shareholders_equity"
        if eq <= 0:
            return None, f"{REASON_DENOMINATOR}:shareholders_equity<=0"
        return ni / eq, None

    def gross_margin_stability(self) -> _Value:
        margins: list[float] = []
        for p in self.snapshot.points[:STABILITY_MAX_POINTS]:
            gp, rev = _get(p.income, "gross_profit"), _get(p.income, "revenue")
            if gp is not None and rev is not None and rev > 0:
                margins.append(gp / rev)
        if len(margins) < STABILITY_MIN_POINTS:
            return None, f"{REASON_HISTORY}:n={len(margins)}<{STABILITY_MIN_POINTS}"
        return -stdev(margins), None

    # -- growth ------------------------------------------------------------

    def revenue_growth_1y(self) -> _Value:
        return _growth(self, "revenue")

    def revenue_cagr_3y(self) -> _Value:
        rev = self.inc("revenue")
        if rev is None:
            return None, f"{REASON_MISSING}:revenue"
        if self.three_back is None:
            fy = self.latest.fiscal_year - 3 if self.latest else None
            return None, f"{REASON_HISTORY}:fy={fy}"
        base = _get(self.three_back.income, "revenue")
        if base is None:
            return None, f"{REASON_MISSING}:revenue_3y_ago"
        if base <= 0 or rev <= 0:
            return None, f"{REASON_DENOMINATOR}:revenue<=0"
        return (rev / base) ** (1.0 / 3.0) - 1.0, None

    def operating_income_growth_1y(self) -> _Value:
        return _growth(self, "operating_income")

    def eps_growth_1y(self) -> _Value:
        need = self._need_prior()
        if need:
            return None, need
        if self.inc("eps_diluted") is not None and self.prior_inc("eps_diluted") is not None:
            return _growth(self, "eps_diluted")
        if self.inc("eps_diluted") is None and self.prior_inc("eps_diluted") is None:
            self.flags.append("eps_growth_from_net_income")
            return _growth(self, "net_income")
        # One side has EPS and the other does not: mixing bases would
        # compare per-share with aggregate numbers.
        missing = "eps_diluted" if self.inc("eps_diluted") is None else "eps_diluted_prior"
        return None, f"{REASON_MISSING}:{missing}"

    def revenue_growth_accel(self) -> _Value:
        g1, reason = _growth(self, "revenue")
        if g1 is None:
            return None, reason
        if self.prior2 is None:
            fy = self.latest.fiscal_year - 2 if self.latest else None
            return None, f"{REASON_HISTORY}:fy={fy}"
        rev_prior = _get(self.prior.income, "revenue") if self.prior else None
        rev_prior2 = _get(self.prior2.income, "revenue")
        if rev_prior2 is None:
            return None, f"{REASON_MISSING}:revenue_2y_ago"
        if rev_prior is None:
            return None, f"{REASON_MISSING}:revenue_prior"
        if rev_prior2 <= 0:
            return None, f"{REASON_DENOMINATOR}:revenue_2y_ago<=0"
        return g1 - (rev_prior / rev_prior2 - 1.0), None

    # -- profitability -----------------------------------------------------

    def gross_margin(self) -> _Value:
        return _over_revenue(self, self.inc("gross_profit"), "gross_profit")

    def operating_margin(self) -> _Value:
        return _over_revenue(self, self.inc("operating_income"), "operating_income")

    def fcf_margin(self) -> _Value:
        return _over_revenue(self, self.cf("free_cash_flow"), "free_cash_flow")

    def operating_margin_change_1y(self) -> _Value:
        om, reason = self.operating_margin()
        if om is None:
            return None, reason
        need = self._need_prior()
        if need:
            return None, need
        assert self.prior is not None
        op_p, rev_p = _get(self.prior.income, "operating_income"), _get(self.prior.income, "revenue")
        if op_p is None:
            return None, f"{REASON_MISSING}:operating_income_prior"
        if rev_p is None:
            return None, f"{REASON_MISSING}:revenue_prior"
        if rev_p <= 0:
            return None, f"{REASON_DENOMINATOR}:revenue_prior<=0"
        return om - op_p / rev_p, None

    # -- efficiency --------------------------------------------------------

    def asset_turnover(self) -> _Value:
        rev = self.inc("revenue")
        if rev is None:
            return None, f"{REASON_MISSING}:revenue"
        return _over_avg_assets(self, rev)

    def opex_ratio(self) -> _Value:
        sga, rd = self.inc("sga"), self.inc("r_and_d")
        if sga is None:
            return None, f"{REASON_MISSING}:sga"
        if rd is None:
            # Not zero-filled. FMP reports 0 for names with no R&D line and
            # that 0 arrives as a real value; only a genuinely absent (or
            # provider-null) line reaches here, and that is unknown.
            return None, f"{REASON_MISSING}:r_and_d"
        return _over_revenue(self, sga + rd, "opex")

    def capex_intensity(self) -> _Value:
        capex = self.cf("capex")
        if capex is None:
            return None, f"{REASON_MISSING}:capex"
        return _over_revenue(self, abs(capex), "capex")

    # -- leverage ----------------------------------------------------------

    def net_debt_to_ebitda(self) -> _Value:
        if self.net_debt is None:
            return None, self.net_debt_reason or f"{REASON_MISSING}:total_debt"
        if self.ebitda is None:
            return None, f"{REASON_MISSING}:ebitda"
        if self.ebitda <= 0:
            return None, f"{REASON_DENOMINATOR}:ebitda<=0"
        return self.net_debt / self.ebitda, None

    def debt_to_equity(self) -> _Value:
        eq = self.bal("shareholders_equity")
        if self.total_debt is None:
            return None, self.total_debt_reason or f"{REASON_MISSING}:total_debt"
        if eq is None:
            return None, f"{REASON_MISSING}:shareholders_equity"
        if eq <= 0:
            return None, f"{REASON_DENOMINATOR}:shareholders_equity<=0"
        return self.total_debt / eq, None

    def interest_coverage(self) -> _Value:
        ebit = self.inc("ebit")
        if ebit is None:
            ebit = self.inc("operating_income")
            if ebit is not None:
                self.flags.append("ebit_from_operating_income")
        interest = self.inc("interest_expense")
        if ebit is None:
            return None, f"{REASON_MISSING}:ebit"
        if interest is None:
            return None, f"{REASON_MISSING}:interest_expense"
        if interest == 0:
            return None, f"{REASON_DENOMINATOR}:interest_expense=0"
        return min(ebit / abs(interest), INTEREST_COVERAGE_CAP), None

    def current_ratio(self) -> _Value:
        ca, cl = self.bal("current_assets"), self.bal("current_liabilities")
        if ca is None:
            return None, f"{REASON_MISSING}:current_assets"
        if cl is None:
            return None, f"{REASON_MISSING}:current_liabilities"
        if cl <= 0:
            return None, f"{REASON_DENOMINATOR}:current_liabilities<=0"
        return ca / cl, None

    # -- capital allocation ------------------------------------------------

    def shareholder_yield(self) -> _Value:
        div, buyback = self.cf("dividends_paid"), self.cf("share_repurchases")
        if div is None and buyback is None:
            return None, f"{REASON_MISSING}:dividends_paid,share_repurchases"
        # Both legs are required: a present partner line does not make the
        # absent one zero, and a yield built on one leg would be ranked
        # against names whose yield has both.
        if div is None:
            return None, f"{REASON_MISSING}:dividends_paid"
        if buyback is None:
            return None, f"{REASON_MISSING}:share_repurchases"
        return _over_mktcap(self, abs(div) + abs(buyback), "shareholder_returns")

    def net_share_change_1y(self) -> _Value:
        cur = self.inc("weighted_avg_shares_diluted")
        if cur is None:
            return None, f"{REASON_MISSING}:weighted_avg_shares_diluted"
        need = self._need_prior()
        if need:
            return None, need
        prev = self.prior_inc("weighted_avg_shares_diluted")
        if prev is None:
            return None, f"{REASON_MISSING}:weighted_avg_shares_diluted_prior"
        if prev <= 0:
            return None, f"{REASON_DENOMINATOR}:shares_prior<=0"
        return cur / prev - 1.0, None

    def sbc_to_revenue(self) -> _Value:
        return _over_revenue(self, self.cf("stock_based_compensation"), "stock_based_compensation")

    def goodwill_to_assets(self) -> _Value:
        gw, ta = self.bal("goodwill"), self.bal("total_assets")
        if gw is None:
            return None, f"{REASON_MISSING}:goodwill"
        if ta is None:
            return None, f"{REASON_MISSING}:total_assets"
        if ta <= 0:
            return None, f"{REASON_DENOMINATOR}:total_assets<=0"
        return gw / ta, None

    # -- earnings quality --------------------------------------------------

    def accruals_ratio(self) -> _Value:
        ni, cfo = self.inc("net_income"), self.cf("cash_from_operations")
        if ni is None:
            return None, f"{REASON_MISSING}:net_income"
        if cfo is None:
            return None, f"{REASON_MISSING}:cash_from_operations"
        return _over_avg_assets(self, ni - cfo)

    def cash_conversion(self) -> _Value:
        return _over_net_income(self, self.cf("cash_from_operations"), "cash_from_operations")

    def fcf_to_net_income(self) -> _Value:
        return _over_net_income(self, self.cf("free_cash_flow"), "free_cash_flow")


# ---------------------------------------------------------------------------
# Shared arithmetic helpers (all return (value, reason))
# ---------------------------------------------------------------------------

def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _get(d: Mapping[str, Any] | None, key: str) -> float | None:
    return _finite(d.get(key)) if d else None


def _finite_or_reason(value: float, name: str) -> _Value:
    if not math.isfinite(value):
        return None, f"{REASON_DENOMINATOR}:{name}_not_finite"
    return value, None


def _total_debt(balance: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:
    """``(total_debt, reason, flag)``: the reported ``total_debt`` line, else
    ``short_term_debt + long_term_debt`` — but only when both halves are
    reported (flagged ``total_debt_from_short_plus_long``). ``ratios.net_debt``
    and the screener treat a missing half as 0; the scorecard does not,
    because an invented 0 would be standardised and ranked as if observed
    (fs-v1 missing-input rule). ``reason`` names the absent line."""
    total = _get(balance, "total_debt")
    if total is not None:
        return total, None, None
    st, lt = _get(balance, "short_term_debt"), _get(balance, "long_term_debt")
    if st is None and lt is None:
        return None, f"{REASON_MISSING}:total_debt", None
    if st is None:
        return None, f"{REASON_MISSING}:short_term_debt", None
    if lt is None:
        return None, f"{REASON_MISSING}:long_term_debt", None
    return st + lt, None, "total_debt_from_short_plus_long"


def _ebitda(income: Mapping[str, Any], cash: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Reported EBITDA line first; else operating income plus D&A, but only
    when both are present — `ratios.ebitda` treats a missing D&A as zero,
    which would label plain operating income as EBITDA."""
    reported = _get(income, "ebitda")
    if reported is not None:
        return reported, None
    op, da = _get(income, "operating_income"), _get(cash, "depreciation_and_amortization")
    if op is None or da is None:
        return None, None
    return op + da, "ebitda_from_operating_income_plus_da"


def _over_mktcap(c: _Calc, numerator: float | None, name: str) -> _Value:
    if numerator is None:
        return None, f"{REASON_MISSING}:{name}"
    if c.price is None or c.price <= 0:
        return None, f"{REASON_MISSING}:price"
    if not c.shares:
        return None, f"{REASON_MISSING}:shares"
    if c.mktcap is None or c.mktcap <= 0:
        return None, f"{REASON_DENOMINATOR}:market_cap<=0"
    return numerator / c.mktcap, None


def _over_ev(c: _Calc, numerator: float) -> _Value:
    if c.mktcap is None:
        return _over_mktcap(c, numerator, "numerator")  # surfaces the price/shares reason
    if c.net_debt is None:
        return None, c.net_debt_reason or f"{REASON_MISSING}:total_debt"
    if c.ev is None or c.ev <= 0:
        return None, f"{REASON_DENOMINATOR}:enterprise_value<=0"
    return numerator / c.ev, None


def _over_revenue(c: _Calc, numerator: float | None, name: str) -> _Value:
    if numerator is None:
        return None, f"{REASON_MISSING}:{name}"
    rev = c.inc("revenue")
    if rev is None:
        return None, f"{REASON_MISSING}:revenue"
    if rev <= 0:
        return None, f"{REASON_DENOMINATOR}:revenue<=0"
    return numerator / rev, None


def _over_avg_assets(c: _Calc, numerator: float) -> _Value:
    if c.avg_assets is None:
        return None, f"{REASON_MISSING}:total_assets"
    if c.avg_assets <= 0:
        return None, f"{REASON_DENOMINATOR}:avg_total_assets<=0"
    return numerator / c.avg_assets, None


def _over_net_income(c: _Calc, numerator: float | None, name: str) -> _Value:
    if numerator is None:
        return None, f"{REASON_MISSING}:{name}"
    ni = c.inc("net_income")
    if ni is None:
        return None, f"{REASON_MISSING}:net_income"
    if ni <= 0:
        return None, f"{REASON_DENOMINATOR}:net_income<=0"
    ratio = numerator / ni
    return max(-CONVERSION_CLIP, min(CONVERSION_CLIP, ratio)), None


def _growth(c: _Calc, line: str) -> _Value:
    cur = c.inc(line)
    if cur is None:
        return None, f"{REASON_MISSING}:{line}"
    need = c._need_prior()
    if need:
        return None, need
    prev = c.prior_inc(line)
    if prev is None:
        return None, f"{REASON_MISSING}:{line}_prior"
    if prev <= 0:
        return None, f"{REASON_DENOMINATOR}:{line}_prior<=0"
    return cur / prev - 1.0, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _feature_functions(c: _Calc) -> dict[str, Callable[[], _Value]]:
    return {f.name: getattr(c, f.name) for f in FEATURE_SPEC}


def compute_features_detailed(
    snapshot: PitSnapshot,
    price_ctx: Mapping[str, Any] | None,
    sector: str | None,
    *,
    spec: tuple[FeatureSpec, ...] = FEATURE_SPEC,
) -> FeatureResult:
    """Evaluate every feature in ``spec`` for one ticker.

    ``sector`` is the provider string (``Company.sector``); it is normalised
    here and the applicability masks are applied on the canonical name.
    Excluded features are None with an ``excluded:`` reason and are
    reported in ``applicable`` so the normaliser does not count them
    against coverage.
    """
    canonical = normalize_sector(sector)
    applicable = applicable_features(canonical, spec)
    calc = _Calc(snapshot, price_ctx)
    calc.context["sector_raw"] = sector
    calc.context["sector"] = canonical
    if sector and canonical is None:
        calc.flags.append("sector_unmatched")

    fns = _feature_functions(calc)
    values: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for f in spec:
        if f.name not in applicable:
            values[f.name] = None
            reasons[f.name] = f"{REASON_EXCLUDED}:sector={canonical}"
            continue
        if snapshot.latest is None:
            values[f.name] = None
            reasons[f.name] = REASON_NO_SNAPSHOT
            continue
        value, reason = fns[f.name]()
        if value is not None and not math.isfinite(value):
            value, reason = None, f"{REASON_DENOMINATOR}:{f.name}_not_finite"
        values[f.name] = value
        if value is None:
            reasons[f.name] = reason or REASON_MISSING
    calc.context["flags"] = sorted(set(calc.flags))
    calc.context["pit_notes"] = dict(snapshot.notes)
    calc.context["pit_exclusions"] = list(snapshot.excluded_rows)
    return FeatureResult(values=values, reasons=reasons, context=calc.context, applicable=applicable)


def compute_features(
    snapshot: PitSnapshot,
    price_ctx: Mapping[str, Any] | None,
    sector: str | None,
) -> dict[str, float | None]:
    """Raw feature values keyed by spec name (None where not computable).
    ``compute_features_detailed`` carries the reasons and context."""
    return compute_features_detailed(snapshot, price_ctx, sector).values
