"""Fundamental Factor Scorecard — pure feature engine (``fs-v1``).

No DB, no I/O, no numpy. Two steps:

1. ``pit_snapshot(rows, as_of)`` turns long-format ``financial_periods``
   rows into a point-in-time view: only rows whose ``available_at`` is on
   or before ``as_of`` survive, grouped into annual points. ``fs-v1``
   ingests annual statements only, so "TTM" here is the latest annual row
   that was knowable at ``as_of``; quarterly rows are ignored and counted.
2. ``compute_features(snapshot, price_ctx, sector)`` evaluates every
   formula in ``scorecard_spec.FEATURE_SPEC`` and returns the raw (unsigned,
   unnormalised) values. Anything that cannot be computed is ``None`` with
   a reason — never a zero, never a neutral placeholder. The reasons are
   the product's "n/a because …" text, so they are stable strings.

Row contract (owned by the persistence slice, mirrored here so this
module can be tested without a database)::

    (statement, line_item, period, period_end, fiscal_year, fiscal_quarter, value, available_at)

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
from datetime import date, datetime
from statistics import stdev
from typing import Any

from app.finance import ratios
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
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return datetime.fromisoformat(str(x)).date()
    except ValueError:
        return None


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
    * ``fiscal_quarter`` set (1-4) → ``quarterly_ignored`` (fs-v1 is annual);
    * ``value`` None / non-finite → ``null_values_dropped``;
    * no fiscal year and no period_end to derive one from → ``rows_unusable``;
    * unknown statement name → ``rows_unusable``;
    * a second row for the same (year, statement, line) → last one wins,
      counted in ``duplicate_rows``.
    """
    notes: dict[str, int] = {
        "rows_seen": 0,
        "missing_available_at": 0,
        "pit_excluded": 0,
        "quarterly_ignored": 0,
        "null_values_dropped": 0,
        "rows_unusable": 0,
        "duplicate_rows": 0,
    }
    by_year: dict[int, dict[str, dict[str, float]]] = {}
    period_end_by_year: dict[int, date | None] = {}
    available_by_year: dict[int, date | None] = {}

    for row in rows:
        notes["rows_seen"] += 1
        statement, line_item, _period, period_end, fiscal_year, fiscal_quarter, value, available_at = row
        if statement not in _STATEMENTS or not line_item:
            notes["rows_unusable"] += 1
            continue
        if fiscal_quarter not in (None, 0):
            notes["quarterly_ignored"] += 1
            continue
        avail = _to_date(available_at)
        if avail is None:
            notes["missing_available_at"] += 1
            continue
        if avail > as_of:
            notes["pit_excluded"] += 1
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
        stmts = by_year.setdefault(fy, {s: {} for s in _STATEMENTS})
        if line_item in stmts[statement]:
            notes["duplicate_rows"] += 1
        stmts[statement][line_item] = float(value)
        prev_pe = period_end_by_year.get(fy)
        if pe is not None and (prev_pe is None or pe > prev_pe):
            period_end_by_year[fy] = pe
        elif fy not in period_end_by_year:
            period_end_by_year[fy] = pe
        prev = available_by_year.get(fy)
        available_by_year[fy] = avail if prev is None or avail > prev else prev

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
    return PitSnapshot(as_of=as_of, points=points, notes=notes)


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
        self.net_debt: float | None = None
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

        debt, debt_flag = _total_debt(balance)
        if debt_flag:
            self.flags.append(debt_flag)
        self.total_debt = debt
        cash = _get(balance, "cash_and_equivalents")
        sti = _get(balance, "short_term_investments")
        if cash is not None and sti is None:
            self.flags.append("short_term_investments_not_reported")
        self.net_debt = (debt - cash - (sti or 0.0)) if (debt is not None and cash is not None) else None
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
        sga = self.inc("sga")
        rd = self.inc("r_and_d")
        if sga is None:
            return None, f"{REASON_MISSING}:sga"
        if rd is None:
            # Many companies (retailers, utilities) have no R&D line at all;
            # FMP reports 0 for them, and a missing line next to a present
            # SG&A is far more likely "not reported" than "unknown". Flagged
            # so the row says so.
            self.flags.append("r_and_d_not_reported")
            rd = 0.0
        return _over_revenue(self, sga + rd, "opex")

    def capex_intensity(self) -> _Value:
        capex = self.cf("capex")
        if capex is None:
            return None, f"{REASON_MISSING}:capex"
        return _over_revenue(self, abs(capex), "capex")

    # -- leverage ----------------------------------------------------------

    def net_debt_to_ebitda(self) -> _Value:
        if self.net_debt is None:
            missing = "total_debt" if self.total_debt is None else "cash_and_equivalents"
            return None, f"{REASON_MISSING}:{missing}"
        if self.ebitda is None:
            return None, f"{REASON_MISSING}:ebitda"
        if self.ebitda <= 0:
            return None, f"{REASON_DENOMINATOR}:ebitda<=0"
        return self.net_debt / self.ebitda, None

    def debt_to_equity(self) -> _Value:
        eq = self.bal("shareholders_equity")
        if self.total_debt is None:
            return None, f"{REASON_MISSING}:total_debt"
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
        if div is None or buyback is None:
            # Same reasoning as R&D: a present partner line makes the
            # missing one "not reported" rather than unknown. Flagged.
            self.flags.append("dividends_paid_not_reported" if div is None else "share_repurchases_not_reported")
        total = abs(div or 0.0) + abs(buyback or 0.0)
        return _over_mktcap(self, total, "shareholder_returns")

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


def _total_debt(balance: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """`total_debt` when reported, else short + long. A missing partner in
    the split is treated as not reported (0) and flagged — the same rule
    `finance.ratios.invested_capital` and the screener use."""
    total = _get(balance, "total_debt")
    if total is not None:
        return total, None
    st, lt = _get(balance, "short_term_debt"), _get(balance, "long_term_debt")
    if st is None and lt is None:
        return None, None
    flag = None
    if st is None:
        flag = "short_term_debt_not_reported"
    elif lt is None:
        flag = "long_term_debt_not_reported"
    return (st or 0.0) + (lt or 0.0), flag


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
        missing = "total_debt" if c.total_debt is None else "cash_and_equivalents"
        return None, f"{REASON_MISSING}:{missing}"
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
    return FeatureResult(values=values, reasons=reasons, context=calc.context, applicable=applicable)


def compute_features(
    snapshot: PitSnapshot,
    price_ctx: Mapping[str, Any] | None,
    sector: str | None,
) -> dict[str, float | None]:
    """Raw feature values keyed by spec name (None where not computable).
    ``compute_features_detailed`` carries the reasons and context."""
    return compute_features_detailed(snapshot, price_ctx, sector).values
