"""FEAT-001 — the Fundamentals Explorer metric catalog.

Pure data + arithmetic: no DB, no numpy, no provider calls. The catalog
is the single place that says *what* a metric is (its inputs in
`FinancialPeriod.line_item` terms, the formula, the unit, the sign
convention) and *how honestly it was obtained* (reported line, derived
arithmetic, or market-derived at the fiscal period end). The series
service (`fundamentals_series_service`) walks this table; the API and
the frontend render it; the commentary prompt quotes `formula_text` so
the model never has to guess what a number means.

Research-process alignment (docs/research/README.md): observed data is
kept separate from interpretation. In machine form that is `kind`
(reported vs derived vs market) plus the per-point `estimated` flag,
which is set only on the documented fallbacks listed in each spec's
`provenance`. A value that cannot be computed is *never* zero or
neutral — `compute` returns `value=None` with a closed-set `reason`.

Expectations ledger (reported consensus / management guidance /
price-implied / our forecast — the four columns in the underwriting
memo) is deliberately **not** in v1. `RESERVED_SOURCE_KINDS` names the
`source_kind` values a later release will use; consensus and guidance
points would come from `FMPProvider` analyst estimates
(`/stable/analyst-estimates`) and price-implied / our-forecast points
from the stored memo's `mispricing_thesis`. Reserving the vocabulary
now means adding them later is additive on the wire.

`CATALOG_VERSION` is part of the series fingerprint and the commentary
cache key: bump it whenever a formula, an input list, a fallback rule or
a unit changes, so cached commentary written against the old arithmetic
is never served for the new one. `test_fundamentals_catalog` pins a
digest of the table so an edit without a bump fails loudly.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from ..finance import ratios as R

CATALOG_VERSION: Final = "2026.09.1"
FREQUENCY: Final = "annual"

# Staleness thresholds applied by the series service to a ticker's
# provenance. 15 months: an annual filer's next fiscal year end plus a
# generous filing lag — beyond that a newer 10-K almost certainly exists
# and has not been ingested. 400 days on `fetched_at`: the nightly sweep
# has not touched the ticker for over a year, whatever the periods say.
STALE_LAST_PERIOD_MONTHS: Final = 15
STALE_FETCHED_AT_DAYS: Final = 400

UnitType = Literal["currency", "percent", "ratio", "multiple", "count"]
MetricKind = Literal["reported", "derived", "market"]
PointReason = Literal[
    "base_nonpositive",         # growth: the prior-period base is <= 0
    "denominator_nonpositive",  # a ratio's denominator is <= 0
    "no_price",                 # market metric: no close on/before period end
    "no_shares",                # market metric: no diluted share count
    "missing_line",             # a required line item is absent for the period
    "not_backfilled",           # the ticker has no annual rows at all
]

UNIT_TYPES: Final[tuple[str, ...]] = ("currency", "percent", "ratio", "multiple", "count")
METRIC_KINDS: Final[tuple[str, ...]] = ("reported", "derived", "market")
POINT_REASONS: Final[tuple[str, ...]] = (
    "base_nonpositive", "denominator_nonpositive", "no_price", "no_shares",
    "missing_line", "not_backfilled",
)
FAMILIES: Final[tuple[str, ...]] = (
    "income", "growth", "margins", "cash_flow", "returns", "balance",
    "per_share", "valuation",
)
# Values a point's `source_kind` will take once the expectations ledger
# ships. v1 emits only what the catalog computes (`kind`); see module doc.
RESERVED_SOURCE_KINDS: Final[tuple[str, ...]] = (
    "consensus", "guidance", "price_implied", "our_forecast",
)

# The share-count line used for every market-derived metric. Not a
# catalog metric input in the ratio sense, but the series service must
# fetch it whenever any selected metric `requires_price`.
SHARES_LINE: Final = "weighted_avg_shares_diluted"

Row = Mapping[str, float | None]


@dataclass(frozen=True)
class MarketContext:
    """Market inputs at one fiscal period end, resolved by the series
    service. `price` is the close on or just before the period end (or
    None → `no_price`); `shares` is the period's diluted count, or the
    company's current share count when the period lacks one, in which
    case `shares_estimated` is True."""
    price: float | None
    shares: float | None
    shares_estimated: bool = False

    @property
    def market_cap(self) -> float | None:
        if self.price is None or self.shares is None:
            return None
        if self.price <= 0 or self.shares <= 0:
            return None
        return self.price * self.shares


@dataclass(frozen=True)
class Computed:
    """One evaluated point. `value is None` always comes with a `reason`."""
    value: float | None
    reason: str | None = None
    estimated: bool = False


ComputeFn = Callable[[Row, Row | None, MarketContext | None], Computed]


@dataclass(frozen=True)
class MetricSpec:
    id: str
    label: str
    family: str
    unit_type: str
    kind: str
    formula_text: str            # in FinancialPeriod.line_item terms
    inputs: tuple[str, ...]      # line items read; must be ingested by history_service
    provenance: str              # where the number comes from + documented fallbacks
    compute: ComputeFn
    sign_note: str | None = None
    requires_price: bool = False  # needs a period-end close (market kind)
    requires_prior: bool = False  # needs the previous fiscal year
    frequency: str = FREQUENCY


# ---------------------------------------------------------------------------
# Arithmetic helpers — every None carries a reason
# ---------------------------------------------------------------------------

MISSING = Computed(None, "missing_line")


def _get(row: Row | None, name: str) -> float | None:
    if row is None:
        return None
    v = row.get(name)
    return None if v is None else float(v)


def _reported(name: str, *, absolute: bool = False) -> ComputeFn:
    def fn(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
        v = _get(cur, name)
        if v is None:
            return MISSING
        return Computed(abs(v) if absolute else v)
    return fn


def _ratio(num: Computed, den: Computed) -> Computed:
    """num / den with the closed-set reasons. Denominator must be > 0:
    a margin on negative revenue or a multiple on a loss says nothing."""
    if num.value is None:
        return Computed(None, num.reason)
    if den.value is None:
        return Computed(None, den.reason)
    if den.value <= 0:
        return Computed(None, "denominator_nonpositive")
    return Computed(num.value / den.value, estimated=num.estimated or den.estimated)


def _line(cur: Row, name: str) -> Computed:
    v = _get(cur, name)
    return MISSING if v is None else Computed(v)


def _margin(num_name: str) -> ComputeFn:
    def fn(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
        return _ratio(_line(cur, num_name), _line(cur, "revenue"))
    return fn


def _yoy_growth(name: str) -> ComputeFn:
    def fn(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
        c = _get(cur, name)
        p = _get(prior, name)
        if c is None or p is None:
            return MISSING
        if p <= 0:
            return Computed(None, "base_nonpositive")
        return Computed((c - p) / p)
    return fn


def _ebitda(cur: Row) -> Computed:
    """Reported `ebitda` line first; else operating_income + D&A (the
    `ratios.ebitda` definition the comps engine uses) flagged estimated."""
    reported = _get(cur, "ebitda")
    if reported is not None:
        return Computed(reported)
    if _get(cur, "operating_income") is None:
        return MISSING
    derived = R.ebitda(dict(cur), dict(cur))
    if derived is None:
        return MISSING
    return Computed(derived, estimated=True)


def _free_cash_flow(cur: Row) -> Computed:
    """Reported `free_cash_flow`; else CFO − |capex| flagged estimated."""
    reported = _get(cur, "free_cash_flow")
    if reported is not None:
        return Computed(reported)
    cfo = _get(cur, "cash_from_operations")
    capex = _get(cur, "capex")
    if cfo is None or capex is None:
        return MISSING
    return Computed(cfo - abs(capex), estimated=True)


def _net_debt(cur: Row) -> Computed:
    """`ratios.net_debt` treats missing lines as zero, which is right for
    a comps table and wrong for a chart: a company with no cash line
    would plot as fully levered. Require a debt figure and a cash figure
    before delegating; the short+long fallback is flagged estimated."""
    total_debt = _get(cur, "total_debt")
    st, lt = _get(cur, "short_term_debt"), _get(cur, "long_term_debt")
    cash = _get(cur, "cash_and_equivalents")
    if cash is None:
        return MISSING
    estimated = False
    if total_debt is None:
        if st is None and lt is None:
            return MISSING
        estimated = True
    return Computed(R.net_debt(dict(cur)), estimated=estimated)


def _market_cap(mkt: MarketContext | None) -> Computed:
    if mkt is None or mkt.price is None:
        return Computed(None, "no_price")
    if mkt.shares is None:
        return Computed(None, "no_shares")
    mc = mkt.market_cap
    if mc is None:
        # Non-positive price or share count: treat as absent rather than
        # producing a negative multiple.
        return Computed(None, "no_shares" if mkt.shares <= 0 else "no_price")
    return Computed(mc, estimated=mkt.shares_estimated)


def _enterprise_value(cur: Row, mkt: MarketContext | None) -> Computed:
    mc = _market_cap(mkt)
    if mc.value is None:
        return mc
    nd = _net_debt(cur)
    if nd.value is None:
        return nd
    return Computed(mc.value + nd.value, estimated=mc.estimated or nd.estimated)


# --- per-metric compute functions ------------------------------------------

def _fcf_margin(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_free_cash_flow(cur), _line(cur, "revenue"))


def _fcf(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _free_cash_flow(cur)


def _fcf_after_sbc(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    fcf = _free_cash_flow(cur)
    sbc = _get(cur, "stock_based_compensation")
    if fcf.value is None:
        return fcf
    if sbc is None:
        return MISSING
    return Computed(fcf.value - abs(sbc), estimated=fcf.estimated)


def _ebitda_metric(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ebitda(cur)


def _net_debt_metric(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _net_debt(cur)


# `ratios.roic_with_provenance` labels → closed-set reasons. A loss-maker
# without a credible tax rate is "missing" the one input that would make
# the number real, not a zero and not a denominator problem.
_ROIC_REASONS: Final[dict[str, str]] = {
    R.ROIC_NO_OPERATING_INCOME: "missing_line",
    R.ROIC_NO_INVESTED_CAPITAL: "denominator_nonpositive",
    R.ROIC_UNKNOWN_LOSS_MAKER: "missing_line",
}


def _roic(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    value, label = R.roic_with_provenance(dict(cur), dict(cur))
    if value is None:
        return Computed(None, _ROIC_REASONS.get(label, "missing_line"))
    return Computed(value, estimated=(label == R.ROIC_STATUTORY_FALLBACK))


def _pe(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_market_cap(mkt), _line(cur, "net_income"))


def _ev_ebitda(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_enterprise_value(cur, mkt), _ebitda(cur))


def _ev_revenue(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_enterprise_value(cur, mkt), _line(cur, "revenue"))


def _p_fcf(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_market_cap(mkt), _free_cash_flow(cur))


def _fcf_yield(cur: Row, prior: Row | None, mkt: MarketContext | None) -> Computed:
    return _ratio(_free_cash_flow(cur), _market_cap(mkt))


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

_REPORTED_PROVENANCE = "Provider statement line as ingested by history_service; no fallback."
_FCF_FALLBACK = (
    "free_cash_flow line when reported; otherwise cash_from_operations − |capex| (estimated)."
)
_MARKET_PROVENANCE = (
    "Market cap = close on/before the fiscal period end (existing price history) × "
    "weighted_avg_shares_diluted for that period; when the period has no share count the "
    "company's current shares_outstanding is used (estimated). No price → no_price; no "
    "shares → no_shares."
)

_SPECS: tuple[MetricSpec, ...] = (
    # --- income --------------------------------------------------------------
    MetricSpec(
        id="revenue", label="Revenue", family="income", unit_type="currency", kind="reported",
        formula_text="revenue", inputs=("revenue",), provenance=_REPORTED_PROVENANCE,
        compute=_reported("revenue"),
    ),
    MetricSpec(
        id="net_income", label="Net income", family="income", unit_type="currency", kind="reported",
        formula_text="net_income", inputs=("net_income",), provenance=_REPORTED_PROVENANCE,
        compute=_reported("net_income"),
        sign_note="Negative = net loss.",
    ),
    MetricSpec(
        id="ebitda", label="EBITDA", family="income", unit_type="currency", kind="derived",
        formula_text="ebitda when reported; else operating_income + depreciation_and_amortization",
        inputs=("ebitda", "operating_income", "depreciation_and_amortization"),
        provenance="Reported ebitda line when present; otherwise derived (estimated) with the "
                   "same definition the comps engine uses (ratios.ebitda).",
        compute=_ebitda_metric,
    ),
    MetricSpec(
        id="eps_diluted", label="Diluted EPS", family="per_share", unit_type="currency", kind="reported",
        formula_text="eps_diluted", inputs=("eps_diluted",), provenance=_REPORTED_PROVENANCE,
        compute=_reported("eps_diluted"),
        sign_note="Per share, in the reporting currency. Negative = loss per share.",
    ),
    # --- growth ---------------------------------------------------------------
    MetricSpec(
        id="revenue_growth_yoy", label="Revenue growth (YoY)", family="growth", unit_type="percent",
        kind="derived",
        formula_text="(revenue − prior-year revenue) / prior-year revenue",
        inputs=("revenue",), requires_prior=True,
        provenance="Derived from two consecutive fiscal years of the reported revenue line; "
                   "the prior year must be > 0 (else base_nonpositive) and must be the "
                   "immediately preceding fiscal year (a gap → missing_line).",
        compute=_yoy_growth("revenue"),
    ),
    # --- margins --------------------------------------------------------------
    MetricSpec(
        id="gross_margin", label="Gross margin", family="margins", unit_type="percent", kind="derived",
        formula_text="gross_profit / revenue", inputs=("gross_profit", "revenue"),
        provenance="Derived from reported lines; revenue must be > 0.",
        compute=_margin("gross_profit"),
    ),
    MetricSpec(
        id="operating_margin", label="Operating margin", family="margins", unit_type="percent",
        kind="derived",
        formula_text="operating_income / revenue", inputs=("operating_income", "revenue"),
        provenance="Derived from reported lines; revenue must be > 0.",
        compute=_margin("operating_income"),
        sign_note="Negative = operating loss.",
    ),
    MetricSpec(
        id="net_margin", label="Net margin", family="margins", unit_type="percent", kind="derived",
        formula_text="net_income / revenue", inputs=("net_income", "revenue"),
        provenance="Derived from reported lines; revenue must be > 0.",
        compute=_margin("net_income"),
        sign_note="Negative = net loss.",
    ),
    MetricSpec(
        id="fcf_margin", label="FCF margin", family="margins", unit_type="percent", kind="derived",
        formula_text="free_cash_flow / revenue",
        inputs=("free_cash_flow", "cash_from_operations", "capex", "revenue"),
        provenance=f"{_FCF_FALLBACK} Revenue must be > 0.",
        compute=_fcf_margin,
    ),
    # --- cash flow ------------------------------------------------------------
    MetricSpec(
        id="operating_cash_flow", label="Operating cash flow", family="cash_flow", unit_type="currency",
        kind="reported",
        formula_text="cash_from_operations", inputs=("cash_from_operations",),
        provenance=_REPORTED_PROVENANCE, compute=_reported("cash_from_operations"),
    ),
    MetricSpec(
        id="capex", label="Capital expenditure", family="cash_flow", unit_type="currency", kind="reported",
        formula_text="|capex|", inputs=("capex",),
        provenance="Reported line. Providers store the outflow as a negative number; the "
                   "absolute value is shown.",
        compute=_reported("capex", absolute=True),
        sign_note="Shown as positive spend even though the statement reports an outflow.",
    ),
    MetricSpec(
        id="free_cash_flow", label="Free cash flow", family="cash_flow", unit_type="currency",
        kind="derived",
        formula_text="free_cash_flow when reported; else cash_from_operations − |capex|",
        inputs=("free_cash_flow", "cash_from_operations", "capex"),
        provenance=_FCF_FALLBACK, compute=_fcf,
        sign_note="Negative = cash burn.",
    ),
    MetricSpec(
        id="fcf_after_sbc", label="FCF after stock comp", family="cash_flow", unit_type="currency",
        kind="derived",
        formula_text="free_cash_flow − |stock_based_compensation|",
        inputs=("free_cash_flow", "cash_from_operations", "capex", "stock_based_compensation"),
        provenance=f"{_FCF_FALLBACK} stock_based_compensation is a reported line; missing → "
                   "missing_line rather than treating SBC as zero.",
        compute=_fcf_after_sbc,
        sign_note="The owner-economic FCF convention: cash left after paying employees in "
                  "stock. Negative = the business does not fund itself once dilution is a cost.",
    ),
    # --- returns --------------------------------------------------------------
    MetricSpec(
        id="roic", label="ROIC", family="returns", unit_type="percent", kind="derived",
        formula_text="operating_income × (1 − tax rate) / (total_debt + shareholders_equity)",
        inputs=("operating_income", "pretax_income", "tax_expense", "total_debt",
                "short_term_debt", "long_term_debt", "shareholders_equity"),
        provenance="Delegates to finance.ratios.roic_with_provenance: effective tax rate "
                   "(tax_expense / pretax_income) when credible; the 21% statutory rate for a "
                   "profitable year without one (estimated); None for a loss-maker without a "
                   "credible rate (missing_line — the rate is the missing input). Invested "
                   "capital ≤ 0 → denominator_nonpositive.",
        compute=_roic,
    ),
    # --- balance --------------------------------------------------------------
    MetricSpec(
        id="net_debt", label="Net debt", family="balance", unit_type="currency", kind="derived",
        formula_text="total_debt − (cash_and_equivalents + short_term_investments)",
        inputs=("total_debt", "short_term_debt", "long_term_debt", "cash_and_equivalents",
                "short_term_investments"),
        provenance="ratios.net_debt on reported lines. total_debt missing → short_term_debt + "
                   "long_term_debt (estimated); no cash line → missing_line.",
        compute=_net_debt_metric,
        sign_note="Negative = net cash.",
    ),
    MetricSpec(
        id="shares_diluted", label="Diluted shares", family="per_share", unit_type="count", kind="reported",
        formula_text="weighted_avg_shares_diluted", inputs=("weighted_avg_shares_diluted",),
        provenance=_REPORTED_PROVENANCE, compute=_reported("weighted_avg_shares_diluted"),
    ),
    # --- valuation (market-derived at each fiscal period end) ----------------
    MetricSpec(
        id="pe_ttm", label="P/E (at fiscal year end)", family="valuation", unit_type="multiple",
        kind="market", requires_price=True,
        formula_text="market cap at period end / net_income",
        inputs=("net_income",),
        provenance=f"{_MARKET_PROVENANCE} net_income ≤ 0 → denominator_nonpositive (the "
                   "comps engine drops negative P/E for the same reason).",
        compute=_pe,
        sign_note="Price at the fiscal year end on that year's earnings — trailing twelve "
                  "months as of that date, not today's price.",
    ),
    MetricSpec(
        id="ev_ebitda", label="EV / EBITDA", family="valuation", unit_type="multiple", kind="market",
        requires_price=True,
        formula_text="(market cap at period end + net_debt) / ebitda",
        inputs=("ebitda", "operating_income", "depreciation_and_amortization", "total_debt",
                "short_term_debt", "long_term_debt", "cash_and_equivalents",
                "short_term_investments"),
        provenance=f"{_MARKET_PROVENANCE} EBITDA and net debt follow their own fallbacks; "
                   "ebitda ≤ 0 → denominator_nonpositive.",
        compute=_ev_ebitda,
    ),
    MetricSpec(
        id="ev_revenue", label="EV / Revenue", family="valuation", unit_type="multiple", kind="market",
        requires_price=True,
        formula_text="(market cap at period end + net_debt) / revenue",
        inputs=("revenue", "total_debt", "short_term_debt", "long_term_debt",
                "cash_and_equivalents", "short_term_investments"),
        provenance=f"{_MARKET_PROVENANCE} Net debt follows its own fallback; revenue must be > 0.",
        compute=_ev_revenue,
    ),
    MetricSpec(
        id="p_fcf", label="Price / FCF", family="valuation", unit_type="multiple", kind="market",
        requires_price=True,
        formula_text="market cap at period end / free_cash_flow",
        inputs=("free_cash_flow", "cash_from_operations", "capex"),
        provenance=f"{_MARKET_PROVENANCE} {_FCF_FALLBACK} FCF ≤ 0 → denominator_nonpositive.",
        compute=_p_fcf,
    ),
    MetricSpec(
        id="fcf_yield", label="FCF yield", family="valuation", unit_type="percent", kind="market",
        requires_price=True,
        formula_text="free_cash_flow / market cap at period end",
        inputs=("free_cash_flow", "cash_from_operations", "capex"),
        provenance=f"{_MARKET_PROVENANCE} {_FCF_FALLBACK}",
        compute=_fcf_yield,
        sign_note="Negative = cash burn relative to market value.",
    ),
)

CATALOG: Final[dict[str, MetricSpec]] = {s.id: s for s in _SPECS}


def metric_ids() -> tuple[str, ...]:
    return tuple(CATALOG)


def get(metric_id: str) -> MetricSpec:
    """Raise on an unknown id — a route maps this to 422; the series
    service must never silently drop a metric the caller asked for."""
    try:
        return CATALOG[metric_id]
    except KeyError:
        raise ValueError(f"unknown metric {metric_id!r}") from None


def required_line_items(metric_ids_: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Union of every selected metric's inputs, plus the share-count
    line when any metric needs a market cap. Order is stable (first
    appearance) so the SQL `IN (...)` is deterministic."""
    seen: dict[str, None] = {}
    for mid in metric_ids_:
        spec = get(mid)
        for line in spec.inputs:
            seen.setdefault(line, None)
        if spec.requires_price:
            seen.setdefault(SHARES_LINE, None)
    return tuple(seen)


def compute(
    metric_id: str, current: Row, prior: Row | None = None,
    market: MarketContext | None = None,
) -> Computed:
    """Evaluate one metric at one fiscal period."""
    spec = get(metric_id)
    out = spec.compute(current, prior, market)
    if out.value is None and out.reason is None:  # pragma: no cover — contract guard
        raise RuntimeError(f"{metric_id}: None value without a reason")
    return out


def catalog_entries() -> list[dict[str, object]]:
    """The wire shape of the catalog (`MetricSpecOut` in schemas)."""
    return [
        {
            "id": s.id,
            "label": s.label,
            "family": s.family,
            "unit_type": s.unit_type,
            "kind": s.kind,
            "formula_text": s.formula_text,
            "inputs": list(s.inputs),
            "sign_note": s.sign_note,
            "provenance": s.provenance,
            "requires_price": s.requires_price,
            "requires_prior": s.requires_prior,
            "frequency": s.frequency,
        }
        for s in _SPECS
    ]


def catalog_digest() -> str:
    """A short hash over everything that changes a computed value. The
    snapshot test pins it next to CATALOG_VERSION so the two move
    together."""
    import hashlib
    import json

    blob = json.dumps(
        [
            [s.id, s.unit_type, s.kind, s.formula_text, list(s.inputs), s.requires_price,
             s.requires_prior, s.frequency]
            for s in _SPECS
        ],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
