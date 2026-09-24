"""Wave 3E — self-historical valuation context for the Comps Analyst.

Today the Comps Analyst compares a target's metrics against a static peer
set. Wave 3E adds a second axis: each metric is also compared against the
target's *own* history (5y / 20-quarter rolling distribution). A stock can
look cheap vs. peers but expensive vs. its own history (or vice versa).
Surfacing both prevents single-axis valuation calls.

The lens is computed from `history_service.get_financial_history` (Wave 2)
for the long-format fundamentals + `market_data_service.get_price_series`
for historical prices. We pivot the long-format rows back into per-period
dicts and re-compute multiples using the same `ratios.py` definitions the
live `CompsRow` already uses, so historical and live numbers are
apples-to-apples.

No new LLM calls — pure deterministic plumbing on top of existing data.
"""
from __future__ import annotations

import logging
from statistics import median
from typing import Any

from ..schemas import CompsHistoryStats, CompsRow
from . import ratios as R

log = logging.getLogger(__name__)


# Same metric vocabulary as `compute_comps`; kept in lockstep so the two
# lenses are interpretable side-by-side on the memo card.
_METRICS = (
    "revenue_growth", "gross_margin", "operating_margin", "ebitda_margin",
    "roic", "pe", "ev_revenue", "ev_ebitda", "p_fcf", "fcf_yield",
)

# Metrics stored as fractions of 1 (a rate, not a multiple). Their gap to the
# own-history median is stated in percentage points: a relative change on a
# margin reads as points, and the saved META v1 comps narrative turned a 6.7%
# relative gap (41.44% vs 38.825%) into "6.7 percentage points" (it is 2.6).
PERCENT_POINT_METRICS = (
    "revenue_growth", "gross_margin", "operating_margin", "ebitda_margin",
    "roic", "fcf_yield",
)

# Line items we need from `history_service` to recompute every metric in
# `_METRICS`. Pre-flattened so the caller makes one trip.
_NEEDED_LINES = (
    # income
    "revenue", "gross_profit", "operating_income", "ebitda", "net_income",
    "r_and_d", "sga",
    # balance
    "shareholders_equity", "total_assets", "cash_and_equivalents",
    "short_term_investments", "short_term_debt", "long_term_debt",
    "total_debt",
    # cash flow
    "depreciation_and_amortization", "free_cash_flow",
)

# Multiples that need a market cap to compute. Shares × price = market cap.
_MARKET_CAP_METRICS = ("pe", "ev_revenue", "ev_ebitda", "p_fcf", "fcf_yield")


def _pivot_long_to_per_period(
    history: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Pivot the long-format `get_financial_history` output back to a list
    of per-period dicts keyed by `period_end`.

    Each dict carries every line item available for that period plus the
    period label / period_end / fiscal_year / fiscal_quarter. Periods
    without a `period_end` are dropped (we need a date to pair with prices).
    """
    by_period: dict[str, dict[str, Any]] = {}
    for line, rows in history.items():
        for r in rows or []:
            pe = r.get("period_end")
            if not pe:
                continue
            bucket = by_period.setdefault(pe, {
                "period_end": pe,
                "period": r.get("period"),
                "fiscal_year": r.get("fiscal_year"),
                "fiscal_quarter": r.get("fiscal_quarter"),
            })
            bucket[line] = r.get("value")
    rows = list(by_period.values())
    rows.sort(key=lambda r: r["period_end"])  # oldest first
    return rows


def _closing_price_for(
    rows: list[dict[str, Any]], target_date: str,
) -> float | None:
    """Pick the closing price on or just before `target_date`.

    `rows` is the chronologically-ordered output of
    `market_data_service.get_price_series`. Linear scan is fine — we only
    do it ~20 times per ticker.
    """
    if not rows or not target_date:
        return None
    chosen: float | None = None
    for r in rows:
        d = str(r.get("date") or "")
        if not d:
            continue
        if d <= target_date:
            close_raw = r.get("close")
            if close_raw is None:
                continue
            try:
                chosen = float(close_raw)
            except (TypeError, ValueError):
                continue
        else:
            break
    return chosen


def _recompute_per_period_row(
    period_row: dict[str, Any], market_cap: float | None,
) -> dict[str, float | None]:
    """Apply `ratios.py` definitions to a single per-period dict, returning
    a flat dict of every `_METRICS` value (None when inputs are missing)."""
    income = period_row
    balance = period_row
    cash_flow = period_row

    rg = None
    out: dict[str, float | None] = {}
    out["gross_margin"] = R.gross_margin(income)
    out["operating_margin"] = R.operating_margin(income)
    out["ebitda_margin"] = R.ebitda_margin(income, cash_flow)
    out["roic"] = R.roic(income, balance)
    out["revenue_growth"] = rg  # filled by the caller (needs prior period)
    if market_cap is not None and market_cap > 0:
        out["pe"] = R.pe_ratio(market_cap, income)
        out["ev_revenue"] = R.ev_revenue(market_cap, balance, income)
        out["ev_ebitda"] = R.ev_ebitda(market_cap, balance, income, cash_flow)
        out["p_fcf"] = R.p_fcf(market_cap, cash_flow)
        out["fcf_yield"] = R.fcf_yield(market_cap, cash_flow)
    else:
        for k in _MARKET_CAP_METRICS:
            out[k] = None
    return out


def _percentile_in(values: list[float], target: float) -> float:
    """Where `target` sits within `values`, 0 (lowest) → 1 (highest).

    Equal-or-below convention so a current value at the cohort max → 1.0.
    """
    if not values:
        return 0.5
    below = sum(1 for v in values if v <= target)
    return below / len(values)


def _interpretation_lines(
    own_median: dict[str, float | None],
    own_p25: dict[str, float | None],
    own_p75: dict[str, float | None],
    current_percentile: dict[str, float],
    current_vs_own_median: dict[str, float],
    lookback_label: str,
) -> str:
    out: list[str] = []
    pct = current_percentile.get("ev_ebitda")
    delta = current_vs_own_median.get("ev_ebitda")
    if pct is not None and delta is not None:
        if pct >= 0.75:
            out.append(
                f"EV/EBITDA at the {pct * 100:.0f}th percentile of its own {lookback_label} "
                f"range ({delta:+.0%} vs own median) — premium to history."
            )
        elif pct <= 0.25:
            out.append(
                f"EV/EBITDA at the {pct * 100:.0f}th percentile of its own {lookback_label} "
                f"range ({delta:+.0%} vs own median) — discount to history."
            )
        else:
            out.append(
                f"EV/EBITDA in the middle of its own {lookback_label} range "
                f"({delta:+.0%} vs own median)."
            )

    om_delta = current_vs_own_median.get("operating_margin")
    if om_delta is not None:
        if om_delta > 0.03:
            out.append("Operating margin running above its own multi-year median.")
        elif om_delta < -0.03:
            out.append("Operating margin below its own multi-year median.")

    rg_pct = current_percentile.get("revenue_growth")
    if rg_pct is not None:
        if rg_pct >= 0.75:
            out.append(f"Revenue growth in the top quartile of its own {lookback_label} range.")
        elif rg_pct <= 0.25:
            out.append(f"Revenue growth in the bottom quartile of its own {lookback_label} range.")

    if not out:
        out.append("Self-historical metrics broadly in line with own multi-year medians.")
    return " ".join(out)


def build_history_stats(
    ticker: str, target_row: CompsRow, *,
    lookback_quarters: int = 20,
    min_periods: int = 8,
) -> CompsHistoryStats | None:
    """Compute the target's own historical valuation/quality distribution.

    Returns None when fewer than `min_periods` usable periods are
    available (i.e., a name that recently IPO'd, or a sparse demo
    dataset). Otherwise: median / p25 / p75 / current percentile /
    current-vs-own-median for every metric in `_METRICS`, plus a short
    English `interpretation` string.

    Multiples that need a market cap are paired with the closing price
    on or just before each period's `period_end`, multiplied by the
    period's `weighted_avg_shares_diluted` if present (else falling back
    to the current `target_row.market_cap` × shares ratio when shares
    are unavailable historically — a defensible approximation for demo
    data with sparse balance-sheet history).
    """
    from ..services.history_service import get_financial_history
    from ..services.market_data_service import get_price_series

    history = get_financial_history(
        ticker, list(_NEEDED_LINES), limit=lookback_quarters,
    )
    period_rows = _pivot_long_to_per_period(history)
    if len(period_rows) < min_periods:
        return None

    # Pull a price series long enough to cover the oldest period_end.
    days = max(lookback_quarters * 95, 252)
    price_rows: list[dict[str, Any]] = []
    try:
        price_rows = get_price_series(ticker, days) or []
    except Exception as exc:  # pragma: no cover — diagnostic only
        log.warning("history price fetch failed for %s: %s", ticker, exc)

    # Estimate diluted shares from the live target_row when we don't have
    # period-level shares (demo annuals don't always carry shares_outstanding
    # at balance level). market_cap_now / price_now ≈ shares; safe approximation.
    fallback_shares: float | None = None
    if target_row.market_cap and price_rows:
        try:
            last_price = float(price_rows[-1].get("close") or 0)
            if last_price > 0:
                fallback_shares = target_row.market_cap / last_price
        except (TypeError, ValueError):
            fallback_shares = None

    # Recompute every period's metrics.
    per_period_metrics: list[dict[str, float | None]] = []
    last_revenue: float | None = None
    for row in period_rows:
        period_end = row["period_end"]
        # Market cap for this period.
        price = _closing_price_for(price_rows, period_end)
        shares = row.get("weighted_avg_shares_diluted") or fallback_shares
        market_cap: float | None = None
        if price and shares:
            market_cap = price * shares

        recomputed = _recompute_per_period_row(row, market_cap)
        # Revenue growth: needs prior period's revenue.
        cur_rev = row.get("revenue")
        if last_revenue is not None and cur_rev is not None and last_revenue != 0:
            recomputed["revenue_growth"] = (cur_rev - last_revenue) / abs(last_revenue)
        last_revenue = cur_rev
        per_period_metrics.append(recomputed)

    # Aggregate distribution stats per metric.
    own_median: dict[str, float | None] = {}
    own_p25: dict[str, float | None] = {}
    own_p75: dict[str, float | None] = {}
    current_percentile: dict[str, float] = {}
    current_vs_own_median: dict[str, float] = {}
    current_minus_own_median_pp: dict[str, float] = {}
    for metric in _METRICS:
        series = [
            v for m in per_period_metrics
            if (v := m.get(metric)) is not None
        ]
        # `pe` should drop sign-flip periods (negative net income makes P/E
        # meaningless) — `safe_div` already returned None for divide-by-zero,
        # but we still want to drop noisy negatives.
        if metric == "pe":
            series = [v for v in series if v is not None and v > 0]
        if len(series) < min_periods:
            own_median[metric] = None
            own_p25[metric] = None
            own_p75[metric] = None
            continue
        own_median[metric] = float(median(series))
        # p25 / p75 via index, since stdlib doesn't ship a quantiles fn pre-3.8.
        sorted_s = sorted(series)
        own_p25[metric] = sorted_s[int(0.25 * (len(sorted_s) - 1))]
        own_p75[metric] = sorted_s[int(0.75 * (len(sorted_s) - 1))]

        target_val = getattr(target_row, metric, None)
        if target_val is None:
            continue
        current_percentile[metric] = round(_percentile_in(series, target_val), 3)
        med_v = own_median[metric]
        if med_v is not None and metric in PERCENT_POINT_METRICS:
            # Defined even at a zero median, where the relative change is not.
            current_minus_own_median_pp[metric] = round((target_val - med_v) * 100, 2)
        if med_v is not None and med_v != 0:
            current_vs_own_median[metric] = round(
                (target_val - med_v) / abs(med_v), 3,
            )

    # Need at least one metric with usable distribution stats; otherwise
    # bailing out is more honest than emitting a sparse all-None block.
    if not any(v is not None for v in own_median.values()):
        return None

    lookback_periods = len(per_period_metrics)
    # Quarterly cadence (FY1Q4) → "Nq"; annual cadence ("FY2024" / "2024") → "Ny".
    is_annual = all(
        not row.get("fiscal_quarter") for row in period_rows[:5]
    )
    lookback_label = (
        f"{lookback_periods}y" if is_annual else f"{lookback_periods} quarters"
    )

    interpretation = _interpretation_lines(
        own_median, own_p25, own_p75,
        current_percentile, current_vs_own_median, lookback_label,
    )

    return CompsHistoryStats(
        lookback_periods=lookback_periods,
        lookback_label=lookback_label,
        own_median=own_median,
        own_p25=own_p25,
        own_p75=own_p75,
        current_percentile=current_percentile,
        current_vs_own_median=current_vs_own_median,
        current_minus_own_median_pp=current_minus_own_median_pp,
        interpretation=interpretation,
    )
