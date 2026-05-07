"""Wave 10 — cycle-position fingerprinting.

For each ticker, classify the current operating margin as **peak /
normal / trough** relative to its own 5-year history. Used by the
DCF default builder + the risk agent to flag cyclical names that
look cheap on trailing earnings but are at the cycle peak.

Heuristic:
- Pull the last 5 years (20 quarters) of operating margin from
  `financial_periods`.
- Compute trailing-3-year median + 5-year max + 5-year min.
- Current margin > 5-year max * 0.95: peak.
- Current margin < 5-year min * 1.05: trough.
- Otherwise: normal.

A cyclical at peak should reasonably revert to the cohort or 5-year
median over the DCF forecast horizon. A cyclical at trough may
warrant a forecast above current.

Read-only against `financial_periods`; never writes.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, Literal, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import FinancialPeriod

log = logging.getLogger(__name__)


CyclePosition = Literal["peak", "normal", "trough", "unknown"]


def _quarterly_op_margin_series(ticker: str, *, lookback_periods: int = 20) -> list[float]:
    """Pull the trailing N quarters of operating margin for a ticker.

    Reads `financial_periods` long-format rows; pivots `operating_income`
    and `revenue` per period and divides. Returns oldest → newest.
    Empty list when there's not enough data.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    with SessionLocal() as db:
        stmt = (
            select(FinancialPeriod)
            .where(
                FinancialPeriod.ticker == ticker.upper(),
                FinancialPeriod.statement == "income",
                FinancialPeriod.line_item.in_(
                    ["operating_income", "revenue"]
                ),
            )
            .order_by(FinancialPeriod.period.desc())
            .limit(lookback_periods * 2 * 3)  # over-fetch to be safe
        )
        for r in db.execute(stmt).scalars().all():
            entry = rows.setdefault(r.period, {})
            entry[r.line_item] = r.value
    margins: list[float] = []
    for period in sorted(rows.keys()):
        e = rows[period]
        revenue = e.get("revenue")
        op = e.get("operating_income")
        if revenue and op is not None and revenue > 0:
            margins.append(float(op) / float(revenue))
    return margins[-lookback_periods:]


def cycle_position(ticker: str) -> Dict[str, Any]:
    """Classify current operating margin position vs 5y history.

    Returns: {
        position: "peak" | "normal" | "trough" | "unknown",
        current: float | None,
        median_5y: float | None,
        max_5y: float | None,
        min_5y: float | None,
        n: int,
        rationale: str
    }
    """
    series = _quarterly_op_margin_series(ticker, lookback_periods=20)
    if len(series) < 4:
        return {
            "ticker": ticker.upper(),
            "position": "unknown",
            "n": len(series),
            "rationale": "Insufficient operating-margin history (need ≥4 quarters).",
        }
    current = series[-1]
    median_5y = statistics.median(series)
    max_5y = max(series)
    min_5y = min(series)
    position: CyclePosition = "normal"
    if current >= max_5y * 0.95 and current > median_5y:
        position = "peak"
    elif current <= min_5y * 1.05 and current < median_5y:
        position = "trough"
    rationale = (
        f"Current op margin {current*100:.1f}%; "
        f"5y median {median_5y*100:.1f}%, range "
        f"{min_5y*100:.1f}%–{max_5y*100:.1f}%."
    )
    return {
        "ticker": ticker.upper(),
        "position": position,
        "current": current,
        "median_5y": median_5y,
        "max_5y": max_5y,
        "min_5y": min_5y,
        "n": len(series),
        "rationale": rationale,
    }
