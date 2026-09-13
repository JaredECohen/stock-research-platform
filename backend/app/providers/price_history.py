"""Shared daily-bar validation; never relabel a provider's price adjustment basis."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any


def history_start(end: date, days: int) -> date | None:
    """Existing trading-bar request convention with calendar-day slack."""
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        return None
    try:
        return end - timedelta(days=math.ceil(days * 1.6))
    except (OverflowError, ValueError):
        return None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def normalize_history(
    rows: list[Any],
    *,
    provider: str,
    ticker: str,
    start: date,
    end: date,
    log: logging.Logger,
) -> list[dict[str, Any]] | None:
    """Validate, deduplicate and sort without silently capping returned history.

    Dates outside the requested interval are retained: some providers ignore
    the filter or return full history. A coverage gap is evidence, not proof of
    a provider truncation (the company might have listed later).
    """
    by_date: dict[str, dict[str, Any]] = {}
    invalid = duplicates = conflicts = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        try:
            day = date.fromisoformat(str(row.get("date", ""))[:10]).isoformat()
        except ValueError:
            invalid += 1
            continue
        close = number(row.get("close"))
        if close is None or close <= 0:
            invalid += 1
            continue
        normalized = {"date": day}
        for key in ("open", "high", "low", "close", "adjusted_close", "volume"):
            normalized[key] = number(row.get(key))
        if normalized["adjusted_close"] is not None and normalized["adjusted_close"] <= 0:
            normalized["adjusted_close"] = None
        if day in by_date:
            duplicates += 1
            if by_date[day] != normalized:
                conflicts += 1
            continue
        by_date[day] = normalized
    if conflicts:
        # Conflicting overlapping pages cannot be resolved without guessing.
        log.warning("price_history_invalid provider=%s ticker=%s conflicting_dates=%d", provider, ticker, conflicts)
        return None
    result = [by_date[day] for day in sorted(by_date)]
    first = result[0]["date"] if result else None
    last = result[-1]["date"] if result else None
    log.info(
        "price_history_result provider=%s ticker=%s requested_start=%s requested_end=%s rows=%d first=%s last=%s invalid=%d duplicates=%d",
        provider,
        ticker,
        start,
        end,
        len(result),
        first,
        last,
        invalid,
        duplicates,
    )
    if (
        invalid
        or not result
        or date.fromisoformat(first) > start + timedelta(days=7)
        or date.fromisoformat(last) < end - timedelta(days=7)
    ):
        log.warning(
            "price_history_coverage provider=%s ticker=%s requested_start=%s requested_end=%s first=%s last=%s invalid=%d completeness=unverified",
            provider,
            ticker,
            start,
            end,
            first,
            last,
            invalid,
        )
    return result or None
