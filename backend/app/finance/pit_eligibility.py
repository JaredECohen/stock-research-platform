"""Pure date eligibility and lossless row diagnostics for strict PIT reads."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any


def as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except (ValueError, TypeError):
        return None


def date_exclusion_reasons(period_end: Any, available_at: Any, as_of: date) -> tuple[str, ...]:
    """Unknown dates are not inferred; impossible availability is never admitted.

    Reasons can overlap. A future-ended row with prematurely recorded availability
    is both future data and invalid provenance, even once its period has ended.
    """
    end, available = as_date(period_end), as_date(available_at)
    reasons = []
    if available is None:
        reasons.append("missing_available_at")
    if end is None:
        reasons.append("missing_period_end")
    if available is not None and end is not None and available < end:
        reasons.append("available_before_period_end")
    if end is not None and end > as_of:
        reasons.append("period_end_after_as_of")
    if available is not None and available > as_of:
        reasons.append("available_after_as_of")
    return tuple(reasons)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # preserve the anomalous value without invalid JSON
    return value


def exclusion_record(row: Mapping[str, Any], reasons: tuple[str, ...], as_of: date) -> dict[str, Any]:
    """Preserve every supplied identity/value/date; absent caller IDs stay explicit."""
    return {"id": None, "ticker": None, "source": None, "currency": None,
            "available_at_source": None, "fetched_at": None,
            **{key: _json_value(value) for key, value in row.items()},
            "as_of": as_of.isoformat(), "reasons": list(reasons)}
