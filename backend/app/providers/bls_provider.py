"""BLS (Bureau of Labor Statistics) provider.

Wraps the BLS Public Data API v2 timeseries endpoint:
    https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}

Without a key the public endpoint accepts ~25 requests/day per IP. With
BLS_API_KEY registered (free), the cap lifts to 500 requests/day plus
multi-series batch support.

Series IDs follow BLS conventions:
    CUUR0000SA0          - All-items CPI, US city average, not seasonally adjusted
    CUUR0000SAF1         - Food at home CPI
    CES4348400001        - Truck transportation employment (CES)
    LAUST060000000000003 - California unemployment rate (LAUS)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)

BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
TIMEOUT = 12.0


class BLSProvider:
    name: str = "bls"

    def __init__(self) -> None:
        self.api_key = getattr(settings, "bls_api_key", "") or ""

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="Using BLS public endpoint (25/day cap); set BLS_API_KEY for 500/day."
            if not self.api_key else "",
            capabilities=["macro", "inflation", "labor"],
        )

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        if not series_id:
            return None
        end_year = date.today().year
        start_year = end_year - 6
        body: Dict[str, Any] = {
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self.api_key:
            body["registrationkey"] = self.api_key

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.post(
                    BASE_URL,
                    content=json.dumps(body),
                    headers={"Content-Type": "application/json"},
                )
                if r.status_code != 200:
                    return None
                data = r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("BLS fetch failed for %s: %s", series_id, exc)
            return None

        status = (data or {}).get("status")
        if status != "REQUEST_SUCCEEDED":
            log.debug("BLS request failed for %s: %s", series_id, status)
            return None

        results = (data.get("Results") or {}).get("series") or []
        if not results:
            return None

        series = results[0]
        raw_points = series.get("data") or []
        points: List[Dict[str, Any]] = []
        for row in raw_points:
            iso = _period_to_iso(row.get("year"), row.get("period"))
            if iso is None:
                continue
            points.append({"date": iso, "value": _coerce_float(row.get("value"))})
        points.sort(key=lambda p: p["date"])
        if not points:
            return None

        return {
            "series_id": series_id,
            "name": series_id,
            "units": "",
            "points": points,
        }

    # ------------------------------------------------------------------
    # BaseProvider stubs
    # ------------------------------------------------------------------

    def get_company_profile(self, ticker: str): return None
    def get_price_history(self, ticker: str, days: int = 252): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_filings(self, ticker: str): return None
    def get_news(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def list_tickers(self) -> List[str]: return []
    def list_macro_series(self) -> List[Dict[str, Any]]:
        from ..data_catalog import SERIES_REGISTRY
        return [
            {"series_id": s.series_id, "name": s.name, "units": s.units, "points": []}
            for s in SERIES_REGISTRY if s.source == "BLS"
        ]


def _period_to_iso(year: Any, period: Any) -> Optional[str]:
    """Convert BLS (year, period) tuples to ISO month-end dates.

    BLS periods: M01..M12 (monthly), Q01..Q04 (quarterly), A01 (annual),
    S01..S03 (semi-annual). We normalize all of these to the last day of
    the underlying month/quarter/year so they sort cleanly alongside
    daily series.
    """
    if not year or not period:
        return None
    try:
        y = int(year)
        p = str(period).upper()
    except (TypeError, ValueError):
        return None
    if p.startswith("M"):
        try:
            month = int(p[1:])
        except ValueError:
            return None
        if not 1 <= month <= 12:
            return None
        return _end_of_month(y, month)
    if p.startswith("Q"):
        try:
            q = int(p[1:])
        except ValueError:
            return None
        end_month = {1: 3, 2: 6, 3: 9, 4: 12}.get(q)
        if not end_month:
            return None
        return _end_of_month(y, end_month)
    if p in {"A01", "AN", "ANN"}:
        return f"{y:04d}-12-31"
    return None


def _end_of_month(year: int, month: int) -> str:
    from datetime import timedelta
    if month == 12:
        return f"{year:04d}-12-31"
    first_next = date(year, month + 1, 1)
    return (first_next - timedelta(days=1)).isoformat()


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", ".", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
