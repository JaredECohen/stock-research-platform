"""US Census Bureau provider.

Covers two endpoint families:
  - Monthly Advance Retail Trade Survey (MARTS) — NAICS-coded retail sales
    series like total retail (44X72), food & beverage stores (445),
    nonstore retailers / e-commerce (454), clothing (448), gasoline
    stations (447), food services (722).
  - Construction Put-In-Place (C30) — residential and nonresidential.

Series IDs in the registry follow a synthetic convention:
    MARTS_<NAICS>     -> MARTS time-series for NAICS code
    RESCONST_TOTAL    -> total private residential construction
    NONRESCONST_TOTAL -> total private nonresidential construction

Without a key the public Census API still works for low-volume requests.
A CENSUS_API_KEY lifts the rate cap and is recommended for any
production deployment.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)

TIMEOUT = 12.0
MARTS_BASE = "https://api.census.gov/data/timeseries/eits/marts"
C30_BASE = "https://api.census.gov/data/timeseries/eits/resconst"


class CensusProvider:
    name: str = "census"

    def __init__(self) -> None:
        self.api_key = getattr(settings, "census_api_key", "") or ""

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="Using public Census endpoints; set CENSUS_API_KEY for rate-cap lift."
            if not self.api_key else "",
            capabilities=["macro", "retail", "construction"],
        )

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        if not series_id:
            return None
        sid = series_id.upper().strip()
        if sid.startswith("MARTS_"):
            naics = sid.split("_", 1)[1]
            return self._fetch_marts(naics=naics, series_id=sid)
        if sid == "RESCONST_TOTAL":
            return self._fetch_construction(category="TOTAL", series_id=sid, residential=True)
        if sid == "NONRESCONST_TOTAL":
            return self._fetch_construction(category="TOTAL", series_id=sid, residential=False)
        return None

    # ------------------------------------------------------------------
    # MARTS — Monthly Advance Retail Trade Survey
    # ------------------------------------------------------------------

    def _fetch_marts(self, *, naics: str, series_id: str) -> Optional[Dict[str, Any]]:
        params = {
            "get": "cell_value,data_type_code,time_slot_id,error_data,category_code",
            "for": "us:*",
            "time": f"from 2018 to {date.today().year}",
            "category_code": naics,
            "data_type_code": "SM",  # seasonally adjusted, not annualized
        }
        if self.api_key:
            params["key"] = self.api_key
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(f"{MARTS_BASE}", params=params)
                if r.status_code != 200:
                    return None
                rows = r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("Census MARTS fetch failed for %s: %s", series_id, exc)
            return None
        if not rows or len(rows) < 2:
            return None
        header = rows[0]
        try:
            cell_idx = header.index("cell_value")
            time_idx = header.index("time")
        except ValueError:
            return None
        points: List[Dict[str, Any]] = []
        for row in rows[1:]:
            value = _coerce_float(row[cell_idx]) if cell_idx < len(row) else None
            period = row[time_idx] if time_idx < len(row) else None
            if not period:
                continue
            iso = _period_to_iso(period)
            if iso is None:
                continue
            points.append({"date": iso, "value": value})
        points.sort(key=lambda p: p["date"])
        if not points:
            return None
        return {
            "series_id": series_id,
            "name": f"MARTS NAICS {naics}",
            "units": "millions $",
            "points": points,
        }

    def _fetch_construction(
        self, *, category: str, series_id: str, residential: bool,
    ) -> Optional[Dict[str, Any]]:
        params = {
            "get": "cell_value,data_type_code,time_slot_id,error_data,category_code",
            "for": "us:*",
            "time": f"from 2018 to {date.today().year}",
            "category_code": category,
            "data_type_code": "ADJ" if residential else "ADJ",
        }
        if self.api_key:
            params["key"] = self.api_key
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(C30_BASE, params=params)
                if r.status_code != 200:
                    return None
                rows = r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("Census C30 fetch failed for %s: %s", series_id, exc)
            return None
        if not rows or len(rows) < 2:
            return None
        header = rows[0]
        try:
            cell_idx = header.index("cell_value")
            time_idx = header.index("time")
        except ValueError:
            return None
        points: List[Dict[str, Any]] = []
        for row in rows[1:]:
            value = _coerce_float(row[cell_idx]) if cell_idx < len(row) else None
            period = row[time_idx] if time_idx < len(row) else None
            if not period:
                continue
            iso = _period_to_iso(period)
            if iso is None:
                continue
            points.append({"date": iso, "value": value})
        points.sort(key=lambda p: p["date"])
        if not points:
            return None
        return {
            "series_id": series_id,
            "name": "Residential Construction" if residential else "Nonresidential Construction",
            "units": "millions $",
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
            for s in SERIES_REGISTRY if s.source == "Census"
        ]


def _period_to_iso(period: str) -> Optional[str]:
    """Convert Census EITS time labels to ISO dates.

    EITS publishes monthly periods as `YYYY-MM`. Returns the last day of
    the month for ordering compatibility with the rest of the platform.
    """
    if not period:
        return None
    s = str(period).strip()
    if len(s) == 7 and s[4] == "-":
        try:
            year = int(s[:4]); month = int(s[5:7])
            # End-of-month so sort order matches other monthly series.
            if month == 12:
                eom = date(year, 12, 31)
            else:
                next_m = date(year, month + 1, 1)
                from datetime import timedelta
                eom = next_m - timedelta(days=1)
            return eom.isoformat()
        except (TypeError, ValueError):
            return None
    if len(s) == 10:
        return s[:10]
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", ".", "X"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
