"""EIA (US Energy Information Administration) provider.

Two access paths:
  1. Keyed API v2 (https://api.eia.gov/v2). Used when EIA_API_KEY is set.
  2. Public dataset endpoints (no key required) as a fallback for the
     weekly storage snapshots and the WTI / Henry Hub spot prices.

Series IDs follow EIA conventions:
  - Petroleum & natural gas weekly: e.g. `PET.WCESTUS1.W`
  - Daily prices: e.g. `PET.RWTC.D`, `NG.RNGWHHD.D`
  - Electricity monthly: e.g. `ELEC.GEN.ALL-US-99.M`

Returns the same shape as FRED: `{series_id, name, units, points: [{date, value}]}`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)

V2_BASE = "https://api.eia.gov/v2"
TIMEOUT = 12.0


def _route_for_series(series_id: str) -> Optional[str]:
    """Map an EIA series_id to a v2 API route.

    EIA's v2 schema is route-based with the series identifier broken
    into `route/data/?facets[series][]=<id>` style requests. We use a
    pragmatic shortcut: peel off the first segment as the route.
    """
    if not series_id:
        return None
    prefix = series_id.split(".", 1)[0].upper()
    if prefix == "PET":
        return "petroleum/stoc/wstk"
    if prefix == "NG":
        return "natural-gas/stor/wkly"
    if prefix == "ELEC":
        return "electricity/electric-power-operational-data"
    return None


class EIAProvider:
    name: str = "eia"

    def __init__(self) -> None:
        self.api_key = getattr(settings, "eia_api_key", "") or ""

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,  # public endpoints work without a key
            healthy=True,
            notes="Using public EIA endpoints; set EIA_API_KEY for v2 API access."
            if not self.api_key else "",
            capabilities=["macro", "energy"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a time series by EIA series_id. Returns the canonical
        `{series_id, name, units, points}` shape used by all providers."""
        if not series_id:
            return None
        # Try the keyed v2 API when configured.
        if self.api_key:
            v2 = self._fetch_v2(series_id)
            if v2 is not None:
                return v2
        # Fall back to the known public snapshots.
        public = self._fetch_public(series_id)
        if public is not None:
            return public
        return None

    def get_petroleum_storage_snapshot(self) -> Optional[Dict[str, Any]]:
        """Latest US crude oil ending stocks (ex SPR) + 1-week & 1-year delta."""
        series = self.get_macro_series("PET.WCESTUS1.W")
        return self._snapshot_from_series(series, headline="US Crude Oil Inventories")

    def get_natgas_storage_snapshot(self) -> Optional[Dict[str, Any]]:
        """Latest working gas in underground storage (Lower 48)."""
        series = self.get_macro_series("NG.NW2_EPG0_SWO_R48_BCF.W")
        return self._snapshot_from_series(series, headline="US Natural Gas Storage")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_v2(self, series_id: str) -> Optional[Dict[str, Any]]:
        route = _route_for_series(series_id)
        if route is None:
            return None
        url = f"{V2_BASE}/{route}/data/"
        params = {
            "api_key": self.api_key,
            "frequency": "weekly" if series_id.endswith(".W") else
                         "daily" if series_id.endswith(".D") else "monthly",
            "data[0]": "value",
            "facets[series][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": "104",
        }
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(url, params=params)
                if r.status_code != 200:
                    return None
                data = r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("EIA v2 fetch failed for %s: %s", series_id, exc)
            return None
        rows = (((data or {}).get("response") or {}).get("data")) or []
        if not rows:
            return None
        points = list(reversed([
            {"date": str(row.get("period") or ""), "value": _coerce_float(row.get("value"))}
            for row in rows
            if row.get("period")
        ]))
        units = next((str(r.get("units") or "") for r in rows if r.get("units")), "")
        return {
            "series_id": series_id,
            "name": series_id,
            "units": units,
            "points": points,
        }

    def _fetch_public(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Public-endpoint fallback for the most-asked series.

        Avoids the API-key wall for the basic storage snapshots and the
        WTI / Henry Hub daily spot prices. Each returns the canonical
        `points` shape.
        """
        sid = series_id.upper()
        if sid == "PET.WCESTUS1.W":
            return self._fetch_public_petroleum_storage()
        if sid == "NG.NW2_EPG0_SWO_R48_BCF.W":
            return self._fetch_public_natgas_storage()
        if sid == "PET.RWTC.D":
            return self._fetch_public_wti()
        if sid == "NG.RNGWHHD.D":
            return self._fetch_public_henry_hub()
        return None

    def _fetch_public_petroleum_storage(self) -> Optional[Dict[str, Any]]:
        url = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
        try:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code != 200:
                    return None
        except Exception as exc:  # pragma: no cover
            log.debug("EIA public petroleum fetch failed: %s", exc)
            return None
        return self._parse_xls_two_column(
            r.content,
            series_id="PET.WCESTUS1.W",
            name="US Crude Oil Inventories",
            units="thousand barrels",
        )

    def _fetch_public_natgas_storage(self) -> Optional[Dict[str, Any]]:
        url = "https://ir.eia.gov/ngs/wngsr.xls"
        try:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code != 200:
                    return None
        except Exception as exc:  # pragma: no cover
            log.debug("EIA public natgas fetch failed: %s", exc)
            return None
        # Parsing the XLS varies sheet-to-sheet. Best-effort: try the
        # generic two-column parser; on failure return None so the
        # snapshot falls back to "unavailable".
        return self._parse_xls_two_column(
            r.content,
            series_id="NG.NW2_EPG0_SWO_R48_BCF.W",
            name="US Natural Gas Storage",
            units="Bcf",
        )

    def _fetch_public_wti(self) -> Optional[Dict[str, Any]]:
        url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
        try:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code != 200:
                    return None
        except Exception as exc:  # pragma: no cover
            log.debug("EIA public WTI fetch failed: %s", exc)
            return None
        return self._parse_xls_two_column(
            r.content,
            series_id="PET.RWTC.D",
            name="WTI Crude Oil Price",
            units="$/bbl",
        )

    def _fetch_public_henry_hub(self) -> Optional[Dict[str, Any]]:
        url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
        try:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code != 200:
                    return None
        except Exception as exc:  # pragma: no cover
            log.debug("EIA public Henry Hub fetch failed: %s", exc)
            return None
        return self._parse_xls_two_column(
            r.content,
            series_id="NG.RNGWHHD.D",
            name="Henry Hub Natural Gas Price",
            units="$/MMBtu",
        )

    # ------------------------------------------------------------------
    # XLS parser (date, value) — graceful no-op when openpyxl/xlrd missing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xls_two_column(
        content: bytes, *, series_id: str, name: str, units: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse an EIA legacy `.xls` (binary) workbook into points.

        EIA still publishes historical CSV/XLS snapshots in the legacy
        binary `.xls` format which requires the optional `xlrd` package.
        When xlrd isn't installed we degrade to None and the caller
        treats the snapshot as unavailable rather than throwing.
        """
        try:
            import io
            import xlrd  # type: ignore
        except ImportError:
            return None
        try:
            book = xlrd.open_workbook(file_contents=content)
            # EIA's legacy workbooks place the series on sheet index 1
            # ("Data 1") with a 3-row header.
            sheet = book.sheet_by_index(1) if book.nsheets > 1 else book.sheet_by_index(0)
            points: List[Dict[str, Any]] = []
            for row_idx in range(3, sheet.nrows):
                row = sheet.row(row_idx)
                if not row or len(row) < 2:
                    continue
                raw_date = row[0].value
                raw_value = row[1].value
                if raw_date in (None, ""):
                    continue
                try:
                    date_tuple = xlrd.xldate_as_tuple(raw_date, book.datemode)
                    iso = f"{date_tuple[0]:04d}-{date_tuple[1]:02d}-{date_tuple[2]:02d}"
                except Exception:
                    iso = str(raw_date)[:10]
                value = _coerce_float(raw_value)
                points.append({"date": iso, "value": value})
            if not points:
                return None
            return {
                "series_id": series_id,
                "name": name,
                "units": units,
                "points": points,
            }
        except Exception as exc:  # pragma: no cover
            log.debug("EIA XLS parse failed for %s: %s", series_id, exc)
            return None

    @staticmethod
    def _snapshot_from_series(
        series: Optional[Dict[str, Any]], *, headline: str,
    ) -> Optional[Dict[str, Any]]:
        if not series:
            return None
        points = series.get("points") or []
        if not points:
            return None
        latest = points[-1]
        value = latest.get("value")
        date = latest.get("date")
        prior_week = points[-2] if len(points) >= 2 else None
        prior_year = points[-53] if len(points) >= 53 else None
        prior_week_delta = (
            (value - prior_week["value"])
            if (value is not None and prior_week and prior_week.get("value") is not None)
            else None
        )
        prior_year_delta = (
            (value - prior_year["value"])
            if (value is not None and prior_year and prior_year.get("value") is not None)
            else None
        )
        return {
            "headline": headline,
            "series_id": series.get("series_id"),
            "units": series.get("units"),
            "latest": {"date": date, "value": value},
            "week_over_week_delta": prior_week_delta,
            "year_over_year_delta": prior_year_delta,
        }

    # ------------------------------------------------------------------
    # BaseProvider stub methods
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
        from ..data_catalog import by_category
        return [s.to_dict() for s in by_category("energy") if s.source == "EIA"]


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", "."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
