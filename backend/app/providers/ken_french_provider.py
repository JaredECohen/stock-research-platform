"""Kenneth French Data Library provider.

Downloads the FF5 (market, size, value, profitability, investment) +
momentum factor return series straight from the Tuck data library, parses
the CSV, and caches the result in `provider_cache`. Returns the canonical
provider shape so the catalog service can fetch it like any other series:

    {series_id, name, units, points: [{date, value}]}

Series IDs (matching SERIES_REGISTRY entries we'll add in Phase 2A):
    KFR.MKT_RF.D    market excess return, daily
    KFR.SMB.D       small-minus-big, daily
    KFR.HML.D       high-minus-low (value), daily
    KFR.RMW.D       robust-minus-weak (profitability), daily
    KFR.CMA.D       conservative-minus-aggressive (investment), daily
    KFR.MOM.D       momentum, daily
    KFR.RF.D        risk-free, daily
    KFR.MKT_RF.M    same set at monthly frequency
    ...

Bundles:
    FF5_DAILY     downloads the FF5 daily zip and emits all 6 series
    FF5_MONTHLY   monthly equivalent
    MOM_DAILY     momentum daily zip
    MOM_MONTHLY   momentum monthly zip

Values are returned as **decimal** returns (e.g. 0.0012 = 12 bps), having
been divided by 100 from the raw CSV's percent values.
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import ProviderStatus

log = logging.getLogger(__name__)

TIMEOUT = 25.0

# Map our synthetic series_ids to (bundle_id, csv_column).
_BUNDLE_URLS: Dict[str, str] = {
    "FF5_DAILY":   "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "FF5_MONTHLY": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "MOM_DAILY":   "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "MOM_MONTHLY": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}

_SERIES_TO_BUNDLE: Dict[str, Tuple[str, str]] = {
    # FF5 columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    "KFR.MKT_RF.D": ("FF5_DAILY",   "Mkt-RF"),
    "KFR.SMB.D":    ("FF5_DAILY",   "SMB"),
    "KFR.HML.D":    ("FF5_DAILY",   "HML"),
    "KFR.RMW.D":    ("FF5_DAILY",   "RMW"),
    "KFR.CMA.D":    ("FF5_DAILY",   "CMA"),
    "KFR.RF.D":     ("FF5_DAILY",   "RF"),
    "KFR.MKT_RF.M": ("FF5_MONTHLY", "Mkt-RF"),
    "KFR.SMB.M":    ("FF5_MONTHLY", "SMB"),
    "KFR.HML.M":    ("FF5_MONTHLY", "HML"),
    "KFR.RMW.M":    ("FF5_MONTHLY", "RMW"),
    "KFR.CMA.M":    ("FF5_MONTHLY", "CMA"),
    "KFR.RF.M":     ("FF5_MONTHLY", "RF"),
    # Momentum
    "KFR.MOM.D":    ("MOM_DAILY",   "Mom"),
    "KFR.MOM.M":    ("MOM_MONTHLY", "Mom"),
}

# In-process cache of parsed bundles. The bundle CSV stays small (~ a few
# hundred KB even for daily) so caching the parsed dict per bundle ID
# avoids re-downloading + re-parsing on every series fetch.
_BUNDLE_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


class KenFrenchProvider:
    name: str = "ken-french"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="Public Tuck data library; no key required.",
            capabilities=["factor_returns"],
        )

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Return the FF5/momentum series in the standard provider shape.

        Routes through `_load_bundle` which downloads + parses the
        appropriate CSV zip on first hit, then caches the parsed payload
        for the lifetime of the process.
        """
        if series_id not in _SERIES_TO_BUNDLE:
            return None
        bundle_id, column = _SERIES_TO_BUNDLE[series_id]
        bundle = self._load_bundle(bundle_id)
        if not bundle:
            return None
        points = bundle.get(column) or []
        if not points:
            return None
        return {
            "series_id": series_id,
            "name": f"Ken French {column} ({bundle_id.replace('_', ' ').title()})",
            "units": "decimal_return",
            "points": points,
        }

    # ------------------------------------------------------------------
    # BaseProvider stubs (factor provider doesn't supply company data)
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
        return [
            {"series_id": sid, "name": f"Ken French {col}", "units": "decimal_return", "points": []}
            for sid, (_, col) in _SERIES_TO_BUNDLE.items()
        ]

    # ------------------------------------------------------------------
    # Bundle loader
    # ------------------------------------------------------------------

    def _load_bundle(self, bundle_id: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        cached = _BUNDLE_CACHE.get(bundle_id)
        if cached is not None:
            return cached
        # Persistent cache via provider_cache so different processes share
        # the parsed payload.
        try:
            from ..services import provider_cache
            persisted = provider_cache.get("factor_returns", bundle_id, ttl_seconds=7 * 86400)
        except Exception:
            persisted = None
        if isinstance(persisted, dict) and persisted:
            _BUNDLE_CACHE[bundle_id] = persisted
            return persisted

        parsed = self._download_and_parse(bundle_id)
        if not parsed:
            return None
        _BUNDLE_CACHE[bundle_id] = parsed
        try:
            from ..services import provider_cache
            provider_cache.put("factor_returns", bundle_id, parsed)
        except Exception:
            pass
        return parsed

    def _download_and_parse(self, bundle_id: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        url = _BUNDLE_URLS.get(bundle_id)
        if not url:
            return None
        try:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "MarketMosaic/1.0"})
                if resp.status_code != 200:
                    log.warning("Ken French download failed (%s): %s", url, resp.status_code)
                    return None
                payload = resp.content
        except Exception as exc:
            log.warning("Ken French download error (%s): %s", url, exc)
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                # Each Ken French zip contains a single CSV.
                inner_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".csv")), None,
                )
                if inner_name is None:
                    return None
                with zf.open(inner_name) as fh:
                    text = fh.read().decode("latin-1", errors="ignore")
        except (zipfile.BadZipFile, KeyError) as exc:
            log.warning("Ken French zip parse failed (%s): %s", bundle_id, exc)
            return None
        return _parse_ken_french_csv(text, monthly="MONTHLY" in bundle_id)


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_ken_french_csv(text: str, *, monthly: bool) -> Dict[str, List[Dict[str, Any]]]:
    """Parse a Ken French CSV into {column_name: [{date, value}, ...]}.

    Layout: a few free-text intro lines, then the header row whose first
    column is the date label and remaining columns are factor names. Data
    rows follow with date in column 0 and percent returns. The file may
    contain MULTIPLE tables (e.g. "Average Equal Weighted Returns —
    Annual" appended). We take the first table only — that's the daily /
    monthly factor return series.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # Header rows always start with at least one comma and contain a
        # known factor column (Mkt-RF, SMB, HML, RMW, CMA, RF, Mom).
        if "," not in line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if any(c in {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"} for c in cells):
            header_idx = i
            break
    if header_idx is None:
        return {}

    header_cells = [c.strip() for c in lines[header_idx].split(",")]
    # Date column is the unlabeled first column. Factor columns start at index 1.
    columns: List[str] = []
    for cell in header_cells[1:]:
        if cell:
            columns.append(cell)
        else:
            break

    out: Dict[str, List[Dict[str, Any]]] = {col: [] for col in columns}

    for line in lines[header_idx + 1:]:
        if not line.strip():
            # Blank line separates the daily/monthly table from the next.
            if any(out.values()):
                break
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2:
            continue
        iso = _coerce_date(cells[0], monthly=monthly)
        if iso is None:
            # Hitting a sub-table header (e.g. "Annual Factors") — stop.
            if any(out.values()):
                break
            continue
        for col_i, col_name in enumerate(columns, start=1):
            if col_i >= len(cells):
                continue
            raw = cells[col_i]
            value = _coerce_pct_to_decimal(raw)
            if value is None:
                continue
            out[col_name].append({"date": iso, "value": value})

    return out


def _coerce_date(token: str, *, monthly: bool) -> Optional[str]:
    """Ken French daily dates are YYYYMMDD; monthly are YYYYMM."""
    s = token.strip()
    if not s:
        return None
    if monthly and len(s) == 6:
        try:
            year = int(s[:4]); month = int(s[4:6])
            if not 1 <= month <= 12:
                return None
            # End-of-month so sort order matches the rest of our monthly series.
            if month == 12:
                eom = date(year, 12, 31)
            else:
                from datetime import timedelta
                eom = date(year, month + 1, 1) - timedelta(days=1)
            return eom.isoformat()
        except (TypeError, ValueError):
            return None
    if (not monthly) and len(s) == 8:
        try:
            year = int(s[:4]); month = int(s[4:6]); day = int(s[6:8])
            return date(year, month, day).isoformat()
        except (TypeError, ValueError):
            return None
    return None


def _coerce_pct_to_decimal(token: str) -> Optional[float]:
    s = token.strip()
    if not s or s in {"-99.99", "-999"}:
        return None
    try:
        return float(s) / 100.0
    except (TypeError, ValueError):
        return None
