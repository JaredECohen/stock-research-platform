"""GDELT news provider — broad international coverage, no API key.

Uses the public GDELT 2.0 DOC API (`https://api.gdeltproject.org/api/v2/doc/doc`).
Returns articles in the same shape every other news source emits:

    [
      {
        "title": "...",
        "url": "...",
        "source": "<publisher domain>",
        "published_at": "<ISO8601>",
        "summary": "<truncated GDELT snippet, if any>",
        "tickers": ["TICKER"],
      },
      ...
    ]

GDELT is most useful as a complement to FMP / Alpha Vantage for
geopolitical, supply-chain, and international stories that US-centric
financial feeds miss. Free, rate-limited but generous, and updates
roughly every 15 minutes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base import ProviderStatus

log = logging.getLogger(__name__)

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TIMEOUT = 12.0


class GDELTProvider:
    name: str = "gdelt"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="Public endpoint; no API key required.",
            capabilities=["news"],
        )

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        if not ticker:
            return None
        return self.search_news(query=ticker.upper(), tickers=[ticker.upper()])

    def search_news(
        self,
        *,
        query: str,
        tickers: Optional[List[str]] = None,
        limit: int = 25,
        max_age_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Run an arbitrary GDELT DOC query and return normalized articles.

        `query` follows GDELT's query syntax (free text, with optional
        operators like `domain:`, `sourcecountry:`, `sourcelang:`).
        For a ticker we wrap the symbol in a few sensible filters so we
        don't accidentally grab a stock ticker that's also a common
        English word ("AI", "BA", "T", "M").
        """
        gdelt_query = self._build_query(query, tickers=tickers)
        params = {
            "query": gdelt_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max(limit, 5), 100)),
            "sort": "datedesc",
        }
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.get(BASE_URL, params=params)
                if resp.status_code != 200:
                    log.debug("GDELT non-200: %s", resp.status_code)
                    return []
                try:
                    data = resp.json()
                except ValueError:
                    # GDELT sometimes returns HTML when overloaded.
                    return []
        except Exception as exc:  # pragma: no cover
            log.debug("GDELT fetch failed for %s: %s", gdelt_query, exc)
            return []

        articles_raw = data.get("articles") if isinstance(data, dict) else None
        if not articles_raw:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        out: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for art in articles_raw:
            if not isinstance(art, dict):
                continue
            url = (art.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            published = _parse_gdelt_ts(art.get("seendate"))
            if published is not None and published < cutoff:
                continue
            out.append({
                "title": (art.get("title") or "").strip(),
                "url": url,
                "source": (art.get("domain") or "").strip(),
                "published_at": published.isoformat() if published else None,
                "summary": (art.get("snippet") or "")[:400],
                "tickers": list(tickers) if tickers else [],
                "language": (art.get("language") or "").strip(),
                "source_country": (art.get("sourcecountry") or "").strip(),
                "tone": _coerce_float(art.get("tone")),
            })
        return out

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
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> List[str]: return []
    def list_macro_series(self) -> List[Dict[str, Any]]: return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_query(self, query: str, *, tickers: Optional[List[str]]) -> str:
        """Construct a GDELT query that is specific to a single ticker.

        For a plain ticker, GDELT will happily match the letters as words
        (e.g. `T` matches "tea"). To narrow down, we look up the company
        name from the companies table when available and combine
        `(ticker OR "Company Name")` plus a sourcelang filter to English.
        """
        symbol = (tickers[0] if tickers else query).upper()
        company_name = self._lookup_company_name(symbol)
        terms: List[str] = []
        if company_name:
            terms.append(f'"{company_name}"')
        # Always include the ticker in cashtag form so we catch "$AAPL" style.
        terms.append(f'"${symbol}"')
        if not company_name:
            # Fall back to bare query if we have no company name.
            terms.append(symbol)
        joined = " OR ".join(terms)
        return f"({joined}) sourcelang:eng"

    def _lookup_company_name(self, ticker: str) -> Optional[str]:
        try:
            from ..database import SessionLocal
            from ..models import Company
            with SessionLocal() as db:
                row = db.query(Company).filter(Company.ticker == ticker).one_or_none()
                if row is not None and row.company_name:
                    return row.company_name
        except Exception:
            return None
        return None


def _parse_gdelt_ts(token: Any) -> Optional[datetime]:
    """GDELT timestamps come as `YYYYMMDDTHHMMSSZ`."""
    if not token:
        return None
    s = str(token).strip()
    if len(s) < 15:
        return None
    try:
        return datetime(
            int(s[0:4]), int(s[4:6]), int(s[6:8]),
            int(s[9:11]), int(s[11:13]), int(s[13:15]),
            tzinfo=timezone.utc,
        )
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
