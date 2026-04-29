"""SEC EDGAR provider for filings and company facts (no API key required).

Uses the SEC submissions and companyfacts JSON endpoints. Requires a
descriptive User-Agent string; configured via SEC_USER_AGENT.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TIMEOUT = 10.0


class SECEdgarProvider:
    name: str = "sec_edgar"

    def __init__(self) -> None:
        self.user_agent = settings.sec_user_agent

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="No API key required; identifies via SEC_USER_AGENT.",
            capabilities=["filings"],
        )

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    def get_filings(self, ticker: str, *, cik: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        if not cik:
            return None
        try:
            cik_padded = str(cik).lstrip("0").zfill(10)
            with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                r = client.get(SUBMISSIONS_URL.format(cik=cik_padded))
                if r.status_code != 200:
                    return None
                data = r.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accs = recent.get("accessionNumber", [])
            primary = recent.get("primaryDocument", [])
            results: List[Dict[str, Any]] = []
            for form, date_, acc, doc in zip(forms, dates, accs, primary):
                if form not in ("10-K", "10-Q", "8-K"):
                    continue
                acc_no_hyphen = acc.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{acc_no_hyphen}/{doc}"
                results.append(dict(
                    type=form,
                    period_end=None,
                    filing_date=date_,
                    accession_number=acc,
                    url=url,
                    business_description=None,
                ))
                if len(results) >= 10:
                    break
            return results
        except Exception as exc:  # pragma: no cover
            log.warning("SEC fetch failed: %s", exc)
            return None

    # All other BaseProvider methods return None
    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]: return None
    def get_price_history(self, ticker: str, days: int = 252): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_news(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> List[str]: return []
