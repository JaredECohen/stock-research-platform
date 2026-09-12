"""Filings service — fetch + chunk filings for retrieval."""
from __future__ import annotations

from .data_service import get_data_service


def get_filings(ticker: str) -> list[dict]:
    """Filings WITH document text. Expensive: up to ten body fetches."""
    return get_data_service().get_filings(ticker) or []


def get_filings_index(ticker: str) -> list[dict]:
    """Filing metadata only — accession numbers, types, dates, no text.

    What a change detector needs, at one provider read per ticker. See
    `DataService.get_filings_index`.
    """
    return get_data_service().get_filings_index(ticker) or []


def invalidate_filings_text(ticker: str) -> int:
    """Forget the cached bodies for `ticker` so the next full read refetches."""
    return get_data_service().invalidate_filings_text(ticker)


def latest_10k(ticker: str) -> dict | None:
    for f in get_filings(ticker):
        if f.get("type") == "10-K":
            return f
    return None
