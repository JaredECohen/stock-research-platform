"""Filings service — fetch + chunk filings for retrieval."""
from __future__ import annotations

from typing import Dict, List

from .data_service import get_data_service


def get_filings(ticker: str) -> List[Dict]:
    return get_data_service().get_filings(ticker) or []


def latest_10k(ticker: str) -> Dict | None:
    for f in get_filings(ticker):
        if f.get("type") == "10-K":
            return f
    return None
