"""Earnings transcript service."""
from __future__ import annotations

from typing import Dict, List, Optional

from .data_service import get_data_service


def get_transcripts(ticker: str) -> List[Dict]:
    return get_data_service().get_earnings_transcripts(ticker) or []


def latest_transcript(ticker: str) -> Optional[Dict]:
    transcripts = get_transcripts(ticker)
    return transcripts[-1] if transcripts else None
