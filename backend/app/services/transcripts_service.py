"""Earnings transcript service."""
from __future__ import annotations

from .data_service import get_data_service


def get_transcripts(ticker: str) -> list[dict]:
    return get_data_service().get_earnings_transcripts(ticker) or []


def latest_transcript(ticker: str) -> dict | None:
    transcripts = get_transcripts(ticker)
    return transcripts[-1] if transcripts else None
