"""News + catalysts service."""
from __future__ import annotations

from typing import Dict, List

from .data_service import get_data_service


def get_news(ticker: str) -> List[Dict]:
    return get_data_service().get_news(ticker) or []
