"""Macro data service."""
from __future__ import annotations

from typing import Dict, List, Optional

from .data_service import get_data_service


def list_series() -> List[Dict]:
    return get_data_service().list_macro_series()


def get_series(series_id: str) -> Optional[Dict]:
    return get_data_service().get_macro_series(series_id)


def latest(series_id: str) -> Optional[float]:
    s = get_series(series_id)
    if not s or not s.get("points"):
        return None
    return s["points"][-1].get("value")


def macro_snapshot() -> Dict[str, float]:
    """Return a small dict of headline macro values for prompts/agents."""
    keys = [
        "FEDFUNDS", "DGS2", "DGS10", "CPIAUCSL", "CORESTICKM159SFRBATL",
        "PCEPI", "UNRATE", "DCOILWTICO", "BAMLH0A0HYM2",
    ]
    snapshot: Dict[str, float] = {}
    for k in keys:
        v = latest(k)
        if v is not None:
            snapshot[k] = v
    return snapshot
