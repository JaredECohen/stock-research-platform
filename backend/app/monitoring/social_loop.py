"""Social monitoring loop — daily per-ticker sentiment scalar."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..agents import social_agent
from ..services.data_service import get_data_service
from . import record_run

log = logging.getLogger(__name__)


def run_once(tickers: Optional[Iterable[str]] = None) -> List[dict]:
    if tickers is None:
        ds = get_data_service()
        tickers = list(ds.list_tickers())[:10]
    out: List[dict] = []
    for t in tickers:
        try:
            payload = social_agent.run(t, force_refresh=True)
            out.append({"ticker": t, "extremity": payload.get("sentiment_extremity")})
        except Exception as exc:  # pragma: no cover
            log.warning("social_agent failed for %s: %s", t, exc)
    record_run("social_loop", note=f"{len(out)} tickers")
    return out


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", days=1, id="social_loop", replace_existing=True)
