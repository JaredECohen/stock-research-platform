"""Read-only review of the curated screener universe.

The universe file (`data/sp500.json`: S&P 500 + curated extensions) is a
hand-reviewed static snapshot. It is deliberately NOT refreshed from an
external feed on a schedule — a constituent change silently rewriting
the file would re-tier companies, drop pinned auto-update names and
change what the screener shows with nobody having looked at it. So the
platform needs the opposite of automation: a cheap way to *see* that the
snapshot is old or has drifted, so an operator decides.

This module answers three questions and changes nothing:

  1. Is the file past its review cadence? (`_last_reviewed` +
     `_review_cadence_days`, stamped in the file itself.)
  2. Does the DB agree with the file? File tickers with no
     `auto_analysis` row mean the seeder hasn't run since the file
     changed; `auto_analysis` rows not in the file mean a stale tier the
     seeder would demote.
  3. Optionally, does the live FMP constituent feed differ from the
     file? Only on request (`compare_feed=True`), only when FMP is
     configured AND live data is enabled, and only through the existing
     provider — no new HTTP code, nothing persisted.

Nothing here writes to the DB or to the JSON files. The one mutating
path is `app.scripts.refresh_universe_lists`, which an operator runs by
hand.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ..config import settings
from ..seed_universe import UniverseFile, load_universe_file

log = logging.getLogger(__name__)

FEED_SOURCE = "fmp"


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def file_status(uf: Optional[UniverseFile] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Staleness of the universe file alone — no DB, no network.

    Cheap enough to embed in `/api/admin/cron-health`. A missing or
    unparseable `_last_reviewed` is reported as stale rather than as
    "unknown": the field exists precisely so that nobody has to guess,
    and a file without it has not been reviewed under this process.
    """
    uf = uf or load_universe_file()
    today = today or date.today()
    last = _parse_date(uf.last_reviewed)
    cadence = uf.review_cadence_days
    days_since = (today - last).days if last else None
    if last is None or cadence is None:
        stale = True
    else:
        stale = days_since > cadence
    return {
        "file": uf.path.name if uf.path else None,
        "last_reviewed": uf.last_reviewed,
        "review_cadence_days": cadence,
        "days_since_review": days_since,
        "stale": stale,
    }


def _db_view(file_tickers: List[str]) -> Dict[str, Any]:
    from ..database import SessionLocal
    from ..models import Company

    file_set = set(file_tickers)
    with SessionLocal() as db:
        rows = db.query(Company.ticker, Company.universe_tier).all()
    auto = {t for t, tier in rows if tier == "auto_analysis"}
    on_demand = sum(1 for _, tier in rows if tier == "analyzed_on_demand")
    return {
        "db": {"auto_analysis_count": len(auto), "on_demand_count": on_demand},
        "diff_vs_db": {
            # "Missing" means no `auto_analysis` row, not merely no row:
            # a file ticker sitting in the DB under another tier is just
            # as invisible to the screener, and the seeder fixes both the
            # same way (it re-tags on warm start).
            "missing_in_db": sorted(file_set - auto),
            "auto_analysis_not_in_file": sorted(auto - file_set),
        },
    }


def _feed_view(file_tickers: List[str]) -> Dict[str, Any]:
    """Diff the file against the live FMP constituent list, read-only.

    `removed` will always include the curated extensions (ADRs, semis)
    because they are not S&P 500 members by design — the point of the
    diff is to make an operator look at the list, not to be actioned
    blindly. Failures are reported in `error` rather than raised: this
    is a report, and a half-report with a reason beats a 500.
    """
    out: Dict[str, Any] = {
        "source": FEED_SOURCE, "fetched_at": None,
        "added": [], "removed": [], "error": None,
    }
    if not settings.fmp_api_key:
        out["error"] = "FMP_API_KEY is not set; the constituent feed was not queried"
        return out
    if not settings.enable_live_data:
        out["error"] = "ENABLE_LIVE_DATA is false; the constituent feed was not queried"
        return out
    from .data_service import get_data_service

    try:
        # Direct provider access on purpose: the capability chain is for
        # per-ticker lookups with fallbacks, and there is no second
        # source for index membership to fall back to.
        feed = get_data_service().fmp.get_sp500_constituents()
    except Exception as exc:  # network / parse failures inside the provider
        log.warning("universe review: FMP constituent fetch failed: %s", exc)
        out["error"] = f"FMP constituent fetch failed: {type(exc).__name__}"
        return out
    out["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    if not feed:
        out["error"] = (
            "FMP returned no constituents (the /stable/sp500-constituent "
            "endpoint requires the Premium tier)"
        )
        return out
    feed_set = {t.upper() for t in feed}
    file_set = set(file_tickers)
    out["added"] = sorted(feed_set - file_set)
    out["removed"] = sorted(file_set - feed_set)
    return out


def review_universe(compare_feed: bool = False, today: Optional[date] = None) -> Dict[str, Any]:
    """Full review report. Read-only; see the module docstring for shape."""
    uf = load_universe_file()
    report: Dict[str, Any] = file_status(uf, today=today)
    report.update({
        "ticker_count": len(uf.tickers),
        "auto_update_count": len(uf.auto_update),
    })
    report.update(_db_view(uf.tickers))
    report["feed"] = _feed_view(uf.tickers) if compare_feed else None
    return report
