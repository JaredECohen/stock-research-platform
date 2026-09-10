"""Phase 6 — one-off / repeatable point-in-time backfills for the scorecard.

Two idempotent passes, both safe to run any number of times and both
no-ops when there is nothing to do:

- `--available-at`: fill NULL `financial_periods.available_at` for rows
  ingested before the column existed (`scorecard_pit.backfill_available_at`).
  Rows that already carry a date are never touched, so restated figures keep
  their original availability.
- `--prices`: populate `price_month_ends` from the 252-day price series the
  app already caches (`scorecard_pit.sync_price_month_ends`), one ticker at a
  time. Reads through `data_service`, so it goes through `provider_cache`
  and the normal provider chain and costs no extra calls when the series is
  warm.

With neither flag both passes run. `--tickers` restricts either pass;
without it the price pass covers the curated `auto_analysis` tier (the
scorecard universe) and the availability pass covers every ticker with a
NULL row.

Usage (from `backend/`)::

    python -m app.scripts.scorecard_backfill
    python -m app.scripts.scorecard_backfill --prices --tickers NVDA,MSFT
    python -m app.scripts.scorecard_backfill --available-at

Exit status is 1 only when a per-ticker price sync raised; an empty
universe or zero NULL rows is a successful no-op, not an error. The worker
runs the same two functions as `run_kind="pit_prepare"` before a scorecard
backfill; this CLI exists for operators and for environments where the
worker is not scheduled.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from sqlalchemy import select

from ..agents.log_safety import safe_exc
from ..database import SessionLocal
from ..models import Company
from ..services import scorecard_pit

log = logging.getLogger(__name__)


def _parse_tickers(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return sorted({t.strip().upper() for t in raw.split(",") if t.strip()})


def _universe_tickers() -> list[str]:
    with SessionLocal() as db:
        rows = db.execute(
            select(Company.ticker).where(Company.universe_tier == "auto_analysis")
        ).all()
    return sorted(r[0] for r in rows)


def run_available_at(tickers: list[str] | None) -> dict[str, Any]:
    return scorecard_pit.backfill_available_at(tickers=tickers)


def run_prices(tickers: list[str] | None, *, days: int = 252) -> dict[str, Any]:
    """Sync month-end closes per ticker; one bad ticker never stops the rest."""
    universe = tickers if tickers is not None else _universe_tickers()
    totals: dict[str, Any] = {"tickers": len(universe), "months": 0, "written": 0, "errors": 0}
    for t in universe:
        try:
            res = scorecard_pit.sync_price_month_ends(t, days=days)
            totals["months"] += res["months"]
            totals["written"] += res["written"]
        except Exception as exc:
            totals["errors"] += 1
            log.warning("scorecard_backfill: price sync failed for %s: %s", t, safe_exc(exc))
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--available-at", action="store_true",
                        help="Fill NULL financial_periods.available_at only.")
    parser.add_argument("--prices", action="store_true",
                        help="Sync price_month_ends only.")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers to restrict either pass to.")
    parser.add_argument("--days", type=int, default=252,
                        help="Price history window to read (default 252 — the app's cached series).")
    args = parser.parse_args(argv)

    do_available = args.available_at or not args.prices
    do_prices = args.prices or not args.available_at
    tickers = _parse_tickers(args.tickers)

    report: dict[str, Any] = {}
    if do_available:
        report["available_at"] = run_available_at(tickers)
    if do_prices:
        report["prices"] = run_prices(tickers, days=args.days)
    print(json.dumps(report, default=str, sort_keys=True))
    return 1 if report.get("prices", {}).get("errors") else 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
