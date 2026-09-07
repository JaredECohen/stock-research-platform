"""Refresh `data/sp500.json` from the live FMP constituent feed.

This is the ONE path that rewrites the universe file, and it only runs
when an operator invokes it. Nothing schedules it: a constituent change
silently landing in the file would re-tier companies and change the
screener with nobody having looked. Check first with
`python -m app.scripts.universe_review --compare-feed` (read-only),
then run this, then `POST /api/seed-universe` to apply it to the DB.

Usage:
    python -m app.scripts.refresh_universe_lists [--dry-run]

The written file keeps the shape the seeder and `universe_review`
expect, stamps `_last_reviewed` = today (a fresh pull is a review) and
carries the existing `_top_10_by_market_cap_2026_05` pin list forward —
dropping it would silently un-pin every auto-update memo on the next
seed. `sp100.json` is left alone: it is the legacy fallback the seeder
uses only when sp500.json is missing.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from ..providers.fmp_provider import FMPProvider

log = logging.getLogger(__name__)


SP500_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "sp500.json"
)
AUTO_UPDATE_KEY = "_top_10_by_market_cap_2026_05"
DEFAULT_REVIEW_CADENCE_DAYS = 120


def _read_existing() -> dict:
    """Metadata from the file about to be overwritten, or {} if absent."""
    if not SP500_PATH.exists():
        return {}
    try:
        return json.loads(SP500_PATH.read_text())
    except ValueError:
        log.warning("existing %s is not valid JSON; not carrying metadata forward", SP500_PATH)
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the fetched list, don't write.")
    args = parser.parse_args(argv)

    fmp = FMPProvider()
    tickers = fmp.get_sp500_constituents()
    if not tickers:
        log.error(
            "FMP returned no SP500 tickers — check FMP_API_KEY + plan tier "
            "(the /stable/sp500-constituent endpoint requires Premium)."
        )
        return 1

    tickers = sorted(set(t.upper() for t in tickers))
    today = date.today().isoformat()
    existing = _read_existing()
    payload = {
        "_doc": (
            "S&P 500 constituents — the curated screener universe. "
            "Static snapshot; nothing refreshes it automatically. Review "
            "with `python -m app.scripts.universe_review`, refresh via "
            f"`python -m app.scripts.refresh_universe_lists`. Last refreshed: {today}."
        ),
        "_dual_class_policy": (
            "When a company has multiple share classes in the index "
            "(Alphabet GOOG/GOOGL, Berkshire BRK.A/BRK.B), FMP usually "
            "returns one canonical ticker; the seeder upserts whatever "
            "shows up. Re-run this script if a class change matters."
        ),
        "as_of": today,
        "_last_reviewed": today,
        "_review_source": "fmp constituent feed via refresh_universe_lists",
        "_review_cadence_days": existing.get("_review_cadence_days", DEFAULT_REVIEW_CADENCE_DAYS),
        "tickers": tickers,
    }
    if existing.get(AUTO_UPDATE_KEY):
        payload[AUTO_UPDATE_KEY] = existing[AUTO_UPDATE_KEY]

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print(f"\n[dry-run] {len(tickers)} tickers — not written.")
        return 0

    SP500_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(tickers)} tickers to {SP500_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
