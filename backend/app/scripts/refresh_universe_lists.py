"""Refresh `data/sp500.json` from the live FMP constituent feed.

Run when SP500 composition changes (a few times per year) or after
the initial deployment to populate the full universe.

Usage:
    python -m app.scripts.refresh_universe_lists [--dry-run]

The file written matches the existing `sp100.json` shape so the
seeder picks it up without further changes. `sp100.json` is left
alone — that file is the "core" cohort used to default the
`auto_update_memo` flag and stays curated rather than constituent-
list-driven.
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
    payload = {
        "_doc": (
            "S&P 500 constituents — the curated screener universe. "
            "Refresh via `python -m app.scripts.refresh_universe_lists`. "
            f"Last refreshed: {date.today().isoformat()}."
        ),
        "_dual_class_policy": (
            "When a company has multiple share classes in the index "
            "(Alphabet GOOG/GOOGL, Berkshire BRK.A/BRK.B), FMP usually "
            "returns one canonical ticker; the seeder upserts whatever "
            "shows up. Re-run this script if a class change matters."
        ),
        "as_of": date.today().isoformat(),
        "tickers": tickers,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print(f"\n[dry-run] {len(tickers)} tickers — not written.")
        return 0

    SP500_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(tickers)} tickers to {SP500_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
