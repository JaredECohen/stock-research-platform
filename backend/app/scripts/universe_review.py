"""Print the curated-universe review report as JSON. No side effects.

Usage (from `backend/`):

    python -m app.scripts.universe_review                 # file + DB drift
    python -m app.scripts.universe_review --compare-feed  # also diff vs FMP

Same report as `GET /api/admin/universe-review`. Deliberately separate
from `refresh_universe_lists` (which rewrites `data/sp500.json`): look
first, then decide, then refresh by hand.
"""
from __future__ import annotations

import argparse
import json

from ..services.universe_review import review_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-feed", action="store_true",
        help="Also diff the file against the live FMP constituent list "
             "(read-only; needs FMP_API_KEY and ENABLE_LIVE_DATA=true).",
    )
    args = parser.parse_args(argv)
    print(json.dumps(review_universe(compare_feed=args.compare_feed), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
