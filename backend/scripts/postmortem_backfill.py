"""Wave 10 — postmortem retroactive backfill.

Cron-shaped + idempotent: walks every memo with a 30d / 90d outcome
that's missing a postmortem and runs the postmortem service. Useful
for:

- Day-0 of the postmortem feature: seed the system with all memos
  already past their horizon windows.
- Recovery after the cron has been off (catch up the missed days).

The same dedupe rules that protect the daily loop apply here — a
backlog run won't write 50 postmortems for the same ticker just
because there are 50 snapshot versions.

Usage from `backend/`:

    python -m scripts.postmortem_backfill                    # full backlog
    python -m scripts.postmortem_backfill --horizon 30       # 30d only
    python -m scripts.postmortem_backfill --limit 20         # cap one run
    python -m scripts.postmortem_backfill --dry-run          # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[30, 90, 180, 365],
        default=None,
        help="Only process this horizon (default: both 30d + 90d).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max postmortems to write per horizon (default 200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts; do not write anything.",
    )
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [30, 90]

    if args.dry_run:
        from app.services.postmortem_service import _due_memos
        for h in horizons:
            due = _due_memos(h, limit=args.limit)
            print(f"horizon={h}d  due (after dedupe) = {len(due)}")
            for item in due[:10]:
                snap = item["snapshot"]
                print(f"  - {snap.ticker} v{snap.version} (gen {snap.generated_at.date()})")
            if len(due) > 10:
                print(f"  ... +{len(due) - 10} more")
        return 0

    from app.services.postmortem_service import run_postmortems
    total_written = 0
    for h in horizons:
        res = run_postmortems(horizon_days=h, limit=args.limit)
        print(
            f"horizon={h}d  due={res['due']}  written={res['written']}  "
            f"skipped={res['skipped']}"
        )
        total_written += res["written"]
    print(f"---\nTotal postmortems written: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
