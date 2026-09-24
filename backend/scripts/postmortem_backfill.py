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

`--dry-run` classifies outcome eligibility first (W6; the only write, to the
derived `memo_outcome_eligibility` ledger) and exits 1 if any snapshot is
still unclassified. Only eligible snapshots are ever postmortem'd.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _dry_run(horizons: list[int], limit: int) -> int:
    """Report what a real run would postmortem. Writes only the W6 ledger.

    Classification runs first, as the real run does: without it every
    snapshot is unclassified, so the eligible-only scan would report nothing
    due and the dry run would understate what the next nightly run spends.
    The ledger is derived data (no memo, outcome or postmortem row changes);
    after the W6 deploy this lists the live memos the eligible-only prior
    dedupe newly lets through, which is bounded strong-route LLM spend
    (limit per horizon per night, 14-day per-ticker rate limit).
    """
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import MemoOutcomeEligibility, MemoSnapshot
    from app.services import outcome_eligibility
    from app.services.postmortem_service import _scan_due

    with SessionLocal() as db:
        print(f"eligibility: {outcome_eligibility.classify_pending(db=db)}")
        unclassified = list(db.execute(
            select(MemoSnapshot.id, MemoSnapshot.ticker)
            .outerjoin(MemoOutcomeEligibility, MemoOutcomeEligibility.memo_snapshot_id == MemoSnapshot.id)
            .where(outcome_eligibility.pending_condition())
            .order_by(MemoSnapshot.id)
        ).all())
    print(f"unclassified snapshots = {len(unclassified)}")
    for sid, ticker in unclassified[:20]:
        print(f"  - {ticker} #{sid}")
    for h in horizons:
        scan = _scan_due(h, limit=limit)
        print(
            f"horizon={h}d  due (after dedupe) = {len(scan.items)}  "
            f"deduped={len(scan.deduped)}  deferred={len(scan.deferred)}  ineligible={scan.ineligible}"
        )
        for item in scan.items[:10]:
            snap = item["snapshot"]
            print(f"  - {snap.ticker} v{snap.version} (snapshot #{snap.id})")
        if len(scan.items) > 10:
            print(f"  ... +{len(scan.items) - 10} more")
    return 1 if unclassified else 0


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
        help="Report counts; writes only the derived W6 eligibility ledger.",
    )
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [30, 90]

    if args.dry_run:
        return _dry_run(horizons, args.limit)

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
