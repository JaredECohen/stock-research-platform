"""Read-only census of what readers now see in each ticker's latest memo (W2a).

    python -m app.scripts.memo_section_census [--json] [--ticker T ...]

For every ticker with a live memo it reads the latest snapshot, presents it
exactly as `GET /api/stocks/{t}/memo` does (`memo_store.present_snapshot`)
and counts section verdicts: per-section status counts across the corpus,
and the tickers where more than half of the classified sections are
unavailable (the "mostly placeholders" memos the owner question in W2a §11
is about). A snapshot that no longer validates is listed, never repaired.

This script only SELECTs. It writes nothing, generates nothing and makes
no LLM or provider call, so it is safe in a Render web shell. It prints no
memo text and no configuration, only tickers, versions and counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MemoSnapshot
from ..services import memo_store

# Sections every memo is judged on for the "mostly unavailable" list. Dynamic
# keys (long-form drill-downs, extra analysts) vary per memo and would make
# the ratio incomparable across tickers.
CORE_SECTIONS: tuple[str, ...] = (
    "final_pm_view", "one_sentence_thesis", "confidence_score", "mispricing_thesis",
    "sector_agent_view", "sector_synthesis", "earnings_agent_view", "filing_agent_view",
    "valuation_agent_view", "macro_sensitivity", "technical_agent_view", "bull_case",
    "bear_case", "key_risks", "catalysts", "risk_committee_challenge", "final_verdict",
)


def live_tickers(db: Session) -> list[str]:
    rows = db.execute(
        select(MemoSnapshot.ticker).where(MemoSnapshot.as_of_date.is_(None)).distinct()
    ).scalars().all()
    return sorted({str(t).upper() for t in rows})


def census(db: Session, tickers: list[str] | None = None) -> dict[str, Any]:
    tickers = tickers or live_tickers(db)
    per_section: dict[str, Counter[str]] = defaultdict(Counter)
    per_ticker: dict[str, Any] = {}
    unreadable: list[dict[str, Any]] = []
    cache: memo_store.ChainCache = {}
    for ticker in tickers:
        snap = memo_store.latest_memo(ticker, db=db)
        if snap is None:
            continue
        try:
            presented = memo_store.present_snapshot(snap, db=db, cache=cache)
        except memo_store.StoredMemoUnreadable as exc:
            unreadable.append({"ticker": ticker, "version": snap.version, "fields": list(exc.fields)})
            continue
        av = presented.section_availability
        for key, entry in av.items():
            per_section[key][entry.status] += 1
        core = [av[k] for k in CORE_SECTIONS if k in av and av[k].reason != "not_produced"]
        hidden = [k for k in CORE_SECTIONS if k in av and av[k].status == "unavailable"
                  and av[k].reason != "not_produced"]
        per_ticker[ticker] = {
            "version": snap.version,
            "trigger": snap.trigger,
            "generation_mode": presented.generation_mode,
            "unavailable": len(hidden),
            "classified": len(core),
            "unavailable_sections": hidden,
        }
    mostly = sorted(
        t for t, row in per_ticker.items()
        if row["classified"] and row["unavailable"] * 2 > row["classified"]
    )
    return {
        "tickers": len(per_ticker),
        "per_section": {k: dict(v) for k, v in sorted(per_section.items())},
        "mostly_unavailable": mostly,
        "per_ticker": per_ticker,
        "unreadable": unreadable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", action="store_true", help="print the full census as JSON")
    parser.add_argument("--ticker", action="append", default=None, help="limit to these tickers")
    args = parser.parse_args(argv)
    with SessionLocal() as db:
        result = census(db, [t.upper() for t in args.ticker] if args.ticker else None)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"memos presented: {result['tickers']}")
    for key, counts in result["per_section"].items():
        print(f"  {key:42s} " + "  ".join(f"{s}={n}" for s, n in sorted(counts.items())))
    print(f"mostly unavailable ({len(result['mostly_unavailable'])}): "
          + (", ".join(result["mostly_unavailable"]) or "none"))
    if result["unreadable"]:
        print("unreadable: " + ", ".join(f"{u['ticker']} v{u['version']}" for u in result["unreadable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
