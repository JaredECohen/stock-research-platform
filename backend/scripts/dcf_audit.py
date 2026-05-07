"""Wave 10 — DCF audit script.

Runs the new DCF (with the Wave 10 reality-check guardrails and the
optional margin-mean-reversion glide) over the curated S&P 100 and
flags names where the model + the live price are materially apart.

The founder's complaint was "valuations seem off." This script is the
diagnostic first step: it tells you which 10-15 tickers to look at
first, and which DCF guardrails are tripping for each.

Usage from `backend/`:

    python -m scripts.dcf_audit                       # full universe
    python -m scripts.dcf_audit --threshold 50        # >50% off only
    python -m scripts.dcf_audit --mean-reversion      # use margin glide
    python -m scripts.dcf_audit --limit 25            # first 25 tickers
    python -m scripts.dcf_audit --tickers NVDA,MSFT   # specific names

Reads only — does not persist DCF results or change any state.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)


def _universe() -> List[str]:
    from app.database import SessionLocal
    from app.models import Company
    with SessionLocal() as db:
        return [
            t for (t,) in db.query(Company.ticker)
            .filter(Company.universe_tier == "auto_analysis").all()
        ]


def _audit_one(ticker: str, *, mean_reversion: bool) -> dict:
    """Run DCF for a single ticker; return audit row dict."""
    from app.finance.dcf import build_full_dcf, derive_default_assumptions
    from app.services.fundamentals_service import get_full_financials
    from app.services.market_data_service import get_current_price

    fin = get_full_financials(ticker)
    if not fin.get("income"):
        return {"ticker": ticker, "error": "no_financials"}

    profile = fin["profile"] or {}
    estimates = None
    try:
        from app.services.data_service import get_data_service
        estimates = get_data_service().get_estimates(ticker)
    except Exception:
        estimates = None

    last_price = get_current_price(ticker) or profile.get("last_price") or 0.0
    diluted_shares = profile.get("shares_outstanding") or 0.0
    cohort_op_margin = None
    try:
        from app.services.sector_research_service import sector_cohort_metrics
        cohort = sector_cohort_metrics(ticker)
        if isinstance(cohort, dict):
            cohort_op_margin = cohort.get("median_operating_margin")
    except Exception:
        cohort_op_margin = None

    try:
        ass = derive_default_assumptions(
            income_statements=fin["income"],
            cash_flows=fin["cash"],
            balance_sheets=fin["balance"],
            current_price=last_price,
            diluted_shares=diluted_shares,
            beta=profile.get("beta") or 1.0,
            analyst_estimates=estimates,
            margin_mean_reversion=mean_reversion,
            cohort_op_margin=cohort_op_margin,
        )
        result = build_full_dcf(ticker, ass)
    except Exception as exc:
        return {"ticker": ticker, "error": f"dcf_failed: {exc}"}

    upside_pct = result.base.upside_pct
    return {
        "ticker": ticker,
        "current_price": last_price,
        "implied_price": result.base.implied_share_price,
        "upside_pct": upside_pct,
        "wacc": result.base.assumptions.wacc,
        "terminal_growth": result.base.assumptions.terminal_growth,
        "guardrails": result.guardrails,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Highlight rows where |upside_pct| exceeds this percentage (default 50).",
    )
    parser.add_argument(
        "--mean-reversion",
        action="store_true",
        help="Use the margin mean-reversion glide (recommended for cyclicals).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many tickers (default: full universe).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list (overrides universe).",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _universe()
    if args.limit:
        tickers = tickers[: args.limit]

    print(
        f"Auditing {len(tickers)} ticker(s) "
        f"{'with' if args.mean_reversion else 'without'} margin mean reversion. "
        f"Threshold: {args.threshold}%."
    )
    print(
        f"{'Ticker':<8} {'Current':>10} {'Implied':>10} {'Upside%':>8} "
        f"{'WACC':>6} {'Tg':>5}  Flags / errors"
    )
    print("-" * 96)
    outliers = 0
    errors = 0
    audited = 0
    for t in tickers:
        row = _audit_one(t, mean_reversion=args.mean_reversion)
        if row.get("error"):
            print(f"{t:<8} {'-':>10} {'-':>10} {'-':>8} {'-':>6} {'-':>5}  ERROR: {row['error']}")
            errors += 1
            continue
        audited += 1
        upside = (row["upside_pct"] or 0.0) * 100
        flag = "*" if abs(upside) > args.threshold else " "
        if abs(upside) > args.threshold:
            outliers += 1
        guards = row.get("guardrails") or []
        guard_summary = ""
        if guards:
            counts = {"warn": 0, "error": 0}
            for g in guards:
                counts[g.severity] = counts.get(g.severity, 0) + 1
            guard_summary = (
                "  ⚠ "
                + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
            )
        print(
            f"{t:<8} {row['current_price']:>10,.2f} "
            f"{row['implied_price']:>10,.2f} {upside:>+7.1f}%{flag} "
            f"{row['wacc']*100:>5.1f}% {row['terminal_growth']*100:>4.1f}%"
            f"{guard_summary}"
        )
    print("-" * 96)
    print(
        f"Audited: {audited}  Outliers (|upside|>{args.threshold}%): {outliers}  "
        f"Errors: {errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
