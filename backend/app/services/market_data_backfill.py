"""Company-wide storage plan and single-company, resumable API backfills."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, or_, select, update

from ..database import SessionLocal
from ..models import Company, DailyPrice, FinancialPeriod, MarketDataSync, MemoOutcome, MemoSnapshot
from .price_history_service import backfill_prices, minimum_start, price_coverage

log = logging.getLogger(__name__)


def backfill_plan(*, today: date | None = None) -> dict:
    """Project small metadata columns; never deserialize stored memo bodies."""
    from .outcome_service import DEFAULT_BENCHMARK, DEFAULT_HORIZONS
    today = today or date.today()
    floor = minimum_start(today)
    with SessionLocal() as db:
        companies = dict(db.execute(select(Company.ticker, Company.company_name)).all())
        snapshots = db.execute(select(MemoSnapshot.id, MemoSnapshot.ticker, MemoSnapshot.generated_at).where(MemoSnapshot.as_of_date.is_(None))).all()
        recorded = set(db.execute(select(MemoOutcome.memo_snapshot_id, MemoOutcome.horizon_days)).all())
        syncs = {ticker: {"status": status, "started_at": started.isoformat(),
                          "completed_at": completed.isoformat() if completed else None}
                 for ticker, status, started, completed in db.execute(select(
                     MarketDataSync.ticker, MarketDataSync.status,
                     MarketDataSync.started_at, MarketDataSync.completed_at,
                 ))}
    starts = {ticker: floor for ticker in companies}
    starts[DEFAULT_BENCHMARK] = floor
    pending = []
    for snapshot_id, ticker, generated_at in snapshots:
        if generated_at is None:
            continue
        generated = generated_at.date()
        required = generated - timedelta(days=7)
        starts[ticker] = min(starts.get(ticker, floor), required)
        starts[DEFAULT_BENCHMARK] = min(starts[DEFAULT_BENCHMARK], required)
        for horizon in DEFAULT_HORIZONS:
            target = generated + timedelta(days=horizon)
            if target <= today and (snapshot_id, horizon) not in recorded:
                pending.append({"ticker": ticker, "memo_snapshot_id": snapshot_id, "horizon_days": horizon,
                                "memo_date": generated.isoformat(), "target_date": target.isoformat()})
    targets = [{"ticker": ticker, "company_name": companies.get(ticker),
                "kind": "company" if ticker in companies else ("benchmark" if ticker == DEFAULT_BENCHMARK else "legacy_memo_symbol"),
                "requested_start": starts[ticker].isoformat(), "requested_end": today.isoformat(),
                "minimum_years": 2, "fundamentals_required": ticker in companies,
                "last_sync": syncs.get(ticker)}
               for ticker in sorted(starts, key=lambda ticker: (ticker != DEFAULT_BENCHMARK, ticker))]
    return {"as_of": today.isoformat(), "minimum_years": 2, "company_count": len(companies),
            "target_count": len(targets), "targets": targets,
            "pending_outcome_pair_count": len(pending), "pending_outcome_pairs": pending,
            "read_only": True, "memo_generation_requests": 0}


def coverage_report(ticker: str | None = None) -> dict:
    from .fundamental_history_service import fundamental_coverage
    plan = backfill_plan()
    targets = plan["targets"]
    if ticker:
        targets = [target for target in targets if target["ticker"] == ticker.upper()]
    with SessionLocal() as db:
        price_counts = dict(db.execute(select(DailyPrice.ticker, func.count(DailyPrice.id)).group_by(DailyPrice.ticker)).all())
        fundamental_counts = dict(db.execute(select(FinancialPeriod.ticker, func.count(FinancialPeriod.id)).group_by(FinancialPeriod.ticker)).all())
    rows = []
    for target in targets:
        start = date.fromisoformat(target["requested_start"])
        prices = price_coverage(target["ticker"], start)
        fundamentals = fundamental_coverage(target["ticker"], start) if target["fundamentals_required"] else None
        rows.append({**target, "prices": prices, "fundamentals": fundamentals,
                     "stored_price_row_count": price_counts.get(target["ticker"], 0),
                     "stored_fundamental_row_count": fundamental_counts.get(target["ticker"], 0)})
    return {**{key: value for key, value in plan.items() if key != "targets"}, "targets": rows,
            "stored_price_row_count": sum(row["stored_price_row_count"] for row in rows),
            "stored_fundamental_row_count": sum(row["stored_fundamental_row_count"] for row in rows),
            "stored_count_note": "Raw database row counts include preexisting and excluded observations; coverage is reported separately.",
            "price_coverage_complete_count": sum(row["prices"]["coverage_complete"] for row in rows),
            "price_incomplete_tickers": [row["ticker"] for row in rows if not row["prices"]["coverage_complete"]]}


def requested_start(ticker: str, *, today: date | None = None) -> date:
    """`backfill_plan`'s start for one target without building the whole plan.

    The two-year floor, extended to a week before the oldest live memo
    snapshot so memo-time context is covered. Pinned equal to the plan.
    """
    today = today or date.today()
    start = minimum_start(today)
    with SessionLocal() as db:
        oldest = db.execute(select(func.min(MemoSnapshot.generated_at)).where(
            MemoSnapshot.ticker == ticker.upper(), MemoSnapshot.as_of_date.is_(None))).scalar_one_or_none()
    if oldest is not None:
        start = min(start, oldest.date() - timedelta(days=7))
    return start


SYNC_SCOPES = ("all", "fundamentals")


def sync_ticker(ticker: str, *, force_refresh: bool = False, scope: str = "all", dry_run: bool = False,
                expected_quarantine: dict[int, str] | None = None, audit_key: str | None = None) -> dict:
    """Import one planned target. `scope="fundamentals"` skips prices.

    `dry_run` (fundamentals only) returns the exact FMP-primary plan from a
    rolled-back run: no `market_data_syncs` claim, no cache invalidation, no
    financial rows or audits persist. The FMP re-pull ledger passes
    `expected_quarantine`/`audit_key` so an executed ticker must match its
    reviewed dry run.
    """
    from .fundamental_history_service import backfill_fundamentals
    if scope not in SYNC_SCOPES:
        raise ValueError("scope must be all or fundamentals")
    if dry_run and scope != "fundamentals":
        raise ValueError("dry_run requires scope=fundamentals")
    ticker = ticker.upper()
    target = next((target for target in backfill_plan()["targets"] if target["ticker"] == ticker), None)
    if target is None:
        raise LookupError("Ticker is not a stored company, benchmark or memo symbol")
    start = date.fromisoformat(target["requested_start"])
    if dry_run:
        fundamentals = (backfill_fundamentals(ticker, start, force_refresh=force_refresh, dry_run=True)
                        if target["fundamentals_required"]
                        else {"status": "not_required", "reason": target["kind"], "success": True})
        return {"ticker": ticker, "requested_start": start.isoformat(), "requested_end": date.today().isoformat(),
                "dry_run": True, "scope": scope, "status": "dry_run", "success": bool(fundamentals.get("success")),
                "prices": {"status": "not_requested", "success": True}, "fundamentals": fundamentals,
                "memo_generation_requests": 0, "outcome_rows_modified": 0}
    started = datetime.utcnow()
    # A durable running record makes interrupted imports visible. Each price
    # and financial write is independently idempotent on its natural key.
    with SessionLocal() as db:
        if db.get_bind().dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        values = {"ticker": ticker, "requested_start": start, "started_at": started,
                  "completed_at": None, "status": "running", "report": {}}
        claim = insert(MarketDataSync).values(**values)
        claimed = db.execute(claim.on_conflict_do_update(
            index_elements=["ticker"],
            set_={key: value for key, value in values.items() if key != "ticker"},
            where=or_(MarketDataSync.status != "running", MarketDataSync.started_at <= started - timedelta(minutes=15)),
        ).returning(MarketDataSync.ticker)).scalar_one_or_none()
        db.commit()
        if claimed is None:
            existing = db.get(MarketDataSync, ticker)
            return {"ticker": ticker, "status": "running", "success": False,
                    "started_at": existing.started_at.isoformat(), "note": "An existing backfill is still in progress."}
    report = {"ticker": ticker, "requested_start": start.isoformat(), "requested_end": date.today().isoformat(),
              "started_at": started.isoformat(), "memo_generation_requests": 0, "outcome_rows_modified": 0,
              "scope": scope}
    for kind, action in (
        ("prices", lambda: backfill_prices(ticker, start, force_refresh=force_refresh)),
        ("fundamentals", lambda: backfill_fundamentals(ticker, start, force_refresh=force_refresh,
                                                       expected_quarantine=expected_quarantine, audit_key=audit_key)),
    ):
        if kind == "prices" and scope == "fundamentals":
            report[kind] = {"status": "not_requested", "success": True}
            continue
        if kind == "fundamentals" and not target["fundamentals_required"]:
            report[kind] = {"status": "not_required", "reason": target["kind"], "success": True}
            continue
        try:
            report[kind] = action()
        except Exception as exc:
            report[kind] = {"success": False, "error_type": type(exc).__name__}
            log.warning("market data backfill failed ticker=%s stage=%s error_type=%s", ticker, kind, type(exc).__name__)
    # Cached pre-repair annual labels/values must not be re-ingested over the
    # corrected durable facts by the other process's next ordinary consumer.
    financials = report["fundamentals"]
    cache = {"success": True, "status": "not_needed", "rows_removed": 0}
    if financials.get("committed"):
        from .provider_cache import invalidate
        try:
            cache = {"success": True, "status": "invalidated", "capability": "financials",
                     "key": ticker, "rows_removed": invalidate("financials", ticker)}
        except Exception as exc:
            cache = {"success": False, "status": "failed", "capability": "financials",
                     "key": ticker, "error_type": type(exc).__name__}
        log.info("market data financial cache ticker=%s result=%s", ticker, cache)
    report["financial_cache_invalidation"] = cache
    # The stock page's cold snapshot was built from the pre-import statements
    # and would otherwise be served for up to 90 days.
    cold = {"success": True, "status": "not_needed", "rows_invalidated": 0}
    if financials.get("committed"):
        from ..cache.snapshots import invalidate as invalidate_snapshots
        try:
            cold = {"success": True, "status": "invalidated", "kind": "company_cold",
                    "rows_invalidated": invalidate_snapshots(ticker, kind="company_cold")}
        except Exception as exc:
            cold = {"success": False, "status": "failed", "kind": "company_cold", "error_type": type(exc).__name__}
    report["company_cold_invalidation"] = cold
    report["success"] = (all(report[kind].get("success", False) for kind in ("prices", "fundamentals"))
                         and cache["success"] and cold["success"])
    report["status"] = "complete" if report["success"] else "incomplete"
    report["completed_at"] = datetime.utcnow().isoformat()
    with SessionLocal() as db:
        completed = db.execute(update(MarketDataSync).where(
            MarketDataSync.ticker == ticker, MarketDataSync.started_at == started,
        ).values(completed_at=datetime.fromisoformat(report["completed_at"]), status=report["status"], report=report))
        db.commit()
        if completed.rowcount != 1:
            report["status"] = "superseded"
            report["success"] = False
            report["note"] = "A newer backfill claim owns the durable result."
    log.info("market data backfill ticker=%s status=%s requested_start=%s", ticker, report["status"], start)
    return report
