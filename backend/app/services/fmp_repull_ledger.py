"""One-time FMP-primary fundamentals re-pull, executed by the deployed worker.

Owner decision 2026-09-24 (item 3): re-pull every company from FMP; other
providers only fill periods FMP lacks; conflicting rows are quarantined, never
deleted. No agent holds `ADMIN_API_TOKEN` or database credentials, so the
production run is in-app code the worker executes, gated on the owner:

1. **Dry run (default after deploy).** The worker's `fmp-repull` thread walks
   every company target. Each ticker's FMP-primary takeover runs inside a
   transaction that is rolled back (`sync_ticker(..., dry_run=True)`), so
   nothing in `financial_periods` or `financial_data_repairs` changes. The
   ledger row `DRY_RUN_ID` records, per ticker, the exact row ids per
   quarantine reason, adoptions, restatements and entitlement denials, and
   a log line is written per ticker and on completion.
2. **Review.** `GET /api/admin/market-data/fmp-repull` returns the
   ledger with a review list and its `result_digest`.
3. **Authorize.** Only the owner starts execution, by
   `POST /api/admin/market-data/fmp-repull/authorize` with that
   `result_digest`, or by setting `FUNDAMENTALS_REPULL_EXECUTE=<result_digest>`
   on the worker in the Render dashboard. Either creates `EXECUTE_ID`.
4. **Execute.** Ticker by ticker, supervised, each in one transaction that
   aborts (`repull_plan_mismatch`, nothing written) when the quarantine set
   differs from the reviewed dry run. Quarantines, adoptions and
   restatements are audited and reversible (`fundamental_quarantine`).

Bounded for the 512 MiB worker: at most `MAX_TICKERS_PER_PASS` tickers and
`PASS_BUDGET_SECONDS` per pass, `PACE_SECONDS` between tickers, and no work in
the 02:00-04:30 UTC window where the nightly loops run. Idempotent and
resumable: a ticker is recorded once; a ticker whose execution committed
before a crash is detected by its deterministic audit id. The ledgers are
`financial_data_repairs` rows (no new table), readable through the existing
bk-repair GET as well. `FUNDAMENTALS_REPULL=off` disables the thread.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialDataRepair
from .bk_fundamental_repair import _digest

log = logging.getLogger(__name__)

LEDGER_KIND = "fmp_repull_ledger"
DRY_RUN_ID = "fmp-repull-2026-09-dry-run"
EXECUTE_ID = "fmp-repull-2026-09-execute"
ENABLE_ENV = "FUNDAMENTALS_REPULL"
EXECUTE_ENV = "FUNDAMENTALS_REPULL_EXECUTE"
MAX_TICKERS_PER_PASS = 6
PASS_BUDGET_SECONDS = 180.0
PACE_SECONDS = 1.0
START_DELAY_SECONDS = 900.0      # let the boot seed and first loops settle
POLL_SECONDS = 60.0              # between passes while a ledger has work
IDLE_SECONDS = 1800.0            # waiting for the owner's authorization
QUIET_WINDOW_UTC = ((2, 0), (4, 30))   # outcome 02:30, postmortem 03:00, history 03:15, scorecard 03:45
# Reasons the design's known-conflict table predicts (W5a §4). Anything else,
# or more than REVIEW_QUARANTINE_THRESHOLD rows outside KNOWN_CASES, is named
# on the review list before the owner authorizes execution.
EXPECTED_REASONS = frozenset({
    "period_end_conflicts_with_primary", "alias_superseded_by_primary",
    "label_conflicts_with_primary_observation", "invalid_availability_precedes_period_end",
    "value_conflicts_with_primary", "secondary_row_in_primary_period"})
REVIEW_QUARANTINE_THRESHOLD = 50
KNOWN_CASES = {"MDT": 32, "MSFT": 128, "NVDA": 128, "ANSS": 149, "LULU": 2, "BK": 0}
UNEXERCISED_ENDPOINT_NOTE = (
    "The re-pull calls only the income, balance-sheet and cash-flow statement endpoints. "
    "The 401/402/403 behaviour of ratios, key-metrics, analyst-estimates and earnings "
    "was never evidenced; their refusals are logged as 'FMP <path> -> <status> symbol=<s>'.")


def _now() -> datetime:
    return datetime.utcnow()


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() not in {"off", "false", "0", "disabled"}


def primary_history_capable() -> bool:
    """True only when FMP, configured, answers the durable history call."""
    from .data_service import get_data_service
    for provider in get_data_service()._live_chain("financials"):
        if getattr(provider, "name", None) != "fmp" or not callable(getattr(provider, "get_financial_history", None)):
            continue
        status = getattr(provider, "status", None)
        return bool(status().configured) if callable(status) else True
    return False


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """The reviewable per-ticker record: counts plus exact row ids."""
    fundamentals = report.get("fundamentals") or {}
    issues = fundamentals.get("issues") or []
    open_issues = [i for i in issues if not i.get("resolved")]
    quarantine = {str(q["id"]): q["reason"] for q in fundamentals.get("quarantined") or []}
    from .fundamental_history_service import BLOCKING_ISSUES
    denied = [i for i in issues if i.get("kind") == "provider_entitlement_denied"]
    mismatch = next((i for i in issues if i.get("kind") == "repull_plan_mismatch"), None)
    dry = bool(report.get("dry_run"))
    return {
        "status": report.get("status"), "success": bool(report.get("success")), "dry_run": dry,
        "fundamentals_success": bool(fundamentals.get("success")),
        "rows_written": fundamentals.get("rows_written", 0), "rows_refreshed": fundamentals.get("rows_refreshed", 0),
        "rows_inserted": fundamentals.get("rows_inserted", 0), "rows_quarantined": len(quarantine),
        "quarantine": quarantine, "quarantine_counts": dict(Counter(quarantine.values())),
        "adoption_ids": sorted({a["id"] for a in fundamentals.get("adoptions") or []}),
        "restatement_ids": sorted({r["id"] for r in fundamentals.get("restatements") or []}),
        "secondary_rows_skipped": fundamentals.get("secondary_rows_skipped", 0),
        # Audit ids are real only when the run committed.
        "quarantine_repair_id": None if dry else fundamentals.get("quarantine_repair_id"),
        "adoption_repair_id": None if dry else fundamentals.get("adoption_repair_id"),
        "restatement_repair_id": None if dry else fundamentals.get("restatement_repair_id"),
        "planned_repair_id": None if dry else fundamentals.get("planned_repair_id"),
        "issue_kinds": dict(Counter(i.get("kind") for i in open_issues)),
        "blocking_issue_kinds": sorted({i.get("kind") for i in open_issues if i.get("kind") in BLOCKING_ISSUES}),
        "entitlement_denied": sorted({f"{i.get('endpoint')}:{i.get('cadence')}:{i.get('status')}:{i.get('symbol')}"
                                      for i in denied if not i.get("resolved")}),
        "entitlement_resolved_by_symbol": sorted({f"{i.get('endpoint')}:{i.get('symbol')}->{i.get('resolved_by_symbol')}"
                                                  for i in denied if i.get("resolved")}),
        "primary_availability_flags": sorted(i["id"] for i in issues
                                             if i.get("kind") == "primary_availability_precedes_period_end"),
        "plan_mismatch": mismatch,
        "error": report.get("error_type") or fundamentals.get("error_type"),
    }


def summarize(compacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Totals and the review list the owner reads before authorizing (pure)."""
    reasons: Counter[str] = Counter()
    endpoints: dict[str, list[str]] = {}
    review, failed, mismatched = [], [], []
    for ticker, item in sorted(compacts.items()):
        counts = item.get("quarantine_counts") or {}
        reasons.update(counts)
        unexpected = sorted(set(counts) - EXPECTED_REASONS)
        count = item.get("rows_quarantined", 0)
        if unexpected or (count > REVIEW_QUARANTINE_THRESHOLD and ticker not in KNOWN_CASES):
            review.append({"ticker": ticker, "rows_quarantined": count, "unexpected_reasons": unexpected})
        if item.get("plan_mismatch"):
            mismatched.append(ticker)
        elif not item.get("fundamentals_success", item.get("success")):
            failed.append({"ticker": ticker, "status": item.get("status"),
                           "blocking_issue_kinds": item.get("blocking_issue_kinds", []), "error": item.get("error")})
        for denial in item.get("entitlement_denied") or []:
            endpoint = ":".join(denial.split(":")[:3])
            endpoints.setdefault(endpoint, []).append(ticker)
    return {
        "tickers": len(compacts),
        "rows_quarantined": sum(i.get("rows_quarantined", 0) for i in compacts.values()),
        "rows_inserted": sum(i.get("rows_inserted", 0) for i in compacts.values()),
        "adoptions": sum(len(i.get("adoption_ids") or []) for i in compacts.values()),
        "restatements": sum(len(i.get("restatement_ids") or []) for i in compacts.values()),
        "quarantine_reasons": dict(reasons),
        "known_cases": {t: {"expected_quarantines": n, "actual": compacts.get(t, {}).get("rows_quarantined")}
                        for t, n in KNOWN_CASES.items()},
        "review_list": review, "failed": failed, "plan_mismatch": mismatched,
        "entitlement_denied_by_endpoint": {k: sorted(v) for k, v in sorted(endpoints.items())},
        "unexercised_endpoints_note": UNEXERCISED_ENDPOINT_NOTE,
    }


def _targets() -> list[dict[str, str]]:
    from .market_data_backfill import backfill_plan
    return [{"ticker": t["ticker"], "requested_start": t["requested_start"]}
            for t in backfill_plan()["targets"] if t["fundamentals_required"]]


def _create(db: Session, ledger_id: str, plan: dict[str, Any]) -> FinancialDataRepair:
    ledger = FinancialDataRepair(id=ledger_id, digest=_digest(plan), status="running", plan=plan,
                                 result={"tickers": {}}, created_at=_now())
    db.add(ledger)
    db.flush()
    return ledger


def _view(ledger: FinancialDataRepair | None) -> dict[str, Any] | None:
    if ledger is None:
        return None
    plan, result = ledger.plan or {}, ledger.result or {}
    targets = plan.get("targets") or []
    done = result.get("tickers") or {}
    return {"id": ledger.id, "status": ledger.status, "digest": ledger.digest, "mode": plan.get("mode"),
            "created_at": plan.get("created_at"), "authorized_by": plan.get("authorized_by"),
            "dry_run_result_digest": plan.get("dry_run_result_digest"),
            "targets": len(targets), "done": len(done), "remaining": [t["ticker"] for t in targets if t["ticker"] not in done],
            "result_digest": result.get("result_digest"), "summary": result.get("summary"),
            "completed_at": result.get("completed_at"), "tickers": done}


def repull_status(*, db: Session | None = None) -> dict[str, Any]:
    """Read-only view of both ledgers for the owner's review."""
    own = db is None
    db = db or SessionLocal()
    try:
        FinancialDataRepair.__table__.create(bind=db.get_bind(), checkfirst=True)
        return {"enabled": enabled(), "execute_env_set": bool(os.environ.get(EXECUTE_ENV, "").strip()),
                "dry_run": _view(db.get(FinancialDataRepair, DRY_RUN_ID)),
                "execute": _view(db.get(FinancialDataRepair, EXECUTE_ID)),
                "authorize": "POST /api/admin/market-data/fmp-repull/authorize {\"result_digest\": <dry_run.result_digest>}",
                "read_only": True}
    finally:
        if own:
            db.close()


def authorize_execution(result_digest: str, *, authorized_by: str = "admin", db: Session | None = None) -> dict[str, Any]:
    """Create the execute ledger from a completed, reviewed dry run.

    LookupError: no dry run. RuntimeError: dry run incomplete, digest does not
    match what was reviewed, or execution already authorized.
    """
    own = db is None
    db = db or SessionLocal()
    try:
        FinancialDataRepair.__table__.create(bind=db.get_bind(), checkfirst=True)
        dry = db.get(FinancialDataRepair, DRY_RUN_ID)
        if dry is None:
            raise LookupError("No FMP re-pull dry run exists")
        expected = (dry.result or {}).get("result_digest")
        if dry.status != "complete" or not expected:
            raise RuntimeError("The dry run has not completed")
        candidate = (result_digest or "").strip().lower()
        if len(candidate) < 12 or not expected.startswith(candidate):
            raise RuntimeError("Digest does not match the reviewed dry run")
        if db.get(FinancialDataRepair, EXECUTE_ID) is not None:
            raise RuntimeError("Execution was already authorized")
        tickers = dry.result.get("tickers") or {}
        plan = {"kind": LEDGER_KIND, "mode": "execute", "created_at": _now().isoformat(),
                "authorized_by": authorized_by, "dry_run_id": DRY_RUN_ID, "dry_run_result_digest": expected,
                "targets": [t for t in dry.plan["targets"] if t["ticker"] in tickers]}
        ledger = _create(db, EXECUTE_ID, plan)
        if own:
            db.commit()
        log.info("fundamentals repull execution authorized by=%s dry_run_digest=%s targets=%d",
                 authorized_by, expected, len(plan["targets"]))
        return _view(ledger) or {}
    except Exception:
        if own:
            db.rollback()
        raise
    finally:
        if own:
            db.close()


def _record(ledger_id: str, ticker: str, compact: dict[str, Any]) -> None:
    with SessionLocal() as db:
        ledger = db.execute(select(FinancialDataRepair).where(FinancialDataRepair.id == ledger_id)
                            .with_for_update()).scalar_one()
        result = dict(ledger.result or {})
        tickers = dict(result.get("tickers") or {})
        tickers[ticker] = compact
        result["tickers"] = tickers
        result["updated_at"] = _now().isoformat()
        ledger.result = result  # reassigned: JSON columns do not track in-place mutation
        db.commit()


def _finish(ledger_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        ledger = db.execute(select(FinancialDataRepair).where(FinancialDataRepair.id == ledger_id)
                            .with_for_update()).scalar_one()
        result = dict(ledger.result or {})
        tickers = result.get("tickers") or {}
        result["summary"] = summarize(tickers)
        result["completed_at"] = _now().isoformat()
        result["result_digest"] = _digest({"ledger_id": ledger_id, "plan_digest": ledger.digest, "tickers": tickers})
        ledger.result = result
        ledger.status = "complete"
        ledger.applied_at = _now()
        db.commit()
        summary = result["summary"]
        log.info("fundamentals repull %s complete: tickers=%d quarantined=%d adoptions=%d restatements=%d "
                 "review=%s failed=%d mismatched=%s result_digest=%s", (ledger.plan or {}).get("mode"),
                 summary["tickers"], summary["rows_quarantined"], summary["adoptions"], summary["restatements"],
                 [r["ticker"] for r in summary["review_list"]], len(summary["failed"]), summary["plan_mismatch"],
                 result["result_digest"])
        return result


def _in_quiet_window(now: datetime) -> bool:
    (start_h, start_m), (end_h, end_m) = QUIET_WINDOW_UTC
    minutes = now.hour * 60 + now.minute
    return start_h * 60 + start_m <= minutes < end_h * 60 + end_m


def run_pass(*, now: datetime | None = None, max_tickers: int = MAX_TICKERS_PER_PASS,
             budget_seconds: float = PASS_BUDGET_SECONDS, sleep: Callable[[float], None] = time.sleep,
             monotonic: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Advance whichever ledger has work by one bounded pass. Never raises for one ticker."""
    from . import fundamental_quarantine as quarantine
    from .market_data_backfill import sync_ticker

    now = now or _now()
    if not enabled():
        return {"skipped_reason": f"{ENABLE_ENV}=off", "remaining": 0}
    if _in_quiet_window(now):
        return {"skipped_reason": "nightly loop window", "remaining": 1}
    if not primary_history_capable():
        return {"skipped_reason": "no configured FMP history provider in the financials chain", "remaining": 0}
    with SessionLocal() as db:
        FinancialDataRepair.__table__.create(bind=db.get_bind(), checkfirst=True)
        dry = db.get(FinancialDataRepair, DRY_RUN_ID)
        if dry is None:
            dry = _create(db, DRY_RUN_ID, {"kind": LEDGER_KIND, "mode": "dry_run", "created_at": now.isoformat(),
                                           "targets": _targets()})
            db.commit()
            log.info("fundamentals repull dry run started targets=%d", len(dry.plan["targets"]))
        execute = db.get(FinancialDataRepair, EXECUTE_ID)
        env_digest = os.environ.get(EXECUTE_ENV, "").strip()
        dry_complete = dry.status == "complete"
    if execute is None and dry_complete and env_digest:
        try:
            authorize_execution(env_digest, authorized_by="env")
        except (LookupError, RuntimeError) as exc:
            log.warning("fundamentals repull %s not accepted: %s", EXECUTE_ENV, exc)
    with SessionLocal() as db:
        dry = db.get(FinancialDataRepair, DRY_RUN_ID)
        execute = db.get(FinancialDataRepair, EXECUTE_ID)
        ledger = dry if dry is not None and dry.status != "complete" else (
            execute if execute is not None and execute.status != "complete" else None)
        if ledger is None:
            state = "complete" if execute is not None else "awaiting_authorization"
            return {"skipped_reason": state, "remaining": 0 if execute is not None else 1}
        ledger_id, mode = ledger.id, ledger.plan["mode"]
        done = set((ledger.result or {}).get("tickers") or {})
        pending = [t["ticker"] for t in ledger.plan["targets"] if t["ticker"] not in done]
        reviewed = (dry.result or {}).get("tickers") or {} if dry is not None else {}
    started = monotonic()
    processed: list[str] = []
    for ticker in pending[:max_tickers]:
        if processed and monotonic() - started >= budget_seconds:
            break
        compact: dict[str, Any]
        if mode == "execute":
            committed_id = quarantine.repair_id_for(ledger_id, quarantine.QUARANTINE_KIND, ticker)
            with SessionLocal() as db:
                committed = db.get(FinancialDataRepair, committed_id) is not None
            if committed:
                compact = {"status": "already_applied", "success": True, "quarantine_repair_id": committed_id,
                           "note": "committed before an interrupted pass; not re-run"}
                _record(ledger_id, ticker, compact)
                processed.append(ticker)
                continue
            expected = {int(k): v for k, v in ((reviewed.get(ticker) or {}).get("quarantine") or {}).items()}
        try:
            if mode == "dry_run":
                report = sync_ticker(ticker, force_refresh=True, scope="fundamentals", dry_run=True)
            else:
                report = sync_ticker(ticker, force_refresh=True, scope="fundamentals",
                                     expected_quarantine=expected, audit_key=ledger_id)
        except LookupError:
            report = {"status": "not_in_plan", "success": False}
        except Exception as exc:
            report = {"status": "error", "success": False, "error_type": type(exc).__name__}
        if report.get("status") == "running":
            log.info("fundamentals repull %s ticker=%s deferred: an import holds the claim", mode, ticker)
            break
        compact = compact_report(report)
        _record(ledger_id, ticker, compact)
        processed.append(ticker)
        log.info("fundamentals repull %s ticker=%s status=%s quarantined=%d counts=%s adoptions=%d restatements=%d "
                 "inserted=%d entitlement_denied=%s mismatch=%s", mode, ticker, compact["status"],
                 compact["rows_quarantined"], compact["quarantine_counts"], len(compact["adoption_ids"]),
                 len(compact["restatement_ids"]), compact["rows_inserted"], compact["entitlement_denied"],
                 bool(compact["plan_mismatch"]))
        sleep(PACE_SECONDS)
    remaining = len(pending) - len(processed)
    result: dict[str, Any] = {"ledger": ledger_id, "mode": mode, "processed": processed, "remaining": remaining}
    if remaining == 0:
        result["result"] = _finish(ledger_id)
    return result


def _loop(shutdown: threading.Event) -> None:
    if shutdown.wait(START_DELAY_SECONDS):
        return
    while True:
        try:
            outcome = run_pass()
        except Exception as exc:  # the thread must survive a bad pass
            log.warning("fundamentals repull pass failed: %s", type(exc).__name__)
            outcome = {"remaining": 1}
        if outcome.get("skipped_reason") == "complete" or (
                outcome.get("remaining") == 0 and outcome.get("skipped_reason", "").endswith("=off")):
            log.info("fundamentals repull thread finished: %s", outcome.get("skipped_reason"))
            return
        delay = POLL_SECONDS if outcome.get("remaining") and not outcome.get("skipped_reason") else IDLE_SECONDS
        if shutdown.wait(delay):
            return


def start_thread(shutdown: threading.Event) -> threading.Thread | None:
    """Start the worker's re-pull thread (daemon; per-ticker transactions make a kill safe)."""
    if not enabled():
        log.info("fundamentals repull disabled (%s=off)", ENABLE_ENV)
        return None
    thread = threading.Thread(target=_loop, args=(shutdown,), name="fmp-repull", daemon=True)
    thread.start()
    return thread
