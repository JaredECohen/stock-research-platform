"""Stock endpoints — list, detail, memo generation."""
from __future__ import annotations

import logging
from datetime import date as _date, datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response
from sqlalchemy import select

from ..agents.graph import run_stock_memo
from ..database import SessionLocal
from ..models import Company
from ..rate_limit import LIMITS, limiter
from ..schemas import CompanyOut, StockMemoOut
from ..seed_universe import ensure_company_in_universe
from ..services import memo_store
from ..services.data_service import get_data_service
from ..services.fundamentals_service import get_full_financials
from ..services.history_service import backfill_ticker
from ..services.market_data_service import get_basic_stats, get_price_series

log = logging.getLogger(__name__)

router = APIRouter()


# In-memory job registry for async memo regeneration. Process-local
# (single-replica only — switch to Redis when scaling out).
#
# `_REGEN_JOBS` tracks live regens: {ticker: started_at_utc}; cleared
# on completion (success OR failure).
# `_REGEN_FAILURES` retains the last failure per ticker so the status
# endpoint can surface it after the job clears. Otherwise a failed
# BackgroundTask silently disappears and the frontend just spins.
#
# Why this exists: a full memo regen takes 5-9 minutes synchronously,
# but Render's HTTP proxy timeout is ~100 seconds. Returning the memo
# inline would always 504 from the user's perspective. Instead the
# analyze endpoint fires-and-returns a 202 with the start time; the
# frontend polls `/analyze/status` for completion.
_REGEN_JOBS: Dict[str, datetime] = {}
_REGEN_FAILURES: Dict[str, Dict[str, Any]] = {}
_REGEN_LOCK = Lock()


def _run_regen_job(ticker: str, scenario: str) -> None:
    """Background worker — runs `run_stock_memo` and clears the job
    entry whether or not it succeeded.

    Catches `BaseException` (not just `Exception`) so we capture
    asyncio.CancelledError too — when FastAPI's response coroutine
    ends, Starlette can cancel the BackgroundTask chain even though
    the threadpool thread keeps running for a moment. The earlier
    `except Exception` missed those, leading to the silent-failure
    pattern where in_progress clears with no failure record and no
    memo. Re-raises SystemExit / KeyboardInterrupt as required.

    Persists failure details (error type + message + duration) to
    `_REGEN_FAILURES[ticker]` so a subsequent /analyze/status call
    can surface what went wrong. Successful runs clear the failure
    entry.
    """
    import traceback
    started = datetime.utcnow()
    log.info("background memo regen STARTING for %s (scenario=%s)", ticker, scenario)
    try:
        run_stock_memo(ticker, scenario=scenario, force_refresh=True)
        # Success — drop any prior failure record so the status
        # endpoint reflects the green state.
        with _REGEN_LOCK:
            _REGEN_FAILURES.pop(ticker, None)
        log.info(
            "background memo regen SUCCEEDED for %s in %.1fs",
            ticker, (datetime.utcnow() - started).total_seconds(),
        )
    except (SystemExit, KeyboardInterrupt):
        # These are intentional process-exit signals — never swallow.
        raise
    except BaseException as exc:
        tb = traceback.format_exc()
        log.error(
            "background memo regen FAILED for %s after %.1fs: %s: %s\n%s",
            ticker, (datetime.utcnow() - started).total_seconds(),
            type(exc).__name__, exc, tb,
        )
        with _REGEN_LOCK:
            _REGEN_FAILURES[ticker] = {
                "ticker": ticker,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "traceback_tail": tb[-1500:],
                "started_at": started.isoformat(),
                "failed_at": datetime.utcnow().isoformat(),
                "duration_seconds": (datetime.utcnow() - started).total_seconds(),
            }
    finally:
        with _REGEN_LOCK:
            _REGEN_JOBS.pop(ticker, None)


def _ensure_lazy_universe(ticker: str) -> str:
    """Resolve `ticker` into the universe.

    Returns the company's tier. Inserts a fresh `analyzed_on_demand`
    row + kicks off a synchronous backfill when the ticker is brand
    new. Raises HTTPException(404) when the live provider chain
    rejects the symbol entirely.
    """
    t = ticker.upper()
    with SessionLocal() as db:
        existing = db.execute(
            select(Company.universe_tier).where(Company.ticker == t)
        ).first()
    if existing:
        return (existing[0] or "data_only")
    # New symbol — try to introduce it.
    profile = ensure_company_in_universe(t)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"{t}: provider chain rejected this symbol.",
        )
    # Heavy load (5yr financials + filings + transcripts) so the agent
    # graph has data to work with. Best-effort — if a single capability
    # 403s, we still return the memo using whatever did land.
    try:
        backfill_ticker(t)
    except Exception:  # pragma: no cover
        pass
    return "analyzed_on_demand"


def _company_tier(ticker: str) -> str:
    """Look up the universe tier for a ticker; returns 'data_only' if unknown."""
    with SessionLocal() as db:
        row = db.execute(
            select(Company.universe_tier).where(Company.ticker == ticker.upper())
        ).first()
    return (row[0] if row else "data_only") or "data_only"


@router.get("/api/stocks", response_model=List[CompanyOut])
def list_stocks() -> List[CompanyOut]:
    """Return every ticker the platform knows about.

    Wave 9b — reads the `companies` table directly so the dropdown gets
    all S&P 100 + analyzed_on_demand entries in one query. The previous
    implementation iterated `data_service.list_tickers()` and made one
    live `get_company_profile` call per ticker; with 100+ universe size
    that was both slow (~10s) and lossy (a single provider miss dropped
    the ticker from the list). The companies row already has every
    field `CompanyOut` exposes, so no provider round-trip is needed.
    """
    fields = (
        "ticker", "company_name", "exchange", "sector", "industry",
        "sub_industry", "country", "currency", "market_cap",
        "business_description", "last_price", "is_etf", "beta",
        "shares_outstanding", "universe_tier",
    )
    with SessionLocal() as db:
        rows = db.execute(select(Company)).scalars().all()
        out: List[CompanyOut] = []
        for row in rows:
            payload = {f: getattr(row, f, None) for f in fields}
            payload["universe_tier"] = payload.get("universe_tier") or "data_only"
            out.append(CompanyOut(**payload))
    out.sort(key=lambda c: c.ticker)
    return out


@router.get("/api/stocks/{ticker}")
def get_stock(ticker: str) -> Dict[str, Any]:
    fin = get_full_financials(ticker.upper())
    if not fin.get("profile"):
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    stats = get_basic_stats(ticker.upper())
    # Overlay a live quote on the profile so the Research page shows
    # an intraday price, not the 7-day-cached profile.last_price.
    profile = dict(fin["profile"])
    quote = get_data_service().get_quote(ticker.upper())
    if quote and quote.get("price") is not None:
        profile["last_price"] = quote["price"]
    return dict(
        profile=profile,
        quote=quote,
        ratios=fin["ratios"],
        income=fin["income"],
        balance=fin["balance"],
        cash=fin["cash"],
        earnings=fin["earnings"],
        market_stats=stats,
    )


@router.get("/api/stocks/{ticker}/prices")
def get_stock_prices(ticker: str, days: int = 252) -> List[Dict[str, Any]]:
    rows = get_price_series(ticker.upper(), days)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No prices for {ticker}")
    return rows


def _parse_as_of(as_of: Optional[str]) -> Optional[_date]:
    """Parse + validate `?as_of=YYYY-MM-DD`. Future dates are rejected."""
    if not as_of:
        return None
    try:
        d = _date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(status_code=422,
                            detail=f"as_of must be YYYY-MM-DD; got {as_of!r}")
    if d > _date.today():
        raise HTTPException(status_code=422,
                            detail=f"as_of {d} is in the future")
    return d


@router.get("/api/stocks/{ticker}/memo", response_model=StockMemoOut)
@limiter.limit(LIMITS["memo_read"])
def get_stock_memo(
    request: Request,
    ticker: str,
    response: Response,
    scenario: str = "soft_landing",
    ondemand: bool = False,
    as_of: Optional[str] = Query(None, description="YYYY-MM-DD; backtest mode"),
) -> StockMemoOut:
    """Return the latest memo for `ticker`.

    Behavior:
    - If a stored snapshot exists, return it (cheap path) and stamp
      `X-Memo-Version` / `X-Memo-Trigger` / `X-Memo-Generated-At` headers
      so the UI can show "updated 2 days ago because of Q1 2026 earnings".
    - If no snapshot exists, run a full memo synchronously. For tickers in
      the `data_only` tier this requires `ondemand=true` to avoid
      surprise-charging the user; without the flag we 409 so the UI can
      surface an explicit "Analyze this stock" affordance.
    - When `as_of=YYYY-MM-DD` is passed (Wave 1C), the memo is reproduced
      as of that historical date. Backtest results are stored separately
      (won't shadow live memos) and skip long-term memory writes.
    """
    t = ticker.upper()
    as_of_date = _parse_as_of(as_of)

    # Backtest path: skip the cached-snapshot shortcut so we always
    # generate a fresh historical memo.
    if as_of_date is None:
        snap = memo_store.latest_memo(t)
        if snap is not None:
            # Wave 9b — Phase 2d. Check if a 10-Q/K or earnings call has
            # landed since this memo was generated; if so, recompute
            # rather than serve a stale cached version.
            freshness = memo_store.memo_freshness(snap)
            if not freshness["stale"]:
                response.headers["X-Memo-Version"] = str(snap.version)
                response.headers["X-Memo-Trigger"] = snap.trigger
                response.headers["X-Memo-Generated-At"] = snap.generated_at.isoformat()
                response.headers["X-Memo-Source"] = "cache"
                return memo_store.memo_to_pydantic(snap)
            # Stale — fall through to a fresh run, advertising why.
            response.headers["X-Memo-Stale-Reason"] = freshness["reason"]
            response.headers["X-Memo-Stale-Trigger"] = freshness["trigger"] or ""

    # Wave 9b — lazy ticker introduction. If `t` isn't in the
    # `companies` table at all, this resolves it via the live profile
    # chain, inserts it as `analyzed_on_demand`, and backfills 5yr of
    # financials so the agent graph has data to work with. Raises 404
    # when the symbol is genuinely unknown.
    tier = _ensure_lazy_universe(t)
    if as_of_date is None and tier == "data_only" and not ondemand:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{t} is in the data-only universe; pass ondemand=true to "
                "trigger the first deep analysis."
            ),
        )
    try:
        memo = run_stock_memo(t, scenario=scenario, as_of_date=as_of_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Promote `data_only` → `analyzed_on_demand` so subsequent calls use
    # the cached memo and don't re-trigger an expensive run automatically.
    # Skip promotion when running a backtest — that's diagnostic and
    # shouldn't change the universe state.
    if as_of_date is None and tier == "data_only":
        with SessionLocal() as db:
            row = db.get(Company, t)
            if row:
                row.universe_tier = "analyzed_on_demand"
                db.commit()

    fresh = memo_store.latest_memo(t, include_backtests=as_of_date is not None)
    if fresh is not None:
        response.headers["X-Memo-Version"] = str(fresh.version)
        response.headers["X-Memo-Trigger"] = fresh.trigger
        response.headers["X-Memo-Generated-At"] = fresh.generated_at.isoformat()
        if fresh.as_of_date:
            response.headers["X-Memo-As-Of"] = fresh.as_of_date.date().isoformat() if hasattr(fresh.as_of_date, "date") else str(fresh.as_of_date)
    return memo


@router.get("/api/stocks/{ticker}/memory")
def get_stock_memory(ticker: str, limit: int = 10) -> Dict[str, Any]:
    """Wave 8D — surface long-term memory entries for the UI.

    Returns the most recent `limit` entries from `memory/companies/<T>.md`
    plus any `structured_facts` blobs Wave 3D extracted from filings /
    transcripts. Read-only; the file itself remains the source of truth.
    """
    from ..memory import CompanyMemory
    from ..memory.longterm import company_memory_path
    cm = CompanyMemory.for_ticker(ticker.upper())
    entries = list(cm.entries[-limit:])
    return {
        "ticker": ticker.upper(),
        "path": str(company_memory_path(ticker.upper())),
        "entry_count": len(cm.entries),
        "historical_context": cm.historical_context or "",
        "entries": [
            {
                "date": e.date,
                "trigger": e.trigger,
                "body": e.body,
                "structured_facts": e.structured_facts,
            }
            for e in reversed(entries)  # newest-first
        ],
    }


@router.get("/api/stocks/{ticker}/memos")
def get_stock_memo_history(ticker: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Memo timeline for `ticker`, newest-first.

    Returns the metadata only (version / trigger / parent_version /
    revision_log / generated_at). Use `?version=N` on the singular memo
    endpoint to fetch a specific version's full body.
    """
    rows = memo_store.memo_history(ticker.upper(), limit=limit)
    return [
        {
            "version": r.version,
            "trigger": r.trigger,
            "parent_version": r.parent_version,
            "generated_at": r.generated_at.isoformat(),
            "revision_log": r.revision_log,
            "rating_label": (r.memo_json or {}).get("rating_label"),
            "confidence_score": (r.memo_json or {}).get("confidence_score"),
        }
        for r in rows
    ]


@router.post("/api/stocks/{ticker}/analyze", status_code=202)
@limiter.limit(LIMITS["memo_analyze"])
def analyze_stock(
    request: Request,
    response: Response,
    ticker: str,
    background_tasks: BackgroundTasks,
    scenario: Optional[str] = None,
    sync: bool = Query(False, description="If True, run synchronously and return the memo (will 504 on prod for full memos > 100s)."),
) -> Dict[str, Any]:
    """Trigger a fresh full memo regeneration.

    Returns 202 immediately by default; the regen runs in a
    BackgroundTasks worker. Frontend polls
    `GET /api/stocks/{ticker}/analyze/status` for completion (or just
    polls `/api/stocks/{ticker}/memo` and watches for the timestamp
    to advance — `latest_memo_at` in the status response is the same
    field).

    Pass `?sync=true` to run inline and return the StockMemoOut.
    Only useful in dev or behind a long-timeout proxy; on Render this
    will 504 after ~100s and the frontend will lose the response (the
    backend may still complete the work; check the status endpoint).
    """
    t = ticker.upper()
    _ensure_lazy_universe(t)
    sc = scenario or "soft_landing"

    if sync:
        try:
            memo = run_stock_memo(t, scenario=sc, force_refresh=True)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        snap = memo_store.latest_memo(t)
        if snap is not None:
            response.headers["X-Memo-Version"] = str(snap.version)
            response.headers["X-Memo-Trigger"] = snap.trigger
            response.headers["X-Memo-Generated-At"] = snap.generated_at.isoformat()
        # FastAPI's response_model coercion is bypassed because we
        # declared the return type as Dict; serialize via model_dump.
        return memo.model_dump()

    # Async path. Coalesce duplicate requests against the same ticker
    # so a frantic-click double-fire doesn't spawn two regens.
    with _REGEN_LOCK:
        already_running = t in _REGEN_JOBS
        if not already_running:
            _REGEN_JOBS[t] = datetime.utcnow()
        started_at = _REGEN_JOBS[t]

    if not already_running:
        background_tasks.add_task(_run_regen_job, t, sc)

    snap = memo_store.latest_memo(t)
    return {
        "ticker": t,
        "status": "in_progress" if already_running else "started",
        "started_at": started_at.isoformat(),
        "current_version": snap.version if snap else None,
        "current_generated_at": snap.generated_at.isoformat() if snap and snap.generated_at else None,
        "note": (
            "Memo regeneration runs in the background (5-9 min typical). "
            "Poll GET /api/stocks/{ticker}/analyze/status for completion."
        ),
    }


@router.get("/api/stocks/{ticker}/analyze/status")
def analyze_status(ticker: str) -> Dict[str, Any]:
    """Poll target for the async analyze flow.

    Returns:
      - `in_progress`: True while a background regen for this ticker
        is still running.
      - `started_at`: when the in-flight regen began (None when idle).
      - `latest_memo_at`: timestamp of the most recent persisted memo.
        Compare against the `started_at` you got from POST /analyze
        — once `latest_memo_at > your_started_at`, the new memo is
        ready to fetch.
      - `latest_version`: memo version for cache busting.
      - `last_failure`: the most recent regen failure for this ticker,
        if any. Cleared on next success. Includes error_type, message,
        and a truncated traceback so a 502/silent-failure is visible
        rather than spinning forever on the frontend.
    """
    t = ticker.upper()
    with _REGEN_LOCK:
        running_started_at = _REGEN_JOBS.get(t)
        last_failure = _REGEN_FAILURES.get(t)
    snap = memo_store.latest_memo(t)
    return {
        "ticker": t,
        "in_progress": running_started_at is not None,
        "started_at": running_started_at.isoformat() if running_started_at else None,
        "latest_memo_at": (
            snap.generated_at.isoformat() if snap and snap.generated_at else None
        ),
        "latest_version": snap.version if snap else None,
        "last_failure": last_failure,
    }
