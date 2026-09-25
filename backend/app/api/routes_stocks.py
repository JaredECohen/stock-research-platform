"""Stock endpoints — list, detail, memo generation."""
from __future__ import annotations

import logging
from datetime import date as _date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..agents.graph import run_stock_memo
from ..auth.entitlements import EntitlementError, Grant, authorize, require_feature
from ..config import settings
from ..database import SessionLocal, get_db
from ..models import Company
from ..rate_limit import LIMITS, limiter
from ..schemas import CompanyOut, StockMemoOut
from ..schemas.quotes import QuotesOut
from ..seed_universe import ensure_company_in_universe
from ..services import memo_sections, memo_store, regen_worker
from ..services.data_service import get_data_service
from ..services.fundamentals_service import get_full_financials
from ..services.history_service import backfill_ticker
from ..services.market_data_service import get_basic_stats, get_price_series
from .gating import customer_wall_on, enforce_scope, feature_disabled, rate_scope

log = logging.getLogger(__name__)

router = APIRouter()


# Async memo regeneration (Theme 5). A full memo regen takes 5-9 minutes
# but Render's HTTP proxy timeout is ~100 seconds, so the analyze
# endpoint enqueues a durable `RegenJob` row and returns 202; a worker
# thread (services/regen_worker.py, started at app startup) drains the
# queue, and the frontend polls `/analyze/status` for completion.
#
# This replaced the in-memory `_REGEN_JOBS` / `_REGEN_FAILURES` dicts +
# request-scoped daemon thread: those died with the process, so OOM
# kills and deploys mid-regen left no trace and the frontend spun
# forever. Job rows survive restarts, orphaned runs are requeued once
# (resuming from MemoRunCheckpoint), and failures persist until the
# next success.


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
    # FEAT-003: place the new company in the GICS registry at introduction
    # so its first memo can find its Industry Group Analyst instead of
    # waiting for the 03:40 UTC classification loop. DB-only (one company
    # read, one current-row read, one insert); a no-op when no taxonomy is
    # active. Best-effort — a classification failure must not turn a
    # successful symbol introduction into a 500.
    try:
        from ..services import industry_classification
        industry_classification.classify_ticker(t)
    except Exception as exc:  # pragma: no cover — diagnostic only
        from ..agents.log_safety import safe_exc
        log.warning("industry classification skipped for %s: %s", t, safe_exc(exc))
    # Heavy load (5yr financials + filings + transcripts) so the agent
    # graph has data to work with. Best-effort — if a single capability
    # 403s, we still return the memo using whatever did land.
    try:
        backfill_ticker(t)
    except Exception as exc:  # preserve optional behavior, expose the failure
        log.warning("symbol introduction backfill failed ticker=%s error_type=%s", t, type(exc).__name__)
    # FIX-005: the legacy ingest above is anonymous and annual-only; ask the
    # worker's fundamentals drain for the durable FMP-primary import. One DB
    # row, no provider call here. Best-effort, like the backfill.
    try:
        from ..services import fundamental_refresh
        fundamental_refresh.request(t, "first_contact")
    except Exception as exc:
        log.warning("fundamentals first-contact request failed ticker=%s error_type=%s", t, type(exc).__name__)
    return "analyzed_on_demand"


def _company_tier(ticker: str) -> str:
    """Look up the universe tier for a ticker; returns 'data_only' if unknown."""
    with SessionLocal() as db:
        row = db.execute(
            select(Company.universe_tier).where(Company.ticker == ticker.upper())
        ).first()
    return (row[0] if row else "data_only") or "data_only"


@router.get("/api/stocks", response_model=list[CompanyOut])
def list_stocks(_rate: None = Depends(rate_scope("data"))) -> list[CompanyOut]:
    """Return every ticker the platform knows about.

    Wave 9b — reads the `companies` table directly so the dropdown gets
    every curated-universe (S&P 500 + extensions) and analyzed_on_demand
    entry in one query. The previous
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
        out: list[CompanyOut] = []
        for row in rows:
            payload = {f: getattr(row, f, None) for f in fields}
            payload["universe_tier"] = payload.get("universe_tier") or "data_only"
            out.append(CompanyOut(**payload))
    out.sort(key=lambda c: c.ticker)
    return out


@router.get("/api/stocks/{ticker}")
def get_stock(ticker: str, _rate: None = Depends(rate_scope("data"))) -> dict[str, Any]:
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


@router.get("/api/quotes", response_model=QuotesOut)
def get_quotes(
    response: Response,
    tickers: str = Query(..., max_length=1000, description="Comma-separated, at most 50"),
    _rate: None = Depends(rate_scope("data")),
) -> QuotesOut:
    """Live quotes for the Research and DCF Lab chips (W5b).

    `quote_service` owns the policy (15 minutes in session, until the next
    open after the close; labelled stale, then the stored close, then
    unavailable). An unknown symbol is never a 404 and never reaches a
    provider: it comes back `unavailable/unknown_ticker`, so the list keeps
    its shape. The browser may reuse an answer for a minute; the shared
    provider cache does the rest.
    """
    from ..finance.market_calendar import CalendarUnavailable
    from ..schemas.quotes import QuoteOut
    from ..services import quote_service

    try:
        keys = quote_service.normalize_tickers(tickers.split(","))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        quotes = quote_service.get_quotes(keys)
        market = quote_service.market_state_out()
    except CalendarUnavailable as exc:
        # tzdata missing from the image: the chip renders nothing on an
        # error, and the Dockerfile build check should have caught it.
        log.error("quotes unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="market calendar unavailable") from exc
    response.headers["Cache-Control"] = "private, max-age=60"
    return QuotesOut(
        quotes=[QuoteOut.model_validate(quotes[k]) for k in keys],
        market=market,  # type: ignore[arg-type]
        ttl_seconds_in_session=quote_service.QUOTE_TTL_SECONDS,
    )


@router.get("/api/stocks/{ticker}/prices")
def get_stock_prices(
    ticker: str, days: int = 252, _rate: None = Depends(rate_scope("series")),
) -> list[dict[str, Any]]:
    rows = get_price_series(ticker.upper(), days)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No prices for {ticker}")
    return rows


def _parse_as_of(as_of: str | None) -> _date | None:
    """Parse + validate `?as_of=YYYY-MM-DD`. Future dates are rejected."""
    if not as_of:
        return None
    try:
        d = _date.fromisoformat(as_of)
    except ValueError as exc:
        raise HTTPException(status_code=422,
                            detail=f"as_of must be YYYY-MM-DD; got {as_of!r}") from exc
    if d > _date.today():
        raise HTTPException(status_code=422,
                            detail=f"as_of {d} is in the future")
    return d


def _stamp_snapshot_headers(response: Response, snap: Any) -> None:
    response.headers["X-Memo-Version"] = str(snap.version)
    response.headers["X-Memo-Trigger"] = snap.trigger
    response.headers["X-Memo-Generated-At"] = snap.generated_at.isoformat()


def _memo_or_unreadable(snap: Any) -> StockMemoOut:
    """The presented memo (`memo_store.present_snapshot`, W2a: template-filled
    sections read "Unavailable in this version."), answering a stored
    snapshot that no longer validates with a structured 422 instead of an
    ASGI 500 (FIX-004). Every stored-memo branch of `GET /memo` — version,
    cache and store-only — goes through here.

    422, not 409: the Research page turns every 409 into the "Analyze
    this stock" gate, which would regenerate (or charge a research run)
    over a row the owner has ruled must be refused as stored. The row is
    not touched; the log names it without quoting it (no exc_info: the
    chained ValidationError carries stored text)."""
    try:
        return memo_store.present_snapshot(snap)
    except memo_store.StoredMemoUnreadable as exc:
        log.error(
            "stored memo unreadable ticker=%s version=%s snapshot_id=%s fields=%s",
            exc.ticker, exc.version, exc.snapshot_id, ",".join(exc.fields),
        )
        raise EntitlementError(
            422, "memo_unreadable",
            f"The stored memo for {exc.ticker} (version {exc.version}) cannot be displayed: "
            "it was saved in a format this version of MarketMosaic cannot read. "
            "The stored record has not been changed.",
            feature="memo_view",
            extra={"ticker": exc.ticker, "version": exc.version, "fields": list(exc.fields)},
        ) from exc


def _memo_from_store_only(
    request: Request,
    response: Response,
    db: Session,
    t: str,
    *,
    scenario: str,
    ondemand: bool,
    as_of: str | None,
) -> Any:
    """`GET /memo` when the web process may not generate (login wall on).

    The agent graph and the provider backfill only ever run in the worker
    here; this function reads `memo_snapshots` and nothing else. The
    branches, in order:

    - `as_of` → 404 `feature_disabled`. Backtests re-run the graph for a
      historical date; there is no stored artefact to serve and no worker
      path for them this phase.
    - no snapshot, `ondemand=false` → 409 `no_memo` with `analyze_path`, so
      the UI can offer a (charged) research run. Nothing is metered: the
      customer saw no memo.
    - no snapshot, `ondemand=true` → the analyze path (a `research_run`
      reservation + a queued job) and 202 with the analyze payload. It
      draws on the `research` rate window (3/hour/user) exactly as
      `POST /analyze` does — the route's own `data` scope is 120/minute,
      which would otherwise be the only ceiling on charged runs started
      through a GET.
    - snapshot → `memo_view` is authorized (Free: 3 distinct tickers a
      month) and the snapshot is served. A stale snapshot is still served,
      flagged with `X-Memo-Stale: true` and the reason, rather than
      regenerated in the request as the wall-off path does.

    `memo_view` is authorized only once a snapshot is known to exist,
    rather than as a route dependency, so the 409 branch never charges:
    the customer saw no memo, and `/api/me/usage` must not list a
    reserve-then-release for it.
    """
    if as_of:
        _parse_as_of(as_of)  # keep today's 422 for a malformed date
        raise feature_disabled(
            f"Backtest memos (as_of) are not available to customer accounts for {t}.",
            feature="memo_view", ticker=t,
        )
    snap = memo_store.latest_memo(t)
    if snap is None:
        if ondemand:
            enforce_scope(request, db, "research")
            return _enqueue_research_run(request, db, t, scenario)
        raise EntitlementError(
            409, "no_memo",
            f"No stored memo for {t} yet. Run research to generate one.",
            feature="memo_view",
            extra={"ticker": t, "analyze_path": f"/api/stocks/{t}/analyze"},
        )

    grant = authorize(request, "memo_view", resource=t, db=db)
    try:
        freshness = memo_store.memo_freshness(snap)
        if freshness["stale"]:
            response.headers["X-Memo-Stale"] = "true"
            response.headers["X-Memo-Stale-Reason"] = freshness["reason"]
            response.headers["X-Memo-Stale-Trigger"] = freshness["trigger"] or ""
        _stamp_snapshot_headers(response, snap)
        response.headers["X-Memo-Source"] = "cache"
        memo = _memo_or_unreadable(snap)
    except Exception:
        grant.release(db)
        raise
    grant.commit(db)
    return memo


@router.get("/api/stocks/{ticker}/memo", response_model=StockMemoOut)
@limiter.limit(LIMITS["memo_read"])
def get_stock_memo(
    request: Request,
    ticker: str,
    response: Response,
    scenario: str = "soft_landing",
    ondemand: bool = False,
    as_of: str | None = Query(None, description="YYYY-MM-DD; backtest mode"),
    version: int | None = Query(None, ge=1, description="Exact stored memo version; never generates"),
    db: Session = Depends(get_db),
    _rate: None = Depends(rate_scope("data")),
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

    FEAT-002: everything after the cheap path above assumes the web
    process may run the agent graph. With the login wall on it never
    does — `memo_view` is metered and generation belongs to the worker —
    and the request is answered from the store alone; see
    `_memo_from_store_only` for the 409 / 202 / stale contract the
    frontend handles. `MEMO_INLINE_GENERATION=false` forces that
    store-only path with the wall off too (worker-only serving with no
    accounts, uncharged since `authorize` is a no-op there). The reverse
    override is deliberately not honoured: `MEMO_INLINE_GENERATION=true`
    under the wall would route customers down the legacy branch, which
    meters nothing and runs the graph plus the provider backfill
    in-request for free.
    """
    t = ticker.upper()
    if version is not None:
        if as_of or ondemand:
            raise HTTPException(status_code=422, detail="version cannot be combined with as_of or ondemand")
        snap = memo_store.memo_version(t, version)
        if snap is None:
            raise HTTPException(status_code=404, detail="Stored memo version not found")
        grant = authorize(request, "memo_view", resource=t, db=db)
        try:
            memo = _memo_or_unreadable(snap)
            _stamp_snapshot_headers(response, snap)
            response.headers["X-Memo-Source"] = "cache"
        except Exception:
            grant.release(db)
            raise
        grant.commit(db)
        return memo
    if customer_wall_on() or not settings.memo_inline_generation_effective:
        return _memo_from_store_only(
            request, response, db, t, scenario=scenario, ondemand=ondemand, as_of=as_of,
        )
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
                return _memo_or_unreadable(snap)
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
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
    # W2a: the fresh run is presented like a stored read. A run is never a
    # news patch, so there is no chain to resolve.
    return memo_sections.present_memo(memo)


@router.get("/api/stocks/{ticker}/memory")
def get_stock_memory(
    ticker: str,
    limit: int = 10,
    _rate: None = Depends(rate_scope("data")),
    _grant: Grant = Depends(require_feature("memo_history")),
) -> dict[str, Any]:
    """Wave 8D — surface long-term memory entries for the UI.

    Returns the most recent `limit` entries from `memory/companies/<T>.md`
    plus any `structured_facts` blobs Wave 3D extracted from filings /
    transcripts. Read-only; the file itself remains the source of truth.

    W7: once learned priors are being injected (effective mode "inject"),
    the trail is the learning ledger instead — company-scope filing
    observations and non-suppressed lessons, each lesson labelled a
    provisional hypothesis, never the audit-only `detail`, in the same
    response shape. In off and shadow modes this route is unchanged.
    """
    from ..learning import context as learning_context
    if learning_context.public_trail_enabled():
        return learning_context.public_trail(ticker, limit=limit)
    from ..memory import CompanyMemory
    from ..memory.longterm import company_memory_path
    cm = CompanyMemory.for_ticker(ticker.upper())
    # W2a (critique delta 3): the trail is a memo exit too. An entry the
    # no-LLM reflection branch wrote restates the memo in a fixed frame, and
    # an entry quoting a section the latest memo hides would re-publish the
    # template text the memo page withholds; both are left out of the
    # response (the file is untouched) and counted.
    hidden = _hidden_memo_texts(ticker.upper())
    shown = [
        e for e in cm.entries
        if not memo_sections.is_reflection_template(e.body)
        and not any(probe in (e.body or "") for probe in hidden)
    ]
    entries = list(shown[-limit:])
    # The condensed block folds old entries in as one-line takeaways (the
    # first 160 characters of each body), so it can quote the same template
    # text; the same probes apply line by line.
    context_lines = (cm.historical_context or "").split("\n")
    context_kept = [
        line for line in context_lines
        if not memo_sections.is_condensed_reflection_template(line)
        and not any(probe in line for probe in hidden)
    ]
    return {
        "ticker": ticker.upper(),
        "path": str(company_memory_path(ticker.upper())),
        "entry_count": len(cm.entries),
        "suppressed_count": len(cm.entries) - len(shown),
        "historical_context": "\n".join(context_kept) if cm.historical_context else "",
        "historical_context_suppressed": len(context_lines) - len(context_kept),
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


def _hidden_memo_texts(ticker: str) -> list[str]:
    """Probes for the prose the latest memo hides: the first 80 characters
    of each unavailable prose section's stored text (the reflection writer
    truncates what it quotes, so a prefix is what an entry can contain).
    Empty when there is no readable memo — the template filter still runs."""
    snap = memo_store.latest_memo(ticker)
    if snap is None:
        return []
    try:
        raw = memo_store.memo_to_pydantic(snap)
        presented = memo_store.present_snapshot(snap)
    except memo_store.StoredMemoUnreadable:
        return []
    texts: list[str] = []
    for key, entry in presented.section_availability.items():
        if (entry.status != "unavailable" or entry.reason == "not_produced"
                or key.endswith(".long_form_report")):
            # A drill-down body is never quoted by a reflection entry.
            continue
        if key in ("final_pm_view", "one_sentence_thesis", "final_verdict"):
            texts.append(getattr(raw, key) or "")
        elif key == "mispricing_thesis":
            texts.extend([raw.mispricing_thesis.our_view, raw.mispricing_thesis.gap])
        elif key.endswith("_view") or key == "macro_sensitivity" or key.startswith("extra_agent_views."):
            finding = (raw.extra_agent_views.get(key.split(".", 1)[1]) if key.startswith("extra_agent_views.")
                       else getattr(raw, key, None))
            if finding is not None:
                texts.extend([finding.headline, finding.summary])
    return [t.strip()[:80] for t in texts if t and len(t.strip()) >= 40]


@router.get("/api/stocks/{ticker}/memos")
def get_stock_memo_history(
    ticker: str,
    limit: int = Query(25, ge=1, le=50),
    _rate: None = Depends(rate_scope("data")),
    _grant: Grant = Depends(require_feature("memo_history")),
) -> list[dict[str, Any]]:
    """Memo timeline for `ticker`, newest-first.

    Returns the metadata only (version / trigger / parent_version /
    revision_log / generated_at). Use `?version=N` on the singular memo
    endpoint to fetch a specific version's full body.

    W2a: each row's confidence is the presented one — `null` with
    `confidence_available: false` when that version's confidence is hidden
    (a template PM view), the number with `true` otherwise, and the raw
    number with `null` when the stored row no longer validates. `limit` is
    bounded (1..50) because every row is now validated and presented; the
    patch-chain walk shares one session and one memo across the rows.
    """
    rows = memo_store.memo_history(ticker.upper(), limit=limit)
    cache: memo_store.ChainCache = {
        (r.ticker, r.version): (r.trigger, r.parent_version, r.revision_log) for r in rows
    }
    with SessionLocal() as db:
        return [_history_row(r, db, cache) for r in rows]


def _history_row(r: Any, db: Session, cache: memo_store.ChainCache) -> dict[str, Any]:
    raw = r.memo_json or {}
    row: dict[str, Any] = {
        "version": r.version,
        "trigger": r.trigger,
        "parent_version": r.parent_version,
        "generated_at": r.generated_at.isoformat(),
        "revision_log": r.revision_log,
        "rating_label": raw.get("rating_label") if isinstance(raw, dict) else None,
        "confidence_score": raw.get("confidence_score") if isinstance(raw, dict) else None,
        "confidence_available": None,
    }
    try:
        presented = memo_store.present_snapshot(r, db=db, cache=cache)
    except memo_store.StoredMemoUnreadable:
        return row
    hidden = memo_sections.is_hidden(presented, "confidence_score")
    row["rating_label"] = presented.rating_label
    row["confidence_score"] = None if hidden else presented.confidence_score
    row["confidence_available"] = not hidden
    return row


def _analyze_payload(t: str, job: dict[str, Any], created: bool, *, charged: bool) -> dict[str, Any]:
    snap = memo_store.latest_memo(t)
    return {
        "ticker": t,
        "status": "started" if created else "in_progress",
        # The polling contract compares `latest_memo_at` against this
        # value, so it must predate the memo the job will persist —
        # enqueue time qualifies even while the job is still queued.
        "started_at": job["started_at"] or job["enqueued_at"],
        "job_id": job["id"],
        "current_version": snap.version if snap else None,
        "current_generated_at": snap.generated_at.isoformat() if snap and snap.generated_at else None,
        # FEAT-002: True only when THIS request reserved a research run.
        # A coalesced request rides on someone else's job for free.
        "charged": charged,
        "note": (
            "Memo regeneration runs in the background (5-9 min typical). "
            "Poll GET /api/stocks/{ticker}/analyze/status for completion."
        ),
    }


def _enqueue_research_run(request: Request, db: Session, t: str, scenario: str) -> JSONResponse:
    """The charged analyze path (login wall on).

    Reserve a `research_run`, then enqueue. The reservation is handed to
    the job when the job was actually created — the worker commits it
    when the memo persists and releases it when the run fails or is
    orphaned — and released right here when the request coalesced onto a
    job that already existed, so the customer pays for their own run and
    never for someone else's.

    The meter key is per call (see `authorize`: a random key means "charge
    each call"). Retry safety comes from the job queue, not from the key:
    a retried POST lands on the still-active job as `created=False` and
    its own reservation is released. A client-supplied key would be
    worse, not better — a replayed key on a job that has already finished
    replays the committed event as "allowed" and starts a second,
    uncharged run.

    Lazy universe resolution (profile lookup + the 5-year backfill for a
    ticker the database has never seen) does not happen here: it is
    provider spend, so it runs in the worker inside the job that was
    charged for it.
    """
    grant = authorize(request, "research_run", resource=t, db=db)
    try:
        job, created = regen_worker.enqueue(
            t, scenario, source="user",
            requested_by_user_id=grant.user_id, usage_event_id=grant.usage_event_id,
        )
    except Exception:
        grant.release(db)
        raise
    if not created:
        grant.release(db)
    return JSONResponse(
        status_code=202,
        content=_analyze_payload(t, job, created, charged=created and grant.charged),
    )


@router.post("/api/stocks/{ticker}/analyze", status_code=202)
@limiter.limit(LIMITS["memo_analyze"])
def analyze_stock(
    request: Request,
    response: Response,
    ticker: str,
    scenario: str | None = None,
    sync: bool = Query(False, description="If True, run synchronously and return the memo (will 504 on prod for full memos > 100s)."),
    db: Session = Depends(get_db),
    _rate: None = Depends(rate_scope("research")),
) -> dict[str, Any]:
    """Trigger a fresh full memo regeneration.

    Returns 202 immediately by default after enqueuing a durable
    `RegenJob`; the worker thread picks it up. Frontend polls
    `GET /api/stocks/{ticker}/analyze/status` for completion (or just
    polls `/api/stocks/{ticker}/memo` and watches for the timestamp
    to advance — `latest_memo_at` in the status response is the same
    field).

    Pass `?sync=true` to run inline and return the StockMemoOut.
    Only useful in dev or behind a long-timeout proxy; on Render this
    will 504 after ~100s and the frontend will lose the response (the
    backend may still complete the work; check the status endpoint).

    FEAT-002, login wall on: `sync=true` is refused (403 — the web process
    never generates for a customer), the run is metered as a
    `research_run` (Free 1 / Pro 20 a month; 402 when spent, with no job
    row), and lazy universe resolution moves into the worker. See
    `_enqueue_research_run`.
    """
    t = ticker.upper()
    sc = scenario or "soft_landing"

    if customer_wall_on():
        if sync:
            raise feature_disabled(
                "Synchronous generation is not available to customer accounts; "
                "POST without sync=true and poll /analyze/status.",
                feature="research_run", status_code=403, ticker=t,
            )
        return _enqueue_research_run(request, db, t, sc)

    _ensure_lazy_universe(t)

    if sync:
        try:
            memo = run_stock_memo(t, scenario=sc, force_refresh=True)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        snap = memo_store.latest_memo(t)
        if snap is not None:
            _stamp_snapshot_headers(response, snap)
        # The route default is 202 (async enqueue). The sync path runs
        # inline and returns the memo, so override back to 200 — otherwise
        # `?sync=true` returns a body with a misleading 202 status.
        response.status_code = 200
        # FastAPI's response_model coercion is bypassed because we
        # declared the return type as Dict; serialize via model_dump.
        # Presented (W2a) like every other exit that serves a memo.
        return memo_sections.present_memo(memo).model_dump()

    # Async path. `enqueue` coalesces duplicate requests against the
    # same ticker so a frantic-click double-fire doesn't queue two
    # regens — and, unlike the old in-memory registry, the coalescing
    # holds across process restarts.
    job, created = regen_worker.enqueue(t, sc)
    return _analyze_payload(t, job, created, charged=False)


@router.get("/api/stocks/{ticker}/analyze/status")
def analyze_status(ticker: str, _rate: None = Depends(rate_scope("data"))) -> dict[str, Any]:
    """Poll target for the async analyze flow.

    Returns:
      - `in_progress`: True while a regen job for this ticker is
        queued or running.
      - `started_at`: when the in-flight regen began (None when idle).
      - `latest_memo_at`: timestamp of the most recent persisted memo.
        Compare against the `started_at` you got from POST /analyze
        — once `latest_memo_at > your_started_at`, the new memo is
        ready to fetch.
      - `latest_version`: memo version for cache busting.
      - `last_failure`: the most recent regen failure for this ticker,
        if any. Cleared on next success. Includes error_type, message,
        and a truncated traceback so a 502/silent-failure is visible
        rather than spinning forever on the frontend. Process deaths
        (OOM kill, deploy) now surface here too, as `WorkerRestart`.
      - `last_progress`: step trace of the most recent regen — worker
        waypoints merged with per-step `MemoRunCheckpoint` completions,
        so a dead run shows exactly which step it reached.
      - `job_id` / `job_status`: the underlying `RegenJob` row, for
        cross-referencing `/api/admin/regen-jobs` telemetry.
    """
    t = ticker.upper()
    state = regen_worker.ticker_status(t)
    snap = memo_store.latest_memo(t)
    return {
        "ticker": t,
        "in_progress": state["in_progress"],
        "started_at": state["started_at"],
        "latest_memo_at": (
            snap.generated_at.isoformat() if snap and snap.generated_at else None
        ),
        "latest_version": snap.version if snap else None,
        "last_failure": state["last_failure"],
        "last_progress": state["progress"][-20:],  # last 20 entries
        "job_id": state["job_id"],
        "job_status": state["job_status"],
    }
