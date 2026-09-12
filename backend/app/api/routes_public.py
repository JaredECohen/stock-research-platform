"""Public samples + analytics intake (FEAT-002, S3).

The logged-out surface of the API: three routes anyone may call and one
admin trigger. `auth/policy.py` already classifies the three as public
and the admin route falls under `admin_auth` by prefix.

    GET  /api/public/samples            list of curated tickers + build state
    GET  /api/public/samples/{ticker}   one sample; ETag / 304; 1h cache
    POST /api/public/events             browser analytics batch; always 200
    POST /api/admin/samples/rebuild     ask the worker to rebuild (admin token)

Nothing here can cost money. The module imports nothing from
`app.agents`, `app.providers` or `app.services.data_service`, the sample
routes read `public_samples` rows only, and a ticker outside
`settings.sample_tickers` is a 404 whether or not a memo exists for it.
`test_public_samples.py` pins all three with an AST check and a
monkeypatch that raises on any LLM / provider call.

Per-IP ceilings come from slowapi (`LIMITS["public_get"]`); the customer
limiter never sees these paths.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth.principal import current_principal
from ..auth.sanitize import safe_logger
from ..database import get_db
from ..rate_limit import LIMITS, limiter
from ..services import analytics_service, public_samples

log = safe_logger(__name__)
router = APIRouter()

SAMPLE_CACHE_CONTROL = "public, max-age=3600"
LIST_CACHE_CONTROL = "public, max-age=300"


def _not_found(ticker: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "not_found", "message": f"{ticker} is not a public sample",
                "sample_tickers": public_samples.allowlist()},
    )


@router.get("/api/public/samples")
@limiter.limit(LIMITS["public_get"])
def list_public_samples(request: Request, response: Response, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Every allowlisted ticker with its build state — built or not, so the
    landing page can render placeholders honestly."""
    response.headers["Cache-Control"] = LIST_CACHE_CONTROL
    return public_samples.list_samples(db)


@router.get("/api/public/samples/{ticker}")
@limiter.limit(LIMITS["public_get"])
def get_public_sample(ticker: str, request: Request, db: Session = Depends(get_db)) -> Response:
    """One curated sample. Strong ETag from the stored rows; `If-None-Match`
    → 304 with no body. Listed-but-unbuilt is 200 with nulls + `degraded`."""
    symbol = ticker.strip().upper()
    if not public_samples.is_listed(symbol):
        raise _not_found(symbol)
    payload, etag = public_samples.assemble(db, symbol)
    headers = {"ETag": etag, "Cache-Control": SAMPLE_CACHE_CONTROL, "Vary": "Accept-Encoding"}
    if public_samples.etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


@router.post("/api/public/events")
@limiter.limit(LIMITS["public_get"])
async def post_public_events(request: Request, response: Response, db: Session = Depends(get_db)) -> dict[str, int]:
    """Browser analytics. Always 200 with `{accepted, rejected}`.

    The body is read raw rather than through a Pydantic model so a
    malformed batch is a count, not a 422 — a public write endpoint
    should not explain its schema to whoever is probing it. Attribution:
    `X-Anon-Id` / `X-Session-Id` are opaque client ids; `user_id` is set
    only when the middleware verified a bearer token.
    """
    length = request.headers.get("content-length")
    try:
        declared = int(length) if length else 0
    except ValueError:
        declared = 0
    if declared > analytics_service.MAX_BODY_BYTES:
        return {"accepted": 0, "rejected": 0}
    raw = await request.body()
    if len(raw) > analytics_service.MAX_BODY_BYTES:
        return {"accepted": 0, "rejected": 0}
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        return {"accepted": 0, "rejected": 0}
    return analytics_service.ingest_batch(
        body, db=db, principal=current_principal(request),
        anon_id=request.headers.get("x-anon-id"), session_id=request.headers.get("x-session-id"),
    )


class SampleRebuildRequest(BaseModel):
    tickers: list[str] | None = Field(default=None, description="Subset of SAMPLE_TICKERS; all when omitted")


@router.post("/api/admin/samples/rebuild")
def rebuild_public_samples(body: SampleRebuildRequest | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Queue a rebuild. The work runs in the worker (`sample_build_loop`)
    on its next poll; this only writes the request row. Admin token
    required (prefix-guarded by `admin_auth`)."""
    tickers = body.tickers if body is not None else None
    try:
        return public_samples.request_rebuild(db, tickers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_ticker", "message": str(exc)}) from exc
