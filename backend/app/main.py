"""FastAPI app for MarketMosaic.

Wires up middleware, routers, and startup tasks (init DB + seed demo data).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .api import (
    routes_account,
    routes_admin,
    routes_billing,
    routes_chat,
    routes_comps,
    routes_data_catalog,
    routes_dcf,
    routes_fundamentals,
    routes_fundamentals_commentary,
    routes_health,
    routes_industries,
    routes_industries_admin,
    routes_learning_admin,
    routes_macro,
    routes_portfolio,
    routes_public,
    routes_scorecard,
    routes_screener,
    routes_stocks,
    routes_unit_economics,
)
from .api.admin_auth import admin_auth_middleware
from .auth.middleware import customer_auth_middleware
from .config import settings
from .log_setup import configure_logging
from .rate_limit import limiter, rate_limit_exceeded_handler

# See `app/log_setup.py`: also silences httpx's per-request logging and
# scrubs query-string credentials, which were leaking into production logs.
configure_logging()
log = logging.getLogger("marketmosaic")


# Paths that don't deserve a UILog row (very noisy + low signal).
_HTTP_LOG_SKIP_PATHS = {"/api/admin/ui-log"}

# Query keys whose VALUES must never be persisted to `ui_logs.payload`.
# The customer auth middleware only reads `Authorization`, so a `?token=`
# is ignored for auth — but this middleware stores every query param, and
# a credential pasted into a URL would otherwise sit in the database.
_REDACTED_QUERY_KEYS = {"token", "authorization", "access_token", "api_key", "apikey", "key", "secret"}


def _safe_query(params) -> dict:
    return {k: ("<redacted>" if k.lower() in _REDACTED_QUERY_KEYS else v) for k, v in dict(params).items()}


async def _http_logging_middleware(request: Request, call_next):
    """Append a `UILog` row + a `marketmosaic.http` log line for every
    backend request. Errors in the logging path are swallowed so they
    never break the actual request."""
    started = time.perf_counter()
    response = None
    error_str = ""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error_str = repr(exc)
        raise
    finally:
        try:
            duration_ms = int((time.perf_counter() - started) * 1000)
            status = response.status_code if response is not None else 500
            path = request.url.path
            method = request.method
            log.info("HTTP %s %s -> %s in %dms", method, path, status, duration_ms)
            if path not in _HTTP_LOG_SKIP_PATHS:
                from .database import SessionLocal
                from .models import UILog
                with SessionLocal() as db:
                    UILog.__table__.create(bind=db.get_bind(), checkfirst=True)
                    db.add(UILog(
                        ts=datetime.utcnow(), source="backend", kind="http",
                        path=path, method=method, status_code=status,
                        duration_ms=duration_ms,
                        session_id=request.headers.get("x-session-id"),
                        payload={
                            "query": _safe_query(request.query_params),
                            "error": error_str or None,
                        },
                    ))
                    db.commit()
        except Exception as exc:  # pragma: no cover — logging must never raise
            log.debug("ui-log write failed: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title="MarketMosaic API",
        description=(
            "Multi-agent equity research and portfolio management platform. "
            "Research / education only — not personalized financial advice."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Per-IP rate limiting. `state.limiter` is the canonical pointer
    # slowapi looks up at request time; `SlowAPIMiddleware` is what
    # actually evaluates the configured per-route limits.
    app.state.limiter = limiter
    # Structured 429 body (`code`, `scope`, `retry_after`, `window_seconds`,
    # `message`) plus a `Retry-After` header, so the per-IP slowapi limits
    # and the per-user DB limits in `auth/ratelimit.py` look identical to
    # the frontend's RateLimitNotice.
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Wave 8G — HTTP request tracing. Every API call emits a structured
    # log line + a UILog row so frontend traces and backend traces sit
    # in one timeline.
    app.middleware("http")(_http_logging_middleware)

    # Admin/ops auth. Starlette runs the LAST-registered
    # `app.middleware("http")` OUTERMOST (verified with a two-middleware
    # probe on starlette 1.0 — an earlier comment here claimed the
    # opposite), so this runs OUTSIDE the request logger: a rejected admin
    # call gets a uvicorn access-log line and the WARNING `admin_auth`
    # emits, but no ui_logs row. Acceptable for a login wall — persisting
    # a row per unauthenticated probe is a cheap way to fill the database.
    # Applied as middleware rather than per-route dependencies so a newly
    # added admin endpoint is covered the moment it is mounted; see
    # `admin_auth` and `test_admin_auth.py`.
    app.middleware("http")(admin_auth_middleware)

    # FEAT-002 customer auth. Registered LAST, so it runs OUTERMOST —
    # before `admin_auth_middleware` and before the request logger.
    # `auth/policy.py` classifies the admin prefix as "not mine", so
    # /api/admin/* passes through untouched for admin_auth to judge: the
    # two guards never overlap, the admin token never satisfies a customer
    # route and a customer JWT never satisfies `/api/admin/*`. Its own
    # refusals (401/503) reach the access log but not ui_logs, for the
    # same reason as above. With AUTH_ENABLED=false it is a pass-through
    # that still attaches an anonymous `request.state.principal` so route
    # code has one code path. See `auth/middleware.py` for the order note.
    app.middleware("http")(customer_auth_middleware)

    app.include_router(routes_health.router, tags=["system"])
    app.include_router(routes_stocks.router, tags=["stocks"])
    app.include_router(routes_screener.router, tags=["screener"])
    app.include_router(routes_chat.router, tags=["chat"])
    app.include_router(routes_dcf.router, tags=["dcf"])
    app.include_router(routes_comps.router, tags=["comps"])
    app.include_router(routes_portfolio.router, tags=["portfolio"])
    app.include_router(routes_macro.router, tags=["macro"])
    app.include_router(routes_data_catalog.router, tags=["data-catalog"])
    app.include_router(routes_admin.router, tags=["admin"])
    # FEAT-002 pricing evidence: observed cost per metered operation, read
    # out of `llm_call_logs` and expressed against the plan allowances. Its
    # own module rather than another handler in `routes_admin` because the
    # report is a service with real arithmetic behind it; the path sits under
    # `/api/admin`, so the token middleware covers it the moment it mounts.
    app.include_router(routes_unit_economics.router, tags=["admin"])
    # FEAT-002: account + public config (S1), public samples/events (S3)
    # and billing (S4). The latter two are stubs until their slices land;
    # including them here means `import app.main` never breaks on a
    # missing module when the slices merge in any order.
    app.include_router(routes_account.router, tags=["account"])
    app.include_router(routes_public.router, tags=["public"])
    app.include_router(routes_billing.router, tags=["billing"])
    # FEAT-001: catalog + series (S2) and commentary (S3 fills the stub).
    # One flag mounts both, so ENABLE_FUNDAMENTALS_EXPLORER=false is the
    # whole rollback: the page sees 404s and nothing else changes.
    if settings.enable_fundamentals_explorer:
        app.include_router(routes_fundamentals.router, tags=["fundamentals"])
        app.include_router(routes_fundamentals_commentary.router, tags=["fundamentals"])
    # Phase 6: scorecard reads + the v1 export. Always mounted so the route
    # set (and the auth policy coverage test) is static; ENABLE_SCORECARD
    # is checked per request and turns the whole surface into 404s.
    app.include_router(routes_scorecard.router, tags=["scorecard"])
    # FEAT-003: Industry Analysis reads (public, `/api/industries/*`) and
    # the ops surface (`/api/admin/industries/*`, guarded by the admin
    # prefix middleware). Always mounted, for the same reason as the
    # scorecard: the route set stays static, and a deployment with no
    # taxonomy imported answers 503 with the remedy rather than 404ing a
    # route the frontend expects to exist. Nothing here generates — the
    # worker owns report generation.
    app.include_router(routes_industries.router, tags=["industries"])
    app.include_router(routes_industries_admin.router, tags=["industries", "admin"])
    # W7 learning ledger ops (status, mode, items, renders, preview, and the
    # S17 corpus census). Under `/api/admin`, so the token middleware guards
    # every one of them; none is browser-called. DB-only — nothing here calls
    # a provider or a model, so it is safe on the web process.
    app.include_router(routes_learning_admin.router, tags=["admin"])

    @app.on_event("startup")
    def _startup() -> None:
        # Fail closed before lifespan completion, even on a web-only process:
        # enqueue/status routes also query the mapped regeneration job schema.
        # This is DB-only and deliberately outside the best-effort seed block.
        from .database import bootstrap_runtime_schema
        bootstrap_runtime_schema()

        # One line saying which provider and which model each role actually
        # resolved to — the answer to "why did the sector agent run on
        # haiku?" without grepping env. Names and booleans only.
        try:
            from .agents.llm import model_summary
            ms = model_summary()
            roles = " ".join(f"{r}={m}" for r, m in ms["role_models"].items())
            cfg = ",".join(k for k, v in ms["configured"].items() if v) or "none"
            log.info(
                "LLM routing: provider=%s choice=%s configured=%s failover=%s %s",
                ms["active_provider"], ms["provider_choice"], cfg,
                "on" if settings.llm_failover_enabled else "off", roles,
            )
        except Exception as exc:  # pragma: no cover - startup hardening
            log.warning("LLM routing summary failed: %s", type(exc).__name__)

        try:
            from .seed_universe import run_full_seed
            summary = run_full_seed()
            log.info("MarketMosaic seeded: %s", summary)
        except Exception as exc:  # pragma: no cover - startup hardening
            log.warning("Seed failed at startup: %s", exc)

        # Theme 5: memo-regen worker. Drains the durable `regen_jobs`
        # queue that POST /analyze writes to. Started after the seed so
        # requeued/recovered jobs see a fully-initialized universe.
        # No-ops under pytest (tests drain the queue via
        # `regen_worker.process_next_job()`), and when
        # ENABLE_REGEN_WORKER=false.
        try:
            from .services.regen_worker import start_worker
            start_worker()
        except Exception as exc:  # pragma: no cover - startup hardening
            log.warning("Regen worker failed to start: %s", exc)

        # Phase 5: register always-on monitoring loops if enabled. Default off
        # in dev/test so the test client doesn't spin up a background scheduler.
        if settings.enable_monitoring:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore

                from .monitoring import register_all
                scheduler = BackgroundScheduler(daemon=True)
                register_all(scheduler)
                scheduler.start()
                app.state.scheduler = scheduler
                log.info("Monitoring scheduler started.")
            except Exception as exc:  # pragma: no cover
                log.warning("Monitoring failed to start: %s", exc)

    return app


app = create_app()
