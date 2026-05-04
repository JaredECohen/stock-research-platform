"""FastAPI app for MarketMosaic.

Wires up middleware, routers, and startup tasks (init DB + seed demo data).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import (
    routes_admin,
    routes_chat,
    routes_comps,
    routes_dcf,
    routes_health,
    routes_macro,
    routes_portfolio,
    routes_screener,
    routes_stocks,
)
from .config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("marketmosaic")


# Paths that don't deserve a UILog row (very noisy + low signal).
_HTTP_LOG_SKIP_PATHS = {"/api/admin/ui-log"}


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
                            "query": dict(request.query_params),
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

    # Wave 8G — HTTP request tracing. Every API call emits a structured
    # log line + a UILog row so frontend traces and backend traces sit
    # in one timeline.
    app.middleware("http")(_http_logging_middleware)

    app.include_router(routes_health.router, tags=["system"])
    app.include_router(routes_stocks.router, tags=["stocks"])
    app.include_router(routes_screener.router, tags=["screener"])
    app.include_router(routes_chat.router, tags=["chat"])
    app.include_router(routes_dcf.router, tags=["dcf"])
    app.include_router(routes_comps.router, tags=["comps"])
    app.include_router(routes_portfolio.router, tags=["portfolio"])
    app.include_router(routes_macro.router, tags=["macro"])
    app.include_router(routes_admin.router, tags=["admin"])

    @app.on_event("startup")
    def _startup() -> None:
        try:
            from .seed_demo_data import run_full_seed
            summary = run_full_seed()
            log.info("MarketMosaic seeded: %s", summary)
        except Exception as exc:  # pragma: no cover - startup hardening
            log.warning("Seed failed at startup: %s", exc)

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
