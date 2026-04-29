"""FastAPI app for MarketMosaic.

Wires up middleware, routers, and startup tasks (init DB + seed demo data).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
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
