"""Portfolio endpoints."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..agents.llm import llm_call_context
from ..agents.log_safety import log_safely
from ..auth.entitlements import Grant, require_feature
from ..auth.principal import current_principal
from ..finance.portfolio_construction import industry_group_exposure
from ..schemas import ModelPortfolio, PortfolioRequest
from ..schemas.portfolio_exposure import ModelPortfolioWithExposure
from ..services.portfolio_service import build_model_portfolio
from .gating import rate_scope

log = logging.getLogger(__name__)
router = APIRouter()


def industry_exposure_block(portfolio: ModelPortfolio) -> dict[str, Any]:
    """FEAT-003 — exposure by industry group for the built holdings.

    Reads only: one SELECT for the holdings' current classifications, the
    registry's cached node list, and the latest persisted cross-industry
    snapshot (one row) for the exposed groups' rows. Never raises into the
    build — a missing taxonomy or a read failure is reported as a labelled
    state, and the portfolio still returns.

    Public: groups are named by MarketMosaic's own labels and addressed by
    slug, and the block is projected through `industry_labels` before it
    leaves (owner decision 2026-09-24 — no licensed names or codes).
    """
    tickers = [h.ticker for h in portfolio.holdings]
    try:
        from ..services import gics_registry, industry_classification, industry_labels, industry_snapshot

        info = gics_registry.active_version()
        if info is None:
            return {"status": "taxonomy_not_imported", "by_group": [], "unmapped": [
                {"ticker": t.upper(), "weight": None, "state": "unclassified"} for t in tickers
            ]}
        current = industry_classification.current_for(tickers, version=info) if tickers else {}
        names = {n.code: industry_labels.label(n.code) for n in gics_registry.industry_groups(version=info)}
        block = industry_group_exposure(portfolio.holdings, current, names)
        block["status"] = "ok"
        block["taxonomy_version"] = info.version_key
        block["mapping_caveat"] = industry_labels.PUBLIC_MAPPING_CAVEAT
        snapshot = industry_snapshot.latest_snapshot(version=info)
        codes = [g["code"] for g in block["by_group"]]
        if snapshot is None:
            block["snapshot"] = {"status": "no_snapshot", "groups": []}
        else:
            block["snapshot"] = {
                "status": "ok",
                "period_key": snapshot.get("period_key"),
                "as_of": snapshot.get("as_of"),
                "macro_regime": ((snapshot.get("payload") or {}).get("regime") or {}).get("macro_regime"),
                "groups": industry_snapshot.group_rows(snapshot, codes),
            }
        block["note"] = (
            "weights are the portfolio's own; group statistics are observed weekly data "
            "and regime labels are rule-based reads — scenario context, not advice"
        )
        # The industry surfaces' projection (with the rollup): this block is
        # built from the same snapshot rows the industry routes serve.
        public: dict[str, Any] = industry_labels.project_public(block, rollup=True)
        return public
    except Exception as exc:
        log_safely(log, "industry exposure block failed (portfolio still returned)", exc, level=logging.DEBUG)
        return {"status": "unavailable", "reason": type(exc).__name__, "by_group": [], "unmapped": []}


@router.post("/api/portfolio/build", response_model=ModelPortfolioWithExposure)
def build(
    request: Request,
    req: PortfolioRequest,
    _rate: None = Depends(rate_scope("llm_light")),
    _grant: Grant = Depends(require_feature("portfolio", resource_param=None)),
) -> ModelPortfolioWithExposure:
    """FEAT-002: Pro only (the middleware refuses Free before this runs);
    the feature adds a two-in-flight lease per user and the `llm_light`
    rate scope because the brief extraction is an LLM call.

    FEAT-003: the response carries `industry_exposure` — weight by
    industry group (our own labels and slugs) from the stored
    classifications plus the exposed groups' rows of the latest
    cross-industry snapshot (reads only)."""
    principal = current_principal(request)
    with llm_call_context(user_id=principal.user_id, feature="portfolio"):
        portfolio = build_model_portfolio(req)
    return ModelPortfolioWithExposure(
        **portfolio.model_dump(), industry_exposure=industry_exposure_block(portfolio),
    )
