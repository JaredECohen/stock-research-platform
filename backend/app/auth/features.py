"""The entitlement matrix — one table the backend enforces and the
pricing page renders, so the copy and the enforcement cannot drift.

Allowance values per plan:
  - an int      → metered, that many per UTC calendar month
  - None        → allowed, unlimited (still metered if `metered`)
  - True/False  → allowed / not allowed, no meter
  - "follows_memo" → Free may use it only for a ticker already counted as
                  a `memo_view` this month (DCF and comps ride on a memo)

Numbers come from DEVPLAN FEAT-002. `ENTITLEMENT_OVERRIDES_JSON` can
change any of them per environment without a deploy:
    {"pm_chat": {"free": 5, "pro": 500}, "memo_view": {"free": 5}}

`memo_view` counts DISTINCT tickers per month, not raw views (opening the
same memo twice is one), which is what "3 full stored-memo views" means
in practice — the pricing page states it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..config import settings
from .sanitize import safe_logger

log = safe_logger(__name__)

Allowance = int | bool | None | str  # see module docstring
FOLLOWS_MEMO = "follows_memo"


@dataclass(frozen=True)
class Shape:
    """Request-shape ceiling for a chart feature (FEAT-001): how many
    companies × metrics one request may draw and how many fiscal years
    back it may reach. `max_years=None` is the full stored history. Kept
    on the feature rather than in the route so the pricing page renders
    the same numbers the route clamps to."""
    max_companies: int
    max_metrics: int
    max_years: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "max_companies": self.max_companies,
            "max_metrics": self.max_metrics,
            "max_years": self.max_years,
        }


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    free: Allowance
    pro: Allowance
    # Metered features reserve a `usage_events` row per use.
    metered: bool = False
    # Money leaves the building when this runs (LLM / provider calls), so
    # an unverified email may not use it.
    cost_bearing: bool = False
    # Concurrency lease per user (`active_actions`), None = no limit.
    max_concurrent: int | None = None
    # Charge once per (user, resource, month) rather than per call.
    distinct_resources: bool = False
    period: str = "month"
    # Per-plan request shape for chart features; None = no shape limit.
    shape_free: Shape | None = None
    shape_pro: Shape | None = None


FEATURES: dict[str, Feature] = {
    f.name: f for f in (
        Feature(
            "memo_view", "Open a stored investment memo",
            free=3, pro=None, metered=True, distinct_resources=True,
        ),
        Feature(
            "research_run", "Run the full agent committee on a ticker",
            free=1, pro=20, metered=True, cost_bearing=True,
        ),
        Feature(
            "pm_chat", "Ask-the-PM chat turn (also macro analysis and the NL screener)",
            free=10, pro=300, metered=True, cost_bearing=True, max_concurrent=2,
        ),
        # FEAT-001. Commentary is one cheap-route LLM call per request:
        # metered per month, at most two in flight per user (DEVPLAN
        # "lightweight LLM actions"), released when the call yields
        # nothing so a degraded answer is free.
        Feature(
            "chart_commentary", "AI commentary on a fundamentals chart",
            free=5, pro=100, metered=True, cost_bearing=True, max_concurrent=2,
        ),
        # The explorer itself is DB reads: allowed on every plan, and the
        # plan only shapes the request (DEVPLAN: Free 2 companies × 2
        # metrics × 5 years; Pro 5 × 4 × the full stored history). With
        # the login wall off the anonymous visitor gets the Pro shape.
        Feature(
            "fundamentals_explorer", "Fundamentals explorer (historical statements and ratios)",
            free=True, pro=True,
            shape_free=Shape(max_companies=2, max_metrics=2, max_years=5),
            shape_pro=Shape(max_companies=5, max_metrics=4, max_years=None),
        ),
        Feature(
            "dcf", "DCF model on a ticker",
            free=FOLLOWS_MEMO, pro=True, cost_bearing=True,
        ),
        Feature(
            "comps", "Comparable-company table",
            free=FOLLOWS_MEMO, pro=True, cost_bearing=True,
        ),
        Feature(
            "portfolio", "Model portfolio builder",
            free=False, pro=True, cost_bearing=True, max_concurrent=2,
        ),
        Feature(
            "macro", "Macro series and scenario analysis",
            free=False, pro=True,
        ),
        Feature(
            "track_record", "Track record and outcome evaluation",
            free=False, pro=True,
        ),
        Feature(
            "memo_history", "Memo version history, agent memory and DCF versions",
            free=False, pro=True,
        ),
        Feature(
            "data_catalog", "Data catalog and sector overlays",
            free=False, pro=True,
        ),
        # Phase 6. DB reads only (the worker computes); Pro because the
        # universe table and the export are the product's quant surface.
        Feature(
            "scorecard", "Fundamental factor scorecard (universe, ticker detail, evaluation, export)",
            free=False, pro=True,
        ),
    )
}


def get(name: str) -> Feature:
    try:
        return FEATURES[name]
    except KeyError:
        raise ValueError(f"unknown feature {name!r}") from None


@lru_cache(maxsize=8)
def _parse_overrides(raw: str) -> dict[str, dict[str, Any]]:
    """Parse ENTITLEMENT_OVERRIDES_JSON once per distinct value. Bad JSON
    is logged and ignored so a typo in the dashboard cannot take the app
    down — the code defaults apply instead."""
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        log.error("ENTITLEMENT_OVERRIDES_JSON is not valid JSON; using code defaults")
        return {}
    if not isinstance(data, dict):
        log.error("ENTITLEMENT_OVERRIDES_JSON must be an object; using code defaults")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, plans in data.items():
        if name not in FEATURES or not isinstance(plans, dict):
            log.warning("ENTITLEMENT_OVERRIDES_JSON: ignoring unknown feature %r", name)
            continue
        out[name] = {k: v for k, v in plans.items() if k in ("free", "pro")}
    return out


def raw_allowance(name: str, plan: str) -> Allowance:
    feat = get(name)
    override = _parse_overrides(settings.entitlement_overrides_json).get(name, {})
    if plan in override:
        return override[plan]
    if plan == "pro":
        return feat.pro
    return feat.free


@dataclass(frozen=True)
class Resolved:
    allowed: bool
    limit: int | None        # None = unlimited (only meaningful when allowed)
    follows_memo: bool = False
    metered: bool = False


def allowance(name: str, plan: str) -> Resolved:
    """Turn a raw allowance into (allowed, limit, follows_memo)."""
    feat = get(name)
    value = raw_allowance(name, plan)
    if value == FOLLOWS_MEMO:
        return Resolved(allowed=True, limit=None, follows_memo=True, metered=False)
    if value is None:
        return Resolved(allowed=True, limit=None, metered=feat.metered)
    if isinstance(value, bool):
        return Resolved(allowed=value, limit=None, metered=False)
    if isinstance(value, int):
        return Resolved(allowed=value > 0, limit=int(value), metered=feat.metered)
    # A string other than "follows_memo" — an override typo. Fail closed.
    log.error("feature %s has an unrecognised allowance %r for plan %s; denying", name, value, plan)
    return Resolved(allowed=False, limit=0)


def shape(name: str, plan: str) -> Shape:
    """The request shape `plan` may ask of feature `name`.

    Free is the only plan with the smaller shape; every other plan value
    — `pro`, and the `unrestricted` plan `authorize()` reports while the
    login wall is off — gets the Pro shape, which is the owner decision
    for anonymous visitors. Raises `ValueError` for a feature that has
    no shape: a route asking for one is a wiring bug, not a request to
    wave through.
    """
    feat = get(name)
    if feat.shape_free is None or feat.shape_pro is None:
        raise ValueError(f"feature {name!r} has no request shape")
    return feat.shape_free if plan == "free" else feat.shape_pro


def registry_for_config() -> dict[str, dict[str, Any]]:
    """The matrix as the pricing page needs it (JSON-safe)."""
    out: dict[str, dict[str, Any]] = {}
    for name, feat in FEATURES.items():
        entry: dict[str, Any] = {
            "description": feat.description,
            "free": raw_allowance(name, "free"),
            "pro": raw_allowance(name, "pro"),
            "metered": feat.metered,
            "period": feat.period,
            "distinct_resources": feat.distinct_resources,
        }
        if feat.shape_free is not None and feat.shape_pro is not None:
            entry["shape"] = {"free": feat.shape_free.as_dict(), "pro": feat.shape_pro.as_dict()}
        out[name] = entry
    return out
