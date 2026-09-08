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
        Feature(
            "chart_commentary", "AI commentary on a chart (reserved for FEAT-001)",
            free=5, pro=100, metered=True, cost_bearing=True,
        ),
        Feature(
            "fundamentals_explorer", "Fundamentals explorer (reserved for FEAT-001)",
            free=True, pro=True,
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


def registry_for_config() -> dict[str, dict[str, Any]]:
    """The matrix as the pricing page needs it (JSON-safe)."""
    out: dict[str, dict[str, Any]] = {}
    for name, feat in FEATURES.items():
        out[name] = {
            "description": feat.description,
            "free": raw_allowance(name, "free"),
            "pro": raw_allowance(name, "pro"),
            "metered": feat.metered,
            "period": feat.period,
            "distinct_resources": feat.distinct_resources,
        }
    return out
