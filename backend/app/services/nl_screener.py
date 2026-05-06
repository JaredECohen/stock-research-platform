"""Wave 10 — natural-language screener.

User types: "show me profitable US-listed semis with falling capex
intensity, beta under 1.5, and meaningful AI exposure."

LLM converts that into a structured `CustomScreenRequest` over the
existing `screener_metrics` table, optionally augmented with theme
exposure filters that didn't exist on the rule-based screener
before. The translation is shown back to the user in the UI so they
can audit / refine before re-running.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..schemas import (
    CustomScreenRequest,
    CustomScreenResult,
    ScreenerMetricName,
    ScreenerOp,
    ScreenerRule,
)

log = logging.getLogger(__name__)

# Subset of metrics + ops the LLM is allowed to emit. Keep the
# vocabulary tight so the rule chain hits the existing custom-screen
# evaluator without surprises.
_ALLOWED_METRICS: List[str] = [
    "pe_ttm", "ev_ebitda", "ev_revenue", "gross_margin",
    "op_margin", "fcf_margin", "roic", "roe", "debt_to_ebitda",
    "revenue_growth_yoy", "market_cap", "beta", "fcf_yield",
]
_ALLOWED_OPS: List[str] = [">", "<", ">=", "<=", "=", "between"]
_SUPPORTED_THEMES = [
    "ai_infrastructure", "ai_applications", "energy_transition", "glp1",
    "china_consumer", "data_center_buildout", "cybersecurity",
    "weight_loss", "long_rates_sensitivity", "consumer_credit",
]


def _llm_translate(query: str) -> Optional[Dict[str, Any]]:
    """Returns a dict with: rules (list), themes (list), sectors
    (list), sort_by, order, rationale. Returns None on any failure."""
    if not getattr(settings, "openai_api_key", None):
        return None
    schema = {
        "rules": "list of {metric, op, value, value2?} — metrics/ops below",
        "themes": "list of theme tags from the supported list",
        "sectors": "list of sector names (Technology, Healthcare, ...)",
        "sort_by": "metric to sort by — same vocabulary as rules.metric",
        "order": "asc or desc",
        "rationale": "1-2 sentence explanation of how the prompt was read",
    }
    prompt = (
        "Convert this natural-language stock screen into a structured "
        "rule chain over the screener metrics table. The user's intent "
        "matters MORE than literal keyword matches — interpret what "
        "they actually want.\n\n"
        f"Allowed metrics (rules.metric): {', '.join(_ALLOWED_METRICS)}\n"
        f"Allowed ops (rules.op): {', '.join(_ALLOWED_OPS)}\n"
        f"Supported themes (themes): {', '.join(_SUPPORTED_THEMES)}\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"User query:\n{query}\n\n"
        "Translation hints:\n"
        "- 'profitable' → op_margin > 0\n"
        "- 'high quality' → roic > 15 AND op_margin > 20\n"
        "- 'cheap' → pe_ttm < 20 OR ev_ebitda < 12\n"
        "- 'high growth' → revenue_growth_yoy > 15\n"
        "- 'low debt' → debt_to_ebitda < 2\n"
        "- 'low beta' or 'defensive' → beta < 1\n"
        "- 'AI exposure' or 'AI play' → themes: ai_infrastructure, ai_applications\n"
        "- 'energy transition' or 'renewables' → themes: energy_transition\n"
        "- Keep value units consistent with the metric (margins as %, "
        "PE as a ratio, market_cap in billions).\n\n"
        "Return strict JSON. Empty arrays where nothing applies."
    )
    from ..agents import llm
    out = llm.chat_json(
        prompt,
        system="You are a buy-side screener. Convert prose to rules accurately.",
        route="cheap",
    )
    return out if isinstance(out, dict) else None


def translate(query: str) -> Tuple[CustomScreenRequest, List[str], str]:
    """Returns (request, themes, rationale).

    Themes are returned alongside because the rule-based evaluator
    doesn't know about the theme_exposure table; the caller layers
    theme filtering on top.
    """
    out = _llm_translate(query) or {}
    rules: List[ScreenerRule] = []
    for raw in (out.get("rules") or []):
        if not isinstance(raw, dict):
            continue
        metric = str(raw.get("metric") or "").strip()
        op = str(raw.get("op") or "").strip()
        if metric not in _ALLOWED_METRICS or op not in _ALLOWED_OPS:
            continue
        try:
            value = float(raw.get("value") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        value2: Optional[float] = None
        if op == "between":
            try:
                value2 = float(raw["value2"])
            except (TypeError, ValueError, KeyError):
                continue
        rules.append(ScreenerRule(
            metric=metric, op=op, value=value, value2=value2,  # type: ignore[arg-type]
        ))

    themes = [t for t in (out.get("themes") or []) if t in _SUPPORTED_THEMES]
    sectors = [str(s) for s in (out.get("sectors") or [])] or None
    sort_by_raw = out.get("sort_by")
    sort_by: ScreenerMetricName = (
        sort_by_raw if sort_by_raw in _ALLOWED_METRICS else "market_cap"
    )  # type: ignore[assignment]
    order = "asc" if (out.get("order") == "asc") else "desc"
    rationale = str(out.get("rationale") or "")

    req = CustomScreenRequest(
        rules=rules, sectors=sectors,
        sort_by=sort_by, order=order, limit=50,
    )
    return req, themes, rationale


def run(query: str) -> Dict[str, Any]:
    """Translate + execute. Returns a payload the chat agent or the
    frontend can render: the inferred request (so the user sees what
    we read), the rationale, and the matching rows.
    """
    req, themes, rationale = translate(query)
    rows: List[Dict[str, Any]] = []
    matched = 0
    try:
        from ..api.routes_screener import _execute_custom_screen
        result: CustomScreenResult = _execute_custom_screen(req)
        rows = [r.model_dump() for r in result.rows]
        matched = result.matched
    except Exception as exc:  # pragma: no cover
        log.warning("nl_screener custom-screen execution failed: %s", exc)
    # Theme overlay: when themes are supplied, intersect rows with the
    # union of theme-exposed tickers.
    if themes:
        try:
            from .theme_exposure_service import top_for_theme
            exposed: set[str] = set()
            for theme in themes:
                for r in top_for_theme(theme, min_score=20.0, limit=80):
                    exposed.add(r["ticker"])
            rows = [r for r in rows if r.get("ticker", "").upper() in exposed]
            matched = len(rows)
        except Exception as exc:  # pragma: no cover
            log.warning("nl_screener theme overlay failed: %s", exc)
    return {
        "query": query,
        "request": req.model_dump(),
        "themes": themes,
        "rationale": rationale,
        "matched": matched,
        "rows": rows[:50],
    }
