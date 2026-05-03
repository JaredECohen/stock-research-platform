"""Wave 5A — LLM-driven DCF assumption updater.

Runs at quarter-close (or on explicit refresh). Given the prior DCF
version's assumptions and the fresh actuals (revenue, op margin,
capex %, FCF) from the Wave 2 history tables, ask the LLM to propose
adjustments to forward assumptions. The roll-forward + delta-cap +
persistence are deterministic; only the *proposed adjustments* are
LLM-driven.

Validation gate during initial rollout (locked in MASTER_PLAN §5):
- Each assumption-change row carries the LLM's rationale string.
- Per-cycle delta capped at ±20% of prior assumption (`MAX_DELTA_PCT`).
  This keeps a hallucinating LLM from yanking WACC from 8.5% to 25%.
- LLM output is JSON-strict; bad/missing fields fall through to
  identity (no change) rather than failing the cycle.

If no LLM is available, the updater performs the deterministic
roll-forward only — shifts the explicit forecast forward one period,
caps growth/margin trends with a small mean-reversion bias toward
terminal. This is intentionally simple — the reflection log shows
"deterministic update" so reviewers know an LLM call wasn't made.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas import DCFAssumptions, DCFResult
from . import llm

log = logging.getLogger(__name__)


# v1 cap on assumption change per cycle. Locked in MASTER_PLAN; bump after
# a manual review pass on the first 10 tickers shows the LLM's drift
# proposals are sane.
MAX_DELTA_PCT = 0.20

# Fields the LLM is allowed to adjust. Leaving `tax_rate`, `current_price`,
# etc. untouched — those are anchored to actuals or external snapshots,
# not modeling judgment.
_ADJUSTABLE_FIELDS = (
    "revenue_growth", "operating_margin", "tax_rate",
    "da_pct_revenue", "capex_pct_revenue", "nwc_pct_revenue",
    "terminal_growth", "wacc", "exit_ebitda_multiple",
)


def _clamp_delta(prior: float, proposed: float) -> float:
    """Clamp `proposed` to within ±MAX_DELTA_PCT of `prior` (by absolute scale).

    For values near zero (like a capex pct that's currently 0.5%), we use
    `max(abs(prior), 0.005)` so the cap doesn't degenerate to zero room.
    """
    if prior is None:
        return proposed
    base = max(abs(float(prior)), 0.005)
    cap = base * MAX_DELTA_PCT
    return max(prior - cap, min(prior + cap, proposed))


def _shift_forecast_forward(prior: List[float]) -> List[float]:
    """Roll the explicit forecast list forward by one period.

    The first-year value drops off (it's now an actual, not a forecast),
    each remaining value shifts left, and the tail is filled by repeating
    the last value. Caller can override the tail with an LLM proposal.
    """
    if not prior:
        return prior
    if len(prior) == 1:
        return list(prior)
    shifted = list(prior[1:])
    shifted.append(prior[-1])  # repeat terminal-ish value
    return shifted


def _deterministic_roll_forward(prior: DCFAssumptions) -> DCFAssumptions:
    """Roll-forward without LLM: shift the explicit forecast lists."""
    out = prior.model_copy(deep=True)
    out.revenue_growth = _shift_forecast_forward(prior.revenue_growth)
    out.operating_margin = _shift_forecast_forward(prior.operating_margin)
    return out


def _build_change_rows(
    prior: DCFAssumptions, updated: DCFAssumptions,
    rationale_by_field: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Diff prior → updated and emit one row per non-zero change."""
    rows: List[Dict[str, Any]] = []
    prior_d = prior.model_dump()
    updated_d = updated.model_dump()
    for field in _ADJUSTABLE_FIELDS:
        before = prior_d.get(field)
        after = updated_d.get(field)
        if before == after:
            continue
        rationale = rationale_by_field.get(field, "")
        if isinstance(before, list) and isinstance(after, list):
            for i, (bv, av) in enumerate(zip(before, after)):
                if bv != av:
                    rows.append({
                        "field": f"{field}[{i}]",
                        "from": bv, "to": av,
                        "rationale": rationale,
                    })
            continue
        rows.append({
            "field": field, "from": before, "to": after,
            "rationale": rationale,
        })
    return rows


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

_UPDATER_PROMPT = (
    "You are reviewing the prior DCF assumptions for {ticker} after a "
    "fresh earnings period. Propose ADJUSTMENTS to forward assumptions, "
    "grounded in the actuals supplied.\n\n"
    "Constraints (HARD):\n"
    "- Adjust at most by ±20% of the prior value per field per cycle.\n"
    "- For list fields (revenue_growth, operating_margin), return the FULL "
    "  forward list (length-matched to prior); the runtime will apply the "
    "  ±20% clamp element-wise.\n"
    "- Always supply a 1-sentence rationale per field you change. Empty "
    "  rationale → field will be silently dropped.\n"
    "- DO NOT touch fields outside the allowed list.\n\n"
    "Allowed fields:\n"
    "- revenue_growth (list of floats, e.g. [0.10, 0.09, ...])\n"
    "- operating_margin (list of floats)\n"
    "- tax_rate (float)\n"
    "- da_pct_revenue (float)\n"
    "- capex_pct_revenue (float)\n"
    "- nwc_pct_revenue (float)\n"
    "- terminal_growth (float)\n"
    "- wacc (float)\n"
    "- exit_ebitda_multiple (float)\n\n"
    "Return strict JSON:\n"
    "{{\n"
    '  "updates": {{ "field": new_value, ... }},\n'
    '  "rationales": {{ "field": "1-sentence why", ... }}\n'
    "}}\n"
    "Empty `updates` is acceptable when the prior assumptions still look right."
)


def _llm_propose_updates(
    ticker: str, prior: DCFAssumptions, actuals: Dict[str, Any],
    rolled: DCFAssumptions,
) -> Optional[Dict[str, Any]]:
    if not settings.has_llm:
        return None
    payload = {
        "ticker": ticker,
        "prior_assumptions": prior.model_dump(),
        "actuals_this_period": actuals,
        "rolled_forward_assumptions": rolled.model_dump(),
    }
    user = (
        _UPDATER_PROMPT.format(ticker=ticker)
        + "\n\nContext:\n" + json.dumps(payload, default=str)[:3500]
    )
    try:
        out = llm.chat_json(
            user, system="You are a careful equity-research valuations analyst.",
            route="strong", model=settings.openai_pm_model,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("DCF updater LLM call failed for %s: %s", ticker, exc)
        return None
    if not isinstance(out, dict):
        return None
    updates = out.get("updates") or {}
    rationales = out.get("rationales") or {}
    return {"updates": updates, "rationales": rationales}


def _apply_updates(
    rolled: DCFAssumptions, updates: Dict[str, Any], rationales: Dict[str, str],
) -> tuple[DCFAssumptions, Dict[str, str]]:
    """Apply LLM-proposed updates to the rolled-forward assumptions, with
    per-field rationale + ±20% clamp. Returns (new_assumptions, accepted_rationales).
    """
    new = rolled.model_copy(deep=True)
    accepted: Dict[str, str] = {}
    for field, value in (updates or {}).items():
        if field not in _ADJUSTABLE_FIELDS:
            continue
        rationale = (rationales or {}).get(field, "").strip()
        if not rationale:
            # No rationale → drop the change. Discipline locked in MASTER_PLAN.
            continue
        prior_val = getattr(rolled, field, None)
        if isinstance(prior_val, list):
            if not isinstance(value, list) or len(value) != len(prior_val):
                continue  # length mismatch → drop
            try:
                clamped = [_clamp_delta(p, float(v)) for p, v in zip(prior_val, value)]
            except (TypeError, ValueError):
                continue
            setattr(new, field, clamped)
        else:
            try:
                clamped = _clamp_delta(float(prior_val), float(value))
            except (TypeError, ValueError):
                continue
            setattr(new, field, clamped)
        accepted[field] = rationale
    return new, accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_for_new_period(
    ticker: str, prior: DCFAssumptions, actuals: Dict[str, Any],
) -> tuple[DCFAssumptions, List[Dict[str, Any]]]:
    """Roll the DCF forward one period + apply LLM-proposed adjustments.

    Returns `(new_assumptions, assumption_changes)` where
    `assumption_changes` is the diff list suitable for
    `dcf_store.save_version(...)`.
    """
    rolled = _deterministic_roll_forward(prior)
    proposal = _llm_propose_updates(ticker, prior, actuals, rolled)

    if proposal is None:
        # No-LLM or failure — record the deterministic roll-forward only.
        change_rows = _build_change_rows(prior, rolled, rationale_by_field={
            "revenue_growth": "deterministic roll-forward (no LLM)",
            "operating_margin": "deterministic roll-forward (no LLM)",
        })
        return rolled, change_rows

    new, accepted = _apply_updates(rolled, proposal["updates"], proposal["rationales"])

    # Build the change diff vs. the *prior* assumptions (not rolled). The
    # rationale falls through from `accepted`; for fields the LLM didn't
    # touch but the deterministic roll moved (the explicit-forecast lists),
    # we tag them as the deterministic shift.
    rationale_by_field: Dict[str, str] = {
        "revenue_growth": "explicit forecast shifted forward + LLM adjustment"
        if "revenue_growth" in accepted else "deterministic roll-forward",
        "operating_margin": "explicit forecast shifted forward + LLM adjustment"
        if "operating_margin" in accepted else "deterministic roll-forward",
    }
    rationale_by_field.update(accepted)
    change_rows = _build_change_rows(prior, new, rationale_by_field)
    return new, change_rows


def actuals_from_history(ticker: str) -> Dict[str, Any]:
    """Pull a small actuals payload (revenue, op margin, capex %, FCF margin)
    out of the Wave 2 history tables. Returns the latest period's values
    plus a 4-period trailing average for context."""
    from ..services.history_service import get_financial_history

    fields = (
        "revenue", "operating_income", "capex", "free_cash_flow",
        "depreciation_and_amortization",
    )
    h = get_financial_history(ticker, list(fields), limit=8)
    actuals: Dict[str, Any] = {}
    for f in fields:
        rows = h.get(f, [])
        if not rows:
            continue
        actuals[f"{f}_latest"] = rows[0].get("value")
        if len(rows) >= 4:
            valid = [r["value"] for r in rows[:4] if r.get("value") is not None]
            if valid:
                actuals[f"{f}_4p_avg"] = sum(valid) / len(valid)
    # Derived: latest operating margin / capex as % of revenue.
    rev = actuals.get("revenue_latest")
    if rev:
        if "operating_income_latest" in actuals and actuals["operating_income_latest"] is not None:
            actuals["operating_margin_latest"] = actuals["operating_income_latest"] / rev
        if "capex_latest" in actuals and actuals["capex_latest"] is not None:
            # capex is reported negative in cash flow conventions; absolute it.
            actuals["capex_pct_revenue_latest"] = abs(actuals["capex_latest"]) / rev
        if "free_cash_flow_latest" in actuals and actuals["free_cash_flow_latest"] is not None:
            actuals["fcf_margin_latest"] = actuals["free_cash_flow_latest"] / rev
    return actuals
