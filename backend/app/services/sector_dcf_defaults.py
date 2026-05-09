"""Wave 10i — sector- and company-specific DCF assumption overrides.

The deterministic `derive_default_assumptions` builder produces a
generic baseline (15.0x exit EBITDA, 2.5% terminal growth, capped
WACC). That's reasonable for a generic mid-cap industrial; it's
materially wrong for a software compounder, a regulated utility, a
commodity producer, or a money-center bank.

This module runs a single LLM call per DCF that takes:
- the company profile (sector, industry, business description, key
  drivers / risks),
- the deterministic baseline assumptions,
- the cycle position (peak / normal / trough),

and emits *sector-appropriate* overrides for the six knobs that
matter most for valuation:

  - exit_ebitda_multiple
  - terminal_growth
  - wacc_adjustment (added on top of CAPM-derived WACC)
  - capex_pct_revenue
  - da_pct_revenue
  - nwc_pct_revenue

Each override is bounded (clamped) so the LLM can't propose anything
absurd, and each carries a one-sentence `rationale` so the audit
trail is preserved on the memo.

LLM-optional: when the API key is missing, returns the baseline
unchanged. The DCF still works; it just uses generic defaults.

The output is *additive* — `derive_default_assumptions` produces
the baseline, this layers sector overrides on top, and the existing
Wave 5A `dcf_pm_adjuster` (which the PM uses to tune assumptions
post-fan-out) still has the final say.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from ..config import settings
from ..schemas import DCFAssumptions

log = logging.getLogger(__name__)


# Bounds — keeps LLM proposals defensible.
_BOUNDS = {
    "exit_ebitda_multiple": (5.0, 30.0),
    "terminal_growth": (0.005, 0.045),
    "wacc_adjustment": (-0.025, 0.025),  # ±250 bps off the deterministic WACC
    "capex_pct_revenue": (0.005, 0.18),
    "da_pct_revenue": (0.005, 0.18),
    "nwc_pct_revenue": (-0.05, 0.10),
}


def _clamp(value: float, key: str) -> float:
    lo, hi = _BOUNDS[key]
    return max(lo, min(hi, value))


def _build_payload(
    profile: Dict[str, Any],
    baseline: DCFAssumptions,
    cycle_position: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ticker": profile.get("ticker"),
        "company_name": profile.get("company_name"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "business_description": (
            profile.get("business_description") or ""
        )[:1500],
        "drivers": (profile.get("drivers") or [])[:5],
        "risks": (profile.get("risks") or [])[:5],
        "cycle_position": cycle_position or "unknown",
        "baseline_assumptions": {
            "wacc": round(baseline.wacc, 4),
            "terminal_growth": round(baseline.terminal_growth, 4),
            "exit_ebitda_multiple": round(baseline.exit_ebitda_multiple, 2),
            "capex_pct_revenue": round(baseline.capex_pct_revenue, 4),
            "da_pct_revenue": round(baseline.da_pct_revenue, 4),
            "nwc_pct_revenue": round(baseline.nwc_pct_revenue, 4),
            "operating_margin_y1": (
                round(baseline.operating_margin[0], 4)
                if baseline.operating_margin else None
            ),
        },
        "bounds": _BOUNDS,
    }


_PROMPT = """You are a senior buy-side valuation analyst tuning a DCF
model's terminal-value + working-capital assumptions for a SPECIFIC
company. The deterministic builder produced generic defaults; your
job is to override them where the company's sector / business model
warrants different terminal economics.

Anchors:
- **High-multiple compounders** (premium software, payment networks,
  luxury brands): exit_ebitda_multiple 18-25x, terminal_growth 3-4%,
  capex 2-5% of revenue, low NWC.
- **Mid-multiple growth** (mid-cap software, healthcare services,
  consumer brands): exit 13-18x, terminal_growth 2.5-3.5%.
- **Mature compounders** (consumer staples, healthcare distribution,
  business services): exit 11-14x, terminal_growth 2-2.5%, low capex,
  modest NWC.
- **Cyclicals** (energy, materials, industrials, autos): exit 6-9x,
  terminal_growth 1.5-2%, high capex 8-15%, NWC swings with cycle.
  WACC adjustment: +50-100 bps for cyclical risk premium.
- **Capital-intensive utilities / telcos**: exit 8-11x, terminal_growth
  2-2.5%, capex 12-18%, high D&A. WACC adjustment: -50-100 bps for
  rate-base regulation predictability (or +50 for exposure to rate
  cases).
- **Banks / insurers**: DCF is a poor model for these. If forced, use
  exit 9-12x and terminal_growth 2.5%; flag in rationale that ROE
  / book value frameworks are more appropriate.
- **Pharma / biotech**: highly bimodal. For diversified pharma exit
  10-13x; for single-asset biotech, baseline is meaningless and you
  should flag it. WACC adjustment: +100-200 bps for trial risk.
- **Auto / transportation**: cyclical-margin reversion + capex 6-10%.
  Exit 7-10x.
- **REITs**: terminal_growth 2.5%, exit 12-16x (FFO basis), low capex.

Adjustments are bounded (see `bounds` object). Stay within them.

For each knob you change, write a one-sentence `rationale` citing
the sector / business model. If the baseline is already correct for
this company, leave the field unchanged (return the baseline value)
with rationale "baseline appropriate".

Return strict JSON:
{
  "exit_ebitda_multiple": <float>,
  "terminal_growth": <float>,
  "wacc_adjustment": <float, additive bps as decimal>,
  "capex_pct_revenue": <float>,
  "da_pct_revenue": <float>,
  "nwc_pct_revenue": <float>,
  "rationales": {
    "exit_ebitda_multiple": "<one sentence>",
    "terminal_growth": "<one sentence>",
    "wacc_adjustment": "<one sentence>",
    "capex_pct_revenue": "<one sentence>",
    "da_pct_revenue": "<one sentence>",
    "nwc_pct_revenue": "<one sentence>"
  },
  "framework_warning": "<empty string OR a sentence flagging when
    DCF is the wrong tool for this name (banks, single-asset
    biotech, etc.)>"
}
"""


def apply_sector_overrides(
    profile: Dict[str, Any],
    baseline: DCFAssumptions,
    *,
    cycle_position: Optional[str] = None,
) -> DCFAssumptions:
    """Wave 10i — layer LLM-judged sector overrides on top of the
    deterministic baseline. Returns the baseline unchanged on any
    failure (no API key, malformed response, network error)."""
    if not getattr(settings, "openai_api_key", None):
        return baseline
    if not (profile.get("sector") or profile.get("business_description")):
        return baseline
    payload = _build_payload(profile, baseline, cycle_position)
    try:
        from ..agents import llm
        out = llm.chat_json(
            _PROMPT
            + "\n\nCompany context:\n"
            + json.dumps(payload, default=str)[:6000],
            system="You are a senior valuation analyst. Be specific.",
            route="cheap",
            model=getattr(settings, "openai_tool_model", None),
            max_tokens=900,
        )
    except Exception as exc:  # pragma: no cover — never block DCF
        log.warning("sector DCF override LLM call failed: %s", exc)
        return baseline
    if not isinstance(out, dict):
        return baseline

    # Build a clamped override dict; fall back to baseline on any
    # missing / unparseable field.
    rationales = out.get("rationales") or {}
    rationales = rationales if isinstance(rationales, dict) else {}
    framework_warning = str(out.get("framework_warning") or "").strip()

    def _override(key: str, baseline_value: float, applies_clamp: bool = True) -> float:
        v = out.get(key)
        if not isinstance(v, (int, float)):
            return baseline_value
        return _clamp(float(v), key) if applies_clamp else float(v)

    new_exit = _override("exit_ebitda_multiple", baseline.exit_ebitda_multiple)
    new_tg = _override("terminal_growth", baseline.terminal_growth)
    new_capex = _override("capex_pct_revenue", baseline.capex_pct_revenue)
    new_da = _override("da_pct_revenue", baseline.da_pct_revenue)
    new_nwc = _override("nwc_pct_revenue", baseline.nwc_pct_revenue)
    wacc_adj = 0.0
    if isinstance(out.get("wacc_adjustment"), (int, float)):
        wacc_adj = _clamp(float(out["wacc_adjustment"]), "wacc_adjustment")
    new_wacc = max(0.04, min(0.18, baseline.wacc + wacc_adj))

    log.info(
        "sector DCF override for %s (%s): "
        "exit %.1fx → %.1fx, tg %.2f%% → %.2f%%, "
        "capex %.2f%% → %.2f%%, wacc %.2f%% → %.2f%%. %s",
        profile.get("ticker"), profile.get("sector"),
        baseline.exit_ebitda_multiple, new_exit,
        baseline.terminal_growth * 100, new_tg * 100,
        baseline.capex_pct_revenue * 100, new_capex * 100,
        baseline.wacc * 100, new_wacc * 100,
        framework_warning or "",
    )
    return baseline.model_copy(update={
        "exit_ebitda_multiple": new_exit,
        "terminal_growth": new_tg,
        "capex_pct_revenue": new_capex,
        "da_pct_revenue": new_da,
        "nwc_pct_revenue": new_nwc,
        "wacc": new_wacc,
    })
