"""Wave 10k — narrative-driven bull / bear DCF scenarios.

Replaces the old symmetric ±400bps mechanical bumps in
`finance/dcf.py::_bull_assumptions` / `_bear_assumptions`. Now bull
and bear assumptions are *tied to specific named drivers* the LLM
identifies for THIS company:

- AI revenue ramp drives the bull's growth + margin together.
- China consumer slowdown drives the bear's growth alone.
- Reg / antitrust hits the bear's WACC + margin.

The LLM picks 2-3 drivers per side, names them, and proposes
assumption changes per driver. The driver list flows back through
`DCFScenario.drivers` and `_bull_case.key_points` so the prose tile
on the memo cites the same drivers the DCF baked in.

Deterministic fallback (no API key) is sector-aware — software
gets smaller bumps than cyclicals, banks get a "DCF is wrong tool"
flag — so the no-LLM path is materially better than the prior
one-size-fits-all approach.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..schemas import DCFAssumptions, ScenarioDriver

log = logging.getLogger(__name__)


# Bounds — keep LLM bumps defensible.
_BUMP_BOUNDS = {
    "revenue_growth_bp": (-1500, 1500),       # ±15 percentage points / year
    "operating_margin_bp": (-1000, 1000),     # ±10 percentage points
    "terminal_growth_bp": (-200, 200),        # ±2 pp
    "wacc_bp": (-200, 300),                   # asymmetric — wacc CAN move +300 bps for bear
}


def _clamp_bp(value: int, key: str) -> int:
    lo, hi = _BUMP_BOUNDS[key]
    return max(lo, min(hi, int(value)))


# Sector-aware deterministic fallback bumps. Used when no LLM is
# available. Tuned per the Wave 10i sector primer.
_SECTOR_FALLBACK_BUMPS: Dict[str, Dict[str, Dict[str, float]]] = {
    # Software / compounders → smaller bumps (margins are stickier).
    "software": {
        "bull": {"growth_bp": +300, "margin_bp": +150, "tg_bp": +50, "wacc_bp": -50},
        "bear": {"growth_bp": -300, "margin_bp": -200, "tg_bp": -50, "wacc_bp": +75},
    },
    # Cyclicals → wider bumps. Cycles drive bigger swings.
    "cyclical": {
        "bull": {"growth_bp": +600, "margin_bp": +400, "tg_bp": +50, "wacc_bp": -75},
        "bear": {"growth_bp": -600, "margin_bp": -500, "tg_bp": -75, "wacc_bp": +150},
    },
    # Mature compounders (staples, distribution) → smaller still.
    "mature": {
        "bull": {"growth_bp": +200, "margin_bp": +100, "tg_bp": +25, "wacc_bp": -25},
        "bear": {"growth_bp": -250, "margin_bp": -150, "tg_bp": -25, "wacc_bp": +50},
    },
    # Banks / insurers — flag in rationale; bump conservatively.
    "financials": {
        "bull": {"growth_bp": +200, "margin_bp": +100, "tg_bp": +25, "wacc_bp": -50},
        "bear": {"growth_bp": -300, "margin_bp": -250, "tg_bp": -25, "wacc_bp": +100},
    },
    # Generic fallback — close to the old symmetric bumps.
    "generic": {
        "bull": {"growth_bp": +400, "margin_bp": +200, "tg_bp": +50, "wacc_bp": -50},
        "bear": {"growth_bp": -400, "margin_bp": -300, "tg_bp": -50, "wacc_bp": +100},
    },
}


def _classify_for_fallback(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    if "tech" in s or "software" in i or "semiconductor" in i:
        return "software"
    if any(k in s for k in ("energy", "materials", "industrials", "consumer discretionary", "real estate")):
        return "cyclical"
    if "financ" in s or "bank" in i or "insurance" in i:
        return "financials"
    if any(k in s for k in ("staples", "health care", "utilities")) or "distribution" in i:
        return "mature"
    return "generic"


def _apply_bumps(
    base: DCFAssumptions,
    growth_bp: int,
    margin_bp: int,
    tg_bp: int,
    wacc_bp: int,
) -> DCFAssumptions:
    """Apply assumption-change bumps to a base, returning a new
    DCFAssumptions. Caps + floors prevent absurd values."""
    a = copy.deepcopy(base)
    delta_growth = growth_bp / 10_000.0
    delta_margin = margin_bp / 10_000.0
    delta_tg = tg_bp / 10_000.0
    delta_wacc = wacc_bp / 10_000.0
    a.revenue_growth = [
        max(-0.30, min(0.50, g + delta_growth)) for g in a.revenue_growth
    ]
    a.operating_margin = [
        max(0.01, min(0.70, m + delta_margin)) for m in a.operating_margin
    ]
    a.terminal_growth = max(0.005, min(0.045, a.terminal_growth + delta_tg))
    a.wacc = max(0.04, min(0.18, a.wacc + delta_wacc))
    # Wave 10j — exit_ebitda_multiple is no longer used by the headline
    # DCF (Gordon-only). Don't bump it; it's dead code in the
    # implied-price path.
    return a


def _deterministic_fallback(
    profile: Dict[str, Any], base: DCFAssumptions,
) -> Tuple[DCFAssumptions, List[ScenarioDriver], DCFAssumptions, List[ScenarioDriver]]:
    """Sector-aware deterministic builder. No LLM cost. Better than
    the prior one-size-fits-all symmetric bumps."""
    bucket = _classify_for_fallback(
        profile.get("sector") or "", profile.get("industry") or "",
    )
    bumps = _SECTOR_FALLBACK_BUMPS.get(bucket, _SECTOR_FALLBACK_BUMPS["generic"])
    bull = _apply_bumps(
        base,
        bumps["bull"]["growth_bp"], bumps["bull"]["margin_bp"],
        bumps["bull"]["tg_bp"], bumps["bull"]["wacc_bp"],
    )
    bear = _apply_bumps(
        base,
        bumps["bear"]["growth_bp"], bumps["bear"]["margin_bp"],
        bumps["bear"]["tg_bp"], bumps["bear"]["wacc_bp"],
    )
    bull_drivers = [ScenarioDriver(
        name=f"{bucket.title()} sector — upside scenario",
        rationale=(
            f"Deterministic {bucket} fallback: growth +{bumps['bull']['growth_bp']}bp, "
            f"margin +{bumps['bull']['margin_bp']}bp, terminal growth "
            f"+{bumps['bull']['tg_bp']}bp, WACC {bumps['bull']['wacc_bp']:+d}bp."
        ),
        assumption_changes=["revenue_growth", "operating_margin", "terminal_growth", "wacc"],
    )]
    bear_drivers = [ScenarioDriver(
        name=f"{bucket.title()} sector — downside scenario",
        rationale=(
            f"Deterministic {bucket} fallback: growth {bumps['bear']['growth_bp']:+d}bp, "
            f"margin {bumps['bear']['margin_bp']:+d}bp, terminal growth "
            f"{bumps['bear']['tg_bp']:+d}bp, WACC {bumps['bear']['wacc_bp']:+d}bp."
        ),
        assumption_changes=["revenue_growth", "operating_margin", "terminal_growth", "wacc"],
    )]
    return bull, bull_drivers, bear, bear_drivers


_PROMPT = """You are a buy-side PM constructing bull and bear DCF
scenarios for a SPECIFIC company. Each scenario must be tied to 2-3
NAMED DRIVERS — not symmetric mechanical bumps.

For each driver:
- name: short label ("AI revenue ramps", "China consumer slowdown")
- rationale: ONE sentence explaining why this driver belongs in
  this scenario, citing the company's actual exposures
- assumption_changes: list which fields this driver moves
  (revenue_growth, operating_margin, terminal_growth, wacc)

Then propose the AGGREGATE bumps to apply across all drivers, in
basis points (positive or negative). Bumps are bounded — see
`bump_bounds` in the context. Use realistic magnitudes:
- Software / mature compounders: smaller bumps (margins sticky).
- Cyclicals: larger bumps (cycles drive wider swings).
- Banks / insurers: DCF is a poor model — flag this in rationale
  and keep bumps conservative; output a `framework_warning`.
- High-growth (>15% baseline): bear should compress growth more,
  bull less (already at the high end).

Output EXACT JSON:
{
  "bull": {
    "drivers": [
      {"name": "...", "rationale": "...", "assumption_changes": ["..."]}
    ],
    "growth_bp": <int>,
    "margin_bp": <int>,
    "terminal_growth_bp": <int>,
    "wacc_bp": <int>
  },
  "bear": {
    "drivers": [...],
    "growth_bp": <int>,
    "margin_bp": <int>,
    "terminal_growth_bp": <int>,
    "wacc_bp": <int>
  },
  "framework_warning": "<empty string OR a sentence flagging when
    DCF is the wrong tool>"
}
"""


def build_bull_bear(
    profile: Dict[str, Any], base: DCFAssumptions,
) -> Tuple[DCFAssumptions, List[ScenarioDriver], DCFAssumptions, List[ScenarioDriver]]:
    """Build bull and bear assumption sets + driver lists.

    LLM-driven when an OpenAI key is configured; sector-aware
    deterministic fallback otherwise.
    """
    if not getattr(settings, "openai_api_key", None):
        return _deterministic_fallback(profile, base)

    payload = {
        "ticker": profile.get("ticker"),
        "company_name": profile.get("company_name"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "business_description": (
            profile.get("business_description") or ""
        )[:1500],
        "drivers": (profile.get("drivers") or [])[:5],
        "risks": (profile.get("risks") or [])[:5],
        "baseline_assumptions": {
            "wacc": round(base.wacc, 4),
            "terminal_growth": round(base.terminal_growth, 4),
            "y1_growth": round(base.revenue_growth[0], 4) if base.revenue_growth else None,
            "y5_growth": round(base.revenue_growth[-1], 4) if base.revenue_growth else None,
            "y1_op_margin": round(base.operating_margin[0], 4) if base.operating_margin else None,
        },
        "bump_bounds": _BUMP_BOUNDS,
    }
    try:
        from ..agents import llm
        out = llm.chat_json(
            _PROMPT
            + "\n\nCompany context:\n"
            + json.dumps(payload, default=str)[:6000],
            system="You are a senior buy-side PM. Be specific and concrete.",
            route="cheap",
            model=getattr(settings, "openai_tool_model", None),
            max_tokens=1200,
        )
    except Exception as exc:  # pragma: no cover — never block DCF
        log.warning("scenario assumptions LLM call failed: %s", exc)
        return _deterministic_fallback(profile, base)

    if not isinstance(out, dict):
        return _deterministic_fallback(profile, base)

    def _parse_side(side: Dict[str, Any]) -> Tuple[DCFAssumptions, List[ScenarioDriver]]:
        if not isinstance(side, dict):
            return base, []
        growth_bp = _clamp_bp(side.get("growth_bp", 0), "revenue_growth_bp")
        margin_bp = _clamp_bp(side.get("margin_bp", 0), "operating_margin_bp")
        tg_bp = _clamp_bp(side.get("terminal_growth_bp", 0), "terminal_growth_bp")
        wacc_bp = _clamp_bp(side.get("wacc_bp", 0), "wacc_bp")
        bumped = _apply_bumps(base, growth_bp, margin_bp, tg_bp, wacc_bp)
        drivers: List[ScenarioDriver] = []
        for d in (side.get("drivers") or [])[:3]:
            if not isinstance(d, dict):
                continue
            drivers.append(ScenarioDriver(
                name=str(d.get("name") or "")[:120],
                rationale=str(d.get("rationale") or "")[:300],
                assumption_changes=[
                    str(c) for c in (d.get("assumption_changes") or [])
                    if isinstance(c, str)
                ][:6],
            ))
        return bumped, drivers

    bull_assumptions, bull_drivers = _parse_side(out.get("bull") or {})
    bear_assumptions, bear_drivers = _parse_side(out.get("bear") or {})

    framework_warning = str(out.get("framework_warning") or "").strip()
    if framework_warning:
        # Prepend a warning driver on the bear side so it surfaces in
        # the memo's bear case key points.
        bear_drivers.insert(0, ScenarioDriver(
            name="Framework warning",
            rationale=framework_warning,
            assumption_changes=[],
        ))

    log.info(
        "scenario assumptions for %s: bull=%s bear=%s",
        profile.get("ticker"),
        [d.name for d in bull_drivers],
        [d.name for d in bear_drivers],
    )
    return bull_assumptions, bull_drivers, bear_assumptions, bear_drivers
