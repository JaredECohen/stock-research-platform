"""Wave 10 — PM-driven DCF assumption adjuster.

After the specialist fan-out (and the Wave 9 dialog rounds) lands, the PM
has the team's full read: sector growth view, earnings tone, filing risks,
comps relative valuation, macro regime. THAT is the moment to update the
DCF assumptions so the final memo's valuation reflects the team's research,
not just the consensus / default-preserve baseline.

Architecture mirrors `dcf_updater.py` (the quarter-close roll-forward),
but the input is different: there it's prior-DCF + new actuals; here it's
default-DCF + the specialists' synthesized findings. The ±20% per-cycle
clamp + rationale-required discipline is reused.

Skip conditions:
- No LLM available (returns initial DCF unchanged).
- `as_of_date` is set (backtest path — don't re-anchor history).
- Initial DCF is None (no fundamentals to anchor).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..schemas import AgentFinding, DCFAssumptions, DCFResult
from . import llm
from .dcf_updater import _apply_updates  # share the clamp + rationale gate

log = logging.getLogger(__name__)


_PM_DCF_SYSTEM = (
    "You are MarketMosaic's senior PM finalizing the valuation step. The "
    "specialist team has filed their reads. Your job: propose targeted "
    "adjustments to the DCF assumptions so the model reflects the team's "
    "research, NOT just consensus defaults."
)


_PM_DCF_PROMPT = (
    "DCF starting point (consensus-anchored / default-preserved):\n"
    "{prior_assumptions}\n\n"
    "Specialist findings (one block per agent — read all before adjusting):\n"
    "{findings_block}\n\n"
    "Discipline:\n"
    "- ONLY adjust an assumption if the team's research warrants it. "
    "If the specialists triangulate to consensus, leave it alone. "
    "Empty `updates` is the correct answer often.\n"
    "- ±20% max change per cycle per field (runtime clamps; you can "
    "  request more, it'll be capped).\n"
    "- For list fields (revenue_growth, operating_margin) return the "
    "  FULL forward list, length-matched to prior.\n"
    "- Every field you change MUST have a 1-sentence rationale citing "
    "  WHICH specialist's finding drove the change (e.g., 'sector agent: "
    "  cohort growth is decelerating; trim '27 from 9% to 7.5%').\n"
    "- Bias: if the team is more bullish than consensus, the model "
    "  should reflect that (e.g., trim margin compression assumption, "
    "  raise growth path). If more bearish, the opposite.\n"
    "- DO NOT touch tax_rate, current_price, base_revenue, net_debt, "
    "  diluted_shares — those are anchored to actuals.\n\n"
    "Allowed fields:\n"
    "- revenue_growth (list of floats, length-matched)\n"
    "- operating_margin (list of floats, length-matched)\n"
    "- da_pct_revenue (float)\n"
    "- capex_pct_revenue (float)\n"
    "- nwc_pct_revenue (float)\n"
    "- terminal_growth (float)\n"
    "- wacc (float)\n"
    "- exit_ebitda_multiple (float)\n\n"
    "Return strict JSON:\n"
    "{{\n"
    '  "updates": {{ "field": new_value, ... }},\n'
    '  "rationales": {{ "field": "1-sentence why, citing which agent", ... }},\n'
    '  "headline": "1-sentence summary of how the model changed (or '
    '\\"no changes — team triangulates to consensus\\")"\n'
    "}}\n"
)


def _format_finding_for_pm(name: str, f: AgentFinding) -> str:
    """Compact rendering of a finding for the PM's adjustment prompt.

    Mirrors `deep_research._format_finding_for_critique` — same shape so
    a reviewer reading the LLMCallLog can compare round 0 critique vs. the
    DCF-adjustment call against the same finding format.
    """
    bullets = "\n".join(f"  - {p}" for p in (f.key_points or [])[:5])
    return (
        f"### {name}\n"
        f"  Headline: {f.headline}\n"
        f"  Summary: {f.summary[:500]}\n"
        f"{bullets}"
    )


def _propose_adjustments(
    *, ticker: str, prior: DCFAssumptions,
    findings: Dict[str, AgentFinding], run_id: str,
) -> Optional[Dict[str, Any]]:
    """Single LLM call that returns proposed updates + rationales + headline.

    Returns None on LLM failure — caller leaves the DCF untouched.
    """
    findings_block = "\n\n".join(
        _format_finding_for_pm(k, v) for k, v in findings.items()
    )
    prompt = _PM_DCF_PROMPT.format(
        prior_assumptions=json.dumps(prior.model_dump(), default=str, indent=2),
        findings_block=findings_block[:6000],
    )
    try:
        from .llm import llm_call_context
        # Use whatever provider is active. Forcing OpenAI hard-failed
        # on deployments configured with only an Anthropic key (100%
        # failure rate observed in prod LLMCallLog before this fix).
        with llm_call_context(
            agent_name="PM DCF Adjuster", run_id=run_id, route="strong",
        ):
            out = llm.chat_json(
                prompt, system=_PM_DCF_SYSTEM, route="strong",
            )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("PM DCF adjuster LLM call failed for %s: %s", ticker, exc)
        return None
    if not isinstance(out, dict):
        return None
    return out


def adjust_dcf_for_pm_view(
    *, ticker: str, initial_dcf: DCFResult,
    findings: Dict[str, AgentFinding], run_id: str,
) -> Tuple[Optional[DCFResult], List[Dict[str, Any]], str]:
    """Re-build the DCF using PM-adjusted assumptions.

    Returns `(adjusted_dcf_or_None, adjustments_audit, headline)`.
    Only the BASE-case assumptions are adjusted — bull / bear are kept as
    PM upside / downside scenarios anchored to the same adjusted base.
    The full 3-scenario DCF is rebuilt via `valuation_service.build_dcf`.

    On any failure path (no LLM, malformed proposal, build failure) the
    function returns `(None, [], "")` and the caller falls through to the
    initial DCF. We never return a half-rebuilt result.
    """
    if not settings.has_llm or initial_dcf is None:
        return None, [], ""

    prior = initial_dcf.base.assumptions
    proposal = _propose_adjustments(
        ticker=ticker, prior=prior, findings=findings, run_id=run_id,
    )
    if proposal is None:
        return None, [], ""

    updates = proposal.get("updates") or {}
    rationales = proposal.get("rationales") or {}
    headline = str(proposal.get("headline") or "").strip()

    # Reuse the dcf_updater clamp + rationale-required gate. `_apply_updates`
    # returns (new_assumptions, accepted_rationales) — fields without a
    # rationale are silently dropped.
    new_assumptions, accepted = _apply_updates(prior, updates, rationales)

    if not accepted:
        # PM proposed nothing actionable — leave the initial DCF alone.
        return None, [], headline or "no changes — team triangulates to consensus"

    # Rebuild the 3-scenario DCF with the adjusted base assumptions.
    # Bull / bear get derived in `build_full_dcf` via the same scenario
    # builder used on the default path, so the upside / downside framing
    # remains consistent — only the base anchor moves.
    try:
        from ..services import valuation_service
        adjusted = valuation_service.build_dcf(
            ticker, assumptions=new_assumptions, force_refresh=False,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("PM-adjusted DCF rebuild failed for %s: %s", ticker, exc)
        return None, [], ""

    if adjusted is None:
        return None, [], ""

    # Audit trail: one row per changed field, with PM's rationale + the
    # before/after values. The frontend uses this to show "Why did the
    # PM change this assumption?" in the DCF Lab.
    audit: List[Dict[str, Any]] = []
    prior_d = prior.model_dump()
    new_d = new_assumptions.model_dump()
    for field, rationale in accepted.items():
        before = prior_d.get(field)
        after = new_d.get(field)
        if isinstance(before, list) and isinstance(after, list):
            for i, (bv, av) in enumerate(zip(before, after)):
                if bv != av:
                    audit.append({
                        "field": f"{field}[{i}]",
                        "from": bv, "to": av,
                        "rationale": rationale,
                    })
        elif before != after:
            audit.append({
                "field": field, "from": before, "to": after,
                "rationale": rationale,
            })

    return adjusted, audit, headline
