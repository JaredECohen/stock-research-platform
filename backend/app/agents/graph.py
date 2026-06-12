"""Agent graph (LangGraph-style structure, hand-rolled).

The graph orchestrates a single stock memo generation:

  classify_intent
        │
        ▼
  fan-out specialists
   ├─ sector_agent
   ├─ earnings_agent
   ├─ filing_agent
   ├─ valuation_agent (uses DCF)
   ├─ comps_agent
   ├─ macro_agent
   └─ risk_agent
        │
        ▼
  draft_memo
        │
        ▼
  critic_agent  (Risk Committee)
        │
        ▼
  pm_synthesis (final view, rating, confidence)

We don't pull in a full LangGraph dependency to keep the container slim; the
shape and naming match the LangGraph mental model and could be swapped in
trivially.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

from ..config import settings
from ..schemas import (
    AgentFinding,
    AgentTrace,
    BullBearCase,
    CatalystItem,
    CompsResult,
    CriticReview,
    DCFResult,
    RiskItem,
    StockMemoOut,
)  # CriticReview imported for the safe-runner fallback path  # noqa: F401
from ..services.fundamentals_service import get_full_financials
from ..services.market_data_service import get_basic_stats
from ..services.transcripts_service import latest_transcript
from ..services.filings_service import get_filings
from ..services.valuation_service import build_comps, build_dcf
from . import llm, prompts
from .comps_agent import run_comps_agent
from .critic_agent import run_critic
from .earnings_agent import run_earnings_agent
from .filing_agent import run_filing_agent
from .macro_agent import run_macro_agent
from .risk_agent import derive_risk_items, run_risk_agent
from .safe_runner import (
    DegradationLog,
    safe_call,
    safe_critic,
    safe_finding,
)
from .sector_agents import run_sector_agent
from .technical_agent import run_technical_agent
from .tools import evidence_quality
from .valuation_agent import run_valuation_agent


# ---------------------------------------------------------------------------
# Memo construction
# ---------------------------------------------------------------------------

_THESIS_ANTI_PATTERN = re.compile(
    r"""^                       # start
        (?P<co>[^—:]{2,80})     # company / ticker prefix
        \s+[—–-]\s+             # em / en / hyphen dash
        (?P<sector>[A-Za-z][\w &/]{2,40})
        \s*/\s*                 # sector / industry slash
        .{2,200}                # industry + regime + any inline clauses
        DCF\ base\ case          # the giveaway phrase
        """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)


def _looks_like_anti_pattern_thesis(text: str) -> bool:
    """True when `text` matches the explicit anti-pattern the PM prompt
    forbids ("{Company} — {Sector} / {industry}, {hook}; DCF base case
    +X% suggests material upside.").

    We use this to detect when the LLM ignored the prompt OR the
    deterministic fallback regressed to the templated form, so we can
    rewrite the sentence before it ships into the memo.
    """
    if not text:
        return False
    return bool(_THESIS_ANTI_PATTERN.match(text.strip()))


def _verdict_word(rating: Optional[str], upside: Optional[float]) -> str:
    """Map the memo's headline rating to the thesis verdict word so the
    one-liner can never contradict the rating badge the reader sees.

    DCF and the multiples/comps read routinely disagree on premium
    compounders (COST: DCF cheap, multiple rich). The prior logic took the
    verdict straight off the DCF base upside in isolation, so the thesis
    could say "undervalued" while the rating badge and the whole valuation
    section said overvalued. Anchoring on the rating fixes that.

    Falls back to the DCF upside sign ONLY when no rating is available —
    the LLM-disabled deterministic path, before the factor blend runs.
    """
    label = (rating or "").strip().lower()
    if "bull" in label:
        return "undervalued"
    if "bear" in label:
        return "overvalued"
    if label:  # explicit "Neutral" (or any other named rating)
        return "fairly priced"
    if upside is None or abs(upside) < 0.10:
        return "fairly priced"
    return "undervalued" if upside > 0 else "overvalued"


def _mispricing_lever_clause(
    verdict_word: str,
    upside: Optional[float],
    drivers: List[str],
    risks: List[str],
) -> str:
    """Sentence-2 lever clause for a mispriced name.

    Cites the DCF number only when its sign agrees with the verdict (else
    it contradicts the call). Names a real driver / risk when we have one,
    and degrades to a clean single clause — never the "core driver
    execution vs. the dominant risk" placeholder — when we don't.
    """
    driver = (drivers[0] if drivers else "").strip()
    risk = (risks[0] if risks else "").strip()
    dcf_agrees = upside is not None and (
        (verdict_word == "undervalued" and upside > 0)
        or (verdict_word == "overvalued" and upside < 0)
    )
    if dcf_agrees:
        sign = "+" if (upside or 0) > 0 else ""
        lead = f"DCF base case implies {sign}{(upside or 0) * 100:.0f}% to fair value"
    elif verdict_word == "overvalued":
        lead = "The multiple already prices in the bull case"
    else:  # undervalued, but the DCF doesn't corroborate the call
        lead = "The market is under-pricing the durable part of the franchise"

    if driver and risk:
        return f"{lead}; the swing factor is {driver} against {risk}."
    if driver:
        return f"{lead}; the swing factor is {driver}."
    if risk:
        return f"{lead}; the key risk is {risk}."
    return f"{lead}."


def _gap_clause_agrees(gap_clause: str, verdict_word: str) -> bool:
    """True when the consensus-gap clause points the same way as the
    verdict. `_market_gap_clause` phrases upside as "upside the market may
    be missing" and downside as "downside the market may be
    underweighting" — keep it from carrying sentence 2 in the direction
    that contradicts the verdict word.
    """
    low = gap_clause.lower()
    if verdict_word == "undervalued":
        return "upside" in low
    if verdict_word == "overvalued":
        return "downside" in low
    return True


def _build_thesis_from_findings(
    profile: Dict,
    findings: Dict[str, "AgentFinding"],
    dcf: Optional[DCFResult],
    ticker: str,
    rating: Optional[str] = None,
) -> str:
    """Compose a short-form thesis (2-3 sentences) from the specialists'
    findings. Mirrors the structure required by PM_SYNTHESIS_PROMPT so
    the deterministic fallback reads the same as the LLM happy path:

        Sentence 1 — VERDICT: ticker + correctly priced / over / under,
                     and the ONE thing that defines the call.
        Sentence 2 — LEVER / PERFORMANCE: if mispriced, the segment or
                     metric where the gap shows up; if correctly priced,
                     what makes it worth owning at current price.

    Used by both the post-LLM anti-pattern rewrite and the deterministic
    fallback in `_pm_synthesis` — keeps the logic in one place so the
    two paths can't drift.
    """
    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []
    ticker_sym = (profile.get("ticker") or ticker or "").upper()

    def _claim_from_finding(f: "Optional[AgentFinding]") -> Optional[str]:
        if f is None:
            return None
        head = (getattr(f, "headline", "") or "").strip().rstrip(".,;:")
        if not head or len(head) < 12:
            return None
        low = head.lower()
        # Reject cohort labels, scenario tags, intake-skip headlines, and
        # the very anti-pattern we're trying to escape — none of these
        # state a claim a reader can defend.
        skip_patterns = (
            "highlights", "view", "profile", "view for", "scenario:",
            "regime:", "cohort placement", "skipped per pm intake",
            "dcf base case", " — ",
        )
        if any(p in low for p in skip_patterns):
            return None
        return head

    sector_finding = findings.get("sector")
    bull_headline: Optional[str] = None
    if sector_finding is not None and isinstance(sector_finding.data, dict):
        bb = sector_finding.data.get("bull_bear_analysis") or {}
        if isinstance(bb, dict):
            bull_case = bb.get("bull_case") or {}
            if isinstance(bull_case, dict):
                bh = (bull_case.get("headline") or "").strip().rstrip(".,;:")
                if 12 <= len(bh) <= 180 and "dcf base case" not in bh.lower():
                    bull_headline = bh

    claim = (
        bull_headline
        or _claim_from_finding(findings.get("valuation"))
        or _claim_from_finding(sector_finding)
        or _claim_from_finding(findings.get("earnings"))
        or _claim_from_finding(findings.get("filing"))
        or (drivers[0] if drivers else None)
    )

    # Drop a leading scenario label ("Bull case:", "Bear case:", …). It
    # reads oddly once the verdict word precedes the claim and can flatly
    # contradict it (a bull-case headline behind an "overvalued" verdict).
    if claim:
        claim = re.sub(
            r"^\s*(bull|bear|base)[\s-]*case\s*[:\-—–]\s*", "", claim,
            flags=re.IGNORECASE,
        ).strip()

    upside = dcf.base.upside_pct if dcf and dcf.base else None

    # --- Sentence 1: VERDICT ---
    # Verdict follows the memo's headline rating, not the DCF base upside
    # in isolation — see `_verdict_word`.
    verdict_word = _verdict_word(rating, upside)

    if claim:
        sentence_1 = f"{ticker_sym} is {verdict_word} — {claim}." if ticker_sym else f"{verdict_word.capitalize()} — {claim}."
    elif verdict_word == "fairly priced":
        sector = (profile.get("sector") or "core").strip().lower()
        sentence_1 = (
            f"{ticker_sym} is fairly priced on our work — no actionable edge in {sector}."
            if ticker_sym
            else f"Fairly priced on our work — no actionable edge in {sector}."
        )
    else:
        # Mispriced per the blended read, but no specialist headline
        # carries the call — stay consistent with the verdict word.
        sentence_1 = (
            f"{ticker_sym} screens {verdict_word} on the blended read, "
            f"though no single specialist headline defines the call."
            if ticker_sym
            else f"Screens {verdict_word} on the blended read, "
            f"though no single specialist headline defines the call."
        )

    # --- Sentence 2: LEVER (if mispriced) or PERFORMANCE PATH (if not) ---
    sentence_2 = ""
    try:
        gap_clause = _market_gap_clause(profile, dcf, ticker_sym)
    except Exception:  # pragma: no cover
        gap_clause = ""

    if verdict_word in ("undervalued", "overvalued"):
        # Only let the consensus-gap clause carry sentence 2 when its
        # direction agrees with the verdict — otherwise it reintroduces
        # the very contradiction we're fixing.
        if gap_clause and _gap_clause_agrees(gap_clause, verdict_word):
            sentence_2 = gap_clause
        else:
            sentence_2 = _mispricing_lever_clause(verdict_word, upside, drivers, risks)
    else:
        # Correctly priced — describe the performance path. Degrades
        # cleanly with no named driver (no "compounding fundamentals
        # compounding" stutter, no template filler).
        if drivers:
            sentence_2 = (
                f"At this price the return comes from {drivers[0].lower()} "
                f"compounding at trend, not a re-rating — own the floor, not the multiple."
            )
        else:
            sentence_2 = (
                "At this price the return comes from steady compounding, "
                "not a re-rating — own the floor, not the multiple."
            )

    return f"{sentence_1} {sentence_2}".strip()


def _market_gap_clause(
    profile: Dict, dcf: Optional[DCFResult], ticker: str,
) -> str:
    """Wave 8R — write the "what the market is missing" sentence.

    Compares model growth vs. analyst consensus growth (5y avg). When
    the gap is material (≥2pp), names which side is leaning + why
    (driven by the company's first-line driver / risk).
    Returns the empty string when no meaningful gap exists.
    """
    if dcf is None:
        return ""
    try:
        model_growths = list(dcf.base.assumptions.revenue_growth)
        if not model_growths:
            return ""
        model_avg = sum(model_growths) / len(model_growths)
    except Exception:
        return ""

    # Pull consensus from the data service via the same helper the engine uses.
    consensus_avg = None
    try:
        from ..finance.dcf import _consensus_growth_path
        from ..services.data_service import get_data_service
        estimates = get_data_service().get_estimates(ticker)
        consensus = _consensus_growth_path(estimates)
        if consensus:
            consensus_avg = sum(consensus) / len(consensus)
    except Exception:
        consensus_avg = None

    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []

    if consensus_avg is None:
        # No consensus visible — make a softer "vs. trend" framing instead.
        return ""

    gap = model_avg - consensus_avg
    if abs(gap) < 0.02:  # within 2pp — not material
        return ""

    if gap > 0:
        # Model is more bullish than the Street.
        driver = drivers[0] if drivers else "core driver execution"
        return (
            f"Model sees ~{model_avg * 100:.0f}% revenue growth vs. consensus "
            f"~{consensus_avg * 100:.0f}% — the **upside the market may be "
            f"missing** is durability of {driver}."
        )
    # Model is more cautious than the Street.
    risk = risks[0] if risks else "execution slip on the dominant driver"
    return (
        f"Model sees ~{model_avg * 100:.0f}% revenue growth vs. consensus "
        f"~{consensus_avg * 100:.0f}% — the **downside the market may be "
        f"underweighting** is {risk}."
    )


def _build_scores_dict(
    *, blended_confidence: float, raw_confidence: float, ev_q: float,
    sector_finding: AgentFinding, valuation_finding: AgentFinding,
    risk_finding: AgentFinding, earnings_finding: Optional[AgentFinding] = None,
    profile: Dict, ratios: Dict, earnings: Dict,
) -> Dict[str, float]:
    """Wave 8M — assemble `memo.scores` so the UI can render every
    category score next to the headline confidence number.

    Three groups of fields ride here:
      - Headline + agent-confidence numbers (existing behavior).
      - `factor_*` — the same seven factor scores the screener uses,
        recomputed from the same ratios/profile inputs so memo and
        screener can't disagree on a name's quality / growth / valuation /
        momentum / risk / macro_fit / catalyst.
      - `factor_pm_score` — the screener's composite (0-100) for
        side-by-side comparison with the LLM-driven confidence.
    """
    from ..finance import factor_scores as fs
    rev_growth = ratios.get("revenue_growth")
    op_margin = ratios.get("operating_margin")
    gross_margin = ratios.get("gross_margin")
    roic = ratios.get("ROIC")
    ev_ebitda = ratios.get("EV_EBITDA")
    p_fcf = ratios.get("PFCF")
    fcf_y = ratios.get("FCF_yield")
    debt_to_ebitda = ratios.get("debt_to_ebitda")
    beta = profile.get("beta")

    quality = fs.quality_score(roic, op_margin, gross_margin)
    growth = fs.growth_score(rev_growth)
    valuation = fs.valuation_score(ev_ebitda, p_fcf, fcf_y)
    surprises = [q.get("surprise_pct", 0) for q in (earnings or {}).get("quarters", [])]
    # Pull the LLM-extracted latest guidance changes so beat-AND-raise
    # registers as a momentum bonus. Falls back to surprise-only when
    # the earnings analyst didn't emit structured output.
    latest_guidance: List[Dict[str, Any]] = []
    if earnings_finding is not None and isinstance(earnings_finding.data, dict):
        structured = earnings_finding.data.get("structured")
        if isinstance(structured, dict):
            raw_changes = structured.get("guidance_changes") or []
            if isinstance(raw_changes, list):
                latest_guidance = raw_changes
    earnings_momentum = fs.earnings_momentum_score(
        surprises, latest_guidance_changes=latest_guidance,
    )
    streak = fs.beat_streak(surprises)
    guidance_net = fs.guidance_net_direction(latest_guidance)
    risk = fs.risk_score(beta, debt_to_ebitda, drawdown=-0.20)

    macro_fit = 60.0  # untilled-theme baseline; theme bias only applies on the screener path
    catalyst = 65.0 if "AI" in (profile.get("description") or "") else 50.0

    pm_score = round(
        quality * 0.25 + growth * 0.20 + valuation * 0.15
        + earnings_momentum * 0.10 + macro_fit * 0.15
        + risk * 0.10 + catalyst * 0.05,
        1,
    )

    return {
        # Headline / agent confidences (existing).
        "confidence": blended_confidence,
        "raw_confidence": raw_confidence,
        "evidence_quality": round(ev_q * 100, 1),
        "sector_confidence": sector_finding.confidence * 100,
        "valuation_confidence": valuation_finding.confidence * 100,
        "risk_confidence": risk_finding.confidence * 100,
        # Wave 8M — quant factor scores (same math as the screener).
        "factor_quality": quality,
        "factor_growth": growth,
        "factor_valuation": valuation,
        "factor_earnings_momentum": earnings_momentum,
        "factor_macro_fit": macro_fit,
        "factor_risk": risk,
        "factor_catalyst": catalyst,
        "factor_pm_score": pm_score,
        # Beat-and-raise transparency. `beat_streak` = consecutive
        # recent EPS beats; `guidance_net` = (raised - lowered) from
        # the latest call's structured guidance changes. UI uses these
        # to render the "🔥 beat & raise" badge on the earnings card
        # when the combination triggers the momentum bonus.
        "beat_streak": float(streak),
        "guidance_net_direction": float(guidance_net),
    }


def _apply_risk_recommendations(
    memo: StockMemoOut, risk_finding: AgentFinding,
) -> List[Dict[str, Any]]:
    """Wave 8H — deterministic enforcement of risk-agent recommendations.

    The PM synthesis prompt is one channel for the risk lens to influence
    the memo (LLM reads the recommendations); this is the second, harder
    channel. For each rec we mutate the memo in place:

    - `target=confidence` direction=lower → clamp confidence_score down
      by 5/10/15 (small/medium/large).
    - `target=rating` direction=lower → shift rating one notch down the
      Bullish→Mixed Positive→Neutral→Mixed Negative→Bearish ladder.
    - `target=thesis_breakers` direction=flag → ensure a matching
      RiskItem is in `memo.thesis_breakers` (severity=high).
    - `target=bear_case` direction=flag → append `detail` to
      `memo.bear_case.key_points` if not already present (this is how
      risk findings actually augment the sector-built bear case).

    Returns the list of recs that were applied so the caller can add an
    audit trail to the memo.
    """
    if not isinstance(risk_finding.data, dict):
        return []
    raw = risk_finding.data.get("recommendations") or []
    if not isinstance(raw, list):
        return []
    applied: List[Dict[str, Any]] = []

    rating_ladder = [
        "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish",
    ]
    confidence_step = {"small": 5.0, "medium": 10.0, "large": 15.0}

    for rec in raw:
        if not isinstance(rec, dict):
            continue
        target = rec.get("target")
        direction = rec.get("direction")
        magnitude = rec.get("magnitude") or "medium"
        detail = (rec.get("detail") or "").strip()
        rationale = (rec.get("rationale") or "").strip()
        # Discipline — recs without a rationale don't apply.
        if not detail or not rationale:
            continue

        if target == "confidence" and direction == "lower":
            delta = confidence_step.get(magnitude, 10.0)
            old = float(memo.confidence_score or 0)
            new = max(20.0, old - delta)
            if new != old:
                memo.confidence_score = new
                applied.append({**rec, "applied_change": {
                    "field": "confidence_score", "from": old, "to": new,
                }})

        elif target == "rating" and direction == "lower":
            current = (memo.rating_label or "").strip()
            if current in rating_ladder:
                idx = rating_ladder.index(current)
                step = {"small": 1, "medium": 1, "large": 2}.get(magnitude, 1)
                new_idx = min(len(rating_ladder) - 1, idx + step)
                if new_idx != idx:
                    new_rating = rating_ladder[new_idx]
                    memo.rating_label = new_rating  # type: ignore[assignment]
                    applied.append({**rec, "applied_change": {
                        "field": "rating_label",
                        "from": current, "to": new_rating,
                    }})

        elif target == "thesis_breakers" and direction == "flag":
            already = any(
                detail.lower()[:60] in (item.title or "").lower()
                for item in memo.thesis_breakers
            )
            if not already:
                memo.thesis_breakers = list(memo.thesis_breakers) + [
                    RiskItem(
                        title=detail[:80],
                        detail=rationale,
                        severity="high",
                        type="thesis_breaker",
                    ),
                ]
                applied.append({**rec, "applied_change": {
                    "field": "thesis_breakers", "appended": detail[:80],
                }})

        elif target == "bear_case" and direction == "flag":
            existing = [p.lower() for p in memo.bear_case.key_points]
            if not any(detail.lower()[:60] in p for p in existing):
                memo.bear_case.key_points = list(memo.bear_case.key_points) + [
                    f"Risk lens: {detail}",
                ]
                applied.append({**rec, "applied_change": {
                    "field": "bear_case.key_points",
                    "appended": f"Risk lens: {detail}",
                }})

    return applied


def _bull_bear_from_sector(sector_finding: AgentFinding) -> Optional[Dict[str, Any]]:
    """Wave 3A: pluck the structured bull_bear_analysis out of the sector
    finding's data payload, if present. Returns the dict (not the Pydantic
    model) so the caller can pull out the raw `bull_case`/`bear_case`
    objects directly into the memo."""
    if not isinstance(sector_finding.data, dict):
        return None
    bb = sector_finding.data.get("bull_bear_analysis")
    return bb if isinstance(bb, dict) else None


def _findings_signal_lines(
    finding: Optional[AgentFinding], *, polarity: str,
    max_items: int = 3, prefix: str = "",
) -> List[str]:
    """Pull the most signal-bearing key_points / signals from an agent
    finding, scoped by polarity.

    Wave 9b — replaces the demo-only `profile.drivers` / `profile.risks`
    fallback in `_bull_case` / `_bear_case` / `_catalysts`. Live FMP
    profiles don't carry those fields; specialist findings do.

    `polarity ∈ {"bull", "bear", "neutral"}` filters lines by simple
    keyword detection (positive: tailwind / leverage / above median /
    accelerat; negative: pressure / headwind / risk / concentration /
    declin). `neutral` returns the first lines unfiltered.
    """
    if finding is None:
        return []
    bull_kw = (
        "tailwind", "leverage", "above median", "above-median",
        "accelerat", "expansion", "outperform", "premium quality",
        "moat", "advantage", "compounder", "growth", "top quartile",
    )
    bear_kw = (
        "pressur", "headwind", "risk", "concentrat", "declin",
        "deteriorat", "compress", "below median", "below-median",
        "elevated", "fragile", "slip", "underperform", "regulator",
        "antitrust", "litigation", "competit",
    )
    out: List[str] = []
    candidates = list(finding.key_points or [])
    # Also consider sentence fragments from the summary as a fallback
    # source — lots of value lives there for short-key_points findings.
    if finding.summary:
        for s in finding.summary.split(". "):
            s = s.strip()
            if 30 <= len(s) <= 220:
                candidates.append(s)
    for line in candidates:
        if not isinstance(line, str):
            continue
        low = line.lower()
        if polarity == "bull" and not any(k in low for k in bull_kw):
            continue
        if polarity == "bear" and not any(k in low for k in bear_kw):
            continue
        text = (prefix + line) if prefix and not line.lower().startswith(prefix.lower().rstrip(": ")) else line
        out.append(text[:240])
        if len(out) >= max_items:
            break
    return out


def _bull_case(profile: Dict, valuation: AgentFinding, dcf: Optional[DCFResult],
               sector_finding: Optional[AgentFinding] = None,
               findings: Optional[Dict[str, AgentFinding]] = None) -> BullBearCase:
    """Build the memo's bull case.

    Preference order:
      1. Sector analyst's structured `bull_bear_analysis` (LLM-generated).
      2. Bull-polarity signals lifted from sector / valuation / earnings
         findings + DCF upside.
      3. Generic but honest "DCF says X" line.
    """
    sector_bb = _bull_bear_from_sector(sector_finding) if sector_finding else None
    # Wave 10k — pull DCF scenario drivers (LLM-named) so the prose
    # tile cites the same drivers the assumption changes baked in.
    dcf_drivers: List[str] = []
    if dcf and dcf.bull and dcf.bull.drivers:
        dcf_drivers = [
            f"DCF driver — {d.name}: {d.rationale}".rstrip(": ")
            for d in dcf.bull.drivers if d.name or d.rationale
        ][:3]
    if sector_bb and isinstance(sector_bb.get("bull_case"), dict):
        bull = sector_bb["bull_case"]
        points = list(bull.get("key_points") or [])
        points.extend(dcf_drivers)
        if dcf:
            points.append(
                f"DCF bull case implies ${dcf.bull.implied_share_price:,.2f} "
                f"({dcf.bull.upside_pct:+.0%})."
            )
        return BullBearCase(
            headline=str(bull.get("headline") or "Bull case from sector synthesis."),
            key_points=points,
        )

    points: List[str] = []
    findings = findings or {}
    points.extend(_findings_signal_lines(findings.get("sector"), polarity="bull", max_items=3))
    points.extend(_findings_signal_lines(findings.get("valuation"), polarity="bull", max_items=2))
    points.extend(_findings_signal_lines(findings.get("earnings"), polarity="bull", max_items=2))
    # Wave 10k — DCF-named drivers from the scenario builder.
    points.extend(dcf_drivers)
    if dcf:
        points.append(
            f"DCF bull case implies ${dcf.bull.implied_share_price:,.2f} "
            f"({dcf.bull.upside_pct:+.0%})."
        )
    if not points:
        points.append("Quality + growth profile supports a premium versus peers.")

    # Pick a headline from the strongest available signal.
    sector_head = (sector_finding.headline if sector_finding else "") or ""
    if "above" in sector_head.lower() or "leader" in sector_head.lower():
        headline = sector_head[:240]
    elif valuation and valuation.headline and "discount" in valuation.headline.lower():
        headline = f"Bull case: {valuation.headline[:200]}"
    else:
        headline = "Bull case: cohort + valuation read both supportive."
    return BullBearCase(headline=headline, key_points=points[:6])


def _bear_case(profile: Dict, dcf: Optional[DCFResult],
               sector_finding: Optional[AgentFinding] = None,
               findings: Optional[Dict[str, AgentFinding]] = None) -> BullBearCase:
    """Build the memo's bear case. Mirror of `_bull_case` — prefers the
    sector LLM's bear, otherwise lifts bear-polarity signals from
    sector / risk / filing findings + DCF downside."""
    sector_bb = _bull_bear_from_sector(sector_finding) if sector_finding else None
    # Wave 10k — DCF bear-case drivers from the scenario builder.
    dcf_drivers: List[str] = []
    if dcf and dcf.bear and dcf.bear.drivers:
        dcf_drivers = [
            f"DCF driver — {d.name}: {d.rationale}".rstrip(": ")
            for d in dcf.bear.drivers if d.name or d.rationale
        ][:3]
    if sector_bb and isinstance(sector_bb.get("bear_case"), dict):
        bear = sector_bb["bear_case"]
        points = list(bear.get("key_points") or [])
        points.extend(dcf_drivers)
        if dcf:
            points.append(
                f"DCF bear case implies ${dcf.bear.implied_share_price:,.2f} "
                f"({dcf.bear.upside_pct:+.0%})."
            )
        return BullBearCase(
            headline=str(bear.get("headline") or "Bear case from sector synthesis."),
            key_points=points,
        )

    points: List[str] = []
    findings = findings or {}
    points.extend(_findings_signal_lines(findings.get("risk"), polarity="bear", max_items=3))
    points.extend(_findings_signal_lines(findings.get("filing"), polarity="bear", max_items=2, prefix="Filing: "))
    points.extend(_findings_signal_lines(findings.get("sector"), polarity="bear", max_items=2))
    # Wave 10k — DCF-named drivers from the scenario builder.
    points.extend(dcf_drivers)
    if dcf:
        points.append(
            f"DCF bear case implies ${dcf.bear.implied_share_price:,.2f} "
            f"({dcf.bear.upside_pct:+.0%})."
        )
    if not points:
        points.append("Cohort positioning leaves modest downside if execution slips.")

    risk_head = (findings.get("risk") and findings["risk"].headline) or ""
    if risk_head and not risk_head.lower().startswith("risk profile for"):
        headline = f"Bear case: {risk_head[:200]}"
    else:
        headline = "Bear case: execution / valuation / regulatory risks if thesis cracks."
    return BullBearCase(headline=headline, key_points=points[:6])


def _catalysts(
    profile: Dict, transcript: Optional[Dict],
    findings: Optional[Dict[str, AgentFinding]] = None,
    earnings: Optional[Dict] = None,
) -> List[CatalystItem]:
    """Surface near-term + medium-term catalysts.

    Wave 9b — derives catalysts from findings (earnings tone, sector
    drivers, news themes) instead of demo-only `profile.drivers`. Adds
    the next earnings date as a concrete near-term watch item when
    we have it (FMP earnings endpoint or AV).
    """
    items: List[CatalystItem] = []
    findings = findings or {}

    # Sector / news positive catalysts.
    sector_signals = _findings_signal_lines(findings.get("sector"), polarity="bull", max_items=2)
    for s in sector_signals:
        items.append(CatalystItem(
            title=s[:80], detail=s, horizon="medium_term", impact="medium",
        ))
    news_signals = _findings_signal_lines(findings.get("news_impact"), polarity="bull", max_items=1)
    for s in news_signals:
        items.append(CatalystItem(
            title=s[:80], detail=s, horizon="near_term", impact="medium",
        ))

    # Concrete next-earnings date when known.
    next_date = None
    if earnings and isinstance(earnings, dict):
        # FMP /stable/earnings returns forward rows with epsActual=null;
        # take the first one with a future-looking date.
        quarters = earnings.get("quarters") or []
        for q in quarters:
            if q.get("eps_actual") is None and q.get("report_date"):
                next_date = q["report_date"]
                break
    if next_date:
        items.append(CatalystItem(
            title=f"Next earnings: {next_date}",
            detail=f"Quarterly print expected on {next_date}; tone + guidance the swing factor.",
            horizon="near_term", impact="medium",
        ))
    elif transcript and transcript.get("period"):
        items.append(CatalystItem(
            title="Next earnings update",
            detail=f"Watch for follow-on commentary on themes from {transcript['period']}.",
            horizon="near_term", impact="medium",
        ))

    # Profile-driven catalysts (demo only — kept for fixture tests).
    if not items:
        for d in (profile.get("drivers") or [])[:3]:
            items.append(CatalystItem(
                title=d[:80], detail=d, horizon="medium_term", impact="medium",
            ))

    return items[:6]


def _pm_synthesis(profile: Dict, findings: Dict[str, AgentFinding], dcf: Optional[DCFResult]) -> Dict:
    # PM uses its dedicated model (OPENAI_PM_MODEL — gpt-5.5-pro by default).
    # Wave 10 — read PM brain + company / sector memory + research_notes.
    from .pm_context import build_pm_context
    pm_ctx = build_pm_context(
        ticker=profile.get("ticker"),
        sector=profile.get("sector"),
        profile=profile,
    )
    llm_out = llm.chat_json(
        prompts.PM_SYNTHESIS_PROMPT
        + ((("\n\n" + pm_ctx) if pm_ctx else ""))
        + "\n\nFindings:\n"
        + json.dumps({k: v.model_dump() for k, v in findings.items()}, default=str)[: settings.max_agent_context_chars],
        system=prompts.PM_SYSTEM, route="strong",
        model=settings.openai_pm_model,
    )
    if llm_out and "rating_label" in llm_out:
        return llm_out

    # Deterministic synthesis
    upside = dcf.base.upside_pct if dcf else 0.0
    pos_signals = sum(1 for f in findings.values() if any(k in (f.headline + f.summary).lower()
                                                          for k in ("constructive", "premium", "outperform", "tailwind")))
    neg_signals = sum(1 for f in findings.values() if any(k in (f.headline + f.summary).lower()
                                                          for k in ("pressured", "underperform", "elevated", "compress")))
    score = pos_signals - neg_signals + (1 if upside > 0.10 else (-1 if upside < -0.10 else 0))
    # Wave 8P — five-label scheme tied to the deterministic Stock-Score
    # mapping. The actual rating gets *overridden* later by
    # `rating_from_stock_score` once the factor blend is computed; this
    # local provides a sensible fallback for the LLM-disabled path.
    if score >= 2:
        rating = "Very Bullish"
    elif score == 1:
        rating = "Bullish"
    elif score == 0:
        rating = "Neutral"
    elif score == -1:
        rating = "Bearish"
    else:
        rating = "Very Bearish"
    confidence = max(40, min(85, 55 + 5 * abs(score)))

    # Build a thesis that distills the actual claim — not a metric
    # recap. Centralized in `_build_thesis_from_findings` so the
    # deterministic fallback and the post-LLM anti-pattern rewrite share
    # one source of truth.
    thesis = _build_thesis_from_findings(
        profile, findings, dcf, profile.get("ticker") or "", rating=rating,
    )
    pm_view = (
        f"Research view: {rating}. {thesis} "
        f"Sector framing supports the cohort thesis; valuation-relative read is the main swing factor. "
        f"The risk committee flagged the dominant downside scenarios; portfolio fit depends on macro view."
    )
    return dict(
        final_pm_view=pm_view,
        one_sentence_thesis=thesis,
        rating_label=rating,
        confidence_score=confidence,
    )


def _portfolio_fit(profile: Dict, rating: str) -> str:
    sector = profile.get("sector", "")
    return (
        f"In a balanced model portfolio, {profile.get('ticker', '')} fits the '{sector}' sleeve. "
        f"With a '{rating}' research view, sizing is governed by the user's max position size and risk level."
    )


# ---------------------------------------------------------------------------
# Wave 8A: per-step checkpointing
# ---------------------------------------------------------------------------
# Each specialist gets a thin checkpointed wrapper that the safe-runner calls.
# When `run_id` is in scope (always, since `run_stock_memo` sets it), the
# decorator caches each step's `AgentFinding` under `(run_id, step_name)` so
# a retried run with the same `run_id` skips the underlying work.
#
# Why thin wrappers vs. decorating each agent at definition: keeping the
# specialist functions un-decorated lets other callers (tests, ad-hoc
# scripts, future workers) use them without checkpoint side effects. The
# checkpoint behavior is deliberately scoped to the graph entry path.

from ..services.checkpoint_store import checkpointed


@checkpointed("graph.fundamentals", return_type=None)
def _checkpointed_fundamentals(ticker: str, *, force_refresh: bool):
    return get_full_financials(ticker, force_refresh=force_refresh)


@checkpointed("graph.dcf", return_type=DCFResult)
def _checkpointed_dcf(ticker: str, *, force_refresh: bool):
    return build_dcf(ticker, force_refresh=force_refresh)


@checkpointed("graph.comps", return_type=CompsResult)
def _checkpointed_comps(ticker: str, *, force_refresh: bool):
    return build_comps(ticker, force_refresh=force_refresh)


@checkpointed("graph.sector_finding", return_type=AgentFinding)
def _checkpointed_sector(profile: Dict, ratios: Dict) -> AgentFinding:
    return run_sector_agent(profile, ratios)


@checkpointed("graph.earnings_finding", return_type=AgentFinding)
def _checkpointed_earnings(profile, transcript, earnings) -> AgentFinding:
    return run_earnings_agent(profile, transcript, earnings)


@checkpointed("graph.filing_finding", return_type=AgentFinding)
def _checkpointed_filing(profile, filings) -> AgentFinding:
    return run_filing_agent(profile, filings)


@checkpointed("graph.valuation_finding", return_type=AgentFinding)
def _checkpointed_valuation(profile, ratios, dcf) -> AgentFinding:
    return run_valuation_agent(profile, ratios, dcf)


@checkpointed("graph.comps_finding", return_type=AgentFinding)
def _checkpointed_comps_agent(profile, comps) -> AgentFinding:
    return run_comps_agent(profile, comps)


@checkpointed("graph.macro_finding", return_type=AgentFinding)
def _checkpointed_macro(profile, scenario: str) -> AgentFinding:
    return run_macro_agent(profile, scenario)


@checkpointed("graph.risk_finding", return_type=AgentFinding)
def _checkpointed_risk(profile, ratios, dcf_summary) -> AgentFinding:
    return run_risk_agent(profile, ratios, dcf_summary)


@checkpointed("graph.technical_finding", return_type=AgentFinding)
def _checkpointed_technical(profile) -> AgentFinding:
    return run_technical_agent(profile)


@checkpointed("graph.critic", return_type=CriticReview)
def _checkpointed_critic(memo_dict: Dict) -> CriticReview:
    return run_critic(memo_dict)


# ---------------------------------------------------------------------------
# Public graph entry point
# ---------------------------------------------------------------------------

def _run_reflection_step(memo: StockMemoOut):
    """Local indirection so safe_call can wrap the reflection step. Imports
    lazily to avoid an import-time cycle (reflection_agent → memory → cache)."""
    from .reflection_agent import run as _reflect
    return _reflect(memo)


def run_stock_memo(
    ticker: str, *, scenario: str = "soft_landing", force_refresh: bool = False,
    run_id: Optional[str] = None,
    as_of_date: Optional[Any] = None,
) -> StockMemoOut:
    """Generate a stock memo. When `force_refresh=True`, every cached snapshot
    in the dependency tree is bypassed; otherwise, fundamentals/sector/comps/DCF
    are read from the snapshot cache when fresh.

    `run_id` (Wave 1A) tags every LLM call made during this memo run for
    cost / trace attribution via `LLMCallLog`. Auto-generated when None.

    `as_of_date` (Wave 1C) reproduces the memo as of a historical date.
    All cache reads/writes inside the call are namespaced by date so
    backtests don't collide with live data; long-term memory writes are
    skipped (a backtest shouldn't pollute the agent's notebook). Future
    PRs will thread per-provider date filtering through the data
    service so backtests truly see only past data.
    """
    import uuid
    from datetime import date as _date_cls, datetime as _dt_cls
    if run_id is None:
        run_id = str(uuid.uuid4())
    # Coerce datetime → date if a caller hands us a datetime.
    if isinstance(as_of_date, _dt_cls):
        as_of_date = as_of_date.date()
    if as_of_date is not None and as_of_date > _date_cls.today():
        raise ValueError(f"as_of_date {as_of_date} is in the future")

    from .llm import llm_call_context
    from ..services.data_service import as_of_context
    with as_of_context(as_of_date), llm_call_context(
        agent_name="run_stock_memo", run_id=run_id,
    ):
        return _run_stock_memo_inner(
            ticker, scenario=scenario, force_refresh=force_refresh,
            run_id=run_id, as_of_date=as_of_date,
        )


def _run_stock_memo_inner(
    ticker: str, *, scenario: str, force_refresh: bool, run_id: str,
    as_of_date: Optional[Any] = None,
) -> StockMemoOut:
    """Indirection so `run_stock_memo` can wrap the entire body in a single
    `llm_call_context` + `as_of_context`. Splitting keeps the public signature clean.

    Wave 8A: each major step (fundamentals, dcf, comps, every specialist,
    critic) runs through a `@checkpointed` wrapper. When `run_id` is reused
    across calls (e.g., a retry after a transient failure), each completed
    step's result is loaded from `MemoRunCheckpoint` instead of re-fired.
    First-time runs see no behavior change; the cache writes are cheap."""
    # Fundamentals MUST succeed — without a profile we can't even identify
    # the company, so this is an unrecoverable error and we re-raise.
    fin = _checkpointed_fundamentals(ticker, force_refresh=force_refresh)
    profile = fin["profile"]
    if not profile:
        raise ValueError(f"Unknown ticker: {ticker}")
    ratios = fin.get("ratios", {}) or {}

    # Everything below this point goes through the safe-runner: a failure in
    # any single specialist becomes a typed fallback rather than killing the
    # memo. Failures are accumulated into `degradation` and surfaced on the
    # memo's `degraded_agents` field.
    degradation = DegradationLog()

    transcript = safe_call(latest_transcript, ticker, fallback=None,
                           name="Transcript Service", log_to=degradation)
    filings = safe_call(get_filings, ticker, fallback=[],
                        name="Filings Service", log_to=degradation)
    earnings = fin.get("earnings", {})

    dcf = safe_call(_checkpointed_dcf, ticker, force_refresh=force_refresh, fallback=None,
                    name="DCF Engine", log_to=degradation)
    comps = safe_call(_checkpointed_comps, ticker, force_refresh=force_refresh, fallback=None,
                      name="Comps Engine", log_to=degradation)

    # Wave 10 — PM intake step. Lets the PM deprioritize up to 3
    # specialists for this memo (e.g., skip technicals on a regulated
    # bank, skip filings re-pass when nothing material has changed).
    # Default = run all 8. Decision is logged on the memo for audit.
    from .intake import run_intake, stub_finding
    intake = run_intake(profile)

    # Each specialist runs with its own llm_call_context so any LLM calls it
    # makes get tagged with the right agent_name in LLMCallLog (Wave 1A).
    from .llm import llm_call_context
    if intake.runs("sector"):
        with llm_call_context(agent_name="Sector Analyst", run_id=run_id):
            sector_finding = safe_finding("Sector Analyst", _checkpointed_sector,
                                          profile, ratios, log_to=degradation)
    else:
        sector_finding = AgentFinding(**stub_finding("sector", intake.rationale))
    if intake.runs("earnings"):
        with llm_call_context(agent_name="Earnings Analyst", run_id=run_id):
            earnings_finding = safe_finding("Earnings Analyst", _checkpointed_earnings,
                                            profile, transcript, earnings, log_to=degradation)
    else:
        earnings_finding = AgentFinding(**stub_finding("earnings", intake.rationale))
    if intake.runs("filing"):
        with llm_call_context(agent_name="Filing Analyst", run_id=run_id):
            filing_finding = safe_finding("Filing Analyst", _checkpointed_filing,
                                          profile, filings, log_to=degradation)
    else:
        filing_finding = AgentFinding(**stub_finding("filing", intake.rationale))
    if intake.runs("valuation"):
        with llm_call_context(agent_name="Valuation Analyst", run_id=run_id):
            valuation_finding = safe_finding("Valuation Analyst", _checkpointed_valuation,
                                             profile, ratios, dcf, log_to=degradation)
    else:
        valuation_finding = AgentFinding(**stub_finding("valuation", intake.rationale))
    if intake.runs("comps"):
        with llm_call_context(agent_name="Comps Analyst", run_id=run_id):
            comps_finding = safe_finding("Comps Analyst", _checkpointed_comps_agent,
                                         profile, comps, log_to=degradation)
    else:
        comps_finding = AgentFinding(**stub_finding("comps", intake.rationale))
    if intake.runs("macro"):
        with llm_call_context(agent_name="Macro Analyst", run_id=run_id):
            macro_finding = safe_finding("Macro Analyst", _checkpointed_macro,
                                         profile, scenario, log_to=degradation)
    else:
        macro_finding = AgentFinding(**stub_finding("macro", intake.rationale))
    if intake.runs("risk"):
        with llm_call_context(agent_name="Risk Analyst", run_id=run_id):
            risk_finding = safe_finding(
                "Risk Analyst", _checkpointed_risk,
                profile, ratios, (dcf.summary if dcf else None), log_to=degradation,
            )
    else:
        risk_finding = AgentFinding(**stub_finding("risk", intake.rationale))
    # Wave 3B — Technical Analyst. By design technicals do NOT influence
    # the rating; they're positioning context only. The agent gets its own
    # llm_call_context so the LLM narrative pass is attributed correctly.
    if intake.runs("technical"):
        with llm_call_context(agent_name="Technical Analyst", run_id=run_id):
            technical_finding = safe_finding(
                "Technical Analyst", _checkpointed_technical, profile,
                log_to=degradation,
            )
    else:
        technical_finding = AgentFinding(**stub_finding("technical", intake.rationale))

    findings = {
        "sector": sector_finding,
        "earnings": earnings_finding,
        "filing": filing_finding,
        "valuation": valuation_finding,
        "comps": comps_finding,
        "macro": macro_finding,
        "risk": risk_finding,
        "technical": technical_finding,
    }

    # Wave 9 — PM↔specialist deep-research dialog. Round 0 is the fan-out
    # above; rounds 1+ critique + re-fire targeted specialists with the
    # PM's question prepended to their prompt. Skipped on backtests
    # (`as_of_date` set) so we don't burn LLM budget retroactively.
    round_findings: List[Any] = []
    if settings.enable_deep_research and as_of_date is None:
        from .deep_research import run_dialog_loop

        def _refire_sector(q: str) -> AgentFinding:
            return run_sector_agent(profile, ratios, prior_round_critique=q)

        def _refire_earnings(q: str) -> AgentFinding:
            return run_earnings_agent(
                profile, transcript, earnings, prior_round_critique=q,
            )

        def _refire_filing(q: str) -> AgentFinding:
            return run_filing_agent(profile, filings, prior_round_critique=q)

        def _refire_valuation(q: str) -> AgentFinding:
            return run_valuation_agent(profile, ratios, dcf, prior_round_critique=q)

        def _refire_comps(q: str) -> AgentFinding:
            return run_comps_agent(profile, comps, prior_round_critique=q)

        def _refire_macro(q: str) -> AgentFinding:
            return run_macro_agent(profile, scenario, prior_round_critique=q)

        def _refire_risk(q: str) -> AgentFinding:
            return run_risk_agent(
                profile, ratios, (dcf.summary if dcf else None),
                prior_round_critique=q,
            )

        def _refire_technical(q: str) -> AgentFinding:
            return run_technical_agent(profile, prior_round_critique=q)

        re_fire = {
            "sector": _refire_sector,
            "earnings": _refire_earnings,
            "filing": _refire_filing,
            "valuation": _refire_valuation,
            "comps": _refire_comps,
            "macro": _refire_macro,
            "risk": _refire_risk,
            "technical": _refire_technical,
        }

        # Loop reads `findings` keyed by short agent name — same as the
        # `re_fire` map. Returns the latest-per-agent findings dict + the
        # full round-by-round audit trail for persistence.
        def _run_loop():
            return run_dialog_loop(
                run_id=run_id,
                initial_findings=findings,
                re_fire=re_fire,
            )

        loop_out = safe_call(
            _run_loop,
            fallback=(findings, []),
            name="Deep Research Loop", log_to=degradation,
        )
        if loop_out:
            current, rounds = loop_out
            round_findings = rounds
            # Replace each agent's finding with the latest-round version so
            # downstream synthesis (PM, critic) sees the freshest read.
            for name, finding in current.items():
                findings[name] = finding
            sector_finding = findings["sector"]
            earnings_finding = findings["earnings"]
            filing_finding = findings["filing"]
            valuation_finding = findings["valuation"]
            comps_finding = findings["comps"]
            macro_finding = findings["macro"]
            risk_finding = findings["risk"]
            technical_finding = findings["technical"]

    # Wave 3C: drill-down long-form reports. The deterministic build is
    # cheap and always populates the field; LLM enrichment runs only when
    # ENABLE_LONG_FORM_REPORTS=true. safe_call wraps so a failure never
    # blocks the memo.
    from .long_form import attach_long_form
    _t = profile.get("ticker", ticker)
    safe_call(attach_long_form, sector_finding, ticker=_t, agent_name="Sector Analyst",
              profile=profile, fallback=None, name="Long-form (Sector)", log_to=degradation)
    safe_call(attach_long_form, earnings_finding, ticker=_t, agent_name="Earnings Analyst",
              profile=profile, fallback=None, name="Long-form (Earnings)", log_to=degradation)
    safe_call(attach_long_form, filing_finding, ticker=_t, agent_name="Filing Analyst",
              profile=profile, fallback=None, name="Long-form (Filing)", log_to=degradation)
    safe_call(attach_long_form, valuation_finding, ticker=_t, agent_name="Valuation Analyst",
              profile=profile, fallback=None, name="Long-form (Valuation)", log_to=degradation)
    safe_call(attach_long_form, comps_finding, ticker=_t, agent_name="Comps Analyst",
              profile=profile, fallback=None, name="Long-form (Comps)", log_to=degradation)
    safe_call(attach_long_form, macro_finding, ticker=_t, agent_name="Macro Analyst",
              profile=profile, fallback=None, name="Long-form (Macro)", log_to=degradation)
    safe_call(attach_long_form, risk_finding, ticker=_t, agent_name="Risk Analyst",
              profile=profile, fallback=None, name="Long-form (Risk)", log_to=degradation)
    safe_call(attach_long_form, technical_finding, ticker=_t, agent_name="Technical Analyst",
              profile=profile, fallback=None, name="Long-form (Technical)", log_to=degradation)

    # Wave 10 — PM-driven DCF assumption adjustment. The PM has the team's
    # full read at this point (round 0 + Wave 9 dialog rounds). Now is when
    # the model should reflect the team's view, not just consensus defaults.
    # Skipped on backtests so retroactive runs use period-appropriate DCF.
    initial_dcf = dcf
    pm_dcf_adjustments: List[Dict[str, Any]] = []
    pm_dcf_headline = ""
    if dcf is not None and as_of_date is None and settings.has_llm:
        from .dcf_pm_adjuster import adjust_dcf_for_pm_view
        adj_out = safe_call(
            adjust_dcf_for_pm_view,
            ticker=_t, initial_dcf=dcf,
            findings=findings, run_id=run_id,
            fallback=(None, [], ""),
            name="PM DCF Adjuster", log_to=degradation,
        )
        if adj_out:
            adjusted_dcf, pm_dcf_adjustments, pm_dcf_headline = adj_out
            if adjusted_dcf is not None and pm_dcf_adjustments:
                # Replace the working DCF — downstream synthesis, bull/bear,
                # factor scoring all see the PM-adjusted version.
                dcf = adjusted_dcf

    bull = safe_call(_bull_case, profile, valuation_finding, dcf, sector_finding, findings,
                     fallback=BullBearCase(headline="Bull case unavailable.", key_points=[]),
                     name="Bull Case Builder", log_to=degradation)
    bear = safe_call(_bear_case, profile, dcf, sector_finding, findings,
                     fallback=BullBearCase(headline="Bear case unavailable.", key_points=[]),
                     name="Bear Case Builder", log_to=degradation)
    catalysts = safe_call(_catalysts, profile, transcript, findings, earnings,
                          fallback=[],
                          name="Catalyst Builder", log_to=degradation)
    risks = safe_call(derive_risk_items, profile, fallback=[],
                      name="Risk Item Builder", log_to=degradation)
    thesis_breakers = [r for r in risks if r.severity == "high"][:3]

    with llm_call_context(agent_name="PM Synthesis", run_id=run_id, route="strong"):
        synth = safe_call(
            _pm_synthesis, profile, findings, dcf,
            fallback={
                "final_pm_view": "PM synthesis unavailable; relying on specialist findings only.",
                "one_sentence_thesis": f"Research draft for {profile.get('ticker', ticker)}.",
                "rating_label": "Neutral",
                "confidence_score": 50,
            },
            name="PM Synthesis", log_to=degradation,
        )
    rating = synth.get("rating_label", "Neutral")
    raw_confidence = float(synth.get("confidence_score", 60))

    def _summarize_dcf(d: Optional[DCFResult]) -> Dict[str, Any]:
        if d is None:
            return {}
        return dict(
            current_price=d.current_price,
            base_implied_price=d.base.implied_share_price,
            bull_implied_price=d.bull.implied_share_price,
            bear_implied_price=d.bear.implied_share_price,
            base_upside=d.base.upside_pct,
            bull_upside=d.bull.upside_pct,
            bear_upside=d.bear.upside_pct,
            wacc=d.base.assumptions.wacc,
            terminal_growth=d.base.assumptions.terminal_growth,
            summary=d.summary,
        )

    dcf_summary = _summarize_dcf(dcf)
    # Wave 10 — keep the consensus-anchored ("initial") DCF on the memo
    # alongside the PM-adjusted version, when they differ. Empty when no
    # PM adjustments fired.
    initial_dcf_summary = (
        _summarize_dcf(initial_dcf) if pm_dcf_adjustments and initial_dcf is not dcf else {}
    )

    sources = [
        f"profile:{profile.get('ticker')}",
        f"financials:{profile.get('ticker')}",
    ]
    if transcript:
        sources.append(f"transcript:{transcript.get('period', '')}")
    for f in filings or []:
        sources.append(f"filing:{f.get('accession_number', f.get('type', ''))}")
    if comps:
        for p in comps.peers:
            sources.append(f"peer:{p.ticker}")
    if dcf:
        sources.append("dcf:base")

    # Dampen PM confidence by source-quality. A memo evidenced by filings +
    # transcripts + financials lands near 1.0; one leaning on news/social
    # gets a meaningful penalty. Prevents over-confident takes from thin evidence.
    ev_q = evidence_quality(sources)
    blended_confidence = max(20.0, min(95.0, raw_confidence * (0.6 + 0.4 * ev_q)))

    # Wave 10 — pull the mispricing thesis off the PM's structured
    # output (PM_SYNTHESIS_PROMPT now requires it). Empty fallback when
    # the deterministic path ran (no LLM) or the PM declined.
    from ..schemas import MispricingThesis
    raw_misp = synth.get("mispricing_thesis") or {}
    if not isinstance(raw_misp, dict):
        raw_misp = {}
    mispricing = MispricingThesis(
        consensus_view=str(raw_misp.get("consensus_view") or "")[:1000],
        our_view=str(raw_misp.get("our_view") or "")[:1000],
        gap=str(raw_misp.get("gap") or "")[:1000],
        falsifiers=[str(x)[:300] for x in (raw_misp.get("falsifiers") or [])][:5],
    )

    # Wave 10 — freeze the memo-time price so the UI can later overlay
    # the live quote and show drift. Best-effort: null when the quote
    # chain misses (the live-overlay path then has nothing to compare
    # against, which is fine).
    price_at_memo: Optional[float] = None
    try:
        from ..services.market_data_service import get_current_price
        price_at_memo = get_current_price(profile.get("ticker", ticker))
    except Exception as exc:  # pragma: no cover — never block a memo
        log.debug("price_at_memo capture failed: %s", exc)

    # Wave 10 — forward catalyst calendar (next 90d). Best-effort —
    # the table may be empty until the cron has run at least once.
    forward_catalysts: List[Dict[str, Any]] = []
    try:
        from ..services.catalyst_service import get_upcoming
        forward_catalysts = get_upcoming(profile.get("ticker", ticker), days_ahead=90)
    except Exception as exc:  # pragma: no cover
        log.debug("forward_catalysts fetch failed: %s", exc)

    # Wave 10 — earnings quarter-over-quarter delta. Reads the
    # earnings agent's structured payload and walks back through the
    # memo history for prior-quarter context. None when no prior data.
    earnings_qoq: Optional[AgentFinding] = None
    try:
        from .earnings_qoq import run_earnings_qoq_delta
        earnings_struct = (earnings_finding.data or {}).get("structured") if earnings_finding else None
        earnings_qoq = run_earnings_qoq_delta(
            profile.get("ticker", ticker), earnings_struct,
        )
    except Exception as exc:  # pragma: no cover
        log.debug("earnings QoQ delta failed: %s", exc)

    # Wave 10 — per-agent influence on the rating. Computed from each
    # finding's confidence + tone; deterministic, no extra LLM cost.
    # Powers per-agent attribution dashboards + the PM's eventual
    # "discount this specialist" feedback loop.
    agent_influence: Dict[str, float] = {}
    try:
        from .influence import compute_influence
        agent_influence = compute_influence(findings)
    except Exception as exc:  # pragma: no cover
        log.debug("agent influence computation failed: %s", exc)

    # Wave 10 — freeze the macro context that produced this rating.
    # Lets postmortem regime-conditional bucketing work even after
    # the macro broadcast cache rolls over.
    macro_snapshot_at_memo: Dict[str, float] = {}
    macro_regime_at_memo: str = ""
    try:
        from ..cache import cache_get
        broadcast = cache_get("macro:global", "macro_broadcast")
        if broadcast and isinstance(broadcast.payload, dict):
            macro_regime_at_memo = str(broadcast.payload.get("regime") or "")
            snap = broadcast.payload.get("snapshot") or {}
            if isinstance(snap, dict):
                macro_snapshot_at_memo = {
                    str(k): float(v) for k, v in snap.items()
                    if isinstance(v, (int, float))
                }
    except Exception as exc:  # pragma: no cover
        log.debug("macro snapshot freeze failed: %s", exc)
    memo = StockMemoOut(
        ticker=profile.get("ticker"),
        company_name=profile.get("company_name", ticker),
        sector=profile.get("sector", ""),
        final_pm_view=synth.get("final_pm_view", ""),
        rating_label=rating,
        confidence_score=blended_confidence,
        one_sentence_thesis=synth.get("one_sentence_thesis", ""),
        mispricing_thesis=mispricing,
        price_at_memo=price_at_memo,
        price_at_memo_at=(datetime.utcnow() if price_at_memo is not None else None),
        business_summary=profile.get("business_description", ""),
        sector_agent_view=sector_finding,
        earnings_agent_view=earnings_finding,
        filing_agent_view=filing_finding,
        valuation_agent_view=valuation_finding,
        comps_agent_view=comps_finding,
        macro_sensitivity=macro_finding,
        technical_agent_view=technical_finding,
        bull_case=bull,
        bear_case=bear,
        catalysts=catalysts,
        key_risks=risks,
        thesis_breakers=thesis_breakers,
        dcf_summary=dcf_summary,
        dcf_initial_summary=initial_dcf_summary,
        dcf_pm_adjustments=pm_dcf_adjustments,
        dcf_pm_adjustment_headline=pm_dcf_headline,
        portfolio_fit=_portfolio_fit(profile, rating),
        # Stub critic seeded here, then replaced by the real critic call below.
        # safe_critic guarantees a typed CriticReview even if the stub raises.
        risk_committee_challenge=safe_critic(run_critic, {}, log_to=None) or CriticReview(
            overall_assessment="Pending critic review.",
        ),
        final_verdict="",
        scores=_build_scores_dict(
            blended_confidence=blended_confidence,
            raw_confidence=raw_confidence,
            ev_q=ev_q,
            sector_finding=sector_finding,
            valuation_finding=valuation_finding,
            risk_finding=risk_finding,
            earnings_finding=earnings_finding,
            profile=profile, ratios=ratios, earnings=earnings,
        ),
        sources_used=sources,
        generated_at=datetime.utcnow(),
        generation_mode="live" if settings.has_llm and settings.enable_live_data else "demo",
        degraded_agents=degradation.degraded_agents(),
        round_findings=round_findings,
        forward_catalysts=forward_catalysts,
        earnings_qoq_delta=earnings_qoq,
        intake_decision=intake.model_dump() if intake.skipped else {},
        agent_influence=agent_influence,
        macro_snapshot_at_memo=macro_snapshot_at_memo,
        macro_regime_at_memo=macro_regime_at_memo,
    )

    # Wave 9 — surface deep-research counters on `memo.scores` so the
    # admin dashboard can chart how often the dialog converges vs. caps
    # out. Round 0 is the fan-out and is always present when the loop
    # ran; rounds 1+ are the PM critique passes.
    if round_findings and isinstance(memo.scores, dict):
        memo.scores = {
            **memo.scores,
            "deep_research_rounds": float(
                max((r.round for r in round_findings), default=0)
            ),
            "deep_research_questions": float(sum(
                len(r.pm_questions) for r in round_findings
            )),
        }

    # Run critic on a draft of the memo (pass dict to avoid recursion).
    # safe_critic upgrades exceptions into a typed "critic unavailable" review
    # so a flaky Anthropic call doesn't kill the memo.
    draft_for_critic = memo.model_dump()
    with llm_call_context(agent_name="Risk Committee", run_id=run_id, route="strong"):
        critic = safe_critic(_checkpointed_critic, draft_for_critic, log_to=degradation)
    if critic:
        memo.risk_committee_challenge = critic
    # Refresh degraded_agents in case the critic recorded a failure.
    memo.degraded_agents = degradation.degraded_agents()

    # Long-term memory: appends a structured entry to the company + sector
    # memory files iff a delta event fired this run (new earnings / new
    # filing / material news). safe_call wraps it so a memory write never
    # blocks the memo from being returned.
    #
    # Wave 1C: skip memory writes when running as a backtest (`as_of_date`
    # set). Backtests are diagnostic — we don't want the agent's notebook
    # polluted with retroactive entries.
    if as_of_date is None:
        safe_call(
            _run_reflection_step, memo,
            fallback=([], []),
            name="Reflection (long-term memory)", log_to=degradation,
        )
        memo.degraded_agents = degradation.degraded_agents()

    # Wave 8H — apply the risk analyst's structured recommendations.
    # Runs AFTER the memo body is assembled but BEFORE final_verdict +
    # persistence so confidence cap / rating downshift / thesis_breaker
    # propagation / bear-case augmentation all flow through to the
    # downstream UI + cache. `applied_recs` rides on the memo's `scores`
    # for transparency.
    applied_risk_recs = _apply_risk_recommendations(memo, risk_finding)
    if applied_risk_recs and isinstance(memo.scores, dict):
        memo.scores = {
            **memo.scores,
            "risk_recs_applied": float(len(applied_risk_recs)),
        }
    # Stash the audit trail on the risk finding's data block so the
    # frontend can render a "Risk recs applied" panel + the long-form
    # report can quote them verbatim.
    if isinstance(risk_finding.data, dict):
        risk_finding.data["applied_recommendations"] = applied_risk_recs

    # Rating blend (Option A) — mix the PM LLM's directional call with
    # the quant factor_pm_score. Weight is `LLM_RATING_WEIGHT` in
    # config.env (default 0.4). At weight=0 this collapses to the
    # prior Wave 8P behavior (factor score is dispositive); at
    # weight=1 the LLM call wins outright. The LLM rating read here
    # is post-risk-rec, so risk_agent downgrades flow into the blend.
    from ..schemas import rating_from_stock_score, score_from_rating_label
    from ..config import settings as _settings
    factor_pm = (memo.scores or {}).get("factor_pm_score")
    if factor_pm is not None:
        w = max(0.0, min(1.0, float(_settings.llm_rating_weight)))
        llm_score = score_from_rating_label(memo.rating_label)
        blended = w * llm_score + (1.0 - w) * float(factor_pm)
        memo.rating_label = rating_from_stock_score(blended)  # type: ignore[assignment]
        if isinstance(memo.scores, dict):
            memo.scores = {
                **memo.scores,
                "llm_rating_score": float(llm_score),
                "llm_rating_weight": float(w),
                "blended_pm_score": round(float(blended), 1),
            }

    # Anti-pattern guard. The PM prompt explicitly forbids the
    # "{Company} — {Sector} / {industry}, {hook}; DCF base case +X%"
    # templated form, but in practice the LLM sometimes ignores it (or
    # the deterministic fallback historically emitted it). Detect and
    # rewrite from the richer specialist findings before the thesis
    # ships into the memo and is persisted to memory.
    if _looks_like_anti_pattern_thesis(memo.one_sentence_thesis):
        try:
            rewritten = _build_thesis_from_findings(
                profile, findings, dcf, ticker, rating=memo.rating_label,
            )
            if rewritten and not _looks_like_anti_pattern_thesis(rewritten):
                memo.one_sentence_thesis = rewritten
        except Exception:  # pragma: no cover — never break the memo
            pass

    # Wave 8R — thesis augmentation. Surface where the model diverges
    # from analyst consensus (the actual *what is the market missing*
    # framing). Compares the DCF's 5-year growth path average against
    # the consensus 5-year average; appends a clause when the gap is
    # material. No-ops cleanly when consensus isn't available. Skip
    # when the thesis already carries the clause (the deterministic
    # path bakes it in via `_build_thesis_from_findings`).
    try:
        delta_clause = _market_gap_clause(profile, dcf, ticker)
        # Only append when it agrees with the verdict word implied by the
        # headline rating — an "upside the market is missing" clause behind
        # an overvalued call would contradict the thesis.
        _verdict = _verdict_word(memo.rating_label, dcf.base.upside_pct if dcf and dcf.base else None)
        if (
            delta_clause
            and delta_clause not in memo.one_sentence_thesis
            and _gap_clause_agrees(delta_clause, _verdict)
        ):
            memo.one_sentence_thesis = (
                memo.one_sentence_thesis.rstrip(".")
                + ". " + delta_clause
            )
    except Exception:  # pragma: no cover — never break a memo on thesis polish
        pass

    # Refresh the rating/confidence-derived locals after enforcement.
    rating = memo.rating_label
    # thesis_breakers may have grown; rebuild the local view used below.
    thesis_breakers = list(memo.thesis_breakers)

    # Phase 6: pull through cross-sector relevance from the sector agent's
    # finding into the PM memo so users see related-name implications without
    # a second model call. Cohort placement is already in the sector view.
    cross_relevance = []
    if isinstance(sector_finding.data, dict):
        cross_relevance = sector_finding.data.get("cross_sector_relevance") or []
    cross_relevance_blurb = (
        f" Cross-sector pull-through: {', '.join(cross_relevance)}." if cross_relevance else ""
    )
    cohort_blurb = ""
    if isinstance(sector_finding.data, dict):
        kpi_placements = sector_finding.data.get("kpi_placements") or {}
        if kpi_placements:
            cohort_blurb = " Cohort placement: see sector view for KPI quartile context."

    # Wave 3A: surface the key disagreement + sector lean in the verdict so
    # readers can see what the bull/bear case actually pivots on.
    sector_lean_blurb = ""
    bb_payload = (
        sector_finding.data.get("bull_bear_analysis")
        if isinstance(sector_finding.data, dict) else None
    )
    if isinstance(bb_payload, dict):
        lean = bb_payload.get("sector_lean")
        disagreement = (bb_payload.get("key_disagreement") or "").strip()
        if lean and lean != "balanced":
            sector_lean_blurb += f" Sector lean: {lean}."
        if disagreement:
            sector_lean_blurb += f" Key disagreement: {disagreement}"

    # Final verdict ties together rating, confidence, and PM view succinctly
    memo.final_verdict = (
        f"PM final view: {rating} (confidence {int(memo.confidence_score)}). "
        f"{memo.one_sentence_thesis}"
        f"{cohort_blurb}{cross_relevance_blurb}{sector_lean_blurb} "
        f"Watch items: {', '.join(r.title for r in thesis_breakers) or 'none flagged.'}"
    )
    if cross_relevance and isinstance(memo.scores, dict):
        memo.scores = {**memo.scores, "cross_sector_relevance_count": float(len(cross_relevance))}

    # Phase F: persist a versioned snapshot. `first_run` only fires when no
    # prior version exists for this ticker; otherwise this is a
    # `full_reanalysis` (the news-driven `incremental_patch` path is owned
    # by the future update-orchestrator, not this code path).
    #
    # Persistence used to be wrapped in safe_call so a DB hiccup wouldn't
    # block the in-memory return value — but for the async regen path
    # that's a silent disaster: the regen looks "successful" while the
    # memo never reaches the database. The user clicks Refresh, sees
    # spinning, then the old memo. Now we let persistence errors raise.
    # The _run_regen_job handler catches BaseException and records the
    # traceback in _REGEN_FAILURES, surfaced via /analyze/status.
    try:
        _persist_memo_snapshot(memo, as_of_date)
    except Exception as exc:
        log.error(
            "memo persistence FAILED for %s: %s: %s",
            ticker, type(exc).__name__, exc,
        )
        # Record on the degradation log so synchronous callers (sync=true
        # path) can still see what happened via memo.degraded_agents.
        degradation.record("Memo store", exc)
        memo.degraded_agents = degradation.degraded_agents()
        raise
    memo.degraded_agents = degradation.degraded_agents()
    return memo


def _persist_memo_snapshot(memo: StockMemoOut, as_of_date: Optional[Any] = None) -> None:
    """Indirection so safe_call wraps DB I/O. Lazy-import keeps graph.py from
    pulling the ORM at module import time (it's already loaded via models).

    Wave 1C: backtest snapshots are persisted with `as_of_date` set so the
    default `latest_memo` lookup excludes them.
    """
    from ..services import memo_store
    # latest_memo defaults to live snapshots only. For backtests we ask
    # for include_backtests so version chains stay continuous within the
    # same as-of date space.
    prior = memo_store.latest_memo(memo.ticker, include_backtests=as_of_date is not None)
    trigger = "first_run" if prior is None else "full_reanalysis"
    parent_version = prior.version if prior is not None else None
    memo_store.save_memo(memo, trigger=trigger, parent_version=parent_version,
                         as_of_date=as_of_date)


# ---------------------------------------------------------------------------
# Agent trace helper
# ---------------------------------------------------------------------------

def default_agent_trace(intent: str) -> List[AgentTrace]:
    base = [
        AgentTrace(agent="PM Orchestrator", status="done", detail=f"Intent classified as {intent}."),
    ]
    if intent in ("single_stock_analysis", "stock_comparison"):
        base += [
            AgentTrace(agent="Sector Analyst", status="done", detail="Sector framework applied."),
            AgentTrace(agent="Earnings Analyst", status="done", detail="Latest transcript reviewed."),
            AgentTrace(agent="Filing Analyst", status="done", detail="10-K/10-Q analyzed."),
            AgentTrace(agent="Valuation Analyst", status="done", detail="DCF + multiples interpreted."),
            AgentTrace(agent="Comps Analyst", status="done", detail="Peer median + premium/discount."),
            AgentTrace(agent="Macro Analyst", status="done", detail="Macro mapping applied."),
            AgentTrace(agent="Risk Committee", status="done", detail="Critic reviewed and flagged challenges."),
        ]
    elif intent == "portfolio_construction":
        base += [
            AgentTrace(agent="Screener Agent", status="done", detail="Universe scored against scenario fit."),
            AgentTrace(agent="Portfolio Construction Agent", status="done", detail="Diversified weights enforced."),
            AgentTrace(agent="Risk Committee", status="done", detail="Concentration + risk reviewed."),
        ]
    elif intent == "thematic_screen":
        base += [AgentTrace(agent="Screener Agent", status="done", detail="Theme bias applied to PM scores.")]
    elif intent == "macro_question":
        base += [AgentTrace(agent="Macro Analyst", status="done", detail="Scenario template + snapshot.")]
    return base
