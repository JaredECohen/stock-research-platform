"""Wave 3A tests — sector-integrated bull/bear analysis.

Covers:
- Schema: `BullBearAnalysis` requires both sides + accepts the sector lean enum.
- LLM-output coercer: malformed payloads return None (graceful fallback);
  well-formed payloads parse cleanly; bad invalidates_side values dropped.
- Deterministic fallback: produces both sides, ≥1 falsifiable test per side,
  cohort-grounded headlines using profile risks/drivers.
- Sector agent: in demo (no-LLM) mode, the finding's `data` carries a
  `bull_bear_analysis` block satisfying the contract.
- Graph integration: when sector finding has the structured block, the
  memo's bull_case / bear_case use it; the verdict surfaces sector lean
  + key_disagreement.
"""
from __future__ import annotations

from app.agents import graph as graph_module
from app.agents.sector_agents import (
    _coerce_bull_bear_analysis,
    _deterministic_bull_bear_analysis,
    run_sector_agent,
)
from app.schemas import (
    AgentFinding,
    BullBearAnalysis,
    BullBearCase,
    FalsifiableTest,
)


def _profile():
    return {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "Technology",
        "sub_industry": "Semiconductors",
        "drivers": ["AI infrastructure demand", "Data-center share take", "Software ecosystem moat"],
        "risks": ["Cyclical hyperscaler capex", "Customer concentration", "Geopolitical exposure"],
    }


def _research():
    return {
        "kpi_placements": {
            "revenue_growth": {"quartile": 1, "target": 0.85, "distribution": {"median": 0.18}},
            "operating_margin": {"quartile": 1, "target": 0.55, "distribution": {"median": 0.20}},
            "EV_EBITDA": {"quartile": 4, "target": 35.0, "distribution": {"median": 18.0}},
        },
        "trends": {"cohort_op_margin_delta": 0.02, "cohort_revenue_growth_recent": 0.21},
        "regime": "expansion",
    }


# ---------------------------------------------------------------------------
# Coercer
# ---------------------------------------------------------------------------

def test_coercer_returns_none_on_missing_required_fields():
    assert _coerce_bull_bear_analysis(None) is None
    assert _coerce_bull_bear_analysis({}) is None
    assert _coerce_bull_bear_analysis({"bull_case": {}, "bear_case": {}}) is None
    # Headlines required.
    assert _coerce_bull_bear_analysis({
        "bull_case": {"headline": "", "key_points": []},
        "bear_case": {"headline": "", "key_points": []},
    }) is None


def test_coercer_parses_well_formed_payload():
    raw = {
        "bull_case": {"headline": "Quality + growth", "key_points": ["Tailwind: AI"]},
        "bear_case": {"headline": "Cycle exposure", "key_points": ["Headwind: capex"]},
        "falsifiable_tests": [
            {"statement": "Cohort growth turns negative", "invalidates_side": "bull"},
            {"statement": "AI capex re-accelerates", "invalidates_side": "bear"},
        ],
        "key_disagreement": "Bulls bet on durability; bears bet on cycle.",
        "sector_synthesis": "Cohort placement supports premium.",
        "sector_lean": "bull",
    }
    parsed = _coerce_bull_bear_analysis(raw)
    assert parsed is not None
    assert parsed.sector_lean == "bull"
    assert len(parsed.falsifiable_tests) == 2
    assert parsed.bear_case.headline == "Cycle exposure"


def test_coercer_drops_bad_falsifiable_test_entries():
    raw = {
        "bull_case": {"headline": "x", "key_points": []},
        "bear_case": {"headline": "y", "key_points": []},
        "falsifiable_tests": [
            {"statement": "valid", "invalidates_side": "bull"},
            {"statement": "missing side"},
            {"invalidates_side": "wrong_value", "statement": "also bad"},
        ],
        "sector_lean": "balanced",
    }
    parsed = _coerce_bull_bear_analysis(raw)
    assert parsed is not None
    assert len(parsed.falsifiable_tests) == 1
    assert parsed.falsifiable_tests[0].invalidates_side == "bull"


def test_coercer_normalizes_bad_lean_to_balanced():
    raw = {
        "bull_case": {"headline": "x", "key_points": []},
        "bear_case": {"headline": "y", "key_points": []},
        "sector_lean": "moonshot",  # not in literal
    }
    parsed = _coerce_bull_bear_analysis(raw)
    assert parsed is not None
    assert parsed.sector_lean == "balanced"


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def test_deterministic_fallback_produces_complete_contract():
    bb = _deterministic_bull_bear_analysis(_profile(), _research())
    assert isinstance(bb, BullBearAnalysis)
    # Both sides must have a headline + ≥1 key point.
    assert bb.bull_case.headline
    assert bb.bear_case.headline
    assert bb.bull_case.key_points
    assert bb.bear_case.key_points
    # Falsifiable tests: at least one per side.
    sides = {t.invalidates_side for t in bb.falsifiable_tests}
    assert "bull" in sides and "bear" in sides
    assert bb.sector_synthesis
    assert bb.key_disagreement


def test_deterministic_fallback_lean_reflects_quartile_placement():
    # Top-quartile growth + margin → bull lean.
    bb = _deterministic_bull_bear_analysis(_profile(), _research())
    assert bb.sector_lean == "bull"

    # Bottom-quartile growth + margin → bear lean.
    bear_research = {
        "kpi_placements": {
            "revenue_growth": {"quartile": 4, "target": -0.05, "distribution": {"median": 0.04}},
            "operating_margin": {"quartile": 4, "target": 0.02, "distribution": {"median": 0.10}},
        },
        "trends": {}, "regime": "contraction",
    }
    bb_bear = _deterministic_bull_bear_analysis(_profile(), bear_research)
    assert bb_bear.sector_lean == "bear"


def test_deterministic_fallback_uses_profile_risks_and_drivers():
    bb = _deterministic_bull_bear_analysis(_profile(), _research())
    bear_text = " ".join([bb.bear_case.headline, *bb.bear_case.key_points]).lower()
    bull_text = " ".join([bb.bull_case.headline, *bb.bull_case.key_points]).lower()
    # Bear pulls from risks; bull pulls from drivers.
    assert any(r.lower().split()[0] in bear_text for r in _profile()["risks"][:1])
    assert "ai" in bull_text or "data-center" in bull_text or "software" in bull_text


# ---------------------------------------------------------------------------
# Sector agent integration (no-LLM demo path)
# ---------------------------------------------------------------------------

def test_sector_agent_attaches_bull_bear_analysis_in_demo_mode():
    finding = run_sector_agent(_profile(), {})
    assert isinstance(finding, AgentFinding)
    assert isinstance(finding.data, dict)
    bb = finding.data.get("bull_bear_analysis")
    assert isinstance(bb, dict)
    assert bb["bull_case"]["headline"]
    assert bb["bear_case"]["headline"]
    assert bb["sector_lean"] in ("bull", "bear", "balanced")


# ---------------------------------------------------------------------------
# Graph integration — memo bull/bear sourced from sector when available
# ---------------------------------------------------------------------------

def _sector_finding_with_bb(lean: str = "bull") -> AgentFinding:
    bb = BullBearAnalysis(
        bull_case=BullBearCase(headline="Sector-integrated bull", key_points=["alpha"]),
        bear_case=BullBearCase(headline="Sector-integrated bear", key_points=["omega"]),
        key_disagreement="bulls vs bears on cycle",
        falsifiable_tests=[FalsifiableTest(statement="X", invalidates_side="bull")],
        sector_synthesis="cohort-grounded synthesis",
        sector_lean=lean,
    )
    return AgentFinding(
        agent="Sector Analyst", headline="h", summary="s", confidence=0.7,
        data={"bull_bear_analysis": bb.model_dump()},
    )


def test_graph_bull_case_uses_sector_block_when_present():
    sector = _sector_finding_with_bb()
    bull = graph_module._bull_case(_profile(), AgentFinding(agent="v", headline="", summary=""), None, sector)
    assert "Sector-integrated bull" in bull.headline
    assert "alpha" in bull.key_points


def test_graph_bear_case_uses_sector_block_when_present():
    sector = _sector_finding_with_bb()
    bear = graph_module._bear_case(_profile(), None, sector)
    assert "Sector-integrated bear" in bear.headline
    assert "omega" in bear.key_points


def test_graph_falls_back_to_template_when_no_sector_block():
    sector = AgentFinding(agent="Sector Analyst", headline="h", summary="s", data={})
    bull = graph_module._bull_case(_profile(), AgentFinding(agent="v", headline="", summary=""), None, sector)
    # Falls back to template using profile drivers.
    assert "Bull case" in bull.headline
    assert any("Tailwind:" in p for p in bull.key_points)


def test_run_stock_memo_surfaces_sector_lean_in_verdict():
    # Full memo path — verifies the verdict picks up sector lean / disagreement.
    memo = graph_module.run_stock_memo("NVDA")
    sector_data = memo.sector_agent_view.data
    bb = sector_data.get("bull_bear_analysis") if isinstance(sector_data, dict) else None
    assert isinstance(bb, dict)
    if bb.get("sector_lean") and bb["sector_lean"] != "balanced":
        assert "Sector lean" in memo.final_verdict
    if (bb.get("key_disagreement") or "").strip():
        assert "disagreement" in memo.final_verdict.lower()
