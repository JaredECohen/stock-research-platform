"""Wave 7C tests — cross-agent research-notes injection.

Covers:
- `build_notes_block_for_agent` returns "" cleanly when no notes match.
- A note tagged for the valuation agent surfaces in the valuation block
  and NOT in the sector block when sector tagging is exclusive.
- Risk agent's finding carries `data["research_notes"]` when matched
  notes exist; the long-form drill-down report renders them.
- Comps agent's finding preserves the existing Wave 3E `history` payload
  AND adds `research_notes` when matched notes exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services import research_notes as rn


def _seed(tmp_path: Path) -> Path:
    """Two notes: one tagged for valuation only, one for the sector only."""
    val = tmp_path / "books" / "val.md"
    val.parent.mkdir(parents=True, exist_ok=True)
    val.write_text(
        "---\n"
        "title: Valuation lens\n"
        "applies_to_agents: [valuation]\n"
        "applies_to_sectors: ['*']\n"
        "weight: 0.8\n"
        "---\n"
        "DCF terminal growth and WACC sensitivity guidance.\n",
        encoding="utf-8",
    )
    sec = tmp_path / "books" / "sec.md"
    sec.write_text(
        "---\n"
        "title: Sector lens\n"
        "applies_to_agents: [sector]\n"
        "applies_to_sectors: ['*']\n"
        "weight: 0.7\n"
        "---\n"
        "Cohort placement and regime narrative.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_build_notes_block_returns_empty_when_no_match(tmp_path):
    out = rn.build_notes_block_for_agent(
        "earnings", {"sector": "Technology"},
        # Override the corpus root to an empty dir.
    )
    assert out == ""


def test_build_notes_block_routes_per_agent(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setattr(rn, "_root_default", lambda: tmp_path)
    profile = {
        "ticker": "NVDA", "sector": "Technology",
        "drivers": ["AI capex"], "risks": ["concentration"],
    }
    val_block = rn.build_notes_block_for_agent("valuation", profile)
    sec_block = rn.build_notes_block_for_agent("sector", profile)
    assert "Valuation lens" in val_block
    assert "Sector lens" not in val_block
    assert "Sector lens" in sec_block
    assert "Valuation lens" not in sec_block


def test_build_notes_block_excludes_unrelated_agents(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setattr(rn, "_root_default", lambda: tmp_path)
    out = rn.build_notes_block_for_agent("earnings", {"sector": "Technology"})
    assert out == ""  # neither seed note is tagged for earnings


def test_risk_agent_attaches_research_notes_to_data(tmp_path, monkeypatch):
    """A note tagged `[risk]` should surface in finding.data['research_notes']."""
    risk_note = tmp_path / "personal" / "x.md"
    risk_note.parent.mkdir(parents=True, exist_ok=True)
    risk_note.write_text(
        "---\n"
        "title: Risk overlay\n"
        "applies_to_agents: [risk]\n"
        "applies_to_sectors: ['*']\n"
        "weight: 0.9\n"
        "---\n"
        "Downside survivable. Stress-test thesis breakers.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rn, "_root_default", lambda: tmp_path)

    from app.agents.risk_agent import run_risk_agent
    finding = run_risk_agent(
        {"ticker": "X", "sector": "Technology", "risks": ["concentration"]},
        ratios={}, dcf_summary=None,
    )
    assert finding.data.get("research_notes")
    assert "Risk overlay" in finding.data["research_notes"]


def test_comps_agent_preserves_history_and_adds_research_notes(monkeypatch, tmp_path):
    """The Wave 3E `data['history']` payload should not regress when
    Wave 7C tucks `research_notes` into the same dict."""
    comps_note = tmp_path / "books" / "c.md"
    comps_note.parent.mkdir(parents=True, exist_ok=True)
    comps_note.write_text(
        "---\n"
        "title: Comps lens\n"
        "applies_to_agents: [comps]\n"
        "applies_to_sectors: ['*']\n"
        "weight: 0.85\n"
        "---\n"
        "Premium / discount durability against own history.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rn, "_root_default", lambda: tmp_path)

    from app.agents.comps_agent import run_comps_agent
    from app.schemas import (
        CompsHistoryStats,
        CompsResult,
        CompsRow,
    )
    target = CompsRow(
        ticker="X", company_name="X", market_cap=100.0,
        ev_ebitda=20.0, operating_margin=0.20,
    )
    median = CompsRow(
        ticker="MEDIAN", company_name="Peer Median",
        ev_ebitda=18.0, operating_margin=0.18,
    )
    history = CompsHistoryStats(
        lookback_periods=20, lookback_label="20 quarters",
        own_median={"ev_ebitda": 18.0},
        own_p25={}, own_p75={},
        current_percentile={"ev_ebitda": 0.85},
        current_vs_own_median={"ev_ebitda": 0.10},
        interpretation="Premium to history.",
    )
    comps = CompsResult(
        target=target, peers=[target], median=median,
        premium_discount={"ev_ebitda": 0.10},
        target_percentiles={}, interpretation="peer interpretation",
        history=history,
    )
    finding = run_comps_agent({"ticker": "X", "sector": "Technology"}, comps)
    assert "history" in finding.data
    assert "research_notes" in finding.data
    assert "Comps lens" in finding.data["research_notes"]


def test_long_form_renders_research_notes_data_block():
    """`_format_data_evidence` should include a `research_notes` markdown
    string when present in `finding.data`."""
    from app.agents.long_form import _format_data_evidence
    out = _format_data_evidence({
        "research_notes": "## Discretionary investment context\n- **X**: y",
    })
    assert "Discretionary investment context" in out
