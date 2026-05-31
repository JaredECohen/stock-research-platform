"""Unit tests for the QA harness rubric scorer + backlog builder."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pytest

# The harness lives outside `app/` (in `backend/qa/`) to make it clear
# it's a sibling tool, not application code. Add backend/ to sys.path
# so `import qa.run_matrix` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa import run_matrix  # noqa: E402


def _good_memo() -> Dict[str, Any]:
    """A reasonably-shaped memo that should pass every rubric."""
    return {
        "rating_label": "Bullish",
        "one_sentence_thesis": "Sustained operating margin expansion + AI infrastructure tailwinds support a multi-year compounding thesis with attractive valuation.",
        "confidence_score": 75,
        "bull_case": {"headline": "Strong moat", "key_points": ["margin expansion", "AI tailwind", "buybacks"]},
        "bear_case": {"headline": "Competitive risk", "key_points": ["new entrants", "regulatory risk", "valuation"]},
        "sector_agent_view": {
            "summary": "The cohort is showing accelerating revenue growth and stable margins, with this name in the top quartile on ROIC.",
            "key_points": ["q1 ROIC", "rev growth +12%", "Tailwind: stable margins"],
            "data": {
                "sector_data_context": {
                    "discovered_catalog": {
                        "sector_relevant": [{"series_id": "DGS10"}],
                        "sub_industry_relevant": [],
                        "geography_relevant": [],
                        "category_relevant": [],
                    },
                    "overlays": {
                        "overlays_run": ["credit", "factor"],
                        "bundles": {
                            "credit": {
                                "available": True,
                                "narrative_hints": ["HY spread at 3.2%, +0.20pp over 3 months"],
                            }
                        },
                    },
                }
            },
        },
        "earnings_agent_view": {
            "summary": "Q3 revenue +18% YoY to $32B, op margin expanded 220bps to 41% — guide raised on stronger enterprise demand.",
            "key_points": ["Q3 revenue +18%", "guide raised", "enterprise demand strong"],
        },
        "filing_agent_view": {
            "summary": "10-K disclosed a new $50B authorization plus a litigation risk on the AI training set.",
            "key_points": ["$50B buyback authorization", "AI litigation flagged"],
        },
        "dcf_summary": {"fair_value": 220.0, "wacc": 0.08, "terminal_growth": 0.025},
        "comps_agent_view": {
            "summary": "Trades at a 15% premium to large-cap software peers on EV/EBITDA.",
            "key_points": ["EV/EBITDA 22x", "vs cohort median 19x"],
        },
        "macro_sensitivity": {
            "summary": "Rate-sensitive given long-duration cash flows; benefits from a more dovish FOMC.",
            "key_points": ["duration tilt", "DXY sensitivity"],
        },
        "key_risks": [
            {"title": "Margin compression", "detail": "If cloud capex moderates...", "severity": "medium", "type": "company"},
            {"title": "Regulatory", "detail": "Antitrust scrutiny", "severity": "low", "type": "regulatory"},
        ],
        "catalysts": [{"title": "AI capex update", "detail": "Q4 print", "horizon": "near_term", "impact": "high"}],
        "risk_committee_challenge": {
            "overall_assessment": "Thesis is internally consistent but understates regulatory tail risk.",
            "challenges": ["What if EU antitrust accelerates?"],
            "underweighted_risks": ["DOJ remedy scope"],
            "suggested_revisions": ["Add a downside scenario tied to forced divestiture."],
        },
        "degraded_agents": [],
    }


def _broken_memo() -> Dict[str, Any]:
    """A memo where most expectations should fail."""
    return {
        "rating_label": "BOGUS",
        "one_sentence_thesis": "",
        "bull_case": {"headline": "", "key_points": []},
        "bear_case": {"headline": "", "key_points": []},
        "sector_agent_view": {"summary": "", "key_points": []},
        "earnings_agent_view": {"summary": "Management tone was constructive.", "key_points": []},
        "filing_agent_view": {"summary": "", "key_points": []},
        "dcf_summary": {},
        "comps_agent_view": {"summary": "", "key_points": []},
        "macro_sensitivity": {"summary": "", "key_points": []},
        "key_risks": [],
        "catalysts": [],
        "risk_committee_challenge": {},
        "degraded_agents": ["sector", "filing"],
    }


def test_score_memo_all_rubrics_pass_on_good_memo():
    expects = list(run_matrix._RUBRIC_FUNCS.keys())
    results = run_matrix.score_memo(_good_memo(), expects)
    failed = [r.expectation for r in results if not r.passed]
    assert not failed, f"Unexpected failures on good memo: {failed}"


def test_score_memo_fails_on_broken_memo():
    expects = list(run_matrix._RUBRIC_FUNCS.keys())
    results = run_matrix.score_memo(_broken_memo(), expects)
    passed = [r.expectation for r in results if r.passed]
    # The broken memo should fail substantially more than it passes.
    assert len(passed) < len(results) / 3


def test_build_backlog_orders_by_priority_then_count():
    counter = Counter({
        "one_sentence_thesis_non_empty": 5,  # P1
        "rating_label_valid": 12,             # P1
        "critic_present": 2,                  # P2
        "comps_finding_present": 8,           # P3
    })
    backlog = run_matrix.build_backlog(counter)
    # P1 should come first, within P1 ordered by descending count.
    assert backlog[0]["expectation"] == "rating_label_valid"
    assert backlog[1]["expectation"] == "one_sentence_thesis_non_empty"
    # P2 next
    assert backlog[2]["expectation"] == "critic_present"
    # P3 last
    assert backlog[-1]["expectation"] == "comps_finding_present"


def test_unknown_expectation_returns_marker_failure():
    results = run_matrix.score_memo(_good_memo(), ["nonexistent_expectation"])
    assert len(results) == 1
    assert results[0].passed is False
    assert "no rubric implementation" in results[0].detail


def test_rubric_critic_added_counts_distinct_kinds():
    memo = _good_memo()
    # Strip all the critic outputs except `overall_assessment`
    memo["risk_committee_challenge"] = {
        "overall_assessment": "Thesis is internally consistent.",
        "challenges": [], "underweighted_risks": [], "suggested_revisions": [],
    }
    res = run_matrix.score_memo(memo, ["critic_present", "critic_added_challenge_or_revision"])
    by_tag = {r.expectation: r for r in res}
    assert by_tag["critic_present"].passed is True
    assert by_tag["critic_added_challenge_or_revision"].passed is False
