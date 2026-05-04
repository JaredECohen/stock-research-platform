"""Wave 9 — deep-research dialog loop tests.

Covers the four exit conditions for `run_dialog_loop`:
  1. PM declares `no_further_questions=True` on round 1 → loop exits after round 1.
  2. PM emits questions but every re-fire raises → loop exits to avoid spinning.
  3. Question budget exhausts (max_rounds reached without PM saying "no more").
  4. PM emits an empty questions list → equivalent to (1).

Plus a smoke test that round 0 always lands in the audit trail unchanged
and that re-fired specialists' findings replace the latest-per-agent view.

The PM critique step is monkey-patched per-test so we don't depend on a
live LLM. The contract is: the loop reads `pm_critique` and `re_fire`
dispatchers — nothing else.
"""
from __future__ import annotations

from typing import List

import pytest

from app.agents import deep_research as dr
from app.schemas import AgentFinding, CritiqueOutput, CritiqueQuestion


def _finding(name: str, summary: str = "stub") -> AgentFinding:
    return AgentFinding(
        agent=name, headline=f"{name} headline",
        summary=summary, key_points=[], confidence=0.7, sources=[],
    )


def _initial_findings() -> dict:
    return {
        "sector": _finding("Sector Analyst"),
        "earnings": _finding("Earnings Analyst"),
        "valuation": _finding("Valuation Analyst"),
    }


def test_round_zero_persisted_as_audit_anchor(monkeypatch):
    """Round 0 must land in the trail with no PM questions, even when the
    PM exits immediately. Reviewers need to see the fan-out happened."""
    monkeypatch.setattr(
        dr, "pm_critique",
        lambda **kw: CritiqueOutput(no_further_questions=True, rationale="all clear"),
    )
    final, rounds = dr.run_dialog_loop(
        run_id="r1",
        initial_findings=_initial_findings(),
        re_fire={},
        max_rounds=3,
    )
    assert rounds[0].round == 0
    assert rounds[0].pm_questions == []
    assert set(rounds[0].findings.keys()) == {"sector", "earnings", "valuation"}
    # PM exit appended as round 1 with early_exit=True so the audit log is honest.
    assert rounds[-1].early_exit is True
    assert rounds[-1].pm_rationale == "all clear"
    # Final = initial (no re-fires).
    assert final["sector"].headline == "Sector Analyst headline"


def test_pm_no_further_questions_stops_loop_after_one_round(monkeypatch):
    calls: List[str] = []

    def fake_critique(**kwargs):
        calls.append("called")
        return CritiqueOutput(no_further_questions=True, rationale="satisfied")

    monkeypatch.setattr(dr, "pm_critique", fake_critique)
    _, rounds = dr.run_dialog_loop(
        run_id="r2", initial_findings=_initial_findings(),
        re_fire={}, max_rounds=5,
    )
    assert len(calls) == 1
    assert len(rounds) == 2  # round 0 + round 1 exit
    assert rounds[1].early_exit is True


def test_pm_empty_questions_list_treated_as_exit(monkeypatch):
    """A PM who says 'no_further_questions=False' but emits 0 questions
    is functionally the same as the explicit-exit path."""
    monkeypatch.setattr(
        dr, "pm_critique",
        lambda **kw: CritiqueOutput(no_further_questions=False, questions=[]),
    )
    _, rounds = dr.run_dialog_loop(
        run_id="r3", initial_findings=_initial_findings(),
        re_fire={}, max_rounds=5,
    )
    assert rounds[-1].early_exit is True


def test_max_rounds_caps_loop(monkeypatch):
    """When the PM keeps asking, the loop terminates at max_rounds."""

    def always_ask(**kwargs):
        return CritiqueOutput(
            questions=[CritiqueQuestion(
                target_agent="sector",
                question="dig deeper on cohort growth",
                why_it_matters="rating-relevant",
            )],
            no_further_questions=False,
            rationale="more depth needed",
        )

    monkeypatch.setattr(dr, "pm_critique", always_ask)

    refire_calls = {"n": 0}

    def refire(q: str) -> AgentFinding:
        refire_calls["n"] += 1
        return _finding("Sector Analyst", summary=f"refired #{refire_calls['n']}: {q[:30]}")

    _, rounds = dr.run_dialog_loop(
        run_id="r4", initial_findings=_initial_findings(),
        re_fire={"sector": refire},
        max_rounds=2,
    )
    # Round 0 + 2 critique rounds.
    assert len(rounds) == 3
    assert refire_calls["n"] == 2
    # Latest-per-agent finding reflects the last re-fire.
    assert "refired #2" in rounds[-1].findings["sector"].summary


def test_all_refires_failing_breaks_loop(monkeypatch):
    """If every re-fire on a round raises, we stop — burning more LLM
    budget on a stuck dispatcher is wasted."""
    monkeypatch.setattr(
        dr, "pm_critique",
        lambda **kw: CritiqueOutput(
            questions=[CritiqueQuestion(
                target_agent="sector",
                question="dig deeper",
                why_it_matters="x",
            )],
        ),
    )

    def boom(q: str):
        raise RuntimeError("simulated re-fire failure")

    _, rounds = dr.run_dialog_loop(
        run_id="r5", initial_findings=_initial_findings(),
        re_fire={"sector": boom}, max_rounds=5,
    )
    # Round 0 + the single failing round (loop exits before issuing
    # another critique call). The finding from the failing re-fire is
    # absent from the round's `findings`.
    assert len(rounds) == 2
    assert rounds[1].findings == {}


def test_unknown_target_agent_skipped_safely(monkeypatch):
    """The PM may target an agent we don't have a dispatcher for; that
    question is dropped silently rather than raising."""
    monkeypatch.setattr(
        dr, "pm_critique",
        lambda **kw: CritiqueOutput(
            questions=[
                CritiqueQuestion(
                    target_agent="filing",  # not in re_fire map
                    question="dig deeper",
                    why_it_matters="x",
                ),
                CritiqueQuestion(
                    target_agent="sector",
                    question="cohort math?",
                    why_it_matters="x",
                ),
            ],
        ),
    )
    fired: List[str] = []

    def refire(q: str) -> AgentFinding:
        fired.append(q)
        return _finding("Sector Analyst", summary="refired")

    final, rounds = dr.run_dialog_loop(
        run_id="r6", initial_findings=_initial_findings(),
        re_fire={"sector": refire}, max_rounds=1,
    )
    assert fired == ["cohort math?"]
    # The sector finding got refreshed; filing did not.
    assert final["sector"].summary == "refired"


def test_pm_critique_failure_short_circuits():
    """`pm_critique` returning no_further_questions=True is the documented
    fallback when the PM's LLM call raises. The loop trusts that signal."""
    # Default pm_critique with no LLM = `no_further_questions=True`.
    final, rounds = dr.run_dialog_loop(
        run_id="r7", initial_findings=_initial_findings(),
        re_fire={}, max_rounds=3,
    )
    # Round 0 + an early-exit round (or just round 0 with no follow-up
    # critique — either is correct as long as we don't loop).
    assert len(rounds) <= 2
    assert all(not r.findings or r.round == 0 for r in rounds)
