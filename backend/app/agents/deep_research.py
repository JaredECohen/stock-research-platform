"""Wave 9 — iterative PM↔specialist deep-research loop.

After the parallel fan-out (round 0) lands, the PM critique step
inspects the findings, names what's underbaked, and emits 0-3
follow-up questions tagged with which specialist should answer. Each
targeted specialist re-fires its runner with the question prepended
to its prompt. Loop until the PM declares "no further questions" OR
the round/question budget is exhausted.

Design points (locked in `docs/DEEP_RESEARCH_DESIGN.md`):
- Round 0 is the existing fan-out — Wave 9 is strictly additive.
- Re-fired agents make REAL LLM calls (no PM imagining their answers).
- PM acts as senior analyst (asking dig-deeper questions), not adversarial
  critic. The Risk Committee critic is unchanged.
- Skipped on `incremental_patch` and backtest paths.
- Per-round LLM calls show up in `LLMCallLog` tagged
  `agent_name="PM Critique"` (round 0 critique) /
  `agent_name="Sector Analyst (round N)"` etc., so cost is observable
  in the existing admin dashboard.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import settings
from ..schemas import (
    AgentFinding,
    CritiqueOutput,
    CritiqueQuestion,
    RoundFindings,
    StockMemoOut,
)
from . import llm
from .llm import llm_call_context

log = logging.getLogger(__name__)


# Map `target_agent` (PM critique vocabulary) → the specialist runner that
# can answer the question. The runners take whatever inputs they need from
# the closure built in `run_dialog_loop`; the dispatch indirection here is
# what lets us keep specialist signatures unchanged.
AgentDispatcher = Callable[[str], AgentFinding]


_PM_CRITIQUE_SYSTEM = (
    "You are MarketMosaic's senior PM running a diligence dialog. You "
    "have read the round-N findings from your specialists below. Your "
    "job is to identify what's UNDERBAKED — places where you'd push back "
    "on a junior analyst's first read — and emit 0-3 follow-up questions "
    "tagged with which specialist should answer. ONE question per "
    "specialist per round."
)


_PM_CRITIQUE_PROMPT_TEMPLATE = (
    "Round {round_num} findings (one block per specialist):\n\n"
    "{findings_block}\n\n"
    "Prior rounds in this dialog:\n{prior_rounds_block}\n\n"
    "Decide: are there any specific dig-deeper questions that would "
    "MATERIALLY change the rating, the confidence, or the key risks?\n\n"
    "If yes, emit up to {max_questions} questions targeting one of: "
    "sector, earnings, filing, valuation, comps, macro, risk, technical. "
    "Each question must be specific (NOT 'tell me more about X' — but "
    "'why does the cohort op margin show compression while the target's "
    "is expanding — what's the cohort outlier driving the median?'). "
    "Each question should also include a one-sentence "
    "`why_it_matters` so reviewers see what your follow-up would change.\n\n"
    "If no questions are warranted, set `no_further_questions: true` and "
    "explain why in `rationale` (one sentence). Do NOT fabricate questions "
    "to fill the budget — empty is the correct answer when the round-N "
    "findings already triangulate.\n\n"
    "Return JSON with this shape:\n"
    "{{\n"
    '  "questions": [\n'
    '    {{ "target_agent": "sector|earnings|...", "question": "...", "why_it_matters": "..." }}\n'
    "  ],\n"
    '  "no_further_questions": false,\n'
    '  "rationale": "..."\n'
    "}}"
)


def _format_finding_for_critique(name: str, f: AgentFinding) -> str:
    bullets = "\n".join(f"  - {p}" for p in (f.key_points or [])[:5])
    return (
        f"### {name}\n"
        f"  Headline: {f.headline}\n"
        f"  Summary: {f.summary[:600]}\n"
        f"{bullets}"
    )


def _format_prior_rounds(rounds: List[RoundFindings]) -> str:
    if not rounds or all(r.round == 0 for r in rounds):
        return "(no prior critique rounds — this is round 1)"
    lines: List[str] = []
    for r in rounds:
        if r.round == 0:
            continue
        for q in r.pm_questions:
            lines.append(f"- Round {r.round}: PM asked {q.target_agent} — {q.question}")
    return "\n".join(lines) if lines else "(no prior critique rounds)"


def pm_critique(
    *, round_num: int, current_findings: Dict[str, AgentFinding],
    rounds_so_far: List[RoundFindings], run_id: str,
) -> CritiqueOutput:
    """Single PM critique step. Inspects current findings + dialog
    history, returns a structured `CritiqueOutput` with 0-N questions.

    On LLM failure (no key, malformed JSON, etc.) returns
    `no_further_questions=True` so the loop exits cleanly rather than
    spinning. The empty-question path is the safe default.
    """
    findings_block = "\n\n".join(
        _format_finding_for_critique(k, v)
        for k, v in current_findings.items()
    )
    prompt = _PM_CRITIQUE_PROMPT_TEMPLATE.format(
        round_num=round_num,
        findings_block=findings_block[:5000],
        prior_rounds_block=_format_prior_rounds(rounds_so_far),
        max_questions=settings.deep_research_max_questions_per_round,
    )
    # Use whatever provider is active — forcing OpenAI here meant the
    # dialog hard-failed on deployments configured with only an
    # Anthropic key, surfacing as "LLM call failed" in the UI.
    with llm_call_context(agent_name="PM Critique", run_id=run_id, route="strong"):
        try:
            out = llm.chat_json(
                prompt, system=_PM_CRITIQUE_SYSTEM, route="strong",
                max_tokens=1200,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("PM critique LLM call failed: %s", exc)
            out = None
    if not isinstance(out, dict):
        return CritiqueOutput(
            no_further_questions=True,
            rationale=(
                "PM critique unavailable (LLM call failed) — "
                "round-0 findings used as-is."
            ),
        )

    questions: List[CritiqueQuestion] = []
    for raw in (out.get("questions") or [])[: settings.deep_research_max_questions_per_round]:
        if not isinstance(raw, dict):
            continue
        target = raw.get("target_agent")
        question = (raw.get("question") or "").strip()
        if not question or target not in {
            "sector", "earnings", "valuation", "comps",
            "risk", "filing", "macro", "technical",
        }:
            continue
        questions.append(CritiqueQuestion(
            target_agent=target,
            question=question[:600],
            why_it_matters=str(raw.get("why_it_matters") or "")[:240],
        ))
    return CritiqueOutput(
        questions=questions,
        no_further_questions=bool(out.get("no_further_questions")),
        rationale=str(out.get("rationale") or "")[:240],
    )


def run_dialog_loop(
    *, run_id: str,
    initial_findings: Dict[str, AgentFinding],
    re_fire: Dict[str, AgentDispatcher],
    max_rounds: Optional[int] = None,
) -> Tuple[Dict[str, AgentFinding], List[RoundFindings]]:
    """Run the PM↔specialist dialog. Returns the final-round findings
    + the full round_findings list for persistence.

    `re_fire` is a per-agent-name dispatcher: given a question string,
    it returns the specialist's NEW finding (re-fired with the question
    as prompt context). The graph builds these closures so each
    specialist gets the right inputs (profile / ratios / dcf / etc.)
    without leaking through the loop's signature.

    Loop exits when ANY of:
    - PM returns `no_further_questions=True`.
    - PM returns 0 questions (functionally the same).
    - Round count hits `max_rounds`.
    - A round's re-fires all fail (we don't burn budget on a stuck loop).
    """
    if max_rounds is None:
        max_rounds = settings.deep_research_max_rounds

    # Round 0 — the existing parallel fan-out. Persist it as round 0 with
    # no PM questions so the audit log is complete.
    rounds: List[RoundFindings] = [
        RoundFindings(round=0, pm_questions=[], findings=dict(initial_findings)),
    ]
    current = dict(initial_findings)

    for r in range(1, max_rounds + 1):
        critique = pm_critique(
            round_num=r - 1, current_findings=current,
            rounds_so_far=rounds, run_id=run_id,
        )
        if critique.no_further_questions or not critique.questions:
            # Persist the no-questions exit so reviewers see why the loop
            # ended. early_exit=True signals "PM was satisfied", not "we
            # ran out of budget".
            rounds.append(RoundFindings(
                round=r, pm_questions=critique.questions,
                findings={}, early_exit=True,
                pm_rationale=critique.rationale,
            ))
            break

        # Re-fire each targeted specialist. Skip questions for agents we
        # don't have a re-fire dispatcher for (defensive).
        new_findings: Dict[str, AgentFinding] = {}
        any_success = False
        for q in critique.questions:
            disp = re_fire.get(q.target_agent)
            if disp is None:
                continue
            try:
                with llm_call_context(
                    agent_name=f"{q.target_agent.title()} (round {r})",
                    run_id=run_id,
                ):
                    new_findings[q.target_agent] = disp(q.question)
                any_success = True
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "Deep-research re-fire failed for %s on round %d: %s",
                    q.target_agent, r, exc,
                )

        rounds.append(RoundFindings(
            round=r, pm_questions=critique.questions,
            findings=new_findings, early_exit=False,
            pm_rationale=critique.rationale,
        ))

        # Update `current` with the latest findings — the next round's
        # critique reads the freshest version of each agent's view.
        for name, finding in new_findings.items():
            current[name] = finding

        if not any_success:
            break

    return current, rounds
