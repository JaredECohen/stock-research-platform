"""Bull/bear debate budget: worst-case admission with reserve accounting
(slice B8-D3; design §13.3; critique #8; owner item 9: MEMO_MAX_USD $5)."""
from __future__ import annotations

import math
import uuid

from app.agents import debate
from app.agents.debate import CallResult
from app.config import settings
from app.database import SessionLocal
from app.models import LLMCallLog
from app.tests import debate_fakes as F


def _probe(run_id: str) -> dict[str, float]:
    """Worst case of each phase's PAIR for the default scripted run (the
    same run id gives the same prompts), from an unconstrained run."""
    call = F.ScriptedCall()
    assert F.run(call, run_id=run_id, inputs=F.inputs(run_id)).status == "complete"
    return {phase: sum(debate.worst_case_usd(r) for r in call.calls(phase=phase)) for phase in debate.PHASES}


def test_worst_case_prices_what_is_sent():
    req = debate._requests("openings", F.ROUTE, "p" * 35_000, "ACME", "12 months", 16_000)["bull"]
    n_in = math.ceil((len(req.prompt) + len(req.system)) / 3.5)
    # claude-opus-5-5 at $4 in / $20 out; 16,000 output is what is sent.
    assert debate.sent_max_tokens("anthropic", "claude-opus-5-5", 16_000) == 16_000
    assert abs(debate.worst_case_usd(req) - (n_in * 4 + 16_000 * 20) / 1e6) < 1e-6
    # The thinking floor lifts a small request to what the layer really sends.
    assert debate.sent_max_tokens("anthropic", "claude-opus-5-5", 1_000) == settings.llm_thinking_max_tokens_floor


def test_budget_never_exceeds_cap_and_skips_rebuttals_symmetrically(monkeypatch):
    F.enable(monkeypatch)
    worst = _probe("run-budget-1")
    # Research settles at its actual cost (2 x $0.01); the openings are
    # admitted at their worst case; the rebuttals' worst case no longer fits.
    cap = 0.02 + worst["openings"] + 0.001
    assert 0.04 + worst["rebuttals"] > cap
    call = F.ScriptedCall()
    budget = F.budget("run-budget-1", cap_usd=cap)
    record = F.run(call, run_id="run-budget-1", inputs=F.inputs("run-budget-1"), budget=budget)
    assert (record.status, record.rebuttal_status) == ("partial", "skipped_budget")
    assert not call.calls(phase="rebuttals"), "neither side's rebuttal is sent"
    assert record.responses == [] and {c.status for c in record.claims} == {"unanswered"}
    assert budget.spent() <= cap
    skipped = [p for p in record.route["phases"] if p["phase"] == "rebuttals"]
    assert skipped == [{"phase": "rebuttals", "attempt": "primary",
                        "sides": {"bull": "skipped_budget", "bear": "skipped_budget"}}]
    assert record.usage["cap_usd"] == cap and record.usage["usd"] <= cap


def test_budget_reserves_worst_case_for_failed_calls(monkeypatch):
    """Two openings raise (billed or not, they report no usage), then the
    retry: each failed call stays charged at its worst case, and the
    reserved total never exceeds the cap."""
    F.enable(monkeypatch, route=F.ROUTE.model_copy(update={"partner_provider": None, "partner_model": None}))
    pair = _probe("run-reserve-1")["openings"]
    script = F.default_script()
    script[("bull", "openings")] = [RuntimeError("timeout"), F.opening("bull")]
    script[("bear", "openings")] = [RuntimeError("timeout"), F.opening("bear")]
    cap = 0.02 + 1.5 * pair  # the pair fits; a retry pair on top of two reserved failures does not
    budget = F.budget("run-reserve-1", cap_usd=cap)
    call = F.ScriptedCall(script)
    record = F.run(call, run_id="run-reserve-1", inputs=F.inputs("run-reserve-1"), budget=budget)
    assert len(call.calls(phase="openings")) == 2, "the retry is refused: failed calls stay reserved"
    assert (record.status, record.reason) == ("unavailable", "side_failed:openings")
    assert budget.spent() <= cap
    assert abs(budget.spent() - (0.02 + pair)) < 1e-6
    retry = [p for p in record.route["phases"] if p["phase"] == "openings" and p["attempt"] == "retry"]
    assert retry and set(retry[0]["sides"].values()) == {"skipped_budget"}


def test_success_replaces_reservation_with_actual_cost():
    b = F.budget("run-settle-1", cap_usd=1.0)
    t_ok, t_fail, t_zero = b.reserve(0.3), b.reserve(0.3), b.reserve(0.3)
    req = debate._requests("openings", F.ROUTE, "p", "ACME", "12 months", 100)["bull"]
    b.settle(t_ok, CallResult({"ok": True}, F.usage_for(req, cost=0.05)))
    b.settle(t_fail, CallResult(None, F.usage_for(req, cost=0.05)))       # truncated: billed, unusable
    b.settle(t_zero, CallResult({"ok": True}, F.usage_for(req, tokens=(0, 0), cost=0.0)))  # no usage row
    assert abs(b.spent() - (0.05 + 0.3 + 0.3)) < 1e-9
    assert not b.admit([0.36]) and b.admit([0.35])


def test_max_calls_enforced(monkeypatch):
    F.enable(monkeypatch)
    budget = F.budget("run-calls-1", max_calls=4)
    call = F.ScriptedCall()
    record = F.run(call, run_id="run-calls-1", inputs=F.inputs("run-calls-1"), budget=budget)
    assert (record.status, record.rebuttal_status) == ("partial", "skipped_budget")
    assert len(call.requests) == 4 and budget.calls() == 4


def _row(run_id: str, agent: str, tokens_out: int) -> LLMCallLog:
    return LLMCallLog(run_id=run_id, agent_name=agent, provider="anthropic", model="claude-opus-5-5",
                      route="strong", tokens_in=0, tokens_out=tokens_out, duration_ms=1, success=True)


def test_memo_admission_reads_cost_per_run(monkeypatch):
    """Spend already on the run (every agent) + the whole debate cap must
    fit MEMO_MAX_USD, read from `llm_call_logs` so both processes agree."""
    F.enable(monkeypatch)
    run_id = f"admit-{uuid.uuid4().hex[:10]}"
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        db.add(_row(run_id, "PM Synthesis", 180_000))   # $3.60 at $20/MTok out
        db.add(_row(run_id, "Bull Advocate", 5_000))    # $0.10, advocate spend of an earlier attempt
        db.commit()
    budget = debate.DebateBudget(run_id, cap_usd=1.50, memo_cap_usd=5.00)
    assert abs(budget.prior_usd - 0.10) < 1e-9 and budget.prior_calls == 1, "prior = advocates only"
    assert settings.memo_max_usd == 5.00
    call = F.ScriptedCall()
    record = F.run(call, run_id=run_id, inputs=F.inputs(run_id), budget=budget)
    assert (record.status, record.reason) == ("unavailable", "budget")
    assert call.requests == []
    assert debate.DebateBudget(run_id, cap_usd=1.20, memo_cap_usd=5.00).memo_admit()


def test_resume_counts_prior_debate_spend_once_in_memo_admission(monkeypatch):
    """A resume must admit what the fresh run admitted: the earlier
    attempt's advocate spend is inside the debate cap, so it is not also
    added on top of the run total."""
    F.enable(monkeypatch)
    run_id = f"admit-resume-{uuid.uuid4().hex[:10]}"
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        db.add(_row(run_id, "PM Synthesis", 150_000))   # analysts etc.: $3.00
        db.commit()
    assert debate.DebateBudget(run_id, cap_usd=1.50, memo_cap_usd=5.00).memo_admit(), "fresh run admitted"
    with SessionLocal() as db:
        db.add(_row(run_id, "Bull Advocate", 22_500))   # $0.45 each side before the crash
        db.add(_row(run_id, "Bear Advocate", 22_500))
        db.commit()
    resumed = debate.DebateBudget(run_id, cap_usd=1.50, memo_cap_usd=5.00)
    assert abs(resumed.prior_usd - 0.90) < 1e-9
    assert resumed.memo_admit(), "3.90 total - 0.90 prior debate + 1.50 cap = 4.50 <= 5.00"
    # And the cap still binds on the resumed debate's own spend.
    assert resumed.admit([0.60]) and not resumed.admit([0.61])
