"""Bull/bear debate symmetry guarantees S1-S8 (design §6) and the lesson
tests L2-L4 (slice B8-D3).

The debate is meant to remove a structural bull tilt, so every place the
two sides could be treated differently has a test: the prompts, the inputs,
the route, the grader, the dispute rule, the presentation order and the
PM block. A test here failing means one side got something the other did
not."""
from __future__ import annotations

import hashlib
import inspect
import re

import pytest

from app.agents import debate
from app.agents import debate_prompts as P
from app.agents.llm_attribution import ACTIONS
from app.config import Settings, settings
from app.schemas.agents import DebateClaim, DebateEvidence, DebateRecord, DebateResponse
from app.tests import debate_fakes as F

HORIZON = "12 months"


def _pair(phase: str, prefix: str = "SHARED PREFIX\n") -> dict[str, debate.DebateRequest]:
    return debate._requests(phase, F.ROUTE, prefix, "ACME", HORIZON, 1000)


# --- S1: mirror-image prompts ----------------------------------------------

@pytest.mark.parametrize("phase", debate.PHASES)
def test_prompts_are_mirror_images(phase):
    reqs = _pair(phase)
    bull, bear = reqs["bull"], reqs["bear"]
    assert bull.prefix == bear.prefix, "the shared prefix must be byte-identical (S2)"
    assert bull.system == bear.system == P.DEBATE_SYSTEM
    assert bull.suffix != bear.suffix
    assert debate.swap_sides(bull.suffix) == bear.suffix
    assert debate.swap_sides(bear.suffix) == bull.suffix


def test_system_prompt_has_no_side_content():
    assert not re.search(r"\b(BULL|BEAR|outperform|underperform)\b", P.DEBATE_SYSTEM)


def test_goal_strings_same_length_and_word_count():
    bull, bear = P.goal("bull", HORIZON), P.goal("bear", HORIZON)
    assert len(bull.split()) == len(bear.split())
    # The owner-adopted L3 verbs ("outperform"/"underperform") are the ONLY
    # difference, so the lengths differ by exactly their difference.
    assert bull.replace("outperform", "VERB") == bear.replace("underperform", "VERB")
    assert len(bear) - len(bull) == len("underperform") - len("outperform")


def test_goal_names_benchmark_and_horizon():
    for side in debate.SIDE_NAMES:
        g = P.goal(side, settings.debate_horizon)
        assert "S&P 500" in g and "total-return" in g and settings.debate_horizon in g
        for phase in ("research", "openings"):
            assert g in _pair(phase)[side].suffix
    assert settings.debate_horizon == "12 months"
    assert "not a bull point" in P.CASE_FILE_HEADER and "not a bear point" in P.CASE_FILE_HEADER


# --- S2/S3: identical inputs, same route and budget --------------------------

def test_rebuttal_prefix_identical_across_sides(monkeypatch):
    F.enable(monkeypatch)
    call = F.ScriptedCall()
    record = F.run(call)
    assert record.status == "complete"
    for phase in debate.PHASES:
        bull, bear = call.calls("bull", phase), call.calls("bear", phase)
        assert len(bull) == len(bear) == 1
        assert bull[0].prefix == bear[0].prefix
    rebut = call.calls("bull", "rebuttals")[0].prefix
    assert "BULL-1" in rebut and "BEAR-1" in rebut, "both openings are in the rebuttal input"
    assert rebut.startswith(call.calls("bull", "openings")[0].prefix)


def test_same_route_and_max_tokens_both_sides(monkeypatch):
    F.enable(monkeypatch)
    call = F.ScriptedCall()
    F.run(call)
    for phase in debate.PHASES:
        bull, bear = call.calls("bull", phase)[0], call.calls("bear", phase)[0]
        for attr in ("provider", "model", "effort", "max_tokens", "system", "ticker"):
            assert getattr(bull, attr) == getattr(bear, attr), (phase, attr)
    assert call.calls("bull", "research")[0].effort == "low"
    assert call.calls("bull", "openings")[0].effort == "high"


def test_no_side_specific_settings():
    names = set(Settings.model_fields)
    assert not {n for n in names if "bull" in n or "bear" in n}, "a per-side knob could split the models"
    for phase in debate.PHASES:
        bull, bear = ACTIONS[debate.action_for("bull", phase)], ACTIONS[debate.action_for("bear", phase)]
        assert (bull.tier, bull.effort_key, bull.kind) == (bear.tier, bear.effort_key, bear.kind)


# --- S4: order independence --------------------------------------------------

def _comparable(record: DebateRecord) -> dict:
    return record.model_dump(exclude={"presentation_order", "disputes", "unanswered", "deterministic_checks"})


def test_record_identical_bull_first_bear_first_parallel(monkeypatch):
    ids = F.run_ids_by_parity()
    records = {}
    for parallel in (False, True):
        F.enable(monkeypatch, parallel=parallel)
        for order, rid in ids.items():
            records[(parallel, order)] = F.run(F.ScriptedCall(), run_id=rid, inputs=F.inputs(rid))
    assert records[(False, "bull_first")].presentation_order == "bull_first"
    assert records[(False, "bear_first")].presentation_order == "bear_first"
    base = _comparable(records[(False, "bull_first")])
    for key, rec in records.items():
        assert rec.status == "complete", key
        assert _comparable(rec) == base, key
        assert set(rec.disputes) == set(records[(False, "bull_first")].disputes)
    # Same run id: sequential and parallel produce the very same record.
    for order in ids:
        assert records[(False, order)] == records[(True, order)]


# --- S6: side-blind grading --------------------------------------------------

def test_grade_side_blind():
    assert "side" not in inspect.signature(debate.grade_claim).parameters
    pool = [DebateEvidence(id="E01", kind="filing", ref="chunk:1", excerpt="Demand is strong.")]
    raw = F.opening("bull")
    raw["claims"][1]["evidence"] = ["E09"]  # unknown: drops to analyst_only
    raw["claims"][2]["quote"] = {"evidence": "E01", "text": "Demand is weak."}  # fails verification
    grades = {}
    for side in debate.SIDE_NAMES:
        _, claims = debate.parse_opening(raw, side, max_claims=5, pool=pool, registry=None,
                                         usable_analysts=["earnings"])
        grades[side] = [(c.grade, c.dropped, c.evidence, c.quote) for c in claims]
    assert grades["bull"] == grades["bear"]
    assert [g[0] for g in grades["bull"]] == ["sourced", "analyst_only", "partially_sourced"]


# --- S7: symmetric dispute selection ------------------------------------------

def _claim(side: str, n: int, **kw) -> DebateClaim:
    base = dict(id=f"{side.upper()}-{n}", side=side, claim=f"claim {n}", materiality="high",
                category="growth", grade="sourced", status="contested")
    base.update(kw)
    return DebateClaim(**base)


def test_decisive_dispute_rule_symmetric():
    specs = [dict(n=1), dict(n=2, grade="partially_sourced"), dict(n=3, status="partial"),
             dict(n=4, materiality="medium", category="valuation"), dict(n=5, materiality="low")]
    claims = [_claim(side, **s) for side in debate.SIDE_NAMES for s in specs]
    d_bull = debate.decisive_disputes(claims, "bull_first")
    d_bear = debate.decisive_disputes(claims, "bear_first")
    assert d_bull == ["BULL-1", "BEAR-1", "BULL-3", "BEAR-3", "BULL-2", "BEAR-2"]
    assert d_bear == [debate.swap_sides(x) for x in d_bull]
    # The mirrored claim set gives the mirrored selection.
    mirrored = [c.model_copy(update={"id": debate.swap_sides(c.id), "side": "bear" if c.side == "bull" else "bull"})
                for c in claims]
    assert debate.decisive_disputes(mirrored, "bear_first") == d_bear


# --- S8: neutral presentation -------------------------------------------------

def test_presentation_order_is_run_hash_parity():
    ids = F.run_ids_by_parity()
    for order, rid in ids.items():
        parity = int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) % 2
        assert order == ("bull_first" if parity == 0 else "bear_first")
        assert debate.presentation_order(rid) == order
    with pytest.raises(ValueError):
        debate.presentation_order("")


def _record(order: str = "bull_first", *, text: str = "x", argument: str = "y", n: int = 3) -> DebateRecord:
    claims, responses = [], []
    for side in debate.SIDE_NAMES:
        opp = "bear" if side == "bull" else "bull"
        for i in range(1, n + 1):
            claims.append(DebateClaim(id=f"{side.upper()}-{i}", side=side, pillar="Demand",
                                      claim=f"{text} {i}", evidence=["E01"], grade="sourced",
                                      materiality="high", status="contested"))
            responses.append(DebateResponse(side=opp, target=f"{side.upper()}-{i}", stance="rebut",
                                            argument=f"{argument} {i}", evidence=["E01"], grade="sourced"))
    rec = DebateRecord(status="complete", presentation_order=order, claims=claims, responses=responses,
                       headlines={"bull": "Up", "bear": "Down"}, cruxes={"bull": "Demand", "bear": "Demand"},
                       evidence=[DebateEvidence(id="E01", kind="filing", ref="chunk:1", excerpt="z" * 600)])
    rec.disputes = debate.decisive_disputes(rec.claims, order)
    return rec


def _mirror(record: DebateRecord) -> DebateRecord:
    swap = {"bull": "bear", "bear": "bull"}
    return record.model_copy(update={
        "presentation_order": "bear_first" if record.presentation_order == "bull_first" else "bull_first",
        "claims": [c.model_copy(update={"id": debate.swap_sides(c.id), "side": swap[c.side]})
                   for c in record.claims],
        "responses": [r.model_copy(update={"target": debate.swap_sides(r.target), "side": swap[r.side]})
                      for r in record.responses],
        "headlines": {swap[k]: v for k, v in record.headlines.items()},
        "cruxes": {swap[k]: v for k, v in record.cruxes.items()},
        "disputes": [debate.swap_sides(d) for d in record.disputes],
    })


def test_debate_block_side_swap_symmetric():
    rec = _record()
    block = debate.render_pm_block(rec, max_chars=14_000)
    assert block
    assert debate.render_pm_block(_mirror(rec), max_chars=14_000) == debate.swap_sides(block)


def test_pm_block_includes_rebuttal_arguments_symmetric():
    rec = _record(argument="because the backlog")
    block = debate.render_pm_block(rec, max_chars=14_000)
    lines = block.splitlines()
    for side, opp in (("BULL", "Bear"), ("BEAR", "Bull")):
        for i in (1, 2, 3):
            at = next(n for n, ln in enumerate(lines) if ln.strip().startswith(f"{side}-{i} "))
            answer = lines[at + 1]
            assert answer.strip().startswith(f"{opp} rebut (sourced; evidence E01): because the backlog {i}"), answer


def test_pm_block_trim_symmetric():
    long_rec = _record(text="t" * 380, argument="a" * 380, n=5)
    full = debate.render_pm_block(long_rec, max_chars=100_000)
    cap = 3_000
    block = debate.render_pm_block(long_rec, max_chars=cap)
    assert len(block) <= cap < len(full)
    for side in ("BULL", "BEAR"):
        for i in range(1, 6):
            assert re.search(rf"^\s+{side}-{i} ", block, re.M), "rows are never dropped"
    # Both sides trimmed identically: the mirror image survives the trim.
    assert debate.render_pm_block(_mirror(long_rec), max_chars=cap) == debate.swap_sides(block)
    # Excerpts go before claim text.
    assert '"zzz' not in block


def test_pm_block_empty_unless_debated():
    for status in ("not_run", "unavailable"):
        assert debate.render_pm_block(DebateRecord(status=status)) == ""
    assert debate.render_pm_block(None) == ""


def test_block_states_disagreement_not_evidence():
    block = debate.render_pm_block(_record(), max_chars=14_000)
    assert "their disagreement is not evidence" in block
    assert "the debate supports Neutral" in block
    assert "Rulings do not set rating_label" in block


# --- L4: openings argue their own side only ------------------------------------

def test_opening_has_no_opponent_text(monkeypatch):
    F.enable(monkeypatch)
    script = F.default_script()
    script[("bear", "research")] = [{"queries": [
        {"corpus": "news", "query": "zebra unicorn channel check", "why": "BEAR SECRET intent"}]}]
    call = F.ScriptedCall(script)
    record = F.run(call)
    assert record.status == "complete"
    for side in debate.SIDE_NAMES:
        req = call.calls(side, "openings")[0]
        assert "The other advocate's case is not available to you" in req.suffix
        assert "zebra unicorn" not in req.prompt and "SECRET" not in req.prompt
        assert not re.search(r"\b(BULL|BEAR)-\d", req.prompt), "no claim ids exist before the openings"


# --- S8 on the advocates' inputs (fixer regression) ------------------------------

def test_rebuttal_input_and_sector_sketches_follow_presentation_order(monkeypatch):
    """The side shown first to the rebuttals, and the sector sketch shown
    first in the case file, is the run's presentation order, for BOTH
    parities (a bull-first constant would still pass every record test)."""
    F.enable(monkeypatch)
    for order, rid in F.run_ids_by_parity().items():
        call = F.ScriptedCall()
        record = F.run(call, run_id=rid, inputs=F.inputs(rid))
        assert record.presentation_order == order
        first, second = debate.ordered(order)
        rebut = call.calls("bull", "rebuttals")[0].prefix
        assert rebut == call.calls("bear", "rebuttals")[0].prefix
        tail = rebut.split("<<<OPENINGS (", 1)[1]
        heads = [P.SIDES[s]["Side"] for s in (first, second)]
        assert tail.index(f"### {heads[0]} opening") < tail.index(f"### {heads[1]} opening"), order
        assert f"shown {first} first this run" in tail
        research = call.calls("bull", "research")[0].prefix
        assert research.index(f"Sector {first} sketch") < research.index(f"Sector {second} sketch"), order


def test_pm_block_default_cap_is_the_setting(monkeypatch):
    """D6 calls render_pm_block without max_chars: the default must be
    DEBATE_PM_BLOCK_MAX_CHARS (14k), not unbounded."""
    rec = _record(text="t" * 380, argument="a" * 380, n=5)
    assert len(debate.render_pm_block(rec, max_chars=100_000)) > 3_000
    assert settings.debate_pm_block_max_chars == 14_000
    assert len(debate.render_pm_block(rec)) <= 14_000
    monkeypatch.setattr(settings, "debate_pm_block_max_chars", 3_000)
    block = debate.render_pm_block(rec)
    assert len(block) <= 3_000
    assert block == debate.render_pm_block(rec, max_chars=3_000)
