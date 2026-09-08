"""RP-003 — the analyst roster is the single declarative source of truth.

What these pin:
- the roster's order and uniqueness (findings dict order, compose order,
  checkpoint names all derive from it);
- every `needs` name is a real `MemoInputs` attribute and every
  `memo_field` is a real `StockMemoOut` field, so a spec cannot ask for
  an input that is never gathered or land on a field that does not exist;
- `KNOWN_STEPS` matches the checkpoint names actually in use — the way
  `monitoring.KNOWN_LOOPS` is pinned against `register_all` — because
  resume and `/analyze/status` key on those strings;
- adding an analyst really is one `AgentSpec` entry: a spec appended
  in-test lands in `extra_agent_views` on a full run, and in
  `degraded_agents` when its runner raises, with no graph.py change.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

from app.agents import graph, intake, roster
from app.agents.memo_context import MemoInputs
from app.agents.roster import AGENTS, KNOWN_STEPS, AgentSpec
from app.schemas import StockMemoOut
from app.tests.factories import make_finding, make_inputs

_EXPECTED_ORDER = (
    "sector", "earnings", "filing", "valuation", "comps", "macro", "risk", "technical",
)


def test_roster_order_is_the_historical_fan_out_order():
    assert tuple(spec.key for spec in AGENTS) == _EXPECTED_ORDER


def test_keys_checkpoints_and_display_names_are_unique():
    for attr in ("key", "checkpoint", "display_name"):
        values = [getattr(spec, attr) for spec in AGENTS]
        assert len(values) == len(set(values)), f"duplicate {attr}: {values}"


def test_checkpoint_names_follow_the_frozen_pattern():
    for spec in AGENTS:
        assert spec.checkpoint == f"graph.{spec.key}_finding"


def test_every_need_is_a_memo_inputs_attribute():
    fields = {f.name for f in dataclasses.fields(MemoInputs)}
    for spec in AGENTS:
        assert spec.needs, f"{spec.key} declares no inputs"
        missing = set(spec.needs) - fields
        assert not missing, f"{spec.key} needs {sorted(missing)} which MemoInputs lacks"


def test_every_memo_field_exists_on_stock_memo_out():
    for spec in AGENTS:
        if spec.memo_field is not None:
            assert spec.memo_field in StockMemoOut.model_fields, spec.memo_field


def test_only_risk_is_unsurfaced():
    """Risk has no memo view by design; nothing else may hide behind
    `NO_MEMO_VIEW` (a new analyst either names a field or rides in
    `extra_agent_views`)."""
    assert roster.NO_MEMO_VIEW == frozenset({"risk"})
    assert roster.AGENTS_BY_KEY["risk"].memo_field is None
    for spec in AGENTS:
        if spec.memo_field is None:
            assert spec.key in roster.NO_MEMO_VIEW


def test_deterministic_round0_analysts_are_comps_and_risk():
    assert {s.key for s in AGENTS if not s.uses_llm_round0} == {"comps", "risk"}


def test_long_form_names_match_the_historical_strings():
    assert [s.long_form_name for s in AGENTS] == [
        "Long-form (Sector)", "Long-form (Earnings)", "Long-form (Filing)",
        "Long-form (Valuation)", "Long-form (Comps)", "Long-form (Macro)",
        "Long-form (Risk)", "Long-form (Technical)",
    ]


def test_specs_are_frozen():
    spec = AGENTS[0]
    try:
        spec.key = "other"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("AgentSpec must be frozen")


def test_intake_derives_its_lists_from_the_roster():
    assert intake.ALL_SPECIALISTS == [s.key for s in AGENTS]
    for spec in AGENTS:
        assert intake.stub_finding(spec.key, "")["agent"] == spec.display_name


# ---------------------------------------------------------------------------
# KNOWN_STEPS — every checkpoint name in use, and nothing else
# ---------------------------------------------------------------------------

def test_known_steps_matches_the_checkpoint_names_in_use():
    """`KNOWN_STEPS` is hand-assembled from the roster plus the four
    non-roster wrappers in graph.py. If either side drifts, an orphaned
    run resumes by recomputing (or the status endpoint stops recognising
    a step), so the two are compared the way KNOWN_LOOPS is."""
    in_graph = set(re.findall(r'checkpointed\(\s*"([^"]+)"', inspect.getsource(graph)))
    in_roster = {spec.checkpoint for spec in AGENTS}
    in_use = in_graph | in_roster
    assert in_use == set(KNOWN_STEPS), (
        "KNOWN_STEPS is out of sync with the wrappers:\n"
        f"  in use but not listed: {sorted(in_use - set(KNOWN_STEPS))}\n"
        f"  listed but not in use: {sorted(set(KNOWN_STEPS) - in_use)}"
    )
    assert len(KNOWN_STEPS) == len(set(KNOWN_STEPS))
    assert set(roster.GATHER_STEPS) <= in_graph and roster.CRITIC_STEP in in_graph


def test_known_steps_is_the_frozen_contract():
    assert isinstance(KNOWN_STEPS, tuple)
    assert KNOWN_STEPS == (
        "graph.fundamentals", "graph.dcf", "graph.comps",
        "graph.sector_finding", "graph.earnings_finding", "graph.filing_finding",
        "graph.valuation_finding", "graph.comps_finding", "graph.macro_finding",
        "graph.risk_finding", "graph.technical_finding",
        "graph.critic",
    )


# ---------------------------------------------------------------------------
# The runner contract
# ---------------------------------------------------------------------------

def test_run_resolves_the_runner_at_call_time(monkeypatch):
    """`spec.run` must look the runner up on `roster` when called, so a
    monkeypatch on `roster.run_<x>_agent` reaches every path (round 0,
    checkpointed, deep-research re-fire). This is the D5 guarantee."""
    seen = []

    def fake(profile, ratios, *, prior_round_critique=None):
        seen.append(prior_round_critique)
        return make_finding("Sector Analyst")

    monkeypatch.setattr(roster, "run_sector_agent", fake)
    inputs = make_inputs("X")
    spec = roster.AGENTS_BY_KEY["sector"]
    spec.run(inputs, None)
    spec.run(inputs, "why is growth durable?")
    assert seen == [None, "why is growth durable?"]


def test_checkpointed_runner_is_built_once_per_spec():
    spec = roster.AGENTS_BY_KEY["sector"]
    assert roster.checkpointed_runner(spec) is roster.checkpointed_runner(spec)


# ---------------------------------------------------------------------------
# One declarative entry is enough for a new analyst
# ---------------------------------------------------------------------------

def _fake_spec(run) -> AgentSpec:
    return AgentSpec(
        key="fake", display_name="Fake Analyst", checkpoint="graph.fake_finding",
        run=run, needs=("profile",), memo_field=None,
    )


def test_appended_spec_lands_in_extra_agent_views(monkeypatch):
    spec = _fake_spec(lambda i, q: make_finding("Fake Analyst", headline="fake view"))
    monkeypatch.setattr(roster, "AGENTS", AGENTS + (spec,))
    memo = graph.run_stock_memo("MSFT")
    assert set(memo.extra_agent_views) == {"fake"}
    assert memo.extra_agent_views["fake"].headline == "fake view"
    assert "Fake Analyst" not in memo.degraded_agents
    # The existing views are untouched by the addition.
    assert memo.sector_agent_view.confidence > 0.0


def test_appended_spec_failure_is_isolated_and_on_the_banner(monkeypatch):
    def boom(i, q):
        raise RuntimeError("fake analyst exploded")

    monkeypatch.setattr(roster, "AGENTS", AGENTS + (_fake_spec(boom),))
    memo = graph.run_stock_memo("MSFT")
    assert "Fake Analyst" in memo.degraded_agents
    assert memo.extra_agent_views["fake"].confidence == 0.0
    assert memo.extra_agent_views["fake"].data.get("degraded") is True
    # Nothing else degraded because of it.
    assert memo.sector_agent_view.confidence > 0.0
    assert [e["agent"] for e in memo.degradation_events] == memo.degraded_agents
