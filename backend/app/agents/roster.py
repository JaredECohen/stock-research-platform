"""The analyst roster — one declarative table behind the memo fan-out (RP-003).

`graph.py` used to name each specialist eight times: a checkpointed
wrapper, a fan-out stanza, a deep-research re-fire closure, a long-form
stanza, the soft-fallback promotion, the compose mapping, plus the intake
module's own copies of the key list and the display names. Adding an
analyst meant editing every one of those, and a `monkeypatch` on
`graph.run_sector_agent` could silently no-op when the wrapper had bound
the runner elsewhere.

Now every consumer loops over `AGENTS`. Adding an analyst is one
`AgentSpec` entry here plus its runner module; `graph.py` does not change.

Contract notes
--------------
* `checkpoint` names are FROZEN. `regen_worker._merged_progress` reads
  them off `MemoRunCheckpoint` rows and `GET /api/stocks/{t}/analyze/status`
  shows them; an orphaned run resumes only if the names match what the
  interrupted run saved. `KNOWN_STEPS` lists every name in use and
  `test_roster` pins it, the way `monitoring.KNOWN_LOOPS` is pinned.
* `run` resolves the runner from this module's globals at call time (the
  lambdas below reference `run_sector_agent` etc. by name), so tests patch
  `roster.run_<x>_agent` and every path — round 0, checkpointed, re-fire —
  sees the patch. `graph.py` no longer imports the runners at all.
* Round 0 calls `spec.run(inputs, None)`; the deep-research loop calls
  `spec.run(inputs, question)`. Same function, so the two can't drift.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..schemas import AgentFinding
from ..services.checkpoint_store import checkpointed
from .comps_agent import run_comps_agent
from .earnings_agent import run_earnings_agent
from .filing_agent import run_filing_agent
from .macro_agent import run_macro_agent
from .risk_agent import run_risk_agent
from .scorecard_context import prompt_block
from .sector_agents import run_sector_agent
from .technical_agent import run_technical_agent
from .valuation_agent import run_valuation_agent

if TYPE_CHECKING:
    from .memo_context import MemoInputs

# (inputs, prior_round_critique) -> finding. `prior_round_critique` is None
# on round 0 and the PM's question on deep-research re-fires.
AgentRunner = Callable[["MemoInputs", str | None], AgentFinding]


@dataclass(frozen=True)
class AgentSpec:
    key: str                          # findings dict key, intake key, re-fire key
    display_name: str                 # llm_call_context agent_name; degraded_agents entry
    checkpoint: str                   # MemoRunCheckpoint step name — FROZEN (see KNOWN_STEPS)
    run: AgentRunner                  # (inputs, prior_round_critique) -> AgentFinding
    needs: tuple[str, ...] = ()       # MemoInputs attribute names the runner reads
    memo_field: str | None = None  # StockMemoOut attribute, or None -> extra_agent_views[key]
    uses_llm_round0: bool = True      # False when round 0 is deterministic by design

    @property
    def long_form_name(self) -> str:
        """The `safe_call` name for this analyst's long-form pass.

        Reproduces the historical "Long-form (Sector)" strings exactly —
        they are what `degraded_agents` shows when enrichment fails.
        """
        return f"Long-form ({self.display_name.replace(' Analyst', '')})"


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        key="sector", display_name="Sector Analyst",
        checkpoint="graph.sector_finding",
        run=lambda i, q: run_sector_agent(i.profile, i.ratios, prior_round_critique=q),
        needs=("profile", "ratios"), memo_field="sector_agent_view",
    ),
    AgentSpec(
        key="earnings", display_name="Earnings Analyst",
        checkpoint="graph.earnings_finding",
        run=lambda i, q: run_earnings_agent(
            i.profile, i.transcript, i.earnings, prior_round_critique=q,
        ),
        needs=("profile", "transcript", "earnings"), memo_field="earnings_agent_view",
    ),
    AgentSpec(
        key="filing", display_name="Filing Analyst",
        checkpoint="graph.filing_finding",
        run=lambda i, q: run_filing_agent(i.profile, i.filings, prior_round_critique=q),
        needs=("profile", "filings"), memo_field="filing_agent_view",
    ),
    AgentSpec(
        key="valuation", display_name="Valuation Analyst",
        checkpoint="graph.valuation_finding",
        # Phase 6: the scorecard read rides as an explicit kwarg (the agent
        # builds its own payload from named ratio keys, so it cannot be
        # smuggled through `ratios`). `prompt_block(None)` is "" and the
        # agent then leaves its payload untouched.
        run=lambda i, q: run_valuation_agent(
            i.profile, i.ratios, i.dcf, prior_round_critique=q,
            scorecard_block=prompt_block(i.scorecard),
        ),
        needs=("profile", "ratios", "dcf", "scorecard"), memo_field="valuation_agent_view",
    ),
    AgentSpec(
        key="comps", display_name="Comps Analyst",
        checkpoint="graph.comps_finding",
        run=lambda i, q: run_comps_agent(i.profile, i.comps, prior_round_critique=q),
        needs=("profile", "comps"), memo_field="comps_agent_view",
        # Round 0 is cohort arithmetic; the LLM only enriches on re-fire.
        uses_llm_round0=False,
    ),
    AgentSpec(
        key="macro", display_name="Macro Analyst",
        checkpoint="graph.macro_finding",
        run=lambda i, q: run_macro_agent(i.profile, i.scenario, prior_round_critique=q),
        needs=("profile", "scenario"), memo_field="macro_sensitivity",
    ),
    AgentSpec(
        key="risk", display_name="Risk Analyst",
        checkpoint="graph.risk_finding",
        # The graph has always handed the risk agent the DCF's summary
        # *string* (it only does `"bear" in x` and str(x) on it).
        run=lambda i, q: run_risk_agent(
            i.profile, i.ratios, (i.dcf.summary if i.dcf else None),
            prior_round_critique=q,
        ),
        needs=("profile", "ratios", "dcf"),
        # No memo view: the risk read feeds scores, recommendations and the
        # bear case rather than rendering as its own card (see NO_MEMO_VIEW).
        memo_field=None,
        # Deterministic at round 0 by design (LLM narrative is enrichment).
        uses_llm_round0=False,
    ),
    AgentSpec(
        key="technical", display_name="Technical Analyst",
        checkpoint="graph.technical_finding",
        run=lambda i, q: run_technical_agent(i.profile, prior_round_critique=q),
        needs=("profile",), memo_field="technical_agent_view",
    ),
)

AGENTS_BY_KEY: dict[str, AgentSpec] = {spec.key: spec for spec in AGENTS}

# Roster keys whose finding is deliberately NOT surfaced on the memo, even
# though `memo_field` is None. The risk analyst predates `extra_agent_views`
# and the memo JSON (persisted per ticker, rendered by the frontend) does
# not carry it as a view; putting it there would change every stored memo
# for no reader benefit. Any future `memo_field=None` analyst that is not
# listed here lands in `extra_agent_views` automatically.
NO_MEMO_VIEW: frozenset = frozenset({"risk"})

# Non-roster steps the graph checkpoints under the same run_id. They stay
# hand-written in graph.py because their return types differ (a dict, a
# DCFResult, a CompsResult, an Optional[CriticReview]).
GATHER_STEPS: tuple[str, ...] = ("graph.fundamentals", "graph.dcf", "graph.comps")
CRITIC_STEP = "graph.critic"

# Every checkpoint name the memo run can write, frozen. Resume, the status
# endpoint and the worker's progress merge all key on these strings.
KNOWN_STEPS: tuple[str, ...] = (
    *GATHER_STEPS,
    *(spec.checkpoint for spec in AGENTS),
    CRITIC_STEP,
)


def _build_checkpointed(spec: AgentSpec) -> Callable[[MemoInputs], AgentFinding]:
    # `run_id` is read by the decorator from `llm_call_context` at call
    # time, so this wrapper checkpoints exactly like the hand-written ones
    # did: cached under (run_id, spec.checkpoint), fall-through without a
    # run_id. Round 0 passes no critique.
    def _round0(inputs: MemoInputs) -> AgentFinding:
        return spec.run(inputs, None)

    _round0.__name__ = f"checkpointed_{spec.key}"
    return checkpointed(spec.checkpoint, return_type=AgentFinding)(_round0)


# Built once at import for the roster; a spec that is not on `AGENTS`
# (a test appending a fake analyst) gets its wrapper built on first use.
_CHECKPOINTED: dict[AgentSpec, Callable[[MemoInputs], AgentFinding]] = {
    spec: _build_checkpointed(spec) for spec in AGENTS
}


def checkpointed_runner(spec: AgentSpec) -> Callable[[MemoInputs], AgentFinding]:
    """The round-0 runner for `spec`, wrapped in the checkpoint store."""
    fn = _CHECKPOINTED.get(spec)
    if fn is None:
        fn = _CHECKPOINTED[spec] = _build_checkpointed(spec)
    return fn
