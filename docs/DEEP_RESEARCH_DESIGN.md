# Wave 9 — Deep Research (PM ↔ Sector iterative loop)

Today every memo runs once: 8 specialists fan out in parallel → PM
synthesizes once. Memos are fast (~3-15s) but shallow on questions
that need follow-up reasoning (e.g., "why does the cohort margin look
that way?", "what's the Y3 catalyst?"). Wave 9 introduces an iterative
PM↔sector dialog so the platform can do real diligence, not just
parallel data collection.

## 1. Problem statement

Concrete examples of what single-pass synthesis misses today:

- Sector says "growth in upper-half quartile" but doesn't say *which
  cohort outliers* are pushing the median or whether NVDA's growth is
  durable — PM has no follow-up channel.
- Earnings analyst flags "constructive tone" but doesn't unpack what
  segments management actually emphasized — PM can't ask for color.
- Risk analyst flags valuation stretch — but PM never asks the
  valuation analyst to *map specifically what would justify* the
  premium.

A senior PM would push back on every one of these. Today our PM
accepts the first read. Wave 9 fixes that.

## 2. Design — the iteration loop

```
sector_finding (round 0)  ┐
                          │
    ┌─────────────────────▼──────────────────────┐
    │  PM critique step                           │
    │  Reads: round-N findings + current memo     │
    │  Outputs: 0-3 follow-up questions tagged    │
    │  with which agent should answer (sector,    │
    │  earnings, valuation, comps, risk).         │
    │  Or: "no further questions" → exit loop     │
    └─────────────────────┬──────────────────────┘
                          │ (questions)
                          ▼
       ┌────────────┴───────┐    ┌──────────────────┐
       │ Re-fire each       │    │ Each re-fire     │
       │ targeted agent     │ ── │ takes the        │
       │ with the question  │    │ question as      │
       │ block prepended    │    │ extra prompt     │
       │ to its prompt      │    │ context          │
       └────────────────────┘    └──────────────────┘
                          │
                          ▼
                    round_n+1 findings
                          │
                          └─→ loop back to PM critique
                              until: no questions OR
                              max rounds OR budget exhausted
```

**Round budget**: max 3 rounds (configurable via
`DEEP_RESEARCH_MAX_ROUNDS=3`). Most memos exit at round 1-2; round 3 is
a hard stop for runaway dialogues.

**Per-round budget**: PM emits at most 3 questions per round; each
question targets one agent. So worst-case extra LLM calls per memo:
3 rounds × (1 PM critique + 3 agent re-fires) = 12 extra LLM calls.

**Token cost estimate**: ~$0.10–$0.30 per memo at gpt-5.5 / Opus 4.7
prices (round 1 typically much cheaper since context is small).

## 3. Agent contract

### PM critique step (`agents/deep_research.py::pm_critique`)

Inputs:
- Current memo draft (rating, scores, key findings)
- All 8 specialist findings from the most recent round
- Conversation history of all prior rounds in this run

Output (structured):
```python
class CritiqueQuestion(BaseModel):
    target_agent: Literal["sector", "earnings", "valuation",
                          "comps", "risk", "filing", "macro", "technical"]
    question: str  # what the PM wants the agent to dig into
    why_it_matters: str  # what the answer would change

class CritiqueOutput(BaseModel):
    questions: List[CritiqueQuestion]  # 0-3 entries
    no_further_questions: bool  # explicit early-exit signal
    rationale: str  # short note for the audit log
```

`questions=[]` + `no_further_questions=True` exits the loop cleanly.

### Specialist re-fire

Each agent's `run_*_agent` function gets a new optional `prior_round_critique: Optional[str]`
parameter. When set, the LLM prompt prepends:

```
A senior PM has reviewed your prior round's finding and asked the
following follow-up. Address it directly with cohort math, ratio
references, or filing/transcript quotes. Do NOT contradict the prior
finding without articulating exactly what changed your read.

PM follow-up: {question}
```

Agents that don't have an LLM call (risk, comps deterministic path)
get a deterministic-augmentation hook that pulls additional structured
data per the question.

## 4. Persistence

Each round's findings stack on `memo.round_findings: List[RoundFindings]`
so the UI / audit log can show the full dialog:

```python
class RoundFindings(BaseModel):
    round: int  # 0 = initial fan-out, 1+ = critique rounds
    pm_questions: List[CritiqueQuestion]  # what PM asked this round
    findings: Dict[str, AgentFinding]  # who answered + the new finding
    early_exit: bool  # PM declared no further questions
```

The final memo's `sector_agent_view` (etc.) is the LATEST round's
finding — back-compat. UI can drill into prior rounds via a new
"Diligence dialog" panel.

## 5. Wiring

### New files
- `backend/app/agents/deep_research.py` — the loop + the PM critique
  prompt + structured output schema.
- `backend/app/tests/test_deep_research.py` — unit tests for
  loop termination, budget cap, no-question exit.

### Touched files
- `backend/app/agents/graph.py::_run_stock_memo_inner` — after the
  initial fan-out, gate on `settings.enable_deep_research` and call
  `deep_research.run_dialog_loop(memo, findings, max_rounds=…)` to
  iterate. Replace `findings` with the loop's final-round output.
- `backend/app/schemas.py` — add `RoundFindings`, `CritiqueQuestion`,
  `StockMemoOut.round_findings: List[RoundFindings]` (optional, default `[]`).
- Each specialist agent runner gets the `prior_round_critique` param
  threaded through (no behavior change when None).
- `backend/app/config.py` — add `enable_deep_research: bool = False`,
  `deep_research_max_rounds: int = 3`, `deep_research_max_questions_per_round: int = 3`.
- `frontend/src/components/MemoCard.tsx` — new collapsible
  "Diligence dialog" panel at the bottom of the memo, rendering each
  round's `pm_questions` + the agent answers as a threaded view.

## 6. Locked design decisions (avoid re-debating)

- **PM acts as a senior analyst, not a critic.** The critique step
  asks "dig deeper" questions, not adversarial ones. The Risk
  Committee critic is unchanged and still runs once at the end.
- **Re-fired agent re-runs ITS OWN LLM call.** No shortcut where the
  PM "imagines" what the agent would say. Real round-trips, real cost,
  real provenance.
- **Round 0 (the existing parallel fan-out) is unchanged.** Wave 9 is
  strictly additive; turning the flag off restores today's behavior.
- **No critique loop on incremental_patch memos** — patches are
  deliberately single-pass per the locked decision in MASTER_PLAN §5.
- **No critique loop on backtest runs** (`as_of_date` set) — that
  would balloon backtest cost without a clear research benefit.
- **Per-question budget:** 1 question = 1 agent re-fire. Even if PM
  asks 3 questions, 3 different agents each fire once. No multi-agent
  re-fan-out per question.
- **Persistence for the dialog**: rides on `memo.round_findings` + the
  existing memo_snapshots store. Each `MemoSnapshot` row carries the
  full dialog — versioned and auditable.

## 7. Telemetry / cost guardrails

- Every round emits an `LLMCallLog` row with `agent_name=PM Critique` so
  the existing `/api/admin/llm-metrics` dashboard surfaces deep-research
  spend separately from base memo spend.
- `memo.scores` gets `deep_research_rounds: int` + `deep_research_questions: int`
  for at-a-glance audit on the memo card.
- Hard kill switch: if any single memo run exceeds
  `deep_research_max_rounds * max_questions_per_round` LLM calls, abort
  the loop and ship the latest-round findings as-is with
  `degraded_agents += ["Deep Research budget exceeded"]`.

## 8. Exit criteria

- `enable_deep_research=true` + run_stock_memo on NVDA → memo carries
  `round_findings` with at least 1 follow-up round when the initial
  sector finding flags a non-trivial signal.
- `pm_critique` produces well-formed structured output with 0-3
  questions per round.
- `enable_deep_research=false` → memo is byte-identical to today's
  output (regression-safety).
- Per-memo cost stays under $0.50 at gpt-5.5 / Opus 4.7 list prices on
  the tier-1 universe.
- Tests verify: loop terminates on `no_further_questions`,
  loop terminates on max rounds, single-question failure doesn't
  poison the rest, deterministic specialist re-fire without an LLM
  produces an enriched finding (not a regression).

## 9. Rollout

1. **Phase A (this wave)** — ship the loop behind `enable_deep_research=false`.
   Run it manually on 5 tier-1 names, eyeball the dialogs, tune prompts.
2. **Phase B** — flip default-on for `auto_analysis` tier ONLY. Other
   tiers stay single-pass to keep on-demand cost predictable.
3. **Phase C** — open up to `analyzed_on_demand` once Phase B has 30
   days of clean cost telemetry.

## 10. Effort

- Backend: ~2.5 days
  - Schemas + loop scaffolding: 0.5 day
  - PM critique prompt + structured output: 0.5 day
  - Per-agent re-fire wiring: 0.5 day
  - Tests + budget guardrails: 0.5 day
  - Wire into graph.py + persistence: 0.5 day
- Frontend: ~0.5 day (Diligence Dialog panel)
- Manual eyeball + prompt tuning: ~0.5 day

**Total: ~3.5 dev-days for a flag-default-off ship-able feature.**

## 11. Out of scope for this wave

- Multi-agent collaboration *between* specialists (e.g., earnings asks
  filing for context). Right now everything routes through the PM as
  the hub. Multi-agent peer-to-peer is its own design problem.
- LLM-driven question prioritization (which of 5 candidate questions
  gets asked given a token budget). The first version uses the PM's
  unconstrained 0-3 output.
- A/B test infrastructure to measure deep-research's actual research
  quality lift vs. the single-pass baseline. Worth doing once we have
  a track-record dataset (Wave 4A) to score against.
