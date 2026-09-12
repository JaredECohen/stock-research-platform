# Research process library

The owner's investment-research process, captured as durable reference material and
wired into the product where it fits. These files are inputs to MarketMosaic's design,
not runtime state (runtime memory lives under `memory/` at the repo root).

| File | What it is | How the product uses it |
| --- | --- | --- |
| `Research_State.md` | Canonical startup and continuity protocol for the 163-sub-industry system, including the causal long-term compounding hierarchy, evidence rules, current research state and local/online synchronization contract. | **Read first for every substantive investment-research session.** Local harnesses use this filesystem copy; ChatGPT sessions use the matching Investment Research project-source copy. |
| `Investment_Universe_163_Handbook.md` | Human-readable master reference for all 163 active GICS sub-industries, governed by the outside-in sequence from world change through valuation. | Primary research handbook after `Research_State.md`; numbers and KPIs test the causal thesis rather than generating it. |
| `Investment_Universe_163_Map.json` | Machine-readable companion to the handbook, including the ordered governing methodology, all 163 briefs, source routes, relationships and identity checks. | Structured input for harnesses and tools that need deterministic stage order and schema-preserving access to the complete universe map. |
| `GICS_74_Industry_Research_Encyclopedia.md` / `.xlsx` | One entry per GICS industry (11 sectors / 25 industry groups / 74 industries): economic engine, core KPIs, leading indicators, moats, capital cycle, valuation lenses, accounting traps, failure modes, ideal compounder and inflection patterns, research priority, thesis half-life. | Parsed by `backend/app/scripts/build_industry_knowledge.py` into `backend/app/data/industry_knowledge/gics_industries_2026.json`, the durable knowledge base the Industry Group analysts load (`backend/app/services/industry_knowledge.py`). |
| `Investment_Research_Atlas.md` / `.xlsx` | Economic sub-industry maps, discovery universe, indicator definitions and observation ledger, historical cycle cases, cross-sector dependency links, ranked research screens. Dated snapshot 2026-09-07. | Source for the Industry Group analysts' economic maps and for the cross-industry dependency context the Portfolio Manager receives. The ranked screens are research-priority queues, not buy lists. |
| `Framework_Update_2026-09-07.md` | Data/provenance decisions for the research framework: no FactSet; SEC/IR as primary facts, Alpha Vantage + FMP for estimates, Polygon/ThetaData for prices; comparison-gate and historical-eligibility controls; NSSC contract-vs-retention and MOD tax-bridge findings. | Provider roles match the platform's existing chain (FMP → Alpha Vantage → Polygon → Tiingo, plus SEC EDGAR). The "same-basis estimate comparison" and "never manufacture a historical dataset" rules govern the expectations ledger and any consensus features. Referenced companion files (`Updated_Project_Instructions.md`, `Updated_Validation_Plan.md`, `Updated_Task_Backlog.json`) were not supplied and live outside this repo. |
| `underwriting-2026-09-07/` | Six-company price-aware underwriting package (NSSC, MOD, POWL, LMAT, APPF, WINA): workbook, memo, machine-readable inputs/results, validation, decision update. | Worked example of the process the memo pipeline is being shaped toward: normalized cash flows, capital-structure bridges, an expectations ledger (reported consensus / guidance / price-implied / our forecast), reverse valuation, separate long-term DCF and two-year payoff tests. Model outputs, not recommendations. |

## The process in one paragraph

Start with `Research_State.md`, then reason **world change → industry consequences →
durable company advantage → value capture → reinvestment runway → long-term per-share
compounding → financial evidence/falsification → valuation**. Find mispriced future cash
flows, not merely good companies or exciting themes. Numbers are evidence and checkpoints,
not the thesis; `current KPI → forecast KPI → EPS → target price` is not the primary
idea-generation process. Answer four questions better than the market: what will happen,
who captures the economic benefit, what is already priced in, and what evidence will reveal
the gap. Two mandates
stay distinct (smaller long-term compounders; shorter-term mispriced inflections), each
with its own checklist and exit discipline. Every serious view keeps four expectation
columns apart: reported consensus, management guidance, price-implied assumptions, and
our own forecast. Observed data is always separated from interpretation; missing
evidence stays visibly unknown rather than being scored as neutral.

## Where it lands in the codebase

- Industry knowledge base: `backend/app/data/industry_knowledge/` (generated, checked in).
- Memo structure: the `mispricing_thesis` block (consensus view / our view / gap /
  falsifiers) is the expectations ledger; DCF outputs render "n/a" when a number cannot
  be computed rather than a misleading zero.
- Industry Group analysts and weekly Industry Analysis reports (FEAT-003) consume the
  encyclopedia and atlas through the registry-driven analyst roster.

Nothing here activates automated trading, scheduled research, or monitoring.
