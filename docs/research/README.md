# Research process library

The owner's investment-research process, captured as durable reference material and
wired into the product where it fits. These files are inputs to MarketMosaic's design,
not runtime state (runtime memory lives under `memory/` at the repo root).

| File | What it is | How the product uses it |
| --- | --- | --- |
| `GICS_74_Industry_Research_Encyclopedia.md` / `.xlsx` | One entry per GICS industry (11 sectors / 25 industry groups / 74 industries): economic engine, core KPIs, leading indicators, moats, capital cycle, valuation lenses, accounting traps, failure modes, ideal compounder and inflection patterns, research priority, thesis half-life. | Parsed by `backend/app/scripts/build_industry_knowledge.py` into `backend/app/data/industry_knowledge/gics_industries_2026.json`, the durable knowledge base the Industry Group analysts load (`backend/app/services/industry_knowledge.py`). |
| `Investment_Research_Atlas.md` / `.xlsx` | Economic sub-industry maps, discovery universe, indicator definitions and observation ledger, historical cycle cases, cross-sector dependency links, ranked research screens. Dated snapshot 2026-09-07. | Source for the Industry Group analysts' economic maps and for the cross-industry dependency context the Portfolio Manager receives. The ranked screens are research-priority queues, not buy lists. |
| `Framework_Update_2026-09-07.md` | Data/provenance decisions for the research framework: no FactSet; SEC/IR as primary facts, Alpha Vantage + FMP for estimates, Polygon/ThetaData for prices; comparison-gate and historical-eligibility controls; NSSC contract-vs-retention and MOD tax-bridge findings. | Provider roles match the platform's existing chain (FMP → Alpha Vantage → Polygon → Tiingo, plus SEC EDGAR). The "same-basis estimate comparison" and "never manufacture a historical dataset" rules govern the expectations ledger and any consensus features. Referenced companion files (`Updated_Project_Instructions.md`, `Updated_Validation_Plan.md`, `Updated_Task_Backlog.json`) were not supplied and live outside this repo. |
| `underwriting-2026-09-07/` | Six-company price-aware underwriting package (NSSC, MOD, POWL, LMAT, APPF, WINA): workbook, memo, machine-readable inputs/results, validation, decision update. | Worked example of the process the memo pipeline is being shaped toward: normalized cash flows, capital-structure bridges, an expectations ledger (reported consensus / guidance / price-implied / our forecast), reverse valuation, separate long-term DCF and two-year payoff tests. Model outputs, not recommendations. |

## The process in one paragraph

Find mispriced future cash flows, not merely good companies or exciting themes. Answer
four questions better than the market: what will happen, who captures the economic
benefit, what is already priced in, and what evidence will reveal the gap. Two mandates
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
