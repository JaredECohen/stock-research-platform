# MarketMosaic — Next Steps

**Date:** 2026-06-11
**Author:** Engineering review (Claude)
**Scope:** Memo-generation correctness, data consistency, reliability, and test coverage.

> **Status update (2026-06-11, second session):** B1–B7 and Themes 1–4
> are now addressed in code. Shipped: `_refresh_dcf_references` after the
> PM DCF adjuster (B2); a `ValuationVerdict` single source of truth on the
> memo schema, computed post-rating-blend and consumed by the thesis
> guard, valuation card, and mispricing fallback (Theme 1); deterministic
> fallbacks now register in `degraded_agents` via
> `DegradationLog.record_soft` when an LLM was supposed to run (B3/Theme
> 2); `key_risks` backfilled from the bear case + wired into the thesis
> builder's profile (B4); comps headline leads with the premium/discount
> magnitude (B5); deterministic `mispricing_thesis` fallback so the card
> never ships blank, plus frontend rendering for it (B6); thesis rewrite
> logging + a verdict-vs-final-rating consistency guard (B7 and the
> post-blend gap B1 left open); `generation_mode` now follows
> `use_demo_data_only` — production was serving LIVE data labeled "demo"
> (Theme 3), and render.yaml sets `ENABLE_LIVE_DATA=true` explicitly;
> `app/tests/test_memo_consistency.py` invariant suite (Theme 4.1);
> frontend: degraded-agents banner + mispricing section + valuation
> verdict + long-form reports in the full memo/PDF, clearer demo/live
> labels. Theme 5 is now also addressed in code: memo regen moved off
> the request-scoped daemon thread onto a durable Postgres-backed
> `regen_jobs` queue drained by a worker thread
> (`services/regen_worker.py`) — jobs survive restarts, orphaned runs
> (OOM kill / deploy) are requeued once with checkpoint resume then
> surfaced as `WorkerRestart` failures, per-step progress merges worker
> waypoints with `MemoRunCheckpoint` rows, telemetry at
> `/api/admin/regen-jobs`, the `/analyze/status` polling contract is
> unchanged, and a nightly regen smoke test
> (`app/tests/test_regen_smoke.py`) runs the full queue path in
> `nightly-live.yml`. Remaining open: Theme 4.2 (frontend test runner)
> and a production redeploy + memo regeneration so stored memos pick
> the fixes up.

This is an evidence-based review. Findings were verified two ways:

1. **Production data** — read the live COST memo from `GET https://marketmosaic.onrender.com/api/stocks/COST/memo` (the same endpoint the `/research` SPA calls).
2. **Source** — traced each symptom to the line in `backend/app/agents/graph.py` and the agent modules that produces it.

The COST memo is the running example throughout because it surfaces several distinct bugs in one report.

---

## Executive summary

The pipeline produces rich, well-written memos, but **the same valuation question gets answered independently in five places and the answers are never reconciled**. One memo can simultaneously say a stock is undervalued (thesis), Neutral (rating), trades at a +77% peer premium (comps), implies +49% upside (valuation card), and implies +17% upside (DCF summary). The single highest-leverage investment is a **consistency / single-source-of-truth pass over the valuation signals**, backed by invariant tests so the memo can never contradict itself again.

Secondary themes: silent agent degradation that isn't reported, empty risk/catalyst extraction that leaks template filler into prose, and ongoing memo-regeneration fragility visible in the recent commit history.

Priority key: **P0** = user-visible incorrectness, ship first. **P1** = quality gap or latent inconsistency. **P2** = hardening / longer-term.

---

## Confirmed bugs

### B1 (P0) — Thesis verdict contradicted the rating and valuation `[FIXED — commit 8a85455]`

**Evidence (production, generated 2026-06-06):**
- `rating_label`: `"Neutral"`
- `one_sentence_thesis`: *"COST is **undervalued** — Bull case: durable execution against sector tailwinds. DCF base case implies +17% to fair value; the gap rests on core driver execution vs. the dominant risk."*

**Root cause:** `_build_thesis_from_findings` derived the verdict word purely from `dcf.base.upside_pct` ([graph.py](backend/app/agents/graph.py)). The DCF base happened to be +17%, so the one-liner asserted "undervalued" while the rating badge said Neutral and the comps said expensive.

**Fix shipped:** verdict word now follows the memo's `rating_label` (bull→undervalued, bear→overvalued, neutral→fairly priced), falling back to DCF only when no rating exists. Also removed the "core driver execution vs. the dominant risk" template filler and stripped the "Bull case:" prefix from the claim.

**Remaining action:** this is the *pattern*, not a one-off. The same single-signal anchoring shows up in B2–B6 below. Production still serves the old memo until redeploy **and** a COST regeneration (the stored memo is cached; it does not self-update on deploy).

---

### B2 (P0) — Valuation finding shows a stale, pre-adjustment DCF number

**Evidence:**
- `dcf_initial_summary.summary`: *"Base case implied price $1,452.82 vs current $971.87 (**+49.5%**)"*
- `valuation_agent_view.headline`: *"P/E 51.7x; EV/EBITDA 30.8x; FCF yield 1.9%; DCF base implies **+49%** vs current"*
- `dcf_summary.base_upside`: **0.17** (the final, displayed DCF)

So the memo prints **two different DCF base upsides** (+49% in the valuation card, +17% in the DCF section) for the same stock.

**Root cause (ordering bug):**
1. The valuation agent runs inside the agent rounds on the original `dcf` — [graph.py:1132](backend/app/agents/graph.py#L1132).
2. The **PM DCF Adjuster** then tempers operating margins + terminal growth and *reassigns* `dcf = adjusted_dcf` — [graph.py:1218-1235](backend/app/agents/graph.py#L1218). For COST it cut the base from +49.5% to +17.0%.
3. Downstream `bull`/`bear` builders use the new `dcf` ([graph.py:1237-1240](backend/app/agents/graph.py#L1237)), which is why their numbers are correct — but `valuation_finding` is **never recomputed**, so it keeps +49%.

**Recommended fix (pick one):**
- **Re-run / patch the valuation finding after the adjuster.** Cleanest: move the valuation-agent call (or a lightweight DCF-number refresh of its headline/summary) to *after* the DCF adjustment.
- **Or** stop baking DCF numbers into agent prose and have the render layer pull the live `dcf_summary` so there is one number on the page.

Add an invariant test: every DCF percentage string in any agent view must equal `dcf_summary.base_upside` (rounded).

---

### B3 (P1) — Valuation agent silently fell back to boilerplate, but `degraded_agents` is empty

**Evidence:**
- `valuation_agent_view.summary` is **verbatim** the deterministic fallback string in [valuation_agent.py:107-111](backend/app/agents/valuation_agent.py#L107): *"DCF triangulates against multiples; the bull/bear range frames the discount-rate sensitivity. Valuation risk increases if terminal growth or margin assumptions slip."* An LLM does not reproduce that boilerplate word for word, so the LLM call returned empty and the deterministic path fired.
- Yet `degraded_agents`: `[]`.

**Why it matters:** the valuation section is one of the most important in the memo, and it quietly shipped a no-insight template while the UI presented it as a real analyst view. Degradation tracking exists (`log_to=degradation` is threaded through `safe_call`), but an LLM call that *succeeds structurally but returns nothing usable* doesn't register as degraded.

**Recommended fix:** treat a deterministic-fallback return inside an agent as a degradation event. Either have `run_valuation_agent` signal "fell back" up to the degradation log, or detect the boilerplate sentinel. Surface degraded agents in the UI so users know which sections are thin.

---

### B4 (P1) — Empty `key_risks` / `thesis_breakers` / `forward_catalysts` leak generic filler into prose

**Evidence:** in the COST memo these fields are all empty arrays: `key_risks: []`, `thesis_breakers: []`, `forward_catalysts: []`. That emptiness is exactly why the pre-fix thesis fell back to *"core driver execution vs. the dominant risk"* — the thesis builder reads `profile["risks"]`/`profile["drivers"]`, which were empty.

**Root cause:** `derive_risk_items` ([graph.py:1246](backend/app/agents/graph.py#L1246)) returned nothing for COST, and nothing downstream notices or compensates. Risk is one of the named specialist roles, so an empty risk list should be loud, not silent.

**Recommended fix:** when risk/catalyst extraction returns empty, either (a) retry with a fallback prompt, (b) derive risks from the bear case (which *was* populated, with specific drivers), or (c) mark the section degraded (see B3). The bear case already contains real risks ("Multiple compression risk: EV/EBITDA 30.8x", "Low FCF yield (1.9%)") — wiring those into `key_risks` would fix both the empty section and the thesis filler.

---

### B5 (P1) — Comps headline is generic while its own summary is detailed

**Evidence:**
- `comps_agent_view.headline`: *"Peer-relative read for COST"* (says nothing)
- `comps_agent_view.summary`: *"Peer set: WMT, TGT, AMZN. Trades at a **+76.6% premium to peers** on EV/EBITDA. Operating margin trails peer median... Growth is above peer median."* (says a lot)

The headline is the part most readers skim, and it's the empty one. The summary already contains the punchline ("+76.6% premium").

**Recommended fix:** generate the comps headline from the summary's strongest fact (the premium/discount magnitude and direction), the same way the thesis builder picks a claim. This is also the comps signal that should feed a unified valuation verdict (see Theme 1).

---

### B6 (P1) — `mispricing_thesis` is entirely empty

**Evidence:** `mispricing_thesis`: `{ consensus_view: "", our_view: "", gap: "", falsifiers: [] }` — every field blank.

This is supposed to be the "what does the market believe vs. what do we believe, and what would prove us wrong" structure — arguably the most valuable part of a differentiated memo. It's shipping empty. Either the generator is failing silently or it isn't wired into the demo path.

**Recommended fix:** trace where `mispricing_thesis` is populated, confirm it runs in the deployed mode, and apply the same degradation visibility as B3. If it can't be filled, it should not render as an empty card.

---

### B7 (P2) — Anti-pattern thesis rewrite likely fired as a false positive

**Context:** the production thesis carried the deterministic builder's fingerprint (the generic filler), which means the LLM's own thesis was discarded by `_looks_like_anti_pattern_thesis` and replaced — [graph.py anti-pattern guard](backend/app/agents/graph.py). The regex `_THESIS_ANTI_PATTERN` matches any `"{Co} — {Sector}/... DCF base case"` shape, which is broad enough to catch legitimate theses.

**Recommended fix:** log every time the rewrite fires (with the original LLM thesis) so the false-positive rate is measurable. Tighten the regex, or only rewrite when the deterministic result actually scores better. Right now a good LLM thesis can be silently downgraded to template prose.

---

## Systemic themes

### Theme 1 (P0) — No single source of truth for "is it cheap or expensive?"

This is the root pattern behind B1, B2, and B5. A single COST memo expresses the valuation call in at least five independently-computed places:

| Signal | COST value | Direction |
|---|---|---|
| `one_sentence_thesis` verdict | "undervalued" (pre-fix) | cheap |
| `rating_label` | Neutral | fair |
| `comps_agent_view` | +76.6% peer premium | expensive |
| `valuation_agent_view` headline | DCF +49% | cheap (and stale) |
| `dcf_summary.base_upside` | +17% | cheap |
| `scores.factor_valuation` | (low, on 51.7x P/E) | expensive |

None of these reconcile against the others. **Recommendation:** introduce one derived `valuation_verdict` object computed once, after the DCF adjuster, that blends DCF, comps, and the factor-valuation score, and have the thesis, rating narrative, and valuation card all read from it. Then add invariant tests (Theme 4) asserting the memo cannot contradict itself.

### Theme 2 (P1) — Silent degradation

B3, B4, and B6 are the same failure mode: a sub-step returns empty/boilerplate and the memo ships anyway with `degraded_agents: []`. The infrastructure to track this exists (`safe_call(..., log_to=degradation)`); the gap is that *structurally-valid-but-empty* results don't count as degradation. Make "fell back to deterministic" and "returned empty" first-class degradation events, and render them so users can calibrate trust.

### Theme 3 (P1) — Production is serving "demo" mode

The COST memo reports `generation_mode: "demo"`, which per [graph.py:1431](backend/app/agents/graph.py#L1431) means `enable_live_data` is false. LLM narrative still runs (the bull case cites real COST specifics), but the financial inputs are demo fixtures. **Confirm this is intentional for the public deployment.** If live data is meant to be on, this is a config bug; if demo is intentional for the showcase, label it in the UI so numbers aren't mistaken for live analysis.

### Theme 4 (P1) — Test coverage shape

- Backend: **53 test files** plus CI (`.github/workflows/ci.yml`, `nightly-live.yml`). Good foundation.
- Frontend: **no test runner** — `frontend/package.json` scripts are only `dev`/`build`/`preview`/`lint`. The memo-rendering components (`FullInvestmentMemo.tsx`) that display all of the above have zero automated coverage.
- **Missing entirely: memo-consistency invariant tests.** None of the bugs above would have been caught because there is no test asserting cross-field agreement.

**Recommended additions:**
1. A `test_memo_consistency.py` golden-memo fixture asserting: thesis verdict agrees with rating direction; every DCF % equals `dcf_summary`; no agent view contains deterministic boilerplate; `key_risks` non-empty when a bear case exists.
2. Frontend: add Vitest + a render test for `FullInvestmentMemo` against a fixture memo.

### Theme 5 (P2) — Memo-regeneration reliability

The recent commit history is dominated by production firefighting around memo regen: OOM kills (`bump web service ... to fix memo regen OOMkills`), async lifecycle (`detach memo regen ... daemon thread`), silent save failures (`let _persist_memo_snapshot raise`), and a stuck circuit breaker (`self-healing circuit breaker ... breaker stuck open caused sub-agents to silently no-op`). This points to a fragile long-running generation path. The cold-start I hit when reading production (Render free-tier spin-down) compounds it.

**Recommendations:** move memo generation to a proper background worker/queue rather than a request-scoped daemon thread; add structured progress + failure telemetry per step (some of this was started in the `diagnose:` commits); and add a health/regen smoke test to the nightly workflow.

---

## Prioritized roadmap

**Now (P0 — correctness, days):**
1. ~~B1 thesis verdict follows rating~~ (done, commit 8a85455 — needs deploy + COST regen to show).
2. B2 recompute/patch valuation finding after the PM DCF adjuster, so one DCF number appears everywhere.
3. Theme 1 first slice: a single derived `valuation_verdict` consumed by thesis + valuation card.
4. Add the memo-consistency invariant test (Theme 4.1) to lock B1/B2 in.

**Next (P1 — quality, 1–2 weeks):**
5. B3 + Theme 2: make deterministic-fallback and empty returns count as degradation, and render `degraded_agents`.
6. B4: populate `key_risks` from the bear case when risk extraction is empty.
7. B5: generate the comps headline from its strongest fact.
8. B6: fix or hide the empty `mispricing_thesis`.
9. Theme 3: confirm/correct `enable_live_data` on production, or label demo mode in the UI.

**Later (P2 — hardening):**
10. B7: log and measure anti-pattern-rewrite false positives, then tighten.
11. ~~Theme 5: move regen to a worker/queue; per-step telemetry; nightly regen smoke test.~~ (done — `regen_jobs` table + `services/regen_worker.py`, `/api/admin/regen-jobs`, `test_regen_smoke.py` in nightly-live.)
12. Theme 4.2: frontend test runner + `FullInvestmentMemo` render tests.

---

## Quick wins (high value, low effort)

- **B5 comps headline** — one prompt/extraction change, immediately better skim experience.
- **B4 risk backfill from bear case** — the data already exists in `bear_case`; just wire it through.
- **Consistency invariant test** — small, and it converts every future regression of this class into a red CI run instead of a user bug report.
- **Label demo mode in the UI** — one line, removes a credibility trap.
