# MarketMosaic — product design review and improvement plan

> **Purpose.** A whole-product audit: what every page, agent, data layer,
> and service does today; where it falls short; and a concrete plan to
> make each one materially better. Anchored to the code (file paths and
> functions cited inline) and to the founder's prioritized feedback
> from May 2026.

> **Status.** Living document. Wave 10 + Wave 11 design lives here.
> When a section ships, leave the "How it works" prose, fold the
> "Improvements" content into a Wave-tagged changelog at the bottom of
> the section, and update the lead bullet so a reader can tell at a
> glance whether the gap still exists.

---

## 0. Vision — from pipeline to portfolio manager

The product was built as a parallel-fan-out + synthesize pipeline:
eight specialists run, the PM aggregates, the user reads a memo. That
gets the user a memo, but it does not get the user a *portfolio
manager*. A real PM doesn't just collate; they push back, ask the
sector specialist a follow-up, recall what the company said two
quarters ago, and tell you why the market is wrong.

The headline shift in this design is **PM-as-brain, not PM-as-router**:

- **PM owns its own memory.** A `memory/pm/notes.md` file the PM reads
  on every turn — investing principles, recent macro takes, lessons
  from past calls that worked or didn't. Today the per-company
  `memory/companies/<T>.md` exists; there is no PM-level memory.
- **PM dynamically calls specialists.** Today an open-ended chat
  question that doesn't match a hard-coded intent lands in a generic
  "answer with cached metrics" path. Going forward the PM is itself a
  tool-using agent: when a question needs the sector view, it calls
  the sector agent live; when it needs the latest 10-K passage, it
  hits the filings RAG; when it doesn't know, it asks one of them
  rather than confabulating.
- **PM reads memos as documents, not as routing keys.** Currently the
  orchestrator pulls cached memos into a "follow-up context" string.
  A real PM would treat the memo as evidence — quote the thesis,
  challenge the rating, look up specific paragraphs.
- **PM articulates mispricing.** The differentiator a serious retail
  investor pays for is *"why is the market wrong about this name?"*
  The current memo gives a rating and a thesis; it rarely says the
  thing a real PM says, which is "consensus thinks X, the data says
  Y, the gap is what creates the alpha." Every memo should carry a
  named **Mispricing Thesis** field with falsifiable claims.

Everything below funnels into this vision. The DCF improvements give
the PM a sharper valuation story to defend; the RAG/vector-DB work
gives the PM a body of evidence to cite; the track-record feedback
loop teaches the PM what kind of mispricing claims actually work out.

---

## 1. Stock Research / Memo (`/research`)

### How it works

`run_stock_memo()` in [`agents/graph.py`](backend/app/agents/graph.py)
fans out eight specialists in parallel
([sector, earnings, filing, valuation, comps, macro, risk,
technical](backend/app/agents/)), runs the cross-family critic on the
draft, and synthesizes a final view through the PM. Each specialist
runs as a `@checkpointed` step so a crashed run resumes without
re-paying for completed work. Memos are versioned in
`memo_snapshots`; updates patch in place when EDGAR / earnings / news
deltas fire.

### Limitations

1. **Static workflow.** Every memo runs the same eight specialists
   regardless of what's interesting about the name. A semis stock
   doesn't need news on regulatory catalysts; a regulated bank doesn't
   need a deep technicals read.
2. **No specialist follow-up by default.** The Wave 9 deep-research
   loop exists in [`agents/deep_research.py`](backend/app/agents/deep_research.py)
   but is gated behind `enable_deep_research=False` in
   [`config.py`](backend/app/config.py). So the standard memo is
   single-pass even though the iterative loop is the more rigorous
   path.
3. **Mispricing thesis is implicit.** The memo carries `thesis` and
   `rating` but no first-class "what does the market get wrong"
   field. The user has to read between the lines.
4. **Section-to-tab navigation is one-way.** A memo's valuation
   section displays the DCF result inline but does not link to
   `/dcf?ticker=<T>` so the user can't drill in.

### Improvements

1. **Dynamic specialist selection.** Add an `intake` step before the
   fan-out that asks the PM: "given this profile + recent news
   alerts, which specialists matter most?" Default = all 8. PM can
   skip up to 3 specialists per run, with a logged rationale. Saves
   cost and sharpens focus on the names where it matters.
2. **Default-on deep research.** Flip `enable_deep_research=True`
   for premium tier; cap at 1 round, max 2 follow-up questions.
   Measured cost increase ~15-25%; quality lift on the questions
   today's single-pass misses (cohort outliers, segment color, what
   would invalidate the thesis).
3. **Mispricing Thesis field.** Add `mispricing_thesis` to the
   `MemoOut` schema. Required structure:
   - **Consensus view:** what does sell-side / market price imply?
   - **Our view:** what does our analysis say?
   - **Gap:** specific number or claim that is different.
   - **Falsifiers:** what observation would prove us wrong?
   PM prompt enforces this — refuses to ship a memo without it.
4. **Click valuation → DCF Lab.** Frontend: wrap the memo's
   Valuation section in a `<Link to="/dcf?ticker={ticker}">`. Same
   for: Comps section → `/comps?ticker=…`, Macro section →
   `/macro?ticker=…`, Earnings → an Earnings tab (see §3) for the
   ticker. Cheap UX win, ships in a single PR.
5. **Memo-time vs. live-time price.** [Just shipped: `f048f3d`]
   The DCF comparison reads from a 60s-cached intraday quote, not
   the 7-day-cached profile. Layer 2 follow-up: persist
   `price_at_memo` per memo and display *both* on the Research page
   ("memo wrote DCF vs $145; current $158, +9% since memo") so the
   user sees price decay of the recommendation.

---

## 2. Ask the PM (`/chat`)

### How it works

[`agents/orchestrator.py::Orchestrator.chat()`](backend/app/agents/orchestrator.py)
runs `classify_intent` (LLM with regex fallback) over 8 hard-coded
intents — `single_stock_analysis`, `stock_comparison`, `dcf_analysis`,
`comps_analysis`, `portfolio_construction`, `thematic_screen`,
`macro_question`, `general_research_chat`. First message routes to
the matching workflow (memo, DCF, comps, screener, portfolio).
Follow-up turns route to an `agents_sdk` chat agent with eight tools
(`get_memo`, `get_dcf_summary`, `get_comps`, `get_macro_snapshot`,
`get_company_lite`, `list_universe`, `screener_query`,
`custom_screen`). When no intent matches and there's prior context,
the orchestrator hands a "lite snapshot" (P/E, margins, ROIC,
growth, beta) of any tickers in scope to the LLM and asks it to
answer.

### Limitations

1. **PM-as-router.** The orchestrator's job today is mostly to
   match the question to a workflow. When it lands in the SDK chat
   agent, that agent has tool access but no investing principles, no
   memory of how it has reasoned before, and no instruction to
   challenge the user's framing.
2. **Open-ended questions get template answers.** The user
   observed: *"the PM needs to do better at answering open ended
   questions instead of giving template response."* Confirmed in
   code — `general_research_chat` falls through to "answer with
   cached metrics" with no synthesis instruction.
3. **No PM memory.** Per-company memory exists at
   `memory/companies/<T>.md`; there is no `memory/pm/` for the PM's
   own evolving views, principles, or hot takes.
4. **Specialists not callable from chat.** The SDK tool surface
   exposes *cached outputs* (memo, DCF, comps) — it does not let the
   PM live-fire a specialist on a new question. So if the user
   asks "what would the sector analyst say if rates fall 100bps,"
   the SDK agent has nothing dynamic to call.
5. **No mispricing-first response shape.** Even when the PM
   answers well, it tends toward "here are the metrics" rather than
   "here's what consensus is missing."

### Improvements

1. **`memory/pm/notes.md` — the PM brain file.** One markdown file,
   curated by the PM itself across runs. Sections: investing
   principles (revisable), recent macro takes, calls that worked /
   didn't, theses currently in conviction. PM reads this on every
   chat turn (~2-4 KB injected as a system block). PM is allowed to
   *write* to it — at the end of each session, append "today I
   learned X" entries. Long-term memory for the PM, not just for the
   companies.
2. **Specialists as live tools.** Add to the SDK tool surface:
   `ask_sector(ticker, question)`, `ask_earnings(ticker, question)`,
   `ask_filings(ticker, question)`, `ask_macro(question)`,
   `ask_valuation(ticker, question)`. Each is a thin wrapper that
   re-fires the corresponding specialist with the question as
   additional prompt context (this is what `deep_research.py`
   already does internally — expose it as a tool). Enforces budget:
   max 2 specialist calls per chat turn.
3. **Mispricing-first response template.** PM system prompt
   amended: *"When the user asks about a specific name, structure
   your answer as: Consensus view → Our view → Gap → Falsifiers.
   When the user asks an open-ended thematic question, structure
   as: Three theses I'd defend → Three I'd reject → Where I'm
   uncertain."* This is the single biggest behavioral lever — it
   forces synthesis over recitation.
4. **PM identity prompt upgrade.** Today's `PM_SYSTEM` prompt is
   light. Replace with a longer identity passage modeled on a
   senior buy-side PM: opinionated, willing to dissent from
   consensus, demands evidence, distinguishes price from value,
   skeptical of LLM hallucinations. Audit this with `/ultrareview`
   on the prompts directory.
5. **Memo-as-evidence in chat.** When a chat question references a
   ticker the user has already researched, inject the relevant
   memo *paragraphs* (not just a header) into the PM context, with
   citations: `[memo §Valuation: "…"]`. PM is then expected to
   challenge or build on its own prior memo, not regurgitate it.

---

## 3. Earnings agent + Earnings UI

### How it works

[`agents/earnings_agent.py::run_earnings_agent`](backend/app/agents/earnings_agent.py)
takes a transcript (prepared remarks + Q&A, capped at ~18 KB total)
and asks an LLM (`OPENAI_TOOL_MODEL`, max_tokens=2000) for headline,
summary, key_points, confidence. Augments with research-notes block
(BM25-ranked chunks on guidance / margins / capex / demand).
Deterministic fallback when the LLM is unavailable just states
transcript size + next earnings date — i.e., almost nothing.

### Limitations

1. **Underexposed in the UI.** The memo shows a paragraph; the user
   has no way to see the full speaker-segmented transcript, the
   guidance the model thinks management gave, or the questions
   analysts asked.
2. **No structured extraction.** Today's output is freeform key
   points. There's no schema enforcing "guidance change vs. last
   quarter," "tone shift in CEO vs CFO," "what segment did
   management defend most aggressively."
3. **Single-pass over 18 KB.** Long calls get truncated. Important
   color (e.g., the third question of the Q&A) may not survive
   the cap.

### Improvements

> Reference: `JaredECohen/earnings-call-LLM-analyzer` — the user has
> a separate repo with a richer earnings-call breakdown. Port the
> high-value pieces here.

1. **Structured earnings schema.** Replace freeform `key_points`
   with a typed extraction:
   ```
   guidance_changes: List[{metric, prior, current, direction}]
   tone_signals: List[{speaker, segment, classification, evidence}]
   q_and_a_themes: List[{theme, analyst, response_quality}]
   most_defended_segment: {name, why}
   most_pressed_segment: {name, why}
   forward_catalysts: List[{event, expected_quarter, materiality}]
   ```
   The agent fills each field with a citation (transcript line
   range). UI renders each as its own card.
2. **Earnings tab.** Add `/earnings/<ticker>` page with: latest call
   summary, full speaker-segmented transcript, guidance-change
   timeline (last 4 quarters of changes side-by-side), tone trend
   chart, Q&A theme heatmap. Linked to from the memo's Earnings
   section (see §1.4).
3. **Multi-pass over long transcripts.** Chunk transcripts by Q&A
   exchange; run the LLM once per chunk; aggregate. Cost ~3-4x but
   nothing material gets dropped.
4. **Quarter-over-quarter delta agent.** New sub-step: load the
   prior-quarter extraction; ask an LLM "what's different this
   quarter vs. last? what's confirmed? what's reversed?" Output
   feeds the PM as one of the strongest synthesis inputs ("CEO
   defended margins last quarter and walked it back this quarter"
   is the kind of signal a real PM lives for).

---

## 4. Filing agent + filings RAG

### How it works

[`agents/filing_agent.py::run_filing_agent`](backend/app/agents/filing_agent.py)
retrieves filing chunks via `retrieval_service` (BM25 with the
hard-coded query "risk factors growth strategy thesis", limit=4),
combines with MD&A (12 KB), risk factors (6 KB), business
description (3 KB), and segments. Passes to LLM
(`OPENAI_TOOL_MODEL`, max_tokens=2400, target 8-12 key_points).
Deterministic fallback returns the top-3 risk factors and segment
names by position. Filing storage is in [`filing_docs`](backend/app/models.py)
with parsed sections.

### Limitations

1. **Retrieval is BM25 with a fixed query.** "risk factors growth
   strategy thesis" is what every filing run searches for. There's
   no question-specific retrieval — if the user asks about
   inventory cycles or supply chain exposure, BM25 has no idea.
2. **No vector embeddings.** The user explicitly called this out:
   *"we should have a Vector database to support RAG on the
   filings."* Today's chunks live in `cached_documents` keyed by
   document, not by semantic meaning.
3. **New filings don't update memory.** When EDGAR fires a new
   10-K, the filing agent re-runs but the *company memory file*
   (`memory/companies/<T>.md`) is not augmented with what's new in
   the filing. So insights decay across runs instead of compounding.
4. **No sector-level insights from filings.** A consistent theme
   across, say, six semiconductor 10-Ks ("NAND demand recovery
   cited") is the kind of pattern that should reach the sector
   analyst's memory. Today it doesn't.

### Improvements

1. **Vector DB for filing chunks.**
   - **Backend:** ChromaDB (embedded; no infra dependency) or
     pgvector (we already use Postgres in production). Embedding
     model: OpenAI `text-embedding-3-large` (3072 dims) or Gemini
     equivalent.
   - **Schema:** chunk_id, ticker, filing_type, accession, section,
     period_end, text, embedding, metadata.
   - **Pipeline:** when EDGAR delivers a new filing, chunk it
     (~500 tokens, 50-token overlap), embed, upsert. Reuse the
     existing `filing_docs` table as the source of truth; vector
     store is a derived index.
   - **Retrieval:** hybrid search — vector similarity for semantic
     match + BM25 boost on entity names + recency boost. Top-K=8
     by default.
2. **Filing → memory pipeline.** New step in the EDGAR webhook /
   poller: after the filing agent runs, it also produces a
   "memory delta" — 3-5 bullet points of what's new versus the
   prior filing of the same type. These get appended to:
   - `memory/companies/<T>.md` under a dated section.
   - `memory/sectors/<sector>.md` if the delta mentions a sector
     theme (LLM-judged).
   So the next time anyone asks about the company or sector, the
   long-term memory carries a *cumulative summary* of recent
   filings, not just the latest one.
3. **Question-specific retrieval.** Drop the fixed
   "risk factors growth strategy thesis" query. Replace with: the
   filing agent's prompt itself is the query embedding. PM's
   chat-time `ask_filings(ticker, question)` tool (see §2.2) uses
   the user's question as the query.
4. **Cross-filing diff agent.** When a new 10-K lands, run a
   pairwise diff against the prior 10-K: which risk factors are
   new? which were dropped? which got longer? Risk-factor
   *additions* and *removals* are extremely high-signal events that
   no LLM today is configured to surface.

---

## 5. Valuation agent + DCF Lab (`/dcf`)

### How it works

[`agents/valuation_agent.py`](backend/app/agents/valuation_agent.py)
takes the profile, ratios, and DCF result, and emits a finding
covering current price (now live, post-`f048f3d`), multiples, FCF
yield, ROIC, and DCF implied prices.

[`finance/dcf.py::build_full_dcf`](backend/app/finance/dcf.py)
runs base/bull/bear scenarios + 3 sensitivity tables.
`derive_default_assumptions()` infers a 5-year revenue path from
analyst estimates (consensus first, historical trend fallback);
holds operating margin flat at trailing 3-yr average; computes
WACC (clamped 6-14%); derives tax / capex / D&A / NWC ratios from
trailing 3-yr averages. Bull = +400 bps growth, +200 bps margin,
+50 bps terminal growth, +3.0x exit EBITDA, -50 bps WACC. Bear is
the symmetric mechanical opposite. Terminal value is a 50/50 blend
of Gordon perpetuity and exit-multiple methods. Sensitivities are
3x 5x5 grids: WACC × Terminal Growth, WACC × Exit EBITDA,
Yr1 Growth × Terminal Op Margin.

### Limitations (the user's feedback was specific here)

1. **Inputs are unreadable.** Five years of revenue and margin
   assumptions are spread across separate inputs without visual
   alignment. User wants them stacked in one column or one row so
   the trajectory is legible at a glance.
2. **Bull/bear are mechanical, not narrative.** ±400 bps is the
   *same* tweak whether the company is a sticky-software cash
   compounder or a cyclical commodity name. The user's words: *"make
   it more clear what assumptions are leading to the bear and bull
   cases — this should also be controllable."*
3. **Sensitivity tables are uncoloured.** Users can't tell at a
   glance which cells imply upside and which imply downside relative
   to today's price.
4. **Valuations seem off.** User's words: *"make the valuations
   better, they seem off."* Likely root causes:
   - Operating margin held flat ignores margin mean reversion.
   - Tax rate / capex / D&A / NWC inferred from 3-yr trailing
     averages — for cyclicals at peak this overstates margins.
   - Terminal value 50/50 blend can disagree with itself by 30%+
     and the model doesn't flag the disagreement.
   - No reality check against trading multiples (does the implied
     EV/EBITDA at year 5 violate cohort distribution?).
5. **No memo-time anchor.** Even after fixing the live price, the
   DCF caches its `current_price` at run time, so the upside_pct
   shown in the *cached* DCF result drifts as the market moves.

### Improvements

1. **Layout: stacked assumption table.**
   Single table, one row per assumption category, columns Y1-Y5 +
   Terminal:
   ```
   |                 | Y1   | Y2   | Y3   | Y4   | Y5   | Terminal |
   | Revenue (% YoY) | 12%  | 10%  | 8%   | 7%   | 6%   | 3%       |
   | Op margin (%)   | 28%  | 29%  | 30%  | 30%  | 30%  | 30%      |
   | Tax rate (%)    | 21%  | 21%  | 21%  | 21%  | 21%  | 21%      |
   | CapEx (% rev)   | 6%   | 5%   | 5%   | 5%   | 5%   | 5%       |
   | D&A (% rev)     | …    | …    | …    | …    | …    | …        |
   | NWC Δ (% Δrev)  | …    | …    | …    | …    | …    | …        |
   | WACC            | (single value, top of table)               |
   ```
   Every cell editable; recompute on blur. Show base / bull / bear
   columns as a tabbed switch above the table.
2. **Narrative bull/bear with sliders.** Replace the symmetric ±
   tweaks with an explicit assumption explorer:
   - Bull case header: *"What has to be true for $XYZ?"*
     User can toggle three drivers (e.g., "margin expands to 35%",
     "growth holds 15% through Y5", "WACC compresses 50 bps") with
     check-and-slider UI; implied price recomputes live.
   - Bear case mirrors with stress drivers.
   - When the user picks drivers, the agent generates a one-paragraph
     *narrative* describing why those conditions might or might not
     hold ("for margin to reach 35%, gross margin needs ~5pp lift —
     plausible if pricing power on AI products holds, but cohort
     median expansion was 1pp YoY").
3. **Red/green sensitivity tables.** Each cell colored on a gradient
   relative to current live price: red for implied price below
   current (downside), green for above (upside), with intensity
   proportional to magnitude. CSS-only change.
4. **Reality-check guardrails.** New deterministic step at end of
   `build_full_dcf`: compute implied year-5 EV/EBITDA, EV/Revenue,
   FCF yield. Compare to cohort distribution from comps_history.
   If implied multiple is outside cohort 90th percentile, flag the
   model with a warning. The PM sees these flags and decides whether
   to defend or revise.
5. **Margin mean reversion option.** Default keeps current behavior.
   New toggle: "revert margin to cohort median over forecast
   period." For cyclicals at peak this prevents the most common DCF
   error.
6. **Live-price overlay on DCF.** [Layer 2 from §1.5] Persist the
   memo-time `price_at_memo`; render both prices alongside on the
   DCF Lab so the user sees price drift since the memo's
   recommendation.
7. **Audit existing valuations.** Before shipping the redesign,
   run the new DCF over the full S&P 100 and compare implied
   prices vs. live prices. Names where the model is more than ±50%
   off are the canaries — investigate, and decide whether to fix
   the model or accept the divergence as alpha signal.

---

## 6. Comps (`/comps`) + comps agent

### How it works

[`agents/comps_agent.py`](backend/app/agents/comps_agent.py) wraps
[`finance/comps.py::compute_comps`](backend/app/finance/comps.py),
which takes a target row and a peer list, computes medians on 10
metrics (revenue growth, margins, ROIC, EV/EBITDA, P/FCF, FCF
yield, etc.), calculates percentile ranks, and emits interpretation
("trades at +15% premium on EV/EBITDA"). Adds self-historical lens
(percentile vs. target's own 3-5 yr median). Confidence bumps to
0.75 when peer and own-history lenses agree, drops to 0.65 when
they diverge (treated as an alpha signal).

Peer set is selected **a priori**, by sub-industry / GICS match, in
[`comps_history.py`](backend/app/finance/comps_history.py). LLM is
not involved in peer selection.

### Limitations

1. **Peer selection is sector-mechanical.** A sub-industry match
   gets you "other large software companies" — fine for SaaS, weak
   for AMZN where the relevant peers are GOOGL and MSFT (different
   sectors, same AI capex exposure) plus WMT and COST (retail
   competition).
2. **No cross-sector exposure peers.** The user's specific feedback:
   *"AMZN, GOOGL, MSFT are all AI-centered despite being in
   different sectors."* Today none of them would appear in each
   other's comp sets.
3. **Direct competitors not always selected.** For
   biotech / semiconductors / restaurant chains, the user wants
   *direct* competitors picked first, not cohort-by-percentile.
4. **Single peer set.** The user looks at Apple, but Apple is two
   businesses (consumer hardware + services). Today Apple gets one
   set of peers — handset makers + services giants merged together.

### Improvements

1. **Two-track peer selection: a priori + LLM.** Pick *both*,
   present *both*.
   - **Track A — direct competitors (a priori).** Rule-based. For
     each ticker, maintain a curated `data/direct_competitors.json`
     of 3-5 hand-picked direct competitors sourced from a one-time
     LLM seeding pass + manual override. Domain-specific rules:
     biotech → same therapy area + similar trial stage; semis →
     same sub-segment (logic / memory / equipment); restaurants →
     same format (QSR / casual / fine dining).
   - **Track B — exposure peers (runtime LLM).** At memo run time,
     ask an LLM: "name 3-5 companies that share material exposure
     to {ticker}'s key drivers (e.g., AI capex, China consumer,
     long-rate sensitivity), regardless of GICS sector." Cache
     these on the company record with a 30-day TTL. The user's
     AMZN/GOOGL/MSFT example falls naturally out of this track.
   - **UI:** show two side-by-side tables. Both contribute to the
     comps agent's reasoning; the PM weights them.
2. **Multi-business companies.** For tickers flagged as
   multi-business (Apple, Amazon, Berkshire, conglomerates), allow
   per-segment peer sets. Schema: `peers[business_segment] =
   List[ticker]`. UI renders one tab per segment.
3. **Comps quality audit.** Periodic job: for each ticker, ask the
   PM to score the current peer set 1-5 on relevance and produce a
   short critique. Sub-3 scores get manually curated.
4. **LLM-driven comps narrative.** The agent today emits structured
   percentile prose. Add an LLM pass on top: *"summarize what these
   comps tell us in one paragraph an investor can use"*. Today's
   prose is technically correct but reads like a spreadsheet.

---

## 7. Sector analyst

### How it works

[`agents/sector_agents.py`](backend/app/agents/sector_agents.py)
calls `run_sector_research()` to build a sub-industry cohort,
computes KPI percentiles, detects regime, aggregates filing themes,
pulls secular trends. Passes the cohort dict through an LLM (with
prior-critique loop support) to emit headline + key_points + a
structured bull/bear analysis. Cross-sector relevance hints come
from a hard-coded adjacency map (AI infra → utilities, banks →
tech).

### Limitations

1. **Cross-sector adjacency is demo data.** The hard-coded map is
   shallow and doesn't update.
2. **Macro broadcast is coarse.** Sector agent subscribes to a
   macro broadcast but the broadcast itself contains rate snapshot
   + scenario tag, not nuance.
3. **No sector memory file.** Per-company memory exists; per-sector
   doesn't. So a sector analyst building a cohort view today has no
   institutional memory of what worked / didn't in this sector last
   quarter.
4. **Bull/bear lean is computed once.** No live re-computation when
   macro regime changes mid-week.

### Improvements

1. **`memory/sectors/<sector>.md`** — per-sector memory file.
   Curated by the sector agent itself. Sections: structural trends,
   recent earnings season takeaways (auto-populated from earnings
   agent QoQ deltas, see §3.4), regulatory state, key cohort
   metrics to watch. Read on every sector run; written at the end.
2. **Real cross-sector exposure model.** Replace the hard-coded
   adjacency map with a learned model: extract co-mention frequency
   from earnings calls and 10-Ks ("supply chain", "AI capex",
   "consumer spend") into a sparse matrix, ticker × theme.
   Adjacency = cosine similarity in theme space. Refreshed weekly.
3. **Live macro re-fire.** When the macro broadcast flips scenario
   (`soft_landing` → `recession`), all sector agents' bull/bear
   lean is invalidated. Mark them stale; refresh on next read.
4. **Deeper synthesis prompt.** User's words: *"make sure the
   sector analyst and PM are thinking deeply to synthesize
   information."* Concretely: replace the current key_points-list
   prompt with a structured synthesis prompt that demands:
   - One-paragraph **What changed this quarter** (vs prior memo).
   - **Cohort outliers** (top 1, bottom 1) with cited metrics.
   - **Sector mispricing thesis** (mirrors the company-level
     mispricing thesis, §1.3).

---

## 8. Macro analyst (`/macro`) + macro agent

### How it works

[`agents/macro_agent.py`](backend/app/agents/macro_agent.py) detects
scenario from `market_view` text (regex with LLM validation: keys
are `soft_landing`, `recession`, `sticky_inflation`, `falling_rates`,
`ai_capex_boom`). Seeds from `SCENARIO_TEMPLATES` (hardcoded sector
impacts). LLM rewrites narrative + suggested_research_views using a
live FRED snapshot (FEDFUNDS, 10Y, CPI, Unemployment). Per-company
`run_macro_agent()` grounds in sector default impact + company
profile.

### Limitations

1. **Five canned scenarios.** Real macro state is continuous and
   often a mix of regimes. The user's words: *"make the macro
   analyst smarter — give it some context on fed speeches and
   economic theory."*
2. **No Fed primary sources.** FOMC minutes, Fed speeches, dot plot
   shifts, SEP releases — none of these flow into the macro agent.
3. **Sector impacts are canned.** The `SCENARIO_TEMPLATES` dict
   defines "in a recession, banks suffer" — but the *magnitude* is
   hardcoded.
4. **No macro memory.** No file capturing how the macro analyst
   has been thinking over time — when it called a regime shift,
   when it was wrong, what it's watching.

### Improvements

1. **`memory/macro/notes.md`.** Per-week macro memory: dated
   entries summarizing the regime view, key data releases, Fed
   speech takeaways, and the analyst's confidence. Read on every
   macro run.
2. **Fed-speech ingestion.** New `fed_speech` data capability:
   poll the Fed's speeches RSS, extract speaker + key sentences via
   LLM, dedupe, store. Macro agent reads recent speeches in its
   prompt. Same shape for FOMC minutes (timed releases, PDF →
   structured extraction). FRED already provides the numerical
   series; this adds the *commentary* the numerical series is
   responding to.
3. **Macroeconomic theory as system context.** Inject a curated
   reference into the macro agent's system prompt: short primer on
   monetary policy transmission, business cycles, supply vs.
   demand inflation, term structure logic. Maintained as a
   markdown file under `prompts/macro_primer.md`. Stops the agent
   from confabulating textbook macro relationships.
4. **Continuous regime score.** Instead of a single scenario tag,
   output probabilities across the 5 regimes (e.g., 0.55 soft,
   0.30 sticky, 0.15 recession). Sector impacts blend across
   regimes weighted by probability. Vastly more useful than a
   single regime tag in genuinely uncertain periods.
5. **Macro-driven memo invalidation.** When the regime probability
   shifts > 20% on a key dimension (rates, growth, inflation),
   trigger a memo refresh on rate-sensitive names. Memo's
   `triggers` field already supports this — wire macro into it.

---

## 9. News + Social agents

### How it works

**News:** [`agents/news_agent.py`](backend/app/agents/news_agent.py)
tries Gemini 2.5 Flash with `google_search` grounding first
(prompt: "find 5 most material news items PRIMARILY about
{company}"; max_tokens=2500 for JSON + citations). Classifies
severity (breaking / material / advisory). Filters by relevance
(title or URL slug must mention ticker / company name tokens —
sector roundups dropped). Falls back to legacy `news_service` if
Gemini missing. Domain allow/block lists from
`app/data/news_domains.json`. Caches alerts 4h.

**Social:** [`agents/social_agent.py`](backend/app/agents/social_agent.py)
exists but the implementation is thin (per the inventory: "(unclear
from code)").

### Limitations

1. **Google grounding is just one Gemini call.** The user's words:
   *"make sure you are leveraging google and the APIs fully."*
   Today's call is 5 items, single shot. We're not using Google
   News API directly, not using SERP API, not using Reddit or X.
2. **Relevance filter is title-only.** Body-text mentions are
   dropped because we can't see the body — a smarter filter would
   read the article.
3. **No catalysts surface.** News produces alerts, but there's no
   "upcoming catalysts" calendar surfaced anywhere (FDA dates,
   earnings, investor days, conferences). User has to remember.
4. **Social media is barely there.** Should be doing two things:
   (a) sentiment / crowdedness on the name; (b) detecting the
   *macro* trends (consumer behavior, product cycles) that shape
   investing.

### Improvements

1. **Multi-source news ingestion.**
   - Google News API (paid tier — more reliable than scraped
     grounding).
   - SERP API for query-based discovery.
   - Direct RSS feeds for top financial pubs (WSJ, FT, Bloomberg,
     Reuters, Barrons).
   - Aggregator: dedupe by URL canonical, keep best source per
     story.
2. **Body-text relevance.** When the news snippet passes the
   title filter but is borderline, fetch the article body
   (cached 7d), re-score with a one-shot LLM relevance prompt.
3. **Catalyst calendar.** New service: `services/catalysts.py`.
   Sources: FMP earnings calendar, FDA pipeline dates, conference
   schedules. Surfaced on:
   - The memo (next 90 days of catalysts box).
   - The PM chat (when asked "what's the catalyst?").
4. **Social — two distinct agents.**
   - **`social_sentiment_agent`:** focused on the ticker — Reddit
     + X (paid tier) + StockTwits. Outputs crowdedness score,
     bull/bear ratio, retail-driven moves. Useful for short-term
     positioning, not for thesis.
   - **`social_trends_agent`:** focused on the *world* — consumer
     behavior, product adoption, brand sentiment, regulatory
     murmurs. Pulls from Reddit (relevant subreddits per industry),
     Twitter/X, Google Trends, app-store reviews. Outputs a weekly
     trend report fed into sector memory (§7.1). The user's
     vision: *"try to understand trends that will shape economic
     activity."*
5. **Editorial governance for sources.** Move
   `news_domains.json` into a DB table with admin UI. Allow per-
   domain trust score (signal-to-noise rating that reweights
   relevance scoring).

---

## 10. Risk, Technical, and Critic agents

### How it works

**Risk:** deterministic — leverage, valuation risk, FCF support,
profile risks. LLM optional.

**Technical:** deterministic SMA / RSI / MACD / Bollinger.
Explicitly *not* a rating driver.

**Critic:** `critic_agent.py` runs a Risk Committee critique with
Claude Opus 4.7. Looks for inconsistencies and rating drift across
specialists.

### Limitations

1. **Risk and Technical have no LLM-grounded narrative.** They
   produce numbers; the user reads them; the PM cites them
   without nuance.
2. **Critic doesn't know about long-term memory.** It compares
   the *current* memo's specialists for internal consistency but
   doesn't compare against the prior memo or the company memory
   for *external* consistency ("you said the opposite three months
   ago — what changed?").

### Improvements

1. **Risk narrative pass.** Optional LLM step on top of the
   deterministic risk numbers — *"in plain English, what's the
   one risk that would invalidate this thesis?"* Output goes into
   the memo's Risk section.
2. **Critic reads memory.** Inject the company memory file +
   prior memo's mispricing thesis into the critic's context.
   Critic's job becomes: are we consistent with our prior view,
   and if not, do we explicitly explain the change?
3. **Technical signals optional in display.** A retail user
   who's a quant person will love it; a fundamentals user will
   tune it out. Add a setting toggle.

---

## 11. Screener (`/screener`)

### How it works

Three views: AI rank (`screener_scores`), factor rank
(`screener_metrics`), and a custom rule builder. AI factor scores
are computed nightly in
[`screener_service.compute_universe_scores`](backend/app/services/screener_service.py)
across 7 factors (quality, growth, valuation, earnings_momentum,
risk, macro_fit, catalyst), blended into `pm_score` with hardcoded
weights. Theme bias from a fixed `THEME_BIAS` dict (7 themes:
ai_infrastructure, falling_rates, sticky_inflation, etc.) applies
sector multipliers. Custom screener reads
[`screener_metrics`](backend/app/services/screener_metrics_service.py)
table directly (15 raw metrics).

### Limitations

1. **Filter intent ≠ filter result.** User: *"the screener needs
   to do better at actually selecting companies exposed to the
   given filter."* If the user asks for "AI-exposed names," the
   screener applies the `ai_infrastructure` theme bias — a
   sector-level multiplier — not a per-company exposure score.
   It returns NVDA and AMD because they're already top by other
   factors, but it misses the AMZN-type cross-sector exposure
   plays.
2. **Themes are hardcoded.** Seven themes is a tiny vocabulary.
3. **No exposure-weighted screening.** "Show me companies with
   30%+ revenue from data centers" is the kind of query no
   surface here can answer.

### Improvements

1. **Per-company theme exposure scores.** New table
   `theme_exposure(ticker, theme, score, evidence)`. Built from:
   - LLM extraction over the 10-K business description for top
     themes (AI, China, energy transition, GLP-1s, etc.).
   - Earnings-call mention frequency.
   - News-tag co-occurrence.
   Refreshed monthly. Custom screener gets a "theme exposure"
   filter alongside numeric metrics.
2. **Open theme vocabulary.** Replace the fixed `THEME_BIAS` with
   a learned one — same LLM extraction across the universe
   produces a vocabulary of ~50-100 themes; PM-curated to ~30
   that are investable.
3. **Natural-language screener.** Single prompt input: "show me
   profitable US-listed semis with falling capex intensity, beta
   under 1.5, and meaningful AI exposure." LLM translates to a
   rule chain over `screener_metrics` + `theme_exposure`. Existing
   custom screener UI surfaces the translation so the user can
   inspect/refine.
4. **Forward-looking metrics.** Today's screener uses trailing
   metrics. Add forward PE / PEG (from estimates) and analyst
   revision momentum. These are some of the strongest screening
   factors empirically.

---

## 12. Portfolio Builder (`/portfolio`)

### How it works

[`services/portfolio_service.py`](backend/app/services/portfolio_service.py)
wraps `build_portfolio()`, which scores candidates via
`candidate_score()`: weighted blend of pm_score (0.40),
quality (0.20), growth (0.10), macro_fit (0.15), risk (0.10),
earnings_momentum (0.05), minus a valuation penalty. Detects
scenario from `market_view` text (same regex keys as macro agent),
applies sector multipliers, selects top N respecting
`max_position_size` and `sector_cap`. Normalizes weights,
clips/redistributes.

### Limitations

1. **Output barely responds to user prompt.** User's words: *"when
   I used it, it was pretty much giving the same portfolio no
   matter what — make it responsive to the actual prompt."*
   Confirmed in code — the scoring weights are fixed, only the
   *scenario tag* shifts (from a 5-key regex), so two prompts that
   land on the same scenario produce nearly identical portfolios.
2. **No real prompt understanding.** "Build me a 10-stock
   portfolio for someone retiring in 5 years" should produce
   something fundamentally different from "build me 10 high-beta
   AI plays" — today they collide on the same scenario tag and
   produce similar holdings.
3. **No optimization.** Greedy selection with constraints; no
   risk parity, no max-Sharpe, no factor balancing.

### Improvements

1. **Prompt → portfolio brief.** New first step: pass the user's
   prompt to an LLM that emits a structured *brief*:
   ```
   horizon: 1y / 5y / 10y
   risk: conservative / balanced / aggressive
   themes: List[str]
   factor_tilts: {growth: 0.0-1.0, value: …, quality: …, momentum: …}
   sector_targets: {…}  // bias, not hard cap
   exclusions: [tickers, sectors]
   beta_target: float | None
   yield_target: float | None
   constraints: [text]  // e.g. "tax-efficient", "ESG-aware"
   ```
   The brief drives every downstream decision. Today's request
   schema is a tiny subset of this.
2. **Brief-driven scoring weights.** `candidate_score()` weights
   become brief-derived (growth weight from the growth tilt, etc.)
   instead of fixed.
3. **Two-stage selection.** Stage 1: filter universe by
   exclusions + sector targets + beta envelope → candidate set.
   Stage 2: optimize weights for the brief's objective (max-score
   subject to risk constraints, or risk-parity, or equal-weight
   per the brief). Use `cvxpy` or a simple QP — overhead trivial
   for ~100 names.
4. **PM rationale per holding.** For each holding in the output,
   one-sentence rationale tied to the brief: "Held NVDA at 8% to
   express the AI exposure target; trimmed to 8% (vs 12%
   unconstrained) because beta cap of 1.4 was binding."
5. **Show the working.** UI displays the inferred brief as an
   editable form *before* the portfolio runs. User sees how the
   prompt was understood, can fix it, re-run.
6. **A/B with the user.** Save the inferred brief alongside the
   portfolio. Periodically pull recent portfolios, ask the user
   to score 1-5 "did this match what you asked for?", feed back
   into the brief-extraction prompt.

---

## 13. Track Record (`/track-record`)

### How it works

`outcome_service.evaluate` runs nightly, computing realized 30 /
90 / 180 / 365-day returns vs. SPY for every memo. Stored in
`memo_outcomes`. The page displays the table.

### Limitations

1. **It's a scoreboard, not a learning loop.** The user's vision:
   *"i want to use this to help monitor the performance of each
   stock the model likes over time. this should be used to allow
   reinforcement learning to make the agents smarter over time."*
   Today nothing reads `memo_outcomes` back into agent prompts.
2. **No per-agent attribution.** When a memo is right or wrong, we
   can't point at which specialist's view drove it.
3. **No regime-conditional accuracy.** The model might be great in
   soft-landing regimes and terrible in recessions; today's table
   pools across all regimes.

### Improvements

1. **Per-memo retrospective field.** New post-mortem step on every
   memo that hits its 90-day mark: an LLM agent reads the original
   memo + the price action + recent news, and writes a one-page
   *what we got right / wrong / why*. Stored in `memo_postmortems`.
2. **Per-agent attribution.** When a postmortem judges a memo
   correct/incorrect, attribute partial credit/blame to each
   specialist's contribution (PM-judged from the original memo's
   `key_points` per agent). Build per-agent accuracy stats over
   time.
3. **Postmortems → memory.** Insights from postmortems append to:
   - `memory/companies/<T>.md` — name-specific lessons.
   - `memory/sectors/<sector>.md` — sector-level patterns.
   - `memory/pm/notes.md` — investing principles refinement.
   This is the reinforcement learning loop the user described —
   not RL in the gradient-descent sense, but in the supervised
   "feed prior outcomes back into the next decision" sense, which
   is what works for LLM-based agents today.
4. **Regime-conditional dashboards.** Track-record page filterable
   by macro regime at memo-creation time. Identifies systematic
   regime weaknesses (e.g., model overestimates growth in
   sticky-inflation environments).
5. **Calibration plot.** For each rating (Strong Buy → Strong
   Sell), plot expected vs. realized excess return distribution.
   A well-calibrated PM has Strong Buy realizations clearly higher
   than Buy realizations.
6. **Public-facing accountability.** Users care that the platform
   keeps score on itself. Link the track-record page from the
   landing surface so it's not a hidden tab.

---

## 14. Cross-cutting infrastructure

### 14.1 Memory architecture

Today: `memory/companies/<T>.md` per ticker. Going forward, a
multi-tier memory system:

| Memory file | Owner | Read by | Written by |
|---|---|---|---|
| `memory/pm/notes.md` | PM | every PM call (chat + synthesis) | PM (end-of-session append) |
| `memory/sectors/<sector>.md` | Sector agent | sector + filing + earnings agents | Sector agent + filing post-pass + earnings QoQ delta |
| `memory/companies/<T>.md` | Company-scoped | every specialist on that ticker | All specialists + filing post-pass |
| `memory/macro/notes.md` | Macro agent | Macro agent | Macro agent (weekly) |
| `prompts/macro_primer.md` | (system constant) | Macro agent | (manual) |
| `prompts/pm_identity.md` | (system constant) | PM | (manual) |

All files are markdown so the user can read / edit them. Total
memory injected per LLM call should stay under a sensible budget
(e.g., 8 KB) — when files grow, summarize and archive the tail.

### 14.2 Vector database for RAG

See §4.1. Single new infra dependency. Recommendation: pgvector
on the production Postgres, ChromaDB embedded for dev. Embedding
model `text-embedding-3-large`. Chunk store backs filings,
transcripts, news, and (eventually) the long-form memos
themselves.

### 14.3 Reinforcement / outcome feedback

See §13.3. Postmortems → memory is the spine. No gradient-based
RL is needed; the LLMs do the credit assignment via the
postmortem prompt.

### 14.4 UX wiring

Several specific items the user called out:

- **Memo Valuation section → DCF Lab link** (§1.4). Cheap.
- **Memo Comps section → Comps tab link.** Same pattern.
- **Memo Earnings section → Earnings tab link** (§3.2).
- **DCF Lab stacked assumption table** (§5.1).
- **DCF Lab red/green sensitivities** (§5.3).
- **DCF Lab live-vs-memo price overlay** (§5.6).
- **Portfolio Builder editable inferred-brief form** (§12.5).
- **Earnings tab speaker-segmented transcript view** (§3.2).

### 14.5 Cost discipline

Most of these improvements add LLM calls. Concretely:

| Improvement | Approx. extra cost / memo |
|---|---|
| Default-on deep research | +15-25% |
| Earnings multi-pass + QoQ delta | +0.05-0.10 USD |
| Filing diff agent | +0.02-0.05 USD |
| Mispricing thesis prompt | negligible (prompt change) |
| Vector retrieval calls | +embedding cost ~$0.0001/call |
| Portfolio brief extraction | +0.005 USD |
| Macro Fed-speech ingestion | +per-week, not per-memo |

Cap headline run cost at ~2x current; monitor via
`llm_call_logs` / cost reporting.

---

## 15. Phasing — what to build first

Ordered by **leverage / effort** (high leverage, low effort first):

### Phase A — the low-hanging UX + correctness wins (1-2 weeks)

1. ✅ Live intraday quote chain (shipped, `f048f3d`).
2. Memo section → tab navigation links (§1.4).
3. DCF Lab stacked assumption table (§5.1).
4. DCF Lab red/green sensitivities (§5.3).
5. PM mispricing-thesis prompt + memo schema field (§1.3, §2.3).
6. PM identity prompt upgrade (§2.4).

### Phase B — the brain-not-router PM (2-4 weeks)

7. `memory/pm/notes.md` + read on every PM call (§2.1).
8. Specialists as live tools in chat (`ask_sector`, etc.) (§2.2).
9. Memo-as-evidence in chat (§2.5).
10. Default-on deep research for premium tier (§1.2).

### Phase C — comps and screener that actually pick the right stocks (2-3 weeks)

11. Two-track comps (a priori + LLM exposure peers) (§6.1).
12. Per-company theme exposure scores (§11.1).
13. Natural-language screener (§11.3).
14. Brief-driven portfolio construction (§12.1-12.5).

### Phase D — RAG and the memory loop (3-5 weeks)

15. Vector DB + filing chunks (§4.1).
16. Filing → memory pipeline (§4.2).
17. Earnings structured schema + multi-pass + QoQ delta (§3.1, §3.3, §3.4).
18. Earnings tab UI (§3.2).
19. Sector memory file (§7.1).
20. Macro memory + Fed speech ingestion + macro primer (§8.1-8.3).

### Phase E — track record as a feedback loop (2-3 weeks)

21. Postmortem agent + table (§13.1).
22. Per-agent attribution (§13.2).
23. Postmortems → memory writes (§13.3).
24. Regime-conditional dashboards + calibration plots (§13.4-13.5).

### Phase F — news, social, catalysts (2-4 weeks)

25. Multi-source news ingestion (§9.1).
26. Catalyst calendar (§9.3).
27. Two-track social agent (§9.4).

---

## 16. Open questions

1. **Vector store choice.** pgvector vs. ChromaDB vs. Pinecone /
   Weaviate. pgvector is the lowest-friction choice given the
   existing Postgres dependency. Decision needed before §4.
2. **Embedding model.** OpenAI vs. Gemini vs. Cohere. Need to
   benchmark on filing chunks + earnings transcripts before
   committing.
3. **Per-segment comps for multi-business companies.** Worth the
   schema complexity? Or does a single hand-curated peer set
   capture 80% of the value for 20% of the work?
4. **Postmortem cadence.** 90 days post-memo is one pulse; should
   we also do 30-day "early read" postmortems that flag drift
   sooner?
5. **Public track record.** How much do we expose to logged-out
   visitors? Bragging-rights argument vs. selection-bias risk
   when a few bad calls dominate the visible window.
6. **Premium gating.** Phase B (deep research, live specialist
   calls in chat) is meaningfully more expensive per request.
   Decision needed: free tier capped at single-pass memos and
   cached chat? Pro tier unlocks deep research + live specialists?

---

## 17. Changelog

*(Sections move here as improvements ship. Each entry: date, wave
tag, what changed, where it landed.)*

- **2026-05-05 — `f048f3d`** — Live intraday quote chain.
  Decoupled valuation `current_price` from 7-day-cached profile.
  Adds `quote` capability (FMP → Tiingo → Polygon, 60s TTL).
  Fixes the NVDA-stale-by-a-week problem the founder reported.
  Section: §1.5, §5.6 (partially).
