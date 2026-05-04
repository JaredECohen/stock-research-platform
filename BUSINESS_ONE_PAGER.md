# MarketMosaic — Your AI Investment Committee

> Eight specialist agents write your stock memo. They keep updating it as filings, earnings, and news land. Then you ask any of them follow-up questions on top.
> Live at **[marketmosaic.ai](https://marketmosaic.ai)**.

---

## The product, in one paragraph

Type a ticker. A team of AI agents — sector analyst, earnings analyst, filings analyst, valuation analyst, comps analyst, macro analyst, risk analyst, technical analyst — fans out, runs in parallel, and produces a structured investment memo with thesis, rating, valuation, comps, risks, and catalysts. A cross-family **Risk Committee critic** challenges the draft before the **PM agent** synthesizes the final view. The memo is **versioned and self-updating**: when a new 10-K hits EDGAR or a material news alert fires, the affected agents re-run and patch the memo in place, with full revision lineage so you can see exactly what changed and why. On top of every memo, you can **chat with the committee** — ask the valuation analyst why WACC is 9.2%, ask the risk analyst what would invalidate the thesis, ask the PM how this name stacks up against your portfolio. This is not ChatGPT-for-stocks. It is the workflow of an institutional research team, automated.

## The user

**Self-directed retail investors with $250k+ in a brokerage account who actively pick single stocks.** They are technical enough to read a DCF, sophisticated enough to demand citations and a balanced bear case, and skeptical of both shallow free tools and hallucinating LLM chat. They cannot justify $20–30k/year for a Bloomberg, FactSet, or Visible Alpha terminal but will pay for tools that respect their time and rigor — Stratechery, Seeking Alpha Premium, premium FinTwit subscriptions. Roughly **2–3M Americans** fit this profile: the demographic that fueled the 2020–2024 retail brokerage explosion, now graduating from "I bought NVDA" to "I want to defend why I bought NVDA."

**A representative user — Alex, 36.** Senior software engineer at a public tech company. $450k in a self-directed Schwab account, 8–15 single names plus ETFs. Reads 10-Ks on weekends. Has a Bloomberg.com tab open and refuses to pay $24k/year for the terminal. Got burned once when ChatGPT cited a revenue figure that was a year stale, and won't trust generic LLM chat for numbers since. Pays for Stratechery, lapsed his Seeking Alpha Premium, follows three FinTwit accounts religiously.

## The problem

Alex's three options today, none of them good:

- **Free tools** (Yahoo, FinViz, Seeking Alpha free) — shallow, fragmented, no synthesis across data. One 10-K takes a Saturday.
- **Premium terminals** (Bloomberg, FactSet, Visible Alpha) — out of reach at $20–30k+/year, sold to institutions.
- **Generic LLM chat** (ChatGPT, Claude, Gemini) — hallucinates numbers, can't run a real DCF, has no retrieval over filings, no critic discipline, and **forgets everything between sessions**. It produces a paragraph, not a memo, and definitely doesn't keep it updated.

The missing product is the *workflow* of an institutional research team — fan out, synthesize, critique, persist, update — at a consumer price.

## The economics

**Pricing.** Two tiers. **Free** — 5 memos/day. **Pro $29/mo** — unlimited memos, follow-up chat, agent-ranked screener, full DCF lab. Live data is used for every user on every tier; there is no "demo only" path on the consumer surface.

**Cost to serve — token math for a single cold memo:**

| Stage | Model class | Tokens (in / out) | Cost |
|---|---|---|---|
| 8 specialist agents + planner (sector, earnings, filings, valuation, comps, macro, risk, technical) | cheap (~$1 / $5 per Mtok) | ~25k / 10k | ~$0.08 |
| Risk Committee critic — Anthropic Opus 4.7, cross-family | strong (~$15 / $75 per Mtok) | ~10k / 2k | ~$0.30 |
| PM synthesis | strong (~$3 / $15 per Mtok) | ~12k / 2.5k | ~$0.07 |
| **Cold memo total** |   | **~50k / 15k** | **~$0.45** |

A **news-patch update** (only affected agents, critic skipped) is **~$0.05**. A **cohort-cache hit** (e.g., NVDA when another user just ran it) is **~$0** — see [`cache/snapshots.py`](backend/app/cache/snapshots.py).

**One Pro user-month, realistic mix:**

| Activity | Volume | Unit cost | Subtotal |
|---|---|---|---|
| Memo views (mostly popular tickers, cohort-cached) | 30 (5 cold + 25 cached) | $0.45 / $0 | $2.25 |
| Chat turns grounded in memos | 50 | ~$0.04 | $2.00 |
| News-patch updates fired by monitoring loops | 8 | ~$0.05 | $0.40 |
| Screener + portfolio runs (pre-computed factor scores) | 15 | ~$0.03 | $0.45 |
| DCF roll-forward + outcome scoring (background) | — | — | $0.20 |
| **Variable cost / user-month** | | | **~$5.30** |

**Fixed costs (infrastructure + data licensing):**

| Line | Cost / mo | Notes |
|---|---|---|
| FMP API (fundamentals, prices, ratios, estimates, news) | $100 | Per-account flat fee; rate-limit-driven, not seat-based |
| Alpha Vantage (transcripts + news/sentiment fallback) | $50 | Premium plan; same — flat fee, scales modestly with traffic |
| Render (web + Postgres + worker) | $15 today; ~$75 at scale | Currently Starter web + basic-256mb Postgres ($14/mo); Standard tier (~$75/mo) once concurrent users grow |
| FRED + SEC EDGAR | $0 | Free government APIs |
| **Total fixed / mo** | **~$165 today → ~$225 at scale** |   |

These costs amortize across the entire user base. **At 100 Pro users**, fixed cost is $1.65/user/mo; **at 1,000 users**, $0.16; **at 10,000+**, negligible. Data-provider rates are flat fees on calls/min, not per-seat — meaningful only at the very low and very high ends of the user count.

**Gross margin at launch (~50 users):** revenue $29 − variable $5.30 − fixed $3.30 ≈ **65%**.
**Gross margin at 1,000 users:** revenue $29 − variable $3.00 − fixed $0.23 ≈ **89%**.
**Gross margin at 10,000+ users:** revenue $29 − variable $2.00 − fixed $0.02 ≈ **93%**.

**Where it breaks.** Long-tail ticker abuse. A user who runs 80 cold memos on small-cap or international names that no one else queries gets no cohort-cache benefit, costs ~$36 against $29 revenue. **Mitigations already in the codebase:** soft cap of ~30 cold memos / Pro user-month, forced cheap-model path on long-tail tickers, BM25 retrieval (not full-doc prompting), and the news-patch path that skips the Opus critic on incremental updates. The Opus critic is the single biggest cost line — every 25% price cut on frontier models drops cost-to-serve by ~$0.07/memo, and frontier prices have fallen ~90% in 24 months.

**Margin gets better as the user base grows — and this is the central unit-economics story.** Two compounding effects: **(1) fixed costs amortize** — $165/mo of data + infra spread across 100 users is $1.65/user, across 10k users is $0.02/user; **(2) variable cost falls because memos are cached across users.** A single $0.45 cold memo on NVDA serves *every subsequent user* who views NVDA for the cache window — the LLM call happens once, the structured output is reused. At 1,000 users overlapping on the top 50 tickers, cohort hit rate climbs to ~70%; cold runs drop from 5 to ~2 / user / month and variable cost falls from $5.30 to ~$3. At 10,000+ users, the top 100 tickers stay continuously warm and the marginal Pro user mostly reads pre-computed memos. Generic LLM-wrapper competitors pay full token cost on every user's every query, every time. Our caching architecture turns concentration in retail attention (everyone wants NVDA, AAPL, MSFT) into a structural cost advantage that compounds with scale.

## Why these technical choices map to the business

- **Eight specialist agents emitting Pydantic JSON, not prose** ([`schemas.py`](backend/app/schemas.py)) — every output is bounded, parseable, and renderable as UI without re-prompting. No streaming-text waste, every dollar of tokens turns into a structured field.
- **Cheap model for extraction, strong model only for PM synth + critic** ([`agents/llm.py`](backend/app/agents/llm.py)) — the quality/cost frontier. We don't pay Opus prices to extract a revenue number.
- **Three-tier cache (cold / warm / hot) with cohort lineage invalidation** ([`cache/snapshots.py`](backend/app/cache/snapshots.py)) — what makes 82% margin survive at scale. Popular tickers amortize across the user base; lineage cascades guarantee freshness when a 10-K drops.
- **News-patch path skips the expensive critic** ([`services/update_orchestrator.py`](backend/app/services/update_orchestrator.py)) — incremental updates are 10× cheaper than full re-runs. This is what lets us promise *always-fresh memos* without burning the margin.
- **Real DCF + real comps engines** ([`finance/dcf.py`](backend/app/finance/dcf.py), [`finance/comps.py`](backend/app/finance/comps.py)) — the moat against a generic LLM wrapper. The rolling DCF updater (capped ±20%/cycle, with audit trail) is what an analyst would actually do; nobody is reproducing this in a weekend.
- **Realized-outcome tracking** ([`services/outcome_service.py`](backend/app/services/outcome_service.py)) — every memo's 30 / 90 / 180 / 365-day forward returns are scored and folded into the company's persistent memory. The product gets *better at picking* over time, not just *more verbose*.
- **Demo-mode fallback for every provider call** ([`providers/demo_provider.py`](backend/app/providers/demo_provider.py)) — the entire app demos zero-key. Free-tier abuse caps at near-zero cost.

## Why now

LLM price/quality crossed the threshold for structured agentic workflows in 2024–2025. Cross-family routing (OpenAI for synthesis, Anthropic for critic, Gemini for long-doc) and aggressive prompt caching make a real research committee feasible at $29/mo — impossible 18 months ago. Premium research's entire moat was *"we have a research team and you don't"*. That moat just got disintermediated.

## What's deployed

**[marketmosaic.ai](https://marketmosaic.ai)** — eight agents, seven workflows (chat, single-stock memo, screener, portfolio builder, DCF lab, comps, macro), versioned memos with full lineage, rolling DCFs, realized-outcome dashboard, three-tier cache, four background monitoring loops, and a track record. Demo mode runs zero-key against a 28-name programmatic dataset; live mode swaps in FMP / Alpha Vantage / FRED / Polygon / Tiingo / SEC EDGAR / OpenAI / Anthropic / Gemini transparently. Built solo.

---

*MarketMosaic is for investment research and education only. It does not provide personalized financial advice. Model portfolios are illustrative scenario constructions.*
