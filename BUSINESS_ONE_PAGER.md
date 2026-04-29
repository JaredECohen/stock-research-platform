# MarketMosaic: Your AI Investment Committee

> Multi-agent equity research and portfolio management for self-directed investors.
> A sector analyst, earnings analyst, filings analyst, valuation analyst, macro analyst,
> risk analyst, and PM orchestrator — collaborating on every memo, screen, and portfolio.

---

## User

Self-directed investors with $100k–$5M+ in equities who actively research single stocks but cannot
justify $20-30k+/year for an institutional research terminal. They are technical enough to read a
DCF, sophisticated enough to want a sector framework, and skeptical enough to demand cited sources
and a critic review.

## Problem

Three options today, none ideal:

- **Free tools** (Yahoo, FinViz, Seeking Alpha): shallow, fragmented, no synthesis, low quality control.
- **Premium terminals** (Bloomberg, FactSet, Visible Alpha): out of reach at \$20-30k+ / year.
- **LLM chat** (ChatGPT, Gemini, Claude): hallucinates, lacks structured outputs, lacks retrieval,
  cannot run a DCF, and has no risk-committee discipline.

There is no product that gives a self-directed investor the *workflow* of an institutional
research team — sector / earnings / valuation / macro / risk — at consumer price points.

## Product

MarketMosaic is an AI-native research terminal:

1. **Ask the PM (chat)** — orchestrator routes to specialist agents, returns structured memos.
2. **Stock research** — full investment memo with sector, earnings, filings, valuation, comps,
   macro, and risk-committee critique.
3. **Agent-ranked screener** — themes (falling rates, AI infrastructure, defensive, sticky inflation,
   margin expansion, reasonable-valuation growth) with PM conviction scores.
4. **DCF Lab** — editable assumptions, base/bull/bear scenarios, sensitivity tables.
5. **Comps** — peer set + premium/discount + interpretation.
6. **Macro** — scenario-driven sector mapping with FRED-compatible time series.
7. **Portfolio Builder** — translate a market view into a diversified scenario portfolio with
   sector caps, max position size, exclusions.

Every output is framed as research/education. No buy/sell imperatives. Risk Committee challenges
the draft memo before the PM publishes the final view.

## Economics

| Tier      | Price        | LLM/API cost / user-month | Gross margin   |
|-----------|--------------|---------------------------|----------------|
| Free      | \$0          | \$0–\$0.50                | ≈ 100% (subsidized) |
| Pro       | \$29 / mo    | \$1–\$3                   | 85–95%         |
| Premium   | \$99 / mo    | \$3–\$8                   | 90–95%         |
| Advisor   | \$299 / mo   | \$5–\$15                  | 90%+           |

## Cost to serve

Built into this codebase:

- **Cached company data** — `services/data_service.py` memoizes provider results.
- **Pre-computed screener scores** — `seed_demo_data.py::seed_screener_scores` runs at startup,
  not per-request.
- **Cheap model for extraction, strong model for synthesis** — `agents/llm.py::route="cheap" |
  "strong"` routes between `OPENAI_CHEAP_MODEL` and `OPENAI_STRONG_MODEL`.
- **Demo / fallback mode** — every provider call cleanly degrades to local fixtures, capping
  cost at \$0 for free-tier overflow.
- **Document retrieval** (`services/retrieval_service.py`) instead of full-document prompting —
  caps tokens per agent.
- **Pydantic structured outputs** — JSON-mode responses, not streaming text.

## Pricing

\$29 / \$99 / \$299 monthly for Pro / Premium / Advisor tiers. Annual discount 15%. 14-day free
trial on Pro+. API access on Advisor tier (rate-limited).

## Margin estimate

- **Variable margin per active Pro user:** \$26–\$28 of \$29 (90%+).
- **Variable margin per active Premium user:** \$91–\$96 of \$99 (92%+).
- Fixed infra (compute, DB, hosting): \~\$2k–\$5k / month at <10k MAU; scales gracefully to
  6-figure MAU on a managed Postgres + container platform.
- Data licensing is the only meaningful gross-margin lever beyond LLM spend; MarketMosaic's
  multi-provider abstraction lets us swap providers when contracts move.

## Why technical choices support the business

- **Multi-agent specialization with structured outputs** is the cost-control mechanism: every
  agent returns Pydantic JSON, so we never pay for verbose prose, and we can deterministically
  render rich UIs from the same payload.
- **Provider abstraction with demo fallback** lets us run a credible product *before* paying for
  data, cap costs during free-tier abuse, and migrate providers without UI work.
- **Risk committee critic** is both an LLM-quality gate and a *legal-risk* gate: it enforces
  the research/education framing on every memo.
- **Editable DCF + comps** give us defensible competitive moat against generic LLM apps —
  *real* finance, not just narrative.
- **Demo dataset shipped in-repo** means the product is fully demoable on day 1, every PM /
  investor / partner can click through without provisioning.

## Why now

- LLM cost / quality crossed a usable threshold for structured agent workflows in 2024–25.
- Self-directed retail flow is at all-time highs and growing; same demographic that fueled the
  retail brokerage explosion now wants institutional-grade *research*.
- Provider APIs (FMP, Alpha Vantage, FRED, Polygon, Tiingo) are mature and affordable.
- Data-licensing precedents exist for AI synthesis use cases (Bloomberg AI, Refinitiv AI).

## Roadmap

- **Now:** v0 demo with 28-stock universe, demo + live providers, 8-agent committee, DCF / comps /
  screener / portfolio builder / macro.
- **Q1 next:** vector RAG over filings + transcripts, custom universes, earnings-preview mode.
- **Q2:** backtest engine for portfolios (per-period replay, drawdown / Sharpe / regime).
- **Q3:** alerts pipeline (thesis-breaker monitoring, news + filings, mobile push).
- **Q4:** advisor features (multi-account, white-label, compliance pack), API access, ETF coverage.

---

*MarketMosaic is for investment research and education only and does not provide personalized
financial advice.*
