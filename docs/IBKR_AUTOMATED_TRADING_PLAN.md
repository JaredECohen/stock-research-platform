# IBKR Automated Trading Platform — Stack & Architecture Plan

> **Scope**: Plan document for a separate repo. Captures the data, infrastructure, risk, and operational stack for a real-time automated trading system running on Interactive Brokers across equities, ETFs, options, and crypto. Implementation will live in its own codebase.

> **Critical mindset shift from a research platform**: research platforms can be wrong; trading platforms can lose money. Every architectural decision below is biased toward **safety, observability, and recoverability** over feature breadth. If a tradeoff exists between "fancy" and "safe," pick safe.

---

## 1. Product Goals

A 24/7 automated trading system that:
1. **Ingests real-time market data** for equities, ETFs, options, and crypto.
2. **Runs strategies** that emit signals → orders.
3. **Routes orders to IBKR** with pre-trade risk checks.
4. **Monitors positions** and manages exits, stops, and corporate-action edge cases.
5. **Records everything** — every tick consumed, every signal, every order, every fill — for postmortem and compliance.
6. **Survives failures** — broker disconnects, partial fills, network drops, daily TWS restarts — without leaking risk.

**Non-goals**:
- HFT / sub-millisecond latency (IBKR is not the right venue).
- Discretionary manual trading (different UX entirely).
- Strategy research / backtesting at scale (use a separate research repo; this is execution).

---

## 2. IBKR-Specific Realities

### What IBKR is great at
- One account → equities, ETFs, options, futures, FX, bonds, crypto (limited — see below), international.
- Best-in-class smart routing (IB SmartRouting, Adaptive algos, Iceberg, VWAP/TWAP).
- Margin rates 3–5× cheaper than retail competitors.
- Paper trading account is API-identical to live — same data subs, same routing logic.
- Data fees often **waived** if you generate $30+/mo in commissions.

### What IBKR is annoying at
- **Always-on requirement**: TWS or IB Gateway must be running, with a forced daily auto-restart and weekly logout. You **must** plan for this (or use Client Portal API to avoid it).
- **Pacing limits**: ~50 messages/sec on most endpoints, stricter on historical data.
- **Stateful API**: order IDs, reqIds — manual lifecycle management. `ib_async` smooths most of this.
- **Crypto coverage is thin**: BTC, ETH, LTC, BCH only, via Paxos, US-only. **For broader crypto, you must integrate a separate exchange (Coinbase, Kraken, etc.).**
- **Options chain endpoint is slow** — pulling full chain across many strikes/expiries takes seconds, not ms. Cache aggressively.
- **No webhooks for fills** — you must subscribe and stay connected.

### API options
| API | Best for | Tradeoffs |
|---|---|---|
| **TWS API (native)** | Full feature set, lowest latency | Requires TWS or IB Gateway running |
| **`ib_async`** (Python wrapper) | Most production trading systems | Maintenance-mode community fork of `ib_insync`. Pleasant async/await. |
| **Client Portal Web API** | Long-running services without gateway | OAuth, REST + WebSocket, slightly fewer features |
| **FIX** | Institutional / co-located | Overkill unless you're a registered firm |

**Recommendation**: Start with **`ib_async`** + IB Gateway (lighter than TWS, headless). Migrate to Client Portal API only if gateway babysitting becomes a real ops burden.

---

## 3. Data Stack

### Real-time market data

| Asset class | Source | Cost | Notes |
|---|---|---|---|
| **US equities + ETFs (NBBO)** | IBKR US Securities Snapshot Bundle | ~$10/mo (waived w/ commissions) | Real-time top-of-book. |
| **US options (OPRA)** | IBKR OPRA Top of Book | ~$1.50/mo (waived w/ commissions) | Same OPRA feed Polygon resells at $199. |
| **Level 2 / depth** | IBKR depth bundles | varies | Only if your strategy uses book depth. |
| **Crypto (BTC/ETH/LTC/BCH)** | IBKR Paxos | included | US-only, limited universe. |
| **Crypto (broader)** | **Coinbase Advanced Trade** or **Kraken** WS | $0 (free public WS) | For altcoins, perps. Coinbase: native USD on-ramp. Kraken: deeper book on majors. |
| **Crypto (perps/derivs)** | **Bybit / Binance** WS | $0 | Only if you need perpetual futures. Geo restrictions matter — check legality. |

**Total real-time data cost (US): ~$15/mo**, often $0 if active. **This is the killer feature of running on IBKR vs. stitching Polygon + a separate broker** ($300+/mo for the same data).

### Reference / historical data
- **IBKR `reqHistoricalData`** for short historical pulls. Pacing-limited; not for bulk.
- **Bulk historical** (for strategy R&D) lives in your **research repo**, not here. Keep concerns separate.
- **Corporate actions** (splits, dividends, mergers, spin-offs) — IBKR pushes events, but **build your own validation layer**. Bad CA handling has killed more strategies than bad alpha.
- **Symbology cache** — IBKR uses `conId` as the canonical contract ID. Cache `symbol → conId` mappings; resolving on every order is wasteful and pacing-limited.

---

## 4. Architecture Overview

### Service decomposition

```
ibkr-trading-platform/
├── services/
│   ├── gateway/              # IB Gateway in Docker (or external)
│   ├── market-data/          # subscribes to IBKR + crypto exchanges, normalizes ticks
│   ├── strategy-runner/      # runs strategies; emits signals
│   ├── risk-engine/          # pre-trade checks; can BLOCK orders
│   ├── order-manager/        # signal → order; tracks lifecycle; handles fills/cancels/rejects
│   ├── position-tracker/     # source of truth for current positions, P&L, exposures
│   ├── reconciler/           # periodic IBKR account state vs. our state diff + alarm
│   ├── api/                  # control plane: start/stop strategies, view state
│   └── web/                  # operator UI (NOT a trading UI for users — this is for you)
├── shared/
│   ├── proto/                # message schemas (protobuf or pydantic)
│   ├── models/               # SQLAlchemy
│   └── lib/                  # logging, metrics, retry, etc.
└── infra/
    ├── docker-compose.yml
    └── ...
```

**Why service decomposition** (vs. a monolith): in a monolith, a bug in the strategy can corrupt position state. In a decomposed system, the **risk engine has veto power** the strategy cannot bypass, and the **position tracker is the only writer to the canonical positions table**. Defense in depth.

### Message bus
- **NATS** or **Redis Streams** for inter-service messaging. Avoid Kafka unless you have throughput needs > 100k msg/sec (you don't).
- Topics: `tick.{exchange}.{symbol}`, `signal.{strategy_id}`, `order.command`, `order.event`, `fill`, `position.update`, `risk.alert`.
- All messages are **persisted** for replay / audit.

### Language & framework
- **Python 3.12+** for strategy/risk/orchestration (faster iteration, mature ecosystem, fine for non-HFT).
- **Rust** for any hot-path component if you ever measure a Python bottleneck (don't preemptively rewrite).
- **`ib_async`** for IBKR connectivity.
- **`ccxt` or native exchange WS clients** for crypto.
- **FastAPI** for the operator API.
- **PostgreSQL** for state, orders, fills, positions.
- **Redis** for hot caches (last quote, working orders, sym→conId).
- **Parquet on S3** for long-term tick archival.

---

## 5. Core Components

### 5.1 Gateway Service
- **IB Gateway** in a Docker container with `IBC` (IBKR's auto-login helper) to handle the daily restart.
- Healthcheck endpoint: did we get a heartbeat from IBKR in the last N seconds?
- **Auto-reconnect** logic with backoff. **Never assume the connection is up.**
- The market-data and order-manager services connect to gateway via TCP.

### 5.2 Market Data Service
- Subscribes to IBKR streams for all symbols in the strategy universe.
- For crypto outside IBKR's coverage, opens WS to Coinbase / Kraken.
- **Normalizes** ticks to a common schema: `{venue, symbol, type (trade|quote|bar), ts, bid, ask, last, size, ...}`.
- Publishes to NATS topic `tick.{venue}.{symbol}`.
- Persists every tick to Parquet (rolling daily files) for replay / postmortem.
- Tracks **subscription health** per symbol — if no tick in N seconds during market hours, alarm.
- Pacing-aware: respects IBKR's 50 msg/s limits, batches subscriptions.

### 5.3 Strategy Runner
- Loads strategies from a registry. Each strategy is a class:
  ```python
  class Strategy:
      def on_tick(self, tick): ...
      def on_bar(self, bar): ...
      def on_fill(self, fill): ...
      def on_position_update(self, position): ...
  ```
- **One strategy per process** by default (isolation). Or co-host a small number if they share state.
- Emits **signals**, not orders directly: `{strategy_id, symbol, side, qty, order_type, urgency, intent}`.
- Risk engine consumes signals before they become orders.
- Strategies are **stateless across restart** unless they explicitly persist state to Postgres. Keep the surface small.

### 5.4 Risk Engine ⚠️ Most important component
**This service has veto power on every order.** No order reaches IBKR without passing here.

Pre-trade checks (every order):
- **Position limits** — per symbol, per asset class, per strategy.
- **Order size sanity** — single-order notional cap (catch fat fingers / runaway loops).
- **Daily loss limit** — kill switch if drawdown exceeds threshold.
- **Order rate limit** — strategy can't send >N orders/sec.
- **Buying power check** — don't send orders that will reject anyway.
- **Symbol whitelist / blacklist** — strategy can only trade approved symbols.
- **Volatility circuit-breaker** — if last-N-min vol > threshold, block new entries.
- **Trading hours** — don't fire equity orders at 3am unless it's an extended-hours order with explicit flag.
- **Earnings / event blackout** (optional) — block opens within X days of earnings.
- **Locate check for shorts** — IBKR enforces this, but check first to avoid rejects.

Each check returns `(pass/fail, reason)`. Failed orders are **logged + alerted**, not silently dropped.

**Global kill switch**: a single env var (or Redis key) flips and the risk engine rejects every order regardless of source. Use it during incidents.

### 5.5 Order Manager
- Receives approved orders from risk engine.
- Translates to IBKR `Order` objects (stocks, options, multi-leg combos, crypto).
- Submits via `ib_async`.
- Tracks **order lifecycle states**: `pending → submitted → working → partially_filled → filled | canceled | rejected`.
- Handles:
  - Partial fills (update position incrementally).
  - Reconnects (re-query open orders on startup; reconcile with our state).
  - Cancels (with timeout — IBKR can be slow; if cancel doesn't confirm in N seconds, alert).
  - Rejects (parse reason, route to alerting; don't auto-retry without explicit retry logic).
- All state changes published to `order.event` topic.

### 5.6 Position Tracker
- **Single source of truth** for current positions.
- Updates from: fills (incremental), reconciliation jobs (full snapshot from IBKR every N min).
- Computes: avg cost, unrealized P&L, realized P&L, exposure, Greeks (for options).
- Exposes a query API for strategies and operator UI.
- Stores history in Postgres for reporting.

### 5.7 Reconciler
- Every N minutes (or on-demand): query IBKR account → diff against our positions table.
- **Any drift = page**. Position drift means either a bug in our fill handling or an order placed outside our system. Both demand attention.
- On startup: full reconciliation **before** any strategy is allowed to fire orders.

### 5.8 Operator UI
- **For you, not customers.** This is a control panel, not a product.
- Views:
  - **Dashboard**: P&L (today, MTD, YTD), open positions, live orders, system health.
  - **Strategies**: start/stop/pause each strategy, recent signals, parameter overrides.
  - **Orders**: searchable log, filter by status / symbol / strategy.
  - **Risk**: kill switch, current limit usage, recent rejections.
  - **Logs**: tail recent events from any service.
  - **Reconciliation**: most recent diff results, last reconciliation time per account.

---

## 6. Asset-Class Specifics

### Equities & ETFs
- Most straightforward. Use SmartRouting (default).
- Decide: regular hours only, or extended-hours-aware? If extended, pass `outsideRth=True` and adjust risk checks.
- Short selling: handle locate failures (IBKR HTB list); cache HTB symbols.

### Options
- Use OCC symbology via IBKR `conId`.
- **Multi-leg orders** (combos): submit as a single combo order, not legged separately — better fills, atomic execution.
- Greek exposure tracking: pull from IBKR or compute via `py_vollib` for risk aggregation.
- **Assignment risk**: monitor short ITM options near expiry. Auto-close before expiry unless explicitly held for assignment.
- **Pin risk**: be careful with options that close near strike on expiry.
- **Early exercise**: short calls on dividend-paying stocks before ex-date — manage explicitly.

### Crypto via IBKR (Paxos)
- BTC, ETH, LTC, BCH only. US-only.
- Order types: Market, Limit, Stop, Stop-Limit.
- 24/7 market — your services must run 24/7.
- Settles in USD. No on-chain wallet — IBKR is custodian.

### Crypto outside IBKR
- **Don't try to unify execution under IBKR's API for non-IBKR coins.** Keep crypto-exchange execution as a separate adapter.
- Common pattern: `OrderManager` has multiple `Broker` adapters (`IBKRBroker`, `CoinbaseBroker`, `KrakenBroker`). Strategy emits signals tagged with venue → router picks adapter.
- **Funding & on-ramp**: separate concern. Manual ACH / wire to exchange; track balances; alert on low cash.
- **Self-custody vs exchange custody**: for automated trading, exchange custody is required (you can't sign with a hardware wallet from a service). Accept the counterparty risk consciously.

---

## 7. State, Storage, and Recovery

### Storage layers

| Data | Store | Retention |
|---|---|---|
| Last quote / hot state | Redis | TTL ~1s for ticks |
| Working orders | Postgres + Redis cache | Until terminal state |
| Order history, fills | Postgres | Forever |
| Positions (current + history) | Postgres | Forever |
| Risk events (rejections, kills) | Postgres | Forever |
| Tick archive | Parquet on S3 | 1 year+ for compliance |
| Logs | Loki / CloudWatch | 90 days |
| Metrics | Prometheus → long-term in Mimir/VictoriaMetrics | 1 year |

### Recovery scenarios — design for them

1. **TWS daily restart** (every ~24h): IBC handles login; gateway service detects disconnect, pauses strategies until reconnected, then reconciles before resuming.
2. **Service crash mid-order**: order-manager queries IBKR open orders on startup; reconciles state; resumes tracking.
3. **Network partition**: strategies pause if no tick in N seconds; risk engine refuses new orders if positions are stale.
4. **Database failure**: strategies and order manager halt (cannot persist state safely). Operator paged.
5. **Drift detected by reconciler**: kill switch flips automatically; operator must inspect.
6. **Bad data tick**: outlier filter (|tick - last| > N stddev) drops the tick rather than reacting to it. Prefer false drops over reacting to a bad print.

---

## 8. Backtesting & Strategy Development

**Don't build deep backtesting in this repo.** Use a separate research repo (or your existing `stock-research-platform`) for that. This repo's job is to **execute** strategies, not develop them.

But you do need:
- **Paper trading mode** — same code, IBKR paper account, identical behavior. Run every strategy on paper for at least N days before going live.
- **Replay mode** — feed historical ticks (from your tick archive) through the strategy to validate behavior without hitting any broker. Useful for postmortem after a bad day.
- **Shadow mode** — run new strategy alongside live ones, generate signals, but route to a "shadow" order manager that logs what would happen without sending to IBKR. Run for days, then promote.

The promotion path: **research → backtest → paper trade → shadow live → live with small size → live full size**.

Skipping steps is how people lose money.

---

## 9. Observability

### Metrics (Prometheus)
- Tick rate per symbol/venue.
- Order submission latency (signal → IBKR ack).
- Fill rate.
- Reject rate (with reason labels).
- Open orders count.
- Position count + total exposure (gross, net).
- P&L (intraday, daily, MTD).
- Risk check pass/fail counts.
- Strategy "alive" heartbeats.
- IBKR connection state.

### Alerts (page immediately)
- IBKR disconnected for > 30s during market hours.
- Reconciliation drift detected.
- Daily loss limit hit (kill switch flipped).
- Risk engine rejecting >X% of orders (likely bug or runaway strategy).
- Strategy crash.
- No ticks for N seconds on subscribed symbol during market hours.
- Order submission latency p99 > threshold (broker-side issue).

### Alerts (Slack / email, no page)
- Individual order reject.
- Strategy paused.
- Reconciliation completed (daily summary).
- Any non-zero P&L day's summary at close.

### Logs
- Structured JSON, every service.
- Every order: full lifecycle log line at submission, ack, fills, terminal state.
- Every risk decision: pass/fail + reason.
- Every reconciliation result.

---

## 10. Compliance & Operational Considerations

- **Pattern Day Trader (PDT)** rules apply if account < $25k. Track day trades; warn if approaching limit.
- **Wash sale tracking** (taxes) — IBKR reports this; don't reinvent, but know it exists.
- **Form 1099 / tax reporting** — IBKR provides; review annually.
- **Self vs. firm**: If you ever trade on behalf of others, you become a registered investment advisor. Don't drift into this accidentally.
- **Logs as evidence**: in the event of a dispute with IBKR or a regulator, your tick archive + order log + risk log is your defense. Keep them.
- **Disaster recovery**: at minimum, weekly Postgres backup to S3. Test restore quarterly.
- **Secrets**: IBKR credentials, exchange API keys. Use a secrets manager (Doppler / AWS Secrets Manager). Never in env files committed to git.
- **Two-machine rule** (mature setup): primary trading box + warm standby on a different cloud region. Manual failover is fine for non-HFT.

---

## 11. Phased Roadmap

### Phase 0 — Account & Data Setup (1 week)
- [ ] Open IBKR account (live + paper).
- [ ] Subscribe to required market data bundles.
- [ ] Set up IB Gateway with IBC for auto-login.
- [ ] Verify connectivity from `ib_async`.
- [ ] Open accounts at any crypto exchanges in scope (Coinbase, Kraken).

### Phase 1 — Foundation (3–4 weeks)
- [ ] Repo scaffold, Docker Compose, Postgres, Redis, NATS.
- [ ] Gateway service with auto-reconnect.
- [ ] Market-data service: IBKR subs + tick normalization + persistence.
- [ ] Position tracker (read from IBKR; no orders yet).
- [ ] Operator UI scaffold.

### Phase 2 — Order Path (3–4 weeks)
- [ ] Order manager: submit equity orders, track lifecycle.
- [ ] Risk engine: position limits, size caps, kill switch.
- [ ] Reconciler: scheduled + on-startup.
- [ ] **Paper trading end-to-end**: a hardcoded "buy 1 SPY at market" works through the full pipeline.

### Phase 3 — Strategies (4–6 weeks)
- [ ] Strategy interface + registry.
- [ ] One simple strategy (e.g., momentum on a 5-symbol universe). Run in paper for 2+ weeks.
- [ ] Replay mode for postmortem.
- [ ] Operator UI: start/stop, view signals, view orders.

### Phase 4 — Multi-Asset (4–6 weeks)
- [ ] Options support (single leg + combos).
- [ ] Crypto via IBKR (Paxos).
- [ ] External crypto adapter (Coinbase or Kraken).
- [ ] Asset-specific risk rules (assignment risk, locate checks, etc.).

### Phase 5 — Production Hardening (ongoing)
- [ ] Full alerting (PagerDuty / Opsgenie).
- [ ] Disaster recovery drills.
- [ ] Shadow mode for new strategies.
- [ ] Two-machine standby.
- [ ] Annual security review.

**Total to first live small-size strategy: ~3–4 months solo.** Going from "first live strategy" to "I trust this with real size" is another 3–6 months of paper + small-size + observation.

---

## 12. Cost Summary

### Monthly recurring (data + infra)

| Item | Cost |
|---|---|
| IBKR US Securities Snapshot Bundle | ~$10 (waived if active) |
| IBKR OPRA Top of Book | ~$1.50 (waived if active) |
| IBKR Crypto (Paxos) | included |
| Coinbase / Kraken WS data | $0 |
| VPS (dedicated, low-latency to NY4 if equities) | $50–$150 |
| Managed Postgres | ~$25 |
| Managed Redis | ~$10 |
| NATS (self-hosted) | included w/ VPS |
| S3 / R2 (tick archive) | ~$20 (grows over time) |
| Sentry, Grafana Cloud (free or low tiers) | $0–$30 |
| PagerDuty / Opsgenie (1 user) | ~$25 |
| **Subtotal** | **~$140–$280/mo** |

### One-time
- IBKR account funding: per your strategy needs (avoid PDT at <$25k for active equities).
- Crypto exchange funding: per strategy.
- Initial setup time: 3–4 months engineering.

### Hidden costs to plan for
- **Slippage** is real and rarely matches backtests. Reserve a budget for "live runs vs. backtest" gap.
- **Borrow fees** on shorts (HTB names can be 10–100% APR).
- **Margin interest** if leveraged.
- **Exchange fees** (especially crypto — adds up fast).

---

## 13. Risks & Open Questions

### Open questions
1. **Is this single-strategy or multi-strategy from day 1?** Multi-strategy doubles the complexity (allocation, conflict, shared risk budget).
2. **What's the strategy's target capacity?** Drives data/latency requirements. $100k AUM has very different needs than $10M.
3. **Are you operating as an individual or a registered entity?** Affects taxes, data fees (pro vs. non-pro), record-keeping requirements.
4. **24/7 ops**: who responds to a 3am page? If solo, accept the limitation; design strategies that can pause cleanly.
5. **Crypto scope**: just BTC/ETH on IBKR, or do you really need altcoin exposure (which forces a second exchange integration and another kill-switch surface)?

### Risks
1. **Bug-class loss > strategy drawdown**: a runaway loop sending orders, a sign-flipped position size, a missing risk check — these dwarf normal P&L variance. Defense in depth + paper-first.
2. **IBKR-specific outages**: scheduled maintenance + occasional unscheduled disconnects. Strategies must pause cleanly, not panic.
3. **Crypto exchange counterparty risk**: 2022 reminded everyone what it means. Don't keep more cash on exchange than you need for active orders.
4. **Regulatory drift**: rules change. Stay current on PDT, options levels, crypto reporting (1099-DA), state-level requirements.
5. **Overconfidence after a good month**: the most expensive failure mode. Size limits stay in place regardless of recent P&L.

---

## 14. References

- IBKR Trader Workstation API: https://interactivebrokers.github.io/tws-api/
- `ib_async`: https://github.com/ib-api-reloaded/ib_async
- IBC (auto-login for IB Gateway): https://github.com/IbcAlpha/IBC
- IBKR Client Portal Web API: https://www.interactivebrokers.com/en/trading/ib-api.php#client-portal-api
- IBKR market data fees: https://www.interactivebrokers.com/en/pricing/market-data-pricing.php
- `ccxt` (multi-exchange crypto): https://github.com/ccxt/ccxt
- Coinbase Advanced Trade API: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
- Kraken WebSocket API: https://docs.kraken.com/api/docs/websocket-v2/
- NATS: https://nats.io/
- `py_vollib` (Greeks): https://github.com/vollib/py_vollib

---

## 15. Final Reminders

1. **Start in paper.** Always.
2. **The risk engine is your only safety net. Test it like your account depends on it — because it does.**
3. **Reconcile early, reconcile often.** Drift is the precursor to disaster.
4. **Don't optimize before you have working.** The first version doesn't need Rust, doesn't need Kafka, doesn't need a cluster. It needs to not blow up.
5. **Logs are forever.** Every tick, every signal, every order. Cheap to store; priceless when something breaks.
6. **Size up slowly.** First live trade: 1 share. Increase only when boring.
