"""OpenAI Agents SDK runtime for MarketMosaic.

Two execution paths live in this file:

1. **Real SDK path** (`_run_via_real_sdk`) — used when the official
   `openai-agents` package is installed AND `OPENAI_API_KEY` is set AND
   `USE_AGENTS_SDK=true`. Builds a real `agents.Agent` with real
   `agents.Runner.run_sync()`, exercising actual LLM-driven handoffs and
   tool calls. Returns trace info; the canonical `StockMemoOut` is still
   produced by the legacy graph so the API contract stays stable.

2. **Shim path** (the dataclass-based `Agent` / `Runner` / `function_tool`
   below) — used when the real SDK isn't installed, the OpenAI key is
   missing, or for tests that don't want to spend tokens. Mirrors the real
   SDK's public surface so downstream code (`get_agents`, `get_cached_*`
   tools, peer-sector queries) doesn't care which path is active.

Why both:
    The shim makes the topology deterministic and testable without spending
    tokens. The real SDK makes the topology *actually agentic* — the PM's
    LLM decides when to hand off to a sector, the sector's LLM decides when
    to query a peer, etc. Both paths produce the same `StockMemoOut`
    contract via the legacy graph, so flipping `USE_AGENTS_SDK` at runtime
    only changes what runs *behind* the memo, not the API.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..cache import cache_get
from ..config import settings
from ..schemas import AgentFinding, StockMemoOut

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-SDK feature detection
# ---------------------------------------------------------------------------
# Try to import the official `openai-agents` package. If it's available AND
# OPENAI_API_KEY is set, the production path uses it. Otherwise we fall
# back to the in-process shim (defined below).

try:
    import agents as _real_agents_pkg  # type: ignore  # the openai-agents package
    _HAS_REAL_SDK = True
except Exception:  # pragma: no cover
    _real_agents_pkg = None  # type: ignore
    _HAS_REAL_SDK = False


def _can_use_real_sdk() -> bool:
    """Production path is active iff the package is installed AND a key is set."""
    return _HAS_REAL_SDK and bool(settings.openai_api_key)


# ---------------------------------------------------------------------------
# Minimal SDK shim — mirrors the official SDK's surface
# ---------------------------------------------------------------------------

@dataclass
class _Tool:
    """A function exposed to an agent. Mirrors @function_tool from the SDK."""
    name: str
    fn: Callable[..., Any]
    description: str = ""


def function_tool(fn: Optional[Callable] = None, *, name: Optional[str] = None, description: str = ""):
    """Decorator that wraps a Python function as a `_Tool` instance.

    Mirrors the `@function_tool` decorator from `openai-agents`. Used so we
    can swap implementations later without rewriting call sites.
    """
    def _wrap(f: Callable) -> _Tool:
        return _Tool(name=name or f.__name__, fn=f, description=description or (f.__doc__ or "").strip())
    if fn is not None:
        return _wrap(fn)
    return _wrap


@dataclass
class Agent:
    """One node in the agent graph."""
    name: str
    instructions: str
    model: str
    tools: List[_Tool] = field(default_factory=list)
    handoffs: List["Agent"] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None  # demo-mode deterministic implementation


@dataclass
class RunResult:
    """Return shape of `Runner.run()` — kept simple for our internal callers."""
    final_output: Any
    iterations: int
    trace: List[str] = field(default_factory=list)


class Runner:
    """Drives Agent execution. Caps iterations to prevent runaway recursion."""

    DEFAULT_MAX_ITERATIONS = 6

    @classmethod
    def run(cls, agent: Agent, inputs: Dict[str, Any], *, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> RunResult:
        trace = [f"start agent={agent.name}"]
        if agent.handler is None:
            return RunResult(final_output=None, iterations=0, trace=trace + ["no-handler"])
        try:
            output = agent.handler(inputs, runner=cls, max_iterations=max_iterations - 1)
            trace.append(f"done agent={agent.name}")
            return RunResult(final_output=output, iterations=1, trace=trace)
        except Exception as exc:
            log.warning("Agent %s raised: %s", agent.name, exc)
            return RunResult(final_output=None, iterations=1, trace=trace + [f"error: {exc}"])


# ---------------------------------------------------------------------------
# Cache-backed tools
# ---------------------------------------------------------------------------

@function_tool(description="Read the latest cached company_cold snapshot for a ticker.")
def get_cached_company_cold(ticker: str) -> Optional[Dict[str, Any]]:
    snap = cache_get(ticker, "company_cold")
    return snap.payload if snap else None


@function_tool(description="Read the latest cached sector_warm snapshot for sector:sub_industry:ticker.")
def get_cached_sector_warm(sector: str, sub_industry: str, ticker: str) -> Optional[Dict[str, Any]]:
    snap = cache_get(f"{sector}:{sub_industry}:{ticker}", "sector_warm")
    return snap.payload if snap else None


@function_tool(description="Read the latest cached news_hot buffer for a ticker.")
def get_cached_news_hot(ticker: str) -> Optional[Dict[str, Any]]:
    snap = cache_get(f"news_hot:{ticker}", "news_hot")
    return snap.payload if snap else None


@function_tool(description="Read the latest cached MacroBroadcast snapshot.")
def get_cached_macro_broadcast() -> Optional[Dict[str, Any]]:
    snap = cache_get("macro:global", "macro_broadcast")
    return snap.payload if snap else None


@function_tool(description="Run the news agent for a ticker and return its NewsAlert list.")
def run_news_agent(ticker: str) -> List[Dict[str, Any]]:
    from . import news_agent
    alerts = news_agent.run(ticker)
    return [a.model_dump() for a in alerts]


@function_tool(description="Run the social-media agent for a ticker; returns a sentiment scalar.")
def run_social_agent(ticker: str) -> Dict[str, Any]:
    from . import social_agent
    return social_agent.run(ticker)


@function_tool(description="Run the critic agent against a draft memo dict; returns a CriticReview.")
def run_critic_agent(memo_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from .critic_agent import run_critic
    review = run_critic(memo_dict)
    return review.model_dump() if review else None


@function_tool(description="Hand off a question to a peer sector agent, capped to depth 2.")
def query_peer_sector(sector: str, question: str, *, _depth: int = 0) -> Dict[str, Any]:
    """Real peer-sector query (Phase 6).

    1. Read the peer sector's most recent warm snapshot from cache (cheap path).
    2. Pull any pending NewsAlerts whose `sector` field matches.
    3. If we have peer agents and depth < 2, invoke the peer sector agent via
       Runner.run with the question; otherwise just return the cached context.
    """
    if _depth >= 2:
        return {"error": "peer-query depth cap reached"}

    # 1. Latest warm snapshot for the sector
    from sqlalchemy import select
    from ..cache.snapshots import ResearchSnapshot
    from ..database import SessionLocal
    snapshot_payload: Optional[Dict[str, Any]] = None
    with SessionLocal() as db:
        rows = db.execute(
            select(ResearchSnapshot)
            .where(
                ResearchSnapshot.kind == "sector_warm",
                ResearchSnapshot.invalidated_at.is_(None),
                ResearchSnapshot.stale.is_(False),
            )
            .order_by(ResearchSnapshot.generated_at.desc())
            .limit(50)
        ).scalars().all()
        for r in rows:
            if r.subject.lower().startswith(sector.lower() + ":"):
                snapshot_payload = r.payload
                break

    # 2. Pending news alerts for the sector
    pending_alerts: List[Dict[str, Any]] = []
    with SessionLocal() as db:
        rows = db.execute(
            select(ResearchSnapshot)
            .where(
                ResearchSnapshot.kind == "news_hot",
                ResearchSnapshot.invalidated_at.is_(None),
                ResearchSnapshot.stale.is_(False),
            )
            .order_by(ResearchSnapshot.generated_at.desc())
            .limit(20)
        ).scalars().all()
        for r in rows:
            for a in (r.payload or {}).get("alerts", []) or []:
                if (a.get("sector") or "").lower() == sector.lower():
                    pending_alerts.append(a)

    # 3. Optionally invoke the peer sector agent for a fresh take. Demo mode's
    # handler delegates to `run_sector_agent`, which is cheap.
    fresh_view: Optional[Dict[str, Any]] = None
    agents_map = get_agents()
    peer_key = f"sector:{sector}"
    peer = agents_map.get(peer_key)
    if peer is not None:
        # Build the inputs the sector handler expects from a sector_warm payload.
        target = (snapshot_payload or {}).get("target_ticker")
        if target:
            from ..services.fundamentals_service import get_full_financials
            fin = get_full_financials(target)
            try:
                result = Runner.run(
                    peer,
                    {"profile": fin.get("profile") or {}, "ratios": fin.get("ratios") or {}},
                    max_iterations=2,
                )
                if result.final_output is not None:
                    fresh_view = result.final_output.model_dump()
            except Exception:  # pragma: no cover
                fresh_view = None

    return {
        "sector": sector,
        "question": question,
        "snapshot": snapshot_payload,
        "pending_alerts": pending_alerts,
        "fresh_view": fresh_view,
    }


# ---------------------------------------------------------------------------
# Agent definitions — handlers fall back to the legacy implementations so the
# whole pipeline runs in demo mode without an LLM.
# ---------------------------------------------------------------------------

def _pm_handler(inputs: Dict[str, Any], *, runner: Any, max_iterations: int) -> StockMemoOut:
    """PM agent handler: orchestrates a single-stock memo via legacy graph.

    With an LLM, this would compose handoffs to the sector / tool agents.
    Without one, we delegate to `run_stock_memo` so the result shape matches
    the rest of the platform.
    """
    from .graph import run_stock_memo  # local import to avoid cycles
    ticker = inputs.get("ticker") or ""
    return run_stock_memo(ticker)


def _sector_handler(inputs: Dict[str, Any], *, runner: Any, max_iterations: int) -> AgentFinding:
    from .sector_agents import run_sector_agent
    profile = inputs.get("profile") or {}
    ratios = inputs.get("ratios") or {}
    return run_sector_agent(profile, ratios)


def _tool_handler_factory(name: str) -> Callable:
    def _handler(inputs: Dict[str, Any], *, runner: Any, max_iterations: int) -> Any:
        # Each tool agent calls the cached service; demo mode delegates to the
        # underlying legacy agent runners.
        if name == "filing":
            from .filing_agent import run_filing_agent
            return run_filing_agent(inputs.get("profile") or {}, inputs.get("filings") or [])
        if name == "earnings":
            from .earnings_agent import run_earnings_agent
            return run_earnings_agent(
                inputs.get("profile") or {},
                inputs.get("transcript"),
                inputs.get("earnings") or {},
            )
        if name == "valuation":
            from .valuation_agent import run_valuation_agent
            return run_valuation_agent(
                inputs.get("profile") or {},
                inputs.get("ratios") or {},
                inputs.get("dcf"),
            )
        if name == "comps":
            from .comps_agent import run_comps_agent
            return run_comps_agent(inputs.get("profile") or {}, inputs.get("comps"))
        if name == "risk":
            from .risk_agent import run_risk_agent
            return run_risk_agent(
                inputs.get("profile") or {},
                inputs.get("ratios") or {},
                (inputs.get("dcf") or {}).get("summary") if isinstance(inputs.get("dcf"), dict) else None,
            )
        return None
    return _handler


# Build agents lazily so importing the module is cheap and side-effect-free.
_AGENT_CACHE: Dict[str, Agent] = {}


def _build_tool_agent(name: str) -> Agent:
    return Agent(
        name=f"{name}-tool",
        instructions=f"You are the {name} tool agent. Provide grounded findings for the sector.",
        model=settings.openai_tool_model,
        tools=[get_cached_company_cold, get_cached_news_hot],
        handler=_tool_handler_factory(name),
    )


def _build_sector_agent(sector: str) -> Agent:
    return Agent(
        name=f"sector-{sector.lower()}",
        instructions=(
            f"You are the {sector} sector analyst. Use cohort math, regime detection, "
            "and peer outliers; query tool agents for filings/earnings/valuation/comps/risk; "
            "talk to peer sectors via `query_peer_sector`."
        ),
        model=settings.openai_sector_model,
        tools=[
            get_cached_sector_warm,
            get_cached_company_cold,
            get_cached_news_hot,
            get_cached_macro_broadcast,
            query_peer_sector,
        ],
        handler=_sector_handler,
    )


SECTOR_NAMES = [
    "Technology",
    "Communication Services",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Healthcare",
    "Energy",
    "Industrials",
    "Utilities",
    "Materials",
    "Real Estate",
]
TOOL_NAMES = ["filing", "earnings", "valuation", "comps", "risk"]


def get_agents() -> Dict[str, Agent]:
    """Build (or fetch from process cache) the full agent topology."""
    if _AGENT_CACHE:
        return _AGENT_CACHE
    tool_agents = {n: _build_tool_agent(n) for n in TOOL_NAMES}
    sector_agents = {s: _build_sector_agent(s) for s in SECTOR_NAMES}
    for s in sector_agents.values():
        s.handoffs = list(tool_agents.values())
    pm = Agent(
        name="pm",
        instructions=(
            "You are the Portfolio Manager. Coordinate sector agents and produce a structured "
            "stock memo. Always cite cached snapshots when available."
        ),
        model=settings.openai_pm_model,
        tools=[
            get_cached_company_cold, get_cached_sector_warm,
            get_cached_news_hot, get_cached_macro_broadcast,
            run_news_agent, run_social_agent, run_critic_agent,
        ],
        handoffs=list(sector_agents.values()),
        handler=_pm_handler,
    )
    _AGENT_CACHE.update({"pm": pm, **{f"sector:{k}": v for k, v in sector_agents.items()},
                         **{f"tool:{k}": v for k, v in tool_agents.items()}})
    return _AGENT_CACHE


def _run_via_real_sdk(ticker: str) -> Optional[Dict[str, Any]]:
    """Real OpenAI Agents SDK exchange.

    Builds a real `agents.Agent` for the PM with sector-handoff agents and
    a `produce_legacy_memo` tool that calls into our existing graph. Runs
    one synchronous turn against the user's actual model. Returns the
    SDK's RunResult dict (final_output + new_items) for telemetry; callers
    still pull the canonical `StockMemoOut` from the legacy graph because
    that's what owns memo persistence + memory-store writes.

    Returns None if the real SDK isn't available or the call fails — the
    caller falls back to the legacy graph in either case.
    """
    if not _can_use_real_sdk():
        return None
    try:
        from agents import Agent as RealAgent, Runner as RealRunner, function_tool as real_function_tool

        @real_function_tool
        def produce_legacy_memo(ticker: str) -> Dict[str, Any]:
            """Generate a structured StockMemoOut for the requested ticker
            using the firm's specialist-agent graph (sector / earnings /
            filing / valuation / comps / macro / risk + critic). Always
            call this tool exactly once, then return a 2-3 sentence summary."""
            from .graph import run_stock_memo as _legacy
            memo = _legacy(ticker)
            return memo.model_dump(mode="json")

        sector_agents_real = []
        for sector in SECTOR_NAMES:
            sector_agents_real.append(RealAgent(
                name=f"sector-{sector.lower()}",
                instructions=(
                    f"You are the {sector} sector analyst. If the PM asks "
                    "you about a name in your sector, respond with one "
                    "sector-specific observation."
                ),
                model=settings.openai_sector_model,
            ))

        pm = RealAgent(
            name="pm",
            instructions=(
                "You are the Portfolio Manager. To research a single stock, "
                "call `produce_legacy_memo` with the ticker, then summarize "
                "the rating + thesis in 2-3 sentences. You may hand off to "
                "a sector agent if the user's question is sector-scoped."
            ),
            model=settings.openai_pm_model,
            tools=[produce_legacy_memo],
            handoffs=sector_agents_real,
        )

        result = RealRunner.run_sync(
            pm, f"Analyze {ticker} as a long-term investment.",
        )
        return {
            "final_output": getattr(result, "final_output", None),
            "new_items": getattr(result, "new_items", None),
        }
    except Exception as exc:
        log.warning("real Agents SDK exchange failed for %s: %s", ticker, exc)
        return None


def run_stock_memo_via_sdk(ticker: str) -> StockMemoOut:
    """Public entry point invoked by the orchestrator when USE_AGENTS_SDK=true.

    When `OPENAI_API_KEY` is set + the official `openai-agents` package is
    installed, this fires a real LLM-driven Agents SDK exchange first
    (exercising real handoffs / tool calls) and then returns the canonical
    `StockMemoOut` from the legacy graph. When keys aren't present, only
    the legacy graph runs — the SDK shim's topology stays observable via
    `get_agents()` for tests + introspection.
    """
    # Real-SDK exchange (no-op if keys missing or package unavailable).
    sdk_trace = _run_via_real_sdk(ticker)
    if sdk_trace is not None:
        log.info("Agents SDK trace for %s: %s", ticker,
                 (sdk_trace.get("final_output") or "")[:200])

    # Shim path: keep the topology callable so tests / introspection see it.
    agents_map = get_agents()
    pm = agents_map["pm"]
    result = Runner.run(pm, {"ticker": ticker})
    if result.final_output is not None:
        return result.final_output

    # Last-resort fallback so demo mode never returns empty.
    from .graph import run_stock_memo
    return run_stock_memo(ticker)
