"""Wave 10 — Freeform chat routed through the OpenAI Agents SDK.

The legacy `_answer_with_memo_context` is a single-shot LLM call that
injects compact memo summaries into one prompt. That works for "compare
MSFT vs GOOGL" (both memos already exist) but breaks down when the user
asks something the prepacked summary doesn't cover ("what's the WACC the
PM used?", "what's the comps median EV/EBITDA?", "what's the latest
macro snapshot?").

This module builds a real SDK Agent with `function_tool`s that lazily
fetch the data the user actually asked for. The agent decides which
tools to call.

Privacy (FIX-020). The turn's `SDKTrace` row (surface='chat') records
WHAT ran — item types, the agent, the tool names — never what the model
wrote: `final_output` is stored empty and no item arguments, outputs or
message text are kept. SDK trace export to OpenAI is disabled for the
whole process at import and again per run, and failures are logged by
exception type only (an SDK exception can quote the model's output).

Attribution: every model response of a turn writes one `llm_call_logs`
row (action `chat.sdk_turn`) through a `RunHooks.on_llm_end` hook, under
the route's `chat:<hex>` run id, so a turn that falls back to the legacy
single-shot answer reads as one timeline.

Skip conditions (caller answers with the legacy single-shot path):
- `CHAT_AGENTS_SDK=false` (plan P14; the legacy `USE_AGENTS_SDK` no longer
  gates chat).
- `openai-agents` not installed, no `OPENAI_API_KEY`, or demo-only mode.

Failure conditions:
- SDK run raises or the model refuses → returns None, caller falls back.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from ..config import settings
from . import llm
from .log_safety import log_safely

log = logging.getLogger(__name__)


def disable_sdk_tracing() -> None:
    """Turn off openai-agents trace export for this process.

    The SDK uploads traces — prompts, the user's message and history, tool
    arguments and outputs — to OpenAI's trace store by default. A per-run
    `RunConfig(tracing_disabled=True)` is passed too, but any future Runner
    call without it would re-enable export, so the global switch is thrown
    at import (attribution critique #18). `set_tracing_disabled` wins over
    the OPENAI_AGENTS_DISABLE_TRACING env var, which production also sets.
    """
    try:
        from agents import set_tracing_disabled
    except Exception:  # pragma: no cover - the package is optional
        return
    set_tracing_disabled(True)


disable_sdk_tracing()


def sdk_run_config() -> Any:
    """The RunConfig every Runner call passes: no trace export, and no
    model/tool data in any trace a processor might still see."""
    from agents import RunConfig
    return RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)


_ITEM_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.:\-]")


def _item_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return _ITEM_NAME_UNSAFE.sub("_", value)[:64]


def trace_items(new_items: Any) -> list[dict[str, Any]]:
    """`new_items` reduced to `{type, agent, tool}` per item.

    What the agent fetched stays reviewable (which tools, which agent,
    in what order); the arguments the model chose, the tool outputs and
    the message text do not reach the database (FIX-020)."""
    out: list[dict[str, Any]] = []
    for item in list(new_items or [])[:200]:
        raw = getattr(item, "raw_item", None)
        tool = raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None)
        out.append({
            "type": _item_name(getattr(item, "type", None)) or type(item).__name__[:64],
            "agent": _item_name(getattr(getattr(item, "agent", None), "name", None)),
            "tool": _item_name(tool),
        })
    return out


def _agent_model_name(agent: Any) -> str:
    model = getattr(agent, "model", None)
    if isinstance(model, str):
        return model
    return str(getattr(model, "model", None) or "") or "unknown"


def _agent_effort(agent: Any) -> str | None:
    reasoning = getattr(getattr(agent, "model_settings", None), "reasoning", None)
    effort = getattr(reasoning, "effort", None)
    return effort if isinstance(effort, str) else None


def sdk_usage_hooks(action: str, *, ticker: str | None = None, run_id: str | None = None) -> Any:
    """RunHooks that write one `llm_call_logs` row per SDK model response
    (attribution critique #4: `RunResult.raw_responses` knows no model, and
    with handoffs each agent runs on its own).

    The hook runs inside the SDK's event loop, which carries a copy of the
    caller's context, so the row picks up the turn's run_id. A telemetry
    failure is logged and swallowed: it must never fail the run.
    `refused` / `responses` / `call_id` are read back by the caller.
    """
    from agents import RunHooks

    class _UsageHooks(RunHooks):  # type: ignore[misc,type-arg]
        def __init__(self) -> None:
            super().__init__()
            self.call_id = uuid.uuid4().hex
            self.responses = 0
            self.refused = False
            self.last_model = ""
            self._started = time.perf_counter()

        async def on_llm_start(self, context: Any, agent: Any, *args: Any, **kwargs: Any) -> None:
            self._started = time.perf_counter()

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            self.responses += 1
            self.last_model = _agent_model_name(agent)
            try:
                self.refused = llm.record_sdk_usage(
                    response, agent=str(getattr(agent, "name", "") or ""),
                    model=self.last_model, action=action, call_id=self.call_id,
                    attempt=self.responses, effort=_agent_effort(agent), ticker=ticker,
                    duration_ms=int((time.perf_counter() - self._started) * 1000),
                )
            except Exception as exc:  # pragma: no cover - telemetry must not fail a run
                log_safely(log, "SDK usage row failed (non-fatal)", exc, level=logging.DEBUG)

        def record_failure(self, exc: BaseException, *, agent: str, model: str) -> None:
            """One error row for a run that raised — unless the raise is the
            refusal the last response's row already recorded."""
            if self.refused:
                return
            where = "response_error" if self.responses else "provider_error"
            try:
                # Called after the run's context has closed: re-open the
                # run id so the error row joins the response rows.
                with llm.llm_call_context(run_id=run_id):
                    llm.record_sdk_usage(
                        None, agent=agent, model=self.last_model or model, action=action,
                        call_id=self.call_id, attempt=self.responses + 1, ticker=ticker,
                        duration_ms=int((time.perf_counter() - self._started) * 1000),
                        error=f"{where}:{type(exc).__name__}",
                    )
            except Exception as rec_exc:  # pragma: no cover
                log_safely(log, "SDK error row failed (non-fatal)", rec_exc, level=logging.DEBUG)

    return _UsageHooks()


def _is_refusal(exc: BaseException) -> bool:
    """openai-agents 0.22 raises ModelRefusalError when the model declines."""
    return type(exc).__name__ == "ModelRefusalError"


def _can_use_sdk() -> bool:
    """Same gate shape as `sdk_runtime._can_use_real_sdk` — kept inline so
    this module's import doesn't pull in the legacy SDK runtime if we end
    up deprecating it.

    Gated on `CHAT_AGENTS_SDK`, not `USE_AGENTS_SDK`: the legacy flag also
    routes chat's inline "analyze X" memo through `sdk_runtime`, which runs
    the memo twice under a run id nothing links to (plan P14). Demo-only
    mode never builds an LLM client in `llm.py`; the SDK builds its own,
    so it has to ask the same question.
    """
    if not settings.chat_agents_sdk:
        return False
    if not settings.openai_api_key:
        return False
    if llm._demo_only():
        return False
    try:
        import agents  # noqa: F401
        return True
    except Exception:
        return False


def chat_model_and_settings() -> tuple[str, Any | None]:
    """(model, ModelSettings or None) for the chat agent.

    `CHAT_MODEL` / `CHAT_EFFORT` through the `chat.sdk_turn` tier route;
    blank keeps today's model (`OPENAI_PM_MODEL`, else the strong route
    default) with the SDK's own defaults for it. The effort goes out as
    `reasoning.effort`, and no temperature is ever set: GPT-6 rejects it
    whenever effort is not `none` (model research 2026-09-25). A
    non-OpenAI CHAT_MODEL is unresolvable — this SDK path speaks only
    OpenAI — so it raises, and the caller answers on the legacy path.
    """
    configured = (settings.chat_model or "").strip()
    if configured and llm.provider_of_model(configured) != "openai":
        raise ValueError(
            f"CHAT_MODEL {configured!r} is not an OpenAI model; the Agents SDK chat speaks only OpenAI"
        )
    route = llm.resolve_action_route("chat.sdk_turn")
    if route.configured and route.model:
        if route.provider != "openai":
            # The tier can resolve elsewhere too: LLM_ACTION_TIER_OVERRIDES=
            # chat.sdk_turn:research with an Opus research model would
            # otherwise send a Claude model name to the OpenAI endpoint.
            raise ValueError(
                f"chat.sdk_turn resolves to {route.provider} model {route.model!r}; "
                "the Agents SDK chat speaks only OpenAI"
            )
        model, effort = route.model, route.effort
    else:
        model = llm.resolve_role_model("pm", provider="openai")
        effort = llm._effort_for("openai", model, settings.chat_effort or None)
    if not effort:
        return model, None
    from agents import ModelSettings
    from openai.types.shared import Reasoning
    return model, ModelSettings(reasoning=Reasoning(effort=effort))  # type: ignore[arg-type]


def _profile_for(fin: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Extract the profile from a `get_full_financials` result, guaranteeing
    a `ticker` key.

    `fin.get("profile") or {}` on its own produced the 2026-08-12 Render
    OOM: when the fundamentals lookup missed (off-universe name, provider
    miss), the specialists received a profile with no ticker and passed
    `profile.get("ticker")` → None into `vector_store.search`, which read
    a falsy ticker as "no filter" and scanned every chunk in the corpus.

    The memo path never hit this because `graph.py` raises on an empty
    profile. Chat can't raise — a follow-up question about a thinly
    covered name is legitimate — so instead of failing, we seed the
    ticker the caller already gave us. The specialist then runs with a
    correctly *scoped* retrieval rather than an unscoped one, which is
    strictly more useful than either OOMing or bailing out.
    """
    profile = dict(fin.get("profile") or {})
    resolved = (ticker or "").strip().upper()
    if resolved and not profile.get("ticker"):
        profile["ticker"] = resolved
    return profile


def _memo_tool_payload(ticker: str) -> dict[str, Any]:
    """`get_memo`'s answer: the chat projection of the PRESENTED memo (W2a),
    so the agent never quotes a section the memo page hides."""
    from ..services.memo_store import latest_memo, present_snapshot
    from .orchestrator import _memo_for_chat_context
    snap = latest_memo((ticker or "").upper())
    if snap is None:
        return {"error": f"No memo cached for {ticker}. The user may need to run an analysis first."}
    try:
        m = present_snapshot(snap)
    except Exception as exc:
        return {"error": f"Failed to load memo for {ticker}: {exc}"}
    return _memo_for_chat_context(m)


SPECIALIST_UNAVAILABLE = "specialist unavailable in this version"


def _specialist_payload(agent: str, ticker: str, finding: Any) -> dict[str, Any]:
    """An `ask_*` tool's answer. A fallback stand-in (deterministic read-out,
    no-input stub, crash placeholder) is refused rather than relayed as the
    specialist's answer — the same rule that hides it on the memo (W2a,
    critique delta 4)."""
    from ..services.memo_sections import finding_is_template
    if finding_is_template(finding):
        return {"error": SPECIALIST_UNAVAILABLE, "agent": agent, "ticker": ticker.upper()}
    return {
        "agent": agent, "ticker": ticker.upper(),
        "headline": finding.headline, "summary": finding.summary,
        "key_points": finding.key_points, "confidence": finding.confidence,
    }


def _build_chat_agent() -> Any | None:
    """Wire an `Agent` with the four data-fetch tools the chat handler
    might need. Returns None if the SDK isn't usable."""
    if not _can_use_sdk():
        return None
    try:
        from agents import Agent, function_tool
        model, model_settings = chat_model_and_settings()

        @function_tool
        def get_memo(ticker: str) -> dict[str, Any]:
            """Return the latest cached investment memo for `ticker`. The
            response is the compact projection that includes the rating,
            stock score, one-sentence thesis, key risks, the bull/bear
            case headlines, and the PM-adjusted DCF summary. Use this
            FIRST for any question about a specific name. Returns
            `{"error": "..."}` if no memo exists yet."""
            return _memo_tool_payload(ticker)

        @function_tool
        def get_dcf_summary(ticker: str) -> dict[str, Any]:
            """Return the latest DCF for `ticker` — both the PM-adjusted
            view (used by the memo's rating) and the consensus-anchored
            initial view, plus the audit trail of which assumptions the
            PM changed and why. Use when the user asks about valuation
            mechanics ("what WACC did you use", "why is the bear case so
            negative", etc.)."""
            from ..services.memo_store import latest_memo, memo_to_pydantic
            snap = latest_memo((ticker or "").upper())
            if snap is None:
                return {"error": f"No memo for {ticker}."}
            try:
                m = memo_to_pydantic(snap)
            except Exception as exc:
                return {"error": f"Failed to load memo: {exc}"}
            return {
                "ticker": m.ticker,
                "dcf_summary_pm_adjusted": m.dcf_summary,
                "dcf_summary_initial": m.dcf_initial_summary or None,
                "pm_adjustments": m.dcf_pm_adjustments or [],
                "pm_adjustment_headline": m.dcf_pm_adjustment_headline or "",
            }

        @function_tool
        def get_comps(ticker: str) -> dict[str, Any]:
            """Return the peer-comparison data for `ticker`: peer set,
            target metrics vs peer median, premium/discount on each
            multiple, and (when available) target's own multi-year
            history percentile. Use when the user asks how a name
            looks vs peers, who its peers are, or whether it's expensive
            on a specific multiple."""
            from ..services.valuation_service import build_comps
            try:
                comps = build_comps((ticker or "").upper())
            except Exception as exc:
                return {"error": f"Failed to build comps: {exc}"}
            if comps is None:
                return {"error": f"No peer set defined for {ticker}."}
            return {
                "ticker": comps.target.ticker,
                "peers": [p.ticker for p in comps.peers],
                "target": comps.target.model_dump(),
                "peer_median": comps.median.model_dump(),
                "premium_discount": comps.premium_discount,
                "interpretation": comps.interpretation,
                "history": comps.history.model_dump() if comps.history else None,
            }

        @function_tool
        def get_macro_snapshot() -> dict[str, Any]:
            """Return the current macro snapshot (FRED data — Fed Funds,
            10y yield, core sticky CPI, unemployment, etc.) plus the
            most recent regime broadcast. Use when the user asks about
            macro context or how a regime change affects a sector/name."""
            from ..cache import cache_get
            from ..services.macro_service import macro_snapshot
            snap = macro_snapshot() or {}
            broadcast = cache_get("macro:global", "macro_broadcast")
            return {
                "fred_snapshot": snap,
                "regime_broadcast": (
                    broadcast.payload if broadcast and isinstance(broadcast.payload, dict)
                    else None
                ),
            }

        @function_tool
        def get_company_lite(ticker: str) -> dict[str, Any]:
            """Return a compact company dossier — sector / industry /
            market cap / business description plus screener-tier metrics
            (P/E, ROIC, margins, growth) and AI-ranked factor scores.
            Use this when the user asks about a name we DON'T have a
            full memo for yet, or when answering comparative questions
            ('which has the best margins?', 'what's the moat?'). This
            doesn't trigger an analysis run. `last_price` comes with
            `last_price_as_of` (UTC) and `last_price_source` (live, stale,
            eod_close or profile_seed): when citing it, say how old it is
            and that live quotes may be delayed up to 15 min."""
            from .orchestrator import _company_lite_snapshot
            snap = _company_lite_snapshot((ticker or "").upper())
            if snap is None:
                return {"error": f"{ticker} not in companies table — outside curated universe."}
            return snap

        @function_tool
        def list_universe(sector: str | None = None) -> dict[str, Any]:
            """List the curated screener universe (S&P 500 + curated extensions). Optionally
            filter by sector ('Technology', 'Healthcare', etc.). Use
            when the user asks "what stocks does the platform cover" or
            "show me tech names available"."""
            from ..database import SessionLocal
            from ..models import Company
            with SessionLocal() as db:
                query = db.query(Company).filter(Company.universe_tier == "auto_analysis")
                if sector:
                    from sqlalchemy import func as _f
                    query = query.filter(_f.lower(Company.sector) == sector.lower())
                rows = query.all()
                return {
                    "count": len(rows),
                    "tickers": [
                        {"ticker": c.ticker, "company_name": c.company_name, "sector": c.sector}
                        for c in rows
                    ],
                }

        @function_tool
        def screener_query(
            sort_by: str | None = None,
            sector: str | None = None,
            theme: str | None = None,
            limit: int = 10,
        ) -> dict[str, Any]:
            """Fetch the AI-ranked screener results. `sort_by` ∈ {pm_score,
            quality, growth, valuation, earnings_momentum, risk,
            macro_fit}; default pm_score. Use when the user asks for
            'top compounders', 'cheapest names', 'highest growth in
            tech', etc. Returns up to `limit` rows with the full factor
            score breakdown for each."""
            from ..services.screener_service import compute_universe_scores
            try:
                result = compute_universe_scores(theme=theme)
            except Exception as exc:
                return {"error": f"screener failed: {exc}"}
            rows = result.rows
            if sector:
                rows = [r for r in rows if sector.lower() in (r.sector or "").lower()]
            allowed = {
                "pm_score", "quality", "growth", "valuation",
                "earnings_momentum", "risk", "macro_fit",
            }
            key = sort_by if sort_by in allowed else "pm_score"
            rows = sorted(rows, key=lambda r: getattr(r, key, 0) or 0, reverse=True)
            return {
                "sort_by": key,
                "sector_filter": sector,
                "theme": theme,
                "count": len(rows),
                "rows": [
                    {
                        "ticker": r.ticker, "name": r.company_name, "sector": r.sector,
                        "pm_score": r.pm_score, "quality": r.quality, "growth": r.growth,
                        "valuation": r.valuation, "risk": r.risk,
                        "thesis": r.one_line_thesis,
                    }
                    for r in rows[:limit]
                ],
            }

        @function_tool
        def custom_screen(rules_json: str, limit: int = 10) -> dict[str, Any]:
            """Run a rule-based custom screen against the 15-metric raw
            snapshot table. `rules_json` is a JSON-encoded list of
            `{"metric": "...", "op": "...", "value": ...}` rules
            (AND-combined). Metrics: pe_ttm, ev_ebitda, ev_revenue,
            gross_margin, op_margin, fcf_margin, roic, roe,
            debt_to_ebitda, revenue_growth_yoy, market_cap, beta. Ops:
            >, <, >=, <=, =, between (with value2). Use when the user
            asks for stocks meeting numeric thresholds ('gross margin
            > 70% and P/E < 25')."""
            try:
                rules = json.loads(rules_json) if isinstance(rules_json, str) else rules_json
            except Exception as exc:
                return {"error": f"rules_json must be a JSON list: {exc}"}
            from ..api.routes_screener import _execute_custom_screen
            from ..schemas import CustomScreenRequest
            try:
                req = CustomScreenRequest(rules=rules, limit=limit)
                result = _execute_custom_screen(req)
            except Exception as exc:
                return {"error": f"custom_screen failed: {exc}"}
            return {
                "matched": result.matched,
                "rule_count": result.rule_count,
                "rows": [r.model_dump() for r in result.rows],
            }

        @function_tool
        def get_industry_context(
            tickers: list[str] | None = None, code: str | None = None,
        ) -> dict[str, Any]:
            """Return the stored weekly Industry Analysis context for the
            industry group(s) of `tickers` (a single name or a whole
            portfolio — the tool takes a list) and/or an explicit
            industry-group `code` (the group's slug, e.g. as returned in a
            previous answer): each group's row from the latest
            cross-industry snapshot (1W/1M/YTD equal-weight returns,
            relative-to-universe, breadth, valuation median, regime label,
            sample size) and a short excerpt of its latest published
            report (analyst view, what changed, whether the edition was
            degraded), plus the dependency-linked groups. Reads stored
            artifacts only — never generates a report or fetches prices.
            Use for "how is the industry doing", "what's my portfolio's
            industry exposure", or cross-industry market-colour questions.
            Returns `status: taxonomy_not_imported` or `snapshot: {status:
            no_snapshot}` when nothing has been published yet. Groups are
            named by their labels; refer to them by those labels, never by
            a numeric code or a third-party classification name."""
            from .pm_context import industry_context_payload
            try:
                return industry_context_payload(tickers=list(tickers or []), code=code)
            except Exception as exc:
                return {"error": f"industry context unavailable: {type(exc).__name__}"}

        # Wave 10 — specialists as live tools. Each `ask_*` re-fires the
        # corresponding specialist with the user's question routed
        # through the existing `prior_round_critique` channel. Bounded
        # at the agent level (the SDK enforces per-turn tool budgets);
        # individually, each tool is one LLM call against a specialist.

        @function_tool
        def ask_sector(ticker: str, question: str) -> dict[str, Any]:
            """Re-fire the sector analyst on `ticker` with a follow-up
            question. Use when you need a sector-grounded view that
            isn't in the cached memo (e.g. "what would change if rates
            fell 100bps?", "is the cohort margin trend reversing?").
            Returns the analyst's headline + key_points."""
            from ..services.fundamentals_service import get_full_financials
            from .sector_agents import run_sector_agent
            try:
                fin = get_full_financials((ticker or "").upper())
                profile = _profile_for(fin, ticker)
                ratios = fin.get("ratios") or {}
                finding = run_sector_agent(profile, ratios, prior_round_critique=question)
            except Exception as exc:
                return {"error": f"sector specialist failed: {exc}"}
            return _specialist_payload("sector", ticker, finding)

        @function_tool
        def ask_earnings(ticker: str, question: str) -> dict[str, Any]:
            """Re-fire the earnings analyst on `ticker` with a follow-up
            question grounded in the latest transcript ("what did the
            CEO defend most?", "did guidance change tone QoQ?"). Returns
            the analyst's headline + key_points."""
            from ..services.data_service import get_data_service
            from ..services.fundamentals_service import get_full_financials
            from .earnings_agent import run_earnings_agent
            try:
                fin = get_full_financials((ticker or "").upper())
                profile = _profile_for(fin, ticker)
                transcripts = get_data_service().get_earnings_transcripts(ticker.upper()) or []
                latest = transcripts[-1] if transcripts else {}
                finding = run_earnings_agent(
                    profile=profile, transcript=latest,
                    earnings=fin.get("earnings") or {},
                    prior_round_critique=question,
                )
            except Exception as exc:
                return {"error": f"earnings specialist failed: {exc}"}
            return _specialist_payload("earnings", ticker, finding)

        @function_tool
        def ask_filings(ticker: str, question: str) -> dict[str, Any]:
            """Re-fire the filings analyst on `ticker` with a follow-up
            grounded in the most recent 10-K/Q ("what's the new risk
            factor?", "did segment X get more disclosure this year?")."""
            from ..services.data_service import get_data_service
            from ..services.fundamentals_service import get_full_financials
            from .filing_agent import run_filing_agent
            try:
                fin = get_full_financials((ticker or "").upper())
                profile = _profile_for(fin, ticker)
                filings = get_data_service().get_filings(ticker.upper()) or []
                finding = run_filing_agent(
                    profile=profile, filings=filings,
                    prior_round_critique=question,
                )
            except Exception as exc:
                return {"error": f"filings specialist failed: {exc}"}
            return _specialist_payload("filings", ticker, finding)

        @function_tool
        def ask_valuation(ticker: str, question: str) -> dict[str, Any]:
            """Re-fire the valuation analyst on `ticker` with a follow-up
            question grounded in the live DCF + ratios ("why is the
            terminal multiple at 15x?", "what would a 50bps WACC change
            do to the implied price?")."""
            from ..services.fundamentals_service import get_full_financials
            from ..services.valuation_service import build_dcf
            from .valuation_agent import run_valuation_agent
            try:
                fin = get_full_financials((ticker or "").upper())
                profile = _profile_for(fin, ticker)
                ratios = fin.get("ratios") or {}
                dcf = build_dcf((ticker or "").upper())
                finding = run_valuation_agent(
                    profile=profile, ratios=ratios, dcf=dcf,
                    prior_round_critique=question,
                )
            except Exception as exc:
                return {"error": f"valuation specialist failed: {exc}"}
            return _specialist_payload("valuation", ticker, finding)

        @function_tool
        def ask_macro(question: str, ticker: str | None = None) -> dict[str, Any]:
            """Re-fire the macro analyst with a follow-up. If `ticker`
            is supplied, run the per-company macro pass (sensitivity to
            current regime); otherwise run the scenario pass."""
            try:
                if ticker:
                    from ..services.fundamentals_service import get_full_financials
                    from .macro_agent import run_macro_agent
                    fin = get_full_financials((ticker or "").upper())
                    profile = _profile_for(fin, ticker)
                    finding = run_macro_agent(
                        profile=profile, scenario=question,
                        prior_round_critique=question,
                    )
                    return _specialist_payload("macro", ticker, finding)
                from .macro_agent import run_macro_scenario
                scenario = run_macro_scenario(question)
                return {
                    "agent": "macro",
                    "scenario": scenario.scenario,
                    "narrative": scenario.narrative,
                    "favored_sectors": scenario.favored_sectors,
                    "pressured_sectors": scenario.pressured_sectors,
                    "risks": scenario.risks,
                }
            except Exception as exc:
                return {"error": f"macro specialist failed: {exc}"}

        return Agent(
            name="chat-pm",
            instructions=(
                "You are MarketMosaic's PM. The user is doing equity "
                "research and may ask anything from 'analyze NVDA' to "
                "'which software names have the strongest moats' to "
                "'what changed for ADBE in the latest 10-K'. You have "
                "tools — call them aggressively to fetch grounding "
                "data, then answer and cite the specific figures you "
                "relied on. "
                "Tool playbook:\n"
                "  • `get_memo(ticker)` — full memo with rating, score, "
                "    DCF, factor scores, key risks. Try first when the "
                "    user names a specific ticker.\n"
                "  • `get_company_lite(ticker)` — fast dossier when no "
                "    memo exists yet (sector, business desc, raw "
                "    metrics, factor scores). Good for comparative "
                "    questions across many names.\n"
                "  • `get_dcf_summary(ticker)` — bull/base/bear prices "
                "    + assumptions + PM adjustments.\n"
                "  • `get_comps(ticker)` — peer set, target vs peer "
                "    median, premium/discount on each multiple.\n"
                "  • `get_macro_snapshot()` — current FRED macro + "
                "    regime broadcast.\n"
                "  • `list_universe(sector?)` — what tickers are "
                "    available, optionally filtered by sector.\n"
                "  • `screener_query(sort_by, sector?, theme?, limit)` — "
                "    AI-ranked list. Use for 'top compounders', "
                "    'cheapest software', 'highest growth healthcare'.\n"
                "  • `custom_screen(rules_json, limit)` — strict "
                "    rule-based filter on raw metrics. Use when the "
                "    user gives numeric thresholds ('gross margin > "
                "    70% and P/E < 25').\n"
                "  • `get_industry_context(tickers?, code?)` — stored "
                "    weekly industry-group statistics, regime reads and "
                "    report excerpts for one name, a portfolio (pass the "
                "    list) or an explicit group slug. Observed data; "
                "    treat analyst views as scenarios.\n"
                "Live specialist follow-ups (use sparingly — each is "
                "a full specialist model call, roughly $0.05-0.15, max "
                "2 per turn):\n"
                "  • `ask_sector(ticker, question)` — re-fire the "
                "    sector analyst on a specific question.\n"
                "  • `ask_earnings(ticker, question)` — earnings "
                "    analyst with a follow-up.\n"
                "  • `ask_filings(ticker, question)` — filings analyst "
                "    with a follow-up.\n"
                "  • `ask_valuation(ticker, question)` — valuation "
                "    analyst with a follow-up.\n"
                "  • `ask_macro(question, ticker?)` — macro analyst.\n"
                "Style:\n"
                "  • Quote SPECIFIC numbers (rating, stock score, DCF "
                "    upside, factor scores, margins, growth). Don't "
                "    invent.\n"
                "  • For open-ended questions, structure the answer "
                "    as: theses you'd defend, theses you'd reject, "
                "    where you're uncertain. Cite the specific "
                "    figures you relied on.\n"
                "  • For mispricing questions, structure as: "
                "    consensus view → our view → gap → falsifiers.\n"
                "  • For comparative questions, rank candidates with "
                "    one-sentence justifications grounded in the data.\n"
                "  • For 'which should I buy', give a directional "
                "    answer grounded in metrics + one sentence on "
                "    what would change your view.\n"
                "  • If a tool returns `error`, briefly explain the "
                "    limitation and proceed with what you have.\n"
                "  • Always end: '_MarketMosaic is for research and "
                "    education only and does not provide personalized "
                "    financial advice._'"
            ),
            # CHAT_MODEL / CHAT_EFFORT (wave H: gpt-6-sol at medium); blank
            # is today's resolution, where an unset OPENAI_PM_MODEL is ""
            # and the real SDK rejects that, so the route default stands in.
            model=model,
            **({"model_settings": model_settings} if model_settings is not None else {}),
            tools=[
                get_memo, get_dcf_summary, get_comps, get_macro_snapshot,
                get_company_lite, list_universe, screener_query, custom_screen,
                get_industry_context,
                ask_sector, ask_earnings, ask_filings, ask_valuation, ask_macro,
            ],
        )
    except Exception as exc:
        log_safely(log, "chat-SDK agent build failed", exc)
        return None


def answer_via_sdk(
    *, message: str, history: list[Any], run_id: str | None = None,
) -> str | None:
    """Run the chat agent and return its final markdown answer, or None
    when the SDK isn't usable, the run failed or the model refused (the
    caller falls back to the legacy single-shot path)."""
    return run_chat_turn(message=message, history=history, run_id=run_id)[0]


def run_chat_turn(
    *, message: str, history: list[Any], run_id: str | None = None,
) -> tuple[str | None, bool]:
    """(answer or None, whether an SDK model run was attempted).

    The second value lets the orchestrator keep to one attempt per
    provider per turn: once the SDK has spent the turn's OpenAI attempt,
    the legacy fallback must not try the SDK again or hop back to OpenAI.

    `run_id` is the turn's `chat:<hex>` id, minted by the chat route and
    read from the call context when not passed (attribution critique #5):
    the SDK usage rows, the `SDKTrace` row and the legacy fallback's rows
    all carry it.
    """
    agent = _build_chat_agent()
    if agent is None:
        return None, False
    run_id = run_id or llm.current_call_context().get("run_id") or f"chat:{uuid.uuid4().hex}"
    started = time.perf_counter()

    # Build the seed prompt from message + recent history. The agent
    # gets the user's question + a few prior turns for context; it
    # decides which tools to call based on what it needs.
    history_block = "\n".join(
        f"- {(h.role if hasattr(h, 'role') else h.get('role', '?'))}: "
        f"{((h.content if hasattr(h, 'content') else h.get('content', '')) or '')[:300]}"
        for h in history[-6:]
    )
    # Wave 10 — inject PM brain + research_notes into the seed so the
    # SDK agent has the same context the legacy path gets. Tickers /
    # sectors mentioned in the message drive memory routing; the PM
    # brain + research_notes load unconditionally.
    pm_ctx = ""
    try:
        from .orchestrator import _extract_tickers
        from .pm_context import build_pm_context
        ticks = _extract_tickers(message + " " + history_block)
        first_ticker = ticks[0] if ticks else None
        first_sector = None
        if first_ticker:
            from ..database import SessionLocal
            from ..models import Company
            with SessionLocal() as db:
                c = db.get(Company, first_ticker)
                if c is not None:
                    first_sector = c.sector
        pm_ctx = build_pm_context(
            ticker=first_ticker, sector=first_sector,
            profile={"ticker": first_ticker, "sector": first_sector},
        )
    except Exception as exc:  # pragma: no cover — never block chat
        log_safely(log, "PM context for SDK seed failed", exc, level=logging.DEBUG)

    seed = (
        ((pm_ctx + "\n\n---\n\n") if pm_ctx else "")
        + f"Conversation so far:\n{history_block}\n\n"
        f"User's new question:\n{message}"
    )

    hooks = sdk_usage_hooks("chat.sdk_turn", run_id=run_id)
    try:
        from agents import Runner as RealRunner
        # An umbrella context: the run id only, never an agent name, so the
        # specialist calls the `ask_*` tools make keep their own agents.
        with llm.llm_call_context(run_id=run_id):
            result = RealRunner.run_sync(agent, seed, hooks=hooks, run_config=sdk_run_config())
    except Exception as exc:
        hooks.record_failure(exc, agent=str(getattr(agent, "name", "") or ""),
                             model=_agent_model_name(agent))
        kind = "refusal" if _is_refusal(exc) else "failure"
        # Type only: an SDK exception can quote the model's output (a
        # refusal's text, a malformed tool call's arguments).
        log.warning("chat-SDK run %s (%s); answering on the legacy path", kind, type(exc).__name__)
        _persist_chat_trace(
            run_id=run_id, new_items=None, error=type(exc).__name__,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return None, True

    final_output = getattr(result, "final_output", None)
    _persist_chat_trace(
        run_id=run_id, new_items=getattr(result, "new_items", None),
        error="refusal" if hooks.refused else "",
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    if hooks.refused:
        log.warning("chat-SDK run refused; answering on the legacy path")
        return None, True
    if not final_output or not str(final_output).strip():
        return None, True
    body = str(final_output).strip()
    if "research and education only" not in body.lower():
        body += (
            "\n\n_MarketMosaic is for research and education only and "
            "does not provide personalized financial advice._"
        )
    return body, True


def _persist_chat_trace(
    *, run_id: str, new_items: Any, error: str = "", duration_ms: int = 0,
) -> None:
    """Write the chat-surface SDKTrace row: what ran, never what the model
    wrote (FIX-020). `final_output` is stored empty and `new_items` is
    reduced to `{type, agent, tool}`. Best-effort, never raises; lives
    here rather than in `sdk_runtime` to avoid an import cycle and to tag
    `surface='chat'` correctly."""
    try:
        from ..database import SessionLocal
        from ..models import SDKTrace
        with SessionLocal() as session:
            session.add(SDKTrace(
                run_id=run_id[:64], ticker=None, surface="chat",
                final_output="",
                new_items=trace_items(new_items),
                error=str(error)[:2000],
                duration_ms=int(duration_ms),
            ))
            session.commit()
    except Exception as exc:  # pragma: no cover — telemetry must not block
        log_safely(log, "chat SDKTrace persistence failed (non-fatal)", exc, level=logging.DEBUG)
