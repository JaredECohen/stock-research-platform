"""Wave 10 — Freeform chat routed through the OpenAI Agents SDK.

The legacy `_answer_with_memo_context` is a single-shot LLM call that
injects compact memo summaries into one prompt. That works for "compare
MSFT vs GOOGL" (both memos already exist) but breaks down when the user
asks something the prepacked summary doesn't cover ("what's the WACC the
PM used?", "what's the comps median EV/EBITDA?", "what's the latest
macro snapshot?").

This module builds a real SDK Agent with `function_tool`s that lazily
fetch the data the user actually asked for. The agent decides which
tools to call. We persist `new_items` to `SDKTrace` (surface='chat') so
reviewers can see what the agent fetched and how it reasoned.

Skip conditions:
- `USE_AGENTS_SDK=false` → caller falls back to legacy single-shot.
- `openai-agents` not installed → same.
- No `OPENAI_API_KEY` → same.

Failure conditions:
- SDK call raises → returns None, caller falls back to legacy path.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..config import settings

log = logging.getLogger(__name__)


def _can_use_sdk() -> bool:
    """Same gate as `sdk_runtime._can_use_real_sdk` — kept inline so this
    module's import doesn't pull in the legacy SDK runtime if we end up
    deprecating it."""
    if not settings.use_agents_sdk:
        return False
    if not settings.openai_api_key:
        return False
    try:
        import agents  # noqa: F401
        return True
    except Exception:
        return False


def _build_chat_agent() -> Optional[Any]:
    """Wire an `Agent` with the four data-fetch tools the chat handler
    might need. Returns None if the SDK isn't usable."""
    if not _can_use_sdk():
        return None
    try:
        from agents import Agent, function_tool

        @function_tool
        def get_memo(ticker: str) -> Dict[str, Any]:
            """Return the latest cached investment memo for `ticker`. The
            response is the compact projection that includes the rating,
            stock score, one-sentence thesis, key risks, the bull/bear
            case headlines, and the PM-adjusted DCF summary. Use this
            FIRST for any question about a specific name. Returns
            `{"error": "..."}` if no memo exists yet."""
            from ..services.memo_store import latest_memo, memo_to_pydantic
            from .orchestrator import _memo_for_chat_context
            snap = latest_memo((ticker or "").upper())
            if snap is None:
                return {"error": f"No memo cached for {ticker}. The user may need to run an analysis first."}
            try:
                m = memo_to_pydantic(snap)
            except Exception as exc:
                return {"error": f"Failed to load memo for {ticker}: {exc}"}
            return _memo_for_chat_context(m)

        @function_tool
        def get_dcf_summary(ticker: str) -> Dict[str, Any]:
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
        def get_comps(ticker: str) -> Dict[str, Any]:
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
        def get_macro_snapshot() -> Dict[str, Any]:
            """Return the current macro snapshot (FRED data — Fed Funds,
            10y yield, core sticky CPI, unemployment, etc.) plus the
            most recent regime broadcast. Use when the user asks about
            macro context or how a regime change affects a sector/name."""
            from ..services.macro_service import macro_snapshot
            from ..cache import cache_get
            snap = macro_snapshot() or {}
            broadcast = cache_get("macro:global", "macro_broadcast")
            return {
                "fred_snapshot": snap,
                "regime_broadcast": (
                    broadcast.payload if broadcast and isinstance(broadcast.payload, dict)
                    else None
                ),
            }

        return Agent(
            name="chat-pm",
            instructions=(
                "You are MarketMosaic's PM answering a follow-up question "
                "from a research user who has already seen full memos for "
                "the tickers they're discussing. You have four tools — "
                "call them as needed to fetch the data you cite. "
                "Discipline:\n"
                "1. ALWAYS call `get_memo(ticker)` first for any name "
                "   the user asks about — that's where rating, score, "
                "   thesis, and key risks live.\n"
                "2. Reach for `get_dcf_summary` when the user asks "
                "   valuation mechanics; `get_comps` for peer-relative; "
                "   `get_macro_snapshot` for regime / rate questions.\n"
                "3. Quote SPECIFIC NUMBERS (rating, stock score, DCF "
                "   upside, factor scores). Do NOT invent data.\n"
                "4. When the user asks 'which should I buy' on multiple "
                "   names, give a directional answer grounded in the "
                "   metrics, then add ONE sentence on what would change "
                "   your view.\n"
                "5. Always end with the disclaimer: '_MarketMosaic is "
                "   for research and education only and does not provide "
                "   personalized financial advice._'"
            ),
            model=settings.openai_pm_model,
            tools=[get_memo, get_dcf_summary, get_comps, get_macro_snapshot],
        )
    except Exception as exc:
        log.warning("chat-SDK agent build failed: %s", exc)
        return None


def answer_via_sdk(
    *, message: str, history: List[Any],
) -> Optional[str]:
    """Run the chat agent and return its final markdown answer.

    Returns None if the SDK isn't usable or the run failed. Caller
    falls back to the legacy single-shot path on None.

    Persists the trace via `SDKTrace` (surface='chat') keyed by a fresh
    `run_id` so the admin viewer can pull it up. The chat trace shares
    no run_id with any memo run — chat is its own surface.
    """
    agent = _build_chat_agent()
    if agent is None:
        return None
    run_id = str(uuid.uuid4())
    started = time.perf_counter()

    # Build the seed prompt from message + recent history. The agent
    # gets the user's question + a few prior turns for context; it
    # decides which tools to call based on what it needs.
    history_block = "\n".join(
        f"- {(h.role if hasattr(h, 'role') else h.get('role', '?'))}: "
        f"{((h.content if hasattr(h, 'content') else h.get('content', '')) or '')[:300]}"
        for h in history[-6:]
    )
    seed = (
        f"Conversation so far:\n{history_block}\n\n"
        f"User's new question:\n{message}"
    )

    try:
        from agents import Runner as RealRunner
        result = RealRunner.run_sync(agent, seed)
    except Exception as exc:
        log.warning("chat-SDK run failed: %s", exc)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _persist_chat_trace(
            run_id=run_id, final_output="", new_items=None,
            error=str(exc), duration_ms=elapsed_ms,
        )
        return None

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    final_output = getattr(result, "final_output", None)
    new_items = getattr(result, "new_items", None)
    _persist_chat_trace(
        run_id=run_id, final_output=final_output or "",
        new_items=new_items, duration_ms=elapsed_ms,
    )

    if not final_output or not str(final_output).strip():
        return None
    body = str(final_output).strip()
    if "research and education only" not in body.lower():
        body += (
            "\n\n_MarketMosaic is for research and education only and "
            "does not provide personalized financial advice._"
        )
    return body


def _persist_chat_trace(
    *, run_id: str, final_output: str, new_items: Any,
    error: str = "", duration_ms: int = 0,
) -> None:
    """Write the chat-surface SDKTrace row. Shares the persistence helper
    contract from `sdk_runtime._persist_sdk_trace` (best-effort, never
    raises) but lives here to avoid an import cycle and to tag
    `surface='chat'` correctly."""
    try:
        from ..database import SessionLocal
        from ..models import SDKTrace
        items_payload: list = []
        for item in (new_items or []):
            if hasattr(item, "model_dump"):
                try:
                    items_payload.append(item.model_dump())
                    continue
                except Exception:
                    pass
            if hasattr(item, "__dict__"):
                try:
                    items_payload.append({
                        k: v for k, v in vars(item).items()
                        if not k.startswith("_")
                    })
                    continue
                except Exception:
                    pass
            items_payload.append({"repr": repr(item)[:500]})
        with SessionLocal() as session:
            session.add(SDKTrace(
                run_id=run_id, ticker=None, surface="chat",
                final_output=str(final_output)[:8000],
                new_items=items_payload[:200],
                error=str(error)[:2000],
                duration_ms=int(duration_ms),
            ))
            session.commit()
    except Exception as exc:  # pragma: no cover — telemetry must not block
        log.debug("chat SDKTrace persistence failed (non-fatal): %s", exc)
