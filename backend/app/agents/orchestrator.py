"""PM Orchestrator — entry point for the chat router.

Classifies intent (LLM if available, regex fallback), dispatches to the
appropriate sub-graph, and synthesizes the final response.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import select

from ..config import settings
from ..finance.dcf import fmt_price, fmt_upside
from ..schemas import (
    ChatMessage,
    ChatResponse,
    DCFResult,
    IntentType,
    MacroScenarioResult,
    ModelPortfolio,
    PortfolioRequest,
    ScreenerResult,
    StockMemoOut,
)
from ..services import memo_sections
from ..services.data_service import get_data_service
from ..services.macro_service import macro_snapshot
from ..services.portfolio_service import build_model_portfolio
from ..services.screener_service import compute_universe_scores
from ..services.valuation_service import build_comps, build_dcf
from . import llm, prompts
from .graph import default_agent_trace, run_stock_memo
from .log_safety import log_safely
from .macro_agent import run_macro_scenario

log = logging.getLogger(__name__)


KNOWN_THEMES = {
    "ai infrastructure": "ai_infrastructure",
    "ai capex": "ai_infrastructure",
    "falling rates": "falling_rates",
    "rate cuts": "falling_rates",
    "sticky inflation": "sticky_inflation",
    "recession": "recession_defense",
    "defensive": "recession_defense",
    "high quality": "high_quality_compounders",
    "compounders": "high_quality_compounders",
    "margin expansion": "margin_expansion",
    "reasonable valuation": "reasonable_valuation_growth",
    "soft landing": "ai_infrastructure",
}


def _ticker_re() -> re.Pattern:
    return re.compile(r"\$?\b([A-Z]{1,5})\b")


def _extract_tickers(text: str) -> list[str]:
    universe = set(get_data_service().list_tickers())
    found = []
    for tok in _ticker_re().findall(text):
        if tok in universe and tok not in found:
            found.append(tok)
    # Map common company name aliases
    aliases = {
        "nvidia": "NVDA", "microsoft": "MSFT", "alphabet": "GOOGL",
        "google": "GOOGL", "meta": "META", "amazon": "AMZN", "apple": "AAPL",
        "broadcom": "AVGO", "amd": "AMD", "jpmorgan": "JPM", "jpm": "JPM",
        "goldman": "GS", "morgan stanley": "MS", "visa": "V",
        "mastercard": "MA", "costco": "COST", "walmart": "WMT",
        "home depot": "HD", "mcdonald": "MCD", "starbucks": "SBUX",
        "nike": "NKE", "lilly": "LLY", "merck": "MRK", "johnson": "JNJ",
        "united health": "UNH", "exxon": "XOM", "nextera": "NEE",
        "caterpillar": "CAT", "salesforce": "CRM",
    }
    low = text.lower()
    for alias, ticker in aliases.items():
        if alias in low and ticker in universe and ticker not in found:
            found.append(ticker)
    return found


def _extract_theme(text: str) -> str | None:
    low = text.lower()
    for k, v in KNOWN_THEMES.items():
        if k in low:
            return v
    return None


def classify_intent(message: str) -> tuple[IntentType, list[str], str | None]:
    """Classify intent. Tries LLM first, falls back to deterministic rules."""
    llm_out = llm.chat_json(
        prompts.INTENT_CLASSIFIER_PROMPT + "\n\nMessage:\n" + message,
        system=prompts.PM_SYSTEM, route="cheap", action="chat.classify",
    )
    if llm_out and llm_out.get("intent"):
        intent = llm_out["intent"]
        tickers = [t.upper() for t in (llm_out.get("tickers") or [])]
        theme = llm_out.get("theme")
        if intent in (
            "single_stock_analysis", "stock_comparison", "thematic_screen",
            "macro_question", "portfolio_construction", "dcf_analysis",
            "comps_analysis", "general_research_chat",
        ):
            return intent, tickers, theme

    # Deterministic fallback
    low = message.lower()
    tickers = _extract_tickers(message)
    theme = _extract_theme(message)

    if "dcf" in low or "discounted cash flow" in low:
        return "dcf_analysis", tickers, theme
    if "comps" in low or "peer" in low or "peer group" in low:
        return "comps_analysis", tickers, theme
    if "compare" in low and len(tickers) >= 2:
        return "stock_comparison", tickers, theme
    # Thematic screens take priority over portfolio construction when the user asks
    # to FIND/SCREEN/SHOW/RANK stocks, even if a theme word like "rates" is present.
    if any(k in low for k in ("find", "show me", "rank", "screen", "list ", "ideas", "stocks that", "names that")):
        return "thematic_screen", tickers, theme
    if "build" in low and ("portfolio" in low or "holdings" in low):
        return "portfolio_construction", tickers, theme
    if "portfolio" in low and "perspective" not in low:
        return "portfolio_construction", tickers, theme
    if any(k in low for k in ("inflation", "recession", "fed funds", "macro", "soft landing", "yield curve", "rate cut")):
        return "macro_question", tickers, theme
    if any(k in low for k in ("high-quality", "high quality", "compounders", "valuation growth")):
        return "thematic_screen", tickers, theme
    if tickers:
        return "single_stock_analysis", tickers, theme
    return "general_research_chat", tickers, theme


# ---------------------------------------------------------------------------
# Helpers to render answers from structured data
# ---------------------------------------------------------------------------

def _confidence_text(memo: StockMemoOut, *, fmt: str) -> str:
    """The confidence as a presented memo shows it: the number, or the
    owner's wording when W2a hides it (a template PM view)."""
    if memo_sections.is_hidden(memo, "confidence_score"):
        return "confidence unavailable in this version"
    return fmt.format(int(memo.confidence_score))


def _rating_note(memo: StockMemoOut) -> str:
    entry = (memo.section_availability or {}).get("rating_label")
    if entry is not None and entry.reason == "pm_view_unavailable":
        return " (PM view unavailable; rating reflects the quantitative factor blend)"
    return ""


DEBATE_OUTCOME_MAX_CHARS = 800
_DEBATE_CRUX_MAX_CHARS = 400
_DEBATE_STATUS_TEXT = {
    "complete": "Bull/bear debate complete.",
    "partial": "Bull/bear debate partly complete (one phase did not finish).",
}


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _debate_outcome(m: StockMemoOut) -> str | None:
    """The chat's bounded read of the bull/bear debate (design §12.4), or
    None when there is nothing to say.

    Built from a PRESENTED memo, so it shows only what the memo page shows:
    nothing when the debate section is hidden or never ran (every memo
    written with DEBATE_MODE off, so today's chat context is unchanged), no
    PM crux or rulings when the PM view is hidden or the PM did not rule,
    and an unanswered count over the claims the page displays (dropped and
    unsupported claims are withheld there). Status, crux, rulings per side,
    unanswered count; at most DEBATE_OUTCOME_MAX_CHARS characters.
    """
    rec = m.debate
    if rec is None or rec.status == "not_run" or memo_sections.is_hidden(m, "debate"):
        return None
    if rec.status == "unavailable":
        return "The bull/bear debate was unavailable for this memo."
    parts = [_DEBATE_STATUS_TEXT.get(rec.status, "Bull/bear debate ran.")]
    res = rec.resolution
    if res.status == "ruled" and not memo_sections.is_hidden(m, "final_pm_view"):
        if res.crux.strip():
            parts.append(f"PM crux: {_clip(res.crux, _DEBATE_CRUX_MAX_CHARS)}")
        tally = {"bull": 0, "bear": 0, "split": 0, "unresolved": 0}
        for ruling in res.rulings:
            key = ruling.ruling if ruling.ruling in tally else "unresolved"
            tally[key] += 1
        parts.append(
            f"PM rulings on the disputes: bull {tally['bull']}, bear {tally['bear']}, "
            f"split {tally['split']}, unresolved {tally['unresolved']}."
        )
    else:
        parts.append("The PM did not adjudicate this debate.")
    shown = {c.id for c in rec.claims if not c.dropped and c.grade != "unsupported"}
    unanswered = sum(1 for claim_id in rec.unanswered if claim_id in shown)
    parts.append(f"Claims left unanswered by the other side: {unanswered}.")
    return _clip(" ".join(parts), DEBATE_OUTCOME_MAX_CHARS)


def _render_memo_answer(memo: StockMemoOut) -> str:
    """Chat rendering of a PRESENTED memo (W2a): hidden prose already reads
    `UNAVAILABLE_TEXT` and hidden list items are already filtered, so each
    line prints what the memo page shows; confidence and the rating note
    consult `section_availability` because the presenter never rewrites a
    number."""
    bullets = []
    bullets.append(f"**{memo.ticker} — {memo.company_name}** ({memo.sector})")
    bullets.append(
        f"Rating: **{memo.rating_label}**{_rating_note(memo)} · "
        f"{_confidence_text(memo, fmt='confidence {}/100')}"
    )
    bullets.append(f"_Thesis:_ {memo.one_sentence_thesis}")
    bullets.append("")
    bullets.append(f"**PM View:** {memo.final_pm_view}")
    bullets.append("")
    bullets.append(f"**Sector ({memo.sector_agent_view.agent}):** {memo.sector_agent_view.summary}")
    bullets.append(f"**Earnings:** {memo.earnings_agent_view.summary}")
    bullets.append(f"**Filing:** {memo.filing_agent_view.summary}")
    bullets.append(f"**Valuation:** {memo.valuation_agent_view.summary}")
    bullets.append(f"**Comps:** {memo.comps_agent_view.summary}")
    bullets.append(f"**Macro:** {memo.macro_sensitivity.summary}")
    bullets.append("")
    bullets.append("**Bull case:**")
    for k in memo.bull_case.key_points[:4]:
        bullets.append(f"- {k}")
    bullets.append("**Bear case:**")
    for k in memo.bear_case.key_points[:4]:
        bullets.append(f"- {k}")
    debate = _debate_outcome(memo)
    if debate:
        bullets.append(f"**Debate:** {debate}")
    bullets.append("")
    bullets.append(f"**Risk Committee:** {memo.risk_committee_challenge.overall_assessment}")
    if memo.risk_committee_challenge.challenges:
        bullets.append("Challenges raised:")
        for c in memo.risk_committee_challenge.challenges[:3]:
            bullets.append(f"- {c}")
    if memo.dcf_summary:
        bullets.append("")
        bullets.append(f"**DCF:** {memo.dcf_summary.get('summary', '')}")
    bullets.append("")
    bullets.append(f"_Final verdict:_ {memo.final_verdict}")
    bullets.append("")
    bullets.append(f"_{memo.disclaimer}_")
    return "\n".join(bullets)


def _render_comparison_answer(memos: list[StockMemoOut]) -> str:
    parts = ["**Cross-comparison from a PM's perspective:**\n"]
    for m in memos:
        parts.append(f"### {m.ticker} — {m.rating_label} ({_confidence_text(m, fmt='confidence {}')})")
        parts.append(m.one_sentence_thesis)
        parts.append(f"- Bull: {m.bull_case.headline}")
        parts.append(f"- Bear: {m.bear_case.headline}")
        if m.dcf_summary:
            parts.append(f"- DCF: {m.dcf_summary.get('summary', '')}")
        parts.append("")
    parts.append("**PM synthesis:** ratings, valuation triangulation, and risk profiles diverge as above. "
                 "Sizing in a model portfolio depends on risk level and macro view.")
    return "\n".join(parts)


def _render_portfolio_answer(p: ModelPortfolio) -> str:
    lines = [f"**Model portfolio: {p.name}** — '{p.market_view}', risk level: {p.risk_level}",
             f"_Expected vol proxy: {p.expected_volatility:.1%}_",
             ""]
    lines.append("**Holdings:**")
    for h in p.holdings:
        lines.append(f"- {h.ticker} ({h.sector}) — {h.weight:.1%}: {h.rationale}")
    lines.append("")
    lines.append("**Sector allocation:**")
    for s, w in p.sector_allocation.items():
        lines.append(f"- {s}: {w:.0%}")
    lines.append("")
    lines.append("**Risk notes:**")
    for n in p.risk_notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("**What could invalidate the portfolio:**")
    for n in p.what_could_invalidate:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("**Watch items:**")
    for n in p.watch_items:
        lines.append(f"- {n}")
    lines.append("")
    lines.append(f"_{p.disclaimer}_")
    return "\n".join(lines)


def _render_macro_answer(s: MacroScenarioResult) -> str:
    lines = [f"**Scenario: {s.scenario}**", "", s.narrative, ""]
    lines.append("**Sector impacts:**")
    for sector, view in s.sector_impacts.items():
        lines.append(f"- {sector}: {view}")
    lines.append("")
    lines.append(f"**Favored sectors:** {', '.join(s.favored_sectors)}")
    lines.append(f"**Pressured sectors:** {', '.join(s.pressured_sectors)}")
    lines.append("")
    lines.append("**Suggested research views:**")
    for v in s.suggested_research_views:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("**Risks:**")
    for r in s.risks:
        lines.append(f"- {r}")
    return "\n".join(lines)


def _render_screener_answer(r: ScreenerResult, *, top_n: int = 7) -> str:
    lines = [f"**Top-ranked ideas{' for theme: ' + r.theme if r.theme else ''}:**", ""]
    for row in r.rows[:top_n]:
        lines.append(
            f"- **{row.ticker}** ({row.sector}) · PM {row.pm_score:.0f} · Q{row.quality:.0f} G{row.growth:.0f} V{row.valuation:.0f} R{row.risk:.0f} — {row.one_line_thesis}"
        )
    return "\n".join(lines)


def _render_dcf_answer(d: DCFResult) -> str:
    lines = [f"**DCF for {d.ticker}**", ""]
    lines.append(f"WACC: {d.base.assumptions.wacc:.2%} · Terminal growth: {d.base.assumptions.terminal_growth:.1%}")
    lines.append("")
    for s in (d.base, d.bull, d.bear):
        # "n/a" rather than a crash or "$0.00" when the share count or
        # quote never reached the model — see `finance.dcf.fmt_price`.
        lines.append(
            f"- {s.name.capitalize()} implied price: "
            f"{fmt_price(s.implied_share_price)} ({fmt_upside(s.upside_pct)})"
        )
    if any(s.tv_clamped for s in (d.base, d.bull, d.bear)):
        lines.append("- ⚠ Terminal value clamped (WACC − terminal growth ≤ 0.5%) — implied prices are not trustworthy")
    lines.append("")
    lines.append(d.summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_FOLLOWUP_HINTS = (
    "which", "why", "how", "compare", "explain", "what about",
    "what's", "what is", "moat", "better", "worse", "cheaper",
    "expensive", "more", "less", "vs", "versus", "differ",
    "of these", "of those", "from above", "from the list",
    "the screener", "the screen",
)


def _is_conceptual_followup(message: str, history: list[ChatMessage]) -> bool:
    """Heuristic — should we route this through the SDK chat agent
    instead of a workflow handler?

    Returns True when:
      • There's prior chat history (any follow-up turn).
      • OR the message starts with / contains a conceptual cue
        ("which", "why", "compare X and Y on …", etc.) — these
        are usually requests to *reason over* prior context, not
        to fire a fresh workflow.
    """
    if history:
        return True
    low = message.lower().strip()
    if any(low.startswith(h) for h in _FOLLOWUP_HINTS):
        return True
    if any(f" {h} " in f" {low} " for h in _FOLLOWUP_HINTS):
        return True
    return False


def _try_sdk_chat(message: str, history: list[ChatMessage] | None) -> tuple[str | None, bool]:
    """Try the OpenAI Agents SDK chat agent (14 tools).

    Returns (answer or None, attempted). None on any failure or refusal so
    callers fall back to the legacy handlers; `attempted` says whether the
    SDK spent this turn's OpenAI attempt (plan C1: at most one attempt per
    provider per turn). Gated on `CHAT_AGENTS_SDK` (plan P14), not on the
    legacy `USE_AGENTS_SDK`."""
    if not settings.chat_agents_sdk:
        return None, False
    try:
        from .chat_sdk import run_chat_turn
        return run_chat_turn(message=message, history=history or [])
    except Exception as exc:
        # Unknown how far it got, so the attempt counts as spent.
        log.warning("chat-SDK turn raised %s; answering on the legacy path", type(exc).__name__)
        return None, True


def _legacy_answer_route(sdk_attempted: bool) -> dict[str, Any]:
    """chat_text kwargs for the single-shot `chat.answer` fallback.

    Before any SDK attempt: today's call (its own failover included). After
    the SDK has run on OpenAI this turn, the fallback makes ONE attempt on a
    provider the turn has not tried: no failover hop (with the research
    tier on, Opus 5.5 would otherwise fail over to gpt-6-sol, a second
    OpenAI attempt), and a route that would itself land on OpenAI moves to
    Anthropic when a key is configured. That move is `spent_provider`, not
    `provider_override`: a configured `chat.answer` tier replaces the
    override, so an OpenAI tier (LLM_RESEARCH_MODEL=gpt-6-sol, or the
    `chat.answer:chat` rollback) used to go straight back to OpenAI.
    An OpenAI-only deployment still gets its one single-shot attempt: the
    SDK run failing (a tool loop, max turns, a refusal) says little about a
    plain completion.
    """
    if not sdk_attempted:
        return {}
    return {"failover": False, "spent_provider": "openai"}


def _stored_memo(ticker: str) -> StockMemoOut | None:
    """The latest persisted memo for `ticker`, or None. Reads only; the
    lazy import mirrors `_answer_with_memo_context` (memo_store imports
    the graph's schemas, and this module is imported early)."""
    from ..services import memo_store
    snap = memo_store.latest_memo(ticker)
    if snap is None:
        return None
    try:
        # W2a: chat is a customer exit; it serves the presented memo.
        return memo_store.present_snapshot(snap)
    except Exception as exc:
        log_safely(log, f"stored memo for {ticker} could not be hydrated for chat", exc)
        return None


def _render_needs_analysis(tickers: list[str]) -> str:
    names = ", ".join(tickers)
    plural = "s" if len(tickers) != 1 else ""
    return (
        f"There is no stored research memo for {names} yet, and the committee does not run "
        f"inside a chat turn. Run research on the ticker{plural} from the Research page "
        "(a research run is charged against your plan's allowance) and ask again once the "
        "memo is ready."
    )


class Orchestrator:
    def chat(
        self,
        message: str,
        history: list[ChatMessage] | None = None,
        *,
        allow_inline_memo: bool = True,
    ) -> ChatResponse:
        """One chat turn.

        `allow_inline_memo` (FEAT-002) says whether this turn may start a
        full memo run in-process for `single_stock_analysis` /
        `stock_comparison`. True is the historical behaviour and the
        default, so nothing changes for a caller that never heard of the
        flag. False — what `routes_chat` passes while the login wall is on
        — answers from `memo_store.latest_memo` and names the tickers
        without a memo in `ChatResponse.needs_analysis` so the UI can offer
        a (charged, worker-side) research run instead. Both inline entry
        points are behind the flag: the legacy `run_stock_memo` and the
        SDK runtime's `run_stock_memo_via_sdk`, which is resolved at call
        time and would otherwise run a full memo under its own run_id.
        """
        intent, tickers, theme = classify_intent(message)
        trace = default_agent_trace(intent)
        sdk_attempted = False

        # Wave 9b — flexible chat routing. When the user is on a
        # follow-up turn (history non-empty) or asking a conceptual
        # question ("which has the best moat?", "why is META
        # cheaper?"), prefer the SDK chat agent over the workflow
        # handlers. The agent has tools to fetch memo/DCF/comps/macro/
        # universe/screener/custom_screen and reasons over the result.
        # Workflow handlers still fire for unambiguous first-message
        # asks ("Analyze NVDA", "Compare MSFT and GOOGL") so the heavy
        # memo path runs only when the user actually wants it.
        if _is_conceptual_followup(message, history or []):
            sdk_answer, sdk_attempted = _try_sdk_chat(message, history)
            if sdk_answer:
                return ChatResponse(
                    intent="general_research_chat",
                    answer=sdk_answer, agent_trace=trace,
                )
            # else: fall through to intent-based routing.

        if intent == "single_stock_analysis" and tickers:
            ticker = tickers[0]
            if not allow_inline_memo:
                memo = _stored_memo(ticker)
                if memo is None:
                    return ChatResponse(
                        intent=intent, answer=_render_needs_analysis([ticker]),
                        agent_trace=trace, needs_analysis=[ticker],
                    )
            # Phase 3: route through the Agents SDK runtime when enabled. The
            # runtime ultimately returns the same StockMemoOut shape, so the
            # downstream rendering / tracing is identical. Only the legacy
            # USE_AGENTS_SDK does this: CHAT_AGENTS_SDK (plan P14) moves the
            # chat agent alone, so the inline memo stays on the graph.
            elif settings.use_agents_sdk:
                from .sdk_runtime import run_stock_memo_via_sdk
                memo = memo_sections.present_memo(run_stock_memo_via_sdk(ticker))
            else:
                memo = memo_sections.present_memo(run_stock_memo(ticker))
            return ChatResponse(
                intent=intent, answer=_render_memo_answer(memo),
                agent_trace=trace, memo=memo, sources=memo.sources_used,
            )

        if intent == "stock_comparison" and len(tickers) >= 2:
            # Defensive: if one ticker's memo blows up, the comparison still
            # renders for the rest. Each `run_stock_memo` is already
            # safe-runner-protected internally, so errors here would only come
            # from the unrecoverable "unknown ticker" case.
            memos = []
            # (c) RP-001 / D8: a dropped ticker used to vanish from the
            # comparison with no trace — the user asked about four names
            # and silently read about three. The note rides on `sources`
            # (appended after the cap so it is never truncated away).
            unavailable: list[str] = []
            # FEAT-002: tickers with no stored memo when generation is not
            # allowed in-request; reported separately so the UI can offer
            # a research run rather than an error.
            missing: list[str] = []
            for t in tickers[:4]:
                if not allow_inline_memo:
                    stored = _stored_memo(t)
                    if stored is None:
                        missing.append(t)
                    else:
                        memos.append(stored)
                    continue
                try:
                    memos.append(memo_sections.present_memo(run_stock_memo(t)))
                except Exception as exc:
                    log_safely(log, f"comparison memo unavailable for {t}", exc)
                    unavailable.append(t)
            unavailable_notes = [f"memo unavailable: {t}" for t in unavailable]
            unavailable_notes += [f"memo not yet generated: {t}" for t in missing]
            if not memos:
                answer = (
                    _render_needs_analysis(missing) if missing
                    else "Could not generate any memos for the requested tickers."
                )
                return ChatResponse(
                    intent=intent, answer=answer,
                    agent_trace=trace, sources=unavailable_notes, needs_analysis=missing,
                )
            answer = _render_comparison_answer(memos)
            if missing:
                answer += "\n\n_" + _render_needs_analysis(missing) + "_"
            return ChatResponse(
                intent=intent, answer=answer,
                agent_trace=trace, memo=memos[0],
                sources=[s for m in memos for s in m.sources_used][:20] + unavailable_notes,
                needs_analysis=missing,
            )

        if intent == "dcf_analysis" and tickers:
            ticker = tickers[0]
            dcf = build_dcf(ticker)
            if dcf is None:
                return ChatResponse(
                    intent=intent, answer=f"Could not build a DCF for {ticker} — try a supported ticker.",
                    agent_trace=trace,
                )
            return ChatResponse(
                intent=intent, answer=_render_dcf_answer(dcf),
                agent_trace=trace, dcf=dcf,
            )

        if intent == "comps_analysis" and tickers:
            ticker = tickers[0]
            comps = build_comps(ticker)
            if comps is None:
                return ChatResponse(
                    intent=intent, answer=f"Could not build comps for {ticker} — try a supported ticker.",
                    agent_trace=trace,
                )
            ans = (
                f"**Comps for {ticker}**\n\n"
                f"Peers: {', '.join(p.ticker for p in comps.peers)}\n\n"
                f"{comps.interpretation}"
            )
            return ChatResponse(intent=intent, answer=ans, agent_trace=trace, comps=comps)

        if intent == "portfolio_construction":
            request = PortfolioRequest(market_view=message, num_holdings=10)
            portfolio = build_model_portfolio(request)
            return ChatResponse(
                intent=intent, answer=_render_portfolio_answer(portfolio),
                agent_trace=trace, portfolio=portfolio,
            )

        if intent == "thematic_screen":
            screener = compute_universe_scores(theme=theme)
            return ChatResponse(
                intent=intent, answer=_render_screener_answer(screener),
                agent_trace=trace, screener=screener,
            )

        if intent == "macro_question":
            scenario = run_macro_scenario(message)
            return ChatResponse(
                intent=intent, answer=_render_macro_answer(scenario),
                agent_trace=trace, macro=scenario,
            )

        # Wave 8S — general_research_chat now actually answers when there's
        # prior context to reason from. Pull the most-recently-discussed
        # tickers from `history` + this message, fetch their latest memos,
        # and ask the LLM to answer the user's question grounded in that
        # data. Falls back to the help-text path only when NO usable
        # context exists (cold start with a vague question).
        contextual = self._answer_with_memo_context(
            message, history or [], sdk_attempted=sdk_attempted,
        )
        if contextual is not None:
            return ChatResponse(intent=intent, answer=contextual, agent_trace=trace)

        snapshot = macro_snapshot()
        snapshot_str = ", ".join(f"{k}: {v}" for k, v in snapshot.items())
        ans = (
            "I'm MarketMosaic — a virtual investment committee.\n\n"
            "Try asking me to:\n"
            "- **Analyze NVDA as a long-term investment**\n"
            "- **Compare MSFT and GOOGL from a PM perspective**\n"
            "- **Find 5 high-quality stocks that benefit from falling rates**\n"
            "- **Build a 10-stock portfolio for a soft landing with continued AI infrastructure spend**\n"
            "- **Run a DCF for MSFT using base-case assumptions**\n"
            "- **Show me reasonable valuation growth stocks**\n"
            "- **What sectors benefit if inflation stays sticky?**\n\n"
            f"_Macro snapshot:_ {snapshot_str}\n\n"
            "_MarketMosaic is for research and education only and does not provide personalized financial advice._"
        )
        return ChatResponse(intent=intent, answer=ans, agent_trace=trace)

    def _answer_with_memo_context(
        self, message: str, history: list[ChatMessage], *, sdk_attempted: bool = False,
    ) -> str | None:
        """Wave 8S — answer a free-form follow-up question using the
        memos already produced in this conversation.

        Returns the answer string (markdown, with the disclaimer
        appended) when there's enough context to reason from, or None
        when the conversation has no prior memo to anchor on (so the
        caller falls through to the help text).

        Wave 10: when `CHAT_AGENTS_SDK=true` + the SDK is installed +
        `OPENAI_API_KEY` is set, route through a real `Agent` with
        `function_tool` access to memo / DCF / comps / macro fetchers.
        The agent decides what to fetch. Falls through to the legacy
        single-shot path on any failure or refusal so the chat handler is
        robust. `sdk_attempted` says the turn already ran the SDK (the
        conceptual-follow-up branch): it is not run twice, and the
        fallback does not return to OpenAI (`_legacy_answer_route`).
        """
        if not sdk_attempted:
            sdk_answer, sdk_attempted = _try_sdk_chat(message, history)
            if sdk_answer:
                return sdk_answer
            # else: fall through to legacy single-shot path
        # Pull tickers mentioned anywhere in the recent transcript.
        all_text = "\n".join([m.content or "" for m in history[-8:]] + [message])
        candidate_tickers = _extract_tickers(all_text)
        # Also pick up tickers from a previous comparison answer (e.g.,
        # "MSFT — Bullish" / "GOOGL — Bullish"). _extract_tickers already
        # handles uppercase symbols.

        # Pull the latest snapshot memos for each candidate. memo_store
        # serves cached snapshots cheaply — no re-running of the graph.
        from ..services.memo_store import latest_memo, present_snapshot
        memos: list[dict[str, Any]] = []
        # Tickers without a memo still get a "lite" company snapshot
        # (sector, industry, business_description + screener_metrics) so
        # the LLM can answer comparative follow-ups like "which has the
        # strongest moat?" without us having to pre-run a full memo for
        # every ticker the user mentions.
        company_lites: list[dict[str, Any]] = []
        seen: set[str] = set()
        for t in candidate_tickers:
            if t in seen:
                continue
            seen.add(t)
            snap = latest_memo(t)
            if snap is not None:
                try:
                    m = present_snapshot(snap)
                    memos.append(_memo_for_chat_context(m))
                    if len(memos) >= 4:
                        break
                    continue
                except Exception as exc:
                    # (a) the chat falls back to the lite company snapshot;
                    # a stored memo that no longer validates is worth a
                    # log line because the row is otherwise unreachable.
                    log_safely(
                        log,
                        f"cached memo snapshot for {t} v{getattr(snap, 'version', '?')} "
                        "could not be hydrated for chat context",
                        exc,
                    )
            lite = _company_lite_snapshot(t, with_price=False)
            if lite is not None:
                company_lites.append(lite)
                if len(company_lites) >= 8:
                    break
        # One quote read for every lite (at most 8), not one per ticker.
        _overlay_lite_prices(company_lites)

        if not memos and not company_lites:
            # Nothing to ground in — let the help text fire.
            return None

        # Build the LLM call. System prompt frames it as a careful PM
        # answering a follow-up using the data the platform has on hand
        # (full memos when available, lighter company snapshots when
        # only screener-tier data exists).
        system = (
            "You are MarketMosaic's PM. The user is asking a follow-up "
            "question about tickers from recent conversation context. "
            "Answer directly using the data provided — quote specific "
            "numbers (rating, scores, margins, ROIC, P/E, EV/EBITDA, "
            "DCF upside, key risks) where they appear. For comparative "
            "questions like 'which has the strongest moat?', reason "
            "from durable advantages — gross / operating margins, ROIC, "
            "scale, network effects implied by the business description "
            "— and rank the candidates with one-sentence justifications. "
            "When asked 'which should I buy', give a directional answer "
            "grounded in the metrics, then add ONE sentence on what "
            "would change your view. Do NOT invent data. Always end "
            "with the disclaimer:\n\n"
            "_MarketMosaic is for research and education only and does "
            "not provide personalized financial advice._"
        )
        context_blocks = []
        if memos:
            context_blocks.append(
                "Full memos (preferred — use these first):\n"
                + json.dumps(memos, default=str, indent=2)[:5000]
            )
        if company_lites:
            context_blocks.append(
                "Company snapshots (use when no memo is available):\n"
                + json.dumps(company_lites, default=str, indent=2)[:5000]
            )
        # Wave 10 — pull the PM brain + memory + research_notes into the
        # context. Picks the first memo's ticker / sector for memory
        # routing; PM brain + research_notes are always loaded.
        from .pm_context import build_pm_context
        first_ticker = (memos[0].get("ticker") if memos else
                        (company_lites[0].get("ticker") if company_lites else None))
        first_sector = (memos[0].get("sector") if memos else
                        (company_lites[0].get("sector") if company_lites else None))
        pm_ctx = build_pm_context(
            ticker=first_ticker, sector=first_sector,
            profile={"ticker": first_ticker, "sector": first_sector},
        )
        prompt = (
            ((pm_ctx + "\n\n") if pm_ctx else "")
            + "\n\n".join(context_blocks)
            + "\n\nConversation history (last few turns):\n"
            + "\n".join(f"- {h.role}: {(h.content or '')[:300]}" for h in history[-6:])
            + f"\n\nUser's new question:\n{message}"
        )
        # Use whatever provider is active. The earlier comment here
        # cited an Anthropic SDK `proxies` kwarg bug; that was fixed
        # by pinning anthropic>=0.42 in requirements.txt, so forcing
        # OpenAI now just hard-fails on deployments configured with
        # only an Anthropic key.
        text = llm.chat_text(
            prompt, system=system, route="strong", action="chat.answer",
            **_legacy_answer_route(sdk_attempted),
        )
        if not text or not text.strip():
            return None
        # Belt-and-suspenders: ensure the disclaimer is present.
        body = text.strip()
        if "research and education only" not in body.lower():
            body += (
                "\n\n_MarketMosaic is for research and education only and "
                "does not provide personalized financial advice._"
            )
        return body


def _lite_quotes(tickers: list[str]) -> dict[str, Any]:
    """Quotes for chat company-lites in ONE `get_quotes` call.

    The stored-close fallback is DB-only and labelled, so the chat model can
    say "as of 4:00 PM ET, last close" rather than present an old number as
    current. A quote is an overlay: any failure leaves the lites unpriced
    here and the per-ticker fallback below takes over.
    """
    if not tickers:
        return {}
    try:
        from ..services import quote_service
        return dict(quote_service.get_quotes(tickers))
    except Exception as exc:  # best-effort overlay, like the old per-ticker one
        log_safely(log, "live quote overlay failed for company lites", exc, level=logging.DEBUG)
        return {}


def _overlay_lite_prices(lites: list[dict[str, Any]]) -> None:
    """Set `last_price`, `last_price_as_of` and `last_price_source` on each lite.

    Order: the live/stale quote or stored close from `quote_service`; else
    the recent price series (the old `get_current_price` fallback); else
    the seed-time `companies.last_price`, labelled as such.
    """
    quotes = _lite_quotes([lite["ticker"] for lite in lites])
    for lite in lites:
        quote = quotes.get(lite["ticker"])
        if quote and quote.get("price") is not None:
            lite.update(last_price=quote["price"], last_price_as_of=quote.get("as_of"),
                        last_price_source=quote.get("source"))
            continue
        try:
            from ..services.market_data_service import get_price_series
            rows = [r for r in get_price_series(lite["ticker"], days=5) if r.get("close") is not None]
        except Exception as exc:  # pragma: no cover — best-effort overlay
            log_safely(log, f"price series fallback failed for {lite['ticker']}", exc, level=logging.DEBUG)
            rows = []
        if rows:
            lite.update(last_price=rows[-1]["close"], last_price_as_of=rows[-1].get("date"),
                        last_price_source="eod_close")
        else:
            lite.update(last_price_as_of=None,
                        last_price_source="profile_seed" if lite.get("last_price") is not None else None)


def _company_lite_snapshot(ticker: str, *, with_price: bool = True) -> dict[str, Any] | None:
    """Compact dossier when no memo exists — sector / industry / market
    cap from the `companies` table, plus screener-tier metrics (P/E,
    margins, ROIC, growth) so the chat LLM can answer comparative
    follow-ups (moat, valuation, growth) without us pre-running a memo
    for every screener row.

    `last_price` is the live quote when there is one, labelled with
    `last_price_as_of` / `last_price_source` so the model can say how old
    it is. `with_price=False` skips the quote read for a caller that prices
    a batch of lites in one call (`_overlay_lite_prices`)."""
    from ..database import SessionLocal
    from ..models import Company, ScreenerMetric, ScreenerScore
    with SessionLocal() as db:
        c = db.get(Company, ticker.upper())
        if c is None:
            return None
        m = db.get(ScreenerMetric, ticker.upper())
        s = db.execute(
            select(ScreenerScore).where(
                ScreenerScore.ticker == ticker.upper(),
                ScreenerScore.theme.is_(None),
            )
        ).scalar_one_or_none()
        lite = _lite_row(c, m, s)
    if with_price:
        _overlay_lite_prices([lite])
    return lite


def _lite_row(c: Any, m: Any, s: Any) -> dict[str, Any]:
    """The DB-only part of a company-lite (Company, ScreenerMetric, ScreenerScore)."""
    return {
        "ticker": c.ticker,
        "name": c.company_name,
        "sector": c.sector,
        "industry": c.industry,
        "market_cap": c.market_cap,
        "business": (c.business_description or "")[:600],
        # Seed-time value until `_overlay_lite_prices` replaces it.
        "last_price": c.last_price,
        "last_price_as_of": None,
        "last_price_source": None,
        "metrics": {
            "pe_ttm": getattr(m, "pe_ttm", None),
            "ev_ebitda": getattr(m, "ev_ebitda", None),
            "gross_margin": getattr(m, "gross_margin", None),
            "op_margin": getattr(m, "op_margin", None),
            "fcf_margin": getattr(m, "fcf_margin", None),
            "roic": getattr(m, "roic", None),
            "roe": getattr(m, "roe", None),
            "debt_to_ebitda": getattr(m, "debt_to_ebitda", None),
            "revenue_growth_yoy": getattr(m, "revenue_growth_yoy", None),
            "beta": getattr(m, "beta", None),
        } if m is not None else None,
        "screener_scores": {
            "pm_conviction": s.pm_conviction,
            "quality": s.quality, "growth": s.growth,
            "valuation": s.valuation, "earnings_momentum": s.earnings_momentum,
            "risk": s.risk, "macro_fit": s.macro_fit,
        } if s is not None else None,
    }


def _memo_for_chat_context(m: StockMemoOut) -> dict[str, Any]:
    """Compact memo projection for the free-form chat prompt. Includes
    the dimensions a PM would actually cite when answering 'which is
    the better investment' — rating, stock score, DCF deltas, key
    risks, valuation read.

    Takes a PRESENTED memo (W2a). A hidden section's key is omitted rather
    than sent as a placeholder, so the LLM has nothing to cite: thesis,
    confidence, valuation summary, mispricing card, and the influence of
    each analyst whose view is hidden. List sections arrive filtered."""
    scores = m.scores or {}
    dcf = m.dcf_summary or {}
    out = _memo_context_fields(m, scores, dcf)
    # "not_produced" is an empty field, sent empty as before; what must not
    # reach the model is template text the memo page withholds.
    hidden = {
        k for k in memo_sections.unavailable_keys(m.section_availability or {})
        if m.section_availability[k].reason != "not_produced"
    }
    omit = {
        "thesis": "one_sentence_thesis", "confidence": "confidence_score",
        "valuation_summary": "valuation_agent_view", "mispricing_thesis": "mispricing_thesis",
    }
    for ctx_key, section in omit.items():
        if section in hidden:
            out.pop(ctx_key, None)
    if hidden:
        out["agent_influence"] = {
            k: v for k, v in (m.agent_influence or {}).items()
            if k == "risk" or _influence_section(k) not in hidden
        }
        out["sections_unavailable"] = sorted(hidden)
    return out


def _influence_section(roster_key: str) -> str:
    """The memo section a roster analyst's view lands in."""
    from . import roster
    for spec in roster.AGENTS:
        if spec.key == roster_key:
            return spec.memo_field or f"extra_agent_views.{roster_key}"
    return f"extra_agent_views.{roster_key}"


def _memo_context_fields(m: StockMemoOut, scores: dict[str, Any], dcf: dict[str, Any]) -> dict[str, Any]:
    fields = _memo_context_base(m, scores, dcf)
    debate = _debate_outcome(m)
    if debate:
        # Only when a debate ran and the page shows it: a memo written with
        # DEBATE_MODE off sends exactly the fields it always did.
        fields["debate_outcome"] = debate
    return fields


def _memo_context_base(m: StockMemoOut, scores: dict[str, Any], dcf: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": m.ticker,
        "name": m.company_name,
        "sector": m.sector,
        "rating": m.rating_label,
        "stock_score": scores.get("factor_pm_score"),
        "confidence": int(m.confidence_score),
        "thesis": m.one_sentence_thesis,
        "factor_scores": {
            k.replace("factor_", ""): v for k, v in scores.items()
            if k.startswith("factor_") and k != "factor_pm_score"
        },
        "dcf": {
            "current_price": dcf.get("current_price"),
            "base_implied": dcf.get("base_implied_price"),
            "base_upside": dcf.get("base_upside"),
            "bull_upside": dcf.get("bull_upside"),
            "bear_upside": dcf.get("bear_upside"),
            "wacc": dcf.get("wacc"),
        },
        "valuation_summary": (m.valuation_agent_view.summary or "")[:240],
        "key_risks": [r.title for r in m.key_risks][:5],
        "thesis_breakers": [r.title for r in m.thesis_breakers][:3],
        "bull_case": [p for p in m.bull_case.key_points][:4],
        "bear_case": [p for p in m.bear_case.key_points][:4],
        # Wave 10 — surface the new memo fields to the chat agent so
        # follow-up questions can quote them.
        "mispricing_thesis": (
            m.mispricing_thesis.model_dump() if m.mispricing_thesis else {}
        ),
        "forward_catalysts": [
            {
                "type": c.get("event_type"),
                "date": c.get("event_date"),
                "title": c.get("title"),
                "materiality": c.get("materiality"),
            }
            for c in (m.forward_catalysts or [])[:5]
        ],
        "agent_influence": m.agent_influence or {},
        "macro_regime_at_memo": m.macro_regime_at_memo or "",
        "price_at_memo": m.price_at_memo,
    }
