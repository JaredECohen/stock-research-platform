"""PM Orchestrator — entry point for the chat router.

Classifies intent (LLM if available, regex fallback), dispatches to the
appropriate sub-graph, and synthesizes the final response.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..schemas import (
    AgentTrace,
    ChatMessage,
    ChatResponse,
    DCFResult,
    IntentType,
    MacroScenarioResult,
    ModelPortfolio,
    PortfolioRequest,
    ScreenerRequest,
    ScreenerResult,
    StockMemoOut,
)
from ..services.data_service import get_data_service
from ..services.macro_service import macro_snapshot
from ..services.portfolio_service import build_model_portfolio
from ..services.screener_service import compute_universe_scores
from ..services.valuation_service import build_comps, build_dcf
from . import llm, prompts
from .graph import default_agent_trace, run_stock_memo
from .macro_agent import run_macro_scenario


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


def _extract_tickers(text: str) -> List[str]:
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


def _extract_theme(text: str) -> Optional[str]:
    low = text.lower()
    for k, v in KNOWN_THEMES.items():
        if k in low:
            return v
    return None


def classify_intent(message: str) -> Tuple[IntentType, List[str], Optional[str]]:
    """Classify intent. Tries LLM first, falls back to deterministic rules."""
    llm_out = llm.chat_json(
        prompts.INTENT_CLASSIFIER_PROMPT + "\n\nMessage:\n" + message,
        system=prompts.PM_SYSTEM, route="cheap",
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

def _render_memo_answer(memo: StockMemoOut) -> str:
    bullets = []
    bullets.append(f"**{memo.ticker} — {memo.company_name}** ({memo.sector})")
    bullets.append(f"Rating: **{memo.rating_label}** · confidence {int(memo.confidence_score)}/100")
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


def _render_comparison_answer(memos: List[StockMemoOut]) -> str:
    parts = ["**Cross-comparison from a PM's perspective:**\n"]
    for m in memos:
        parts.append(f"### {m.ticker} — {m.rating_label} (confidence {int(m.confidence_score)})")
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
    lines.append(f"- Base implied price: ${d.base.implied_share_price:,.2f} ({d.base.upside_pct:+.1%})")
    lines.append(f"- Bull implied price: ${d.bull.implied_share_price:,.2f} ({d.bull.upside_pct:+.1%})")
    lines.append(f"- Bear implied price: ${d.bear.implied_share_price:,.2f} ({d.bear.upside_pct:+.1%})")
    lines.append("")
    lines.append(d.summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def chat(self, message: str, history: Optional[List[ChatMessage]] = None) -> ChatResponse:
        intent, tickers, theme = classify_intent(message)
        trace = default_agent_trace(intent)

        if intent == "single_stock_analysis" and tickers:
            ticker = tickers[0]
            # Phase 3: route through the Agents SDK runtime when enabled. The
            # runtime ultimately returns the same StockMemoOut shape, so the
            # downstream rendering / tracing is identical.
            if settings.use_agents_sdk:
                from .sdk_runtime import run_stock_memo_via_sdk
                memo = run_stock_memo_via_sdk(ticker)
            else:
                memo = run_stock_memo(ticker)
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
            for t in tickers[:4]:
                try:
                    memos.append(run_stock_memo(t))
                except Exception:
                    continue
            if not memos:
                return ChatResponse(
                    intent=intent,
                    answer="Could not generate any memos for the requested tickers.",
                    agent_trace=trace,
                )
            return ChatResponse(
                intent=intent, answer=_render_comparison_answer(memos),
                agent_trace=trace, memo=memos[0],
                sources=[s for m in memos for s in m.sources_used][:20],
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

        # general_research_chat fallback — explain capabilities
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
