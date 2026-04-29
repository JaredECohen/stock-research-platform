"""Tools available to agents.

Tool access is *scoped* per agent role to enforce the evidence-discipline rule:
filings + financials + transcripts are primary sources; news / search / social
are bounded modifiers used only by the news, risk, and macro agents.

Source-weight constants encode the relative trust we put in each source type
when computing memo `evidence_quality`. They are referenced by
`graph.py::_pm_synthesis` and the citation linter.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Set

from ..services import (
    fundamentals_service,
    macro_service,
    news_service,
    retrieval_service,
    transcripts_service,
    valuation_service,
)
from ..services.data_service import get_data_service
from ..services.market_data_service import get_basic_stats


# ---------------------------------------------------------------------------
# Source-weight registry — weights memo evidence quality.
# Higher = more trustworthy as primary investment evidence.
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS: Dict[str, float] = {
    "filing": 1.0,         # 10-K / 10-Q / 8-K — primary, audited
    "financials": 1.0,     # IS / BS / CF rows from a fundamentals provider
    "transcript": 0.95,    # Earnings call prepared remarks + Q&A
    "ratios": 0.9,         # Derived from audited financials
    "estimates": 0.7,      # Sell-side consensus
    "macro": 0.85,         # FRED + central-bank data
    "peer": 0.85,          # Peer fundamentals (transitively from primary)
    "press_release": 0.7,  # Company IR — primary but unaudited
    "regulatory": 0.85,    # FDA / SEC / DOJ / FTC announcements
    "tier1_news": 0.7,     # Bloomberg / Reuters / WSJ / FT
    "news": 0.5,           # Generic news wire
    "sell_side_note": 0.6,
    "blog": 0.3,
    "social": 0.2,         # Twitter / Reddit / StockTwits
    "search_trend": 0.4,   # Google Trends — alt-data, narrow utility
    "short_report": 0.7,   # Treated as high-signal even when wrong
    "dcf": 0.95,           # Derived from primary financials
    "sector_config": 0.6,  # Internal sector framework
    "profile": 0.85,
    "default": 0.5,
}


def source_weight(source_id: str) -> float:
    """Return the trust weight for a source pointer like 'filing:0001-DEMO-10K'."""
    if not source_id:
        return SOURCE_WEIGHTS["default"]
    head = source_id.split(":", 1)[0].lower()
    return SOURCE_WEIGHTS.get(head, SOURCE_WEIGHTS["default"])


def evidence_quality(sources: List[str]) -> float:
    """Average source weight across a list of source pointers (0..1)."""
    if not sources:
        return 0.5
    weights = [source_weight(s) for s in sources]
    return sum(weights) / len(weights)


# ---------------------------------------------------------------------------
# Tool-access scoping
# ---------------------------------------------------------------------------

# Map agent role → set of tool names it is allowed to call.
# Filing / valuation / comps / earnings / sector agents get NO news or social
# tools. The news + risk + macro agents get bounded access.
AGENT_TOOL_SCOPE: Dict[str, Set[str]] = {
    "sector":    {"profile", "fundamentals", "price_stats", "retrieve"},
    "earnings":  {"profile", "fundamentals", "transcript", "retrieve"},
    "filing":    {"profile", "fundamentals", "filings", "retrieve"},
    "valuation": {"profile", "fundamentals", "dcf", "comps"},
    "comps":     {"profile", "fundamentals", "comps"},
    "macro":     {"macro_snapshot", "news_recent"},  # macro news only
    "news":      {"news_recent", "search_trend", "social_sentiment", "profile"},
    "risk":      {"profile", "fundamentals", "filings", "news_recent",
                  "social_sentiment", "short_reports", "retrieve"},
    "portfolio": {"profile", "fundamentals", "dcf", "comps"},
    "screener":  {"profile", "fundamentals", "ratios"},
    "critic":    {"retrieve"},  # only to verify citations against indexed corpus
}


def is_tool_allowed(agent_role: str, tool_name: str) -> bool:
    """Defensive check used by tool wrappers below."""
    allowed = AGENT_TOOL_SCOPE.get(agent_role)
    return bool(allowed and tool_name in allowed)


# ---------------------------------------------------------------------------
# Universal tools (low-cost data lookups; used by most agents)
# ---------------------------------------------------------------------------

def get_company_profile(ticker: str) -> Optional[Dict[str, Any]]:
    return get_data_service().get_company_profile(ticker)


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    return fundamentals_service.get_full_financials(ticker)


def get_price_stats(ticker: str) -> Dict[str, Any]:
    return get_basic_stats(ticker)


def get_dcf(ticker: str) -> Optional[Dict[str, Any]]:
    res = valuation_service.build_dcf(ticker)
    return res.model_dump() if res else None


def get_comps(ticker: str) -> Optional[Dict[str, Any]]:
    res = valuation_service.build_comps(ticker)
    return res.model_dump() if res else None


def get_transcript(ticker: str) -> Optional[Dict[str, Any]]:
    return transcripts_service.latest_transcript(ticker)


def macro_snapshot() -> Dict[str, float]:
    return macro_service.macro_snapshot()


def retrieve(ticker: str, query: str, *, limit: int = 4) -> List[Dict[str, Any]]:
    return retrieval_service.search(ticker, query, limit=limit)


# ---------------------------------------------------------------------------
# Restricted tools — news / social / search trend
# These are used only by the news, macro, and risk agents.
# News is *bounded to events since the last filing*: older news is already
# reflected in the latest 10-Q/10-K and re-citing it adds noise.
# ---------------------------------------------------------------------------

CONSUMER_FACING_SECTORS = {
    "Consumer Discretionary",
    "Consumer Staples",
    "Communication Services",
}


def _latest_filing_date(ticker: str) -> Optional[date]:
    filings = get_data_service().get_filings(ticker) or []
    dates: List[date] = []
    for f in filings:
        d = f.get("filing_date") or f.get("period_end")
        if not d:
            continue
        try:
            dates.append(datetime.fromisoformat(str(d)[:10]).date())
        except Exception:
            continue
    return max(dates) if dates else None


def get_news_recent(ticker: str, *, since_last_filing: bool = True) -> List[Dict[str, Any]]:
    """News bounded to (latest filing date, today). Older news is already in the filing."""
    items = news_service.get_news(ticker) or []
    if not since_last_filing:
        return items
    last_filing = _latest_filing_date(ticker)
    if not last_filing:
        # Fall back to last 60 days
        cutoff = date.today() - timedelta(days=60)
    else:
        cutoff = last_filing
    out: List[Dict[str, Any]] = []
    for n in items:
        pub = n.get("published_at")
        if not pub:
            out.append(n)
            continue
        try:
            d = datetime.fromisoformat(str(pub).replace("Z", "+00:00")).date()
        except Exception:
            out.append(n)
            continue
        if d >= cutoff:
            out.append(n)
    return out


def get_search_trend(ticker: str) -> Optional[Dict[str, Any]]:
    """Google-Trends-style alt-data. Wired only for consumer-facing tickers.

    Returns None for B2B / financials / energy / industrials — search-trend
    signal there is too noisy to be useful.
    """
    profile = get_data_service().get_company_profile(ticker) or {}
    if profile.get("sector") not in CONSUMER_FACING_SECTORS:
        return None
    # Demo mode: deterministic stub. Wire a real Google Trends provider here.
    return dict(
        ticker=ticker,
        provider="demo_stub",
        period="trailing_12_weeks",
        index_value=72.0,
        delta_4w=4.5,
        note="Search-trend signal is enabled only for consumer-facing tickers.",
    )


def get_social_sentiment(ticker: str) -> Dict[str, Any]:
    """Aggregate social into a single contrarian-flag scalar.

    Per the discipline rule, we never quote individual tweets/posts. We return
    a sentiment-extremity score (0..100) plus a contrarian flag. Consumers
    should treat extremes as a *contrarian* signal, not a momentum signal.
    """
    # Demo mode: deterministic stub derived from ticker hash so it's stable.
    h = sum(ord(c) for c in ticker) % 100
    extremity = 50 + ((h - 50) * 0.6)  # range ~20..80
    extremity = max(0.0, min(100.0, extremity))
    contrarian = "neutral"
    if extremity > 80:
        contrarian = "bearish_setup"  # crowd is unanimously bullish
    elif extremity < 20:
        contrarian = "bullish_setup"  # crowd is unanimously bearish
    return dict(
        ticker=ticker,
        sentiment_extremity=round(extremity, 1),
        contrarian_flag=contrarian,
        note="Sentiment scalar only — never quote individual posts. Contrarian read.",
    )


def get_short_reports(ticker: str) -> List[Dict[str, Any]]:
    """Curated short-report feed (Hindenburg, Muddy Waters, Citron, etc.).

    In demo mode, returns an empty list. Wire a real feed here in production.
    """
    return []


# ---------------------------------------------------------------------------
# Citation linter — used by the critic to challenge memos that lean on
# low-quality sources for thesis-bearing claims.
# ---------------------------------------------------------------------------

def lint_citations(sources: List[str], *, min_quality: float = 0.7) -> Dict[str, Any]:
    """Score the trust profile of a memo's sources.

    Returns a dict with overall quality, primary-source ratio, and a flag
    indicating whether the memo's thesis-bearing evidence is too thin.
    """
    if not sources:
        return dict(quality=0.0, primary_ratio=0.0, flag="no_sources")
    weights = [source_weight(s) for s in sources]
    primary = sum(1 for w in weights if w >= 0.85)
    quality = sum(weights) / len(weights)
    ratio = primary / len(weights)
    flag = "ok"
    if quality < min_quality:
        flag = "low_quality"
    elif ratio < 0.4:
        flag = "thin_primary_evidence"
    return dict(quality=round(quality, 3), primary_ratio=round(ratio, 3), flag=flag)
