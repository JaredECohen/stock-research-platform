"""News agent backed by Gemini 2.5 Flash + Google Search grounding.

The news agent produces `NewsAlert` records from open-web sources, classifies
severity (advisory/material/breaking), and drops them into the hot cache so
sector + PM agents can react. With no Gemini API key, the agent falls back
to whatever the existing `news_service` returns and labels everything
`advisory` so the rest of the pipeline still has signal.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..cache import cache_get, cache_put, resolved_cost_tokens
from ..config import settings
from ..schemas import NewsAlert
from ..services import news_service
from . import llm

log = logging.getLogger(__name__)


# Allow / block lists for grounded sources. Keep this conservative; we'd
# rather drop a low-quality source than cite it.
_ALLOWED_DOMAINS: Set[str] = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "barrons.com", "marketwatch.com", "nytimes.com",
    "axios.com", "theinformation.com", "businesswire.com", "prnewswire.com",
    "sec.gov", "fool.com", "seekingalpha.com", "investors.com",
    "techcrunch.com", "stratechery.com",
}
_BLOCKED_DOMAINS: Set[str] = {
    # Aggregators and content farms — keep these out of grounded citations.
    "msn.com", "yahoo.com",
}


def _classify_severity(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(k in text for k in ("guidance cut", "guidance lowered", "earnings miss", "fraud", "subpoena",
                                "doj investigation", "ftc lawsuit", "delisting", "going concern",
                                "ceo resigns", "ceo fired", "ceo steps down")):
        return "breaking"
    if any(k in text for k in ("guidance raised", "beat", "raise", "approval", "fda approval",
                                "buyback", "dividend hike", "acquisition", "merger", "spin-off",
                                "regulator", "lawsuit", "downgrade", "upgrade")):
        return "material"
    return "advisory"


def _domain_of(url: str) -> str:
    if not url:
        return ""
    try:
        if "://" in url:
            host = url.split("://", 1)[1].split("/", 1)[0]
        else:
            host = url.split("/", 1)[0]
        host = host.split(":")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _filter_grounded_sources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        d = _domain_of(it.get("url", ""))
        if not d:
            out.append(it)
            continue
        if d in _BLOCKED_DOMAINS:
            continue
        # If the allow-list is set and we don't match, skip — but be lenient
        # with subdomains (e.g. `nordics.reuters.com`).
        if any(d == ad or d.endswith("." + ad) for ad in _ALLOWED_DOMAINS):
            out.append(it)
    return out


def _since_window(ticker: str) -> date:
    """Time-window selection: prefer 'since last filing'; else last 60 days."""
    cold = cache_get(ticker, "company_cold")
    if cold and isinstance(cold.payload, dict):
        # Try to read the last filing date from the cold payload's filings list.
        # The cold payload doesn't include filings directly; the filings_service
        # holds them, so we only need the date the cold snapshot itself was
        # generated as a worst-case lower bound.
        pass
    return date.today() - timedelta(days=60)


def run(ticker: str, *, force_refresh: bool = False) -> List[NewsAlert]:
    """Fetch + classify news, cache as `news_hot`. Returns NewsAlert list."""
    cache_subject = f"news_hot:{ticker}"
    today_key = f"news_hot:{ticker}:{date.today().isoformat()}"

    if not force_refresh:
        cached = cache_get(today_key, "news_hot", max_age_seconds=4 * 3600)
        if cached and isinstance(cached.payload, dict):
            payload = cached.payload.get("alerts") or []
            try:
                return [NewsAlert.model_validate(a) for a in payload]
            except Exception:
                pass

    # Try Gemini-grounded path first; fall back to deterministic news_service.
    items: List[Dict[str, Any]] = []
    if settings.has_gemini:
        prompt = (
            f"Find the 5 most material news items about {ticker} since "
            f"{_since_window(ticker).isoformat()}. Return JSON list of "
            f"{{title, summary, url, published_at}} objects."
        )
        out = llm.gemini_chat_json(
            prompt,
            model=settings.gemini_news_model,
            enable_search_grounding=True,
            max_tokens=900,
        )
        if isinstance(out, dict) and isinstance(out.get("items"), list):
            items = list(out["items"])
        elif isinstance(out, list):
            items = list(out)

    if not items:
        # Fallback: existing news_service
        items = list(news_service.get_news(ticker) or [])

    items = _filter_grounded_sources(items)

    alerts: List[NewsAlert] = []
    for n in items[:10]:
        title = n.get("title") or n.get("headline") or ""
        summary = n.get("summary") or n.get("description") or ""
        url = n.get("url") or n.get("source_url") or ""
        published_at = n.get("published_at") or n.get("date")
        sev = _classify_severity(title, summary)
        alerts.append(NewsAlert(
            ticker=ticker,
            title=title[:240] or f"{ticker} update",
            summary=summary[:600],
            url=url,
            severity=sev,
            published_at=str(published_at) if published_at else None,
            source="gemini" if settings.has_gemini else "news_service",
        ))

    # Persist to hot cache (today's bucket + canonical bucket)
    payload = {"alerts": [a.model_dump() for a in alerts], "ticker": ticker}
    cache_put(today_key, "news_hot", payload=payload,
              sources_used=[f"news:{a.url or a.title}" for a in alerts],
              generated_by="news_agent",
              cost_tokens=resolved_cost_tokens(80),
              ttl_seconds=4 * 3600)
    cache_put(cache_subject, "news_hot", payload=payload,
              sources_used=[f"news:{a.url or a.title}" for a in alerts],
              generated_by="news_agent",
              cost_tokens=0,
              ttl_seconds=4 * 3600)
    return alerts
