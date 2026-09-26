"""News agent backed by a Search-grounded Gemini model (`GEMINI_NEWS_MODEL`).

The news agent produces `NewsAlert` records from open-web sources, classifies
severity (advisory/material/breaking), and drops them into the hot cache so
sector + PM agents can react. With no Gemini API key, the agent falls back
to whatever the existing `news_service` returns and labels everything
`advisory` so the rest of the pipeline still has signal.

Gemini items are model-written JSON, not publisher text: the title, url and
date are whatever the model typed after reading its search results. So the
Gemini branch is governed more strictly than the provider feed (N34, news
trace 2026-09-25): an item with no URL is dropped because the domain
allow-list cannot judge it, the allow-list runs BEFORE the provider-fallback
decision so a relevant-but-blocked answer falls back instead of producing
zero alerts, and a date the parser cannot read is kept for display but
flagged `date_unknown` so the patch path never treats it as fresh.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..cache import cache_get, cache_put, resolved_cost_tokens
from ..config import settings
from ..schemas import NewsAlert
from ..services import news_service
from . import llm

log = logging.getLogger(__name__)


# Wave 6C: domain governance moved to `app/data/news_domains.json` so
# editorial calls about which sources to cite live in a reviewable JSON
# file, not Python constants. The file is the source of truth — edit +
# commit; the cached read below auto-picks up changes on next process boot.

_NEWS_DOMAINS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "news_domains.json"
)


def _load_domain_lists() -> tuple[set[str], set[str]]:
    """Read the governance file. Returns `(allowed, blocked)` sets, lower-cased.

    Falls back to empty sets if the file is missing or malformed — the
    filter then applies the conservative "skip-when-no-allow-list" rule
    in `_filter_grounded_sources` (every grounded source dropped),
    which is the safe behavior.
    """
    try:
        with open(_NEWS_DOMAINS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning("news_domains.json unreadable (%s) — defaulting to empty lists", exc)
        return set(), set()
    allowed = {str(d).strip().lower() for d in (data.get("allowed") or []) if d}
    blocked = {str(d).strip().lower() for d in (data.get("blocked") or []) if d}
    return allowed, blocked


@lru_cache(maxsize=1)
def _domain_lists_cached() -> tuple[set[str], set[str]]:
    return _load_domain_lists()


def reload_domain_lists() -> tuple[set[str], set[str]]:
    """Force-reload the governance file. Useful for tests + admin tooling
    after the JSON has been edited live."""
    _domain_lists_cached.cache_clear()
    return _domain_lists_cached()


# Public for tests + admin use.
def allowed_domains() -> set[str]:
    return _domain_lists_cached()[0]


def blocked_domains() -> set[str]:
    return _domain_lists_cached()[1]


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


_NAME_SUFFIX_RE = re.compile(
    r"\b(inc|inc\.|incorporated|corp|corp\.|corporation|company|co\.|"
    r"ltd|ltd\.|limited|plc|holdings|group|the)\b\.?",
    re.IGNORECASE,
)


def _company_name(ticker: str) -> str:
    """Look up the company display name. Lazy import — `data_service` pulls
    in the provider chain, which we don't want loaded at module import time."""
    try:
        from ..services.data_service import get_data_service
        profile = get_data_service().get_company_profile(ticker) or {}
        return str(profile.get("company_name") or "").strip()
    except Exception:
        return ""


# Words too common in company names to identify one on their own. Matched
# alone they let sector roundups through: BK's name tokens used to be
# ["bank", "york", "mellon"], so "Big bank stocks rally" and "New York Fed
# survey ..." counted as BK news while "BNY beats estimates" did not.
_GENERIC_NAME_TOKENS = frozenset({
    "america", "american", "bank", "bancorp", "brands", "capital",
    "communications", "energy", "financial", "first", "general", "global",
    "industries", "insurance", "international", "investment", "investments",
    "national", "partners", "platforms", "products", "resources",
    "services", "solutions", "systems",
    "technologies", "technology", "trust", "united", "york",
})

_NEWS_ALIASES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "news_aliases.json"
)


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, tuple[str, ...]]:
    """ticker -> press names from `app/data/news_aliases.json` (reviewable,
    like the domain lists). A missing or malformed file means no aliases:
    the filter then matches on ticker and legal name only, as it did."""
    try:
        with open(_NEWS_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning("news_aliases.json unreadable (%s) — no press aliases", exc)
        return {}
    raw = data.get("aliases") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        str(t).strip().upper(): tuple(str(a) for a in names if str(a).strip())
        for t, names in raw.items() if isinstance(names, list)
    }


def _normalized_text(value: str) -> str:
    """Lowercase words separated by single spaces, padded, so a URL slug
    (`/bny-q3-results`) and a title compare the same way and a whole-word
    test is a plain substring test on `" term "`."""
    return " " + " ".join(re.split(r"[^a-z0-9&]+", (value or "").lower())).strip() + " "


def _mentions(text: str, term: str) -> bool:
    """Whole-word (or whole-phrase) mention of `term` in normalised `text`.

    Substring matching was the bug: "bank" matched "bankruptcy" and
    "morgan" matched "jpmorgan"."""
    t = _normalized_text(term).strip()
    return bool(t) and f" {t} " in text


def _ticker_aliases(ticker: str) -> list[str]:
    """Other names that identify `ticker` in a headline: curated press names
    plus verified same-security ticker renames (BK -> BNY)."""
    from ..services.ticker_symbols import market_data_symbols
    out = list(_alias_map().get(ticker.upper(), ()))
    for sym in market_data_symbols(ticker):
        # Short symbols ("BK", "C") are ordinary words and initials in
        # headlines; only 3+ characters identify a company on their own.
        if sym.upper() != ticker.upper() and len(sym) >= 3:
            out.append(sym)
    return out


def _name_tokens(name: str) -> list[str]:
    """Terms from a company name that identify it in a headline.

    Corporate suffixes are stripped. Distinctive words of 4+ characters
    count on their own; generic ones (`_GENERIC_NAME_TOKENS`) do not. The
    run-together form of a multi-word name counts too ("Exxon Mobil" ->
    "exxonmobil", as the press writes it). A name with no distinctive word
    ("Bank of America", "American International Group", "BNY") falls back
    to the whole name as one phrase."""
    cleaned = _NAME_SUFFIX_RE.sub(" ", name or "")
    words = [t for t in re.split(r"[^a-z0-9&]+", cleaned.lower()) if t]
    distinctive = [t for t in words if len(t) >= 4 and t not in _GENERIC_NAME_TOKENS]
    if not distinctive:
        phrase = " ".join(words)
        return [phrase] if len(phrase) >= 3 else []
    if len(words) > 1:
        distinctive.append("".join(words))
    return distinctive


def _is_about_company(item: dict[str, Any], ticker: str, name_tokens: list[str]) -> bool:
    """Drop grounded items that don't mention the ticker, a press alias or
    a distinctive term of the company name. Gemini's `google_search` tool
    sometimes returns sector roundups or peer-comparison articles where the
    target ticker is barely a footnote — those are noise for a per-ticker
    alert.

    We only check the title and the URL slug. Summary-text mentions are too
    permissive: dividend-list articles like "Cardinal Health Among 9 Companies
    …" cite Apple in the body but aren't *about* Apple. Headlines and URL
    slugs reflect the article's primary subject. Every term is matched as a
    whole word (`_mentions`)."""
    title = str(item.get("title") or item.get("headline") or "")
    url = str(item.get("url") or item.get("source_url") or "")
    text = _normalized_text(f"{title} {url}")
    if len(ticker) >= 3 and _mentions(text, ticker):
        return True
    if any(_mentions(text, alias) for alias in _ticker_aliases(ticker)):
        return True
    return any(_mentions(text, tok) for tok in name_tokens)


def _filter_grounded_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = allowed_domains()
    blocked = blocked_domains()
    out: list[dict[str, Any]] = []
    for it in items:
        # Same `url or source_url` fallback the alert is built with: reading
        # only `url` let an item whose link sat under `source_url` pass as
        # URL-less, past the block and allow lists alike.
        d = _domain_of(str(it.get("url") or it.get("source_url") or ""))
        if not d:
            out.append(it)
            continue
        if d in blocked:
            continue
        # If the allow-list is set and we don't match, skip — but be lenient
        # with subdomains (e.g. `nordics.reuters.com`).
        if any(d == ad or d.endswith("." + ad) for ad in allowed):
            out.append(it)
    return out


# How far back the grounded Gemini prompt asks for news (assumption A10 of
# the 2026-09-25 plan). It was 60 days: with "the 5 most material items" that
# turned news_hot into a rolling greatest-hits list, and every 2-hourly pass
# re-surfaced weeks-old "material" stories into the patch path.
GEMINI_WINDOW_DAYS = 7

# A published date further in the future than this is not a date anyone
# published; it is a model typo or a timezone-free guess. One day of slack
# covers provider timestamps written in a zone ahead of UTC.
_FUTURE_SLACK = timedelta(days=1)

# Provider formats seen in the feeds, plus what a model plausibly types.
# Alpha Vantage's `time_published` is the compact `20260921T214039`.
_DATETIME_FORMATS = ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M")
_DATE_FORMATS = ("%Y%m%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y")


def _parse_published(value: Any, now: datetime | None = None) -> tuple[datetime, bool] | None:
    """(naive-UTC instant, has_time) for a published-date value, or None.

    Aware values are converted to UTC; naive ones are taken as UTC (the
    persisted convention). A date with no time keeps `has_time=False` so
    callers can tell "that day" from "midnight that day". `now` (default:
    the wall clock) is what "in the future" is measured against.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt, has_time = value, True
    elif isinstance(value, date):
        dt, has_time = datetime.combine(value, time.min), False
    else:
        s = str(value).strip()
        if not s:
            return None
        parsed: datetime | None = None
        has_time = True
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            has_time = len(s) > 10
        except ValueError:
            parsed = None
        if parsed is None:
            for fmt in _DATETIME_FORMATS:
                try:
                    parsed = datetime.strptime(s, fmt).replace(tzinfo=UTC)
                    break
                except ValueError:
                    continue
        if parsed is None:
            for fmt in _DATE_FORMATS:
                try:
                    parsed = datetime.strptime(s, fmt).replace(tzinfo=UTC)
                    has_time = False
                    break
                except ValueError:
                    continue
        if parsed is None:
            try:
                parsed = parsedate_to_datetime(s)  # RFC 2822, as RSS feeds write it
            except (TypeError, ValueError, IndexError):
                parsed = None
        if parsed is None:
            return None
        dt = parsed
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    if dt - (now or datetime.utcnow()) > _FUTURE_SLACK:
        return None
    return dt, has_time


def parse_published_at(value: Any, *, now: datetime | None = None) -> datetime | None:
    """The LATEST naive-UTC instant a published-date value can mean, or None.

    A date-only value maps to the end of that day, so an age gate never
    rejects a same-day story because its time was not given. None means the
    value cannot be read as a date (or lies in the future): callers decide
    what an undated item is worth, and for model-written items the answer
    is "not fresh".
    """
    parsed = _parse_published(value, now)
    if parsed is None:
        return None
    dt, has_time = parsed
    return dt if has_time else datetime.combine(dt.date(), time.max)


def _normalized_published_at(value: Any) -> str | None:
    """ISO form of a parseable published date (`...Z` with a time, a bare
    date without), else the original text, else None. Normalising lets every
    reader (`industry_snapshot`, the NewsAlert panel) parse what the feeds
    wrote in their own formats; an unreadable value is kept verbatim so the
    user still sees what the source said."""
    parsed = _parse_published(value)
    if parsed is None:
        return str(value) if value not in (None, "") else None
    dt, has_time = parsed
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if has_time else dt.date().isoformat()


def _since_window(ticker: str) -> date:
    """Start of the window the Gemini prompt asks about (`GEMINI_WINDOW_DAYS`)."""
    return date.today() - timedelta(days=GEMINI_WINDOW_DAYS)


def _gemini_skip_reason() -> str | None:
    """Why a grounded Gemini call made now would be skipped without a
    response, or None. Read BEFORE the call, so a failure the call itself
    causes (the third one trips the breaker) is not counted as a skip.

    Both short-circuits return None exactly like an empty answer, so
    without this the news_loop note cannot tell "Gemini was never asked"
    from "Gemini had nothing" (news critique: the shared breaker silently
    turns Gemini news off for 120 s). The grounding cap is the llm core's
    grounded-call daily cap (GEMINI_GROUNDED_MAX_PER_DAY).
    """
    try:
        breaker = getattr(llm, "breaker_open", None) or llm._breaker_open
        if breaker("gemini"):
            return "breaker_open"
    except Exception:  # pragma: no cover — diagnostics must not break news
        pass
    # The llm core (M1) names it `_grounding_cap_reached`; probe the public
    # spelling first, as the breaker probe does, so a later rename to a
    # public name keeps working.
    cap = getattr(llm, "grounding_cap_reached", None) or getattr(llm, "_grounding_cap_reached", None)
    if callable(cap):
        try:
            if cap():
                return "grounding_cap"
        except Exception:  # pragma: no cover
            pass
    return None


def _gemini_items(
    ticker: str, name: str, name_tokens: list[str], report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ask the grounded Gemini model for news; return only the items that
    pass relevance, carry a URL and pass the domain allow-list.

    An empty return means "use the provider feed". `report` records a
    short-circuited call for the news_loop note.
    """
    company_clause = f"{name} ({ticker})" if name else ticker
    prompt = (
        f"Find the 5 most material news items PRIMARILY about {company_clause} "
        f"since {_since_window(ticker).isoformat()}. Each item must be a story "
        f"whose main subject is {company_clause} — exclude sector roundups, "
        f"peer-comparison articles, dividend lists, or pieces where {ticker} "
        f"is only mentioned in passing. Keep each summary under 200 characters. "
        f"Give published_at as an ISO 8601 date. "
        f"Return a JSON object "
        f'{{"items": [{{title, summary, url, published_at}}, ...]}}.'
    )
    skip = _gemini_skip_reason()
    # Grounded responses include citations + thinking-token overhead, so
    # the budget needs to comfortably fit 5 items + grounding metadata.
    # 900 tokens truncates mid-JSON on 2.5-flash with `google_search`.
    out = llm.gemini_chat_json(
        prompt,
        model=settings.gemini_news_model,
        enable_search_grounding=True,
        max_tokens=2500,
    )
    if out is None and skip:
        report["gemini_skipped"] = skip
    items: list[Any] = []
    if isinstance(out, dict) and isinstance(out.get("items"), list):
        items = list(out["items"])
    elif isinstance(out, list):
        items = list(out)
    items = [it for it in items if isinstance(it, dict)]
    if not items:
        return []
    relevant = [it for it in items if _is_about_company(it, ticker, name_tokens)]
    if not relevant:
        log.info(
            "news_agent: all %d Gemini items for %s failed relevance filter; "
            "falling back to news_service", len(items), ticker,
        )
        return []
    # `_filter_grounded_sources` passes URL-less items through (a provider
    # row without a link is still publisher text). A model-written item
    # without a URL has no source anyone can check and would bypass the
    # allow-list entirely, so the Gemini branch drops it first.
    with_url = [
        it for it in relevant if _domain_of(str(it.get("url") or it.get("source_url") or ""))
    ]
    # The allow-list runs BEFORE the fallback decision: relevant items that
    # are all unsourced or off-list fall back to the provider feed instead
    # of leaving the ticker with zero alerts.
    governed = _filter_grounded_sources(with_url)
    if not governed:
        log.info(
            "news_agent: %d relevant Gemini items for %s had no URL or a disallowed "
            "domain; falling back to news_service", len(relevant), ticker,
        )
    return governed


def run(
    ticker: str, *, force_refresh: bool = False, report: dict[str, Any] | None = None,
) -> list[NewsAlert]:
    """Fetch + classify news, cache as `news_hot`. Returns NewsAlert list.

    `report`, when given, is filled with where the alerts came from
    (`origin`: gemini / provider / empty / cache) and, when the Gemini call
    was short-circuited, why (`gemini_skipped`: breaker_open /
    grounding_cap). The news_loop note counts these; the return value is
    unchanged, so every other caller is unaffected.
    """
    report = report if report is not None else {}
    cache_subject = f"news_hot:{ticker}"
    today_key = f"news_hot:{ticker}:{date.today().isoformat()}"

    if not force_refresh:
        cached = cache_get(today_key, "news_hot", max_age_seconds=4 * 3600)
        if cached and isinstance(cached.payload, dict):
            payload = cached.payload.get("alerts") or []
            try:
                cached_alerts = [NewsAlert.model_validate(a) for a in payload]
            except Exception:
                cached_alerts = None
            if cached_alerts is not None:
                report["origin"] = "cache"
                return cached_alerts

    # Try Gemini-grounded path first; fall back to deterministic news_service.
    items: list[dict[str, Any]] = []
    name = _company_name(ticker)
    name_tokens = _name_tokens(name)
    used_gemini = False
    if settings.has_gemini:
        items = _gemini_items(ticker, name, name_tokens, report)
        used_gemini = bool(items)

    if not items:
        # Fallback: existing news_service. Re-apply the relevance filter — the
        # underlying providers (Alpha Vantage NEWS_SENTIMENT, etc.) often
        # return sector or peer items keyed off the requested ticker.
        raw = list(news_service.get_news(ticker) or [])
        items = [it for it in raw if _is_about_company(it, ticker, name_tokens)] if name_tokens else raw
        items = _filter_grounded_sources(items)

    alerts: list[NewsAlert] = []
    undated: list[bool] = []
    for n in items[:10]:
        title = n.get("title") or n.get("headline") or ""
        summary = n.get("summary") or n.get("description") or ""
        url = n.get("url") or n.get("source_url") or ""
        raw_published = n.get("published_at") or n.get("date")
        sev = _classify_severity(title, summary)
        alerts.append(NewsAlert(
            ticker=ticker,
            title=title[:240] or f"{ticker} update",
            summary=summary[:600],
            url=url,
            severity=sev,
            published_at=_normalized_published_at(raw_published),
            source="gemini" if used_gemini else "news_service",
        ))
        # L8: a model-written date the parser cannot read stays visible (it
        # is what the model claimed) but is flagged, so no reader takes it
        # for a fresh story. Provider rows keep their existing leniency.
        undated.append(used_gemini and parse_published_at(raw_published) is None)

    report["origin"] = ("gemini" if used_gemini else "provider") if alerts else "empty"

    # Persist to hot cache (today's bucket + canonical bucket). `date_unknown`
    # is written only where true, so dated payloads keep their exact shape.
    alert_dicts = [
        {**a.model_dump(), "date_unknown": True} if flag else a.model_dump()
        for a, flag in zip(alerts, undated)
    ]
    payload = {"alerts": alert_dicts, "ticker": ticker}
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
