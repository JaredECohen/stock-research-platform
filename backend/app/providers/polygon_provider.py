"""Polygon.io provider — prices and news."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

import httpx

from ..config import settings
from .base import ProviderStatus, log_safely
from .price_history import history_start, normalize_history

log = logging.getLogger(__name__)
BASE = "https://api.polygon.io"
TIMEOUT = 10.0


class PolygonProvider:
    name: str = "polygon"
    price_history_provenance = {
        "provider": "polygon", "endpoint": "/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}",
        "close_basis": "split_adjusted", "adjusted_close_basis": "split_adjusted_not_dividend_adjusted",
        "date_basis": "America/New_York aggregate session date",
    }

    def __init__(self) -> None:
        self.api_key = settings.polygon_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set POLYGON_API_KEY to enable.",
            capabilities=["prices", "quote", "news"],
        )

    def _get(self, path: str, **params: Any) -> Any | None:
        if not self.api_key:
            return None
        try:
            params["apiKey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(f"{BASE}{path}", params=params)
                if r.status_code != 200:
                    log.warning("Polygon request path=%s status=%s", path, r.status_code)
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover
            log_safely(log, f"Polygon fetch failed for {path}", exc)
            return None

    def get_quote(self, ticker: str) -> dict[str, Any] | None:
        """Snapshot endpoint — last trade + previous-day close.

        Free tier limits to 5 calls/min; paid tiers are real-time.
        """
        data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}")
        if not data or "ticker" not in data:
            return None
        snap = data["ticker"]
        last_trade = (snap.get("lastTrade") or {})
        prev_day = (snap.get("prevDay") or {})
        day = (snap.get("day") or {})
        price = last_trade.get("p") or day.get("c")
        prev_close = prev_day.get("c")
        return dict(
            ticker=snap.get("ticker"),
            price=price,
            previous_close=prev_close,
            change=(price - prev_close) if (price is not None and prev_close) else None,
            change_pct=snap.get("todaysChangePerc"),
            day_low=day.get("l"),
            day_high=day.get("h"),
            volume=day.get("v"),
            timestamp=last_trade.get("t"),
        )

    def get_price_history(self, ticker: str, days: int = 252) -> list[dict[str, Any]] | None:
        end = date.today()
        start = history_start(end, days)
        if start is None or not self.api_key:
            return None
        prefix = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        path = f"{prefix}{start.isoformat()}/{end.isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        rows = []
        seen_pages = set()
        # At most one nonempty page per calendar day plus a terminal page.
        # Exceeding this bound fails explicitly; no partial history is returned.
        for _ in range((end - start).days + 2):
            page_key = (path, tuple(sorted((str(k), str(v)) for k, v in params.items())))
            if page_key in seen_pages:
                log.warning("Polygon price history pagination cycle ticker=%s", ticker)
                return None
            seen_pages.add(page_key)
            data = self._get(path, **params)
            if not isinstance(data, dict) or data.get("status") in ("ERROR", "NOT_AUTHORIZED"):
                log.warning("Polygon price history page unavailable ticker=%s page=%d", ticker, len(seen_pages))
                return None
            if data.get("adjusted") is False:
                log.warning("Polygon price history adjustment mismatch ticker=%s", ticker)
                return None
            results = data.get("results", [])
            if not isinstance(results, list):
                log.warning("Polygon price history invalid payload ticker=%s", ticker)
                return None
            for r in results:
                try:
                    if not isinstance(r, dict) or isinstance(r.get("t"), bool):
                        raise ValueError("invalid bar")
                    day = datetime.fromtimestamp(float(r["t"]) / 1000, ZoneInfo("America/New_York")).date().isoformat()
                    rows.append(dict(
                        date=day, open=r.get("o"), high=r.get("h"), low=r.get("l"),
                        close=r.get("c"), adjusted_close=r.get("c"), volume=r.get("v"),
                    ))
                except (KeyError, TypeError, ValueError, OverflowError, OSError):
                    rows.append(None)  # normalized report includes every invalid bar
            next_url = data.get("next_url")
            if not next_url:
                return normalize_history(rows, provider=self.name, ticker=ticker, start=start, end=end, log=log)
            if not isinstance(next_url, str):
                log.warning("Polygon price history invalid next page ticker=%s", ticker)
                return None
            try:
                parsed = urlsplit(next_url)
                valid = (
                    parsed.scheme == "https" and parsed.hostname in {"api.polygon.io", "api.massive.com"}
                    and parsed.port in (None, 443) and not parsed.username and not parsed.password
                    and parsed.path.startswith(prefix) and not parsed.fragment
                )
            except ValueError:
                valid = False
            if not valid:
                log.warning("Polygon price history rejected next page ticker=%s", ticker)
                return None
            # Reuse the configured provider host; never forward credentials to
            # an arbitrary next_url. Query secrets from the response are ignored.
            path = parsed.path
            params = {k: v for k, v in parse_qsl(parsed.query) if k.lower() != "apikey"}
            params.update(adjusted="true", sort="asc", limit=50000)
        log.warning("Polygon price history pagination limit ticker=%s pages=%d", ticker, len(seen_pages))
        return None

    def get_news(self, ticker: str) -> list[dict[str, Any]] | None:
        data = self._get("/v2/reference/news", ticker=ticker.upper(), limit=20)
        if not data or "results" not in data:
            return None
        return [
            dict(
                title=n.get("title"),
                source=(n.get("publisher") or {}).get("name"),
                published_at=n.get("published_utc"),
                url=n.get("article_url"),
                summary=n.get("description"),
                tickers=n.get("tickers", [ticker]),
                sentiment="neutral",
                relevance_score=0.6,
            )
            for n in data["results"]
        ]

    # Stubs
    def get_company_profile(self, ticker: str): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_filings(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> list[str]: return []
