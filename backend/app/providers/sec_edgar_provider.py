"""SEC EDGAR provider for filings — metadata + primary-document text.

Wave 9b — fetches the actual document body for the latest 10-K and
10-Q so the Filing Analyst, fact-extraction, and retrieval pipelines
have something real to work with. Older 8-Ks and the long tail of
historical filings stay metadata-only to keep the SEC request budget
in check.

SEC requires a descriptive User-Agent and rate-limits to ~10 req/sec
per identifier. We fetch sequentially with a small sleep between
calls so the universe-wide backfill stays under that ceiling.
"""
from __future__ import annotations

import html
import logging
import re
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
TIMEOUT = 30.0          # filings can be a few MB; 10s is too tight
DOC_TIMEOUT = 60.0
RATE_LIMIT_SLEEP = 0.12  # ~8 req/sec — under SEC's 10/sec ceiling
MAX_TEXT_BYTES = 250_000  # ~50k tokens; trim huge filings so we don't blow the DB


class _HTMLStripper(HTMLParser):
    """Minimal HTML→text stripper. SEC documents are heavy on inline
    styles + tables; we drop tags and non-content scripts/styles, decode
    entities, and collapse whitespace."""

    _SKIP_TAGS = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("p", "br", "tr", "li", "div", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = " ".join(self._chunks)
        # Collapse runs of whitespace + non-breaking spaces; drop empty lines.
        raw = re.sub(r"[  ​]+", " ", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _strip_html(raw: str) -> str:
    parser = _HTMLStripper()
    try:
        parser.feed(raw)
    except Exception:  # pragma: no cover — malformed filings shouldn't kill the fetch
        return html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return parser.text()


# Regex for SEC 10-K / 10-Q item headings ("Item 1A. Risk Factors", etc).
# Multi-line aware; case-insensitive; tolerates the long-S "ITEM" all-caps
# variant and optional trailing punctuation.
_ITEM_HEADER = re.compile(
    r"^[ \t]*item[ \t]+(\d{1,2}[a-z]?)[\.\s:\-]+([^\n\r]{1,120})$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_sections(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Slice the filing text into Item-keyed sections.

    Returns (sections_dict, risk_factors_bullets). Falls back to empty
    sections when the document doesn't expose Item headers (some 8-Ks).
    """
    matches = list(_ITEM_HEADER.finditer(text))
    if not matches:
        return {}, []
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        item_num = m.group(1).lower()
        item_title = m.group(2).strip().rstrip(".").lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Map a few well-known headings into stable keys the rest of the
        # platform expects (history_service / filing_agent read these).
        if "risk factor" in item_title:
            sections["risk_factors"] = body
        elif "management's discussion" in item_title or item_title.startswith("management"):
            sections["mda"] = body
        elif "business" in item_title and item_num.startswith("1"):
            sections["business_description"] = body
        elif "legal" in item_title:
            sections["legal_or_regulatory"] = body
        # Keep an item-keyed view for retrieval to consume.
        sections.setdefault(f"item_{item_num}", body)

    # Bullet extraction from the risk-factor section. SEC 10-Ks
    # typically structure each risk as one paragraph (caption + body);
    # split on double-newlines and keep paragraphs that look like risk
    # statements (100-1500 chars). Caps to 15 bullets for LLM context.
    risks_text = sections.get("risk_factors", "")
    bullets: List[str] = []
    if risks_text:
        # Split on blank lines OR sentence-end-then-cap-letter (reflows a
        # filing whose paragraphs got run together by HTML stripping).
        paragraphs = re.split(r"\n\s*\n|(?<=[.!?])\s{2,}(?=[A-Z])", risks_text)
        for p in paragraphs:
            p = p.strip()
            if 80 <= len(p) <= 1500:
                bullets.append(p)
                if len(bullets) >= 15:
                    break
        # Fall back to first ~12 lines if paragraph splitting fails.
        if not bullets:
            for line in risks_text.split("\n"):
                line = line.strip()
                if 40 <= len(line) <= 600:
                    bullets.append(line)
                    if len(bullets) >= 12:
                        break
    return sections, bullets


class SECEdgarProvider:
    name: str = "sec_edgar"

    def __init__(self) -> None:
        self.user_agent = settings.sec_user_agent
        self._ticker_cik_map: Optional[Dict[str, str]] = None

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="No API key required; identifies via SEC_USER_AGENT.",
            capabilities=["filings"],
        )

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": accept}

    def lookup_cik(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper().replace(".", "-")  # SEC uses BRK-B format
        if self._ticker_cik_map is None:
            try:
                with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                    r = client.get(TICKER_LOOKUP_URL)
                    if r.status_code != 200:
                        log.warning("SEC ticker lookup -> %s", r.status_code)
                        return None
                    data = r.json()
                self._ticker_cik_map = {
                    str(row["ticker"]).upper(): str(row["cik_str"]).zfill(10)
                    for row in data.values()
                }
            except Exception as exc:  # pragma: no cover
                log.warning("SEC ticker map fetch failed: %s", exc)
                return None
        return self._ticker_cik_map.get(ticker)

    def fetch_filing_text(self, url: str) -> Optional[str]:
        """Download a primary filing document and return plain text.

        Caps the result at MAX_TEXT_BYTES so a 5MB 10-K doesn't blow up
        the DB. Returns None on network / parse failure so the caller
        can still persist the filing's metadata."""
        if not url:
            return None
        try:
            with httpx.Client(timeout=DOC_TIMEOUT, headers=self._headers("text/html")) as client:
                r = client.get(url, follow_redirects=True)
                if r.status_code != 200:
                    log.warning("SEC doc %s -> %s", url, r.status_code)
                    return None
                body = r.text
        except Exception as exc:  # pragma: no cover
            log.warning("SEC doc fetch failed for %s: %s", url, exc)
            return None
        text = _strip_html(body)
        if len(text) > MAX_TEXT_BYTES:
            text = text[:MAX_TEXT_BYTES] + "\n\n…[truncated]"
        return text

    def get_filings(
        self, ticker: str, *, cik: Optional[str] = None,
        fetch_text: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return the latest 10-K/10-Q/8-K metadata, with text body for
        the latest 10-K and latest 10-Q (the highest-leverage forms for
        agent context). 8-Ks stay metadata-only — they're short, plentiful,
        and rarely thesis-relevant unless the news agent flags one.

        Pass `fetch_text=False` to skip the slow per-document fetches
        (used by `data_service._lookup_cik` which only needs the
        accession list)."""
        if not cik:
            cik = self.lookup_cik(ticker)
        if not cik:
            return None
        try:
            cik_padded = str(cik).lstrip("0").zfill(10)
            with httpx.Client(timeout=TIMEOUT, headers=self._headers()) as client:
                r = client.get(SUBMISSIONS_URL.format(cik=cik_padded))
                if r.status_code != 200:
                    return None
                data = r.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accs = recent.get("accessionNumber", [])
            primary = recent.get("primaryDocument", [])
            period_ends = recent.get("reportDate", []) or recent.get("primaryDocDescription", [])
            results: List[Dict[str, Any]] = []
            for form, date_, acc, doc, pe in zip(forms, dates, accs, primary, period_ends):
                if form not in ("10-K", "10-Q", "8-K"):
                    continue
                acc_no_hyphen = acc.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{acc_no_hyphen}/{doc}"
                results.append(dict(
                    type=form,
                    period_end=pe if pe and re.match(r"^\d{4}-\d{2}-\d{2}$", str(pe)) else None,
                    filing_date=date_,
                    accession_number=acc,
                    url=url,
                    business_description=None,
                    raw_text="",
                ))
                if len(results) >= 10:
                    break
        except Exception as exc:  # pragma: no cover
            log.warning("SEC submissions fetch failed: %s", exc)
            return None

        if not fetch_text:
            return results

        # Fetch document body for the latest 10-K + latest 10-Q only —
        # those are the high-leverage docs for memo context. SEC limits
        # to ~10 req/sec; sleep between calls to stay under.
        latest_10k = next((f for f in results if f["type"] == "10-K"), None)
        latest_10q = next((f for f in results if f["type"] == "10-Q"), None)
        for filing in (latest_10k, latest_10q):
            if filing is None:
                continue
            time.sleep(RATE_LIMIT_SLEEP)
            text = self.fetch_filing_text(filing["url"])
            if not text:
                continue
            filing["raw_text"] = text
            sections, risks = _extract_sections(text)
            if sections.get("business_description"):
                filing["business_description"] = sections["business_description"][:4000]
            if sections.get("mda"):
                filing["mda"] = sections["mda"][:8000]
            if sections.get("risk_factors"):
                filing["risk_factors"] = risks  # bullet list
            if sections.get("legal_or_regulatory"):
                filing["legal_or_regulatory"] = [sections["legal_or_regulatory"][:2000]]
        return results

    # All other BaseProvider methods return None
    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]: return None
    def get_price_history(self, ticker: str, days: int = 252): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_news(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> List[str]: return []
