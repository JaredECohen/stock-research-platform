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

import codecs
import html
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import httpx

from ..config import settings
from .base import ProviderStatus, log_safely
from .sec_text import BoundedHTMLStripper, decoded_bytes

log = logging.getLogger(__name__)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
TIMEOUT = 30.0          # filings can be a few MB; 10s is too tight
DOC_TIMEOUT = 60.0
RATE_LIMIT_SLEEP = 0.12  # ~8 req/sec — under SEC's 10/sec ceiling
MAX_TEXT_BYTES = 250_000  # ~50k tokens; trim huge filings so we don't blow the DB


@dataclass(frozen=True)
class FilingTextResult:
    text: str
    observed_chars: int
    retained_chars: int
    bytes_read: int
    oversized_tokens: int

    @property
    def truncated(self) -> bool:
        return self.observed_chars > self.retained_chars


class _HTMLStripper(HTMLParser):
    """Minimal HTML→text stripper. SEC documents are heavy on inline
    styles + tables; we drop tags and non-content scripts/styles, decode
    entities, and collapse whitespace."""

    _SKIP_TAGS = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
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
    r"^[ \t]*item[ \t]+(\d{1,2}[a-z]?)[\.\s:\-\u2013\u2014]+([^\n\r]{1,120})$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Slice the filing text into Item-keyed sections.

    Returns (sections_dict, risk_factors_bullets). Falls back to empty
    sections when the document doesn't expose Item headers (some 8-Ks).
    """
    matches = list(_ITEM_HEADER.finditer(text))
    if not matches:
        return {}, []
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        item_num = m.group(1).lower()
        item_title = m.group(2).strip().rstrip(".").lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Map a few well-known headings into stable keys the rest of the
        # platform expects (history_service / filing_agent read these).
        if "risk factor" in item_title:
            if len(body) > len(sections.get("risk_factors", "")):
                sections["risk_factors"] = body
        elif "management's discussion" in item_title or item_title.startswith("management"):
            if len(body) > len(sections.get("mda", "")):
                sections["mda"] = body
        elif "business" in item_title and item_num.startswith("1"):
            if len(body) > len(sections.get("business_description", "")):
                sections["business_description"] = body
        elif "legal" in item_title:
            if len(body) > len(sections.get("legal_or_regulatory", "")):
                sections["legal_or_regulatory"] = body
        # Keep an item-keyed view for retrieval to consume. A 10-K's table
        # of contents repeats every heading before the real section, so the
        # first match is usually a page-number stub; the longest body for a
        # given item is the actual section.
        key = f"item_{item_num}"
        if len(body) > len(sections.get(key, "")):
            sections[key] = body

    # Bullet extraction from the risk-factor section. SEC 10-Ks
    # typically structure each risk as one paragraph (caption + body);
    # split on double-newlines and keep paragraphs that look like risk
    # statements (100-1500 chars). Caps to 15 bullets for LLM context.
    risks_text = sections.get("risk_factors", "")
    bullets: list[str] = []
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
        self._ticker_cik_map: dict[str, str] | None = None

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=True,
            healthy=True,
            notes="No API key required; identifies via SEC_USER_AGENT.",
            capabilities=["filings"],
        )

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": accept}

    def lookup_cik(self, ticker: str) -> str | None:
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
                log_safely(log, "SEC ticker map fetch failed", exc)
                return None
        return self._ticker_cik_map.get(ticker)

    def fetch_filing_text(self, url: str) -> str | None:
        """Compatibility view of the bounded, instrumented document fetch."""
        result = self.fetch_filing_document(url)
        return result.text if result is not None else None

    def fetch_filing_document(self, url: str) -> FilingTextResult | None:
        """Stream HTML while retaining the existing normalized-text prefix.

        The former full response, decoded body, fragment list and regex
        copies coexisted before truncation. A 20 MiB offline HTML response
        reproduced a 272 MiB peak increase despite the 250k output cap.
        Count the whole streamed document so every omitted character and
        its source remain visible without holding the whole body in RAM.
        """
        if not url:
            return None
        from ..services.memory_probe import log_rss
        parser = BoundedHTMLStripper(MAX_TEXT_BYTES)
        log_rss("sec_filing_fetch_start", url=url)
        bytes_read = 0
        try:
            headers = {**self._headers("text/html"), "Accept-Encoding": "identity"}
            with httpx.Client(timeout=DOC_TIMEOUT, headers=headers) as client:
                with client.stream("GET", url, follow_redirects=True) as response:
                    if response.status_code != 200:
                        log.warning("SEC doc %s -> %s", url, response.status_code)
                        return None
                    decoder = codecs.getincrementaldecoder(response.encoding or "utf-8")(errors="replace")
                    for fragment in decoded_bytes(
                        response.iter_raw(chunk_size=4096),
                        response.headers.get("content-encoding", ""),
                    ):
                        parser.feed(decoder.decode(fragment))
                    parser.feed(decoder.decode(b"", final=True))
                    bytes_read = response.num_bytes_downloaded
            parser.close()
        except Exception as exc:  # pragma: no cover
            log_safely(log, f"SEC doc fetch failed for {url}", exc)
            return None
        finally:
            log_rss("sec_filing_fetch_end", url=url)
        prefix = parser.output.text()
        observed = parser.output.observed_chars
        truncated = observed > len(prefix)
        result = FilingTextResult(
            text=prefix + ("\n\n…[truncated]" if truncated else ""),
            observed_chars=observed, retained_chars=len(prefix),
            bytes_read=bytes_read, oversized_tokens=parser.oversized_tokens,
        )
        log.info(
            "SEC document %s bytes_read=%d observed_chars=%d retained_chars=%d "
            "truncated_chars=%d oversized_markup_tokens=%d",
            url, bytes_read, observed, len(prefix), max(0, observed - len(prefix)),
            parser.oversized_tokens,
        )
        return result

    def get_filings(
        self, ticker: str, *, cik: str | None = None,
        fetch_text: bool = True,
    ) -> list[dict[str, Any]] | None:
        """Return up to 10 recent filings with full text body.

        Wave 9b — fetches the document body for every form returned
        (latest 10-K, latest 10-Q, and recent 8-Ks). Earlier passes
        skipped 8-K text on the assumption they were rarely thesis-
        relevant; in practice they carry material disclosures
        (M&A, executive changes, dividend / buyback announcements,
        Regulation FD updates) that the news-impact agent cares about.

        SEC limits to ~10 req/sec per User-Agent, so we pace at
        RATE_LIMIT_SLEEP between document fetches. ~10 docs/ticker max.

        Pass `fetch_text=False` to skip the per-document fetches entirely
        and return metadata only. That is how `data_service
        .get_filings_index` reads this method: change detection needs the
        accession numbers and nothing else, and the 30-minute filing poll
        across the curated universe is only affordable at one
        submissions.json read per ticker rather than ten document bodies."""
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
            results: list[dict[str, Any]] = []
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
            log_safely(log, "SEC submissions fetch failed", exc)
            return None

        if not fetch_text:
            return results

        # Fetch document body for every filing in the list. SEC limits
        # to ~10 req/sec per User-Agent; sleep between calls to stay
        # under. With max 10 filings × ~0.12s pacing = ~1.5s overhead
        # per ticker on cold load (cached after that via provider_cache).
        for filing in results:
            time.sleep(RATE_LIMIT_SLEEP)
            document = self.fetch_filing_document(filing["url"])
            if document is None or not document.text:
                continue
            text = document.text
            filing["raw_text"] = text
            filing.update(
                text_truncated=document.truncated,
                text_observed_chars=document.observed_chars,
                text_retained_chars=document.retained_chars,
                text_bytes_read=document.bytes_read,
                html_oversized_tokens=document.oversized_tokens,
            )
            # Section extraction targets 10-K/10-Q Item-N headers; 8-Ks
            # use a different structure (Item 2.02, Item 7.01, etc.)
            # without long bodies, so the section dict will mostly stay
            # empty and the agent reads `raw_text` directly. The Item
            # headers for 8-Ks (e.g., "Item 8.01 Other Events") still
            # land in the generic `item_*` keys for retrieval.
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
    def get_company_profile(self, ticker: str) -> dict[str, Any] | None: return None
    def get_price_history(self, ticker: str, days: int = 252): return None
    def get_financial_statements(self, ticker: str): return None
    def get_ratios(self, ticker: str): return None
    def get_key_metrics(self, ticker: str): return None
    def get_earnings(self, ticker: str): return None
    def get_earnings_transcripts(self, ticker: str): return None
    def get_news(self, ticker: str): return None
    def get_estimates(self, ticker: str): return None
    def get_macro_series(self, series_id: str): return None
    def list_tickers(self) -> list[str]: return []
