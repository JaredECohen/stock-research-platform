"""Offline SEC EDGAR provider tests against trimmed synthetic fixtures.

Fixtures live in `fixtures/sec_edgar/` — a CIK lookup map, a submissions
index with a mix of forms, and four trimmed filing bodies (a 10-K with a
duplicated table of contents, one missing Item 7, one with nbsp /
casing / whitespace-mangled headers, and an 8-K). Every HTTP call goes
through an `httpx.MockTransport`, so real `httpx` request / response
objects flow through the provider but no socket is ever opened.

The section-parser tests pin what `_extract_sections` actually does
today. Two gaps found while writing these tests (TOC stubs winning over
real sections; en-dash separators) were fixed in the provider rather than
papered over here, and are asserted below.
"""
from __future__ import annotations

import pathlib
import socket
from typing import Any

import httpx
import pytest

from app.providers import sec_edgar_provider as sec
from app.providers.sec_edgar_provider import (
    SECEdgarProvider,
    _extract_sections,
    _strip_html,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "sec_edgar"
CIK = "0000123456"
SUBMISSIONS = sec.SUBMISSIONS_URL.format(cik=CIK)
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/123456"
MAPPED_KEYS = {"business_description", "risk_factors", "legal_or_regulatory", "mda"}


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


class _Route:
    def __init__(self, body: str, status: int, exc: Exception | None) -> None:
        self.body, self.status, self.exc = body, status, exc


class _Router:
    """URL → canned response table used as the MockTransport handler."""

    def __init__(self) -> None:
        self.routes: dict[str, _Route] = {}
        self.calls: list[httpx.Request] = []

    def add(self, url: str, body: str = "", *, status: int = 200,
            exc: Exception | None = None) -> None:
        self.routes[url] = _Route(body, status, exc)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        route = self.routes.get(str(request.url))
        if route is None:
            return httpx.Response(404, text="not routed by test")
        if route.exc is not None:
            raise route.exc
        return httpx.Response(
            route.status, stream=httpx.ByteStream(route.body.encode()),
            headers={"content-type": "text/html; charset=utf-8"},
        )


class _FakeHttpx:
    """Stands in for the `httpx` module attribute inside the provider so
    every `httpx.Client(...)` it opens is backed by the router."""

    def __init__(self, router: _Router) -> None:
        self._router = router

    def Client(self, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._router), **kwargs)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline fixture test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    # The inter-document pacing sleep is wall-clock, not behaviour.
    monkeypatch.setattr(sec.time, "sleep", lambda *_: None)


@pytest.fixture
def router(monkeypatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(sec, "httpx", _FakeHttpx(r))
    return r


def _sections(name: str):
    return _extract_sections(_strip_html(_load(name)))


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_drops_non_content_and_normalises_whitespace():
    text = _strip_html(_load("10k_trimmed.html"))
    assert "should never appear" not in text          # <script> body dropped
    assert "padding" not in text                       # <style> body dropped
    assert "<" not in text and ">" not in text
    assert "\xa0" not in text                          # nbsp collapsed
    assert "June 30, 2025" in text
    assert "MD&A" in text                              # entities decoded
    assert "\n\n\n" not in text


def test_strip_html_on_empty_inputs():
    assert _strip_html("") == ""
    assert _strip_html("<html><body></body></html>") == ""


# ---------------------------------------------------------------------------
# _extract_sections
# ---------------------------------------------------------------------------

def test_extract_sections_maps_the_known_items():
    sections, _ = _sections("10k_trimmed.html")
    assert MAPPED_KEYS <= set(sections)
    assert "TOC-SENTINEL-BUSINESS" in sections["business_description"]
    assert "TOC-SENTINEL-RISK" in sections["risk_factors"]
    assert "TOC-SENTINEL-LEGAL" in sections["legal_or_regulatory"]
    assert "TOC-SENTINEL-MDA" in sections["mda"]
    # Each body stops at the next header rather than running to EOF.
    assert "page F-1" not in sections["mda"]
    assert "TOC-SENTINEL-MDA" not in sections["risk_factors"]


def test_toc_duplicates_do_not_win_for_mapped_keys():
    text = _strip_html(_load("10k_trimmed.html"))
    # Precondition: the fixture really does carry every header twice.
    assert text.count("Risk Factors") >= 2
    assert text.lower().count("management's discussion") >= 2
    sections, bullets = _extract_sections(text)
    for key in MAPPED_KEYS:
        assert len(sections[key]) > 100, key   # not a page-number stub
    assert bullets and all(b in sections["risk_factors"] for b in bullets)


def test_item_keys_prefer_the_real_section_over_the_toc_stub():
    sections, _ = _sections("10k_trimmed.html")
    assert "TOC-SENTINEL-RISK" in sections["item_1a"]
    assert "TOC-SENTINEL-MDA" in sections["item_7"]


def test_risk_factor_bullets_are_real_paragraphs():
    sections, bullets = _sections("10k_trimmed.html")
    assert 1 <= len(bullets) <= 15
    assert all(b in sections["risk_factors"] for b in bullets)
    assert all(len(b) >= 40 for b in bullets)
    assert "Short line." not in bullets
    assert any("41% of consolidated revenue" in b for b in bullets)


def test_missing_item7_is_reported_not_fabricated():
    sections, bullets = _sections("10k_missing_item7.html")
    assert "mda" not in sections
    assert "item_7" not in sections
    assert "TOC-SENTINEL-RISK" in sections["risk_factors"]
    assert bullets


def test_mangled_headers_still_map():
    """nbsp inside the header, ALL-CAPS / lower-case titles, runs of
    whitespace and a curly apostrophe in "Management's" all survive."""
    sections, bullets = _sections("10k_mangled_headers.html")
    assert MAPPED_KEYS <= set(sections)
    assert "TOC-SENTINEL-BUSINESS" in sections["business_description"]
    assert "TOC-SENTINEL-RISK" in sections["risk_factors"]
    assert "TOC-SENTINEL-LEGAL" in sections["legal_or_regulatory"]
    assert "TOC-SENTINEL-MDA" in sections["mda"]
    assert len(bullets) == 2


def test_extract_sections_needs_strip_html_to_normalise_nbsp():
    """The header regex only accepts ASCII whitespace; nbsp-separated
    headers are recognised solely because `_strip_html` collapses them.
    Callers feeding raw text must route through `_strip_html` first."""
    raw = (
        "Item\xa01A.\xa0Risk Factors\n"
        "Our results depend heavily on a small number of customers whose "
        "capital budgets move with the freight cycle.\n"
        "Item\xa02. Properties\nWe lease our headquarters.\n"
    )
    assert _extract_sections(raw) == ({}, [])
    sections, _ = _extract_sections(_strip_html(raw))
    assert "risk_factors" in sections


def test_en_dash_separator_maps_mda():
    text = (
        "Item 7 – Management’s Discussion and Analysis\n"
        "Revenue increased 6.9% in fiscal 2025.\n"
        "Item 8. Financial Statements\nSee page F-1.\n"
    )
    sections, _ = _extract_sections(text)
    assert "mda" in sections


def test_8k_yields_item_keys_but_no_mapped_sections():
    sections, bullets = _sections("8k_trimmed.html")
    assert {"item_2", "item_9"} <= set(sections)
    assert not (MAPPED_KEYS & set(sections))
    assert "TOC-SENTINEL-8K" in sections["item_2"]
    assert bullets == []


def test_extract_sections_on_empty_document():
    assert _extract_sections("") == ({}, [])
    assert _extract_sections("No item headers anywhere in this text.") == ({}, [])


# ---------------------------------------------------------------------------
# lookup_cik
# ---------------------------------------------------------------------------

def test_lookup_cik_normalises_tickers_and_caches_the_map(router):
    router.add(sec.TICKER_LOOKUP_URL, _load("company_tickers.json"))
    p = SECEdgarProvider()
    assert p.lookup_cik("acme") == "0000123456"
    assert p.lookup_cik("BRK.B") == "0000007890"      # dot → dash, zero-padded
    assert p.lookup_cik("ZETA") == "0000055555"       # map keys upper-cased
    assert p.lookup_cik("NOPE") is None
    assert len(router.calls) == 1                     # one fetch, then cached
    assert router.calls[0].headers["User-Agent"] == p.user_agent


def test_lookup_cik_non_200_returns_none_and_leaves_cache_cold(router):
    router.add(sec.TICKER_LOOKUP_URL, "forbidden", status=403)
    p = SECEdgarProvider()
    assert p.lookup_cik("ACME") is None
    router.add(sec.TICKER_LOOKUP_URL, _load("company_tickers.json"))
    assert p.lookup_cik("ACME") == "0000123456"       # retried, not poisoned
    assert len(router.calls) == 2


@pytest.mark.parametrize("body, exc", [
    ("<html>maintenance</html>", None),
    ("[]", None),                                     # list, not the keyed map
    ("", httpx.ReadTimeout("timed out")),
])
def test_lookup_cik_bad_responses_return_none(router, body, exc):
    router.add(sec.TICKER_LOOKUP_URL, body, exc=exc)
    assert SECEdgarProvider().lookup_cik("ACME") is None


# ---------------------------------------------------------------------------
# get_filings
# ---------------------------------------------------------------------------

def test_get_filings_metadata_only(router):
    router.add(SUBMISSIONS, _load("submissions_CIK0000123456.json"))
    rows = SECEdgarProvider().get_filings("ACME", cik="123456", fetch_text=False)
    assert rows is not None
    # Only 10-K / 10-Q / 8-K survive; S-8 and Form 4 are dropped, order kept.
    assert [r["type"] for r in rows] == ["10-K", "8-K", "10-Q", "10-Q"]
    ten_k = rows[0]
    assert ten_k["accession_number"] == "0000123456-25-000031"
    assert ten_k["filing_date"] == "2025-08-28"
    assert ten_k["period_end"] == "2025-06-30"
    assert ten_k["url"] == f"{ARCHIVE}/000012345625000031/acme-20250630.htm"
    assert ten_k["raw_text"] == "" and ten_k["business_description"] is None
    assert rows[3]["period_end"] is None              # "20241231" fails the ISO check
    # Explicit CIK skips the ticker lookup; the index is fetched once.
    assert [str(c.url) for c in router.calls] == [SUBMISSIONS]


def test_index_mode_includes_periodic_extras_and_body_window_is_unchanged(router):
    """FIX-005: the metadata read also lists what the issuer reported that
    the body set cannot show (10-Q/A, 20-F(/A)) and deregistration notices
    (evidence only), while the 10-K/10-Q/8-K rows stay exactly the body
    window so the poller's seen-set diff cannot re-fire on deploy."""
    globex = sec.SUBMISSIONS_URL.format(cik="0000654321")
    router.add(globex, _load("submissions_CIK0000654321.json"))
    rows = SECEdgarProvider().get_filings("GLBX", cik="654321", fetch_text=False)
    assert rows is not None
    assert [r["type"] for r in rows] == ["8-K", "10-Q/A", "25-NSE", "10-Q", "20-F", "15-12B", "20-F/A"]
    by_type = {r["type"]: r for r in rows}
    assert by_type["20-F"]["period_end"] == "2025-12-31" and by_type["20-F"]["filing_date"] == "2026-04-28"
    assert by_type["10-Q/A"]["accession_number"] == "0000654321-26-000039"
    assert by_type["25-NSE"]["period_end"] is None  # an empty reportDate stays None
    # The body read of the same index returns only the body forms, in order.
    bodies = SECEdgarProvider().get_filings("GLBX", cik="654321", fetch_text=True)
    assert [r["type"] for r in bodies] == ["8-K", "10-Q"]
    assert [r["accession_number"] for r in bodies] == [
        r["accession_number"] for r in rows if r["type"] in sec.BODY_FORMS]


def test_index_extras_have_separate_caps_and_never_shift_the_body_window(router):
    """Twelve body forms interleaved with twelve amendments and seven 25-NSE
    notices: the body window is the first ten body forms either way, the
    extras stop at ten, and deregistration notices at their own five so they
    cannot crowd an amendment out."""
    forms, dates, accs, reports, docs = [], [], [], [], []
    for i in range(12):
        for form in ("10-Q", "10-Q/A", "25-NSE") if i < 7 else ("10-Q", "10-Q/A"):
            n = len(forms)
            forms.append(form)
            dates.append(f"2026-{12 - i:02d}-01")
            accs.append(f"0000777777-26-{n:06d}")
            reports.append("2026-06-30" if form != "25-NSE" else "")
            docs.append(f"doc{n}.htm")
    import json
    body = json.dumps({"filings": {"recent": {"accessionNumber": accs, "filingDate": dates, "reportDate": reports,
                                              "form": forms, "primaryDocument": docs}}})
    router.add(sec.SUBMISSIONS_URL.format(cik="0000777777"), body)
    rows = SECEdgarProvider().get_filings("CAPS", cik="777777", fetch_text=False)
    assert rows is not None
    counts = {t: sum(r["type"] == t for r in rows) for t in ("10-Q", "10-Q/A", "25-NSE")}
    assert counts == {"10-Q": sec.MAX_BODY_FORMS, "10-Q/A": sec.MAX_INDEX_ONLY_FORMS,
                      "25-NSE": sec.MAX_DEREGISTRATION_FORMS}
    first_ten_bodies = [a for a, f in zip(accs, forms) if f == "10-Q"][:10]
    assert [r["accession_number"] for r in rows if r["type"] == "10-Q"] == first_ten_bodies
    bodies = SECEdgarProvider().get_filings("CAPS", cik="777777", fetch_text=True)
    assert [r["accession_number"] for r in bodies] == first_ten_bodies


def test_get_filings_fetches_text_and_extracts_sections(router):
    router.add(SUBMISSIONS, _load("submissions_CIK0000123456.json"))
    router.add(f"{ARCHIVE}/000012345625000031/acme-20250630.htm", _load("10k_trimmed.html"))
    router.add(f"{ARCHIVE}/000012345625000022/acme-8k-20250729.htm", _load("8k_trimmed.html"))
    router.add(f"{ARCHIVE}/000012345625000019/acme-20250331.htm", _load("10k_missing_item7.html"))
    router.add(f"{ARCHIVE}/000012345625000009/acme-20241231.htm", "gateway timeout", status=504)
    rows = SECEdgarProvider().get_filings("ACME", cik="123456")
    assert rows is not None and len(rows) == 4
    ten_k, eight_k, ten_q, dead_q = rows
    assert dead_q["text_fetch_error"] == "http_status_504"

    assert "TOC-SENTINEL-RISK" in ten_k["raw_text"]
    assert "TOC-SENTINEL-MDA" in ten_k["mda"] and len(ten_k["mda"]) <= 8000
    assert "TOC-SENTINEL-BUSINESS" in ten_k["business_description"]
    assert isinstance(ten_k["risk_factors"], list) and ten_k["risk_factors"]
    assert isinstance(ten_k["legal_or_regulatory"], list)
    assert "TOC-SENTINEL-LEGAL" in ten_k["legal_or_regulatory"][0]

    assert "TOC-SENTINEL-8K" in eight_k["raw_text"]
    assert "mda" not in eight_k and "risk_factors" not in eight_k
    assert eight_k["business_description"] is None    # metadata default, not extracted

    assert "risk_factors" in ten_q and "mda" not in ten_q

    # A dead document keeps its metadata row rather than dropping the filing.
    assert dead_q["raw_text"] == "" and "risk_factors" not in dead_q
    assert dead_q["accession_number"] == "0000123456-25-000009"

    doc_requests = [c for c in router.calls if str(c.url).startswith(ARCHIVE)]
    assert len(doc_requests) == 4
    assert all(c.headers["Accept"] == "text/html" for c in doc_requests)


def test_get_filings_resolves_cik_via_lookup(router):
    router.add(sec.TICKER_LOOKUP_URL, _load("company_tickers.json"))
    router.add(SUBMISSIONS, _load("submissions_CIK0000123456.json"))
    rows = SECEdgarProvider().get_filings("acme", fetch_text=False)
    assert rows and rows[0]["type"] == "10-K"
    assert [str(c.url) for c in router.calls] == [sec.TICKER_LOOKUP_URL, SUBMISSIONS]


def test_get_filings_unknown_ticker_returns_none_without_index_fetch(router):
    router.add(sec.TICKER_LOOKUP_URL, _load("company_tickers.json"))
    assert SECEdgarProvider().get_filings("NOPE", fetch_text=False) is None
    assert [str(c.url) for c in router.calls] == [sec.TICKER_LOOKUP_URL]


@pytest.mark.parametrize("body, status, exc", [
    ("not found", 404, None),
    ("<html>rate limited</html>", 200, None),         # malformed JSON body
    ("", 200, httpx.ReadTimeout("timed out")),
])
def test_get_filings_bad_index_returns_none(router, body, status, exc):
    router.add(SUBMISSIONS, body, status=status, exc=exc)
    assert SECEdgarProvider().get_filings("ACME", cik="123456", fetch_text=False) is None


def test_get_filings_index_without_recent_block_is_empty_not_none(router):
    router.add(SUBMISSIONS, '{"cik": "123456", "filings": {}}')
    assert SECEdgarProvider().get_filings("ACME", cik="123456", fetch_text=False) == []


# ---------------------------------------------------------------------------
# fetch_filing_text
# ---------------------------------------------------------------------------

def test_fetch_filing_text_truncates_oversized_documents(router, monkeypatch):
    url = f"{ARCHIVE}/000012345625000031/acme-20250630.htm"
    router.add(url, _load("10k_trimmed.html"))
    monkeypatch.setattr(sec, "MAX_TEXT_BYTES", 300)
    text = SECEdgarProvider().fetch_filing_text(url)
    assert text is not None and text.endswith("…[truncated]")
    assert len(text) <= 300 + len("\n\n…[truncated]")


def test_fetch_filing_text_failure_modes(router):
    url = f"{ARCHIVE}/000012345625000031/acme-20250630.htm"
    p = SECEdgarProvider()
    assert p.fetch_filing_text("") is None
    router.add(url, "", status=503)
    assert p.fetch_filing_text(url) is None
    router.add(url, "", exc=httpx.ConnectTimeout("no route"))
    assert p.fetch_filing_text(url) is None


def test_status_is_keyless_and_static():
    s = SECEdgarProvider().status()
    assert s.configured is True and s.healthy is True
    assert s.capabilities == ["filings"]
