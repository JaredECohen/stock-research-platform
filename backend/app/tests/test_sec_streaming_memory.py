"""SEC extraction bounds must hold before and after HTML/decompression."""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.providers import sec_edgar_provider as sec
from app.providers.sec_text import PARSER_BUFFER_CHARS, BoundedHTMLStripper, decoded_bytes


def _parse(text, *, size=4096, limit=250_000):
    parser = BoundedHTMLStripper(limit)
    for offset in range(0, len(text), size):
        parser.feed(text[offset:offset + size])
    parser.close()
    return parser


@pytest.mark.parametrize("size", [1, 17, 4096, 65536])
def test_transport_boundaries_do_not_split_words_entities_or_sections(size):
    raw = (
        "<html><head><title>omit</title></head><body>"
        "<h2>Item 7. Management’s Discussion</h2>"
        "<p>revenue &amp; earnings &#x1F642; <b>rose</b> 12%.</p>"
        "<table><tr><td>Gross margin</td><td>42.3%</td></tr></table>"
        "<script>omit scripts</script><style>omit styles</style>"
        "<p>Next paragraph.</p></body></html>"
    )
    parser = _parse(raw, size=size)
    text = parser.output.text()
    assert text == _parse(raw, size=len(raw)).output.text()
    assert text.split() == sec._strip_html(raw).split()
    assert "\n" in text
    assert "& earnings 🙂" in text
    assert "omit" not in text
    assert parser.output.observed_chars == len(text)


def test_existing_sec_fixtures_preserve_all_words_and_mapped_sections():
    for path in (Path(__file__).parent / "fixtures" / "sec_edgar").glob("*.html"):
        raw = path.read_text()
        old = sec._strip_html(raw)
        parser = _parse(raw, size=17)
        new = parser.output.text()
        assert new.split() == old.split(), path.name
        assert sec._extract_sections(new)[0].keys() == sec._extract_sections(old)[0].keys()


@pytest.mark.parametrize("shape", ["visible", "attribute", "comment", "script", "entity", "unclosed_tag", "self_closing"])
def test_giant_unbroken_tokens_cannot_accumulate_a_whole_body(shape):
    size = 2 * 1024 * 1024
    data = "x" * size
    forms = {
        "visible": data,
        "attribute": '<div title="' + data + '>still attribute">Visible after tag</div>',
        "comment": "<!--" + data + "-->Visible after comment",
        "script": "<script>" + data + "</script>Visible after script",
        "entity": "&" + data + "; visible",
        "unclosed_tag": '<div title="' + data,
        "self_closing": '<style title="' + data + '"/>Visible after self-closing tag',
    }
    parser = _parse(forms[shape])
    assert parser.max_pending_chars <= PARSER_BUFFER_CHARS * 2
    assert sum(map(len, parser.output.parts)) <= 250_000
    if shape == "visible":
        assert parser.output.observed_chars == size
        assert parser.output.text() == "x" * 250_000
    elif shape in ("attribute", "comment", "script", "self_closing"):
        assert parser.output.text().startswith("Visible after")
        assert parser.oversized_tokens == 1
    elif shape == "entity":
        assert parser.output.observed_chars == size + len("&; visible")
    else:
        assert parser.output.observed_chars == 0
        assert parser.oversized_tokens == 1


def test_compressed_transport_cannot_expand_one_fragment_without_bound():
    payload = b"a" * (5 * 1024 * 1024)
    compressed = gzip.compress(payload)
    pieces = list(decoded_bytes([compressed], "gzip"))
    assert max(map(len, pieces)) <= PARSER_BUFFER_CHARS
    assert b"".join(pieces) == payload


def test_fetch_uses_streaming_only_and_reports_exact_truncation(monkeypatch, caplog):
    class NoBufferResponse(httpx.Response):
        def read(self):
            raise AssertionError("whole response read")

        @property
        def text(self):
            raise AssertionError("whole response text")

        @property
        def content(self):
            raise AssertionError("whole response content")

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"<p>"
            for _ in range(200):
                yield b"abcdefghij" * 4096
            yield b"</p>"

    def handler(request):
        assert request.headers["accept-encoding"] == "identity"
        return NoBufferResponse(200, stream=Stream())

    monkeypatch.setattr(sec, "httpx", SimpleNamespace(
        Client=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
    ))
    url = "https://www.sec.gov/Archives/synthetic-large.htm"
    with caplog.at_level(logging.INFO):
        result = sec.SECEdgarProvider().fetch_filing_document(url)
    assert result is not None
    assert result.truncated
    assert result.observed_chars == 200 * 40960
    assert result.retained_chars == sec.MAX_TEXT_BYTES
    assert result.bytes_read == 200 * 40960 + len("<p></p>")
    assert result.text == ("abcdefghij" * 25000) + "\n\n…[truncated]"
    assert url in caplog.text
    assert "truncated_chars=7942000" in caplog.text
    assert "sec_filing_fetch_start" in caplog.text
    assert "sec_filing_fetch_end" in caplog.text
