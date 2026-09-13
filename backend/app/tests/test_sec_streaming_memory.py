"""SEC extraction bounds must hold before and after HTML/decompression."""
from __future__ import annotations

import gzip
import logging
import zlib
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


@pytest.mark.parametrize("size", [1, 7, 4096])
@pytest.mark.parametrize("raw", [
    "S&P Global", "before &foo after", "before &notit after",
    "before &foo; after", "before &notit; after",
    "<script>&P;</script> S&P Global &amp; earnings",
])
def test_optional_entity_semicolons_preserve_the_original_text(raw, size):
    assert _parse(raw, size=size).output.text().split() == sec._strip_html(raw).split()


@pytest.mark.parametrize("size", [1, 7, 4096])
@pytest.mark.parametrize("raw, expected", [
    ("&amp;&unknown; &notin;&notit", "&&unknown; ∉¬it"), ("&amp", "&"), ("S&P", "S&P"),
])
def test_named_reference_at_eof_keeps_standard_unescape_semantics(raw, expected, size):
    assert _parse(raw, size=size).output.text() == expected


@pytest.mark.parametrize("digits, expected", [("1" * 10000, "�"), ("0" * 10000 + "65", "A")])
def test_long_numeric_reference_at_eof_keeps_unescape_semantics_without_int_limit(digits, expected):
    assert _parse("before &#" + digits).output.text() == "before " + expected


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


@pytest.mark.parametrize("size", [1, 7, 4096])
def test_raw_deflate_matches_the_original_http_client_compatibility(size):
    payload = b"raw deflate payload" * 100
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(payload) + compressor.flush()
    chunks = (compressed[i:i + size] for i in range(0, len(compressed), size))
    assert b"".join(decoded_bytes(chunks, "deflate")) == payload


@pytest.mark.parametrize("shape", ["comment", "numeric", "script_end", "marked"])
def test_oversized_token_terminators_preserve_the_text_after_them(shape):
    large = "x" * (PARSER_BUFFER_CHARS * 2 + 1)
    raw = {
        "comment": "<p>before</p><!--" + large + "--!><p>after</p>",
        "numeric": "before &#" + "1" * len(large) + "; after",
        "script_end": "before<script>" + large + "</script" + " " * len(large) + ">after",
        "marked": "before<![CDATA[" + large + ">hiddenCDATAtail]]>after",
    }[shape]
    parser = _parse(raw)
    expected = ["before", "�", "after"] if shape == "numeric" else ["before", "after"]
    assert parser.output.text().split() == expected
    assert parser.max_pending_chars <= PARSER_BUFFER_CHARS * 2


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
