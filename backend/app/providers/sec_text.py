"""Bounded incremental text extraction for large SEC HTML documents."""
from __future__ import annotations

import html
import re
import zlib
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser

PARSER_BUFFER_CHARS = 64 * 1024


def decoded_bytes(chunks: Iterable[bytes], encoding: str) -> Iterator[bytes]:
    """Bound decompression output as well as transport input.

    HTTPX's normal decoder can expand a compressed network chunk into one
    enormous bytes object before iter_text applies its chunk size.
    """
    encoding = encoding.strip().lower()
    if encoding in ("", "identity"):
        yield from chunks
        return
    if encoding not in ("gzip", "x-gzip", "deflate"):
        raise ValueError(f"unsupported SEC content encoding: {encoding}")
    decoder = None
    for raw in chunks:
        if decoder is None:
            window = 31 if encoding in ("gzip", "x-gzip") else 15
            decoder = zlib.decompressobj(window)
        pending = raw
        while pending:
            expanded = decoder.decompress(pending, PARSER_BUFFER_CHARS)
            if expanded:
                yield expanded
            pending = decoder.unconsumed_tail
            if decoder.eof and decoder.unused_data:
                pending = decoder.unused_data
                decoder = zlib.decompressobj(31 if encoding in ("gzip", "x-gzip") else 15)
    if decoder is not None and not decoder.eof:
        raise ValueError("incomplete SEC compressed response")


class TextPrefix:
    """Keep a normalized prefix while counting the entire visible document."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.retained = 0
        self.observed = 0
        self.trailing = 0
        self.started = False
        self.previous = ""
        self.newlines = 0

    def append(self, text: str) -> None:
        normalized: list[str] = []
        for char in text:
            if char in "\u00a0\u2009\u200b\t":
                char = " "
            if char == " " and self.previous in (" ", "\n"):
                continue
            if char == "\n":
                if self.newlines >= 2:
                    continue
                self.newlines += 1
            else:
                self.newlines = 0
            self.previous = char
            if not self.started:
                if char.isspace():
                    continue
                self.started = True
            self.observed += 1
            self.trailing = self.trailing + 1 if char.isspace() else 0
            if self.retained < self.limit:
                normalized.append(char)
                self.retained += 1
        if normalized:
            self.parts.append("".join(normalized))

    @property
    def observed_chars(self) -> int:
        return self.observed - self.trailing

    def text(self) -> str:
        prefix = "".join(self.parts)
        return prefix.rstrip() if self.observed_chars <= self.limit else prefix


class BoundedHTMLStripper(HTMLParser):
    """HTMLParser with bounded output and bounded unfinished-token state.

    HTMLParser normally retains an entire unfinished tag, comment, entity,
    or script body. Huge inline attributes and malformed filings must not
    reintroduce whole-document buffering behind an HTTP streaming facade.
    Attributes/comments are never research text, so oversized markup is
    drained incrementally while preserving the tag name and quoted `>`.
    """

    _SKIP_TAGS = {"script", "style", "head"}
    _BLOCK_TAGS = {"p", "br", "tr", "li", "div", "h1", "h2", "h3", "h4"}

    def __init__(self, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self.output = TextPrefix(limit)
        self.skip_depth = 0
        self.oversized_tokens = 0
        self.max_pending_chars = 0
        self._discard: str | None = None
        self._quote = ""
        self._comment_tail = ""
        self._cdata_overflow = False
        self._oversized_open_tag: str | None = None
        self._markup_last = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        self.output.append(" ")
        if tag in self._SKIP_TAGS:
            self.skip_depth += 1
        elif tag in self._BLOCK_TAGS and not self.skip_depth:
            self.output.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        self.output.append(" ")
        self._cdata_overflow = False

    def handle_comment(self, data: str) -> None:
        self.output.append(" ")

    def handle_decl(self, decl: str) -> None:
        self.output.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            # Physical HTTP chunks are not semantic boundaries: adding a
            # separator on every callback would split words at 64 KiB.
            self.output.append(data)

    def _drain_markup(self, data: str) -> str:
        if self._discard == "comment":
            combined = self._comment_tail + data
            end = combined.find("-->")
            if end < 0:
                self._comment_tail = combined[-2:]
                return ""
            self._comment_tail = ""
            self._discard = None
            return combined[end + 3:]
        for index, char in enumerate(data):
            if self._quote:
                if char == self._quote:
                    self._quote = ""
            elif char in ("'", '"'):
                self._quote = char
            elif char == ">":
                self._discard = None
                if self._markup_last == "/" and self._oversized_open_tag:
                    self.handle_endtag(self._oversized_open_tag)
                    self.clear_cdata_mode()
                self._oversized_open_tag = None
                return data[index + 1:]
            elif not char.isspace():
                self._markup_last = char
        return ""

    def feed(self, data: str) -> None:
        # Bound feed size even when an HTTP transport yields one enormous
        # decoded fragment. No extra full-fragment copy is retained.
        for start in range(0, len(data), PARSER_BUFFER_CHARS):
            piece = data[start:start + PARSER_BUFFER_CHARS]
            if self._discard:
                piece = self._drain_markup(piece)
            if not piece:
                continue
            super().feed(piece)
            self.max_pending_chars = max(self.max_pending_chars, len(self.rawdata))
            if len(self.rawdata) <= PARSER_BUFFER_CHARS:
                continue
            pending = self.rawdata
            self.rawdata = ""
            if self.cdata_elem:
                if not self._cdata_overflow:
                    self.oversized_tokens += 1
                    self._cdata_overflow = True
                # Keep enough suffix to recognize a closing tag that
                # straddles the next feed; all earlier content is skipped.
                self.rawdata = pending[-128:]
            elif not pending.startswith("<"):
                # No HTML entity name can be 64 KiB long. It is literal
                # visible text, not a reason to buffer the rest of a filing.
                self.oversized_tokens += 1
                self.handle_data(html.unescape(pending))
            else:
                self.oversized_tokens += 1
                self._discard = "comment" if pending.startswith("<!--") else "tag"
                self._markup_last = ""
                if self._discard == "tag":
                    match = re.match(r"<\s*(/?)\s*([a-zA-Z][\w:.-]*)", pending[:256])
                    if match:
                        tag = match[2].lower()
                        if match[1]:
                            self.handle_endtag(tag)
                        else:
                            self._oversized_open_tag = tag
                            self.handle_starttag(tag, [])
                            if tag in self.CDATA_CONTENT_ELEMENTS:
                                self.set_cdata_mode(tag)
                remainder = self._drain_markup(pending)
                if remainder:
                    super().feed(remainder)
