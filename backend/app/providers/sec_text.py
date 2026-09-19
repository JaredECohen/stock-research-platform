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
    prefix = b""
    for raw in chunks:
        if decoder is None:
            prefix += raw
            if encoding == "deflate" and len(prefix) < 2:
                continue
            raw, prefix = prefix, b""
            if encoding in ("gzip", "x-gzip"):
                window = 31
            else:
                # Some servers send raw DEFLATE despite the HTTP header's
                # zlib-wrapper convention; HTTPX supports both forms too.
                zlib_header = raw[0] & 15 == 8 and int.from_bytes(raw[:2], "big") % 31 == 0
                window = 15 if zlib_header else -15
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
        super().__init__(convert_charrefs=False)
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
        self._numeric_base = 10
        self._numeric_value = 0
        self._source_index = 0

    def updatepos(self, i: int, j: int) -> int:
        # HTMLParser calls this immediately before each entity callback.
        # Keep its bounded-buffer cursor so optional semicolons remain exact.
        self._source_index = j
        return super().updatepos(i, j)

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

    def unknown_decl(self, data: str) -> None:
        self.output.append(" ")

    def handle_pi(self, data: str) -> None:
        self.output.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            # Physical HTTP chunks are not semantic boundaries: adding a
            # separator on every callback would split words at 64 KiB.
            self.output.append(data)

    def handle_entityref(self, name: str) -> None:
        end = self._source_index + len(name) + 1
        suffix = ";" if self.rawdata[end:end + 1] == ";" else ""
        self.handle_data(html.unescape(f"&{name}{suffix}"))

    @staticmethod
    def _numeric_reference_value(name: str) -> int:
        hexadecimal = name.lower().startswith("x")
        digits = (name[1:] if hexadecimal else name).lstrip("0") or "0"
        # Avoid Python's integer-string conversion limit. Any significant
        # numeric reference this long is outside Unicode, regardless of radix.
        return 0x110000 if len(digits) > 7 else int(digits, 16 if hexadecimal else 10)

    def handle_charref(self, name: str) -> None:
        self.handle_data(html.unescape(f"&#{self._numeric_reference_value(name)};"))

    def _drain_markup(self, data: str) -> str:
        if self._discard in ("comment", "marked"):
            combined = self._comment_tail + data
            pattern = r"--!?>" if self._discard == "comment" else r"\]\]>"
            match = re.search(pattern, combined)
            if match is None:
                self._comment_tail = combined[-3:]
                return ""
            self._comment_tail = ""
            self._discard = None
            return combined[match.end():]
        if self._discard == "numeric":
            digits = "0123456789abcdefABCDEF" if self._numeric_base == 16 else "0123456789"
            for index, char in enumerate(data):
                if char not in digits:
                    self.handle_data(html.unescape(f"&#{self._numeric_value};"))
                    self._discard = None
                    return data[index + (char == ";"):]
                self._numeric_value = min(
                    0x110000, self._numeric_value * self._numeric_base + int(char, self._numeric_base),
                )
            return ""
        if self._discard == "cdata_close":
            for index, char in enumerate(data):
                if char == ">":
                    self.handle_endtag(self.cdata_elem)
                    self.clear_cdata_mode()
                    self._discard = None
                    return data[index + 1:]
                if not char.isspace():
                    # It was script content resembling an incomplete end
                    # tag. Keep skipping script text until a real end tag.
                    self._discard = None
                    return data[index:]
            return ""
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

    # HTMLParser keeps unparsed input in `rawdata`, and since CPython
    # 3.12.14 / 3.13.15 also in a second buffer: `feed` parks input in
    # `_pending` and only joins it into `rawdata` once `_parse_threshold`
    # is reached, so an unterminated token is not rescanned on every call.
    # Bounding unfinished-token state means accounting for both. Measuring
    # `rawdata` alone let the deferred half grow unmeasured, and let a
    # document tail reach `goahead` *after* `close` had already rewritten
    # `rawdata` — a 10,000-digit character reference leaked its digits into
    # the extracted text. `_pending` is absent on older interpreters.

    def _buffered_chars(self) -> int:
        """Characters the parser still holds, across both of its buffers."""
        return len(self.rawdata) + getattr(self, "_pending_len", 0)

    def _take_buffered(self) -> str:
        """Drain every unparsed character and let parsing resume at once.

        Leaving `_parse_threshold` where CPython set it would re-park the
        next buffer's worth of input in `_pending` unmeasured, which is
        precisely the bound this class exists to enforce.
        """
        data = self.rawdata
        pending = getattr(self, "_pending", None)
        if pending is not None:
            data += "".join(pending)
            pending.clear()
            self._pending_len = 0
            self._parse_threshold = 1
        self.rawdata = ""
        return data

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
            buffered = self._buffered_chars()
            self.max_pending_chars = max(self.max_pending_chars, buffered)
            if buffered <= PARSER_BUFFER_CHARS:
                continue
            pending = self._take_buffered()
            if self.cdata_elem:
                if not self._cdata_overflow:
                    self.oversized_tokens += 1
                    self._cdata_overflow = True
                end_tag = re.search(rf"</\s*{re.escape(self.cdata_elem)}\s*$", pending, re.IGNORECASE)
                if end_tag:
                    self._discard = "cdata_close"
                else:
                    # Keep enough suffix to recognize a closing tag that
                    # straddles the next feed; earlier script text is skipped.
                    self.rawdata = pending[-128:]
            elif not pending.startswith("<"):
                # No HTML entity name can be 64 KiB long. It is literal
                # visible text, not a reason to buffer the rest of a filing.
                self.oversized_tokens += 1
                numeric = re.search(r"&#([xX]?)([0-9a-fA-F]+)$", pending)
                if numeric and (numeric[1] or numeric[2].isdigit()):
                    self.handle_data(pending[:numeric.start()])
                    self._discard = "numeric"
                    self._numeric_base = 16 if numeric[1] else 10
                    self._numeric_value = 0
                    self._drain_markup(numeric[2])
                else:
                    self.handle_data(html.unescape(pending))
            else:
                self.oversized_tokens += 1
                self._discard = (
                    "comment" if pending.startswith("<!--") else
                    "marked" if pending.startswith("<![CDATA[") else "tag"
                )
                self.output.append(" ")
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

    def close(self) -> None:
        if self._discard == "numeric":
            self.handle_data(html.unescape(f"&#{self._numeric_value};"))
            self._discard = None
        # Let HTMLParser's automatic conversion handle pending EOF references
        # (including unknown names) exactly like html.unescape. Normalize the
        # numeric spelling first to avoid Python's decimal-int digit limit.
        # The rewrite has to cover the deferred buffer too: `HTMLParser.close`
        # appends it to `rawdata`, so anything left there would arrive
        # unnormalized after this line had already run.
        self.rawdata = re.sub(
            r"&#([xX][0-9a-fA-F]+|[0-9]+);?",
            lambda match: f"&#{self._numeric_reference_value(match[1])};",
            self._take_buffered(),
        )
        self.convert_charrefs = True
        try:
            super().close()
        finally:
            self.convert_charrefs = False
