"""W2b 7(a) — number-to-source check for company memos.

Every figure the memo prints is checked against the facts the analysts
were actually given (`source_ledger.FactRegistry`). The check is
deterministic, costs no LLM call, and is a PROVENANCE check, not a truth
or currency check: a figure traced to the initial DCF is traced, and a
correctly traced number attributed to the wrong segment is out of scope.

Statuses (per occurrence, `NumberClaim.status`):

* ``traced`` — the value matches a registered fact AND the claim is
  anchored to it (the words next to the number name the fact's metric),
  or it matches a verbatim (non-derived) fact at 4+ significant digits,
  where a coincidence is negligible. The critique's measurement is why
  anchoring is required: a bare value match on a 1-2 digit percentage
  accepts most fabricated values (60-87% false support).
* ``weak`` — a value match nobody can vouch for: not anchored, and the
  words next to the number name no metric the registry knows.
* ``mis_anchored`` — the claim names a registered metric but its value
  matches only something else (GOOGL's "$66 gap" between two DCF
  scenarios whose real gap is $92).
* ``untraceable`` — no registered fact carries the value.
* ``assumption`` — a forward figure the PM declared as a forecast
  assumption whose basis resolves in the ledger (labelled, not counted
  as untraceable).
* ``threshold`` — a forward condition (falsifiers, "FCF yield > 4%"):
  counted, never checked.

Only ``untraceable`` and ``mis_anchored`` are FLAGGED (and can withhold a
list item or lower confidence); ``weak`` is counted and listed.

Derived facts (growth rates, margins, scenario deltas, peer gaps; see
`source_ledger`) support only anchored claims — a derivation is a
legitimate reading of a number only when the prose says which one — and
a percentage-point claim never matches a relative change: "6.7
percentage points" is not a 6.7% relative increase (META v1).

This module is pure: no DB, no settings reads except where the caller
passes a value in.
"""
from __future__ import annotations

import bisect
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # source_ledger imports this module's parser at runtime
    from ..schemas import NumberCheck
    from .source_ledger import Fact, FactRegistry

METHOD_VERSION = "1"

FLAGGED_STATUSES = frozenset({"untraceable", "mis_anchored"})
# Statuses a claim of fact can end with (thresholds and assumptions are
# counted separately).
FACT_STATUSES = ("traced", "weak", "mis_anchored", "untraceable")

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Masks: spans replaced by equal-length spaces so offsets survive. The first
# three are the same patterns as `industry_report_validator` (lines 64-66),
# copied rather than imported: that module is W2a's publication gate and
# must not change shape under this check.
_MASKS: tuple[re.Pattern[str], ...] = tuple(re.compile(p, re.I) for p in (
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b",                   # ISO date
    r"\b\d{4}-W\d{2}\b",                                                      # ISO week
    r"\b\d+\.\d+\.\d+(?:[-\w.]*)?\b|\bv\d+(?:\.\d+)*\b",                      # versions
    # "may" is also the English verb ("revenue may 30% higher"), so only a
    # capitalised "May" is a month; and a day number is never followed by a
    # percent sign or a decimal ("Mar 12.5%" is a figure, not a date).
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|(?-i:May)|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?!\s?%|\.\d)(?:,?\s+\d{4})?\b",                                     # "Sep 24, 2026"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                                            # 9/24/2026
    # fiscal / calendar periods
    r"\b(?:FY|CY)\s?'?\d{2,4}\b",
    r"\b[QH][1-4]\s?(?:FY\s?)?'?\d{2,4}\b",
    r"\b\d{4}\s?[QH][1-4]\b",
    r"\b[QH][1-4]\b",
    r"\bfiscal(?:\s+year)?\s+\d{4}\b",
    r"\b(?:19|20)\d{2}\s?[-–/]\s?(?:(?:19|20)\d{2}|\d{2})\b",                 # 2026-2027, 2025/26
    r"\b(?:19|20)\d0s\b",                                                     # 2030s
    # filing forms, index names, identifiers
    r"\b(?:10-K|10-Q|8-K|20-F|6-K|40-F|S-1|S-3|S-4|F-1|DEF\s?14A|13[FDG]|11-K)\b",
    r"\bItem\s+\d+[A-Z]?\b",
    r"\bForm\s+\d+\b",
    r"\bS&P\s?\d+\b|\bRussell\s?\d{4}\b|\bNasdaq[- ]?\d+\b|\bFTSE\s?\d+\b|\bDAX\s?\d+\b|\bDow\s?\d+\b",
    r"\b\d{10}-\d{2}-\d{6}\b",                                                # SEC accession
    r"§\s?\d+(?:\.\d+)*",
    r"\b\d+(?:st|nd|rd|th)\b",                                                # ordinals
    r"\b\d+-for-\d+\b",                                                       # splits
    # indicator parameters (the parameter is not a claim; the reading is)
    # (case-sensitive: "ma" and "roc" are English)
    r"(?-i:\b(?:RSI|ATR|ADX|CCI|MFI|ROC|SMA|EMA|WMA|MA)\s?\(\s?\d+(?:\s?,\s?\d+)*\s?\))",
    r"(?-i:\bMACD\s?\(\s?\d+\s?,\s?\d+\s?,\s?\d+\s?\))",
    r"(?-i:\b(?:SMA|EMA|MA)\s?\d+(?:\s?/\s?\d+)?\b)",
    r"(?-i:\bRSI[- ]?\d+\b)",
    # score denominators: "25/100" keeps the numerator, drops "/100"
    r"/\s?(?:100|10|5)\b",
))

_UNIT_PATTERN = (
    r"(?P<unit>"
    r"\s?(?:%|percent\b|per\s?cent\b)"
    r"|[\s-]?percentage[\s-]points?\b"
    r"|[\s-]?basis[\s-]points?\b"
    r"|\s?(?:pp\b|p\.p\.|pts?\b|bps\b|bp\b)"
    r"|\s?×"
    r"|x(?![A-Za-z0-9])"
    r"|\s?(?:thousand|million|billion|trillion)\b"
    r"|(?:k|mm|mn|mil|m|bn|b|tn|t)(?![A-Za-z0-9])"
    r")?"
)
_TOKEN_RE = re.compile(
    r"(?P<cur>US\$|\$|€|£|¥)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?P<dec>\.\d+)?"
    + _UNIT_PATTERN,
    re.I,
)
_SCALE = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mm": 1e6, "mn": 1e6, "mil": 1e6,
    "billion": 1e9, "b": 1e9, "bn": 1e9,
    "trillion": 1e12, "t": 1e12, "tn": 1e12,
}
_SIGN_CHARS = "-+−"
_SIGN_OPENERS = set(" \t\n([{,~≈:;/")
_RANGE_SEP_RE = re.compile(r"^\s?[-–—]\s?[~≈]?$|^\s+to\s+[~≈]?$", re.I)
_DURATION_RE = re.compile(
    r"^[\s-]?(?:days?|weeks?|months?|quarters?|years?|hours?|minutes?|decades?|sessions?|bars?"
    r"|consecutive|straight|successive)\b", re.I)
# A comparison symbol in front of a figure makes it a threshold. An ASCII
# arrow ("12% -> 97%", "=>") is a change between two stated values, so the
# value after it is a claim of fact like any other.
_THRESHOLD_RE = re.compile(r"(?:>=|<=|(?<![-=])>|<|[≥≤])\s?$")
_COUNT_MAX = 12


@dataclass(frozen=True, slots=True)
class ParsedClaim:
    raw: str
    start: int
    end: int
    value: float          # abs(number) x scale; percent and pp as printed
    unit: str             # "pct" | "pp" | "multiple" | "usd" | "currency" | "number"
    decimals: int
    scale: float
    sig_digits: int
    tol: float
    cls: str = "fact"     # "fact" | "threshold" | "exempt"
    exempt_reason: str = ""
    signed: bool = False


def _mask(text: str) -> str:
    out = text
    for rx in _MASKS:
        out = rx.sub(lambda m: " " * (m.end() - m.start()), out)
    return out


def _sig_digits(num: str, dec: str) -> int:
    digits = num.replace(",", "")
    if dec:
        allds = (digits + dec).lstrip("0")
        return max(1, len(allds))
    stripped = digits.lstrip("0").rstrip("0")
    return max(1, len(stripped))


@dataclass
class _Tok:
    start: int
    end: int
    num: float
    decimals: int
    sig: int
    int_digits: str
    unit_raw: str
    cur: str
    signed: bool
    range_prev: bool = False     # joined to the previous token by a range separator
    rejected: str = ""
    threshold: bool = False


def _unit_of(tok: _Tok) -> tuple[str, float]:
    u = " ".join(tok.unit_raw.lower().replace("-", " ").split())
    if u in ("%", "percent", "per cent", "percent."):
        return "pct", 1.0
    if u.startswith("percentage point"):
        return "pp", 1.0
    if u in ("pp", "p.p.", "pt", "pts"):
        return "pp", 1.0
    if u in ("bps", "bp") or u.startswith("basis point"):
        return "pp", 0.01
    if u in ("x", "×"):
        return "multiple", 1.0
    scale = _SCALE.get(u, 1.0)
    if tok.cur:
        return ("usd" if "$" in tok.cur else "currency"), scale
    return "number", scale


def _tokens(masked: str) -> list[_Tok]:
    toks: list[_Tok] = []
    for m in _TOKEN_RE.finditer(masked):
        start, end = m.start(), m.end()
        num, dec = m["num"], (m["dec"] or "")[1:]
        cur = m["cur"] or ""
        unit_raw = (m["unit"] or "").strip()
        tok = _Tok(start=start, end=end, num=float(num.replace(",", "") + ("." + dec if dec else "")),
                   decimals=len(dec), sig=_sig_digits(num, dec), int_digits=num.replace(",", ""),
                   unit_raw=unit_raw, cur=cur, signed=False)
        before = masked[start - 1] if start > 0 else " "
        # Lead guard: glued to a letter, a dot or a slash is an identifier
        # (H100, v1.2, 3/4); "-" glued to a word is a compound (COVID-19),
        # after a digit it is a range separator, after a space a sign.
        if not cur:
            if before.isalpha() or before in "./_" or before.isdigit():
                tok.rejected = "identifier"
            elif before in _SIGN_CHARS:
                b2 = masked[start - 2] if start > 1 else " "
                if b2.isalpha():
                    tok.rejected = "identifier"
                elif b2.isdigit() or (b2 == " " and toks and toks[-1].end >= start - 3):
                    tok.range_prev = True
                elif b2 in _SIGN_OPENERS or start - 1 == 0:
                    tok.signed = True
                    tok.start = start - 1
            elif before in "–—":
                tok.range_prev = True
        else:
            if before in _SIGN_CHARS:
                b2 = masked[start - 2] if start > 1 else " "
                if b2 in _SIGN_OPENERS or start - 1 == 0:
                    tok.signed = True
                    tok.start = start - 1
                elif b2.isdigit():
                    tok.range_prev = True
            elif before.isalpha():
                tok.rejected = "identifier"
        # Trail guard: a unit-less number glued to letters/digits is an
        # identifier ("52w", "5G", "10b5-1").
        after = masked[tok.end] if tok.end < len(masked) else " "
        if not unit_raw and (after.isalpha() or after.isdigit()):
            tok.rejected = tok.rejected or "identifier"
        if not tok.rejected:
            lead = masked[max(0, tok.start - 3):tok.start]
            if _THRESHOLD_RE.search(lead):
                tok.threshold = True
        toks.append(tok)
    # " to " ranges ("from $205 to 220")
    for i in range(1, len(toks)):
        a, b = toks[i - 1], toks[i]
        if not b.range_prev and _RANGE_SEP_RE.match(masked[a.end:b.start] or "x"):
            b.range_prev = True
    return toks


def extract_claims(text: str, *, threshold: bool = False) -> list[ParsedClaim]:
    """Every numeric token in `text` as a `ParsedClaim` (exempt ones included,
    with `cls="exempt"`). Offsets index `text` itself: masks replace spans
    with equal-length blanks, so `text[c.start:c.end] == c.raw`."""
    if not text or not any(ch.isdigit() for ch in text):
        return []
    masked = _mask(text)
    toks = _tokens(masked)
    # Range inheritance: "$205–220" (currency), "4-4.5%" (unit), "$1.2–1.5B" (scale).
    for i in range(1, len(toks)):
        a, b = toks[i - 1], toks[i]
        if not b.range_prev or a.rejected or b.rejected:
            continue
        if not a.unit_raw and b.unit_raw:
            a.unit_raw = b.unit_raw
        if not b.cur and a.cur:
            b.cur = a.cur
        if not a.cur and b.cur:
            a.cur = b.cur
        if not b.unit_raw and a.unit_raw:
            b.unit_raw = a.unit_raw
        b.threshold = b.threshold or a.threshold
    out: list[ParsedClaim] = []
    for tok in toks:
        if tok.rejected:
            continue
        unit, scale = _unit_of(tok)
        value = abs(tok.num) * scale
        # Printed precision. An integer with trailing zeros and a scale word
        # ("1,200 million") is rounded to its significant digits, never
        # looser than two of them.
        z = len(tok.int_digits) - len(tok.int_digits.rstrip("0"))
        z_eff = min(z, max(0, len(tok.int_digits) - 2)) if (tok.decimals == 0 and scale >= 1e3) else 0
        tol = 0.5 * (10 ** (z_eff - tok.decimals)) * scale + 1e-9
        raw = text[tok.start:tok.end]
        cls, reason = "fact", ""
        if threshold or tok.threshold:
            cls = "threshold"
        elif unit == "number" and scale == 1.0 and tok.decimals == 0:
            follow = masked[tok.end:tok.end + 14]
            if tok.num <= _COUNT_MAX:
                cls, reason = "exempt", "count"
            elif len(tok.int_digits) == 4 and 1900 <= tok.num <= 2100:
                cls, reason = "exempt", "year"
            elif _DURATION_RE.match(follow):
                cls, reason = "exempt", "duration"
        out.append(ParsedClaim(
            raw=raw, start=tok.start, end=tok.end, value=value, unit=unit,
            decimals=tok.decimals, scale=scale, sig_digits=tok.sig, tol=tol,
            cls=cls, exempt_reason=reason, signed=tok.signed,
        ))
    return out


# ---------------------------------------------------------------------------
# Metric vocabulary (anchoring)
# ---------------------------------------------------------------------------

# Closed synonym table: phrase pattern -> canonical tokens. Applied to prose
# and to structured key paths (split on `.`, `_`, `[ ]` and camelCase), so
# "EV/EBITDA", "ev_ebitda" and "EV to EBITDA" all anchor to one another.
# ORDER MATTERS: a matched span is consumed, so compounds come first —
# "EV/EBITDA" must not also read as "EBITDA" (it would anchor to
# debt/EBITDA), nor "FCF yield" as "FCF" (the FCF margin).
METRIC_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"bull\s*(?:/|to|-|vs\.?)?\s*bear\s+ratio", ("bull_bear_ratio",)),
    (r"\bev\s*/?\s*ebitda\b|\bev\s+to\s+ebitda\b|\bebitda\s+multiples?\b", ("ev_ebitda",)),
    (r"\bev\s*/?\s*(?:revenue|sales)\b|\bev\s+to\s+(?:revenue|sales)\b", ("ev_revenue",)),
    (r"\bev\s*/?\s*ebit\b", ("ev_ebit",)),
    (r"\bp\s*/\s*fcf\b|\bpfcf\b|\bp\s+fcf\b|price[- ]to[- ]free[- ]cash(?:[- ]flow)?", ("pfcf",)),
    (r"\bforward\s+p\s*/?\s*e\b|\bforward\s+pe\b", ("forward_pe",)),
    (r"\bp\s*/\s*e\b|\bpe\b|price[- ]to[- ]earnings|earnings\s+multiple", ("pe",)),
    (r"\bp\s*/\s*s\b|\bps\b|price[- ]to[- ]sales", ("ps",)),
    (r"\bp\s*/\s*b\b|\bpb\b|price[- ]to[- ]book", ("pb",)),
    (r"\bfcf\s+yield\b|free[- ]cash[- ]flow\s+yield", ("fcf_yield",)),
    (r"\bfcf\s+margins?\b|free[- ]cash[- ]flow\s+margins?", ("fcf_margin", "margin")),
    (r"\bebitda\s+margins?\b", ("ebitda_margin", "margin")),
    (r"\b(?:operating|op|ebit)\s+margins?\b", ("operating_margin", "margin")),
    (r"\bgross\s+margins?\b", ("gross_margin", "margin")),
    (r"\bnet\s+margins?\b", ("net_margin", "margin")),
    (r"\binterest\s+coverage\b", ("interest_coverage",)),
    (r"\bnet\s+debt\s+to\s+ebitda\b|\bdebt\s*(?:/|to)\s*ebitda\b", ("debt_to_ebitda",)),
    (r"operating\s+cash\s+flow|\bocf\b|cash\s+from\s+operations", ("ocf",)),
    (r"\bfcff?\b|free[- ]cash[- ]flows?\b", ("fcf",)),
    (r"\bebitda\b", ("ebitda",)),
    (r"\boperating\s+(?:income|profit)\b|\bebit\b", ("operating_income",)),
    (r"\bgross\s+profit\b", ("gross_profit",)),
    (r"\bnet\s+(?:income|earnings|profit)\b", ("net_income",)),
    (r"\beps\b|earnings\s+per\s+share", ("eps",)),
    (r"market\s+cap(?:italization)?\b|\bmkt\s*cap\b", ("market_cap",)),
    (r"enterprise\s+value", ("ev",)),
    (r"\brevenues?\b|\bsales\b|\btop[- ]line\b|\bturnover\b", ("revenue",)),
    (r"\bcapex\b|capital\s+(?:expenditures?|spending|spend)", ("capex",)),
    (r"\broic\b|return\s+on\s+invested\s+capital", ("roic",)),
    (r"\broe\b|return\s+on\s+equity", ("roe",)),
    (r"\broa\b|return\s+on\s+assets", ("roa",)),
    (r"\bwacc\b|discount\s+rate|cost\s+of\s+capital", ("wacc",)),
    (r"\bexit\s+(?:ebitda\s+)?multiple\b", ("exit_multiple",)),
    (r"\bterminal\b|\btgr?\b", ("terminal",)),
    (r"\bdcf\b|discounted[- ]cash[- ]flow", ("dcf",)),
    (r"\bbull(?:ish)?\b|\bupside\s+(?:scenario|case)\b", ("bull",)),
    (r"\bbear(?:ish)?\b|\bdownside\s+(?:scenario|case)\b", ("bear",)),
    (r"\bbase\b", ("base",)),
    (r"\bupside\b|\bdownside\b|fair\s+value|\bimplie[sd]\b|\bimply\b|\bintrinsic\b|\bmispric", ("upside",)),
    (r"\bprices?\b|\bpriced\b|\bquote\b|\blast\s+close\b|\bclose\b", ("price",)),
    (r"\bgrowth\b|\bgrew\b|\bgrow(?:s|ing)?\b|\brose\b|\brise[sn]?\b|\bincreas|\bdecreas|\byoy\b"
     r"|year[- ]over[- ]year|\bcagr\b|\bqoq\b|\bdeclin|\bfell\b|\bexpan|\bcontract", ("growth",)),
    (r"\bmargins?\b", ("margin",)),
    (r"\bpremium\b|\bdiscount\b", ("premium",)),
    (r"\bpeers?\b|\bmedian\b|\bcohort\b|\bquartile\b|\bcomps?\b|\bcomparables?\b", ("peer",)),
    (r"\bdebt\b|\bleverage\b|\bborrowings?\b", ("debt",)),
    (r"\bdividends?\b|\bpayout\b", ("dividend",)),
    (r"\bbuybacks?\b|\brepurchas", ("buyback",)),
    (r"\bshare\s+price", ("price",)),
    (r"\bshares?\b|\bdiluted\b|share\s+count", ("shares",)),
    (r"\bbeta\b", ("beta",)),
    (r"\brsi\b", ("rsi",)),
    (r"\bmacd\b", ("macd",)),
    (r"\bbollinger\b|\bbb\b", ("bollinger",)),
    (r"\bsma\d*\b|\bema\d*\b|\bvwma\d*\b|moving\s+average", ("sma",)),
    (r"\b52\s?w(?:ee)?k\b|\b52w\b|\byear\s+(?:high|low)\b|trailing\s+(?:high|low)|\brange\b", ("range52",)),
    (r"valuation\s+(?:factor|score)|factor\s+valuation", ("valuation_factor",)),
    (r"\bexpenses?\b|\bopex\b|\bsg\s?&?\s?a\b|\br\s?&\s?d\b|research\s+and\s+development", ("expenses",)),
    (r"\busers?\b|\bdap\b|\bmau\b|\bdau\b|\bsubscribers?\b", ("users",)),
    (r"\bimpressions?\b", ("impressions",)),
    (r"\bads?\b|\badvertising\b", ("ads",)),
    (r"\bcloud\b|\bazure\b|\baws\b|\bgcp\b", ("cloud",)),
    (r"\bsegments?\b|\bdivisions?\b", ("segment",)),
    (r"\bspread\b|\bgap\b|\bdifference\b|\bdelta\b", ("spread",)),
    (r"\bratio\b", ("ratio",)),
    (r"fed(?:eral)?\s+funds|\bfedfunds\b", ("fed_funds",)),
    (r"\b10y\b|\b10[- ]year\b|\bdgs10\b|\btreasury\b|\byields?\s+curve", ("rates",)),
    (r"\bcpi\b|\binflation\b|\bcorestick|\bpce\b", ("inflation",)),
    (r"\bunemploy|\bunrate\b|\bjobless", ("unemployment",)),
    (r"credit\s+spreads?|high[- ]yield|\bhy\b|\boas\b", ("credit",)),
    (r"\bhhi\b|herfindahl|concentration|top[- ]3", ("concentration",)),
    (r"\btax(?:es)?\b|tax\s+rate", ("tax",)),
    (r"\binterest\b", ("interest",)),
    (r"\bguidance\b|\bguided\b|\boutlook\b", ("guidance",)),
    (r"\bbacklog\b|\brpo\b|remaining\s+performance", ("backlog",)),
    (r"\binventor", ("inventory",)),
    (r"\bsurprise\b|\bbeat\b|\bmiss(?:ed)?\b", ("surprise",)),
    (r"\bestimates?\b|\bconsensus\b", ("estimates",)),
    (r"\bpercentile\b|\brank\b", ("percentile",)),
    (r"\bworking\s+capital\b|\bnwc\b", ("working_capital",)),
    (r"\bdepreciation\b|\bd\s?&\s?a\b", ("da",)),
    (r"\bstock[- ]based\s+comp|\bsbc\b", ("sbc",)),
    (r"\bcash\b", ("cash",)),
    (r"\bassets?\b", ("assets",)),
    (r"\bequity\b", ("equity",)),
)
_SYN_RE = tuple((re.compile(p, re.I), toks) for p, toks in METRIC_SYNONYMS)
# Qualifiers say WHOSE number it is (which DCF scenario, peers, a spread or
# ratio between two), not WHICH metric; the rest of the vocabulary names a
# metric. GENERIC metrics are too common to anchor a claim that names a
# more specific one (W2b §4.3): "revenue growth 22%" is not "EPS growth 22%".
SCENARIOS = frozenset({"base", "bull", "bear"})
QUALIFIERS = SCENARIOS | frozenset({"dcf", "peer", "ratio", "spread"})
GENERIC = frozenset({"growth", "price", "margin", "upside"})

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SPLIT_RE = re.compile(r"[._\[\]\s]+")


def context_tokens(text: str) -> frozenset[str]:
    """Canonical metric tokens named in `text`."""
    if not text:
        return frozenset()
    low = text.lower()
    out: set[str] = set()
    for rx, toks in _SYN_RE:
        if rx.search(low):
            out.update(toks)
            # Consume the span so a compound is not also read as its parts.
            low = rx.sub(lambda m: " " * (m.end() - m.start()), low)
    return frozenset(out)


def key_tokens(path: Iterable[str]) -> frozenset[str]:
    """Canonical tokens of a structured key path (`premium_discount.ev_ebitda`)."""
    words: list[str] = []
    for seg in path:
        s = str(seg)
        if s.isdigit():
            continue
        s = _CAMEL_RE.sub(" ", s)
        words.extend(w for w in _KEY_SPLIT_RE.split(s) if w)
    return context_tokens(" ".join(words))


# Clause boundaries for anchoring: sentence ends, semicolons, newlines, em
# dashes, spaced en dashes and pipes. NOT colons or parentheses: "Op margin:
# 61%" and "$73.90 (-37%)" state the metric across them.
_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|;|\n|—|\s–\s|\|")
_WORD_RE = re.compile(r"[A-Za-z0-9&/$%'’.-]+")
NEAR_WORDS = 8
WIDE_WORDS = 12


def _words_tail(s: str, n: int) -> str:
    return " ".join(_WORD_RE.findall(s)[-n:])


def _words_head(s: str, n: int) -> str:
    return " ".join(_WORD_RE.findall(s)[:n])


@dataclass(frozen=True, slots=True)
class ClaimContext:
    left: frozenset[str]
    right: frozenset[str]
    wide: frozenset[str]


# Multiples are written value-first: "46.8x earnings", "22.3x sales",
# "34.4x EBITDA", "12x book". The noun after the "x" names the MULTIPLE
# (P/E, P/S, EV/EBITDA ...), not the underlying metric — "22.3x sales" is
# not a claim about revenue, so without this binding a correct multiple
# reads as mis_anchored against the revenue fact.
MULTIPLE_METRICS = frozenset({"pe", "forward_pe", "ps", "pb", "pfcf", "ev_ebitda", "ev_revenue",
                              "ev_ebit", "debt_to_ebitda", "exit_multiple", "interest_coverage"})
_PERIOD_WORD = r"(?:(?:trailing|ttm|ltm|current|forward|fwd|ntm|next[- ]year'?s?|this[- ]year'?s?)\s+)?"
_MULTIPLE_NOUNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(p, re.I), toks) for p, toks in (
        (r"(?:forward|fwd|ntm|next[- ]year'?s?)\s+(?:earnings|eps)\b", ("forward_pe",)),
        (_PERIOD_WORD + r"(?:earnings|eps)\b(?!\s+(?:growth|per|yield))", ("pe",)),
        # "x sales" is written for both P/S and EV/sales.
        (_PERIOD_WORD + r"(?:sales|revenues?)\b(?!\s+growth)", ("ps", "ev_revenue")),
        (_PERIOD_WORD + r"ebitda\b(?!\s+(?:margin|growth))", ("ev_ebitda",)),
        (_PERIOD_WORD + r"ebit\b(?!\s+(?:margin|growth))", ("ev_ebit",)),
        (r"(?:tangible\s+)?book(?:\s+value)?\b", ("pb",)),
        (_PERIOD_WORD + r"(?:fcf|free[- ]cash[- ]flows?)\b(?!\s+(?:yield|margin|growth))", ("pfcf",)),
    ))
# The words a multiple binds to stop at punctuation: in "P/E 46.8x, EV/EBITDA
# 34.4x" the EV/EBITDA after the comma belongs to the next value.
_BIND_CUT_RE = re.compile(r"[,;:()\[\]]|[.!?](?=\s|$)|\s[-–—]\s|—|\n")
_BIND_LEAD_RE = re.compile(r"^\s*(?:times\s+)?", re.I)


def _multiple_binding(after: str, left: frozenset[str]) -> frozenset[str]:
    """The metric a multiple-unit claim names directly after its "x"
    ("46.8x earnings" -> pe; "34.4x EV/EBITDA" -> ev_ebitda), or empty."""
    cut = _BIND_CUT_RE.search(after)
    seg = after[:cut.start()] if cut else after
    seg = seg[_BIND_LEAD_RE.match(seg).end():]  # type: ignore[union-attr]
    if not seg:
        return frozenset()
    for rx, toks in _MULTIPLE_NOUNS:
        if rx.match(seg):
            if toks == ("ev_ebitda",) and left & {"debt", "debt_to_ebitda"}:
                return frozenset({"debt_to_ebitda"})   # "net debt at 1.2x EBITDA"
            return frozenset(toks)
    # An explicit multiple name right after the value ("46.8x P/E").
    low = seg.lower()
    for rx, toks in _SYN_RE:
        if rx.match(low):
            return frozenset(toks) & MULTIPLE_METRICS
    return frozenset()


# Separators that end one value's phrase in a list: "46.8x P/E, 34.4x
# EV/EBITDA", "a 61% operating margin and 97.6% gross margins".
_LIST_SEP_RE = re.compile(r",|\band\b|\bwith\b|\bwhile\b|\bbut\b|\bversus\b|\bvs\b\.?|\bagainst\b", re.I)


def claim_contexts(text: str, claims: list[ParsedClaim]) -> list[ClaimContext]:
    """Anchoring context for each claim, in order: `left` (the words between
    the previous number and this one, within the clause), `right` (up to the
    next number), `wide` (±12 words of the clause)."""
    bounds = [0] + [m.end() for m in _BOUNDARY_RE.finditer(text)] + [len(text) + 1]
    out: list[ClaimContext] = []
    # Whether each claim took its metric from the words AFTER it
    # (value-then-metric prose) rather than from the words before it.
    value_first: list[bool] = []
    for i, c in enumerate(claims):
        k = bisect.bisect_right(bounds, c.start) - 1
        cs, ce = bounds[k], min(len(text), bounds[k + 1] if k + 1 < len(bounds) else len(text))
        prev_end = claims[i - 1].end if i > 0 and claims[i - 1].end > cs else cs
        next_start = claims[i + 1].start if i + 1 < len(claims) and claims[i + 1].start < ce else ce
        between = text[prev_end:c.start]
        left = context_tokens(_words_tail(between, NEAR_WORDS))
        if prev_end > cs and out and _RANGE_SEP_RE.match(between or "x"):
            # The second end of a range reads what the first end reads.
            out.append(out[-1])
            value_first.append(value_first[-1])
            continue
        # "16.4x EV/EBITDA with a 2.77% FCF yield", "46.8x P/E, 34.4x
        # EV/EBITDA": when the previous number took its metric from the
        # words after it, the words up to the first list separator are ITS
        # metric, not this one's; this number reads only what follows the
        # separator.
        if prev_end > cs and out and value_first[-1] and (left - QUALIFIERS):
            sep = _LIST_SEP_RE.search(between)
            if sep is not None:
                left = context_tokens(_words_tail(between[sep.end():], NEAR_WORDS))
            else:
                head = context_tokens(_words_head(between, 2)) - QUALIFIERS
                tail = context_tokens(_words_tail(between, 2)) - QUALIFIERS
                if head and not tail:
                    left = frozenset()
        first = not (left - QUALIFIERS)
        if c.unit == "multiple":
            bound = _multiple_binding(text[c.end:next_start], left)
            if bound:
                # The noun after the "x" is what the multiple is OF; it beats
                # anything named further left ("trades at a premium, 46.8x
                # earnings"). Qualifiers on the left (peers, bull) still apply.
                left = bound | (left & QUALIFIERS)
                first = True
        right = context_tokens(_words_head(text[c.end:next_start], NEAR_WORDS))
        wide = context_tokens(_words_tail(text[cs:c.start], WIDE_WORDS) + " "
                              + _words_head(text[c.end:ce], WIDE_WORDS))
        out.append(ClaimContext(left=left, right=right, wide=wide))
        value_first.append(first)
    return out


def fact_window_tokens(text: str, start: int, end: int) -> frozenset[str]:
    """A source fact's context: ±12 words around it (no clause cut; source
    prose is not ours to parse finely)."""
    return context_tokens(_words_tail(text[max(0, start - 400):start], WIDE_WORDS) + " "
                          + _words_head(text[end:end + 400], WIDE_WORDS))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_NO_WIDE = SCENARIOS | frozenset({"spread", "ratio"})


def effective_tokens(ctx: ClaimContext) -> frozenset[str]:
    """What the claim is about: the metric named nearest to it plus the
    qualifiers around it ("DCF base case", "vs peers").

    The metric is the one named on the left; with nothing at all on the
    left, the one on the right ("a 25% EV/EBITDA premium"); otherwise the
    clause's — unless the number is qualified by a DCF scenario or a
    relation ("base $148", "the $66 gap"), where the clause's other metrics
    ("... driven by terminal growth") are the explanation, not the figure."""
    near = ctx.left | ctx.right
    if ctx.left - QUALIFIERS:
        metrics = ctx.left - QUALIFIERS
    elif not ctx.left and ctx.right - QUALIFIERS:
        metrics = ctx.right - QUALIFIERS
    elif not (near & _NO_WIDE):
        metrics = ctx.wide - QUALIFIERS
    else:
        metrics = frozenset()
    quals = (near & QUALIFIERS) or (ctx.wide & QUALIFIERS)
    return metrics | quals


def anchored(tokens: frozenset[str], fact_ctx: frozenset[str]) -> bool:
    """Does a claim about `tokens` read a fact about `fact_ctx`?

    * different DCF scenarios never anchor (bull is not base);
    * a shared specific metric anchors (ev_ebitda, wacc, revenue ...);
    * a shared GENERIC metric anchors only when the claim names nothing more
      specific ("growth +300bp" reads a growth delta; "revenue growth" does
      not read an EPS growth);
    * a claim naming only qualifiers ("the bull vs base spread") anchors to
      a fact with the same qualifiers and no specific metric of its own."""
    ts, fs = tokens & SCENARIOS, fact_ctx & SCENARIOS
    if ts and fs and not (ts & fs):
        return False
    tm, fm = tokens - QUALIFIERS, fact_ctx - QUALIFIERS
    shared = tm & fm
    if shared - GENERIC:
        return True
    if shared:
        return not (tm - GENERIC)
    if tm:
        return False
    return bool(tokens & fact_ctx) and not (fm - GENERIC)


def _named_metrics(tokens: frozenset[str]) -> frozenset[str]:
    """What a claim names that the registry could hold: its specific metrics
    and its scenario / relation qualifiers ("bull", "spread")."""
    return tokens - GENERIC - {"dcf", "peer"}


def classify(claim: ParsedClaim, ctx: ClaimContext, registry: FactRegistry) -> tuple[str, list[Fact]]:
    """(status, supporting facts) for one claim of fact."""
    status, facts, _ = _classify(claim, ctx, registry)
    return status, facts


def _classify(claim: ParsedClaim, ctx: ClaimContext,
              registry: FactRegistry) -> tuple[str, list[Fact], list[Fact]]:
    """`classify`, plus every fact that could on its own give the claim its
    status: for a traced claim, the anchored facts AND (at 4+ significant
    digits) the non-derived verbatim ones, because either rule traces it;
    for a weak one, its unanchored matches. `classify`'s facts are the
    ones that decided it (anchored first), which is what is credited."""
    matches = registry.value_matches(claim.unit, claim.value, claim.tol)
    tokens = effective_tokens(ctx)
    hits = [f for f in matches if anchored(tokens, f.ctx)]
    # Derived facts support only anchored claims (critique delta 1).
    verbatim = [f for f in matches if not f.derived]
    verbatim_traces = bool(verbatim) and claim.sig_digits >= 4
    if hits:
        return "traced", hits, hits + (verbatim if verbatim_traces else [])
    if verbatim_traces:
        return "traced", verbatim, verbatim
    if verbatim:
        named = _named_metrics(tokens)
        if named and registry.knows_any(named):
            return "mis_anchored", [], []
        return "weak", verbatim, verbatim
    return "untraceable", [], []


@dataclass(frozen=True, slots=True)
class CheckedClaim:
    claim: ParsedClaim
    status: str
    sources: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    # EVERY registered source with a fact that would on its own give this
    # figure its status, sorted: for a traced claim the anchored facts and,
    # at 4+ significant digits, the non-derived verbatim matches too (either
    # rule traces it, so a figure anchored to a debate passage that ALSO
    # matches analyst data verbatim is not "debate-only"); for a weak one
    # the unanchored matches. `sources` is the capped, primary-first list
    # that is stored; this is uncapped, so a caller can tell a figure that
    # traces ONLY to some refs (debate-registered passages, critique #7)
    # from one that would still trace without them.
    support: tuple[str, ...] = ()


def check_text(
    text: str, registry: FactRegistry, *, threshold: bool = False,
    assumptions: Iterable[dict[str, Any]] = (),
) -> list[CheckedClaim]:
    """Every non-exempt claim in `text` with its status."""
    claims = [c for c in extract_claims(text, threshold=threshold) if c.cls != "exempt"]
    if not claims:
        return []
    ctxs = claim_contexts(text, claims)
    declared = list(assumptions)
    out: list[CheckedClaim] = []
    for c, ctx in zip(claims, ctxs):
        if c.cls == "threshold":
            out.append(CheckedClaim(c, "threshold"))
            continue
        status, facts, supporting = _classify(c, ctx, registry)
        if status in FLAGGED_STATUSES and _is_declared(c, declared):
            out.append(CheckedClaim(c, "assumption"))
            continue
        traced = status == "traced"
        sources = _credited(facts) if traced else ()
        support = tuple(sorted({f.source for f in supporting}))
        # Kinds come from EVERY supporting fact, not only the stored refs:
        # the same note block is registered once per agent (notes:comps,
        # notes:earnings ...), and truncating to eight refs must not drop the
        # transcript that also carries the figure (primary-kind credit feeds
        # the no_primary_trace / single_primary_kind caps).
        kinds = tuple(sorted({f.kind for f in facts})) if traced else ()
        out.append(CheckedClaim(c, status, sources, kinds, support))
    return out


MAX_CREDITED_REFS = 8


def _credited(facts: list[Fact]) -> tuple[str, ...]:
    """The refs stored for a traced claim: primary sources first, then the
    rest, alphabetically within each group, at most eight."""
    from .source_ledger import PRIMARY_KINDS
    primary = sorted({f.source for f in facts if f.kind in PRIMARY_KINDS})
    other = sorted({f.source for f in facts} - set(primary))
    return tuple(primary + other)[:MAX_CREDITED_REFS]


def _is_declared(c: ParsedClaim, declared: list[dict[str, Any]]) -> bool:
    for a in declared:
        raw = a.get("value")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(raw):
            continue
        v = abs(float(raw))
        unit = str(a.get("unit") or "")
        if unit and unit != c.unit:
            continue
        if abs(v - c.value) <= max(c.tol, 1e-9):
            return True
    return False


# ---------------------------------------------------------------------------
# Memo fields
# ---------------------------------------------------------------------------

POLICY_PARAGRAPH = "paragraph"
POLICY_WITHHOLD = "list_withhold"
POLICY_FLAG = "list_flag"
POLICY_THRESHOLD = "threshold"
EXPANSION_MARKER = "### Analyst expansion\n"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    path: str                 # dotted path the claim's `field` names
    text: str
    policy: str
    list_path: str = ""       # the list the item belongs to (withhold policy)
    index: int = -1
    offset_base: int = 0      # text is text_of(path)[offset_base:]


def _view_fields(prefix: str, finding: Any) -> Iterator[FieldSpec]:
    if finding is None:
        return
    yield FieldSpec(f"{prefix}.headline", finding.headline or "", POLICY_PARAGRAPH)
    yield FieldSpec(f"{prefix}.summary", finding.summary or "", POLICY_PARAGRAPH)
    for i, kp in enumerate(finding.key_points or []):
        yield FieldSpec(f"{prefix}.key_points[{i}]", str(kp), POLICY_WITHHOLD,
                        list_path=f"{prefix}.key_points", index=i)
    lf = finding.long_form_report or ""
    at = lf.find(EXPANSION_MARKER)
    if at >= 0:
        base = at + len(EXPANSION_MARKER)
        yield FieldSpec(f"{prefix}.long_form_report", lf[base:], POLICY_PARAGRAPH, offset_base=base)


def view_paths(memo: Any) -> list[tuple[str, Any]]:
    """(field prefix, finding) for every analyst view the memo shows."""
    out: list[tuple[str, Any]] = []
    for name in ("sector_agent_view", "earnings_agent_view", "filing_agent_view",
                 "valuation_agent_view", "comps_agent_view", "macro_sensitivity",
                 "technical_agent_view"):
        f = getattr(memo, name, None)
        if f is not None:
            out.append((name, f))
    for key, f in sorted((memo.extra_agent_views or {}).items()):
        out.append((f"extra_agent_views.{key}", f))
    if memo.earnings_qoq_delta is not None:
        out.append(("earnings_qoq_delta", memo.earnings_qoq_delta))
    return out


def iter_fields(memo: Any) -> Iterator[FieldSpec]:
    """Every checked memo field with its disposition (W2b §4.4)."""
    yield FieldSpec("final_pm_view", memo.final_pm_view or "", POLICY_PARAGRAPH)
    yield FieldSpec("one_sentence_thesis", memo.one_sentence_thesis or "", POLICY_PARAGRAPH)
    mt = memo.mispricing_thesis
    for k in ("consensus_view", "our_view", "gap"):
        yield FieldSpec(f"mispricing_thesis.{k}", getattr(mt, k) or "", POLICY_PARAGRAPH)
    for i, f in enumerate(mt.falsifiers or []):
        yield FieldSpec(f"mispricing_thesis.falsifiers[{i}]", str(f), POLICY_THRESHOLD)
    for side in ("bull_case", "bear_case"):
        case = getattr(memo, side)
        yield FieldSpec(f"{side}.headline", case.headline or "", POLICY_PARAGRAPH)
        for i, kp in enumerate(case.key_points or []):
            yield FieldSpec(f"{side}.key_points[{i}]", str(kp), POLICY_WITHHOLD,
                            list_path=f"{side}.key_points", index=i)
    for prefix, finding in view_paths(memo):
        yield from _view_fields(prefix, finding)
    for i, c in enumerate(memo.catalysts or []):
        # One catalyst is one list item: title and detail withhold together.
        for part in ("title", "detail"):
            yield FieldSpec(f"catalysts[{i}].{part}", getattr(c, part) or "", POLICY_WITHHOLD,
                            list_path="catalysts", index=i)
    for lname in ("key_risks", "thesis_breakers"):
        for i, r in enumerate(getattr(memo, lname) or []):
            for part in ("title", "detail"):
                yield FieldSpec(f"{lname}[{i}].{part}", getattr(r, part) or "", POLICY_FLAG,
                                list_path=lname, index=i)
    yield FieldSpec("dcf_pm_adjustment_headline", memo.dcf_pm_adjustment_headline or "", POLICY_PARAGRAPH)
    for i, adj in enumerate(memo.dcf_pm_adjustments or []):
        if isinstance(adj, dict) and adj.get("rationale"):
            yield FieldSpec(f"dcf_pm_adjustments[{i}].rationale", str(adj["rationale"]), POLICY_PARAGRAPH)
    sc = memo.scorecard
    if sc is not None and getattr(sc, "reconciliation", None):
        yield FieldSpec("scorecard.reconciliation", str(sc.reconciliation), POLICY_PARAGRAPH)
    q = memo.quality
    rec = q.rating_reconciliation if q is not None else None
    if rec is not None and rec.reason:
        yield FieldSpec("quality.rating_reconciliation.reason", rec.reason, POLICY_PARAGRAPH)
    yield from _debate_fields(memo)


def _debate_case_texts(memo: Any) -> dict[str, set[str]]:
    """Each side's case key points as the presenter joins claims to them."""
    return {side: {p.strip() for p in (getattr(memo, f"{side}_case").key_points or [])
                   if isinstance(p, str)}
            for side in ("bull", "bear")}


def _debate_fields(memo: Any) -> Iterator[FieldSpec]:
    """The debate texts a reader is shown (design-bullbear-final §12.2), only
    for a debate that is shown (complete or partial).

    Claim texts are NOT read here: a shown claim is displayed as its side's
    case key point, which `bull_case.*` / `bear_case.*` already check (and
    withhold), so reading it twice would double its weight in the ratio cap.
    Responses are FLAG, never withheld: withholding one would orphan the
    claim id the matrix points at. Only the parts of the record the
    presenter shows are read: a claim is shown when it is displayable AND
    its side's case carries its text (the presenter's join), so a response
    to, or the falsifier of, any other claim is never shown and must not
    move the untraceable caps either. A claim the withholding then removes
    from its case leaves the same way (`_drop_withheld_debate_fields`). The
    deterministic checks are computed facts and are not read."""
    from ..services.memo_sections import DEBATE_SHOWN_STATUSES, debate_claim_displayable

    debate = getattr(memo, "debate", None)
    if debate is None or debate.status not in DEBATE_SHOWN_STATUSES:
        return
    case_texts = _debate_case_texts(memo)
    shown = {i: c for i, c in enumerate(debate.claims or [])
             if debate_claim_displayable(c) and (c.claim or "").strip() in case_texts.get(c.side, set())}
    shown_ids = {c.id for c in shown.values()}
    for i, r in enumerate(debate.responses or []):
        if r.target in shown_ids:
            yield FieldSpec(f"debate.responses[{i}].argument", r.argument or "", POLICY_FLAG,
                            list_path="debate.responses", index=i)
    for side in ("bull", "bear"):
        crux = (debate.cruxes or {}).get(side)
        if isinstance(crux, str):
            yield FieldSpec(f"debate.cruxes.{side}", crux, POLICY_PARAGRAPH)
    for i, c in shown.items():
        if c.falsifier:
            yield FieldSpec(f"debate.claims[{i}].falsifier", c.falsifier, POLICY_THRESHOLD)
    yield FieldSpec("debate.resolution.crux", debate.resolution.crux or "", POLICY_PARAGRAPH)


_DEBATE_CLAIM_FIELD = re.compile(r"^debate\.(?P<list>claims|responses)\[(?P<index>\d+)\]")


def _drop_withheld_debate_fields(memo: Any, results: list[FieldResult],
                                 plan: WithholdPlan) -> list[FieldResult]:
    """`results` without the debate responses and falsifiers of the claims
    the plan withholds from their case: the presenter joins claims to the
    case as withheld, so those parts are never shown (see `_debate_fields`).

    Taken after the plan, in one pass: the dropped fields are FLAG and
    threshold texts, which never withhold anything themselves, so removing
    them cannot change which case items the plan withholds (a rebuttal
    whose words equal a case item's could only have blocked it, and that
    blocked item then stays flagged, as the plan says)."""
    debate = getattr(memo, "debate", None)
    if debate is None or not plan.items:
        return results
    withheld: dict[str, set[str]] = {}
    for side in ("bull", "bear"):
        points = getattr(memo, f"{side}_case").key_points or []
        withheld[side] = {str(points[i]).strip() for i in plan.items.get(f"{side}_case.key_points", [])
                          if i < len(points)}
    if not any(withheld.values()):
        return results
    gone_ids = {c.id for c in debate.claims if (c.claim or "").strip() in withheld.get(c.side, set())}
    kept: list[FieldResult] = []
    for r in results:
        m = _DEBATE_CLAIM_FIELD.match(r.spec.path)
        if m is not None:
            i = int(m.group("index"))
            if m.group("list") == "claims":
                target = debate.claims[i].id if i < len(debate.claims) else None
            else:
                target = debate.responses[i].target if i < len(debate.responses) else None
            if target in gone_ids:
                continue
        kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# The memo check
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    spec: FieldSpec
    claims: list[CheckedClaim] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return any(c.status in FLAGGED_STATUSES for c in self.claims)


@dataclass
class WithholdPlan:
    """List items to remove: {list_path: sorted original indexes}."""
    items: dict[str, list[int]] = field(default_factory=dict)
    lists_not_withheld: list[str] = field(default_factory=list)


@dataclass
class MemoCheck:
    fields: list[FieldResult]
    plan: WithholdPlan
    registry_facts: int
    registry_sources: int
    incomplete: tuple[str, ...]


# Fields the PM writes. A declared forecast assumption labels matching
# figures HERE only: it covers the PM's own forward numbers, never an
# analyst's key point or a risk that happens to print the same value
# (owner decision 1/9: grounding is not exempted).
PM_FIELDS = frozenset({
    "final_pm_view", "one_sentence_thesis", "mispricing_thesis.consensus_view",
    "mispricing_thesis.our_view", "mispricing_thesis.gap", "dcf_pm_adjustment_headline",
    "quality.rating_reconciliation.reason",
})
# The debate resolution is the PM's ruling (design §12.2), so the PM's own
# declared forecast assumptions apply to its figures.
_PM_FIELD_PREFIXES = ("dcf_pm_adjustments[", "debate.resolution.")


def is_pm_field(path: str) -> bool:
    return path in PM_FIELDS or path.startswith(_PM_FIELD_PREFIXES)


# Two list items carry the same text when one contains the other: the memo
# copies analyst key points into bull/bear points and catalysts, truncated
# (`title=s[:80]`, `text[:240]`) or prefixed.
_SAME_TEXT_MIN = 30


def _same_text(a: str, b: str) -> bool:
    if a == b:
        return True
    return min(len(a), len(b)) >= _SAME_TEXT_MIN and (a in b or b in a)


def check_memo(memo: Any, registry: FactRegistry, *, withhold: bool,
               assumptions: Iterable[dict[str, Any]] = ()) -> MemoCheck:
    """Check every field of `memo` and plan the withholding (not applied)."""
    declared = list(assumptions)
    results: list[FieldResult] = []
    for spec in iter_fields(memo):
        claims = check_text(spec.text, registry, threshold=spec.policy == POLICY_THRESHOLD,
                            assumptions=declared if is_pm_field(spec.path) else ())
        results.append(FieldResult(spec, claims))
    plan = WithholdPlan()
    if withhold:
        _plan_withholding(results, plan)
        results = _drop_withheld_debate_fields(memo, results, plan)
    return MemoCheck(fields=results, plan=plan, registry_facts=len(registry.facts),
                     registry_sources=len(registry.sources), incomplete=registry.incomplete_steps)


def _plan_withholding(results: list[FieldResult], plan: WithholdPlan) -> None:
    """Decide which flagged list items to remove.

    A withheld text must appear nowhere else in the memo, and the memo
    copies analyst key points into bull/bear points and catalysts. So the
    decision is taken per TEXT: a flagged item is withheld only when every
    copy of it sits in a list that may withhold it. A copy in a list the
    half-list guard keeps, or in a list that is never withheld (risks,
    thesis breakers), keeps every copy — flagged — rather than leave a
    withheld record that contradicts what the reader sees."""
    items: dict[tuple[str, int], list[str]] = {}
    flagged: dict[tuple[str, int], bool] = {}
    policy: dict[str, str] = {}
    for r in results:
        if r.spec.policy not in (POLICY_WITHHOLD, POLICY_FLAG):
            continue
        key = (r.spec.list_path, r.spec.index)
        policy[r.spec.list_path] = r.spec.policy
        if r.spec.text.strip():
            items.setdefault(key, []).append(r.spec.text.strip())
        else:
            items.setdefault(key, [])
        flagged[key] = flagged.get(key, False) or r.flagged
    bad = {k for k, f in flagged.items() if f}
    if not bad:
        return

    def copies(key: tuple[str, int]) -> set[tuple[str, int]]:
        mine = items[key]
        return {k for k, texts in items.items()
                if k == key or any(_same_text(a, b) for a in mine for b in texts)}

    # Every copy of a flagged text is flagged with it (the same words read
    # the same way; this only matters for truncated copies).
    closure = set(bad)
    for k in bad:
        closure |= copies(k)
    sizes: dict[str, int] = {}
    for lp, _ in items:
        sizes[lp] = sizes.get(lp, 0) + 1
    by_list: dict[str, list[int]] = {}
    for lp, i in closure:
        by_list.setdefault(lp, []).append(i)
    # Half-list guard: when most of a list fails, the likelier cause is a
    # registry gap, not a hallucinating agent — flag, keep all. Lists that
    # never withhold (risks) keep everything by policy.
    kept_lists = {lp for lp, idx in by_list.items()
                  if policy[lp] != POLICY_WITHHOLD or len(idx) > sizes[lp] // 2}
    kept = {k for k in closure if k[0] in kept_lists}
    blocked = set(kept)
    for k in kept:
        blocked |= copies(k)
    for lp, idx in sorted(by_list.items()):
        if policy[lp] != POLICY_WITHHOLD:
            continue
        gone = sorted(i for i in idx if (lp, i) not in blocked)
        if gone:
            plan.items[lp] = gone
        if len(gone) < len(idx):
            plan.lists_not_withheld.append(lp)


@dataclass(frozen=True, slots=True)
class FigureSupport:
    """One checked figure and the refs that support it. `start`/`end` index
    the field's text as `resolve_field` returns it (the stored offsets)."""
    field: str
    start: int
    end: int
    raw: str
    status: str
    refs: tuple[str, ...]


def figure_support(result: MemoCheck) -> list[FigureSupport]:
    """Every checked figure (thresholds included, with no refs) and ALL the
    registered sources that carry its value, in field order.

    The stored record keeps traced figures only as tallies and credited
    refs (`summarize`), which cannot answer "which figures trace only via
    X". Registering debate passages widens the ledger for the whole memo
    (critique #7), so the review flow (R1) counts the figures traced via
    debate-registered refs alone from this."""
    return [
        FigureSupport(field=fr.spec.path, start=fr.spec.offset_base + cc.claim.start,
                      end=fr.spec.offset_base + cc.claim.end, raw=cc.claim.raw, status=cc.status,
                      refs=cc.support)
        for fr in result.fields for cc in fr.claims
    ]


def traced_only_via(result: MemoCheck, refs: Iterable[str]) -> list[FigureSupport]:
    """Traced figures whose every supporting source is in `refs`: without
    those refs registered, each would not have traced (its `support` holds
    every source that could trace it by either rule, anchored or verbatim)."""
    allowed = frozenset(refs)
    return [f for f in figure_support(result)
            if f.status == "traced" and f.refs and set(f.refs) <= allowed]


def distinct_flagged(checked: Iterable[CheckedClaim]) -> int:
    """Distinct (unit, value) among flagged claims: a repeated figure is one
    defect, as the industry validator counts them."""
    return len({(c.claim.unit, round(c.claim.value, 6)) for c in checked
                if c.status in FLAGGED_STATUSES})


# ---------------------------------------------------------------------------
# Result record, withholding and offsets
# ---------------------------------------------------------------------------

MAX_STORED_CLAIMS = 100
MAX_SOURCES_CITED = 30
_STORE_ORDER = {"untraceable": 0, "mis_anchored": 1, "assumption": 2, "weak": 3}


def summarize(result: MemoCheck, *, assumptions: list[dict[str, Any]], notes: list[str]) -> NumberCheck:
    """The `NumberCheck` record for a completed check (before withholding).

    Stored claims are the ones a reader must be told about — flagged,
    declared assumptions, then weak matches — capped at 100; traced ones
    are counted, and credited in `sources_cited` / `primary_kinds_cited`."""
    from ..schemas import NumberCheck, NumberClaim
    from .source_ledger import PRIMARY_KINDS

    counts: dict[str, int] = {s: 0 for s in (*FACT_STATUSES, "assumption", "threshold")}
    stored: list[NumberClaim] = []
    cited: dict[str, None] = {}
    kinds: set[str] = set()
    every: list[CheckedClaim] = []
    for fr in result.fields:
        for cc in fr.claims:
            every.append(cc)
            counts[cc.status] = counts.get(cc.status, 0) + 1
            if cc.status == "traced":
                for src in cc.sources:
                    cited.setdefault(src, None)
                kinds.update(k for k in cc.kinds if k in PRIMARY_KINDS)
                continue
            if cc.status in _STORE_ORDER:
                c = cc.claim
                stored.append(NumberClaim(
                    field=fr.spec.path, start=fr.spec.offset_base + c.start,
                    end=fr.spec.offset_base + c.end, raw=c.raw, value=c.value, unit=c.unit,
                    status=cc.status,  # type: ignore[arg-type]
                ))
    stored.sort(key=lambda n: _STORE_ORDER.get(n.status, 9))
    counts["claims_total"] = sum(counts[s] for s in FACT_STATUSES)
    counts["flagged_distinct"] = distinct_flagged(every)
    counts["fields_checked"] = len(result.fields)
    counts["registry_facts"] = result.registry_facts
    counts["registry_sources"] = result.registry_sources
    return NumberCheck(
        checked=True, method_version=METHOD_VERSION, counts=counts,
        claims=stored[:MAX_STORED_CLAIMS],
        lists_not_withheld=list(result.plan.lists_not_withheld),
        sources_cited=list(cited)[:MAX_SOURCES_CITED],
        primary_kinds_cited=sorted(kinds),
        assumptions=[{**a, "status": "assumption"} for a in assumptions],
        notes=list(notes),
    )


_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve_field(memo: Any, path: str) -> Any:
    """The object at a claim's field path (`catalysts[2].title`,
    `extra_agent_views.industry_group.key_points[1]`), or None."""
    cur: Any = memo
    for name, idx in _PATH_TOKEN_RE.findall(path):
        if cur is None:
            return None
        if idx:
            i = int(idx)
            cur = cur[i] if isinstance(cur, list) and 0 <= i < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(name)
        else:
            cur = getattr(cur, name, None)
    return cur


def _set_list(memo: Any, list_path: str, items: list[Any]) -> None:
    parent_path, _, attr = list_path.rpartition(".")
    parent = resolve_field(memo, parent_path) if parent_path else memo
    if isinstance(parent, dict):
        parent[attr] = items
    else:
        setattr(parent, attr, items)


def _index_of(path: str, list_path: str) -> tuple[int, str] | None:
    m = re.match(re.escape(list_path) + r"\[(\d+)\](.*)$", path)
    return (int(m.group(1)), m.group(2)) if m else None


def shift_field(nc: NumberCheck | None, field_path: str, delta: int) -> None:
    """Move every stored claim on `field_path` by `delta` characters (a
    preface was written in front of the checked text)."""
    if nc is None or not delta:
        return
    nc.claims = [c.model_copy(update={"start": c.start + delta, "end": c.end + delta})
                 if c.field == field_path else c for c in nc.claims]


def drop_stale_claims(memo: Any, nc: NumberCheck | None) -> int:
    """Remove stored claims whose offsets no longer index their field's
    text (`text[start:end] == raw` is the contract a renderer relies on).
    Returns how many were dropped; the caller logs it — it should be 0."""
    if nc is None:
        return 0
    kept = []
    for c in nc.claims:
        text = resolve_field(memo, c.field)
        if isinstance(text, str) and 0 <= c.start < c.end <= len(text) and text[c.start:c.end] == c.raw:
            kept.append(c)
    dropped = len(nc.claims) - len(kept)
    nc.claims = kept
    return dropped


def apply_withholding(memo: Any, nc: NumberCheck | None, plan: WithholdPlan | None) -> list[str]:
    """Remove the planned list items from `memo`, record them verbatim in
    `nc.withheld` with their claims, renumber the claims of later items,
    rebuild each affected analyst's long-form "### Key points" block and
    strip the withheld text from the `round_findings` audit copies — so a
    withheld figure appears nowhere but `quality.number_check.withheld`.
    Returns the withheld texts."""
    from ..schemas import WithheldItem
    from .long_form import replace_key_points_block

    if nc is None or plan is None or not plan.items:
        return []
    texts: list[str] = []
    for list_path, indexes in sorted(plan.items.items()):
        items = resolve_field(memo, list_path)
        if not isinstance(items, list) or not indexes:
            continue
        gone = sorted({i for i in indexes if 0 <= i < len(items)})
        if not gone:
            continue
        mine = [(c, _index_of(c.field, list_path)) for c in nc.claims]
        for i in gone:
            item = items[i]
            parts: dict[str, int]
            if isinstance(item, str):
                text, parts = item, {"": 0}
            else:   # a catalyst: title and detail withhold together
                title = str(getattr(item, "title", "") or "")
                detail = str(getattr(item, "detail", "") or "")
                text = title if not detail or detail == title else f"{title} — {detail}"
                parts = {".title": 0} if text == title else {".title": 0, ".detail": len(title) + 3}
            claims = []
            for c, at in mine:
                if at is None or at[0] != i or at[1] not in parts:
                    continue
                off = parts[at[1]]
                claims.append(c.model_copy(update={"start": c.start + off, "end": c.end + off}))
            nc.withheld.append(WithheldItem(field=list_path, index=i, text=text, claims=claims))
            texts.append(text)
        keep_claims = []
        for c, at in mine:
            if at is None:
                keep_claims.append(c)
            elif at[0] not in gone:
                new_i = at[0] - sum(1 for g in gone if g < at[0])
                keep_claims.append(c.model_copy(update={"field": f"{list_path}[{new_i}]{at[1]}"}))
        nc.claims = keep_claims
        remaining = [x for j, x in enumerate(items) if j not in gone]
        _set_list(memo, list_path, remaining)
        # An analyst's own list: its deterministic long-form body repeats the
        # key points verbatim, before the checked "Analyst expansion".
        if list_path.endswith(".key_points"):
            finding = resolve_field(memo, list_path.rpartition(".")[0])
            report = getattr(finding, "long_form_report", None)
            if isinstance(report, str) and report:
                new = replace_key_points_block(report, remaining)
                if new != report:
                    finding.long_form_report = new
                    shift_field(nc, list_path.rpartition(".")[0] + ".long_form_report",
                                len(new) - len(report))
    if texts:
        _scrub_round_findings(memo, set(texts))
    return texts


def _scrub_round_findings(memo: Any, texts: set[str]) -> None:
    """The deep-research audit trail keeps each round's finding; a withheld
    key point must not survive there either."""
    from .long_form import replace_key_points_block
    for r in memo.round_findings or []:
        for finding in (r.findings or {}).values():
            kps = list(finding.key_points or [])
            kept = [k for k in kps if k not in texts]
            if len(kept) != len(kps):
                finding.key_points = kept
                if finding.long_form_report:
                    finding.long_form_report = replace_key_points_block(finding.long_form_report, kept)
