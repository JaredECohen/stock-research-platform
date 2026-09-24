"""MarketMosaic's own public industry labels (owner decision 2026-09-24).

The internal taxonomy keeps its codes and names — classification, the
registry, the report store and every audit row are keyed by them — but no
public surface may show GICS branding or codes (licensing). This module is
the one place that turns the internal taxonomy into what a reader sees:

* ``label(code)`` / ``slug(code)`` — our label and URL slug for a SECTOR
  (2-digit) or INDUSTRY GROUP (4-digit) code. Anything else raises
  ``UnknownLabel``: industries (6-digit) and sub-industries (8-digit) are
  never named publicly, and callers show the data provider's own industry
  string instead. One accessor contract, so no caller invents a label for a
  level that has none.
* ``code_for(slug_or_code)`` — the inverse, for routes that accept either.
* ``scrub_text(text, keep=...)`` — the ONLY scrubber in the codebase (the
  W4 design's second one was deliberately not built). It removes codes, the
  brand and the registry's multi-word names from prose while leaving
  ordinary numeric prose alone: a year (``19xx``/``20xx``), a number
  followed by ``%``, ``.d``, ``,d`` or a hyphen (``10-K``, ``50-day``), and
  a sector-sized number followed by a unit or an ordinary word (``sector
  10 years``, ``sector 25 bps``). The one year-shaped rewrite is a group
  code sitting in brackets right after its own registry name
  (``"Transportation (2030)"``): the name proves it is a code, and three
  group codes (2010/2020/2030) are also years. ``keep`` names phrases the
  caller has already established as public (the data provider's own
  industry string) so a provider string that happens to spell a registry
  name is shown as the provider wrote it.
* ``project_public(obj)`` / ``project_for_prompt(obj)`` — a JSON walker
  that applies the rules to a whole payload (keys as well as strings).

The labels live in ``data/industry_knowledge/public_labels.json`` (checked
in, keyed by internal code). The code index the scrubber matches against is
built from the bundled knowledge base — the same single taxonomy source the
registry imports from — so this module never needs a database session.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import industry_knowledge

log = logging.getLogger(__name__)

LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "industry_knowledge" / "public_labels.json"

# The display name a routed memo shows for the analyst. The telemetry key
# (`IndustryAnalyst.display_name`, "Industry Group Analyst 4530") stays
# code-based so LLMCallLog cost-by-agent keeps its continuity; this is the
# name a READER sees.
_ANALYST_NAME = "Industry Group Analyst"
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]+$")
_REQUIRED_TEXT = ("label_set_version", "public_taxonomy_prefix", "public_taxonomy_key", "attribution",
                  "mapping_caveat", "brief_attribution", "security_reference_caveat")


class LabelsMissing(RuntimeError):
    """The labels file is absent or malformed. A build defect: a silent
    fallback would put registry names back on public pages."""


class UnknownLabel(LookupError):
    """A code (or slug) with no public label: not a known sector or
    industry-group code. 6/8-digit codes always raise here by design."""


@dataclass(frozen=True)
class Labels:
    """The parsed label file."""

    version: str
    prefix: str
    public_taxonomy_key: str
    attribution: str
    mapping_caveat: str
    brief_attribution: str
    security_reference_caveat: str
    # code -> (label, slug)
    sectors: dict[str, tuple[str, str]]
    groups: dict[str, tuple[str, str]]
    by_slug: dict[str, str]

    def entry(self, code: str) -> tuple[str, str] | None:
        table = self.sectors if len(code) == 2 else self.groups if len(code) == 4 else {}
        return table.get(code)


def _parse(path: Path) -> Labels:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LabelsMissing(f"public industry labels file not found: {path.name}") from exc
    except (OSError, ValueError) as exc:
        raise LabelsMissing(f"public industry labels file unreadable: {path.name}") from exc
    if not isinstance(raw, dict):
        raise LabelsMissing("public industry labels file must be a JSON object")
    for key in _REQUIRED_TEXT:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise LabelsMissing(f"public industry labels file lacks {key!r}")
    by_slug: dict[str, str] = {}
    tables: dict[str, dict[str, tuple[str, str]]] = {}
    for level, width in (("sectors", 2), ("industry_groups", 4)):
        entries = raw.get(level)
        if not isinstance(entries, dict) or not entries:
            raise LabelsMissing(f"public industry labels file lacks {level!r}")
        table: dict[str, tuple[str, str]] = {}
        for code, entry in entries.items():
            label_text = (entry or {}).get("label") if isinstance(entry, dict) else None
            slug_text = (entry or {}).get("slug") if isinstance(entry, dict) else None
            if not (isinstance(code, str) and code.isdigit() and len(code) == width):
                raise LabelsMissing(f"{level} key {code!r} is not a {width}-digit code")
            if not (isinstance(label_text, str) and label_text.strip()):
                raise LabelsMissing(f"{level} {code} has no label")
            if not (isinstance(slug_text, str) and _SLUG_RE.match(slug_text)):
                raise LabelsMissing(f"{level} {code} has an invalid slug")
            if slug_text in by_slug:
                raise LabelsMissing(f"slug {slug_text!r} is used twice")
            by_slug[slug_text] = code
            table[code] = (label_text.strip(), slug_text)
        tables[level] = table
    for code in tables["industry_groups"]:
        if code[:2] not in tables["sectors"]:
            raise LabelsMissing(f"industry group {code} has no labelled sector")
    return Labels(
        version=raw["label_set_version"], prefix=raw["public_taxonomy_prefix"],
        public_taxonomy_key=raw["public_taxonomy_key"], attribution=raw["attribution"],
        mapping_caveat=raw["mapping_caveat"], brief_attribution=raw["brief_attribution"],
        security_reference_caveat=raw["security_reference_caveat"],
        sectors=tables["sectors"], groups=tables["industry_groups"], by_slug=by_slug,
    )


@lru_cache(maxsize=1)
def _default_labels() -> Labels:
    return _parse(LABELS_PATH)


def load(path: Path | None = None) -> Labels:
    """The label set (cached for the checked-in file). Raises ``LabelsMissing``
    when the file is absent or malformed."""
    return _default_labels() if path is None else _parse(Path(path))


# Loaded at import on purpose: the file ships in the image next to the
# knowledge base, and a deploy without it must fail loudly at start-up —
# not render registry names on the first memo that needs a label.
_LABELS = load()
PUBLIC_TAXONOMY_KEY: str = _LABELS.public_taxonomy_key
PUBLIC_ATTRIBUTION: str = _LABELS.attribution
PUBLIC_MAPPING_CAVEAT: str = _LABELS.mapping_caveat
PUBLIC_BRIEF_ATTRIBUTION: str = _LABELS.brief_attribution
PUBLIC_SECURITY_REFERENCE_CAVEAT: str = _LABELS.security_reference_caveat


# --- accessors -----------------------------------------------------------------


def _norm(code: Any) -> str:
    return str(code if code is not None else "").strip()


def _entry(code: Any) -> tuple[str, str]:
    key = _norm(code)
    found = load().entry(key) if key.isdigit() else None
    if found is None:
        raise UnknownLabel(
            f"no public label for {key!r}: labels exist for sector and industry-group codes only"
        )
    return found


def label(code: Any) -> str:
    """Our public label for a 2-digit sector or 4-digit group code."""
    return _entry(code)[0]


def slug(code: Any) -> str:
    """The public URL slug for a 2-digit sector or 4-digit group code."""
    return _entry(code)[1]


def code_for(slug_or_code: Any) -> str:
    """The internal code for a public slug; a known sector/group code passes
    through unchanged (old links and admin callers use codes). Raises
    ``UnknownLabel`` for anything else."""
    key = _norm(slug_or_code)
    labels = load()
    if key.isdigit():
        if labels.entry(key) is None:
            raise UnknownLabel(f"{key!r} is not a known sector or industry-group code")
        return key
    code = labels.by_slug.get(key.lower())
    if code is None:
        raise UnknownLabel(f"{key!r} is not a known industry slug")
    return code


def public_version_key(version_key: Any) -> str:
    """``gics-2026-04`` → ``mm-2026-04``. An internal key without the
    ``gics-`` prefix maps to ``mm-<sha8>`` so it is still deterministic and
    still says nothing about the internal naming."""
    key = _norm(version_key)
    prefix = load().prefix
    if key.startswith(prefix):
        return key
    if key.startswith("gics-") and len(key) > len("gics-"):
        return prefix + key[len("gics-"):]
    return prefix + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def public_display_name(code: Any) -> str:
    """``"Industry Group Analyst (Chips & Chipmaking Equipment)"``."""
    return f"{_ANALYST_NAME} ({label(code)})"


# --- the code index (from the bundled knowledge base) ----------------------------


@dataclass(frozen=True)
class _Index:
    names: dict[str, str]            # every known code (2/4/6/8, retired too) -> registry name
    public_names: dict[str, str]     # multi-word registry names with & or , -> label (see `rollup`)
    public_names_re: re.Pattern[str] | None
    exact: tuple[tuple[str, str], ...]
    rollup: bool                     # does `public_names` also roll industry names up to groups?


def _exact_table(labels: Labels) -> tuple[tuple[str, str], ...]:
    """Fixed sentences that carry the brand, mapped to their public
    equivalents. Longest first, so a sentence that contains another is
    replaced whole rather than half-rewritten."""
    # Imported here: the registry pulls the ORM in, and it imports this
    # module lazily from `display()`; a top-level import would be circular.
    from . import gics_registry

    pairs = {
        gics_registry.ATTRIBUTION: labels.attribution,
        gics_registry.MAPPING_CAVEAT: labels.mapping_caveat,
        industry_knowledge.BRIEF_ATTRIBUTION: labels.brief_attribution,
        industry_knowledge.SECURITY_REFERENCE_CAVEAT: labels.security_reference_caveat,
        # routes_industries._SOURCE_LABELS and the report writer's
        # constituents template; both are stored in editions/responses.
        "research map (economic research examples; not licensed issuer GICS mapping)":
            "research map (economic research examples; not an issuer classification)",
        "(mapping derived from provider classification, not licensed GICS security assignments)":
            f"({labels.mapping_caveat})",
    }
    return tuple(sorted(pairs.items(), key=lambda kv: -len(kv[0])))


@lru_cache(maxsize=2)
def _index(rollup: bool = False) -> _Index:
    labels = load()
    payload = industry_knowledge.load_industry_knowledge()
    names: dict[str, str] = {}
    public_names: dict[str, str] = {}
    for sector in payload.get("sectors") or []:
        names[str(sector["code"])] = str(sector["name"])
        for group in sector.get("industry_groups") or []:
            names[str(group["code"])] = str(group["name"])
            for industry in group.get("industries") or []:
                names[str(industry["code"])] = str(industry["name"])
                for sub in industry.get("sub_industries") or []:
                    names[str(sub["code"])] = str(sub["name"])
    for retired in payload.get("retired_sub_industries") or []:
        names.setdefault(str(retired.get("code", "")), str(retired.get("name", "")))
        if retired.get("parent_code"):
            names.setdefault(str(retired["parent_code"]), str(retired.get("parent_name", "")))
    names.pop("", None)
    for code, name in names.items():
        entry = labels.entry(code)
        # Only multi-word names carrying "&" or "," are unambiguous enough to
        # replace in free prose: "Banks", "Energy" or "Real Estate" are
        # ordinary English and rewriting them would mangle sentences.
        if entry is not None and ("&" in name or "," in name):
            public_names[name] = entry[0]
    # The ROLLUP (`rollup=True` only): a distinctive industry / sub-industry
    # name ("Semiconductor Materials & Equipment") reads as the label of the
    # group it rolls up to — what `gics_registry.display()` shows for those
    # levels. It is the net for LEGACY industry-report prose (new editions
    # are rejected by the validator's L1 rule instead), so only the industry
    # surfaces ask for it: the read routes, the PM's report excerpts and the
    # report writer's facts. It is NOT the default, because in company-memo
    # prose the same strings are ordinary industry words — and often a
    # provider's industry string ("Aerospace & Defense") — and rewriting
    # them into a broader group label changes what a sentence says ("Unlike
    # Oil & Gas Exploration & Production peers, XOM refines" would name
    # XOM's own group as the contrast). A name two groups share is ambiguous
    # and left for L1; a sector/group name keeps its own mapping above.
    if rollup:
        targets_of: dict[str, set[str]] = {}
        for code, name in names.items():
            if len(code) in (6, 8) and ("&" in name or "," in name) and name not in public_names:
                group = labels.entry(code[:4])
                if group is not None:
                    targets_of.setdefault(name, set()).add(group[0])
        for name, targets in targets_of.items():
            if len(targets) == 1:
                public_names[name] = next(iter(targets))
    pattern = None
    if public_names:
        alternation = "|".join(re.escape(n) for n in sorted(public_names, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\w&])(?:{alternation})(?![\w])")
    return _Index(names=names, public_names=public_names, public_names_re=pattern,
                  exact=_exact_table(labels), rollup=rollup)


def clear_cache() -> None:
    """Drop the cached label file and code index (tests that swap the file).
    The module constants keep the values read at import."""
    _default_labels.cache_clear()
    _index.cache_clear()
    _phrase_index.cache_clear()


def registry_names() -> dict[str, str]:
    """Every known taxonomy code (2/4/6/8-digit, retired included) → its
    registry name. A copy: the report validator's L1 rule and the
    public-surface tests read it, and neither may edit the cached index."""
    return dict(_index().names)


@lru_cache(maxsize=1)
def _phrase_index() -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    """(the industry/sub-industry registry-phrase pattern, our labels) —
    the phrase set the report validator's L1 rule rejects in new prose.

    A phrase is the multi-word registry name of a 6/8-digit node that is
    not also a sector/group name; single words ("Software", "Restaurants")
    are ordinary English. Case-sensitive: "Office REITs" is the taxonomy's
    name, "office REITs" is a description."""
    names = _index().names
    upper = {n for c, n in names.items() if len(c) in (2, 4)}
    phrases = sorted({n for c, n in names.items()
                      if len(c) in (6, 8) and n not in upper and len(n.split()) > 1}, key=len, reverse=True)
    phrase_re = (re.compile(r"(?<![\w&])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![\w])")
                 if phrases else None)
    labels = load()
    ours = sorted({lab for lab, _slug in (*labels.sectors.values(), *labels.groups.values())},
                  key=len, reverse=True)
    ours_re = (re.compile(r"(?<![\w&])(?:" + "|".join(re.escape(lab) for lab in ours) + r")(?![\w])")
               if ours else None)
    return phrase_re, ours_re


def registry_phrase_hits(text: Any) -> list[re.Match[str]]:
    """Every industry/sub-industry registry phrase in `text` that is not
    part of one of OUR labels ("Software & IT Services" is our label and
    contains the registry name "IT Services"; a label is what prose is
    asked to say, so it is never the leak).

    Matched on the raw text, and a hit is discarded only when it lies
    wholly inside a label occurrence. Blanking the labels out first (the
    earlier approach) also blanked the label WORDS inside longer registry
    names — "Health Care Technology" lost "Technology", "Technology
    Distributors" became "Distributors" — so those names were never seen."""
    if not isinstance(text, str) or not text:
        return []
    phrase_re, ours_re = _phrase_index()
    if phrase_re is None:
        return []
    spans = [m.span() for m in ours_re.finditer(text)] if ours_re is not None else []
    return [m for m in phrase_re.finditer(text)
            if not any(a <= m.start() and m.end() <= b for a, b in spans)]


def _plain_word(word: str) -> str:
    """"Office" → "office"; an acronym ("IT", "REITs") keeps its capitals."""
    return word if sum(ch.isupper() for ch in word) > 1 else word.lower()


def plain_registry_phrases(text: str) -> str:
    """Write every registry phrase `registry_phrase_hits` finds as an
    ordinary lower-case description ("Office REITs lease space" → "office
    REITs lease space"), for text a model is PROMPTED with.

    The mandate's own prose names sub-industries ("Office REITs",
    "Industrial REITs" in the Property REITs mandate); a model that reads
    the capitalised name repeats it, and L1 then rejects the edition. The
    description keeps its meaning — unlike rolling it up to the group
    label, which would turn four different REIT types into one — and is
    not the taxonomy's name. Length-preserving (ASCII case change only), so
    a budget measured on the text still holds after it."""
    hits = registry_phrase_hits(text)
    if not hits:
        return text
    out: list[str] = []
    last = 0
    for m in hits:
        out.append(text[last:m.start()])
        out.append(re.sub(r"[A-Za-z]+", lambda w: _plain_word(w.group(0)), m.group(0)))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# --- scrub_text ----------------------------------------------------------------

_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
# A digit run that is a whole token: not glued to letters/digits, not the
# tail of a decimal or thousands group, not the head of one, not a
# percentage. This is what keeps "20% share", "453,010" and "2030.5" safe.
_TOKEN_GUARD_BEFORE = r"(?<![\w$.,])"
# A hyphen/dash glued to a following letter or digit makes the number part
# of a compound ("10-K", "50-day", "10-year") — ordinary prose, not a code.
_TOKEN_GUARD_AFTER = r"(?![\w%]|[.,]\d|[-\u2010\u2011\u2013]\w)"
_DIGITS_RE = re.compile(_TOKEN_GUARD_BEFORE + r"(\d{2}|\d{4}|\d{6}|\d{8})" + _TOKEN_GUARD_AFTER)
# `(\s?)`, not `(\s*)`: an unanchored `\s*` rescans the rest of a whitespace
# run from every position in it (quadratic on a long run of model output);
# the one space it swallows is all the removal needs, and the multi-space
# tidy below handles the rest.
_BRACKET_RE = re.compile(
    r"(\s?)([\[(])\s*(\d{2,8}(?:\s*[,;/]\s*\d{2,8})*)\s*([\])])"
)
_ANALYST_RE = re.compile(r"\bIndustry Group Analyst (\d{4})" + _TOKEN_GUARD_AFTER)
_PREFIX_RE = re.compile(
    r"\b(sector|industry group|group|industry)(\s+)(\d{2}|\d{4})" + _TOKEN_GUARD_AFTER, re.IGNORECASE,
)
_LONG_CODE_RE = re.compile(r"(\s?)" + _TOKEN_GUARD_BEFORE + r"(\d{6}|\d{8})" + _TOKEN_GUARD_AFTER)
_INTERNAL_KEY_RE = re.compile(r"\bgics-\d{4}-\d{2}\b", re.IGNORECASE)
_BRAND_BEFORE_WORD_RE = re.compile(r"\bGICS\b®?\s+(?=[A-Za-z])", re.IGNORECASE)
# Letters only on either side, not `\b`: an underscore or a digit is a word
# character, so `\bGICS\b` let the brand through inside an identifier
# ("gics_industries_2026.json", "import_gics_taxonomy") — a string the
# public surfaces did carry, in the taxonomy source and the 503 remedy.
_BRAND_RE = re.compile(r"(?<![A-Za-z])GICS(?![A-Za-z])®?", re.IGNORECASE)


def has_brand(text: Any) -> bool:
    """Does `text` name the brand? Letters on neither side, so "Biologics"
    (and any other word that happens to contain the four letters) is not
    the brand: a substring test dropped a "biologics" primary source and
    flagged a "Biologics production" dependency edge."""
    return isinstance(text, str) and _BRAND_RE.search(text) is not None
# A code in a basis or source reference: `mandate:4530`, `mapping:4530`,
# `industry:453010`. The namespace is an allowlist because an arbitrary
# `word:NN` is as often a count or a ratio as a code ("limit:10", "year:2020");
# a taxonomy namespace proves the number is a code, including the three
# year-shaped group codes (2010/2020/2030).
_REF_NAMESPACES = ("mandate", "mapping", "industry", "industry_group", "group", "industry_knowledge",
                   "sector", "gics", "taxonomy", "code")
_REF_CODE_RE = re.compile(
    r"(?<![\w:])(" + "|".join(_REF_NAMESPACES) + r"):(\d{8}|\d{6}|\d{4}|\d{2})" + _TOKEN_GUARD_AFTER,
    re.IGNORECASE,
)
# The brand followed directly by a code or a list of codes ("GICS 4010",
# "GICS® 2030", "GICS 4510, 4520", "GICS 4510/451020"). Longest width
# first, so a 4-digit code is never read as a 2-digit one plus a remainder.
_CODE_TOKEN = r"(?:\d{8}|\d{6}|\d{4}|\d{2})"
_BRAND_CODES_RE = re.compile(
    r"\b(GICS\b®?)(\s+)(" + _CODE_TOKEN + r"(?:\s*[,;/]\s*" + _CODE_TOKEN + r")*)" + _TOKEN_GUARD_AFTER,
    re.IGNORECASE,
)
_CODE_LIST_SPLIT_RE = re.compile(r"(\s*[,;/]\s*)")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
# The lookbehind pins a match to the START of a space run, so a long run not
# followed by punctuation is scanned once rather than once per position.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<![ \t])[ \t]+([,.;:)\]])")
# What follows a prefixed code decides whether it is a code. A unit or time
# word makes the number a quantity ("group 2550 units", "sector 25 bps");
# a 2-digit sector code must additionally be followed by the end of the
# text, punctuation or a taxonomy noun, because "sector 10 names" or "the
# sector 20 largest" read as counts far more often than as codes.
_NEXT_WORD_RE = re.compile(r"[ \t]*([A-Za-z]+|[^\sA-Za-z0-9]|$)")
_UNIT_WORDS = frozenset({
    "day", "days", "week", "weeks", "month", "months", "year", "years", "yr", "yrs",
    "quarter", "quarters", "qtr", "qtrs", "hour", "hours", "minute", "minutes",
    "bp", "bps", "basis", "percent", "pct", "percentage", "points", "pts", "pp",
    "x", "times", "k", "q", "m", "mm", "b", "bn", "million", "billion", "trillion",
    "thousand", "units", "dollars", "cents", "usd",
})
_SECTOR_CODE_NOUNS = frozenset({
    "constituents", "companies", "stocks", "equities", "peers", "members", "index",
    "indices", "benchmark", "analyst", "analysts", "coverage", "universe", "exposure",
    "weighting", "weight", "cohort", "basket", "classification", "code",
})
# After the brand, one number followed by one of these counts taxonomy
# levels ("the GICS 11 sectors"; "GICS 10 sectors" before 2016) rather than
# naming one. "sub" is what _NEXT_WORD_RE reads from "sub-industries".
_TAXONOMY_COUNT_NOUNS = frozenset({
    "sectors", "groups", "industries", "sub", "subindustries", "levels", "tiers", "codes",
})
# A year after the brand that names a revision of the standard ("the GICS
# 2020 changes") is a year, even where it is also a group code.
_REVISION_NOUNS = frozenset({
    "revision", "revisions", "reclassification", "reclassifications", "restructuring",
    "change", "changes", "update", "updates", "methodology", "structure", "edition",
    "version", "review", "reshuffle", "overhaul",
})


def _is_year(token: str) -> bool:
    return bool(_YEAR_RE.match(token))


def _bracket_sub(m: re.Match[str], names: dict[str, str]) -> str:
    opener, closer = m.group(2), m.group(4)
    if (opener, closer) not in (("[", "]"), ("(", ")")):
        return m.group(0)
    tokens = re.split(r"\s*[,;/]\s*", m.group(3))
    for tok in tokens:
        if tok not in names or len(tok) not in (2, 4, 6, 8) or (len(tok) == 4 and _is_year(tok)):
            return m.group(0)
        # A bare 2-digit number in brackets is far more often a count
        # ("(45)") or a note reference ("[10]") than a sector code. W1 rule
        # (b) removes 6/8-digit provenance lists; 4-digit non-year group
        # codes are unambiguous enough to join them. A 2-digit code is only
        # trusted next to its own sector name (`_pair_subs`).
        if len(tok) == 2:
            return m.group(0)
    return ""


def _pair_subs(text: str, idx: _Index, labels: Labels, kept: frozenset[str] = frozenset()) -> str:
    """``"{name} ({code})"`` / ``"{code} {name}"`` → our label for a
    sector/group, or nothing for an industry/sub-industry. Only the codes
    actually present in the text are tried, so the common case costs one
    regex scan. The registry name next to the code proves it is a code, so
    this is the one rule that may rewrite a year-shaped group code
    ("Transportation (2030)")."""
    for code in dict.fromkeys(m.group(1) for m in _DIGITS_RE.finditer(text)):
        name = idx.names.get(code)
        if not name:
            continue
        entry = labels.entry(code)
        # A kept name (the provider's own string) stays; only its code goes.
        replacement = name if name in kept else entry[0] if entry is not None else ""
        # A 2-digit number in parentheses after a sector name is usually a
        # count ("Energy (10), Materials (15) names"); only the square-
        # bracket provenance form is trusted at that width.
        forms = [f"{name} [{code}]"] if len(code) == 2 else [f"{name} ({code})", f"{name} [{code}]"]
        # "code name" is only trusted where the number cannot be a count or
        # a year: "top 10 Energy names" and "in 2010 Capital Goods ..." are
        # prose, "453010 Semiconductors & ..." is not.
        if len(code) >= 4 and not _is_year(code):
            forms.append(f"{code} {name}")
        for form in forms:
            if form in text:
                text = text.replace(form, replacement)
    return text


def _keepable(phrase: Any) -> bool:
    """A phrase a caller may exempt from scrubbing: non-blank text with no
    digit and no brand — a provider industry string, never a code."""
    return (isinstance(phrase, str) and bool(phrase.strip())
            and not any(ch.isdigit() for ch in phrase) and not has_brand(phrase))


def _keep_set(keep: Any) -> frozenset[str]:
    if isinstance(keep, str):
        keep = (keep,)
    return frozenset(p.strip() for p in (keep or ()) if _keepable(p))


def scrub_text(text: str, *, keep: Any = (), rollup: bool = False) -> str:
    """Remove taxonomy codes, the brand and registry group names from
    prose. A registry name EQUAL to a ``keep`` phrase is left as written
    (the caller's already-public provider industry string: "Household &
    Personal Products" is FMP's industry for PG and ALSO a registry group
    name, and rewriting it would report our label as the provider's); a
    longer registry name that merely contains one ("Semiconductors &
    Semiconductor Equipment" vs "Semiconductors") is still rewritten, and a
    code next to a kept name is still removed. Keep phrases carry no digit
    and no brand, so no other rule can touch them.
    Deterministic and idempotent; order matters:

    a. fixed branded sentences → their public equivalents;
    b. bracketed code lists (``[453010]``, ``[451020,451030]``, ``(4530)``)
       → removed with the space before them;
    c. ``name (code)`` / ``code name`` pairs → our label (sector/group) or
       nothing (industry/sub-industry); ``Industry Group Analyst dddd`` →
       the public display name; ``sector 45`` / ``group 4530`` → ``sector
       Technology`` / ``group Chips & Chipmaking Equipment``; the brand
       as a code's own prefix (``GICS 4010``, ``GICS® 2030``, ``GICS 4510,
       4520``) → ``industry <label>`` per 2/4-digit code, 6/8-digit codes
       in the run removed;
    d. standalone known 6/8-digit codes → removed. A bare 4-digit token is
       never touched (2020 and 2030 are years as well as group codes);
    e. multi-word registry sector/group names containing ``&`` or ``,`` →
       our label (case-sensitive); with ``rollup=True`` (industry-report
       surfaces only — see ``_index``) a distinctive industry/sub-industry
       name → the label of its group as well;
    f. the internal taxonomy key → the public key; any remaining "GICS" is
       dropped before a noun ("the GICS sector" → "the sector") and
       otherwise read as "industry".

    Years (``19xx``/``20xx``) and numbers followed by ``%``, ``.d``, ``,d``
    or a hyphenated word are never altered by the contextual rules (b,
    c-prefix, d); a prefixed code followed by a unit word is a quantity,
    and a 2-digit one must be followed by punctuation, the end of the text
    or a taxonomy noun. A year-shaped number is rewritten only where
    something next to it proves it is a code: the registry name (rule c's
    pair form) or the brand (rule c's brand form, unless a revision noun
    follows: "the GICS 2020 changes").
    """
    if not isinstance(text, str) or not text:
        return text
    kept = _keep_set(keep)
    labels = load()
    idx = _index(rollup)
    out = text
    for old, new in idx.exact:
        if old in out:
            out = out.replace(old, new)
    if any(ch.isdigit() for ch in out):
        # References before everything else: the standalone-code rule would
        # strip the 6-digit half of "industry:453010" and leave a dangling
        # "industry:".
        out = _REF_CODE_RE.sub(lambda m: _ref_sub(m, idx, labels), out)
        # Pairs first: a bracket rule that ran earlier would strip the
        # "(4010)" off "Banks (4010)" and leave the registry name behind.
        out = _pair_subs(out, idx, labels, kept)
        out = _ANALYST_RE.sub(
            lambda m: public_display_name(m.group(1)) if labels.entry(m.group(1)) else m.group(0), out,
        )
        if "gics" in out.lower():
            # Before the bracket, prefix and long-code rules and before rule
            # f: the brand is the one evidence that "2030" in "GICS 2030" is
            # a code and not a year. Once rule f rewrites it to "industry",
            # or rule d strips a 6-digit member out of a list, that evidence
            # is gone and the code stays next to our own noun.
            out = _BRAND_CODES_RE.sub(lambda m: _brand_codes_sub(m, idx, labels), out)
        out = _BRACKET_RE.sub(lambda m: _bracket_sub(m, idx.names), out)
        out = _PREFIX_RE.sub(lambda m: _prefix_sub(m, labels), out)
        out = _LONG_CODE_RE.sub(lambda m: "" if m.group(2) in idx.names else m.group(0), out)
    if idx.public_names_re is not None:
        out = idx.public_names_re.sub(
            lambda m: m.group(0) if m.group(0) in kept else idx.public_names[m.group(0)], out,
        )
    out = _INTERNAL_KEY_RE.sub(lambda m: public_version_key(m.group(0).lower()), out)
    if "gics" in out.lower():
        out = _BRAND_BEFORE_WORD_RE.sub("", out)
        out = _BRAND_RE.sub("industry", out)
    if out != text:
        # Only a string this function changed is tidied, so a caller's own
        # spacing survives untouched text.
        out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _MULTISPACE_RE.sub(" ", out))
    return out


def _brand_codes_sub(m: re.Match[str], idx: _Index, labels: Labels) -> str:
    """``GICS <code>[, <code>...]`` → ``industry <label>[, <label>...]``.

    The brand proves the number is a code, so neither the year guard nor
    the 2-digit noun guard of the bare prefix form applies ("GICS 2030
    rerates", "GICS 45 names"). A single number followed by a unit word, a
    taxonomy count ("GICS 11 sectors") or, when year-shaped, a revision
    noun ("GICS 2020 changes") is left for rule f. Codes are rewritten left
    to right; from the first token that is not a known code on, the run
    stays as written, since a list that stops being codes is not a code
    list."""
    brand, gap, run = m.group(1), m.group(2), m.group(3)
    parts = _CODE_LIST_SPLIT_RE.split(run)
    tokens, seps = parts[0::2], parts[1::2]
    if len(tokens) == 1:
        nxt = _NEXT_WORD_RE.match(m.string, m.end())
        word = (nxt.group(1) if nxt else "").lower()
        if word in _UNIT_WORDS or word in _TAXONOMY_COUNT_NOUNS:
            return m.group(0)
        if _is_year(run) and word in _REVISION_NOUNS:
            return m.group(0)
    rewritten: list[str] = []
    rest = ""
    for i, tok in enumerate(tokens):
        entry = labels.entry(tok)
        if tok not in idx.names or (len(tok) <= 4 and entry is None):
            # The unrecognised remainder, with the separator before it.
            rest = "".join(parts[2 * i - 1:]) if i else run
            break
        if entry is not None:
            rewritten.append(entry[0])
        # A 6/8-digit code has no entry and is removed outright (rule d).
    if not rewritten:
        # Nothing to label: the brand is left for rule f and the remainder
        # as written ("GICS 45301020 names" → "GICS names" → "names").
        return f"{brand}{gap}{rest}" if rest else brand
    # Between the surviving labels, the list's own first separator.
    joiner = seps[0] if seps else ", "
    return f"industry{gap}{joiner.join(rewritten)}{rest}"


def _ref_sub(m: re.Match[str], idx: _Index, labels: Labels) -> str:
    """``mandate:4530`` → ``mandate:<group slug>``; a 6/8-digit code rolls
    up to its group's slug (the only level with a public name). An unknown
    number is left alone — it is not a code this taxonomy has."""
    namespace, code = m.group(1), m.group(2)
    if code not in idx.names:
        return m.group(0)
    entry = labels.entry(code if len(code) <= 4 else code[:4])
    return f"{namespace}:{entry[1]}" if entry is not None else m.group(0)


def _prefix_sub(m: re.Match[str], labels: Labels) -> str:
    noun, gap, code = m.group(1), m.group(2), m.group(3)
    want = 2 if noun.lower() == "sector" else 4
    if len(code) != want or (len(code) == 4 and _is_year(code)):
        return m.group(0)
    nxt = _NEXT_WORD_RE.match(m.string, m.end())
    word = (nxt.group(1) if nxt else "").lower()
    if word in _UNIT_WORDS:
        return m.group(0)
    if want == 2 and word.isalpha() and word not in _SECTOR_CODE_NOUNS:
        return m.group(0)
    entry = labels.entry(code)
    return f"{noun}{gap}{entry[0]}" if entry is not None else m.group(0)


def scrub_strings(obj: Any, *, keep: Any = ()) -> Any:
    """``scrub_text`` over every string inside a nested dict/list, keys and
    structure untouched — for LLM output whose SHAPE is a contract (the
    sector analyst's ``bull_bear_analysis``)."""
    if isinstance(obj, str):
        return scrub_text(obj, keep=keep)
    if isinstance(obj, dict):
        return {k: scrub_strings(v, keep=keep) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_strings(v, keep=keep) for v in obj]
    return obj


# --- project_public ------------------------------------------------------------

# Single-code keys and the name key that travels with each.
_CODE_KEYS: dict[str, str] = {
    "code": "name",
    "sector_code": "sector_name",
    "industry_group_code": "industry_group_name",
    "group_code": "group_name",
    "parent_code": "parent_name",
}
# Industry / sub-industry identities are never public, whatever their value.
_DROP_KEYS = frozenset({"industry_code", "sub_industry_code", "industry_name", "sub_industry_name"})
_CONSTANT_KEYS = ("attribution", "mapping_caveat", "security_reference_caveat")
_VERSION_KEYS = frozenset({"taxonomy_version", "key", "knowledge_version", "version_key"})
_SOURCE_LIST_KEYS = frozenset({"primary_sources", "map_sources"})
_BRANDED_SOURCE_RE = re.compile(r"(?<![A-Za-z])gics(?![A-Za-z])|msci\.com|spglobal\.com", re.IGNORECASE)
_PUBLIC_REF = "MarketMosaic industry research knowledge base"
# Values that are public by design and shown exactly as the source wrote
# them (W1 §4.5: the provider's industry string stands in for the
# sub-industry). Scrubbing one would rewrite a provider string that spells a
# registry name into OUR label, and the card would misreport the provider.
_PASSTHROUGH_KEYS = frozenset({"provider_industry"})


# Lists that name groups by bare code under a key that does not end in
# `_codes`: the cross-industry snapshot's `insufficient_sample_groups` is a
# list of code strings, and `missing_groups` is one in older snapshots (a
# list of `{code, name, reason}` in newer ones). A bare "4530" string is
# never touched by `scrub_text` (four digits can be a year), so these keys
# have to be named.
_GROUP_LIST_KEYS = frozenset({"insufficient_sample_groups", "missing_groups"})


def _code_list_slugs(values: list[Any], idx: _Index, labels: Labels, keep: Any) -> list[Any]:
    """Each known code → the slug of its group (6/8-digit codes roll up to
    their 4-digit prefix); unknown values are kept (strings scrubbed, other
    values projected); duplicates collapse."""
    out: list[Any] = []
    for v in values:
        key = _norm(v)
        if isinstance(v, str) and key.isdigit() and key in idx.names:
            entry = labels.entry(key if len(key) <= 4 else key[:4])
            if entry is None:
                continue
            v = entry[1]
        else:
            v = _project(v, idx, labels, keep)
        if v not in out:
            out.append(v)
    return out


def _constant_for(key: str, value: Any, idx: _Index, labels: Labels, keep: Any) -> Any:
    default = {"attribution": labels.attribution, "mapping_caveat": labels.mapping_caveat,
               "security_reference_caveat": labels.security_reference_caveat}[key]
    if not isinstance(value, str):
        return default
    # A known branded sentence maps to its own public twin (the brief
    # attribution is not the taxonomy attribution); any other text that
    # names the brand gets the key's public constant, not a half-scrub.
    exact = dict(idx.exact)
    if value in exact:
        return exact[value]
    return default if has_brand(value) else scrub_text(value, keep=keep, rollup=idx.rollup)


def _project_dict(obj: dict[Any, Any], idx: _Index, labels: Labels, keep: Any) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    drop: set[Any] = set()
    renamed: dict[Any, str] = {}
    for code_key, name_key in _CODE_KEYS.items():
        value = obj.get(code_key)
        key = _norm(value)
        if not (isinstance(value, str) and key.isdigit() and key in idx.names):
            continue
        entry = labels.entry(key)
        if entry is not None:
            renamed[code_key] = entry[1]
            if name_key in obj:
                renamed[name_key] = entry[0]
        else:
            drop.update((code_key, name_key))
    for k, v in obj.items():
        if k in drop or k in _DROP_KEYS:
            continue
        if isinstance(k, str) and k.isdigit() and k in idx.names:
            entry = labels.entry(k)
            if entry is None:
                continue          # a 6/8-digit keyed entry is dropped outright
            k = entry[1]
        if k in renamed:
            out[k] = renamed[k]
        elif k in _PASSTHROUGH_KEYS and (v is None or _keepable(v)):
            out[k] = v
        elif k in _CONSTANT_KEYS:
            out[k] = _constant_for(k, v, idx, labels, keep)
        elif k == "provenance" and isinstance(v, dict):
            out[k] = {}
        elif k in _VERSION_KEYS and isinstance(v, str) and v.lower().startswith("gics-"):
            out[k] = public_version_key(v.lower())
        elif isinstance(k, str) and (k == "codes" or k.endswith("_codes") or k in _GROUP_LIST_KEYS) \
                and isinstance(v, list):
            out[k] = _code_list_slugs(v, idx, labels, keep)
        elif k in _SOURCE_LIST_KEYS and isinstance(v, list):
            kept = [s for s in v if not _BRANDED_SOURCE_RE.search(json.dumps(s, default=str))]
            out[k] = [_project(s, idx, labels, keep) for s in kept]
            if len(kept) != len(v):
                out[f"{k}_withheld"] = len(v) - len(kept)
        elif k == "ref" and isinstance(v, str) and has_brand(v):
            out[k] = _PUBLIC_REF
        elif isinstance(v, list):
            # A list entry that was nothing BUT a non-public identity (an
            # overview `boundaries` row is `{code, name}` of an industry)
            # projects to `{}`; a page would render a row of nothing. It is
            # dropped and counted, the same way a withheld source is.
            projected = [_project(item, idx, labels, keep) for item in v]
            emptied = {i for i, (a, b) in enumerate(zip(v, projected))
                       if isinstance(a, dict) and a and b == {}}
            out[k] = [p for i, p in enumerate(projected) if i not in emptied]
            if emptied:
                out[f"{k}_withheld"] = len(emptied)
        else:
            out[k] = _project(v, idx, labels, keep)
    return out


def _project(obj: Any, idx: _Index, labels: Labels, keep: Any = ()) -> Any:
    if isinstance(obj, str):
        return scrub_text(obj, keep=keep, rollup=idx.rollup)
    if isinstance(obj, dict):
        return _project_dict(obj, idx, labels, keep)
    if isinstance(obj, (list, tuple)):
        return [_project(v, idx, labels, keep) for v in obj]
    return copy.deepcopy(obj)


def project_public(obj: Any, *, keep: Any = (), rollup: bool = False) -> Any:
    """A public copy of a JSON-shaped payload (the input is never mutated):

    1. ``code`` / ``sector_code`` / ``industry_group_code`` / ``group_code``
       / ``parent_code`` holding a known 2/4-digit code → its slug, and the
       sibling ``*name`` → our label; holding a 6/8-digit code → both keys
       removed.
    2. ``industry_code`` / ``sub_industry_code`` (and their names) → removed;
       ``codes`` / ``*_codes`` lists, and the snapshot's
       ``insufficient_sample_groups`` / ``missing_groups`` → group slugs for
       their bare-code entries, de-duplicated.
    3. dict keys equal to a known 2/4-digit code → slug; 6/8-digit keys →
       entry dropped.
    4. ``attribution`` / ``mapping_caveat`` / ``security_reference_caveat``
       → the public constants; ``provenance`` → ``{}``; internal taxonomy
       keys → the public key.
    5. ``primary_sources`` / ``map_sources`` entries naming the brand or its
       publishers → removed, counted in ``<key>_withheld``; a branded
       ``ref`` → the knowledge-base reference.
    6. ``provider_industry`` → kept verbatim (public by design).
    7. every remaining string → ``scrub_text`` (with ``keep`` and
       ``rollup``). ``rollup=True`` is for industry-report surfaces only
       (the read routes, the PM's report excerpts): see ``_index``.
    """
    return _project(obj, _index(rollup), load(), keep)


def project_for_prompt(obj: Any, *, keep: Any = ()) -> Any:
    """The same projection, applied to facts sent to a model: a prompt that
    never contains a code or registry name cannot be echoed into prose.
    A separate name so a prompt-only rule can diverge without moving what
    API responses carry. Its only caller is the industry report writer,
    whose prose L1 gates, so the industry rollup applies."""
    return _project(obj, _index(rollup=True), load(), keep)
