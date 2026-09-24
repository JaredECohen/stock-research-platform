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
* ``scrub_text(text)`` — the ONLY scrubber in the codebase (the W4 design's
  second one was deliberately not built). It removes codes, the brand and
  the registry's multi-word names from prose while never altering a year
  (``19xx``/``20xx``) or a number followed by ``%``, ``.d`` or ``,d`` —
  ``"industry 2030 targets"`` and ``"sector 20% share"`` are ordinary
  English, and three group codes (2010/2020/2030) are also years.
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
    public_names: dict[str, str]     # multi-word sector/group registry names with & or , -> label
    public_names_re: re.Pattern[str] | None
    exact: tuple[tuple[str, str], ...]


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


@lru_cache(maxsize=1)
def _index() -> _Index:
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
    pattern = None
    if public_names:
        alternation = "|".join(re.escape(n) for n in sorted(public_names, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\w&])(?:{alternation})(?![\w])")
    return _Index(names=names, public_names=public_names, public_names_re=pattern,
                  exact=_exact_table(labels))


def clear_cache() -> None:
    """Drop the cached label file and code index (tests that swap the file).
    The module constants keep the values read at import."""
    _default_labels.cache_clear()
    _index.cache_clear()


# --- scrub_text ----------------------------------------------------------------

_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
# A digit run that is a whole token: not glued to letters/digits, not the
# tail of a decimal or thousands group, not the head of one, not a
# percentage. This is what keeps "20% share", "453,010" and "2030.5" safe.
_TOKEN_GUARD_BEFORE = r"(?<![\w$.,])"
_TOKEN_GUARD_AFTER = r"(?![\w%]|[.,]\d)"
_DIGITS_RE = re.compile(_TOKEN_GUARD_BEFORE + r"(\d{2}|\d{4}|\d{6}|\d{8})" + _TOKEN_GUARD_AFTER)
_BRACKET_RE = re.compile(
    r"(\s*)([\[(])\s*(\d{2,8}(?:\s*[,;/]\s*\d{2,8})*)\s*([\])])"
)
_ANALYST_RE = re.compile(r"\bIndustry Group Analyst (\d{4})" + _TOKEN_GUARD_AFTER)
_PREFIX_RE = re.compile(
    r"\b(sector|industry group|group|industry)(\s+)(\d{2}|\d{4})" + _TOKEN_GUARD_AFTER, re.IGNORECASE,
)
_LONG_CODE_RE = re.compile(r"(\s?)" + _TOKEN_GUARD_BEFORE + r"(\d{6}|\d{8})" + _TOKEN_GUARD_AFTER)
_INTERNAL_KEY_RE = re.compile(r"\bgics-\d{4}-\d{2}\b", re.IGNORECASE)
_BRAND_BEFORE_WORD_RE = re.compile(r"\bGICS\b®?\s+(?=[A-Za-z])", re.IGNORECASE)
_BRAND_RE = re.compile(r"\bGICS\b®?", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:)\]])")


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
        # A lone 2-digit number in parentheses is far more often a count
        # than a sector ("(45)"); only square brackets are provenance marks.
        if opener == "(" and len(tok) == 2:
            return m.group(0)
    return ""


def _pair_subs(text: str, idx: _Index, labels: Labels) -> str:
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
        replacement = entry[0] if entry is not None else ""
        for form in (f"{name} ({code})", f"{name} [{code}]", f"{code} {name}"):
            if form in text:
                text = text.replace(form, replacement)
    return text


def scrub_text(text: str) -> str:
    """Remove taxonomy codes, the brand and registry group names from
    prose. Deterministic and idempotent; order matters:

    a. fixed branded sentences → their public equivalents;
    b. bracketed code lists (``[453010]``, ``[451020,451030]``, ``(4530)``)
       → removed with the space before them;
    c. ``name (code)`` / ``code name`` pairs → our label (sector/group) or
       nothing (industry/sub-industry); ``Industry Group Analyst dddd`` →
       the public display name; ``sector 45`` / ``group 4530`` → ``sector
       Technology`` / ``group Chips & Chipmaking Equipment``;
    d. standalone known 6/8-digit codes → removed. A bare 4-digit token is
       never touched (2020 and 2030 are years as well as group codes);
    e. multi-word registry sector/group names containing ``&`` or ``,`` →
       our label (case-sensitive);
    f. the internal taxonomy key → the public key; any remaining "GICS" is
       dropped before a noun ("the GICS sector" → "the sector") and
       otherwise read as "industry".

    Years (``19xx``/``20xx``) and numbers followed by ``%``, ``.d`` or
    ``,d`` are never altered by the contextual rules (b, c-prefix, d).
    """
    if not isinstance(text, str) or not text:
        return text
    labels = load()
    idx = _index()
    out = text
    for old, new in idx.exact:
        if old in out:
            out = out.replace(old, new)
    if any(ch.isdigit() for ch in out):
        # Pairs first: a bracket rule that ran earlier would strip the
        # "(4010)" off "Banks (4010)" and leave the registry name behind.
        out = _pair_subs(out, idx, labels)
        out = _ANALYST_RE.sub(
            lambda m: public_display_name(m.group(1)) if labels.entry(m.group(1)) else m.group(0), out,
        )
        out = _BRACKET_RE.sub(lambda m: _bracket_sub(m, idx.names), out)
        out = _PREFIX_RE.sub(lambda m: _prefix_sub(m, labels), out)
        out = _LONG_CODE_RE.sub(lambda m: "" if m.group(2) in idx.names else m.group(0), out)
    if idx.public_names_re is not None:
        out = idx.public_names_re.sub(lambda m: idx.public_names[m.group(0)], out)
    out = _INTERNAL_KEY_RE.sub(lambda m: public_version_key(m.group(0).lower()), out)
    if "gics" in out.lower():
        out = _BRAND_BEFORE_WORD_RE.sub("", out)
        out = _BRAND_RE.sub("industry", out)
    if out != text:
        # Only a string this function changed is tidied, so a caller's own
        # spacing survives untouched text.
        out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _MULTISPACE_RE.sub(" ", out))
    return out


def _prefix_sub(m: re.Match[str], labels: Labels) -> str:
    noun, gap, code = m.group(1), m.group(2), m.group(3)
    want = 2 if noun.lower() == "sector" else 4
    if len(code) != want or (len(code) == 4 and _is_year(code)):
        return m.group(0)
    entry = labels.entry(code)
    return f"{noun}{gap}{entry[0]}" if entry is not None else m.group(0)


def scrub_strings(obj: Any) -> Any:
    """``scrub_text`` over every string inside a nested dict/list, keys and
    structure untouched — for LLM output whose SHAPE is a contract (the
    sector analyst's ``bull_bear_analysis``)."""
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {k: scrub_strings(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_strings(v) for v in obj]
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
_BRANDED_SOURCE_RE = re.compile(r"gics|msci\.com|spglobal\.com", re.IGNORECASE)
_PUBLIC_REF = "MarketMosaic industry research knowledge base"


def _code_list_slugs(values: list[Any], idx: _Index, labels: Labels) -> list[Any]:
    """Each known code → the slug of its group (6/8-digit codes roll up to
    their 4-digit prefix); unknown values are kept; duplicates collapse."""
    out: list[Any] = []
    for v in values:
        key = _norm(v)
        if isinstance(v, str) and key.isdigit() and key in idx.names:
            entry = labels.entry(key if len(key) <= 4 else key[:4])
            if entry is None:
                continue
            v = entry[1]
        elif isinstance(v, str):
            v = scrub_text(v)
        if v not in out:
            out.append(v)
    return out


def _constant_for(key: str, value: Any, labels: Labels) -> Any:
    default = {"attribution": labels.attribution, "mapping_caveat": labels.mapping_caveat,
               "security_reference_caveat": labels.security_reference_caveat}[key]
    if not isinstance(value, str):
        return default
    # A known branded sentence maps to its own public twin (the brief
    # attribution is not the taxonomy attribution); any other text that
    # names the brand gets the key's public constant, not a half-scrub.
    exact = dict(_index().exact)
    if value in exact:
        return exact[value]
    return default if "gics" in value.lower() else scrub_text(value)


def _project_dict(obj: dict[Any, Any], idx: _Index, labels: Labels) -> dict[Any, Any]:
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
        elif k in _CONSTANT_KEYS:
            out[k] = _constant_for(k, v, labels)
        elif k == "provenance" and isinstance(v, dict):
            out[k] = {}
        elif k in _VERSION_KEYS and isinstance(v, str) and v.lower().startswith("gics-"):
            out[k] = public_version_key(v.lower())
        elif isinstance(k, str) and (k == "codes" or k.endswith("_codes")) and isinstance(v, list):
            out[k] = _code_list_slugs(v, idx, labels)
        elif k in _SOURCE_LIST_KEYS and isinstance(v, list):
            kept = [s for s in v if not _BRANDED_SOURCE_RE.search(json.dumps(s, default=str))]
            out[k] = [_project(s, idx, labels) for s in kept]
            if len(kept) != len(v):
                out[f"{k}_withheld"] = len(v) - len(kept)
        elif k == "ref" and isinstance(v, str) and "gics" in v.lower():
            out[k] = _PUBLIC_REF
        else:
            out[k] = _project(v, idx, labels)
    return out


def _project(obj: Any, idx: _Index, labels: Labels) -> Any:
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return _project_dict(obj, idx, labels)
    if isinstance(obj, (list, tuple)):
        return [_project(v, idx, labels) for v in obj]
    return copy.deepcopy(obj)


def project_public(obj: Any) -> Any:
    """A public copy of a JSON-shaped payload (the input is never mutated):

    1. ``code`` / ``sector_code`` / ``industry_group_code`` / ``group_code``
       / ``parent_code`` holding a known 2/4-digit code → its slug, and the
       sibling ``*name`` → our label; holding a 6/8-digit code → both keys
       removed.
    2. ``industry_code`` / ``sub_industry_code`` (and their names) → removed;
       ``codes`` / ``*_codes`` lists → group slugs, de-duplicated.
    3. dict keys equal to a known 2/4-digit code → slug; 6/8-digit keys →
       entry dropped.
    4. ``attribution`` / ``mapping_caveat`` / ``security_reference_caveat``
       → the public constants; ``provenance`` → ``{}``; internal taxonomy
       keys → the public key.
    5. ``primary_sources`` / ``map_sources`` entries naming the brand or its
       publishers → removed, counted in ``<key>_withheld``; a branded
       ``ref`` → the knowledge-base reference.
    6. every remaining string → ``scrub_text``.
    """
    return _project(obj, _index(), load())


def project_for_prompt(obj: Any) -> Any:
    """The same projection, applied to facts sent to a model: a prompt that
    never contains a code or registry name cannot be echoed into prose.
    A separate name so a prompt-only rule can diverge without moving what
    API responses carry."""
    return _project(obj, _index(), load())
