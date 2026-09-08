"""Compile the industry research encyclopedia into a JSON knowledge base.

The GICS Industry Group analysts (FEAT-003) need each industry's economic
engine, KPIs, leading indicators, moats, valuation lenses and failure
modes at prompt-assembly time. That knowledge is authored by hand as
markdown in ``docs/research/GICS_74_Industry_Research_Encyclopedia.md``
because prose is what an analyst maintains; the runtime wants a keyed
structure it can load once and index by GICS code. This script is the
bridge, and the *only* thing that should ever read the markdown: ``docs/``
is not copied into the production image (the Dockerfile copies ``backend``
alone), so the generated JSON under ``app/data/industry_knowledge/`` is
what ships, and it is checked in next to this script.

Usage (from ``backend/``)::

    python -m app.scripts.build_industry_knowledge          # rebuild the JSON
    python -m app.scripts.build_industry_knowledge --check  # exit 1 if stale

Source layout the parser understands::

    # <Sector>
    ## <Industry Group>
    ### <Industry> (<6-digit GICS code>)
    **Economic engine.** prose ...
    **Core KPIs.** prose ...
    **Research priority:** 5/5. **Cadence:** Monthly. **Preferred archetype:** ...

Every bold ``**Label.**`` / ``**Label:**`` inside an industry section becomes
a snake_case field, and several labels may share one paragraph. The label
set is discovered from the document rather than hard-coded here, but it
must be identical across all industries: the encyclopedia is a deliberately
uniform template, so a lone deviation is a typo rather than an enrichment,
and a mistyped label would otherwise ship as a differently-named key the
analysts never read. H1/H2 titles that own no industry sections (the
document's own title, the source library, the maintenance rule) are not
sectors and are dropped; the ten universal research rules and the primary
source library are carried along because the analysts use them too.

Anything that would produce a partial or wrong file raises ``SystemExit``
with a message instead of writing: heading counts other than 11 / 25 / 74,
an industry heading without a code, duplicate codes, an industry whose code
prefix disagrees with the heading it sits under, prose in a position the
parser would otherwise drop, empty or duplicate fields. A silently partial
knowledge base would only surface weeks later as an analyst reasoning
confidently from a blank field — the same failure shape as the
``written=0`` cron loops this repo has already been bitten by.

Output is byte-deterministic for a given source, so re-running is a no-op
and ``--check`` doubles as a drift test; ``source_sha256`` records which
revision of the markdown the file came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # scripts -> app -> backend -> repo root
DEFAULT_SOURCE = (
    REPO_ROOT / "docs" / "research" / "GICS_74_Industry_Research_Encyclopedia.md"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "data" / "industry_knowledge" / "gics_industries_2026.json"
)

TAXONOMY_VERSION = "gics-2026-04"

# The April 2026 GICS structure. The build refuses to emit anything else, so
# adopting a future GICS review is a deliberate edit here (plus
# TAXONOMY_VERSION and the output filename), never an accident of a
# half-edited markdown file.
EXPECTED_SECTORS = 11
EXPECTED_INDUSTRY_GROUPS = 25
EXPECTED_INDUSTRIES = 74

# Fields the analysts depend on unconditionally. The rest of the label set is
# discovered from the document.
REQUIRED_FIELDS = ("economic_engine", "core_kpis")

UNIVERSAL_RULES_HEADING = "Universal research rules"
PRIMARY_SOURCES_HEADING = "Primary source library"

_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
_INDUSTRY_TITLE = re.compile(r"^(?P<name>.+?)\s+\((?P<code>\d{6})\)$")
_LABEL = re.compile(r"\*\*(?P<label>[^*\n]+?)\*\*")
_NUMBERED_ITEM = re.compile(r"^\d+\.\s+(?P<text>.+?)\s*$")
_SOURCE_ITEM = re.compile(
    r"^-\s+\*\*(?P<domain>[^*]+?)\s+—\s+(?P<name>[^*]+?):\*\*"
    r"\s+(?P<url>\S+)\s+—\s+(?P<description>.+?)\s*$"
)


def _die(message: str) -> None:
    raise SystemExit(f"build_industry_knowledge: {message}")


@dataclass
class _Section:
    """One markdown heading plus the body lines up to the next heading."""

    level: int
    title: str
    line_no: int
    lines: list[str] = field(default_factory=list)


def _split_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = _HEADING.match(line)
        if match:
            current = _Section(
                level=len(match.group("hashes")),
                title=match.group("title"),
                line_no=line_no,
            )
            sections.append(current)
        elif current is not None:
            current.lines.append(line)
        # Text before the first heading is the document's own front matter.
    return sections


def _paragraphs(lines: Iterable[str]) -> list[str]:
    """Blank-line-separated blocks, soft line wraps joined with a space."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if line.strip():
            buffer.append(line.strip())
        elif buffer:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return paragraphs


def _snake(label: str) -> str:
    """``Capital cycle / supply response.`` -> ``capital_cycle_supply_response``."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _parse_fields(section: _Section) -> dict[str, str]:
    where = f"industry '{section.title}' (line {section.line_no})"
    fields: dict[str, str] = {}
    for paragraph in _paragraphs(section.lines):
        labels = list(_LABEL.finditer(paragraph))
        if not labels or paragraph[: labels[0].start()].strip():
            _die(
                f"unlabelled prose under {where}: {paragraph[:80]!r} — every "
                "paragraph in an industry section must open with a **Label.**"
            )
        for index, match in enumerate(labels):
            end = labels[index + 1].start() if index + 1 < len(labels) else len(paragraph)
            key = _snake(match.group("label"))
            value = paragraph[match.end():end].strip()
            if not key:
                _die(f"label {match.group('label')!r} under {where} has no letters")
            if key in fields:
                _die(f"duplicate field '{key}' under {where}")
            if not value:
                _die(f"empty field '{key}' under {where}")
            fields[key] = value
    if not fields:
        _die(f"no labelled fields under {where}")
    return fields


def _parse_numbered(section: _Section) -> list[str]:
    items: list[str] = []
    for line in section.lines:
        if not line.strip():
            continue
        match = _NUMBERED_ITEM.match(line.strip())
        if not match:
            _die(
                f"unrecognised line under '{section.title}': {line.strip()[:80]!r}; "
                "expected a numbered '1. ...' item"
            )
        items.append(match.group("text"))
    if not items:
        _die(f"'{section.title}' is present but holds no numbered items")
    return items


def _parse_primary_sources(section: _Section) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for line in section.lines:
        if not line.strip():
            continue
        match = _SOURCE_ITEM.match(line.strip())
        if not match:
            _die(
                f"unrecognised entry under '{section.title}': {line.strip()[:80]!r}; "
                "expected '- **<Domain> — <Name>:** <url> — <description>'"
            )
        items.append(
            {
                "domain": match.group("domain").strip(),
                "name": match.group("name").strip(),
                "url": match.group("url"),
                "description": match.group("description").strip(),
            }
        )
    if not items:
        _die(f"'{section.title}' is present but holds no entries")
    return items


def _build_tree(
    sections: list[_Section],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    """Nest H2 under H1 and coded H3 under H2, in document order.

    Codes are assigned later, once each node's industries are known; the
    ``_lines`` / ``_line_no`` scaffolding is stripped there too.
    """
    sectors: list[dict[str, Any]] = []
    sector: dict[str, Any] | None = None
    group: dict[str, Any] | None = None
    universal_rules: list[str] = []
    primary_sources: list[dict[str, str]] = []

    for section in sections:
        if section.level == 1:
            sector = {
                "name": section.title,
                "industry_groups": [],
                "_lines": section.lines,
                "_line_no": section.line_no,
            }
            group = None
            sectors.append(sector)
            if section.title == PRIMARY_SOURCES_HEADING:
                primary_sources = _parse_primary_sources(section)
        elif section.level == 2:
            if sector is None:
                _die(f"'## {section.title}' (line {section.line_no}) precedes any '# <Sector>'")
            group = {
                "name": section.title,
                "industries": [],
                "_lines": section.lines,
                "_line_no": section.line_no,
            }
            sector["industry_groups"].append(group)
            if section.title == UNIVERSAL_RULES_HEADING:
                universal_rules = _parse_numbered(section)
        elif section.level == 3:
            match = _INDUSTRY_TITLE.match(section.title)
            if not match:
                _die(
                    f"industry heading without a 6-digit GICS code at line "
                    f"{section.line_no}: '### {section.title}'"
                )
            if group is None:
                _die(
                    f"'### {section.title}' (line {section.line_no}) precedes any "
                    "'## <Industry Group>'"
                )
            group["industries"].append(
                {
                    "code": match.group("code"),
                    "name": match.group("name"),
                    "fields": _parse_fields(section),
                }
            )
        else:
            _die(f"unexpected heading depth {section.level} at line {section.line_no}")

    return sectors, universal_rules, primary_sources


def _reject_prose(node: dict[str, Any], kind: str) -> None:
    stray = [line for line in node["_lines"] if line.strip()]
    if stray:
        _die(
            f"prose directly under {kind} heading '{node['name']}' (line "
            f"{node['_line_no']}) would be dropped: {stray[0][:80]!r}. Only "
            "'### <Industry> (code)' sections carry fields."
        )


def _finalize(raw_sectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop headings that own no industries and derive codes from prefixes.

    A sector or group code is the shared prefix of its industries' codes,
    so the heading an industry sits under is checked against the GICS
    hierarchy for free: a mis-filed industry shows up as a mixed prefix.
    """
    sectors: list[dict[str, Any]] = []
    for raw_sector in raw_sectors:
        groups: list[dict[str, Any]] = []
        for raw_group in raw_sector["industry_groups"]:
            if not raw_group["industries"]:
                continue
            _reject_prose(raw_group, "industry group")
            prefixes = sorted({ind["code"][:4] for ind in raw_group["industries"]})
            if len(prefixes) != 1:
                _die(
                    f"industry group '{raw_group['name']}' mixes GICS group prefixes "
                    f"{prefixes}: an industry is filed under the wrong heading"
                )
            groups.append(
                {"code": prefixes[0], "name": raw_group["name"], "industries": raw_group["industries"]}
            )
        if not groups:
            continue
        _reject_prose(raw_sector, "sector")
        prefixes = sorted({grp["code"][:2] for grp in groups})
        if len(prefixes) != 1:
            _die(
                f"sector '{raw_sector['name']}' mixes GICS sector prefixes {prefixes}: "
                "an industry group is filed under the wrong heading"
            )
        sectors.append({"code": prefixes[0], "name": raw_sector["name"], "industry_groups": groups})
    return sectors


def _validate(
    sectors: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    industries: list[dict[str, Any]],
) -> None:
    counts = (len(sectors), len(groups), len(industries))
    expected = (EXPECTED_SECTORS, EXPECTED_INDUSTRY_GROUPS, EXPECTED_INDUSTRIES)
    if counts != expected:
        _die(
            f"parsed {counts[0]} sectors / {counts[1]} industry groups / "
            f"{counts[2]} industries; {TAXONOMY_VERSION} has "
            f"{expected[0]} / {expected[1]} / {expected[2]}. Refusing to write a "
            "partial knowledge base — fix the headings in the source, or update "
            "EXPECTED_* if GICS itself changed."
        )
    for level, items in (("sector", sectors), ("industry group", groups), ("industry", industries)):
        seen: dict[str, str] = {}
        for item in items:
            if item["code"] in seen:
                _die(
                    f"duplicate {level} code {item['code']}: "
                    f"'{seen[item['code']]}' and '{item['name']}'"
                )
            seen[item["code"]] = item["name"]

    reference = set(industries[0]["fields"])
    for industry in industries:
        label = f"industry {industry['code']} '{industry['name']}'"
        missing = [name for name in REQUIRED_FIELDS if not industry["fields"].get(name)]
        if missing:
            _die(f"{label} lacks required field(s) {missing}")
        present = set(industry["fields"])
        if present != reference:
            _die(
                f"{label} has fields {sorted(present ^ reference)} that differ from "
                f"industry {industries[0]['code']}; the template is uniform, so a "
                "one-off label is a typo"
            )


def _relative_source(source: Path) -> str:
    try:
        return source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return source.name


def build_payload(source: Path) -> dict[str, Any]:
    """Parse ``source`` into the knowledge-base dict, or raise SystemExit."""
    raw = source.read_bytes()
    sections = _split_sections(raw.decode("utf-8"))
    raw_sectors, universal_rules, primary_sources = _build_tree(sections)
    sectors = _finalize(raw_sectors)
    groups = [grp for sec in sectors for grp in sec["industry_groups"]]
    industries = [ind for grp in groups for ind in grp["industries"]]
    if not industries:
        _die(f"no '### <Industry> (<code>)' sections found in {source}")
    _validate(sectors, groups, industries)

    generated_from = _relative_source(source)
    return {
        "_doc": (
            f"Provenance: generated from {generated_from} by "
            "app/scripts/build_industry_knowledge.py; do not edit by hand — rerun "
            "`python -m app.scripts.build_industry_knowledge` from backend/. "
            f"GICS structure April 2026: {len(sectors)} sectors / {len(groups)} "
            f"industry groups / {len(industries)} industries; codes are the public "
            "GICS numeric hierarchy (sector=2 digits, group=4, industry=6). The "
            "descriptive text is MarketMosaic's own analyst research, not licensed "
            "GICS content."
        ),
        "taxonomy_version": TAXONOMY_VERSION,
        "generated_from": generated_from,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "sector_count": len(sectors),
        "industry_group_count": len(groups),
        "industry_count": len(industries),
        "field_names": list(industries[0]["fields"]),
        "universal_research_rules": universal_rules,
        "primary_sources": primary_sources,
        "sectors": sectors,
    }


def render(payload: dict[str, Any]) -> bytes:
    """Serialize deterministically: fixed key order, UTF-8, trailing newline.

    ``ensure_ascii=False`` keeps the analysts' arrows, multiplication signs
    and em dashes readable in diffs instead of as ``\\u2192`` escapes.
    """
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Parse ``source`` and write ``output``; returns the payload written."""
    payload = build_payload(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render(payload))
    return payload


def _summary(payload: dict[str, Any]) -> str:
    return (
        f"{payload['sector_count']} sectors / {payload['industry_group_count']} "
        f"industry groups / {payload['industry_count']} industries / "
        f"{len(payload['field_names'])} fields per industry"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the industry research encyclopedia into JSON.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"markdown encyclopedia (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"JSON to write (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--check", action="store_true",
                        help="Write nothing; exit 1 if the output is missing or "
                             "differs from what the source produces.")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        _die(f"source not found: {args.source}")
    payload = build_payload(args.source)
    rendered = render(payload)

    if args.check:
        current = args.output.read_bytes() if args.output.is_file() else None
        if current == rendered:
            print(f"{args.output} is up to date ({_summary(payload)})")
            return 0
        print(
            f"{args.output} is stale or missing — rerun without --check "
            f"({_summary(payload)})",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    print(f"Wrote {_summary(payload)} to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
