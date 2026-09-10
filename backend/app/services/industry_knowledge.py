"""Read-only access to the GICS industry knowledge base.

``data/industry_knowledge/gics_industries_2026.json`` is generated from the
analyst-authored encyclopedia by ``app.scripts.build_industry_knowledge``;
the markdown itself never ships (``docs/`` is outside the image), so this
module is the runtime's only way in. It reads the file once per process,
indexes it by GICS code and answers every lookup from memory, so an
Industry Group analyst assembling a prompt pays nothing per call.

Counts, names and codes all come from the file. Nothing here knows how many
industry groups exist: FEAT-003's taxonomy decision was explicit that 24/25
must never be hard-coded (nor 163 sub-industries), so a future GICS review
lands by regenerating the JSON, not by editing this module.

The file carries four levels. Sector / industry group / industry and their
research fields come from the encyclopedia; the 8-digit sub-industries with
their original briefs, the retired rows, the cross-industry relationships,
the governing methodology, the source register and the economic security
reference come from the user-authored universe map, merged by
``build_industry_knowledge`` so the runtime reads exactly one document.
Every brief is original analysis and is labelled as such wherever it is
displayed; the security reference is "economic research examples, not
official licensed issuer GICS mapping" (its own metadata says so).

Lookups return deep copies. The payload from ``load_industry_knowledge()``
is the shared cache and must be treated as read-only.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

KNOWLEDGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "industry_knowledge" / "gics_industries_2026.json"
)

# Shown next to every sub-industry brief and security reference so the
# provenance the map's author insisted on travels with the text.
BRIEF_ATTRIBUTION = (
    "Original MarketMosaic analyst research; GICS codes and names are factual "
    "taxonomy, the descriptive text is not licensed GICS content."
)
SECURITY_REFERENCE_CAVEAT = (
    "Economic research examples; not official licensed issuer GICS mapping."
)


@lru_cache(maxsize=1)
def load_industry_knowledge() -> dict[str, Any]:
    """The whole generated payload, read once per process.

    Raises ``FileNotFoundError`` if the JSON has not been built. A missing
    knowledge base is a build defect, and an empty-dict fallback would
    surface weeks later as an analyst reasoning from blank fields — the
    silent-partial failure shape this repo has already paid for.
    """
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Flat ``code -> record`` maps for industries and groups, in document order.

    Each record carries its parents' code and name so a caller holding a
    6-digit code never has to walk the tree to learn which sector it is in.
    """
    industries: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for sector in load_industry_knowledge()["sectors"]:
        for group in sector["industry_groups"]:
            groups[group["code"]] = {
                "code": group["code"],
                "name": group["name"],
                "sector_code": sector["code"],
                "sector_name": sector["name"],
                "industries": group["industries"],
            }
            for industry in group["industries"]:
                industries[industry["code"]] = {
                    "code": industry["code"],
                    "name": industry["name"],
                    "industry_group_code": group["code"],
                    "industry_group_name": group["name"],
                    "sector_code": sector["code"],
                    "sector_name": sector["name"],
                    "fields": industry["fields"],
                    "sub_industries": industry.get("sub_industries", []),
                }
    return industries, groups


@lru_cache(maxsize=1)
def _sub_industry_index() -> dict[str, dict[str, Any]]:
    """Flat ``code8 -> record`` for the active sub-industries, in document
    order, each carrying its full parent chain so a caller holding an
    8-digit code learns its industry, group and sector in one lookup."""
    out: dict[str, dict[str, Any]] = {}
    for industry in _indexes()[0].values():
        for sub in industry["sub_industries"]:
            out[sub["code"]] = {
                "code": sub["code"],
                "name": sub["name"],
                "industry_code": industry["code"],
                "industry_name": industry["name"],
                "industry_group_code": industry["industry_group_code"],
                "industry_group_name": industry["industry_group_name"],
                "sector_code": industry["sector_code"],
                "sector_name": industry["sector_name"],
                "fields": sub["fields"],
                "source_ids": sub.get("source_ids", []),
                "cross_industry_themes": sub.get("cross_industry_themes", []),
                "framework_status": sub.get("framework_status", ""),
                "last_framework_review": sub.get("last_framework_review", ""),
                "attribution": BRIEF_ATTRIBUTION,
            }
    return out


def _normalize(code: Any) -> str:
    return str(code or "").strip()


def get_industry(code: str) -> dict[str, Any] | None:
    """One industry by 6-digit GICS code, with its group/sector identity and
    all research fields; ``None`` for an unknown code."""
    entry = _indexes()[0].get(_normalize(code))
    return copy.deepcopy(entry) if entry is not None else None


def get_industry_group(code4: str) -> dict[str, Any] | None:
    """One industry group by 4-digit GICS code, including its industries
    (each with fields); ``None`` for an unknown code."""
    entry = _indexes()[1].get(_normalize(code4))
    return copy.deepcopy(entry) if entry is not None else None


def list_industry_groups() -> list[dict[str, Any]]:
    """Every industry group as ``{code, name, sector_code, sector_name,
    industry_count}``, in taxonomy (document) order."""
    return [
        {
            "code": group["code"],
            "name": group["name"],
            "sector_code": group["sector_code"],
            "sector_name": group["sector_name"],
            "industry_count": len(group["industries"]),
        }
        for group in _indexes()[1].values()
    ]


def search_industries(text: str) -> list[str]:
    """Codes of industries whose name contains ``text`` (case-insensitive),
    in taxonomy order. A blank query matches nothing rather than everything."""
    needle = _normalize(text).lower()
    if not needle:
        return []
    return [
        code for code, industry in _indexes()[0].items()
        if needle in industry["name"].lower()
    ]


# --- the sub-industry layer (universe map) -----------------------------------


def get_sub_industry(code8: str) -> dict[str, Any] | None:
    """One active sub-industry by 8-digit GICS code, with its industry /
    group / sector identity, the original brief ``fields`` and the map's
    per-entry bookkeeping; ``None`` for an unknown or retired code."""
    entry = _sub_industry_index().get(_normalize(code8))
    return copy.deepcopy(entry) if entry is not None else None


def list_sub_industries(code: str | None = None) -> list[dict[str, Any]]:
    """Active sub-industries as ``{code, name, industry_code, industry_group_code,
    sector_code}`` in taxonomy order.

    ``code`` narrows by prefix at any level — a 6-digit industry, a 4-digit
    group or a 2-digit sector — and ``None`` lists every entry. An unknown
    prefix matches nothing; a blank one (after stripping) is treated as
    "no filter" only when ``None`` was passed, so ``""`` returns ``[]``.
    """
    if code is None:
        prefix = ""
    else:
        prefix = _normalize(code)
        if not prefix:
            return []
    return [
        {
            "code": sub["code"],
            "name": sub["name"],
            "industry_code": sub["industry_code"],
            "industry_group_code": sub["industry_group_code"],
            "sector_code": sub["sector_code"],
        }
        for sub in _sub_industry_index().values()
        if sub["code"].startswith(prefix)
    ]


def retired_sub_industries() -> list[dict[str, Any]]:
    """The rows the official structure discontinued, as ``{code, name,
    parent_code, parent_name, status}``. They are not in the active index;
    the registry imports them as inactive nodes so an old classification
    can still be named."""
    return copy.deepcopy(load_industry_knowledge().get("retired_sub_industries", []))


def security_reference(symbol: str) -> dict[str, Any] | None:
    """``{symbol, codes, as_of, source_id, caveat}`` for a symbol in the map's
    economic security reference, or ``None``. ``codes`` are active 8-digit
    sub-industry codes (several when the author recorded more than one
    exposure); ``caveat`` restates that these are research examples, not a
    licensed issuer assignment."""
    sym = _normalize(symbol).upper()
    if not sym:
        return None
    entry = load_industry_knowledge().get("security_reference", {}).get(sym)
    if entry is None:
        return None
    return {
        "symbol": sym,
        "codes": list(entry.get("codes", [])),
        "as_of": entry.get("as_of", ""),
        "source_id": entry.get("source_id", ""),
        "caveat": SECURITY_REFERENCE_CAVEAT,
    }


def security_reference_as_of() -> str:
    """The map edition the security reference came from (``map_as_of``) —
    the provenance stamp a classification records as its author's date."""
    return str(load_industry_knowledge().get("map_as_of", ""))


def cross_industry_relationships() -> list[dict[str, Any]]:
    """The map's themed cross-industry relationships, each ``{theme,
    mechanism, codes, monitor, failure_of_inference, source_ids}`` with
    ``codes`` resolving to active sub-industries."""
    return copy.deepcopy(load_industry_knowledge().get("cross_industry_relationships", []))


def governing_methodology() -> dict[str, Any]:
    """The eight-stage causal order (``thesis_construction_order``), the
    per-link requirements, the operating rules and the named anti-patterns —
    loaded, never retyped, so prompts and validators share one spine."""
    return copy.deepcopy(load_industry_knowledge().get("governing_methodology", {}))


def sources() -> list[dict[str, Any]]:
    """The map's source register (``S01``…): id, title, url, purpose,
    review scope, cadence and access date. Distinct from
    ``primary_sources`` (the encyclopedia's source library)."""
    return copy.deepcopy(load_industry_knowledge().get("sources", []))


def map_metadata() -> dict[str, Any]:
    """Provenance of the sub-industry layer: ``as_of``, the official structure
    effective date, the workbook hash and the security-assignment caveat."""
    return copy.deepcopy(load_industry_knowledge().get("map_metadata", {}))
