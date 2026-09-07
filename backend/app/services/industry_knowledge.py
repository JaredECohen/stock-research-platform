"""Read-only access to the GICS industry knowledge base.

``data/industry_knowledge/gics_industries_2026.json`` is generated from the
analyst-authored encyclopedia by ``app.scripts.build_industry_knowledge``;
the markdown itself never ships (``docs/`` is outside the image), so this
module is the runtime's only way in. It reads the file once per process,
indexes it by GICS code and answers every lookup from memory, so an
Industry Group analyst assembling a prompt pays nothing per call.

Counts, names and codes all come from the file. Nothing here knows how many
industry groups exist: FEAT-003's taxonomy decision was explicit that 24/25
must never be hard-coded, so a future GICS review lands by regenerating the
JSON, not by editing this module.

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
                }
    return industries, groups


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
