"""Owner decision 2026-09-24 — MarketMosaic's own public industry labels.

Pinned here (`services/industry_labels.py`, contract C3 of the 2026-09-24
integration plan):

- the label file covers every sector and industry group the bundled
  knowledge base carries (counts come from the data, never a literal), its
  slugs are unique and URL-safe, and no label is the registry's own name —
  the point of the file is that nothing public reads as the licensed
  taxonomy;
- `label` / `slug` answer for 2- and 4-digit codes only; industries and
  sub-industries raise, because they are never named publicly;
- `code_for` round-trips every slug and passes known codes through;
- `scrub_text` removes codes, the brand and registry group names while
  leaving years and percentages alone ("industry 2030 targets", "sector 20%
  share");
- `project_public` rewrites a payload's keys as well as its strings;
- a missing or malformed file raises instead of falling back to names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services import gics_registry as reg
from app.services import industry_knowledge as ik
from app.services import industry_labels as il


def _knowledge_nodes() -> dict[str, str]:
    """Every sector and industry-group code the knowledge base carries,
    with its registry name."""
    out: dict[str, str] = {}
    for sector in ik.load_industry_knowledge()["sectors"]:
        out[sector["code"]] = sector["name"]
        for group in sector["industry_groups"]:
            out[group["code"]] = group["name"]
    return out


# --- the label file ----------------------------------------------------------------


def test_every_sector_and_group_has_a_label_and_nothing_else_does():
    nodes = _knowledge_nodes()
    labels = il.load()
    payload = ik.load_industry_knowledge()
    assert set(labels.sectors) == {c for c in nodes if len(c) == 2}
    assert set(labels.groups) == {c for c in nodes if len(c) == 4}
    # The counts are the data's own, not a literal.
    assert len(labels.sectors) == payload["sector_count"]
    assert len(labels.groups) == payload["industry_group_count"]
    for code in nodes:
        assert il.label(code) and il.slug(code)


def test_slugs_are_unique_url_safe_and_disjoint_between_levels():
    labels = il.load()
    slugs = [s for _, s in labels.sectors.values()] + [s for _, s in labels.groups.values()]
    assert len(slugs) == len(set(slugs))
    assert all(re.fullmatch(r"[a-z][a-z0-9-]+", s) for s in slugs)
    assert not {s for _, s in labels.sectors.values()} & {s for _, s in labels.groups.values()}


def test_no_label_is_a_registry_name():
    """The licensing decision in one assertion: a public label must never
    BE the taxonomy's name for that node — or for any other node at the
    two public levels (a label equal to another group's registry name would
    read as that group)."""
    nodes = _knowledge_nodes()
    registry_names = {n.casefold() for n in nodes.values()}
    for code, name in nodes.items():
        public = il.label(code)
        assert public.casefold() != name.casefold(), code
        assert public.casefold() not in registry_names, code
        assert "gics" not in public.lower() and not re.search(r"\d", public), code


def test_the_public_constants_carry_no_brand_and_the_key_follows_the_bundled_version():
    for text in (il.PUBLIC_ATTRIBUTION, il.PUBLIC_MAPPING_CAVEAT, il.PUBLIC_BRIEF_ATTRIBUTION,
                 il.PUBLIC_SECURITY_REFERENCE_CAVEAT, il.PUBLIC_TAXONOMY_KEY):
        assert text and "gics" not in text.lower() and "msci" not in text.lower()
    bundled = ik.load_industry_knowledge()["taxonomy_version"]
    assert il.PUBLIC_TAXONOMY_KEY == il.public_version_key(bundled) == "mm-2026-04"
    other = il.public_version_key("some-internal-key")
    assert re.fullmatch(r"mm-[0-9a-f]{8}", other) and other == il.public_version_key("some-internal-key")
    assert il.public_version_key("mm-2026-04") == "mm-2026-04"


# --- the accessors -----------------------------------------------------------------


@pytest.mark.parametrize("code", ["453010", "45301020", "0000", "99", "", None, "semis", "453"])
def test_label_and_slug_answer_for_sector_and_group_codes_only(code):
    with pytest.raises(il.UnknownLabel):
        il.label(code)
    with pytest.raises(il.UnknownLabel):
        il.slug(code)


def test_code_for_round_trips_every_slug_and_passes_codes_through():
    for code in _knowledge_nodes():
        assert il.code_for(il.slug(code)) == code
        assert il.code_for(il.slug(code).upper()) == code   # a slug in a URL is case-insensitive
        assert il.code_for(code) == code                   # old links carry codes
    for bad in ("no-such-slug", "453010", "0000", ""):
        with pytest.raises(il.UnknownLabel):
            il.code_for(bad)


def test_public_display_name_names_the_label_not_the_code():
    assert il.public_display_name("4530") == f"Industry Group Analyst ({il.label('4530')})"
    with pytest.raises(il.UnknownLabel):
        il.public_display_name("453010")


# --- a missing or malformed file raises ----------------------------------------------


def test_a_missing_labels_file_raises(tmp_path):
    with pytest.raises(il.LabelsMissing):
        il.load(tmp_path / "public_labels.json")


def _write(tmp_path: Path, mutate) -> Path:
    raw = json.loads(il.LABELS_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "public_labels.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.pop("attribution"),
    lambda raw: raw["industry_groups"].__setitem__("4530", {"label": "", "slug": "chips"}),
    lambda raw: raw["industry_groups"].__setitem__("4530", {"label": "Chips", "slug": "technology"}),
    lambda raw: raw["industry_groups"].__setitem__("453010", {"label": "Chips", "slug": "chips-x"}),
    lambda raw: raw["industry_groups"].__setitem__("9910", {"label": "Orphan", "slug": "orphan"}),
    lambda raw: raw.__setitem__("sectors", {}),
], ids=["no-attribution", "empty-label", "duplicate-slug", "six-digit-key", "orphan-group", "no-sectors"])
def test_a_malformed_labels_file_raises(tmp_path, mutate):
    with pytest.raises(il.LabelsMissing):
        il.load(_write(tmp_path, mutate))


def test_an_unreadable_labels_file_raises(tmp_path):
    path = tmp_path / "public_labels.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(il.LabelsMissing):
        il.load(path)


# --- scrub_text ----------------------------------------------------------------------

_T, _C, _B = il.label("45"), il.label("4530"), il.label("4010")

SCRUB_TABLE = [
    # (input, expected) — the contextual rules
    ("sector 45 constituents", f"sector {_T} constituents"),
    ("group 4530 leads", f"group {_C} leads"),
    ("the industry group 4530 read", f"the industry group {_C} read"),
    # years and percentages are never altered
    ("industry 2030 targets", "industry 2030 targets"),
    ("sector 20% share", "sector 20% share"),
    ("group 2030 grew", "group 2030 grew"),
    ("(2030) guidance", "(2030) guidance"),
    ("revenue of 4510 units in 2026", "revenue of 4510 units in 2026"),
    ("453010.5 and 453,010", "453010.5 and 453,010"),
    ("sector 45.5 multiple", "sector 45.5 multiple"),
    # provenance brackets and bare long codes
    ("KPIs [453010] matter", "KPIs matter"),
    ("KPIs [451030, 451020] matter", "KPIs matter"),
    ("Semis (4530) lead", "Semis lead"),
    ("unknown [123456] stays", "unknown [123456] stays"),
    ("a count (45) stays", "a count (45) stays"),
    ("sub-industry 45301020 is cyclical", "sub-industry is cyclical"),
    # names next to codes, and the multi-word registry names
    ("Semiconductors & Semiconductor Equipment (4530) leads", f"{_C} leads"),
    ("4530 Semiconductors & Semiconductor Equipment", _C),
    ("Banks (4010)", _B),
    ("Transportation (2030) is cyclical", f"{il.label('2030')} is cyclical"),
    ("Software & Services names", f"{il.label('4510')} names"),
    ("Industry Group Analyst 4530 said", f"Industry Group Analyst ({_C}) said"),
    # the brand and the internal key
    ("the GICS® sector", "the sector"),
    ("GICS industry group 4530", f"industry group {_C}"),
    ("taxonomy gics-2026-04", "taxonomy mm-2026-04"),
    # ordinary English is left alone — a count or a year before a name is
    # not a code
    ("Banks and energy names rallied", "Banks and energy names rallied"),
    ("the top 10 Energy names", "the top 10 Energy names"),
    ("in 2010 Capital Goods orders fell", "in 2010 Capital Goods orders fell"),
    ("", ""),
]


@pytest.mark.parametrize("text,expected", SCRUB_TABLE)
def test_scrub_text_table(text, expected):
    assert il.scrub_text(text) == expected
    assert il.scrub_text(expected) == expected   # idempotent


def test_scrub_text_maps_the_fixed_branded_sentences_to_their_public_twins():
    assert il.scrub_text(reg.ATTRIBUTION) == il.PUBLIC_ATTRIBUTION
    assert il.scrub_text(reg.MAPPING_CAVEAT) == il.PUBLIC_MAPPING_CAVEAT
    assert il.scrub_text(ik.BRIEF_ATTRIBUTION) == il.PUBLIC_BRIEF_ATTRIBUTION
    assert il.scrub_text(ik.SECURITY_REFERENCE_CAVEAT) == il.PUBLIC_SECURITY_REFERENCE_CAVEAT
    text = ("Constituents on file: 9, from research_map (mapping derived from provider "
            "classification, not licensed GICS security assignments). Largest: NVDA.")
    assert il.scrub_text(text) == (
        f"Constituents on file: 9, from research_map ({il.PUBLIC_MAPPING_CAVEAT}). Largest: NVDA."
    )


def test_scrub_text_leaves_untouched_text_byte_identical():
    text = "Margins  expanded ,  2030 targets hold (see 10-K)."
    assert il.scrub_text(text) == text


def test_scrub_strings_keeps_the_shape():
    obj = {"a": ["KPIs [453010] matter", 3, None], "b": {"c": "the GICS sector"}, "n": 1.5}
    assert il.scrub_strings(obj) == {"a": ["KPIs matter", 3, None], "b": {"c": "the sector"}, "n": 1.5}


# --- project_public ------------------------------------------------------------------


def test_project_public_rewrites_keys_and_strings_and_never_mutates_its_input():
    payload = {
        "code": "4530", "name": "Semiconductors & Semiconductor Equipment",
        "sector_code": "45", "sector_name": "Information Technology",
        "industry_code": "453010", "industry_name": "Semiconductors & Semiconductor Equipment",
        "sub_industry_code": "45301020",
        "industry_codes": ["453010", "451020", "453010"],
        "not_coverable_codes": ["4530", "no-such"],
        "sub_industry": {"code": "45301020", "name": "Semiconductors", "note": "ok"},
        "report_versions": {"4530": 3, "45301020": 1},
        "attribution": reg.ATTRIBUTION,
        "mapping_caveat": reg.MAPPING_CAVEAT,
        "brief": {"attribution": ik.BRIEF_ATTRIBUTION},
        "provenance": {"taxonomy": "gics-2026-04"},
        "taxonomy_version": "gics-2026-04",
        "primary_sources": [{"name": "SEC EDGAR", "url": "https://www.sec.gov"},
                            {"name": "MSCI GICS workbook", "url": "https://www.msci.com/x"}],
        "evidence": [{"ref": "GICS structure", "excerpt": "KPIs [453010] matter"}],
        "status": 404,
    }
    before = json.dumps(payload, sort_keys=True)
    out = il.project_public(payload)
    assert json.dumps(payload, sort_keys=True) == before
    assert out["code"] == il.slug("4530") and out["name"] == il.label("4530")
    assert out["sector_code"] == il.slug("45") and out["sector_name"] == il.label("45")
    for gone in ("industry_code", "industry_name", "sub_industry_code"):
        assert gone not in out
    assert out["industry_codes"] == [il.slug("4530"), il.slug("4510")]
    assert out["not_coverable_codes"] == [il.slug("4530"), "no-such"]
    assert out["sub_industry"] == {"note": "ok"}
    assert out["report_versions"] == {il.slug("4530"): 3}
    assert out["attribution"] == il.PUBLIC_ATTRIBUTION
    assert out["mapping_caveat"] == il.PUBLIC_MAPPING_CAVEAT
    assert out["brief"]["attribution"] == il.PUBLIC_BRIEF_ATTRIBUTION
    assert out["provenance"] == {} and out["taxonomy_version"] == il.PUBLIC_TAXONOMY_KEY
    assert out["primary_sources"] == [{"name": "SEC EDGAR", "url": "https://www.sec.gov"}]
    assert out["primary_sources_withheld"] == 1
    assert out["evidence"] == [{"ref": "MarketMosaic industry research knowledge base", "excerpt": "KPIs matter"}]
    assert out["status"] == 404
    blob = json.dumps(out)
    assert "gics" not in blob.lower() and not re.search(r"\b\d{6}(\d{2})?\b", blob)
    assert il.project_for_prompt(payload) == out
