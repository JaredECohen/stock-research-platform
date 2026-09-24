"""FEAT-003 slice 1 — the versioned GICS taxonomy registry.

What can rot here: the registry importing a structure that disagrees
with the knowledge JSON, a re-import rewriting history under an existing
key, two versions active at once, a process-local "active" flag hiding a
flip made by the other process, and a retired sub-industry quietly
reappearing as active. Each has a test.

No test asserts a literal node count — every count is read from the
knowledge JSON and compared with what the registry holds.
"""
from __future__ import annotations

import copy
import uuid

import pytest
from sqlalchemy import delete, func, select, update

from app.database import SessionLocal
from app.models import GicsNode, TaxonomyVersion
from app.services import gics_registry as reg
from app.services import industry_knowledge as ik


@pytest.fixture(scope="module")
def bundled() -> reg.VersionInfo:
    """The bundled knowledge JSON, imported and active (idempotent)."""
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    return info


@pytest.fixture()
def restore_active(bundled):
    """Whatever a test does to the active flag, the bundled version is
    active again afterwards — other modules' tests read it."""
    yield
    reg.activate_version(bundled.version_key)


def _active_rows() -> list[str]:
    with SessionLocal() as db:
        return list(db.execute(
            select(TaxonomyVersion.version_key).where(TaxonomyVersion.is_active.is_(True))
        ).scalars().all())


def _drop_version(key: str) -> None:
    with SessionLocal() as db:
        vid = db.execute(
            select(TaxonomyVersion.id).where(TaxonomyVersion.version_key == key)
        ).scalar_one_or_none()
        if vid is not None:
            db.execute(delete(GicsNode).where(GicsNode.taxonomy_version_id == vid))
            db.execute(delete(TaxonomyVersion).where(TaxonomyVersion.id == vid))
            db.commit()
    reg.invalidate_cache()


def _payload_under_key(key: str) -> dict:
    payload = copy.deepcopy(ik.load_industry_knowledge())
    payload["taxonomy_version"] = key
    return payload


def _tree_counts(payload: dict) -> dict[str, int]:
    sectors = payload["sectors"]
    groups = [g for s in sectors for g in s["industry_groups"]]
    industries = [i for g in groups for i in g["industries"]]
    subs = [s for i in industries for s in i.get("sub_industries", [])]
    return {
        "sector": len(sectors), "industry_group": len(groups),
        "industry": len(industries), "sub_industry": len(subs),
        "inactive": len(payload.get("retired_sub_industries", [])),
    }


# --- the node list derived from the knowledge JSON ------------------------


def test_nodes_from_payload_holds_four_levels_with_consistent_prefixes():
    payload = ik.load_industry_knowledge()
    nodes = reg.nodes_from_payload(payload)
    expected = _tree_counts(payload)

    active = [n for n in nodes if n["is_active"]]
    by_level = {level: sum(1 for n in active if n["level"] == level) for level in reg.NODE_LEVELS}
    assert by_level == {k: v for k, v in expected.items() if k != "inactive"}
    assert sum(1 for n in nodes if not n["is_active"]) == expected["inactive"]
    assert by_level["sub_industry"] == payload["sub_industry_count"]

    keys = [(n["level"], n["code"]) for n in nodes]
    assert len(keys) == len(set(keys)), "duplicate (level, code)"
    codes_active = {(n["level"], n["code"]) for n in active}
    for n in active:
        length = {"sector": 2, "industry_group": 4, "industry": 6, "sub_industry": 8}[n["level"]]
        assert len(n["code"]) == length and n["code"].isdigit(), n
        if n["level"] == "sector":
            assert n["parent_code"] is None
        else:
            parent_level = reg.NODE_LEVELS[reg.NODE_LEVELS.index(n["level"]) - 1]
            assert n["parent_code"] == n["code"][: length - 2], n
            assert (parent_level, n["parent_code"]) in codes_active, n
    for n in nodes:
        if not n["is_active"]:
            assert n["level"] == "sub_industry"
            assert n["effective_to"] is not None and n["effective_from"] is None
            assert n["attributes"]["status"] == "discontinued"
    # Sort order is document order, so the tree renders without a sort key.
    assert [n["sort_order"] for n in nodes] == list(range(1, len(nodes) + 1))


def test_checksum_is_structural_not_textual():
    payload = ik.load_industry_knowledge()
    nodes = reg.nodes_from_payload(payload)
    reordered = list(reversed(nodes))
    assert reg.checksum_for(nodes) == reg.checksum_for(reordered)
    renamed = copy.deepcopy(nodes)
    renamed[0]["name"] = renamed[0]["name"] + " (renamed)"
    assert reg.checksum_for(renamed) != reg.checksum_for(nodes)


# --- import -----------------------------------------------------------------


def test_bundled_import_matches_the_knowledge_json(bundled):
    payload = ik.load_industry_knowledge()
    assert bundled.version_key == payload["taxonomy_version"]
    assert bundled.checksum == reg.checksum_for(reg.nodes_from_payload(payload))
    assert bundled.source == reg.SOURCE_LABEL
    assert bundled.provenance["map_source_sha256"] == payload["map_source_sha256"]
    assert bundled.provenance["map_as_of"] == payload["map_as_of"]
    assert bundled.effective_from is not None

    expected = _tree_counts(payload)
    assert reg.counts(bundled) == expected
    assert bundled.node_counts == expected
    with SessionLocal() as db:
        stored = db.execute(
            select(func.count(GicsNode.id)).where(GicsNode.taxonomy_version_id == bundled.id)
        ).scalar_one()
    assert stored == sum(expected.values())


def test_reimport_of_the_same_structure_is_a_no_op(bundled):
    with SessionLocal() as db:
        before = db.execute(select(func.count(GicsNode.id))).scalar_one()
    result = reg.import_from_knowledge_json()
    assert result["imported"] is False
    assert result["nodes_inserted"] == 0
    assert result["version_id"] == bundled.id
    assert result["checksum"] == bundled.checksum
    assert result["activated"] is True  # was already active; import never deactivates
    with SessionLocal() as db:
        after = db.execute(select(func.count(GicsNode.id))).scalar_one()
    assert after == before


def test_a_changed_structure_under_an_existing_key_is_refused():
    key = f"test-{uuid.uuid4().hex[:8]}"
    try:
        first = reg.import_from_knowledge_json(payload=_payload_under_key(key))
        assert first["imported"] is True and first["activated"] is False

        changed = _payload_under_key(key)
        changed["sectors"][0]["industry_groups"][0]["name"] += " (renamed)"
        with pytest.raises(reg.TaxonomyChecksumMismatch, match="new version key"):
            reg.import_from_knowledge_json(payload=changed)
        # ...and nothing was written for the refused payload.
        assert reg.resolve_version(key).checksum == first["checksum"]
    finally:
        _drop_version(key)


def test_import_refuses_an_empty_or_unversioned_payload():
    with pytest.raises(ValueError, match="taxonomy_version"):
        reg.import_from_knowledge_json(payload={"sectors": []})
    with pytest.raises(ValueError, match="no taxonomy nodes"):
        reg.import_from_knowledge_json(payload={"taxonomy_version": "x", "sectors": []})


# --- activation and the two-process rule -----------------------------------


def test_activation_switches_exactly_one_active_version(bundled, restore_active):
    key = f"test-{uuid.uuid4().hex[:8]}"
    try:
        reg.import_from_knowledge_json(payload=_payload_under_key(key))
        assert _active_rows() == [bundled.version_key]

        info = reg.activate_version(key)
        assert info.is_active and info.activated_at is not None
        assert _active_rows() == [key]
        assert reg.active_version().version_key == key

        reg.activate_version(bundled.version_key)
        assert _active_rows() == [bundled.version_key]
    finally:
        reg.activate_version(bundled.version_key)
        _drop_version(key)


def test_active_version_is_read_from_the_database_every_call(bundled, restore_active):
    """A flip made by the *other* process (simulated with a raw UPDATE,
    bypassing this module) must be visible on the next read without any
    cache invalidation."""
    key = f"test-{uuid.uuid4().hex[:8]}"
    try:
        alt = reg.import_from_knowledge_json(payload=_payload_under_key(key))
        assert reg.active_version().id == bundled.id
        with SessionLocal() as db:
            db.execute(update(TaxonomyVersion).values(is_active=False))
            db.execute(
                update(TaxonomyVersion)
                .where(TaxonomyVersion.id == alt["version_id"])
                .values(is_active=True)
            )
            db.commit()
        assert reg.active_version().version_key == key
        # The node cache is keyed by version id, so reads follow the flip too.
        assert reg.counts()["industry_group"] == reg.counts(bundled)["industry_group"]
    finally:
        reg.activate_version(bundled.version_key)
        _drop_version(key)


def test_require_active_version_raises_when_nothing_is_active(bundled, restore_active):
    with SessionLocal() as db:
        db.execute(update(TaxonomyVersion).values(is_active=False))
        db.commit()
    assert reg.active_version() is None
    with pytest.raises(reg.TaxonomyNotImported, match="not imported"):
        reg.require_active_version()
    with pytest.raises(reg.TaxonomyNotImported):
        reg.industry_groups()


def test_ensure_taxonomy_activates_the_bundled_version_when_nothing_is_active(bundled, restore_active):
    with SessionLocal() as db:
        db.execute(update(TaxonomyVersion).values(is_active=False))
        db.commit()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None and info.id == bundled.id and info.is_active
    # A second call never re-imports or re-activates anything.
    assert reg.ensure_taxonomy(activate=True).id == bundled.id


def _renamed_payload() -> dict:
    """The bundled payload with one industry group renamed — the same
    version key, a different structure; what a regenerated map ships."""
    payload = copy.deepcopy(ik.load_industry_knowledge())
    payload["sectors"][0]["industry_groups"][0]["name"] += " (renamed)"
    return payload


def test_bundled_drift_is_none_while_the_registry_matches_the_json(bundled):
    assert reg.bundled_drift(bundled) is None
    assert reg.bundled_drift() is None  # resolves the active version itself


def test_bundled_drift_names_a_changed_structure_under_the_same_key(bundled):
    payload = _renamed_payload()
    drift = reg.bundled_drift(bundled, payload=payload)
    assert drift is not None
    assert drift["kind"] == "same_key_changed_structure"
    assert drift["active_version_key"] == drift["bundled_version_key"] == bundled.version_key
    assert drift["active_checksum"] == bundled.checksum
    assert drift["bundled_checksum"] == reg.checksum_for(reg.nodes_from_payload(payload))
    assert drift["bundled_checksum"] != drift["active_checksum"]
    assert drift["active_node_counts"] == bundled.node_counts
    assert drift["bundled_node_counts"] == bundled.node_counts  # a rename changes no count
    assert "--version-key" in drift["remedy"]

    payload["taxonomy_version"] = bundled.version_key + "-next"
    other = reg.bundled_drift(bundled, payload=payload)
    assert other is not None and other["kind"] == "bundled_version_not_active"
    assert other["bundled_version_key"] == bundled.version_key + "-next"
    assert "--activate" in other["remedy"]


def test_bundled_drift_needs_an_active_version(bundled, restore_active):
    with SessionLocal() as db:
        db.execute(update(TaxonomyVersion).values(is_active=False))
        db.commit()
    assert reg.bundled_drift() is None


def test_ensure_taxonomy_warns_when_the_bundled_json_drifted(bundled, monkeypatch, caplog):
    """Regression: a regenerated JSON under the active key used to be
    returned as the active version with no signal at all."""
    monkeypatch.setattr(reg, "_load_payload", lambda path: _renamed_payload())
    with caplog.at_level("WARNING", logger=reg.__name__):
        info = reg.ensure_taxonomy(activate=True)
    assert info is not None and info.id == bundled.id  # never re-imports or flips
    assert reg.active_version().checksum == bundled.checksum
    messages = [r.getMessage() for r in caplog.records if "taxonomy drift" in r.getMessage()]
    assert len(messages) == 1
    assert bundled.version_key in messages[0] and "same_key_changed_structure" in messages[0]
    assert "--version-key" in messages[0]

    caplog.clear()
    monkeypatch.setattr(reg, "_load_payload", lambda path: copy.deepcopy(ik.load_industry_knowledge()))
    with caplog.at_level("WARNING", logger=reg.__name__):
        reg.ensure_taxonomy(activate=True)
    assert not [r for r in caplog.records if "taxonomy drift" in r.getMessage()]


def test_unknown_version_raises():
    with pytest.raises(reg.UnknownTaxonomyVersion):
        reg.resolve_version("no-such-version")
    with pytest.raises(reg.UnknownTaxonomyVersion):
        reg.resolve_version(-1)
    with pytest.raises(reg.UnknownTaxonomyVersion):
        reg.activate_version("no-such-version")


# --- reads ------------------------------------------------------------------


def test_group_reads_come_from_the_registry(bundled):
    payload = ik.load_industry_knowledge()
    expected = _tree_counts(payload)

    groups = reg.industry_groups()
    assert len(groups) == expected["industry_group"]
    assert [g.code for g in groups] == [
        g["code"] for s in payload["sectors"] for g in s["industry_groups"]
    ]
    sectors = {s.code for s in reg.sectors()}
    assert len(sectors) == expected["sector"]
    for g in groups:
        assert g.level == "industry_group" and g.parent_code in sectors
        assert g.sector_code == g.parent_code
        assert g.industry_group_code == g.code and g.industry_code is None

    first = groups[0]
    assert reg.group(first.code) == first
    assert reg.group(first.code, version=bundled.version_key) == first
    assert reg.group(first.code, version=bundled.id) == first
    with pytest.raises(reg.UnknownNode, match="not found in taxonomy"):
        reg.group("0000")
    industries = reg.industries_of_group(first.code)
    assert industries and all(i.level == "industry" and i.code[:4] == first.code for i in industries)
    with pytest.raises(reg.UnknownNode):
        reg.group(industries[0].code)  # an industry code is not a group


def test_node_lookups_by_code_length(bundled):
    subs = reg.sub_industries_of(reg.industry_groups()[0].code)
    assert subs and all(n.level == "sub_industry" for n in subs)
    sub = subs[0]
    assert reg.node(sub.code) == sub
    assert reg.node(sub.code[:6]).level == "industry"
    assert reg.node(sub.code[:4]).level == "industry_group"
    assert reg.node(sub.code[:2]).level == "sector"
    assert reg.group_code_for(sub.code) == sub.code[:4]
    assert reg.group_code_for(sub.code[:6]) == sub.code[:4]
    assert reg.group_code_for("10") is None
    assert reg.node("") is None
    assert reg.node("12345") is None  # odd length is not a GICS code
    assert reg.node("abcd") is None
    assert reg.node("99999999") is None
    assert reg.children(sub.code[:6]) == reg.sub_industries_of(sub.code[:6])
    assert sub.as_dict()["display"] == f"{sub.name} ({sub.code})"


def test_retired_sub_industries_are_inactive_nodes(bundled):
    retired = ik.retired_sub_industries()
    assert retired, "the map records retired rows; the registry must carry them"
    for row in retired:
        assert reg.node(row["code"]) is None, row
        node = reg.node(row["code"], include_inactive=True)
        assert node is not None and node.is_active is False
        assert node.level == "sub_industry"
        assert node.parent_code == row["parent_code"] == row["code"][:6]
        assert node.effective_to == bundled.effective_from
        assert node.attributes["status"] == "discontinued"
        assert row["code"] not in {n.code for n in reg.sub_industries_of(row["code"][:6])}
        assert row["code"] in {
            n.code for n in reg.sub_industries_of(row["code"][:6], include_inactive=True)
        }
    active_subs = reg.nodes("sub_industry")
    assert len(active_subs) == ik.load_industry_knowledge()["sub_industry_count"]
    assert len(reg.nodes("sub_industry", include_inactive=True)) == len(active_subs) + len(retired)


def test_tree_nests_four_levels_with_attribution(bundled):
    tree = reg.tree()
    version = tree["taxonomy_version"]
    assert version["key"] == bundled.version_key and version["is_active"] is True
    assert "MSCI" in version["attribution"] and "S&P" in version["attribution"]
    assert version["mapping_caveat"] == reg.MAPPING_CAVEAT
    assert version["display_mode"] == "codes_and_names"

    sectors = tree["sectors"]
    assert len(sectors) == reg.counts()["sector"]
    groups = [g for s in sectors for g in s["industry_groups"]]
    industries = [i for g in groups for i in g["industries"]]
    subs = [x for i in industries for x in i["sub_industries"]]
    counts = reg.counts()
    assert (len(groups), len(industries), len(subs)) == (
        counts["industry_group"], counts["industry"], counts["sub_industry"],
    )
    assert all(x["is_active"] for x in subs)  # retired rows are not rendered


def _node(code: str, name: str) -> reg.NodeInfo:
    return reg.NodeInfo(
        code=code, name=name, level=reg.LEVEL_BY_CODE_LENGTH[len(code)], parent_code=code[:-2] or None,
        is_active=True, effective_from=None, effective_to=None, sort_order=1,
    )


def test_display_mode_is_validated(monkeypatch):
    assert reg.display_mode() == "codes_and_names"
    monkeypatch.setattr(reg.settings, "gics_display_mode", "internal_labels")
    assert reg.display_mode() == "internal_labels"
    monkeypatch.setattr(reg.settings, "gics_display_mode", "bogus")
    assert reg.display_mode() == "codes_and_names"
    assert reg.display(_node("4530", "Semis")) == "Semis (4530)"


def test_internal_labels_mode_displays_our_label_and_names_no_lower_level(monkeypatch):
    """Owner decision 2026-09-24: `internal_labels` is built — it renders
    MarketMosaic's own label. Industries and sub-industries are never
    named publicly, so they display as the group they roll up to."""
    from app.services import industry_labels as il

    monkeypatch.setattr(reg.settings, "gics_display_mode", "internal_labels")
    assert reg.display(_node("4530", "Semiconductors & Semiconductor Equipment")) == il.label("4530")
    assert reg.display(_node("45", "Information Technology")) == il.label("45")
    assert reg.display(_node("453010", "Semiconductors & Semiconductor Equipment")) == il.label("4530")
    assert reg.display(_node("45301020", "Semiconductors")) == il.label("4530")


def test_codes_and_names_mode_warns_once_per_process(monkeypatch, caplog):
    """The code-and-name rendering carries licensed names; it still works
    (admin/internal) but says so — once, not once per row."""
    monkeypatch.setattr(reg, "_CODES_AND_NAMES_WARNED", False)
    with caplog.at_level("WARNING", logger=reg.log.name):
        assert reg.display(_node("4530", "Semis")) == "Semis (4530)"
        reg.display(_node("4510", "Soft"))
    warnings = [r for r in caplog.records if "codes_and_names" in r.getMessage()]
    assert len(warnings) == 1


# --- the CLI ----------------------------------------------------------------


def test_import_cli_is_idempotent_and_reports_status(bundled, capsys):
    from app.scripts import import_gics_taxonomy as cli

    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert '"imported": false' in out

    assert cli.main(["--status"]) == 0
    status = capsys.readouterr().out
    assert bundled.version_key in status
    assert '"industry_group"' in status and '"classification"' in status
