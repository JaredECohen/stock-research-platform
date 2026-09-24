"""Stored legacy cases remain readable; malformed new patches cannot publish."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.agents import news_impact_agent
from app.models import MemoSnapshot
from app.services import memo_store, update_orchestrator
from app.tests.test_update_orchestrator import _stub_alert, _stub_memo


@pytest.fixture
def memo_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-memos.db'}")
    MemoSnapshot.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(memo_store, "SessionLocal", factory)
    yield factory
    engine.dispose()


def _legacy(factory, *, field="bull_case", value=None):
    payload = _stub_memo("ABBV").model_dump(mode="json")
    payload[field] = value if value is not None else [{"key_point": "Original stored point."}]
    with factory() as db:
        row = MemoSnapshot(ticker="ABBV", version=7, trigger="incremental_patch", memo_json=payload)
        db.add(row)
        db.commit()
        return row.id, deepcopy(payload)


@pytest.mark.parametrize("field", ["bull_case", "bear_case"])
@pytest.mark.parametrize("value", [
    ["exact text", "  spaces remain  "], [{"key_point": "exact text"}], [],
    [{"key_point": "exact text", "evidence": {"source": "Original source", "value": 12},
      "headline": "A point-level label cannot supply the whole case headline."}],
])
def test_legacy_projection_preserves_text_and_original_row(memo_db, field, value):
    row_id, original = _legacy(memo_db, field=field, value=value)
    with pytest.raises(ValidationError):
        # Shape from ABBV's web traceback; its complete case text was unavailable.
        type(_stub_memo()).model_validate(original)
    snap = memo_store.latest_memo("ABBV")
    read = memo_store.memo_to_pydantic(snap)
    expected = value if all(isinstance(p, str) for p in value) else [p["key_point"] for p in value]
    assert getattr(read, field).headline == ""
    assert getattr(read, field).key_points == expected
    assert read.degraded_agents == ["Stored memo compatibility"]
    assert read.degradation_events[0] == {
        "agent": "Stored memo compatibility", "error_type": "LegacyCaseShape",
        "message": f"{field} was a legacy list; exact points retained, headline unavailable.",
        "field": field, "source_snapshot_id": row_id, "source_snapshot_version": 7,
        "source_snapshot_ticker": "ABBV", "original_value": value,
    }
    assert snap.memo_json == original
    with memo_db() as db:
        assert db.get(MemoSnapshot, row_id).memo_json == original
        assert len(db.scalars(select(MemoSnapshot)).all()) == 1


@pytest.mark.parametrize("value", [
    ["mixed", {"key_point": "shape"}], [{"key_point": 123}], [None], [{"unexpected": "point"}],
    [{"key_point": None}],
])
def test_ambiguous_legacy_cases_still_fail_without_rewriting(memo_db, value):
    # FIX-004 residual: still refused (owner ruling, never coerced), but as a
    # typed error naming the row and field path rather than a raw
    # ValidationError that quotes the stored text and 500s the memo route.
    row_id, original = _legacy(memo_db, value=value)
    with pytest.raises(memo_store.StoredMemoUnreadable) as info:
        memo_store.memo_to_pydantic(memo_store.latest_memo("ABBV"))
    assert isinstance(info.value, ValueError)
    assert (info.value.ticker, info.value.version, info.value.snapshot_id) == ("ABBV", 7, row_id)
    assert info.value.fields == ("bull_case",)
    assert str(info.value) == f"stored memo ABBV v7 (snapshot {row_id}) does not validate at bull_case"
    with memo_db() as db:
        assert db.get(MemoSnapshot, row_id).memo_json == original


def test_news_alert_on_unreadable_prior_raises_typed_error_and_writes_nothing(memo_db, monkeypatch):
    # news_loop's per-ticker catch-all records this as an update failure named
    # by type; the alert must never reach the LLM or publish a patched version.
    row_id, original = _legacy(memo_db, value=["text", {"key_point": "x"}])

    def assess(*_a):
        raise AssertionError("an unreadable prior memo must not be assessed")

    monkeypatch.setattr(news_impact_agent, "assess", assess)
    with pytest.raises(memo_store.StoredMemoUnreadable):
        update_orchestrator.on_news_alert("ABBV", _stub_alert())
    with memo_db() as db:
        rows = db.scalars(select(MemoSnapshot)).all()
        assert [r.id for r in rows] == [row_id]
        assert rows[0].memo_json == original


def test_typed_case_is_unchanged_and_not_marked_degraded(memo_db):
    memo = _stub_memo("ABBV")
    snap = memo_store.save_memo(memo)
    assert memo_store.memo_to_pydantic(snap).model_dump() == memo.model_dump()


def test_news_update_reads_legacy_prior_and_publishes_only_new_valid_version(memo_db, monkeypatch):
    row_id, original = _legacy(memo_db)
    seen = []

    def assess(prior, alert):
        seen.append(prior)
        return {"material": True, "patch": {"bull_case": {"key_points": ["New exact point."]}},
                "rationales": {"bull_case": "A new disclosed fact."}, "delta_summary": "New point added."}

    monkeypatch.setattr(news_impact_agent, "assess", assess)
    result = update_orchestrator.on_news_alert("ABBV", _stub_alert())
    assert result["patched"] is True and result["version"] == 8
    assert seen[0].bull_case.key_points == ["Original stored point."]
    new = memo_store.memo_to_pydantic(memo_store.latest_memo("ABBV"))
    assert new.bull_case.key_points == ["Original stored point.", "New exact point."]
    assert new.bull_case.headline == ""
    assert new.rating_label == original["rating_label"]
    assert new.confidence_score == original["confidence_score"]
    assert new.degradation_events[0]["source_snapshot_id"] == row_id
    assert new.degradation_events[0]["original_value"] == original["bull_case"]
    with memo_db() as db:
        assert db.get(MemoSnapshot, row_id).memo_json == original
        assert [r.version for r in db.scalars(select(MemoSnapshot).order_by(MemoSnapshot.version))] == [7, 8]


@pytest.mark.parametrize("patch", [
    {"bull_case": [{"key_point": "Malformed new LLM case."}]},
    {"bear_case": ["A list is still a malformed new patch."]},
    {"bull_case": {"key_points": "Do not split into letters."}},
    {"bear_case": {"key_points": [{"key_point": "Do not stringify a dictionary."}]}},
    {"bull_case": {"headline": 5}}, {"bull_case": {"key_point": "Unknown singular key."}},
    {"rating_label": "Not a rating"}, {"final_pm_view": ["Not a string"]},
])
def test_invalid_new_patch_never_publishes(memo_db, monkeypatch, patch):
    saved = memo_store.save_memo(_stub_memo("ABBV"))
    original = deepcopy(saved.memo_json)
    monkeypatch.setattr(news_impact_agent, "assess", lambda *a: {
        "material": True, "patch": patch, "rationales": {k: "reason" for k in patch},
    })
    with pytest.raises((ValueError, ValidationError)):
        update_orchestrator.on_news_alert("ABBV", _stub_alert())
    with memo_db() as db:
        rows = db.scalars(select(MemoSnapshot)).all()
        assert len(rows) == 1 and rows[0].memo_json == original


@pytest.mark.parametrize("field,value", [
    ("bull_case", [{"key_point": "Do not normalize new publication."}]),
    ("rating_label", "Not a rating"),
])
def test_publication_revalidates_assignment_before_opening_database(monkeypatch, field, value):
    memo = _stub_memo()
    setattr(memo, field, value)  # Pydantic assignment itself does not reject this.
    calls = []
    monkeypatch.setattr(memo_store, "SessionLocal", lambda: calls.append(True))
    with pytest.raises(ValidationError):
        memo_store.save_memo(memo)
    assert calls == []
