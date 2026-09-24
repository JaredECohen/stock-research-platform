"""Memo contract C1 (S2, 2026-09-24): the expand-only half of W2a + W2b.

This slice adds every `StockMemoOut` field W2a (hide template-filled sections)
and W2b (research-quality guards) need, and nothing writes them yet. That
ordering is the point: the widened `ValuationVerdict.verdict` ("mixed") and
the new optional records are live at least one deploy before any code stores
them, so reverting a later writer can never strand stored memos as
`memo_unreadable`. These tests pin:

  * old stored shapes (the pre-S2 payload, the ABBV v7 legacy lists) read
    exactly as before, and the ambiguous shapes are still refused;
  * the rollback direction: a memo stored by a W2b writer ("mixed",
    evidence basis, a full quality record) reads back under this schema;
  * `save_memo` refuses a presented memo (non-empty `section_availability`)
    before any DB work, and never persists the read-time map, while the
    write-time `section_provenance` and `quality` are persisted.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models import MemoSnapshot
from app.schemas import (
    ConfidenceAssessment,
    ConfidenceCap,
    CriticReview,
    MemoQuality,
    NumberCheck,
    NumberClaim,
    RatingReconciliation,
    SectionAvailability,
    StockMemoOut,
    ValuationVerdict,
    WithheldItem,
)
from app.services import memo_store, public_samples, regen_lease
from app.tests.gating_helpers import purge_memos
from app.tests.test_memo_store import _stub_memo
from app.tests.test_memo_unreadable import ABBV_BEAR, AMBIGUOUS, LINZESS

# A memo exactly as `save_memo` stored it on 0c5b491, before this slice:
# synthetic values, captured with `model_dump(mode="json")` at that commit.
PRE_S2 = Path(__file__).parent / "fixtures" / "memo_contract" / "pre_s2_memo.json"

NEW_TOP_LEVEL = {"section_provenance", "section_availability", "quality"}

TICKERS = ("ZZC1QUAL", "ZZC1PRES", "ZZC1BACK", "ZZC1MIX", "ZZC1OWN", "ZZC1SAMP")


@pytest.fixture(autouse=True)
def _purge_rows():
    purge_memos(*TICKERS)
    yield
    purge_memos(*TICKERS)


def _snap(payload: Any, *, ticker: str = "ZZPRE", version: int = 1) -> MemoSnapshot:
    """An in-memory row, as `memo_to_pydantic` receives one from the DB."""
    return MemoSnapshot(id=1, ticker=ticker, version=version, trigger="full_reanalysis",
                        memo_json=payload)


def _pre_s2() -> dict[str, Any]:
    return json.loads(PRE_S2.read_text())


def _assert_new_fields_default(memo: StockMemoOut) -> None:
    assert memo.section_provenance == {}
    assert memo.section_availability == {}
    assert memo.quality is None
    # "rating" is the truth for every stored verdict: all of them were
    # derived from the rating by `graph._verdict_word`.
    assert memo.valuation_verdict.basis == "rating"
    assert memo.valuation_verdict.signals == {}
    assert memo.risk_committee_challenge.valuation_divergence_assessment == "not_assessed"


def _full_quality() -> MemoQuality:
    claim = NumberClaim(field="one_sentence_thesis", start=4, end=11, raw="$12.3B",
                        value=12.3e9, unit="usd", status="untraceable",
                        source_refs=["filing:10-K:2025"])
    return MemoQuality(
        number_check=NumberCheck(
            checked=True, counts={"traced": 3, "untraceable": 1},
            claims=[claim],
            withheld=[WithheldItem(field="bull_case.key_points", index=1, text="t", claims=[claim])],
            lists_not_withheld=["catalysts"], unchecked_fields=["key_risks[2]"],
            sources_cited=["filing:10-K:2025"], primary_kinds_cited=["filing"],
            assumptions=[{"value": 0.05, "unit": "pct", "basis_ref": "guidance:FY26"}],
            notes=["n"],
        ),
        rating_reconciliation=RatingReconciliation(
            outcome="downgraded", pm_rating="Bullish", pm_confidence=72.0,
            blended_rating="Bullish", final_rating="Neutral", valuation_verdict="overvalued",
            divergence=True, reason="", reason_checks={"substantive": False},
            critic_assessment="unsupported", note="rating set to Neutral",
        ),
        confidence=ConfidenceAssessment(
            raw=72.0, final=55.0,
            caps=[ConfidenceCap(code="divergence_unreviewed", cap=55.0, detail="d"),
                  ConfidenceCap(code="critic_not_live", cap=60.0)],
            binding="divergence_unreviewed",
        ),
    )


# ---------------------------------------------------------------------------
# 1. Old rows still read
# ---------------------------------------------------------------------------

def test_stored_shapes_validate_unchanged():
    original = _pre_s2()
    assert not NEW_TOP_LEVEL & set(original), "fixture must be the pre-S2 shape"
    assert "basis" not in original["valuation_verdict"]
    assert "valuation_divergence_assessment" not in original["risk_committee_challenge"]

    memo = memo_store.memo_to_pydantic(_snap(deepcopy(original)))
    _assert_new_fields_default(memo)
    dumped = memo.model_dump(mode="json")
    # Every stored value reads back exactly; the only additions are the new
    # defaulted fields, top level and nested.
    for key, value in original.items():
        if key in ("valuation_verdict", "risk_committee_challenge"):
            assert {k: dumped[key][k] for k in value} == value, key
        else:
            assert dumped[key] == value, key
    assert set(dumped) - set(original) == NEW_TOP_LEVEL
    assert set(dumped["valuation_verdict"]) - set(original["valuation_verdict"]) == {"basis", "signals"}
    assert (set(dumped["risk_committee_challenge"]) - set(original["risk_committee_challenge"])
            == {"valuation_divergence_assessment"})

    # ABBV v7: the two unambiguous legacy case lists still project as before.
    legacy = deepcopy(original)
    legacy["bull_case"] = [{"key_point": LINZESS}]
    legacy["bear_case"] = ABBV_BEAR
    abbv = memo_store.memo_to_pydantic(_snap(legacy, ticker="ZZABBV", version=7))
    assert abbv.bull_case.model_dump() == {"headline": "", "key_points": [LINZESS]}
    assert abbv.bear_case.model_dump() == ABBV_BEAR
    assert abbv.degraded_agents == ["Stored memo compatibility"]
    _assert_new_fields_default(abbv)

    # The ambiguous shapes stay refused: widening the contract must not have
    # loosened anything that FIX-004 decided to reject.
    for value in AMBIGUOUS:
        bad = deepcopy(original)
        bad["bull_case"] = value
        with pytest.raises(memo_store.StoredMemoUnreadable) as exc:
            memo_store.memo_to_pydantic(_snap(bad))
        assert exc.value.fields == ("bull_case",)


_REVIEWS = Path(os.environ.get(
    "MM_REVIEWS_DIR", Path(__file__).resolve().parents[3] / "docs" / "reviews",
))
_REVIEW_BODIES = {
    "2026-09-13-META-v1.json": lambda o: {"META": o},
    "2026-09-13-ABBV-v7-compatibility.json": lambda o: {"ABBV": o["post_deploy_response"]},
    "2026-09-13-baseline-memo-evidence.json":
        lambda o: {t: entry["memo"] for t, entry in o["memos"].items()},
}


@pytest.mark.skipif(not all((_REVIEWS / f).is_file() for f in _REVIEW_BODIES),
                    reason="local-only production evidence (docs/reviews is not in the public repo)")
def test_production_evidence_bodies_validate_unchanged():
    """The saved production bodies (AAPL, AMZN, AVGO, MSFT, GOOGL, META,
    ABBV v7) validate under the widened contract with the new fields at
    their defaults. Local-only: the files hold production prose."""
    seen = []
    for name, pick in _REVIEW_BODIES.items():
        for ticker, body in pick(json.loads((_REVIEWS / name).read_text())).items():
            assert not NEW_TOP_LEVEL & set(body), ticker
            memo = StockMemoOut.model_validate(deepcopy(body))
            _assert_new_fields_default(memo)
            assert memo.valuation_verdict.verdict == body["valuation_verdict"]["verdict"]
            seen.append(ticker)
    assert len(seen) == 7, seen


# ---------------------------------------------------------------------------
# 2. Rollback safety: what a W2b writer will store reads under this schema
# ---------------------------------------------------------------------------

def test_valuation_verdict_accepts_mixed_and_defaults_basis_rating():
    assert ValuationVerdict().basis == "rating"
    assert ValuationVerdict().verdict == "fairly_priced"
    mixed = ValuationVerdict(verdict="mixed", basis="evidence",
                             signals={"comps": -1, "valuation_family": 1, "dcf_final_upside": 0.31})
    assert mixed.verdict == "mixed"
    with pytest.raises(ValidationError):
        ValuationVerdict(verdict="cheap")
    with pytest.raises(ValidationError):
        ValuationVerdict(basis="vibes")
    assert CriticReview(overall_assessment="ok").valuation_divergence_assessment == "not_assessed"
    with pytest.raises(ValidationError):
        CriticReview(overall_assessment="ok", valuation_divergence_assessment="maybe")

    # A memo stored by the (later) evidence-verdict writer, then read back
    # after that writer is reverted but this schema stays: still readable.
    memo = _stub_memo("ZZC1MIX")
    memo.valuation_verdict = mixed
    memo.risk_committee_challenge = CriticReview(
        overall_assessment="ok", review_mode="live", valuation_divergence_assessment="supported",
    )
    snap = memo_store.save_memo(memo)
    stored = memo_store.memo_to_pydantic(memo_store.memo_version("ZZC1MIX", snap.version))
    assert stored.valuation_verdict == mixed
    assert stored.risk_committee_challenge.valuation_divergence_assessment == "supported"


def test_quality_optional_roundtrip():
    # Absent on the stub (no writer sets it in this slice) and stored as null.
    plain = memo_store.save_memo(_stub_memo("ZZC1QUAL"))
    assert plain.memo_json["quality"] is None
    assert memo_store.memo_to_pydantic(plain).quality is None

    memo = _stub_memo("ZZC1QUAL")
    memo.quality = _full_quality()
    memo.section_provenance = {"v": 1, "llm_configured": True, "thesis": "pm",
                               "mispricing": "fallback", "confidence": "earned"}
    snap = memo_store.save_memo(memo)
    back = memo_store.memo_to_pydantic(memo_store.memo_version("ZZC1QUAL", snap.version))
    assert back.quality == _full_quality()
    assert back.section_provenance == memo.section_provenance
    # And through the JSON wire, as the API and the frontend see it.
    assert StockMemoOut.model_validate_json(back.model_dump_json()).quality == _full_quality()
    # Nested records stay optional on their own.
    assert MemoQuality().model_dump() == {
        "v": 1, "number_check": None, "rating_reconciliation": None, "confidence": None,
    }
    assert NumberCheck().checked is False


# ---------------------------------------------------------------------------
# 3. save_memo and the read-time map
# ---------------------------------------------------------------------------

def _row_count(ticker: str) -> int:
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(MemoSnapshot)
                         .where(MemoSnapshot.ticker == ticker)) or 0


def test_save_memo_refuses_presented_memo():
    memo = _stub_memo("ZZC1PRES")
    memo.section_availability = {
        "final_pm_view": SectionAvailability(status="unavailable", reason="template_fallback",
                                             basis=["signature:pm_view_tail"]),
    }
    with pytest.raises(ValueError, match="presented memo") as exc:
        memo_store.save_memo(memo)
    assert not isinstance(exc.value, memo_store.StoredMemoUnreadable)
    assert _row_count("ZZC1PRES") == 0

    # A presented memo arriving as a plain dict map (assignment bypasses
    # validation) is caught too, because save_memo re-validates first.
    memo.section_availability = {"bull_case": {"status": "degraded", "headline_hidden": True}}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="presented memo"):
        memo_store.save_memo(memo)
    assert _row_count("ZZC1PRES") == 0


def test_save_memo_refuses_presented_memo_before_any_db_work(monkeypatch):
    # The refusal must come before the lease check and the insert, not merely
    # before the commit: a caller that passes its own session and commits
    # after catching the error would otherwise persist the presented memo.
    def _db_work(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("save_memo touched the DB before refusing a presented memo")

    monkeypatch.setattr(regen_lease, "assert_current", _db_work)
    monkeypatch.setattr(memo_store, "_next_version", _db_work)
    memo = _stub_memo("ZZC1OWN")
    memo.section_availability = {
        "bull_case": SectionAvailability(status="degraded", reason="template_fallback"),
    }
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="presented memo"):
            memo_store.save_memo(memo, db=db)
        assert not db.new and not db.dirty
        db.commit()
    assert _row_count("ZZC1OWN") == 0


def test_public_sample_payload_carries_the_presented_map():
    # public_samples stores a memo dump of its own. The W2a serve path treats
    # a sample row WITHOUT the key as built before presentation and presents
    # it. Since S11 the build step stores the PRESENTED copy, so the key is
    # present and never empty: an empty map would pass for "already
    # presented" while hiding nothing. (This supersedes S2's pin that the key
    # was absent, which held only while nothing presented the sample.)
    memo = _stub_memo("ZZC1SAMP")
    memo.section_provenance = {"v": 1, "llm_configured": True, "thesis": "pm"}
    memo_store.save_memo(memo)
    with SessionLocal() as db:
        payload, source_ref, degraded = public_samples._build_memo("ZZC1SAMP", db)
    assert payload is not None and degraded == []
    assert source_ref is not None and source_ref.startswith("memo_snapshot:")
    assert payload["section_availability"]
    # Write-time facts still reach the sample, and it still reads as a memo.
    assert payload["section_provenance"] == memo.section_provenance
    assert "quality" in payload
    assert StockMemoOut.model_validate(payload).section_availability


def test_save_memo_never_persists_section_availability():
    memo = _stub_memo("ZZC1BACK")
    memo.section_provenance = {"v": 1, "llm_configured": False}
    snap = memo_store.save_memo(memo)
    with SessionLocal() as db:
        stored = db.scalars(select(MemoSnapshot).where(MemoSnapshot.id == snap.id)).one().memo_json
    # Write-time facts persist; the read-time map is never stored, not even
    # empty, so no stored value can be mistaken for a current verdict.
    assert "section_availability" not in stored
    assert stored["section_provenance"] == {"v": 1, "llm_configured": False}
    assert "quality" in stored
    assert memo_store.memo_to_pydantic(snap).section_availability == {}
