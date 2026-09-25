"""D2 (2026-09-25) — the bull/bear debate record's contract.

`StockMemoOut.debate` ships expand-only, a deploy wave before the debate
engine (D3) and the graph wiring (D6) write it behind `DEBATE_MODE`. These
tests pin what that ordering relies on:

  * every stored memo, none of which has a `debate` key, reads back with
    `debate=None` and nothing else moved (so reverting a later writer can
    never strand a stored memo as `memo_unreadable`);
  * a fully populated record survives `save_memo` -> the DB -> the reader
    byte-for-byte, including the pieces later slices depend on: a refusal
    recorded in `route.phases` (P16) and the L1 counterfactual's non-integer
    `cf_*` values in `outcome`;
  * the closed vocabularies refuse off-list values instead of coercing them.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.models import MemoSnapshot
from app.schemas import (
    DebateClaim,
    DebateEvidence,
    DebateRecord,
    DebateResolution,
    DebateResponse,
    DebateReview,
    DebateRuling,
    StockMemoOut,
)
from app.schemas.agents import DEBATE_EXCERPT_MAX_CHARS
from app.services import memo_store
from app.tests.gating_helpers import purge_memos
from app.tests.test_memo_store import _stub_memo

# A memo exactly as `save_memo` stored it before S2 (and so before D2).
PRE_S2 = Path(__file__).parent / "fixtures" / "memo_contract" / "pre_s2_memo.json"
TICKER = "ZZD2DEB"


@pytest.fixture(autouse=True)
def _purge_rows():
    purge_memos(TICKER)
    yield
    purge_memos(TICKER)


def populated_record() -> DebateRecord:
    """A debate that exercised every part of the record: two evidence items,
    a claim per side (one dropped), rebuttals, a PM ruling, a refusal on the
    first route and the pair failover that recovered, a news patch, and the
    counterfactual PM's outcome."""
    return DebateRecord(
        status="partial",
        reason="rebuttals_unavailable",
        rebuttal_status="unavailable",
        research_status="directed",
        presentation_order="bear_first",
        route={
            "provider": "openai", "model": "gpt-6-sol", "effort": "high", "failed_over": True,
            "phases": [
                {"phase": "research", "provider": "anthropic", "model": "claude-opus-5-5",
                 "status": "refused:bio", "calls": 2},
                {"phase": "research", "provider": "openai", "model": "gpt-6-sol",
                 "status": "ok", "calls": 2},
                {"phase": "openings", "provider": "openai", "model": "gpt-6-sol",
                 "status": "ok", "calls": 2},
                {"phase": "rebuttals", "status": "unavailable", "calls": 0},
            ],
        },
        headlines={"bull": "Share gains outrun the multiple", "bear": "Capex cycle peaks in FY27"},
        cruxes={"bull": "Networking attach rate", "bear": "Hyperscaler capex duration"},
        research={"bull": [{"query": "networking revenue share", "why": "attach"}],
                  "bear": [{"query": "capex guidance cuts", "why": "cycle"}]},
        evidence=[
            DebateEvidence(id="E01", kind="filing", ref="chunk:0001045810-25-000023:mdna:3f2a",
                           title="10-K MD&A", date="2025-02-26", excerpt="Networking revenue grew.",
                           found_by=["bull"], query="networking revenue share"),
            DebateEvidence(id="E02", kind="news", ref="news:abc123", title="Capex guide",
                           date="2026-09-20", excerpt="Two hyperscalers trimmed guides.",
                           found_by=["bull", "bear"], query="capex guidance cuts"),
        ],
        claims=[
            DebateClaim(id="BULL-1", side="bull", pillar="growth", claim="Networking attach rises.",
                        category="growth", materiality="high", evidence=["E01"],
                        quote={"evidence": "E01", "text": "Networking revenue grew.", "verified": True},
                        analyst_refs=["analyst:sector"], contests_analyst=None,
                        falsifier="Attach rate below 20% for two quarters",
                        grade="sourced", figures={"attach_pct": 20}, status="contested"),
            DebateClaim(id="BEAR-1", side="bear", claim="Capex peaks next year.",
                        evidence=["E02"], contests_analyst="analyst:valuation",
                        grade="partially_sourced", status="partial"),
            DebateClaim(id="BEAR-2", side="bear", claim="Unsourced aside.", dropped=True,
                        drop_reason="no_evidence"),
        ],
        responses=[
            DebateResponse(side="bear", target="BULL-1", stance="rebut",
                           argument="Attach gains are one-off.", evidence=["E02"],
                           grade="partially_sourced"),
            DebateResponse(side="bull", target="BEAR-1", stance="partial"),
        ],
        disputes=["D1"],
        unanswered=["BEAR-2"],
        resolution=DebateResolution(
            status="ruled", crux="Capex duration",
            rulings=[DebateRuling(dispute="D1", claim="BULL-1", ruling="split",
                                  basis=["E01", "E02"], flags=["relied_partial"])],
            unresolved=["D2"], relied_unsupported=[],
        ),
        deterministic_checks=["one_sided_ruling_share"],
        # Integer tallies beside the L1 counterfactual's float values.
        outcome={"conceded": 0, "contested": 1, "cf_rating_score": 0.5,
                 "cf_confidence": 71.5, "cf_shift": -1},
        news_since=[{"at": "2026-09-21T10:00:00", "headline": "Guide cut",
                     "alert_ref": "news:def456"}],
        usage={"calls": 6, "tokens_in": 41000, "tokens_out": 9000, "usd": 0.84, "cap_usd": 1.5},
    )


def test_legacy_memo_without_debate_validates():
    """A pre-D2 stored memo (no `debate` key, a critic review with none of the
    item-8 fields) reads through the store's reader with `debate=None`, and
    `debate` is the only new top-level key its dump gains."""
    original = json.loads(PRE_S2.read_text())
    assert "debate" not in original, "fixture must be a pre-D2 shape"
    snap = MemoSnapshot(id=1, ticker="ZZPRE", version=1, trigger="full_reanalysis",
                        memo_json=deepcopy(original))
    memo = memo_store.memo_to_pydantic(snap)
    assert memo.debate is None
    assert memo.bull_case.model_dump() == original["bull_case"]
    assert memo.bear_case.model_dump() == original["bear_case"]
    dumped = memo.model_dump(mode="json")
    assert "debate" in set(dumped) - set(original)
    assert dumped["debate"] is None
    # Plain validation (no store adapter) agrees.
    assert StockMemoOut.model_validate(deepcopy(original)).debate is None


def test_debate_sits_after_bear_case():
    """The design places the record right after the cases it extends; the
    field order is also the serialized key order the fixtures pin."""
    fields = list(StockMemoOut.model_fields)
    assert fields[fields.index("bear_case") + 1] == "debate"


def test_debate_record_roundtrip_and_defaults():
    empty = DebateRecord()
    assert empty.protocol_version == 1
    assert empty.status == "not_run"
    assert empty.presentation_order == "bull_first"
    assert empty.research_status == "directed"
    assert empty.resolution == DebateResolution()
    assert empty.resolution.status == "not_applicable"
    assert (empty.route, empty.outcome, empty.usage, empty.news_since) == ({}, {}, {}, [])
    assert (empty.evidence, empty.claims, empty.responses) == ([], [], [])
    assert DebateReview() == DebateReview(dispute_views=[], unaddressed=[], one_sided="")

    record = populated_record()
    wire = json.loads(record.model_dump_json())
    again = DebateRecord.model_validate(wire)
    assert again == record
    assert json.loads(again.model_dump_json()) == wire
    # A refusal is recordable in the persisted route (P16).
    assert wire["route"]["phases"][0]["status"] == "refused:bio"
    # `outcome` keeps the counterfactual's floats as floats and the tallies
    # as ints; the design's `dict[str, int]` would have refused 71.5.
    assert again.outcome["cf_confidence"] == 71.5
    assert again.outcome["cf_rating_score"] == 0.5
    assert type(again.outcome["contested"]) is int
    assert type(again.outcome["cf_shift"]) is int


def test_memo_store_roundtrip_with_populated_debate():
    """`save_memo` validates, stores JSON and the reader rebuilds the record
    exactly — the path every debated memo takes."""
    memo = _stub_memo(TICKER)
    memo.debate = populated_record()
    snap = memo_store.save_memo(memo, trigger="first_run")
    stored = memo_store.memo_version(TICKER, snap.version)
    assert stored is not None
    assert stored.memo_json["debate"]["route"]["phases"][0]["status"] == "refused:bio"
    back = memo_store.memo_to_pydantic(stored)
    assert back.debate == memo.debate
    assert back.model_dump(mode="json")["debate"] == json.loads(memo.debate.model_dump_json())


def test_memo_without_debate_stores_null():
    snap = memo_store.save_memo(_stub_memo(TICKER), trigger="first_run")
    stored = memo_store.memo_version(TICKER, snap.version)
    assert stored is not None
    assert stored.memo_json["debate"] is None
    assert memo_store.memo_to_pydantic(stored).debate is None


def _claim(**over: Any) -> dict[str, Any]:
    return {"id": "BULL-1", "side": "bull", "claim": "c", **over}


@pytest.mark.parametrize("model, payload", [
    (DebateRecord, {"status": "done"}),
    (DebateRecord, {"presentation_order": "random"}),
    (DebateClaim, _claim(side="neutral")),
    (DebateClaim, _claim(grade="well_sourced")),
    (DebateClaim, _claim(status="won")),
    (DebateClaim, _claim(materiality="critical")),
    (DebateResponse, {"side": "bull", "target": "BEAR-1", "stance": "ignore"}),
    (DebateRuling, {"dispute": "D1", "claim": "BULL-1", "ruling": "draw"}),
    (DebateResolution, {"status": "skipped"}),
    (DebateEvidence, {"id": "E01", "kind": "news", "ref": "news:x", "found_by": ["pm"]}),
])
def test_debate_enums_refuse_off_list_values(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_evidence_excerpt_bound():
    """Quotes are verified against the excerpt, so the bound is contractual."""
    DebateEvidence(id="E01", kind="filing", ref="r", excerpt="x" * DEBATE_EXCERPT_MAX_CHARS)
    with pytest.raises(ValidationError):
        DebateEvidence(id="E01", kind="filing", ref="r", excerpt="x" * (DEBATE_EXCERPT_MAX_CHARS + 1))
