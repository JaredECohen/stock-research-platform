"""W7: what the postmortem path may write into the learning ledger.

Only outcome-grounded, W6-eligible, non-template, 90d+ postmortems create
lessons, and only in the grounded grammar `{condition, observable}`. Writes
are idempotent, reported truthfully in the postmortem report, and with the
ledger off nothing changes: `_llm_postmortem` is still called with three
arguments (every existing stub's arity) and the prompt carries no
hypothesis contract.

Each test uses a private sqlite file (exact counts) and a canned
`llm.chat_json`; a throwaway key opens `has_llm`, and the suite's netguard
refuses any real call.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.agents import llm
from app.config import settings
from app.learning import ledger
from app.models import LearningItem, MemoPostmortem
from app.services import postmortem_service as pm
from app.tests.factories import make_finding
from app.tests.learning_helpers import add_company, add_outcome, add_snapshot, classify, learning_db

GEN = datetime(2026, 6, 20)
HYP = {"condition": "management raises full-year gross margin guidance twice in a row",
       "observable": "outperform"}
PEER = {"condition": "group peers report falling inventory days alongside rising prices",
        "observable": "underperform"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "learning_ledger_writes", True)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")
    yield sessions
    engine.dispose()


@pytest.fixture
def canned(monkeypatch):
    """`llm.chat_json` returns `canned.reply`; every prompt is kept."""
    class Canned:
        reply: Any = {"lesson": "We said buy. It rallied. Next time, weigh guidance.",
                      "agent_attribution": {}, "regime_at_memo": "", "sector_lesson": "",
                      "hypothesis": HYP, "peer_hypothesis": PEER}
        prompts: list[str] = []

    box = Canned()
    box.prompts = []

    def _chat(prompt: str, **kwargs: Any):
        box.prompts.append(prompt)
        return box.reply
    monkeypatch.setattr(llm, "chat_json", _chat)
    return box


def _seed(sessions, ticker: str, *, company: bool = True, **memo: Any):
    with sessions() as s:
        if company:
            add_company(s, ticker)
        snap = add_snapshot(s, ticker, generated_at=GEN, **memo)
        add_outcome(s, snap, horizon=90, alpha=0.08)
        return snap.id


def _items(sessions) -> list[LearningItem]:
    with sessions() as s:
        rows = s.query(LearningItem).order_by(LearningItem.id).all()
        s.expunge_all()
        return rows


def test_eligible_90d_postmortem_writes_company_and_group_lessons(db, canned):
    with db() as s:
        classify(s, "LRNA", state="mapped", group="4510")
    sid = _seed(db, "LRNA")
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert report["written"] == 1
    assert (report["learning_written"], report["learning_skipped"], report["learning_failed"]) == (1, 0, 0)

    company, group = _items(db)
    with db() as s:
        pm_id = s.query(MemoPostmortem.id).filter_by(memo_snapshot_id=sid).scalar()
    assert (company.kind, company.scope_type, company.scope_key) == ("lesson", "company", "LRNA")
    assert (company.condition, company.observable) == (HYP["condition"], "outperform")
    assert company.text == (f"When {HYP['condition']}, expect the stock to outperform the benchmark "
                            "over 90 days.")
    assert (company.origin_kind, company.origin_ref, company.origin_snapshot_id) == ("postmortem", str(pm_id), sid)
    assert company.source_date == datetime(2026, 9, 19).date()        # outcome.evaluated_at, not today
    assert company.detail.startswith("We said buy.")
    assert (group.scope_type, group.scope_key, group.origin_kind) == ("industry_group", "4510", "postmortem_sector")
    assert "4510" not in group.text and "group peers" in group.text
    # The strong call proposed; it was never asked to judge anything.
    prompt = canned.prompts[0]
    assert '"hypothesis"' in prompt and '"peer_hypothesis"' in prompt
    assert "hypotheses_to_judge" not in prompt and "hypothesis_judgments" not in prompt


def test_unclassified_ticker_falls_back_to_sector_scope(db, canned):
    _seed(db, "LRNB")
    pm.run_postmortems(horizon_days=90, limit=10)
    peer = [i for i in _items(db) if i.origin_kind == "postmortem_sector"]
    assert [(i.scope_type, i.scope_key) for i in peer] == [("sector", "information_technology")]


def test_fallback_state_ticker_uses_sector_scope(db, canned):
    """A fallback classification knows its sector, not its group — even when
    a group code is on the row, it must not scope a group lesson."""
    with db() as s:
        classify(s, "LRNC", state="fallback", group="4510")
    _seed(db, "LRNC")
    pm.run_postmortems(horizon_days=90, limit=10)
    peer = [i for i in _items(db) if i.origin_kind == "postmortem_sector"]
    assert [(i.scope_type, i.scope_key) for i in peer] == [("sector", "information_technology")]


def test_ineligible_or_unclassified_snapshot_writes_nothing(db):
    view = ledger.MemoView(True, None, {"sector": "Technology"})
    out = {"hypothesis": HYP}
    for eligible, name in ((False, "LRND"), (None, "LRNE")):
        with db() as s:
            snap = add_snapshot(s, name, generated_at=GEN, eligible=eligible)
            add_outcome(s, snap)
            s.add(MemoPostmortem(memo_snapshot_id=snap.id, ticker=name, horizon_days=90, verdict="right",
                                 lesson="x", agent_attribution={}))
            s.commit()
            sid = snap.id
        res = ledger.record_postmortem(snapshot_id=sid, ticker=name, horizon_days=90,
                                       evaluated_at=GEN, llm_out=out, view=view)
        assert (res["status"], res["reason"], res["items"]) == ("skipped", "ineligible", 0)
    assert _items(db) == []


def test_deterministic_postmortem_writes_nothing(db, canned):
    canned.reply = None
    _seed(db, "LRNF")
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert report["written"] == 1 and report["learning_written"] == 0
    assert report["learning_skip_reasons"] == {"deterministic": 1}
    assert _items(db) == []


@pytest.mark.parametrize("memo, reason", [
    ({"degraded_agents": ["PM Synthesis"]}, "template_pm"),
    ({"thesis": ""}, "unavailable:one_sentence_thesis"),
])
def test_template_pm_memo_skipped(db, canned, memo, reason):
    _seed(db, "LRNG", **memo)
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert report["learning_written"] == 0 and report["learning_skip_reasons"] == {reason: 1}
    assert _items(db) == []
    # A memo that may not teach is not even asked for a hypothesis.
    assert '"hypothesis"' not in canned.prompts[0]


def test_learning_payload_goes_through_the_presenter(db, canned):
    """W2a: a template-filled section reads "Unavailable in this version."
    in the learning payload; its template text never reaches the model."""
    template = make_finding("Filing Analyst", summary="TEMPLATE FILLER FROM THE FALLBACK",
                            data={"deterministic_fallback": True})
    _seed(db, "LRNH", filing_agent_view=template)
    pm.run_postmortems(horizon_days=90, limit=10)
    prompt = canned.prompts[0]
    assert "TEMPLATE FILLER FROM THE FALLBACK" not in prompt
    assert "Unavailable in this version." in prompt


def test_gics_or_code_text_rejected(db, canned):
    with db() as s:
        classify(s, "LRNI", state="mapped", group="4510")
    canned.reply = {**canned.reply,
                    "hypothesis": {"condition": "GICS sector momentum fades into earnings", "observable": "underperform"},
                    "peer_hypothesis": {"condition": "industry group 4510 guides capex down", "observable": "outperform"}}
    _seed(db, "LRNI")
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert report["learning_rejected"] == 2 and report["learning_written"] == 0
    assert report["learning_skip_reasons"] == {"rejected": 1}
    assert _items(db) == []


def test_rerun_is_duplicate_not_failure(db, canned):
    sid = _seed(db, "LRNJ")
    pm.run_postmortems(horizon_days=90, limit=10)
    before = len(_items(db))
    with db() as s:
        from app.models import MemoSnapshot
        snap = s.get(MemoSnapshot, sid)
        s.expunge(snap)
    view = ledger.present_for_learning(snap)
    res = ledger.record_postmortem(snapshot_id=sid, ticker="LRNJ", horizon_days=90, evaluated_at=GEN,
                                   llm_out=canned.reply, view=view)
    assert (res["status"], res["items"], res["error_type"]) == ("duplicate", 0, None)
    assert len(_items(db)) == before == 2


def test_writes_off_calls_llm_with_three_args_and_report_zeros(db, canned, monkeypatch):
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    calls: list[int] = []

    def three(memo, outcome, horizon_days):   # the arity every existing stub has
        calls.append(horizon_days)
        return None
    monkeypatch.setattr(pm, "_llm_postmortem", three)
    _seed(db, "LRNK")
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert calls == [90]
    assert {k: report[k] for k in ("learning_written", "learning_skipped", "learning_rejected",
                                   "learning_failed")} == dict.fromkeys(
        ("learning_written", "learning_skipped", "learning_rejected", "learning_failed"), 0)
    assert report["learning_failed_memos"] == [] and report["learning_skip_reasons"] == {}
    assert _items(db) == []


def test_writes_off_prompt_has_no_hypothesis_contract(db, canned, monkeypatch):
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    _seed(db, "LRNL")
    pm.run_postmortems(horizon_days=90, limit=10)
    assert '"hypothesis"' not in canned.prompts[0]
    assert canned.prompts[0].startswith("Write a 90-day postmortem for this memo.")


def test_30d_writes_nothing(db, monkeypatch):
    calls: list[int] = []

    def three(memo, outcome, horizon_days):
        calls.append(horizon_days)
        return {"lesson": "x", "hypothesis": HYP}
    monkeypatch.setattr(pm, "_llm_postmortem", three)
    with db() as s:
        add_company(s, "LRNM")
        snap = add_snapshot(s, "LRNM", generated_at=GEN)
        add_outcome(s, snap, horizon=30)
    report = pm.run_postmortems(horizon_days=30, limit=10)
    assert calls == [30] and report["written"] == 1
    assert report["learning_written"] == report["learning_skipped"] == 0
    assert _items(db) == []
