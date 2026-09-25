"""W7: priors update from later, out-of-sample, eligible outcomes.

The nightly judge (`ledger.judge_due`) is decoupled from the postmortem:
it covers every W6-eligible 90d+ outcome whose window has in-scope,
knowable-at-memo-time, not-in-sample hypotheses without evidence — even when
the postmortem deduped that memo as "rating unchanged". The model answers
only whether a hypothesis's CONDITION applied; held / failed come from
realized alpha. One evidence row per lesson per scope window, so three peers
in one window count once. Spend is capped in calls and dollars.
"""
from __future__ import annotations

import itertools
from datetime import date, datetime, timedelta
from typing import Any

import pytest

from app.agents import llm
from app.config import settings
from app.learning import ledger
from app.models import LearningEvidence, LearningItem
from app.services import postmortem_service as pm
from app.tests.learning_helpers import add_company, add_outcome, add_snapshot, learning_db

NOW = datetime(2026, 12, 1)
_REFS = itertools.count(1)
ORIGIN_EVAL = datetime(2026, 3, 1)    # when the lesson became knowable


@pytest.fixture
def db(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "learning_ledger_writes", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "use_demo_data", False)   # a live model is required to judge
    yield sessions
    engine.dispose()


@pytest.fixture
def judge(monkeypatch):
    """Canned judge: answers `applies` per id from `box.answers` (default
    "yes"); records every prompt."""
    class Box:
        answers: dict[str, str] = {}
        extra: list[dict[str, Any]] = []
        prompts: list[str] = []
        reply_none = False

    box = Box()
    box.answers, box.extra, box.prompts = {}, [], []

    def _chat(prompt: str, **kwargs: Any):
        assert kwargs.get("route") == "cheap"
        box.prompts.append(prompt)
        if box.reply_none:
            return None
        import json
        payload = json.loads(prompt.split("Hypotheses and memo:\n", 1)[1])
        rows = [{"id": h["id"], "applies": box.answers.get(h["id"], "yes"), "why": "the memo says so"}
                for h in payload["hypotheses"]]
        return {"judgments": rows + box.extra}
    monkeypatch.setattr(llm, "chat_json", _chat)
    return box


def _lesson(sessions, *, scope_type: str = "company", scope_key: str = "JDGA", observable: str = "outperform",
            condition: str = "management raises gross margin guidance", origin_snapshot_id: int | None = None,
            source_date: date = ORIGIN_EVAL.date()) -> int:
    with sessions() as s:
        item = ledger._new_item(
            s, kind="lesson", scope_type=scope_type, scope_key=scope_key,
            text=f"When {condition}, expect it.", condition=condition, observable=observable,
            origin_kind="postmortem", origin_ref=f"pm-{next(_REFS)}",
            origin_ticker=scope_key, origin_snapshot_id=origin_snapshot_id, source_date=source_date,
            now=ORIGIN_EVAL,
        )
        s.commit()
        return item.id


def _memo(sessions, ticker: str, generated_at: datetime, *, alpha: float = 0.08, version: int = 1,
          company: bool = True, sector: str = "Technology", evaluated_at: datetime | None = None,
          **memo: Any):
    with sessions() as s:
        if company:
            add_company(s, ticker, sector)
        snap = add_snapshot(s, ticker, generated_at=generated_at, version=version, sector=sector, **memo)
        add_outcome(s, snap, horizon=90, alpha=alpha, evaluated_at=evaluated_at)
        return snap.id


def _evidence(sessions, item_id: int | None = None) -> list[LearningEvidence]:
    with sessions() as s:
        q = s.query(LearningEvidence)
        if item_id is not None:
            q = q.filter_by(item_id=item_id)
        rows = q.order_by(LearningEvidence.id).all()
        s.expunge_all()
        return rows


def _run(**kw: Any) -> dict[str, Any]:
    return ledger.judge_due(max_calls=kw.pop("max_calls", 20), max_usd=kw.pop("max_usd", 0.25), now=NOW, **kw)


def test_hypotheses_exclude_future_source_date_in_sample_and_judged_bucket(db):
    later = datetime(2026, 5, 1)
    sid = _memo(db, "JDGA", later)
    knowable = _lesson(db)
    _lesson(db, condition="future lesson learned after the memo", source_date=date(2026, 6, 1))
    _lesson(db, condition="lesson learned from this very memo", origin_snapshot_id=sid)
    with db() as s:
        scopes = ledger.scopes_for(s, "JDGA")
        cands = ledger.hypotheses_to_judge(s, scopes=scopes, snapshot_id=sid, generated_at=later, horizon_days=90)
        assert [c.item_id for c in cands] == [knowable]
        s.add(LearningEvidence(item_id=knowable, verdict="held", ticker="JDGA", horizon_days=90,
                               independence_key=cands[0].key, observed_at=later))
        s.commit()
        assert ledger.hypotheses_to_judge(s, scopes=scopes, snapshot_id=sid, generated_at=later,
                                          horizon_days=90) == []


def test_judgments_become_evidence_unknown_ids_ignored(db, judge):
    sid = _memo(db, "JDGA", datetime(2026, 5, 1), alpha=0.08)
    item = _lesson(db)
    judge.extra = [{"id": "L-99999", "applies": "yes", "why": "not offered"}]
    report = _run()
    assert (report["calls"], report["evidence"], report["failed"]) == (1, 1, 0)
    [row] = _evidence(db)
    assert (row.item_id, row.verdict, row.applies, row.memo_snapshot_id) == (item, "held", "yes", sid)
    assert row.alpha == pytest.approx(0.08) and row.observed_at == datetime(2026, 5, 1) + timedelta(days=91)
    assert row.rationale == "the memo says so"


def test_one_evidence_row_per_window(db, judge):
    """MSFT is at v108: re-issues inside one window give one row per lesson."""
    item = _lesson(db)
    _memo(db, "JDGA", datetime(2026, 5, 1))
    _memo(db, "JDGA", datetime(2026, 5, 2), version=2, company=False)
    report = _run()
    assert report["calls"] == 1 and len(_evidence(db, item)) == 1


def test_deduped_eligible_outcome_still_produces_evidence(db, judge, monkeypatch):
    """The postmortem skips v2 as "rating unchanged"; the judge still uses
    its outcome — dedupe used to starve the ledger of evidence."""
    monkeypatch.setattr(pm, "_llm_postmortem", lambda memo, outcome, horizon_days, *a: None)
    _memo(db, "JDGA", datetime(2026, 2, 1), rating="Bullish")
    v2 = _memo(db, "JDGA", datetime(2026, 6, 20), version=2, company=False, rating="Bullish")
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert [m["memo_snapshot_id"] for m in report["deduped_memos"]] == [v2]
    item = _lesson(db, source_date=date(2026, 5, 10))   # knowable for v2 only
    _run()
    assert [(r.item_id, r.memo_snapshot_id) for r in _evidence(db)] == [(item, v2)]


def test_verdict_follows_alpha_not_model(db, judge):
    """The judge "says" held; realized alpha of -10% on an outperform
    hypothesis makes it failed. The model's opinion of the outcome is
    never read."""
    _memo(db, "JDGA", datetime(2026, 5, 1), alpha=-0.10)
    item = _lesson(db)
    judge.extra = []
    report = _run()
    assert report["evidence"] == 1
    [row] = _evidence(db, item)
    assert row.verdict == "failed"
    assert "alpha" not in judge.prompts[0] and "outperform" not in judge.prompts[0]


def test_absent_condition_is_irrelevant(db, judge):
    item_no = _lesson(db, condition="the company announces a large buyback")
    item_unclear = _lesson(db, condition="channel inventory is rising at distributors")
    _memo(db, "JDGA", datetime(2026, 5, 1), alpha=0.2)
    judge.answers = {f"L-{item_no}": "no", f"L-{item_unclear}": "maybe"}
    report = _run()
    assert (report["evidence"], report["irrelevant"]) == (0, 2)
    with db() as s:
        post = ledger.posteriors(s, [item_no, item_unclear], now=NOW)
    assert {r.verdict for r in _evidence(db)} == {"irrelevant"}
    assert all(p.stance == "untested" and p.n_eff == 0 for p in post.values())


def test_fourth_failure_auto_retires_with_history(db, judge):
    item = _lesson(db)
    # Evaluated at NOW, so every weight is exactly 1 and the design's
    # "4 failed -> retire" row applies exactly.
    fresh = NOW
    for i, month in enumerate((4, 7, 10)):
        _memo(db, "JDGA", datetime(2026, month, 1), alpha=-0.2, version=i + 1, company=(i == 0),
              evaluated_at=fresh)
    _run()
    with db() as s:
        assert s.get(LearningItem, item).status == "active"      # 3 failures: not yet
    _memo(db, "JDGA", datetime(2026, 1, 5) + timedelta(days=360), alpha=-0.2, version=4, company=False,
          evaluated_at=fresh)
    report = _run()
    assert report["retired"] == 1
    with db() as s:
        row = s.get(LearningItem, item)
        assert row.status == "retired"
        assert row.status_history[-1]["reason"] == "auto:posterior"
        assert row.status_history[-1]["actor"] == "auto"


def test_group_lesson_judged_on_peer_ticker(db, judge):
    item = _lesson(db, scope_type="sector", scope_key="information_technology",
                   condition="enterprise software budgets are cut mid-year", observable="underperform")
    sid = _memo(db, "JDGP", datetime(2026, 5, 1), alpha=-0.04)
    _run()
    [row] = _evidence(db, item)
    assert (row.ticker, row.memo_snapshot_id, row.verdict) == ("JDGP", sid, "held")
    assert row.independence_key.startswith("information_technology:90:")


def test_three_peers_one_window_neff_le_1(db, judge):
    item = _lesson(db, scope_type="sector", scope_key="information_technology",
                   condition="enterprise software budgets are cut mid-year", observable="outperform")
    for ticker in ("JDP1", "JDP2", "JDP3"):
        _memo(db, ticker, datetime(2026, 5, 1), alpha=0.2)
    report = _run()
    assert report["calls"] == 1 and len(_evidence(db, item)) == 1
    with db() as s:
        assert ledger.posteriors(s, [item], now=NOW)[item].n_eff <= 1.0


def test_hypotheses_survive_oversized_memo(db, judge):
    items = [_lesson(db, condition=f"condition number {n} holds for the quarter") for n in range(3)]
    _memo(db, "JDGA", datetime(2026, 5, 1), thesis="Enormous thesis. " * 20000)
    _run()
    prompt = judge.prompts[0]
    for n, item in enumerate(items):
        assert f"L-{item}" in prompt and f"condition number {n} holds" in prompt
    assert prompt.index('"hypotheses"') < prompt.index('"memo"')
    assert len(prompt) < 40_000        # each memo field is clipped on its own


def test_judge_caps_stop_spend(db, judge):
    _lesson(db, scope_type="sector", scope_key="information_technology",
            condition="enterprise software budgets are cut mid-year")
    for i, ticker in enumerate(("JDC1", "JDC2", "JDC3")):
        _memo(db, ticker, datetime(2026, 4 + 3 * i, 1))   # three separate windows
    report = _run(max_calls=2)
    assert (report["calls"], report["deferred"], report["stopped_reason"]) == (2, 1, "max_calls")
    assert len(judge.prompts) == 2
    report = _run(max_usd=1e-9)
    assert (report["calls"], report["deferred"], report["stopped_reason"]) == (0, 1, "max_usd")
    assert len(judge.prompts) == 2       # nothing spent past the dollar cap


def test_judge_off_or_without_live_model_makes_no_calls(db, judge, monkeypatch):
    _lesson(db)
    _memo(db, "JDGA", datetime(2026, 5, 1))
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    assert _run()["status"] == "off"
    monkeypatch.setattr(settings, "learning_ledger_writes", True)
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    assert _run()["status"] == "no_llm"
    assert judge.prompts == [] and _evidence(db) == []


def test_failed_judge_call_is_counted_and_retried(db, judge):
    _lesson(db)
    _memo(db, "JDGA", datetime(2026, 5, 1))
    judge.reply_none = True
    report = _run()
    assert (report["failed"], report["status"], report["evidence"]) == (1, "partial", 0)
    judge.reply_none = False
    assert _run()["evidence"] == 1


def test_one_bad_memo_does_not_end_the_night(db, judge, monkeypatch):
    item = _lesson(db, scope_type="sector", scope_key="information_technology",
                   condition="enterprise software budgets are cut mid-year")
    bad = _memo(db, "JDB1", datetime(2026, 4, 1))
    good = _memo(db, "JDB2", datetime(2026, 7, 1))
    real = ledger.present_for_learning

    def flaky(snap, db=None):
        if snap.id == bad:
            raise RuntimeError("presenter bug")
        return real(snap, db)
    monkeypatch.setattr(ledger, "present_for_learning", flaky)
    report = _run()
    assert report["failed_memos"] == [{"ticker": "JDB1", "memo_snapshot_id": bad, "error_type": "RuntimeError"}]
    assert [(r.item_id, r.memo_snapshot_id) for r in _evidence(db)] == [(item, good)]
