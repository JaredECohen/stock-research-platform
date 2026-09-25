"""W7: the nightly ledger pass — backfill, integrity (K1), expiry, capacity.

The backfill is DB-only (no LLM), idempotent, and reads only postmortems
written BEFORE the ledger epoch that the live path never learned from: a
live postmortem whose hypothesis was empty was deliberately not learned, and
re-learning its narrative would duplicate exactly what the epoch prevents.
Contamination cannot persist: an item whose origin snapshot stops being
eligible is retired, and if it reached an inject render the DB mode is
demoted automatically. Filing observations have no origin snapshot and are
out of K1's (and G1's) scope.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from app.agents import llm
from app.config import settings
from app.learning import control, ledger
from app.models import LearningControlEvent, LearningItem, LearningRender, MemoPostmortem
from app.monitoring import postmortem_loop
from app.services import postmortem_service as pm
from app.tests.eligibility_helpers import mark
from app.tests.learning_helpers import add_company, add_outcome, add_snapshot, learning_db

NOW = datetime(2026, 10, 1)
GEN = datetime(2026, 6, 20)
LESSON = ("We rated it Bullish on a margin recovery that management had promised for two quarters. "
          "The stock lagged SPY by 6% over the window as the recovery slipped again. Pricing was weaker "
          "than guided. Next time, weigh distributor inventory before trusting margin guidance.")


@pytest.fixture
def db(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "learning_ledger_writes", True)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    yield sessions
    engine.dispose()


def _postmortem(sessions, ticker: str, *, lesson: str = LESSON, created_at: datetime = NOW - timedelta(days=5),
                eligible: bool | None = True, horizon: int = 90, **memo: Any) -> tuple[int, int]:
    with sessions() as s:
        snap = add_snapshot(s, ticker, generated_at=GEN, eligible=eligible, **memo)
        add_outcome(s, snap, horizon=horizon)
        row = MemoPostmortem(memo_snapshot_id=snap.id, ticker=ticker, horizon_days=horizon, verdict="wrong",
                             lesson=lesson, agent_attribution={}, created_at=created_at)
        s.add(row)
        s.commit()
        return snap.id, row.id


def _epoch(sessions, at: datetime) -> None:
    with sessions() as s:
        control.ensure_epoch(s, now=at)
        s.commit()


def _items(sessions, **filters: Any) -> list[LearningItem]:
    with sessions() as s:
        rows = s.query(LearningItem).filter_by(**filters).order_by(LearningItem.id).all()
        s.expunge_all()
        return rows


def test_sync_backfills_eligible_llm_lessons_trailing_sentences(db):
    sid, pm_id = _postmortem(db, "NTA")
    out = ledger.nightly(now=NOW)
    assert out["backfilled"] == 1 and out["backfill_failed"] == 0
    [item] = _items(db)
    assert (item.kind, item.scope_type, item.scope_key, item.origin_kind) == (
        "lesson", "company", "NTA", "postmortem_backfill")
    assert item.origin_ref == str(pm_id) and item.origin_snapshot_id == sid
    # The whole lesson is 262 chars; the longest whole-sentence tail that fits 240 is kept.
    assert len(LESSON) > ledger.LESSON_MAX
    assert item.text == ("The stock lagged SPY by 6% over the window as the recovery slipped again. Pricing was "
                         "weaker than guided. Next time, weigh distributor inventory before trusting margin "
                         "guidance.")
    assert len(item.text) <= ledger.LESSON_MAX and item.detail == LESSON
    # Historical narrative is never judged: it carries no observable.
    assert item.observable is None and item.condition is None
    assert item.source_date == (NOW - timedelta(days=5)).date()


def test_sync_skips_deterministic_signature_and_ineligible(db):
    _postmortem(db, "NTB", lesson=pm._deterministic_lesson(
        {"rating_label": "Bullish"}, type("O", (), {"alpha": 0.1, "forward_return": 0.2,
                                                    "benchmark_return": 0.1})(), "right", 90))
    _postmortem(db, "NTC", eligible=False)
    _postmortem(db, "NTD", eligible=None)                    # unclassified is ineligible (fail closed)
    _postmortem(db, "NTE", degraded_agents=["PM Synthesis"])
    _postmortem(db, "NTF", horizon=30)
    out = ledger.nightly(now=NOW)
    assert out["backfilled"] == 0
    assert out["backfill_skipped"] == {"deterministic": 1, "template_pm": 1}
    assert _items(db) == []


def test_sync_idempotent(db):
    _postmortem(db, "NTG")
    assert ledger.nightly(now=NOW)["backfilled"] == 1
    assert ledger.nightly(now=NOW + timedelta(days=1))["backfilled"] == 0
    assert len(_items(db)) == 1


def test_sync_skips_postmortems_recorded_live(db):
    """Two ways a postmortem belongs to the live era: it was written after
    the epoch, or the live path already stored an item for it."""
    _epoch(db, NOW - timedelta(days=10))
    _postmortem(db, "NTH", created_at=NOW - timedelta(days=2))        # after the epoch
    sid, pm_id = _postmortem(db, "NTI", created_at=NOW - timedelta(days=20))
    with db() as s:
        ledger._new_item(s, kind="lesson", scope_type="industry_group", scope_key="4510",
                         text="When x, expect group peers to outperform the benchmark over 90 days.",
                         condition="x", observable="outperform", origin_kind="postmortem_sector",
                         origin_ref=str(pm_id), origin_ticker="NTI", origin_snapshot_id=sid,
                         source_date=date(2026, 9, 1), now=NOW - timedelta(days=20))
        s.commit()
    assert ledger.nightly(now=NOW)["backfilled"] == 0
    assert [i.origin_kind for i in _items(db)] == ["postmortem_sector"]


def test_empty_hypothesis_is_not_backfilled(db, monkeypatch):
    """The live path wrote a postmortem whose hypothesis was null, so it
    learned nothing — on purpose. The backfill must not later learn its
    narrative tail instead."""
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")
    monkeypatch.setattr(llm, "chat_json", lambda prompt, **k: {
        "lesson": LESSON, "agent_attribution": {}, "regime_at_memo": "", "sector_lesson": "",
        "hypothesis": None, "peer_hypothesis": None})
    with db() as s:
        add_company(s, "NTJ")
        snap = add_snapshot(s, "NTJ", generated_at=GEN)
        add_outcome(s, snap)
    report = pm.run_postmortems(horizon_days=90, limit=10)
    assert report["written"] == 1 and report["learning_skip_reasons"] == {"no_hypothesis": 1}
    assert ledger.nightly(now=datetime.utcnow() + timedelta(days=1))["backfilled"] == 0
    assert _items(db) == []


def test_integrity_retires_ineligible_origin_and_demotes_if_injected(db):
    _postmortem(db, "NTK")
    ledger.nightly(now=NOW)
    [item] = _items(db)
    with db() as s:
        s.add(LearningControlEvent(mode="inject", actor="admin", reason="soaked", gates={}, forced=True,
                                   created_at=NOW - timedelta(days=3)))
        s.add(LearningRender(run_id="r1", consumer="pm_memo", ticker="NTK", mode="inject", chars=300,
                             items=[{"ref": f"L-{item.id}", "item_id": item.id}], dropped=[],
                             created_at=NOW - timedelta(days=2)))
        mark(s, item.origin_snapshot_id, eligible=False, reason="demo_dev_copy_2026_05_04")
    out = ledger.integrity_check(now=NOW)
    assert out == {"retired": [item.id], "demoted": True}
    [retired] = _items(db)
    assert retired.status == "retired" and retired.status_history[-1]["reason"] == "origin_ineligible"
    with db() as s:
        assert control.db_mode(s) == "shadow"
        latest = s.query(LearningControlEvent).order_by(LearningControlEvent.id.desc()).first()
        assert (latest.actor, latest.reason) == ("auto", f"contamination: [{item.id}]")


def test_demotion_never_raises_an_off_mode(db):
    _postmortem(db, "NTL")
    ledger.nightly(now=NOW)
    [item] = _items(db)
    with db() as s:
        s.add(LearningControlEvent(mode="off", actor="admin", reason="stop", gates={}, created_at=NOW))
        s.add(LearningRender(run_id="r1", consumer="pm_memo", ticker="NTL", mode="inject", chars=10,
                             items=[{"item_id": item.id}], dropped=[], created_at=NOW - timedelta(days=1)))
        mark(s, item.origin_snapshot_id, eligible=False, reason="demo_dev_copy_2026_05_04")
    assert ledger.integrity_check(now=NOW)["demoted"] is False
    with db() as s:
        assert control.db_mode(s) == "off"


def test_observation_survives_integrity_and_does_not_block_g1(db):
    with db() as s:
        ledger._new_item(s, kind="observation", scope_type="company", scope_key="NTM",
                         text="What's new in 10-Q filed 2026-09-01: backlog grew", origin_kind="filing_delta",
                         origin_ref="0001", origin_ticker="NTM", source_date=date(2026, 9, 1),
                         expires_at=NOW + timedelta(days=300), now=NOW)
        s.commit()
    assert ledger.integrity_check(now=NOW)["retired"] == []
    [obs] = _items(db)
    assert obs.status == "active"
    g = control.gates(now=NOW)
    assert g["G1"] == {"ok": True, "contaminated_active_items": 0} and g["G4"]["ok"] is True


def test_observation_expiry_and_scope_capacity(db, monkeypatch):
    monkeypatch.setattr(ledger, "MAX_ACTIVE_LESSONS_PER_SCOPE", 3)
    with db() as s:
        ledger._new_item(s, kind="observation", scope_type="company", scope_key="NTN", text="old filing fact",
                         origin_kind="filing_delta", origin_ref="A1", origin_ticker="NTN",
                         expires_at=NOW - timedelta(days=1), now=NOW - timedelta(days=401))
        tested = None
        for n in range(5):
            item = ledger._new_item(s, kind="lesson", scope_type="company", scope_key="NTN", text=f"lesson {n}",
                                    condition=f"c{n}", observable="outperform", origin_kind="postmortem",
                                    origin_ref=f"p{n}", origin_ticker="NTN", source_date=date(2026, 1, 1),
                                    now=NOW - timedelta(days=100 - n))
            s.flush()
            if n == 0:
                tested = item.id
        s.add_all([ledger.LearningEvidence(item_id=tested, verdict="held", ticker="NTN", horizon_days=90,
                                           independence_key=f"NTN:90:{k}", observed_at=NOW) for k in range(2)])
        s.commit()
    out = ledger.expire_and_cap(now=NOW)
    assert out == {"expired": 1, "capacity": 2}
    statuses = {i.text: (i.status, i.status_history[-1]["reason"]) for i in _items(db)}
    assert statuses["old filing fact"] == ("retired", "expired")
    # The oldest UNTESTED lessons go; the tested one keeps its place.
    assert statuses["lesson 0"][0] == "active"
    assert [statuses[f"lesson {n}"][0] for n in (1, 2, 3, 4)] == ["retired", "retired", "active", "active"]


# ---------------------------------------------------------------------------
# postmortem_loop: the note and the success rule
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch):
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(postmortem_loop, "record_run", lambda name, **k: rows.append(k))
    monkeypatch.setattr(pm, "_llm_postmortem", lambda memo, outcome, horizon_days, *a: None)
    return rows


def test_loop_note_reports_learning_and_stays_green(db, recorded):
    # The loop's epoch is the real clock, so the historical row predates it.
    _postmortem(db, "NTO", created_at=datetime.utcnow() - timedelta(days=5))
    postmortem_loop.run_once()
    [row] = recorded
    assert row["success"] is True
    note = row["note"]
    assert "learning_written=0" in note and "learning_failed=0" in note
    assert "learning backfilled=1 backfill_failed=0 retired=0 demoted=no expired=0 capacity=0" in note
    assert "judge status=no_llm" in note                     # CI: no key, no spend


def test_loop_is_red_when_the_learning_pass_fails(db, recorded, monkeypatch):
    def boom(**k):
        raise RuntimeError("ledger down")
    monkeypatch.setattr(ledger, "nightly", boom)
    postmortem_loop.run_once()
    assert recorded[0]["success"] is False
    assert "learning nightly failed: RuntimeError: ledger down" in recorded[0]["note"]


def test_loop_is_red_when_a_learning_write_fails(db, recorded, monkeypatch):
    monkeypatch.setattr(postmortem_loop, "run_postmortems", lambda horizon_days, limit: {
        "due": 1, "written": 1, "learning_failed": 1 if horizon_days == 90 else 0,
        "learning_failed_memos": [{"ticker": "NTP", "memo_snapshot_id": 9, "reason": "exception:OperationalError"}]
        if horizon_days == 90 else []})
    postmortem_loop.run_once()
    assert recorded[0]["success"] is False
    assert "learning_failed memos: NTP#9 (exception:OperationalError)" in recorded[0]["note"]


def test_loop_with_learning_off_says_so(db, recorded, monkeypatch):
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    postmortem_loop.run_once()
    assert recorded[0]["success"] is True and recorded[0]["note"].endswith("; learning off")
