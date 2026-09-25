"""W7: the learning mode switch works across processes and fails safe.

Effective mode = min(env ceiling, DB mode), the DB mode defaulting to shadow.
The ceiling "off" never touches the database. A DB read failure is "off"
(prompts revert to today's), never open. The per-process cache is a 60 s
cache OF the database — a demotion written by the other service is seen once
it expires. Promotion to inject is gated (G1-G4) and a forced promotion is
recorded on the event.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event

from app.config import settings
from app.learning import control, ledger
from app.models import LearningControlEvent, LearningRender
from app.tests.learning_helpers import learning_db

NOW = datetime(2026, 10, 20)
_RUNS = itertools.count(1)


@pytest.fixture
def db(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "learning_mode_max", "inject")
    clock = {"t": 1000.0}
    monkeypatch.setattr(control, "_clock", lambda: clock["t"])
    control._cache_clear()
    yield sessions, engine, clock
    control._cache_clear()
    engine.dispose()


def _event(sessions, mode: str, *, at: datetime = NOW, reason: str = "test") -> None:
    with sessions() as s:
        s.add(LearningControlEvent(mode=mode, actor="admin", reason=reason, gates={}, created_at=at))
        s.commit()


def _item(sessions) -> None:
    with sessions() as s:
        ledger._new_item(s, kind="observation", scope_type="company", scope_key="CTL", text="fact",
                         origin_kind="filing_delta", origin_ref="A1", origin_ticker="CTL", now=NOW)
        s.commit()


def _renders(sessions, n: int, *, tickers: tuple[str, ...] = ("CTA", "CTB", "CTC"), linked: bool = True,
             at: datetime = NOW - timedelta(days=1), **kw: Any) -> None:
    with sessions() as s:
        for i in range(n):
            s.add(LearningRender(
                run_id=f"run-{next(_RUNS)}", consumer="pm_memo", ticker=tickers[i % len(tickers)], mode="shadow",
                memo_snapshot_id=100 + i if linked else None, chars=kw.get("chars", 800), items=[], dropped=[],
                error_type=kw.get("error_type"), created_at=at,
            ))
        s.commit()


def test_default_db_mode_is_shadow(db):
    assert control.effective_mode() == "shadow"


def test_effective_mode_is_min_of_ceiling_and_db(db, monkeypatch):
    sessions, _, clock = db
    _event(sessions, "inject")
    assert control.effective_mode() == "inject"
    monkeypatch.setattr(settings, "learning_mode_max", "shadow")
    assert control.effective_mode() == "shadow"             # the ceiling caps the DB
    monkeypatch.setattr(settings, "learning_mode_max", "inject")
    _event(sessions, "off", at=NOW + timedelta(seconds=1))
    clock["t"] += control.CACHE_SECONDS
    assert control.effective_mode() == "off"                # the DB caps the ceiling


def test_ceiling_off_never_reads_the_database(db, monkeypatch):
    _, engine, _ = db
    monkeypatch.setattr(settings, "learning_mode_max", "off")
    statements: list[str] = []

    def count(conn, cursor, statement, *args):
        statements.append(statement)
    event.listen(engine, "before_cursor_execute", count)
    try:
        assert control.effective_mode() == "off"
    finally:
        event.remove(engine, "before_cursor_execute", count)
    assert statements == []


def test_db_error_fails_closed_to_off(db, monkeypatch):
    def broken():
        raise RuntimeError("database unreachable")
    monkeypatch.setattr(control, "SessionLocal", broken)
    assert control.effective_mode() == "off"


def test_unknown_stored_mode_fails_closed(db):
    sessions, _, _ = db
    _event(sessions, "loud")
    assert control.effective_mode() == "off"


def test_cache_ttl_with_injected_clock(db):
    sessions, _, clock = db
    assert control.effective_mode() == "shadow"
    _event(sessions, "inject")                               # written by "the other process"
    clock["t"] += control.CACHE_SECONDS - 1
    assert control.effective_mode() == "shadow"             # still cached
    clock["t"] += 1
    assert control.effective_mode() == "inject"             # expired: re-read


def test_demotion_visible_to_fresh_session(db):
    """A demotion appended by one service reaches the other within the TTL:
    the mode lives in Postgres, not in a module-level flag."""
    sessions, _, clock = db
    _event(sessions, "inject")
    assert control.effective_mode() == "inject"
    with sessions() as other_process:
        assert control.demote_if_injecting(other_process, reason="contamination: [1]", now=NOW) is True
        other_process.commit()
    clock["t"] += control.CACHE_SECONDS
    assert control.effective_mode() == "shadow"
    with sessions() as fresh:
        assert control.db_mode(fresh) == "shadow"


def test_epoch_marker_is_not_a_mode(db):
    sessions, _, _ = db
    _event(sessions, "off")
    with sessions() as s:
        control.ensure_epoch(s, now=NOW + timedelta(days=1))
        s.commit()
        assert control.db_mode(s) == "off"
        assert control.epoch(s) == NOW + timedelta(days=1)
        control.ensure_epoch(s, now=NOW + timedelta(days=9))   # never moved
        s.commit()
        assert control.epoch(s) == NOW + timedelta(days=1)


def test_inject_refused_until_gates_green_then_allowed(db):
    sessions, _, _ = db
    with pytest.raises(control.GatesNotMet) as refused:
        control.set_mode("inject", reason="too early", now=NOW)
    gates = refused.value.gates
    assert gates["promotable"] is False and gates["G4"]["ok"] is False and gates["G2"]["ok"] is False
    with sessions() as s:
        assert s.query(LearningControlEvent).count() == 0     # a refusal writes nothing

    with sessions() as s:
        control.ensure_epoch(s, now=NOW - timedelta(days=8))  # the soak starts at the epoch
        s.commit()
    _item(sessions)
    _renders(sessions, 9)
    with pytest.raises(control.GatesNotMet):                  # 9 linked runs < 10
        control.set_mode("inject", reason="almost", now=NOW)
    _renders(sessions, 5, linked=False)                       # unlinked renders never count (G2)
    with pytest.raises(control.GatesNotMet):
        control.set_mode("inject", reason="almost", now=NOW)
    _renders(sessions, 1, tickers=("CTD",), at=NOW - timedelta(hours=1))
    out = control.set_mode("inject", reason="W7 after soak", now=NOW)
    assert out["gates"]["promotable"] is True and out["effective_mode"] == "inject"
    with sessions() as s:
        row = s.query(LearningControlEvent).filter_by(mode="inject").one()
        assert (row.actor, row.forced, row.reason) == ("admin", False, "W7 after soak")


def test_g3_blocks_on_error_type_or_over_budget(db):
    sessions, _, _ = db
    with sessions() as s:
        control.ensure_epoch(s, now=NOW - timedelta(days=8))
        s.commit()
    _item(sessions)
    _renders(sessions, 12)
    assert control.gates(now=NOW)["promotable"] is True
    _renders(sessions, 1, error_type="OperationalError")
    g = control.gates(now=NOW)
    assert g["G3"]["ok"] is False and len(g["G3"]["errored"]) == 1 and g["promotable"] is False
    _renders(sessions, 1, chars=control.RENDER_BUDGETS["pm_memo"] + 1)
    assert len(control.gates(now=NOW)["G3"]["over_budget"]) == 1


def test_force_is_recorded(db):
    sessions, _, _ = db
    out = control.set_mode("inject", reason="owner override", force=True, now=NOW)
    assert out["gates"]["promotable"] is False and out["effective_mode"] == "inject"
    with sessions() as s:
        row = s.query(LearningControlEvent).one()
        assert (row.mode, row.forced) == ("inject", True)
        assert row.gates["promotable"] is False              # the red gates it overrode


@pytest.mark.parametrize("mode, reason", [("loud", "x"), ("shadow", ""), ("shadow", "   ")])
def test_bad_mode_or_empty_reason_raises(db, mode, reason):
    with pytest.raises(ValueError):
        control.set_mode(mode, reason=reason, now=NOW)
