"""W2a at the HTTP exits: every memo a customer can read is presented, and
the stored row is never written.

Covers each exit the integration plan (S11, critique delta 12) names: the
four `GET /memo` branches (version, cache, store-only, fresh run),
`POST /analyze?sync=true`, the `/memos` history, and the patch-chain walk
that decides which patched fields are credited.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError

from app.database import SessionLocal, engine
from app.main import app
from app.models import MemoSnapshot
from app.schemas import StockMemoOut
from app.services import memo_sections, memo_store
from app.services.memo_sections import UNAVAILABLE_TEXT
from app.tests.gating_helpers import purge_memos

FIXTURES = Path(__file__).parent / "fixtures" / "memo_sections"
PM_TAIL = memo_sections.SIG["pm_view_tail"].text
TICKERS = ("ZZW2A", "ZZW2B", "ZZW2C", "ZZW2D", "ZZW2E", "ZZW2F", "ZZW2G")


def _fixture(name: str, ticker: str) -> StockMemoOut:
    memo = StockMemoOut.model_validate(json.loads((FIXTURES / f"{name}.json").read_text()))
    return memo.model_copy(update={"ticker": ticker})


@pytest.fixture(autouse=True)
def _purge():
    purge_memos(*TICKERS)
    yield
    purge_memos(*TICKERS)


class _Grant:
    def commit(self, db):
        pass

    def release(self, db):
        pass


def _forbidden(*a, **k):
    raise AssertionError("reading a stored memo must neither generate nor backfill")


def _route_setup(monkeypatch, path: str) -> None:
    from app.api import routes_stocks
    from app.config import settings

    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(routes_stocks, "customer_wall_on", lambda: False)
    monkeypatch.setattr(routes_stocks, "authorize", lambda *a, **k: _Grant())
    monkeypatch.setattr(routes_stocks, "run_stock_memo", _forbidden)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", _forbidden)
    monkeypatch.setattr(memo_store, "memo_freshness",
                        lambda *a, **k: {"stale": False, "reason": "", "trigger": None})
    monkeypatch.setattr(settings, "memo_inline_generation", path != "store_only")


def _insert(ticker: str, memo: StockMemoOut, *, version: int, trigger: str = "full_reanalysis",
            parent: int | None = None, revision_log: list[dict[str, Any]] | None = None) -> int:
    """A snapshot as a writer left it (raw memo, no presentation)."""
    with SessionLocal() as db:
        row = MemoSnapshot(ticker=ticker, version=version, trigger=trigger, parent_version=parent,
                           memo_json=json.loads(memo.model_dump_json(exclude={"section_availability"})),
                           revision_log=revision_log or [{"version": version, "trigger": trigger}])
        db.add(row)
        db.commit()
        return row.id


def _stored_json(ticker: str) -> list[Any]:
    with SessionLocal() as db:
        return [deepcopy(r.memo_json) for r in db.scalars(
            select(MemoSnapshot).where(MemoSnapshot.ticker == ticker).order_by(MemoSnapshot.version))]


@pytest.mark.parametrize("path", ["version", "latest_inline", "store_only"])
def test_get_memo_all_branches_present_but_store_untouched(monkeypatch, path):
    _route_setup(monkeypatch, path)
    _insert("ZZW2A", _fixture("googl_live_prepflag", "ZZW2A"), version=1)
    before = _stored_json("ZZW2A")
    url = "/api/stocks/ZZW2A/memo" + ("?version=1" if path == "version" else "")
    with TestClient(app) as client:
        r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["one_sentence_thesis"] == UNAVAILABLE_TEXT
    assert body["section_availability"]["one_sentence_thesis"]["status"] == "unavailable"
    assert PM_TAIL not in r.text
    assert _stored_json("ZZW2A") == before
    assert "section_availability" not in before[0]


def test_get_memo_fresh_run_branch_is_presented(monkeypatch):
    from app.api import routes_stocks
    _route_setup(monkeypatch, "latest_inline")
    raw = _fixture("googl_live_prepflag", "ZZW2B")

    def run(ticker, **_k):
        # What the graph does: persist the raw memo, return it.
        memo_store.save_memo(raw, trigger="first_run")
        return raw

    monkeypatch.setattr(routes_stocks, "run_stock_memo", run)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", lambda t: "auto_analysis")
    with TestClient(app) as client:
        r = client.get("/api/stocks/ZZW2B/memo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final_pm_view"] == UNAVAILABLE_TEXT and PM_TAIL not in r.text
    assert body["section_availability"]["final_pm_view"]["reason"] == "template_fallback"
    stored = _stored_json("ZZW2B")
    assert len(stored) == 1 and "section_availability" not in stored[0]
    assert PM_TAIL in stored[0]["final_pm_view"]


def test_analyze_sync_is_presented(monkeypatch):
    from app.api import routes_stocks
    _route_setup(monkeypatch, "latest_inline")
    raw = _fixture("aapl_demo", "ZZW2C")

    def run(ticker, **_k):
        memo_store.save_memo(raw, trigger="first_run")
        return raw

    monkeypatch.setattr(routes_stocks, "run_stock_memo", run)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", lambda t: "auto_analysis")
    with TestClient(app) as client:
        r = client.post("/api/stocks/ZZW2C/analyze?sync=true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["one_sentence_thesis"] == UNAVAILABLE_TEXT
    assert body["section_availability"]["key_risks"]["status"] == "unavailable"
    assert body["key_risks"] == []
    assert raw.section_availability == {}  # the graph's memo is not mutated


def test_save_memo_never_stores_section_availability():
    presented = memo_sections.present_memo(_fixture("meta_v1", "ZZW2D"))
    with pytest.raises(ValueError, match="presented memo"):
        memo_store.save_memo(presented)
    assert _stored_json("ZZW2D") == []
    raw = _fixture("meta_v1", "ZZW2D")
    memo_store.save_memo(raw)
    assert "section_availability" not in _stored_json("ZZW2D")[0]


def _patch_log(version: int, parent: int, fields: list[str]) -> list[dict[str, Any]]:
    return [{"version": version, "trigger": "incremental_patch", "parent_version": parent,
             "fields_patched": fields}]


def _patch_chain(ticker: str) -> tuple[StockMemoOut, StockMemoOut]:
    """v1 template base (GOOGL shape), v2 patches the thesis, v3 the bull case."""
    base = _fixture("googl_live_prepflag", ticker)
    _insert(ticker, base, version=1)
    v2 = base.model_copy(deep=True)
    v2.one_sentence_thesis = "The PM's patched thesis: ad demand held through the quarter."
    _insert(ticker, v2, version=2, trigger="incremental_patch", parent=1,
            revision_log=_patch_log(2, 1, ["one_sentence_thesis"]))
    v3 = v2.model_copy(deep=True)
    v3.bull_case.key_points = [*v3.bull_case.key_points, "A patched LLM bull point."]
    _insert(ticker, v3, version=3, trigger="incremental_patch", parent=2,
            revision_log=_patch_log(3, 2, ["bull_case"]))
    return base, v3


def _snap(ticker: str, version: int) -> MemoSnapshot:
    snap = memo_store.memo_version(ticker, version)
    assert snap is not None
    return snap


def test_patch_chain_restores_patched_thesis_only():
    _patch_chain("ZZW2E")
    chain = memo_store.patch_chain_for(_snap("ZZW2E", 3))
    assert chain.complete and chain.fields == {"one_sentence_thesis", "bull_case"}
    assert chain.base is not None and PM_TAIL in chain.base.final_pm_view
    shown = memo_store.present_snapshot(_snap("ZZW2E", 3))
    av = shown.section_availability
    assert av["one_sentence_thesis"].status == "available"
    assert shown.one_sentence_thesis.startswith("The PM's patched thesis")
    assert av["confidence_score"].status == "unavailable"  # the patch never re-ran the PM
    assert av["final_pm_view"].status == "unavailable"
    assert "A patched LLM bull point." in shown.bull_case.key_points

    # A chain with a missing hop degrades toward hiding: v2's thesis patch
    # can no longer be credited.
    with SessionLocal() as db:
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == "ZZW2E",
                                      MemoSnapshot.version == 2).delete()
        db.commit()
    broken = memo_store.patch_chain_for(_snap("ZZW2E", 3))
    assert not broken.complete and broken.base is None and broken.fields == {"bull_case"}
    shown = memo_store.present_snapshot(_snap("ZZW2E", 3))
    assert shown.section_availability["one_sentence_thesis"].status == "unavailable"
    assert shown.one_sentence_thesis == UNAVAILABLE_TEXT


def test_patch_chain_evaluates_base_snapshot_through_the_store():
    """The patched PM view is credited; the confidence stays hidden because
    the BASE PM view was the template; the fallback mispricing card is found
    by the base thesis it quotes."""
    base = _fixture("googl_live_prepflag", "ZZW2F")
    _insert("ZZW2F", base, version=1)
    v2 = base.model_copy(deep=True)
    v2.final_pm_view = "The PM re-read the news: ad demand held."
    v2.one_sentence_thesis = "GOOGL is fairly priced — ad demand held."
    _insert("ZZW2F", v2, version=2, trigger="incremental_patch", parent=1,
            revision_log=_patch_log(2, 1, ["final_pm_view", "one_sentence_thesis"]))
    shown = memo_store.present_snapshot(_snap("ZZW2F", 2))
    av = shown.section_availability
    assert av["final_pm_view"].status == "available"
    assert shown.final_pm_view == v2.final_pm_view
    assert av["confidence_score"].status == "unavailable"
    assert av["rating_label"].status == "degraded"
    assert av["mispricing_thesis"].status == "unavailable"


def test_history_nulls_hidden_confidence(monkeypatch):
    _route_setup(monkeypatch, "latest_inline")
    _insert("ZZW2G", _fixture("googl_live_prepflag", "ZZW2G"), version=1)
    clean = _fixture("meta_v1", "ZZW2G")
    _insert("ZZW2G", clean, version=2)
    with TestClient(app) as client:
        rows = client.get("/api/stocks/ZZW2G/memos").json()
    by_version = {r["version"]: r for r in rows}
    assert by_version[1]["confidence_score"] is None
    assert by_version[1]["confidence_available"] is False
    assert by_version[2]["confidence_score"] == pytest.approx(clean.confidence_score)
    assert by_version[2]["confidence_available"] is True


def test_history_limit_clamped_and_walk_is_projected(monkeypatch):
    """limit is bounded 1..50; the chain walk selects only
    (trigger, parent_version, revision_log) and each ancestor at most once
    across the rows of one call, plus one read of the base body."""
    _route_setup(monkeypatch, "latest_inline")
    _patch_chain("ZZW2E")
    with TestClient(app) as client:
        assert client.get("/api/stocks/ZZW2E/memos?limit=0").status_code == 422
        assert client.get("/api/stocks/ZZW2E/memos?limit=51").status_code == 422
        # limit=1 returns only v3, so v2 and v1 must be walked from the DB.
        statements: list[str] = []

        def capture(conn, cursor, statement, params, context, executemany):
            if "memo_snapshots" in statement:
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            rows = client.get("/api/stocks/ZZW2E/memos?limit=1").json()
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert [r["version"] for r in rows] == [3]
        assert rows[0]["confidence_available"] is False
    hops = [s for s in statements if "memo_snapshots.revision_log" in s and "memo_json" not in s]
    bodies = [s for s in statements if "memo_json" in s]
    # One projected hop per ancestor (v2, v1); the listing itself and the base
    # body are the only reads of memo_json.
    assert len(hops) == 2
    assert len(bodies) == 2
    with TestClient(app) as client:
        assert len(client.get("/api/stocks/ZZW2E/memos?limit=50").json()) == 3


def test_history_db_error_is_conservative(monkeypatch):
    _route_setup(monkeypatch, "latest_inline")
    _patch_chain("ZZW2E")
    real = memo_store.patch_chain_for

    def failing(snap, *, db=None, max_hops=50, cache=None):
        class _Broken:
            def execute(self, *a, **k):
                raise OperationalError("select", {}, Exception("db down"))

            def close(self):
                pass
        return real(snap, db=_Broken(), max_hops=max_hops, cache={})  # type: ignore[arg-type]

    monkeypatch.setattr(memo_store, "patch_chain_for", failing)
    with TestClient(app) as client:
        rows = {r["version"]: r for r in client.get("/api/stocks/ZZW2E/memos").json()}
    # The patch rows cannot be walked: their confidence is hidden, not shown.
    assert rows[3]["confidence_available"] is False and rows[3]["confidence_score"] is None
