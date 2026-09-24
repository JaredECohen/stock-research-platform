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
TICKERS = ("ZZW2A", "ZZW2B", "ZZW2C", "ZZW2D", "ZZW2E", "ZZW2F", "ZZW2G", "ZZW2H", "ZZW2I",
           "ZZW2J", "ZZW2K", "ZZW2L", "ZZW2M")


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


class _Savepoint:
    def __init__(self, log: list[str]):
        self.log = log

    def __enter__(self):
        self.log.append("savepoint")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append("rollback to savepoint" if exc_type else "release savepoint")
        return False


class _BrokenSession:
    """A caller's session whose every statement fails (the database is down)."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def begin_nested(self):
        return _Savepoint(self.log)

    def execute(self, *a, **k):
        self.log.append("execute")
        raise OperationalError("select", {}, Exception("db down"))

    def close(self):
        self.log.append("close")


def test_chain_walk_error_rolls_back_only_its_savepoint():
    """On Postgres a failed statement aborts the caller's transaction, so the
    walk's error must be contained in a SAVEPOINT on a session the caller
    owns (history's shared session, the commentary request, the sample
    build) — and that session is the caller's to close, not the walk's."""
    _patch_chain("ZZW2E")
    broken = _BrokenSession()
    chain = memo_store.patch_chain_for(_snap("ZZW2E", 3), db=broken, cache={})  # type: ignore[arg-type]
    assert not chain.complete and chain.base is None
    assert broken.log == ["savepoint", "execute", "rollback to savepoint"]


def test_history_db_error_is_conservative(monkeypatch):
    _route_setup(monkeypatch, "latest_inline")
    _patch_chain("ZZW2E")
    real = memo_store.patch_chain_for

    def failing(snap, *, db=None, max_hops=50, cache=None):
        return real(snap, db=_BrokenSession(), max_hops=max_hops, cache={})  # type: ignore[arg-type]

    monkeypatch.setattr(memo_store, "patch_chain_for", failing)
    with TestClient(app) as client:
        rows = {r["version"]: r for r in client.get("/api/stocks/ZZW2E/memos").json()}
    # The patch rows cannot be walked: their confidence is hidden, not shown.
    assert rows[3]["confidence_available"] is False and rows[3]["confidence_score"] is None


def _insert_json(ticker: str, memo_json: dict[str, Any], *, version: int,
                 trigger: str = "full_reanalysis", parent: int | None = None,
                 revision_log: list[dict[str, Any]] | None = None) -> None:
    with SessionLocal() as db:
        db.add(MemoSnapshot(ticker=ticker, version=version, trigger=trigger, parent_version=parent,
                            memo_json=memo_json,
                            revision_log=revision_log or [{"version": version, "trigger": trigger}]))
        db.commit()


def _unreadable_json(ticker: str) -> dict[str, Any]:
    """A stored row in the AMBIGUOUS legacy case shape `memo_to_pydantic`
    refuses (see test_memo_unreadable)."""
    body = json.loads(_fixture("meta_v1", ticker).model_dump_json(exclude={"section_availability"}))
    body["bull_case"] = [{"key_point": None}]
    return body


def test_history_serves_an_unreadable_row_raw(monkeypatch):
    """Every history row is now validated; one row the schema outgrew must
    not turn the listing into a 500. It keeps its raw number with
    `confidence_available: null`; its readable neighbours are presented."""
    _route_setup(monkeypatch, "latest_inline")
    _insert("ZZW2H", _fixture("googl_live_prepflag", "ZZW2H"), version=1)
    bad = _unreadable_json("ZZW2H")
    _insert_json("ZZW2H", bad, version=2)
    with TestClient(app) as client:
        r = client.get("/api/stocks/ZZW2H/memos")
    assert r.status_code == 200, r.text
    rows = {row["version"]: row for row in r.json()}
    assert rows[2]["confidence_available"] is None
    assert rows[2]["confidence_score"] == pytest.approx(bad["confidence_score"])
    assert rows[1]["confidence_available"] is False and rows[1]["confidence_score"] is None


def _llm_base(ticker: str) -> StockMemoOut:
    return _fixture("meta_v1", ticker)


@pytest.mark.parametrize("exit_", ["unlogged_hop", "no_parent", "unreadable_base", "max_hops"])
def test_patch_chain_incomplete_exits_hide_confidence(exit_):
    """Each way a chain cannot be walked to its base is conservative: over an
    LLM base (so the template-PM rule cannot mask it) the confidence is
    hidden with basis `patch_chain:incomplete`, and a hop whose fields were
    never logged credits none."""
    t = {"unlogged_hop": "ZZW2I", "no_parent": "ZZW2J", "unreadable_base": "ZZW2K",
         "max_hops": "ZZW2L"}[exit_]
    base = _llm_base(t)
    v2 = base.model_copy(update={"one_sentence_thesis": "A patched thesis after the news."})
    if exit_ == "unreadable_base":
        _insert_json(t, _unreadable_json(t), version=1)
    else:
        _insert(t, base, version=1)
    log = _patch_log(2, 1, ["one_sentence_thesis"])
    if exit_ == "unlogged_hop":
        log = [{"version": 2, "trigger": "incremental_patch", "parent_version": 1}]
    _insert(t, v2, version=2, trigger="incremental_patch",
            parent=None if exit_ == "no_parent" else 1, revision_log=log)
    target = 2
    kwargs: dict[str, Any] = {}
    if exit_ == "max_hops":
        _insert(t, v2, version=3, trigger="incremental_patch", parent=2,
                revision_log=_patch_log(3, 2, ["bull_case"]))
        target, kwargs = 3, {"max_hops": 1}
    chain = memo_store.patch_chain_for(_snap(t, target), **kwargs)
    assert not chain.complete and chain.base is None
    if exit_ == "unlogged_hop":
        assert chain.fields == frozenset()
    if exit_ != "max_hops":
        shown = memo_store.present_snapshot(_snap(t, target))
        av = shown.section_availability
        assert av["confidence_score"].status == "unavailable"
        assert av["confidence_score"].basis == ["patch_chain:incomplete"]
        if exit_ == "unlogged_hop":
            # Nothing credited: the thesis is judged as written, not as patched.
            assert av["one_sentence_thesis"].basis != ["patched:one_sentence_thesis"]
    # A complete two-hop walk over the same rows is not incomplete.
    if exit_ == "max_hops":
        assert memo_store.patch_chain_for(_snap(t, 3)).complete


def test_history_chain_walk_is_shared_across_rows(monkeypatch):
    """One history call walks each ancestor at most once across its rows:
    every listed row pre-seeds the hop cache (no hop SELECT), and the base
    body v2 and v3 both rest on is read once."""
    _route_setup(monkeypatch, "latest_inline")
    _patch_chain("ZZW2E")
    statements: list[str] = []

    def capture(conn, cursor, statement, params, context, executemany):
        if "memo_snapshots" in statement:
            statements.append(statement)

    with TestClient(app) as client:
        event.listen(engine, "before_cursor_execute", capture)
        try:
            rows = client.get("/api/stocks/ZZW2E/memos?limit=50").json()
        finally:
            event.remove(engine, "before_cursor_execute", capture)
    assert [r["version"] for r in rows] == [3, 2, 1]
    hops = [s for s in statements if "memo_snapshots.revision_log" in s and "memo_json" not in s]
    bodies = [s for s in statements if "memo_json" in s]
    assert hops == []
    assert len(bodies) == 2  # the listing, and the one shared base body


def test_patched_thesis_keeps_the_base_verdict_hidden_through_the_store(monkeypatch):
    """A pinned ticker's news patch replaces the thesis; the verdict still
    quotes the base builder thesis and stays hidden on GET /memo."""
    _route_setup(monkeypatch, "latest_inline")
    base = _llm_base("ZZW2M")
    thesis = ("ZZW2M screens undervalued on the blended read, though no single specialist headline "
              "defines the call. The market is under-pricing the durable part of the franchise.")
    base = base.model_copy(update={
        "one_sentence_thesis": thesis,
        "final_verdict": f"PM final view: Bullish (confidence 62). {thesis} Watch items: none flagged.",
        "section_provenance": {"v": 1, "llm_configured": True, "thesis": "rewrite", "mispricing": "pm"},
    })
    _insert("ZZW2M", base, version=1)
    v2 = base.model_copy(update={"one_sentence_thesis": "Ad pricing held through the quarter."})
    _insert("ZZW2M", v2, version=2, trigger="incremental_patch", parent=1,
            revision_log=_patch_log(2, 1, ["one_sentence_thesis"]))
    assert memo_store.present_snapshot(_snap("ZZW2M", 1)).final_verdict == UNAVAILABLE_TEXT
    with TestClient(app) as client:
        r = client.get("/api/stocks/ZZW2M/memo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["one_sentence_thesis"] == "Ad pricing held through the quarter."
    assert body["final_verdict"] == UNAVAILABLE_TEXT
    assert body["section_availability"]["final_verdict"]["basis"] == ["derived:base_thesis"]
    assert thesis not in r.text


def test_earned_confidence_reaches_history(monkeypatch):
    """The W2b hook end to end: an earned confidence over a template PM is a
    number in the history listing and the chat context, not a null."""
    from app.agents import orchestrator as orch_mod
    _route_setup(monkeypatch, "latest_inline")
    memo = _fixture("googl_live_prepflag", "ZZW2G").model_copy(
        update={"section_provenance": {"confidence": "earned"}})
    _insert("ZZW2G", memo, version=1)
    with TestClient(app) as client:
        rows = client.get("/api/stocks/ZZW2G/memos").json()
    assert rows[0]["confidence_available"] is True
    assert rows[0]["confidence_score"] == pytest.approx(memo.confidence_score)
    ctx = orch_mod._memo_for_chat_context(memo_store.present_snapshot(_snap("ZZW2G", 1)))
    assert "confidence" in ctx and "thesis" not in ctx
