"""W6 / FIX-007 / FIX-003 — the outcome-eligibility ledger and its rule.

Owner decision 2026-09-24: outcomes on the demo-mode memos copied from the
development laptop on 2026-05-04 are EXCLUDED from the track record and the
learning loop, never deleted; FIX-003's migrated test fixtures likewise. Every
exact-count test here runs on a private sqlite engine: the suite's shared
database would make a count a statement about test ordering.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import sessionmaker

from app.models import (
    Company,
    MemoOutcome,
    MemoOutcomeEligibility,
    MemoPostmortem,
    MemoSnapshot,
)
from app.services import market_data_backfill, outcome_service
from app.services import outcome_eligibility as oe
from app.services.outcome_eligibility import Classification
from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS, FIXTURE_SNAPSHOTS
from app.tests.eligibility_helpers import add_outcome, add_snapshot, classify_all, isolated_sessions, mark

# Evidence rows used as dev-copy seeds: the guard only accepts the enumerated
# (id, ticker) set, and fingerprints the exact generated_at.
_DEV = {sid: (ticker, datetime.fromisoformat(g)) for sid, ticker, g in DEV_COPY_SNAPSHOTS}
NVDA_DEV_ID = next(sid for sid, (t, _) in sorted(_DEV.items()) if t == "NVDA")
MSFT_DEV_ID = next(sid for sid, (t, _) in sorted(_DEV.items()) if t == "MSFT")


@pytest.fixture
def iso(tmp_path, monkeypatch):
    sessions, engine = isolated_sessions(tmp_path, monkeypatch, outcome_service, market_data_backfill)
    yield sessions, engine
    engine.dispose()


def _dev_copy(db, sid, *, mode="demo"):
    ticker, generated = _DEV[sid]
    return add_snapshot(
        db, id=sid, ticker=ticker, generated_at=generated, mode=mode,
        memo_generated_at=generated.isoformat(), version=sid,
    )


def _ledger(sessions) -> dict[int, MemoOutcomeEligibility]:
    with sessions() as db:
        return {r.memo_snapshot_id: r for r in db.execute(select(MemoOutcomeEligibility)).scalars()}


# ---------------------------------------------------------------------------
# The pure rule
# ---------------------------------------------------------------------------

def _rule(sid, ticker="XYZ", *, analysis=None, generated=None, mode="demo", trigger="full_reanalysis",
          as_of=None, parent=None, parent_id=None):
    return oe.classify_snapshot(
        snapshot_id=sid, ticker=ticker, snapshot_generated_at=generated or analysis,
        analysis_generated_at=analysis, generation_mode=mode, trigger=trigger,
        as_of_date=as_of, parent_classification=parent, parent_snapshot_id=parent_id,
    )


ELIG_LIVE = Classification(True, oe.REASON_LIVE)
EXCL_DEV = Classification(False, oe.REASON_DEV_COPY)


@pytest.mark.parametrize("kwargs, expected", [
    # FIX-003 evidence boundaries.
    (dict(sid=299, analysis=datetime(2026, 5, 4, 8, 0)), (False, oe.REASON_DEV_COPY)),
    # BAC 559: absent from the laptop file, labelled demo in production, and
    # copied at 07:59Z — part of the dev copy (600 rows on 300 snapshots).
    (dict(sid=559, ticker="BAC", analysis=datetime(2026, 5, 4, 7, 59, 14)), (False, oe.REASON_DEV_COPY)),
    (dict(sid=582, analysis=datetime(2026, 5, 11, 23, 59)), (False, oe.REASON_DEV_COPY)),
    # Production-native ids that reused gaps (17-28) and the first 583+.
    (dict(sid=17, analysis=datetime(2026, 5, 12, 23, 53)), (True, oe.REASON_LABEL_PREDATES_FIX)),
    (dict(sid=583, analysis=datetime(2026, 5, 31)), (True, oe.REASON_LABEL_PREDATES_FIX)),
    (dict(sid=583, analysis=datetime(2026, 5, 4)), (True, oe.REASON_LABEL_PREDATES_FIX)),
    # The label fix (7961289) reached main 2026-06-15T00:59:06Z.
    (dict(sid=700, analysis=datetime(2026, 6, 15, 0, 59, 59)), (True, oe.REASON_LABEL_PREDATES_FIX)),
    (dict(sid=700, analysis=datetime(2026, 6, 15, 1, 0)), (False, oe.REASON_NO_LLM_OR_DEMO)),
    # Worker-without-LLM-keys memos (AAPL v7, AMZN v2, AVGO v1, GOOGL v7).
    (dict(sid=640, ticker="AAPL", analysis=datetime(2026, 9, 10)), (False, oe.REASON_NO_LLM_OR_DEMO)),
    (dict(sid=700, analysis=None, generated=datetime(2026, 7, 1)), (False, oe.REASON_NO_LLM_OR_DEMO)),
    # Mode normalisation and fail-closed modes.
    (dict(sid=700, analysis=datetime(2026, 7, 1), mode="LIVE "), (True, oe.REASON_LIVE)),
    (dict(sid=300, analysis=datetime(2026, 5, 4), mode="live"), (True, oe.REASON_LIVE)),  # laptop live memo
    (dict(sid=700, analysis=datetime(2026, 7, 1), mode=None), (False, oe.REASON_UNRECORDED)),
    (dict(sid=700, analysis=datetime(2026, 7, 1), mode=""), (False, oe.REASON_UNRECORDED)),
    (dict(sid=700, analysis=datetime(2026, 7, 1), mode="backtest"), (False, oe.REASON_UNRECOGNIZED)),
    # Backtests are excluded whatever their mode.
    (dict(sid=700, analysis=datetime(2026, 7, 1), mode="live", as_of=datetime(2025, 6, 30)),
     (False, oe.REASON_BACKTEST)),
    # FIX-003 fixtures take precedence over every rule but the backtest one...
    (dict(sid=562, ticker="TSTONE", analysis=datetime(2026, 1, 4, 8, 2), mode=None), (False, oe.REASON_TEST_FIXTURE)),
    # ASOFT1 525 is test_as_of_date.py's v1 (no as_of_date, with pending
    # 30d/90d pairs in FIX-003 §2); its v2 at 526 is a backtest the evaluator
    # never logged, so it is named a backtest, not an unlisted fixture.
    (dict(sid=525, ticker="ASOFT1", analysis=datetime(2026, 5, 4), mode="live"), (False, oe.REASON_TEST_FIXTURE)),
    (dict(sid=526, ticker="ASOFT1", analysis=datetime(2026, 5, 4), mode="live", as_of=datetime(2025, 6, 30)),
     (False, oe.REASON_BACKTEST)),
    (dict(sid=577, ticker="TSTPATCH", analysis=datetime(2026, 5, 4), trigger="incremental_patch"),
     (False, oe.REASON_TEST_FIXTURE)),
    # ...but only for rows the copy could have carried.
    (dict(sid=700, ticker="TSTONE", analysis=datetime(2026, 6, 20), mode="live"), (True, oe.REASON_LIVE)),
    (dict(sid=562, ticker="TSTONE", analysis=datetime(2026, 5, 6), mode="live"), (True, oe.REASON_LIVE)),
    (dict(sid=562, ticker="TSTONEX", analysis=datetime(2026, 1, 4), mode=None), (False, oe.REASON_UNRECORDED)),
])
def test_rule_table(kwargs, expected):
    result = _rule(**kwargs)
    assert (result.eligible, result.reason) == expected
    assert result.inherited_from_snapshot_id is None


@pytest.mark.parametrize("parent, own_mode, expected", [
    # A patch copies its parent's body: a post-fix "demo" label on a patch of
    # a live memo is the parent's lineage, not new evidence (eligible)...
    (ELIG_LIVE, "demo", (True, oe.REASON_LIVE)),
    # ...and a "live" label on a patch of the dev copy is still the dev copy.
    (EXCL_DEV, "live", (False, oe.REASON_DEV_COPY)),
    (Classification(True, oe.REASON_LABEL_PREDATES_FIX), None, (True, oe.REASON_LABEL_PREDATES_FIX)),
    (None, "live", (False, oe.REASON_PATCH_PARENT_MISSING)),
])
def test_rule_table_patch_inheritance(parent, own_mode, expected):
    result = _rule(
        700, analysis=datetime(2026, 7, 1), mode=own_mode, trigger="incremental_patch",
        parent=parent, parent_id=None if parent is None else 42,
    )
    assert (result.eligible, result.reason) == expected
    assert result.inherited_from_snapshot_id == (None if parent is None else 42)


def test_parse_timestamp_converts_offsets_and_rejects_garbage():
    assert oe.parse_timestamp("2026-06-15T02:00:00+02:00") == datetime(2026, 6, 15, 0, 0)
    assert oe.parse_timestamp("2026-06-15T01:00:00Z") == datetime(2026, 6, 15, 1, 0)
    assert oe.parse_timestamp("not a date") is None
    assert oe.parse_timestamp(None) is None


# ---------------------------------------------------------------------------
# Rating source
# ---------------------------------------------------------------------------

KEYWORD_VIEW = (
    "Research view: Bullish. A thesis. Sector framing supports the cohort thesis; "
    "valuation-relative read is the main swing factor. The risk committee flagged the dominant downside "
    "scenarios; portfolio fit depends on macro view."
)


@pytest.mark.parametrize("trigger, mode, view, degraded, expected", [
    ("incremental_patch", "live", "anything", None, oe.SOURCE_PATCH),
    ("full_reanalysis", "live", "PM synthesis unavailable; relying on specialist findings only.", None,
     oe.SOURCE_FALLBACK_PM),
    ("full_reanalysis", "live", KEYWORD_VIEW + " PM rationale before those adjustments (rating Bullish).", None,
     oe.SOURCE_KEYWORD_PM),
    ("full_reanalysis", "live", "An LLM-written view.", '["PM Synthesis"]', oe.SOURCE_KEYWORD_PM),
    ("full_reanalysis", "demo", "An LLM-looking view.", None, oe.SOURCE_KEYWORD_PM),
    ("full_reanalysis", "live", "An LLM-written view.", '["Fundamental Scorecard"]', oe.SOURCE_LLM_PM),
    ("full_reanalysis", "live", "", None, oe.SOURCE_UNKNOWN),
    ("full_reanalysis", None, "An LLM-written view.", None, oe.SOURCE_UNKNOWN),
])
def test_rating_source_rules(trigger, mode, view, degraded, expected):
    assert oe.rating_source(
        trigger=trigger, generation_mode=mode, final_pm_view=view, degraded_agents=degraded,
    ) == expected


def test_rating_source_projection(iso):
    """The sweep derives rating_source from JSON-path strings, not bodies."""
    sessions, _ = iso
    at = datetime(2026, 8, 1)
    with sessions() as db:
        kw = add_snapshot(db, ticker="KW", generated_at=at, final_pm_view=KEYWORD_VIEW)
        llm = add_snapshot(db, ticker="LLM", generated_at=at, final_pm_view="A real view.",
                           degraded_agents=["Fundamental Scorecard"])
        deg = add_snapshot(db, ticker="DEG", generated_at=at, final_pm_view="x", degraded_agents=["PM Synthesis"])
        fb = add_snapshot(db, ticker="FB", generated_at=at,
                          final_pm_view="PM synthesis unavailable; relying on specialist findings only.")
        patch = add_snapshot(db, ticker="LLM", version=2, parent_version=1, trigger="incremental_patch",
                             generated_at=at + timedelta(days=1), final_pm_view="A real view.")
        db.commit()
        classify_all(db)
    ledger = _ledger(sessions)
    assert {t: ledger[s.id].rating_source for t, s in
            (("kw", kw), ("llm", llm), ("deg", deg), ("fb", fb), ("patch", patch))} == {
        "kw": "keyword_pm", "llm": "llm_pm", "deg": "keyword_pm", "fb": "fallback_pm", "patch": "patch",
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def test_sweep_resolves_lineage_and_fixtures(iso):
    sessions, _ = iso
    with sessions() as db:
        nvda = _dev_copy(db, NVDA_DEV_ID)
        bac = _dev_copy(db, 559)
        # Patch of the dev copy: its own label says "live", it is still excluded.
        nvda_patch = add_snapshot(db, id=600, ticker="NVDA", version=nvda.version + 1,
                                  parent_version=nvda.version, trigger="incremental_patch",
                                  generated_at=datetime(2026, 6, 20), mode="live")
        msft = add_snapshot(db, id=583, ticker="MSFT", version=1000, generated_at=datetime(2026, 6, 1))
        msft_patch = add_snapshot(db, id=601, ticker="MSFT", version=1001, parent_version=1000,
                                  trigger="incremental_patch", generated_at=datetime(2026, 7, 1), mode="demo")
        msft_patch2 = add_snapshot(db, id=602, ticker="MSFT", version=1002, parent_version=1001,
                                   trigger="incremental_patch", generated_at=datetime(2026, 7, 2), mode="demo")
        orphan_patch = add_snapshot(db, id=603, ticker="AMZN", version=3, parent_version=2,
                                    trigger="incremental_patch", generated_at=datetime(2026, 7, 3))
        # A patch whose parent sits at a HIGHER id (production reused gaps
        # 17-28 before the sequence jumped past the copy).
        gap_parent = add_snapshot(db, id=590, ticker="AMD", version=1, generated_at=datetime(2026, 5, 20),
                                  mode="demo", memo_generated_at="2026-05-20T10:00:00")
        gap_patch = add_snapshot(db, id=18, ticker="AMD", version=2, parent_version=1,
                                 trigger="incremental_patch", generated_at=datetime(2026, 5, 21))
        native = add_snapshot(db, id=17, ticker="ABBV", generated_at=datetime(2026, 5, 12, 23, 53), mode="demo")
        worker_no_llm = add_snapshot(db, id=640, ticker="AAPL", version=7, generated_at=datetime(2026, 9, 10),
                                     mode="demo")
        backtest = add_snapshot(db, id=610, ticker="NVDA", version=5000, generated_at=datetime(2026, 7, 1),
                                as_of_date=datetime(2025, 6, 30))
        fixture = add_snapshot(db, id=562, ticker="TSTONE", generated_at=datetime(2026, 1, 4, 8, 2, 41), mode=None)
        same_ticker_later = add_snapshot(db, id=620, ticker="TSTONE", version=2, generated_at=datetime(2026, 6, 20))
        db.commit()
        summary = oe.classify_pending(db=db, batch_size=3)
    ledger = _ledger(sessions)
    got = {s.id: (ledger[s.id].eligible, ledger[s.id].reason, ledger[s.id].inherited_from_snapshot_id)
           for s in (nvda, bac, nvda_patch, msft, msft_patch, msft_patch2, orphan_patch, gap_parent, gap_patch,
                     native, worker_no_llm, backtest, fixture, same_ticker_later)}
    assert got == {
        nvda.id: (False, oe.REASON_DEV_COPY, None),
        bac.id: (False, oe.REASON_DEV_COPY, None),
        nvda_patch.id: (False, oe.REASON_DEV_COPY, nvda.id),
        msft.id: (True, oe.REASON_LIVE, None),
        msft_patch.id: (True, oe.REASON_LIVE, msft.id),
        msft_patch2.id: (True, oe.REASON_LIVE, msft_patch.id),
        orphan_patch.id: (False, oe.REASON_PATCH_PARENT_MISSING, None),
        gap_parent.id: (True, oe.REASON_LABEL_PREDATES_FIX, None),
        gap_patch.id: (True, oe.REASON_LABEL_PREDATES_FIX, gap_parent.id),
        native.id: (True, oe.REASON_LABEL_PREDATES_FIX, None),
        worker_no_llm.id: (False, oe.REASON_NO_LLM_OR_DEMO, None),
        backtest.id: (False, oe.REASON_BACKTEST, None),
        fixture.id: (False, oe.REASON_TEST_FIXTURE, None),
        same_ticker_later.id: (True, oe.REASON_LIVE, None),
    }
    assert summary["classified"] == 14
    assert summary["listed"][oe.REASON_LABEL_PREDATES_FIX] == sorted([gap_parent.id, gap_patch.id, native.id])
    assert summary["listed"][oe.REASON_NO_LLM_OR_DEMO] == [worker_no_llm.id]
    # Identity columns are the snapshot's own.
    assert ledger[nvda.id].ticker == "NVDA" and ledger[nvda.id].snapshot_generated_at == _DEV[NVDA_DEV_ID][1]


def test_fix003_fixture_set_is_excluded_and_not_counted_by_the_due_scan(iso, monkeypatch):
    """FIX-003: the migrated fixtures stop driving the nightly pending ERROR.

    All 34 enumerated fixture snapshots are classified with the explicit
    reason and are neither priced nor counted as due; a same-ticker snapshot
    written later (id > 582) is evaluated as ordinary research.
    """
    sessions, _ = iso
    with sessions() as db:
        for sid, ticker in FIXTURE_SNAPSHOTS:
            generated = datetime(2026, 1, 4, 8, 2) if ticker.startswith("TST") and sid in range(562, 572) \
                else datetime(2026, 5, 4, 7, 0)
            add_snapshot(db, id=sid, ticker=ticker, version=sid, generated_at=generated, mode=None)
        later = add_snapshot(db, id=700, ticker="TSTONE", version=9000, generated_at=datetime(2026, 6, 1))
        db.commit()
    requests = []

    def no_prices(ticker, days):
        requests.append(ticker)
        return []

    from app.services import market_data_service, price_history_service
    monkeypatch.setattr(market_data_service, "get_price_series", no_prices)
    monkeypatch.setattr(price_history_service, "read_prices", lambda ticker, **k: requests.append(ticker) or [])
    report = outcome_service.evaluate_all_due(horizons=[30, 90], today=date(2026, 9, 24))
    ledger = _ledger(sessions)
    fixture_ids = {sid for sid, row in ledger.items() if row.reason == oe.REASON_TEST_FIXTURE}
    assert fixture_ids == {sid for sid, _ in FIXTURE_SNAPSHOTS}
    assert report["ineligible_snapshots_by_reason"] == {oe.REASON_TEST_FIXTURE: len(FIXTURE_SNAPSHOTS)}
    assert report["ineligible"] == 2 * len(FIXTURE_SNAPSHOTS)
    # Only the later TSTONE snapshot is due, priced, and pending.
    assert report["due"] == report["data_unavailable"] == 2
    assert report["unavailable_pairs"] == [f"TSTONE:snap={later.id}:30d:ticker_prices_unavailable",
                                           f"TSTONE:snap={later.id}:90d:ticker_prices_unavailable"]
    assert requests == ["TSTONE", "TSTONE"]
    assert report["unclassified"] == 0


def test_expected_set_guard_aborts_the_whole_sweep(iso):
    """Over- and under-exclusion both roll back every batch and raise."""
    sessions, _ = iso
    # Over-exclusion: a demo snapshot in the copy's id range that is NOT in
    # the enumerated evidence would be classified as the dev copy.
    with sessions() as db:
        add_snapshot(db, id=5, ticker="GOOD", generated_at=datetime(2026, 8, 1))
        stray = next(i for i in range(400, 583) if i not in _DEV)
        add_snapshot(db, id=stray, ticker="ZZZ", generated_at=datetime(2026, 5, 4), mode="demo")
        db.commit()
        with pytest.raises(oe.ExclusionSetMismatch, match="outside the enumerated set"):
            oe.classify_pending(db=db, batch_size=1)
    assert _ledger(sessions) == {}, "a mismatch must leave nothing written, not even earlier batches"


def test_expected_set_guard_catches_under_exclusion(iso):
    sessions, _ = iso
    with sessions() as db:
        _dev_copy(db, MSFT_DEV_ID, mode="live")   # the exact evidence row, but not demo
        db.commit()
        with pytest.raises(oe.ExclusionSetMismatch, match="classified as live_generation"):
            oe.classify_pending(db=db)
    assert _ledger(sessions) == {}


def test_fixture_ticker_backtest_does_not_trip_the_fixture_guard(iso):
    """The copy carried ASOFT1 v1 (525, enumerated) AND its v2 backtest (526).

    526 never reached a log (the evaluator skips as_of rows), so it is not in
    the enumerated fixture set. Named a fixture, it tripped the guard and the
    first production sweep aborted, writing nothing.
    """
    sessions, _ = iso
    with sessions() as db:
        v1 = add_snapshot(db, id=525, ticker="ASOFT1", version=1, generated_at=datetime(2026, 5, 2, 12, 0))
        v2 = add_snapshot(db, id=526, ticker="ASOFT1", version=2, parent_version=1,
                          generated_at=datetime(2026, 5, 2, 12, 0, 1), as_of_date=datetime(2025, 6, 30))
        live = add_snapshot(db, id=900, ticker="AAPL", generated_at=datetime(2026, 8, 1))
        db.commit()
        summary = oe.classify_pending(db=db)
    assert summary["classified"] == 3
    ledger = _ledger(sessions)
    assert {sid: (row.eligible, row.reason) for sid, row in ledger.items()} == {
        v1.id: (False, oe.REASON_TEST_FIXTURE),
        v2.id: (False, oe.REASON_BACKTEST),
        live.id: (True, oe.REASON_LIVE),
    }


def test_expected_set_guard_catches_unlisted_fixture(iso):
    sessions, _ = iso
    with sessions() as db:
        add_snapshot(db, id=100, ticker="TSTONE", generated_at=datetime(2026, 1, 4), mode=None)
        db.commit()
        with pytest.raises(oe.ExclusionSetMismatch, match="fixture classification"):
            oe.classify_pending(db=db)


def _all_rows(db):
    return (
        db.execute(select(*MemoSnapshot.__table__.c).order_by(MemoSnapshot.id)).all(),
        db.execute(select(*MemoOutcome.__table__.c).order_by(MemoOutcome.id)).all(),
        db.execute(select(*MemoPostmortem.__table__.c).order_by(MemoPostmortem.id)).all(),
    )


def test_classify_pending_is_idempotent_and_writes_nothing_else(iso):
    """Exclusion, never deletion: only the derived ledger is ever written."""
    sessions, engine = iso
    with sessions() as db:
        dev = _dev_copy(db, NVDA_DEV_ID)
        live = add_snapshot(db, ticker="LIVE", generated_at=datetime(2026, 6, 1))
        for snap in (dev, live):
            add_outcome(db, snap, horizon=30, forward_return=0.1, alpha=0.02)
        db.add(MemoPostmortem(memo_snapshot_id=dev.id, ticker=dev.ticker, horizon_days=30, verdict="right"))
        db.commit()
        before = _all_rows(db)
    writes = []

    def capture(conn, cursor, statement, params, context, executemany):
        head = statement.lstrip().split(None, 1)[0].upper()
        if head in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}:
            writes.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with sessions() as db:
            first = oe.classify_pending(db=db)
        first_writes = list(writes)
        writes.clear()
        with sessions() as db:
            second = oe.classify_pending(db=db)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert first["classified"] == 2 and second["classified"] == 0
    assert first_writes and all("memo_outcome_eligibility" in s for s in first_writes)
    assert writes == [], "a second sweep must not write anything"
    with sessions() as db:
        assert _all_rows(db) == before


def test_classify_pending_reads_no_bodies_into_python(iso):
    sessions, engine = iso
    with sessions() as db:
        add_snapshot(db, ticker="BODY", generated_at=datetime(2026, 6, 1), unused_body="x" * 200_000)
        db.commit()
    statements = []

    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with sessions() as db:
            oe.classify_pending(db=db)
            facts = oe._project(db, MemoSnapshot.ticker == "BODY")
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    body_refs = [s for s in statements if "memo_snapshots.memo_json" in s]
    assert body_refs, "the projection should appear in the capture"
    for s in body_refs:
        # Every reference is inside a JSON path extraction.
        assert s.count("memo_snapshots.memo_json") == s.count("JSON_EXTRACT(memo_snapshots.memo_json"), s
    for value in (facts[0].generation_mode, facts[0].sector, facts[0].final_pm_view, facts[0].degraded_agents):
        assert value is None or isinstance(value, str)


def test_rule_version_bump_reclassifies_once(iso, monkeypatch):
    sessions, _ = iso
    with sessions() as db:
        for i in range(3):
            add_snapshot(db, ticker=f"V{i}", generated_at=datetime(2026, 6, 1))
        db.commit()
        assert oe.classify_pending(db=db)["classified"] == 3
    monkeypatch.setattr(oe, "RULE_VERSION", 2)
    monkeypatch.setattr(oe, "classify_snapshot", lambda **kw: Classification(False, "stub_rule_v2"))
    with sessions() as db:
        assert oe.classify_pending(db=db)["classified"] == 3
        assert oe.classify_pending(db=db)["classified"] == 0
    assert {(r.rule_version, r.reason) for r in _ledger(sessions).values()} == {(2, "stub_rule_v2")}


def test_concurrent_writer_is_harmless(iso, monkeypatch):
    """Another sweep writing the same rows between our two queries is fine."""
    sessions, _ = iso
    with sessions() as db:
        target = add_snapshot(db, ticker="RACE", generated_at=datetime(2026, 6, 1))
        add_snapshot(db, ticker="RACE2", generated_at=datetime(2026, 6, 1))
        db.commit()
    real = oe._project
    raced = []

    def project_after_other_writer(db, where):
        if not raced:
            raced.append(True)
            with sessions() as other:
                mark(other, target.id, eligible=True, reason=oe.REASON_LIVE, rating_source="unknown")
        return real(db, where)

    monkeypatch.setattr(oe, "_project", project_after_other_writer)
    with sessions() as db:
        oe.classify_pending(db=db)
    ledger = _ledger(sessions)
    assert len(ledger) == 2 and ledger[target.id].eligible is True


def test_no_progress_raises(iso, monkeypatch):
    sessions, _ = iso
    with sessions() as db:
        add_snapshot(db, ticker="STUCK", generated_at=datetime(2026, 6, 1))
        db.commit()
    monkeypatch.setattr(oe, "_upsert", lambda *a, **k: None)
    with sessions() as db, pytest.raises(RuntimeError, match="no progress"):
        oe.classify_pending(db=db)


def test_unsupported_dialect_raises():
    class _Bind:
        class dialect:  # noqa: N801 - mimics the SQLAlchemy attribute
            name = "mysql"

    class _Session:
        def get_bind(self):
            return _Bind()

    with pytest.raises(NotImplementedError, match="mysql"):
        oe.classify_pending(db=_Session())  # type: ignore[arg-type]


def test_snapshot_delete_cascades_ledger(tmp_path):
    from sqlalchemy import create_engine

    from app.database import Base

    fk = MemoOutcomeEligibility.__table__.c.memo_snapshot_id.foreign_keys
    assert {f.ondelete for f in fk} == {"CASCADE"}
    engine = create_engine(f"sqlite:///{tmp_path / 'fk.db'}")
    event.listen(engine, "connect", lambda conn, _r: conn.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        snap = add_snapshot(db, ticker="GONE", generated_at=datetime(2026, 6, 1))
        db.commit()
        oe.classify_pending(db=db)
        # FIX-003 Option B deletes snapshots; the derived row goes with them.
        db.execute(text("DELETE FROM memo_snapshots WHERE id = :i"), {"i": snap.id})
        db.commit()
        assert db.execute(select(MemoOutcomeEligibility)).first() is None
    engine.dispose()


def test_identity_mismatch_reclassifies(iso):
    """sqlite reuses a deleted max rowid; a stale ledger row must not vouch
    for the stranger that now holds its id."""
    sessions, _ = iso
    with sessions() as db:
        add_snapshot(db, ticker="KEEP", generated_at=datetime(2026, 6, 1))
        old = add_snapshot(db, ticker="REUSE", generated_at=datetime(2026, 6, 2), mode="live")
        old_id = old.id
        db.commit()
        oe.classify_pending(db=db)
        assert oe.lookup(db, old_id) == Classification(True, oe.REASON_LIVE)
        db.execute(text("DELETE FROM memo_snapshots WHERE id = :i"), {"i": old_id})  # FKs off: row survives
        db.commit()
        db.expunge_all()
        new = add_snapshot(db, ticker="REUSE", generated_at=datetime(2026, 9, 1), mode="demo")
        db.commit()
        assert new.id == old_id, "precondition: sqlite reused the max rowid"
        # The stale row says eligible; every reader must say otherwise.
        assert oe.lookup(db, new.id) is None
        assert db.execute(oe.eligible_only(select(MemoSnapshot.id), MemoSnapshot.id)
                          .where(MemoSnapshot.id == new.id)).first() is None
        pending = db.execute(
            select(MemoSnapshot.id).outerjoin(
                MemoOutcomeEligibility, MemoOutcomeEligibility.memo_snapshot_id == MemoSnapshot.id,
            ).where(oe.pending_condition())
        ).scalars().all()
        assert pending == [new.id]
        assert oe.classify_pending(db=db)["classified"] == 1
        assert oe.lookup(db, new.id) == Classification(False, oe.REASON_NO_LLM_OR_DEMO)


def test_backfill_plan_skips_known_ineligible(iso):
    sessions, _ = iso
    with sessions() as db:
        dev = _dev_copy(db, NVDA_DEV_ID)
        unclassified = add_snapshot(db, ticker="UNCL", generated_at=datetime(2026, 6, 1))
        db.commit()
        mark(db, dev.id, eligible=False, reason=oe.REASON_DEV_COPY)
    plan = market_data_backfill.backfill_plan(today=date(2026, 9, 24))
    pending_ids = {p["memo_snapshot_id"] for p in plan["pending_outcome_pairs"]}
    assert dev.id not in pending_ids
    assert unclassified.id in pending_ids, "unclassified stays in scope (conservative)"
    with sessions() as db:
        assert db.execute(select(MemoOutcomeEligibility).where(
            MemoOutcomeEligibility.memo_snapshot_id == unclassified.id)).first() is None, "the plan never classifies"


def test_eligible_only_is_the_shared_predicate(iso):
    sessions, _ = iso
    with sessions() as db:
        good = add_snapshot(db, ticker="GOODX", generated_at=datetime(2026, 6, 1))
        bad = _dev_copy(db, NVDA_DEV_ID)
        unknown = add_snapshot(db, ticker="UNKX", generated_at=datetime(2026, 6, 1))
        for snap in (good, bad, unknown):
            add_outcome(db, snap, horizon=30, forward_return=0.1, alpha=0.01)
            db.add(MemoPostmortem(memo_snapshot_id=snap.id, ticker=snap.ticker, horizon_days=30, verdict="right"))
        db.commit()
        mark(db, good.id)
        mark(db, bad.id, eligible=False, reason=oe.REASON_DEV_COPY)
        outcomes = db.execute(oe.eligible_only(select(MemoOutcome.memo_snapshot_id),
                                               MemoOutcome.memo_snapshot_id)).scalars().all()
        postmortems = db.execute(oe.eligible_only(select(MemoPostmortem.memo_snapshot_id),
                                                  MemoPostmortem.memo_snapshot_id)).scalars().all()
        # Joining MemoSnapshot in the outer query must not collide.
        joined = db.execute(oe.eligible_only(
            select(MemoOutcome.memo_snapshot_id).join(MemoSnapshot, MemoSnapshot.id == MemoOutcome.memo_snapshot_id),
            MemoOutcome.memo_snapshot_id,
        )).scalars().all()
    assert outcomes == postmortems == joined == [good.id]


def test_universe_table_is_available_for_coverage(iso):
    """Guard for the coverage block's denominator on an isolated engine."""
    sessions, _ = iso
    with sessions() as db:
        db.add(Company(ticker="AAA", company_name="A", sector="Tech", industry="Soft", is_etf=False))
        db.add(Company(ticker="SPY", company_name="SPY", sector="ETF", industry="ETF", is_etf=True))
        db.commit()
        assert outcome_service._universe_companies(db) == 1


def test_postmortem_backfill_dry_run_classifies_first(tmp_path, monkeypatch, capsys):
    """The dry run reports what the real run would spend, so it sweeps first
    (otherwise every snapshot is unclassified and it reports nothing due)."""
    import scripts.postmortem_backfill as backfill
    from app import database
    from app.services import postmortem_service

    sessions, engine = isolated_sessions(tmp_path, monkeypatch, database, postmortem_service)
    try:
        with sessions() as db:
            dev = _dev_copy(db, NVDA_DEV_ID)
            live = add_snapshot(db, ticker="DRYLIVE", generated_at=datetime(2026, 6, 1))
            for snap in (dev, live):
                add_outcome(db, snap, horizon=90, forward_return=0.1, alpha=0.05)
            db.commit()
        assert backfill._dry_run([90], 25) == 0
        out = capsys.readouterr().out
        assert "unclassified snapshots = 0" in out
        assert "due (after dedupe) = 1" in out and "ineligible=1" in out
        assert f"DRYLIVE v1 (snapshot #{live.id})" in out
        with sessions() as db:
            assert db.query(MemoPostmortem).count() == 0, "a dry run writes no postmortem"
            assert len(_ledger(sessions)) == 2
    finally:
        engine.dispose()
