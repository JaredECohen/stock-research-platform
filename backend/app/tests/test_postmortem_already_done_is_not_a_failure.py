"""`postmortem_loop` reported failure every night for work that was done.

Production, nightly, for months::

    30d due=25 written=2 skipped=23; 90d due=25 written=2 skipped=23  success=False

and in the worker log, twenty-three times a night::

    WARNING app.services.postmortem_service: postmortem persist failed for
    memo 512: (psycopg2.errors.UniqueViolation) duplicate key value violates
    unique constraint "uq_memo_postmortem_snapshot_horizon"

None of those twenty-three was a failure and none was a coverage gap. They
were memos whose postmortem **already existed**. The due query handed them
back, the insert was rejected by the database, the handler counted the
rejection as a skipped postmortem, and `success = skipped == 0` turned a
healthy backlog into a standing red light.

Two separate defects, and both are fixed here because either one alone
leaves a trap:

**The due set and the constraint disagreed about what a duplicate is.**
`memo_postmortems` is unique on `(memo_snapshot_id, horizon_days)`. The due
list was keyed by *outcome row* — one entry per `memo_outcomes` row that
passed a per-row lookup — and nothing compared the pairs it was about to
write against each other or re-read them at write time. The moment the same
`(snapshot, horizon)` appears twice in one list, or is written by anything
else during a pass that makes a `route="strong"` LLM call per memo, the
identical insert is attempted twice. Now the exclusion is a `NOT EXISTS` on
exactly the constraint's two columns, the list is deduplicated on that same
key, and the existence check is repeated immediately before the LLM call
rather than discovered by the database afterwards.

**The telemetry had no word for "already done".** A memo whose postmortem
exists is neither written nor skipped, and a loop that cries wolf every
night is a loop nobody reads — which is precisely how the dead filings
pipeline stayed dead for the system's entire life. So the report counts
`written` / `already_done` / `deduped` / `skipped` separately, and only
`skipped` — work that should have been written and was not — fails the
loop.
"""
from __future__ import annotations

import socket
from datetime import datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import MemoOutcome, MemoPostmortem, MemoSnapshot
from app.monitoring import postmortem_loop
from app.services import postmortem_service as pm


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Pin the LLM to its no-answer path and count the calls.

    Counting matters as much as pinning here: the cost of the old code was
    not only a false alarm, it was a `route="strong"` completion per
    already-done memo, spent to discover what a SELECT could have said.
    """
    calls: list[int] = []

    def _none(memo, outcome, horizon_days):
        calls.append(horizon_days)
        return None

    monkeypatch.setattr(pm, "_llm_postmortem", _none)
    return calls


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    return tmp_path


def _seed(ticker: str, *, versions: int, horizon: int) -> list[int]:
    """`versions` snapshots for one ticker, each with an outcome at `horizon`.

    Ratings alternate so the rating-unchanged dedupe does not hold them
    back, and the snapshots are dated far enough apart that the 14-day
    per-ticker rate limit is the only policy in play.
    """
    ratings = ["Bullish", "Bearish", "Neutral", "Very Bullish"]
    ids: list[int] = []
    with SessionLocal() as db:
        db.query(MemoPostmortem).filter(MemoPostmortem.ticker == ticker).delete()
        db.query(MemoOutcome).filter(MemoOutcome.ticker == ticker).delete()
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.commit()
        for v in range(1, versions + 1):
            snap = MemoSnapshot(
                ticker=ticker, version=v, trigger="first_run",
                memo_json={
                    "ticker": ticker, "rating_label": ratings[v % len(ratings)],
                    "confidence_score": 70.0, "sector": "Technology",
                },
                revision_log=[],
                generated_at=datetime.utcnow() - timedelta(days=200 - v),
            )
            db.add(snap)
            db.flush()
            db.add(MemoOutcome(
                memo_snapshot_id=snap.id, ticker=ticker, rating_at_memo="Bullish",
                confidence_at_memo=70.0, price_at_memo=100.0, horizon_days=horizon,
                forward_return=0.2, benchmark_return=0.06, alpha=0.14,
                thesis_held=True,
            ))
            ids.append(snap.id)
        db.commit()
    return ids


def _drain(horizon: int = 90) -> None:
    """Clear any memo left due by another module, so counts here are absolute.

    The suite shares one database; without this, "written == 1" would be
    an assertion about test ordering rather than about this fix.
    """
    for _ in range(5):
        report = pm.run_postmortems(horizon_days=horizon, limit=1000)
        if report["due"] == 0:
            return


def _postmortems(ticker: str, horizon: int) -> list[MemoPostmortem]:
    with SessionLocal() as db:
        return db.query(MemoPostmortem).filter(
            MemoPostmortem.ticker == ticker,
            MemoPostmortem.horizon_days == horizon,
        ).all()


# ---------------------------------------------------------------------------
# The second run
# ---------------------------------------------------------------------------

def test_a_second_run_over_already_postmortemd_memos_reports_healthy(
    memory_dir, _no_llm,
):
    """The headline requirement: run twice, stay green, attempt nothing."""
    _drain(90)
    ticker = "ZZPMH1"
    _seed(ticker, versions=1, horizon=90)

    first = pm.run_postmortems(horizon_days=90, limit=25)
    assert first["written"] == 1 and first["skipped"] == 0
    assert len(_postmortems(ticker, 90)) == 1
    calls_after_first = len(_no_llm)

    second = pm.run_postmortems(horizon_days=90, limit=25)

    assert second["skipped"] == 0, (
        "an already-written postmortem was reported as work the loop failed "
        "to do; that is the false telemetry, not a cosmetic detail"
    )
    assert second["written"] == 0
    assert len(_postmortems(ticker, 90)) == 1
    assert len(_no_llm) == calls_after_first, (
        "the second run spent an LLM call re-deriving a postmortem that "
        "already existed"
    )


def test_the_loop_reports_success_on_a_backlog_that_is_already_done(
    memory_dir, _no_llm, monkeypatch,
):
    """`postmortem_loop`, end to end, is the surface that was lying."""
    _drain(30)
    _drain(90)
    for horizon in (30, 90):
        _seed(f"ZZPML{horizon}", versions=1, horizon=horizon)

    calls: list[tuple] = []
    monkeypatch.setattr(
        postmortem_loop, "record_run",
        lambda *a, **k: calls.append((a, k)),
    )

    postmortem_loop.run_once(limit_per_horizon=25)
    postmortem_loop.run_once(limit_per_horizon=25)

    second = calls[-1][1]
    assert second["success"] is True, (
        f"the loop still reports failure for finished work: {second['note']}"
    )
    assert "already_done=" in second["note"], (
        "the note has no word for 'already done', which is how 23 finished "
        f"memos read as 23 failures: {second['note']}"
    )
    assert "skipped=0" in second["note"], second["note"]


def test_already_done_is_reported_apart_from_written_and_skipped(
    memory_dir, _no_llm, monkeypatch,
):
    """A memo the due query hands back after its postmortem landed.

    This is the production shape reproduced directly: the due list is a
    snapshot taken before the first of `limit` strong-route LLM calls, and
    the row appears underneath it. It must come back as `already_done` —
    not `written`, not `skipped` — and must cost no model round-trip.
    """
    _drain(90)
    ticker = "ZZPMR1"
    (snap_id,) = _seed(ticker, versions=1, horizon=90)
    due, _ = pm._scan_due(90, limit=25)
    assert [d["outcome"].memo_snapshot_id for d in due] == [snap_id]

    # Another writer gets there first, between the query and the write.
    with SessionLocal() as db:
        db.add(MemoPostmortem(
            memo_snapshot_id=snap_id, ticker=ticker, horizon_days=90,
            verdict="right", lesson="written by someone else",
            agent_attribution={}, created_at=datetime.utcnow(),
        ))
        db.commit()

    monkeypatch.setattr(pm, "_scan_due", lambda h, *, limit: (due, 0))
    before = len(_no_llm)
    report = pm.run_postmortems(horizon_days=90, limit=25)

    assert report["already_done"] == 1
    assert report["written"] == 0
    assert report["skipped"] == 0
    assert len(_no_llm) == before
    rows = _postmortems(ticker, 90)
    assert len(rows) == 1 and rows[0].lesson == "written by someone else"


def test_one_pass_never_attempts_the_same_key_twice(memory_dir, _no_llm, monkeypatch):
    """A duplicated entry in the due list must not become a failed insert.

    The due list used to be keyed by outcome row while the constraint keys
    on `(snapshot, horizon)`. Any divergence — a duplicated `memo_outcomes`
    row is the obvious one — put the same pair in the list twice, and the
    second insert was rejected and counted as a failure. The pass must
    recognise its own work instead.
    """
    _drain(90)
    ticker = "ZZPMD1"
    (snap_id,) = _seed(ticker, versions=1, horizon=90)
    due, _ = pm._scan_due(90, limit=25)
    assert len(due) == 1
    doubled = due + due

    monkeypatch.setattr(pm, "_scan_due", lambda h, *, limit: (doubled, 0))
    report = pm.run_postmortems(horizon_days=90, limit=25)

    assert report["due"] == 2
    assert report["written"] == 1
    assert report["already_done"] == 1
    assert report["skipped"] == 0, (
        "the pass counted its own second attempt at the same key as a failure"
    )
    assert len(_postmortems(ticker, 90)) == 1
    assert len(_no_llm) == 1, "the duplicate cost a second strong-route call"


def test_the_due_scan_is_deduplicated_on_the_constraints_own_key(memory_dir):
    """Whatever the query returns, the list is one entry per (snapshot, horizon)."""
    ticker = "ZZPMK1"
    ids = _seed(ticker, versions=3, horizon=90)
    due, _ = pm._scan_due(90, limit=25)

    keys = [(d["outcome"].memo_snapshot_id, 90) for d in due]
    assert len(keys) == len(set(keys))
    assert set(k[0] for k in keys) <= set(ids)


def test_the_scan_excludes_what_the_constraint_would_reject(memory_dir):
    ticker = "ZZPMX1"
    ids = _seed(ticker, versions=3, horizon=90)
    with SessionLocal() as db:
        db.add(MemoPostmortem(
            memo_snapshot_id=ids[0], ticker=ticker, horizon_days=90,
            verdict="right", lesson="already", agent_attribution={},
            created_at=datetime.utcnow(),
        ))
        db.commit()

    due, _ = pm._scan_due(90, limit=25)
    assert ids[0] not in [d["outcome"].memo_snapshot_id for d in due]

    # Another horizon is a different key and stays due.
    _seed(ticker + "B", versions=1, horizon=30)
    assert [d["outcome"].memo_snapshot_id for d in pm._scan_due(30, limit=25)[0]]


def test_persist_classifies_the_three_outcomes_it_can_have(memory_dir):
    """`_persist_postmortem` is where "already done" is distinguished, and
    the guard against fixing the alarm by silencing it.

    An integrity error that leaves the row present is an answer. One that
    leaves no row behind is still a postmortem the loop owed and did not
    write, and must keep failing the loop.
    """
    _drain(90)
    ticker = "ZZPMP1"
    (snap_id,) = _seed(ticker, versions=1, horizon=90)

    def _row(**over):
        kwargs = dict(
            memo_snapshot_id=snap_id, ticker=ticker, horizon_days=90,
            verdict="right", lesson="body", agent_attribution={},
            created_at=datetime.utcnow(),
        )
        kwargs.update(over)
        return MemoPostmortem(**kwargs)

    assert pm._persist_postmortem(_row()) == "written"
    assert pm._persist_postmortem(_row()) == "already_done"
    # A different integrity failure — the row is not there afterwards, so
    # this is a real miss and has to say so.
    assert pm._persist_postmortem(
        _row(memo_snapshot_id=snap_id + 10_000, ticker=None),
    ) == "failed"
    assert len(_postmortems(ticker, 90)) == 1


def test_a_real_failure_still_fails_the_loop(memory_dir, _no_llm, monkeypatch):
    _drain(90)
    ticker = "ZZPMF1"
    _seed(ticker, versions=1, horizon=90)
    monkeypatch.setattr(pm, "_persist_postmortem", lambda row: "failed")

    report = pm.run_postmortems(horizon_days=90, limit=25)
    assert report["skipped"] == report["due"] == 1 and report["written"] == 0

    calls: list[tuple] = []
    monkeypatch.setattr(
        postmortem_loop, "record_run", lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(
        postmortem_loop, "run_postmortems",
        lambda **kw: {"due": 1, "written": 0, "already_done": 0,
                      "deduped": 0, "skipped": 1},
    )
    postmortem_loop.run_once()
    assert calls[-1][1]["success"] is False
    assert "skipped=1" in calls[-1][1]["note"]


def test_the_policy_dedupe_is_counted_rather_than_only_logged(memory_dir, _no_llm):
    """Rating-unchanged and rate-limited memos are answers too.

    They used to reach a DEBUG log and nothing else, so an operator reading
    cron-health could not tell a horizon with nothing due from one holding
    two dozen memos back on purpose.
    """
    _drain(90)
    ticker = "ZZPMDD"
    with SessionLocal() as db:
        db.query(MemoPostmortem).filter(MemoPostmortem.ticker == ticker).delete()
        db.query(MemoOutcome).filter(MemoOutcome.ticker == ticker).delete()
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.commit()
        for v in (1, 2):
            snap = MemoSnapshot(
                ticker=ticker, version=v, trigger="first_run",
                memo_json={"ticker": ticker, "rating_label": "Bullish",
                           "confidence_score": 70.0},
                revision_log=[],
                generated_at=datetime.utcnow() - timedelta(days=200 - v),
            )
            db.add(snap)
            db.flush()
            db.add(MemoOutcome(
                memo_snapshot_id=snap.id, ticker=ticker, rating_at_memo="Bullish",
                confidence_at_memo=70.0, price_at_memo=100.0, horizon_days=90,
                forward_return=0.2, benchmark_return=0.06, alpha=0.14,
            ))
        db.commit()

    report = pm.run_postmortems(horizon_days=90, limit=25)

    assert report["deduped"] >= 1, (
        "v2 repeats v1's rating, so the rating-unchanged rule held it back — "
        "and the report has to say so"
    )
    assert report["skipped"] == 0
