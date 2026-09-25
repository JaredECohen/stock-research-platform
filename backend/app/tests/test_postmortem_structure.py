"""Structural tests for `services/postmortem_service.py`.

Covers the pure verdict classifier, the dedupe guard (`_should_postmortem`:
rating-unchanged skip + 14-day per-(ticker, horizon) window), and the
`run_postmortems` driver end-to-end with a seeded outcome and no LLM —
the deterministic lesson must land in `memo_postmortems` and, on the
90-day cadence, in the company memory file. Memory writes are pointed
at `tmp_path`. The LLM is *pinned* to its no-answer path via
`_llm_postmortem` rather than assumed absent: with a developer `.env`
in place the driver would otherwise issue a billable `route="strong"`
completion per due memo. A separate test configures a throwaway key and
routes `llm.chat_json` through a canned reply to prove that seam is the
only one the driver uses; an autouse socket guard backs both up.

Seeding mirrors `test_outcome_tracking._seed_snapshot`, but keeps
earlier versions when asked so the rating-change rule can be probed.
"""
from __future__ import annotations

import socket
from datetime import datetime, timedelta
from typing import Any

import pytest

from app.agents import llm
from app.config import settings
from app.database import SessionLocal
from app.models import MemoOutcome, MemoPostmortem, MemoSnapshot
from app.services import postmortem_service as pm
from app.tests.eligibility_helpers import mark


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline structural test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


def _seed_snapshot(
    ticker: str, *, version: int = 1, rating: str | None = "Bullish",
    days_ago: int = 120, regime: str = "", clear: bool = True,
) -> MemoSnapshot:
    with SessionLocal() as db:
        if clear:
            db.query(MemoPostmortem).filter(MemoPostmortem.ticker == ticker).delete()
            db.query(MemoOutcome).filter(MemoOutcome.ticker == ticker).delete()
            db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        memo = {"ticker": ticker, "confidence_score": 70.0, "sector": "Technology",
                "macro_regime_at_memo": regime, "generation_mode": "live"}
        if rating is not None:
            memo["rating_label"] = rating
        snap = MemoSnapshot(
            ticker=ticker, version=version, trigger="first_run", memo_json=memo,
            revision_log=[], generated_at=datetime.utcnow() - timedelta(days=days_ago),
        )
        db.add(snap)
        db.commit()
        # W6: selection and the prior-version dedupe read only eligible
        # snapshots; most tests here call them without a sweep.
        mark(db, snap.id)
        db.refresh(snap)
        db.expunge(snap)
        return snap


def _seed_outcome(snap: MemoSnapshot, *, horizon: int, fwd: float, bench: float) -> MemoOutcome:
    with SessionLocal() as db:
        row = MemoOutcome(
            memo_snapshot_id=snap.id, ticker=snap.ticker,
            rating_at_memo=snap.memo_json.get("rating_label", ""),
            confidence_at_memo=70.0, price_at_memo=100.0, horizon_days=horizon,
            forward_return=fwd, benchmark_return=bench, alpha=fwd - bench,
            thesis_held=(fwd - bench) > 0,
            regime_at_memo=snap.memo_json.get("macro_regime_at_memo") or None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


def _seed_postmortem(ticker: str, snap_id: int, horizon: int, *, days_ago: int = 0) -> None:
    with SessionLocal() as db:
        db.add(MemoPostmortem(
            memo_snapshot_id=snap_id, ticker=ticker, horizon_days=horizon,
            verdict="mixed", lesson="seed", agent_attribution={},
            created_at=datetime.utcnow() - timedelta(days=days_ago),
        ))
        db.commit()


def _postmortems(ticker: str, horizon: int):
    with SessionLocal() as db:
        return db.query(MemoPostmortem).filter_by(ticker=ticker, horizon_days=horizon).all()


# ---------------------------------------------------------------------------
# _classify_verdict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rating, alpha, expected", [
    ("Bullish", None, "pending"),
    ("Bullish", 0.021, "right"),
    ("Bullish", 0.02, "mixed"),          # strict > 0.02
    ("Bullish", -0.05, "mixed"),         # strict < -0.05
    ("Bullish", -0.051, "wrong"),
    ("Very Bullish", 0.5, "right"),
    ("Bearish", -0.021, "right"),
    ("Bearish", -0.02, "mixed"),
    ("Bearish", 0.05, "mixed"),
    ("Bearish", 0.051, "wrong"),
    ("Very Bearish", 0.2, "wrong"),
    ("Neutral", 0.049, "right"),
    ("Neutral", -0.049, "right"),
    ("Neutral", 0.05, "mixed"),
    ("Neutral", -0.5, "mixed"),
    ("", 0.0, "right"),                  # unknown label → neutral rules
    ("Mixed Positive", 0.3, "mixed"),
])
def test_classify_verdict_boundaries(rating, alpha, expected):
    assert pm._classify_verdict(rating, alpha) == expected


# ---------------------------------------------------------------------------
# _should_postmortem
# ---------------------------------------------------------------------------

def test_should_postmortem_dedupe_rules():
    t = "TSTPMDEDUP"
    v1 = _seed_snapshot(t, version=1, rating="Bullish")
    with SessionLocal() as db:
        assert pm._should_postmortem(db, v1, 90) == (True, "ok")   # first memo always proceeds

    v2 = _seed_snapshot(t, version=2, rating="Bullish", clear=False)
    with SessionLocal() as db:
        ok, reason = pm._should_postmortem(db, v2, 90)
    assert ok is False and reason.startswith("rating unchanged (Bullish)") and "v1" in reason

    v3 = _seed_snapshot(t, version=3, rating="Neutral", clear=False)
    with SessionLocal() as db:
        assert pm._should_postmortem(db, v3, 90) == (True, "ok")

    # A fresh postmortem for (ticker, 90) blocks 90 but not 30.
    _seed_postmortem(t, v3.id, 90, days_ago=1)
    with SessionLocal() as db:
        ok, reason = pm._should_postmortem(db, v3, 90)
        assert ok is False and reason.startswith("recent postmortem exists")
        assert f"within {pm._DEDUPE_WINDOW_DAYS}d" in reason
        assert pm._should_postmortem(db, v3, 30) == (True, "ok")

    # Outside the window the guard opens again.
    with SessionLocal() as db:
        db.query(MemoPostmortem).filter_by(ticker=t).update(
            {"created_at": datetime.utcnow() - timedelta(days=pm._DEDUPE_WINDOW_DAYS + 1)}
        )
        db.commit()
    with SessionLocal() as db:
        assert pm._should_postmortem(db, v3, 90) == (True, "ok")


def test_rating_unchanged_skip_needs_both_labels():
    t = "TSTPMNOLBL"
    _seed_snapshot(t, version=1, rating=None)                 # prior has no label
    v2 = _seed_snapshot(t, version=2, rating="Bullish", clear=False)
    with SessionLocal() as db:
        assert pm._should_postmortem(db, v2, 90) == (True, "ok")
    v3 = _seed_snapshot(t, version=3, rating=None, clear=False)  # new has no label
    with SessionLocal() as db:
        assert pm._should_postmortem(db, v3, 90) == (True, "ok")


def test_due_memos_excludes_already_written_and_deduped():
    t = "TSTPMDUE"
    snap = _seed_snapshot(t, rating="Bullish")
    _seed_outcome(snap, horizon=90, fwd=0.2, bench=0.05)
    due = [d for d in pm._due_memos(90, limit=500) if d["snapshot"].ticker == t]
    assert len(due) == 1 and due[0]["outcome"].memo_snapshot_id == snap.id
    _seed_postmortem(t, snap.id, 90)
    assert not [d for d in pm._due_memos(90, limit=500) if d["snapshot"].ticker == t]


# ---------------------------------------------------------------------------
# run_postmortems — deterministic path
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    return tmp_path


@pytest.fixture
def no_llm(monkeypatch) -> list[dict[str, Any]]:
    """Pin `_llm_postmortem` to None so the deterministic branch runs
    regardless of which keys the environment carries; returns the calls
    so a test can prove the driver asked."""
    calls: list[dict[str, Any]] = []

    def _none(memo, outcome, horizon_days):
        calls.append({"ticker": memo.get("ticker"), "horizon": horizon_days})
        return None
    monkeypatch.setattr(pm, "_llm_postmortem", _none)
    return calls


def test_llm_gate_is_closed_under_blank_keys(monkeypatch):
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    assert settings.has_llm is False

    def _never(*_a, **_k):
        raise AssertionError("chat_json must not be called without a key")
    monkeypatch.setattr(llm, "chat_json", _never)
    outcome = MemoOutcome(forward_return=0.1, benchmark_return=0.0, alpha=0.1)
    assert pm._llm_postmortem({"ticker": "X"}, outcome, 90) is None


def test_deterministic_lesson_text():
    outcome = MemoOutcome(forward_return=0.2, benchmark_return=0.06, alpha=0.14)
    lesson = pm._deterministic_lesson({"rating_label": "Bullish"}, outcome, "right", 90)
    assert lesson == (
        "90d postmortem (right). Memo rated Bullish; alpha 14.0% vs benchmark "
        "over the window. Realized return 20.0%, benchmark 6.0%."
    )
    no_alpha = MemoOutcome(forward_return=0.0, benchmark_return=0.0, alpha=None)
    assert "alpha unavailable" in pm._deterministic_lesson({}, no_alpha, "pending", 30)


def test_run_postmortems_writes_row_and_memory_on_90d(memory_dir, no_llm):
    t = "TSTPMRUN"
    snap = _seed_snapshot(t, rating="Bullish", regime="soft_landing")
    outcome = _seed_outcome(snap, horizon=90, fwd=0.20, bench=0.06)

    report = pm.run_postmortems(horizon_days=90, limit=500)
    assert set(report) == {
        "horizon_days", "due", "written", "already_done", "deduped", "skipped",
        "deduped_memos", "deferred", "deferred_memos", "ineligible",
        "skipped_memos", "memory_written", "memory_written_memos",
        "memory_failed", "memory_failed_memos", "memory_disabled", "memory_disabled_memos",
        "memory_not_requested", "memory_not_requested_memos", "classification_error",
        # W7: always present, zero while the learning ledger is off.
        "learning_written", "learning_skipped", "learning_skip_reasons", "learning_rejected",
        "learning_failed", "learning_failed_memos",
    }
    assert (report["learning_written"], report["learning_skipped"], report["learning_failed"]) == (0, 0, 0)
    assert report["horizon_days"] == 90 and report["written"] >= 1
    assert report["classification_error"] is None
    assert {"ticker": t, "horizon": 90} in no_llm       # the LLM was asked, and declined

    rows = _postmortems(t, 90)
    assert len(rows) == 1
    row = rows[0]
    assert row.memo_snapshot_id == snap.id
    assert row.verdict == "right"
    assert row.lesson == pm._deterministic_lesson(snap.memo_json, outcome, "right", 90)
    assert row.agent_attribution == {}                 # no LLM → no attribution
    assert row.regime_at_memo == "soft_landing"        # memo's own tag, not an LLM guess
    assert row.realized_return == 0.20 and row.benchmark_return == 0.06
    assert row.written_to_memory is True

    memory_file = memory_dir / "companies" / f"{t}.md"
    assert memory_file.exists()
    assert "90d postmortem (right)" in memory_file.read_text()

    # Idempotent: the (snapshot, horizon) pair is never written twice.
    again = pm.run_postmortems(horizon_days=90, limit=500)
    assert not [d for d in pm._due_memos(90, limit=500) if d["snapshot"].ticker == t]
    assert len(_postmortems(t, 90)) == 1
    assert (
        again["written"] + again["already_done"] + again["skipped"] == again["due"]
    )
    # Nothing was re-attempted and nothing reads as a failure.
    assert again["due"] == again["written"] == again["skipped"] == 0


def test_run_postmortems_30d_early_read_stays_out_of_memory(memory_dir, no_llm):
    t = "TSTPM30"
    snap = _seed_snapshot(t, rating="Bearish", days_ago=45)
    _seed_outcome(snap, horizon=30, fwd=0.10, bench=0.01)    # bearish call, stock up → wrong

    report = pm.run_postmortems(horizon_days=30, limit=500)
    assert report["written"] >= 1
    rows = _postmortems(t, 30)
    assert len(rows) == 1
    assert rows[0].verdict == "wrong"
    assert rows[0].written_to_memory is False
    assert not (memory_dir / "companies" / f"{t}.md").exists()
    assert _postmortems(t, 90) == []                   # other horizon untouched


def test_run_postmortems_with_a_configured_key_uses_only_the_chat_json_seam(memory_dir, monkeypatch):
    """Regression for the review finding that the driver tests relied on
    blank keys: give `settings` a throwaway key so `has_llm` opens the
    gate, route `llm.chat_json` through a canned reply, and check the
    reply is what lands. The socket guard makes any attempt to build a
    real client and call out fail loudly instead of silently billing."""
    t = "TSTPMLLM"
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")
    assert settings.has_llm is True
    seen: list[dict[str, Any]] = []

    def _chat_json(prompt: str, **kwargs: Any):
        seen.append(kwargs)
        return {
            "lesson": "LLM lesson body",
            "agent_attribution": {"sector": 0.5, "valuation": -0.25},
            "regime_at_memo": "llm_guess",
            "sector_lesson": "",
        }
    monkeypatch.setattr(llm, "chat_json", _chat_json)

    snap = _seed_snapshot(t, rating="Bullish", regime="")
    _seed_outcome(snap, horizon=90, fwd=0.20, bench=0.06)
    report = pm.run_postmortems(horizon_days=90, limit=500)
    assert report["written"] >= 1
    assert len(seen) == 1 and seen[0]["route"] == "strong"

    rows = _postmortems(t, 90)
    assert len(rows) == 1
    assert rows[0].lesson == "LLM lesson body"
    assert rows[0].agent_attribution == {"sector": 0.5, "valuation": -0.25}
    assert rows[0].regime_at_memo == "llm_guess"          # memo had no tag → LLM guess used
    assert "LLM lesson body" in (memory_dir / "companies" / f"{t}.md").read_text()


# ---------------------------------------------------------------------------
# W6 / FIX-007 — eligible-only selection (isolated engine: exact counts)
# ---------------------------------------------------------------------------

@pytest.fixture
def w6_pm(tmp_path, monkeypatch):
    from app.tests.eligibility_helpers import isolated_sessions

    sessions, engine = isolated_sessions(tmp_path, monkeypatch, pm)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    calls: list[str] = []

    def record(memo, outcome, horizon_days):
        calls.append(outcome.ticker)
        return None

    monkeypatch.setattr(pm, "_llm_postmortem", record)
    yield sessions, calls
    engine.dispose()


def _nvda_dev_copy(db):
    from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS
    from app.tests.eligibility_helpers import add_snapshot

    sid, ticker, generated = next(row for row in DEV_COPY_SNAPSHOTS if row[1] == "NVDA")
    return add_snapshot(db, id=sid, ticker=ticker, version=sid, mode="demo",
                        generated_at=datetime.fromisoformat(generated), memo_generated_at=generated)


def test_scan_due_skips_ineligible_and_counts_them(w6_pm):
    """A dev-copy outcome is never selected for strong-route LLM spend."""
    from app.tests.eligibility_helpers import add_outcome, add_snapshot

    sessions, calls = w6_pm
    with sessions() as db:
        dev = _nvda_dev_copy(db)
        live = add_snapshot(db, ticker="PMLIVE", generated_at=datetime(2026, 5, 20))
        for snap in (dev, live):
            add_outcome(db, snap, horizon=90, forward_return=0.2, alpha=0.1)
        db.commit()
    report = pm.run_postmortems(horizon_days=90, limit=25)
    assert report["ineligible"] == 1
    assert report["due"] == report["written"] == 1
    assert calls == ["PMLIVE"]
    with sessions() as db:
        assert [r.memo_snapshot_id for r in db.query(MemoPostmortem).all()] == [live.id]


def test_prior_demo_version_does_not_dedupe_live_memo(w6_pm):
    """Before W6 the dev-copy prior (same rating) suppressed the first live
    memo's postmortem as "rating unchanged". NVDA has hundreds of them."""
    from app.tests.eligibility_helpers import add_outcome, add_snapshot

    sessions, calls = w6_pm
    with sessions() as db:
        dev = _nvda_dev_copy(db)
        live = add_snapshot(db, ticker="NVDA", version=dev.version + 1, rating="Bullish",
                            generated_at=datetime(2026, 6, 20))
        add_outcome(db, live, horizon=90, forward_return=0.2, alpha=0.1)
        db.commit()
    report = pm.run_postmortems(horizon_days=90, limit=25)
    assert report["deduped"] == 0, report["deduped_memos"]
    assert report["written"] == 1 and calls == ["NVDA"]


def test_failed_sweep_still_postmortems_classified_memos(w6_pm, monkeypatch):
    """An aborted eligibility sweep is reported and turns the loop red, but
    memos the ledger already classifies still get their postmortems; the one
    the sweep could not classify is skipped (fail closed). Before, the
    exception escaped run_postmortems ahead of the scan."""
    from app.monitoring import postmortem_loop
    from app.services import outcome_eligibility as oe
    from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS
    from app.tests.eligibility_helpers import add_outcome, add_snapshot, classify_all

    sessions, calls = w6_pm
    dev_ids = {row[0] for row in DEV_COPY_SNAPSHOTS}
    with sessions() as db:
        live = add_snapshot(db, ticker="PMLIVE", generated_at=datetime(2026, 5, 20))
        add_outcome(db, live, horizon=90, forward_return=0.2, alpha=0.1)
        db.commit()
        classify_all(db)
        stray = add_snapshot(db, id=next(i for i in range(400, 583) if i not in dev_ids), ticker="PMSTRAY",
                             generated_at=datetime(2026, 5, 4), mode="demo")
        add_outcome(db, stray, horizon=90, forward_return=0.2, alpha=0.1)
        db.commit()
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(postmortem_loop, "record_run", lambda *a, **k: recorded.append(k))
    monkeypatch.setattr(postmortem_loop, "run_postmortems",
                        lambda horizon_days, limit: pm.run_postmortems(horizon_days=horizon_days, limit=limit)
                        if horizon_days == 90 else {"due": 0, "written": 0})
    postmortem_loop.run_once()
    assert calls == ["PMLIVE"]
    assert recorded[0]["success"] is False
    assert f"classification_error=ExclusionSetMismatch: dev-copy classification outside the enumerated set: " \
           f"PMSTRAY#{stray.id}" in recorded[0]["note"]
    with sessions() as db:
        assert [r.memo_snapshot_id for r in db.query(MemoPostmortem).all()] == [live.id]
        assert oe.lookup(db, stray.id) is None


# ---------------------------------------------------------------------------
# W7 — the loop note carries the learning counts, and a failed learning
# write turns the loop red (the postmortem itself was still written)
# ---------------------------------------------------------------------------

def _loop_report(horizon_days: int, *, learning_failed: int = 0) -> dict[str, Any]:
    return {
        "horizon_days": horizon_days, "due": 1, "written": 1, "already_done": 0, "deduped": 0,
        "skipped": 0, "deferred": 0, "ineligible": 0, "classification_error": None,
        "learning_written": 1 - learning_failed, "learning_skipped": 0, "learning_rejected": 0,
        "learning_failed": learning_failed,
        "learning_failed_memos": [{"ticker": "TSTPMLRN", "memo_snapshot_id": 1, "reason": "exception:X"}]
        if learning_failed else [],
    }


@pytest.mark.parametrize("learning_failed, success", [(0, True), (1, False)])
def test_loop_success_requires_no_learning_failures(monkeypatch, learning_failed, success):
    from app.monitoring import postmortem_loop

    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(postmortem_loop, "record_run", lambda name, **k: recorded.append(k))
    monkeypatch.setattr(postmortem_loop, "run_postmortems", lambda horizon_days, limit: _loop_report(
        horizon_days, learning_failed=learning_failed if horizon_days == 90 else 0))
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    postmortem_loop.run_once()
    [row] = recorded
    assert row["success"] is success
    assert (f"90d due=1 written=1 already_done=0 deduped=0 skipped=0 deferred=0 ineligible=0 "
            f"memory_written=0 memory_disabled=0 memory_failed=0 memory_not_requested=0 "
            f"learning_written={1 - learning_failed} learning_skipped=0 learning_rejected=0 "
            f"learning_failed={learning_failed}") in row["note"]
    if learning_failed:
        assert "learning_failed memos: TSTPMLRN#1 (exception:X)" in row["note"]
    assert row["note"].endswith("; learning off")
