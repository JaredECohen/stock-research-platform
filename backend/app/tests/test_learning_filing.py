"""W7: filing observations persist the post-pass's existing LLM diff.

`_llm_diff` is a strong-route call that is paid for today and — with the
file backend off in production — dropped. With the ledger on, its bullets
become one dated company observation per filing (the next filing of the same
type supersedes it), and its sector pattern becomes a group or sector
observation (at most 10 active per scope). The first filing of a type and
the deterministic fallback diff write nothing, demo accessions are refused,
and a ledger failure is reported in the post-pass errors like any other
memory write. No new LLM call is made.
"""
from __future__ import annotations

import itertools
from datetime import date
from typing import Any

import pytest

from app.config import settings
from app.learning import ledger
from app.models import FilingDoc, LearningItem
from app.services import filing_memory
from app.tests.learning_helpers import add_company, classify, learning_db

_IDS = itertools.count(1)


@pytest.fixture
def db(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "learning_ledger_writes", True)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    monkeypatch.setattr(settings, "use_demo_data", False)
    monkeypatch.setattr(filing_memory, "index_filing", lambda filing: 0)   # no vector store here
    yield sessions
    engine.dispose()


@pytest.fixture
def diff(monkeypatch):
    """`_llm_diff` returns `box.reply` (None means the LLM declined)."""
    class Box:
        reply: Any = {"bullets": ["Revenue guidance raised to 12% growth", "New buyback of $2B"],
                      "sector_relevant": False, "sector_pattern": ""}
        calls = 0

    box = Box()

    def _diff(prior, filing):
        box.calls += 1
        return box.reply
    monkeypatch.setattr(filing_memory, "_llm_diff", _diff)
    return box


def _filing(sessions, ticker: str, filed: date, *, ftype: str = "10-Q", accession: str | None = None) -> FilingDoc:
    with sessions() as s:
        row = FilingDoc(ticker=ticker, accession_number=accession or f"0000-{ticker}-{next(_IDS):06d}",
                        filing_type=ftype, filing_date=filed, raw_text="body", sections={}, word_count=1)
        s.add(row)
        s.commit()
        s.refresh(row)
        s.expunge(row)
        return row


def _items(sessions, **filters: Any) -> list[LearningItem]:
    with sessions() as s:
        rows = s.query(LearningItem).filter_by(**filters).order_by(LearningItem.id).all()
        s.expunge_all()
        return rows


def test_llm_delta_supersedes_prior_same_type(db, diff):
    with db() as s:
        add_company(s, "FLGA")
    _filing(db, "FLGA", date(2026, 2, 1))
    q2 = _filing(db, "FLGA", date(2026, 5, 1))
    report = filing_memory.post_pass(q2, {"sector": "Technology"})
    assert report["memory_writes"]["ledger"] == {"status": "written", "error_type": None}
    [first] = _items(db)
    assert (first.kind, first.scope_type, first.scope_key, first.status) == ("observation", "company", "FLGA", "active")
    assert first.text == ("What's new in 10-Q filed 2026-05-01: Revenue guidance raised to 12% growth; "
                          "New buyback of $2B")
    assert first.supersede_key == "filing_delta:FLGA:10-Q" and first.origin_snapshot_id is None
    assert first.expires_at.date() == date(2027, 6, 5)                     # filing date + 400 days

    q3 = _filing(db, "FLGA", date(2026, 8, 1))
    filing_memory.post_pass(q3, {"sector": "Technology"})
    old, new = _items(db)
    assert (old.status, new.status) == ("superseded", "active")
    assert old.status_history[-1]["reason"] == f"superseded_by:{q3.accession_number}"
    # A 10-K does not supersede a 10-Q.
    _filing(db, "FLGA", date(2026, 3, 1), ftype="10-K")
    k2 = _filing(db, "FLGA", date(2026, 9, 1), ftype="10-K")
    filing_memory.post_pass(k2, {"sector": "Technology"})
    assert [i.status for i in _items(db)] == ["superseded", "active", "active"]
    assert diff.calls == 3                          # one existing diff call per filing, no new call


def test_newest_first_batch_keeps_the_newest_observation_active(db, diff):
    """EDGAR's `filings.recent` is newest-first and a batch is post-passed in
    that order. Supersession follows the FILING date: the Aug 10-Q stays
    live, the May one is stored already superseded (with its history)."""
    with db() as s:
        add_company(s, "FLGN")
    aug = _filing(db, "FLGN", date(2026, 8, 1))
    may = _filing(db, "FLGN", date(2026, 5, 1))
    _filing(db, "FLGN", date(2026, 2, 1))
    filing_memory.post_pass(aug, {"sector": "Technology"})
    report = filing_memory.post_pass(may, {"sector": "Technology"})
    assert report["memory_writes"]["ledger"] == {"status": "written", "error_type": None}
    rows = {i.source_date: i for i in _items(db, origin_kind="filing_delta")}
    assert [d for d, i in rows.items() if i.status == "active"] == [date(2026, 8, 1)]
    older = rows[date(2026, 5, 1)]
    assert older.status == "superseded"
    assert [h["to"] for h in older.status_history] == ["active", "superseded"]
    assert older.status_history[-1]["reason"] == f"superseded_by:{aug.accession_number}"
    assert rows[date(2026, 8, 1)].status_history[-1]["to"] == "active"


def test_scope_cap_keeps_the_newest_filings_not_the_last_processed(db, diff):
    diff.reply = {"bullets": ["Backlog grew"], "sector_relevant": True,
                  "sector_pattern": "Cloud buyers are pushing renewals into the next fiscal year."}
    for n in reversed(range(12)):          # newest filing processed first
        ticker = f"FR{n:02d}"
        with db() as s:
            add_company(s, ticker)
            classify(s, ticker, state="mapped", group="4510")
        _filing(db, ticker, date(2026, 1, 1))
        filing_memory.post_pass(_filing(db, ticker, date(2026, 2, 1 + n)))
    patterns = _items(db, origin_kind="filing_pattern")
    active = sorted(i.source_date.day for i in patterns if i.status == "active")
    assert active == list(range(3, 13))    # Feb 3..12 kept; Feb 1 and 2 capped


def test_first_of_type_and_deterministic_diff_write_nothing(db, diff):
    first = _filing(db, "FLGB", date(2026, 2, 1))
    report = filing_memory.post_pass(first)
    assert "ledger" not in report["memory_writes"] and diff.calls == 0
    diff.reply = None                                # the LLM declined: deterministic fallback
    second = _filing(db, "FLGB", date(2026, 5, 1))
    report = filing_memory.post_pass(second)
    assert diff.calls == 1
    assert report["memory_writes"]["ledger"] == {"status": "not_requested", "error_type": None}
    assert _items(db) == []


def test_demo_accession_refused(db, diff, monkeypatch):
    _filing(db, "FLGC", date(2026, 2, 1))
    demo = _filing(db, "FLGC", date(2026, 5, 1), accession="DEMO-FLGC-0001")
    report = filing_memory.post_pass(demo)
    assert report["memory_writes"]["ledger"]["status"] == "refused"
    live = _filing(db, "FLGC", date(2026, 8, 1))
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    assert filing_memory.post_pass(live)["memory_writes"]["ledger"]["status"] == "refused"
    assert _items(db) == []


def test_sector_pattern_goes_to_group_scope_with_cap(db, diff):
    diff.reply = {"bullets": ["Backlog grew"], "sector_relevant": True,
                  "sector_pattern": "Cloud buyers are pushing renewals into the next fiscal year."}
    for n in range(12):
        ticker = f"FG{n:02d}"
        with db() as s:
            add_company(s, ticker)
            classify(s, ticker, state="mapped", group="4510")
        _filing(db, ticker, date(2026, 1, 1))
        filing_memory.post_pass(_filing(db, ticker, date(2026, 2, 1 + n)))
    patterns = _items(db, origin_kind="filing_pattern")
    assert {(i.scope_type, i.scope_key) for i in patterns} == {("industry_group", "4510")}
    active = [i for i in patterns if i.status == "active"]
    assert len(patterns) == 12 and len(active) == ledger.MAX_OBS_PER_SCOPE
    assert {i.status_history[-1]["reason"] for i in patterns if i.status == "superseded"} == {"scope_cap"}
    assert min(i.id for i in active) == patterns[2].id      # the two oldest went


def test_no_profile_post_pass_resolves_sector(db, diff):
    """The ingest post-pass passes no profile (`history_service`); the
    sector comes from the universe row, not from nowhere."""
    diff.reply = {"bullets": ["Pipeline readout moved to Q3"], "sector_relevant": True,
                  "sector_pattern": "Trial readouts are slipping a quarter across mid-cap biotech."}
    with db() as s:
        add_company(s, "FLGD", sector="Health Care")
    _filing(db, "FLGD", date(2026, 2, 1))
    filing_memory.post_pass(_filing(db, "FLGD", date(2026, 5, 1)))
    [pattern] = _items(db, origin_kind="filing_pattern")
    assert (pattern.scope_type, pattern.scope_key) == ("sector", "health_care")


def test_ledger_failure_reported_in_post_pass_errors(db, diff, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("scope lookup down")
    monkeypatch.setattr(ledger, "scopes_for", boom)
    _filing(db, "FLGE", date(2026, 2, 1))
    report = filing_memory.post_pass(_filing(db, "FLGE", date(2026, 5, 1)))
    assert report["memory_writes"]["ledger"] == {"status": "failed", "error_type": "RuntimeError"}
    assert {"stage": "ledger_memory", "error_type": "RuntimeError"} in report["errors"]


def test_writes_off_leaves_the_report_unchanged(db, diff, monkeypatch):
    monkeypatch.setattr(settings, "learning_ledger_writes", False)
    _filing(db, "FLGF", date(2026, 2, 1))
    report = filing_memory.post_pass(_filing(db, "FLGF", date(2026, 5, 1)))
    assert "ledger" not in report["memory_writes"] and not report["errors"]
    assert _items(db) == []


def test_code_or_brand_in_an_observation_is_rejected(db, diff):
    diff.reply = {"bullets": ["GICS reclassified the company"], "sector_relevant": False, "sector_pattern": ""}
    _filing(db, "FLGG", date(2026, 2, 1))
    report = filing_memory.post_pass(_filing(db, "FLGG", date(2026, 5, 1)))
    assert report["memory_writes"]["ledger"]["status"] == "rejected"
    assert _items(db) == []
