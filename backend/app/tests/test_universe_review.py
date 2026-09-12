"""Curated-universe review: staleness, DB drift, and the read-only feed diff.

The universe file is a hand-reviewed snapshot that nothing refreshes on
a schedule. These tests pin the two properties that make that safe:
the review can always tell you the file is old or has drifted, and the
review itself never touches the file, the DB, or the network unless
explicitly asked — and even then only reads.

Anything that asserts a date or a count builds its own `UniverseFile`
(`small_file`) instead of reading the shipped sp500.json: the operator
workflow this slice documents rewrites that file's `_last_reviewed` and
ticker list, and a refresh must not turn CI red. The shipped file is
checked for the *shape* of its metadata only.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal, session_scope
from app.main import app
from app.models import Company
from app.providers.fmp_provider import FMPProvider
from app.scripts import refresh_universe_lists
from app.scripts import universe_review as cli
from app.seed_universe import UniverseFile, _load_universe, load_universe_file
from app.services import universe_review

DATA_DIR = Path(universe_review.__file__).resolve().parent.parent / "data"
SP500_PATH = DATA_DIR / "sp500.json"

TOKEN = "test-admin-token-universe"


def _company(ticker: str, tier: str) -> Company:
    return Company(
        ticker=ticker, company_name=ticker, exchange="NYSE",
        sector="Test", industry="Test", universe_tier=tier,
    )


@pytest.fixture()
def drift_rows():
    """Three deterministic rows: a file ticker tagged auto_analysis, an
    auto_analysis row the file does not list, and an on-demand row."""
    with session_scope() as db:
        aapl = db.get(Company, "AAPL")
        prior_tier = aapl.universe_tier if aapl else None
        if aapl is None:
            db.add(_company("AAPL", "auto_analysis"))
        else:
            aapl.universe_tier = "auto_analysis"
        for t in ("ZZRVW1", "ZZRVW2"):
            db.query(Company).filter(Company.ticker == t).delete()
        db.add(_company("ZZRVW1", "auto_analysis"))
        db.add(_company("ZZRVW2", "analyzed_on_demand"))
    yield
    with session_scope() as db:
        db.query(Company).filter(Company.ticker.in_(("ZZRVW1", "ZZRVW2"))).delete(
            synchronize_session=False,
        )
        aapl = db.get(Company, "AAPL")
        if prior_tier is None and aapl is not None:
            db.delete(aapl)
        elif aapl is not None:
            aapl.universe_tier = prior_tier


@pytest.fixture()
def small_file(monkeypatch):
    """Point the review at a 3-ticker universe so diffs are exact."""
    uf = UniverseFile(
        path=SP500_PATH, tickers=["AAPL", "MSFT", "ZZRVW3"], auto_update=["AAPL"],
        as_of="2026-09-01", last_reviewed="2026-09-01",
        review_source="test", review_cadence_days=30,
    )
    monkeypatch.setattr(universe_review, "load_universe_file", lambda: uf)
    return uf


@pytest.fixture()
def feed_never_called(monkeypatch):
    def _boom(self):  # pragma: no cover — reaching this is the failure
        raise AssertionError("FMP constituent feed must not be queried")
    monkeypatch.setattr(FMPProvider, "get_sp500_constituents", _boom)


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

def _assert_review_contract(uf: UniverseFile) -> None:
    """The metadata contract both the shipped file and a refresh must meet.

    Deliberately no literal dates or counts: `refresh_universe_lists`
    rewrites `_last_reviewed` and the ticker list, and the file being
    refreshed is the documented workflow, not a regression.
    """
    assert uf.last_reviewed, "sp500.json must carry _last_reviewed"
    assert date.fromisoformat(uf.last_reviewed) <= date.today()
    assert isinstance(uf.review_cadence_days, int) and uf.review_cadence_days > 0
    assert uf.review_source and uf.review_source.strip()
    assert uf.tickers, "universe must not be empty"
    assert len(uf.tickers) == len(set(uf.tickers)), "duplicate tickers"
    assert all(t == t.upper() for t in uf.tickers)
    # A pinned auto-update name outside the universe would never be seeded.
    assert set(uf.auto_update) <= set(uf.tickers)


def test_shipped_file_carries_review_metadata():
    uf = load_universe_file()
    assert uf.path == SP500_PATH
    assert not uf.legacy_fallback
    _assert_review_contract(uf)


def test_shipped_file_review_is_internally_consistent():
    """`days_since_review` is derived from the file, never asserted as a
    literal — so this holds on the day the file is refreshed and on every
    day after."""
    uf = load_universe_file()
    today = date(2030, 1, 1)
    status = universe_review.file_status(uf, today=today)
    assert status["last_reviewed"] == uf.last_reviewed
    assert status["days_since_review"] == (today - date.fromisoformat(uf.last_reviewed)).days
    assert status["stale"] is (status["days_since_review"] > uf.review_cadence_days)


def test_tuple_api_still_matches_dataclass():
    tickers, auto_update = _load_universe()
    uf = load_universe_file()
    assert tickers == uf.tickers
    assert auto_update == uf.auto_update


def test_legacy_file_is_marked_superseded():
    cfg = json.loads((DATA_DIR / "sp100.json").read_text())
    assert cfg["_status"].startswith("legacy")
    assert "sp500.json" in cfg["_status"]


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def _freeze_today(monkeypatch, frozen: date) -> None:
    class _FrozenDate(date):
        @classmethod
        def today(cls):
            return frozen

    monkeypatch.setattr(universe_review, "date", _FrozenDate)


def test_fresh_file_is_not_stale(small_file):
    # small_file: reviewed 2026-09-01, cadence 30
    report = universe_review.review_universe(today=date(2026, 9, 2))
    assert report["file"] == "sp500.json"
    assert report["last_reviewed"] == small_file.last_reviewed
    assert report["review_cadence_days"] == small_file.review_cadence_days
    assert report["days_since_review"] == 1
    assert report["stale"] is False


def test_stale_when_last_review_is_older_than_the_cadence(monkeypatch, small_file):
    _freeze_today(monkeypatch, date(2026, 10, 6))  # 35 days after 2026-09-01
    report = universe_review.review_universe()
    assert report["days_since_review"] == 35
    assert report["stale"] is True


def test_exactly_at_the_cadence_is_not_yet_stale(monkeypatch, small_file):
    # The cadence is "review within N days"; day N itself is on time.
    _freeze_today(monkeypatch, date(2026, 10, 1))  # 30 days after 2026-09-01
    report = universe_review.review_universe()
    assert report["days_since_review"] == small_file.review_cadence_days
    assert report["stale"] is False


def test_missing_review_stamp_counts_as_stale():
    uf = UniverseFile(path=SP500_PATH, tickers=["AAPL"], last_reviewed=None)
    status = universe_review.file_status(uf)
    assert status["stale"] is True
    assert status["days_since_review"] is None


# ---------------------------------------------------------------------------
# DB drift
# ---------------------------------------------------------------------------

def test_report_shape_and_db_diff(drift_rows, small_file):
    report = universe_review.review_universe()
    assert set(report) == {
        "file", "last_reviewed", "review_cadence_days", "days_since_review",
        "stale", "ticker_count", "auto_update_count", "db", "diff_vs_db", "feed",
    }
    assert report["ticker_count"] == 3
    assert report["auto_update_count"] == 1
    assert report["feed"] is None
    assert report["db"]["auto_analysis_count"] >= 2
    assert report["db"]["on_demand_count"] >= 1
    diff = report["diff_vs_db"]
    assert "ZZRVW3" in diff["missing_in_db"]
    assert "AAPL" not in diff["missing_in_db"]
    assert "ZZRVW1" in diff["auto_analysis_not_in_file"]
    assert "ZZRVW2" not in diff["auto_analysis_not_in_file"]  # on-demand rows are fine


# ---------------------------------------------------------------------------
# Feed comparison — gated, read-only
# ---------------------------------------------------------------------------

def test_feed_skipped_without_fmp_key(monkeypatch, feed_never_called):
    monkeypatch.setattr(settings, "fmp_api_key", "")
    monkeypatch.setattr(settings, "enable_live_data", True)
    feed = universe_review.review_universe(compare_feed=True)["feed"]
    assert feed["source"] == "fmp"
    assert "FMP_API_KEY" in feed["error"]
    assert feed["added"] == [] and feed["removed"] == []
    assert feed["fetched_at"] is None


def test_feed_skipped_when_live_data_disabled(monkeypatch, feed_never_called):
    monkeypatch.setattr(settings, "fmp_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "enable_live_data", False)
    feed = universe_review.review_universe(compare_feed=True)["feed"]
    assert "ENABLE_LIVE_DATA" in feed["error"]


def test_feed_not_queried_unless_asked(monkeypatch, feed_never_called):
    monkeypatch.setattr(settings, "fmp_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "enable_live_data", True)
    assert universe_review.review_universe(compare_feed=False)["feed"] is None


def test_feed_diff_is_computed_and_nothing_is_written(monkeypatch, small_file):
    monkeypatch.setattr(settings, "fmp_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(
        FMPProvider, "get_sp500_constituents",
        lambda self: ["aapl", "MSFT", "NEWCO"],
    )
    before_bytes = SP500_PATH.read_bytes()
    before_mtime = SP500_PATH.stat().st_mtime_ns
    with SessionLocal() as db:
        rows_before = db.query(Company).count()

    feed = universe_review.review_universe(compare_feed=True)["feed"]

    assert feed["error"] is None
    assert feed["fetched_at"]
    assert feed["added"] == ["NEWCO"]
    assert feed["removed"] == ["ZZRVW3"]
    assert SP500_PATH.read_bytes() == before_bytes
    assert SP500_PATH.stat().st_mtime_ns == before_mtime
    with SessionLocal() as db:
        assert db.query(Company).count() == rows_before


def test_feed_empty_response_is_reported_not_raised(monkeypatch, small_file):
    monkeypatch.setattr(settings, "fmp_api_key", "not-a-real-key")
    monkeypatch.setattr(settings, "enable_live_data", True)
    monkeypatch.setattr(FMPProvider, "get_sp500_constituents", lambda self: None)
    feed = universe_review.review_universe(compare_feed=True)["feed"]
    assert "no constituents" in feed["error"]
    assert feed["added"] == [] and feed["removed"] == []


# ---------------------------------------------------------------------------
# Admin endpoint + cron-health
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    # Bare TestClient: the lifespan form runs the seeder on every test.
    return TestClient(app)


def test_endpoint_requires_the_admin_token(monkeypatch, client, feed_never_called, small_file):
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)
    assert client.get("/api/admin/universe-review").status_code == 401
    resp = client.get(
        "/api/admin/universe-review", headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["file"] == "sp500.json"
    assert body["last_reviewed"] == small_file.last_reviewed
    assert body["ticker_count"] == len(small_file.tickers)
    assert body["feed"] is None
    assert "missing_in_db" in body["diff_vs_db"]


def test_endpoint_compare_feed_without_key_explains(monkeypatch, client, feed_never_called):
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)
    monkeypatch.setattr(settings, "fmp_api_key", "")
    resp = client.get(
        "/api/admin/universe-review", params={"compare_feed": "true"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200
    assert "FMP_API_KEY" in resp.json()["feed"]["error"]


def test_cron_health_surfaces_universe_staleness(monkeypatch, client, small_file):
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)
    _freeze_today(monkeypatch, date(2026, 10, 6))  # past small_file's 30-day cadence
    resp = client.get(
        "/api/admin/cron-health", headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["universe_review"]) == {"last_reviewed", "days_since_review", "stale"}
    assert body["universe_review"]["last_reviewed"] == small_file.last_reviewed
    assert body["universe_review"]["days_since_review"] == 35
    assert body["universe_review"]["stale"] is True
    # Loops only — a stale file is a review task, not a cron failure.
    assert body["stale_count"] == sum(1 for r in body["loops"] if r["stale"])


# ---------------------------------------------------------------------------
# Refresh round-trip — the documented operator step must not break the review
# ---------------------------------------------------------------------------

def test_refreshed_file_meets_the_same_contract_and_reads_as_fresh(monkeypatch, tmp_path):
    """Regression: the shipped-file assertions above must hold for a file
    `refresh_universe_lists` has just written, or the operator workflow
    breaks CI. Writes only to tmp_path; the shipped file is untouched."""
    target = tmp_path / "sp500.json"
    target.write_text(json.dumps({
        "_review_cadence_days": 45,
        "_top_10_by_market_cap_2026_05": ["AAPL"],
        "tickers": ["AAPL", "OLDCO"],
    }))
    monkeypatch.setattr(refresh_universe_lists, "SP500_PATH", target)
    monkeypatch.setattr(
        FMPProvider, "get_sp500_constituents", lambda self: ["msft", "AAPL", "NEWCO"],
    )
    shipped_before = SP500_PATH.read_bytes()

    assert refresh_universe_lists.main([]) == 0

    assert SP500_PATH.read_bytes() == shipped_before
    uf = load_universe_file(target)
    _assert_review_contract(uf)
    assert uf.tickers == ["AAPL", "MSFT", "NEWCO"]
    assert uf.auto_update == ["AAPL"]                 # pin list carried forward
    assert uf.review_cadence_days == 45               # cadence carried forward
    assert uf.last_reviewed == date.today().isoformat()
    assert "refresh_universe_lists" in uf.review_source
    status = universe_review.file_status(uf)
    assert status["days_since_review"] == 0
    assert status["stale"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_prints_the_report_as_json(capsys, feed_never_called):
    before = SP500_PATH.read_bytes()
    assert cli.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["file"] == "sp500.json"
    assert out["feed"] is None
    assert SP500_PATH.read_bytes() == before
