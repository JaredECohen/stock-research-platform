"""Phase 6 (slice C) — `services/scorecard_service`: the run body with
lineage, the version registry, the point-in-time rule at run level, the
skip rule, failure atomicity, the readers and the frozen export.

Ten synthetic tickers across three sectors, seeded by
`scorecard_helpers.seed_universe`; the clock is pinned through
`scorecard_service._utcnow`; the daily price series seam returns nothing
unless a test supplies one, so no provider is ever reached.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime

import pytest

from app.database import SessionLocal
from app.finance import scorecard_spec
from app.models import ScorecardScore, ScorecardVersion
from app.services import scorecard_pit, scorecard_queue
from app.services import scorecard_service as svc
from app.tests.scorecard_helpers import REQUESTED_BY, insert_run, purge, score_rows_for_run, seed_universe

UNIVERSE = {
    "Technology": ["ZSC0", "ZSC1", "ZSC2", "ZSC3"],
    "Financial Services": ["ZSC4", "ZSC5", "ZSC6"],
    "Healthcare": ["ZSC7", "ZSC8", "ZSC9"],
}
TICKERS = sorted(t for ts in UNIVERSE.values() for t in ts)
AS_OF = date(2026, 6, 30)            # FY2025 (available 2026-03-16) is knowable
AS_OF_EARLY = date(2026, 2, 28)      # FY2025 is NOT yet knowable; FY2024 is
NOW = datetime(2026, 7, 15, 12, 0, 0)


@pytest.fixture(scope="module", autouse=True)
def _universe():
    purge(TICKERS)
    seed_universe(UNIVERSE, price_through=AS_OF, price_months=20)
    yield
    purge(TICKERS)


@pytest.fixture(autouse=True)
def _clock_and_no_series(monkeypatch):
    # The queue stamps `finished_at` (the export's `generated_at`); the
    # service stamps `created_at` and judges staleness. Pin both.
    monkeypatch.setattr(svc, "_utcnow", lambda: NOW)
    monkeypatch.setattr(scorecard_queue, "_utcnow", lambda: NOW)
    monkeypatch.setattr(svc, "_price_series", lambda ticker: None)
    purge(requested_by=(REQUESTED_BY,))
    yield
    purge(requested_by=(REQUESTED_BY,))


def _run(as_of: date = AS_OF, **kw):
    return svc.run_scorecard(None, as_of, tickers=TICKERS, requested_by=REQUESTED_BY, **kw)


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

def test_ensure_version_registered_is_idempotent_and_hashes_the_code_spec():
    first = svc.ensure_version_registered()
    second = svc.ensure_version_registered()
    assert first["version_key"] == second["version_key"] == scorecard_spec.VERSION_KEY
    assert first["spec_hash"] == second["spec_hash"] == scorecard_spec.spec_hash()
    assert second["created"] is False and second["changed"] is False
    with SessionLocal() as db:
        rows = db.query(ScorecardVersion).filter(ScorecardVersion.version_key == scorecard_spec.VERSION_KEY).all()
    assert len(rows) == 1 and rows[0].is_active and rows[0].spec_json["version_key"] == scorecard_spec.VERSION_KEY
    active = svc.active_version()
    assert active["source"] == "registry" and active["spec_hash"] == scorecard_spec.spec_hash()


def test_resolve_version_falls_back_to_code_and_rejects_unknown_keys():
    assert svc.resolve_version(scorecard_spec.VERSION_KEY)["version_key"] == scorecard_spec.VERSION_KEY
    with pytest.raises(svc.UnknownVersion):
        svc.resolve_version("fs-does-not-exist")
    spec = svc.spec_view(None)
    assert spec["spec_hash"] == scorecard_spec.spec_hash()
    assert "50 = z of 0" in spec["score_scale"]
    assert {f["name"] for fam in spec["families"] for f in fam["features"]} == set(scorecard_spec.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# run_scorecard end to end
# ---------------------------------------------------------------------------

def test_run_scorecard_writes_one_row_per_ticker_with_lineage():
    out = _run()
    assert out["status"] == scorecard_queue.STATUS_SUCCEEDED, out
    assert out["universe_size"] == 10 and out["scored_count"] == 10
    assert "written=10" in out["note"] and "as_of=2026-06-30" in out["note"] and "month_end=1" in out["note"]
    assert "price_store=10" in out["note"] and "no_price=0" in out["note"]
    assert out["params"]["spec_hash"] == scorecard_spec.spec_hash() and out["params"]["is_month_end"] is True
    assert out["inputs_hash"] and len(out["inputs_hash"]) == 64

    rows = score_rows_for_run(out["id"])
    assert [r.ticker for r in rows] == TICKERS
    for r in rows:
        assert r.is_month_end is True and r.as_of == AS_OF and r.latest_period == "FY2025"
        assert r.data_available_at == date(2026, 3, 16) and r.price_date == AS_OF
        assert r.overall_z is not None and r.overall_score is not None
        assert 0.0 < r.universe_percentile <= 100.0
        assert 0.0 < r.sector_percentile <= 100.0
        assert 0.0 < r.coverage <= 1.0
        assert r.inputs_hash and len(r.inputs_hash) == 64
        # Observed layer is the raw ratio; interpretation layer is the z. Metadata
        # rides under underscore keys and never mixes with feature names.
        values, meta = svc._split_meta(r.feature_raw)
        assert set(values) == set(scorecard_spec.FEATURE_NAMES)
        assert set(meta) == {"_reasons", "_context"}
        assert meta["_context"]["price_basis"] == "month_end_store"
        zs, zmeta = svc._split_meta(r.feature_z)
        assert set(zs) == set(scorecard_spec.FEATURE_NAMES) and set(zmeta) == {"_basis", "_universe"}
        cats, cmeta = svc._split_meta(r.category_z)
        assert set(cats) == set(scorecard_spec.FAMILY_NAMES)
        pcts, pmeta = svc._split_meta(r.category_percentile)
        assert set(pcts) == set(scorecard_spec.FAMILY_NAMES) and "_sector" in pmeta
        assert r.top_positive and r.top_negative
        assert all(set(it) == {"feature", "family", "z", "contribution"} for it in r.top_positive)

    # Financials never get EV / leverage features, and the exclusion is a
    # reason, not a zero.
    fin = next(r for r in rows if r.ticker == "ZSC4")
    assert fin.sector == "Financials"
    assert fin.feature_raw["ebitda_ev_yield"] is None
    assert fin.feature_raw["_reasons"]["ebitda_ev_yield"].startswith("excluded:")
    assert fin.feature_z["_basis"]["net_debt_to_ebitda"] == "n/a:excluded"


def test_percentiles_are_rank_based_and_scores_map_z_to_100():
    from app.finance.factor_scores import _z_to_100
    out = _run()
    rows = score_rows_for_run(out["id"])
    by_score = sorted(rows, key=lambda r: r.overall_z)
    assert by_score[-1].universe_percentile == 100.0
    assert by_score[0].universe_percentile == pytest.approx(10.0)
    for r in rows:
        assert r.overall_score == pytest.approx(_z_to_100(r.overall_z))


def test_point_in_time_exclusion_changes_the_score():
    late = _run(AS_OF)
    early = _run(AS_OF_EARLY)
    assert early["status"] == scorecard_queue.STATUS_SUCCEEDED
    assert "pit_excluded=" in early["note"] and "pit_excluded=0" not in early["note"]
    late_rows = {r.ticker: r for r in score_rows_for_run(late["id"])}
    early_rows = {r.ticker: r for r in score_rows_for_run(early["id"])}
    assert all(r.latest_period == "FY2024" for r in early_rows.values())
    assert all(r.data_available_at == date(2025, 3, 16) for r in early_rows.values())
    changed = [t for t in TICKERS if late_rows[t].overall_score != early_rows[t].overall_score]
    assert changed, "excluding a fiscal year that was not yet public must move scores"
    assert late["inputs_hash"] != early["inputs_hash"]


def test_identical_inputs_are_skipped_without_writing_rows():
    first = _run()
    second = _run()
    assert second["status"] == scorecard_queue.STATUS_SKIPPED
    assert "skipped=identical_inputs" in second["note"] and first["run_id"] in second["note"]
    assert second["inputs_hash"] == first["inputs_hash"]
    assert score_rows_for_run(second["id"]) == []
    # The reader still resolves the first (succeeded) run.
    assert svc.latest_score("ZSC0")["run_id"] == first["run_id"]


def test_failed_run_leaves_no_score_rows(monkeypatch):
    from app.finance import scorecard_normalize

    def boom(*a, **k):
        raise RuntimeError("normalisation exploded with token=sk-secretsecretsecret")

    monkeypatch.setattr(scorecard_normalize, "normalize_universe", boom)
    out = _run()
    assert out["status"] == scorecard_queue.STATUS_FAILED
    assert out["error_type"] == "RuntimeError" and "sk-secret" not in out["error_message"]
    assert score_rows_for_run(out["id"]) == []
    assert svc.latest_score("ZSC0") is None, "a failed run must be invisible to readers"


def test_unsupported_version_fails_the_run_instead_of_raising():
    out = svc.run_scorecard("fs-v0", AS_OF, tickers=TICKERS, requested_by=REQUESTED_BY)
    assert out["status"] == scorecard_queue.STATUS_FAILED and out["error_type"] == "UnsupportedVersion"


def test_empty_explicit_universe_succeeds_with_nothing_written():
    out = svc.run_scorecard(None, AS_OF, tickers=[], requested_by=REQUESTED_BY)
    assert out["status"] == scorecard_queue.STATUS_SUCCEEDED and "written=0" in out["note"]
    assert out["universe_size"] == 0


def test_ticker_without_statements_is_written_as_insufficient_not_dropped():
    out = svc.run_scorecard(None, AS_OF, tickers=TICKERS + ["ZSCX"], requested_by=REQUESTED_BY)
    rows = {r.ticker: r for r in score_rows_for_run(out["id"])}
    assert "ZSCX" in rows
    ghost = rows["ZSCX"]
    assert ghost.overall_z is None and ghost.overall_score is None and ghost.coverage == 0.0
    assert ghost.feature_raw["_reasons"]["revenue_growth_1y"] == "no_snapshot"
    assert "insufficient=1" in out["note"] and "no_snapshot=1" in out["note"] and "sector_missing" in (
        ghost.category_z["_notes"]
    )


def test_daily_as_of_prefers_the_cached_series_and_records_the_basis(monkeypatch):
    series = [{"date": "2026-06-10", "close": 50.0}, {"date": "2026-06-12", "close": 52.0},
              {"date": "2026-06-20", "close": 99.0}]
    monkeypatch.setattr(svc, "_price_series", lambda ticker: series)
    out = _run(date(2026, 6, 15), run_kind=scorecard_queue.KIND_SCHEDULED)
    assert out["status"] == scorecard_queue.STATUS_SUCCEEDED and "month_end=0" in out["note"]
    assert "price_series=10" in out["note"]
    rows = score_rows_for_run(out["id"])
    assert all(r.is_month_end is False and r.price_date == date(2026, 6, 12) for r in rows)
    assert all(r.feature_raw["_context"]["price"] == 52.0 for r in rows)
    # Daily rows fall back to the (stale) store when the series is missing,
    # and say so in price_date rather than pretending.
    monkeypatch.setattr(svc, "_price_series", lambda ticker: None)
    fallback = _run(date(2026, 6, 16), run_kind=scorecard_queue.KIND_SCHEDULED)
    rows = score_rows_for_run(fallback["id"])
    assert all(r.price_date == date(2026, 5, 31) for r in rows)
    assert all(r.feature_raw["_context"]["price_basis"] == "month_end_store" for r in rows)


def test_price_context_reports_none_honestly():
    ctx, basis = svc.price_context("ZSC0", AS_OF, store_price=None, prefer_store=True, shares_fallback=None, today=NOW.date())
    assert ctx["price"] is None and basis == "none"
    ctx, basis = svc.price_context("ZSC0", AS_OF, store_price=(10.0, AS_OF), prefer_store=False, shares_fallback=1.0,
                                   today=NOW.date())
    assert (ctx["price"], ctx["price_date"], basis) == (10.0, AS_OF, "month_end_store")


# ---------------------------------------------------------------------------
# pit_prepare
# ---------------------------------------------------------------------------

def test_pit_prepare_records_failed_tickers_without_aborting(monkeypatch):
    def fake_sync(ticker, **kw):
        if ticker == "ZSC1":
            raise scorecard_pit.PriceSeriesUnavailable("chain exhausted")
        if ticker == "ZSC2":
            raise RuntimeError("boom")
        return {"months": 2, "written": 1, "skipped": 0}

    monkeypatch.setattr(scorecard_pit, "sync_price_month_ends", fake_sync)
    run_id = scorecard_queue.create_running_row(
        version_key=svc.VERSION_KEY, as_of=AS_OF, kind=scorecard_queue.KIND_PIT_PREPARE, requested_by=REQUESTED_BY,
        params={"tickers": TICKERS},
    )
    out = svc.pit_prepare(run_row_id=run_id, tickers=TICKERS)
    assert out["status"] == scorecard_queue.STATUS_SUCCEEDED
    assert "failed=2" in out["note"] and "unavailable=1" in out["note"] and "months=16" in out["note"]
    assert out["params"]["failed_tickers"] == ["ZSC1", "ZSC2"]


def test_pit_prepare_fails_only_when_every_ticker_failed(monkeypatch):
    def all_fail(ticker, **kw):
        raise scorecard_pit.PriceSeriesUnavailable("outage")

    monkeypatch.setattr(scorecard_pit, "sync_price_month_ends", all_fail)
    run_id = scorecard_queue.create_running_row(
        version_key=svc.VERSION_KEY, as_of=AS_OF, kind=scorecard_queue.KIND_PIT_PREPARE, requested_by=REQUESTED_BY,
    )
    out = svc.pit_prepare(run_row_id=run_id, tickers=TICKERS[:3])
    assert out["status"] == scorecard_queue.STATUS_FAILED and out["error_type"] == "PriceSeriesUnavailable"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_latest_score_summary_and_detail():
    out = _run()
    row = svc.latest_score("ZSC0")
    assert row is not None and row["run_id"] == out["run_id"] and row["as_of"] == AS_OF
    assert row["stale"] is False
    assert svc.latest_score("ZSC0", as_of=date(2026, 1, 1)) is None

    summary = svc.latest_summary("ZSC0")
    assert summary is not None and summary.version_key == svc.VERSION_KEY
    assert set(summary.categories) == set(scorecard_spec.FAMILY_NAMES)
    val = summary.categories["valuation"]
    assert val.weight == pytest.approx(1 / 8) and val.n_features == 4 and val.n_available == 4
    assert val.score is not None and val.percentile is not None
    assert summary.top_positive and summary.top_negative and summary.disagreement is None
    assert "compounder" in summary.profiles and "inflection" in summary.profiles

    detail = svc.ticker_detail("ZSC4", months=12)
    assert detail is not None and detail.ticker == "ZSC4" and detail.sector == "Financials"
    assert detail.sector_raw == "Financial Services" and detail.company_name == "ZSC4 Test Co"
    assert detail.spec_hash == scorecard_spec.spec_hash()
    names = [f.name for f in detail.features]
    assert names == list(scorecard_spec.FEATURE_NAMES)
    ev = next(f for f in detail.features if f.name == "ebitda_ev_yield")
    assert ev.applicable is False and ev.raw is None and ev.z is None and ev.reason.startswith("excluded:")
    total = sum(f.contribution for f in detail.features if f.contribution is not None)
    assert total == pytest.approx(detail.overall_z, abs=1e-9), "contributions must add up to the overall z"
    assert [p.as_of for p in detail.history] == [AS_OF]
    assert svc.ticker_detail("ZSCZ") is None
    with pytest.raises(svc.UnknownVersion):
        svc.ticker_detail("ZSC0", version_key="fs-nope")


def test_universe_table_sorts_filters_and_rejects_unknown_columns():
    _run()
    table = svc.universe_table(as_of=AS_OF, sort_by="overall_score", order="desc", limit=100)
    assert table is not None and table.as_of == AS_OF and table.universe_size == 10 and table.scored == 10
    scores = [r.overall_score for r in table.rows]
    assert scores == sorted(scores, reverse=True) and [r.rank for r in table.rows] == list(range(1, 11))
    asc = svc.universe_table(as_of=AS_OF, sort_by="valuation", order="asc", limit=3)
    vals = [r.category_z["valuation"] for r in asc.rows]
    assert len(vals) == 3 and vals == sorted(vals)
    fin = svc.universe_table(as_of=AS_OF, sector="financ")
    assert {r.ticker for r in fin.rows} == {"ZSC4", "ZSC5", "ZSC6"}
    assert svc.universe_table(as_of=date(1990, 1, 1)) is None
    with pytest.raises(ValueError):
        svc.universe_table(as_of=AS_OF, sort_by="__class__")


def test_score_history_returns_month_end_rows_oldest_first():
    _run(AS_OF_EARLY)
    _run(AS_OF)
    _run(date(2026, 6, 15), run_kind=scorecard_queue.KIND_SCHEDULED)   # daily: excluded from history
    points = svc.score_history("ZSC0", version_key=svc.VERSION_KEY, months=36)
    assert [p["as_of"] for p in points] == [AS_OF_EARLY, AS_OF]
    assert svc.score_history("ZSC0", version_key=svc.VERSION_KEY, months=1)[0]["as_of"] == AS_OF


def test_evaluation_view_always_carries_caveats():
    from app.finance.scorecard_evaluation_math import EVALUATION_CAVEATS
    view = svc.evaluation_view("fs-v1")
    assert list(view.caveats) == list(EVALUATION_CAVEATS)
    if not view.evaluations:
        assert "no evaluation has run" in view.note


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_gc_deletes_old_dailies_and_keeps_month_ends():
    old_daily = insert_run(version_key=svc.VERSION_KEY, as_of=date(2026, 4, 15), kind="scheduled")
    old_month = insert_run(version_key=svc.VERSION_KEY, as_of=date(2026, 4, 30), kind="scheduled")
    with SessionLocal() as db:
        db.add(ScorecardScore(run_id=old_daily, version_key=svc.VERSION_KEY, as_of=date(2026, 4, 15),
                              ticker="ZSC0", is_month_end=False, coverage=1.0))
        db.add(ScorecardScore(run_id=old_month, version_key=svc.VERSION_KEY, as_of=date(2026, 4, 30),
                              ticker="ZSC0", is_month_end=True, coverage=1.0))
        db.commit()
    deleted = svc.gc_daily_rows(today=date(2026, 7, 15), retention_days=45)
    assert deleted == 1
    assert score_rows_for_run(old_daily) == [] and len(score_rows_for_run(old_month)) == 1


# ---------------------------------------------------------------------------
# Export contract
# ---------------------------------------------------------------------------

FROZEN_HEADER = (
    "contract_version,version_key,spec_hash,run_id,as_of,price_date,data_available_at,ticker,sector,coverage,"
    "overall_z,overall_score,universe_percentile,sector_percentile,z_valuation,z_quality,z_growth,"
    "z_profitability,z_efficiency,z_leverage,z_capital_allocation,z_earnings_quality,pct_valuation,pct_quality,"
    "pct_growth,pct_profitability,pct_efficiency,pct_leverage,pct_capital_allocation,pct_earnings_quality,"
    "top_positive_1,top_positive_2,top_positive_3,top_negative_1,top_negative_2,top_negative_3,generated_at\n"
)


def test_export_csv_columns_are_frozen_and_values_are_six_dp():
    assert ",".join(svc.EXPORT_COLUMNS_V1) + "\n" == FROZEN_HEADER
    out = _run()
    run = svc.export_run(as_of=AS_OF)
    assert run["run_id"] == out["run_id"] and run["spec_hash"] == scorecard_spec.spec_hash()
    chunks = list(svc.iter_export_csv(run))
    assert chunks[0] == FROZEN_HEADER
    body = "".join(chunks)
    rows = list(csv.DictReader(io.StringIO(body)))
    assert [r["ticker"] for r in rows] == TICKERS
    first = rows[0]
    assert first["contract_version"] == "v1" and first["run_id"] == out["run_id"]
    assert first["as_of"] == "2026-06-30" and first["generated_at"] == "2026-07-15T12:00:00"
    assert len(first["overall_score"].split(".")[1]) == 6
    fin = next(r for r in rows if r["ticker"] == "ZSC4")
    assert fin["z_leverage"] == "" and fin["pct_leverage"] == "", "nulls are empty cells, never zero"
    assert first["top_positive_1"] in scorecard_spec.FEATURE_NAMES


def test_export_json_shares_the_column_keys_and_can_carry_features():
    _run()
    run = svc.export_run(as_of=AS_OF)
    payload = json.loads("".join(svc.iter_export_json(run, include_features=True)))
    assert payload["contract"] == "v1" and payload["columns"] == list(svc.EXPORT_COLUMNS_V1)
    assert len(payload["rows"]) == 10
    row = payload["rows"][0]
    assert list(row)[: len(svc.EXPORT_COLUMNS_V1)] == list(svc.EXPORT_COLUMNS_V1)
    assert set(row["feature_raw"]) == set(scorecard_spec.FEATURE_NAMES) and "_reasons" not in row["feature_raw"]
    assert isinstance(row["overall_score"], float)
    plain = json.loads("".join(svc.iter_export_json(run)))
    assert "feature_raw" not in plain["rows"][0]


def test_export_of_a_run_with_no_rows_is_header_only():
    empty = insert_run(version_key=svc.VERSION_KEY, as_of=date(2018, 6, 30), kind="manual")
    run = svc.export_run(as_of=date(2018, 6, 30))
    assert run["id"] == empty
    assert list(svc.iter_export_csv(run)) == [FROZEN_HEADER]
    assert json.loads("".join(svc.iter_export_json(run)))["rows"] == []
