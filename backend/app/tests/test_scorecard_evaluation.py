"""Phase 6 (slice C) — `services/scorecard_evaluation`: the worker job that
builds the month-end panel from `scorecard_scores` + `price_month_ends`,
runs the quintile / FF6 / double-LASSO arithmetic and persists the rows.

A planted panel (numpy `RandomState(0)`) under a synthetic version key:
next-month returns rise with the score, so the top quintile outperforms,
the FF6 alpha is positive and the LASSO reads the score as independent of
the controls. The Ken French series are synthetic and injected;
`factor_analytics._full_series_points` is replaced with a tripwire so a
provider can never be reached.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

import numpy as np
import pytest

from app.database import SessionLocal
from app.finance import double_selection_lasso as dsl
from app.finance import scorecard_evaluation_math as sem
from app.models import Company, PriceMonthEnd, ScorecardEvaluation, ScorecardRun, ScorecardScore
from app.services import factor_analytics, scorecard_queue
from app.services import scorecard_evaluation as ev
from app.tests.scorecard_helpers import next_month_end, prev_month_end, purge

VK = "fs-evaltest"
VK_SMALL = "fs-evaltiny"
# The seeded panel's runs carry their own requester so the per-test purge of
# evaluation runs never removes the panel itself.
PANEL_BY = "test-scorecard-panel"
SECTORS = ("Information Technology", "Financials", "Health Care")
FETCHED = datetime(2026, 9, 1)


def _month_ends(n: int, start: date = date(2022, 1, 31)) -> list[date]:
    out = [start]
    for _ in range(n - 1):
        out.append(next_month_end(out[-1]))
    return out


def seed_panel(version: str, *, prefix: str, n_names: int, n_months: int, slope: float = 0.02,
               noise: float = 0.03, seed: int = 0) -> list[str]:
    """Month-end succeeded runs + score rows + a price path whose next-month
    return is `slope·score + noise`. Prices start 13 months before the first
    score month so 12-1 momentum exists for every panel row."""
    rng = np.random.RandomState(seed)
    months = _month_ends(n_months)
    tickers = [f"{prefix}{i:03d}" for i in range(n_names)]
    pre = [prev_month_end(months[0], k) for k in range(13, 0, -1)]
    all_months = pre + months + [next_month_end(months[-1])]
    prices = {t: {} for t in tickers}
    with SessionLocal() as db:
        for i, t in enumerate(tickers):
            db.merge(Company(ticker=t, company_name=f"{t} Co", exchange="TEST", sector=SECTORS[i % 3],
                             industry="Test", universe_tier="data_only", beta=float(rng.uniform(0.6, 1.6))))
            level = 100.0
            for m in pre:
                level *= 1.0 + rng.normal(scale=0.02)
                prices[t][m] = level
        for k, as_of in enumerate(months):
            run = ScorecardRun(run_id=str(uuid.uuid4()), version_key=version, as_of=as_of, run_kind="month_end",
                               status="succeeded", attempts=1, requested_by=PANEL_BY, enqueued_at=FETCHED,
                               started_at=FETCHED, finished_at=FETCHED, params={"is_month_end": True})
            db.add(run)
            db.flush()
            scores = rng.normal(size=n_names)
            rets = slope * scores + rng.normal(scale=noise, size=n_names)
            for i, t in enumerate(tickers):
                if k == 0:
                    prices[t][as_of] = prices[t][pre[-1]] * (1.0 + rng.normal(scale=0.02))
                prices[t][next_month_end(as_of)] = prices[t][as_of] * (1.0 + float(rets[i]))
                db.add(ScorecardScore(
                    run_id=run.id, version_key=version, as_of=as_of, ticker=t, sector=SECTORS[i % 3],
                    overall_z=float(scores[i]), overall_score=50.0 + 20.0 * float(scores[i]), coverage=1.0,
                    universe_percentile=50.0, category_z={}, category_percentile={},
                    feature_raw={"roa": float(rng.uniform(0.0, 0.2)),
                                 "_context": {"market_cap": float(rng.uniform(1e9, 1e11))}},
                    feature_z={}, top_positive=[], top_negative=[], is_month_end=True, created_at=FETCHED,
                ))
        for t in tickers:
            for m in all_months:
                p = prices[t][m]
                db.add(PriceMonthEnd(ticker=t, month_end=m, price_date=m, close=p, adjusted_close=p,
                                     source="test", fetched_at=FETCHED))
        db.commit()
    return tickers


def synthetic_factors(months: list[str], *, seed: int = 1) -> dict[str, list[dict]]:
    rng = np.random.RandomState(seed)
    out: dict[str, list[dict]] = {}
    for sid in sem.KFR_MONTHLY_FACTOR_IDS.values():
        out[sid] = [{"date": f"{sem.return_month(m)}-28", "value": float(rng.normal(scale=0.03))} for m in months]
    out[sem.KFR_MONTHLY_RF_ID] = [{"date": f"{sem.return_month(m)}-28", "value": 0.002} for m in months]
    return out


BIG = {"prefix": "ZEV", "n_names": 90, "n_months": 28}
SMALL = {"prefix": "ZET", "n_names": 5, "n_months": 3}


@pytest.fixture(scope="module", autouse=True)
def _panels():
    big = [f"ZEV{i:03d}" for i in range(BIG["n_names"])]
    small = [f"ZET{i:03d}" for i in range(SMALL["n_names"])]
    purge(big + small, versions=(VK, VK_SMALL))
    seed_panel(VK, **BIG)
    seed_panel(VK_SMALL, **SMALL, slope=0.0)
    # Register the synthetic version (inactive) so the read model resolves it.
    from app.finance import scorecard_spec
    from app.models import ScorecardVersion
    with SessionLocal() as db:
        db.add(ScorecardVersion(version_key=VK, spec_hash="test", is_active=False,
                                spec_json=scorecard_spec.spec_as_dict(version_key=VK)))
        db.commit()
    yield
    purge(big + small, versions=(VK, VK_SMALL))


@pytest.fixture(autouse=True)
def _no_providers(monkeypatch):
    def tripwire(series_id):
        raise AssertionError(f"evaluation reached a provider for {series_id}")

    monkeypatch.setattr(factor_analytics, "_full_series_points", tripwire)
    _purge_evaluate_runs()
    yield
    _purge_evaluate_runs()


def _purge_evaluate_runs() -> None:
    """Only the rows `run_evaluation` creates (kind=evaluate), never the panel."""
    with SessionLocal() as db:
        db.query(ScorecardRun).filter(
            ScorecardRun.version_key.in_((VK, VK_SMALL)), ScorecardRun.run_kind == "evaluate",
        ).delete(synchronize_session=False)
        db.commit()


def _rows(version: str) -> dict[str, ScorecardEvaluation]:
    with SessionLocal() as db:
        rows = db.query(ScorecardEvaluation).filter(ScorecardEvaluation.version_key == version).order_by(
            ScorecardEvaluation.id.desc()).all()
        db.expunge_all()
    out: dict[str, ScorecardEvaluation] = {}
    for r in rows:
        out.setdefault(r.eval_kind, r)
    return out


def _clear_rows(version: str) -> None:
    with SessionLocal() as db:
        db.query(ScorecardEvaluation).filter(ScorecardEvaluation.version_key == version).delete()
        db.commit()


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def test_build_panel_joins_forward_returns_and_controls():
    panel = ev.build_panel(VK)
    assert panel["n_rows"] == BIG["n_names"] * BIG["n_months"] and panel["n_missing_return"] == 0
    assert len(panel["months"]) == BIG["n_months"]
    obs, ctl = panel["observations"][0], panel["controls"][0]
    assert obs.forward_return is not None and obs.score is not None and obs.coverage == 1.0
    assert all(ctl[k] is not None for k in ev.LASSO_CONTROLS) and ctl["sector"] in SECTORS
    # Forward return is the stored price path, not a recomputation.
    with SessionLocal() as db:
        p0 = db.query(PriceMonthEnd).filter_by(ticker=obs.ticker, month_end=date.fromisoformat(obs.as_of)).one().close
        p1 = db.query(PriceMonthEnd).filter_by(ticker=obs.ticker,
                                               month_end=next_month_end(date.fromisoformat(obs.as_of))).one().close
    assert obs.forward_return == pytest.approx(p1 / p0 - 1.0)


def test_build_panel_uses_adjusted_close_only_when_both_ends_have_one():
    t = "ZEV000"
    as_of = date(2022, 1, 31)
    with SessionLocal() as db:
        start = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=as_of).one()
        end = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=next_month_end(as_of)).one()
        old = (start.adjusted_close, end.adjusted_close)
        start.adjusted_close, end.adjusted_close = start.close * 0.5, None   # a split on one side only
        db.commit()
    try:
        obs = next(o for o in ev.build_panel(VK)["observations"] if o.ticker == t and o.as_of == as_of.isoformat())
        with SessionLocal() as db:
            c0 = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=as_of).one().close
            c1 = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=next_month_end(as_of)).one().close
        assert obs.forward_return == pytest.approx(c1 / c0 - 1.0), "mixed bases must fall back to raw closes"
    finally:
        with SessionLocal() as db:
            start = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=as_of).one()
            end = db.query(PriceMonthEnd).filter_by(ticker=t, month_end=next_month_end(as_of)).one()
            start.adjusted_close, end.adjusted_close = old
            db.commit()


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def test_run_evaluation_writes_quintile_ff6_and_lasso_rows_on_the_planted_panel():
    _clear_rows(VK)
    months = ev.build_panel(VK)["months"]
    out = ev.run_evaluation(VK, factor_loader=lambda: synthetic_factors(months))
    assert out["run"]["status"] == scorecard_queue.STATUS_SUCCEEDED, out
    assert set(out["rows"]) == set(ev.EVAL_KINDS) and "written=3" in out["note"]
    rows = _rows(VK)
    assert set(rows) == set(ev.EVAL_KINDS)
    for r in rows.values():
        assert r.result["caveats"] == list(sem.EVALUATION_CAVEATS), "caveats travel verbatim on every row"
        assert r.params["min_leg"] == 15 and r.params["min_months"] == 24 and r.params["min_obs"] == 2000
        assert r.params["min_coverage"] == 0.6 and r.params["n_quantiles"] == 5
        assert r.sample_start == date(2022, 1, 31) and r.sample_end == date.fromisoformat(months[-1])
        assert r.run_id == out["run"]["run_id"]

    quint = rows[ev.KIND_QUINTILE].result
    assert quint["n_months"] == BIG["n_months"] and quint["n_skipped"] == 0
    assert quint["mean_spread"] > 0 and quint["t_stat"] > 3 and quint["monotonic"] is True
    assert [q["q"] for q in quint["quantile_table"]] == [1, 2, 3, 4, 5]
    assert all(m["n_long"] >= 15 and m["n_short"] >= 15 for m in quint["months"])
    assert quint["cumulative_spread"] is not None and quint["stats_note"] is None
    assert rows[ev.KIND_QUINTILE].n_obs == BIG["n_names"] * BIG["n_months"]

    ff6 = rows[ev.KIND_FF6].result
    assert ff6["insufficient"] is False and ff6["n_months"] == BIG["n_months"]
    assert set(ff6["betas"]) == set(sem.FF6_FACTOR_NAMES) and all(v is not None for v in ff6["betas"].values())
    assert ff6["alpha_monthly"] > 0 and ff6["alpha_t"] is not None and ff6["aligned"]["n_missing"] == 0

    lasso = rows[ev.KIND_LASSO].result
    assert lasso["verdict"] in dsl.VERDICTS
    assert lasso["verdict"] == dsl.VERDICT_INDEPENDENT and lasso["direction"] == "positive"
    assert lasso["n_obs"] >= 2000 and lasso["n_months"] == BIG["n_months"]
    assert lasso["se_hc1"] is not None and lasso["se_cluster_month"] is not None
    assert lasso["controls"][: len(ev.LASSO_CONTROLS)] == list(ev.LASSO_CONTROLS)
    assert any(c.startswith("sector:") for c in lasso["controls"])
    assert "scenario" in " ".join(lasso["caveats"]).lower() or "recommendation" in " ".join(lasso["caveats"]).lower()

    # The read model serves exactly these rows, caveats attached.
    from app.services import scorecard_service
    view = scorecard_service.evaluation_view(VK)
    assert set(view.evaluations) == set(ev.EVAL_KINDS) and view.note == ""
    assert view.evaluations[ev.KIND_LASSO].result["verdict"] == dsl.VERDICT_INDEPENDENT
    assert view.evaluations[ev.KIND_QUINTILE].params["min_leg"] == 15
    assert view.caveats == list(sem.EVALUATION_CAVEATS)


def test_thin_panel_writes_honest_insufficient_rows():
    _clear_rows(VK_SMALL)
    out = ev.run_evaluation(VK_SMALL, factor_loader=lambda: synthetic_factors(ev.build_panel(VK_SMALL)["months"]))
    assert out["run"]["status"] == scorecard_queue.STATUS_SUCCEEDED
    rows = _rows(VK_SMALL)
    quint = rows[ev.KIND_QUINTILE].result
    assert quint["n_months"] == 0 and quint["n_skipped"] == SMALL["n_months"]
    assert quint["mean_spread"] is None and "no months qualified" in quint["stats_note"]
    assert all("needed for 15 per leg" in s["reason"] for s in quint["skipped_months"])
    ff6 = rows[ev.KIND_FF6].result
    assert ff6["insufficient"] is True and ff6["alpha_monthly"] is None and ff6["reasons"]
    lasso = rows[ev.KIND_LASSO].result
    assert lasso["verdict"] == dsl.VERDICT_INSUFFICIENT and lasso["coef_d"] is None
    assert "lasso_verdict=insufficient_data" in out["note"]


def test_factor_series_outage_is_an_insufficient_regression_not_a_failed_run():
    _clear_rows(VK)

    def broken():
        raise RuntimeError("ken french down apikey=sk-abcdefghijklmnop")

    out = ev.run_evaluation(VK, factor_loader=broken)
    assert out["run"]["status"] == scorecard_queue.STATUS_SUCCEEDED
    ff6 = _rows(VK)[ev.KIND_FF6].result
    assert ff6["insufficient"] is True and ff6["reasons"] == ["factor series unavailable: RuntimeError"]
    assert "sk-abc" not in str(ff6)
    assert _rows(VK)[ev.KIND_QUINTILE].result["n_months"] == BIG["n_months"], "the spread is still measured"


def test_a_raising_panel_fails_the_run_with_the_type(monkeypatch):
    def boom(version_key, **kw):
        raise ValueError("panel broke")

    monkeypatch.setattr(ev, "build_panel", boom)
    out = ev.run_evaluation(VK, factor_loader=lambda: {})
    assert out["run"]["status"] == scorecard_queue.STATUS_FAILED and out["run"]["error_type"] == "ValueError"
    assert out["rows"] == {}


def test_default_loader_is_the_catalog_series_and_never_runs_here():
    with pytest.raises(AssertionError, match="reached a provider"):
        ev._load_factor_points()


def test_jsonable_scrubs_numpy_and_non_finite():
    out = ev._jsonable({"a": np.float64(1.5), "b": float("nan"), "c": [np.int64(2), (1, 2)], "d": np.array([1.0])})
    assert out == {"a": 1.5, "b": None, "c": [2, [1, 2]], "d": [1.0]}
