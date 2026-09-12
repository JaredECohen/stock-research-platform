"""`services/unit_economics.py` — the observed cost of a metered operation.

Every test seeds `llm_call_logs` rows directly and pins the clock into 2019,
so the window can never pick up a row another test wrote with `utcnow()`, and
the teardown removes them by the marker agent name. No LLM or provider call
is made anywhere in this module — one test proves it by patching every entry
point to raise.

Costs are engineered to be checkable by hand: `claude-haiku-4-5` is priced at
$0.50 per million input tokens (`llm_metrics.MODEL_PRICES_PER_MTOK`), so
20,000 input tokens and no output tokens is exactly one cent.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import LLMCallLog
from app.services import unit_economics as ue

MARKER = "unit_economics_test_agent"
MODEL = "claude-haiku-4-5"          # $0.50 / MTok in, $2.50 / MTok out
TOKENS_PER_CENT = 20_000            # 20,000 × 0.50 / 1e6 = $0.01
# Far enough in the past that no other test's rows can land in the window.
CLOCK = datetime(2019, 5, 15, 12, 0, 0)
ADMIN_TOKEN = "admin-token-unit-economics-tests"


def _seed(rows: list[dict[str, Any]]) -> None:
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        for i, row in enumerate(rows):
            db.add(LLMCallLog(
                run_id=row.get("run_id"),
                agent_name=MARKER,
                provider=row.get("provider", "anthropic"),
                model=row.get("model", MODEL),
                route="cheap",
                tokens_in=row.get("tokens_in", 0),
                tokens_out=row.get("tokens_out", 0),
                duration_ms=100,
                success=row.get("success", True),
                feature=row["feature"],
                # Ordered backwards from the clock so "newest first" is
                # well-defined for the scan-cap test.
                generated_at=CLOCK - timedelta(minutes=i + 1),
            ))
        db.commit()


def _cents(n: int) -> dict[str, int]:
    """A call costing exactly `n` cents."""
    return {"tokens_in": TOKENS_PER_CENT * n, "tokens_out": 0}


@pytest.fixture(autouse=True)
def _clean_marker_rows():
    def wipe() -> None:
        with SessionLocal() as db:
            LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            db.query(LLMCallLog).filter(LLMCallLog.agent_name == MARKER).delete()
            db.commit()
    wipe()
    yield
    wipe()


def _report(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("now", CLOCK + timedelta(minutes=1))
    return ue.build_report(**kwargs)


# ---------------------------------------------------------------------------
# Percentile arithmetic
# ---------------------------------------------------------------------------

def test_percentile_is_nearest_rank_and_hand_checkable():
    values = [float(i) for i in range(1, 21)]        # 1 … 20
    # ceil(0.5 × 20) = 10 → the 10th value; ceil(0.9 × 20) = 18 → the 18th.
    assert ue.percentile(values, 0.5) == 10.0
    assert ue.percentile(values, 0.9) == 18.0
    assert ue.percentile(values, 0.0) == 1.0         # rank clamps to 1
    assert ue.percentile(values, 1.0) == 20.0
    # Order of the input must not matter.
    assert ue.percentile(list(reversed(values)), 0.5) == 10.0
    # No interpolation: every answer is a value that was actually observed.
    assert ue.percentile([1.0, 2.0], 0.5) == 1.0


def test_percentile_refuses_an_empty_sample_and_a_bad_quantile():
    with pytest.raises(ValueError):
        ue.percentile([], 0.5)
    with pytest.raises(ValueError):
        ue.percentile([1.0], 1.5)


def test_chart_commentary_percentiles_over_a_hand_checked_fixture():
    """30 commentaries costing 1¢ … 30¢: median is the 15th, p90 the 27th."""
    _seed([{"feature": "chart_commentary", **_cents(i)} for i in range(1, 31)])
    block = _report()["operations"]["chart_commentary"]
    assert block["status"] == "observed"
    assert block["n_units"] == 30
    assert block["n_calls_in_units"] == 30
    assert block["basis"] == ue.BASIS_CALL
    cost = block["cost_usd_per_unit"]
    assert cost["median"] == pytest.approx(0.15)
    assert cost["p90"] == pytest.approx(0.27)
    assert cost["mean"] == pytest.approx(0.155)
    assert cost["min"] == pytest.approx(0.01)
    assert cost["max"] == pytest.approx(0.30)
    assert block["reasons"] == {}


# ---------------------------------------------------------------------------
# The insufficient-sample path — the behaviour that matters most
# ---------------------------------------------------------------------------

def test_a_thin_sample_is_insufficient_with_the_count_never_a_number():
    _seed([
        {"feature": "research_run", "run_id": f"ue-run-{i}", **_cents(10)}
        for i in range(5)
    ])
    report = _report()
    block = report["operations"]["research_run"]
    assert block["status"] == "insufficient_sample"
    assert block["n_units"] == 5
    assert block["cost_usd_per_unit"] == {
        "median": None, "p90": None, "mean": None, "min": None, "max": None,
    }
    # The count and the threshold are both in the reason.
    assert "5 usable unit" in block["reasons"]["median"]
    assert str(ue.MIN_UNITS_FOR_MEDIAN) in block["reasons"]["median"]
    assert block["thresholds"] == {"median": ue.MIN_UNITS_FOR_MEDIAN,
                                   "p90": ue.MIN_UNITS_FOR_P90}

    # It must not leak into the plan projection as a number either.
    for plan in ("free", "pro"):
        term = report["plans"][plan]["terms"]["research_run"]
        assert term["monthly_usd_median"] is None
        assert term["monthly_usd_p90"] is None
        assert "5 usable unit" in term["reason"]
        total = report["plans"][plan]["monthly_variable_cost_usd"]
        assert total["median"] is None and total["p90"] is None
        assert any(u["operation"] == "research_run"
                   for u in report["plans"][plan]["unpriced_terms"])
        # Nothing measured at all: the subtotal is null, not $0.00.
        assert report["plans"][plan]["measured_terms"] == []
        assert report["plans"][plan]["measured_subtotal_usd_median"] is None
    assert report["plans"]["pro"]["gross_margin_pct_at_median"] is None


def test_p90_demands_a_larger_sample_than_the_median():
    _seed([{"feature": "chart_commentary", **_cents(i)} for i in range(1, 21)])
    block = _report()["operations"]["chart_commentary"]
    assert block["status"] == "observed"
    assert block["cost_usd_per_unit"]["median"] == pytest.approx(0.10)   # 10th of 20
    assert block["cost_usd_per_unit"]["p90"] is None
    assert "20 usable unit" in block["reasons"]["p90"]
    assert str(ue.MIN_UNITS_FOR_P90) in block["reasons"]["p90"]
    assert "median" not in block["reasons"]
    # The monthly figure follows: a median projection, no p90 projection.
    free = _report()["plans"]["free"]["terms"]["chart_commentary"]
    assert free["monthly_usd_median"] == pytest.approx(5 * 0.10)
    assert free["monthly_usd_p90"] is None
    assert "20 usable unit" in free["reason"]


# ---------------------------------------------------------------------------
# Unit boundaries, and what is deliberately not counted as one
# ---------------------------------------------------------------------------

def test_a_memo_run_is_a_run_id_and_its_calls_are_summed():
    rows = []
    for i in range(1, 31):
        # Three calls of 1¢, 2¢ and (i)¢ → the run costs (i + 3)¢.
        for cost in (1, 2, i):
            rows.append({"feature": "research_run", "run_id": f"ue-run-{i}", **_cents(cost)})
    _seed(rows)
    block = _report()["operations"]["research_run"]
    assert block["basis"] == ue.BASIS_RUN_ID
    assert block["n_units"] == 30
    assert block["n_calls_in_units"] == 90
    # Runs cost 4¢ … 33¢; the 15th is 18¢ and the 27th is 30¢.
    assert block["cost_usd_per_unit"]["median"] == pytest.approx(0.18)
    assert block["cost_usd_per_unit"]["p90"] == pytest.approx(0.30)


def test_calls_that_belong_to_no_run_are_counted_not_dropped():
    rows = [{"feature": "research_run", "run_id": f"ue-run-{i}", **_cents(5)}
            for i in range(20)]
    rows += [{"feature": "research_run", "run_id": None, **_cents(7)} for _ in range(3)]
    _seed(rows)
    block = _report()["operations"]["research_run"]
    assert block["n_units"] == 20
    assert block["unattributed_calls"]["n"] == 3
    assert "run_id" in block["unattributed_calls"]["reason"]


def test_a_model_missing_from_the_price_table_excludes_its_unit_and_says_so():
    rows = [{"feature": "chart_commentary", **_cents(5)} for _ in range(20)]
    rows += [{"feature": "chart_commentary", "provider": "acme", "model": "acme-ultra-9",
              "tokens_in": 500_000, "tokens_out": 1_000}]
    _seed(rows)
    block = _report()["operations"]["chart_commentary"]
    # Priced at $0 it would have dragged the median down; instead it is out,
    # counted, and the model is named so the price table can be fixed.
    assert block["n_units"] == 20
    assert block["n_units_excluded"] == 1
    assert block["excluded_units"]["unpriced_model"]["n"] == 1
    assert "price table" in block["excluded_units"]["unpriced_model"]["reason"]
    assert block["unpriced_models"] == ["acme/acme-ultra-9"]
    assert block["cost_usd_per_unit"]["median"] == pytest.approx(0.05)


def test_a_successful_call_with_no_token_counts_is_unknown_not_zero():
    rows = [{"feature": "chart_commentary", **_cents(5)} for _ in range(20)]
    rows += [{"feature": "chart_commentary", "tokens_in": 0, "tokens_out": 0}]
    _seed(rows)
    block = _report()["operations"]["chart_commentary"]
    assert block["n_units"] == 20
    assert block["excluded_units"]["missing_token_counts"]["n"] == 1
    assert "unknown rather than zero" in block["excluded_units"]["missing_token_counts"]["reason"]


def test_a_failed_call_is_not_a_delivered_unit():
    rows = [{"feature": "chart_commentary", **_cents(5)} for _ in range(20)]
    rows += [{"feature": "chart_commentary", "success": False, "tokens_in": 0, "tokens_out": 0}
             for _ in range(4)]
    _seed(rows)
    block = _report()["operations"]["chart_commentary"]
    assert block["n_units"] == 20
    assert block["failed_calls"] == 4
    assert block["cost_usd_per_unit"]["median"] == pytest.approx(0.05)


def test_the_scan_cap_counts_the_rows_it_dropped():
    _seed([{"feature": "chart_commentary", **_cents(i)} for i in range(1, 11)])
    report = _report(max_rows=4)
    scan = report["scan"]
    assert scan["rows_in_window"] == 10
    assert scan["rows_scanned"] == 4
    assert scan["rows_dropped_by_cap"] == 6
    assert "6 row(s)" in scan["dropped_reason"]
    # The kept rows cover a shorter period than the window claims, and the
    # report says from when.
    assert scan["scanned_from"] is not None
    assert scan["scanned_from"] > report["window"]["since"]


def test_rows_outside_the_window_are_not_in_the_sample():
    _seed([{"feature": "chart_commentary", **_cents(5)} for _ in range(20)])
    # A one-hour window ending before any seeded row was written.
    old = ue.build_report(window_days=1, now=CLOCK - timedelta(days=90))
    assert old["operations"]["chart_commentary"]["n_units"] == 0
    assert old["scan"]["rows_in_window"] == 0


# ---------------------------------------------------------------------------
# Plan projection
# ---------------------------------------------------------------------------

def _seed_every_metered_operation() -> None:
    """30 units each, all costing 10¢, across the three metered operations."""
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append({"feature": "research_run", "run_id": f"ue-run-{i}", **_cents(10)})
        rows.append({"feature": "pm_chat", **_cents(10)})
        rows.append({"feature": "chart_commentary", **_cents(10)})
    _seed(rows)


def test_plan_allowances_are_priced_from_the_observed_cost():
    _seed_every_metered_operation()
    report = _report()
    free, pro = report["plans"]["free"], report["plans"]["pro"]

    # Free: 1 research_run + 10 pm_chat + 5 chart_commentary, all at 10¢.
    assert free["terms"]["research_run"]["limit"] == 1
    assert free["terms"]["pm_chat"]["limit"] == 10
    assert free["terms"]["chart_commentary"]["limit"] == 5
    assert free["monthly_variable_cost_usd"]["median"] == pytest.approx(1.6)
    assert free["monthly_variable_cost_usd"]["p90"] == pytest.approx(1.6)
    assert free["price_usd_per_month"] == 0.0
    assert free["threshold_usd"] == 1.50
    assert free["verdict"]["under_threshold_at_median"] is False   # $1.60 > $1.50

    # Pro: 20 + 300 + 100 at 10¢ = $42.00 against a $29.99 price.
    assert pro["monthly_variable_cost_usd"]["median"] == pytest.approx(42.0)
    assert pro["price_usd_per_month"] == ue.PRO_PRICE_USD
    assert pro["verdict"]["under_threshold_at_median"] is False
    assert pro["gross_margin_pct_at_median"] < 0
    assert pro["unpriced_terms"] == []


def test_a_per_call_operation_is_flagged_as_a_floor_not_a_ceiling():
    """pm_chat has no per-turn id in the log; the projection must say so."""
    _seed_every_metered_operation()
    report = _report()
    assert report["operations"]["pm_chat"]["understates_unit"] is True
    for plan in ("free", "pro"):
        term = report["plans"][plan]["terms"]["pm_chat"]
        assert term["floor_only"] is True
        assert "classify_intent" in term["floor_reason"]
        assert report["plans"][plan]["verdict"]["is_a_floor"] is True
        assert any("lower bound" in note for note in report["plans"][plan]["notes"])


def test_an_industry_report_is_costed_but_carries_no_per_user_allowance():
    _seed([{"feature": "industry_report", "run_id": f"ue-ind-{i}", **_cents(i + 1)}
           for i in range(30)])
    report = _report()
    block = report["operations"]["industry_report"]
    assert block["status"] == "observed"
    assert block["plan_feature"] is None
    assert block["units_per_30d"] == pytest.approx(30.0)
    # No plan term: a report is generated once for everybody, not per user.
    assert "industry_report" not in report["plans"]["pro"]["terms"]
    assert "industry_report" not in report["plans"]["free"]["terms"]


def test_features_this_report_does_not_price_are_named_with_a_reason():
    names = {t["feature"]: t["reason"] for t in _report()["terms_not_covered"]}
    # Cost-bearing but unmetered — no allowance to multiply.
    for name in ("dcf", "comps", "portfolio"):
        assert "not metered" in names[name]
    # Metered but not cost-bearing — a read of what a research_run paid for.
    assert "not cost-bearing" in names["memo_view"]


def test_an_unlimited_allowance_is_reported_as_unpriceable_not_free(monkeypatch):
    """An allowance with no ceiling has no monthly cost to state. Reported
    as such, with the reason — and the plan total goes null with it, because
    "unlimited" is not "$0"."""
    monkeypatch.setattr(settings, "entitlement_overrides_json",
                        '{"research_run": {"pro": null}}')
    _seed_every_metered_operation()
    pro = _report()["plans"]["pro"]
    term = pro["terms"]["research_run"]
    assert term["allowed"] is True
    assert term["limit"] is None
    assert term["monthly_usd_median"] is None
    assert "unlimited" in term["reason"]
    assert pro["monthly_variable_cost_usd"]["median"] is None
    assert pro["gross_margin_pct_at_median"] is None
    # What IS measured is still reported: pm_chat + chart_commentary.
    assert pro["measured_subtotal_usd_median"] == pytest.approx(300 * 0.10 + 100 * 0.10)
    assert sorted(pro["measured_terms"]) == ["chart_commentary", "pm_chat"]


def test_a_feature_a_plan_does_not_have_is_not_a_cost():
    monkeypatch_value = '{"chart_commentary": {"free": 0}}'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "entitlement_overrides_json", monkeypatch_value)
        _seed_every_metered_operation()
        free = _report()["plans"]["free"]
        term = free["terms"]["chart_commentary"]
        assert term["allowed"] is False
        assert "not available" in term["reason"]
        assert term["monthly_usd_median"] is None
        # Denied, not unmeasured: it does not block the plan total.
        assert free["monthly_variable_cost_usd"]["median"] == pytest.approx(1.1)


def test_build_report_rejects_a_nonsense_window():
    with pytest.raises(ValueError):
        ue.build_report(window_days=0)
    with pytest.raises(ValueError):
        ue.build_report(max_rows=0)


# ---------------------------------------------------------------------------
# It must never spend money
# ---------------------------------------------------------------------------

def test_the_report_makes_no_llm_or_provider_call(monkeypatch):
    from app.agents import llm as llm_mod
    from app.services import data_service

    def boom(*_a: Any, **_k: Any):
        raise AssertionError("unit_economics called a provider or a model")

    for name in ("chat_json", "chat_text", "gemini_chat_text",
                 "_openai_client", "_anthropic_client", "_gemini_client"):
        monkeypatch.setattr(llm_mod, name, boom)
    monkeypatch.setattr(data_service, "get_data_service", boom)

    _seed_every_metered_operation()
    with SessionLocal() as db:
        before = db.query(LLMCallLog).count()
    report = _report()
    assert report["operations"]["research_run"]["status"] == "observed"
    with SessionLocal() as db:
        # A real model call would have written a row of its own.
        assert db.query(LLMCallLog).count() == before


# ---------------------------------------------------------------------------
# The admin route
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    # Bare TestClient (no lifespan): the startup event runs a full universe
    # seed, and this endpoint needs none of it.
    return TestClient(app)


@pytest.fixture()
def admin(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def test_route_requires_the_admin_token(client, admin):
    assert client.get("/api/admin/unit-economics").status_code == 401
    assert client.get("/api/admin/unit-economics", headers=admin).status_code == 200


def test_route_returns_the_report(client, admin):
    body = client.get("/api/admin/unit-economics?window_days=7", headers=admin).json()
    assert body["window"]["days"] == 7
    assert set(body["operations"]) == {op.key for op in ue.OPERATIONS}
    assert set(body["plans"]) == {"free", "pro"}
    assert body["scan"]["max_rows_scanned"] == ue.MAX_ROWS_SCANNED
    for block in body["operations"].values():
        # With no seeded traffic in a recent window every figure must be
        # absent with a reason — never a confident zero.
        if block["status"] == "insufficient_sample":
            assert block["cost_usd_per_unit"]["median"] is None
            assert block["reasons"]["median"]


def test_route_bounds_the_window(client, admin):
    assert client.get("/api/admin/unit-economics?window_days=0", headers=admin).status_code == 422
    assert client.get("/api/admin/unit-economics?window_days=91", headers=admin).status_code == 422
