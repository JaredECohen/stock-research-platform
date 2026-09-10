"""FEAT-001 — the metric catalog is closed, honest and pinned.

- every id is unique and every input is a line item `history_service`
  actually ingests (a metric that reads a line nobody writes would render
  as missing forever);
- unit / kind / family / reason vocabularies are closed sets;
- every spec explains itself (`formula_text`, `provenance` non-empty);
- the metrics the feature brief requires are all present;
- `CATALOG_VERSION` + a digest of the arithmetic are pinned together, so
  editing a formula without bumping the version fails here rather than
  serving stale cached commentary;
- `compute` returns a reason for every None and flags exactly the
  documented fallbacks as `estimated`.
"""
from __future__ import annotations

import math

import pytest

from app.finance import ratios as R
from app.schemas.fundamentals import CatalogOut, MetricSpecOut
from app.services import fundamentals_catalog as C
from app.services import history_service as hs

WHITELIST = set(hs._INCOME_LINES) | set(hs._BALANCE_LINES) | set(hs._CASH_LINES)

REQUIRED_IDS = {
    "revenue", "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin",
    "net_income", "eps_diluted", "operating_cash_flow", "capex", "free_cash_flow",
    "fcf_margin", "fcf_after_sbc", "roic", "net_debt", "shares_diluted",
    "pe_ttm", "ev_ebitda", "ev_revenue", "p_fcf", "fcf_yield",
}

# Bump BOTH when a formula, input list, fallback or unit changes. The
# digest covers exactly the fields that change a computed value.
PINNED_VERSION = "2026.09.1"
PINNED_DIGEST = "564bd05966f53f57"


def _close(a, b):
    return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_ids_unique_and_required_metrics_present():
    ids = C.metric_ids()
    assert len(ids) == len(set(ids))
    assert REQUIRED_IDS <= set(ids), REQUIRED_IDS - set(ids)


@pytest.mark.parametrize("metric_id", C.metric_ids())
def test_inputs_are_ingested_line_items(metric_id):
    spec = C.get(metric_id)
    assert spec.inputs, metric_id
    assert set(spec.inputs) <= WHITELIST, set(spec.inputs) - WHITELIST


@pytest.mark.parametrize("metric_id", C.metric_ids())
def test_enums_closed_and_text_present(metric_id):
    spec = C.get(metric_id)
    assert spec.unit_type in C.UNIT_TYPES
    assert spec.kind in C.METRIC_KINDS
    assert spec.family in C.FAMILIES
    assert spec.frequency == "annual"
    assert spec.formula_text.strip()
    assert spec.provenance.strip()
    # Market metrics are the only ones that need a price; nothing else may.
    assert spec.requires_price == (spec.kind == "market")


def test_share_line_is_ingested_and_added_for_market_metrics():
    assert C.SHARES_LINE in WHITELIST
    assert C.SHARES_LINE not in C.required_line_items(["revenue", "gross_margin"])
    assert C.SHARES_LINE in C.required_line_items(["pe_ttm"])
    # Stable, first-appearance order — the SQL IN list is deterministic.
    assert C.required_line_items(["gross_margin", "revenue"]) == ("gross_profit", "revenue")


def test_reason_vocabulary_matches_schema():
    from typing import get_args

    from app.schemas.fundamentals import PointReason, UnitType
    assert set(get_args(PointReason)) == set(C.POINT_REASONS)
    assert set(get_args(UnitType)) == set(C.UNIT_TYPES)


def test_expectations_ledger_reserved_not_implemented():
    assert set(C.RESERVED_SOURCE_KINDS) == {"consensus", "guidance", "price_implied", "our_forecast"}
    assert not (set(C.RESERVED_SOURCE_KINDS) & set(C.METRIC_KINDS))
    assert "not" in C.__doc__ and "v1" in C.__doc__


def test_catalog_version_and_digest_snapshot():
    assert C.CATALOG_VERSION == PINNED_VERSION
    assert C.catalog_digest() == PINNED_DIGEST, (
        "catalog arithmetic changed: bump CATALOG_VERSION and re-pin PINNED_DIGEST together"
    )


def test_catalog_entries_validate_against_the_wire_schema():
    entries = [MetricSpecOut(**e) for e in C.catalog_entries()]
    assert [e.id for e in entries] == list(C.metric_ids())
    out = CatalogOut(catalog_version=C.CATALOG_VERSION, as_of="2026-09-09T00:00:00",
                     families=list(C.FAMILIES), metrics=entries,
                     reserved_source_kinds=list(C.RESERVED_SOURCE_KINDS))
    assert out.frequency == "annual"


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        C.get("bogus")
    with pytest.raises(ValueError):
        C.required_line_items(["revenue", "bogus"])


# ---------------------------------------------------------------------------
# Arithmetic — every None has a reason, estimated only on fallbacks
# ---------------------------------------------------------------------------

CUR = {
    "revenue": 1000.0, "gross_profit": 600.0, "operating_income": 250.0, "net_income": 180.0,
    "ebitda": 300.0, "depreciation_and_amortization": 50.0, "eps_diluted": 1.8,
    "cash_from_operations": 320.0, "capex": -70.0, "free_cash_flow": 250.0,
    "stock_based_compensation": 40.0, "pretax_income": 240.0, "tax_expense": 60.0,
    "total_debt": 400.0, "short_term_debt": 80.0, "long_term_debt": 320.0,
    "shareholders_equity": 600.0, "cash_and_equivalents": 150.0, "short_term_investments": 50.0,
    "weighted_avg_shares_diluted": 100.0,
}
MKT = C.MarketContext(price=50.0, shares=100.0)   # market cap 5000


def test_reported_lines_pass_through_and_capex_is_absolute():
    assert C.compute("revenue", CUR).value == 1000.0
    assert C.compute("capex", CUR).value == 70.0
    assert C.compute("revenue", {}) == C.Computed(None, "missing_line")


def test_margins_and_reasons():
    assert _close(C.compute("gross_margin", CUR).value, 0.6)
    assert _close(C.compute("net_margin", CUR).value, 0.18)
    assert C.compute("gross_margin", {**CUR, "revenue": 0.0}).reason == "denominator_nonpositive"
    assert C.compute("gross_margin", {**CUR, "revenue": -5.0}).reason == "denominator_nonpositive"
    assert C.compute("operating_margin", {**CUR, "operating_income": None}).reason == "missing_line"


def test_growth_needs_a_positive_prior():
    assert _close(C.compute("revenue_growth_yoy", CUR, {"revenue": 800.0}).value, 0.25)
    assert C.compute("revenue_growth_yoy", CUR, {"revenue": 0.0}).reason == "base_nonpositive"
    assert C.compute("revenue_growth_yoy", CUR, {"revenue": -10.0}).reason == "base_nonpositive"
    assert C.compute("revenue_growth_yoy", CUR, None).reason == "missing_line"
    assert C.compute("revenue_growth_yoy", CUR, {}).reason == "missing_line"


def test_fcf_after_sbc_arithmetic_and_sbc_never_defaults_to_zero():
    out = C.compute("fcf_after_sbc", CUR)
    assert _close(out.value, 250.0 - 40.0) and not out.estimated
    assert C.compute("fcf_after_sbc", {**CUR, "stock_based_compensation": None}).reason == "missing_line"
    # FCF fallback carries through as estimated: CFO − |capex| − SBC.
    derived = C.compute("fcf_after_sbc", {**CUR, "free_cash_flow": None})
    assert _close(derived.value, 320.0 - 70.0 - 40.0) and derived.estimated


def test_free_cash_flow_fallback_is_estimated():
    assert C.compute("free_cash_flow", CUR) == C.Computed(250.0)
    d = C.compute("free_cash_flow", {**CUR, "free_cash_flow": None})
    assert _close(d.value, 250.0) and d.estimated
    assert C.compute("free_cash_flow", {"cash_from_operations": 1.0}).reason == "missing_line"


def test_ebitda_fallback_matches_ratios_definition():
    assert C.compute("ebitda", CUR) == C.Computed(300.0)
    d = C.compute("ebitda", {**CUR, "ebitda": None})
    assert _close(d.value, R.ebitda(CUR, CUR)) and d.estimated
    assert C.compute("ebitda", {"depreciation_and_amortization": 5.0}).reason == "missing_line"


def test_net_debt_requires_cash_and_flags_the_split_fallback():
    assert _close(C.compute("net_debt", CUR).value, R.net_debt(CUR))
    assert C.compute("net_debt", {**CUR, "cash_and_equivalents": None}).reason == "missing_line"
    split = C.compute("net_debt", {**CUR, "total_debt": None})
    assert _close(split.value, 400.0 - 200.0) and split.estimated
    no_debt = C.compute("net_debt", {"cash_and_equivalents": 10.0})
    assert no_debt.reason == "missing_line"


def test_roic_delegates_to_ratios_with_provenance(monkeypatch):
    real = C.compute("roic", CUR)
    assert _close(real.value, R.roic(CUR, CUR)) and not real.estimated

    statutory = C.compute("roic", {**CUR, "tax_expense": None})
    assert _close(statutory.value, 250.0 * (1 - R.STATUTORY_TAX_RATE_FALLBACK) / 1000.0)
    assert statutory.estimated

    calls: list[tuple] = []

    def fake(income, balance, tax_rate=None):
        calls.append((dict(income), dict(balance)))
        return None, R.ROIC_UNKNOWN_LOSS_MAKER

    monkeypatch.setattr(R, "roic_with_provenance", fake)
    out = C.compute("roic", CUR)
    assert calls and out == C.Computed(None, "missing_line")


def test_roic_reason_mapping_for_every_none_label():
    assert C.compute("roic", {**CUR, "operating_income": None}).reason == "missing_line"
    zero_capital = {**CUR, "total_debt": 0.0, "short_term_debt": 0.0, "long_term_debt": 0.0,
                    "shareholders_equity": 0.0}
    assert C.compute("roic", zero_capital).reason == "denominator_nonpositive"
    loss = {**CUR, "operating_income": -50.0, "pretax_income": -60.0, "tax_expense": -12.0}
    assert C.compute("roic", loss).reason == "missing_line"


def test_market_metrics_need_price_then_shares():
    assert C.compute("pe_ttm", CUR, market=None).reason == "no_price"
    assert C.compute("pe_ttm", CUR, market=C.MarketContext(price=None, shares=100.0)).reason == "no_price"
    assert C.compute("pe_ttm", CUR, market=C.MarketContext(price=50.0, shares=None)).reason == "no_shares"
    assert C.compute("pe_ttm", CUR, market=C.MarketContext(price=50.0, shares=0.0)).reason == "no_shares"


def test_market_multiples_match_ratios_definitions():
    mc = MKT.market_cap
    assert mc == 5000.0
    assert _close(C.compute("pe_ttm", CUR, market=MKT).value, R.pe_ratio(mc, CUR))
    assert _close(C.compute("ev_ebitda", CUR, market=MKT).value, R.ev_ebitda(mc, CUR, CUR, CUR))
    assert _close(C.compute("ev_revenue", CUR, market=MKT).value, R.ev_revenue(mc, CUR, CUR))
    assert _close(C.compute("p_fcf", CUR, market=MKT).value, R.p_fcf(mc, CUR))
    assert _close(C.compute("fcf_yield", CUR, market=MKT).value, R.fcf_yield(mc, CUR))


def test_market_multiples_refuse_nonpositive_denominators():
    assert C.compute("pe_ttm", {**CUR, "net_income": -1.0}, market=MKT).reason == "denominator_nonpositive"
    assert C.compute("p_fcf", {**CUR, "free_cash_flow": 0.0}, market=MKT).reason == "denominator_nonpositive"
    assert C.compute("ev_ebitda", {**CUR, "ebitda": -5.0, "operating_income": -60.0},
                     market=MKT).reason == "denominator_nonpositive"


def test_estimated_shares_propagate_to_market_metrics():
    est = C.MarketContext(price=50.0, shares=100.0, shares_estimated=True)
    out = C.compute("pe_ttm", CUR, market=est)
    assert _close(out.value, 5000.0 / 180.0) and out.estimated
    # fcf_yield with an estimated *numerator* (derived FCF) is estimated too.
    y = C.compute("fcf_yield", {**CUR, "free_cash_flow": None}, market=MKT)
    assert y.estimated and _close(y.value, 250.0 / 5000.0)
