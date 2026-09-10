"""Scorecard feature engine (fs-v1): point-in-time snapshot rules and every
formula against hand-computed values on a synthetic company.

Pure module — no DB, no providers. Rows use the persistence contract
``(statement, line_item, period, period_end, fiscal_year, fiscal_quarter,
value, available_at)``.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from statistics import stdev

import pytest

from app.finance import scorecard_features as F
from app.finance import scorecard_spec as S

AS_OF = date(2025, 3, 1)
PRICE_CTX = {"price": 20.0, "price_date": date(2025, 2, 28), "shares_fallback": 55.0}

# Fiscal-year statements. FY2024 is the "TTM" row; FY2023 the prior; FY2021
# is three years back (revenue 512 → 1000 is a clean 25% CAGR).
FY = {
    2024: {
        "income": {
            "revenue": 1000.0, "gross_profit": 600.0, "sga": 200.0, "r_and_d": 100.0,
            "operating_income": 300.0, "ebit": 290.0, "ebitda": 350.0, "net_income": 200.0,
            "eps_diluted": 4.0, "weighted_avg_shares_diluted": 50.0, "interest_expense": 20.0,
            "pretax_income": 250.0, "tax_expense": 50.0,
        },
        "balance": {
            "total_assets": 2000.0, "shareholders_equity": 800.0, "cash_and_equivalents": 100.0,
            "short_term_investments": 50.0, "total_debt": 400.0, "goodwill": 300.0,
            "current_assets": 500.0, "current_liabilities": 250.0,
        },
        "cash": {
            "cash_from_operations": 260.0, "capex": -80.0, "free_cash_flow": 180.0,
            "depreciation_and_amortization": 50.0, "dividends_paid": -40.0,
            "share_repurchases": -60.0, "stock_based_compensation": 30.0,
        },
    },
    2023: {
        "income": {
            "revenue": 800.0, "gross_profit": 440.0, "operating_income": 200.0,
            "net_income": 160.0, "eps_diluted": 3.2, "weighted_avg_shares_diluted": 52.0,
        },
        "balance": {"total_assets": 1800.0},
        "cash": {},
    },
    2022: {"income": {"revenue": 700.0, "gross_profit": 350.0}, "balance": {}, "cash": {}},
    2021: {"income": {"revenue": 512.0, "gross_profit": 256.0}, "balance": {}, "cash": {}},
}


def _rows(fy_data=FY, *, available=None, quarter=None):
    """Long-format rows; every FY becomes available on Feb 15 of the next
    calendar year unless ``available`` overrides it."""
    out = []
    for fy, stmts in fy_data.items():
        avail = available(fy) if available else date(fy + 1, 2, 15)
        for statement, lines in stmts.items():
            for line, value in lines.items():
                out.append((statement, line, f"FY{fy}", date(fy, 12, 31), fy, quarter, value, avail))
    return out


def _snapshot(as_of=AS_OF, **kw):
    return F.pit_snapshot(_rows(**kw), as_of)


def _features(sector="Technology", price_ctx=PRICE_CTX, snapshot=None):
    return F.compute_features_detailed(snapshot or _snapshot(), price_ctx, sector)


def _close(a, b):
    return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# pit_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_orders_annual_points_newest_first():
    snap = _snapshot()
    assert [p.fiscal_year for p in snap.points] == [2024, 2023, 2022, 2021]
    assert snap.latest.fiscal_year == 2024
    assert snap.prior.fiscal_year == 2023
    assert snap.years_back(3).fiscal_year == 2021
    assert snap.latest.period_end == date(2024, 12, 31)
    assert snap.latest.available_at == date(2025, 2, 15)
    assert snap.notes["annual_points"] == 4
    assert snap.notes["pit_excluded"] == 0


def test_snapshot_excludes_rows_not_yet_available():
    # As of Jan 2025 the FY2024 10-K (available Feb 15) is not knowable.
    snap = F.pit_snapshot(_rows(), date(2025, 1, 31))
    assert snap.latest.fiscal_year == 2023
    assert snap.notes["pit_excluded"] == sum(len(v) for v in FY[2024].values())


def test_snapshot_availability_is_inclusive_on_the_as_of_date():
    snap = F.pit_snapshot(_rows(), date(2025, 2, 15))
    assert snap.latest.fiscal_year == 2024


def test_snapshot_accepts_datetime_and_iso_string_availability():
    rows = [
        ("income", "revenue", "FY2024", date(2024, 12, 31), 2024, None, 1.0, datetime(2025, 2, 15, 16, 5)),
        ("income", "net_income", "FY2024", date(2024, 12, 31), 2024, None, 2.0, "2025-02-15"),
        ("income", "sga", "FY2024", date(2024, 12, 31), 2024, None, 3.0, "2025-02-15T09:00:00"),
    ]
    snap = F.pit_snapshot(rows, date(2025, 2, 15))
    assert snap.latest.income == {"revenue": 1.0, "net_income": 2.0, "sga": 3.0}


def test_snapshot_rows_without_availability_are_excluded_and_counted():
    rows = [("income", "revenue", "FY2024", date(2024, 12, 31), 2024, None, 1.0, None)]
    snap = F.pit_snapshot(rows, AS_OF)
    assert snap.points == ()
    assert snap.notes["missing_available_at"] == 1


def test_snapshot_ignores_quarterly_rows_in_fs_v1():
    snap = F.pit_snapshot(_rows(quarter=4), AS_OF)
    assert snap.points == ()
    assert snap.notes["quarterly_ignored"] == snap.notes["rows_seen"]


def test_snapshot_drops_null_and_non_finite_values_never_zeroing_them():
    rows = [
        ("income", "revenue", "FY2024", date(2024, 12, 31), 2024, None, None, date(2025, 2, 15)),
        ("income", "net_income", "FY2024", date(2024, 12, 31), 2024, None, float("nan"), date(2025, 2, 15)),
        ("income", "sga", "FY2024", date(2024, 12, 31), 2024, None, 5.0, date(2025, 2, 15)),
    ]
    snap = F.pit_snapshot(rows, AS_OF)
    assert snap.latest.income == {"sga": 5.0}
    assert snap.notes["null_values_dropped"] == 2


def test_snapshot_derives_fiscal_year_from_period_end_and_counts_unusable():
    rows = [
        ("income", "revenue", "2024", date(2024, 12, 31), None, None, 1.0, date(2025, 2, 15)),
        ("income", "revenue", "2023", None, None, None, 1.0, date(2024, 2, 15)),   # no year at all
        ("bogus", "revenue", "FY2024", date(2024, 12, 31), 2024, None, 1.0, date(2025, 2, 15)),
    ]
    snap = F.pit_snapshot(rows, AS_OF)
    assert [p.fiscal_year for p in snap.points] == [2024]
    assert snap.notes["rows_unusable"] == 2


def test_snapshot_duplicate_rows_last_wins_and_are_counted():
    rows = [
        ("income", "revenue", "FY2024", date(2024, 12, 31), 2024, None, 1.0, date(2025, 2, 15)),
        ("income", "revenue", "2024", date(2024, 12, 31), 2024, None, 2.0, date(2025, 2, 20)),
    ]
    snap = F.pit_snapshot(rows, AS_OF)
    assert snap.latest.income["revenue"] == 2.0
    assert snap.latest.available_at == date(2025, 2, 20)
    assert snap.notes["duplicate_rows"] == 1


def test_years_back_requires_the_adjacent_fiscal_year():
    data = {2024: FY[2024], 2022: FY[2022]}
    snap = F.pit_snapshot(_rows(data), AS_OF)
    assert snap.prior is None                      # 2023 missing: no "one-year" growth
    assert snap.years_back(2).fiscal_year == 2022


def test_inputs_hash_is_deterministic_and_input_sensitive():
    snap = _snapshot()
    h1 = F.inputs_hash(snap, PRICE_CTX)
    assert h1 == F.inputs_hash(_snapshot(), dict(PRICE_CTX))
    assert h1 != F.inputs_hash(snap, {**PRICE_CTX, "price": 21.0})
    assert h1 != F.inputs_hash(F.pit_snapshot(_rows(), date(2025, 1, 31)), PRICE_CTX)
    assert len(h1) == 64


# ---------------------------------------------------------------------------
# Formulas — hand-computed on the synthetic company
# ---------------------------------------------------------------------------

EXPECTED = {
    # market cap = 20 * 50 = 1000; net debt = 400 - 100 - 50 = 250; EV = 1250
    "earnings_yield": 200 / 1000,
    "fcf_yield": 180 / 1000,
    "ebitda_ev_yield": 350 / 1250,
    "sales_ev_yield": 1000 / 1250,
    # tax rate 50/250 = 0.2 → NOPAT 240; invested = 400 + 800
    "roic": 240 / 1200,
    "roa": 200 / 1900,                       # avg assets (2000 + 1800) / 2
    "roe": 200 / 800,
    "gross_margin_stability": -stdev([0.6, 0.55, 0.5, 0.5]),
    "revenue_growth_1y": 1000 / 800 - 1,
    "revenue_cagr_3y": (1000 / 512) ** (1 / 3) - 1,
    "operating_income_growth_1y": 300 / 200 - 1,
    "eps_growth_1y": 4.0 / 3.2 - 1,
    "revenue_growth_accel": (1000 / 800 - 1) - (800 / 700 - 1),
    "gross_margin": 0.6,
    "operating_margin": 0.3,
    "fcf_margin": 0.18,
    "operating_margin_change_1y": 0.3 - 200 / 800,
    "asset_turnover": 1000 / 1900,
    "opex_ratio": 300 / 1000,
    "capex_intensity": 80 / 1000,           # abs() of the negative outflow
    "net_debt_to_ebitda": 250 / 350,
    "debt_to_equity": 400 / 800,
    "interest_coverage": 290 / 20,
    "current_ratio": 500 / 250,
    "shareholder_yield": (40 + 60) / 1000,  # abs() of both outflows
    "net_share_change_1y": 50 / 52 - 1,
    "sbc_to_revenue": 30 / 1000,
    "goodwill_to_assets": 300 / 2000,
    "accruals_ratio": (200 - 260) / 1900,
    "cash_conversion": 260 / 200,
    "fcf_to_net_income": 180 / 200,
}


def test_expected_table_covers_every_spec_feature():
    assert set(EXPECTED) == set(S.FEATURE_NAMES)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_formula_hand_computed(name):
    res = _features()
    assert _close(res.values[name], EXPECTED[name]), (name, res.values[name], res.reasons.get(name))


def test_full_coverage_has_no_reasons_and_records_context():
    res = _features()
    assert res.reasons == {}
    assert res.applicable == frozenset(S.FEATURE_NAMES)
    ctx = res.context
    assert ctx["market_cap"] == 1000.0
    assert ctx["enterprise_value"] == 1250.0
    assert ctx["shares"] == 50.0 and ctx["shares_source"] == "income_statement"
    assert ctx["sector"] == S.SECTOR_INFORMATION_TECHNOLOGY and ctx["sector_raw"] == "Technology"
    assert ctx["fiscal_year"] == 2024 and ctx["prior_fiscal_year"] == 2023
    assert ctx["available_at"] == "2025-02-15" and ctx["price_date"] == "2025-02-28"
    assert ctx["roic_provenance"] == "effective_rate"
    assert ctx["avg_assets_basis"] == "average_two_years"
    assert ctx["flags"] == []
    assert ctx["pit_notes"]["annual_points"] == 4


def test_compute_features_plain_dict_matches_detailed_values():
    plain = F.compute_features(_snapshot(), PRICE_CTX, "Technology")
    assert plain == _features().values
    assert set(plain) == set(S.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Null rules: missing evidence is None with a reason, never zero
# ---------------------------------------------------------------------------


def _without(fy: int, statement: str, *lines: str):
    data = {y: {s: dict(v) for s, v in stmts.items()} for y, stmts in FY.items()}
    for line in lines:
        data[fy][statement].pop(line, None)
    return data


def test_no_price_nulls_market_cap_features_with_reason():
    res = _features(price_ctx={"price": None, "price_date": None, "shares_fallback": 55.0})
    for name in ("earnings_yield", "fcf_yield", "ebitda_ev_yield", "sales_ev_yield", "shareholder_yield"):
        assert res.values[name] is None
        assert res.reasons[name] == "missing:price"
    assert res.values["roe"] == pytest.approx(0.25)  # price-free features unaffected
    assert res.context["market_cap"] is None


def test_shares_fall_back_to_company_shares_outstanding_and_are_tagged():
    snap = F.pit_snapshot(_rows(_without(2024, "income", "weighted_avg_shares_diluted")), AS_OF)
    res = _features(snapshot=snap)
    assert res.context["shares_source"] == "company_shares_outstanding"
    assert res.context["market_cap"] == 20.0 * 55.0
    assert _close(res.values["earnings_yield"], 200 / 1100)
    assert res.values["net_share_change_1y"] is None
    assert res.reasons["net_share_change_1y"] == "missing:weighted_avg_shares_diluted"


def test_no_shares_anywhere_is_missing_not_zero():
    snap = F.pit_snapshot(_rows(_without(2024, "income", "weighted_avg_shares_diluted")), AS_OF)
    res = _features(snapshot=snap, price_ctx={"price": 20.0, "price_date": None, "shares_fallback": None})
    assert res.values["earnings_yield"] is None
    assert res.reasons["earnings_yield"] == "missing:shares"
    assert res.context["shares_source"] == "none"


def test_negative_equity_nulls_roe_and_debt_to_equity():
    data = _without(2024, "balance")
    data[2024]["balance"]["shareholders_equity"] = -50.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["roe"] is None
    assert res.reasons["roe"] == "denominator:shareholders_equity<=0"
    assert res.values["debt_to_equity"] is None
    assert res.reasons["debt_to_equity"] == "denominator:shareholders_equity<=0"


def test_loss_maker_conversion_ratios_are_null():
    data = _without(2024, "income")
    data[2024]["income"]["net_income"] = -10.0
    data[2024]["income"]["pretax_income"] = -12.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["cash_conversion"] is None
    assert res.reasons["cash_conversion"] == "denominator:net_income<=0"
    assert res.values["fcf_to_net_income"] is None
    # earnings yield is a legitimate negative, not a null
    assert _close(res.values["earnings_yield"], -10 / 1000)
    # ROIC still resolves: operating income is positive, so the shared
    # rule falls back to the statutory rate rather than a fabricated one.
    assert res.context["roic_provenance"] == "statutory_fallback"
    assert _close(res.values["roic"], 300 * 0.79 / 1200)


def test_roic_null_for_loss_maker_without_credible_rate_carries_provenance():
    data = _without(2024, "income", "pretax_income", "tax_expense")
    data[2024]["income"]["operating_income"] = -30.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["roic"] is None
    assert res.reasons["roic"] == "missing:unknown_loss_maker"
    assert res.context["roic_provenance"] == "unknown_loss_maker"


def test_conversion_ratios_are_clipped_to_plus_minus_three():
    data = _without(2024, "income")
    data[2024]["income"]["net_income"] = 1.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["cash_conversion"] == 3.0
    assert res.values["fcf_to_net_income"] == 3.0


def test_interest_coverage_is_capped_and_null_on_zero_interest():
    data = _without(2024, "income")
    data[2024]["income"]["interest_expense"] = 0.1
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["interest_coverage"] == 50.0
    data[2024]["income"]["interest_expense"] = 0.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["interest_coverage"] is None
    assert res.reasons["interest_coverage"] == "denominator:interest_expense=0"


def test_interest_coverage_falls_back_to_operating_income_when_ebit_missing():
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "income", "ebit")), AS_OF))
    assert _close(res.values["interest_coverage"], 300 / 20)
    assert "ebit_from_operating_income" in res.context["flags"]


def test_ebitda_derived_from_operating_income_plus_da_only_when_both_present():
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "income", "ebitda")), AS_OF))
    assert _close(res.values["ebitda_ev_yield"], 350 / 1250)   # 300 + 50
    assert "ebitda_from_operating_income_plus_da" in res.context["flags"]
    data = _without(2024, "income", "ebitda")
    data[2024]["cash"].pop("depreciation_and_amortization")
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["ebitda_ev_yield"] is None
    assert res.reasons["ebitda_ev_yield"] == "missing:ebitda"
    assert res.reasons["net_debt_to_ebitda"] == "missing:ebitda"


def test_negative_ebitda_nulls_net_debt_to_ebitda_only():
    data = _without(2024, "income")
    data[2024]["income"]["ebitda"] = -5.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["net_debt_to_ebitda"] is None
    assert res.reasons["net_debt_to_ebitda"] == "denominator:ebitda<=0"
    assert _close(res.values["ebitda_ev_yield"], -5 / 1250)


def test_total_debt_falls_back_to_short_plus_long_and_flags_missing_partner():
    data = _without(2024, "balance", "total_debt")
    data[2024]["balance"]["long_term_debt"] = 400.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert _close(res.values["debt_to_equity"], 0.5)
    assert "short_term_debt_not_reported" in res.context["flags"]
    data[2024]["balance"].pop("long_term_debt")
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["debt_to_equity"] is None
    assert res.reasons["debt_to_equity"] == "missing:total_debt"
    assert res.reasons["ebitda_ev_yield"] == "missing:total_debt"


def test_missing_prior_year_nulls_growth_and_change_features():
    snap = F.pit_snapshot(_rows({2024: FY[2024], 2022: FY[2022], 2021: FY[2021]}), AS_OF)
    res = _features(snapshot=snap)
    for name in ("revenue_growth_1y", "operating_income_growth_1y", "eps_growth_1y",
                 "revenue_growth_accel", "operating_margin_change_1y", "net_share_change_1y"):
        assert res.values[name] is None, name
        assert res.reasons[name] == "no_prior_period:fy=2023", name
    # avg assets falls back to the single balance sheet, and says so
    assert _close(res.values["roa"], 200 / 2000)
    assert res.context["avg_assets_basis"] == "latest_only"
    assert "avg_assets_latest_only" in res.context["flags"]
    # 3y CAGR only needs FY2021
    assert _close(res.values["revenue_cagr_3y"], EXPECTED["revenue_cagr_3y"])


def test_negative_prior_base_nulls_growth_rather_than_inventing_a_sign():
    data = _without(2023, "income")
    data[2023]["income"]["operating_income"] = -50.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert res.values["operating_income_growth_1y"] is None
    assert res.reasons["operating_income_growth_1y"] == "denominator:operating_income_prior<=0"


def test_eps_growth_falls_back_to_net_income_only_when_eps_absent_both_years():
    data = _without(2024, "income", "eps_diluted")
    data = {y: {s: dict(v) for s, v in stmts.items()} for y, stmts in data.items()}
    data[2023]["income"].pop("eps_diluted")
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert _close(res.values["eps_growth_1y"], 200 / 160 - 1)
    assert "eps_growth_from_net_income" in res.context["flags"]
    # one side only: mixing per-share with aggregate is refused
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "income", "eps_diluted")), AS_OF))
    assert res.values["eps_growth_1y"] is None
    assert res.reasons["eps_growth_1y"] == "missing:eps_diluted"


def test_gross_margin_stability_needs_three_annual_points():
    snap = F.pit_snapshot(_rows({2024: FY[2024], 2023: FY[2023]}), AS_OF)
    res = _features(snapshot=snap)
    assert res.values["gross_margin_stability"] is None
    assert res.reasons["gross_margin_stability"] == "insufficient_history:n=2<3"
    assert res.reasons["revenue_cagr_3y"] == "insufficient_history:fy=2021"
    assert res.reasons["revenue_growth_accel"] == "insufficient_history:fy=2022"


def test_gross_margin_stability_uses_at_most_five_points():
    data = {y: FY[y] for y in FY}
    data[2020] = {"income": {"revenue": 100.0, "gross_profit": 10.0}, "balance": {}, "cash": {}}   # 10% margin
    data[2019] = {"income": {"revenue": 100.0, "gross_profit": 90.0}, "balance": {}, "cash": {}}   # would blow up the stdev
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    assert _close(res.values["gross_margin_stability"], -stdev([0.6, 0.55, 0.5, 0.5, 0.1]))


def test_optional_partner_lines_are_treated_as_not_reported_and_flagged():
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "income", "r_and_d")), AS_OF))
    assert _close(res.values["opex_ratio"], 200 / 1000)
    assert "r_and_d_not_reported" in res.context["flags"]
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "cash", "share_repurchases")), AS_OF))
    assert _close(res.values["shareholder_yield"], 40 / 1000)
    assert "share_repurchases_not_reported" in res.context["flags"]
    res = _features(snapshot=F.pit_snapshot(_rows(_without(2024, "cash", "share_repurchases", "dividends_paid")), AS_OF))
    assert res.values["shareholder_yield"] is None
    assert res.reasons["shareholder_yield"] == "missing:dividends_paid,share_repurchases"


def test_single_line_features_null_when_their_line_is_missing():
    snap = F.pit_snapshot(_rows(_without(2024, "cash", "stock_based_compensation")), AS_OF)
    res = _features(snapshot=snap)
    assert res.values["sbc_to_revenue"] is None
    assert res.reasons["sbc_to_revenue"] == "missing:stock_based_compensation"
    snap = F.pit_snapshot(_rows(_without(2024, "balance", "goodwill")), AS_OF)
    res = _features(snapshot=snap)
    assert res.values["goodwill_to_assets"] is None
    assert res.reasons["goodwill_to_assets"] == "missing:goodwill"


def test_zero_revenue_nulls_every_revenue_denominator():
    data = _without(2024, "income")
    data[2024]["income"]["revenue"] = 0.0
    res = _features(snapshot=F.pit_snapshot(_rows(data), AS_OF))
    for name in ("gross_margin", "operating_margin", "fcf_margin", "opex_ratio", "capex_intensity", "sbc_to_revenue"):
        assert res.values[name] is None, name
        assert res.reasons[name] == "denominator:revenue<=0", name


def test_empty_snapshot_yields_all_null_with_no_snapshot_reason():
    snap = F.pit_snapshot([], AS_OF)
    res = F.compute_features_detailed(snap, PRICE_CTX, "Technology")
    assert all(v is None for v in res.values.values())
    assert set(res.reasons.values()) == {"no_snapshot"}
    assert res.context["fiscal_year"] is None


# ---------------------------------------------------------------------------
# Applicability masks
# ---------------------------------------------------------------------------


def test_financials_are_excluded_from_ev_and_leverage_features():
    res = _features(sector="Financial Services")
    excluded = {n for n, r in res.reasons.items() if r.startswith("excluded:")}
    assert excluded == set(S.FEATURE_NAMES) - S.applicable_features(S.SECTOR_FINANCIALS)
    assert res.reasons["net_debt_to_ebitda"] == "excluded:sector=Financials"
    assert res.applicable == S.applicable_features(S.SECTOR_FINANCIALS)
    # everything that applies is still computed from the same numbers
    assert _close(res.values["earnings_yield"], EXPECTED["earnings_yield"])
    assert _close(res.values["roe"], EXPECTED["roe"])


def test_unmatched_sector_applies_every_feature_and_flags_it():
    res = _features(sector="No Such Sector")
    assert res.reasons == {}
    assert res.context["sector"] is None
    assert "sector_unmatched" in res.context["flags"]
    res = _features(sector=None)
    assert res.reasons == {}
    assert "sector_unmatched" not in res.context["flags"]


def test_utilities_lose_only_the_current_ratio():
    res = _features(sector="Utilities")
    assert {n for n, r in res.reasons.items() if r.startswith("excluded:")} == {"current_ratio"}
