"""DCF engine tests."""
from __future__ import annotations

import pytest

from app.finance.dcf import (
    build_full_dcf,
    check_dcf_realism,
    fmt_price,
    fmt_upside,
    run_dcf,
)
from app.schemas import DCFAssumptions
from app.services.valuation_service import default_dcf_assumptions


def _msft_assumptions() -> DCFAssumptions:
    a = default_dcf_assumptions("MSFT")
    assert a is not None
    return a


def test_default_assumptions_are_within_sane_bounds():
    a = _msft_assumptions()
    assert 0.06 <= a.wacc <= 0.14
    assert 0.0 <= a.tax_rate <= 0.30
    assert 0.005 <= a.capex_pct_revenue <= 0.20
    assert a.base_revenue > 0
    assert a.diluted_shares > 0


def test_dcf_produces_three_scenarios_with_ordering():
    a = _msft_assumptions()
    res = build_full_dcf("MSFT", a)
    assert res.bull.implied_share_price >= res.base.implied_share_price
    assert res.base.implied_share_price >= res.bear.implied_share_price
    # Bear should still produce a positive price for a profitable mega-cap
    assert res.bear.implied_share_price > 0


def test_dcf_projections_have_explicit_horizon():
    a = _msft_assumptions()
    res = run_dcf(a)
    assert len(res.projections) == 5
    for p in res.projections:
        assert p.revenue > 0
        assert p.discount_factor > 0


def test_sensitivity_tables_built():
    a = _msft_assumptions()
    res = build_full_dcf("MSFT", a)
    # Wave 10j — added the exit-multiple cross-check (5 multiples × 3
    # scenarios = 15 cells), bringing the total to 4 sensitivities.
    # The original 3 still produce 25 cells (5×5 grids); the new one
    # produces 15.
    assert len(res.sensitivities) == 4
    grid_counts = sorted(len(s.cells) for s in res.sensitivities)
    assert grid_counts == [15, 25, 25, 25]
    exit_check = next(
        s for s in res.sensitivities
        if s.name.lower().startswith("exit multiple sensitivity")
    )
    # The cross-check spans 5 multiples × 3 scenarios. Cell labels
    # carry the multiple ("9.0x") and scenario ("bear" / "base" /
    # "bull"); the renderer keys off these directly.
    assert {c.col_label for c in exit_check.cells} == {"bear", "base", "bull"}
    assert len({c.row_label for c in exit_check.cells}) == 5


# ---------------------------------------------------------------------------
# None, not 0.0, when the number cannot be computed
# ---------------------------------------------------------------------------

def test_run_dcf_without_shares_yields_none_not_zero():
    """No diluted share count → no implied price, no upside. The old 0.0
    read as "-100% upside" downstream, which is a lie, not a value."""
    a = _msft_assumptions().model_copy(update={"diluted_shares": 0.0})
    s = run_dcf(a)
    assert s.implied_share_price is None
    assert s.upside_pct is None
    assert s.equity_value != 0  # the model still ran; only the per-share step is impossible
    rails = check_dcf_realism(s)
    unavailable = [g for g in rails if g.metric == "implied_share_price_unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0].severity == "error"
    assert unavailable[0].value is None
    assert "share count" in unavailable[0].message


def test_run_dcf_without_price_yields_implied_but_no_upside():
    """No quote → the implied price is still meaningful, the upside isn't."""
    a = _msft_assumptions().model_copy(update={"current_price": 0.0})
    s = run_dcf(a)
    assert s.implied_share_price is not None and s.implied_share_price > 0
    assert s.upside_pct is None
    # The runaway check must not trip on a missing price either.
    assert not [g for g in check_dcf_realism(s) if g.metric.startswith("implied_share_price")]


def test_healthy_dcf_is_not_flagged_as_clamped():
    a = _msft_assumptions()
    s = run_dcf(a)
    assert s.tv_clamped is False
    assert s.upside_pct is not None
    assert not [g for g in check_dcf_realism(s) if g.metric == "terminal_value_clamped"]


def test_degenerate_gordon_denominator_is_flagged():
    """WACC == terminal growth: the Gordon denominator is floored at 50bp,
    so the terminal value is an artefact of the floor. The scenario must
    carry `tv_clamped`, the guardrail must warn, and the summary must say
    so rather than presenting the capped price as a valuation."""
    a = _msft_assumptions().model_copy(update={"wacc": 0.06, "terminal_growth": 0.06})
    s = run_dcf(a)
    assert s.tv_clamped is True
    rails = check_dcf_realism(s)
    clamped = [g for g in rails if g.metric == "terminal_value_clamped"]
    assert len(clamped) == 1
    assert clamped[0].severity == "warn"
    assert clamped[0].value == pytest.approx(0.0)
    assert "0.5%" in clamped[0].message
    assert "capped" in clamped[0].message

    res = build_full_dcf("MSFT", a)
    assert res.base.tv_clamped is True
    assert "clamped" in res.summary.lower()
    assert any(g.metric == "terminal_value_clamped" for g in res.guardrails)


def test_build_full_dcf_summary_says_na_when_unavailable():
    a = _msft_assumptions().model_copy(update={"diluted_shares": 0.0, "current_price": 0.0})
    res = build_full_dcf("MSFT", a)
    assert res.current_price is None
    assert res.base.implied_share_price is None
    assert res.bull.upside_pct is None
    assert "n/a" in res.summary
    assert "$0.00" not in res.summary
    assert "+0.0%" not in res.summary
    # Every sensitivity cell — including the exit-multiple cross-check —
    # carries None rather than a fabricated $0.00.
    assert all(c.value is None for s in res.sensitivities for c in s.cells)
    assert any(g.metric == "implied_share_price_unavailable" for g in res.guardrails)


def test_summary_reads_na_for_current_when_price_missing():
    """Shares present, quote missing: the implied price prints, the
    comparison is explicitly n/a — the reader sees WHY the upside is n/a."""
    a = _msft_assumptions().model_copy(update={"current_price": 0.0})
    res = build_full_dcf("MSFT", a)
    assert "vs current n/a" in res.summary
    assert "(n/a)" in res.summary
    assert res.base.implied_share_price is not None


def test_fmt_helpers_render_na_for_none():
    assert fmt_price(None) == "n/a"
    assert fmt_upside(None) == "n/a"
    assert fmt_price(1234.5) == "$1,234.50"
    assert fmt_upside(0.1234) == "+12.3%"
    assert fmt_upside(-0.05, decimals=0) == "-5%"
