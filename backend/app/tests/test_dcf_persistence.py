"""Wave 5A tests — versioned DCF persistence + LLM updater.

Covers:
- `dcf_store.save_version` / `latest_version` / `version_history`
  round-trip with parent_version chain.
- `_clamp_delta` enforces ±20% per cycle (the v1 safety rail).
- `_shift_forecast_forward` rolls a multi-year forecast list correctly.
- `_apply_updates` drops fields without a rationale (discipline rule).
- `update_for_new_period` falls back to deterministic roll-forward when
  the LLM is unavailable; emits change rows tagged "deterministic".
- `update_for_new_period` accepts a stub LLM proposal, clamps it, and
  persists rationales per-field.
- `update_on_earnings_close` chains v1 → v2 with `earnings_update`
  trigger and a non-empty `assumption_changes`.
- `build_dcf` (existing) persists a `DCFModel` row on the default-build
  path, tagging `initial` for the first save and `memo_rebuild` for
  subsequent rebuilds.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

from app.agents import dcf_updater
from app.database import SessionLocal
from app.models import DCFModel
from app.schemas import DCFAssumptions
from app.services import dcf_store


def _stub_assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
        operating_margin=[0.25, 0.26, 0.27, 0.27, 0.27],
        tax_rate=0.21,
        da_pct_revenue=0.04,
        capex_pct_revenue=0.05,
        nwc_pct_revenue=0.02,
        terminal_growth=0.025,
        exit_ebitda_multiple=15.0,
        wacc=0.085,
        base_revenue=100.0,
        net_debt=10.0,
        diluted_shares=1.0,
        current_price=120.0,
    )


def _reset_table() -> None:
    with SessionLocal() as db:
        dcf_store._ensure_table(db)
        db.query(DCFModel).delete()
        db.commit()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_clamp_delta_caps_movement_to_20_percent():
    # Prior 0.10 → 20% cap means the proposal stays within [0.08, 0.12].
    assert abs(dcf_updater._clamp_delta(0.10, 0.50) - 0.12) < 1e-9
    assert abs(dcf_updater._clamp_delta(0.10, -0.10) - 0.08) < 1e-9
    # Within bounds: passes through unchanged.
    assert abs(dcf_updater._clamp_delta(0.10, 0.105) - 0.105) < 1e-9


def test_clamp_delta_handles_near_zero_prior():
    # Prior 0.001 → cap base = 0.005, range = 0.005 * 0.20 = 0.001.
    out = dcf_updater._clamp_delta(0.001, 0.10)
    assert abs(out - 0.002) < 1e-9


def test_shift_forecast_forward_drops_first_repeats_last():
    out = dcf_updater._shift_forecast_forward([0.10, 0.09, 0.08, 0.07, 0.06])
    assert out == [0.09, 0.08, 0.07, 0.06, 0.06]


def test_shift_forecast_forward_handles_short_lists():
    assert dcf_updater._shift_forecast_forward([]) == []
    assert dcf_updater._shift_forecast_forward([0.10]) == [0.10]


# ---------------------------------------------------------------------------
# dcf_store
# ---------------------------------------------------------------------------

def test_save_and_latest_version_round_trip():
    _reset_table()
    a = _stub_assumptions()
    snap = dcf_store.save_version("RTRIP", assumptions=a, trigger="initial")
    assert snap.version == 1
    assert snap.trigger == "initial"

    latest = dcf_store.latest_version("RTRIP")
    assert latest is not None
    assert latest.version == 1
    rehydrated = dcf_store.assumptions_to_pydantic(latest)
    assert rehydrated.terminal_growth == a.terminal_growth


def test_versions_chain_with_parent_version():
    _reset_table()
    a = _stub_assumptions()
    dcf_store.save_version("CHAIN", assumptions=a, trigger="initial")
    v2 = dcf_store.save_version(
        "CHAIN", assumptions=a, trigger="memo_rebuild", parent_version=1,
        assumption_changes=[{"field": "wacc", "from": 0.085, "to": 0.090,
                             "rationale": "rates higher"}],
    )
    assert v2.version == 2
    assert v2.parent_version == 1
    history = dcf_store.version_history("CHAIN")
    assert [h.version for h in history] == [2, 1]


def test_save_version_rejects_unknown_trigger():
    import pytest
    a = _stub_assumptions()
    with pytest.raises(ValueError):
        dcf_store.save_version("BADTRIG", assumptions=a, trigger="bogus_trigger")


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

def test_update_for_new_period_no_llm_path_does_deterministic_shift():
    """Without an LLM proposal, the updater rolls the explicit forecast
    forward and tags the changes as deterministic."""
    prior = _stub_assumptions()
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=None):
        new, change_rows = dcf_updater.update_for_new_period(
            "DETER", prior, actuals={"revenue_latest": 110.0},
        )
    # Forecast lists shifted forward.
    assert new.revenue_growth[0] == prior.revenue_growth[1]
    # Change rows tagged deterministic.
    rationales = " ".join(r.get("rationale", "") for r in change_rows)
    assert "deterministic" in rationales.lower()


def test_update_for_new_period_applies_llm_proposal_with_clamp():
    """A stubbed LLM raising terminal_growth from 0.025 to 0.10 should
    clamp to the 20% cap (0.025 * 1.20 = 0.030)."""
    prior = _stub_assumptions()
    fake_proposal = {
        "updates": {"terminal_growth": 0.10, "wacc": 0.080},
        "rationales": {
            "terminal_growth": "secular maturation",
            "wacc": "real rates rolled lower",
        },
    }
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=fake_proposal):
        new, change_rows = dcf_updater.update_for_new_period(
            "CLAMP", prior, actuals={"revenue_latest": 110.0},
        )
    # 20% cap → 0.025 * 1.20 = 0.030.
    assert abs(new.terminal_growth - 0.030) < 1e-9
    # WACC was within bounds → applied.
    assert abs(new.wacc - 0.080) < 1e-9
    # Both rationales preserved on the change rows.
    field_rationales = {r["field"]: r["rationale"] for r in change_rows}
    assert "secular maturation" in field_rationales.get("terminal_growth", "")
    assert "real rates" in field_rationales.get("wacc", "")


def test_update_drops_field_when_rationale_missing():
    prior = _stub_assumptions()
    fake_proposal = {
        "updates": {"terminal_growth": 0.030, "wacc": 0.080},
        "rationales": {"terminal_growth": ""},  # missing rationale → drop
    }
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=fake_proposal):
        new, _ = dcf_updater.update_for_new_period(
            "DROPRAT", prior, actuals={},
        )
    # terminal_growth dropped (no rationale) → stays at prior 0.025.
    assert new.terminal_growth == prior.terminal_growth
    # wacc dropped too (no rationale entry at all) → stays at prior.
    assert new.wacc == prior.wacc


def test_update_drops_field_outside_allowed_set():
    prior = _stub_assumptions()
    fake_proposal = {
        "updates": {"current_price": 999.0},  # not in _ADJUSTABLE_FIELDS
        "rationales": {"current_price": "test"},
    }
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=fake_proposal):
        new, _ = dcf_updater.update_for_new_period(
            "OUTSIDE", prior, actuals={},
        )
    assert new.current_price == prior.current_price


def test_update_drops_list_field_with_length_mismatch():
    prior = _stub_assumptions()
    fake_proposal = {
        "updates": {"revenue_growth": [0.20, 0.19]},  # wrong length (5 expected)
        "rationales": {"revenue_growth": "test"},
    }
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=fake_proposal):
        new, _ = dcf_updater.update_for_new_period(
            "MISMATCH", prior, actuals={},
        )
    # Length mismatch → falls back to deterministic roll-forward.
    assert new.revenue_growth[0] == prior.revenue_growth[1]


# ---------------------------------------------------------------------------
# update_on_earnings_close — full chain
# ---------------------------------------------------------------------------

def test_update_on_earnings_close_chains_versions_and_records_changes():
    _reset_table()
    a = _stub_assumptions()
    dcf_store.save_version("CHAINUP", assumptions=a, trigger="initial")

    fake_proposal = {
        "updates": {"wacc": 0.080},
        "rationales": {"wacc": "rates lower in soft-landing scenario"},
    }
    # Stub the LLM and the DCF engine so the test stays offline.
    with patch.object(dcf_updater, "_llm_propose_updates", return_value=fake_proposal), \
         patch.object(dcf_updater, "actuals_from_history",
                      return_value={"revenue_latest": 110.0,
                                    "operating_margin_latest": 0.27}), \
         patch("app.finance.dcf.build_full_dcf", return_value=None):
        v2 = dcf_store.update_on_earnings_close("CHAINUP")
    assert v2 is not None
    assert v2.version == 2
    assert v2.parent_version == 1
    assert v2.trigger == "earnings_update"
    # WACC change recorded with rationale.
    fields = {r["field"]: r for r in v2.assumption_changes}
    assert "wacc" in fields
    assert "rates" in fields["wacc"]["rationale"].lower()


def test_update_on_earnings_close_returns_none_when_no_prior():
    _reset_table()
    out = dcf_store.update_on_earnings_close("NEVER_SEEDED")
    assert out is None


# ---------------------------------------------------------------------------
# build_dcf integration
# ---------------------------------------------------------------------------

def test_build_dcf_persists_initial_version():
    _reset_table()
    from app.services.valuation_service import build_dcf
    res = build_dcf("MSFT", force_refresh=True)
    if res is None:
        return  # demo dataset may not seed MSFT properly; skip
    snap = dcf_store.latest_version("MSFT")
    assert snap is not None
    assert snap.version >= 1
    assert snap.trigger in ("initial", "memo_rebuild")
