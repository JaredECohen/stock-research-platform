"""Three services gated their LLM path on `settings.openai_api_key`.

`llm.chat_json` routes to whichever provider is configured, so the
OpenAI-only check silently disabled bull/bear scenario drivers, the
mispricing audit, and the NL screener translation on Anthropic-only
deployments — while each returned its deterministic fallback as if the
call had simply produced nothing. The gate is now `settings.has_llm`.
Every case here patches `llm.chat_json`; nothing reaches a provider.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents import llm
from app.config import settings
from app.schemas import DCFAssumptions
from app.services import mispricing_audit, nl_screener, scenario_assumptions


@pytest.fixture
def anthropic_only(monkeypatch):
    """The deployment shape that exposed the bug."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-anthropic")
    assert settings.has_llm and settings.active_llm_provider == "anthropic"


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert not settings.has_llm


def _base_assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
        operating_margin=[0.25, 0.25, 0.26, 0.26, 0.27],
        wacc=0.09,
        terminal_growth=0.025,
    )


_PROFILE = {"ticker": "T", "company_name": "Test Co", "sector": "Technology"}


# ---------------------------------------------------------------------------
# scenario_assumptions.build_bull_bear
# ---------------------------------------------------------------------------

def test_bull_bear_uses_the_llm_on_anthropic_only(anthropic_only):
    fake = {
        "bull": {"growth_bp": 200, "drivers": [{"name": "Share gains", "rationale": "r"}]},
        "bear": {"margin_bp": -300, "drivers": [{"name": "Pricing pressure", "rationale": "r"}]},
    }
    with patch.object(llm, "chat_json", return_value=fake) as call:
        bull, bull_drivers, bear, bear_drivers = scenario_assumptions.build_bull_bear(
            _PROFILE, _base_assumptions(),
        )
    assert call.call_count == 1
    assert [d.name for d in bull_drivers] == ["Share gains"]
    assert [d.name for d in bear_drivers] == ["Pricing pressure"]
    assert bull.revenue_growth[0] > _base_assumptions().revenue_growth[0]


def test_bull_bear_stays_deterministic_without_an_llm(no_llm):
    with patch.object(llm, "chat_json") as call:
        out = scenario_assumptions.build_bull_bear(_PROFILE, _base_assumptions())
    call.assert_not_called()
    expected = scenario_assumptions._deterministic_fallback(_PROFILE, _base_assumptions())
    assert [d.name for d in out[1]] == [d.name for d in expected[1]]
    assert [d.name for d in out[3]] == [d.name for d in expected[3]]


def test_bull_bear_falls_back_when_the_llm_returns_nothing(anthropic_only):
    with patch.object(llm, "chat_json", return_value=None):
        out = scenario_assumptions.build_bull_bear(_PROFILE, _base_assumptions())
    expected = scenario_assumptions._deterministic_fallback(_PROFILE, _base_assumptions())
    assert [d.name for d in out[1]] == [d.name for d in expected[1]]


# ---------------------------------------------------------------------------
# mispricing_audit.run_audit
# ---------------------------------------------------------------------------

_MEMOS = [{"ticker": "T", "version": 2, "thesis": "x"}]


def test_audit_runs_on_anthropic_only(anthropic_only):
    fake = {
        "per_memo": [{"ticker": "T", "version": 2, "specificity": 4,
                      "differentiation": 3, "falsifiability": 5, "improvement": "tighten"}],
        "pattern_observation": "theses are generic",
    }
    with patch.object(mispricing_audit, "_gather_memos", return_value=_MEMOS), \
         patch.object(llm, "chat_json", return_value=fake) as call:
        out = mispricing_audit.run_audit(limit=5)
    assert call.call_count == 1
    assert out["audited"] == 1
    assert out["per_memo"][0]["specificity"] == 4
    assert out["pattern_observation"] == "theses are generic"


def test_audit_is_skipped_without_an_llm(no_llm):
    with patch.object(mispricing_audit, "_gather_memos", return_value=_MEMOS), \
         patch.object(llm, "chat_json") as call:
        out = mispricing_audit.run_audit(limit=5)
    call.assert_not_called()
    assert out["audited"] == 1 and out["per_memo"] == []
    assert "skipped" in out["pattern_observation"]


# ---------------------------------------------------------------------------
# nl_screener._llm_translate
# ---------------------------------------------------------------------------

def test_nl_translate_runs_on_anthropic_only(anthropic_only):
    fake = {"rules": [{"metric": "pe", "op": "lt", "value": 20}], "themes": [],
            "sectors": [], "sort_by": "pe", "order": "asc", "rationale": "cheap"}
    with patch.object(llm, "chat_json", return_value=fake) as call:
        out = nl_screener._llm_translate("cheap tech")
    assert call.call_count == 1
    assert out == fake


def test_nl_translate_returns_none_without_an_llm(no_llm):
    with patch.object(llm, "chat_json") as call:
        assert nl_screener._llm_translate("cheap tech") is None
    call.assert_not_called()
