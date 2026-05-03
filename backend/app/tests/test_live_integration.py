"""Wave 4B — opt-in live integration tests.

Every test in this file is gated behind `pytest.mark.live` and is
skipped unless `RUN_LIVE_TESTS=1` is set in the environment. The
session-default `ENABLE_LIVE_DATA=false` (set by conftest.py) is also
flipped on a per-test basis via the `live_settings` fixture so the
data_service routes through real providers.

Why opt-in:
- Real API calls cost money. CI runs nightly with secrets; dev shouldn't
  pay every time `pytest` is invoked.
- Real APIs go down. A flaky network shouldn't fail unrelated PRs.

Each test guards on `settings.<provider>_api_key` presence so a
partially-configured environment skips per-provider rather than
exploding. Tests that need a key but don't have one are `pytest.skip`-ed
in-place so the live suite is self-documenting (any test that ran without
its key would be lying about provider coverage).
"""
from __future__ import annotations

import os

import pytest

from app.config import settings


pytestmark = pytest.mark.live


def _require_key(name: str, attr: str) -> str:
    val = getattr(settings, attr, None) or ""
    if not val:
        pytest.skip(f"{name} not configured (set {name} in .env to enable)")
    return val


# ---------------------------------------------------------------------------
# Provider-level smoke tests
# ---------------------------------------------------------------------------

def test_live_fmp_returns_profile(live_settings):
    """FMP profile endpoint should return a non-empty record for AAPL."""
    _require_key("FMP_API_KEY", "fmp_api_key")
    from app.providers.fmp_provider import FMPProvider
    p = FMPProvider()
    profile = p.get_company_profile("AAPL")
    assert profile is not None, "expected FMP to return a profile for AAPL"
    assert profile.get("ticker", "").upper() == "AAPL"
    assert profile.get("market_cap")


def test_live_fmp_returns_price_history(live_settings):
    _require_key("FMP_API_KEY", "fmp_api_key")
    from app.providers.fmp_provider import FMPProvider
    p = FMPProvider()
    rows = p.get_price_history("AAPL", days=30)
    assert rows, "expected ≥1 price row from FMP for AAPL/30d"
    sample = rows[0]
    assert "date" in sample and "close" in sample


def test_live_fred_returns_macro_series(live_settings):
    """FRED unemployment series should return a non-empty observation list."""
    _require_key("FRED_API_KEY", "fred_api_key")
    from app.providers.fred_provider import FREDProvider
    p = FREDProvider()
    series = p.get_macro_series("UNRATE")
    assert series is not None
    assert series.get("series_id") == "UNRATE"
    assert series.get("points"), "expected non-empty observations from FRED"


def test_live_sec_edgar_returns_filings(live_settings):
    """SEC EDGAR is keyless but rate-limited; just check the shape."""
    from app.providers.sec_edgar_provider import SECEdgarProvider
    p = SECEdgarProvider()
    # Apple's CIK is 0000320193.
    rows = p.get_filings("AAPL", cik="0000320193")
    if not rows:
        # SEC sometimes returns empty under heavy load; skip rather than fail.
        pytest.skip("SEC returned no filings; possibly rate-limited")
    sample = rows[0]
    assert "type" in sample or "filing_type" in sample


# ---------------------------------------------------------------------------
# LLM provider smoke tests
# ---------------------------------------------------------------------------

def test_live_openai_chat_returns_text(live_settings):
    _require_key("OPENAI_API_KEY", "openai_api_key")
    from app.agents.llm import chat_text
    out = chat_text(
        "Say the word OK and nothing else.",
        system="You answer with one word only.", route="cheap",
    )
    assert out is not None
    assert len(out) > 0


def test_live_anthropic_chat_returns_text(live_settings):
    _require_key("ANTHROPIC_API_KEY", "anthropic_api_key")
    from app.agents.llm import chat_text
    out = chat_text(
        "Say the word OK and nothing else.",
        system="You answer with one word only.",
        route="strong",  # Anthropic is the "strong" route
    )
    assert out is not None
    assert len(out) > 0


# ---------------------------------------------------------------------------
# End-to-end memo run on real providers
# ---------------------------------------------------------------------------

def test_live_memo_run_on_aapl(live_settings):
    """A full live memo run should produce a non-empty memo with sources_used
    that reference actual filings (not just demo placeholders)."""
    _require_key("FMP_API_KEY", "fmp_api_key")
    from app.agents.graph import run_stock_memo
    memo = run_stock_memo("AAPL")
    assert memo.ticker == "AAPL"
    assert memo.sector
    assert memo.final_pm_view
    assert memo.rating_label in {
        "Bullish", "Mixed Positive", "Neutral", "Mixed Negative", "Bearish",
    }
    # Live mode should produce sources that include profile + financials at minimum.
    assert any(s.startswith("profile:") for s in memo.sources_used)
    assert any(s.startswith("financials:") for s in memo.sources_used)
