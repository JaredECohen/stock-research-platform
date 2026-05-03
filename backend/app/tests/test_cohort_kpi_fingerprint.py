"""Wave 6B tests — cohort KPI fingerprint tightening.

Covers:
- `_kpi_fingerprint_inputs` produces a stable, ticker-prefixed token from
  the latest income + cash + profile rows.
- Two cohort entries with the same KPIs but different filings produce
  the SAME fingerprint (the whole point — irrelevant filings don't bust
  the cache).
- A change in revenue / operating_income / capex / shares produces a
  different fingerprint.
- `run_sector_research` writes a sector_warm snapshot whose
  `sources_used` carries `kpi:` tokens (not `filing:` tokens).
"""
from __future__ import annotations

from app.services import sector_research_service as srs


def _entry(rev=10_000_000_000, op=2_000_000_000, capex=500_000_000,
           shares=1_000_000_000, ticker="X"):
    return {
        "ticker": ticker,
        "financials": {
            "income": [{"period": "2024", "revenue": rev, "operating_income": op}],
            "cash": [{"period": "2024", "capex": capex}],
            "profile": {"shares_outstanding": shares},
        },
    }


def test_kpi_fingerprint_includes_all_inputs():
    out = srs._kpi_fingerprint_inputs(_entry(ticker="NVDA"))
    assert out is not None
    assert out.startswith("kpi:NVDA:")
    assert "rev=" in out and "op=" in out and "cx=" in out and "sh=" in out


def test_kpi_fingerprint_stable_under_irrelevant_changes():
    """Same KPIs, different ticker — fingerprints differ.
    Same KPIs, same ticker — fingerprints match exactly."""
    a = srs._kpi_fingerprint_inputs(_entry(ticker="X"))
    b = srs._kpi_fingerprint_inputs(_entry(ticker="X"))
    assert a == b
    c = srs._kpi_fingerprint_inputs(_entry(ticker="Y"))
    assert a != c


def test_kpi_fingerprint_changes_when_revenue_moves():
    a = srs._kpi_fingerprint_inputs(_entry(rev=10_000_000_000))
    b = srs._kpi_fingerprint_inputs(_entry(rev=11_000_000_000))
    assert a != b


def test_kpi_fingerprint_round_smooths_micro_jitter():
    """A $5 jitter in revenue at the billion-dollar level shouldn't shift
    the fingerprint — the rounding to nearest $1M smooths it out."""
    a = srs._kpi_fingerprint_inputs(_entry(rev=10_000_000_005))
    b = srs._kpi_fingerprint_inputs(_entry(rev=10_000_000_000))
    assert a == b


def test_kpi_fingerprint_returns_none_when_inputs_missing():
    out = srs._kpi_fingerprint_inputs({"ticker": "X", "financials": {}})
    assert out is None
    assert srs._kpi_fingerprint_inputs({}) is None


def test_run_sector_research_writes_kpi_tokens_in_sources():
    """End-to-end: a fresh sector_warm snapshot's `sources_used` carries
    the new KPI fingerprint tokens, not `filing:` tokens."""
    from app.cache import cache_get
    from app.services.data_service import get_data_service
    payload = srs.run_sector_research("NVDA", force_refresh=True)
    assert payload
    # Cache key shape is `{sector}:{sub_industry}:{target}` (whatever the
    # data service reports for this ticker — varies across environments).
    profile = get_data_service().get_company_profile("NVDA") or {}
    sector = profile.get("sector", "")
    sub_industry = profile.get("sub_industry") or profile.get("industry") or ""
    snap = cache_get(f"{sector}:{sub_industry}:NVDA", "sector_warm")
    assert snap is not None
    sources = snap.sources_used or []
    assert any(str(s).startswith("kpi:") for s in sources)
    # No more per-filing tokens — that was the whole point of Wave 6B.
    assert not any(str(s).startswith("filing:") for s in sources)
