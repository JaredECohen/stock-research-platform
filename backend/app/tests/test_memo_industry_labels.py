"""Owner decision 2026-09-24 — memo-side industry surfaces carry our own
labels, never a GICS code, the brand or a registry name.

Routing is still OFF in production (it flips in a later slice), so these
tests turn it on explicitly and check every memo-side surface the Industry
Group analyst and the sector analyst produce when it is on:

- `industry_group_summary` (on both findings' `data`),
- the analyst finding (headline, summary, key points, evidence refs,
  sources, the whole `data` payload) on the deterministic AND the LLM path,
- the company-context block and the system prompt the models are sent,
- the sector analyst's LLM output when the industry block was spliced in,
- and the failure path: a label accessor that raises must degrade the
  sector card, not turn it into a crash stub.

`display_name` (the LLMCallLog telemetry key) keeps its code on purpose.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pytest

from app.agents import graph, sector_agents
from app.agents import industry_analysts as ia
from app.config import settings
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.services import industry_knowledge as ik
from app.services import industry_labels as il
from app.tests.gating_helpers import seed_demo_universe

TICKERS = ("MSFT", "NVDA", "JPM")


@pytest.fixture(scope="module", autouse=True)
def _universe():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(tickers=list(TICKERS))
    yield info


@pytest.fixture(autouse=True)
def _fresh_cache():
    ia.clear_cache()
    yield
    ia.clear_cache()


def _strings(obj: Any) -> list[str]:
    """Every string in a JSON-shaped object — dict KEYS included, since a
    code used as a key leaks just as well as one used as a value."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def _taxonomy_codes(row: dict, analyst: ia.IndustryAnalyst) -> set[str]:
    codes = {row.get("industry_group_code"), row.get("industry_code"), row.get("sub_industry_code"),
             *(row.get("sub_industry_codes") or []), *analyst.mandate.industry_codes,
             *(s.code for s in analyst.mandate.sub_industries)}
    return {str(c) for c in codes if c}


def _leaks(obj: Any, row: dict, analyst: ia.IndustryAnalyst) -> list[str]:
    """What a reader must never see: the brand, an internal taxonomy key, a
    4/6/8-digit code this company's classification or mandate carries, the
    sector code in a code position, or a registry name that differs from
    our label (the group's, and the sub-industry's where it is shown as one)."""
    found: list[str] = []
    codes = _taxonomy_codes(row, analyst)
    sector = str(row.get("sector_code") or "")
    sub = ik.get_sub_industry(str(row.get("sub_industry_code") or "")) or {}
    # The provider's own industry string is public by design (it stands in
    # for the sub-industry) even where it happens to spell a registry name —
    # FMP calls NVDA's industry "Semiconductors" and JPM's "Banks". The
    # whitelist is the ROW's column, never `_sub_industry_of` itself: a
    # function under test cannot be its own oracle.
    provider = row.get("source_industry")
    # Where the registry sub-industry name differs from the provider's
    # string (MSFT "Systems Software" vs "Software", JPM "Diversified Banks"
    # vs "Banks") it must appear nowhere, in any position.
    sub_name = sub.get("name") or ""
    distinct_sub_name = sub_name if sub_name and sub_name not in (provider or "") else ""
    for text in _strings(obj):
        if distinct_sub_name and distinct_sub_name in text:
            found.append(f"registry sub-industry name: {text[:80]}")
        if text == provider:
            continue
        low = text.lower()
        if "gics" in low:
            found.append(f"brand: {text[:80]}")
        if analyst.taxonomy_version_key in text:
            found.append(f"internal key: {text[:80]}")
        for code in codes:
            if re.search(rf"(?<!\d){code}(?!\d)", text):
                found.append(f"code {code}: {text[:80]}")
        for form in (f"sector {sector}", f"({sector})", f"[{sector}]"):
            if sector and form in text:
                found.append(f"sector code: {text[:80]}")
        if analyst.name != analyst.label and (
            analyst.name in text if ("&" in analyst.name or "," in analyst.name)
            else f"{analyst.name} (" in text or text == analyst.name
        ):
            found.append(f"registry group name: {text[:80]}")
        if sub.get("name") and (f"sub-industry {sub['name']}" in text or text == sub["name"]):
            found.append(f"registry sub-industry name: {text[:80]}")
    return found


def _row(ticker: str) -> dict:
    return ic.current_for([ticker])[ticker]


# --- the deterministic surfaces ----------------------------------------------------


@pytest.mark.parametrize("ticker", TICKERS)
def test_summary_finding_evidence_and_company_context_carry_no_code_or_registry_name(ticker):
    row = _row(ticker)
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    profile = {"ticker": ticker, "company_name": ticker}

    summary = ia.industry_group_summary(row, analyst)
    assert _leaks(summary, row, analyst) == []
    assert summary["label"] == analyst.label and summary["slug"] == analyst.slug

    finding = ia.run_industry_group_agent(profile, {"roic": 0.2}, classification=row)
    assert _leaks(finding.model_dump(), row, analyst) == []
    assert finding.headline.startswith(f"{analyst.label}: mandate read for {ticker}")
    assert [e.ref for e in finding.evidence] == [f"industry_group:{analyst.slug}"]
    assert finding.sources == [f"industry_knowledge:{analyst.slug}"]
    assert all("industry_code" not in k for k in finding.data["kpis_to_watch"])
    assert finding.data["attribution"] == il.PUBLIC_BRIEF_ATTRIBUTION

    context = analyst.company_context_block(profile, row)
    assert _leaks(context, row, analyst) == []
    assert analyst.label in context and il.PUBLIC_MAPPING_CAVEAT in context

    block, sector_summary, _ = ia.sector_prompt_block(profile, row)
    assert _leaks([block, sector_summary], row, analyst) == []

    system = analyst.system_prompt()
    assert _leaks(system, row, analyst) == []

    # The telemetry key keeps its code; the reader-facing name does not.
    assert analyst.display_name == f"Industry Group Analyst {analyst.code}"
    assert analyst.public_display_name == f"Industry Group Analyst ({analyst.label})"


def test_provider_industry_is_the_rows_own_column_in_precedence_order():
    """Pins `_sub_industry_of` to the row's DATA (review of d623e59: the
    leak tests used the function as their own oracle, so returning the
    registry sub-industry name passed everything)."""
    for ticker in TICKERS:
        row = _row(ticker)
        assert row["source_industry"]
        assert ia._sub_industry_of(row) == row["source_industry"]
    labels = {"industry": "Labels Industry", "sub_industry": "Labels Sub"}
    assert ia._sub_industry_of({"source_industry": "Column", "evidence": {"labels": labels}}) == "Column"
    assert ia._sub_industry_of({"evidence": {"labels": labels}, "source_sub_industry": "Sub Column"}) \
        == "Labels Industry"
    assert ia._sub_industry_of({"source_sub_industry": "Sub Column", "evidence": {"labels": {
        "sub_industry": "Labels Sub"}}}) == "Sub Column"
    assert ia._sub_industry_of({"evidence": {"labels": {"sub_industry": "Labels Sub"}}}) == "Labels Sub"


@pytest.mark.parametrize("row", [
    None, "not-a-dict", {}, {"evidence": "not-a-dict"}, {"evidence": {"labels": "x"}},
    {"evidence": {"labels": ["x"]}}, {"evidence": None, "source_industry": "   "},
])
def test_sub_industry_of_never_raises(row):
    assert ia._sub_industry_of(row) is None


def test_malformed_evidence_does_not_hide_the_provider_column_or_fail_the_finding():
    base = _row("MSFT")
    row = {**base, "evidence": "not-a-dict"}
    assert ia._sub_industry_of(row) == base["source_industry"]
    bare = {k: v for k, v in row.items() if k not in ("source_industry", "source_sub_industry")}
    finding = ia.run_industry_group_agent({"ticker": "MSFT"}, {}, classification=bare)
    assert finding.data["industry_group"]["provider_industry"] is None
    assert "no provider industry recorded" in finding.summary


def _pg_shaped_row() -> dict:
    """A provider-alias row whose provider industry string is ALSO a
    registry group name: FMP files PG and CL under "Household & Personal
    Products", the registry's name for group 3030."""
    base = _row("MSFT")
    return {
        **base, "ticker": "PG", "source": ic.SOURCE_PROVIDER_ALIAS, "author": "alias map",
        "sector_code": "30", "industry_group_code": "3030", "industry_code": "303010",
        "sub_industry_code": "30301010", "sub_industry_codes": ["30301010"],
        "source_industry": "Household & Personal Products", "source_sub_industry": None,
        "evidence": {"labels": {"sector": "Consumer Defensive", "industry": "Household & Personal Products"}},
    }


def test_provider_industry_that_spells_a_registry_name_is_shown_as_the_provider_wrote_it(monkeypatch):
    """REGRESSION (review of d623e59): the finding scrubbed the provider's
    string into our label, so it reported "Household & Personal Care
    Brands" as the PROVIDER's industry while the sector card, in the same
    memo, said "Household & Personal Products"."""
    row = _pg_shaped_row()
    provider = row["source_industry"]
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None and analyst.code == "3030" and analyst.name == provider
    profile = {"ticker": "PG", "company_name": "Procter & Gamble"}

    _, sector_summary, _ = ia.sector_prompt_block(profile, row)
    finding = ia.run_industry_group_agent(profile, {}, classification=row)
    assert sector_summary is not None
    assert finding.data["industry_group"]["provider_industry"] == sector_summary["provider_industry"] == provider
    assert finding.headline == f"{analyst.label}: mandate read for PG — {provider}"
    assert f"provider industry: {provider}." in finding.summary
    assert f"provider industry: {provider}." in finding.data["placement"]
    assert f"Provider industry: {provider}." in analyst.company_context_block(profile, row)

    # The LLM path and the spliced sector output keep it too, while a code
    # next to it and the brand are still scrubbed.
    monkeypatch.setattr(type(settings), "has_llm", property(lambda self: True))
    monkeypatch.setattr(ia.llm, "chat_json", lambda *a, **k: {
        "headline": f"{provider} (3030) demand holds", "summary": f"The GICS {provider} peers", "confidence": 0.6,
    })
    finding = ia.run_industry_group_agent(profile, {}, classification=row)
    assert finding.headline == f"{provider} demand holds"
    assert finding.summary == f"The {provider} peers"
    assert finding.data["industry_group"]["provider_industry"] == provider

    out = sector_agents._scrub_spliced_output(
        {"headline": f"{provider} (3030) volumes", "summary": "Software & Services lag"},
        provider_industry=provider,
    )
    assert out == {"headline": f"{provider} volumes", "summary": f"{il.label('4510')} lag"}


# --- the LLM path ----------------------------------------------------------------------


def _leaky_prose(analyst: ia.IndustryAnalyst, sub_code: str) -> str:
    return (f"The GICS® {analyst.name} ({analyst.code}) group [{sub_code}] rerates; "
            f"Industry Group Analyst {analyst.code} sees sub-industry {sub_code} leading.")


def test_llm_prose_is_scrubbed_and_structured_codes_are_dropped(monkeypatch):
    row = _row("NVDA")
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    sub_code = str(row["sub_industry_code"])
    leaky = _leaky_prose(analyst, sub_code)
    seen: dict[str, str] = {}

    def fake_chat_json(prompt, *, system=None, **kwargs):
        seen["prompt"], seen["system"] = prompt, system or ""
        return {
            "headline": leaky, "summary": leaky, "key_points": [leaky, "Wafer starts [453010]"],
            "confidence": 0.8, "causal_chain": [{"stage": "world_change", "text": leaky}],
            "placement": leaky, "mandate_type": "inflection",
            "kpis_to_watch": [{"kpi": "Wafer starts", "industry_code": "453010", "why": leaky}],
            "falsifiers": [leaky], "traps": [leaky],
        }

    monkeypatch.setattr(ia.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(type(settings), "has_llm", property(lambda self: True))
    finding = ia.run_industry_group_agent({"ticker": "NVDA", "company_name": "NVIDIA"}, {}, classification=row)

    assert finding.confidence == 0.8                        # the LLM path, not the fallback
    assert _leaks(finding.model_dump(), row, analyst) == []
    assert analyst.label in finding.headline
    assert finding.key_points[1] == "Wafer starts"
    kpi = finding.data["kpis_to_watch"][0]
    assert "industry_code" not in kpi and kpi["kpi"] == "Wafer starts"
    # The model was never shown a code, the brand or a version key either.
    assert _leaks([seen["prompt"], seen["system"]], row, analyst) == []
    assert '"industry_code"' not in seen["prompt"]


def test_headline_fallback_uses_the_label(monkeypatch):
    row = _row("MSFT")
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    monkeypatch.setattr(ia.llm, "chat_json", lambda *a, **k: {"summary": "s", "confidence": 0.6})
    monkeypatch.setattr(type(settings), "has_llm", property(lambda self: True))
    finding = ia.run_industry_group_agent({"ticker": "MSFT"}, {}, classification=row)
    assert finding.headline == f"{analyst.label} read for MSFT"


# --- the sector analyst -----------------------------------------------------------------


def _sector_llm(leaky: str):
    def fake(prompt, **kwargs):
        fake.prompt = prompt  # type: ignore[attr-defined]
        return {
            "headline": leaky, "summary": leaky, "key_points": [leaky], "confidence": 0.7,
            "macro_alignment": leaky,
            "bull_bear_analysis": {
                "bull_case": {"headline": leaky, "key_points": [leaky]},
                "bear_case": {"headline": leaky, "key_points": [leaky]},
                "falsifiable_tests": [{"statement": leaky, "invalidates_side": "bull"}],
                "key_disagreement": leaky, "sector_synthesis": leaky, "sector_lean": "bull",
            },
        }
    return fake


def test_sector_output_is_scrubbed_only_when_the_industry_block_was_spliced(monkeypatch):
    row = _row("NVDA")
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    leaky = _leaky_prose(analyst, str(row["sub_industry_code"]))
    profile = {"ticker": "NVDA", "company_name": "NVIDIA", "sector": "Technology"}
    fake = _sector_llm(leaky)
    monkeypatch.setattr(sector_agents.llm, "chat_json", fake)
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)

    finding = sector_agents.run_sector_agent(profile, {}, industry_group=row)
    displayed = {
        "headline": finding.headline, "summary": finding.summary, "key_points": finding.key_points,
        "macro_alignment": finding.data["macro_alignment"],
        "bull_bear_analysis": finding.data["bull_bear_analysis"],
        "industry_group": finding.data["industry_group"],
    }
    assert _leaks(displayed, row, analyst) == []
    assert analyst.label in finding.headline
    assert "bull_bear_parse_failed" not in finding.data     # scrubbing kept the contract's shape
    assert analyst.label in fake.prompt and "gics" not in fake.prompt.lower()

    # Routing off: no block was spliced, so the model's output is untouched
    # (the routing-off memo must stay byte-identical to the base golden).
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", False)
    finding = sector_agents.run_sector_agent(profile, {}, industry_group=None)
    assert finding.headline == leaky[:len(finding.headline)] and "GICS" in finding.headline


def test_label_accessor_failure_does_not_stub_the_sector_card(monkeypatch):
    """REGRESSION (W4 critique, high): the sector analyst's except path
    built its provenance summary INSIDE the handler, with the same accessor
    that had just failed. A label the file lacks raised twice, and the
    whole sector card became an "unavailable" crash stub. It now degrades
    to the sector-only read with the error on its provenance block."""
    def boom(code):
        raise il.UnknownLabel(f"no public label for {code!r}")

    monkeypatch.setattr(il, "label", boom)
    monkeypatch.setattr(il, "slug", boom)
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    row = _row("MSFT")

    finding = sector_agents.run_sector_agent({"ticker": "MSFT", "company_name": "Microsoft"}, {},
                                             industry_group=row)
    assert finding.agent == "Sector Analyst"
    assert finding.headline != "Sector Analyst unavailable"
    ig = finding.data["industry_group"]
    assert ig["error"] == "industry group block unavailable"
    assert ig["state"] == "mapped" and ig["label"] is None
    assert ig["mapping_caveat"] == il.PUBLIC_MAPPING_CAVEAT
    _assert_no_code_value(ig, row)

    # End to end: the memo keeps a real sector card; only the industry
    # analyst (whose label is genuinely unavailable) degrades.
    memo = graph.run_stock_memo("MSFT")
    assert memo.sector_agent_view.headline != "Sector Analyst unavailable"
    assert "Sector Analyst" not in memo.degraded_agents
    assert memo.sector_agent_view.data["industry_group"]["error"] == "industry group block unavailable"


def _assert_no_code_value(obj: Any, row: dict | None) -> None:
    """No value (and no key) equals one of the row's 2/4/6/8-digit codes —
    `_leaks` only catches a sector code in 'sector 45' / '(45)' form."""
    row = row or {}
    codes = {str(c) for c in (row.get("sector_code"), row.get("industry_group_code"),
                              row.get("industry_code"), row.get("sub_industry_code"),
                              *(row.get("sub_industry_codes") or [])) if c}
    values = [v for v in _strings(obj)] + [v for v in _values(obj) if not isinstance(v, str)]
    assert not [v for v in values if str(v) in codes], (obj, codes)


def _values(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _values(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _values(v)]
    return [obj]


def test_unavailable_group_summary_cannot_raise():
    for bad in (None, {}, {"state": "mapped", "industry_group_code": "9999"},
                {"state": "stale", "evidence": "not-a-dict", "industry_group_code": "4530"},
                {**_row("NVDA"), "industry_group_code": "9999"}):
        out = ia.unavailable_group_summary(bad, "x")
        assert out["error"] == "x" and "gics" not in json.dumps(out).lower()
        _assert_no_code_value(out, bad if isinstance(bad, dict) else None)


_PUBLIC_SUMMARY_KEYS = {
    "slug", "label", "sector_label", "name", "state", "routed_state", "source", "source_label",
    "author", "source_as_of", "provider_industry", "taxonomy_version", "report_version",
    "mapping_caveat",
}


@pytest.mark.parametrize("ticker", TICKERS)
def test_industry_group_summary_emits_public_values_only(ticker):
    """The summary's exact shape: slug, label, public caveat and key plus
    classification provenance — no code key, no code value, and already a
    fixed point of the public projection."""
    row = _row(ticker)
    analyst = ia.analyst_for_classification(row)
    for summary in (ia.industry_group_summary(row, analyst), ia.industry_group_summary(row, None)):
        assert set(summary) == _PUBLIC_SUMMARY_KEYS
        _assert_no_code_value(summary, row)
        assert il.project_public(summary) == summary
    assert ia.industry_group_summary(row, analyst)["taxonomy_version"] == il.PUBLIC_TAXONOMY_KEY


# --- a routed demo memo, end to end ---------------------------------------------------------


@pytest.mark.parametrize("ticker", TICKERS)
def test_routed_demo_memo_industry_surfaces_carry_no_code_or_registry_name(monkeypatch, ticker):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    memo = graph.run_stock_memo(ticker)
    row = _row(ticker)
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    finding = memo.extra_agent_views["industry_group"]
    assert _leaks(finding.model_dump(), row, analyst) == []     # long_form_report included
    assert _leaks(memo.sector_agent_view.data["industry_group"], row, analyst) == []
    assert memo.sector_agent_view.data["industry_group"]["label"] == analyst.label
