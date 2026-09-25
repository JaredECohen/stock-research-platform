"""W2a: template-filled memo sections are hidden on the way out.

Owner decision 2 (2026-09-24): publish the memo, but a section a template
filled reads "Unavailable in this version." — stored payloads unchanged.

The scenario tests run the classifier over five saved production memos,
minimized for this public repository (`app/scripts/trim_memo_section_fixtures.py`
kept the signature-bearing template text, flags and numbers and replaced
every piece of analyst prose with synthetic sentences). META v1, the one fully
agentic memo, is the false-positive control: it must trip nothing.

`test_signatures_match_live_producers` is the drift guard: each signature is
produced verbatim by calling its live producer deterministically, so a
reworded fallback fails here instead of silently un-hiding template text.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

from app.agents import graph, llm
from app.config import settings
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CatalystItem,
    CriticReview,
    MemoQuality,
    MispricingThesis,
    NumberCheck,
    NumberClaim,
    RiskItem,
    StockMemoOut,
    TechnicalSignals,
    ValuationVerdict,
    WithheldItem,
)
from app.services import memo_sections as ms
from app.services.memo_sections import SIG, UNAVAILABLE_TEXT, compute_availability, present_memo
from app.tests.factories import make_finding, make_memo

FIXTURES = Path(__file__).parent / "fixtures" / "memo_sections"
PM_TAIL = SIG["pm_view_tail"].text
KD = SIG["bb_key_disagreement"].text


def _fixture(name: str) -> StockMemoOut:
    return StockMemoOut.model_validate(json.loads((FIXTURES / f"{name}.json").read_text()))


def _status(av: dict, key: str) -> tuple[str, str | None]:
    return av[key].status, av[key].reason


@pytest.fixture
def no_llm(monkeypatch):
    """Every producer takes its deterministic branch, keys or not."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: None)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Scenario tests over the minimized production memos
# ---------------------------------------------------------------------------

def test_meta_v1_agentic_sections_all_available():
    """The false-positive control: a fully agentic memo trips no signature."""
    memo = _fixture("meta_v1")
    av = compute_availability(memo)
    for key in ("sector_agent_view", "earnings_agent_view", "filing_agent_view",
                "valuation_agent_view", "comps_agent_view", "macro_sensitivity",
                "one_sentence_thesis", "final_pm_view", "confidence_score", "rating_label",
                "mispricing_thesis", "bull_case", "bear_case", "sector_synthesis", "key_risks",
                "thesis_breakers", "catalysts", "final_verdict", "valuation_verdict", "dcf_summary"):
        assert av[key].status == "available", (key, av[key])
    assert _status(av, "technical_agent_view") == ("unavailable", "skipped_by_intake")
    assert _status(av, "risk_committee_challenge") == ("unavailable", "critic_not_run")
    assert _status(av, "portfolio_fit") == ("unavailable", "template_always")
    shown = present_memo(memo)
    assert shown.bull_case == memo.bull_case and shown.bear_case == memo.bear_case
    assert shown.one_sentence_thesis == memo.one_sentence_thesis
    assert shown.final_verdict == memo.final_verdict


def test_googl_prepflag_template_pm_hidden_without_any_flag():
    """GOOGL (2026-09-10) predates every fallback flag: signatures alone must
    find the deterministic PM view, mispricing card and sector read."""
    memo = _fixture("googl_live_prepflag")
    assert memo.degraded_agents == [] and memo.degradation_events == []
    assert memo.section_provenance == {}
    av = compute_availability(memo)
    for key in ("final_pm_view", "one_sentence_thesis", "confidence_score"):
        assert _status(av, key) == ("unavailable", "template_fallback"), key
        assert "signature:pm_view_tail" in av[key].basis
    assert _status(av, "rating_label") == ("degraded", "pm_view_unavailable")
    assert _status(av, "mispricing_thesis") == ("unavailable", "template_fallback")
    assert "signature:mispricing_gap_dcf" in av["mispricing_thesis"].basis
    assert _status(av, "sector_agent_view") == ("unavailable", "template_fallback")
    assert _status(av, "sector_synthesis") == ("unavailable", "derived_from_hidden")
    bull = av["bull_case"]
    assert (bull.status, bull.headline_hidden, bull.hidden_items) == ("degraded", True, 1)
    assert _status(av, "risk_committee_challenge") == ("unavailable", "critic_not_run")
    assert _status(av, "final_verdict") == ("unavailable", "derived_from_hidden")
    shown = present_memo(memo)
    # The LLM-named DCF drivers stay; the template item and headline go.
    assert shown.bull_case.headline == UNAVAILABLE_TEXT
    assert all(p.startswith(("DCF driver — Synthetic", "DCF bull case implies"))
               for p in shown.bull_case.key_points)
    assert len(shown.bull_case.key_points) == len(memo.bull_case.key_points) - 1
    assert PM_TAIL not in json.dumps(shown.model_dump(mode="json"))


def test_msft_llm_pm_kept_while_template_sector_hidden():
    memo = _fixture("msft_live")
    av = compute_availability(memo)
    for key in ("one_sentence_thesis", "final_pm_view", "confidence_score", "rating_label",
                "mispricing_thesis"):
        assert av[key].status == "available", key
    assert _status(av, "sector_agent_view") == ("unavailable", "template_fallback")
    assert _status(av, "valuation_agent_view") == ("unavailable", "template_fallback")
    assert "flag:deterministic_fallback" in av["valuation_agent_view"].basis
    assert _status(av, "filing_agent_view") == ("unavailable", "skipped_by_intake")
    assert _status(av, "technical_agent_view") == ("unavailable", "skipped_by_intake")
    shown = present_memo(memo)
    assert "Execution risk on the dominant driver." not in [r.detail for r in shown.key_risks]
    assert shown.key_risks and all(r.detail.startswith("DCF driver — Synthetic") for r in shown.key_risks)
    assert av["key_risks"].status == "degraded" and av["key_risks"].hidden_items == 1
    assert [c.title for c in shown.catalysts] == [c.title for c in memo.catalysts
                                                  if c.title.startswith("Next earnings: ")]
    # Critique delta 11: the verdict keeps the LLM thesis and loses only the
    # canned blurbs the hidden sector view contributed.
    assert _status(av, "final_verdict") == ("degraded", "partial_template")
    assert KD in memo.final_verdict and KD not in shown.final_verdict
    assert "Cohort placement: see sector view" not in shown.final_verdict
    assert memo.one_sentence_thesis in shown.final_verdict


def test_aapl_demo_memo_template_end_to_end():
    memo = _fixture("aapl_demo")
    av = compute_availability(memo)
    for key in ("sector_agent_view", "filing_agent_view", "valuation_agent_view",
                "macro_sensitivity", "technical_agent_view", "final_pm_view",
                "one_sentence_thesis", "confidence_score", "mispricing_thesis", "sector_synthesis",
                "risk_committee_challenge", "final_verdict", "key_risks", "bull_case"):
        assert av[key].status == "unavailable", key
    assert _status(av, "earnings_agent_view") == ("unavailable", "no_source_data")
    for key in ("valuation_verdict", "dcf_summary", "comps_agent_view", "scorecard",
                "business_summary"):
        assert av[key].status == "available", key
    bear = av["bear_case"]
    assert (bear.status, bear.reason, bear.headline_hidden, bear.hidden_items) == (
        "degraded", "partial_template", True, 2)
    shown = present_memo(memo)
    assert shown.bear_case.key_points == [p for p in memo.bear_case.key_points
                                          if p.startswith(("DCF bear case implies", "Risk lens: "))]
    assert shown.key_risks == []
    assert [c.title for c in shown.catalysts] == [c.title for c in memo.catalysts
                                                  if c.title.startswith("Next earnings: ")]
    assert shown.comps_agent_view == memo.comps_agent_view.model_copy(
        update={"long_form_report": None})


def test_abbv_v7_legacy_shapes():
    memo = _fixture("abbv_v7_patch")
    av = compute_availability(memo)
    # The FIX-004 compatibility projection (headline "", a patched LLM point)
    # is not a template.
    assert av["bull_case"].status == "available"
    assert _status(av, "bear_case") == ("unavailable", "template_fallback")
    assert _status(av, "one_sentence_thesis") == ("unavailable", "derived_from_hidden")
    # An LLM patch imitating "Research view: …" is not the template tail.
    assert memo.final_pm_view.startswith("Research view: ")
    assert av["final_pm_view"].status == "available"
    for key, sig in (("earnings_agent_view", "earnings_det_headline"),
                     ("filing_agent_view", "filing_det_pair"),
                     ("macro_sensitivity", "macro_det_pair"),
                     ("technical_agent_view", "technical_det_tail"),
                     ("valuation_agent_view", "valuation_det_summary")):
        assert av[key].status == "unavailable" and f"signature:{sig}" in av[key].basis, key
    assert _status(av, "mispricing_thesis") == ("unavailable", "not_produced")


def test_legacy_signatures_pinned_to_evidence():
    """Removed producers still classify through the stored evidence."""
    googl = _fixture("googl_live_prepflag")
    assert googl.risk_committee_challenge.overall_assessment == SIG["critic_legacy_rule_based"].text
    assert "signature:critic_legacy_rule_based" in compute_availability(googl)[
        "risk_committee_challenge"].basis
    abbv = _fixture("abbv_v7_patch")
    # The older filing extract wrote "<type> dated <date>:" (colon).
    assert abbv.filing_agent_view.summary.startswith("10-K dated 2026-02-20:")
    assert SIG["filing_det_summary"].matches(abbv.filing_agent_view.summary)


@pytest.mark.skipif(
    not (Path(os.environ.get("MM_EVIDENCE_DIR") or Path(__file__).parents[3] / "docs" / "reviews")
         / "2026-09-13-META-v1.json").exists(),
    reason="local-only: the full evidence files are gitignored",
)
def test_minimized_fixtures_classify_like_evidence():
    """The committed fixtures prove what the full production memos prove."""
    from app.scripts import trim_memo_section_fixtures as trim
    source = Path(os.environ.get("MM_EVIDENCE_DIR") or trim.DEFAULT_SOURCE_DIR)
    for name, memo in trim.load_evidence(source).items():
        full = compute_availability(StockMemoOut.model_validate(memo))
        assert trim.same_verdicts(full, compute_availability(_fixture(name))), name
        committed = (FIXTURES / f"{name}.json").read_text()
        assert json.loads(committed) == trim.minimize(name, memo), name


# ---------------------------------------------------------------------------
# Signatures pinned to their live producers
# ---------------------------------------------------------------------------

def _producer_texts() -> dict[str, Any]:
    """{signature id: callable returning the producer's text} — each call
    runs the real producer on its deterministic branch."""
    from app.agents import (
        critic_agent,
        earnings_agent,
        filing_agent,
        industry_analysts,
        macro_agent,
        safe_runner,
        sector_agents,
        technical_agent,
        valuation_agent,
    )
    from app.agents.earnings_qoq import run_earnings_qoq_delta
    from app.services import scenario_assumptions
    from app.services.valuation_service import build_dcf

    profile = {"ticker": "PINT", "sector": "Technology", "sub_industry": "Software", "company_name": "Pin"}

    def bb():
        research = {
            "kpi_placements": {"EV_EBITDA": {"quartile": 4}, "revenue_growth": {"quartile": 1},
                               "operating_margin": {"quartile": 2}},
            "trends": {"cohort_op_margin_delta": -0.02}, "regime": "mixed",
        }
        return sector_agents._deterministic_bull_bear_analysis(profile, research)

    def bb_empty():
        return sector_agents._deterministic_bull_bear_analysis(profile, {})

    def thesis(rating: str, findings: dict[str, AgentFinding] | None = None) -> str:
        # W2b: the builder takes the verdict word; a rating-derived word is
        # what every legacy (signature-bearing) thesis was built with.
        from app.agents.memo_quality import rating_word
        return graph._build_thesis_from_findings(dict(profile), findings or {}, None, "PINT",
                                                  verdict_word=rating_word(rating))

    def mispricing(verdict: str, upside: float | None) -> MispricingThesis:
        memo = make_memo(ticker="PINT", one_sentence_thesis="",
                         valuation_verdict=ValuationVerdict(verdict=verdict, dcf_base_upside=upside))
        return graph._build_mispricing_fallback(memo)

    def dcf_driver(side: str) -> str:
        dcf = build_dcf("NVDA")
        assert dcf is not None
        _, bull, _, bear = scenario_assumptions._deterministic_fallback(
            {"sector": "Technology", "industry": "Software"}, dcf.base.assumptions)
        d = (bull if side == "bull" else bear)[0]
        return f"DCF driver — {d.name}: {d.rationale}".rstrip(": ")

    def qoq() -> AgentFinding:
        import app.agents.earnings_qoq as eq
        orig_prior, orig_llm = eq._prior_structured_extraction, eq._llm_delta
        eq._prior_structured_extraction = lambda *_a, **_k: {"period": "2025Q1", "overall_tone": "measured"}
        eq._llm_delta = lambda *_a, **_k: None
        try:
            out = run_earnings_qoq_delta("PINT", {"period": "2025Q2", "overall_tone": "cautious"})
        finally:
            eq._prior_structured_extraction, eq._llm_delta = orig_prior, orig_llm
        assert out is not None
        return out

    def critic_crash():
        def crash():
            raise RuntimeError("boom")
        review = safe_runner.safe_critic(crash)
        assert review is not None
        return review.overall_assessment

    signals = TechnicalSignals(trend="up", momentum="positive", sma_50=10.0, sma_200=9.0,
                               sma_50_above_200=True, last_price=11.0, rsi_14=55.0)
    filing = [{"type": "10-K", "filing_date": "2026-02-20", "mda": "Revenue rose on demand. " * 40,
               "risk_factors": []}]
    return {
        "pm_view_tail": lambda: graph._pm_synthesis(dict(profile), {}, None)["final_pm_view"],
        "thesis_builder_underpricing": lambda: thesis("Bullish", {"valuation": make_finding(
            "Valuation Analyst", headline="Durable cash generation outruns the multiple")}),
        "thesis_builder_multiple": lambda: thesis("Bearish", {"valuation": make_finding(
            "Valuation Analyst", headline="Durable cash generation outruns the multiple")}),
        "thesis_builder_no_headline": lambda: thesis("Bullish"),
        "thesis_builder_floor": lambda: thesis("Neutral", {"valuation": make_finding(
            "Valuation Analyst", headline="Durable cash generation outruns the multiple")}),
        "thesis_builder_no_edge": lambda: thesis("Neutral"),
        "mispricing_gap_fair": lambda: mispricing("fairly_priced", None).gap,
        "mispricing_gap_dcf": lambda: mispricing("undervalued", -0.54).gap,
        "mispricing_gap_blend": lambda: mispricing("overvalued", None).gap,
        "mispricing_our_view_pointer": lambda: mispricing("undervalued", None).our_view,
        "bb_key_disagreement": lambda: bb().key_disagreement,
        "bb_bull_headline": lambda: bb().bull_case.headline,
        "bb_bear_headline": lambda: bb().bear_case.headline,
        "bb_bear_valuation_quartile": lambda: bb().bear_case.key_points[-1],
        "bb_bear_margin_compressing": lambda: bb().bear_case.key_points[0],
        "bb_bear_execution": lambda: bb_empty().bear_case.key_points[0],
        "bb_bull_growth_quartile": lambda: bb().bull_case.key_points[0],
        "bb_bull_margin": lambda: bb().bull_case.key_points[1],
        "case_bull_last_resort": lambda: graph._bull_case({}, make_finding("Valuation Analyst"), None, None, {}).key_points[0],
        "case_bear_last_resort": lambda: graph._bear_case({}, None, None, {}).key_points[0],
        "case_headline_bull_blend": lambda: graph._bull_case({}, make_finding("Valuation Analyst"), None, None, {}).headline,
        "case_headline_bear_blend": lambda: graph._bear_case({}, None, None, {}).headline,
        "case_headline_bull_synthesis": lambda: graph._bull_case(
            {}, make_finding("Valuation Analyst"), None,
            make_finding(data={"bull_bear_analysis": {"bull_case": {"key_points": []}}}), {}).headline,
        "case_headline_bear_synthesis": lambda: graph._bear_case(
            {}, None, make_finding(data={"bull_bear_analysis": {"bear_case": {"key_points": []}}}), {}).headline,
        "case_profile_tailwind": lambda: graph._bull_case({"drivers": ["Cloud"]}, make_finding(
            "Valuation Analyst", headline="x", summary=""), None, None, {}).key_points[0],
        "case_profile_headwind": lambda: sector_agents._deterministic_bull_bear_analysis(
            {**profile, "risks": ["Pricing"]}, {}).bear_case.key_points[0],
        "dcf_template_driver": lambda: dcf_driver("bull"),
        "catalyst_next_update": lambda: graph._catalysts({}, {"period": "2025Q2"}, {}, None)[0].title,
        "critic_rule_based": lambda: critic_agent.run_critic(
            {"ticker": "PINT", "rating_label": "Neutral"}).overall_assessment,
        "critic_unavailable": critic_crash,
        "earnings_no_transcript": lambda: earnings_agent.run_earnings_agent(profile, None, None).headline,
        "earnings_det_headline": lambda: earnings_agent.run_earnings_agent(
            profile, {"period": "2025Q2", "prepared_remarks": "Revenue grew. " * 200}, {}).headline,
        "filing_no_filings": lambda: filing_agent.run_filing_agent(profile, []).headline,
        "filing_det_headline": lambda: filing_agent.run_filing_agent(profile, filing).headline,
        "filing_det_summary": lambda: filing_agent.run_filing_agent(profile, filing).summary,
        "valuation_det_summary": lambda: valuation_agent.run_valuation_agent(
            profile, {"PE": 20.0, "EV_EBITDA": 12.0}, None).summary,
        "macro_det_headline": lambda: macro_agent.run_macro_agent(profile, "soft landing").headline,
        "macro_det_summary": lambda: macro_agent.run_macro_agent(profile, "soft landing").summary,
        "technical_det_tail": lambda: technical_agent._deterministic_summary(profile, signals)["summary"],
        "technical_no_ticker": lambda: technical_agent.run_technical_agent({}).headline,
        "industry_no_mapping": lambda: industry_analysts._unmapped_finding("PINT", None).headline,
        "qoq_no_llm_points": lambda: qoq().key_points[0],
        "qoq_no_llm_summary": lambda: qoq().summary,
    }


def _source_pins() -> dict[str, tuple[Any, str]]:
    """{signature id: (producer, literal it must still contain)} for texts
    no deterministic call can reach: compose-stage stand-ins, a price feed
    that must fail, the sector read-out that needs the cohort pipeline."""
    from app.agents import industry_analysts, sector_agents, technical_agent
    return {
        "pm_hard_fallback": (graph._compose_memo, SIG["pm_hard_fallback"].text),
        "thesis_research_draft": (graph._compose_memo, 'f"Research draft for {profile.get('),
        "case_headline_bull_unavailable": (graph._compose_memo,
                                           SIG["case_headline_bull_unavailable"].text),
        "case_headline_bear_unavailable": (graph._compose_memo,
                                           SIG["case_headline_bear_unavailable"].text),
        "critic_pending": (graph._compose_memo, SIG["critic_pending"].text),
        "technical_no_prices": (technical_agent.run_technical_agent,
                                ": price series unavailable for technical read."),
        "technical_short_history": (technical_agent.run_technical_agent,
                                    ": insufficient price history for technical read."),
        "sector_det_summary": (sector_agents.run_sector_agent,
                               "Cohort of {cohort['size']} peers selected on "
                               "{cohort['selection_basis']} basis."),
        "industry_det_chain": (industry_analysts._deterministic_finding,
                               SIG["industry_det_chain"].text),
    }


# Pinned through stored evidence: the producer was removed (9da68fd).
_EVIDENCE_PINNED = {"critic_legacy_rule_based"}
_SOURCE_PINNED = {
    "pm_hard_fallback", "thesis_research_draft", "case_headline_bull_unavailable",
    "case_headline_bear_unavailable", "critic_pending", "technical_no_prices",
    "technical_short_history", "sector_det_summary", "industry_det_chain",
}


def test_every_signature_has_a_pin():
    assert set(_source_pins()) == _SOURCE_PINNED
    assert set(SIG) == set(_producer_texts()) | _SOURCE_PINNED | _EVIDENCE_PINNED


@pytest.mark.parametrize("sig_id", sorted(set(SIG) - _EVIDENCE_PINNED - _SOURCE_PINNED))
def test_signatures_match_live_producers(sig_id, no_llm):
    text = _producer_texts()[sig_id]()
    assert SIG[sig_id].matches(text), (sig_id, text)


@pytest.mark.parametrize("sig_id", sorted(_SOURCE_PINNED))
def test_source_pinned_signatures_still_in_producers(sig_id):
    producer, literal = _source_pins()[sig_id]
    assert literal in inspect.getsource(producer), sig_id


def test_no_edge_signature_ignores_prompt_phrase():
    """The PM prompt offers "fairly priced on our work, no actionable edge"
    as a legitimate LLM call; only the builder's em-dash sentence matches."""
    from app.agents import prompts
    assert "fairly priced on our work, no actionable edge" in " ".join(
        prompts.PM_SYNTHESIS_PROMPT.split())
    sig = SIG["thesis_builder_no_edge"]
    assert sig.matches("PINT is fairly priced on our work — no actionable edge in technology. Own it.")
    for llm_text in (
        "MSFT is fairly priced on our work, no actionable edge — the floor is hard.",
        "Fairly priced on our work, no actionable edge; watch renewal rates.",
        "We think MSFT is fairly priced on our work — no actionable edge in the near term",
    ):
        assert not sig.matches(llm_text), llm_text
    memo = make_memo(one_sentence_thesis="MSFT is fairly priced on our work, no actionable edge — "
                                         "the floor is hard.")
    assert compute_availability(memo)["one_sentence_thesis"].status == "available"


# ---------------------------------------------------------------------------
# Presenter contract
# ---------------------------------------------------------------------------

_FIXTURE_NAMES = ("aapl_demo", "abbv_v7_patch", "googl_live_prepflag", "meta_v1", "msft_live")


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_present_is_pure_and_idempotent(name):
    memo = _fixture(name)
    before = memo.model_dump(mode="json")
    once = present_memo(memo)
    assert memo.model_dump(mode="json") == before, "the input must not be mutated"
    assert present_memo(once) == once
    assert once.section_availability
    StockMemoOut.model_validate(once.model_dump(mode="json"))
    # Numbers never change; renderers consult the map instead.
    assert once.confidence_score == memo.confidence_score
    assert once.scores == memo.scores and once.dcf_summary == memo.dcf_summary
    for key, _name in ms.LLM_ANALYST_FIELDS.items():
        a, b = getattr(memo, key), getattr(once, key)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.confidence == b.confidence
    assert once.degraded_agents == memo.degraded_agents
    assert once.section_provenance == memo.section_provenance


def test_placeholder_text_is_owner_wording():
    assert UNAVAILABLE_TEXT == "Unavailable in this version."
    shown = present_memo(_fixture("googl_live_prepflag"))
    for value in (shown.final_pm_view, shown.one_sentence_thesis, shown.final_verdict,
                  shown.portfolio_fit, shown.sector_agent_view.headline,
                  shown.sector_agent_view.summary, shown.risk_committee_challenge.overall_assessment,
                  shown.mispricing_thesis.our_view, shown.mispricing_thesis.gap,
                  shown.mispricing_thesis.consensus_view, shown.bull_case.headline,
                  shown.bear_case.headline):
        assert value == UNAVAILABLE_TEXT
    assert shown.sector_agent_view.key_points == [] and shown.mispricing_thesis.falsifiers == []
    assert shown.risk_committee_challenge.challenges == []


def _sector_template_memo(**data: Any) -> StockMemoOut:
    sector = make_finding(
        "Sector Analyst", headline="Software cohort, regime: mixed",
        summary="Tech / Software regime read: mixed. Cohort of 5 peers selected on sub_industry basis.",
        key_points=["operating_margin: 30% vs cohort median 20% — top quartile"],
        long_form_report="## Sector\n\n### Sector synthesis\nCanned.\n\n### Analyst expansion\nLLM words.",
        data={"deterministic_fallback": "Sector LLM returned no usable output.", **data},
    )
    return make_memo(sector_agent_view=sector, one_sentence_thesis="TEST is fairly priced — cloud.")


def test_hidden_finding_data_is_allowlisted():
    memo = _sector_template_memo(
        bull_bear_analysis={"bull_case": {"headline": "b", "key_points": []},
                            "bear_case": {"headline": "c", "key_points": []},
                            "key_disagreement": KD, "sector_synthesis": "s"},
        structured={"x": 1}, narrative="LLM narrative", kpi_placements={"ROIC": {"quartile": 1}},
        macro_broadcast={"regime": "mixed"}, a_future_key="leaks?",
    )
    shown = present_memo(memo)
    data = shown.sector_agent_view.data
    for gone in ("bull_bear_analysis", "structured", "narrative", "a_future_key"):
        assert gone not in data
    assert data["kpi_placements"] == {"ROIC": {"quartile": 1}}
    assert data["macro_broadcast"] == {"regime": "mixed"}
    assert data["deterministic_fallback"]
    assert shown.sector_agent_view.long_form_report is None


def test_hidden_sector_keeps_news_alerts_and_macro_alignment():
    alerts = [{"title": "Company wins contract", "severity": "material"}]
    shown = present_memo(_sector_template_memo(pending_news_alerts=alerts, macro_alignment="favored"))
    assert shown.sector_agent_view.headline == UNAVAILABLE_TEXT
    assert shown.sector_agent_view.data["pending_news_alerts"] == alerts
    assert shown.sector_agent_view.data["macro_alignment"] == "favored"


def test_long_form_shows_only_analyst_expansion(monkeypatch):
    """Critique delta 1: the drill-down's deterministic body restates the
    finding in canned frames; only the LLM expansion is the analyst's."""
    from app.agents import long_form
    finding = make_finding("Valuation Analyst", headline="Cash flow outruns the multiple",
                           summary="The model reads cheap.", key_points=["P/E 12x"])
    body = long_form.deterministic_long_form(finding, ticker="TEST", agent_name="Valuation Analyst")
    assert "### Analyst expansion" not in body  # drift guard on the marker
    monkeypatch.setattr(settings, "enable_long_form_reports", True)
    monkeypatch.setattr(long_form, "_enriched_long_form", lambda *a, **k: "The analyst's own words.")
    enriched = long_form.build_long_form_report(finding, ticker="TEST", agent_name="Valuation Analyst")
    with_lf = finding.model_copy(update={"long_form_report": enriched})
    without_lf = finding.model_copy(update={"long_form_report": body})
    memo = make_memo(valuation_agent_view=with_lf, earnings_agent_view=without_lf)
    av = compute_availability(memo)
    shown = present_memo(memo)
    assert shown.valuation_agent_view.long_form_report == "The analyst's own words."
    assert av["valuation_agent_view.long_form_report"].status == "available"
    assert shown.earnings_agent_view.long_form_report is None
    assert _status(av, "earnings_agent_view.long_form_report") == ("unavailable", "template_fallback")
    # The canned sector-synthesis / KD block never survives, whatever the
    # sector view's status.
    sector = make_finding("Sector Analyst", long_form_report=(
        f"## S\n\n### Sector synthesis\n{KD}\n\n### Analyst expansion\nWords."))
    kept = present_memo(make_memo(sector_agent_view=sector))
    assert kept.sector_agent_view.long_form_report == "Words."


def test_quality_number_check_drops_hidden_claims():
    claim = NumberClaim(field="one_sentence_thesis", start=0, end=3, raw="12x", status="traced")
    kept = NumberClaim(field="comps_agent_view.summary", start=0, end=3, raw="12x", status="traced")
    memo = make_memo(
        final_pm_view=f"Research view: Neutral. X. {PM_TAIL}", one_sentence_thesis="X is 12x.",
        quality=MemoQuality(number_check=NumberCheck(
            checked=True, claims=[claim, kept],
            withheld=[WithheldItem(field="final_pm_view", index=0, text="t")])),
    )
    shown = present_memo(memo)
    assert shown.quality is not None and shown.quality.number_check is not None
    assert shown.quality.number_check.claims == [kept]
    assert shown.quality.number_check.withheld == []
    assert memo.quality is not None and memo.quality.number_check is not None
    assert len(memo.quality.number_check.claims) == 2  # input untouched


def test_legacy_case_original_value_filtered():
    template_point = "Quality + growth profile supports a premium versus peers."
    memo = make_memo(
        bull_case=BullBearCase(headline="", key_points=[template_point, "LLM patch point."]),
        degradation_events=[{"agent": "Stored memo compatibility", "error_type": "LegacyCaseShape",
                             "field": "bull_case",
                             "original_value": [template_point, {"key_point": "LLM patch point."}]}],
        degraded_agents=["Stored memo compatibility"],
    )
    shown = present_memo(memo)
    assert shown.bull_case.key_points == ["LLM patch point."]
    assert shown.degradation_events[0]["original_value"] == [{"key_point": "LLM patch point."}]


def test_degraded_agents_union_with_events():
    """A banner entry with no event (rows written before events existed)
    and an event with no banner entry both mark the analyst."""
    only_banner = make_memo(degraded_agents=["Sector Analyst"])
    assert _status(compute_availability(only_banner), "sector_agent_view") == (
        "unavailable", "template_fallback")
    only_event = make_memo(degradation_events=[
        {"agent": "Earnings Analyst", "error_type": "DeterministicFallback", "message": "m"}])
    assert _status(compute_availability(only_event), "earnings_agent_view") == (
        "unavailable", "template_fallback")
    # A retrieval failure is recorded under the analyst too, but it is not a
    # fallback: the finding's own flags decide.
    retrieval = make_memo(
        degraded_agents=["Filing Analyst"],
        degradation_events=[{"agent": "Filing Analyst", "error_type": "OperationalError", "message": "m"}],
        filing_agent_view=make_finding("Filing Analyst", data={"retrieval_failed": True}),
    )
    assert _status(compute_availability(retrieval), "filing_agent_view") == ("degraded", "reduced_inputs")


def test_qoq_without_llm_is_unavailable():
    qoq = AgentFinding(agent="Earnings QoQ Delta", headline="QoQ delta vs 2025Q1: no material change",
                       summary="QoQ delta computed vs 2025Q1; no major reversals detected.",
                       key_points=["No material differences."], confidence=0.5)
    av = compute_availability(make_memo(earnings_qoq_delta=qoq))
    assert _status(av, "earnings_qoq_delta") == ("unavailable", "template_fallback")
    assert present_memo(make_memo(earnings_qoq_delta=qoq)).earnings_qoq_delta.headline == UNAVAILABLE_TEXT


def test_patch_chain_evaluates_base_snapshot():
    """Critique delta 6: a patch never re-runs the PM synthesis, so the PM
    rules read the chain's base; a patched field is restored unless an exact
    signature still matches it."""
    base = _fixture("googl_live_prepflag")
    patched = base.model_copy(deep=True)
    patched.final_pm_view = "The PM re-read the news: ad demand held."
    patched.one_sentence_thesis = "GOOGL is fairly priced — ad demand held."
    av = compute_availability(patched, patched_fields=frozenset({"final_pm_view", "one_sentence_thesis"}),
                              base=base)
    assert av["final_pm_view"].status == "available" and av["final_pm_view"].basis == ["patched:final_pm_view"]
    assert av["one_sentence_thesis"].status == "available"
    # The confidence and rating still rest on the template PM (±15 clamp).
    assert _status(av, "confidence_score") == ("unavailable", "template_fallback")
    assert any(b.startswith("base:") for b in av["confidence_score"].basis)
    assert _status(av, "rating_label") == ("degraded", "pm_view_unavailable")
    # The fallback card quotes the BASE thesis, not the patched one.
    assert _status(av, "mispricing_thesis") == ("unavailable", "template_fallback")
    # Exact signatures override "patched".
    sig_thesis = patched.model_copy(update={
        "one_sentence_thesis": "GOOGL is undervalued — x. The market is under-pricing the durable "
                               "part of the franchise."})
    av2 = compute_availability(sig_thesis, patched_fields=frozenset({"one_sentence_thesis"}), base=base)
    assert av2["one_sentence_thesis"].status == "unavailable"
    # An unwalkable chain hides the confidence even over an LLM base.
    meta = _fixture("meta_v1")
    av3 = compute_availability(meta, chain_complete=False)
    assert _status(av3, "confidence_score") == ("unavailable", "template_fallback")
    assert av3["confidence_score"].basis == ["patch_chain:incomplete"]


def test_rewritten_thesis_with_llm_headline_is_degraded():
    """Critique delta 11: a rewrite whose claim is an available analyst's
    headline keeps the claim (shown with a note); one with no LLM claim is
    the builder's template."""
    val = make_finding("Valuation Analyst", headline="Durable cash generation outruns the multiple")
    llm_claim = make_memo(
        valuation_agent_view=val,
        one_sentence_thesis="TEST is undervalued — Durable cash generation outruns the multiple. "
                            "The market is under-pricing the durable part of the franchise.",
        section_provenance={"v": 1, "llm_configured": True, "thesis": "rewrite", "mispricing": "pm"},
    )
    assert _status(compute_availability(llm_claim), "one_sentence_thesis") == ("degraded", "partial_template")
    no_claim = llm_claim.model_copy(update={
        "one_sentence_thesis": "TEST screens undervalued on the blended read, though no single "
                               "specialist headline defines the call."})
    assert _status(compute_availability(no_claim), "one_sentence_thesis") == (
        "unavailable", "template_fallback")


def test_mispricing_provenance_and_final_verdict_for_new_memos():
    memo = make_memo(
        mispricing_thesis=MispricingThesis(consensus_view="c", our_view="o", gap="g"),
        section_provenance={"v": 1, "llm_configured": True, "thesis": "pm", "mispricing": "fallback"},
        one_sentence_thesis="A real thesis.",
        final_verdict="PM final view: Neutral (confidence 60). A real thesis. Watch items: none flagged.",
    )
    av = compute_availability(memo)
    assert _status(av, "mispricing_thesis") == ("unavailable", "template_fallback")
    assert av["final_verdict"].status == "available"


def test_computed_roster_keys_match_roster():
    from app.agents import roster
    assert ms.COMPUTED_ROSTER_KEYS == {s.key for s in roster.AGENTS if not s.uses_llm_round0}
    fields = {s.memo_field: s.display_name for s in roster.AGENTS if s.memo_field}
    for field, name in {**ms.LLM_ANALYST_FIELDS, **ms.COMPUTED_ANALYST_FIELDS}.items():
        assert fields[field] == name


def test_unexpected_shape_classifies_available_not_500(monkeypatch):
    def boom(*_a, **_k):
        raise KeyError("surprise")
    monkeypatch.setattr(ms, "_classify", boom)
    memo = make_memo()
    av = compute_availability(memo)
    assert all(v.status == "available" and v.basis == ["unclassified"] for v in av.values())
    assert present_memo(memo).one_sentence_thesis == memo.one_sentence_thesis


def test_new_memo_llm_off_hides_llm_sections():
    """A memo written in CI (no keys, demo data) records the provenance the
    payload cannot recover later, and the presenter hides what a template
    wrote while keeping what is computed by design."""
    assert not settings.has_llm
    memo = graph.run_stock_memo("UNH")
    prov = memo.section_provenance
    assert prov["v"] == 1 and prov["llm_configured"] is False
    assert prov["thesis"] in ("pm", "rewrite") and prov["mispricing"] == "fallback"
    av = compute_availability(memo)
    for key in ("final_pm_view", "one_sentence_thesis", "mispricing_thesis",
                "sector_agent_view", "valuation_agent_view", "macro_sensitivity"):
        assert av[key].status == "unavailable", key
    for key in ("comps_agent_view", "dcf_summary", "valuation_verdict"):
        assert av[key].status in ("available", "degraded"), key
    # W2b 7(c) (contract C2): the confidence is EARNED — capped for the
    # template PM and sections — so it is shown, not hidden with the PM view.
    assert prov["confidence"] == "earned"
    assert av["confidence_score"].status == "available"
    assert memo.quality is not None and memo.quality.confidence is not None
    assert memo.confidence_score == memo.quality.confidence.final <= 40.0
    shown = present_memo(memo)
    assert shown.one_sentence_thesis == UNAVAILABLE_TEXT
    assert shown.comps_agent_view.headline == memo.comps_agent_view.headline


def test_finding_is_template_c2():
    assert ms.finding_is_template(make_finding(data={"deterministic_fallback": "x"}))
    assert ms.finding_is_template(make_finding(data={"no_mapping": True}))
    assert ms.finding_is_template(make_finding(headline="Earnings transcript unavailable."))
    assert not ms.finding_is_template(make_finding(headline="Cloud margins expand", summary="Real read."))


def test_unavailable_keys_and_is_hidden():
    shown = present_memo(_fixture("googl_live_prepflag"))
    keys = ms.unavailable_keys(shown.section_availability)
    assert isinstance(keys, set) and "one_sentence_thesis" in keys
    assert ms.is_hidden(shown, "one_sentence_thesis")
    assert not ms.is_hidden(make_memo(), "one_sentence_thesis")  # no map: available


def test_case_headline_built_from_a_hidden_finding_is_hidden():
    """`graph._bull_case` labels a hidden valuation headline ("Bull case: …");
    the label does not launder the template text."""
    val = make_finding("Valuation Analyst", headline="P/E 12.0x; EV/EBITDA 8.0x; trades at a discount",
                       data={"deterministic_fallback": "Valuation LLM returned no usable output."})
    memo = make_memo(valuation_agent_view=val, bull_case=BullBearCase(
        headline="Bull case: P/E 12.0x; EV/EBITDA 8.0x; trades at a discount",
        key_points=["An analyst's real bull point."]))
    av = compute_availability(memo)
    assert av["bull_case"].headline_hidden and av["bull_case"].status == "degraded"


def test_catalysts_and_critic_rules():
    memo = make_memo(
        catalysts=[CatalystItem(title="Next earnings update", detail="Watch for follow-on commentary."),
                   CatalystItem(title="Next earnings: 2026-10-28", detail="Quarterly print.")],
        risk_committee_challenge=CriticReview(overall_assessment="Critic agent unavailable for this run.",
                                              review_mode="unavailable"),
        key_risks=[RiskItem(title="Cohort positioning", detail=SIG["case_bear_last_resort"].text)],
    )
    shown = present_memo(memo)
    assert [c.title for c in shown.catalysts] == ["Next earnings: 2026-10-28"]
    assert shown.risk_committee_challenge.overall_assessment == UNAVAILABLE_TEXT
    assert shown.risk_committee_challenge.review_mode == "unavailable"
    assert shown.key_risks == []


# ---------------------------------------------------------------------------
# Fixer round: rules the first pass left untested or applied partially
# ---------------------------------------------------------------------------

_DET_BODY = "Synthetic deterministic drill-down body"


@pytest.mark.parametrize("name", ["msft_live", "meta_v1", "aapl_demo", "googl_live_prepflag"])
def test_round_findings_follow_the_long_form_rule(name):
    """Critique delta 1 applies to every drill-down a client can read, and
    GET /memo serves round_findings: a shown round finding keeps only the
    analyst's expansion, never the deterministic body in front of it."""
    raw = _fixture(name)
    shown = present_memo(raw)
    for r_raw, r_shown in zip(raw.round_findings, shown.round_findings, strict=True):
        for key, f in r_shown.findings.items():
            before = r_raw.findings[key].long_form_report
            if f.headline == UNAVAILABLE_TEXT:
                assert f.long_form_report is None
            elif before:
                assert f.long_form_report == ms._long_form_expansion(before), (r_shown.round, key)
    assert _DET_BODY not in json.dumps([r.model_dump(mode="json") for r in shown.round_findings])


def test_round_copies_inherit_the_grid_verdict_and_lose_the_template_block():
    """Round 0 is the fan-out of the grid views (and an agent's last round is
    its final finding): a copy of a finding the grid hides by memo-level
    evidence is hidden too, and a template bull/bear block never survives in
    a round sector copy."""
    raw = _fixture("meta_v1")
    sector_top = raw.sector_agent_view
    banner = raw.model_copy(update={"degraded_agents": [*raw.degraded_agents, "Sector Analyst"]})
    assert raw.round_findings[0].findings["sector"].headline == sector_top.headline
    av = compute_availability(banner)
    assert _status(av, "sector_agent_view") == ("unavailable", "template_fallback")
    shown = present_memo(banner)
    assert shown.sector_agent_view.headline == UNAVAILABLE_TEXT
    assert shown.round_findings[0].findings["sector"].headline == UNAVAILABLE_TEXT
    assert sector_top.headline not in shown.model_dump_json()

    # Sector view shown, synthesis hidden by the parse flag: the grid AND the
    # round copy drop the block.
    parse_failed = raw.model_copy(deep=True)
    parse_failed.sector_agent_view.data["bull_bear_parse_failed"] = True
    parse_failed.round_findings[0].findings["sector"].data["bull_bear_parse_failed"] = True
    av = compute_availability(parse_failed)
    assert av["sector_agent_view"].status == "available"
    assert _status(av, "sector_synthesis") == ("unavailable", "template_fallback")
    shown = present_memo(parse_failed)
    assert "bull_bear_analysis" not in shown.sector_agent_view.data
    assert "bull_bear_analysis" not in shown.round_findings[0].findings["sector"].data
    # META itself (LLM block) keeps both.
    clean = present_memo(raw)
    assert "bull_bear_analysis" in clean.round_findings[0].findings["sector"].data


def test_template_bull_bear_block_dropped_from_an_available_sector_view():
    """The sector card renders `data.bull_bear_analysis` (MemoCard's
    BullBearAnalysisBlock). An LLM sector view with a canned block keeps the
    view and loses the block, whichever trigger marks the block."""
    base = _fixture("meta_v1")
    bb = base.sector_agent_view.data["bull_bear_analysis"]

    def with_data(**extra: Any) -> StockMemoOut:
        m = base.model_copy(deep=True)
        m.sector_agent_view.data.update(extra)
        return m

    cases = {
        "signature": with_data(bull_bear_analysis={**bb, "key_disagreement": KD}),
        "parse_failed": with_data(bull_bear_parse_failed=True),
        "llm_off": base.model_copy(update={"section_provenance": {"v": 1, "llm_configured": False}}),
    }
    for label, memo in cases.items():
        av = compute_availability(memo)
        assert av["sector_synthesis"].status == "unavailable", label
        shown = present_memo(memo)
        assert "bull_bear_analysis" not in shown.sector_agent_view.data, label
        if label != "llm_off":  # llm_off hides every LLM analyst view
            assert av["sector_agent_view"].status == "available", label
            assert shown.sector_agent_view.headline == base.sector_agent_view.headline, label
    assert "bull_bear_analysis" in present_memo(base).sector_agent_view.data


def test_number_check_paths_follow_dropped_items():
    """Dropping items from a list section moves every later item up: a claim
    on a dropped item goes, a later claim's index is renumbered, and a
    withheld item whose text is template is dropped like a stored one."""
    raw = _fixture("msft_live")
    plan = ms._classify(raw, patched_fields=frozenset(), base=None, chain_complete=True)
    gone = plan.list_hidden["key_risks"]
    assert gone and plan.avail["key_risks"].status == "degraded"
    last = len(raw.key_risks) - 1
    assert last not in gone
    first_gone = gone[0]
    claims = [
        NumberClaim(field=f"key_risks[{first_gone}].detail", start=0, end=1, raw="1", status="traced"),
        NumberClaim(field=f"key_risks[{last}].detail", start=0, end=1, raw="1", status="traced"),
        NumberClaim(field="dcf_summary.base_implied_price", start=0, end=1, raw="1", status="traced"),
    ]
    withheld = [
        WithheldItem(field="key_risks", index=9, text=SIG["case_bear_last_resort"].text),
        WithheldItem(field="key_risks", index=10, text="A real risk whose figure did not trace."),
    ]
    memo = raw.model_copy(update={"quality": MemoQuality(number_check=NumberCheck(
        checked=True, claims=claims, withheld=withheld))})
    shown = present_memo(memo)
    nc = shown.quality.number_check
    moved = last - sum(1 for g in gone if g < last)
    assert [c.field for c in nc.claims] == [f"key_risks[{moved}].detail", "dcf_summary.base_implied_price"]
    assert shown.key_risks[moved] == raw.key_risks[last]
    assert [w.text for w in nc.withheld] == ["A real risk whose figure did not trace."]
    # Case key points and a hidden case headline follow the same rule.
    assert ms._renumbered("bull_case.key_points[3]", {"bull_case": [0, 2]}, set()) == "bull_case.key_points[1]"
    assert ms._renumbered("bull_case.key_points[2]", {"bull_case": [0, 2]}, set()) is None
    assert ms._renumbered("bull_case.headline", {}, {"bull_case"}) is None
    assert ms._renumbered("bull_case.headline", {}, set()) == "bull_case.headline"


def _builder_base() -> StockMemoOut:
    """An LLM-PM memo whose thesis the verdict guard rewrote with the builder
    (no LLM headline claimed), so the thesis is hidden and the confidence is
    not; the final verdict quotes the thesis verbatim."""
    base = _fixture("meta_v1")
    thesis = ("META screens undervalued on the blended read, though no single specialist headline "
              "defines the call. The market is under-pricing the durable part of the franchise.")
    return base.model_copy(update={
        "one_sentence_thesis": thesis,
        "final_verdict": f"PM final view: Bullish (confidence 62). {thesis} Watch items: none flagged.",
        "section_provenance": {"v": 1, "llm_configured": True, "thesis": "rewrite", "mispricing": "pm"},
    })


def test_patched_thesis_does_not_unhide_the_base_verdict():
    """A news patch may replace the thesis but never the final verdict, which
    still quotes the base thesis: that verdict stays hidden."""
    base = _builder_base()
    av_base = compute_availability(base)
    assert av_base["one_sentence_thesis"].status == "unavailable"
    assert av_base["confidence_score"].status == "available"
    assert av_base["final_verdict"].status == "unavailable"
    patched = base.model_copy(update={"one_sentence_thesis": "Ad pricing held through the quarter."})
    fields = frozenset({"one_sentence_thesis"})
    av = compute_availability(patched, patched_fields=fields, base=base)
    assert av["one_sentence_thesis"].status == "available"
    assert _status(av, "final_verdict") == ("unavailable", "derived_from_hidden")
    assert av["final_verdict"].basis == ["derived:base_thesis"]
    shown = present_memo(patched, patched_fields=fields, base=base)
    assert shown.final_verdict == UNAVAILABLE_TEXT
    assert base.one_sentence_thesis not in shown.model_dump_json()
    # An unknown base errs toward hiding; an LLM base thesis keeps the verdict.
    av = compute_availability(patched, patched_fields=fields, base=None, chain_complete=False)
    assert av["final_verdict"].status == "unavailable"
    meta = _fixture("meta_v1")
    meta_patched = meta.model_copy(update={"one_sentence_thesis": "A patched thesis."})
    av = compute_availability(meta_patched, patched_fields=fields, base=meta)
    assert av["final_verdict"].status == compute_availability(meta)["final_verdict"].status != "unavailable"


def test_patched_fallback_card_found_by_the_base_thesis_it_quotes():
    """Critique delta 6, isolated: over an LLM base (so the template-PM rule
    cannot decide), the legacy fallback card quotes the BASE thesis, which a
    patch replaced; the base-thesis match alone hides it."""
    base = _fixture("meta_v1").model_copy(update={"mispricing_thesis": MispricingThesis(
        consensus_view="The street sees a steady compounder.",
        our_view=_fixture("meta_v1").one_sentence_thesis,
        gap="The blended read calls the name undervalued.")})
    assert compute_availability(base)["mispricing_thesis"].status == "unavailable"
    patched = base.model_copy(update={"one_sentence_thesis": "A patched thesis after the news."})
    av = compute_availability(patched, patched_fields=frozenset({"one_sentence_thesis"}), base=base)
    assert av["final_pm_view"].status == "available"  # the PM is not a template
    assert _status(av, "mispricing_thesis") == ("unavailable", "template_fallback")
    # Without the base, the patched thesis no longer matches the card.
    assert compute_availability(patched)["mispricing_thesis"].status == "available"


def test_final_verdict_strips_hidden_watch_items_and_sector_lean():
    """`graph._build_verdict` ends with the thesis breakers' titles and may
    carry the canned sector lean; a hidden breaker's title and a template
    lean are stripped, the rest of the verdict kept."""
    meta = _fixture("meta_v1")
    real = RiskItem(title="Cloud price war", detail="Hyperscalers cut prices into a slowdown.")
    template = RiskItem(title="Cohort positioning", detail=SIG["case_bear_last_resort"].text)
    head = f"PM final view: Neutral (confidence 55). {meta.one_sentence_thesis}"
    memo = meta.model_copy(update={
        "thesis_breakers": [template, real],
        "final_verdict": f"{head} Watch items: Cohort positioning, Cloud price war",
    })
    av = compute_availability(memo)
    assert av["thesis_breakers"].status == "degraded"
    assert _status(av, "final_verdict") == ("degraded", "partial_template")
    assert "stripped:watch_items" in av["final_verdict"].basis
    shown = present_memo(memo)
    assert shown.final_verdict == f"{head} Watch items: Cloud price war"
    assert [r.title for r in shown.thesis_breakers] == ["Cloud price war"]
    only_template = memo.model_copy(update={
        "thesis_breakers": [template], "final_verdict": f"{head} Watch items: Cohort positioning"})
    assert present_memo(only_template).final_verdict == f"{head} Watch items: none flagged."

    bb = meta.sector_agent_view.data["bull_bear_analysis"]
    lean_memo = meta.model_copy(deep=True)
    lean_memo.sector_agent_view.data["bull_bear_analysis"] = {**bb, "key_disagreement": KD,
                                                              "sector_lean": "bearish"}
    lean_memo.final_verdict = f"{head} Sector lean: bearish. Watch items: none flagged."
    av = compute_availability(lean_memo)
    assert "stripped:sector_lean" in av["final_verdict"].basis
    assert present_memo(lean_memo).final_verdict == f"{head} Watch items: none flagged."


def test_earned_confidence_is_shown_over_a_template_pm():
    """The W2b contract hook: a write-time `section_provenance.confidence ==
    "earned"` shows the confidence even when the PM view was a template."""
    memo = _fixture("googl_live_prepflag").model_copy(update={"section_provenance": {"confidence": "earned"}})
    av = compute_availability(memo)
    assert av["final_pm_view"].status == "unavailable"
    assert av["confidence_score"].status == "available"
    assert av["confidence_score"].basis == ["provenance:confidence=earned"]
    assert compute_availability(_fixture("googl_live_prepflag"))["confidence_score"].status == "unavailable"


def test_is_reflection_template_pins_the_no_llm_entry(no_llm):
    """Pinned to the producer: the deterministic branch of
    `reflection_agent._compose_company_entry` is a template entry, the LLM
    branch's labelled entry is not."""
    from app.agents import reflection_agent
    memo = _fixture("meta_v1")
    det = reflection_agent._compose_company_entry(memo, [{"label": "memo run"}])
    assert ms.is_reflection_template(det)
    llm_entry = ("**Trigger:** memo run\n\n**Observation:** Ads grew faster than the model assumed.\n\n"
                 "**Update to thesis:** Held the view; confidence unchanged.\n\n**Watch next:** capex.")
    assert not ms.is_reflection_template(llm_entry)


def test_condensed_history_line_template_probe():
    det_obs = "Technology / Internet Content regime read: mixed. Cohort of 5 peers selected on sub_indus"
    assert ms.is_condensed_reflection_template(
        f"- 2026-09-01 (memo_run): **Trigger:** memo run  **Observation:** {det_obs}")
    assert not ms.is_condensed_reflection_template(
        "- 2026-09-01 (memo_run): **Trigger:** memo run  **Observation:** Ads grew faster than modelled.")
    assert not ms.is_condensed_reflection_template(
        "**Condensed 2026-08-01 → 2026-09-01** (3 entries; memo_run=3)")
