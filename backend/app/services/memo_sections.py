"""Hide template-filled memo sections on the way out (W2a, owner decision 2).

The owner's ruling (2026-09-24): publish the memo, but a section that a
template filled (the deterministic fallback) is shown as "Unavailable in this
version." instead of reading as analysis. Stored payloads are never changed.

Two public functions do the work:

* `compute_availability(memo)` returns a per-section verdict
  (`available | degraded | unavailable`, a closed-vocabulary reason and the
  `basis` evidence it rests on). Its inputs are facts the payload already
  carries, in this order: the write-time `section_provenance` (new memos),
  the existing flags (`degradation_events`, finding `data` flags, critic
  `review_mode`), and a closed catalogue of EXACT fallback signatures for the
  memos written before those flags existed. There is no similarity scoring:
  a signature is text only the fallback producer can emit, and each one is
  pinned to its producer by `test_memo_sections.py`. When nothing matches, a
  section is available — this module never guesses.
* `present_memo(memo)` returns a deep copy in which hidden prose reads
  `UNAVAILABLE_TEXT`, template items are removed from list sections and
  `section_availability` is filled in. Numbers never change (a rewritten
  confidence would be a false value); renderers consult the map instead.

Both are pure: no I/O, no settings reads (the no-LLM case comes from
`section_provenance`), no mutation of the input. This module imports only
`..schemas`, so the public-samples request path may use it without gaining a
route to `app.agents` or the providers. A payload shape this module does not
expect is classified `available` with basis `unclassified` rather than
raising: a memo that validated must never 500 on the read path.

Every customer-facing exit calls the presenter — see
`memo_store.present_snapshot` and the callers it names. Internal readers
(outcomes, calibration, postmortems, the critic's prior context, the news
impact assessment) keep reading the raw payload on purpose.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

from ..schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    DebateClaim,
    DebateRecord,
    MispricingThesis,
    NumberClaim,
    ReviewIssue,
    SectionAvailability,
    StockMemoOut,
)
from ..schemas.agents import CRITIC_REVIEW_ITEM8_FIELDS

# The owner's wording (decision 2), with the full stop the integration plan
# (C2) fixed. Every hidden prose field reads exactly this, so a renderer that
# knows nothing about `section_availability` (the pre-S12 UI, chat, the PDF,
# API clients) still shows the placeholder rather than a blank.
UNAVAILABLE_TEXT = "Unavailable in this version."

# Bumped whenever the classification rules or the presented shape change.
# Caches of text derived from a presented memo (chart commentary) key on it,
# so a rule change never serves prose built from the old presentation.
#   2 (D4, 2026-09-25; integration plan P12): the `debate` section and its
#     projection, and the item-8 review projection. On a memo written before
#     the debate existed the only visible change is the review label: a
#     review that was not live now reads "not independently reviewed"
#     (`review_status`). A legacy live review stays unlabelled: only the
#     item-8 reviewer's stored label says "independent".
PRESENTATION_VERSION = 2

# Stable vocabulary shared by the backend, the frontend and W2b, in display
# order. Dynamic keys are added per memo: `extra_agent_views.<roster key>` and
# `<finding key>.long_form_report` for each shown analyst drill-down.
SECTION_KEYS: tuple[str, ...] = (
    "final_pm_view", "one_sentence_thesis", "rating_label", "confidence_score",
    "mispricing_thesis", "valuation_verdict", "business_summary",
    "sector_agent_view", "sector_synthesis", "earnings_agent_view", "filing_agent_view",
    "valuation_agent_view", "comps_agent_view", "macro_sensitivity", "technical_agent_view",
    "earnings_qoq_delta", "bull_case", "bear_case", "debate", "catalysts", "key_risks",
    "thesis_breakers", "forward_catalysts", "dcf_summary", "risk_committee_challenge",
    "portfolio_fit", "final_verdict", "scorecard", "round_findings",
)

# Memo field -> the display name its producer records in the degradation log.
LLM_ANALYST_FIELDS: dict[str, str] = {
    "sector_agent_view": "Sector Analyst",
    "earnings_agent_view": "Earnings Analyst",
    "filing_agent_view": "Filing Analyst",
    "valuation_agent_view": "Valuation Analyst",
    "macro_sensitivity": "Macro Analyst",
    "technical_agent_view": "Technical Analyst",
}
# Deterministic at round 0 by design (`roster.AgentSpec.uses_llm_round0=False`);
# never hidden for being deterministic. `test_memo_sections` pins this set
# against the roster so a new computed analyst cannot be missed here.
COMPUTED_ANALYST_FIELDS: dict[str, str] = {"comps_agent_view": "Comps Analyst"}
COMPUTED_ROSTER_KEYS: frozenset[str] = frozenset({"comps", "risk"})

# Hidden-finding `data` allowlist. Computed keys a renderer may still show,
# plus the provenance flags that say why the finding is hidden. Everything
# else (LLM prose such as `bull_bear_analysis`, `structured`, `narrative`,
# `causal_chain`) is dropped; an allowlist so an unknown future key cannot
# leak template text.
_ALLOWED_DATA_KEYS: frozenset[str] = frozenset({
    "macro_broadcast", "kpi_placements", "cohort", "trends", "regime", "outliers",
    "industry_structure", "sector", "sub_industry", "industry", "target_ticker", "signals",
    "history", "industry_group", "sector_data_context", "valuation_exclusions",
    # Critique delta 2: live news items and the macro alignment label are data
    # the sector card shows beside the view, not the view itself.
    "pending_news_alerts", "macro_alignment",
})
_PROVENANCE_DATA_KEYS: frozenset[str] = frozenset({
    "deterministic_fallback", "bull_bear_parse_failed", "degraded", "error",
    "intake_skipped", "intake_rationale", "retrieval_failed", "no_mapping",
})

_LONG_FORM_MARKER = "### Analyst expansion"  # long_form.build_long_form_report


# ---------------------------------------------------------------------------
# The signature catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signature:
    """Text only one fallback producer emits. `producer` is the code that
    emits it (or the stored evidence for a removed producer), `since` the
    commit that introduced it; the pinning tests call each live producer."""
    id: str
    section: str
    kind: Literal["exact", "prefix", "suffix", "contains", "regex"]
    text: str
    producer: str
    since: str = ""

    def matches(self, value: Any) -> bool:
        if not isinstance(value, str) or not value:
            return False
        if self.kind == "exact":
            return value == self.text
        if self.kind == "prefix":
            return value.startswith(self.text)
        if self.kind == "suffix":
            return value.endswith(self.text)
        if self.kind == "contains":
            return self.text in value
        return _compiled(self.text).search(value) is not None


@cache
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


SIGNATURES: tuple[Signature, ...] = (
    # --- PM synthesis --------------------------------------------------------
    Signature("pm_view_tail", "final_pm_view", "contains",
              "Sector framing supports the cohort thesis; valuation-relative read is the main swing "
              "factor. The risk committee flagged the dominant downside scenarios; portfolio fit "
              "depends on macro view.", "graph._pm_synthesis", "bc34765"),
    Signature("pm_hard_fallback", "final_pm_view", "contains",
              "PM synthesis unavailable; relying on specialist findings only.",
              "graph._compose_memo synth_fallback", "f1f9575"),
    # --- thesis ----------------------------------------------------------------
    Signature("thesis_research_draft", "one_sentence_thesis", "regex",
              r"^Research draft for \S+\.$", "graph._compose_memo synth_fallback"),
    Signature("thesis_builder_underpricing", "one_sentence_thesis", "contains",
              "The market is under-pricing the durable part of the franchise",
              "graph._mispricing_lever_clause"),
    Signature("thesis_builder_multiple", "one_sentence_thesis", "contains",
              "The multiple already prices in the bull case", "graph._mispricing_lever_clause"),
    Signature("thesis_builder_no_headline", "one_sentence_thesis", "contains",
              "though no single specialist headline defines the call",
              "graph._build_thesis_from_findings"),
    Signature("thesis_builder_floor", "one_sentence_thesis", "contains",
              "own the floor, not the multiple", "graph._build_thesis_from_findings"),
    # Anchored (critique delta 10): the PM prompt itself offers "fairly priced
    # on our work, no actionable edge" as a valid LLM call (prompts.py), so
    # only the builder's exact sentence-1 form — em dash, "in <sector>." — is
    # a signature.
    Signature("thesis_builder_no_edge", "one_sentence_thesis", "regex",
              r"^(?:\S+ is fairly priced|Fairly priced) on our work — no actionable edge in [^.]+\.",
              "graph._build_thesis_from_findings"),
    # --- mispricing fallback (graph._build_mispricing_fallback) -----------------
    Signature("mispricing_gap_fair", "mispricing_thesis", "exact",
              "No material mispricing on our work — the signals offset and the blended rating "
              "lands at fair value.", "graph._build_mispricing_fallback"),
    Signature("mispricing_gap_dcf", "mispricing_thesis", "regex",
              r"^Our DCF base case implies [+-]\d+% to fair value; the blended read calls the name "
              r"(?:undervalued|overvalued)\.$", "graph._build_mispricing_fallback"),
    Signature("mispricing_gap_blend", "mispricing_thesis", "regex",
              r"^The blended read calls the name (?:undervalued|overvalued)\.$",
              "graph._build_mispricing_fallback"),
    Signature("mispricing_our_view_pointer", "mispricing_thesis", "regex",
              r"^See the \S+ thesis above\.$", "graph._build_mispricing_fallback"),
    # --- sector bull/bear (sector_agents._deterministic_bull_bear_analysis) ------
    Signature("bb_key_disagreement", "sector_synthesis", "exact",
              "Bears price in cohort margin compression flowing through to this name; bulls price "
              "in this name continuing to outpace cohort on the dominant driver.",
              "sector_agents._deterministic_bull_bear_analysis", "472a01d"),
    Signature("bb_bull_headline", "bull_case", "regex",
              r"^Bull case: durable execution against .+\.$",
              "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bear_headline", "bear_case", "regex",
              r"^Bear case: .+ thesis breaks on .+\.$",
              "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bear_valuation_quartile", "bear_case", "regex",
              r"^Valuation in the (?:top|upper) cohort quartile — any execution slip re-rates "
              r"the multiple\.$", "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bear_margin_compressing", "bear_case", "regex",
              r"^Cohort op margin compressing \([+-]\d+\.\dpp multi-year\) — competitive "
              r"intensity is rising\.$", "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bear_execution", "bear_case", "exact", "Execution risk on the dominant driver.",
              "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bull_growth_quartile", "bull_case", "regex",
              r"^Revenue growth in the (?:top|upper-half) cohort quartile — share-take story is "
              r"empirical, not narrative\.$", "sector_agents._deterministic_bull_bear_analysis"),
    Signature("bb_bull_margin", "bull_case", "exact",
              "Operating margin above cohort median — quality premium is earned.",
              "sector_agents._deterministic_bull_bear_analysis"),
    # --- memo bull/bear builders (graph._bull_case / _bear_case) ----------------
    Signature("case_bull_last_resort", "bull_case", "exact",
              "Quality + growth profile supports a premium versus peers.",
              "graph._bull_case, sector_agents._deterministic_bull_bear_analysis"),
    Signature("case_bear_last_resort", "bear_case", "exact",
              "Cohort positioning leaves modest downside if execution slips.", "graph._bear_case"),
    Signature("case_headline_bull_blend", "bull_case", "exact",
              "Bull case: cohort + valuation read both supportive.", "graph._bull_case"),
    Signature("case_headline_bear_blend", "bear_case", "exact",
              "Bear case: execution / valuation / regulatory risks if thesis cracks.",
              "graph._bear_case"),
    Signature("case_headline_bull_synthesis", "bull_case", "exact",
              "Bull case from sector synthesis.", "graph._bull_case"),
    Signature("case_headline_bear_synthesis", "bear_case", "exact",
              "Bear case from sector synthesis.", "graph._bear_case"),
    Signature("case_headline_bull_unavailable", "bull_case", "exact", "Bull case unavailable.",
              "graph._compose_memo safe_call fallback"),
    Signature("case_headline_bear_unavailable", "bear_case", "exact", "Bear case unavailable.",
              "graph._compose_memo safe_call fallback"),
    # Profile driver/risk lines. Identity slots only, but an LLM could write a
    # "Tailwind:" bullet too, so these count only when the case was NOT built
    # from an LLM-authored sector bull/bear block (see `_case_item_template`).
    Signature("case_profile_tailwind", "bull_case", "regex", r"^Tailwind: .+$",
              "graph._bull_case, sector_agents._deterministic_bull_bear_analysis"),
    Signature("case_profile_headwind", "bear_case", "regex", r"^Headwind: .+$",
              "sector_agents._deterministic_bull_bear_analysis"),
    # Self-labelled template scenario drivers (scenario_assumptions).
    Signature("dcf_template_driver", "bull_case", "regex",
              r"^DCF driver — .+ sector — (?:upside|downside) scenario: Deterministic \w+ "
              r"fallback: ", "scenario_assumptions deterministic scenario"),
    # --- catalysts ---------------------------------------------------------------
    Signature("catalyst_next_update", "catalysts", "exact", "Next earnings update",
              "graph._catalysts"),
    # --- critic ------------------------------------------------------------------
    Signature("critic_legacy_rule_based", "risk_committee_challenge", "exact",
              "Memo is structurally sound; balance and source citations should be tightened.",
              "stored evidence (removed in 9da68fd)"),
    Signature("critic_rule_based", "risk_committee_challenge", "prefix",
              "Rule-based check only; no live critic review was completed.",
              "critic_agent.run_critic"),
    Signature("critic_unavailable", "risk_committee_challenge", "exact",
              "Critic agent unavailable for this run.", "safe_runner.safe_critic"),
    Signature("critic_pending", "risk_committee_challenge", "exact", "Pending critic review.",
              "graph._compose_memo"),
    # --- analyst stand-ins (matched on a finding field; see _ANALYST_RULES) -----
    Signature("earnings_no_transcript", "earnings_agent_view", "exact",
              "Earnings transcript unavailable.", "earnings_agent.run_earnings_agent"),
    Signature("earnings_det_headline", "earnings_agent_view", "regex",
              r"^\S*: transcript pending LLM analysis\.$", "earnings_agent.run_earnings_agent",
              "26c3454"),
    Signature("filing_no_filings", "filing_agent_view", "exact",
              "No filings cached for this ticker.", "filing_agent.run_filing_agent"),
    Signature("filing_det_headline", "filing_agent_view", "regex", r"^\S* [\w/-]+ highlights$",
              "filing_agent.run_filing_agent"),
    # Older extracts wrote "10-K dated <date>: business spans ..." (colon).
    Signature("filing_det_summary", "filing_agent_view", "regex",
              r"^[\w/-]+ dated [\w/—-]+[.:](?:\s|$)", "filing_agent.run_filing_agent"),
    Signature("valuation_det_summary", "valuation_agent_view", "contains",
              "DCF triangulates against multiples; the bull/bear range frames the discount-rate "
              "sensitivity.", "valuation_agent.run_valuation_agent"),
    Signature("macro_det_headline", "macro_sensitivity", "regex", r"^Macro scenario: .+$",
              "macro_agent.run_macro_agent"),
    Signature("macro_det_summary", "macro_sensitivity", "prefix", "Scenario read: ",
              "macro_agent.run_macro_agent"),
    Signature("sector_det_summary", "sector_agent_view", "regex",
              r"^.+ / .+ regime read: .+\. Cohort of \d+ peers selected on .+ basis\.",
              "sector_agents.run_sector_agent"),
    Signature("technical_det_tail", "technical_agent_view", "contains",
              "Technical signals are positioning context for the fundamental thesis, not a "
              "standalone trade signal.", "technical_agent._deterministic_summary"),
    Signature("technical_no_prices", "technical_agent_view", "regex",
              r"^\S*: price series unavailable for technical read\.$",
              "technical_agent.run_technical_agent"),
    Signature("technical_short_history", "technical_agent_view", "regex",
              r"^\S*: insufficient price history for technical read\.$",
              "technical_agent.run_technical_agent"),
    Signature("technical_no_ticker", "technical_agent_view", "exact",
              "Technical analysis unavailable.", "technical_agent.run_technical_agent"),
    Signature("industry_no_mapping", "extra_agent_views", "exact",
              "Industry group read unavailable: no mapping.", "industry_analysts._unmapped_finding"),
    Signature("industry_det_chain", "extra_agent_views", "exact",
              "n/a: deterministic edition — not asserted", "industry_analysts._deterministic_finding"),
    # --- earnings QoQ without an LLM (earnings_qoq.run_earnings_qoq_delta) ------
    Signature("qoq_no_llm_points", "earnings_qoq_delta", "exact", "No material differences.",
              "earnings_qoq.run_earnings_qoq_delta"),
    Signature("qoq_no_llm_summary", "earnings_qoq_delta", "regex",
              r"^(?:Overall tone \S+ → \S+\.|QoQ delta computed vs .+; no major reversals "
              r"detected\.)$", "earnings_qoq.run_earnings_qoq_delta"),
)

SIG: dict[str, Signature] = {s.id: s for s in SIGNATURES}

_CASE_HEADLINE_SIGS = (
    "case_headline_bull_blend", "case_headline_bear_blend", "case_headline_bull_synthesis",
    "case_headline_bear_synthesis", "case_headline_bull_unavailable",
    "case_headline_bear_unavailable", "bb_bull_headline", "bb_bear_headline",
)
_CASE_ITEM_SIGS = (
    "case_bull_last_resort", "case_bear_last_resort", "bb_bear_valuation_quartile",
    "bb_bear_margin_compressing", "bb_bear_execution", "bb_bull_growth_quartile", "bb_bull_margin",
)
_THESIS_BUILDER_SIGS = (
    "thesis_builder_underpricing", "thesis_builder_multiple", "thesis_builder_no_headline",
    "thesis_builder_floor", "thesis_builder_no_edge",
)
_MISPRICING_GAP_SIGS = ("mispricing_gap_fair", "mispricing_gap_dcf", "mispricing_gap_blend")
_CRITIC_SIGS = ("critic_legacy_rule_based", "critic_rule_based", "critic_unavailable",
                "critic_pending")

# The computed lines the case builders append; never template.
_COMPUTED_CASE_LINE = re.compile(r"^(?:DCF (?:bull|bear) case implies |Risk lens: )")
_DCF_IMPLIES_LINE = re.compile(r"^DCF (?:bull|bear) case implies ")
_DCF_DRIVER_LINE = re.compile(r"^DCF driver — ")
_NEXT_EARNINGS = re.compile(r"^Next earnings: ")
_CASE_LABEL = re.compile(r"^\s*(?:bull|bear|base)[\s-]*case\s*[:\-—–]\s*", re.IGNORECASE)


def _finding_rule(f: AgentFinding) -> str | None:
    """The id of the analyst stand-in signature `f` matches, or None.

    No-input stubs (`*_no_*`) are checked by `_no_input_rule`; this covers the
    deterministic read-outs that stand in for an LLM analyst."""
    h, s = getattr(f, "headline", "") or "", getattr(f, "summary", "") or ""
    if SIG["earnings_det_headline"].matches(h):
        return "earnings_det_headline"
    if SIG["filing_det_headline"].matches(h) and SIG["filing_det_summary"].matches(s):
        return "filing_det_pair"
    if SIG["valuation_det_summary"].matches(s):
        return "valuation_det_summary"
    if SIG["macro_det_headline"].matches(h) and SIG["macro_det_summary"].matches(s):
        return "macro_det_pair"
    if SIG["sector_det_summary"].matches(s):
        return "sector_det_summary"
    if SIG["technical_det_tail"].matches(s):
        return "technical_det_tail"
    data = getattr(f, "data", None)
    chain = data.get("causal_chain") if isinstance(data, dict) else None
    if isinstance(chain, list) and any(
        isinstance(c, dict) and SIG["industry_det_chain"].matches(c.get("text")) for c in chain
    ):
        return "industry_det_chain"
    if (SIG["qoq_no_llm_summary"].matches(s)
            and list(getattr(f, "key_points", None) or []) == [SIG["qoq_no_llm_points"].text]):
        return "qoq_no_llm"
    return None


def _no_input_rule(f: AgentFinding) -> str | None:
    h = getattr(f, "headline", "") or ""
    for sid in ("earnings_no_transcript", "filing_no_filings", "technical_no_prices",
                "technical_short_history", "technical_no_ticker", "industry_no_mapping"):
        if SIG[sid].matches(h):
            return sid
    return None


def finding_is_template(finding: AgentFinding) -> bool:
    """True when `finding` is a stand-in rather than an analyst's read (C2).

    Used where a single finding is shown on its own — the chat `ask_*` tools
    — so a fallback never reaches a reader as a specialist's answer.
    Duck-typed (only `data`/`headline`/`summary`/`key_points` are read)."""
    data = getattr(finding, "data", None)
    d = data if isinstance(data, dict) else {}
    if any(d.get(k) for k in ("deterministic_fallback", "degraded", "intake_skipped", "no_mapping")):
        return True
    return _no_input_rule(finding) is not None or _finding_rule(finding) is not None


def is_reflection_template(body: str) -> bool:
    """True for a long-term-memory entry written by the deterministic branch
    of `reflection_agent._compose_company_entry` (no LLM), which restates the
    memo's sector summary, rating/confidence/thesis and valuation summary in a
    fixed frame. The LLM branch never writes a "**Valuation read:**" line nor
    the "Rating=" form."""
    return (
        isinstance(body, str)
        and "\n\n**Update to thesis:** Rating=" in body
        and "\n\n**Valuation read:** " in body
    )


# A line `longterm._deterministic_summary` folds a reflection entry into:
# "- <date> (<trigger>): <first 160 chars of the body>".
_CONDENSED_LINE = re.compile(r"^- \S+ \([^)]*\): (?P<takeaway>.*)$")
# The start of `sector_agents.run_sector_agent`'s deterministic summary
# (signature `sector_det_summary`), which the condenser truncates before the
# "Cohort of N peers" clause the full signature needs.
_SECTOR_DET_PREFIX = re.compile(r"^[^/*]+ / [^*]+ regime read: ")


def is_condensed_reflection_template(line: str) -> bool:
    """True for a condensed-history line that folded in a deterministic
    reflection entry whose observation is the template sector summary.
    The frame alone does not decide (the LLM branch writes the same
    `**Trigger:**`/`**Observation:**` labels); the observation text does."""
    m = _CONDENSED_LINE.match(line or "")
    if m is None:
        return False
    _, sep, obs = m.group("takeaway").partition("**Observation:** ")
    return bool(sep) and _SECTOR_DET_PREFIX.match(obs.strip()) is not None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _av(status: str, reason: str | None = None, basis: Iterable[str] = (), *,
        hidden_items: int = 0, headline_hidden: bool = False) -> SectionAvailability:
    return SectionAvailability(
        status=status,  # type: ignore[arg-type]
        reason=reason,  # type: ignore[arg-type]
        basis=list(dict.fromkeys(basis)), hidden_items=hidden_items,
        headline_hidden=headline_hidden,
    )


_AVAILABLE = SectionAvailability()


@dataclass
class _Plan:
    """What the presenter must do, decided on the input before any blanking."""
    avail: dict[str, SectionAvailability] = field(default_factory=dict)
    hidden_findings: set[str] = field(default_factory=set)       # field / extra_agent_views.<k>
    long_forms: dict[str, str | None] = field(default_factory=dict)
    drop_bull_bear_block: bool = False
    case_hidden: dict[str, tuple[bool, list[int]]] = field(default_factory=dict)
    list_hidden: dict[str, list[int]] = field(default_factory=dict)
    round_hidden: set[tuple[int, str]] = field(default_factory=set)
    round_bb_drop: set[tuple[int, str]] = field(default_factory=set)
    final_verdict: str | None = None                                # replacement text
    # What `_case_item_template` needs, kept so the number-check filter can
    # judge a withheld item's text by the same rules as a stored item.
    fragments: set[str] = field(default_factory=set)
    scenarios_templated: bool = False
    llm_bb: bool = False


@dataclass
class _Ctx:
    memo: StockMemoOut
    eval_memo: StockMemoOut          # the chain's base for PM / mispricing rules
    patched: frozenset[str]
    chain_complete: bool
    llm_off: bool
    events: set[tuple[str, str]]
    event_agents: set[str]
    presence: set[str]
    degraded_only: set[str]          # in degraded_agents with no event (pre-RP-001 rows)


def _events(memo: StockMemoOut) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for e in memo.degradation_events or []:
        if isinstance(e, dict) and isinstance(e.get("agent"), str):
            out.add((e["agent"], str(e.get("error_type") or "")))
    return out


def _presence(memo: StockMemoOut) -> set[str]:
    """Agents the memo records as degraded: `degraded_agents` ∪ event agents
    (critique delta 10). Stored rows exist with either list missing."""
    agents = {a for a in (memo.degraded_agents or []) if isinstance(a, str)}
    return agents | {a for a, _ in _events(memo)}


def _pm_template(m: StockMemoOut) -> tuple[bool, bool, list[str]]:
    """(template, hard, basis) for the PM synthesis of memo `m`."""
    basis: list[str] = []
    hard = False
    if (m.section_provenance or {}).get("llm_configured") is False:
        basis.append("provenance:llm_configured=false")
    evs = _events(m)
    for agent, kind in evs:
        if agent == "PM Synthesis":
            basis.append(f"event:PM Synthesis/{kind}")
            hard = hard or kind != "DeterministicFallback"
    if "PM Synthesis" in (m.degraded_agents or []) and not any(a == "PM Synthesis" for a, _ in evs):
        basis.append("degraded_agents:PM Synthesis")
    view = m.final_pm_view or ""
    if SIG["pm_view_tail"].matches(view):
        basis.append("signature:pm_view_tail")
    if SIG["pm_hard_fallback"].matches(view):
        basis.append("signature:pm_hard_fallback")
        hard = True
    return bool(basis), hard, basis


def _classify_finding(f: AgentFinding | None, ctx: _Ctx, *, computed: bool = False,
                      refire: bool = False, memo_level: bool = True) -> SectionAvailability:
    """The analyst-view rules (W2a §4.3), first match wins.

    `memo_level=False` for a diligence-round finding: the degradation log
    describes the FINAL finding per agent, so an agent's banner entry says
    nothing about its earlier rounds; those are judged on their own flags
    and signatures only."""
    if f is None:
        return _av("unavailable", "not_produced")
    try:
        d = f.data if isinstance(f.data, dict) else {}
        agent = f.agent or ""
        if d.get("intake_skipped"):
            return _av("unavailable", "skipped_by_intake", ["flag:intake_skipped"])
        if d.get("degraded") is True and f.headline == f"{agent} unavailable":
            return _av("unavailable", "agent_failed", ["flag:degraded"])
        no_input = _no_input_rule(f)
        if no_input or d.get("no_mapping") or (agent == "Technical Analyst" and d.get("degraded")):
            basis = [f"signature:{no_input}"] if no_input else (
                ["flag:no_mapping"] if d.get("no_mapping") else ["flag:degraded"])
            return _av("unavailable", "no_source_data", basis)
        if computed:
            if d.get("deterministic_fallback"):
                # A comps/risk re-fire that fell through re-states the round-0
                # read: the section keeps it, the dialog answer is not one.
                if refire:
                    return _av("unavailable", "follow_up_unanswered", ["flag:deterministic_fallback"])
                return _av("degraded", "follow_up_unanswered", ["flag:deterministic_fallback"])
            return _AVAILABLE.model_copy()
        if d.get("deterministic_fallback"):
            return _av("unavailable", "template_fallback", ["flag:deterministic_fallback"])
        if memo_level and (agent, "DeterministicFallback") in ctx.events:
            return _av("unavailable", "template_fallback",
                       [f"event:{agent}/DeterministicFallback"])
        if memo_level and agent in ctx.degraded_only:
            return _av("unavailable", "template_fallback", [f"degraded_agents:{agent}"])
        if ctx.llm_off:
            return _av("unavailable", "template_fallback", ["provenance:llm_configured=false"])
        legacy = _finding_rule(f)
        if legacy:
            return _av("unavailable", "template_fallback", [f"signature:{legacy}"])
        if d.get("retrieval_failed"):
            return _av("degraded", "reduced_inputs", ["flag:retrieval_failed"])
        return _AVAILABLE.model_copy()
    except Exception:  # an unexpected shape never 500s a memo that validated
        return _av("available", None, ["unclassified"])


def _finding_fragments(f: AgentFinding) -> set[str]:
    """The texts `graph._findings_signal_lines` can lift from a finding:
    headline, key points, and summary sentences of 30-220 chars, each also
    truncated to 240 (the builder's cap)."""
    out: set[str] = set()
    candidates: list[str] = [f.headline or ""]
    candidates.extend(p for p in (f.key_points or []) if isinstance(p, str))
    for s in (f.summary or "").split(". "):
        s = s.strip()
        if 30 <= len(s) <= 220:
            candidates.append(s)
    for c in candidates:
        c = c.strip()
        if c:
            out.add(c)
            out.add(c[:240])
    return out


def _normalized_headline(h: str) -> str:
    return _CASE_LABEL.sub("", h or "").strip().rstrip(".,;:").strip()


def _bb_fragments(bb: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for side in ("bull_case", "bear_case"):
        case = bb.get(side)
        if not isinstance(case, dict):
            continue
        head = case.get("headline")
        if isinstance(head, str) and head.strip():
            out.add(head.strip())
            norm = _normalized_headline(head)
            if len(norm) >= 24:
                out.add(norm)
        out.update(p.strip() for p in (case.get("key_points") or []) if isinstance(p, str) and p.strip())
    for key in ("sector_synthesis", "key_disagreement"):
        v = bb.get(key)
        if isinstance(v, str) and v.strip():
            out.add(v.strip())
    for t in bb.get("falsifiable_tests") or []:
        st = t.get("statement") if isinstance(t, dict) else None
        if isinstance(st, str) and st.strip():
            out.add(st.strip())
    return out


def _contains_any(text: str, fragments: Iterable[str], min_len: int) -> str | None:
    for frag in fragments:
        if len(frag) >= min_len and frag in text:
            return frag
    return None


def _finding_fields(memo: StockMemoOut) -> list[tuple[str, AgentFinding | None, bool]]:
    """(section key, finding, computed) for every analyst view on the memo."""
    out: list[tuple[str, AgentFinding | None, bool]] = []
    for key in LLM_ANALYST_FIELDS:
        out.append((key, getattr(memo, key), False))
    for key in COMPUTED_ANALYST_FIELDS:
        out.append((key, getattr(memo, key), True))
    for key, f in (memo.extra_agent_views or {}).items():
        out.append((f"extra_agent_views.{key}", f, key in COMPUTED_ROSTER_KEYS))
    return out


def _classify(memo: StockMemoOut, *, patched_fields: frozenset[str], base: StockMemoOut | None,
              chain_complete: bool) -> _Plan:
    plan = _Plan()
    av = plan.avail
    events = _events(memo)
    event_agents = {a for a, _ in events}
    degraded_names = {a for a in (memo.degraded_agents or []) if isinstance(a, str)}
    ctx = _Ctx(
        memo=memo, eval_memo=base if base is not None else memo, patched=patched_fields,
        chain_complete=chain_complete,
        llm_off=(memo.section_provenance or {}).get("llm_configured") is False,
        events=events, event_agents=event_agents, presence=degraded_names | event_agents,
        degraded_only=degraded_names - event_agents,
    )

    # --- analyst views --------------------------------------------------------
    unavailable_findings: list[AgentFinding] = []
    available_findings: list[AgentFinding] = []
    hidden_identities: set[tuple[str, str, str]] = set()
    for key, finding, computed in _finding_fields(memo):
        verdict = _classify_finding(finding, ctx, computed=computed)
        av[key] = verdict
        if finding is None:
            continue
        if verdict.status == "unavailable":
            plan.hidden_findings.add(key)
            hidden_identities.add(_identity(finding))
            # Only a stand-in's text is template. A skipped analyst's
            # "summary" is the PM's intake rationale, which other sections
            # may legitimately echo, so it seeds no fragments.
            if verdict.reason != "skipped_by_intake":
                unavailable_findings.append(finding)
        else:
            available_findings.append(finding)
            if finding.long_form_report:
                expansion = _long_form_expansion(finding.long_form_report)
                plan.long_forms[key] = expansion
                av[f"{key}.long_form_report"] = (
                    _av("available", None, ["long_form:analyst_expansion"]) if expansion
                    else _av("unavailable", "template_fallback", ["long_form:no_analyst_expansion"])
                )
    if memo.technical_agent_view is None:
        # Optional field: an absent technical view is not produced, not hidden.
        av["technical_agent_view"] = _av("unavailable", "not_produced")

    # --- sector bull/bear block and the template-fragment set F ---------------
    sector = memo.sector_agent_view
    sector_data = sector.data if isinstance(sector.data, dict) else {}
    bb = sector_data.get("bull_bear_analysis")
    bb = bb if isinstance(bb, dict) else None
    bb_basis: list[str] = []
    if bb is not None:
        if av["sector_agent_view"].status == "unavailable":
            bb_basis.append("derived:sector_agent_view")
        if sector_data.get("bull_bear_parse_failed"):
            bb_basis.append("flag:bull_bear_parse_failed")
        if SIG["bb_key_disagreement"].matches((bb.get("key_disagreement") or "").strip()):
            bb_basis.append("signature:bb_key_disagreement")
        if ctx.llm_off:
            bb_basis.append("provenance:llm_configured=false")
    bb_template = bool(bb_basis)
    if bb is None:
        av["sector_synthesis"] = _av("unavailable", "not_produced")
    elif bb_template:
        reason = ("derived_from_hidden" if av["sector_agent_view"].status == "unavailable"
                  else "template_fallback")
        av["sector_synthesis"] = _av("unavailable", reason, bb_basis)
        plan.drop_bull_bear_block = True
    else:
        av["sector_synthesis"] = _AVAILABLE.model_copy()

    fragments: set[str] = {SIG["case_bull_last_resort"].text, SIG["case_bear_last_resort"].text}
    if bb is not None and bb_template:
        fragments |= _bb_fragments(bb)
    hidden_finding_fragments: set[str] = set()
    for f in unavailable_findings:
        hidden_finding_fragments |= _finding_fragments(f)
    fragments |= hidden_finding_fragments

    # --- PM view, thesis, confidence, rating ----------------------------------
    pm_template, pm_hard, pm_basis = _pm_template(ctx.eval_memo)
    if base is not None and pm_basis:
        pm_basis = [f"base:{b}" for b in pm_basis]

    view = memo.final_pm_view or ""
    view_sig = next((s for s in ("pm_hard_fallback", "pm_view_tail") if SIG[s].matches(view)), None)
    if view_sig:
        av["final_pm_view"] = _av("unavailable",
                                  "agent_failed" if view_sig == "pm_hard_fallback" else "template_fallback",
                                  [f"signature:{view_sig}"])
    elif "final_pm_view" in patched_fields:
        av["final_pm_view"] = _av("available", None, ["patched:final_pm_view"])
    elif pm_template:
        av["final_pm_view"] = _av("unavailable", "agent_failed" if pm_hard else "template_fallback",
                                  pm_basis)
    elif not view.strip():
        av["final_pm_view"] = _av("unavailable", "not_produced")
    else:
        av["final_pm_view"] = _AVAILABLE.model_copy()

    av["one_sentence_thesis"] = _classify_thesis(memo, ctx, pm_template, pm_basis, fragments,
                                                 available_findings, bb if not bb_template else None)

    if (memo.section_provenance or {}).get("confidence") == "earned":
        av["confidence_score"] = _av("available", None, ["provenance:confidence=earned"])
    elif pm_template:
        av["confidence_score"] = _av("unavailable", "template_fallback", pm_basis)
    elif not chain_complete:
        # A patch whose chain could not be walked to its base: the base PM
        # view is unknown, and a patch only moves confidence ±15 around it.
        av["confidence_score"] = _av("unavailable", "template_fallback", ["patch_chain:incomplete"])
    else:
        av["confidence_score"] = _AVAILABLE.model_copy()

    if "rating_label" in patched_fields:
        av["rating_label"] = _av("available", None, ["patched:rating_label"])
    elif pm_template:
        # Never hidden: 60% of it is the quant factor blend, and track record
        # and screens read it. The note says the PM input was a template.
        av["rating_label"] = _av("degraded", "pm_view_unavailable", pm_basis)
    else:
        av["rating_label"] = _AVAILABLE.model_copy()

    av["mispricing_thesis"] = _classify_mispricing(memo, ctx, pm_template, fragments)

    vv = memo.valuation_verdict
    vv_hard = [k for a, k in events if a == "Valuation Verdict" and k != "DeterministicFallback"]
    if vv_hard:
        av["valuation_verdict"] = _av("unavailable", "agent_failed", [f"event:Valuation Verdict/{vv_hard[0]}"])
    elif not (vv.summary or "").strip():
        av["valuation_verdict"] = _av("unavailable", "not_produced")
    else:
        av["valuation_verdict"] = _AVAILABLE.model_copy()

    av["business_summary"] = (_AVAILABLE.model_copy() if (memo.business_summary or "").strip()
                              else _av("unavailable", "not_produced"))

    # --- bull / bear / risks / catalysts --------------------------------------
    llm_bb = bb is not None and not bb_template
    scenarios_templated = ctx.llm_off or "DCF Scenarios" in ctx.presence
    plan.fragments, plan.scenarios_templated, plan.llm_bb = fragments, scenarios_templated, llm_bb
    hidden_case_texts: set[str] = set()
    # The case texts hidden because the debate failed, NOT because they are
    # template output: a risk hidden for restating one says so, rather than
    # "template text was removed" (they are real analysis).
    debate_hidden_texts: set[str] = set()
    debate_av = _classify_debate(memo.debate)
    for key in ("bull_case", "bear_case"):
        case = getattr(memo, key)
        if debate_av.reason == "debate_unavailable":
            # §12.1: with the debate on and failed, the stored cases are the
            # legacy builders' (the payload is never rewritten), and they
            # must not read as the debate's output. Both go, whole; a risk
            # that restates a hidden point goes with it (below). The sector
            # block is untouched.
            points = [p.strip() for p in (case.key_points or []) if isinstance(p, str)]
            debate_hidden_texts.update(p for p in points if p)
            plan.case_hidden[key] = (True, list(range(len(case.key_points or []))))
            # No `hidden_items`: the frontend words that count as "template
            # items not shown", which these are not.
            av[key] = _av("unavailable", "debate_unavailable", debate_av.basis)
            continue
        head = case.headline or ""
        # `_bull_case` also builds "Bull case: <valuation headline>", so a
        # label-stripped copy of a hidden finding's headline counts too.
        head_hidden = bool(head.strip()) and (
            head.strip() in fragments
            or _CASE_LABEL.sub("", head).strip()[:200] in {f[:200] for f in hidden_finding_fragments}
            or any(SIG[s].matches(head) for s in _CASE_HEADLINE_SIGS)
        )
        hidden_idx: list[int] = []
        substantive_kept = False
        for i, item in enumerate(case.key_points or []):
            if _case_item_template(item, fragments, scenarios_templated, llm_bb):
                hidden_idx.append(i)
                hidden_case_texts.add(item)
            elif not _DCF_IMPLIES_LINE.match(item or ""):
                substantive_kept = True
        plan.case_hidden[key] = (head_hidden, hidden_idx)
        av[key] = _list_status(
            hidden=len(hidden_idx), headline_hidden=head_hidden,
            survivor=(bool(head.strip()) and not head_hidden) or substantive_kept,
            basis=(["fragments:template"] if hidden_idx or head_hidden else []),
        )
    av["debate"] = debate_av

    # A risk or breaker is hidden as template output when it restates a
    # template case item (or is one); as the failed debate's when it only
    # restates a case text the failed debate hid (§12.1).
    risk_hidden_details: set[str] = set()
    risk_debate_details: set[str] = set()
    idx: list[int] = []
    n_debate, kept = 0, False
    for i, r in enumerate(memo.key_risks or []):
        text = (r.detail or r.title or "").strip()
        if (text in hidden_case_texts
                or _case_item_template(text, fragments, scenarios_templated, llm_bb)):
            idx.append(i)
            risk_hidden_details.add(text)
        elif text in debate_hidden_texts:
            idx.append(i)
            risk_debate_details.add(text)
            n_debate += 1
        else:
            kept = True
    plan.list_hidden["key_risks"] = idx
    av["key_risks"] = _derived_list_status(
        template=len(idx) - n_debate, debate=n_debate, survivor=kept,
        basis=["derived:bear_case"], debate_basis=debate_av.basis)

    idx, n_debate, kept = [], 0, False
    for i, r in enumerate(memo.thesis_breakers or []):
        text = (r.detail or r.title or "").strip()
        if (text in risk_hidden_details or text in hidden_case_texts
                or _case_item_template(text, fragments, scenarios_templated, llm_bb)):
            idx.append(i)
        elif text in risk_debate_details or text in debate_hidden_texts:
            idx.append(i)
            n_debate += 1
        else:
            kept = True
    plan.list_hidden["thesis_breakers"] = idx
    av["thesis_breakers"] = _derived_list_status(
        template=len(idx) - n_debate, debate=n_debate, survivor=kept,
        basis=["derived:key_risks"], debate_basis=debate_av.basis)

    idx, kept = [], False
    for i, c in enumerate(memo.catalysts or []):
        title, detail = (c.title or "").strip(), (c.detail or "").strip()
        if _NEXT_EARNINGS.match(title):
            kept = True
            continue
        if (SIG["catalyst_next_update"].matches(title) or detail in hidden_finding_fragments
                or title in hidden_finding_fragments):
            idx.append(i)
        else:
            kept = True
    plan.list_hidden["catalysts"] = idx
    av["catalysts"] = _list_status(hidden=len(idx), survivor=kept,
                                   basis=["derived:hidden_findings"] if idx else [])

    # --- computed-by-design sections -------------------------------------------
    av["forward_catalysts"] = (_AVAILABLE.model_copy() if memo.forward_catalysts
                               else _av("unavailable", "not_produced"))
    dcf_hard = [k for a, k in events if a == "DCF Engine" and k != "DeterministicFallback"]
    if not memo.dcf_summary:
        av["dcf_summary"] = (_av("unavailable", "agent_failed", [f"event:DCF Engine/{dcf_hard[0]}"])
                             if dcf_hard else _av("unavailable", "not_produced"))
    elif "DCF Scenarios" in ctx.presence:
        av["dcf_summary"] = _av("degraded", "templated_scenarios", ["event:DCF Scenarios"])
    else:
        av["dcf_summary"] = _AVAILABLE.model_copy()
    if memo.scorecard is None:
        no_row = ("Fundamental Scorecard", "DataUnavailable") in events
        av["scorecard"] = (_av("unavailable", "no_source_data", ["event:Fundamental Scorecard/DataUnavailable"])
                           if no_row else _av("unavailable", "not_produced"))
    else:
        av["scorecard"] = _AVAILABLE.model_copy()
    if memo.earnings_qoq_delta is None:
        av["earnings_qoq_delta"] = _av("unavailable", "not_produced")
    else:
        av["earnings_qoq_delta"] = _classify_finding(memo.earnings_qoq_delta, ctx)
        if av["earnings_qoq_delta"].status == "unavailable":
            plan.hidden_findings.add("earnings_qoq_delta")

    # --- critic, portfolio fit ---------------------------------------------------
    rc = memo.risk_committee_challenge
    mode = rc.review_mode
    if mode == "live":
        av["risk_committee_challenge"] = _AVAILABLE.model_copy()
    elif mode in ("rule_based", "unavailable", "pending"):
        av["risk_committee_challenge"] = _av("unavailable", "critic_not_run", [f"review_mode:{mode}"])
    else:
        hit = next((s for s in _CRITIC_SIGS if SIG[s].matches(rc.overall_assessment or "")), None)
        av["risk_committee_challenge"] = (
            _av("unavailable", "critic_not_run", [f"signature:{hit}"]) if hit
            else _AVAILABLE.model_copy()
        )
    av["portfolio_fit"] = (_av("unavailable", "template_always", ["producer:graph._portfolio_fit"])
                           if (memo.portfolio_fit or "").strip() else _av("unavailable", "not_produced"))

    # --- final verdict ---------------------------------------------------------
    _classify_final_verdict(memo, plan, fragments, bb if bb_template else None,
                            sector_hidden=av["sector_agent_view"].status == "unavailable",
                            embedded_thesis=_embedded_thesis_verdict(memo, patched_fields, base))

    # --- diligence rounds --------------------------------------------------------
    # Every round's findings are blanked by the same rules (round 0 holds the
    # fan-out copies of the grid views, template text included, and API
    # clients read it), but only rounds >= 1 count toward the section's
    # status: the dialog renders the PM's follow-ups, not the fan-out.
    hidden_rounds = 0
    for r_i, rnd in enumerate(memo.round_findings or []):
        for k, f in (rnd.findings or {}).items():
            computed = k in COMPUTED_ROSTER_KEYS
            if f is not None and _identity(f) in hidden_identities:
                # The same finding the grid hides (round 0 is the fan-out,
                # and an agent's last round IS its final finding): it
                # inherits the memo-level verdict, which the degradation
                # log may have decided on its own.
                hidden = True
            else:
                v = _classify_finding(f, ctx, computed=computed, refire=computed and rnd.round >= 1,
                                      memo_level=False)
                hidden = v.status == "unavailable"
            if hidden:
                plan.round_hidden.add((r_i, k))
                hidden_rounds += int(rnd.round >= 1)
            elif f is not None and _round_bb_template(f, ctx, plan.drop_bull_bear_block):
                plan.round_bb_drop.add((r_i, k))
    av["round_findings"] = (_av("degraded", "partial_template", ["rounds:hidden_findings"],
                                hidden_items=hidden_rounds) if hidden_rounds else _AVAILABLE.model_copy())
    return plan


def _identity(f: AgentFinding) -> tuple[str, str, str]:
    return (f.agent or "", f.headline or "", f.summary or "")


def _round_bb_template(f: AgentFinding, ctx: _Ctx, grid_block_dropped: bool) -> bool:
    """True when a shown round finding carries a template sector bull/bear
    block. A sector copy loses it whenever the grid's block was dropped (it
    is the same block, or an earlier draft of it), and otherwise by the
    same flags and signature the grid is judged on."""
    d = f.data if isinstance(f.data, dict) else {}
    bb = d.get("bull_bear_analysis")
    if not isinstance(bb, dict):
        return False
    if grid_block_dropped and (f.agent or "") == "Sector Analyst":
        return True
    return bool(
        d.get("bull_bear_parse_failed") or ctx.llm_off
        or SIG["bb_key_disagreement"].matches((bb.get("key_disagreement") or "").strip())
    )


def _embedded_thesis_verdict(memo: StockMemoOut, patched_fields: frozenset[str],
                             base: StockMemoOut | None) -> list[str]:
    """Basis for hiding the final verdict because it quotes a hidden BASE thesis.

    `graph._build_verdict` embeds the thesis verbatim, and a news patch may
    replace `one_sentence_thesis` but never `final_verdict`. So after a
    thesis patch the verdict still carries the base memo's thesis, which the
    current thesis rules no longer see. Judge that base thesis on the base
    snapshot; an unknown base errs toward hiding."""
    if "one_sentence_thesis" not in patched_fields or "final_verdict" in patched_fields:
        return []
    if base is None:
        return ["patch_chain:incomplete"]
    base_thesis = (base.one_sentence_thesis or "").strip()
    if not base_thesis or base_thesis not in (memo.final_verdict or ""):
        # The verdict does not quote the base thesis (a producer change, or
        # a base with no thesis): nothing of it to hide.
        return []
    try:
        status = _classify(base, patched_fields=frozenset(), base=None,
                           chain_complete=True).avail["one_sentence_thesis"].status
    except Exception:
        return ["unclassified:base_thesis"]
    return ["derived:base_thesis"] if status == "unavailable" else []


def _long_form_expansion(text: str) -> str | None:
    """The analyst's own words in a long-form drill-down: the LLM text after
    `### Analyst expansion` (critique delta 1). Everything before it is the
    deterministic `long_form.deterministic_long_form` body, which restates
    the finding in canned frames (and the canned sector synthesis/KD block)."""
    at = text.find(_LONG_FORM_MARKER)
    if at < 0:
        return None
    tail = text[at + len(_LONG_FORM_MARKER):].strip()
    return tail or None


def _case_item_template(item: str, fragments: set[str], scenarios_templated: bool,
                        llm_bb: bool) -> bool:
    text = (item or "").strip()
    if not text or _COMPUTED_CASE_LINE.match(text):
        return False
    if text in fragments or text.removeprefix("Filing: ") in fragments:
        return True
    if any(SIG[s].matches(text) for s in _CASE_ITEM_SIGS):
        return True
    if SIG["dcf_template_driver"].matches(text):
        return True
    if scenarios_templated and _DCF_DRIVER_LINE.match(text):
        return True
    if not llm_bb and (SIG["case_profile_tailwind"].matches(text)
                       or SIG["case_profile_headwind"].matches(text)):
        return True
    return False


def _list_status(*, hidden: int, survivor: bool, basis: list[str],
                 headline_hidden: bool = False) -> SectionAvailability:
    if not hidden and not headline_hidden:
        return _AVAILABLE.model_copy()
    if survivor:
        return _av("degraded", "partial_template", basis, hidden_items=hidden,
                   headline_hidden=headline_hidden)
    return _av("unavailable", "template_fallback", basis, hidden_items=hidden,
               headline_hidden=headline_hidden)


def _derived_list_status(*, template: int, debate: int, survivor: bool, basis: list[str],
                         debate_basis: list[str]) -> SectionAvailability:
    """`_list_status` for a list derived from the cases (risks, breakers),
    whose items can also be hidden for restating a case text the failed
    debate hid. Those are real analysis, so they are never counted or
    worded as template items: with no template item hidden the reason is
    the debate's own (`debate_unavailable`, basis `debate:<reason>`); with
    some, the template reason and count cover only the template items and
    the basis records the debate too."""
    if template:
        return _list_status(hidden=template, survivor=survivor,
                            basis=[*basis, *(debate_basis if debate else [])])
    if not debate:
        return _AVAILABLE.model_copy()
    return _av("degraded" if survivor else "unavailable", "debate_unavailable",
               [*debate_basis, *basis])


def _classify_thesis(memo: StockMemoOut, ctx: _Ctx, pm_template: bool, pm_basis: list[str],
                     fragments: set[str], available_findings: list[AgentFinding],
                     llm_bb: dict[str, Any] | None) -> SectionAvailability:
    thesis = memo.one_sentence_thesis or ""
    builder_hit = next((s for s in _THESIS_BUILDER_SIGS if SIG[s].matches(thesis)), None)
    exact_hit = ("thesis_research_draft" if SIG["thesis_research_draft"].matches(thesis)
                 else builder_hit)
    if "one_sentence_thesis" in ctx.patched and not exact_hit:
        # Exact signatures override "patched" (critique delta 6).
        return _av("available", None, ["patched:one_sentence_thesis"])
    if pm_template:
        return _av("unavailable", "template_fallback", pm_basis)
    if SIG["thesis_research_draft"].matches(thesis):
        return _av("unavailable", "agent_failed", ["signature:thesis_research_draft"])
    rewritten = (memo.section_provenance or {}).get("thesis") == "rewrite"
    if rewritten or builder_hit:
        basis = ["provenance:thesis=rewrite"] if rewritten else [f"signature:{builder_hit}"]
        frag = _contains_any(thesis, fragments, 24)
        if frag:
            return _av("unavailable", "derived_from_hidden", [*basis, "fragment:template"])
        # A rewrite whose claim is an LLM-authored headline keeps the
        # analyst's claim; the lever sentence is the builder's. Shown with a
        # note (critique delta 11), not hidden.
        claims = [_normalized_headline(f.headline) for f in available_findings]
        if llm_bb is not None:
            bull = llm_bb.get("bull_case")
            if isinstance(bull, dict):
                claims.append(_normalized_headline(str(bull.get("headline") or "")))
        if any(len(c) >= 12 and c in thesis for c in claims):
            return _av("degraded", "partial_template", [*basis, "claim:llm_headline"])
        return _av("unavailable", "template_fallback", basis)
    frag = _contains_any(thesis, fragments, 24)
    if frag:
        return _av("unavailable", "derived_from_hidden", ["fragment:template"])
    if not thesis.strip():
        return _av("unavailable", "not_produced")
    return _AVAILABLE.model_copy()


def _classify_mispricing(memo: StockMemoOut, ctx: _Ctx, pm_template: bool,
                         fragments: set[str]) -> SectionAvailability:
    mt = memo.mispricing_thesis
    if not (mt.consensus_view or mt.our_view or mt.gap):
        return _av("unavailable", "not_produced")
    if (ctx.eval_memo.section_provenance or {}).get("mispricing") == "fallback":
        return _av("unavailable", "template_fallback", ["provenance:mispricing=fallback"])
    gap_hit = next((s for s in _MISPRICING_GAP_SIGS if SIG[s].matches(mt.gap or "")), None)
    if gap_hit:
        ours = (mt.our_view or "").strip()
        theses = {(memo.one_sentence_thesis or "").strip(),
                  (ctx.eval_memo.one_sentence_thesis or "").strip()} - {""}
        # Relaxed legacy rule (critique delta 6): the fallback quotes the
        # final thesis, which a later patch or rewrite may have replaced.
        derived = (
            ours in theses
            or SIG["mispricing_our_view_pointer"].matches(ours)
            or any(SIG[s].matches(ours) for s in _THESIS_BUILDER_SIGS)
            or _contains_any(ours, fragments, 24) is not None
            or pm_template
        )
        if derived:
            return _av("unavailable", "template_fallback", [f"signature:{gap_hit}", "legacy:our_view"])
    return _AVAILABLE.model_copy()


def _classify_final_verdict(memo: StockMemoOut, plan: _Plan, fragments: set[str],
                            template_bb: dict[str, Any] | None, *, sector_hidden: bool,
                            embedded_thesis: list[str]) -> None:
    av = plan.avail
    text = memo.final_verdict or ""
    if not text.strip():
        av["final_verdict"] = _av("unavailable", "not_produced")
        return
    if (av["one_sentence_thesis"].status == "unavailable"
            or av["confidence_score"].status == "unavailable"):
        # The thesis and confidence are embedded verbatim (graph._build_verdict).
        av["final_verdict"] = _av("unavailable", "derived_from_hidden", ["derived:thesis_or_confidence"])
        return
    if embedded_thesis:
        # A patched thesis is shown, but the verdict still quotes the base's.
        av["final_verdict"] = _av("unavailable", "derived_from_hidden", embedded_thesis)
        return
    stripped = text
    basis: list[str] = []
    if template_bb is not None:
        kd = (template_bb.get("key_disagreement") or "").strip()
        if kd and f" Key disagreement: {kd}" in stripped:
            stripped = stripped.replace(f" Key disagreement: {kd}", "")
            basis.append("stripped:key_disagreement")
        lean = template_bb.get("sector_lean")
        if lean and lean != "balanced" and f" Sector lean: {lean}." in stripped:
            stripped = stripped.replace(f" Sector lean: {lean}.", "")
            basis.append("stripped:sector_lean")
    if sector_hidden:
        cohort = " Cohort placement: see sector view for KPI quartile context."
        if cohort in stripped:
            stripped = stripped.replace(cohort, "")
            basis.append("stripped:cohort_pointer")
        m = re.search(r" Cross-sector pull-through: [^.]*\.", stripped)
        if m:
            stripped = stripped.replace(m.group(0), "")
            basis.append("stripped:cross_sector")
    hidden_breakers = [memo.thesis_breakers[i].title for i in plan.list_hidden.get("thesis_breakers", [])]
    if hidden_breakers:
        titles = [r.title for r in memo.thesis_breakers]
        old = f"Watch items: {', '.join(titles) or 'none flagged.'}"
        if stripped.endswith(old):
            kept = [t for i, t in enumerate(titles)
                    if i not in set(plan.list_hidden.get("thesis_breakers", []))]
            stripped = stripped[: -len(old)] + f"Watch items: {', '.join(kept) or 'none flagged.'}"
            basis.append("stripped:watch_items")
    if _contains_any(stripped, fragments, 40):
        av["final_verdict"] = _av("unavailable", "derived_from_hidden", [*basis, "fragment:template"])
        return
    if basis:
        plan.final_verdict = stripped
        breakers = av.get("thesis_breakers")
        if basis == ["stripped:watch_items"] and breakers is not None and breakers.reason == "debate_unavailable":
            # The only text removed is watch items the failed debate hid
            # (real analysis, not template): say so, as the breakers do.
            av["final_verdict"] = _av("degraded", "debate_unavailable", [*breakers.basis, *basis])
            return
        av["final_verdict"] = _av("degraded", "partial_template", basis)
        return
    av["final_verdict"] = _AVAILABLE.model_copy()


# ---------------------------------------------------------------------------
# The bull/bear debate (D4; design-bullbear-final §12.1)
# ---------------------------------------------------------------------------

# Debate states whose record is shown (and whose texts the number check reads).
DEBATE_SHOWN_STATUSES: frozenset[str] = frozenset({"complete", "partial"})
# The claims each side's case carries (design §4.3: at most five per side).
DEBATE_CASE_MAX_POINTS = 5
# The evidence map served beside a shown debate: the pool is E01..E16.
DEBATE_EVIDENCE_MAP_MAX = 16
_ID_NUMBER = re.compile(r"(\d+)")


def _classify_debate(debate: DebateRecord | None) -> SectionAvailability:
    """The `debate` section, by the record's own status (§12.1).

    None is every memo written with `DEBATE_MODE` off and every memo that
    pre-dates the field: nothing was produced, and the cases keep today's
    rules. `not_run` (no model, a backtest) also keeps today's case rules;
    `unavailable` hides both cases (see `_classify`)."""
    if debate is None:
        return _av("unavailable", "not_produced")
    basis = [f"debate:{debate.reason or debate.status}"]
    if debate.status == "not_run":
        return _av("unavailable", "not_run", basis)
    if debate.status == "unavailable":
        return _av("unavailable", "debate_unavailable", basis)
    if debate.status == "partial":
        # Openings argued, rebuttals lost: shown, with a note saying so.
        # "degraded" is the presenter's "shown with a note" state.
        return _av("degraded", "rebuttals_unavailable", basis)
    return _AVAILABLE.model_copy()


def debate_claim_displayable(claim: DebateClaim) -> bool:
    """A claim a reader may see: not dropped by the grader and not graded
    `unsupported`. The rest stay in the stored record for the audit only."""
    return not claim.dropped and claim.grade != "unsupported"


def _id_order(claim_id: str) -> tuple[str, int, str]:
    m = _ID_NUMBER.search(claim_id or "")
    return ((claim_id or "")[: m.start()] if m else claim_id or "",
            int(m.group(1)) if m else -1, claim_id or "")


def debate_case_projection(record: DebateRecord, side: str) -> BullBearCase:
    """The case a complete or partial debate writes into `bull_case` /
    `bear_case` (§12.1): the side's final headline, and its displayable
    claims' texts verbatim, in id order, at most five.

    The presenter joins the debate's shown claims to the case by that exact
    text (`_present_debate`), so the writer (D6) builds the case with this
    function: a claim the number check withholds from the case then leaves
    the debate panel too, and a withheld figure appears nowhere else."""
    claims = sorted((c for c in record.claims if c.side == side and debate_claim_displayable(c)),
                    key=lambda c: _id_order(c.id))
    return BullBearCase(
        headline=(record.headlines or {}).get(side, "") or "",
        key_points=[c.claim for c in claims[:DEBATE_CASE_MAX_POINTS]],
    )


def _hidden_debate(record: DebateRecord) -> DebateRecord:
    """What a reader gets of a debate whose section is unavailable: why,
    and how it was routed. The deterministic checks stay: they are facts
    the code computed about the memo, not an advocate's argument."""
    return DebateRecord(
        protocol_version=record.protocol_version, status=record.status, reason=record.reason,
        presentation_order=record.presentation_order, route=dict(record.route or {}),
        deterministic_checks=list(record.deterministic_checks or []),
    )


def _present_debate(
    record: DebateRecord, bull: BullBearCase, bear: BullBearCase,
) -> tuple[DebateRecord, dict[int, int], dict[int, int]]:
    """The presented debate and the stored->presented index maps of its
    claims and responses (for the number-check paths).

    Shown claims are the displayable ones whose text the PRESENTED case of
    their side still carries: the join drops a claim the number check
    withheld from the case, or the presenter's template rules hid there.
    Responses to a claim that is not shown go with it (they would point at
    an id the reader cannot find), as do those claims' `unanswered` entries.
    The evidence map keeps the pool entries the shown text cites (≤16),
    without the retrieval audit (`found_by`, `query`); the research plans
    and the usage record are audit data, and the L1 counterfactual
    (`outcome.cf_*`) is stored, never displayed."""
    case_texts = {
        "bull": {p.strip() for p in bull.key_points if isinstance(p, str)},
        "bear": {p.strip() for p in bear.key_points if isinstance(p, str)},
    }
    shown = [(i, c) for i, c in enumerate(record.claims)
             if debate_claim_displayable(c) and (c.claim or "").strip() in case_texts.get(c.side, set())]
    claim_map = {i: n for n, (i, _) in enumerate(shown)}
    shown_ids = {c.id for _, c in shown}
    hidden_ids = {c.id for c in record.claims} - shown_ids
    responses = [(j, r) for j, r in enumerate(record.responses) if r.target in shown_ids]
    response_map = {j: n for n, (j, _) in enumerate(responses)}

    cited: set[str] = set()
    for _, c in shown:
        cited.update(c.evidence)
        quoted = (c.quote or {}).get("evidence")
        if isinstance(quoted, str):
            cited.add(quoted)
    for _, r in responses:
        cited.update(r.evidence)
    for ruling in record.resolution.rulings:
        cited.update(ruling.basis)
    prose = " ".join([
        record.resolution.crux or "", *(v for v in (record.cruxes or {}).values() if isinstance(v, str)),
        *(r.argument or "" for _, r in responses), *(c.falsifier or "" for _, c in shown),
    ])
    evidence = []
    for ev in record.evidence:
        if ev.id in cited or (ev.id and re.search(rf"(?<![\w-]){re.escape(ev.id)}(?![\w-])", prose)):
            evidence.append(ev.model_copy(update={"found_by": [], "query": ""}))
        if len(evidence) >= DEBATE_EVIDENCE_MAP_MAX:
            break
    out = record.model_copy(deep=True)
    out.claims = [c.model_copy(deep=True) for _, c in shown]
    out.responses = [r.model_copy(deep=True) for _, r in responses]
    out.unanswered = [u for u in record.unanswered if u not in hidden_ids]
    out.evidence = evidence
    out.research = {}
    out.usage = {}
    out.outcome = {k: v for k, v in (record.outcome or {}).items() if not k.startswith("cf_")}
    return out, claim_map, response_map


_DEBATE_ITEM_PATH = re.compile(r"^debate\.(?P<list>claims|responses)\[(?P<index>\d+)\](?P<rest>.*)$")


def _debate_renumbered(path: str, claim_map: dict[int, int], response_map: dict[int, int]) -> str | None:
    """A number-check path into the debate after the presenter filtered its
    claims and responses: None when it pointed at an item not shown, else
    the path with the item's presented index."""
    m = _DEBATE_ITEM_PATH.match(path)
    if m is None:
        return path
    mapping = claim_map if m.group("list") == "claims" else response_map
    new = mapping.get(int(m.group("index")))
    return None if new is None else f"debate.{m.group('list')}[{new}]{m.group('rest')}"


# ---------------------------------------------------------------------------
# The item-8 review (D4; owner decision 2026-09-25 item 8)
# ---------------------------------------------------------------------------

def presented_review_status(review: CriticReview, *, live: bool) -> str:
    """The review label a reader sees: `independent`, `not_independent`
    (the W2a "not independently reviewed" label), `rule_based`, or "".

    `live` is the presenter's verdict on the review section (available).
    A label the writer stored wins, except that a review which was not
    live is never shown as independent. A review that was not live
    (rule-based, failed, pending) is "not independently reviewed" however
    old it is (plan P12).

    "independent" is a claim about WHO reviewed, so it comes only from the
    writer that knew (the item-8 reviewer stores it). A live review with no
    stored label, which is every legacy critic review, stays unlabelled:
    that critic crossed provider families only when Anthropic was
    configured and otherwise ran on the author's own provider, and the
    stored review does not record which happened."""
    stored = review.review_status
    if stored and not (stored == "independent" and not live):
        return stored
    return "" if live else "not_independent"


def issue_is_open(issue: ReviewIssue, review: CriticReview) -> bool:
    """True unless the re-check resolved it (P5). The PM's own revision
    never closes an issue: a status of `resolved` counts only when the
    completed re-check lists the id as resolved (and not as open), so a
    superficial edit cannot escape the open-issue display or its cap."""
    if issue.status != "resolved":
        return True
    recheck = review.revision.recheck if review.revision is not None else None
    return not (recheck is not None and recheck.status == "complete"
                and issue.id in recheck.resolved and issue.id not in recheck.open)


def open_issues(review: CriticReview) -> list[ReviewIssue]:
    """The reviewer's open issues, material first (the display order)."""
    return [i for i in _ordered_issues(review) if issue_is_open(i, review)]


def _ordered_issues(review: CriticReview) -> list[ReviewIssue]:
    return sorted(review.issues, key=lambda i: (not issue_is_open(i, review), i.severity != "material"))


def _present_review(review: CriticReview, *, live: bool) -> CriticReview:
    """The item-8 review projection. A review that was not live loses its
    prose (the legacy critic's template text, W2a) but keeps what item 8
    must show: the label, the model asked, and any verdict and issues,
    which are structured findings a cap may rest on, so hiding them would
    show a capped confidence with no reason. Issues are ordered open
    material first; one the re-check did not resolve reads as addressed by
    the PM, i.e. still open."""
    if not live:
        review = CriticReview(
            overall_assessment=UNAVAILABLE_TEXT, review_mode=review.review_mode, challenges=[],
            underweighted_risks=[], suggested_revisions=[],
            advice_compliance_check=review.advice_compliance_check,
            valuation_divergence_assessment=review.valuation_divergence_assessment,
            **{k: getattr(review, k) for k in CRITIC_REVIEW_ITEM8_FIELDS},
        )
    else:
        review = review.model_copy(deep=True)
    issues = [
        i.model_copy(update={"status": "addressed_by_pm"})
        if i.status == "resolved" and issue_is_open(i, review) else i.model_copy()
        for i in _ordered_issues(review)
    ]
    review.review_status = presented_review_status(review, live=live)  # type: ignore[assignment]
    review.issues = issues
    return review


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_availability(
    memo: StockMemoOut, *, patched_fields: frozenset[str] = frozenset(),
    base: StockMemoOut | None = None, chain_complete: bool = True,
) -> dict[str, SectionAvailability]:
    """Per-section verdicts for a RAW memo, keyed by `SECTION_KEYS` (+ dynamic keys).

    `patched_fields`, `base` and `chain_complete` describe a news-patch chain
    (`memo_store.patch_chain_for`): fields a patch replaced are available
    unless an exact signature still matches them, and the PM-view and
    mispricing rules read the chain's base snapshot, because a patch never
    re-runs the PM synthesis."""
    if memo.section_availability:
        return {k: v.model_copy() for k, v in memo.section_availability.items()}
    try:
        return _classify(memo, patched_fields=frozenset(patched_fields), base=base,
                         chain_complete=chain_complete).avail
    except Exception:  # never 500 a memo that validated
        return {k: _av("available", None, ["unclassified"]) for k in SECTION_KEYS}


def unavailable_keys(av: dict[str, SectionAvailability]) -> set[str]:
    """Section keys whose verdict is `unavailable` (the W2b confidence-cap hook)."""
    return {k for k, v in av.items() if v.status == "unavailable"}


def is_hidden(memo: StockMemoOut, key: str) -> bool:
    """True when a PRESENTED memo marks `key` unavailable. Absent map or key
    reads as available, which keeps pre-W2a callers and fixtures working."""
    entry = (memo.section_availability or {}).get(key)
    return entry is not None and entry.status == "unavailable"


def _allowlisted(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keep = _ALLOWED_DATA_KEYS | _PROVENANCE_DATA_KEYS
    return {k: v for k, v in data.items() if k in keep}


def _blank_finding(f: AgentFinding) -> AgentFinding:
    return AgentFinding(
        agent=f.agent, headline=UNAVAILABLE_TEXT, summary=UNAVAILABLE_TEXT, key_points=[],
        confidence=f.confidence, sources=list(f.sources or []), evidence=[],
        data=_allowlisted(f.data), long_form_report=None,
    )


def _finding_at(memo: StockMemoOut, key: str) -> AgentFinding | None:
    if key.startswith("extra_agent_views."):
        return (memo.extra_agent_views or {}).get(key.split(".", 1)[1])
    return getattr(memo, key, None)


def _set_finding(memo: StockMemoOut, key: str, value: AgentFinding | None) -> None:
    if key.startswith("extra_agent_views."):
        if value is not None:
            memo.extra_agent_views[key.split(".", 1)[1]] = value
        return
    setattr(memo, key, value)


def _section_of_field(name: str) -> str:
    """Map a number-check field path ("bull_case.key_points[2]",
    "extra_agent_views.industry_group.summary") to its section key."""
    if name.startswith("sector_agent_view.data.bull_bear_analysis"):
        return "sector_synthesis"
    head = re.split(r"[.\[]", name, maxsplit=1)[0]
    if head == "extra_agent_views":
        parts = name.split(".")
        return ".".join(parts[:2]) if len(parts) > 1 else head
    return head


def _present_unclassified(memo: StockMemoOut) -> StockMemoOut:
    """The memo as stored, when the classifier failed: nothing can be
    judged template, so every section is shown ("unclassified").

    What was never for display still does not leave: the debate goes
    through its own projection (`_present_debate` against the stored cases,
    which drops the dropped and unsupported claims, the research plans, the
    usage record and the L1 counterfactual), and the review carries its
    label. Neither depends on the classifier; if the debate projection
    fails too, only the record's routing is shown."""
    out = memo.model_copy(deep=True)
    av = {k: _av("available", None, ["unclassified"]) for k in SECTION_KEYS}
    debate_av = _classify_debate(memo.debate)
    av["debate"] = debate_av.model_copy(update={"basis": [*debate_av.basis, "unclassified"]})
    if memo.debate is not None:
        try:
            if debate_av.status == "unavailable":
                out.debate = _hidden_debate(memo.debate)
            else:
                out.debate = _present_debate(memo.debate, out.bull_case, out.bear_case)[0]
        except Exception:
            out.debate = _hidden_debate(memo.debate)
            av["debate"] = _av("unavailable", "unclassified", [*debate_av.basis, "unclassified"])
    review = out.risk_committee_challenge
    review.review_status = presented_review_status(  # type: ignore[assignment]
        review, live=review.review_mode == "live")
    out.section_availability = av
    return out


def present_memo(
    memo: StockMemoOut, *, patched_fields: frozenset[str] = frozenset(),
    base: StockMemoOut | None = None, chain_complete: bool = True,
) -> StockMemoOut:
    """A copy of `memo` with template-filled sections hidden and the map set.

    Idempotent: a memo that already carries `section_availability` is a
    presented memo, and is returned (copied) as it is — the map was
    computed from the stored text and must never be recomputed from
    placeholders. Pure: the input is not mutated."""
    if memo.section_availability:
        return memo.model_copy(deep=True)
    try:
        plan = _classify(memo, patched_fields=frozenset(patched_fields), base=base,
                         chain_complete=chain_complete)
    except Exception:  # never 500 a memo that validated
        return _present_unclassified(memo)
    av = plan.avail
    out = memo.model_copy(deep=True)

    # Analyst views: blank the hidden, reduce the shown drill-downs.
    for key in plan.hidden_findings:
        f = _finding_at(out, key)
        if f is not None:
            _set_finding(out, key, _blank_finding(f))
    for key, expansion in plan.long_forms.items():
        f = _finding_at(out, key)
        if f is not None and key not in plan.hidden_findings:
            f.long_form_report = expansion
    if plan.drop_bull_bear_block and "sector_agent_view" not in plan.hidden_findings:
        out.sector_agent_view.data = {
            k: v for k, v in (out.sector_agent_view.data or {}).items() if k != "bull_bear_analysis"
        }

    # Prose sections.
    for key in ("final_pm_view", "one_sentence_thesis", "portfolio_fit", "final_verdict"):
        if av[key].status == "unavailable" and av[key].reason != "not_produced":
            setattr(out, key, UNAVAILABLE_TEXT)
    if plan.final_verdict is not None and av["final_verdict"].status == "degraded":
        out.final_verdict = plan.final_verdict
    if av["mispricing_thesis"].status == "unavailable" and av["mispricing_thesis"].reason != "not_produced":
        out.mispricing_thesis = MispricingThesis(
            consensus_view=UNAVAILABLE_TEXT, our_view=UNAVAILABLE_TEXT, gap=UNAVAILABLE_TEXT,
            falsifiers=[],
        )
    out.risk_committee_challenge = _present_review(
        out.risk_committee_challenge, live=av["risk_committee_challenge"].status != "unavailable")

    # Lists.
    hidden_case_points: dict[str, set[str]] = {}
    for key, (head_hidden, idx) in plan.case_hidden.items():
        case = getattr(out, key)
        drop = set(idx)
        hidden_case_points[key] = {p for i, p in enumerate(case.key_points) if i in drop}
        case.key_points = [p for i, p in enumerate(case.key_points) if i not in drop]
        if head_hidden:
            case.headline = UNAVAILABLE_TEXT
    for key, idx in plan.list_hidden.items():
        drop = set(idx)
        setattr(out, key, [x for i, x in enumerate(getattr(out, key)) if i not in drop])

    # The debate, joined to the cases as they are now presented.
    claim_map: dict[int, int] = {}
    response_map: dict[int, int] = {}
    if out.debate is not None:
        if av["debate"].status == "unavailable":
            out.debate = _hidden_debate(out.debate)
        else:
            out.debate, claim_map, response_map = _present_debate(out.debate, out.bull_case, out.bear_case)

    # Diligence rounds.
    for r_i, rnd in enumerate(out.round_findings or []):
        for k, f in list((rnd.findings or {}).items()):
            if (r_i, k) in plan.round_hidden:
                rnd.findings[k] = _blank_finding(f)
                continue
            # The same drill-down rule as the grid (critique delta 1): only
            # the analyst's expansion is shown, in every round.
            if f.long_form_report:
                f.long_form_report = _long_form_expansion(f.long_form_report)
            if (r_i, k) in plan.round_bb_drop:
                f.data = {dk: dv for dk, dv in (f.data or {}).items() if dk != "bull_bear_analysis"}

    # The compatibility event carries the legacy list verbatim; filter the
    # same template points out of it (critique delta 10).
    if any(hidden_case_points.values()):
        events: list[dict[str, Any]] = []
        for e in out.degradation_events:
            if (isinstance(e, dict) and e.get("error_type") == "LegacyCaseShape"
                    and hidden_case_points.get(str(e.get("field")))
                    and isinstance(e.get("original_value"), list)):
                gone = hidden_case_points[str(e["field"])]
                e = {**e, "original_value": [
                    v for v in e["original_value"]
                    if not ((isinstance(v, str) and v in gone)
                            or (isinstance(v, dict) and v.get("key_point") in gone))
                ]}
            events.append(e)
        out.degradation_events = events

    # W2b number check: drop withheld items and claims tied to hidden sections.
    nc = out.quality.number_check if out.quality is not None else None
    if nc is not None:
        hidden = unavailable_keys(av)
        dropped: dict[str, list[int]] = {
            **{k: sorted(v[1]) for k, v in plan.case_hidden.items()},
            **{k: sorted(v) for k, v in plan.list_hidden.items()},
        }
        hidden_heads = {k for k, (h, _) in plan.case_hidden.items() if h}

        def moved_path(path: str) -> str | None:
            if path.startswith("debate."):
                return _debate_renumbered(path, claim_map, response_map)
            return _renumbered(path, dropped, hidden_heads)

        claims = []
        for c in nc.claims:
            if _section_of_field(c.field) in hidden:
                continue
            if c.field.endswith(".long_form_report"):
                rebased = _rebased_long_form_claim(memo, out, c)
                if rebased is None:
                    continue
                c = rebased
            moved = moved_path(c.field)
            if moved is not None:
                claims.append(c if moved == c.field else c.model_copy(update={"field": moved}))
        nc.claims = claims
        # Fields a patch changed name list items by their STORED index; they
        # move with the presenter's dropped items exactly as claims do.
        nc.unchecked_fields = [
            moved for p in nc.unchecked_fields
            if _section_of_field(p) not in hidden
            and (moved := moved_path(p)) is not None
        ]
        # A withheld item was removed before storage, so its index refers to
        # the pre-check list and is left alone; its TEXT is judged by the
        # same rules as a stored item of that list.
        nc.withheld = [
            w for w in nc.withheld
            if _section_of_field(w.field) not in hidden
            and not (_section_of_field(w.field) in _LIST_SECTIONS
                     and _case_item_template(w.text, plan.fragments, plan.scenarios_templated,
                                             plan.llm_bb))
        ]

    out.section_availability = av
    return out


_LIST_SECTIONS = frozenset({"key_risks", "thesis_breakers", "catalysts", "bull_case", "bear_case"})
_LIST_ITEM_PATH = re.compile(
    r"^(?P<section>key_risks|thesis_breakers|catalysts|(?:bull|bear)_case)"
    r"(?P<mid>\.key_points)?\[(?P<index>\d+)\](?P<rest>.*)$"
)


def _rebased_long_form_claim(memo: StockMemoOut, out: StockMemoOut, c: NumberClaim) -> NumberClaim | None:
    """A long-form claim moved from the STORED report to the presented one.

    The number check stores long-form offsets into the full report (the
    deterministic body, then the marker, then the analyst's expansion); the
    presenter shows only the stripped expansion (`_long_form_expansion`).
    The claim moves back by everything the presenter cut in front of it, and
    is dropped when it no longer indexes the shown text — a renderer relies
    on `text[start:end] == raw`."""
    key = c.field[: -len(".long_form_report")]
    f_stored, f_shown = _finding_at(memo, key), _finding_at(out, key)
    stored = f_stored.long_form_report if f_stored is not None else None
    shown = f_shown.long_form_report if f_shown is not None else None
    if not isinstance(stored, str) or not isinstance(shown, str):
        return None
    if stored == shown:
        return c if shown[c.start:c.end] == c.raw else None
    at = stored.find(_LONG_FORM_MARKER)
    if at < 0:
        return None
    rest = stored[at + len(_LONG_FORM_MARKER):]
    cut = at + len(_LONG_FORM_MARKER) + (len(rest) - len(rest.lstrip()))
    start, end = c.start - cut, c.end - cut
    if start < 0 or end > len(shown) or shown[start:end] != c.raw:
        return None
    return c.model_copy(update={"start": start, "end": end})


def _renumbered(path: str, dropped: dict[str, list[int]], head_hidden: set[str]) -> str | None:
    """A number-check field path after the presenter drops list items: None
    when it pointed at a dropped item (or a hidden case headline), else the
    path with its index moved down past the dropped items before it."""
    section = _section_of_field(path)
    if section in head_hidden and path.startswith(f"{section}.headline"):
        return None
    m = _LIST_ITEM_PATH.match(path)
    gone = dropped.get(section) or []
    if m is None or not gone:
        return path
    if section in ("bull_case", "bear_case") and m.group("mid") is None:
        return path  # not a key_points path
    i = int(m.group("index"))
    if i in gone:
        return None
    new = i - sum(1 for g in gone if g < i)
    return f"{section}{m.group('mid') or ''}[{new}]{m.group('rest')}"
