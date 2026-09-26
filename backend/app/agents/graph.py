"""Agent graph (LangGraph-style structure, hand-rolled).

The graph orchestrates a single stock memo generation:

  classify_intent
        │
        ▼
  fan-out specialists
   ├─ sector_agent
   ├─ earnings_agent
   ├─ filing_agent
   ├─ valuation_agent (uses DCF)
   ├─ comps_agent
   ├─ macro_agent
   └─ risk_agent
        │
        ▼
  draft_memo
        │
        ▼
  critic_agent  (Risk Committee)
        │
        ▼
  pm_synthesis (final view, rating, confidence)

We don't pull in a full LangGraph dependency to keep the container slim; the
shape and naming match the LangGraph mental model and could be swapped in
trivially.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, get_args

from ..config import settings
from ..finance.dcf import fmt_price, fmt_upside
from ..schemas import (
    AgentFinding,
    AgentTrace,
    BullBearCase,
    CatalystItem,
    CompsResult,
    ConfidenceAssessment,
    ConfidenceCap,
    CriticReview,
    CritiqueQuestion,
    DCFResult,
    MemoQuality,
    MispricingThesis,
    NumberCheck,
    RatingLabel,
    RatingReconciliation,
    RiskItem,
    RoundFindings,
    ScorecardSummary,
    StockMemoOut,
    ValuationVerdict,
    score_from_rating_label,
)  # CriticReview imported for the safe-runner fallback path  # noqa: F401
from ..services.checkpoint_store import checkpointed
from ..services.filings_service import get_filings
from ..services.fundamentals_service import get_full_financials
from ..services.transcripts_service import latest_transcript
from ..services.valuation_service import build_comps, build_dcf
from . import llm, memo_quality, news_context, number_check, prompts, roster, scorecard_context
from .critic_agent import run_critic
from .log_safety import redact
from .memo_context import (
    AnalystRound,
    DCFStage,
    DegradationNote,
    MemoInputs,
    PMOpinion,
    QualityOutcome,
    VerdictOutcome,
)
from .risk_agent import derive_risk_items, risk_item_from_text
from .safe_runner import (
    DegradationLog,
    note_soft,
    safe_call,
    safe_critic,
    safe_finding,
)
from .source_ledger import SourceLedger, active_ledger, register_source
from .tools import evidence_quality

if TYPE_CHECKING:  # the ORM stays a lazy import at runtime (see _persist_memo_snapshot)
    from ..models import MemoSnapshot

log = logging.getLogger(__name__)

T = TypeVar("T")

# W2b 7(a) source registration for DCF results: prose is read only under
# these keys (the engine's own labels, summary and guardrail messages;
# `name` is for DCFSensitivity.name). Scenario drivers are written by an
# LLM in live runs (`scenario_assumptions._parse_side` copies each driver's
# name, rationale and assumption_changes from its JSON), so the whole
# `drivers` subtree is never a source for the figures it quotes — a figure
# in a driver name would otherwise trace to "dcf:initial" (the memo prints
# it as "DCF driver — {name}: ..."). The other keys stay excluded wherever
# they appear.
DCF_TEXT_KEYS = ("summary", "message", "label", "name", "row_axis", "col_axis", "metric")
DCF_LLM_KEYS = ("drivers", "rationale", "assumption_changes")


# ---------------------------------------------------------------------------
# Memo construction
# ---------------------------------------------------------------------------

_THESIS_ANTI_PATTERN = re.compile(
    r"""^                       # start
        (?P<co>[^—:]{2,80})     # company / ticker prefix
        \s+[—–-]\s+             # em / en / hyphen dash
        (?P<sector>[A-Za-z][\w &/]{2,40})
        \s*/\s*                 # sector / industry slash
        .{2,200}                # industry + regime + any inline clauses
        DCF\ base\ case          # the giveaway phrase
        """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)


def _looks_like_anti_pattern_thesis(text: str) -> bool:
    """True when `text` matches the explicit anti-pattern the PM prompt
    forbids ("{Company} — {Sector} / {industry}, {hook}; DCF base case
    +X% suggests material upside.").

    We use this to detect when the LLM ignored the prompt OR the
    deterministic fallback regressed to the templated form, so we can
    rewrite the sentence before it ships into the memo.
    """
    if not text:
        return False
    return bool(_THESIS_ANTI_PATTERN.match(text.strip()))


# A bull/bear price ratio the valuation analyst computed itself from the
# DCF it was shown ("The 3.9x bull/bear ratio", "3.9x bull-to-bear
# spread", "a bull/bear ratio of 3.9x"). Only a multiple attached to
# explicit bull/bear wording is a candidate: "27.5x earnings" or
# "16.4x EV/EBITDA" never is.
_BULL_BEAR = r"bull(?:\s*/\s*|\s*-\s*to\s*-\s*|\s+to\s+|\s*[-–—]\s*)bear\b"
_RATIO_BEFORE_BULL_BEAR = re.compile(
    rf"(?<![\w.])(?P<num>\d+(?:\.\d+)?)[x×](?=\s+{_BULL_BEAR})", re.IGNORECASE,
)
_RATIO_AFTER_BULL_BEAR = re.compile(
    rf"\b{_BULL_BEAR}\s+(?:ratio|multiple)\s+(?:of\s+)?"
    rf"(?P<num>\d+(?:\.\d+)?)[x×](?!\w|\.\d)",
    re.IGNORECASE,
)
# A formatted DCF figure must stand alone in the prose: "9.4%" is not the
# tail of "19.4%", and "$373" is not the head of "$373.50" or "$3,730".
# The lead guard applies only to a figure that starts with a digit; one
# led by "$" or a sign is already delimited, even straight after a comma
# ("+9.4%,-43.5%").
_FIGURE_LEAD = r"(?:(?=\d)(?<![\d.,])|(?!\d))"
_FIGURE_TAIL = r"(?!\d|[.,]\d)"
_UPDOWN_WORD = r"(?:(?P<ws>\s+)(?P<word>(?i:upside|downside))\b)?"
# An integer magnitude ("23% upside") is too common a phrase to trust on
# its own; it counts as a DCF figure only inside a clause that names a
# scenario (or quotes an old DCF price, checked separately).
_CLAUSE_BREAK = re.compile(r"[;\n]|[.!?](?=\s)")
_DCF_CLAUSE_WORD = re.compile(r"\b(?:base|bull|bear|dcf)\b", re.IGNORECASE)


def _bull_bear_ratio(d: DCFResult) -> float | None:
    """Bull implied price over bear implied price, or None when either is
    unavailable or non-positive (a ratio of a negative price means nothing)."""
    bull, bear = d.bull.implied_share_price, d.bear.implied_share_price
    if bull is None or bear is None or bull <= 0 or bear <= 0:
        return None
    return bull / bear


def _refresh_dcf_references(
    finding: AgentFinding | None,
    old: DCFResult | None,
    new: DCFResult | None,
) -> None:
    """Rewrite stale DCF numbers baked into an agent finding's prose (B2).

    The valuation agent runs inside the agent rounds on the original DCF
    and formats its numbers into headline/summary/key_points (and the
    long-form drill-down rendered from them). The PM DCF Adjuster then
    replaces the working DCF, so without this pass the memo prints two
    different DCF base upsides (e.g. +49% in the valuation card, +17% in
    the DCF section). We substitute every formatted variant of the old
    scenario numbers with the new ones, in place, and append a
    transparency note so the reader knows the figures are PM-adjusted.

    FIX-008: figures DERIVED from the old scenarios (the bull/bear price
    ratio, "N% downside" magnitudes) are recomputed from `new` and printed
    at the precision the prose used. A derived figure is rewritten only
    when it matches the old model at that precision; the replacement
    always comes from the structured `new` DCF, never from the prose.
    Substitution is one pass, so a new value equal to another scenario's
    old value is not rewritten twice, and a formatted string that two
    scenarios share with different new values is left alone as ambiguous.
    A worded magnitude whose scenario changed sign is rebuilt whole from
    the new value ("22.0% upside" -> "5.0% downside"), and an integer one
    ("23% upside") is trusted only in a clause about the DCF.
    """
    if finding is None or old is None or new is None or old is new:
        return

    def _pct_strs(x: float | None) -> tuple[str, ...]:
        # Signed forms first (what the deterministic path emits), then the
        # one-decimal unsigned form LLM prose tends to use. The unsigned
        # integer form ("49%") is deliberately excluded — too collision-
        # prone with margins/percentages that aren't DCF upside — unless
        # "upside"/"downside" follows it (see _magnitude_strs).
        # An unavailable number has no formatted variants: "n/a" is far
        # too common a token to substitute, and we cannot invent a
        # replacement for a figure that was never printed.
        if x is None:
            return ()
        return (f"{x:+.0%}", f"{x:+.1%}", f"{x * 100:+.0f}%",
                f"{x * 100:+.1f}%", f"{x * 100:.1f}%")

    def _usd_strs(x: float | None) -> tuple[str, ...]:
        if x is None:
            return ()
        return (f"${x:,.2f}", f"${x:,.0f}")

    def _magnitude_strs(x: float | None) -> tuple[str, ...]:
        # Unsigned "63% downside" / "22% upside": the magnitude is only a
        # DCF figure when the direction word follows it, and the integer
        # form only inside a clause about the DCF (_in_dcf_clause).
        if x is None:
            return ()
        return (f"{abs(x) * 100:.0f}%", f"{abs(x) * 100:.1f}%")

    def _direction(x: float | None) -> str | None:
        if x is None or x == 0:
            return None
        return "upside" if x > 0 else "downside"

    # Every old string collects every new string it maps to; only a
    # one-to-one mapping to a different string is substituted.
    targets: dict[str, set[str]] = {}
    worded_targets: dict[tuple[str, str], set[tuple[str, str]]] = {}
    old_prices: set[str] = set()
    for o_s, n_s in ((old.base, new.base), (old.bull, new.bull), (old.bear, new.bear)):
        if o_s is None or n_s is None:
            continue
        old_prices.update(_usd_strs(o_s.implied_share_price))
        # Settle this scenario's own mapping first, first form winning: a
        # negative's one-decimal form IS its signed form ("-5.0%"), and
        # when the new value is positive the two new forms differ ("+3.2%"
        # vs "3.2%"). That is one figure, not two scenarios disagreeing,
        # so it must not reach the cross-scenario ambiguity check below.
        # zip() stops at the shorter tuple, so a None on either side yields
        # no pairs for that field rather than a half-substituted memo.
        own: dict[str, str] = {}
        for o_str, n_str in zip(_pct_strs(o_s.upside_pct), _pct_strs(n_s.upside_pct)):
            own.setdefault(o_str, n_str)
        for o_str, n_str in zip(_usd_strs(o_s.implied_share_price),
                                _usd_strs(n_s.implied_share_price)):
            own.setdefault(o_str, n_str)
        for o_str, n_str in own.items():
            targets.setdefault(o_str, set()).add(n_str)
        o_dir, n_dir = _direction(o_s.upside_pct), _direction(n_s.upside_pct)
        if o_dir is None or n_dir is None:
            continue
        # A worded figure carries its direction: when the adjustment flips
        # the scenario's sign, "22.0% upside" becomes "5.0% downside", both
        # halves taken from the new value, so neither the old magnitude nor
        # an inverted direction survives.
        for o_str, n_str in zip(_magnitude_strs(o_s.upside_pct),
                                _magnitude_strs(n_s.upside_pct)):
            worded_targets.setdefault((o_str, o_dir), set()).add((n_str, n_dir))
    subs = {k: next(iter(v)) for k, v in targets.items() if len(v) == 1 and k not in v}
    worded = {k: next(iter(v)) for k, v in worded_targets.items()
              if len(v) == 1 and k not in v}
    old_ratio, new_ratio = _bull_bear_ratio(old), _bull_bear_ratio(new)
    if not subs and not worded and (
        old_ratio is None or new_ratio is None or old_ratio == new_ratio
    ):
        return

    def _alternation(strs: set[str]) -> str:
        return "|".join(re.escape(t) for t in sorted(strs, key=len, reverse=True))

    tokens = set(subs) | {tok for tok, _ in worded}
    figure_re = (
        re.compile(_FIGURE_LEAD + "(?P<tok>" + _alternation(tokens) + ")"
                   + _FIGURE_TAIL + _UPDOWN_WORD)
        if tokens else None
    )
    old_price_re = (
        re.compile(_FIGURE_LEAD + "(?:" + _alternation(old_prices) + ")" + _FIGURE_TAIL)
        if old_prices else None
    )

    def _in_dcf_clause(m: re.Match[str]) -> bool:
        text, lo, hi = m.string, 0, len(m.string)
        for brk in _CLAUSE_BREAK.finditer(text):
            if brk.end() <= m.start():
                lo = brk.end()
            elif brk.start() >= m.end():
                hi = brk.start()
                break
        clause = text[lo:hi]
        return bool(_DCF_CLAUSE_WORD.search(clause)) or bool(
            old_price_re is not None and old_price_re.search(clause)
        )

    def _ratio_repl(m: re.Match[str]) -> str:
        num = m["num"]
        places = len(num.partition(".")[2])
        if old_ratio is None or new_ratio is None or f"{old_ratio:.{places}f}" != num:
            # Bull/bear wording, but not the old model's ratio at the
            # printed precision: some other figure, so leave it.
            return m.group(0)
        text, at = m.group(0), m.start("num") - m.start()
        return text[:at] + f"{new_ratio:.{places}f}" + text[at + len(num):]

    def _figure_repl(m: re.Match[str]) -> str:
        token, ws, word = m["tok"], m["ws"] or "", m["word"]
        if word is None:
            return subs.get(token, token)
        hit = worded.get((token, word.lower()))
        # The one-decimal magnitude is specific enough to trust anywhere;
        # the integer one only in a clause about the DCF (see above).
        if hit is not None and ("." in token or _in_dcf_clause(m)):
            n_tok, n_dir = hit
            if n_dir != word.lower():
                log.info("DCF refresh rewrote %r as %r: the PM adjustment "
                         "flipped that scenario's sign", f"{token} {word}",
                         f"{n_tok} {n_dir}")
            return n_tok + ws + (n_dir.capitalize() if word[:1].isupper() else n_dir)
        return subs.get(token, token) + ws + word

    def _sub(text: str) -> str:
        # Ratios first, identified against the untouched prose; they end in
        # "x", so the figure pass below can never touch their output.
        text = _RATIO_BEFORE_BULL_BEAR.sub(_ratio_repl, text)
        text = _RATIO_AFTER_BULL_BEAR.sub(_ratio_repl, text)
        return figure_re.sub(_figure_repl, text) if figure_re is not None else text

    finding.headline = _sub(finding.headline or "")
    finding.summary = _sub(finding.summary or "")
    finding.key_points = [_sub(p) for p in (finding.key_points or [])]
    # The drill-down is rendered from the fields above before the PM
    # adjustment runs (attach_long_form in the analyst round), so it
    # carries the same stale figures.
    if finding.long_form_report:
        finding.long_form_report = _sub(finding.long_form_report)
    for ev in (getattr(finding, "evidence", None) or []):
        try:
            ev.excerpt = _sub(ev.excerpt or "")
        except Exception as exc:  # pragma: no cover — citations are best-effort
            # (a) the citation keeps its pre-adjustment number; the finding
            # prose above already carries the corrected figures.
            log.debug("citation excerpt substitution skipped: %s", type(exc).__name__)
    note = (
        f"DCF figures reflect PM-adjusted assumptions "
        f"(base case {fmt_upside(new.base.upside_pct, decimals=0)} vs current)."
    )
    if note not in finding.key_points:
        finding.key_points = list(finding.key_points) + [note]


def _risk_items_from_bear_case(bear: BullBearCase | None) -> list[RiskItem]:
    """Backfill `key_risks` from the bear case when profile-driven risk
    extraction returned nothing (B4).

    Live FMP profiles carry no `risks` field, so `derive_risk_items`
    routinely comes back empty — which both ships a memo with no risk
    section and starves the thesis builder into template filler. The bear
    case is built from real bear-polarity specialist findings, so its key
    points are the de-facto risk list. The pure price-target line is
    skipped (a DCF bear print is not a risk statement).
    """
    if bear is None:
        return []
    items: list[RiskItem] = []
    seen: set = set()
    for point in bear.key_points or []:
        text = (point or "").strip()
        low = text.lower()
        if not text or low.startswith("dcf bear case implies"):
            continue
        if low[:60] in seen:
            continue
        seen.add(low[:60])
        items.append(risk_item_from_text(text))
        if len(items) >= 4:
            break
    return items


def _build_mispricing_fallback(memo: StockMemoOut) -> MispricingThesis:
    """Deterministic mispricing thesis when the PM left the structure
    blank (B6 — no LLM, LLM failure, or the PM declined).

    Built from the reconciled `valuation_verdict` plus the final thesis
    and risk list, so it can never contradict the rest of the memo. When
    the verdict is fairly_priced it says so explicitly rather than
    rendering an empty card.
    """
    vv = memo.valuation_verdict
    ticker = memo.ticker

    consensus_bits: list[str] = []
    if vv.comps_ev_ebitda_premium is not None:
        d = "premium" if vv.comps_ev_ebitda_premium > 0 else "discount"
        consensus_bits.append(
            f"pays a {abs(vv.comps_ev_ebitda_premium):.0%} EV/EBITDA {d} "
            f"versus peers"
        )
    consensus_view = (
        f"The market {' and '.join(consensus_bits)} for {ticker}."
        if consensus_bits
        else f"Consensus pricing for {ticker} embeds steady execution at the current multiple."
    )

    our_view = memo.one_sentence_thesis or f"See the {ticker} thesis above."

    if vv.basis == "evidence":
        # W2b 7(b): the verdict is an evidence read now, not the blended
        # rating, so the card must not say "the blended read calls" it.
        if vv.verdict == "mixed":
            gap = (f"Valuation signals conflict ({vv.summary.removeprefix('Net read: ')}); "
                   f"no single mispricing call.")
        elif vv.verdict == "fairly_priced":
            gap = ("No material mispricing on our work — the valuation evidence does not "
                   "point either way.")
        elif vv.dcf_base_upside is not None:
            gap = (f"Our DCF base case implies {vv.dcf_base_upside:+.0%} to fair value; "
                   f"the valuation evidence reads {vv.verdict.replace('_', ' ')}.")
        else:
            gap = f"The valuation evidence reads {vv.verdict.replace('_', ' ')}."
    # A rating-derived verdict (every memo stored before W2b, and the
    # fixtures that pin the presenter's legacy signatures) keeps the
    # wording that was true of it.
    elif vv.verdict == "fairly_priced":
        gap = (
            "No material mispricing on our work — the signals offset and "
            "the blended rating lands at fair value."
        )
    elif vv.dcf_base_upside is not None:
        gap = (
            f"Our DCF base case implies {vv.dcf_base_upside:+.0%} to fair "
            f"value; the blended read calls the name {vv.verdict.replace('_', ' ')}."
        )
    else:
        gap = f"The blended read calls the name {vv.verdict.replace('_', ' ')}."

    falsifiers = [
        r.title for r in (memo.thesis_breakers or memo.key_risks or [])[:3]
    ]
    return MispricingThesis(
        consensus_view=consensus_view[:1000],
        our_view=our_view[:1000],
        gap=gap[:1000],
        falsifiers=falsifiers,
    )


def _mispricing_lever_clause(
    verdict_word: str,
    upside: float | None,
    drivers: list[str],
    risks: list[str],
) -> str:
    """Sentence-2 lever clause for a mispriced name.

    Cites the DCF number only when its sign agrees with the verdict (else
    it contradicts the call). Names a real driver / risk when we have one,
    and degrades to a clean single clause — never the "core driver
    execution vs. the dominant risk" placeholder — when we don't.
    """
    driver = (drivers[0] if drivers else "").strip()
    risk = (risks[0] if risks else "").strip()
    dcf_agrees = upside is not None and (
        (verdict_word == "undervalued" and upside > 0)
        or (verdict_word == "overvalued" and upside < 0)
    )
    if dcf_agrees:
        sign = "+" if (upside or 0) > 0 else ""
        lead = f"DCF base case implies {sign}{(upside or 0) * 100:.0f}% to fair value"
    elif verdict_word == "overvalued":
        lead = "The multiple already prices in the bull case"
    else:  # undervalued, but the DCF doesn't corroborate the call
        lead = "The market is under-pricing the durable part of the franchise"

    if driver and risk:
        return f"{lead}; the swing factor is {driver} against {risk}."
    if driver:
        return f"{lead}; the swing factor is {driver}."
    if risk:
        return f"{lead}; the key risk is {risk}."
    return f"{lead}."


def _gap_clause_agrees(gap_clause: str, verdict_word: str) -> bool:
    """True when the consensus-gap clause points the same way as the
    verdict. `_market_gap_clause` phrases upside as "upside the market may
    be missing" and downside as "downside the market may be
    underweighting" — keep it from carrying sentence 2 in the direction
    that contradicts the verdict word.
    """
    low = gap_clause.lower()
    if verdict_word == "undervalued":
        return "upside" in low
    if verdict_word == "overvalued":
        return "downside" in low
    return True


def _build_thesis_from_findings(
    profile: dict,
    findings: dict[str, AgentFinding],
    dcf: DCFResult | None,
    ticker: str,
    *,
    verdict_word: str | None,
) -> str:
    """Compose a short-form thesis (2-3 sentences) from the specialists'
    findings. Mirrors the structure required by PM_SYNTHESIS_PROMPT so
    the deterministic fallback reads the same as the LLM happy path:

        Sentence 1 — VERDICT: ticker + correctly priced / over / under,
                     and the ONE thing that defines the call.
        Sentence 2 — LEVER / PERFORMANCE: if mispriced, the segment or
                     metric where the gap shows up; if correctly priced,
                     what makes it worth owning at current price.

    Used by both the post-LLM anti-pattern rewrite and the deterministic
    fallback in `_pm_synthesis` — keeps the logic in one place so the
    two paths can't drift.
    """
    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []
    ticker_sym = (profile.get("ticker") or ticker or "").upper()

    def _claim_from_finding(f: AgentFinding | None) -> str | None:
        if f is None:
            return None
        head = (getattr(f, "headline", "") or "").strip().rstrip(".,;:")
        if not head or len(head) < 12:
            return None
        low = head.lower()
        # Reject cohort labels, scenario tags, intake-skip headlines, and
        # the very anti-pattern we're trying to escape — none of these
        # state a claim a reader can defend.
        skip_patterns = (
            "highlights", "view", "profile", "view for", "scenario:",
            "regime:", "cohort placement", "skipped per pm intake",
            "dcf base case", " — ",
        )
        if any(p in low for p in skip_patterns):
            return None
        return head

    sector_finding = findings.get("sector")
    bull_headline: str | None = None
    if sector_finding is not None and isinstance(sector_finding.data, dict):
        bb = sector_finding.data.get("bull_bear_analysis") or {}
        if isinstance(bb, dict):
            bull_case = bb.get("bull_case") or {}
            if isinstance(bull_case, dict):
                bh = (bull_case.get("headline") or "").strip().rstrip(".,;:")
                if 12 <= len(bh) <= 180 and "dcf base case" not in bh.lower():
                    bull_headline = bh

    claim = (
        bull_headline
        or _claim_from_finding(findings.get("valuation"))
        or _claim_from_finding(sector_finding)
        or _claim_from_finding(findings.get("earnings"))
        or _claim_from_finding(findings.get("filing"))
        or (drivers[0] if drivers else None)
    )

    # Drop a leading scenario label ("Bull case:", "Bear case:", …). It
    # reads oddly once the verdict word precedes the claim and can flatly
    # contradict it (a bull-case headline behind an "overvalued" verdict).
    if claim:
        claim = re.sub(
            r"^\s*(bull|bear|base)[\s-]*case\s*[:\-—–]\s*", "", claim,
            flags=re.IGNORECASE,
        ).strip()

    upside = dcf.base.upside_pct if dcf and dcf.base else None

    # --- Sentence 1: VERDICT ---
    # The caller decides the word (W2b 7(b)): the valuation EVIDENCE word
    # (`memo_quality.verdict_word`), or the rating's word when there is no
    # evidence or the PM's divergence was accepted. None is a `mixed`
    # verdict: the signals conflict, so no single word is stated.
    if verdict_word is None:
        lead = f"{ticker_sym}: valuation signals are mixed" if ticker_sym else "Valuation signals are mixed"
        sentence_1 = f"{lead} — {claim}." if claim else f"{lead}; no single mispricing call."
    elif claim:
        sentence_1 = f"{ticker_sym} is {verdict_word} — {claim}." if ticker_sym else f"{verdict_word.capitalize()} — {claim}."
    elif verdict_word == "fairly priced":
        sector = (profile.get("sector") or "core").strip().lower()
        sentence_1 = (
            f"{ticker_sym} is fairly priced on our work — no actionable edge in {sector}."
            if ticker_sym
            else f"Fairly priced on our work — no actionable edge in {sector}."
        )
    else:
        # Mispriced on our valuation read, but no specialist headline
        # carries the call — stay consistent with the verdict word.
        sentence_1 = (
            f"{ticker_sym} screens {verdict_word} on our valuation read, "
            f"though no single specialist headline defines the call."
            if ticker_sym
            else f"Screens {verdict_word} on our valuation read, "
            f"though no single specialist headline defines the call."
        )

    # --- Sentence 2: LEVER (if mispriced) or PERFORMANCE PATH (if not) ---
    sentence_2 = ""
    try:
        gap_clause = _market_gap_clause(profile, dcf, ticker_sym)
    except Exception as exc:  # pragma: no cover
        # (b) sentence 2 of the thesis silently loses the consensus-gap
        # framing and falls back to the generic lever clause. That changes
        # what the reader sees, so it belongs on the memo banner, not only
        # in a debug log. `note_soft` no-ops outside a memo run.
        log.warning("market-gap clause failed for %s: %s", ticker_sym, type(exc).__name__)
        note_soft(
            "Thesis Builder",
            f"consensus-gap clause unavailable: {redact(exc)}",
            kind=type(exc).__name__,
        )
        gap_clause = ""

    if verdict_word in ("undervalued", "overvalued"):
        # Only let the consensus-gap clause carry sentence 2 when its
        # direction agrees with the verdict — otherwise it reintroduces
        # the very contradiction we're fixing.
        if gap_clause and _gap_clause_agrees(gap_clause, verdict_word):
            sentence_2 = gap_clause
        else:
            sentence_2 = _mispricing_lever_clause(verdict_word, upside, drivers, risks)
    else:
        # Correctly priced — describe the performance path. Degrades
        # cleanly with no named driver (no "compounding fundamentals
        # compounding" stutter, no template filler).
        if drivers:
            sentence_2 = (
                f"At this price the return comes from {drivers[0].lower()} "
                f"compounding at trend, not a re-rating — own the floor, not the multiple."
            )
        else:
            sentence_2 = (
                "At this price the return comes from steady compounding, "
                "not a re-rating — own the floor, not the multiple."
            )

    return f"{sentence_1} {sentence_2}".strip()


def _market_gap_clause(
    profile: dict, dcf: DCFResult | None, ticker: str,
) -> str:
    """Wave 8R — write the "what the market is missing" sentence.

    Compares model growth vs. analyst consensus growth (5y avg). When
    the gap is material (≥2pp), names which side is leaning + why
    (driven by the company's first-line driver / risk).
    Returns the empty string when no meaningful gap exists.
    """
    if dcf is None:
        return ""
    try:
        model_growths = list(dcf.base.assumptions.revenue_growth)
        if not model_growths:
            return ""
        model_avg = sum(model_growths) / len(model_growths)
    except Exception as exc:
        # (a) no growth path on the DCF means there is no gap to describe.
        log.debug("market-gap clause: no model growth path for %s: %s", ticker, type(exc).__name__)
        return ""

    # Pull consensus from the data service via the same helper the engine uses.
    consensus_avg = None
    try:
        from ..finance.dcf import _consensus_growth_path
        from ..services.data_service import get_data_service
        estimates = get_data_service().get_estimates(ticker)
        consensus = _consensus_growth_path(estimates)
        if consensus:
            consensus_avg = sum(consensus) / len(consensus)
        # W2b 7(a): the thesis quotes the consensus path and its average.
        register_source("estimates", f"estimates:{ticker}", {
            "consensus_revenue_growth": consensus, "consensus_revenue_growth_average": consensus_avg,
            "estimates": estimates,
        })
    except Exception as exc:
        # (a) the clause degrades to its "vs. trend" framing below, which is
        # the same output as "no consensus published" — not a memo change.
        log.debug("market-gap clause: consensus unavailable for %s: %s", ticker, type(exc).__name__)
        consensus_avg = None

    drivers = profile.get("drivers") or []
    risks = profile.get("risks") or []

    if consensus_avg is None:
        # No consensus visible — make a softer "vs. trend" framing instead.
        return ""

    gap = model_avg - consensus_avg
    if abs(gap) < 0.02:  # within 2pp — not material
        return ""

    if gap > 0:
        # Model is more bullish than the Street.
        driver = drivers[0] if drivers else "core driver execution"
        return (
            f"Model sees ~{model_avg * 100:.0f}% revenue growth vs. consensus "
            f"~{consensus_avg * 100:.0f}% — the **upside the market may be "
            f"missing** is durability of {driver}."
        )
    # Model is more cautious than the Street.
    risk = risks[0] if risks else "execution slip on the dominant driver"
    return (
        f"Model sees ~{model_avg * 100:.0f}% revenue growth vs. consensus "
        f"~{consensus_avg * 100:.0f}% — the **downside the market may be "
        f"underweighting** is {risk}."
    )


def _build_scores_dict(
    *, blended_confidence: float, raw_confidence: float, ev_q: float,
    sector_finding: AgentFinding, valuation_finding: AgentFinding,
    risk_finding: AgentFinding, earnings_finding: AgentFinding | None = None,
    profile: dict, ratios: dict, earnings: dict,
) -> dict[str, float]:
    """Wave 8M — assemble `memo.scores` so the UI can render every
    category score next to the headline confidence number.

    Three groups of fields ride here:
      - Headline + agent-confidence numbers (existing behavior).
      - `factor_*` — the same seven factor scores the screener uses,
        recomputed from the same ratios/profile inputs so memo and
        screener can't disagree on a name's quality / growth / valuation /
        momentum / risk / macro_fit / catalyst.
      - `factor_pm_score` — the screener's composite (0-100) for
        side-by-side comparison with the LLM-driven confidence.
    """
    from ..finance import factor_scores as fs
    rev_growth = ratios.get("revenue_growth")
    op_margin = ratios.get("operating_margin")
    gross_margin = ratios.get("gross_margin")
    roic = ratios.get("ROIC")
    ev_ebitda = ratios.get("EV_EBITDA")
    p_fcf = ratios.get("PFCF")
    fcf_y = ratios.get("FCF_yield")
    debt_to_ebitda = ratios.get("debt_to_ebitda")
    beta = profile.get("beta")

    quality = fs.quality_score(roic, op_margin, gross_margin)
    growth = fs.growth_score(rev_growth)
    valuation = fs.valuation_score(ev_ebitda, p_fcf, fcf_y)
    surprises = [q.get("surprise_pct", 0) for q in (earnings or {}).get("quarters", [])]
    # Pull the LLM-extracted latest guidance changes so beat-AND-raise
    # registers as a momentum bonus. Falls back to surprise-only when
    # the earnings analyst didn't emit structured output.
    latest_guidance: list[dict[str, Any]] = []
    if earnings_finding is not None and isinstance(earnings_finding.data, dict):
        structured = earnings_finding.data.get("structured")
        if isinstance(structured, dict):
            raw_changes = structured.get("guidance_changes") or []
            if isinstance(raw_changes, list):
                latest_guidance = raw_changes
    earnings_momentum = fs.earnings_momentum_score(
        surprises, latest_guidance_changes=latest_guidance,
    )
    streak = fs.beat_streak(surprises)
    guidance_net = fs.guidance_net_direction(latest_guidance)
    risk = fs.risk_score(beta, debt_to_ebitda, drawdown=-0.20)

    macro_fit = 60.0  # untilled-theme baseline; theme bias only applies on the screener path
    catalyst = 65.0 if "AI" in (profile.get("description") or "") else 50.0

    pm_score = round(
        quality * 0.25 + growth * 0.20 + valuation * 0.15
        + earnings_momentum * 0.10 + macro_fit * 0.15
        + risk * 0.10 + catalyst * 0.05,
        1,
    )

    return {
        # Headline / agent confidences (existing).
        "confidence": blended_confidence,
        "raw_confidence": raw_confidence,
        "evidence_quality": round(ev_q * 100, 1),
        "sector_confidence": sector_finding.confidence * 100,
        "valuation_confidence": valuation_finding.confidence * 100,
        "risk_confidence": risk_finding.confidence * 100,
        # Wave 8M — quant factor scores (same math as the screener).
        "factor_quality": quality,
        "factor_growth": growth,
        "factor_valuation": valuation,
        "factor_earnings_momentum": earnings_momentum,
        "factor_macro_fit": macro_fit,
        "factor_risk": risk,
        "factor_catalyst": catalyst,
        "factor_pm_score": pm_score,
        # Beat-and-raise transparency. `beat_streak` = consecutive
        # recent EPS beats; `guidance_net` = (raised - lowered) from
        # the latest call's structured guidance changes. UI uses these
        # to render the "🔥 beat & raise" badge on the earnings card
        # when the combination triggers the momentum bonus.
        "beat_streak": float(streak),
        "guidance_net_direction": float(guidance_net),
    }


def _apply_risk_recommendations(
    memo: StockMemoOut, risk_finding: AgentFinding,
) -> list[dict[str, Any]]:
    """Wave 8H — deterministic enforcement of risk-agent recommendations.

    The PM synthesis prompt is one channel for the risk lens to influence
    the memo (LLM reads the recommendations); this is the second, harder
    channel. For each rec we mutate the memo in place:

    - `target=confidence` direction=lower → clamp confidence_score down
      by 5/10/15 (small/medium/large).
    - `target=rating` direction=lower → shift rating one notch down the
      Bullish→Mixed Positive→Neutral→Mixed Negative→Bearish ladder.
    - `target=thesis_breakers` direction=flag → ensure a matching
      RiskItem is in `memo.thesis_breakers` (severity=high).
    - `target=bear_case` direction=flag → append `detail` to
      `memo.bear_case.key_points` if not already present (this is how
      risk findings actually augment the sector-built bear case).

    Returns the list of recs that were applied so the caller can add an
    audit trail to the memo.
    """
    if not isinstance(risk_finding.data, dict):
        return []
    raw = risk_finding.data.get("recommendations") or []
    if not isinstance(raw, list):
        return []
    applied: list[dict[str, Any]] = []

    rating_ladder = [
        "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish",
    ]
    confidence_step = {"small": 5.0, "medium": 10.0, "large": 15.0}

    for rec in raw:
        if not isinstance(rec, dict):
            continue
        target = rec.get("target")
        direction = rec.get("direction")
        magnitude = rec.get("magnitude") or "medium"
        detail = (rec.get("detail") or "").strip()
        rationale = (rec.get("rationale") or "").strip()
        # Discipline — recs without a rationale don't apply.
        if not detail or not rationale:
            continue

        if target == "confidence" and direction == "lower":
            delta = confidence_step.get(magnitude, 10.0)
            old = float(memo.confidence_score or 0)
            new = max(20.0, old - delta)
            if new != old:
                memo.confidence_score = new
                applied.append({**rec, "applied_change": {
                    "field": "confidence_score", "from": old, "to": new,
                }})

        elif target == "rating" and direction == "lower":
            current = (memo.rating_label or "").strip()
            if current in rating_ladder:
                idx = rating_ladder.index(current)
                step = {"small": 1, "medium": 1, "large": 2}.get(magnitude, 1)
                new_idx = min(len(rating_ladder) - 1, idx + step)
                if new_idx != idx:
                    new_rating = rating_ladder[new_idx]
                    memo.rating_label = new_rating  # type: ignore[assignment]
                    applied.append({**rec, "applied_change": {
                        "field": "rating_label",
                        "from": current, "to": new_rating,
                    }})

        elif target == "thesis_breakers" and direction == "flag":
            already = any(
                detail.lower()[:60] in (item.title or "").lower()
                for item in memo.thesis_breakers
            )
            if not already:
                memo.thesis_breakers = list(memo.thesis_breakers) + [
                    RiskItem(
                        title=detail[:80],
                        detail=rationale,
                        severity="high",
                        type="thesis_breaker",
                    ),
                ]
                applied.append({**rec, "applied_change": {
                    "field": "thesis_breakers", "appended": detail[:80],
                }})

        elif target == "bear_case" and direction == "flag":
            existing = [p.lower() for p in memo.bear_case.key_points]
            if not any(detail.lower()[:60] in p for p in existing):
                memo.bear_case.key_points = list(memo.bear_case.key_points) + [
                    f"Risk lens: {detail}",
                ]
                applied.append({**rec, "applied_change": {
                    "field": "bear_case.key_points",
                    "appended": f"Risk lens: {detail}",
                }})

    return applied


def _bull_bear_from_sector(sector_finding: AgentFinding) -> dict[str, Any] | None:
    """Wave 3A: pluck the structured bull_bear_analysis out of the sector
    finding's data payload, if present. Returns the dict (not the Pydantic
    model) so the caller can pull out the raw `bull_case`/`bear_case`
    objects directly into the memo."""
    if not isinstance(sector_finding.data, dict):
        return None
    bb = sector_finding.data.get("bull_bear_analysis")
    return bb if isinstance(bb, dict) else None


def _findings_signal_lines(
    finding: AgentFinding | None, *, polarity: str,
    max_items: int = 3, prefix: str = "",
) -> list[str]:
    """Pull the most signal-bearing key_points / signals from an agent
    finding, scoped by polarity.

    Wave 9b — replaces the demo-only `profile.drivers` / `profile.risks`
    fallback in `_bull_case` / `_bear_case` / `_catalysts`. Live FMP
    profiles don't carry those fields; specialist findings do.

    `polarity ∈ {"bull", "bear", "neutral"}` filters lines by simple
    keyword detection (positive: tailwind / leverage / above median /
    accelerat; negative: pressure / headwind / risk / concentration /
    declin). `neutral` returns the first lines unfiltered.
    """
    if finding is None:
        return []
    bull_kw = (
        "tailwind", "leverage", "above median", "above-median",
        "accelerat", "expansion", "outperform", "premium quality",
        "moat", "advantage", "compounder", "growth", "top quartile",
    )
    bear_kw = (
        "pressur", "headwind", "risk", "concentrat", "declin",
        "deteriorat", "compress", "below median", "below-median",
        "elevated", "fragile", "slip", "underperform", "regulator",
        "antitrust", "litigation", "competit",
    )
    out: list[str] = []
    candidates = list(finding.key_points or [])
    # Also consider sentence fragments from the summary as a fallback
    # source — lots of value lives there for short-key_points findings.
    if finding.summary:
        for s in finding.summary.split(". "):
            s = s.strip()
            if 30 <= len(s) <= 220:
                candidates.append(s)
    for line in candidates:
        if not isinstance(line, str):
            continue
        low = line.lower()
        if polarity == "bull" and not any(k in low for k in bull_kw):
            continue
        if polarity == "bear" and not any(k in low for k in bear_kw):
            continue
        text = (prefix + line) if prefix and not line.lower().startswith(prefix.lower().rstrip(": ")) else line
        out.append(text[:240])
        if len(out) >= max_items:
            break
    return out


def _bull_case(profile: dict, valuation: AgentFinding, dcf: DCFResult | None,
               sector_finding: AgentFinding | None = None,
               findings: dict[str, AgentFinding] | None = None) -> BullBearCase:
    """Build the memo's bull case.

    Preference order:
      1. Sector analyst's structured `bull_bear_analysis` (LLM-generated).
      2. Bull-polarity signals lifted from sector / valuation / earnings
         findings + DCF upside.
      3. Generic but honest "DCF says X" line.
    """
    sector_bb = _bull_bear_from_sector(sector_finding) if sector_finding else None
    # Wave 10k — pull DCF scenario drivers (LLM-named) so the prose
    # tile cites the same drivers the assumption changes baked in.
    dcf_drivers: list[str] = []
    if dcf and dcf.bull and dcf.bull.drivers:
        dcf_drivers = [
            f"DCF driver — {d.name}: {d.rationale}".rstrip(": ")
            for d in dcf.bull.drivers if d.name or d.rationale
        ][:3]
    if sector_bb and isinstance(sector_bb.get("bull_case"), dict):
        bull = sector_bb["bull_case"]
        points: list[str] = list(bull.get("key_points") or [])
        points.extend(dcf_drivers)
        if dcf:
            points.append(
                f"DCF bull case implies {fmt_price(dcf.bull.implied_share_price)} "
                f"({fmt_upside(dcf.bull.upside_pct, decimals=0)})."
            )
        return BullBearCase(
            headline=str(bull.get("headline") or "Bull case from sector synthesis."),
            key_points=points,
        )

    points = []
    findings = findings or {}
    points.extend(_findings_signal_lines(findings.get("sector"), polarity="bull", max_items=3))
    points.extend(_findings_signal_lines(findings.get("valuation"), polarity="bull", max_items=2))
    points.extend(_findings_signal_lines(findings.get("earnings"), polarity="bull", max_items=2))
    # Wave 10k — DCF-named drivers from the scenario builder.
    points.extend(dcf_drivers)
    if dcf:
        points.append(
            f"DCF bull case implies {fmt_price(dcf.bull.implied_share_price)} "
            f"({fmt_upside(dcf.bull.upside_pct, decimals=0)})."
        )
    if not points:
        # Template fallback: cite the profile's own thesis drivers as
        # "Tailwind:" points rather than emitting a generic filler line.
        drivers = [d for d in (profile.get("drivers") or []) if d]
        points.extend(f"Tailwind: {d}" for d in drivers[:3])
    if not points:
        points.append("Quality + growth profile supports a premium versus peers.")

    # Pick a headline from the strongest available signal.
    sector_head = (sector_finding.headline if sector_finding else "") or ""
    if "above" in sector_head.lower() or "leader" in sector_head.lower():
        headline = sector_head[:240]
    elif valuation and valuation.headline and "discount" in valuation.headline.lower():
        headline = f"Bull case: {valuation.headline[:200]}"
    else:
        headline = "Bull case: cohort + valuation read both supportive."
    return BullBearCase(headline=headline, key_points=points[:6])


def _bear_case(profile: dict, dcf: DCFResult | None,
               sector_finding: AgentFinding | None = None,
               findings: dict[str, AgentFinding] | None = None) -> BullBearCase:
    """Build the memo's bear case. Mirror of `_bull_case` — prefers the
    sector LLM's bear, otherwise lifts bear-polarity signals from
    sector / risk / filing findings + DCF downside."""
    sector_bb = _bull_bear_from_sector(sector_finding) if sector_finding else None
    # Wave 10k — DCF bear-case drivers from the scenario builder.
    dcf_drivers: list[str] = []
    if dcf and dcf.bear and dcf.bear.drivers:
        dcf_drivers = [
            f"DCF driver — {d.name}: {d.rationale}".rstrip(": ")
            for d in dcf.bear.drivers if d.name or d.rationale
        ][:3]
    if sector_bb and isinstance(sector_bb.get("bear_case"), dict):
        bear = sector_bb["bear_case"]
        points: list[str] = list(bear.get("key_points") or [])
        points.extend(dcf_drivers)
        if dcf:
            points.append(
                f"DCF bear case implies {fmt_price(dcf.bear.implied_share_price)} "
                f"({fmt_upside(dcf.bear.upside_pct, decimals=0)})."
            )
        return BullBearCase(
            headline=str(bear.get("headline") or "Bear case from sector synthesis."),
            key_points=points,
        )

    points = []
    findings = findings or {}
    points.extend(_findings_signal_lines(findings.get("risk"), polarity="bear", max_items=3))
    points.extend(_findings_signal_lines(findings.get("filing"), polarity="bear", max_items=2, prefix="Filing: "))
    points.extend(_findings_signal_lines(findings.get("sector"), polarity="bear", max_items=2))
    # Wave 10k — DCF-named drivers from the scenario builder.
    points.extend(dcf_drivers)
    if dcf:
        points.append(
            f"DCF bear case implies {fmt_price(dcf.bear.implied_share_price)} "
            f"({fmt_upside(dcf.bear.upside_pct, decimals=0)})."
        )
    if not points:
        points.append("Cohort positioning leaves modest downside if execution slips.")

    risk_head = (findings.get("risk") and findings["risk"].headline) or ""
    if risk_head and not risk_head.lower().startswith("risk profile for"):
        headline = f"Bear case: {risk_head[:200]}"
    else:
        headline = "Bear case: execution / valuation / regulatory risks if thesis cracks."
    return BullBearCase(headline=headline, key_points=points[:6])


def _catalysts(
    profile: dict, transcript: dict | None,
    findings: dict[str, AgentFinding] | None = None,
    earnings: dict | None = None,
) -> list[CatalystItem]:
    """Surface near-term + medium-term catalysts.

    Wave 9b — derives catalysts from findings (earnings tone, sector
    drivers, news themes) instead of demo-only `profile.drivers`. Adds
    the next earnings date as a concrete near-term watch item when
    we have it (FMP earnings endpoint or AV).
    """
    items: list[CatalystItem] = []
    findings = findings or {}

    # Sector / news positive catalysts.
    sector_signals = _findings_signal_lines(findings.get("sector"), polarity="bull", max_items=2)
    for s in sector_signals:
        items.append(CatalystItem(
            title=s[:80], detail=s, horizon="medium_term", impact="medium",
        ))
    news_signals = _findings_signal_lines(findings.get("news_impact"), polarity="bull", max_items=1)
    for s in news_signals:
        items.append(CatalystItem(
            title=s[:80], detail=s, horizon="near_term", impact="medium",
        ))

    # Concrete next-earnings date when known.
    next_date = None
    if earnings and isinstance(earnings, dict):
        # FMP /stable/earnings returns forward rows with epsActual=null;
        # take the first one with a future-looking date.
        quarters = earnings.get("quarters") or []
        for q in quarters:
            if q.get("eps_actual") is None and q.get("report_date"):
                next_date = q["report_date"]
                break
    if next_date:
        items.append(CatalystItem(
            title=f"Next earnings: {next_date}",
            detail=f"Quarterly print expected on {next_date}; tone + guidance the swing factor.",
            horizon="near_term", impact="medium",
        ))
    elif transcript and transcript.get("period"):
        items.append(CatalystItem(
            title="Next earnings update",
            detail=f"Watch for follow-on commentary on themes from {transcript['period']}.",
            horizon="near_term", impact="medium",
        ))

    # Profile-driven catalysts (demo only — kept for fixture tests).
    if not items:
        for d in (profile.get("drivers") or [])[:3]:
            items.append(CatalystItem(
                title=d[:80], detail=d, horizon="medium_term", impact="medium",
            ))

    return items[:6]


class PMView(NamedTuple):
    """How the PM synthesis sees this run's findings (contract C7).

    `digests` — text placed after `pm_ctx` and before "Findings:", outside
    the `max_agent_context_chars` cut; `findings` — what the capped JSON
    carries, and what the no-LLM keyword heuristic counts; `visible` —
    everything the PM reads either way, which is what `agent_influence`
    scores (a withheld read has no stored pull on a rating it never saw)."""

    digests: list[str]
    findings: dict[str, AgentFinding]
    visible: dict[str, AgentFinding]


def _pm_view(findings: dict[str, AgentFinding]) -> PMView:
    """Split `findings` by the roster's `pm_digest` declarations.

    A spec with a digest is NEVER put in the JSON: on a live memo its entry
    sits past the 60k cut (it is the roster tail), and on a small memo it
    would be read twice. Its non-empty digest is read instead; an empty one
    withholds the finding from the PM entirely. Without such a spec in
    `findings` the JSON set is `findings` itself, same keys in the same
    order, so the PM prompt is byte-identical to what it was before."""
    # Read off `roster.AGENTS` per call rather than `AGENTS_BY_KEY`, so a
    # spec a test appends to the roster is honoured here too.
    specs = {spec.key: spec for spec in roster.AGENTS}
    digests: list[str] = []
    rest: dict[str, AgentFinding] = {}
    visible: dict[str, AgentFinding] = {}
    for key, finding in findings.items():
        spec = specs.get(key)
        if spec is None or spec.pm_digest is None:
            rest[key] = finding
            visible[key] = finding
            continue
        try:
            text = spec.pm_digest(finding)
        except Exception as exc:
            # A digest that cannot be built withholds that one read; it must
            # not cost the memo its PM synthesis (`safe_call` would ship the
            # "synthesis unavailable" fallback for the whole memo).
            log.warning("PM digest for %s failed; read withheld from the PM: %s",
                        key, type(exc).__name__)
            text = ""
        if text:
            digests.append(text)
            visible[key] = finding
    return PMView(digests, rest, visible)


# L6 (TradingAgents lessons, 2026-09-25): the five labels, by their
# case-folded form. `RatingLabel` is a strict Literal, so an off-enum label
# from the PM ("bullish", "Bullish ", "Moderately Bullish", "Buy") used to
# raise a ValidationError when the memo was built and lose the whole run.
_RATING_LABELS: tuple[str, ...] = get_args(RatingLabel)
_RATING_BY_FOLDED: dict[str, str] = {label.casefold(): label for label in _RATING_LABELS}

# `_pm_synthesis` marks where its rating came from under this key; the
# compose stage pops it into `scores` (P6 recording) before building the memo.
RATING_SOURCE_KEY = "_rating_source"


def normalize_rating_label(value: Any) -> str | None:
    """The canonical label for `value`, or None when it is not one.

    Strip, collapse inner whitespace, case-fold, then an EXACT match against
    the five labels. Never a substring or prefix match: "Moderately Bullish"
    is not "Bullish", and "Sell-side" is not "Sell" (the misread TradingAgents
    #1383 shipped). None is an explicit state the caller must handle, never
    a silent default."""
    if not isinstance(value, str):
        return None
    return _RATING_BY_FOLDED.get(" ".join(value.split()).casefold())


def _pm_synthesis(
    profile: dict, findings: dict[str, AgentFinding], dcf: DCFResult | None,
    *, scorecard: Any | None = None, valuation_evidence: ValuationVerdict | None = None,
    news: news_context.NewsContext | None = None,
) -> dict:
    # PM uses its dedicated model (OPENAI_PM_MODEL — gpt-5.5-pro by default).
    # Wave 10 — read PM brain + company / sector memory + research_notes.
    # Phase 6 — plus the scorecard block (<= 600 chars; "" when no row), so
    # the synthesis prompt can ask for `scorecard_reconciliation`.
    # W7 — `learning_consumer` asks for the learned-priors block. It renders
    # only in a live memo run; off / shadow leave pm_ctx byte-identical.
    from .pm_context import build_pm_context
    pm_ctx = build_pm_context(
        ticker=profile.get("ticker"),
        sector=profile.get("sector"),
        profile=profile,
        scorecard_block=scorecard_context.prompt_block(scorecard),
        learning_consumer="pm_memo",
    )
    view = _pm_view(findings)
    # C7 assembly order: static template + pm_ctx, then the routed digests,
    # then the capped JSON. The digests sit outside the cut on purpose (see
    # `_pm_view`), and "" when there are none keeps the prompt byte-identical.
    digest_block = ("\n\n" + "\n\n".join(view.digests)) if view.digests else ""
    # FIX-018, C7: the run's news block goes after the digests and before the
    # evidence block. The PM used to get news only nested in the sector
    # entry of the Findings JSON with no instruction to weigh it. "" with no
    # news keeps the prompt byte-identical.
    news_text = news_context.render_block(news, "pm")
    news_block = ("\n\n" + news_text) if news_text else ""
    # W2b 7(b), C7: the deterministic valuation-evidence read goes after the
    # digests and before "Findings:" — volatile, outside the cached prefix and
    # outside the JSON cut. "" without evidence keeps the prompt byte-identical.
    evidence = memo_quality.valuation_evidence_block(valuation_evidence)
    evidence_block = ("\n\n" + evidence) if evidence else ""
    # W2b 7(a): the source refs a declared forecast assumption may name as
    # its basis. Volatile, after the evidence block; "" outside a memo run
    # (no active ledger) keeps the prompt byte-identical.
    refs = _source_refs_block()
    refs_block = ("\n\n" + refs) if refs else ""
    json_findings = {k: v.model_dump() for k, v in view.findings.items()}
    if news_text:
        # The sector finding stores the same items as `pending_news_alerts`
        # (both come from `inputs.news`); with the block present that copy
        # is a duplicate read of up to ~1,100 tokens, so it is dropped from
        # this JSON copy only. The stored memo keeps it.
        sector_json = json_findings.get("sector")
        if isinstance(sector_json, dict) and isinstance(sector_json.get("data"), dict):
            sector_json["data"] = {k: v for k, v in sector_json["data"].items()
                                   if k != "pending_news_alerts"}
    # The synthesis template is byte-stable across memos; declare it as the
    # cached prefix so each PM call reads it instead of re-paying for it. The
    # volatile pm_ctx / digests / findings follow the "\n\n" join and stay
    # uncached.
    with llm.llm_call_context(static_prefix_chars=len(prompts.PM_SYNTHESIS_PROMPT) + 2):
        llm_out = llm.chat_json(
            prompts.PM_SYNTHESIS_PROMPT
            + (("\n\n" + pm_ctx) if pm_ctx else "")
            + digest_block
            + news_block
            + evidence_block
            + refs_block
            + "\n\nFindings:\n"
            + json.dumps(json_findings, default=str)[: settings.max_agent_context_chars],
            system=prompts.PM_SYSTEM, route="strong",
            model=settings.openai_pm_model,
            action="pm.synthesis", ticker=profile.get("ticker"),
        )
    if isinstance(llm_out, dict) and llm_out.get("priors_considered") is not None:
        # W7: which shown priors the PM applied or contradicted, kept on the
        # run's inject render row (ids it was not shown are dropped). The
        # memo picks `synth` fields explicitly, so the key never reaches it.
        from ..learning import context as learning_context
        safe_call(
            learning_context.record_considered, llm.current_call_context().get("run_id"),
            llm_out.get("priors_considered"), fallback=0, name="Learning considered", log_to=None,
        )
    invalid_label = False
    if isinstance(llm_out, dict) and "rating_label" in llm_out:
        label = normalize_rating_label(llm_out.get("rating_label"))
        if label is not None:
            return {**llm_out, "rating_label": label, RATING_SOURCE_KEY: "llm"}
        # L6: an unreadable rating is an explicit PM failure, not a crash in
        # compose and not a guessed label. The memo completes on the
        # deterministic view and says so.
        invalid_label = True

    if invalid_label:
        note_soft(
            "PM Synthesis",
            "LLM returned an invalid rating_label; deterministic PM view shipped",
        )
    elif settings.has_llm:
        # (b) The PM view is the memo's headline. Templated prose standing in
        # for it while an LLM was configured is a degradation the reader must
        # see; in deterministic mode (no keys) this path IS the design, so it
        # is not flagged there. `_pm_synthesis` has no log handle — the
        # contextvar set by `run_stock_memo` carries it.
        note_soft(
            "PM Synthesis",
            "LLM returned no usable synthesis; deterministic PM view shipped",
        )

    # Deterministic synthesis
    # None (no DCF, or a DCF that could not price the shares) contributes
    # nothing to the score — it is an absent signal, not a neutral one.
    upside = dcf.base.upside_pct if dcf else None
    # The keyword heuristic counts the JSON set only. A digested read is
    # the PM model's input, not a vote here: the mandate text of some groups
    # says "premium", and routing must not move the no-LLM fallback rating.
    pos_signals = sum(1 for f in view.findings.values() if any(k in (f.headline + f.summary).lower()
                                                               for k in ("constructive", "premium", "outperform", "tailwind")))
    neg_signals = sum(1 for f in view.findings.values() if any(k in (f.headline + f.summary).lower()
                                                               for k in ("pressured", "underperform", "elevated", "compress")))
    dcf_signal = 0
    if upside is not None:
        dcf_signal = 1 if upside > 0.10 else (-1 if upside < -0.10 else 0)
    score = pos_signals - neg_signals + dcf_signal
    # Wave 8P — five-label scheme tied to the deterministic Stock-Score
    # mapping. The actual rating gets *overridden* later by
    # `rating_from_stock_score` once the factor blend is computed; this
    # local provides a sensible fallback for the LLM-disabled path.
    if score >= 2:
        rating = "Very Bullish"
    elif score == 1:
        rating = "Bullish"
    elif score == 0:
        rating = "Neutral"
    elif score == -1:
        rating = "Bearish"
    else:
        rating = "Very Bearish"
    confidence = max(40, min(85, 55 + 5 * abs(score)))

    # Build a thesis that distills the actual claim — not a metric
    # recap. Centralized in `_build_thesis_from_findings` so the
    # deterministic fallback and the post-LLM anti-pattern rewrite share
    # one source of truth.
    # The thesis states the valuation EVIDENCE word when there is evidence
    # (W2b 7(b)): the keyword rating is a separate call about return, and the
    # reconciliation stage squares the two. With no evidence, the rating's
    # word (the pre-W2b behaviour).
    word = (
        memo_quality.verdict_word(valuation_evidence.verdict)
        if valuation_evidence is not None and memo_quality.evidence_available(valuation_evidence)
        else memo_quality.rating_word(rating)
    )
    thesis = _build_thesis_from_findings(
        profile, findings, dcf, profile.get("ticker") or "", verdict_word=word,
    )
    pm_view = (
        f"Research view: {rating}. {thesis} "
        f"Sector framing supports the cohort thesis; valuation-relative read is the main swing factor. "
        f"The risk committee flagged the dominant downside scenarios; portfolio fit depends on macro view."
    )
    return {
        "final_pm_view": pm_view,
        "one_sentence_thesis": thesis,
        "rating_label": rating,
        "confidence_score": confidence,
        RATING_SOURCE_KEY: "keyword",
    }


MAX_SOURCE_REFS_IN_PROMPT = 40


def _source_refs_block() -> str:
    """The ledger's source refs, for the PM's `forecast_assumptions`."""
    ledger = active_ledger()
    if ledger is None:
        return ""
    refs = ledger.source_refs()[:MAX_SOURCE_REFS_IN_PROMPT]
    if not refs:
        return ""
    return "## Source refs (for forecast_assumptions basis_ref)\n" + ", ".join(refs)


def _portfolio_fit(profile: dict, rating: str) -> str:
    sector = profile.get("sector", "")
    return (
        f"In a balanced model portfolio, {profile.get('ticker', '')} fits the '{sector}' sleeve. "
        f"With a '{rating}' research view, sizing is governed by the user's max position size and risk level."
    )


# ---------------------------------------------------------------------------
# Wave 8A: per-step checkpointing
# ---------------------------------------------------------------------------
# Each step the memo run can resume from gets a thin checkpointed wrapper.
# When `run_id` is in scope (always, since `run_stock_memo` sets it), the
# decorator caches the step's result under `(run_id, step_name)` so a
# retried run with the same `run_id` skips the underlying work.
#
# The eight analysts' wrappers live on the roster (`roster.checkpointed_runner`)
# and are built from `AgentSpec.checkpoint`; only the four steps whose
# return types differ from `AgentFinding` stay hand-written here. Their
# step names are the `roster.GATHER_STEPS` / `roster.CRITIC_STEP` literals —
# frozen, because resume and the status endpoint key on them.
#
# Why thin wrappers vs. decorating each function at definition: keeping the
# underlying functions un-decorated lets other callers (tests, ad-hoc
# scripts, future workers) use them without checkpoint side effects. The
# checkpoint behavior is deliberately scoped to the graph entry path.


@checkpointed("graph.fundamentals", return_type=None)
def _checkpointed_fundamentals(ticker: str, *, force_refresh: bool) -> dict[str, Any]:
    return get_full_financials(ticker, force_refresh=force_refresh)


@checkpointed("graph.dcf", return_type=DCFResult)
def _checkpointed_dcf(ticker: str, *, force_refresh: bool) -> DCFResult | None:
    return build_dcf(ticker, force_refresh=force_refresh)


@checkpointed("graph.comps", return_type=CompsResult)
def _checkpointed_comps(ticker: str, *, force_refresh: bool) -> CompsResult | None:
    return build_comps(ticker, force_refresh=force_refresh)


@checkpointed("graph.critic", return_type=CriticReview)
def _checkpointed_critic(memo_dict: dict[str, Any]) -> CriticReview | None:
    # None is a legitimate outcome (ENABLE_AGENT_CRITIC=false); `safe_critic`
    # passes it through rather than treating it as a failure.
    return run_critic(memo_dict)


# ---------------------------------------------------------------------------
# Public graph entry point
# ---------------------------------------------------------------------------

def _run_reflection_step(memo: StockMemoOut):
    """Local indirection so safe_call can wrap the reflection step. Imports
    lazily to avoid an import-time cycle (reflection_agent → memory → cache)."""
    from .reflection_agent import run as _reflect
    return _reflect(memo)


def run_stock_memo(
    ticker: str, *, scenario: str = "soft_landing", force_refresh: bool = False,
    run_id: str | None = None,
    as_of_date: Any | None = None,
) -> StockMemoOut:
    """Generate a stock memo. When `force_refresh=True`, every cached snapshot
    in the dependency tree is bypassed; otherwise, fundamentals/sector/comps/DCF
    are read from the snapshot cache when fresh.

    `run_id` (Wave 1A) tags every LLM call made during this memo run for
    cost / trace attribution via `LLMCallLog`. Auto-generated when None.

    `as_of_date` (Wave 1C) reproduces the memo as of a historical date.
    All cache reads/writes inside the call are namespaced by date so
    backtests don't collide with live data; long-term memory writes are
    skipped (a backtest shouldn't pollute the agent's notebook). Future
    PRs will thread per-provider date filtering through the data
    service so backtests truly see only past data.
    """
    import uuid
    from datetime import date as _date_cls
    from datetime import datetime as _dt_cls
    if run_id is None:
        run_id = str(uuid.uuid4())
    # Coerce datetime → date if a caller hands us a datetime.
    if isinstance(as_of_date, _dt_cls):
        as_of_date = as_of_date.date()
    if as_of_date is not None and as_of_date > _date_cls.today():
        raise ValueError(f"as_of_date {as_of_date} is in the future")

    from ..services import memory_probe
    from ..services.data_service import as_of_context
    from .llm import llm_call_context
    # RSS breadcrumbs around the most memory-hungry operation in the
    # process. A Render OOM-kill is a SIGKILL, so Python never gets to log
    # anything on the way down — these two lines are what turns the next
    # one from "the instance restarted" into "it restarted during TICKER's
    # memo, having already grown N MB".
    memory_probe.log_rss("memo_start", ticker=ticker, run_id=run_id)
    # The degradation log is created here and *activated* for the whole run
    # (RP-001): service code and helpers with no handle on it — the thesis
    # builder, PM synthesis, the PM DCF adjuster, the valuation service —
    # report soft failures through `safe_runner.note_soft`, which writes to
    # whichever log is active in this context. Activating in the outermost
    # `with` keeps the guarantee identical to the other two contexts: every
    # line of the memo run is covered, and the token is reset in `finally`
    # so the regen worker's next memo in the same thread starts empty.
    degradation = DegradationLog()
    # W2b 7(a): the source ledger — every fact the analysts are given is
    # registered where its payload is built (`register_source`), and the
    # quality stage checks the memo's figures against it. Same activation
    # discipline as the degradation log: one per run, reset in `finally`.
    ledger = SourceLedger()
    try:
        # The run context is an UMBRELLA: it sets run_id and ticker, never
        # an agent (attribution critique #1). A named agent here was
        # credited with every call nested under it that opened no context of
        # its own; each call now names its action, and the stages that are
        # an agent (`_run_analyst_round`, `_compose_memo`, `_review_memo`)
        # open their own agent context.
        with as_of_context(as_of_date), llm_call_context(
            run_id=run_id, ticker=ticker,
        ), degradation.activate(), ledger.activate():
            return _run_stock_memo_inner(
                ticker, scenario=scenario, force_refresh=force_refresh,
                run_id=run_id, as_of_date=as_of_date, degradation=degradation,
            )
    finally:
        memory_probe.log_rss("memo_end", ticker=ticker, run_id=run_id)


def _run_stock_memo_inner(
    ticker: str, *, scenario: str, force_refresh: bool, run_id: str,
    as_of_date: Any | None = None,
    degradation: DegradationLog | None = None,
) -> StockMemoOut:
    """The memo pipeline as a sequence of stage calls (RP-002).

    Indirection so `run_stock_memo` can wrap the entire body in a single
    `llm_call_context` + `as_of_context` + `DegradationLog.activate()`;
    every stage below runs under that one activated log.

    Stage boundaries and the objects that cross them are the dataclasses
    in `memo_context` — read its mutation contract: `inputs.profile` and
    `analysts.findings` are shared and edited in place by stages 2-5, and
    the memo holds the same finding objects. The verdict and quality stages
    are pure: each returns an outcome (`VerdictOutcome`, `QualityOutcome`)
    applied here, after review, because both read the post-blend,
    post-reconciliation rating. The confidence-bearing texts are rendered
    once, after both (`_render_final_texts`), and memory reflection runs on
    that final memo (W2b §6.1).

    Wave 8A: each resumable step (fundamentals, dcf, comps, every
    specialist, critic) runs through a `@checkpointed` wrapper. When
    `run_id` is reused across calls (a retry after a transient failure),
    each completed step's result is loaded from `MemoRunCheckpoint`
    instead of re-fired. First-time runs see no behavior change.
    """
    # Everything after fundamentals goes through the safe-runner: a failure
    # in any single specialist becomes a typed fallback rather than killing
    # the memo. Failures accumulate on `degradation` and surface on the
    # memo's `degraded_agents` / `degradation_events`. `run_stock_memo` — the
    # only production caller — passes the log it activated; a direct caller
    # without one gets a private log, and `note_soft` is then a no-op.
    if degradation is None:
        degradation = DegradationLog()
    inputs = _gather_inputs(
        ticker, scenario=scenario, force_refresh=force_refresh, run_id=run_id,
        as_of_date=as_of_date, degradation=degradation,
    )
    analysts = _run_analyst_round(inputs)
    dcf_stage = _adjust_dcf(inputs, analysts)
    # The evidence verdict is computed here, BEFORE the PM, and the PM's
    # divergence reason is captured on `memo.quality` (W2b 7(b)).
    memo = _compose_memo(inputs, analysts, dcf_stage)
    initial = PMOpinion(memo.rating_label, float(memo.confidence_score), memo.final_pm_view)
    # Critic, risk recommendations, blend, then the rating reconciliation.
    memo = _review_memo(memo, inputs, analysts)
    verdict = _build_verdict(
        memo, comps=inputs.comps, dcf=dcf_stage.dcf, profile=inputs.profile,
        findings=analysts.findings, ticker=ticker,
    )
    verdict.apply(memo, degradation)
    # 7(c): the earned-confidence caps. A checker bug must neither kill the
    # memo nor ship it uncapped, so the fallback still caps (`_quality_fallback`).
    quality = safe_call(
        _assess_quality, memo, inputs, analysts,
        fallback=_quality_fallback(memo), name="Memo Quality", log_to=degradation,
    )
    quality.apply(memo, degradation)
    # The two confidence-bearing strings are rendered once, from final values.
    _render_final_texts(memo, verdict, initial)
    # Memory learns from the final, checked memo (it used to write the
    # pre-blend rating and confidence).
    _run_reflection(memo, inputs)
    _attach_scorecard_disagreement(memo)
    return _persist(memo, inputs)


def _attach_scorecard_disagreement(memo: StockMemoOut) -> None:
    """Phase 6 — fill `memo.scorecard.disagreement` from the FINAL memo.

    Runs after the verdict is applied because the detector reads the
    post-blend rating and the reconciled valuation verdict; a flag computed
    off the compose-stage draft could name a rating the reader never sees.
    A finding, not an outage: no degradation entry, and on a detector crash
    the summary is kept without a flag (the memo is still whole).
    """
    summary = memo.scorecard
    if summary is None:
        return

    def _with_flag() -> ScorecardSummary:
        return scorecard_context.summarize(summary, memo) or summary

    memo.scorecard = safe_call(_with_flag, fallback=summary, name="Scorecard Summary", log_to=None)


# ---------------------------------------------------------------------------
# Stage 1 — gather
# ---------------------------------------------------------------------------

def _gather_inputs(
    ticker: str, *, scenario: str, force_refresh: bool, run_id: str,
    as_of_date: Any | None, degradation: DegradationLog,
) -> MemoInputs:
    """Fundamentals, transcript, filings, DCF and comps for one ticker.

    Owns the `graph.fundamentals` / `graph.dcf` / `graph.comps` checkpoints.
    Raises `ValueError` on an unknown ticker; everything else degrades.
    """
    # Fundamentals MUST succeed — without a profile we can't even identify
    # the company, so this is an unrecoverable error and we re-raise.
    fin = _checkpointed_fundamentals(ticker, force_refresh=force_refresh)
    profile = fin["profile"]
    if not profile:
        raise ValueError(f"Unknown ticker: {ticker}")
    ratios = fin.get("ratios", {}) or {}
    # W2b 7(a): the fundamentals every analyst reads (profile text, the
    # statements, ratios, earnings history). Derived growth and margins
    # (D1/D2) are computed at registration.
    register_source("financials", f"financials:{ticker}", fin)

    # Failover events are context-local and the regen worker runs memos
    # back to back in one long-lived thread, so whatever the previous run
    # left undrained would otherwise be pinned on this memo. Discard it.
    llm.consume_failover_events()

    transcript = safe_call(latest_transcript, ticker, fallback=None,
                           name="Transcript Service", log_to=degradation)
    filings: list[dict[str, Any]] = safe_call(
        get_filings, ticker, fallback=[], name="Filings Service", log_to=degradation,
    )
    earnings = fin.get("earnings", {})

    dcf = safe_call(_checkpointed_dcf, ticker, force_refresh=force_refresh, fallback=None,
                    name="DCF Engine", log_to=degradation)
    comps = safe_call(_checkpointed_comps, ticker, force_refresh=force_refresh, fallback=None,
                      name="Comps Engine", log_to=degradation)
    if dcf is not None:
        register_source("dcf", "dcf:initial", dcf, text_keys=DCF_TEXT_KEYS,
                        exclude_keys=DCF_LLM_KEYS)
    if comps is not None:
        # `exposure_rationale` is an LLM's pick of cross-sector peers.
        register_source("comps", f"comps:{ticker}", comps, exclude_keys=("exposure_rationale",))

    # Phase 6 — the scorecard read, point-in-time at `as_of_date`. A crash
    # here is a hard "Fundamental Scorecard" degradation (the read failed);
    # a clean None is the soft one, recorded in the compose stage where the
    # memo's degradation fields are assembled. With the kill switch off
    # `load_for_memo` returns None without touching the DB and nothing
    # downstream records anything — the memo is what it was before.
    scorecard = None
    seeds: list[CritiqueQuestion] = []
    if settings.enable_scorecard:
        scorecard = safe_call(
            scorecard_context.load_for_memo, ticker, as_of_date, fallback=None,
            name=scorecard_context.AGENT_NAME, log_to=degradation,
        )
        if scorecard is not None:
            register_source("scorecard", f"scorecard:{ticker}", scorecard)
        # Review seeds are only meaningful where the dialog will run (live
        # memo, deep research on) — the same gate `_run_analyst_round` uses.
        if settings.enable_deep_research and as_of_date is None:
            seeds = safe_call(
                scorecard_context.pending_seed_questions, ticker, fallback=[],
                name="Scorecard Review Seeds", log_to=None,
            )

    # FEAT-003 — the company's industry-group classification, read once
    # (one SELECT, or the on-demand hook for a symbol no loop has seen).
    # Only with routing on: off, nothing downstream reads it.
    industry_group = None
    if settings.enable_industry_analyst_routing:
        from .industry_analysts import AGENT_NAME as _IG_NAME
        from .industry_analysts import lookup_classification
        industry_group = safe_call(lookup_classification, ticker, fallback=None,
                                   name=_IG_NAME, log_to=degradation)

    # FIX-018 — the run's one news read (N1), fetched at memo time when a
    # live run finds nothing on file (N2). ALWAYS a context: a failed read is
    # an empty one, so the sector analyst never falls back to a read of its
    # own that the ledger would not hold. Registered here, once, with exactly
    # the items every reader is shown.
    news = safe_call(
        news_context.load_for_memo, ticker, as_of_date=as_of_date,
        fallback=news_context.NewsContext.empty(ticker), name="News Context", log_to=None,
    )
    news_context.register(news)

    # `profile` is shared with every later stage and mutated in place — see
    # the mutation contract in `memo_context`.
    return MemoInputs(
        ticker=ticker, run_id=run_id, scenario=scenario, force_refresh=force_refresh,
        as_of_date=as_of_date, fin=fin, profile=profile, ratios=ratios,
        earnings=earnings, transcript=transcript, filings=filings,
        dcf=dcf, comps=comps, degradation=degradation,
        scorecard=scorecard, scorecard_seeds=seeds, industry_group=industry_group,
        news=news, ledger=active_ledger(),
    )


# ---------------------------------------------------------------------------
# Stage 2 — analyst round (fan-out, deep research, long-form)
# ---------------------------------------------------------------------------

def _run_analyst_round(inputs: MemoInputs) -> AnalystRound:
    """Round 0 fan-out over the roster, the PM deep-research dialog, the
    deterministic-fallback promotion and the long-form pass.

    Owns the `graph.<key>_finding` checkpoints. The returned `findings`
    dict is in roster order and is the object every later stage mutates.
    """
    degradation = inputs.degradation
    profile = inputs.profile
    run_id = inputs.run_id

    # Wave 10 — PM intake step. Lets the PM deprioritize up to 3
    # specialists for this memo (e.g., skip technicals on a regulated
    # bank, skip filings re-pass when nothing material has changed).
    # Default = run the whole applicable roster. Decision is logged on
    # the memo for audit.
    from .intake import run_intake, stub_finding
    specs = roster.applicable(inputs)  # each spec's `applies_to`, once per run
    # The PM chooses among THIS run's roster: a spec whose predicate said no
    # is not on the run, so offering it would waste one of the three skips
    # and put an absent agent in the memo's intake audit line.
    # FIX-018: intake sees the run's headlines (it was always told "no news"
    # while its prompt skips specialists on "no recent material news").
    intake = run_intake(
        profile, news_alerts=(inputs.news.alerts() if inputs.news is not None else None),
        specialists=[spec.key for spec in specs],
    )

    # Round 0 fan-out, in roster order. Each specialist runs with its own
    # llm_call_context so any LLM calls it makes get tagged with the right
    # agent_name in LLMCallLog (Wave 1A). (Technicals, by design, do NOT
    # influence the rating — positioning context only.)
    from .llm import llm_call_context
    findings: dict[str, AgentFinding] = {}
    for spec in specs:
        if not intake.runs(spec.key):
            findings[spec.key] = AgentFinding(**stub_finding(spec.key, intake.rationale))
            continue
        with llm_call_context(agent_name=spec.display_name, run_id=run_id, role="analyst"):
            findings[spec.key] = safe_finding(
                spec.display_name, roster.checkpointed_runner(spec), inputs,
                log_to=degradation,
            )

    # Wave 9 — PM↔specialist deep-research dialog. Round 0 is the fan-out
    # above; rounds 1+ critique + re-fire targeted specialists with the
    # PM's question prepended to their prompt. Skipped on backtests
    # (`as_of_date` set) so we don't burn LLM budget retroactively.
    round_findings: list[RoundFindings] = []
    if settings.enable_deep_research and inputs.as_of_date is None:
        from .deep_research import run_dialog_loop

        # Same runner as round 0, with the PM's question threaded through.
        def _refire_for(spec: roster.AgentSpec) -> Callable[[str], AgentFinding]:
            return lambda q: spec.run(inputs, q)

        re_fire = {spec.key: _refire_for(spec) for spec in specs}

        # Loop reads `findings` keyed by short agent name — same as the
        # `re_fire` map. Returns the latest-per-agent findings dict + the
        # full round-by-round audit trail for persistence.
        # Phase 6 — scorecard review seeds force a round-1 re-fire (see
        # `run_dialog_loop`); None keeps the loop's original behavior.
        seeds = list(inputs.scorecard_seeds) or None

        def _run_loop() -> tuple[dict[str, AgentFinding], list[RoundFindings]]:
            return run_dialog_loop(
                run_id=run_id,
                initial_findings=findings,
                re_fire=re_fire,
                seed_questions=seeds,
            )

        no_rounds: list[RoundFindings] = []
        loop_out = safe_call(
            _run_loop,
            fallback=(findings, no_rounds),
            name="Deep Research Loop", log_to=degradation,
        )
        if loop_out:
            current, rounds = loop_out
            round_findings = rounds
            # Replace each agent's finding with the latest-round version so
            # downstream synthesis (PM, critic) sees the freshest read.
            for name, finding in current.items():
                findings[name] = finding
            # The seeds were asked only if round 1 actually re-fired (the
            # loop's fallback above returns no rounds); the persist stage
            # closes the queued review rows on that flag.
            if seeds and any(r.round == 1 and not r.early_exit for r in rounds):
                inputs.scorecard_seeds_consumed = True

    # B3 / Theme 2 — promote silent deterministic fallbacks into the
    # degradation log. An agent whose LLM call returned nothing usable
    # ships boilerplate while presenting as a real analyst view; that is
    # a degradation event the UI must surface, same as a crash. Only when
    # an LLM was supposed to run — in deterministic mode (no keys) the
    # fallback IS the expected path, not a degradation — and only for an
    # analyst an LLM was expected of: comps and risk are deterministic at
    # round 0 by design (`uses_llm_round0=False`), so their flag counts
    # only once the PM re-fired them in a deep-research round.
    if settings.has_llm:
        refired = {
            key for r in round_findings if r.round > 0 for key in r.findings
        }
        for spec in specs:
            _f = findings[spec.key]
            if not (spec.uses_llm_round0 or spec.key in refired):
                continue
            if isinstance(_f.data, dict) and _f.data.get("deterministic_fallback"):
                degradation.record_soft(_f.agent, str(_f.data["deterministic_fallback"]))

    # A specialist that ran on the backup vendor after a failover produced
    # a real view, but not the one the routing config asked for — surface
    # it on the same banner. Drained here after the round and again just
    # before persistence, because PM synthesis, the critic, reflection and
    # the long-form/DCF enrichment all make LLM calls after this point.
    _absorb_failover_events(degradation)

    # Wave 3C: drill-down long-form reports. The deterministic build is
    # cheap and always populates the field; LLM enrichment runs only when
    # ENABLE_LONG_FORM_REPORTS=true. safe_call wraps so a failure never
    # blocks the memo. Mutates each finding's `long_form_report` in place.
    from .long_form import attach_long_form
    _t = profile.get("ticker", inputs.ticker)
    for spec in specs:
        safe_call(attach_long_form, findings[spec.key], ticker=_t,
                  agent_name=spec.display_name, profile=profile, fallback=None,
                  name=spec.long_form_name, log_to=degradation)

    # From here on no entry of `findings` is replaced, only mutated in place.
    return AnalystRound(findings=findings, intake=intake, round_findings=round_findings)


# ---------------------------------------------------------------------------
# Stage 3 — PM DCF adjustment
# ---------------------------------------------------------------------------

def _adjust_dcf(inputs: MemoInputs, analysts: AnalystRound) -> DCFStage:
    """Wave 10 — PM-driven DCF assumption adjustment.

    The PM has the team's full read at this point (round 0 + Wave 9 dialog
    rounds). Now is when the model should reflect the team's view, not
    just consensus defaults. Skipped on backtests so retroactive runs use
    period-appropriate DCF, and without an LLM (nothing to adjust with).
    Mutates `findings["valuation"]` in place when the DCF changed (B2).
    """
    dcf = inputs.dcf
    initial_dcf = dcf
    pm_dcf_adjustments: list[dict[str, Any]] = []
    pm_dcf_headline = ""
    if dcf is not None and inputs.as_of_date is None and settings.has_llm:
        from .dcf_pm_adjuster import adjust_dcf_for_pm_view
        no_adjustment: tuple[DCFResult | None, list[dict[str, Any]], str] = (None, [], "")
        adj_out = safe_call(
            adjust_dcf_for_pm_view,
            ticker=inputs.profile.get("ticker", inputs.ticker), initial_dcf=dcf,
            findings=analysts.findings, run_id=inputs.run_id,
            fallback=no_adjustment,
            name="PM DCF Adjuster", log_to=inputs.degradation,
        )
        if adj_out:
            adjusted_dcf, pm_dcf_adjustments, pm_dcf_headline = adj_out
            if adjusted_dcf is not None and pm_dcf_adjustments:
                # Replace the working DCF — downstream synthesis, bull/bear,
                # factor scoring all see the PM-adjusted version.
                dcf = adjusted_dcf
                # W2b 7(a): the final model is a source; the adjuster's
                # from/to values are too, its rationale (LLM prose) is not.
                register_source("dcf", "dcf:pm_adjusted", dcf, text_keys=DCF_TEXT_KEYS,
                                exclude_keys=DCF_LLM_KEYS)
                register_source("dcf_adjustment", "dcf:adjustments", pm_dcf_adjustments,
                                text_keys=(), exclude_keys=("rationale",))
                # B2 — the valuation agent already ran on the pre-adjustment
                # DCF and baked those numbers into its prose. Rewrite the
                # stale references so ONE DCF appears everywhere in the memo.
                _refresh_dcf_references(analysts.findings["valuation"], initial_dcf, dcf)
    return DCFStage(
        dcf=dcf, initial_dcf=initial_dcf,
        pm_adjustments=pm_dcf_adjustments, pm_headline=pm_dcf_headline,
    )


# ---------------------------------------------------------------------------
# Stage 4 — compose
# ---------------------------------------------------------------------------

def _summarize_dcf(d: DCFResult | None) -> dict[str, Any]:
    if d is None:
        return {}
    return dict(
        current_price=d.current_price,
        base_implied_price=d.base.implied_share_price,
        bull_implied_price=d.bull.implied_share_price,
        bear_implied_price=d.bear.implied_share_price,
        base_upside=d.base.upside_pct,
        bull_upside=d.bull.upside_pct,
        bear_upside=d.bear.upside_pct,
        wacc=d.base.assumptions.wacc,
        terminal_growth=d.base.assumptions.terminal_growth,
        # Any scenario whose Gordon denominator hit the floor taints
        # the three prices the memo prints side by side, so the UI
        # badge keys off "any", not just the base case.
        tv_clamped=any(s.tv_clamped for s in (d.base, d.bull, d.bear)),
        summary=d.summary,
    )


def _evidence_verdict(inputs: MemoInputs, dcf_stage: DCFStage) -> ValuationVerdict:
    """Collect the valuation evidence and call `memo_quality`'s rule on it.

    * valuation family: the fs-v1 scorecard row's valuation category
      (universe percentile; its own coverage, else the row's);
    * comps: the EV/EBITDA premium to the peer median, with both
      multiples and the sector so a negative multiple or a bank/REIT
      (where EV is not meaningful) records the premium without a vote;
    * DCF: the INITIAL (consensus-anchored) model votes, the PM-adjusted
      one is recorded — the PM who rates the name also moved that model;
    * `factor_valuation`: the identical call `_build_scores_dict` makes, so
      `valuation_verdict.factor_valuation == scores["factor_valuation"]`
      (display only; the absolute factor never votes).
    """
    from ..finance import factor_scores as fs
    fam_pct: float | None = None
    fam_cov: float | None = None
    summary = inputs.scorecard
    if summary is not None:
        cat = (summary.categories or {}).get("valuation")
        if cat is not None and cat.percentile is not None:
            fam_pct = float(cat.percentile)
            cov = cat.coverage if cat.coverage is not None else summary.coverage
            fam_cov = float(cov) if cov is not None else None
    comps = inputs.comps
    prem = (comps.premium_discount or {}).get("ev_ebitda") if comps is not None else None
    target_multiple = comps.target.ev_ebitda if comps is not None else None
    median_multiple = comps.median.ev_ebitda if comps is not None else None
    initial = dcf_stage.initial_dcf
    init_summary = _summarize_dcf(initial)
    ratios = inputs.ratios or {}
    return memo_quality.valuation_evidence_verdict(
        family_pct=fam_pct, family_coverage=fam_cov, comps_premium=prem,
        dcf_initial_upside=init_summary.get("base_upside"),
        dcf_initial_tv_clamped=bool(init_summary.get("tv_clamped")),
        dcf_final_upside=_summarize_dcf(dcf_stage.dcf).get("base_upside"),
        factor_valuation=fs.valuation_score(
            ratios.get("EV_EBITDA"), ratios.get("PFCF"), ratios.get("FCF_yield")),
        comps_target_multiple=target_multiple, comps_peer_median_multiple=median_multiple,
        sector=(inputs.profile or {}).get("sector"),
    )


def _compose_memo(inputs: MemoInputs, analysts: AnalystRound, dcf_stage: DCFStage) -> StockMemoOut:
    """Bull/bear/catalysts/risks, PM synthesis, enrichments, then the memo.

    Mutates `inputs.profile["risks"]` when the profile carries none (B4) —
    the verdict stage's thesis builder reads it. The memo holds the same
    finding objects as `analysts.findings`.
    """
    degradation = inputs.degradation
    profile = inputs.profile
    findings = analysts.findings
    ticker = inputs.ticker
    dcf = dcf_stage.dcf
    sector_finding = findings["sector"]
    earnings_finding = findings["earnings"]
    valuation_finding = findings["valuation"]
    risk_finding = findings["risk"]

    bull = safe_call(_bull_case, profile, valuation_finding, dcf, sector_finding, findings,
                     fallback=BullBearCase(headline="Bull case unavailable.", key_points=[]),
                     name="Bull Case Builder", log_to=degradation)
    bear = safe_call(_bear_case, profile, dcf, sector_finding, findings,
                     fallback=BullBearCase(headline="Bear case unavailable.", key_points=[]),
                     name="Bear Case Builder", log_to=degradation)
    catalysts: list[CatalystItem] = safe_call(
        _catalysts, profile, inputs.transcript, findings, inputs.earnings,
        fallback=[], name="Catalyst Builder", log_to=degradation,
    )
    risks: list[RiskItem] = safe_call(
        derive_risk_items, profile, fallback=[], name="Risk Item Builder", log_to=degradation,
    )
    # B4 — live profiles carry no `risks` field, so the profile-driven
    # extraction routinely returns nothing. The bear case is built from
    # real bear-polarity findings; backfill from it so the memo never
    # ships an empty risk section while a populated bear case exists.
    if not risks:
        risks = _risk_items_from_bear_case(bear)
        if risks and not profile.get("risks"):
            # The thesis builder reads profile["risks"] for its lever
            # clause — feed it the same list so it names a real risk
            # instead of degrading to template filler.
            profile["risks"] = [r.detail for r in risks]
    thesis_breakers = [r for r in risks if r.severity == "high"][:3]

    # W2b 7(b): the valuation verdict from evidence alone, BEFORE the PM
    # writes — so the PM reads it (and must argue any divergence) and so the
    # verdict can never be derived from the rating it is meant to check. A
    # crash is a hard "Valuation Verdict" entry (the same banner name the
    # verdict stage used), with a placeholder card rather than a silent one.
    valuation_verdict: ValuationVerdict = safe_call(
        _evidence_verdict, inputs, dcf_stage,
        fallback=ValuationVerdict(basis="evidence", summary=memo_quality.VERDICT_UNAVAILABLE_SUMMARY),
        name="Valuation Verdict", log_to=degradation,
    )

    from .llm import llm_call_context
    synth_fallback: dict[str, Any] = {
        "final_pm_view": "PM synthesis unavailable; relying on specialist findings only.",
        "one_sentence_thesis": f"Research draft for {profile.get('ticker', ticker)}.",
        "rating_label": "Neutral",
        "confidence_score": 50,
    }
    with llm_call_context(agent_name="PM Synthesis", run_id=inputs.run_id, route="strong"):
        synth: dict[str, Any] = safe_call(
            _pm_synthesis, profile, findings, dcf, scorecard=inputs.scorecard,
            valuation_evidence=valuation_verdict, news=inputs.news,
            fallback=synth_fallback, name="PM Synthesis", log_to=degradation,
        )
    # L6 / P6 recording: where the rating came from. Only the LLM PM's own
    # label is "llm"; the keyword synthesis and the crash fallback are not.
    # Recorded as float score keys only: no schema change, no published change.
    rating_source_llm = 1.0 if synth.pop(RATING_SOURCE_KEY, None) == "llm" else 0.0
    rating = synth.get("rating_label", "Neutral")
    raw_confidence = float(synth.get("confidence_score", 60))
    # The PM's stated reason for rating against the evidence. Memo content
    # (the reader sees it with the reconciliation note), capped like the
    # other PM strings.
    divergence_reason = str(synth.get("valuation_divergence_reason") or "").strip()[:1200]
    # W2b 7(a): forward figures the PM declared as assumptions (at most 5,
    # shape-checked here; the quality stage keeps only those whose basis
    # resolves in the ledger).
    inputs.forecast_assumptions = memo_quality.parse_forecast_assumptions(
        synth.get("forecast_assumptions"))

    # Phase 6 — the scorecard summary the memo carries. No row on file is a
    # SOFT degradation the reader must see (the section says n/a and why),
    # recorded before the memo's degradation fields are assembled below;
    # with the kill switch off nothing is recorded and the field stays
    # None. `record_soft` dedupes against a hard entry from a failed read.
    memo_scorecard = safe_call(
        scorecard_context.for_memo, inputs.scorecard,
        reconciliation=synth.get("scorecard_reconciliation"),
        fallback=None, name="Scorecard Summary", log_to=None,
    )
    if memo_scorecard is None and settings.enable_scorecard:
        degradation.record_soft(
            scorecard_context.AGENT_NAME,
            f"no scorecard row on file for {profile.get('ticker', ticker)}; section renders n/a",
            kind="DataUnavailable",
        )

    dcf_summary = _summarize_dcf(dcf)
    # Wave 10 — keep the consensus-anchored ("initial") DCF on the memo
    # alongside the PM-adjusted version, when they differ. Empty when no
    # PM adjustments fired.
    initial_dcf_summary = (
        _summarize_dcf(dcf_stage.initial_dcf)
        if dcf_stage.pm_adjustments and dcf_stage.initial_dcf is not dcf else {}
    )

    sources = [
        f"profile:{profile.get('ticker')}",
        f"financials:{profile.get('ticker')}",
    ]
    if inputs.transcript:
        sources.append(f"transcript:{inputs.transcript.get('period', '')}")
    for f in inputs.filings or []:
        sources.append(f"filing:{f.get('accession_number', f.get('type', ''))}")
    if inputs.comps:
        for p in inputs.comps.peers:
            sources.append(f"peer:{p.ticker}")
    if dcf:
        sources.append("dcf:base")

    # Dampen PM confidence by source-quality. A memo evidenced by filings +
    # transcripts + financials lands near 1.0; one leaning on news/social
    # gets a meaningful penalty. Prevents over-confident takes from thin evidence.
    ev_q = evidence_quality(sources)
    blended_confidence = max(20.0, min(95.0, raw_confidence * (0.6 + 0.4 * ev_q)))

    # Wave 10 — pull the mispricing thesis off the PM's structured
    # output (PM_SYNTHESIS_PROMPT now requires it). Empty fallback when
    # the deterministic path ran (no LLM) or the PM declined.
    raw_misp = synth.get("mispricing_thesis") or {}
    if not isinstance(raw_misp, dict):
        raw_misp = {}
    mispricing = MispricingThesis(
        consensus_view=str(raw_misp.get("consensus_view") or "")[:1000],
        our_view=str(raw_misp.get("our_view") or "")[:1000],
        gap=str(raw_misp.get("gap") or "")[:1000],
        falsifiers=[str(x)[:300] for x in (raw_misp.get("falsifiers") or [])][:5],
    )

    # Wave 10 — freeze the memo-time price so the UI can later overlay
    # the live quote and show drift. Best-effort: null when the quote
    # chain misses (the live-overlay path then has nothing to compare
    # against, which is fine).
    price_at_memo: float | None = None
    try:
        from ..services.market_data_service import get_current_price
        price_at_memo = get_current_price(profile.get("ticker", ticker))
    except Exception as exc:  # pragma: no cover — never block a memo
        log.debug("price_at_memo capture failed: %s", exc)
    if price_at_memo is not None:
        register_source("price", f"price:{ticker}", {"price_at_memo": price_at_memo})

    # Wave 10 — forward catalyst calendar (next 90d). Best-effort —
    # the table may be empty until the cron has run at least once.
    forward_catalysts: list[dict[str, Any]] = []
    try:
        from ..services.catalyst_service import get_upcoming
        forward_catalysts = get_upcoming(profile.get("ticker", ticker), days_ahead=90)
        register_source("catalyst_calendar", f"catalysts:{ticker}", forward_catalysts)
    except Exception as exc:  # pragma: no cover
        # (a) the catalyst tile is legitimately empty before the calendar
        # cron has run, so an empty tile is not a memo degradation — but a
        # *failed* read should be visible in the log, not a debug line.
        log.warning("forward_catalysts fetch failed for %s: %s", ticker, type(exc).__name__)

    # Wave 10 — earnings quarter-over-quarter delta. Reads the
    # earnings agent's structured payload and walks back through the
    # memo history for prior-quarter context. None when no prior data.
    earnings_qoq: AgentFinding | None = None
    try:
        from .earnings_qoq import run_earnings_qoq_delta
        earnings_struct = (earnings_finding.data or {}).get("structured") if earnings_finding else None
        earnings_qoq = run_earnings_qoq_delta(
            profile.get("ticker", ticker), earnings_struct,
        )
    except Exception as exc:  # pragma: no cover
        # (b) the QoQ tile silently vanishes from the memo — the reader
        # cannot tell "no prior quarter" from "the delta crashed". Record
        # it so the banner says which.
        log.warning("earnings QoQ delta failed for %s: %s", ticker, type(exc).__name__)
        degradation.record_soft(
            "Earnings QoQ", f"quarter-over-quarter delta unavailable: {redact(exc)}",
            kind=type(exc).__name__,
        )

    # Wave 10 — per-agent influence on the rating. Computed from each
    # finding's confidence + tone; deterministic, no extra LLM cost.
    # Powers per-agent attribution dashboards + the PM's eventual
    # "discount this specialist" feedback loop.
    agent_influence: dict[str, float] = {}
    try:
        from .influence import compute_influence
        # Scored over what the PM read: a withheld read (a template stand-in
        # `_pm_view` kept out of the synthesis) must not be stored as a pull
        # on a rating it never reached.
        agent_influence = compute_influence(_pm_view(findings).visible)
    except Exception as exc:  # pragma: no cover
        log.debug("agent influence computation failed: %s", exc)

    # Wave 10 — freeze the macro context that produced this rating.
    # Lets postmortem regime-conditional bucketing work even after
    # the macro broadcast cache rolls over.
    macro_snapshot_at_memo: dict[str, float] = {}
    macro_regime_at_memo: str = ""
    try:
        from ..cache import cache_get
        broadcast = cache_get("macro:global", "macro_broadcast")
        if broadcast and isinstance(broadcast.payload, dict):
            macro_regime_at_memo = str(broadcast.payload.get("regime") or "")
            snap = broadcast.payload.get("snapshot") or {}
            if isinstance(snap, dict):
                macro_snapshot_at_memo = {
                    str(k): float(v) for k, v in snap.items()
                    if isinstance(v, (int, float))
                }
                # FRED series are printed in percent already.
                register_source("macro", "macro:snapshot_at_memo", macro_snapshot_at_memo, pct=True)
    except Exception as exc:  # pragma: no cover
        log.debug("macro snapshot freeze failed: %s", exc)

    # Each analyst lands on the memo field its spec names; an analyst with
    # no dedicated field rides in `extra_agent_views` (none today — the risk
    # read is deliberately unsurfaced, see `roster.NO_MEMO_VIEW`).
    views: dict[str, Any] = {
        spec.memo_field: findings[spec.key] for spec in roster.AGENTS
        if spec.memo_field and spec.key in findings
    }
    extra_views: dict[str, AgentFinding] = {
        spec.key: findings[spec.key] for spec in roster.AGENTS
        if spec.memo_field is None and spec.key not in roster.NO_MEMO_VIEW and spec.key in findings
    }
    memo = StockMemoOut(
        ticker=profile.get("ticker"),
        company_name=profile.get("company_name", ticker),
        sector=profile.get("sector", ""),
        final_pm_view=synth.get("final_pm_view", ""),
        rating_label=rating,
        confidence_score=blended_confidence,
        one_sentence_thesis=synth.get("one_sentence_thesis", ""),
        mispricing_thesis=mispricing,
        price_at_memo=price_at_memo,
        price_at_memo_at=(datetime.utcnow() if price_at_memo is not None else None),
        business_summary=profile.get("business_description", ""),
        **views,
        extra_agent_views=extra_views,
        bull_case=bull,
        bear_case=bear,
        catalysts=catalysts,
        key_risks=risks,
        thesis_breakers=thesis_breakers,
        dcf_summary=dcf_summary,
        dcf_initial_summary=initial_dcf_summary,
        dcf_pm_adjustments=dcf_stage.pm_adjustments,
        dcf_pm_adjustment_headline=dcf_stage.pm_headline,
        portfolio_fit=_portfolio_fit(profile, rating),
        # Composition only seeds a typed placeholder. The review stage runs
        # the critic once, against the complete draft, before persistence.
        risk_committee_challenge=CriticReview(
            overall_assessment="Pending critic review.", review_mode="pending",
        ),
        final_verdict="",
        scores={
            **_build_scores_dict(
                blended_confidence=blended_confidence,
                raw_confidence=raw_confidence,
                ev_q=ev_q,
                sector_finding=sector_finding,
                valuation_finding=valuation_finding,
                risk_finding=risk_finding,
                earnings_finding=earnings_finding,
                profile=profile, ratios=inputs.ratios, earnings=inputs.earnings,
            ),
            # P6 recording (bullish-skew diagnosis; L6): 1.0 when the LLM PM
            # produced the label, else 0.0, and the PM's label as a bucket
            # centre BEFORE risk recommendations, the blend and 7(b) move it.
            # Lets the track record and the learning ledger segment by where
            # a rating came from instead of measuring the keyword fallback as
            # if it were the committee.
            "rating_source_llm": rating_source_llm,
            "pm_rating_score": score_from_rating_label(rating),
        },
        sources_used=sources,
        generated_at=datetime.utcnow(),
        # Label follows the SAME flag that gates the data path
        # (`use_demo_data_only`), not `enable_live_data` alone. Production
        # sets USE_DEMO_DATA=false without ENABLE_LIVE_DATA, which made the
        # old check label genuinely live-data memos as "demo" (Theme 3).
        generation_mode="live" if settings.has_llm and not settings.use_demo_data_only else "demo",
        degraded_agents=degradation.degraded_agents(),
        degradation_events=degradation.events(),
        round_findings=analysts.round_findings,
        forward_catalysts=forward_catalysts,
        earnings_qoq_delta=earnings_qoq,
        intake_decision=analysts.intake.model_dump() if analysts.intake.skipped else {},
        agent_influence=agent_influence,
        macro_snapshot_at_memo=macro_snapshot_at_memo,
        macro_regime_at_memo=macro_regime_at_memo,
        scorecard=memo_scorecard,
        valuation_verdict=valuation_verdict,
        # W2b: the PM's side of the 7(b) reconciliation, recorded now; the
        # review stage completes it against the post-blend rating, and the
        # critic reads it from the draft.
        quality=MemoQuality(rating_reconciliation=RatingReconciliation(
            pm_rating=str(rating), pm_confidence=raw_confidence,
            valuation_verdict=valuation_verdict.verdict, reason=divergence_reason,
        )),
        # W2a write-time provenance: facts the payload cannot recover later.
        # Without keys every LLM-intended section is the deterministic
        # stand-in by design, and the presenter hides it from readers; the
        # verdict stage adds "thesis" and "mispricing" (VerdictOutcome.apply).
        section_provenance={"v": 1, "llm_configured": bool(settings.has_llm)},
    )

    # W2b 7(a): the quant factor block the memo prints beside the rating.
    # The confidence entries are the PM's own numbers, never a source.
    register_source("factor_scores", f"factor_scores:{ticker}", {
        k: v for k, v in (memo.scores or {}).items()
        if k.startswith("factor_") or k in ("beat_streak", "guidance_net_direction")
    })

    # Wave 9 — surface deep-research counters on `memo.scores` so the
    # admin dashboard can chart how often the dialog converges vs. caps
    # out. Round 0 is the fan-out and is always present when the loop
    # ran; rounds 1+ are the PM critique passes.
    round_findings = analysts.round_findings
    if round_findings and isinstance(memo.scores, dict):
        memo.scores = {
            **memo.scores,
            "deep_research_rounds": float(
                max((r.round for r in round_findings), default=0)
            ),
            "deep_research_questions": float(sum(
                len(r.pm_questions) for r in round_findings
            )),
        }
    return memo


# ---------------------------------------------------------------------------
# Stage 5 — review (critic, reflection, risk recs, rating blend)
# ---------------------------------------------------------------------------

def _review_memo(memo: StockMemoOut, inputs: MemoInputs, analysts: AnalystRound) -> StockMemoOut:
    """Critic, risk recommendations, rating blend, rating reconciliation.

    Owns the `graph.critic` checkpoint. Mutates `memo` in place (and
    `findings["risk"].data["applied_recommendations"]`) and refreshes the
    memo's degradation fields after the critic, which makes an LLM call
    that can record failures.

    Two things moved out (W2b §6.1): long-term-memory reflection now runs
    after the quality stage (`_run_reflection`), so memory records the final
    rating and confidence; and the "Final rating after ..." preface on the
    PM view is rendered once from final values (`_render_final_texts`).
    """
    degradation = inputs.degradation
    risk_finding = analysts.findings["risk"]

    # Run critic on a draft of the memo (pass dict to avoid recursion).
    # safe_critic upgrades exceptions into a typed "critic unavailable" review
    # so a flaky Anthropic call doesn't kill the memo.
    from .llm import llm_call_context
    draft_for_critic = memo.model_dump()
    with llm_call_context(agent_name="Risk Committee", run_id=inputs.run_id, route="strong"):
        critic = safe_critic(_checkpointed_critic, draft_for_critic, log_to=degradation)
    if critic:
        memo.risk_committee_challenge = critic
    # Refresh degraded_agents in case the critic recorded a failure.
    _sync_degradation(memo, degradation)

    # Wave 8H — apply the risk analyst's structured recommendations.
    # Runs AFTER the memo body is assembled but BEFORE final_verdict +
    # persistence so confidence cap / rating downshift / thesis_breaker
    # propagation / bear-case augmentation all flow through to the
    # downstream UI + cache. `applied_recs` rides on the memo's `scores`
    # for transparency.
    applied_risk_recs = _apply_risk_recommendations(memo, risk_finding)
    if applied_risk_recs and isinstance(memo.scores, dict):
        memo.scores = {
            **memo.scores,
            "risk_recs_applied": float(len(applied_risk_recs)),
        }
    # Stash the audit trail on the risk finding's data block so the
    # frontend can render a "Risk recs applied" panel + the long-form
    # report can quote them verbatim.
    if isinstance(risk_finding.data, dict):
        risk_finding.data["applied_recommendations"] = applied_risk_recs

    _blend_rating(memo)
    # W2b 7(b) binds the PUBLISHED rating, so it runs on the post-blend one.
    _reconcile_rating(memo, degradation)
    if isinstance(memo.scores, dict):
        memo.scores = {**memo.scores, "confidence": float(memo.confidence_score)}
    return memo


def _reconcile_rating(memo: StockMemoOut, degradation: DegradationLog) -> None:
    """Rule 7(b) on the post-blend rating (`memo_quality.reconcile_rating`).

    Writes `memo.rating_label` (enforce mode only) and
    `memo.quality.rating_reconciliation`. A reason check that crashed
    fails closed (the rating moves to Neutral) and is a soft "Rating Check"
    degradation, so the reader sees why. A memo without `quality` (a
    fixture) gets a reconciliation with an empty reason.
    """
    quality = memo.quality or MemoQuality()
    critic = memo.risk_committee_challenge
    enforce = settings.rating_reconciliation_mode != "record"
    try:
        rec = memo_quality.reconcile_rating(
            blended_rating=str(memo.rating_label), verdict=memo.valuation_verdict,
            pm=quality.rating_reconciliation, critic=critic, enforce=enforce,
        )
    except Exception as exc:
        # Fail closed here too: a divergent rating nobody could check does
        # not ship as if it had passed.
        log.warning("rating check failed for %s: %s", memo.ticker, type(exc).__name__)
        blended = str(memo.rating_label)
        divergent = memo_quality.diverges(blended, memo.valuation_verdict.verdict)
        rec = RatingReconciliation(
            outcome="downgraded" if divergent else "not_applicable",
            pm_rating=(quality.rating_reconciliation.pm_rating
                       if quality.rating_reconciliation else ""),
            blended_rating=blended,
            final_rating="Neutral" if (divergent and enforce) else blended,
            valuation_verdict=memo.valuation_verdict.verdict, divergence=divergent,
            reason_checks={"check_failed": True},
            note="The rating check could not run; a divergent rating was set to Neutral.",
        )
    if rec.reason_checks.get("check_failed"):
        degradation.record_soft(
            "Rating Check", "divergence reason could not be verified; the check failed closed",
            kind="RatingCheckFailed",
        )
        _sync_degradation(memo, degradation)
    memo.rating_label = rec.final_rating  # type: ignore[assignment]
    memo.quality = quality.model_copy(update={"rating_reconciliation": rec})
    log.info(
        "rating check %s: pm=%s blended=%s verdict=%s outcome=%s final=%s mode=%s",
        memo.ticker, rec.pm_rating, rec.blended_rating, rec.valuation_verdict,
        rec.outcome, rec.final_rating, settings.rating_reconciliation_mode,
    )


def _blend_rating(memo: StockMemoOut) -> None:
    """Rating blend (Option A) — mix the PM LLM's directional call with
    the quant factor_pm_score. Weight is `LLM_RATING_WEIGHT` in
    config.env (default 0.4). At weight=0 this collapses to the
    prior Wave 8P behavior (factor score is dispositive); at
    weight=1 the LLM call wins outright. The LLM rating read here
    is post-risk-rec, so risk_agent downgrades flow into the blend.

    Reads `rating_label` and `scores["factor_pm_score"]` and nothing else
    — in particular not `memo.scorecard`. The Phase 6 scorecard informs
    the memo (context, section, disagreement flag) but does not move the
    rating in this phase; `test_memo_consistency` pins that.
    """
    from ..schemas import rating_from_stock_score, score_from_rating_label
    factor_pm = (memo.scores or {}).get("factor_pm_score")
    if factor_pm is not None:
        w = max(0.0, min(1.0, float(settings.llm_rating_weight)))
        llm_score = score_from_rating_label(memo.rating_label)
        blended = w * llm_score + (1.0 - w) * float(factor_pm)
        memo.rating_label = rating_from_stock_score(blended)  # type: ignore[assignment]
        if isinstance(memo.scores, dict):
            memo.scores = {
                **memo.scores,
                "llm_rating_score": float(llm_score),
                "llm_rating_weight": float(w),
                "blended_pm_score": round(float(blended), 1),
            }


# ---------------------------------------------------------------------------
# Stage 6 — verdict (pure)
# ---------------------------------------------------------------------------

def _build_verdict(
    memo: StockMemoOut, *,
    comps: CompsResult | None, dcf: DCFResult | None,
    profile: dict[str, Any], findings: dict[str, AgentFinding], ticker: str,
) -> VerdictOutcome:
    """Reconcile the memo's valuation call, thesis, mispricing card and
    final verdict from the post-review memo.

    Pure: reads `memo` / the inputs and writes nothing — the caller applies
    the returned `VerdictOutcome` (`VerdictOutcome.apply`). Failures the
    stage swallows on the reader's behalf ride on `outcome.degradations`
    in the order they occurred, so a fixture memo can exercise every
    branch without a pipeline run or an activated log. The one side
    channel is `_build_thesis_from_findings`, whose consensus-gap clause
    reports through `note_soft` on the run's active log (a no-op outside
    a memo run).
    """
    notes: list[DegradationNote] = []

    def _guarded(name: str, fn: Callable[..., T], *args: Any, fallback: T) -> T:
        # `safe_call` without a log: the same warning + redacted record shape
        # as `DegradationLog.record`, but collected on `notes` so the stage
        # constructs no log of its own (the run has exactly one — the
        # failover-attribution test pins that) and stays pure.
        try:
            return fn(*args)
        except Exception as exc:
            log.warning("Safe call %s failed: %s", name, type(exc).__name__)
            log.debug("Safe call %s failed — traceback follows", name, exc_info=True)
            notes.append(DegradationNote(name, type(exc).__name__, redact(exc), soft=False))
            return fallback

    def _note_soft(agent: str, reason: str, exc: BaseException) -> None:
        notes.append(DegradationNote(agent, type(exc).__name__, reason[:300], soft=True))

    # W2b 7(b): the valuation verdict is the evidence read the compose stage
    # stored (before the PM wrote); it is passed through, never recomputed
    # here — recomputing it from the post-blend memo is how it used to follow
    # the rating badge. Everything downstream (thesis guard, mispricing
    # fallback, UI valuation card) reads it.
    valuation_verdict = memo.valuation_verdict
    rec = memo.quality.rating_reconciliation if memo.quality is not None else None
    accepted = rec is not None and rec.outcome == "accepted"

    # Anti-pattern guard + verdict-consistency guard. The PM prompt forbids
    # the "{Company} — {Sector} / {industry}, {hook}; DCF base case +X%"
    # templated form, but in practice the LLM sometimes ignores it (or the
    # deterministic fallback historically emitted it). The stated verdict
    # word is rewritten only when it contradicts BOTH the final rating's
    # direction and the evidence verdict (either is a defensible reading),
    # and never for an accepted divergence — the PM argued that one.
    thesis = memo.one_sentence_thesis
    is_anti_pattern = _looks_like_anti_pattern_thesis(thesis)
    stated_word = next(
        (w for w in ("undervalued", "overvalued", "fairly priced")
         if w in (thesis or "").lower()),
        None,
    )
    rewrite_fired, expected_word = memo_quality.thesis_rewrite_word(
        stated=stated_word, rating=memo.rating_label, verdict=valuation_verdict,
        accepted=accepted, anti_pattern=is_anti_pattern,
    )
    # W2a: whether the builder's text actually replaced the PM's thesis.
    # `rewrite_fired` alone is not enough — the rewrite is rejected when it
    # would itself be the anti-pattern, and the PM's words then stand.
    thesis_rewritten = False
    if rewrite_fired:
        # B7 — log every rewrite so the false-positive rate of this guard is
        # measurable in prod logs. The thesis is PM model output, so the line
        # carries its length and sha1, never its text (attribution critique
        # #15): the stored memo version holds the words, and the sha1 finds
        # the one a line is about.
        original = thesis or ""
        log.info(
            "thesis rewrite fired for %s (anti_pattern=%s, stated=%r, "
            "expected=%r); original_len=%d original_sha1=%s",
            ticker, is_anti_pattern, stated_word, expected_word,
            len(original), hashlib.sha1(original.encode("utf-8")).hexdigest(),
        )
        try:
            rewritten = _build_thesis_from_findings(
                profile, findings, dcf, ticker, verdict_word=expected_word,
            )
            if rewritten and not _looks_like_anti_pattern_thesis(rewritten):
                thesis_rewritten = rewritten != thesis
                thesis = rewritten
        except Exception as exc:  # pragma: no cover — never break the memo
            # (b) the thesis the reader sees keeps the anti-pattern form or
            # the wrong verdict word — exactly what this guard exists to
            # prevent. Surface it instead of swallowing it.
            log.warning("thesis rewrite failed for %s: %s", ticker, type(exc).__name__)
            _note_soft("Thesis Builder", f"thesis rewrite failed: {redact(exc)}", exc)
    # The word the final thesis stands on: the rewrite's, else what the
    # thesis states, else the expected one. The gap clause must agree with it.
    thesis_word = (expected_word if thesis_rewritten else stated_word) or expected_word or "fairly priced"

    # Wave 8R — thesis augmentation. Surface where the model diverges
    # from analyst consensus (the actual *what is the market missing*
    # framing). Compares the DCF's 5-year growth path average against
    # the consensus 5-year average; appends a clause when the gap is
    # material. No-ops cleanly when consensus isn't available. Skip
    # when the thesis already carries the clause (the deterministic
    # path bakes it in via `_build_thesis_from_findings`).
    try:
        delta_clause = _market_gap_clause(profile, dcf, ticker)
        # Only append when it agrees with the verdict word implied by the
        # headline rating — an "upside the market is missing" clause behind
        # an overvalued call would contradict the thesis.
        if (
            delta_clause
            and delta_clause not in thesis
            and _gap_clause_agrees(delta_clause, thesis_word)
        ):
            thesis = thesis.rstrip(".") + ". " + delta_clause
    except Exception as exc:  # pragma: no cover — never break a memo on thesis polish
        # (b) the "what is the market missing" clause is the part of the
        # thesis a reader pays for; losing it silently is a memo change.
        # Soft notes dedupe per agent on apply, so an earlier "Thesis
        # Builder" entry from the rewrite guard above is not doubled.
        log.warning("thesis gap-clause polish failed for %s: %s", ticker, type(exc).__name__)
        _note_soft("Thesis Builder", f"consensus-gap clause polish failed: {redact(exc)}", exc)

    # B6 — never ship an empty mispricing card. When the PM declined (or
    # the deterministic path ran), build the consensus-vs-us structure
    # from the reconciled verdict + final thesis + risk list. Runs after
    # the thesis guards so `our_view` quotes the final thesis — hence the
    # draft carrying the fields decided above; `memo` itself is untouched.
    mispricing = memo.mispricing_thesis
    mispricing_fallback = False
    if not (mispricing.consensus_view or mispricing.our_view or mispricing.gap):
        draft = memo.model_copy(
            update={"valuation_verdict": valuation_verdict, "one_sentence_thesis": thesis},
        )
        # A crash here used to ship an empty mispricing card — the field B6
        # exists to never leave empty — with no banner entry (RP-001).
        mispricing = _guarded(
            "Mispricing Fallback", _build_mispricing_fallback, draft, fallback=mispricing,
        )
        # W2a: the card is the template only when the fallback built one; a
        # crashed fallback leaves the (empty) PM card, which reads not_produced.
        mispricing_fallback = bool(mispricing.consensus_view or mispricing.our_view or mispricing.gap)

    # Phase 6: pull through cross-sector relevance from the sector agent's
    # finding into the PM memo so users see related-name implications without
    # a second model call. Cohort placement is already in the sector view.
    sector_finding = findings["sector"]
    cross_relevance: list[str] = []
    if isinstance(sector_finding.data, dict):
        cross_relevance = sector_finding.data.get("cross_sector_relevance") or []
    cross_relevance_blurb = (
        f" Cross-sector pull-through: {', '.join(cross_relevance)}." if cross_relevance else ""
    )
    cohort_blurb = ""
    if isinstance(sector_finding.data, dict):
        kpi_placements = sector_finding.data.get("kpi_placements") or {}
        if kpi_placements:
            cohort_blurb = " Cohort placement: see sector view for KPI quartile context."

    # Wave 3A: surface the key disagreement + sector lean in the verdict so
    # readers can see what the bull/bear case actually pivots on.
    sector_lean_blurb = ""
    bb_payload = (
        sector_finding.data.get("bull_bear_analysis")
        if isinstance(sector_finding.data, dict) else None
    )
    if isinstance(bb_payload, dict):
        lean = bb_payload.get("sector_lean")
        disagreement = (bb_payload.get("key_disagreement") or "").strip()
        if lean and lean != "balanced":
            sector_lean_blurb += f" Sector lean: {lean}."
        if disagreement:
            sector_lean_blurb += f" Key disagreement: {disagreement}"

    # Final verdict ties together rating, confidence, and PM view succinctly.
    # Only the BODY is built here: its "PM final view: <rating> (confidence
    # N)." lead is rendered once, after the quality stage has set the final
    # confidence (`VerdictOutcome.render`, `_render_final_texts`).
    # thesis_breakers are read post-review (risk recs may have grown them).
    final_verdict_body = (
        f"{thesis}"
        f"{cohort_blurb}{cross_relevance_blurb}{sector_lean_blurb} "
        f"Watch items: {', '.join(r.title for r in memo.thesis_breakers) or 'none flagged.'}"
    )
    extra_scores: dict[str, float] = (
        {"cross_sector_relevance_count": float(len(cross_relevance))} if cross_relevance else {}
    )
    return VerdictOutcome(
        valuation_verdict=valuation_verdict,
        one_sentence_thesis=thesis,
        mispricing_thesis=mispricing,
        final_verdict_body=final_verdict_body,
        extra_scores=extra_scores,
        thesis_rewrite_fired=rewrite_fired,
        thesis_rewritten=thesis_rewritten,
        mispricing_fallback=mispricing_fallback,
        degradations=notes,
    )


# ---------------------------------------------------------------------------
# Stage 6b — quality (pure): 7(a) number check + 7(c) earned confidence
# ---------------------------------------------------------------------------

def _check_numbers(
    memo: StockMemoOut, inputs: MemoInputs, notes: list[DegradationNote],
) -> tuple[NumberCheck | None, number_check.WithholdPlan | None]:
    """7(a): the memo's figures against the run's source ledger.

    None when there is no ledger (a direct stage call; nothing was
    registered, so nothing can be judged). An incomplete registry (a
    resumed step without stored sources, a registration that failed) is
    "not checked": no flags, no withholding, the `figures_unchecked` cap —
    judging figures against a partial registry would flag real ones. A
    crash is a hard "Number Check" degradation with the same unchecked
    result. `UntraceableNumbers` are recorded in `quality`, never on the
    degraded-agents banner: an untraceable figure is a finding about the
    memo, not an analyst outage."""
    ledger = getattr(inputs, "ledger", None)
    if ledger is None:
        return None, None
    try:
        registry = ledger.snapshot()
        declared, dropped = _resolve_assumptions(getattr(inputs, "forecast_assumptions", []), registry)
        if not registry.complete:
            steps = ", ".join(registry.incomplete_steps[:3])
            log.info("number check %s: not checked, source registry incomplete (%s)", memo.ticker, steps)
            return NumberCheck(
                checked=False, method_version=number_check.METHOD_VERSION,
                counts={"registry_facts": len(registry.facts), "registry_sources": len(registry.sources)},
                assumptions=[{**a, "status": "assumption"} for a in declared],
                notes=[f"source registry incomplete ({steps})", *dropped],
            ), None
        result = number_check.check_memo(
            memo, registry, withhold=settings.number_check_withhold, assumptions=declared)
        nc = number_check.summarize(result, assumptions=declared, notes=dropped)
    except Exception as exc:
        log.warning("number check failed for %s: %s", memo.ticker, type(exc).__name__)
        notes.append(DegradationNote(agent="Number Check", error_type=type(exc).__name__,
                                     message=redact(exc), soft=False))
        return NumberCheck(checked=False, notes=[f"the number check crashed ({type(exc).__name__})"]), None
    c = nc.counts
    log.info(
        "number check %s: fields=%d claims=%d traced=%d weak=%d mis_anchored=%d untraceable=%d "
        "distinct=%d planned_withheld=%d registry=%d facts/%d sources",
        memo.ticker, c.get("fields_checked", 0), c.get("claims_total", 0), c.get("traced", 0),
        c.get("weak", 0), c.get("mis_anchored", 0), c.get("untraceable", 0),
        c.get("flagged_distinct", 0), sum(len(v) for v in result.plan.items.values()),
        c.get("registry_facts", 0), c.get("registry_sources", 0),
    )
    return nc, result.plan


def _resolve_assumptions(
    declared: list[dict[str, Any]], registry: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Declared forecast assumptions whose `basis_ref` names a registered
    source; the rest are dropped with a note (their figures stay unchecked
    claims, i.e. untraceable)."""
    kept: list[dict[str, Any]] = []
    notes: list[str] = []
    for a in declared or []:
        if registry.resolves(a.get("basis_ref", "")):
            kept.append(a)
        else:
            notes.append(f"forecast assumption not accepted: basis_ref "
                         f"{str(a.get('basis_ref'))[:60]!r} is not a registered source")
    return kept, notes


def _assess_quality(memo: StockMemoOut, inputs: MemoInputs, analysts: AnalystRound) -> QualityOutcome:
    """7(a) the number-to-source check and 7(c) the caps on the PM's
    confidence, from the post-verdict memo.

    Pure: reads the memo and the inputs; the orchestrator applies the
    returned `QualityOutcome` (withholding included). Template-filled
    sections come from the W2a presenter's `compute_availability` (contract
    C2) so the caps and the "unavailable in this version" placeholders can
    never disagree about which section was a template. A section that map
    could not classify raises (`memo_quality.template_filled`), so
    `_quality_fallback` caps the memo instead of the template caps silently
    dropping out. The number-based caps apply only when the number check
    ran (`memo_quality.number_caps`).
    """
    from ..services.memo_sections import compute_availability

    notes: list[DegradationNote] = []
    nc, plan = _check_numbers(memo, inputs, notes)
    pm_template, template_sections = memo_quality.template_filled(compute_availability(memo))
    filing = analysts.findings.get("filing")
    filing_skipped = bool(
        filing is not None and isinstance(filing.data, dict) and filing.data.get("intake_skipped")
    )
    rec = memo.quality.rating_reconciliation if memo.quality is not None else None
    # Only a divergence 7(b) ACCEPTED on a reason no live critic supported.
    # A downgraded one no longer diverges in enforce mode; in record mode it
    # still does, but the kill switch must leave published output (rating
    # AND confidence) as if 7(b) were off, and "a reason no live critic
    # reviewed" would misdescribe a missing or critic-rejected reason.
    divergence_unreviewed = (
        rec is not None and rec.divergence and rec.outcome == "accepted"
        and memo_quality.diverges(memo.rating_label, memo.valuation_verdict.verdict)
        and rec.critic_assessment != "supported"
    )
    confidence = memo_quality.earned_confidence(
        raw=float(memo.confidence_score),
        pm_template=pm_template,
        template_sections=template_sections,
        critic_mode=memo.risk_committee_challenge.review_mode,
        transcript_given=inputs.transcript is not None,
        filing_reviewed=bool(inputs.filings) and not filing_skipped,
        divergence_unreviewed=divergence_unreviewed,
        number_check=nc,
    )
    log.info(
        "confidence check %s: raw=%.1f final=%.1f binding=%s caps=%s",
        memo.ticker, confidence.raw, confidence.final, confidence.binding,
        ",".join(f"{c.code}:{c.cap:g}" for c in confidence.caps) or "none",
    )
    return QualityOutcome(confidence=confidence, number_check=nc, withhold=plan, degradations=notes)


def _quality_fallback(memo: StockMemoOut) -> QualityOutcome:
    """What ships when `_assess_quality` crashes: never an uncapped memo.

    Nothing is known about which caps apply, so the most conservative
    section cap stands in (the "Memo Quality" banner entry says why)."""
    raw = float(memo.confidence_score)
    cap = ConfidenceCap(
        code="quality_check_failed", cap=memo_quality.CAP_TEMPLATE_SECTIONS[3],
        detail="The confidence checks could not run for this memo.",
    )
    final = min(raw, cap.cap)
    return QualityOutcome(confidence=ConfidenceAssessment(
        raw=raw, final=final, caps=[cap], binding=cap.code if final < raw else None,
    ))


def _render_final_texts(memo: StockMemoOut, verdict: VerdictOutcome, initial: PMOpinion) -> None:
    """Render the two strings that embed rating and confidence, once.

    Runs after the quality stage, so both read the FINAL rating and the
    earned confidence (rendering them earlier and patching later is the
    5aa1b74 stale-string class):
      * `final_verdict` = lead + the verdict stage's body;
      * the PM-view preface, written when the rating or confidence moved
        since the PM wrote. It names each step that moved something; with
        only the risk review and blend involved the wording is unchanged.
    """
    memo.final_verdict = verdict.render(memo)
    nc = memo.quality.number_check if memo.quality is not None else None
    if memo.rating_label == initial.rating and float(memo.confidence_score) == initial.confidence:
        memo.final_pm_view = initial.text
        return
    q = memo.quality
    rec = q.rating_reconciliation if q is not None else None
    steps = ["risk review", "factor blend"]
    if rec is not None and rec.outcome == "downgraded" and rec.final_rating != rec.blended_rating:
        steps.append("valuation check")
    conf = q.confidence if q is not None else None
    if conf is not None and conf.final < conf.raw:
        steps.append("evidence cap on confidence")
    after = " and ".join(steps) if len(steps) == 2 else ", ".join(steps[:-1]) + " and " + steps[-1]
    prefix = (
        f"Final rating after {after}: {memo.rating_label} "
        f"(confidence {memo.confidence_score:g}/100).\n\n"
        f"PM rationale before those adjustments (rating {initial.rating}; "
        f"confidence {initial.confidence:g}/100):\n"
    )
    memo.final_pm_view = prefix + initial.text
    # 7(a) claims on the PM view were located in the PM's own text; the
    # preface moves them (`text[start:end] == raw` must hold as stored).
    number_check.shift_field(nc, "final_pm_view", len(prefix))


def _run_reflection(memo: StockMemoOut, inputs: MemoInputs) -> None:
    """Long-term memory reflection on the FINAL memo (W2b §6.4).

    Appends a structured entry to the company + sector memory files iff a
    delta event fired this run (new earnings / new filing / material news).
    It runs after the quality stage so memory records the published rating
    and earned confidence, not the pre-blend draft. safe_call wraps it so a
    memory write never blocks the memo. Skipped on backtests (`as_of_date`
    set): the agent's notebook must not collect retroactive entries.
    """
    if inputs.as_of_date is not None:
        return
    safe_call(
        _run_reflection_step, memo,
        fallback=([], []),
        name="Reflection (long-term memory)", log_to=inputs.degradation,
    )
    _sync_degradation(memo, inputs.degradation)


# ---------------------------------------------------------------------------
# Stage 7 — persist
# ---------------------------------------------------------------------------

def _persist(memo: StockMemoOut, inputs: MemoInputs) -> StockMemoOut:
    """Persist a versioned snapshot and return the memo.

    `first_run` only fires when no prior version exists for this ticker;
    otherwise this is a `full_reanalysis` (the news-driven
    `incremental_patch` path is owned by the update-orchestrator, not this
    code path).

    Persistence used to be wrapped in safe_call so a DB hiccup wouldn't
    block the in-memory return value — but for the async regen path
    that's a silent disaster: the regen looks "successful" while the
    memo never reaches the database. The user clicks Refresh, sees
    spinning, then the old memo. Now we let persistence errors raise.
    The regen worker (services/regen_worker.py) catches BaseException
    and records the traceback on the RegenJob row, surfaced via
    /analyze/status and /api/admin/regen-jobs.
    """
    degradation = inputs.degradation
    # Last LLM call is behind us: pick up any failover the later stages
    # recorded so the persisted memo says which vendor actually wrote it.
    _absorb_failover_events(degradation)
    _sync_degradation(memo, degradation)
    try:
        snapshot = _persist_memo_snapshot(memo, inputs.as_of_date)
    except Exception as exc:
        log.error(
            "memo persistence FAILED for %s: %s: %s",
            inputs.ticker, type(exc).__name__, exc,
        )
        # Record on the degradation log so synchronous callers (sync=true
        # path) can still see what happened via memo.degraded_agents.
        degradation.record("Memo store", exc)
        _sync_degradation(memo, degradation)
        raise
    _sync_degradation(memo, degradation)

    # Phase 6 — the disagreement row keyed on the snapshot just written,
    # and the review rows this run answered. Finding records, not memo
    # content: a failure is logged, never raised and never a degradation
    # (the memo is already saved and whole).
    snapshot_id = getattr(snapshot, "id", None)
    # Live memos only: a backtest (`as_of_date` set) still carries the
    # flag on `memo.scorecard.disagreement` — that is memo content — but
    # writes no finding row, because an `open` row is what
    # `handle_scorecard_disagreements` turns into a present-day review
    # regen, and a reproduced historical disagreement must not spend one
    # of the day's regen slots.
    if memo.scorecard is not None and inputs.as_of_date is None:
        safe_call(
            scorecard_context.persist_disagreement, memo, snapshot_id,
            fallback=None, name="Scorecard Disagreement", log_to=None,
        )
    # Independent of whether the summary survived: the seeds were asked in
    # round 1 whatever the later read returned (a GC'd row, a DB hiccup),
    # and a queued_review row left open re-fires on every later memo run.
    if inputs.scorecard_seeds_consumed:
        safe_call(
            scorecard_context.mark_reviewed, inputs.ticker, snapshot_id,
            fallback=0, name="Scorecard Review", log_to=None,
        )
    # W7 — link this run's learned-priors audit rows to the snapshot, so
    # "why did the agent see this?" answers per memo (and the promotion
    # gate counts only linked renders). Live memos only: a backtest renders
    # nothing, and an audit failure never touches the saved memo.
    if inputs.as_of_date is None:
        from ..learning import context as learning_context
        safe_call(
            learning_context.link_run, inputs.run_id, snapshot_id,
            fallback=0, name="Learning link", log_to=None,
        )
    return memo


def _sync_degradation(memo: StockMemoOut, degradation: DegradationLog) -> None:
    """Copy the log onto the memo — both the names and the reasons.

    `degraded_agents` and `degradation_events` are two views of the same
    accumulator and must never disagree, so every refresh point goes
    through here rather than assigning one field and forgetting the other.
    """
    memo.degraded_agents = degradation.degraded_agents()
    memo.degradation_events = degradation.events()


def _absorb_failover_events(degradation: DegradationLog) -> None:
    """Move this context's LLM failover events onto the memo's degradation log."""
    for _ev in llm.consume_failover_events():
        degradation.record_soft(
            "LLM provider",
            f"failed over from {_ev['from']} to {_ev['to']}: {_ev['reason']}",
            kind="ProviderFailover",
        )


def _persist_memo_snapshot(memo: StockMemoOut, as_of_date: Any | None = None) -> MemoSnapshot:
    """Indirection so safe_call wraps DB I/O. Lazy-import keeps graph.py from
    pulling the ORM at module import time (it's already loaded via models).

    Wave 1C: backtest snapshots are persisted with `as_of_date` set so the
    default `latest_memo` lookup excludes them.

    Returns the saved `MemoSnapshot` (Phase 6) so the persist stage can key
    the `scorecard_disagreements` row on its id instead of re-querying
    `latest_memo` and hoping nothing landed in between.
    """
    from ..services import memo_store
    # latest_memo defaults to live snapshots only. For backtests we ask
    # for include_backtests so version chains stay continuous within the
    # same as-of date space.
    prior = memo_store.latest_memo(memo.ticker, include_backtests=as_of_date is not None)
    trigger = "first_run" if prior is None else "full_reanalysis"
    parent_version = prior.version if prior is not None else None
    return memo_store.save_memo(memo, trigger=trigger, parent_version=parent_version,
                                as_of_date=as_of_date)


# ---------------------------------------------------------------------------
# Agent trace helper
# ---------------------------------------------------------------------------

def default_agent_trace(intent: str) -> list[AgentTrace]:
    base = [
        AgentTrace(agent="PM Orchestrator", status="done", detail=f"Intent classified as {intent}."),
    ]
    if intent in ("single_stock_analysis", "stock_comparison"):
        base += [
            AgentTrace(agent="Sector Analyst", status="done", detail="Sector framework applied."),
            AgentTrace(agent="Earnings Analyst", status="done", detail="Latest transcript reviewed."),
            AgentTrace(agent="Filing Analyst", status="done", detail="10-K/10-Q analyzed."),
            AgentTrace(agent="Valuation Analyst", status="done", detail="DCF + multiples interpreted."),
            AgentTrace(agent="Comps Analyst", status="done", detail="Peer median + premium/discount."),
            AgentTrace(agent="Macro Analyst", status="done", detail="Macro mapping applied."),
            AgentTrace(agent="Risk Committee", status="done", detail="Critic reviewed and flagged challenges."),
        ]
    elif intent == "portfolio_construction":
        base += [
            AgentTrace(agent="Screener Agent", status="done", detail="Universe scored against scenario fit."),
            AgentTrace(agent="Portfolio Construction Agent", status="done", detail="Diversified weights enforced."),
            AgentTrace(agent="Risk Committee", status="done", detail="Concentration + risk reviewed."),
        ]
    elif intent == "thematic_screen":
        base += [AgentTrace(agent="Screener Agent", status="done", detail="Theme bias applied to PM scores.")]
    elif intent == "macro_question":
        base += [AgentTrace(agent="Macro Analyst", status="done", detail="Scenario template + snapshot.")]
    return base
