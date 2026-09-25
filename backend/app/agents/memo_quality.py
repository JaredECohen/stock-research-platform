"""Research-quality guards for company memos: owner decision 7(b) and 7(c).

7(b) — the rating must match the valuation. A Bullish (or Very Bullish)
final rating on an *overvalued* valuation verdict, or a Bearish one on an
*undervalued* verdict, ships only with a substantive reason the PM stated
and nobody independent rejected; otherwise the rating becomes Neutral.

That rule was vacuous until now: `ValuationVerdict.verdict` used to be
derived FROM the rating (`graph._verdict_word`, deleted with this module),
so "Bullish + overvalued" could not happen by construction. The verdict is
now an evidence read that never looks at the rating
(`valuation_evidence_verdict`).

7(c) — earned confidence. The PM's confidence is capped (never raised)
when the PM synthesis or core analyst sections were template-filled, when
no live critic reviewed the memo, when the transcript or the filing
review is missing, or when a divergence was accepted without independent
review (`earned_confidence`). Caps based on the number-to-source check
arrive with that check (S15) and are skipped while it has not run.

Everything here is pure: no DB, no LLM, no settings reads except where a
caller passes the value in. `graph.py` owns when these run; this module
owns what they decide, so each rule is testable on plain values.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..finance import scorecard_spec
from ..schemas import (
    ConfidenceAssessment,
    ConfidenceCap,
    CriticReview,
    RatingReconciliation,
    SectionAvailability,
    StockMemoOut,
    ValuationVerdict,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 7(b) evidence-verdict constants
#
# These define a VERDICT RULE, not the rating calibration. Owner decision
# 7(d) keeps the rating blend (`LLM_RATING_WEIGHT`, the label thresholds)
# untouched until a calibration proposal is approved; nothing here feeds
# the blend. The rule is centred (W3 P7): the absolute quant valuation
# factor is NOT a vote, because its fixed multiple thresholds read every
# mega cap as expensive; a universe percentile and a peer-relative premium
# are centred by construction. Re-tune from the stored `signals` once live
# memos accumulate, never to hit a target distribution.
# ---------------------------------------------------------------------------

METHOD = "centred-v1"
# fs-v1 valuation family, universe percentile. Every valuation feature is a
# yield with sign +1 (`scorecard_spec`: earnings/fcf/ebitda-ev/sales-ev
# yield), so a HIGH percentile is a CHEAP name. Pinned by a test.
FAMILY_MIN_COVERAGE = 0.6
FAMILY_CHEAP_PCT = 80.0
FAMILY_RICH_PCT = 20.0
# Comps EV/EBITDA premium versus the peer median (ratio: 0.15 = 15%).
COMPS_BAND = 0.15
# Sectors where the comps EV/EBITDA premium never votes: enterprise value is
# not meaningful for banks, insurers or REITs. Read from the scorecard's own
# `ebitda_ev_yield` exclusion so the two valuation reads cannot drift apart.
COMPS_EXCLUDED_SECTORS: frozenset[str] = frozenset(next(
    f.exclude_sectors for f in scorecard_spec.FEATURE_SPEC if f.name == "ebitda_ev_yield"))
# `comps.compute_comps` divides by |peer median|, so a negative multiple
# (negative EBITDA or EV) on either side yields a large, meaningless
# "discount" or "premium". Comps is the pivotal vote (the DCF may only agree
# with it), so a meaningless multiple must not vote. A discount of 100% or
# more is only reachable with a negative target multiple: the guard for
# stored bodies that carry the premium but not the multiples.
COMPS_MIN_PREMIUM = -1.0
# The INITIAL (consensus-anchored) DCF is a corroborating vote only: the
# raw DCF sign is biased negative across the covered names (median base
# upside -36% over the 25 live runs), so it counts only when it is large,
# not terminal-value clamped, and agrees with a non-zero comps vote. The
# PM-adjusted DCF is recorded and never votes: the PM who rates the name
# also moved that model, so it cannot check the rating.
DCF_VOTE_BAND = 0.40
# A directional verdict needs this many agreeing votes and no opposing one.
MIN_AGREEING_VOTES = 2

SIGNAL_FAMILY = "valuation_family"
SIGNAL_COMPS = "comps_ev_ebitda"
SIGNAL_DCF = "dcf_initial"
SIGNAL_DCF_FINAL = "dcf_pm_adjusted"
VOTING_SIGNALS: tuple[str, ...] = (SIGNAL_FAMILY, SIGNAL_COMPS, SIGNAL_DCF)

_SIGNAL_LABEL = {
    SIGNAL_FAMILY: "the scorecard valuation-family rank",
    SIGNAL_COMPS: "the EV/EBITDA premium to peers",
    SIGNAL_DCF: "the consensus DCF",
}

VERDICT_UNAVAILABLE_SUMMARY = "Valuation verdict unavailable for this run."


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _num(v: Any) -> float | None:
    """A finite float, or None. Stored JSON can carry strings or NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _family_display(pct: float) -> str:
    return f"{_ordinal(int(round(pct)))} percentile"


def _comps_display(prem: float) -> str:
    return f"{abs(prem):.0%} {'premium' if prem > 0 else 'discount'}"


def _dcf_display(u: float) -> str:
    return f"{u:+.0%}"


def valuation_evidence_verdict(
    *,
    family_pct: float | None,
    family_coverage: float | None,
    comps_premium: float | None,
    dcf_initial_upside: float | None,
    dcf_initial_tv_clamped: bool = False,
    dcf_final_upside: float | None = None,
    factor_valuation: float | None = None,
    comps_target_multiple: float | None = None,
    comps_peer_median_multiple: float | None = None,
    sector: str | None = None,
) -> ValuationVerdict:
    """The memo's valuation verdict from evidence alone. Never reads a rating.

    Votes (+1 cheap, -1 rich, 0 inside the band):
      * valuation family: universe percentile >= 80 -> +1, <= 20 -> -1,
        counted only at coverage >= 0.6;
      * comps: EV/EBITDA premium <= -15% -> +1, >= +15% -> -1; no vote
        (recorded with the reason) when either multiple is <= 0, the
        premium is <= -100%, or the sector is one where EV is not
        meaningful (`COMPS_EXCLUDED_SECTORS`);
      * initial DCF: +/-1 only when |upside| >= 40%, no terminal-value
        clamp, and the same sign as a non-zero comps vote.
    `overvalued` needs >= 2 rich votes and no cheap one; `undervalued` is the
    mirror; opposing votes are `mixed`; anything else is `fairly_priced`.
    Because the DCF only ever agrees with comps, no single signal can make
    a directional verdict. `dcf_final_upside` (PM-adjusted) and
    `factor_valuation` (absolute) are recorded for display, never voted.
    """
    signals: dict[str, Any] = {"method": METHOD}
    votes: dict[str, int] = {}
    available: list[str] = []
    parts: list[str] = []

    pct, cov = _num(family_pct), _num(family_coverage)
    if pct is not None:
        available.append(SIGNAL_FAMILY)
        entry: dict[str, Any] = {"percentile": pct, "coverage": cov, "display": _family_display(pct)}
        if cov is not None and cov >= FAMILY_MIN_COVERAGE:
            vote = 1 if pct >= FAMILY_CHEAP_PCT else (-1 if pct <= FAMILY_RICH_PCT else 0)
            votes[SIGNAL_FAMILY] = vote
            entry["vote"] = vote
            parts.append(f"valuation family at the {entry['display']} of the universe{_tag(vote)}")
        else:
            entry["vote"] = None
            cov_txt = "unknown" if cov is None else f"{cov:.0%}"
            parts.append(f"valuation family at the {entry['display']} (coverage {cov_txt}, "
                         f"below the {FAMILY_MIN_COVERAGE:.0%} floor; no vote)")
        signals[SIGNAL_FAMILY] = entry

    prem = _num(comps_premium)
    comps_vote = 0
    if prem is not None:
        available.append(SIGNAL_COMPS)
        no_vote = _comps_no_vote(prem, comps_target_multiple, comps_peer_median_multiple, sector)
        if no_vote:
            # No "display": the number is not a valuation read, so the PM
            # prompt must not list it as one (and nothing can quote it).
            signals[SIGNAL_COMPS] = {"premium": prem, "vote": None, "no_vote": no_vote}
            parts.append(f"EV/EBITDA vs peers not meaningful ({no_vote}; no vote)")
        else:
            comps_vote = 1 if prem <= -COMPS_BAND else (-1 if prem >= COMPS_BAND else 0)
            votes[SIGNAL_COMPS] = comps_vote
            signals[SIGNAL_COMPS] = {"premium": prem, "vote": comps_vote, "display": _comps_display(prem)}
            parts.append(f"EV/EBITDA {_comps_display(prem)} vs peers{_tag(comps_vote)}")

    u = _num(dcf_initial_upside)
    if u is not None:
        available.append(SIGNAL_DCF)
        clamped = bool(dcf_initial_tv_clamped)
        direction = 1 if u > 0 else (-1 if u < 0 else 0)
        dcf_vote = (
            direction
            if abs(u) >= DCF_VOTE_BAND and not clamped and comps_vote != 0 and direction == comps_vote
            else 0
        )
        votes[SIGNAL_DCF] = dcf_vote
        signals[SIGNAL_DCF] = {"upside": u, "tv_clamped": clamped, "vote": dcf_vote,
                               "display": _dcf_display(u)}
        why = ""
        if dcf_vote == 0 and abs(u) >= DCF_VOTE_BAND:
            why = ("; terminal value clamped, no vote" if clamped
                   else "; counts only with the peer multiple's agreement")
        parts.append(f"consensus DCF base case {_dcf_display(u)} to fair value{_tag(dcf_vote)}{why}")

    uf = _num(dcf_final_upside)
    if uf is not None:
        signals[SIGNAL_DCF_FINAL] = {"upside": uf, "vote": None, "display": _dcf_display(uf)}
        if u is None or abs(uf - u) > 1e-9:
            parts.append(f"PM-adjusted DCF {_dcf_display(uf)} (recorded, not a vote)")
    if u is None and uf is None:
        # An unpriced DCF is "n/a", never a 0% neutral signal.
        parts.append("DCF unavailable")

    rich = [k for k, v in votes.items() if v < 0]
    cheap = [k for k, v in votes.items() if v > 0]
    if rich and cheap:
        verdict = "mixed"
    elif len(rich) >= MIN_AGREEING_VOTES:
        verdict = "overvalued"
    elif len(cheap) >= MIN_AGREEING_VOTES:
        verdict = "undervalued"
    else:
        verdict = "fairly_priced"
    signals["votes"] = votes
    signals["available"] = available

    if not available:
        summary = "Net read: fairly priced — valuation evidence unavailable (DCF unavailable; no peer multiple; no scorecard rank)."
    elif verdict == "mixed":
        summary = (
            f"Net read: mixed — {_names(cheap)} point cheap while {_names(rich)} point rich; "
            f"the rating is reconciled separately ({'; '.join(parts)})."
        )
    else:
        summary = f"Net read: {verdict.replace('_', ' ')} ({'; '.join(parts)})."
    return ValuationVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        basis="evidence",
        signals=signals,
        dcf_base_upside=uf if uf is not None else u,
        comps_ev_ebitda_premium=prem,
        factor_valuation=_num(factor_valuation),
        summary=summary,
    )


def _comps_no_vote(prem: float, target: Any, median: Any, sector: str | None) -> str | None:
    """Why the comps premium cannot vote, or None when it can."""
    if scorecard_spec.normalize_sector(sector) in COMPS_EXCLUDED_SECTORS:
        return "enterprise value is not meaningful for banks, insurers or REITs"
    t, m = _num(target), _num(median)
    if (t is not None and t <= 0) or (m is not None and m <= 0) or prem <= COMPS_MIN_PREMIUM:
        return "negative EV/EBITDA multiple"
    return None


def _tag(vote: int) -> str:
    return ", cheap" if vote > 0 else (", rich" if vote < 0 else "")


def _names(keys: Iterable[str]) -> str:
    return " and ".join(_SIGNAL_LABEL.get(k, k) for k in keys)


def evidence_available(vv: ValuationVerdict | None) -> bool:
    """True when `vv` is an evidence verdict with at least one input."""
    if vv is None or vv.basis != "evidence":
        return False
    return bool((vv.signals or {}).get("available"))


def verdict_word(verdict: str | None) -> str | None:
    """The thesis word for an evidence verdict; None for `mixed`."""
    return {
        "undervalued": "undervalued", "overvalued": "overvalued",
        "fairly_priced": "fairly priced",
    }.get(verdict or "fairly_priced")


def rating_direction(rating: str | None) -> int:
    label = (rating or "").lower()
    return 1 if "bull" in label else (-1 if "bear" in label else 0)


def rating_word(rating: str | None) -> str:
    """What a rating says about price: Bullish reads undervalued, Bearish
    overvalued, anything else fairly priced."""
    return {1: "undervalued", -1: "overvalued"}.get(rating_direction(rating), "fairly priced")


def diverges(rating: str | None, verdict: str | None) -> bool:
    """Bullish/Very Bullish on `overvalued`, or Bearish/Very Bearish on
    `undervalued`. Neutral, `mixed` and `fairly_priced` never diverge."""
    d = rating_direction(rating)
    return (d > 0 and verdict == "overvalued") or (d < 0 and verdict == "undervalued")


def valuation_evidence_block(vv: ValuationVerdict | None) -> str:
    """The PM prompt's volatile "Valuation evidence" block (contract C7).

    "" when there is no evidence, so the assembled prompt is byte-identical
    to the pre-S14 one in that case. The per-signal values are listed at
    display precision because a divergence reason must quote the one it
    overrides at that precision (`assess_divergence_reason`)."""
    if not evidence_available(vv):
        return ""
    assert vv is not None
    word = verdict_word(vv.verdict) or "mixed"
    lines = [
        "## Valuation evidence (deterministic read of the scorecard valuation rank, "
        "the peer multiple and the consensus DCF; not a rating)",
        f"Verdict: {word}. {vv.summary}",
    ]
    quoted = []
    for key in VOTING_SIGNALS:
        entry = (vv.signals or {}).get(key)
        if isinstance(entry, dict) and entry.get("display"):
            quoted.append(f"{_SIGNAL_LABEL[key]} {entry['display']}")
    if quoted:
        lines.append("Values at display precision: " + "; ".join(quoted) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7(b) divergence reason and reconciliation
# ---------------------------------------------------------------------------

MIN_REASON_WORDS = 15
MIN_REASON_CHARS = 80

_SIGNAL_NAME_RE: dict[str, re.Pattern[str]] = {
    SIGNAL_FAMILY: re.compile(
        r"\b(?:valuation[- ]family|scorecard|percentile|fs-v1|valuation rank)\b", re.I),
    SIGNAL_COMPS: re.compile(
        r"\b(?:ev\s*/\s*ebitda|ebitda multiple|peer multiple|multiple|premium|discount|peers?|comps)\b",
        re.I),
    SIGNAL_DCF: re.compile(r"\b(?:dcf|discounted[- ]cash[- ]flow|intrinsic value|fair value)\b", re.I),
}
# A number as printed: optional sign, digits (comma groups), decimals, and a
# unit the value check reads ("%", "percent", an ordinal suffix, "percentile").
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-+−])?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?P<unit>\s*%|\s*percent\b|\s*pct\b|st\b|nd\b|rd\b|th\b|\s+percentile\b)?",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_PERCENT_UNITS = frozenset({"%", "percent", "pct"})
_RANK_UNITS = frozenset({"st", "nd", "rd", "th", "percentile"})
# A quoted number counts only this close (characters) to a mention of the
# signal it quotes: "10 years" at the far end of the reason is not the
# 10th-percentile rank.
QUOTE_WINDOW = 60
# The word right after a quoted magnitude that states its direction, as
# (word when the value is positive, word when negative).
_DIRECTION_WORDS: dict[str, tuple[str, str]] = {
    SIGNAL_COMPS: ("premium", "discount"),
    SIGNAL_DCF: ("upside", "downside"),
}
_DIRECTION_LOOKAHEAD = 30


def _quotes(text: str, expected: float, *, name_re: re.Pattern[str], percent_unit: bool,
            directions: tuple[str, str] | None = None) -> bool:
    """True when `text` prints `expected`, at the precision it prints it,
    within `QUOTE_WINDOW` characters of a mention of the signal (`name_re`).

    `percent_unit=True` (comps and DCF) needs a "%"/"percent" after the
    number and compares magnitudes, since prose often drops the sign ("a 44%
    premium"), but a printed sign must agree ("+54%" is not a -54% DCF) and
    so must a direction word right after it ("54% upside" is not a -54%
    DCF, "44% discount" is not a 44% premium). Otherwise (the percentile)
    the number must carry an ordinal or "percentile" unit: "10 years" is
    not the 10th percentile."""
    names = [m.span() for m in name_re.finditer(text)]
    for m in _NUMBER_RE.finditer(text):
        unit = (m["unit"] or "").strip().lower()
        if unit not in (_PERCENT_UNITS if percent_unit else _RANK_UNITS):
            continue
        digits = m["num"].replace(",", "")
        dec = (m["dec"] or "")[1:]
        places = min(len(dec), 2)
        printed = f"{digits}.{dec[:places]}" if places else digits
        if f"{abs(expected):.{places}f}" != printed:
            continue
        lo, hi = m.span()
        if not any(n_lo - QUOTE_WINDOW <= hi and lo <= n_hi + QUOTE_WINDOW for n_lo, n_hi in names):
            continue
        if percent_unit and expected != 0:
            if m["sign"] and (m["sign"] == "+") != (expected > 0):
                continue
            if directions is not None:
                pos_word, neg_word = directions
                ahead = re.search(rf"\b({pos_word}|{neg_word})\b",
                                  text[hi:hi + _DIRECTION_LOOKAHEAD], re.I)
                if ahead and ahead.group(1).lower() != directions[0 if expected > 0 else 1]:
                    continue
        return True
    return False


def _opposing_signals(verdict: ValuationVerdict, rating: str | None) -> list[str]:
    """Signals that actually voted against `rating`'s direction."""
    d = rating_direction(rating)
    votes = (verdict.signals or {}).get("votes") or {}
    return [k for k in VOTING_SIGNALS if d != 0 and isinstance(votes.get(k), int) and votes[k] == -d]


def assess_divergence_reason(reason: str, *, verdict: ValuationVerdict, rating: str | None) -> dict[str, bool]:
    """The self-contained checks on a PM's divergence reason.

    * `substantive`: not a placeholder (`is_real_falsifier`), at least 15
      words and 80 characters. "Quality deserves a premium" is not a reason.
    * `names_signal`: names a signal that voted against the rating.
    * `quotes_value`: quotes that same signal's value at display precision,
      near a mention of it, with the right unit and (for comps and DCF) no
      contradicting sign or direction word, checked against
      `verdict.signals` — a reason must engage with the number it
      overrides, not just its name (`_quotes`).
    Raises on malformed input; `reconcile_rating` turns that into a
    fail-closed downgrade."""
    from .industry_report_validator import is_real_falsifier

    text = " ".join(str(reason or "").split())
    substantive = (
        is_real_falsifier(text)
        and len(_WORD_RE.findall(text)) >= MIN_REASON_WORDS
        and len(text) >= MIN_REASON_CHARS
    )
    names = False
    quotes = False
    for key in _opposing_signals(verdict, rating):
        if not _SIGNAL_NAME_RE[key].search(text):
            continue
        names = True
        entry = verdict.signals[key]
        name_re = _SIGNAL_NAME_RE[key]
        if key == SIGNAL_FAMILY:
            ok = _quotes(text, float(entry["percentile"]), name_re=name_re, percent_unit=False)
        elif key == SIGNAL_COMPS:
            ok = _quotes(text, float(entry["premium"]) * 100.0, name_re=name_re, percent_unit=True,
                         directions=_DIRECTION_WORDS[key])
        else:
            ok = _quotes(text, float(entry["upside"]) * 100.0, name_re=name_re, percent_unit=True,
                         directions=_DIRECTION_WORDS[key])
        if ok:
            quotes = True
            break
    return {"substantive": bool(substantive), "names_signal": names, "quotes_value": quotes}


REASON_CHECKS: tuple[str, ...] = ("substantive", "names_signal", "quotes_value")


def reconcile_rating(
    *,
    blended_rating: str,
    verdict: ValuationVerdict,
    pm: RatingReconciliation | None,
    critic: CriticReview | None,
    enforce: bool = True,
) -> RatingReconciliation:
    """Rule 7(b) on the POST-BLEND rating. Pure; never raises.

    | condition                                             | outcome        | final     |
    | no valuation evidence                                 | not_applicable | unchanged |
    | not diverges(blended, verdict)                        | consistent     | unchanged |
    | diverges, all reason checks pass, critic not negative | accepted       | unchanged |
    | diverges, anything else (incl. a check crash)         | downgraded     | Neutral   |

    A critic assessment counts only from a live review. Without one the
    reason is accepted and labelled "not independently reviewed" (the
    confidence cap `divergence_unreviewed` applies). `enforce=False` is the
    `rating_reconciliation_mode=record` kill switch: the outcome is
    recorded but the published rating is left as blended.
    """
    base = pm or RatingReconciliation()
    reason = base.reason or ""
    out = RatingReconciliation(
        pm_rating=base.pm_rating or "", pm_confidence=base.pm_confidence,
        blended_rating=blended_rating, final_rating=blended_rating,
        valuation_verdict=verdict.verdict, reason=reason,
    )
    if not evidence_available(verdict):
        out.outcome = "not_applicable"
        out.note = "No valuation evidence was available, so the rating was not checked against it."
        return out
    if not diverges(blended_rating, verdict.verdict):
        out.outcome = "consistent"
        return out

    out.divergence = True
    try:
        checks = assess_divergence_reason(reason, verdict=verdict, rating=blended_rating)
    except Exception as exc:  # fail closed: an unverifiable reason is no reason
        log.warning("rating check: divergence reason could not be verified: %s", type(exc).__name__)
        checks = {k: False for k in REASON_CHECKS}
        checks["check_failed"] = True
    out.reason_checks = checks
    live = critic is not None and critic.review_mode == "live"
    out.critic_assessment = critic.valuation_divergence_assessment if (live and critic) else "not_assessed"
    passed = all(checks.get(k) for k in REASON_CHECKS)
    against = f"the valuation evidence reads {verdict.verdict.replace('_', ' ')}"
    if passed and out.critic_assessment != "unsupported":
        out.outcome = "accepted"
        review = ("critic: supported" if out.critic_assessment == "supported"
                  else "not independently reviewed")
        out.note = (f"Rated {blended_rating} although {against}. PM's reason: {reason} ({review}).")
        return out

    out.outcome = "downgraded"
    if not reason.strip():
        why = "no reason was given"
    elif checks.get("check_failed"):
        why = "the stated reason could not be verified"
    elif passed:
        why = "the live critic judged the stated reason unsupported"
    else:
        failed = ", ".join(k for k in REASON_CHECKS if not checks.get(k))
        why = f"the stated reason was not substantive enough (failed: {failed})"
    note = (f"Rating set to Neutral: the blended rating was {blended_rating} but {against}, "
            f"and {why}. {verdict.summary}")
    if enforce:
        out.final_rating = "Neutral"
    else:
        note = "Recorded only (rating_reconciliation_mode=record); not enforced. " + note
    out.note = note
    return out


def thesis_rewrite_word(
    *, stated: str | None, rating: str | None, verdict: ValuationVerdict,
    accepted: bool, anti_pattern: bool,
) -> tuple[bool, str | None]:
    """(rewrite?, word the rewrite states) for the thesis guard.

    Rewrites only on the anti-pattern, or when the stated verdict word
    contradicts BOTH the rating's direction and the evidence verdict — a
    thesis that agrees with either is a defensible reading. An accepted
    divergence is never rewritten for its word (the PM argued it), and a
    `mixed` verdict contradicts no word. The rewrite states the evidence
    word, except for an accepted divergence (the rating's word) or when no
    evidence exists (the rating's word, as before this rule)."""
    r_word = rating_word(rating)
    has_evidence = evidence_available(verdict)
    e_word = verdict_word(verdict.verdict) if has_evidence else None
    word: str | None
    if accepted or not has_evidence:
        word = r_word
    else:
        word = e_word
    if anti_pattern:
        return True, word
    if accepted or stated is None:
        return False, word
    contradicts_rating = stated != r_word
    contradicts_evidence = e_word is not None and stated != e_word
    if not has_evidence:
        # Legacy behaviour with no evidence at all: the rating decides.
        contradicts_evidence = True
    return (contradicts_rating and contradicts_evidence), word


# ---------------------------------------------------------------------------
# Patch path (news incremental patches never re-run the PM or the critic)
# ---------------------------------------------------------------------------

LAST_FULL_RUN_CAP = "last_full_run"


@dataclass
class PatchGuard:
    rating_downgraded: bool = False
    confidence_clamped: bool = False


def enforce_after_patch(
    prior: StockMemoOut, patched: StockMemoOut, fields_patched: Iterable[str], *,
    enforce: bool = True,
) -> tuple[StockMemoOut, PatchGuard]:
    """Re-apply 7(b) and the 7(c) ceiling to a news-patched memo.

    No-op when the memo carries no `quality` (every memo written before
    these guards): those keep today's behaviour exactly. Otherwise:
      * (b) a patched rating that diverges from the STORED evidence verdict
        carries no reason (a patch cannot state one), so it becomes Neutral —
        unless the last full run's divergence was accepted and the patch
        keeps its direction;
      * (c) confidence may move down but never above the last full run's
        earned value (anti-ratchet: ABBV went 44.6 -> 90 in five patches).
        The ceiling is recorded as a `last_full_run` cap the first time, so
        later patches in the chain keep reading the full run's value.
    """
    guard = PatchGuard()
    quality = patched.quality
    if quality is None:
        return patched, guard
    fields = set(fields_patched)
    out = patched.model_copy(deep=True)
    q = out.quality
    assert q is not None

    vv = out.valuation_verdict
    if "rating_label" in fields and vv.basis == "evidence" and diverges(out.rating_label, vv.verdict):
        prior_rec = prior.quality.rating_reconciliation if prior.quality else None
        keep = (
            prior_rec is not None and prior_rec.outcome == "accepted"
            and rating_direction(prior_rec.final_rating) == rating_direction(out.rating_label)
        )
        if not keep:
            patched_to = out.rating_label
            note = (f"A news update moved the rating to {patched_to} against the valuation evidence "
                    f"({vv.verdict.replace('_', ' ')}) without a valuation reason; rating set to Neutral.")
            if enforce:
                out.rating_label = "Neutral"  # type: ignore[assignment]
                guard.rating_downgraded = True
            else:
                note = "Recorded only (rating_reconciliation_mode=record); not enforced. " + note
            q.rating_reconciliation = RatingReconciliation(
                outcome="downgraded", pm_rating=patched_to, blended_rating=patched_to,
                final_rating=out.rating_label, valuation_verdict=vv.verdict, divergence=True,
                note=note,
            )

    conf = q.confidence
    if conf is not None:
        caps = list(conf.caps)
        ceiling_cap = next((c for c in caps if c.code == LAST_FULL_RUN_CAP), None)
        if ceiling_cap is None:
            ceiling_cap = ConfidenceCap(
                code=LAST_FULL_RUN_CAP, cap=float(conf.final),
                detail="A news update may lower confidence but never raise it above the last full run.",
            )
            caps.append(ceiling_cap)
        proposed = float(out.confidence_score)
        final = min(proposed, ceiling_cap.cap)
        guard.confidence_clamped = final < proposed
        q.confidence = ConfidenceAssessment(
            raw=proposed, final=final, caps=caps,
            binding=LAST_FULL_RUN_CAP if guard.confidence_clamped else None,
        )
        out.confidence_score = final
        if isinstance(out.scores, dict):
            out.scores = {**out.scores, "confidence": final}
    return out, guard


# ---------------------------------------------------------------------------
# 7(c) earned confidence
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 20.0
CAP_PM_TEMPLATE = 40.0
CAP_TEMPLATE_SECTIONS = {1: 65.0, 2: 55.0, 3: 45.0}   # 3 means ">= 3"
CAP_CRITIC_NOT_LIVE = 60.0
CAP_NO_TRANSCRIPT = 75.0
CAP_NO_FILING_REVIEW = 75.0
CAP_DIVERGENCE_UNREVIEWED = 55.0

# CORE analyst sections for `template_sections` (W2b §5, industry_group
# included per W2b over W4). Comps and risk are deterministic at round 0 by
# design, and macro/technical are context, not the thesis.
CORE_SECTIONS: tuple[str, ...] = (
    "sector_agent_view", "earnings_agent_view", "filing_agent_view",
    "valuation_agent_view", "extra_agent_views.industry_group",
)
# Availability reasons that mean "a template stood in for the analyst".
# `skipped_by_intake` and `no_source_data` are not templates: the missing
# filing / transcript caps cover those, and an unmapped industry group is
# simply not applicable.
_TEMPLATE_REASONS = frozenset({"template_fallback", "agent_failed"})
# The basis `memo_sections` records when it could not classify a section.
UNCLASSIFIED = "unclassified"


def template_filled(av: Mapping[str, SectionAvailability]) -> tuple[bool, list[str]]:
    """(PM synthesis template-filled?, CORE sections template-filled) from
    the W2a presenter's `compute_availability` map (contract C2).

    Raises when the PM view or a CORE section is `unclassified`: the
    presenter maps a classifier crash to "available" so a reader's page
    never 500s, but for the caps that would read "no template" and drop
    `pm_template` / `template_sections` silently. Raising lets the quality
    stage's fallback cap the memo and put "Memo Quality" on the banner."""
    unclassified = [
        key for key in ("final_pm_view", *CORE_SECTIONS)
        if (entry := av.get(key)) is not None and UNCLASSIFIED in (entry.basis or [])
    ]
    if unclassified:
        raise ValueError(f"section availability unclassified for: {', '.join(unclassified)}")
    pm = av.get("final_pm_view")
    pm_template = pm is not None and pm.status == "unavailable"
    sections = [
        key for key in CORE_SECTIONS
        if (entry := av.get(key)) is not None and entry.status == "unavailable"
        and entry.reason in _TEMPLATE_REASONS
    ]
    return pm_template, sections


def earned_confidence(
    *,
    raw: float,
    pm_template: bool,
    template_sections: list[str],
    critic_mode: str,
    transcript_given: bool,
    filing_reviewed: bool,
    divergence_unreviewed: bool,
    number_check: Any | None = None,
) -> ConfidenceAssessment:
    """Deterministic caps on the PM's confidence; the minimum binds.

    `final = min(raw, max(20, min(caps)))`: a cap can lower confidence to
    no less than 20, and nothing here ever raises `raw`. `number_check` is
    accepted and ignored until the number-to-source check lands (S15):
    number-based caps must not fire on a check that did not run."""
    caps: list[ConfidenceCap] = []
    if pm_template:
        caps.append(ConfidenceCap(code="pm_template", cap=CAP_PM_TEMPLATE,
                                  detail="The PM synthesis was template-filled."))
    k = len(template_sections)
    if k:
        caps.append(ConfidenceCap(
            code="template_sections", cap=CAP_TEMPLATE_SECTIONS[min(k, 3)],
            detail=f"{k} core analyst section(s) template-filled: {', '.join(template_sections)}.",
        ))
    if critic_mode != "live":
        caps.append(ConfidenceCap(code="critic_not_live", cap=CAP_CRITIC_NOT_LIVE,
                                  detail=f"No live critic review (review mode: {critic_mode})."))
    if not transcript_given:
        caps.append(ConfidenceCap(code="no_transcript", cap=CAP_NO_TRANSCRIPT,
                                  detail="No earnings-call transcript was available."))
    if not filing_reviewed:
        caps.append(ConfidenceCap(code="no_filing_review", cap=CAP_NO_FILING_REVIEW,
                                  detail="No filing was reviewed by the filing analyst."))
    if divergence_unreviewed:
        caps.append(ConfidenceCap(
            code="divergence_unreviewed", cap=CAP_DIVERGENCE_UNREVIEWED,
            detail="The rating diverges from the valuation evidence on a reason no live critic reviewed.",
        ))
    raw_f = float(raw)
    if not caps:
        return ConfidenceAssessment(raw=raw_f, final=raw_f, caps=[], binding=None)
    lowest = min(caps, key=lambda c: c.cap)   # first in table order on a tie
    final = min(raw_f, max(CONFIDENCE_FLOOR, lowest.cap))
    return ConfidenceAssessment(
        raw=raw_f, final=final, caps=caps,
        binding=lowest.code if final < raw_f else None,
    )
