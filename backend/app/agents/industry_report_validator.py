"""FEAT-003 — the gate an Industry Analysis report passes before it is saved.

The report payload has two layers: ``facts`` the server computed and an
``interpretation`` an analyst (LLM or deterministic template) wrote on top.
The validator enforces the research-process contract between them:

* every section is present, in the frozen order, and ``facts`` are exactly
  what the server handed the writer (an LLM never writes into facts);
* every number the interpretation quotes is present in the facts at the
  precision shown (rounding tolerance), so a report cannot invent a
  return, a multiple or a sample size;
* every sentence that asserts a cause is registered as a
  ``causal_inference`` claim with a basis and a falsifier — the
  methodology's "required for each link" made mechanical;
* the drivers section follows the eight-stage causal order loaded from the
  knowledge base and does not open with a KPI forecast (the handbook's
  named anti-pattern);
* no advice phrasing, and the disclaimer is present;
* no licensed-taxonomy brand, code or industry/sub-industry registry name
  in anything the page prints (rule L1, owner decision 2026-09-24);
* the outlook's forward numbers are REGISTERED forecast assumptions
  (owner decision 1, 2026-09-24; rules F1-F10 below): each is declared
  once as a ``forecast_assumption`` claim with an id, a rate or multiple,
  a horizon, an anchor that resolves to an observed fact of the same unit
  family, and a falsifier. The outlook is checked against ITS OWN facts
  (which carry the anchors catalogue) and those declarations — never the
  whole pack, where a sector code "45" licensed "45x" and a breadth of
  0.75 licensed a "75%" bull case nobody declared. Grounding is not
  exempted: a forward number is still traceable, to the observation it
  departs from.

``validate`` returns a list of error strings; empty means the payload may
be published. A rejected payload fails the job, which retries and, on the
final attempt, runs the deterministic writer so a week is never blank.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..services.industry_group_knowledge import thesis_stages

SECTION_ORDER: tuple[str, ...] = (
    "overview", "drivers", "kpis", "performance", "companies", "statistics",
    "themes", "cross_industry", "outlook", "risks", "what_changed", "sources", "metadata",
)
# Sections that carry an analyst interpretation; the rest are facts-only.
INTERPRETED_SECTIONS: tuple[str, ...] = tuple(
    s for s in SECTION_ORDER if s not in ("statistics", "sources", "metadata")
)
CLAIM_TYPES: tuple[str, ...] = ("observed_fact", "causal_inference", "forecast_assumption")

FORBIDDEN_PHRASES: tuple[str, ...] = (
    "you should buy", "you should sell", "you should hold", "you should invest",
    "we recommend", "our recommendation", "recommend buying", "recommend selling",
    "buy now", "sell now", "strong buy", "strong sell", "buy rating", "sell rating",
    "price target", "must buy", "must sell", "you must", "you need to buy", "you need to sell",
    "guaranteed return", "cannot lose",
)
CAUSAL_MARKERS: tuple[str, ...] = (
    "because", "leads to", "lead to", "led to", "drives", "driven by", "as a result",
    "due to", "causes", "caused by", "therefore", "results in", "resulting in",
    "which means", "so that", "translates into", "feeds through",
)
# The anti-pattern: a drivers section that opens on a KPI forecast rather
# than naming the world change. Applied to the first sentence only.
_KPI_FORECAST_OPENERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(we\s+)?(forecast|expect|project|model|estimate|see|target)\b", re.I),
    re.compile(r"^[^.]{0,80}\b(will|should|to|could)\s+(grow|rise|reach|expand|compress|hit|increase|"
               r"decline|accelerate|decelerate|climb|fall)\b[^.]*\d", re.I),
    re.compile(r"^\s*(kpi|kpis|eps|revenue|arr|margins?|growth|earnings)\b[^.]{0,80}\b(forecast|target|"
               r"estimate|guide|guidance)\b", re.I),
    re.compile(r"^[^.]{0,40}\b(forecast|target price|price target|eps estimate)\b", re.I),
)

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")
_ISO_WEEK_RE = re.compile(r"\b\d{4}-W\d{2}\b")
_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:[-\w.]*)?\b|\bv\d+(?:\.\d+)*\b", re.I)
# Comma groups count as thousands separators only in 3-digit runs, so a
# provenance list like "451020,451030" reads as two codes, not one number.
_NUMBER_RE = re.compile(
    r"(?<![\w./-])[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?"
    r"(?:%|x|bps|pp|bp|pts?|bn|mm|m|k|tn|t|b)?(?![\w/])", re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A ratio of ±1000% is already an extreme; beyond it a fact is a rendered
# figure (a multiple, a price, a market cap), not something to scale.
_RATIO_CEILING = 10.0

# --- registered forecast assumptions (owner decision 1, 2026-09-24) -------------
#
# A forward number cannot be in a backward-looking facts pack, so the
# outlook may state one only as a DECLARED assumption: a rate or a multiple,
# with a horizon, anchored to an observed fact of the same unit family and
# quoted next to it, so the size of the departure is visible. Exogenous
# levels (commodity prices, policy rates, currency amounts, counts) have no
# anchor and are not admissible; they must be stated qualitatively.
FORWARD_SECTIONS: tuple[str, ...] = ("outlook",)
MAX_FORECAST_ASSUMPTIONS = 12
_FA_ID_RE = re.compile(r"^FA(?:[1-9]|1[0-2])$")
# F4 is a WHOLE-FIELD grammar, not a search: the horizon is printed verbatim
# in the assumptions table, so anything a search let through beside the
# horizon phrase ("next 4 quarters, oil 95", "... ; strong buy", "... for
# GICS group 4530") would reach the page unchecked. A count is capped at
# 12 — the prose rule's count exemption — so a horizon the field accepts is
# one the assumption's own text can quote ("the next 24 months" would pass
# here and then fail as an unregistered "24" in the prose); longer horizons
# are stated in years or as a fiscal year.
_HORIZON_RE = re.compile(
    r"(?:(?:over|within|through|by|in|to|for)\s+)?(?:the\s+)?"
    r"(?:next\s+(?:(?:[1-9]|1[0-2])\s+)?(?:quarters?|years?|months?|weeks?)"
    r"|(?:[1-9]|1[0-2])\s+(?:quarters?|years?|months?|weeks?)"
    r"|(?:end\s+of\s+)?(?:FY\s?(?:\d{2}|20\d{2})|[HQ][1-4]\s?(?:FY\s?)?(?:\d{2}|20\d{2})|20\d{2}))",
    re.I,
)
_MAX_HORIZON_CHARS = 60
RATE_UNITS: tuple[str, ...] = ("%", "bps", "bp", "pp")
MULTIPLE_UNITS: tuple[str, ...] = ("x",)
# Where an anchor may live, per unit family. Order matters: it is the
# catalogue's order, and the catalogue is what the writer shows the model.
ANCHOR_PREFIXES: dict[str, tuple[str, ...]] = {
    "rate": ("performance.returns.", "performance.benchmark_relative.", "statistics.breadth.",
             "statistics.dispersion.", "statistics.fundamentals."),
    "multiple": ("statistics.valuation.", "outlook.expectations_ledger.price_implied.value."),
}
# The catalogue is capped so the outlook call's facts stay small. Headline
# leaves are listed before the quantiles so a cap never drops a whole family
# (the valuation multiples come last in prefix order).
MAX_ANCHORS = 40
_HEADLINE_LEAVES = frozenset({"median", "equal_weight", "value", "pct_positive", "share", "stdev"})
# The leaf names a rate or a multiple actually lives at under the anchor
# prefixes. A WHITELIST, not a count blacklist: those subtrees also carry
# bookkeeping keyed by something other than a measure — breadth's
# `excluded_by_reason` maps a reason code ("window_too_short") to a number
# of constituents — and a count reached through a key that names no count
# would otherwise be offered to the model as a rate, anchor an assumption,
# and print on the page as "200.00%".
_MEASURE_LEAVES = _HEADLINE_LEAVES | {"market_cap_weight", "iqr", "range", "p25", "p75"}
# A leaf naming a count is a sample size or a window, never a rate: `n`,
# `n_mcw`, `benchmark_n`, `window_sessions`, `n_days`. Same token rule as
# the UI's `unitFor` (frontend/src/components/industries/format.ts).
_COUNT_LEAF_TOKENS = frozenset({"n", "count", "sessions", "days"})
# The outlook's anchors catalogue and its truncation count are the writer's
# bookkeeping. The catalogue's values are copies of other sections' facts
# and the count is a number nobody observed, so neither may license a
# number anywhere — the outlook's own set and the whole-pack set both
# leave them out.
_CATALOGUE_KEYS = frozenset({"anchors", "anchors_truncated"})
FORECAST_POLICY = (
    "Forward numbers are registered assumptions anchored to an observed fact; they are "
    "labelled as assumptions, not forecasts of record."
)


# --- helpers -------------------------------------------------------------------


def _strings(obj: Any) -> list[str]:
    """Every string reachable inside `obj` (dict/list/scalars)."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_strings(v))
    return out


def _numbers(obj: Any) -> set[float]:
    """Every numeric value reachable inside `obj`, plus numeric tokens found
    in its strings (mandate prose carries "5/5", "≥200bps", …)."""
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for tok in numeric_tokens(obj):
            out.add(tok[1])
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out |= _numbers(k) | _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers(v)
    return out


def numeric_tokens(text: str) -> list[tuple[str, float, int]]:
    """`(raw token, value, decimals)` for every number in `text`, skipping
    dates, ISO weeks and version strings (identifiers, not measurements)."""
    scrubbed = _DATE_RE.sub(" ", text)
    scrubbed = _ISO_WEEK_RE.sub(" ", scrubbed)
    scrubbed = _VERSION_RE.sub(" ", scrubbed)
    out: list[tuple[str, float, int]] = []
    for m in _NUMBER_RE.finditer(scrubbed):
        raw = m.group(0).strip()
        core = re.sub(r"[^\d.+-]", "", raw.replace(",", ""))
        core = re.sub(r"(?<=\d)\.$", "", core)
        if not core or core in ("+", "-", "."):
            continue
        try:
            value = float(core)
        except ValueError:
            continue
        decimals = len(core.split(".")[1]) if "." in core else 0
        out.append((raw, value, decimals))
    return out


def _is_exempt(raw: str) -> bool:
    """Bare counts and ordinals ("2 quarters", "stage 3") and bare 4-digit
    years are identifiers, not measurements.

    The exemption is decided on the RAW token, never the parsed value: a
    unit suffix, a sign or a currency mark turns the same digits into a
    measurement the facts must carry. "8 quarters" is a count; "8%",
    "10x", "12 bps" and "$5" are claims. `str.isdigit()` is the whole
    test — it rejects a suffix, a sign, a decimal point and a thousands
    comma in one go — so a bare token is always integral and only a
    4-digit one can be a year.
    """
    if not raw.isdigit():
        return False
    v = int(raw)
    if 0 <= v <= 12:
        return True
    return len(raw) == 4 and 1900 <= v <= 2100


def _percent_form(a: float) -> float | None:
    """`a` rendered as a percentage, when `a` is plausibly a RATIO.

    Returns and breadth live in the facts as decimals (-0.031), and the
    writer renders them as "-3.1%", so the gate has to accept the scaled
    form. It must not accept it for every fact: the facts always carry
    small counts — ``n_constituents``, ``research_priority``, the number
    of industries — and scaling those by 100 manufactured support for
    "Revenue grew 1700%" out of ``n_constituents == 17``.

    A ratio is either inside [-1, 1] or fractional (a single name up 180%
    is stored as 1.8); a bare integer above 1 is a count, never a ratio.
    ``_RATIO_CEILING`` stops a large non-integer fact (a median EV/EBITDA
    of 26.5) from licensing "2650".
    """
    if a != a or a in (float("inf"), float("-inf")):  # NaN / inf are not ratios
        return None
    if abs(a) <= 1.0 or (a != int(a) and abs(a) <= _RATIO_CEILING):
        return a * 100.0
    return None


def _supported(raw: str, value: float, decimals: int, allowed: set[float]) -> bool:
    if _is_exempt(raw):
        return True
    return _matches(value, decimals, allowed)


def _matches(value: float, decimals: int, allowed: set[float]) -> bool:
    """`value`, shown at `decimals`, is one of `allowed` (or its percent
    form), whatever unit the token carries. No count/year exemption; the
    caller decides that. An anchor quoted as an observation goes through
    `states_observed` instead, which does look at the unit."""
    tol = 0.5 * (10 ** -decimals) + 1e-9
    for a in allowed:
        if abs(a - value) <= tol:
            return True
        percent = _percent_form(a)
        if percent is not None and abs(percent - value) <= tol:
            return True
    return False


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


# A falsifier is a statement of the observation that would break the claim.
# These openings are the strings that occupy the field without being one:
# "n/a", "none", "tbd" — and, just as empty, "n/a: no dated falsifier in
# this edition", which explains the ABSENCE of a falsifier rather than
# supplying one. A reason belongs where a missing value is reported; it is
# not a falsifier, and the gate must not read it as one.
_PLACEHOLDER_FALSIFIER_RE = re.compile(
    r"^(n\s*/?\s*a|none|nil|null|tbd|tba|todo|unknown|pending|"
    r"not\s+applicable|not\s+available|no\s+falsifier|see\s+above|same\s+as\s+above)\b",
    re.I,
)
# A usefully specific falsifier names an observation someone could go and
# check, which does not fit in a handful of characters. The floors are
# deliberately low — the gate's job is to reject a field that was filled in
# to get past it, not to grade prose.
_MIN_FALSIFIER_CHARS = 24
_MIN_FALSIFIER_WORDS = 4
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")


def is_real_falsifier(value: Any) -> bool:
    """True when `value` states what observation would disprove the claim.

    The "every causal link carries a falsifier" gate used to be satisfied
    by ANY non-empty string, so the literal "n/a" passed it and the gate
    proved nothing. Three things disqualify a string: it opens like a
    placeholder, it is too short to name an observation, or it is blank
    once stripped. Everything else is taken at face value — the gate
    cannot judge whether a stated observation is the RIGHT one, and
    pretending otherwise would be a worse lie than the one it replaces.
    """
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return False
    if _PLACEHOLDER_FALSIFIER_RE.match(text):
        return False
    return len(text) >= _MIN_FALSIFIER_CHARS and len(_WORD_RE.findall(text)) >= _MIN_FALSIFIER_WORDS


def _has_causal_marker(sentence: str) -> bool:
    low = f" {sentence.lower()} "
    return any(f" {m} " in low or f" {m}," in low for m in CAUSAL_MARKERS)


def _claims_of(interp: dict[str, Any]) -> list[dict[str, Any]]:
    raw = interp.get("claims")
    return [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []


def _claim_supports(sentence: str, claims: list[dict[str, Any]]) -> bool:
    target = _norm(sentence)
    for claim in claims:
        if claim.get("type") != "causal_inference":
            continue
        if not claim.get("basis") or not is_real_falsifier(claim.get("falsifier")):
            continue
        text = _norm(str(claim.get("text") or ""))
        if not text:
            continue
        if text == target or text in target or target in text:
            return True
    return False


def _interpretation_texts(interp: dict[str, Any]) -> list[str]:
    """Free text the reader sees: `text`, stage texts, scenario texts,
    claim texts. Basis references are not prose."""
    texts: list[str] = []
    if isinstance(interp.get("text"), str):
        texts.append(interp["text"])
    for stage in interp.get("stages") or []:
        if isinstance(stage, dict) and isinstance(stage.get("text"), str):
            texts.append(stage["text"])
    scenarios = interp.get("scenarios")
    if isinstance(scenarios, dict):
        for sc in scenarios.values():
            if isinstance(sc, dict):
                if isinstance(sc.get("text"), str):
                    texts.append(sc["text"])
                texts.extend(str(f) for f in (sc.get("falsifiers") or []))
    for claim in _claims_of(interp):
        if isinstance(claim.get("text"), str):
            texts.append(claim["text"])
        if isinstance(claim.get("falsifier"), str):
            texts.append(claim["falsifier"])
    return texts


def opens_with_kpi_forecast(text: str) -> bool:
    first = _sentences(text)[:1]
    if not first:
        return False
    return any(p.search(first[0]) for p in _KPI_FORECAST_OPENERS)


# --- forecast assumptions ---------------------------------------------------------


class _Missing:
    """Sentinel for a fact path that names nothing (a fact may be `None`)."""

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return "<missing>"


_MISSING: Any = _Missing()


def resolve_fact_path(facts: Any, path: Any) -> Any:
    """The value at dot `path` in the per-section facts, or `_MISSING`.

    Integer segments index lists. A dict key may itself contain dots — the
    factor benchmark is keyed ``KFR.MKT_RF.D`` — so at each dict the
    longest run of remaining segments that is a key wins, backtracking if
    it leads nowhere.
    """
    if not isinstance(path, str) or not path.strip():
        return _MISSING
    return _resolve(facts, path.strip().split("."))


def _resolve(node: Any, segs: list[str]) -> Any:
    if not segs:
        return node
    if isinstance(node, dict):
        for j in range(len(segs), 0, -1):
            key = ".".join(segs[:j])
            if key in node:
                found = _resolve(node[key], segs[j:])
                if found is not _MISSING:
                    return found
        return _MISSING
    if isinstance(node, list) and segs[0].isdigit() and int(segs[0]) < len(node):
        return _resolve(node[int(segs[0])], segs[1:])
    return _MISSING


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _is_count_leaf(key: str) -> bool:
    return bool(set(str(key).lower().split("_")) & _COUNT_LEAF_TOKENS)


def _numeric_leaves(node: Any, path: str) -> list[tuple[str, str, Any]]:
    """`(path, leaf key, value)` for every finite number under a dict tree."""
    out: list[tuple[str, str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if _is_number(value):
                out.append((f"{path}.{key}", str(key), value))
            elif isinstance(value, dict):
                out.extend(_numeric_leaves(value, f"{path}.{key}"))
    return out


def anchor_catalog(facts: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The observed facts a forecast assumption may be anchored to, and how
    many the cap dropped.

    Every finite numeric leaf under ``ANCHOR_PREFIXES`` except counts, as
    ``{path, value, family}``. The writer stores it in the outlook's facts
    (so the model sees exactly what it may anchor to) and the validator
    recomputes it from the server's facts, so the two cannot disagree: the
    stored copy is covered by the facts-mutation guard like every other
    fact. A path outside the catalogue is not an anchor, even if it
    resolves — the model was never shown it, and the page could not print
    its value without re-resolving paths.
    """
    rows: list[tuple[bool, dict[str, Any]]] = []
    for family, prefixes in ANCHOR_PREFIXES.items():
        for prefix in prefixes:
            root = prefix.rstrip(".")
            node = resolve_fact_path(facts, root)
            if node is _MISSING:
                continue
            for path, leaf, value in _numeric_leaves(node, root):
                if leaf not in _MEASURE_LEAVES or _is_count_leaf(leaf):
                    continue
                rows.append((leaf in _HEADLINE_LEAVES, {"path": path, "value": value, "family": family}))
    ordered = [r for headline, r in rows if headline] + [r for headline, r in rows if not headline]
    return ordered[:MAX_ANCHORS], max(0, len(ordered) - MAX_ANCHORS)


def _token_family(raw: str) -> str | None:
    """`rate`, `multiple`, `other` (a currency or a size unit) or `None`
    for a bare number."""
    s = raw.strip().lower()
    if "$" in s:
        return "other"
    m = re.search(r"(bps|bp|pp|%|x|pts?|bn|mm|m|k|tn|t|b)$", s)
    if not m:
        return None
    unit = m.group(1)
    if unit in RATE_UNITS:
        return "rate"
    if unit in MULTIPLE_UNITS:
        return "multiple"
    return "other"


def unit_family(raw: Any) -> str | None:
    """`rate` or `multiple` for a single unit-suffixed figure, else None."""
    fam = _token_family(str(raw or ""))
    return fam if fam in ("rate", "multiple") else None


def _canonical(raw: str, value: float, decimals: int) -> tuple[float, float]:
    """`(value, tolerance)` in the family's common unit: basis points are
    hundredths of a percentage point, so "150 bps" and "1.5%" compare."""
    factor = 0.01 if re.search(r"bps?$", raw.strip().lower()) else 1.0
    return value * factor, (0.5 * (10 ** -decimals)) * factor + 1e-9


def _single_measure(raw: Any) -> tuple[str, float, str] | None:
    """`(raw, canonical value, family)` when `raw` is exactly one rate or
    multiple (``"21%"``, ``"-150 bps"``, ``"18x"``) and nothing else."""
    if not isinstance(raw, str):
        return None
    toks = numeric_tokens(raw)
    if len(toks) != 1 or toks[0][0] != raw.strip():
        return None
    fam = unit_family(toks[0][0])
    if fam is None:
        return None
    value, _ = _canonical(*toks[0])
    return toks[0][0], value, fam


@dataclass(frozen=True)
class ForecastAssumption:
    """One declared, valid forecast assumption (all of F2-F7 passed)."""

    id: str
    value: float          # canonical: percentage points for a rate, times for a multiple
    family: str
    anchor: str
    anchor_value: float
    bounds: tuple[float, float] | None = None

    def states_value(self, raw: str, value: float, decimals: int) -> bool:
        """`raw` is this assumption's value at the precision `raw` shows."""
        if _token_family(raw) != self.family:
            return False
        canon, tol = _canonical(raw, value, decimals)
        return abs(canon - self.value) <= tol

    def states_anchor(self, raw: str, value: float, decimals: int) -> bool:
        """`raw` is the anchor's observed value, in the unit `raw` shows."""
        return states_observed(raw, value, decimals, self.anchor_value, self.family)

    def states_bound(self, raw: str, value: float, decimals: int) -> bool:
        if self.bounds is None or _token_family(raw) != self.family:
            return False
        canon, tol = _canonical(raw, value, decimals)
        return any(abs(canon - b) <= tol for b in self.bounds)


def states_observed(raw: str, value: float, decimals: int, observed: float, family: str) -> bool:
    """`raw` states the observed fact `observed` (of unit `family`) at the
    precision it shows, IN THE UNIT IT SHOWS.

    A rate is stored as a decimal and shown as a percentage, so "%" and
    "pp" compare with ``observed * 100`` and "bps" with ``observed *
    10000`` — never with the raw decimal, which let "0.2%" pass for an
    observed 23.4%. A multiple is shown at face value and never scaled, so
    "520x" cannot pass for 5.2x. A bare figure is read in the family's
    displayed unit; a bare count or year ("the next 4 quarters") is an
    identifier, not a statement of any observation.
    """
    fam = _token_family(raw)
    if fam not in (None, family) or _is_exempt(raw):
        return False
    if family == "multiple":
        shown = observed
    elif re.search(r"bps?$", raw.strip().lower()):
        shown = observed * 10000.0
    else:
        shown = observed * 100.0
    return abs(value - shown) <= 0.5 * (10 ** -decimals) + 1e-9


def _horizon_ok(horizon: Any) -> bool:
    """An explicit horizon and nothing else: the field is printed beside
    the assumption, so it must match the horizon grammar whole."""
    if not isinstance(horizon, str) or not horizon.strip() or len(horizon) > _MAX_HORIZON_CHARS:
        return False
    return _HORIZON_RE.fullmatch(" ".join(horizon.split()).rstrip(".")) is not None


def _bounds_of(raw: Any, measure: tuple[str, float, str] | None) -> tuple[float, float] | None:
    """`[lo, hi]` in the value's unit family, around the value."""
    if measure is None or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    parsed = [_single_measure(b) for b in raw]
    if any(p is None or p[2] != measure[2] for p in parsed):
        return None
    lo, hi = parsed[0][1], parsed[1][1]  # type: ignore[index]
    if not lo < hi or not lo <= measure[1] <= hi:
        return None
    return lo, hi


def _forecast_assumptions(name: str, interp: dict[str, Any], facts: dict[str, Any],
                          anchors: list[dict[str, Any]],
                          errors: list[str]) -> tuple[set[str], dict[str, ForecastAssumption]]:
    """F2-F7 over the section's `forecast_assumption` claims. Returns the
    ids declared (for F9/F10) and the assumptions that passed every rule
    (the only ones that may support a number)."""
    catalogue = {a["path"]: a for a in anchors}
    fa_claims = [c for c in _claims_of(interp) if c.get("type") == "forecast_assumption"]
    if len(fa_claims) > MAX_FORECAST_ASSUMPTIONS:
        errors.append(f"{name}: {len(fa_claims)} forecast assumptions declared; at most "
                      f"{MAX_FORECAST_ASSUMPTIONS} are accepted")
    seen: set[str] = set()
    declared: set[str] = set()
    valid: dict[str, ForecastAssumption] = {}
    for claim in fa_claims:
        fid = claim.get("id")
        if not isinstance(fid, str) or not _FA_ID_RE.match(fid) or fid in seen:
            errors.append(f"{name}: forecast assumption id {fid!r} is missing, malformed or repeated")
            valid.pop(str(fid), None)
            continue
        seen.add(fid)
        declared.add(fid)
        ok = True
        measure = _single_measure(claim.get("value"))  # F3
        if measure is None:
            errors.append(f"{name}: {fid} value {claim.get('value')!r} is not a single rate or multiple")
            ok = False
        if not _horizon_ok(claim.get("horizon")):  # F4
            errors.append(f"{name}: {fid} has no explicit horizon")
            ok = False
        anchor = claim.get("anchor")  # F5
        entry = catalogue.get(anchor) if isinstance(anchor, str) else None
        resolved = resolve_fact_path(facts, anchor)
        family = measure[2] if measure else None
        if (entry is None or (family is not None and entry["family"] != family)
                or not _is_number(resolved) or resolved != entry["value"]):
            errors.append(f"{name}: {fid} anchor {anchor!r} is not an observed {family or 'rate or multiple'} "
                          "in this edition's facts")
            ok = False
            entry = None
        bounds = None
        if claim.get("bounds") is not None:
            bounds = _bounds_of(claim.get("bounds"), measure)
            if bounds is None:
                errors.append(f"{name}: {fid} bounds must be [lo, hi] in its value's unit family, "
                              "around its value")
                ok = False
        fa = (ForecastAssumption(fid, measure[1], measure[2], str(anchor), float(entry["value"]), bounds)
              if measure is not None and entry is not None else None)
        if fa is not None:  # F6
            toks = numeric_tokens(str(claim.get("text") or ""))
            if not (any(fa.states_value(*t) for t in toks) and any(fa.states_anchor(*t) for t in toks)):
                errors.append(f"{name}: {fid} text must state its value and its anchor's observed value")
                ok = False
        if not is_real_falsifier(claim.get("falsifier")):  # F7
            errors.append(f"{name}: {fid} has no usable falsifier")
            ok = False
        if ok and fa is not None:
            valid[fid] = fa
    return declared, valid


def _basis_resolves(basis: Any, facts: dict[str, Any], declared: set[str]) -> bool:
    """F9: a causal basis names something checkable — a fact path below a
    section, a mandate item, or a declared assumption."""
    for ref in basis if isinstance(basis, list) else []:
        r = str(ref).strip()
        if r.startswith("mandate:") and len(r) > len("mandate:"):
            return True
        if r in declared:
            return True
        if "." in r and resolve_fact_path(facts, r) is not _MISSING:
            return True
    return False


def _scenario_ids(sc: dict[str, Any]) -> list[Any]:
    ids = sc.get("assumption_ids")
    return list(ids) if isinstance(ids, list) else []


def _forward_errors(name: str, interp: dict[str, Any], facts: dict[str, Any],
                    errors: list[str]) -> None:
    """F1-F10 for a forward section. Its numbers are checked against the
    section's OWN facts (the anchors catalogue included) and its valid
    assumptions — never the whole pack — and a scenario's numbers only
    against the assumptions that scenario lists."""
    anchors = anchor_catalog(facts)[0]
    declared, fas = _forecast_assumptions(name, interp, facts, anchors, errors)
    claims = _claims_of(interp)

    for claim in claims:  # F9
        if claim.get("type") == "causal_inference" and not _basis_resolves(claim.get("basis"), facts, declared):
            errors.append(f"{name}: causal claim basis names nothing in the facts, the mandate or a "
                          "registered assumption")

    scenarios = interp.get("scenarios")
    scenarios = scenarios if isinstance(scenarios, dict) else {}
    for key, sc in scenarios.items():  # F10
        if not isinstance(sc, dict):
            continue
        if sc.get("assumption_ids") is not None and not isinstance(sc.get("assumption_ids"), list):
            errors.append(f"{name}: scenario {key} assumption_ids is not a list")
        for fid in _scenario_ids(sc):
            if not isinstance(fid, str) or fid not in declared:
                errors.append(f"{name}: scenario {key} references undeclared assumption {fid}")

    section_facts = facts.get(name)
    own_facts: dict[str, Any] = section_facts if isinstance(section_facts, dict) else {}
    own = _numbers({k: v for k, v in own_facts.items() if k not in _CATALOGUE_KEYS})

    def general(text: str) -> None:  # F8
        for raw, value, decimals in numeric_tokens(text):
            # An anchor is quoted in its own unit (states_observed), not by
            # the unit-agnostic rule the section's other facts get.
            if _supported(raw, value, decimals, own) or any(
                    states_observed(raw, value, decimals, float(a["value"]), a["family"]) for a in anchors) or any(
                    fa.states_value(raw, value, decimals) for fa in fas.values()):
                continue
            errors.append(f"{name}: number {raw!r} is not in the facts or a registered assumption")

    if isinstance(interp.get("text"), str):
        general(interp["text"])
    for stage in interp.get("stages") or []:
        if isinstance(stage, dict) and isinstance(stage.get("text"), str):
            general(stage["text"])
    for claim in claims:
        if isinstance(claim.get("text"), str):
            general(claim["text"])
        falsifier = claim.get("falsifier")
        if not isinstance(falsifier, str):
            continue
        if claim.get("type") != "forecast_assumption":
            general(falsifier)
            continue
        fa = fas.get(str(claim.get("id")))
        # An assumption's falsifier may name only its own registered
        # numbers: its bounds, its value, its anchor's observed value.
        for raw, value, decimals in numeric_tokens(falsifier):
            if _is_exempt(raw) or (fa is not None and (
                    fa.states_bound(raw, value, decimals) or fa.states_value(raw, value, decimals)
                    or fa.states_anchor(raw, value, decimals))):
                continue
            errors.append(f"{name}: {claim.get('id')} falsifier number {raw!r} is not its bounds, "
                          "value or anchor value")

    # A scenario may quote only the assumptions it lists — their values or
    # their anchors' observed values. Anything else, even a number that
    # happens to be somewhere in the facts, is an undeclared forecast.
    for key, sc in scenarios.items():
        if not isinstance(sc, dict):
            continue
        listed = [fas[i] for i in _scenario_ids(sc) if isinstance(i, str) and i in fas]
        texts = [sc.get("text")] + list(sc.get("falsifiers") or [])
        for text in texts:
            for raw, value, decimals in numeric_tokens(str(text or "")):
                if _is_exempt(raw) or any(
                        fa.states_value(raw, value, decimals) or fa.states_anchor(raw, value, decimals)
                        for fa in listed):
                    continue
                errors.append(f"{name}: scenario {key} number {raw!r} is not the value or anchor "
                              "of an assumption it lists")


def _pack_numbers(facts: dict[str, Any]) -> set[float]:
    """Every number in the whole facts pack, less the outlook's catalogue
    bookkeeping (`_CATALOGUE_KEYS`): its truncation count would otherwise
    license "22%" in every backward-looking section."""
    outlook = facts.get("outlook")
    if isinstance(outlook, dict) and _CATALOGUE_KEYS & outlook.keys():
        facts = {**facts, "outlook": {k: v for k, v in outlook.items() if k not in _CATALOGUE_KEYS}}
    return _numbers(facts)


def _claim_field_texts(interp: dict[str, Any]) -> list[str]:
    """The structured claim fields the page prints beside the prose — an
    assumption's value, horizon, anchor and bounds, and every claim's
    basis. They are not prose, so their numbers are checked by F3-F5 (and
    a basis is a reference, not a quotation); the advice-phrase scan and
    L1 still have to see them, because the page prints them."""
    out: list[str] = []
    for claim in _claims_of(interp):
        for key in ("value", "horizon", "anchor"):
            if isinstance(claim.get(key), str):
                out.append(claim[key])
        bounds = claim.get("bounds")
        if isinstance(bounds, list):
            out.extend(str(b) for b in bounds)
        basis = claim.get("basis")
        if isinstance(basis, list):
            out.extend(str(b) for b in basis)
    return out


# --- L1: no licensed taxonomy in new prose (owner decision 2026-09-24) ----------
#
# Public surfaces never show the licensed taxonomy's brand, codes or names.
# The read API projects every body through `industry_labels`, which is the
# net for legacy editions; this rule is the gate for NEW ones, so a
# validated edition is clean as written and the projection has nothing to
# rescue. Deliberately narrow, because a false positive costs a retry:
#
# * the brand, anywhere (letters on neither side — "gics_x" counts);
# * a bracketed list of 6/8-digit numbers (the old template's provenance);
# * a standalone 6/8-digit number that IS a known industry/sub-industry
#   code (not the tail of a decimal or a thousands group);
# * the group's own 4-digit code as "(dddd)", "group dddd" or "Industry
#   Group Analyst dddd". A code that is also a year (2010/2020/2030) is
#   never flagged: "(2030)" in a Transportation report is far more often
#   the year, and the projection still rewrites the unambiguous
#   name-plus-code form;
# * a multi-word registry name of an industry or sub-industry that is not
#   also a sector or group name, matched case-sensitively as a phrase on
#   the raw text (`industry_labels.registry_phrase_hits`; a hit wholly
#   inside one of OUR labels is exempt) and quoted in the reason so a
#   retry knows what to change. Single words ("Software", "Restaurants",
#   "Semiconductors") are ordinary English — the mandate's own prose uses
#   them — and a sector/group name is the projection's to relabel, not a
#   rejection. The mandate prompt writes these phrases in lower case
#   (`plain_registry_phrases`), so a model is never shown one to repeat.
L1_MESSAGE = "prints an internal taxonomy code or third-party classification mark"
_L1_BRAND_RE = re.compile(r"(?<![A-Za-z])GICS(?![A-Za-z])", re.I)
_L1_BRACKET_RE = re.compile(r"[\[(]\s*\d{6,8}(?:\s*[,;/]\s*\d{6,8})*\s*[\])]")
_L1_LONG_CODE_RE = re.compile(r"(?<![\w$.,])(\d{8}|\d{6})(?![\w%]|[.,]\d)")
_L1_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")


@lru_cache(maxsize=1)
def _l1_long_codes() -> frozenset[str]:
    """The known 6/8-digit codes, from the same bundled index the public
    scrubber uses — one taxonomy source."""
    from ..services import industry_labels

    return frozenset(c for c in industry_labels.registry_names() if len(c) in (6, 8))


def taxonomy_leaks(text: str, *, group_code: str | None = None) -> list[str]:
    """What in `text` would put the licensed taxonomy on the page (L1), as
    short reasons; empty when clean.

    A registry-name reason QUOTES the phrase. The reason is all a repair
    retry is told, and "registry name" alone does not say which words to
    change — a model re-reading its own paragraph cannot guess that
    "Office REITs" is the taxonomy's name and "office landlords" is not."""
    from ..services import industry_labels

    if not isinstance(text, str) or not text:
        return []
    found: list[str] = []
    if _L1_BRAND_RE.search(text):
        found.append("classification brand")
    if _L1_BRACKET_RE.search(text):
        found.append("bracketed code list")
    if any(m.group(1) in _l1_long_codes() for m in _L1_LONG_CODE_RE.finditer(text)):
        found.append("industry or sub-industry code")
    code = str(group_code or "")
    if len(code) == 4 and code.isdigit() and not _L1_YEAR_RE.match(code):
        own = re.compile(
            rf"\(\s*{code}\s*\)|\bgroup\s+{code}(?!\d)|\bIndustry Group Analyst\s+{code}(?!\d)", re.I,
        )
        if own.search(text):
            found.append("the group's own code")
    # Our own labels are exempt ("Software & IT Services" contains the
    # industry name "IT Services"); `registry_phrase_hits` discards only a
    # hit that lies wholly inside a label, so a registry name that merely
    # CONTAINS a label word ("Health Care Technology") is still caught.
    for phrase in dict.fromkeys(m.group(0) for m in industry_labels.registry_phrase_hits(text)):
        found.append(f"industry or sub-industry registry name {phrase!r}")
    return found


# --- the gate -----------------------------------------------------------------


def validate(payload: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """Errors that block publication; empty list means publishable.

    `facts` is the per-section facts dict the server built (the same object
    handed to the writer); `payload` is the writer's output.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        return ["payload.sections missing"]

    for name in SECTION_ORDER:
        if name not in sections or not isinstance(sections[name], dict):
            errors.append(f"section missing: {name}")
    if list(payload.get("section_order") or []) != list(SECTION_ORDER):
        errors.append("section_order does not match the frozen order")

    disclaimer = str(payload.get("disclaimer") or "")
    if "research" not in disclaimer.lower() or "education" not in disclaimer.lower():
        errors.append("disclaimer missing")

    # Facts are the server's; any drift is a rejection.
    for name in SECTION_ORDER:
        if name not in sections or not isinstance(sections[name], dict):
            continue
        if name in facts and sections[name].get("facts") != facts[name]:
            errors.append(f"facts mutated: {name}")

    allowed = _pack_numbers(facts)
    stage_ids = [s["id"] for s in thesis_stages()]
    overview_facts = facts.get("overview")
    group_code = overview_facts.get("code") if isinstance(overview_facts, dict) else None

    for name in INTERPRETED_SECTIONS:
        section = sections.get(name)
        if not isinstance(section, dict):
            continue
        interp = section.get("interpretation")
        if not isinstance(interp, dict):
            errors.append(f"interpretation missing: {name}")
            continue
        claims = _claims_of(interp)
        for claim in claims:
            if claim.get("type") not in CLAIM_TYPES:
                errors.append(f"{name}: claim type {claim.get('type')!r} is not one of {CLAIM_TYPES}")
            # Named as its own rejection rather than left to surface as
            # "unsupported causal claim": a registered link whose falsifier
            # is a placeholder is a different defect from one nobody
            # registered, and the writer that emitted it needs to hear so.
            if claim.get("type") == "causal_inference" and not is_real_falsifier(claim.get("falsifier")):
                errors.append(
                    f"{name}: causal_inference claim has no usable falsifier "
                    f"({str(claim.get('falsifier') or '')[:40]!r})"
                )
            # F1: a registered forward number belongs to the outlook. A
            # backward-looking section quotes observations, full stop.
            if claim.get("type") == "forecast_assumption" and name not in FORWARD_SECTIONS:
                errors.append(f"{name}: forecast_assumption claims are accepted only in outlook")
        forward = name in FORWARD_SECTIONS
        if forward:
            _forward_errors(name, interp, facts, errors)
        for text in _claim_field_texts(interp):
            low = text.lower()
            for phrase in FORBIDDEN_PHRASES:
                if phrase in low:
                    errors.append(f"{name}: advice phrasing {phrase!r}")
        # L1 — over everything the page prints for this section: the prose
        # and the structured claim fields (an assumption's value, horizon,
        # anchor, bounds; every basis).
        leaks: set[str] = set()
        for text in _interpretation_texts(interp) + _claim_field_texts(interp):
            leaks.update(taxonomy_leaks(text, group_code=group_code))
        if leaks:
            errors.append(f"{name}: {L1_MESSAGE} ({', '.join(sorted(leaks))})")
        for text in _interpretation_texts(interp):
            low = text.lower()
            for phrase in FORBIDDEN_PHRASES:
                if phrase in low:
                    errors.append(f"{name}: advice phrasing {phrase!r}")
            if not forward:  # the forward section's numbers were checked above
                for raw, value, decimals in numeric_tokens(text):
                    if not _supported(raw, value, decimals, allowed):
                        errors.append(f"{name}: number {raw!r} is not in the facts")
            for sentence in _sentences(text):
                if _has_causal_marker(sentence) and not _claim_supports(sentence, claims):
                    errors.append(f"{name}: unsupported causal claim {sentence[:80]!r}")

        if name == "drivers":
            stages = interp.get("stages")
            if not isinstance(stages, list) or not stages:
                errors.append("drivers: stages missing")
            else:
                ids = [str(s.get("id", "")) for s in stages if isinstance(s, dict)]
                if stage_ids and ids != stage_ids:
                    errors.append(f"drivers: stages are not in the methodology order {stage_ids}")
                first = stages[0] if isinstance(stages[0], dict) else {}
                first_text = str(first.get("text") or "").strip()
                if not first_text:
                    errors.append("drivers: the first stage (world change) is empty")
                elif opens_with_kpi_forecast(first_text):
                    errors.append("drivers: opens with a KPI forecast before naming the world change")
            text = interp.get("text")
            if isinstance(text, str) and opens_with_kpi_forecast(text):
                errors.append("drivers: text opens with a KPI forecast before naming the world change")

    # De-duplicate while preserving order — a repeated number in three
    # scenarios is one defect, not three.
    seen: set[str] = set()
    unique: list[str] = []
    for e in errors:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique
