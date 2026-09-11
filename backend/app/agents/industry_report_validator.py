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
* no advice phrasing, and the disclaimer is present.

``validate`` returns a list of error strings; empty means the payload may
be published. A rejected payload fails the job, which retries and, on the
final attempt, runs the deterministic writer so a week is never blank.
"""
from __future__ import annotations

import re
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
        if not claim.get("basis") or not str(claim.get("falsifier") or "").strip():
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

    allowed = _numbers(facts)
    stage_ids = [s["id"] for s in thesis_stages()]

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
        for text in _interpretation_texts(interp):
            low = text.lower()
            for phrase in FORBIDDEN_PHRASES:
                if phrase in low:
                    errors.append(f"{name}: advice phrasing {phrase!r}")
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
