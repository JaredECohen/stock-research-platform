"""W2b 7(a) — the source ledger: every fact the memo's analysts were given.

A memo run activates one `SourceLedger` (a `ContextVar`, like
`DegradationLog.activate`). Wherever an analyst builds the payload it
hands its model, it calls `register_source(kind, ref, payload)`; the
ledger extracts the numbers from it EAGERLY (structured values by key
path, prose via `number_check.extract_claims`) and keeps no reference to
the payload, so a 12 KB MD&A costs a few hundred small tuples, not the
text. After the memo is composed, `snapshot()` freezes the facts into a
`FactRegistry` the number check queries.

What is registered is the evidence. What is deliberately NOT registered
(it would launder unverified numbers into "sources"): long-term memory
(priors, not evidence — owner decision 11), specialist findings and any
other LLM output (PM text, critic draft, long-form input, the DCF
adjuster's rationale, scenario-driver prose, the earnings multi-pass
addendum, the structured earnings extraction as evidence of the call),
and sector bull/bear analyses. The earnings QoQ tile's two structured
extractions ARE its inputs, so they are registered under the labelled,
non-primary kind `prior_extraction`: a figure traced there is traced to
an earlier model reading, never to a filing or a transcript.

Registration can never hurt the analyst that calls it:
`register_source` contains every error except the closed-vocabulary
check (a programming error the tests enforce), marks the ledger
incomplete (the number check then reports "figures not checked" instead
of flagging real figures against a partial registry) and returns False.

Outside a memo run (chat, screener, a direct agent call) there is no
active ledger and `register_source` is a no-op.
"""
from __future__ import annotations

import bisect
import contextlib
import hashlib
import logging
import math
import re
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from . import number_check as nc

log = logging.getLogger(__name__)

SOURCE_KINDS = frozenset({
    "financials", "filing", "transcript", "news", "research_note", "dcf",
    "dcf_adjustment", "comps", "scorecard", "sector_research", "macro", "technical",
    "industry", "estimates", "price", "catalyst_calendar", "factor_scores",
    "prior_extraction",
})
PRIMARY_KINDS = frozenset({"financials", "filing", "transcript"})

# Units a fact can carry. "ratio" is any structured number (unitless, as
# stored); the others are prose figures as printed.
FACT_UNITS = ("ratio", "pct", "pp", "multiple", "usd", "currency", "number")

# Structured keys whose value is already in percent ("surprise_pct" can be
# either; the ratio-like x100 path still covers a fraction stored there).
_PCT_KEY_RE = re.compile(r"(?:^|_)(?:pct|percent|percentage)$", re.I)
# Keys that name a DIFFERENCE of two rates: the only structured values a
# percentage-point claim may match (besides pp facts).
_DELTA_KEY_RE = re.compile(r"(?:^|_)(?:delta|spread|diff)(?:_|$)", re.I)
# Keys whose value is already in percentage points.
_PP_KEY_RE = re.compile(r"(?:^|_)pp$", re.I)
# Tokens that make a structured fact a RELATIVE change, which a
# percentage-point claim must never match.
_REL_TOKENS = frozenset({"growth", "premium", "upside"})
_REL_KEY_RE = re.compile(r"(?:change|return|surprise|reaction)", re.I)
# Tokens every fact of a kind carries, so "DCF base case" anchors to
# `base.implied_share_price` registered under kind "dcf".
_KIND_TOKENS: dict[str, frozenset[str]] = {
    "dcf": frozenset({"dcf"}),
    "dcf_adjustment": frozenset({"dcf"}),
    "comps": frozenset({"peer"}),
    "technical": frozenset({"price"}),
}
# Walk bounds: a registered object is small (a payload), but a caller could
# hand over something huge; the ledger must never be the memo's OOM.
MAX_DEPTH = 12
MAX_FACTS_PER_REGISTER = 20_000
MAX_TEXT_CHARS = 60_000


@dataclass(frozen=True, slots=True)
class Fact:
    value: float                 # abs value in the fact's own unit
    unit: str                    # FACT_UNITS member
    kind: str                    # SOURCE_KINDS member; derived facts keep their base kind
    source: str                  # ref, e.g. "transcript:2025Q3", "dcf:initial"
    ctx: frozenset[str]          # canonical metric tokens (number_check.METRIC_SYNONYMS)
    derived: str = ""            # "D1".."D4" when computed; "" when registered verbatim
    pct_key: bool = False        # structured key says the value is already in percent
    rel: bool = False            # a relative change (growth, premium, upside)
    delta: bool = False          # a difference of two rates (key says delta/spread)


def _ratio_like(v: float) -> bool:
    """`industry_report_validator._percent_form`'s rule: a stored fraction.
    Counts never support percentages: |a| <= 1, or a fractional |a| <= 10."""
    a = abs(v)
    return a <= 1.0 or (a <= 10.0 and a != int(a))


class FactRegistry:
    """Immutable snapshot of the ledger with bisect indexes per match class.

    Lookup is O(log facts) per claim; no numpy (the project keeps numpy a
    lazy import)."""

    def __init__(self, facts: Iterable[Fact], sources: dict[str, str],
                 incomplete: Iterable[str] = ()):
        self.facts: tuple[Fact, ...] = tuple(facts)
        self.sources: dict[str, str] = dict(sources)
        self.incomplete_steps: tuple[str, ...] = tuple(incomplete)
        idx: dict[str, list[tuple[float, int]]] = {"abs": [], "mult": [], "pct": [], "pp": []}
        tokens: set[str] = set()
        for i, f in enumerate(self.facts):
            tokens |= f.ctx
            v = f.value
            if f.unit in ("ratio", "usd", "number", "currency"):
                idx["abs"].append((v, i))
            if f.unit in ("ratio", "multiple", "number"):
                idx["mult"].append((v, i))
            if f.unit == "pct":
                idx["pct"].append((v, i))
            elif f.unit == "ratio":
                # A "_pct" key is NOT reliably in percent here: `upside_pct`
                # and `surprise_pct` hold fractions. Such a fact is indexed
                # both ways; the value match is still bound by anchoring.
                if f.pct_key:
                    idx["pct"].append((v, i))
                if _ratio_like(v):
                    idx["pct"].append((v * 100.0, i))
            if f.unit == "pp":
                idx["pp"].append((v, i))
            elif f.unit == "ratio" and f.delta and not f.rel and _ratio_like(v):
                idx["pp"].append((v * 100.0, i))
        self._keys = {k: [x[0] for x in sorted(lst)] for k, lst in idx.items()}
        self._ids = {k: [x[1] for x in sorted(lst)] for k, lst in idx.items()}
        self._tokens = frozenset(tokens)

    @property
    def complete(self) -> bool:
        return not self.incomplete_steps

    def knows_any(self, tokens: Iterable[str]) -> bool:
        """True when some registered fact carries one of `tokens`."""
        return any(t in self._tokens for t in tokens)

    def value_matches(self, unit: str, value: float, tol: float) -> list[Fact]:
        cls = {"usd": "abs", "currency": "abs", "number": "abs", "multiple": "mult",
               "pct": "pct", "pp": "pp"}.get(unit, "abs")
        keys, ids = self._keys[cls], self._ids[cls]
        lo = bisect.bisect_left(keys, value - tol)
        hi = bisect.bisect_right(keys, value + tol)
        return [self.facts[ids[j]] for j in range(lo, hi)]

    def resolves(self, ref: str) -> bool:
        return bool(ref) and ref in self.sources


class _Walker:
    """Extracts facts from one registered object without keeping it."""

    def __init__(self, kind: str, ref: str, *, text_keys: frozenset[str] | None,
                 exclude_keys: frozenset[str], pct: bool):
        self.kind, self.ref = kind, ref
        self.text_keys, self.exclude_keys, self.pct = text_keys, exclude_keys, pct
        self.kind_ctx = _KIND_TOKENS.get(kind, frozenset())
        self.facts: list[Fact] = []

    def walk(self, obj: Any, path: tuple[str, ...] = (), depth: int = 0, text_ok: bool = True) -> None:
        if depth > MAX_DEPTH or len(self.facts) >= MAX_FACTS_PER_REGISTER:
            return
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            obj = obj.model_dump()
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            f = float(obj)
            if math.isfinite(f):
                self._number(f, path)
            return
        if isinstance(obj, str):
            if text_ok:
                self._text(obj, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                if key in self.exclude_keys:
                    continue
                # With `text_keys`, prose is read only under those keys
                # (numbers are always taken): a DCF's labels, never its
                # LLM-written scenario rationale.
                ok = True if self.text_keys is None else key in self.text_keys
                self.walk(v, path + (key,), depth + 1, ok)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self.walk(v, path + (str(i),), depth + 1, text_ok)
            return

    def _number(self, v: float, path: tuple[str, ...]) -> None:
        leaf = path[-1] if path else ""
        ctx = nc.key_tokens(path) | self.kind_ctx
        # A key that says percentage points holds a value already in points
        # (`current_vs_own_median_pp`: 2.61 means 2.61 points).
        if _PP_KEY_RE.search(leaf) or any(_PP_KEY_RE.search(p) for p in path[-2:-1]):
            self.facts.append(Fact(value=abs(v), unit="pp", kind=self.kind, source=self.ref, ctx=ctx))
            return
        self.facts.append(Fact(
            value=abs(v), unit="ratio", kind=self.kind, source=self.ref, ctx=ctx,
            pct_key=self.pct or bool(_PCT_KEY_RE.search(leaf)),
            rel=bool(ctx & _REL_TOKENS) or bool(_REL_KEY_RE.search(leaf)),
            delta=bool(_DELTA_KEY_RE.search(leaf)),
        ))

    def _text(self, s: str, path: tuple[str, ...]) -> None:
        text = s[:MAX_TEXT_CHARS]
        if not any(ch.isdigit() for ch in text):
            return
        key_ctx = nc.key_tokens(path[-1:]) if path else frozenset()
        for c in nc.extract_claims(text):
            if c.cls == "exempt":
                continue
            ctx = nc.fact_window_tokens(text, c.start, c.end) | key_ctx | self.kind_ctx
            unit = c.unit
            self.facts.append(Fact(
                value=c.value, unit=unit, kind=self.kind, source=self.ref, ctx=ctx,
                rel=unit == "pct" and bool(ctx & _REL_TOKENS),
            ))


# ---------------------------------------------------------------------------
# Derived facts (bounded, labelled): D1/D2 financials, D3 DCF, D4 comps
# ---------------------------------------------------------------------------

_ANNUAL_RE = re.compile(r"^(?:FY\s?)?(?:19|20)\d{2}$", re.I)
_QUARTER_RE = re.compile(r"(?:Q[1-4]|[1-4]Q)", re.I)
_META_KEYS = frozenset({"period", "date", "period_end", "fiscal_year", "fiscal_quarter",
                        "calendar_year", "symbol", "ticker", "currency", "filing_date",
                        "accepted_date", "cik", "link", "final_link", "reported_currency"})


def _rows_by_cadence(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Annual vs quarterly by the row's period label, the same split
    `fundamentals_series_service._classify_rows` makes for stored rows
    (an explicit FY/year label is annual; a quarter label is quarterly;
    anything else is left out rather than guessed)."""
    out: dict[str, list[dict[str, Any]]] = {"annual": [], "quarterly": []}
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = str(r.get("period") or r.get("date") or "").strip()
        if _QUARTER_RE.search(p):
            out["quarterly"].append(r)
        elif _ANNUAL_RE.match(p):
            out["annual"].append(r)
    for k in out:
        out[k].sort(key=lambda r: str(r.get("period") or r.get("date") or ""))
    return out


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _derive_financials(obj: Any, add) -> None:
    if not isinstance(obj, dict):
        return
    stmts = {k: obj.get(k) for k in ("income", "balance", "cash") if isinstance(obj.get(k), list)}
    revenue_by_period: dict[str, float] = {}
    for row in stmts.get("income") or []:
        if isinstance(row, dict) and _num(row.get("revenue")):
            revenue_by_period[str(row.get("period") or row.get("date") or "")] = float(row["revenue"])
    for stmt, rows in stmts.items():
        for cadence, series in _rows_by_cadence(rows or []).items():
            lag_steps = (1, 4) if cadence == "quarterly" else (1,)
            for i, row in enumerate(series):
                period = str(row.get("period") or row.get("date") or "")
                for key, raw in row.items():
                    if key in _META_KEYS:
                        continue
                    v = _num(raw)
                    if v is None:
                        continue
                    # D1: growth vs the previous same-cadence period (and YoY for quarters).
                    for lag in lag_steps:
                        if i - lag < 0:
                            continue
                        prev = _num(series[i - lag].get(key))
                        if prev is not None and prev > 0:
                            add("D1", v / prev - 1.0, (stmt, key, "growth"), rel=True)
                    # D2: line item / revenue (margins, capex intensity, SBC share).
                    rev = revenue_by_period.get(period)
                    if rev and rev > 0 and key != "revenue":
                        add("D2", v / rev, (stmt, key, "margin"))
            # D2: FCF = OCF - |capex| and its margin.
            if stmt == "cash":
                for row in series:
                    ocf, capex = _num(row.get("cash_from_operations")), _num(row.get("capex"))
                    if ocf is None or capex is None:
                        continue
                    fcf = ocf - abs(capex)
                    add("D2", fcf, ("free_cash_flow",))
                    rev = revenue_by_period.get(str(row.get("period") or row.get("date") or ""))
                    if rev and rev > 0:
                        add("D2", fcf / rev, ("fcf", "margin"))


def _derive_dcf(obj: Any, add) -> None:
    d = obj.model_dump() if hasattr(obj, "model_dump") else obj
    if not isinstance(d, dict):
        return
    price = _num(d.get("current_price"))
    scen: dict[str, dict[str, Any]] = {
        k: v for k in ("base", "bull", "bear") if isinstance(v := d.get(k), dict)}
    implied = {k: _num(s.get("implied_share_price")) for k, s in scen.items()}
    for k, p in implied.items():
        if p is not None and price:
            add("D3", p - price, (k, "implied", "price", "spread"))
    bull, base, bear = implied.get("bull"), implied.get("base"), implied.get("bear")
    if bull and bear and bear > 0:
        add("D3", bull / bear, ("bull", "bear", "ratio"))
    if bull is not None and bear is not None:
        add("D3", bull - bear, ("bull", "bear", "spread"))
    if bull is not None and base is not None:
        add("D3", bull - base, ("bull", "base", "spread"))
    if base is not None and bear is not None:
        add("D3", base - bear, ("base", "bear", "spread"))
    base_a = (scen.get("base") or {}).get("assumptions") or {}
    for name in ("bull", "bear"):
        a = (scen.get(name) or {}).get("assumptions") or {}
        for key, bv in base_a.items():
            sv = a.get(key)
            if isinstance(bv, list) and isinstance(sv, list) and bv and sv:
                b0, s0 = _num(bv[0]), _num(sv[0])
                if b0 is not None and s0 is not None:
                    add("D3", (s0 - b0) * 100.0, (name, key, "delta"), unit="pp")
            else:
                b0, s0 = _num(bv), _num(sv)
                if b0 is not None and s0 is not None and _ratio_like(b0) and _ratio_like(s0):
                    add("D3", (s0 - b0) * 100.0, (name, key, "delta"), unit="pp")
    for name, s in scen.items():
        g = (s.get("assumptions") or {}).get("revenue_growth")
        if isinstance(g, list):
            vals = [x for x in (_num(v) for v in g) if x is not None]
            if vals:
                add("D3", sum(vals) / len(vals), (name, "revenue", "growth", "average"), rel=True)


def _derive_comps(obj: Any, add) -> None:
    d = obj.model_dump() if hasattr(obj, "model_dump") else obj
    if not isinstance(d, dict):
        return
    target, median = d.get("target") or {}, d.get("median") or {}
    if isinstance(target, dict) and isinstance(median, dict):
        for key, tv in target.items():
            t, m = _num(tv), _num(median.get(key))
            if t is None or m is None:
                continue
            if _ratio_like(t) and _ratio_like(m):
                add("D4", (t - m) * 100.0, (key, "peer", "median", "delta"), unit="pp")
            if m > 0:
                add("D4", t / m - 1.0, (key, "peer", "median", "premium"), rel=True)
    hist = d.get("history") or {}
    own_median = hist.get("own_median") if isinstance(hist, dict) else None
    if isinstance(own_median, dict) and isinstance(target, dict):
        for key, raw_med in own_median.items():
            med = _num(raw_med)
            t = _num(target.get(key))
            if med is None or t is None:
                continue
            if _ratio_like(t) and _ratio_like(med):
                add("D4", (t - med) * 100.0, (key, "history", "median", "delta"), unit="pp")
            if med > 0:
                add("D4", t / med - 1.0, (key, "history", "median", "premium"), rel=True)


_DERIVERS = {"financials": _derive_financials, "dcf": _derive_dcf, "comps": _derive_comps}


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

_ACTIVE: ContextVar[SourceLedger | None] = ContextVar("source_ledger", default=None)


class SourceLedger:
    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._seen: set[tuple[str, str, str]] = set()
        self._sources: dict[str, str] = {}
        self._incomplete: list[str] = []
        self._captures: list[list[Fact]] = []

    # -- lifecycle ------------------------------------------------------
    @contextlib.contextmanager
    def activate(self) -> Iterator[SourceLedger]:
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    @contextlib.contextmanager
    def capture(self) -> Iterator[list[Fact]]:
        """Facts registered inside the block (for checkpoint persistence)."""
        bucket: list[Fact] = []
        self._captures.append(bucket)
        try:
            yield bucket
        finally:
            self._captures.remove(bucket)

    # -- writes ---------------------------------------------------------
    def register(self, kind: str, ref: str, obj: Any, *, text_keys: frozenset[str] | None = None,
                 exclude_keys: Iterable[str] = (), pct: bool = False) -> int:
        """Extract the facts of `obj` under (kind, ref). Returns the number added.

        Raises ValueError on a kind outside `SOURCE_KINDS`. Keeps no reference
        to `obj`. Dedupes on (kind, ref, digest of the extracted facts), so a
        deep-research re-fire that registers the same payload adds nothing."""
        if kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind: {kind!r}")
        ref = str(ref or kind)[:200]
        walker = _Walker(kind, ref, text_keys=text_keys, exclude_keys=frozenset(exclude_keys), pct=pct)
        walker.walk(obj)
        facts = walker.facts
        derive = _DERIVERS.get(kind)
        if derive is not None:
            base_ctx = _KIND_TOKENS.get(kind, frozenset())

            def add(label: str, value: float, path: tuple[str, ...], *, unit: str = "ratio",
                    rel: bool = False) -> None:
                if not math.isfinite(value) or len(facts) >= MAX_FACTS_PER_REGISTER:
                    return
                ctx = nc.key_tokens(path) | base_ctx
                leaf = path[-1] if path else ""
                facts.append(Fact(value=abs(value), unit=unit, kind=kind, source=ref, ctx=ctx,
                                  derived=label, rel=rel or unit == "ratio" and bool(ctx & _REL_TOKENS),
                                  delta=unit == "ratio" and bool(_DELTA_KEY_RE.search(leaf))))
            derive(obj, add)
        digest = hashlib.sha1(repr(sorted(
            (f.value, f.unit, tuple(sorted(f.ctx)), f.derived) for f in facts
        )).encode()).hexdigest()
        key = (kind, ref, digest)
        self._sources.setdefault(ref, kind)
        if key in self._seen:
            return 0
        self._seen.add(key)
        self._add(facts)
        return len(facts)

    def _add(self, facts: list[Fact]) -> None:
        self._facts.extend(facts)
        for bucket in self._captures:
            bucket.extend(facts)

    def mark_incomplete(self, step: str) -> None:
        if step not in self._incomplete:
            self._incomplete.append(step)

    # -- checkpoint export / replay --------------------------------------
    @staticmethod
    def export(facts: Iterable[Fact]) -> dict[str, Any]:
        """Compact JSON form: string tables plus one row per fact
        `[value, unit, kind, source, ctx, derived, flags]` (indexes into
        `strings`; flags bit 0 pct_key, 1 rel, 2 delta)."""
        strings: list[str] = []
        at: dict[str, int] = {}

        def s(x: str) -> int:
            if x not in at:
                at[x] = len(strings)
                strings.append(x)
            return at[x]

        rows = [[f.value, s(f.unit), s(f.kind), s(f.source), s(" ".join(sorted(f.ctx))), s(f.derived),
                 (1 if f.pct_key else 0) | (2 if f.rel else 0) | (4 if f.delta else 0)] for f in facts]
        return {"v": 1, "strings": strings, "facts": rows}

    def replay(self, exported: Any) -> int:
        """Re-add facts a checkpointed step registered in an earlier attempt.
        A malformed payload raises (the caller marks the registry incomplete)."""
        if not isinstance(exported, dict) or exported.get("v") != 1:
            raise ValueError("unrecognised checkpoint sources payload")
        strings = list(exported.get("strings") or [])
        facts: list[Fact] = []
        for row in exported.get("facts") or []:
            value, unit, kind, source, ctx, derived, flags = row
            k = strings[kind]
            if k not in SOURCE_KINDS or strings[unit] not in FACT_UNITS:
                raise ValueError("checkpoint sources carry an unknown kind or unit")
            facts.append(Fact(
                value=float(value), unit=strings[unit], kind=k, source=strings[source],
                ctx=frozenset(t for t in strings[ctx].split() if t), derived=strings[derived],
                pct_key=bool(flags & 1), rel=bool(flags & 2), delta=bool(flags & 4),
            ))
        for f in facts:
            self._sources.setdefault(f.source, f.kind)
        self._add(facts)
        return len(facts)

    # -- reads ------------------------------------------------------------
    def snapshot(self) -> FactRegistry:
        return FactRegistry(self._facts, self._sources, self._incomplete)

    def source_refs(self) -> list[str]:
        return sorted(self._sources)

    @property
    def fact_count(self) -> int:
        return len(self._facts)


def active_ledger() -> SourceLedger | None:
    return _ACTIVE.get()


def register_source(kind: str, ref: str, obj: Any, *, text_keys: Iterable[str] | None = None,
                    exclude_keys: Iterable[str] = (), pct: bool = False) -> bool:
    """Register `obj` on the active ledger. False (a no-op) outside a memo run.

    Contains every error except an unknown `kind` (a programming error,
    enforced by the tests): a walker failure marks the ledger incomplete,
    logs, and returns False, so the analyst's finding is never affected."""
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind: {kind!r}")
    ledger = _ACTIVE.get()
    if ledger is None:
        return False
    try:
        ledger.register(kind, ref, obj,
                        text_keys=frozenset(text_keys) if text_keys is not None else None,
                        exclude_keys=exclude_keys, pct=pct)
        return True
    except Exception as exc:
        ledger.mark_incomplete(f"register:{kind}:{str(ref)[:60]}")
        log.warning("source ledger: registering %s/%s failed (%s); figures will be reported unchecked",
                    kind, str(ref)[:60], type(exc).__name__)
        return False
