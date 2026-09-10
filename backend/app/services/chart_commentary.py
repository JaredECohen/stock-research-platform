"""FEAT-001 — chart commentary for the Fundamentals Explorer (slice S3).

`generate(req, principal, *, request, db, years)` turns a commentary
request into `CommentaryOut`: two labelled sections the frontend keeps
apart — "Observed in the data" (computed here, in Python, from the
recomputed series) and "From stored memos" (one cheap-route LLM call
that relates each stored memo's view to the displayed data) — plus the
caveats that say what was dropped, truncated, stale or missing.

The order of operations is the cost control, so it is spelled out:

  1. Recompute the series with the same code path the chart used
     (`fundamentals_series_service.build_series`) and compare its
     fingerprint with the one the client sent. A mismatch is
     `SeriesChanged` (409): commentary is only ever written about data
     the client is showing, and client-supplied numbers are never
     trusted. `normalize` is not in the request, so both views are
     tried; the one that matches is the one displayed.
  2. Build the deterministic observations. These are free and always
     returned, degraded or not.
  3. Anonymous with the wall off and `FUNDAMENTALS_ANON_COMMENTARY`
     unset → the degraded shape (`"commentary requires an account"`),
     no LLM call, no charge, nothing stored.
  4. Cache lookup by `cache_key` — everything that changes the answer:
     the series fingerprint, the selection, `CATALOG_VERSION`,
     `PROMPT_VERSION` and the memo versions quoted. A hit is served from
     the `chart_commentaries` row without an LLM call and without a
     charge (the table is shared by every web replica, so a hit here is
     a hit everywhere). A stored *degraded* row is never a hit. Memo
     staleness is deliberately *not* in the key — a filing that lands
     after the row was written does not change the memo version, so it
     would not move the key anyway — and is therefore overlaid live on
     every hit: `memo_stale` / `memo_stale_reason` on each memo_view item
     and the "memo … is stale" caveats reflect now, not generation time.
  5. Nothing to interpret (no series has a value, or no selected ticker
     has a stored memo) or no LLM (provider unconfigured, breaker open)
     → the degraded shape, free.
  6. `authorize()` — the FEAT-002 seam — reserves the `chart_commentary`
     meter and the two-in-flight lease BEFORE the call. A database
     failure inside the meter is `UsageUnavailable` (503): nothing was
     charged and nothing generated, and the caller can retry. Then one
     `llm.chat_json(route="cheap")` inside `llm_call_context` so the
     spend is attributed to the user and the feature. Nothing usable
     back → `grant.release()` (the degraded answer is free); otherwise
     `grant.commit()` and the row is stored.

What the model sees is exactly the displayed series plus bounded memo
excerpts (`prompts/chart_commentary`) — never the research pipeline,
filings or the vector store. What it returns is validated: memo_view
sentences for a ticker outside the selection, or one without a stored
memo, are dropped and counted in `caveats`.

`memo_stale` combines `memo_store.memo_freshness` (a newer filing or
transcript exists) with "the memo predates the last displayed fiscal
period end", each with its reason. Missing evidence is disclosed, never
scored as neutral: a ticker without a memo is named in the caveats, and
a series without values is an observation saying so.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.requests import Request

from ..agents import llm
from ..agents.log_safety import log_safely
from ..auth.entitlements import Grant, authorize
from ..auth.principal import Principal
from ..config import settings
from ..models import ChartCommentary, MemoSnapshot
from ..prompts import chart_commentary as prompt_mod
from ..schemas.fundamentals import (
    CommentaryOut,
    CommentaryRef,
    CommentaryRequest,
    MemoViewItem,
    MetricSeries,
    ObservedItem,
    SeriesResponse,
)
from . import fundamentals_catalog as catalog
from . import memo_store
from .fundamentals_series_service import build_series

log = logging.getLogger(__name__)

FEATURE = "chart_commentary"
AGENT_NAME = "Chart Commentary"
MAX_TOKENS = 900

# Degraded reasons — closed set, each a sentence fragment the panel can
# show. `REASON_NO_ACCOUNT` is the exact string the owner decision names.
REASON_NO_ACCOUNT = "commentary requires an account"
REASON_NO_LLM = "llm_unavailable: no LLM provider is configured"
REASON_BREAKER = "breaker_open: the LLM provider's circuit breaker is open"
REASON_INVALID = "invalid_output: the model returned nothing usable"
REASON_NO_DATA = "no_displayed_data: no selected series has a value"
REASON_NO_MEMOS = "no_stored_memos: none of the selected companies has a stored memo"

# Observations are bounded so a 5 × 4 chart cannot flood the panel.
MAX_OBSERVED = 48


class SeriesChanged(Exception):
    """The recomputed fingerprint differs from the client's (409)."""

    def __init__(self, actual: str, requested: str) -> None:
        super().__init__("series changed")
        self.actual = actual
        self.requested = requested


class UsageUnavailable(Exception):
    """The usage meter could not be read or written (503). Nothing was
    charged and no commentary was generated."""


def _utcnow() -> datetime:
    """Clock seam — tests pin it instead of freezing time."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Formatting (unit-aware, used by both the observations and the prompt)
# ---------------------------------------------------------------------------

def _scale(v: float) -> str:
    a = abs(v)
    if a >= 1e12:
        return f"{v / 1e12:.2f}T"
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.2f}"


def fmt_value(value: float | None, unit_type: str, currency: str | None, *, indexed: bool = False) -> str:
    if value is None:
        return "n/a"
    if indexed:
        return f"{value:.1f}"
    if unit_type == "percent":
        return f"{value * 100:.1f}%"
    if unit_type == "multiple":
        return f"{value:.1f}x"
    if unit_type == "ratio":
        return f"{value:.2f}"
    if unit_type == "currency":
        return f"{currency} {_scale(value)}" if currency else _scale(value)
    return _scale(value)


def _render_point(value: float | None, reason: str | None, unit_type: str, currency: str | None,
                  *, indexed: bool) -> str:
    if value is None:
        return f"n/a({reason or 'unknown'})"
    return fmt_value(value, unit_type, currency, indexed=indexed)


# ---------------------------------------------------------------------------
# Observed in the data — deterministic
# ---------------------------------------------------------------------------

def _change_text(first: float, last: float, n_years: int, unit_type: str, indexed: bool) -> str:
    """The change between two valued points in the unit's own terms.
    Percent-type metrics move in percentage points; currency/count
    change is relative (with a CAGR when the base is positive and more
    than a year apart); multiples/ratios are stated as a plain delta."""
    if unit_type == "percent" and not indexed:
        return f"{(last - first) * 100:+.1f} pp"
    if unit_type in ("currency", "count") or indexed:
        if first > 0:
            rel = (last - first) / first
            if n_years >= 2 and last > 0:
                cagr = (last / first) ** (1.0 / n_years) - 1.0
                return f"{rel * 100:+.1f}%, {cagr * 100:.1f}% CAGR over {n_years} years"
            return f"{rel * 100:+.1f}%"
        return "relative change not meaningful from a non-positive base"
    return f"{last - first:+.2f}"


def _year_of(period: str) -> int | None:
    s = period.upper().removeprefix("FY")
    return int(s) if s.isdigit() else None


def _series_observation(s: MetricSeries, label: str) -> ObservedItem | None:
    valued = [p for p in s.points if p.value is not None]

    def ref(p: Any) -> CommentaryRef:
        return CommentaryRef(ticker=s.ticker, metric=s.metric, period=p.period)

    if not valued:
        reasons = Counter(p.reason or "unknown" for p in s.points)
        why = ", ".join(f"{r} ×{n}" for r, n in reasons.most_common())
        return ObservedItem(
            text=f"{s.ticker} {label}: no value in any displayed period ({why}).",
            refs=[ref(p) for p in s.points[:1]],
        )
    first, last = valued[0], valued[-1]
    unit = s.unit_type
    if len(valued) == 1:
        return ObservedItem(
            text=f"{s.ticker} {label}: one valued period, {first.period} = "
                 f"{fmt_value(first.value, unit, s.currency, indexed=s.indexed)}"
                 f"{' (indexed)' if s.indexed else ''}.",
            refs=[ref(first)],
        )
    y0, y1 = _year_of(first.period), _year_of(last.period)
    n_years = (y1 - y0) if (y0 is not None and y1 is not None) else len(valued) - 1
    assert first.value is not None and last.value is not None
    change = _change_text(first.value, last.value, n_years, unit, s.indexed)
    est = " Some points use a documented fallback (estimated)." if any(p.estimated for p in valued) else ""
    idx = " (indexed to 100 at the first valued period)" if s.indexed else ""
    return ObservedItem(
        text=(
            f"{s.ticker} {label}{idx}: {first.period} "
            f"{fmt_value(first.value, unit, s.currency, indexed=s.indexed)} → {last.period} "
            f"{fmt_value(last.value, unit, s.currency, indexed=s.indexed)} ({change}).{est}"
        ),
        refs=[ref(first), ref(last)],
    )


def _extremes_observation(s: MetricSeries, label: str) -> ObservedItem | None:
    valued = [p for p in s.points if p.value is not None]
    if len(valued) < 3:
        return None
    lo = min(valued, key=lambda p: float(p.value or 0.0))
    hi = max(valued, key=lambda p: float(p.value or 0.0))
    if lo.period in (valued[0].period, valued[-1].period) and hi.period in (valued[0].period, valued[-1].period):
        return None  # the first/last observation already shows the range
    return ObservedItem(
        text=(
            f"{s.ticker} {label}: low {fmt_value(lo.value, s.unit_type, s.currency, indexed=s.indexed)} "
            f"in {lo.period}, high {fmt_value(hi.value, s.unit_type, s.currency, indexed=s.indexed)} in {hi.period}."
        ),
        refs=[CommentaryRef(ticker=s.ticker, metric=s.metric, period=lo.period),
              CommentaryRef(ticker=s.ticker, metric=s.metric, period=hi.period)],
    )


def _gap_observation(s: MetricSeries, label: str) -> ObservedItem | None:
    missing = [p for p in s.points if p.value is None]
    if not missing or len(missing) == len(s.points):
        return None
    reasons = Counter(p.reason or "unknown" for p in missing)
    why = ", ".join(f"{r} ×{n}" for r, n in reasons.most_common())
    return ObservedItem(
        text=f"{s.ticker} {label}: {len(missing)} of {len(s.points)} displayed periods have no value ({why}).",
        refs=[CommentaryRef(ticker=s.ticker, metric=s.metric, period=p.period) for p in missing[:3]],
    )


def _ranking_observation(metric: str, label: str, series: list[MetricSeries], period: str) -> ObservedItem | None:
    """Companies ranked on one metric at the last shared period. Currency
    values are compared only when every company reports in the same
    currency — otherwise the comparison is stated as not made."""
    rows: list[tuple[str, float, str | None]] = []
    for s in series:
        if s.metric != metric:
            continue
        pt = next((p for p in s.points if p.period == period), None)
        if pt is not None and pt.value is not None:
            rows.append((s.ticker, pt.value, s.currency))
    if len(rows) < 2:
        return None
    unit = next(s.unit_type for s in series if s.metric == metric)
    indexed = any(s.indexed for s in series if s.metric == metric)
    currencies = {c for _, _, c in rows if c}
    if unit == "currency" and not indexed and len(currencies) > 1:
        return ObservedItem(
            text=f"{label} at {period} is reported in different currencies ({', '.join(sorted(currencies))}); "
                 "no cross-company comparison is made.",
            refs=[CommentaryRef(ticker=t, metric=metric, period=period) for t, _, _ in rows],
        )
    rows.sort(key=lambda r: r[1], reverse=True)
    ranked = ", ".join(f"{t} {fmt_value(v, unit, c, indexed=indexed)}" for t, v, c in rows)
    return ObservedItem(
        text=f"{label} at {period}, highest to lowest: {ranked}.",
        refs=[CommentaryRef(ticker=t, metric=metric, period=period) for t, _, _ in rows],
    )


def observed_items(out: SeriesResponse) -> list[ObservedItem]:
    """The deterministic "Observed in the data" section, computed from the
    recomputed series only. Bounded by `MAX_OBSERVED`."""
    items: list[ObservedItem] = []
    labels = {m: catalog.get(m).label for m in {s.metric for s in out.series}}
    for s in out.series:
        label = labels[s.metric]
        for fn in (_series_observation, _extremes_observation, _gap_observation):
            item = fn(s, label)
            if item is not None:
                items.append(item)
    if out.periods:
        last = out.periods[-1]
        seen: list[str] = []
        for s in out.series:
            if s.metric not in seen:
                seen.append(s.metric)
        for metric in seen:
            item = _ranking_observation(metric, labels[metric], out.series, last)
            if item is not None:
                items.append(item)
    for u in out.unavailable:
        items.append(ObservedItem(
            text=f"{u.ticker}: no stored financial history ({u.reason}); nothing is displayed for it.",
            refs=[],
        ))
    return items[:MAX_OBSERVED]


# ---------------------------------------------------------------------------
# Memo excerpts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoExcerpt:
    ticker: str
    version: int
    generated_at: datetime
    stale: bool
    stale_reason: str | None
    block: prompt_mod.MemoBlock


def _last_displayed_end(out: SeriesResponse, ticker: str) -> tuple[date | None, str | None]:
    """`(period_end, period)` of the latest valued point for `ticker`
    that carries a stored period end. None when no displayed point has
    a date — then the predates rule cannot be applied and is not."""
    best: tuple[date, str] | None = None
    for s in out.series:
        if s.ticker != ticker:
            continue
        for p in s.points:
            if p.value is None or p.period_end is None:
                continue
            if best is None or p.period_end > best[0]:
                best = (p.period_end, p.period)
    return (best[0], best[1]) if best else (None, None)


def memo_staleness(snap: MemoSnapshot, out: SeriesResponse, db: Session) -> tuple[bool, str | None]:
    """`memo_stale` for the response: memo_store freshness (a newer
    filing or transcript exists) OR the memo predates the last displayed
    fiscal period end. Both reasons are kept when both apply."""
    reasons: list[str] = []
    try:
        fresh = memo_store.memo_freshness(snap, db=db)
    except SQLAlchemyError as exc:
        # Freshness is a disclosure, not a gate: an unreadable filings
        # table must not turn a commentary into a 500. Say we could not check.
        log_safely(log, f"chart commentary: memo freshness check failed for {snap.ticker}", exc)
        fresh = {"stale": False, "reason": ""}
        reasons.append("freshness could not be checked")
    if fresh.get("stale"):
        reasons.append(str(fresh.get("reason") or "newer source document exists"))
    last_end, last_period = _last_displayed_end(out, snap.ticker)
    generated = snap.generated_at
    if last_end is not None and generated is not None and generated.date() < last_end:
        reasons.append(
            f"memo generated {generated.date().isoformat()} predates the last displayed period "
            f"{last_period} (ended {last_end.isoformat()})"
        )
    stale = any(r != "freshness could not be checked" for r in reasons)
    return stale, ("; ".join(reasons) or None)


def memo_excerpt(ticker: str, out: SeriesResponse, db: Session) -> MemoExcerpt | None:
    """The bounded excerpt of the latest live memo for `ticker`, or None
    when there is no snapshot (or it no longer validates — an old memo
    the schema outgrew is reported as missing, not quoted blind)."""
    snap = memo_store.latest_memo(ticker, db=db)
    if snap is None:
        return None
    try:
        memo = memo_store.memo_to_pydantic(snap)
    except Exception as exc:  # pydantic ValidationError or a corrupt JSON column
        log_safely(log, f"chart commentary: stored memo for {ticker} does not validate; treated as missing", exc)
        return None
    stale, reason = memo_staleness(snap, out, db)
    mt = memo.mispricing_thesis
    vv = memo.valuation_verdict
    fields = (
        ("rating", str(memo.rating_label)),
        ("thesis", memo.one_sentence_thesis),
        ("consensus view", mt.consensus_view),
        ("our view", mt.our_view),
        ("gap", mt.gap),
        ("valuation verdict", f"{vv.verdict}" + (f" — {vv.summary}" if vv.summary else "")),
        ("bull case", memo.bull_case.headline),
        ("bear case", memo.bear_case.headline),
    )
    block = prompt_mod.MemoBlock(
        ticker=ticker, version=int(snap.version), generated_at=snap.generated_at.date().isoformat(),
        stale=stale, stale_reason=reason, fields=fields,
    )
    return MemoExcerpt(ticker, int(snap.version), snap.generated_at, stale, reason, block)


def memos_used(excerpts: dict[str, MemoExcerpt | None]) -> dict[str, dict[str, Any] | None]:
    """The `memo_versions` column: version + generated_at per ticker,
    None for a ticker without a memo (so the key still records that the
    row was written knowing there was none)."""
    return {
        t: (None if e is None else {"version": e.version, "generated_at": e.generated_at.isoformat()})
        for t, e in excerpts.items()
    }


# ---------------------------------------------------------------------------
# Cache key, availability
# ---------------------------------------------------------------------------

def cache_key(
    tickers: list[str], metrics: list[str], years: int | None, normalize: str, fingerprint: str,
    memo_versions: dict[str, dict[str, Any] | None],
) -> str:
    payload = {
        "catalog_version": catalog.CATALOG_VERSION,
        "prompt_version": prompt_mod.PROMPT_VERSION,
        "tickers": tickers, "metrics": metrics, "years": years, "normalize": normalize,
        "fingerprint": fingerprint,
        "memo_versions": {t: (v["version"] if v else None) for t, v in memo_versions.items()},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def llm_unavailable_reason() -> str | None:
    """Why no LLM call would succeed right now, or None. Agrees with the
    `llm.degraded` flag on `/api/providers/status` for the conditions
    that apply to one call: no provider configured, or the active
    provider's breaker open (the status page also counts a recent
    failover, which does not stop a call)."""
    provider = settings.active_llm_provider
    if provider == "none":
        return REASON_NO_LLM
    breakers = llm.get_breaker_state(include_failover=False)
    state = breakers.get(provider)
    if state and state.get("is_open"):
        # A failover partner could still answer; `chat_json` will try it.
        # Only report the breaker when there is no partner to fall to.
        partner = llm._failover_partner(provider)
        if partner is None or (breakers.get(partner) or {}).get("is_open"):
            return REASON_BREAKER
    return None


# ---------------------------------------------------------------------------
# Series recomputation
# ---------------------------------------------------------------------------

def recompute_series(req: CommentaryRequest, *, years: int | None, db: Session) -> SeriesResponse:
    """The series the client is showing, by fingerprint. The request has
    no `normalize`, so the plain view is tried first and the indexed view
    second; neither matching is `SeriesChanged`."""
    plain = build_series(req.tickers, req.metrics, years=years, normalize="none", db=db)
    if plain.fingerprint == req.fingerprint:
        return plain
    indexed = build_series(req.tickers, req.metrics, years=years, normalize="indexed", db=db)
    if indexed.fingerprint == req.fingerprint:
        return indexed
    raise SeriesChanged(actual=plain.fingerprint, requested=req.fingerprint)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _degraded(
    out: SeriesResponse, observed: list[ObservedItem], reason: str, caveats: list[str],
    *, commentary_id: int | None = None, now: datetime,
) -> CommentaryOut:
    return CommentaryOut(
        commentary_id=commentary_id, fingerprint=out.fingerprint, cache_hit=False,
        degraded=True, degraded_reason=reason, observed=observed, memo_view=[],
        caveats=caveats, generated_at=now, model=None,
    )


def _series_lines(out: SeriesResponse) -> list[prompt_mod.SeriesLine]:
    return [
        prompt_mod.SeriesLine(
            ticker=s.ticker, metric=s.metric, unit_type=s.unit_type, currency=s.currency, indexed=s.indexed,
            points=tuple(
                (p.period, _render_point(p.value, p.reason, s.unit_type, s.currency, indexed=s.indexed))
                for p in s.points
            ),
            stale=s.provenance.stale, stale_reason=s.provenance.stale_reason,
        )
        for s in out.series
    ]


def build_prompt_for(
    req: CommentaryRequest, out: SeriesResponse, observed: list[ObservedItem],
    excerpts: dict[str, MemoExcerpt | None],
) -> prompt_mod.BuiltPrompt:
    metrics = [
        prompt_mod.MetricLine(id=m, label=spec.label, unit_type=spec.unit_type, kind=spec.kind,
                              formula_text=spec.formula_text)
        for m in req.metrics for spec in (catalog.get(m),)
    ]
    return prompt_mod.build_prompt(
        tickers=req.tickers, metrics=metrics, periods=out.periods, normalize=out.normalize,
        series=_series_lines(out), observed=[o.text for o in observed],
        memos={t: (e.block if e else None) for t, e in excerpts.items()},
        unavailable={u.ticker: u.reason for u in out.unavailable},
    )


@dataclass(frozen=True)
class Validated:
    memo_view: list[MemoViewItem]
    model_caveats: list[str]
    dropped: int      # outside the selection, or no stored memo
    overflow: int     # beyond the per-ticker / total caps


def _validate_output(
    raw: Any, req: CommentaryRequest, excerpts: dict[str, MemoExcerpt | None],
) -> Validated | None:
    """The usable part of the model's answer, or None when nothing usable
    came back. An item is dropped when its ticker is outside the
    selection or has no stored memo — the model was told to omit those,
    and a sentence "from a memo" that does not exist must never render.
    Items beyond the caps are counted separately so the caveat says what
    actually happened."""
    parsed = prompt_mod.parse_output(raw)
    if parsed is None:
        return None
    pairs, model_caveats = parsed
    selected = set(req.tickers)
    per_ticker: Counter[str] = Counter()
    items: list[MemoViewItem] = []
    dropped = overflow = 0
    for ticker, text in pairs:
        excerpt = excerpts.get(ticker)
        if ticker not in selected or excerpt is None:
            dropped += 1
            continue
        if per_ticker[ticker] >= prompt_mod.MAX_MEMO_VIEW_PER_TICKER or len(items) >= prompt_mod.MAX_MEMO_VIEW_TOTAL:
            overflow += 1
            continue
        per_ticker[ticker] += 1
        items.append(MemoViewItem(
            text=text, ticker=ticker, memo_version=excerpt.version, memo_generated_at=excerpt.generated_at,
            memo_stale=excerpt.stale, memo_stale_reason=excerpt.stale_reason,
        ))
    if not items:
        return None
    return Validated(items, model_caveats, dropped, overflow)


def _lookup(db: Session, key: str) -> ChartCommentary | None:
    return db.execute(select(ChartCommentary).where(ChartCommentary.cache_key == key)).scalar_one_or_none()


def _store(
    db: Session, *, key: str, req: CommentaryRequest, years: int | None, out: CommentaryOut,
    memo_versions: dict[str, Any], usage: dict[str, Any], user_id: int | None, existing: ChartCommentary | None,
) -> int | None:
    """Insert (or refresh a degraded row for the same key). Returns the
    row id, or None when the write failed — a cache miss next time is
    the only consequence, so a storage error never fails the response."""
    payload = out.model_dump(mode="json")
    values = dict(
        fingerprint=out.fingerprint, user_id=user_id, tickers=list(req.tickers), metrics=list(req.metrics),
        years=years, catalog_version=catalog.CATALOG_VERSION, memo_versions=memo_versions, output=payload,
        provider=str(usage.get("provider") or "")[:32], model=str(usage.get("model") or "")[:64],
        tokens_in=int(usage.get("input_tokens") or 0), tokens_out=int(usage.get("output_tokens") or 0),
        degraded=out.degraded, degraded_reason=(out.degraded_reason or "")[:200], created_at=out.generated_at,
    )
    try:
        if existing is not None:
            for k, v in values.items():
                setattr(existing, k, v)
            db.commit()
            return int(existing.id)
        row = ChartCommentary(cache_key=key, **values)
        db.add(row)
        db.commit()
        return int(row.id)
    except IntegrityError:
        # Two replicas generated the same chart at once; the other's row
        # stands and this answer is still the right one to return.
        db.rollback()
        try:
            winner = _lookup(db, key)
            return int(winner.id) if winner is not None else None
        except SQLAlchemyError:
            return None
    except SQLAlchemyError as exc:
        log_safely(log, "chart commentary: could not store the commentary row", exc)
        try:
            db.rollback()
        except SQLAlchemyError:  # pragma: no cover
            pass
        return None


def _from_row(row: ChartCommentary) -> CommentaryOut:
    out = CommentaryOut.model_validate(row.output)
    return out.model_copy(update={"commentary_id": int(row.id), "cache_hit": True})


def _stale_caveat(ticker: str, excerpt: MemoExcerpt) -> str:
    """The one caveat whose truth moves without the cache key moving; kept
    in one place so `_with_live_staleness` can recognise the stored copy."""
    return f"{ticker} memo v{excerpt.version} is stale: {excerpt.stale_reason}."


_STALE_CAVEAT_RE = re.compile(r"^\S+ memo v\d+ is stale: ")


def _with_live_staleness(
    cached: CommentaryOut, excerpts: dict[str, MemoExcerpt | None], request_caveats: list[str],
) -> CommentaryOut:
    """A cache hit with today's staleness verdict instead of the one
    frozen when the row was written.

    The row is keyed on the memo *versions*, and a newer 10-K or
    transcript arriving after generation leaves the version unchanged —
    so the stored `memo_stale=false` would otherwise be served for up to
    90 days while `memo_store.memo_freshness` already says otherwise. The
    excerpts were just recomputed for the key anyway, so overlay their
    verdict per ticker, and rebuild the caveat list as: the caveats this
    request computed (unavailable tickers, missing memos, *current*
    staleness) followed by whatever the generation itself added (drops,
    truncation, model notes), minus the stale caveats of that time. The
    text sentences, versions and `generated_at` are untouched: they are
    what was generated, and the badge says whether to still trust them."""
    memo_view = []
    for item in cached.memo_view:
        e = excerpts.get(item.ticker)
        if e is None or e.version != item.memo_version:
            # Cannot happen while versions are in the key; if it ever
            # does, the stored verdict is the honest one to keep.
            memo_view.append(item)
            continue
        memo_view.append(item.model_copy(update={"memo_stale": e.stale, "memo_stale_reason": e.stale_reason}))
    generation_caveats = [
        c for c in cached.caveats if c not in request_caveats and not _STALE_CAVEAT_RE.match(c)
    ]
    return cached.model_copy(update={"memo_view": memo_view, "caveats": [*request_caveats, *generation_caveats]})


def generate(
    req: CommentaryRequest, principal: Principal, *, request: Request, db: Session, years: int | None,
) -> CommentaryOut:
    """See the module docstring for the order of operations. `years` is
    the plan-capped range the route resolved (None = full history);
    raises `SeriesChanged`, `UsageUnavailable` and `EntitlementError`."""
    now = _utcnow()
    out = recompute_series(req, years=years, db=db)
    observed = observed_items(out)
    caveats: list[str] = []
    for u in out.unavailable:
        caveats.append(f"{u.ticker}: {u.remedy}")

    # 3. The anonymous default: no commentary generation for the
    # logged-out surface unless the operator opted in.
    if principal.is_anon and not settings.fundamentals_anon_commentary:
        return _degraded(out, observed, REASON_NO_ACCOUNT, caveats, now=now)

    # Memo excerpts first: their versions are part of the cache key.
    excerpts = {t: memo_excerpt(t, out, db) for t in req.tickers}
    versions = memos_used(excerpts)
    for t, e in excerpts.items():
        if e is None:
            caveats.append(f"No stored memo for {t}; the memo section cannot cover it. Run research on it to add one.")
        elif e.stale:
            caveats.append(_stale_caveat(t, e))

    key = cache_key(req.tickers, req.metrics, years, out.normalize, out.fingerprint, versions)
    existing = _lookup(db, key)
    if existing is not None and not existing.degraded:
        try:
            cached = _from_row(existing)
        except Exception as exc:  # a row written by an older contract
            log_safely(log, "chart commentary: cached row does not validate; regenerating", exc)
        else:
            return _with_live_staleness(cached, excerpts, caveats)

    # 5. Nothing to interpret, or no way to interpret it.
    if not any(p.value is not None for s in out.series for p in s.points):
        return _degraded(out, observed, REASON_NO_DATA, caveats, now=now)
    if all(e is None for e in excerpts.values()):
        return _degraded(out, observed, REASON_NO_MEMOS, caveats, now=now)
    reason = llm_unavailable_reason()
    if reason is not None:
        return _degraded(out, observed, reason, caveats, now=now)

    built = build_prompt_for(req, out, observed, excerpts)
    for t in built.truncated_tickers:
        caveats.append(f"{t} memo excerpts were shortened to fit the prompt budget.")
    if built.memos_dropped:
        # Without the memos there is nothing for the model to relate.
        caveats.append("Memo excerpts did not fit the prompt budget; no memo commentary was generated.")
        return _degraded(out, observed, REASON_INVALID, caveats, now=now)

    # 6. Charge, then call. The meter and lease live in the database; if
    # it cannot answer, nothing is charged and nothing is generated.
    pk_user = principal.user_id if principal.is_user else None
    try:
        grant: Grant = authorize(
            request, FEATURE, db=db, now=now,
            # The chart itself is the natural idempotency key: a retry of
            # the same chart in the same month replays the reservation
            # rather than charging twice. `EntitlementError` (402/429)
            # propagates to the route untouched.
            idempotency_key=(f"{pk_user}:{FEATURE}:{now:%Y-%m}:{key}" if pk_user is not None else None),
        )
    except SQLAlchemyError as exc:
        log_safely(log, "chart commentary: usage meter unavailable", exc)
        raise UsageUnavailable() from exc

    raw: Any = None
    usage: dict[str, Any] = {}
    try:
        try:
            with llm.llm_call_context(
                agent_name=AGENT_NAME, run_id=f"commentary:{key[:12]}", route="cheap",
                user_id=pk_user, feature=FEATURE,
            ):
                raw = llm.chat_json(
                    built.text, system=prompt_mod.SYSTEM_PROMPT, route="cheap", max_tokens=MAX_TOKENS,
                    model=settings.fundamentals_commentary_model or None,
                )
        except Exception as exc:  # the wrappers return None on failure; belt and braces
            log_safely(log, "chart commentary: LLM call raised", exc)
            raw = None
        usage = llm.last_usage() or {}
        validated = _validate_output(raw, req, excerpts)
    except BaseException:
        # Anything unexpected between the reservation and its outcome
        # must not leave the customer charged (and the lease held) for
        # a commentary they never received.
        grant.release(db)
        raise
    if validated is None:
        grant.release(db)
        degraded_reason = llm_unavailable_reason() or REASON_INVALID
        body = _degraded(out, observed, degraded_reason, caveats, now=now)
        row_id = _store(db, key=key, req=req, years=years, out=body, memo_versions=versions, usage=usage,
                        user_id=pk_user, existing=existing)
        return body.model_copy(update={"commentary_id": row_id})

    grant.commit(db)
    if validated.dropped:
        caveats.append(
            f"{validated.dropped} model sentence(s) referred to a company outside the selection or without a "
            "stored memo and were dropped."
        )
    if validated.overflow:
        caveats.append(f"{validated.overflow} model sentence(s) beyond the per-memo limit were not shown.")
    caveats.extend(f"Model note: {c}" for c in validated.model_caveats)
    body = CommentaryOut(
        commentary_id=None, fingerprint=out.fingerprint, cache_hit=False, degraded=False, degraded_reason=None,
        observed=observed, memo_view=validated.memo_view, caveats=caveats, generated_at=now,
        model=str(usage.get("model") or "") or None,
    )
    row_id = _store(db, key=key, req=req, years=years, out=body, memo_versions=versions, usage=usage,
                    user_id=pk_user, existing=existing)
    return body.model_copy(update={"commentary_id": row_id})
