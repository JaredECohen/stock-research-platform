"""Curated public samples (FEAT-002, S3).

Two halves, deliberately in one file so the boundary is visible:

**Request side** (everything above the "worker side" marker) is what
`api/routes_public.py` calls. It reads `public_samples` rows, assembles
the page payload, derives the expectations ledger from the stored memo
copy, and computes ETags. It imports nothing from `app.agents`,
`app.providers` or `app.services.data_service` — anonymous traffic must
not be able to cause LLM or provider spend, and the cheapest way to
guarantee that is for the code path to have no route to those modules.
`test_public_samples.py` enforces it with an AST check.

**Worker side** (`build_for_ticker` and the `_build_*` helpers named in
`WORKER_ONLY_FUNCTIONS`) runs inside `monitoring/sample_build_loop.py`
on the worker process. It reads the stored memo and DCF, computes comps,
pulls a price series through the provider cache, and optionally asks the
LLM for a short commentary. Its provider/LLM imports are lazy, inside
those functions only, so importing this module from the web process
stays as cheap as importing the ORM.

Per-kind payloads are capped at `PAYLOAD_CAP_BYTES`. Memo bodies are the
only kind that gets near it, and only because of the long-form agent
reports, which the public copy strips anyway — the marketing page shows
the committee's verdict and reasoning, not eight essays.

A ticker outside `settings.sample_tickers` is never served, even when a
memo exists for it: the allowlist is the product decision about what a
logged-out visitor may read, and a stored memo is not consent.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models.public import PublicSample
from ..models.universe import Company

log = logging.getLogger(__name__)

KINDS: tuple[str, ...] = ("memo", "dcf", "comps", "fundamentals", "prices", "screener_row", "commentary")

PAYLOAD_CAP_BYTES = 1_500_000

# Bump when `assemble()` changes shape so cached ETags roll over even
# though the underlying rows did not change.
_ASSEMBLY_VERSION = "v1"

# The admin "rebuild" trigger crosses the web→worker boundary through the
# same table the samples live in: one control row keyed by a ticker no
# listing can ever match (allowlist entries are upper-cased symbols). The
# request-side readers filter by the allowlist, so it never leaks.
CONTROL_TICKER = "_control"
REBUILD_KIND = "rebuild_request"

# A row whose `built_at` trails the newest row for the same ticker by more
# than this was kept from an earlier build because its kind failed this
# time. Say so instead of presenting it as current.
KEPT_ROW_SLACK = timedelta(days=1)

# Fundamentals series shown on the sample page: a handful of headline
# lines, annual where available, straight from `financial_periods`.
FUNDAMENTAL_LINES: tuple[tuple[str, str], ...] = (
    ("income", "revenue"),
    ("income", "gross_profit"),
    ("income", "operating_income"),
    ("income", "net_income"),
    ("cash", "free_cash_flow"),
)
FUNDAMENTAL_POINTS = 12
PRICE_POINTS = 260

RESEARCH_ONLY_TEXT = (
    "MarketMosaic is for investment research and education only. Sample pages are model "
    "outputs from stored research runs, not recommendations, and not personalized financial, "
    "investment, legal, or tax advice."
)
OBSERVED_VS_INTERPRETATION_TEXT = (
    "Observed cells quote stored data (prices, filings, transcripts). Interpretation cells are "
    "the research committee's view. A blank means the evidence was not obtained, not that it is zero."
)


# ---------------------------------------------------------------------------
# Request side — allowlist, assembly, ETags, expectations ledger
# ---------------------------------------------------------------------------

def allowlist() -> list[str]:
    return settings.sample_tickers_list


def is_listed(ticker: str) -> bool:
    return ticker.upper() in allowlist()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def payload_etag(payload: Any) -> str:
    return hashlib.sha1(canonical_json(payload)).hexdigest()


def disclosures_for(built_at: datetime | None) -> list[str]:
    out = [RESEARCH_ONLY_TEXT, OBSERVED_VS_INTERPRETATION_TEXT]
    if built_at is not None:
        out.append(f"Built from stored research on {built_at.isoformat(timespec='seconds')}Z; not updated in real time.")
    else:
        out.append("This sample has not been built yet.")
    return out


def rows_for(db: Session, ticker: str) -> dict[str, PublicSample]:
    rows = db.execute(select(PublicSample).where(PublicSample.ticker == ticker.upper())).scalars().all()
    return {r.kind: r for r in rows if r.kind in KINDS}


def _company_meta(db: Session, ticker: str, memo_payload: dict | None) -> tuple[str | None, str | None]:
    """Name and sector: from the stored memo copy when built, else from the
    universe table, else unknown (never invented)."""
    if isinstance(memo_payload, dict):
        name = memo_payload.get("company_name")
        sector = memo_payload.get("sector")
        if name or sector:
            return (name or None), (sector or None)
    company = db.get(Company, ticker.upper())
    if company is not None:
        return (company.company_name or None), (company.sector or None)
    return None, None


def list_samples(db: Session) -> list[dict[str, Any]]:
    """One entry per allowlisted ticker, built or not, in allowlist order."""
    out: list[dict[str, Any]] = []
    for ticker in allowlist():
        rows = rows_for(db, ticker)
        built = [k for k in KINDS if k in rows]
        built_at = max((rows[k].built_at for k in built if rows[k].built_at), default=None)
        memo_payload = rows["memo"].payload if "memo" in rows else None
        name, sector = _company_meta(db, ticker, memo_payload)
        out.append({
            "ticker": ticker,
            "company_name": name,
            "sector": sector,
            "built_at": built_at.isoformat() if built_at else None,
            "kinds": built,
        })
    return out


def compute_etag(rows: dict[str, PublicSample]) -> str:
    """Quoted strong ETag over the per-row hashes. Stable across processes
    (rows are the only input) and distinct for the unbuilt state."""
    parts = [_ASSEMBLY_VERSION]
    for kind in KINDS:
        row = rows.get(kind)
        parts.append(f"{kind}:{row.etag if row is not None else '-'}")
    return '"' + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest() + '"'


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """RFC 7232 `If-None-Match`: a comma list of (possibly weak) tags or `*`."""
    if not if_none_match:
        return False
    for raw in if_none_match.split(","):
        candidate = raw.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        if candidate == etag:
            return True
    return False


def assemble(db: Session, ticker: str) -> tuple[dict[str, Any], str]:
    """The `GET /api/public/samples/{ticker}` body and its ETag.

    Listed-but-unbuilt is a 200 with nulls and a `degraded` list, not an
    error: the page renders what exists and says what does not.
    """
    ticker = ticker.upper()
    rows = rows_for(db, ticker)
    degraded: list[str] = []
    built_at_values = [r.built_at for r in rows.values() if r.built_at]
    newest = max(built_at_values) if built_at_values else None

    payload: dict[str, Any] = {"ticker": ticker}
    memo_payload = rows["memo"].payload if "memo" in rows else None
    name, sector = _company_meta(db, ticker, memo_payload)
    payload["company_name"] = name
    payload["sector"] = sector
    payload["built_at"] = newest.isoformat() if newest else None

    for kind in KINDS:
        row = rows.get(kind)
        if row is None:
            payload[kind] = None
            degraded.append(f"{kind}: not built")
            continue
        payload[kind] = row.payload if row.payload else None
        if not row.payload:
            degraded.append(f"{kind}: empty")
        for note in (row.degraded or []):
            if isinstance(note, str) and note not in degraded:
                degraded.append(note)
        if newest is not None and row.built_at is not None and newest - row.built_at > KEPT_ROW_SLACK:
            degraded.append(f"{kind}: kept from an earlier build ({row.built_at.isoformat(timespec='seconds')}Z)")

    memo_dict = payload.get("memo")
    payload["expectations_ledger"] = build_expectations_ledger(memo_dict if isinstance(memo_dict, dict) else None)
    payload["kinds_built"] = [k for k in KINDS if k in rows]
    payload["degraded"] = degraded
    payload["disclosures"] = disclosures_for(newest)
    return payload, compute_etag(rows)


# --- expectations ledger ------------------------------------------------------

def _cell(label: str, value: Any, basis: str, source: str) -> dict[str, Any]:
    return {"label": label, "value": value, "basis": basis, "source": source}


def _available(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "available", "items": items, "reason": None}


def _missing(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "items": [], "reason": reason}


def build_expectations_ledger(memo: dict[str, Any] | None) -> dict[str, Any]:
    """The four expectation columns from the research process
    (docs/research/README.md), derived ONLY from fields the stored memo
    already carries. Every cell is tagged `observed` (quoted data) or
    `interpretation` (the committee's view); a leg the memo does not
    carry is `not_captured` or `n/a` with a reason — never a zero.

      reported consensus   ← mispricing_thesis.consensus_view
      management guidance  ← earnings_agent_view.data.structured.guidance_changes
      price-implied        ← price_at_memo + valuation_verdict.dcf_base_upside
      our forecast         ← mispricing_thesis.our_view / gap / falsifiers
    """
    ledger: dict[str, Any] = {
        "columns": ["reported_consensus", "management_guidance", "price_implied", "our_forecast"],
        "note": OBSERVED_VS_INTERPRETATION_TEXT,
    }
    if not isinstance(memo, dict):
        reason = "no stored memo"
        ledger["reported_consensus"] = _missing("n/a", reason)
        ledger["management_guidance"] = _missing("not_captured", reason)
        ledger["price_implied"] = _missing("n/a", reason)
        ledger["our_forecast"] = _missing("n/a", reason)
        return ledger

    thesis = memo.get("mispricing_thesis") if isinstance(memo.get("mispricing_thesis"), dict) else {}

    # Reported consensus — the PM's restatement of what the street expects.
    # It is written by the committee, so it is an interpretation of
    # consensus rather than a quoted estimate feed.
    consensus = (thesis.get("consensus_view") or "").strip()
    if consensus:
        ledger["reported_consensus"] = _available([
            _cell("Consensus view", consensus, "interpretation", "mispricing_thesis.consensus_view"),
        ])
    else:
        ledger["reported_consensus"] = _missing("not_captured", "memo carries no consensus view")

    # Management guidance — extracted from the transcript, so observed.
    earnings = memo.get("earnings_agent_view") if isinstance(memo.get("earnings_agent_view"), dict) else {}
    data = earnings.get("data") if isinstance(earnings.get("data"), dict) else {}
    structured = data.get("structured") if isinstance(data.get("structured"), dict) else {}
    changes = structured.get("guidance_changes") if isinstance(structured.get("guidance_changes"), list) else []
    items: list[dict[str, Any]] = []
    for ch in changes:
        if not isinstance(ch, dict) or not ch.get("metric"):
            continue
        items.append(_cell(
            str(ch.get("metric")),
            {
                "prior": ch.get("prior"), "current": ch.get("current"),
                "direction": ch.get("direction") or "unclear", "rationale": ch.get("rationale") or "",
            },
            "observed", "earnings_agent_view.data.structured.guidance_changes",
        ))
    if items:
        ledger["management_guidance"] = _available(items)
    else:
        ledger["management_guidance"] = _missing("not_captured", "not captured")

    # Price-implied — the price is observed; what the DCF says about it is
    # the model's reading.
    price = memo.get("price_at_memo")
    verdict = memo.get("valuation_verdict") if isinstance(memo.get("valuation_verdict"), dict) else {}
    upside = verdict.get("dcf_base_upside")
    if not isinstance(price, int | float) or isinstance(price, bool):
        ledger["price_implied"] = _missing("n/a", "memo has no price snapshot")
    elif not isinstance(upside, int | float) or isinstance(upside, bool):
        ledger["price_implied"] = _missing("n/a", "no DCF base-case upside was computed for this memo")
    else:
        price_items = [
            _cell("Price at memo", float(price), "observed", "price_at_memo"),
            _cell("DCF base-case upside vs. price", float(upside), "interpretation", "valuation_verdict.dcf_base_upside"),
        ]
        when = memo.get("price_at_memo_at")
        if when:
            price_items[0]["as_of"] = str(when)
        dcf_summary = memo.get("dcf_summary") if isinstance(memo.get("dcf_summary"), dict) else {}
        implied = dcf_summary.get("base_implied_price")
        if isinstance(implied, int | float) and not isinstance(implied, bool):
            price_items.append(_cell("DCF base-case implied price", float(implied), "interpretation", "dcf_summary.base_implied_price"))
        ledger["price_implied"] = _available(price_items)

    # Our forecast — the committee's view, gap and what would falsify it.
    ours = (thesis.get("our_view") or "").strip()
    gap = (thesis.get("gap") or "").strip()
    falsifiers = [f for f in (thesis.get("falsifiers") or []) if isinstance(f, str) and f.strip()]
    if ours or gap or falsifiers:
        forecast_items = []
        if ours:
            forecast_items.append(_cell("Our view", ours, "interpretation", "mispricing_thesis.our_view"))
        if gap:
            forecast_items.append(_cell("Gap vs. consensus", gap, "interpretation", "mispricing_thesis.gap"))
        if falsifiers:
            forecast_items.append(_cell("What would prove us wrong", falsifiers, "interpretation", "mispricing_thesis.falsifiers"))
        ledger["our_forecast"] = _available(forecast_items)
    else:
        ledger["our_forecast"] = _missing("not_captured", "the committee did not commit a view on this memo")
    return ledger


# --- public memo copy + size cap ---------------------------------------------

def _drop_key_recursive(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {k: _drop_key_recursive(v, key) for k, v in value.items() if k != key}
    if isinstance(value, list):
        return [_drop_key_recursive(v, key) for v in value]
    return value


def _blank_key_recursive(value: Any, key: str, replacement: Any) -> Any:
    """Like `_drop_key_recursive` but keeps the key with an empty value, so
    the result still validates against the schema the frontend expects."""
    if isinstance(value, dict):
        return {k: (replacement if k == key else _blank_key_recursive(v, key, replacement)) for k, v in value.items()}
    if isinstance(value, list):
        return [_blank_key_recursive(v, key, replacement) for v in value]
    return value


def strip_for_public(memo: dict[str, Any]) -> dict[str, Any]:
    """The memo as the marketing page shows it: verdict, bull/bear,
    mispricing thesis and the agent findings — minus the long-form agent
    reports and the diligence-dialog transcript, which are Pro reading and
    most of the bytes."""
    out = _drop_key_recursive(memo, "long_form_report")
    out["round_findings"] = []
    return out


def _memo_reducers() -> list[tuple[str, Any]]:
    """Progressively cheaper memo copies, applied in order until one fits."""
    def no_evidence(m: dict) -> dict:
        return _blank_key_recursive(m, "evidence", [])

    def no_agent_data(m: dict) -> dict:
        out = dict(m)
        for k, v in list(out.items()):
            if isinstance(v, dict) and "agent" in v and "data" in v:
                out[k] = {**v, "data": {}}
        extra = out.get("extra_agent_views")
        if isinstance(extra, dict):
            out["extra_agent_views"] = {k: {**v, "data": {}} for k, v in extra.items() if isinstance(v, dict)}
        return out

    return [("evidence citations", no_evidence), ("agent data blocks", no_agent_data)]


def fit_payload(kind: str, payload: dict[str, Any], *, cap: int = PAYLOAD_CAP_BYTES) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a payload under `cap` bytes plus the notes describing what
    was stripped to get there, or `(None, [reason])` when nothing fits.
    Only the memo has reducers; other kinds either fit or are dropped."""
    notes: list[str] = []
    size = len(canonical_json(payload))
    if size <= cap:
        return payload, notes
    if kind == "memo":
        current = payload
        for label, reducer in _memo_reducers():
            current = reducer(current)
            notes.append(f"memo: stripped {label} to fit the public payload cap")
            if len(canonical_json(current)) <= cap:
                return current, notes
    return None, [f"{kind}: payload exceeds {cap} bytes even after stripping; omitted"]


# --- admin rebuild trigger ----------------------------------------------------

def request_rebuild(db: Session, tickers: list[str] | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Write the control row the worker polls for. Idempotent: a second
    request before the worker picks the first up replaces it (union of
    tickers), which is what an operator clicking twice means."""
    now = now or datetime.utcnow()
    listed = allowlist()
    wanted = [t.upper() for t in (tickers or listed)]
    unknown = sorted({t for t in wanted if t not in listed})
    if unknown:
        raise ValueError(f"not in SAMPLE_TICKERS: {', '.join(unknown)}")
    row = db.get(PublicSample, (CONTROL_TICKER, REBUILD_KIND))
    previous = list((row.payload or {}).get("tickers") or []) if row is not None else []
    merged = [t for t in listed if t in set(previous) | set(wanted)]
    payload = {"tickers": merged, "requested_at": now.isoformat()}
    if row is None:
        row = PublicSample(ticker=CONTROL_TICKER, kind=REBUILD_KIND, payload=payload,
                           etag=payload_etag(payload), built_at=now, built_by="web", degraded=[])
        db.add(row)
    else:
        row.payload = payload
        row.etag = payload_etag(payload)
        row.built_at = now
        row.built_by = "web"
    db.commit()
    return {"queued": True, "tickers": merged, "requested_at": payload["requested_at"]}


def pending_request(db: Session) -> dict[str, Any] | None:
    row = db.get(PublicSample, (CONTROL_TICKER, REBUILD_KIND))
    if row is None:
        return None
    return dict(row.payload or {})


def clear_request(db: Session, *, requested_at: str | None = None) -> bool:
    """Delete the control row — only if it is still the request we served
    (`requested_at` matches), so a request that arrived mid-build is kept
    for the next tick."""
    row = db.get(PublicSample, (CONTROL_TICKER, REBUILD_KIND))
    if row is None:
        return False
    if requested_at is not None and (row.payload or {}).get("requested_at") != requested_at:
        return False
    db.delete(row)
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Worker side — builds rows. Nothing below is reachable from a request.
# ---------------------------------------------------------------------------

# The functions allowed to import from `app.agents`, `app.providers` or
# `app.services.data_service`. `test_public_samples.py` walks this module's
# AST and fails on any such import outside these bodies.
WORKER_ONLY_FUNCTIONS: frozenset[str] = frozenset({
    "build_for_ticker", "_build_comps", "_build_prices", "_build_commentary",
})


def _upsert_row(
    db: Session, ticker: str, kind: str, payload: dict[str, Any], *,
    source_ref: str | None, degraded: list[str], now: datetime, built_by: str,
) -> None:
    row = db.get(PublicSample, (ticker, kind))
    if row is None:
        row = PublicSample(ticker=ticker, kind=kind)
        db.add(row)
    row.payload = payload
    row.source_ref = (source_ref or "")[:64] or None
    row.etag = payload_etag(payload)
    row.built_at = now
    row.built_by = built_by[:32]
    row.degraded = list(degraded)
    db.commit()


def _build_memo(ticker: str, db: Session) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    from .memo_store import latest_memo, memo_to_pydantic

    snap = latest_memo(ticker, db=db)
    if snap is None:
        return None, None, ["memo: no stored memo"]
    memo = memo_to_pydantic(snap).model_dump(mode="json")
    return strip_for_public(memo), f"memo_snapshot:{snap.id}", []


def _build_dcf(ticker: str, db: Session) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    from .dcf_store import latest_version, result_to_pydantic

    row = latest_version(ticker, db=db)
    if row is None:
        return None, None, ["dcf: no stored DCF"]
    result = result_to_pydantic(row)
    if result is None:
        return None, None, ["dcf: stored DCF has no result payload"]
    return result.model_dump(mode="json"), f"dcf_model:{row.id}", []


def _build_comps(ticker: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    # Worker only: comps read every peer's financials through the provider
    # chain and may make one cheap LLM call for exposure peers.
    from .valuation_service import build_comps

    result = build_comps(ticker)
    if result is None:
        return None, None, ["comps: financials incomplete; no comps table"]
    return result.model_dump(mode="json"), "company_warm:comps", []


def _build_fundamentals(ticker: str, db: Session) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    from ..models.documents import FinancialPeriod

    series: list[dict[str, Any]] = []
    for statement, line in FUNDAMENTAL_LINES:
        rows = db.execute(
            select(FinancialPeriod).where(
                FinancialPeriod.ticker == ticker, FinancialPeriod.statement == statement,
                FinancialPeriod.line_item == line,
            )
        ).scalars().all()
        annual = [r for r in rows if r.fiscal_quarter is None]
        chosen = annual or rows
        chosen.sort(key=lambda r: (r.period_end or date.min, r.period))
        points = [
            {"period": r.period, "period_end": r.period_end.isoformat() if r.period_end else None, "value": r.value}
            for r in chosen[-FUNDAMENTAL_POINTS:]
        ]
        if points:
            series.append({"metric": line, "statement": statement, "cadence": "annual" if annual else "quarterly", "points": points})
    if not series:
        return None, None, ["fundamentals: no financial periods stored"]
    return {"series": series}, "financial_periods", []


def _build_prices(ticker: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    # Worker only: goes through the provider cache (1-day TTL), so a weekly
    # build is one provider call per ticker at most.
    from .data_service import get_data_service

    rows = get_data_service().get_price_history(ticker, days=PRICE_POINTS + 10) or []
    points = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        close = r.get("close", r.get("adjusted_close"))
        if r.get("date") is None or not isinstance(close, int | float) or isinstance(close, bool):
            continue
        points.append({"date": str(r["date"])[:10], "close": float(close)})
    if not points:
        return None, None, ["prices: no price history available"]
    return {"points": points[-PRICE_POINTS:]}, "provider_cache:prices", []


def _build_screener_row(ticker: str, db: Session) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    from sqlalchemy import func

    from ..models.universe import ScreenerScore

    # Scores are appended per seed, so the table holds every run; rank
    # against each ticker's newest row only.
    newest_ids = select(func.max(ScreenerScore.id)).group_by(ScreenerScore.ticker)
    scores = db.execute(select(ScreenerScore).where(ScreenerScore.id.in_(newest_ids))).scalars().all()
    latest: dict[str, ScreenerScore] = {s.ticker.upper(): s for s in scores}
    mine = latest.get(ticker)
    if mine is None:
        return None, None, ["screener_row: ticker not in the stored screener scores"]
    ranked = sorted(latest.values(), key=lambda s: s.pm_conviction or 0.0, reverse=True)
    rank = next((i for i, s in enumerate(ranked, start=1) if s.ticker.upper() == ticker), 0)
    company = db.get(Company, ticker)
    row = {
        "rank": rank,
        "ticker": ticker,
        "company_name": company.company_name if company is not None else ticker,
        "sector": company.sector if company is not None else "",
        "pm_score": mine.pm_conviction,
        "quality": mine.quality,
        "growth": mine.growth,
        "valuation": mine.valuation,
        "earnings_momentum": mine.earnings_momentum,
        "risk": mine.risk,
        "macro_fit": mine.macro_fit,
        "one_line_thesis": mine.one_line_thesis or "",
        "main_catalyst": mine.main_catalyst or "",
        "main_risk": mine.main_risk or "",
        "theme": mine.theme,
        "universe_size": len(ranked),
        "scored_at": mine.generated_at.isoformat() if mine.generated_at else None,
    }
    return row, f"screener_score:{mine.id}", []


def _build_commentary(ticker: str, memo: dict[str, Any] | None, *, now: datetime) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """One short LLM paragraph framing the sample. Optional: skipped
    outright when no LLM is configured (CI, demo deployments), and a
    failure or empty answer is a degraded note, never an exception."""
    from ..agents.safe_runner import note_soft

    if not settings.has_llm:
        note_soft("public_samples", "commentary skipped: no LLM configured")
        return None, None, ["commentary: skipped (no LLM configured)"]
    if not memo:
        return None, None, ["commentary: skipped (no stored memo to comment on)"]

    from ..agents import llm

    thesis = memo.get("mispricing_thesis") if isinstance(memo.get("mispricing_thesis"), dict) else {}
    verdict = memo.get("valuation_verdict") if isinstance(memo.get("valuation_verdict"), dict) else {}
    prompt = (
        f"Company: {memo.get('company_name') or ticker} ({ticker}), sector {memo.get('sector') or 'n/a'}.\n"
        f"Rating: {memo.get('rating_label')}; confidence {memo.get('confidence_score')}.\n"
        f"Thesis: {memo.get('one_sentence_thesis') or ''}\n"
        f"Consensus view: {thesis.get('consensus_view') or 'not captured'}\n"
        f"Our view: {thesis.get('our_view') or 'not captured'}\n"
        f"Valuation verdict: {verdict.get('verdict') or 'n/a'} — {verdict.get('summary') or ''}\n\n"
        "Write one paragraph (max 120 words) for a logged-out visitor explaining what this research "
        "memo concluded and what evidence would change the view. No recommendations, no price targets, "
        "no claims about returns."
    )
    system = (
        "You write neutral, research-and-education-only commentary for an investment research tool. "
        "Never give personalized advice or tell the reader to buy or sell."
    )
    try:
        text = llm.chat_text(prompt, system=system, route="cheap", max_tokens=300)
    except Exception as exc:
        note_soft("public_samples", f"commentary failed: {type(exc).__name__}")
        return None, None, [f"commentary: LLM call failed ({type(exc).__name__})"]
    if not text or not text.strip():
        return None, None, ["commentary: LLM returned nothing"]
    return {
        "text": text.strip()[:2000],
        "generated_at": now.isoformat(),
        "model": f"{settings.active_llm_provider}:cheap",
    }, "llm:cheap", []


def build_for_ticker(
    ticker: str, *, db: Session | None = None, now: datetime | None = None, built_by: str = "worker",
) -> dict[str, Any]:
    """Build every kind for one allowlisted ticker and upsert the rows.

    Each kind is independent: a failure in one is recorded as a degraded
    note and the kind's last good row is left in place (the marketing
    page shows something dated rather than nothing). Returns
    `{ticker, built: [kinds], degraded: [notes]}`.
    """
    from ..database import SessionLocal

    ticker = ticker.upper()
    if not is_listed(ticker):
        raise ValueError(f"{ticker} is not in SAMPLE_TICKERS")
    now = now or datetime.utcnow()
    own = db is None
    if own:
        db = SessionLocal()
    built: list[str] = []
    degraded: list[str] = []
    memo_payload: dict[str, Any] | None = None
    try:
        builders: list[tuple[str, Any]] = [
            ("memo", lambda: _build_memo(ticker, db)),
            ("dcf", lambda: _build_dcf(ticker, db)),
            ("comps", lambda: _build_comps(ticker)),
            ("fundamentals", lambda: _build_fundamentals(ticker, db)),
            ("prices", lambda: _build_prices(ticker)),
            ("screener_row", lambda: _build_screener_row(ticker, db)),
            ("commentary", lambda: _build_commentary(ticker, memo_payload, now=now)),
        ]
        for kind, fn in builders:
            try:
                payload, source_ref, notes = fn()
            except Exception as exc:
                # Last good row stays; the failure is visible in the loop's
                # note and on the page as a "kept from an earlier build" flag.
                log.warning("public sample %s/%s failed: %s", ticker, kind, type(exc).__name__, exc_info=True)
                db.rollback()
                degraded.append(f"{kind}: build failed ({type(exc).__name__}); previous row kept")
                continue
            degraded.extend(notes)
            if payload is None:
                continue
            fitted, fit_notes = fit_payload(kind, payload)
            degraded.extend(fit_notes)
            if fitted is None:
                continue
            _upsert_row(db, ticker, kind, fitted, source_ref=source_ref, degraded=fit_notes, now=now, built_by=built_by)
            built.append(kind)
            if kind == "memo":
                memo_payload = fitted
        return {"ticker": ticker, "built": built, "degraded": degraded}
    finally:
        if own:
            db.close()
