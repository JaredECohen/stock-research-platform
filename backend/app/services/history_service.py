"""Wave 2 — Financial history service.

Reads + writes the three new history tables (`FinancialPeriod`,
`FilingDoc`, `EarningsTranscript`) introduced for deeper backtesting and
agent context. The data layer above this still routes through the
provider-aware `data_service`; this service is the *cache + history of
record* on top of it.

Why a separate service vs. extending fundamentals_service:
- `fundamentals_service.get_full_financials` returns whatever the
  provider exposes for the *current* call. It's pull-on-demand and
  scoped to one ticker × all-statements.
- The history service is push-on-backfill: on schedule (or on first
  touch), we flatten provider rows into long-format `FinancialPeriod`
  rows so a 10-year `revenue` query is one indexed SELECT.

Backfill is idempotent — re-running against the same provider data
upserts on `(ticker, period, statement, line_item)` so a daily job
never duplicates rows.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date as _date
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import EarningsTranscript, FilingDoc, FinancialPeriod
from . import scorecard_pit

log = logging.getLogger(__name__)


# Lines we always lift from a provider statement payload. Anything else in
# the row is ignored at backfill time. Vocabulary matches the demo dataset
# / common provider convention so `get_financial_history` returns rows
# keyed by the same names downstream readers (ratios.py, comps_history)
# already use — no per-caller renaming.
_INCOME_LINES = (
    "revenue", "cost_of_revenue", "gross_profit", "r_and_d", "sga",
    "operating_income", "ebit", "ebitda", "net_income", "eps_diluted",
    "weighted_avg_shares_diluted", "interest_expense", "pretax_income",
    "tax_expense",
)
_BALANCE_LINES = (
    "total_assets", "total_liabilities", "shareholders_equity",
    "cash_and_equivalents", "short_term_investments",
    "short_term_debt", "long_term_debt", "total_debt",
    "goodwill", "current_assets", "current_liabilities",
)
_CASH_LINES = (
    "cash_from_operations", "capex", "free_cash_flow",
    "depreciation_and_amortization", "dividends_paid",
    "share_repurchases", "stock_based_compensation",
)


def _ensure_tables(db: Session) -> None:
    """Lazy-create the three Wave 2 tables. Mirrors the pattern in
    cache/snapshots.py so direct importers (tests, scripts) work without
    needing to call `init_db()` first."""
    bind = db.get_bind()
    FinancialPeriod.__table__.create(bind=bind, checkfirst=True)
    FilingDoc.__table__.create(bind=bind, checkfirst=True)
    EarningsTranscript.__table__.create(bind=bind, checkfirst=True)


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _parse_period(period: Any) -> tuple[int | None, int | None]:
    """Best-effort extract `(fiscal_year, fiscal_quarter)` from a period
    label. Accepts `2024Q4`, `2024-Q4`, `FY2024`, `2024`, integers."""
    if period is None:
        return None, None
    s = str(period).upper().replace("-", "").replace(" ", "")
    if "FY" in s:
        s = s.replace("FY", "")
    if "Q" in s:
        try:
            year_part, q_part = s.split("Q", 1)
            return int(year_part), int(q_part[:1])
        except (ValueError, IndexError):
            return None, None
    try:
        return int(s), None
    except ValueError:
        return None, None


def _coerce_date(d: Any) -> _date | None:
    if d is None:
        return None
    if isinstance(d, _date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    try:
        return _date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def _upsert_financial_period(
    db: Session, *, ticker: str, period: str, statement: str,
    line_item: str, value: float | None, period_end: _date | None,
    fiscal_year: int | None, fiscal_quarter: int | None,
    source: str,
    available_at: _date | None = None,
    available_at_source: str | None = None,
    fetched_at: datetime | None = None,
) -> bool:
    """Insert-or-update one row. Returns True if an actual write happened.

    `available_at` / `available_at_source` (Phase 6, point-in-time) are
    written on INSERT only. A restatement overwrites `value` and
    `period_end` but never moves `available_at`: the figure was first
    knowable when it was first published, and re-dating it to the
    restatement would let a historical scorecard "know" a number before
    anyone did. Rows that predate the column keep NULL here and are filled
    by `scorecard_pit.backfill_available_at`.
    """
    existing = db.execute(
        select(FinancialPeriod).where(
            FinancialPeriod.ticker == ticker,
            FinancialPeriod.period == period,
            FinancialPeriod.statement == statement,
            FinancialPeriod.line_item == line_item,
        )
    ).scalar_one_or_none()
    now = fetched_at or datetime.utcnow()
    if existing is not None:
        # Cheap value compare with float tolerance to avoid spurious rewrites.
        if existing.value == value and existing.period_end == period_end:
            return False
        existing.value = value
        existing.period_end = period_end
        existing.fiscal_year = fiscal_year
        existing.fiscal_quarter = fiscal_quarter
        existing.source = source
        existing.fetched_at = now
        return True
    db.add(FinancialPeriod(
        ticker=ticker, period=period, statement=statement,
        line_item=line_item, value=value, period_end=period_end,
        fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        source=source, fetched_at=now,
        available_at=available_at, available_at_source=available_at_source,
    ))
    return True


def _ingest_statement_rows(
    db: Session, ticker: str, statement: str, rows: Iterable[dict[str, Any]],
    line_whitelist: tuple[str, ...], source: str,
) -> int:
    # Dedupe by period — providers occasionally return two rows for the
    # same fiscal period after a restatement (e.g., JNJ FY2023). The
    # later row wins; the earlier one would otherwise collide on the
    # `(ticker, period, statement, line_item)` unique index because we
    # haven't flushed yet inside this transaction.
    by_period: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        period = str(row.get("period") or row.get("date") or "").strip()
        if not period:
            continue
        by_period[period] = row
    written = 0
    # Point-in-time availability is derived once per period row (every
    # line of a statement was published together). The ticker's periodic
    # filings are loaded once here rather than per line; `backfill_ticker`
    # ingests filings BEFORE statements so a first pass can match them.
    filings = scorecard_pit.load_filing_dates(db, ticker) if by_period else []
    fetched_at = datetime.utcnow()
    for period, row in by_period.items():
        period_end = _coerce_date(row.get("period_end") or row.get("date") or row.get("period"))
        fy, fq = _parse_period(period)
        available_at, available_at_source = scorecard_pit.derive_available_at(
            ticker=ticker, period_end=period_end, fiscal_year=fy, fiscal_quarter=fq,
            provider_date=row.get("filing_date") or row.get("accepted_date"),
            fetched_at=fetched_at, filings=filings,
        )
        for line in line_whitelist:
            if line not in row:
                continue
            v = row.get(line)
            try:
                value = None if v is None else float(v)
            except (TypeError, ValueError):
                value = None
            if _upsert_financial_period(
                db, ticker=ticker, period=period, statement=statement,
                line_item=line, value=value, period_end=period_end,
                fiscal_year=fy, fiscal_quarter=fq, source=source,
                available_at=available_at, available_at_source=available_at_source,
                fetched_at=fetched_at,
            ):
                written += 1
    return written


def _filing_word_count(sections: dict[str, Any], raw: str) -> int:
    if raw:
        return len(raw.split())
    total = 0
    for v in (sections or {}).values():
        if isinstance(v, str):
            total += len(v.split())
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    total += len(item.split())
    return total


def _ingest_filings(db: Session, ticker: str, filings: list[dict[str, Any]]) -> int:
    """Idempotently store filings. Returns the count of rows that actually
    changed — an insert, or an update whose content differs from what is
    already stored. Re-ingesting an unchanged filing counts zero and writes
    nothing, `_upsert_financial_period`'s contract.

    That distinction is not cosmetic. This number is what `backfill_ticker`
    reports and what the nightly `history_backfill` note shows, and while it
    counted every UPDATE as a write the note read `filings=1660` every single
    night — 166 tickers x the SEC provider's hard 10-filing cap — against a
    pipeline that had not ingested a new filing since the deployment's first
    day. The provider cache was frozen, the poller was diffing against a
    frozen list, and the only telemetry anyone had said 1,660 writes a night.
    A counter that cannot say "nothing happened" cannot report that anything
    did.

    Wave 10 — after a NEW filing row is inserted (not on updates), we
    fire `filing_memory.post_pass` to (a) index its chunks into the
    vector store and (b) write the diff against the prior filing of
    the same type into company / sector memory. Failures are swallowed
    — memory updates are non-critical to the ingest pipeline.
    """
    new_filing_ids: list[int] = []
    written = 0
    for f in filings or []:
        accession = f.get("accession_number") or f.get("accession") or ""
        if not accession:
            continue
        filing_type = f.get("type") or f.get("filing_type") or "UNKNOWN"
        filing_date = _coerce_date(f.get("filing_date"))
        period_end = _coerce_date(f.get("period_end"))
        url = f.get("url") or ""
        sections = {
            k: f.get(k) for k in (
                "business_description", "risk_factors", "mda",
                "segments", "legal_or_regulatory", "financial_highlights",
            ) if k in f
        }
        raw_text = f.get("raw_text") or ""
        wc = _filing_word_count(sections, raw_text)
        existing = db.execute(
            select(FilingDoc).where(FilingDoc.accession_number == accession)
        ).scalar_one_or_none()
        if existing is not None:
            # Compare before writing, so an unchanged re-ingest is a true
            # no-op rather than a refreshed `fetched_at` counted as a write.
            unchanged = (
                existing.ticker == ticker
                and existing.filing_type == filing_type
                and existing.filing_date == filing_date
                and existing.period_end == period_end
                and existing.raw_text == raw_text
                and existing.sections == sections
                and existing.word_count == wc
                and existing.url == url
            )
            if unchanged:
                continue
            existing.ticker = ticker
            existing.filing_type = filing_type
            existing.filing_date = filing_date
            existing.period_end = period_end
            existing.raw_text = raw_text
            existing.sections = sections
            existing.word_count = wc
            existing.url = url
            existing.fetched_at = datetime.utcnow()
            written += 1
            continue
        new_row = FilingDoc(
            ticker=ticker, accession_number=accession, filing_type=filing_type,
            filing_date=filing_date, period_end=period_end,
            raw_text=raw_text, sections=sections, word_count=wc, url=url,
            fetched_at=datetime.utcnow(),
        )
        db.add(new_row)
        db.flush()  # populate new_row.id without committing the outer txn
        new_filing_ids.append(new_row.id)
        written += 1
    if new_filing_ids:
        # Defer the post-pass until after the outer commit — running it
        # here would happen inside the same session. We just stash the
        # IDs on the session via a hook and fire after commit.
        try:
            from . import filing_memory
            db.flush()
            for fid in new_filing_ids:
                row = db.get(FilingDoc, fid)
                if row is not None:
                    filing_memory.post_pass(row)
        except Exception as exc:  # pragma: no cover — never block ingest
            log.warning("filing_memory post_pass failed: %s", exc)
    return written


def _transcript_blocks(t: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Render a transcript payload into structured blocks + concatenated text.

    Accepts three shapes:
      - `blocks`: a list of `{speaker, role, segment, text}` dicts (demo).
      - `prepared_remarks` / `qa` as a list of dicts (legacy demo).
      - `prepared_remarks` / `qa` as a single concatenated string (live AV).

    The string-input path is the one that bit us: when AV returns
    `prepared_remarks` as one big string, iterating it yields one char
    per "remark" — a 50KB transcript blew up into ~68k single-char
    blocks (5MB JSON per row). Now strings collapse to a single block.
    """
    if isinstance(t.get("blocks"), list) and t["blocks"]:
        blocks = t["blocks"]
        text = "\n".join(
            (b.get("text") or "") for b in blocks if isinstance(b, dict)
        )
        return blocks, text

    blocks: list[dict[str, Any]] = []
    parts: list[str] = []

    def _ingest(field: Any, *, segment: str, default_speaker: str, role: str) -> None:
        if not field:
            return
        if isinstance(field, str):
            # Single concatenated string → one block. Strip to keep the
            # JSON payload tight; the full text is stored separately.
            stripped = field.strip()
            if stripped:
                blocks.append({
                    "speaker": default_speaker, "role": role,
                    "segment": segment, "text": stripped,
                })
                parts.append(stripped)
            return
        if isinstance(field, list):
            for r in field:
                if isinstance(r, dict):
                    speaker = r.get("speaker") or default_speaker
                    text = r.get("text") or r.get("content") or ""
                else:
                    speaker = default_speaker
                    text = str(r)
                blocks.append({
                    "speaker": speaker, "role": role,
                    "segment": segment, "text": text,
                })
                if text:
                    parts.append(str(text))

    _ingest(t.get("prepared_remarks"), segment="prepared_remarks",
            default_speaker="Management", role="exec")
    _ingest(t.get("qa"), segment="qa",
            default_speaker="Analyst", role="analyst")
    return blocks, "\n".join(parts)


def _ingest_transcripts(
    db: Session, ticker: str, transcripts: list[dict[str, Any]],
) -> int:
    """Persist transcripts → EarningsTranscript table. New rows are
    embedded into `doc_chunks` via `filing_memory.index_transcript`
    so the earnings analyst can retrieve speaker-attributed Q&A
    without re-fetching the call. Re-ingest of an existing period that
    has genuinely changed refreshes the row but skips re-indexing
    (idempotent — the same period yields the same chunks); re-ingest of
    an unchanged period writes nothing at all.

    Returns the count of rows that actually changed, for the same reason
    `_ingest_filings` does: the number is telemetry, and one that cannot
    report zero reports nothing.
    """
    written = 0
    new_transcript_ids: list[int] = []
    for t in transcripts or []:
        period = str(t.get("period") or "").strip()
        if not period:
            continue
        fy, fq = _parse_period(period)
        call_date = _coerce_date(t.get("date") or t.get("call_date"))
        blocks, full_text = _transcript_blocks(t)
        wc = len(full_text.split())
        existing = db.execute(
            select(EarningsTranscript).where(
                EarningsTranscript.ticker == ticker,
                EarningsTranscript.period == period,
            )
        ).scalar_one_or_none()
        if existing is not None:
            unchanged = (
                existing.fiscal_year == fy
                and existing.fiscal_quarter == fq
                and existing.call_date == call_date
                and existing.blocks == blocks
                and existing.full_text == full_text
                and existing.word_count == wc
            )
            if unchanged:
                continue
            existing.fiscal_year = fy
            existing.fiscal_quarter = fq
            existing.call_date = call_date
            existing.blocks = blocks
            existing.full_text = full_text
            existing.word_count = wc
            existing.fetched_at = datetime.utcnow()
            written += 1
            continue
        row = EarningsTranscript(
            ticker=ticker, period=period, fiscal_year=fy, fiscal_quarter=fq,
            call_date=call_date, blocks=blocks, full_text=full_text,
            word_count=wc, fetched_at=datetime.utcnow(),
        )
        db.add(row)
        db.flush()  # populate row.id without committing
        new_transcript_ids.append(row.id)
        written += 1
    if new_transcript_ids:
        try:
            from . import filing_memory
            for tid in new_transcript_ids:
                t_row = db.get(EarningsTranscript, tid)
                if t_row is not None:
                    filing_memory.index_transcript(t_row)
        except Exception as exc:  # pragma: no cover — never block ingest
            log.warning("filing_memory.index_transcript failed: %s", exc)
    return written


# The two reads in `backfill_ticker` that cost real money on a miss, and so
# are worth a budget when something fans this function out over the whole
# curated universe. `filings` pulls up to ten document bodies of a few MB
# each and every filing it newly inserts fires `filing_memory.post_pass`
# (embeddings plus an LLM diff); `transcripts` costs four AlphaVantage
# requests. The third read, `get_financial_statements`, is deliberately NOT
# in this list: it is one JSON response with no bodies and no LLM behind it,
# and it is the only thing that ever refreshes fundamentals, so slowing it
# down would trade a cost problem for a correctness one.
_BUDGETED_BACKFILL_CAPABILITIES = ("filings", "transcripts")


def backfill_hits_provider(ticker: str) -> bool:
    """True when `backfill_ticker(ticker, prefer_cached=True)` would still
    have to consult a live provider for one of its expensive reads.

    A caller that sweeps the universe uses this to spend a bounded number of
    cold reads per pass instead of discovering the cost after the fact — see
    `monitoring/history_backfill.py`.
    """
    from .data_service import get_data_service
    ds = get_data_service()
    key = ticker.upper()
    return not all(
        ds.reads_from_cache(cap, key) for cap in _BUDGETED_BACKFILL_CAPABILITIES
    )


def backfill_ticker(
    ticker: str, *, db: Session | None = None, prefer_cached: bool = False,
) -> dict[str, int]:
    """Full backfill of one ticker against the data_service.

    Returns a `{financial_periods, filings, transcripts}` dict of net
    write counts — rows inserted, plus rows whose stored content actually
    differed from the provider's. Idempotent: re-running on unchanged data
    writes nothing and returns zeros for all three, which is what makes the
    nightly `history_backfill` note a usable signal rather than a constant.

    `prefer_cached=True` reads filings and transcripts at whatever age they
    are already cached at, reaching a provider only when there is no cached
    row at all. That is for the nightly universe-wide sweep, which is a
    reconciliation pass and not a freshness driver: filing bodies are
    refreshed by `edgar_poller` invalidating the ticker it saw a new
    accession for, and transcripts by `transcripts_poller`'s event, both of
    which are capped per pass. Left at the default, every other caller —
    a regen job, a user opening a ticker — still gets the capability's own
    TTL applied.
    """
    from .data_service import get_data_service
    ticker = ticker.upper()
    ds = get_data_service()
    statements = ds.get_financial_statements(ticker) or {}
    filings = ds.get_filings(ticker, prefer_cached=prefer_cached) or []
    transcripts = ds.get_earnings_transcripts(
        ticker, prefer_cached=prefer_cached,
    ) or []

    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        source = ds.mode()
        # Filings first: a statement row's point-in-time `available_at`
        # falls back to the matching 10-K / 10-Q filing date when the
        # provider did not supply one, and `_ingest_filings` flushes new
        # rows, so the same pass can see them. Order is otherwise
        # irrelevant — the three ingests share nothing else.
        n_filings = _ingest_filings(db, ticker, filings)
        n_fp = 0
        n_fp += _ingest_statement_rows(
            db, ticker, "income", statements.get("income", []),
            _INCOME_LINES, source,
        )
        n_fp += _ingest_statement_rows(
            db, ticker, "balance", statements.get("balance", []),
            _BALANCE_LINES, source,
        )
        n_fp += _ingest_statement_rows(
            db, ticker, "cash", statements.get("cash", []),
            _CASH_LINES, source,
        )
        n_tx = _ingest_transcripts(db, ticker, transcripts)
        db.commit()
        return {
            "financial_periods": n_fp,
            "filings": n_filings,
            "transcripts": n_tx,
        }
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def get_financial_history(
    ticker: str, line_items: list[str], *, limit: int = 40,
    statement: str | None = None, db: Session | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return long-format history for the requested line items.

    Output shape: `{line_item: [{period, period_end, value, fiscal_year,
    fiscal_quarter}]}` with newest period first. `limit` caps each line's
    series independently. `statement` (optional) restricts to one of
    income/balance/cash for tighter queries.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        out: dict[str, list[dict[str, Any]]] = {}
        for line in line_items:
            stmt = (
                select(FinancialPeriod)
                .where(
                    FinancialPeriod.ticker == ticker.upper(),
                    FinancialPeriod.line_item == line,
                )
            )
            if statement:
                stmt = stmt.where(FinancialPeriod.statement == statement)
            stmt = stmt.order_by(
                FinancialPeriod.period_end.desc().nulls_last(),
                FinancialPeriod.period.desc(),
            ).limit(limit)
            rows = db.execute(stmt).scalars().all()
            out[line] = [
                {
                    "period": r.period,
                    "period_end": r.period_end.isoformat() if r.period_end else None,
                    "value": r.value,
                    "fiscal_year": r.fiscal_year,
                    "fiscal_quarter": r.fiscal_quarter,
                    "currency": r.currency,
                    "statement": r.statement,
                }
                for r in rows
            ]
        return out
    finally:
        if own:
            db.close()


def get_recent_filings(
    ticker: str, *, limit: int = 12, filing_type: str | None = None,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        stmt = select(FilingDoc).where(FilingDoc.ticker == ticker.upper())
        if filing_type:
            stmt = stmt.where(FilingDoc.filing_type == filing_type)
        stmt = stmt.order_by(FilingDoc.filing_date.desc().nulls_last()).limit(limit)
        rows = db.execute(stmt).scalars().all()
        return [
            {
                "accession_number": r.accession_number,
                "ticker": r.ticker,
                "filing_type": r.filing_type,
                "filing_date": r.filing_date.isoformat() if r.filing_date else None,
                "period_end": r.period_end.isoformat() if r.period_end else None,
                "word_count": r.word_count,
                "sections": list((r.sections or {}).keys()),
                "url": r.url,
            }
            for r in rows
        ]
    finally:
        if own:
            db.close()


def get_filing_text(
    ticker: str, accession_number: str, *,
    section: str | None = None, db: Session | None = None,
) -> dict[str, Any] | None:
    """Return the full filing record (or one section's text)."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        row = db.execute(
            select(FilingDoc).where(
                FilingDoc.ticker == ticker.upper(),
                FilingDoc.accession_number == accession_number,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if section is not None:
            return {
                "accession_number": row.accession_number,
                "section": section,
                "text": (row.sections or {}).get(section, ""),
            }
        return {
            "accession_number": row.accession_number,
            "ticker": row.ticker,
            "filing_type": row.filing_type,
            "filing_date": row.filing_date.isoformat() if row.filing_date else None,
            "period_end": row.period_end.isoformat() if row.period_end else None,
            "raw_text": row.raw_text,
            "sections": row.sections,
            "word_count": row.word_count,
            "url": row.url,
        }
    finally:
        if own:
            db.close()


def get_transcript(
    ticker: str, period: str | None = None, *,
    db: Session | None = None,
) -> dict[str, Any] | None:
    """Return the transcript for `period`, or the most recent if `period` is None."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        stmt = select(EarningsTranscript).where(
            EarningsTranscript.ticker == ticker.upper(),
        )
        if period:
            stmt = stmt.where(EarningsTranscript.period == period)
        else:
            stmt = stmt.order_by(
                EarningsTranscript.call_date.desc().nulls_last(),
                EarningsTranscript.period.desc(),
            ).limit(1)
        row = db.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        return {
            "ticker": row.ticker,
            "period": row.period,
            "fiscal_year": row.fiscal_year,
            "fiscal_quarter": row.fiscal_quarter,
            "call_date": row.call_date.isoformat() if row.call_date else None,
            "blocks": row.blocks,
            "full_text": row.full_text,
            "word_count": row.word_count,
        }
    finally:
        if own:
            db.close()


def get_recent_transcripts(
    ticker: str, *, limit: int = 10, db: Session | None = None,
) -> list[dict[str, Any]]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        rows = db.execute(
            select(EarningsTranscript)
            .where(EarningsTranscript.ticker == ticker.upper())
            .order_by(EarningsTranscript.call_date.desc().nulls_last())
            .limit(limit)
        ).scalars().all()
        return [
            {
                "ticker": r.ticker,
                "period": r.period,
                "fiscal_year": r.fiscal_year,
                "fiscal_quarter": r.fiscal_quarter,
                "call_date": r.call_date.isoformat() if r.call_date else None,
                "word_count": r.word_count,
            }
            for r in rows
        ]
    finally:
        if own:
            db.close()
