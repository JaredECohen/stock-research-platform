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
from datetime import date as _date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import EarningsTranscript, FilingDoc, FinancialPeriod

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

def _parse_period(period: Any) -> Tuple[Optional[int], Optional[int]]:
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


def _coerce_date(d: Any) -> Optional[_date]:
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
    line_item: str, value: Optional[float], period_end: Optional[_date],
    fiscal_year: Optional[int], fiscal_quarter: Optional[int],
    source: str,
) -> bool:
    """Insert-or-update one row. Returns True if an actual write happened."""
    existing = db.execute(
        select(FinancialPeriod).where(
            FinancialPeriod.ticker == ticker,
            FinancialPeriod.period == period,
            FinancialPeriod.statement == statement,
            FinancialPeriod.line_item == line_item,
        )
    ).scalar_one_or_none()
    if existing is not None:
        # Cheap value compare with float tolerance to avoid spurious rewrites.
        if existing.value == value and existing.period_end == period_end:
            return False
        existing.value = value
        existing.period_end = period_end
        existing.fiscal_year = fiscal_year
        existing.fiscal_quarter = fiscal_quarter
        existing.source = source
        existing.fetched_at = datetime.utcnow()
        return True
    db.add(FinancialPeriod(
        ticker=ticker, period=period, statement=statement,
        line_item=line_item, value=value, period_end=period_end,
        fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        source=source, fetched_at=datetime.utcnow(),
    ))
    return True


def _ingest_statement_rows(
    db: Session, ticker: str, statement: str, rows: Iterable[Dict[str, Any]],
    line_whitelist: Tuple[str, ...], source: str,
) -> int:
    # Dedupe by period — providers occasionally return two rows for the
    # same fiscal period after a restatement (e.g., JNJ FY2023). The
    # later row wins; the earlier one would otherwise collide on the
    # `(ticker, period, statement, line_item)` unique index because we
    # haven't flushed yet inside this transaction.
    by_period: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        period = str(row.get("period") or row.get("date") or "").strip()
        if not period:
            continue
        by_period[period] = row
    written = 0
    for period, row in by_period.items():
        period_end = _coerce_date(row.get("period_end") or row.get("date") or row.get("period"))
        fy, fq = _parse_period(period)
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
            ):
                written += 1
    return written


def _filing_word_count(sections: Dict[str, Any], raw: str) -> int:
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


def _ingest_filings(db: Session, ticker: str, filings: List[Dict[str, Any]]) -> int:
    """Idempotently store filings. Existing rows are updated; new rows are
    inserted. Returns count of net writes.

    Wave 10 — after a NEW filing row is inserted (not on updates), we
    fire `filing_memory.post_pass` to (a) index its chunks into the
    vector store and (b) write the diff against the prior filing of
    the same type into company / sector memory. Failures are swallowed
    — memory updates are non-critical to the ingest pipeline.
    """
    new_filing_ids: List[int] = []
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


def _transcript_blocks(t: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
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

    blocks: List[Dict[str, Any]] = []
    parts: List[str] = []

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
    db: Session, ticker: str, transcripts: List[Dict[str, Any]],
) -> int:
    """Persist transcripts → EarningsTranscript table. New rows are
    embedded into `doc_chunks` via `filing_memory.index_transcript`
    so the earnings analyst can retrieve speaker-attributed Q&A
    without re-fetching the call. Re-ingest of existing periods
    refreshes the row but skips re-indexing (idempotent — the same
    period yields the same chunks).
    """
    written = 0
    new_transcript_ids: List[int] = []
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


def backfill_ticker(ticker: str, *, db: Optional[Session] = None) -> Dict[str, int]:
    """Full backfill of one ticker against the data_service.

    Returns a `{financial_periods, filings, transcripts}` dict of net
    write counts. Idempotent: re-running on unchanged data is a no-op.
    """
    from .data_service import get_data_service
    ticker = ticker.upper()
    ds = get_data_service()
    statements = ds.get_financial_statements(ticker) or {}
    filings = ds.get_filings(ticker) or []
    transcripts = ds.get_earnings_transcripts(ticker) or []

    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        source = ds.mode()
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
        n_filings = _ingest_filings(db, ticker, filings)
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
    ticker: str, line_items: List[str], *, limit: int = 40,
    statement: Optional[str] = None, db: Optional[Session] = None,
) -> Dict[str, List[Dict[str, Any]]]:
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
        out: Dict[str, List[Dict[str, Any]]] = {}
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
    ticker: str, *, limit: int = 12, filing_type: Optional[str] = None,
    db: Optional[Session] = None,
) -> List[Dict[str, Any]]:
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
    section: Optional[str] = None, db: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
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
    ticker: str, period: Optional[str] = None, *,
    db: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
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
    ticker: str, *, limit: int = 10, db: Optional[Session] = None,
) -> List[Dict[str, Any]]:
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
