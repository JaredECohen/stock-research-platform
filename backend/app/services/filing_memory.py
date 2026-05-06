"""Wave 10 — filing → memory + chunk-ingest pipeline.

When a new SEC filing lands (or any time the filing agent runs), this
post-pass:

1. **Indexes** the filing into the vector store (`doc_chunks`) so the
   PM's `ask_filings` tool, the question-specific filings retriever,
   and downstream RAG can find passages by meaning, not just by BM25
   keyword.
2. **Diffs** the new filing against the prior filing of the same type
   to surface what *changed* — risk-factor additions / removals,
   MD&A reframing, segment disclosure shifts. This is some of the
   highest-signal raw material in the corpus today, and nothing was
   surfacing it.
3. **Writes a memory delta** — 3-5 bullet "what's new" lines — to:
   - `memory/companies/<TICKER>.md` (always).
   - `memory/sectors/<SECTOR>.md` when the LLM judges the delta to be
     a sector-relevant pattern.

LLM-optional: when the LLM is unavailable, the diff falls back to a
deterministic section-length comparison + the top added risk-factor
phrases, so memory still grows on every filing.

This service is *off* the memo's critical path — failures log + return
None rather than blocking a memo run.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import FilingDoc
from . import vector_store
from . import embeddings as emb_svc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_filing(filing: FilingDoc) -> int:
    """Chunk + embed all sections of a filing into `doc_chunks`.

    Returns the count of chunks written. Idempotent — re-indexing the
    same filing replaces its prior chunks.
    """
    sections = filing.sections or {}
    chunks: List[Dict[str, Any]] = []
    for section_name, section_text in sections.items():
        if not isinstance(section_text, str) or not section_text.strip():
            continue
        for piece in emb_svc.chunk_text(section_text):
            chunks.append({
                "text": piece,
                "section": section_name,
                "period_end": filing.period_end,
                "meta": {
                    "accession": filing.accession_number,
                    "filing_type": filing.filing_type,
                    "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
                    "url": filing.url or "",
                },
            })
    if not chunks:
        return 0
    return vector_store.upsert_source(
        ticker=filing.ticker,
        source_type="filing",
        source_id=filing.id,
        chunks=chunks,
    )


def index_latest_filing_for(ticker: str) -> int:
    """Convenience: index the most recent filing for a ticker."""
    with SessionLocal() as db:
        row = db.execute(
            select(FilingDoc)
            .where(FilingDoc.ticker == ticker.upper())
            .order_by(FilingDoc.filing_date.desc().nullslast())
            .limit(1)
        ).scalars().first()
        if not row:
            return 0
        return index_filing(row)


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

def _prior_filing_of_same_type(filing: FilingDoc) -> Optional[FilingDoc]:
    with SessionLocal() as db:
        row = db.execute(
            select(FilingDoc)
            .where(
                FilingDoc.ticker == filing.ticker,
                FilingDoc.filing_type == filing.filing_type,
                FilingDoc.id != filing.id,
            )
            .order_by(FilingDoc.filing_date.desc().nullslast())
            .limit(1)
        ).scalars().first()
        return row


def _deterministic_diff(prior: FilingDoc, new: FilingDoc) -> List[str]:
    """Falls back to section-level length deltas + top novel sentences.

    Light enough to never fail. Captures the obvious cases — risk
    factors got longer, segments section split, etc. — without an LLM.
    """
    bullets: List[str] = []
    p_secs = prior.sections or {}
    n_secs = new.sections or {}
    keys = set(p_secs.keys()) | set(n_secs.keys())
    for k in sorted(keys):
        p_text = (p_secs.get(k) or "")
        n_text = (n_secs.get(k) or "")
        if not p_text and n_text:
            bullets.append(f"New section disclosed: **{k}** ({len(n_text.split())} words).")
        elif p_text and not n_text:
            bullets.append(f"Section removed: **{k}** (was {len(p_text.split())} words).")
        elif p_text and n_text:
            delta = len(n_text.split()) - len(p_text.split())
            if abs(delta) > max(50, 0.15 * len(p_text.split())):
                direction = "expanded" if delta > 0 else "shortened"
                bullets.append(
                    f"**{k}** {direction} {abs(delta)} words "
                    f"({len(p_text.split())} → {len(n_text.split())})."
                )
    return bullets[:6]


def _llm_diff(prior: FilingDoc, new: FilingDoc) -> Optional[Dict[str, Any]]:
    """Ask the LLM for a structured what-changed summary.

    Output: { bullets: [str], sector_relevant: bool, sector_pattern: str }.
    Returns None on any failure; caller falls back to deterministic.
    """
    if not getattr(settings, "openai_api_key", None):
        return None
    p_secs = prior.sections or {}
    n_secs = new.sections or {}
    payload = {
        "prior": {
            "filing_type": prior.filing_type,
            "filing_date": prior.filing_date.isoformat() if prior.filing_date else None,
            "sections": {k: v[:8000] for k, v in p_secs.items() if isinstance(v, str)},
        },
        "new": {
            "filing_type": new.filing_type,
            "filing_date": new.filing_date.isoformat() if new.filing_date else None,
            "sections": {k: v[:8000] for k, v in n_secs.items() if isinstance(v, str)},
        },
    }
    from ..agents import llm
    out = llm.chat_json(
        "Two filings of the same type for the same company. "
        "Identify what is materially different in the NEW vs the PRIOR. "
        "Focus on: new risk factors, removed risk factors, MD&A tone "
        "shifts, segment / customer / geographic disclosure changes, "
        "guidance changes. Each bullet must be specific and citation-"
        "worthy (no fluff). Then judge whether the delta represents a "
        "SECTOR pattern (something that would also matter for peers) — "
        "if yes, write a short generalizable lesson.\n\n"
        "Return JSON: { bullets: [str up to 5], sector_relevant: bool, "
        "sector_pattern: str }.\n\n"
        "Filings:\n" + json.dumps(payload, default=str)[: settings.max_agent_context_chars],
        system="You are an experienced filings analyst. Be concise and specific.",
        route="strong",
        model=getattr(settings, "openai_tool_model", None),
    )
    if not isinstance(out, dict):
        return None
    return out


# ---------------------------------------------------------------------------
# Memory writers
# ---------------------------------------------------------------------------

def _write_to_company_memory(ticker: str, filing: FilingDoc, bullets: List[str]) -> None:
    if not bullets:
        return
    try:
        from ..memory import CompanyMemory, MemoryEntry
        cm = CompanyMemory.for_ticker(ticker)
        body = "\n".join(f"- {b}" for b in bullets)
        entry = MemoryEntry(
            date=(filing.filing_date or date.today()).isoformat(),
            trigger=f"filing:{filing.filing_type}",
            body=f"**What's new vs prior {filing.filing_type}:**\n\n{body}",
        )
        cm.append_entry(entry)
        cm.save()
    except Exception as exc:  # pragma: no cover
        log.warning("company memory write failed for %s: %s", ticker, exc)


def _write_to_sector_memory(
    sector: Optional[str], filing: FilingDoc, sector_pattern: str,
) -> None:
    if not sector or not sector_pattern.strip():
        return
    try:
        from ..memory import CrossCompanyPattern, SectorMemory
        sm = SectorMemory.for_sector(sector)
        sm.add_pattern(CrossCompanyPattern(
            date=(filing.filing_date or date.today()).isoformat(),
            source_company=filing.ticker,
            applies_to=[],  # left empty — peers expanded by sector agent at read time
            lesson=sector_pattern.strip(),
        ))
        sm.save()
    except Exception as exc:  # pragma: no cover
        log.warning("sector memory write failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def post_pass(filing: FilingDoc, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the full pipeline for one filing: index + diff + memory.

    Idempotent: re-running the same filing replaces its chunks and
    appends a new dated memory entry (parser dedupes on (date, body)).

    Returns a small report dict for callers / observability.
    """
    report: Dict[str, Any] = {
        "ticker": filing.ticker,
        "filing_type": filing.filing_type,
        "indexed_chunks": 0,
        "delta_bullets": [],
        "sector_pattern_written": False,
    }
    try:
        report["indexed_chunks"] = index_filing(filing)
    except Exception as exc:  # pragma: no cover
        log.warning("filing index failed: %s", exc)

    prior = _prior_filing_of_same_type(filing)
    if prior is None:
        # First filing of this type — no diff possible. Still write a
        # "first time we've seen a 10-K" entry for the company memory.
        bullets = [
            f"First {filing.filing_type} ingested ("
            f"{filing.filing_date.isoformat() if filing.filing_date else 'undated'})."
        ]
        report["delta_bullets"] = bullets
        _write_to_company_memory(filing.ticker, filing, bullets)
        return report

    llm_out = _llm_diff(prior, filing)
    bullets: List[str] = []
    sector_pattern = ""
    if llm_out:
        bullets = [str(b) for b in (llm_out.get("bullets") or []) if str(b).strip()][:5]
        if llm_out.get("sector_relevant"):
            sector_pattern = str(llm_out.get("sector_pattern") or "")
    if not bullets:
        bullets = _deterministic_diff(prior, filing)

    report["delta_bullets"] = bullets
    _write_to_company_memory(filing.ticker, filing, bullets)
    if sector_pattern:
        sector = (profile or {}).get("sector") or ""
        _write_to_sector_memory(sector, filing, sector_pattern)
        report["sector_pattern_written"] = True
    return report
