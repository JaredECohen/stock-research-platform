"""Reflection agent — turns one memo run into long-term memory entries.

Runs after the critic. Decides whether a *delta event* fired this run:
    - new earnings (transcript period not seen in memory)
    - new SEC filing (10-K / 10-Q / 8-K accession not seen in memory)
    - material or breaking news alert in the hot cache
If any did, appends a structured entry to the company memory file. After
the company entry, the sector reflection appends a self-evaluation entry
to the sector memory file (e.g., "AI capex regime call paid off on NVDA;
underweighted utility pull-through again").

LLM-backed when keys are present; deterministic templates fall through in
demo mode so the memory still grows in the absence of an LLM.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..cache import cache_get
from ..config import settings
from ..memory import CompanyMemory, MemoryEntry, SectorMemory
from ..memory.longterm import CrossCompanyPattern
from ..schemas import StockMemoOut
from . import llm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

def _memory_blob(mem: CompanyMemory) -> str:
    """Concatenate every searchable surface in the file so the seen-detector
    catches accession / period mentions whether they were stored as triggers,
    in entry bodies, or already folded into the historical context."""
    parts: List[str] = [mem.historical_context or ""]
    for e in mem.entries:
        parts.append(e.trigger or "")
        parts.append(e.body or "")
    return "\n".join(parts)


def _seen_accessions(mem: CompanyMemory) -> set[str]:
    """Pull accession-like tokens we've already written about.

    SEC's real format is `\\d{7,12}-\\d{2}-\\d{6}` but the demo dataset and
    some providers use shapes like `0001-DEMO-10K`. Match `filing:<token>`
    where <token> is any word-ish run, then also match the bare SEC pattern
    so older entries written without the `filing:` prefix still count.
    """
    out: set[str] = set()
    blob = _memory_blob(mem)
    import re
    # Anything explicitly tagged `filing:<token>` in the entry triggers / body
    for m in re.finditer(r"filing:([A-Za-z0-9_\-\./]+)", blob):
        out.add(m.group(1).rstrip(".,;:)"))
    # SEC-style bare accessions (legacy entries / historical context)
    for m in re.finditer(r"\b\d{7,12}-\d{2}-\d{6}\b", blob):
        out.add(m.group(0))
    return out


def _normalize_period(s: str) -> str:
    """Normalize transcript period strings to a canonical form.

    Accepts `Q3-2024`, `Q3 2024`, `2024Q3`, `2024-Q3`, etc.; emits `2024Q3`.
    Anything that doesn't match the QnYYYY/YYYYQn family is uppercased and
    stripped, so unfamiliar shapes still compare correctly to themselves.
    """
    import re
    t = (s or "").upper().strip()
    m = re.match(r"^Q([1-4])[- ]?(\d{4})$", t)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    m = re.match(r"^(\d{4})[- ]?Q([1-4])$", t)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    return t


def _seen_transcript_periods(mem: CompanyMemory) -> set[str]:
    out: set[str] = set()
    blob = _memory_blob(mem)
    import re
    # Anything explicitly tagged `transcript:<period>` or `earnings:<period>`
    # in the entry triggers / body
    for m in re.finditer(r"(?:transcript|earnings):([A-Za-z0-9\-]+)", blob):
        out.add(_normalize_period(m.group(1)))
    # Bare period tokens (Q3-2024, 2024Q3)
    for m in re.finditer(r"\bQ[1-4][- ]?20\d{2}\b", blob):
        out.add(_normalize_period(m.group(0)))
    for m in re.finditer(r"\b20\d{2}Q[1-4]\b", blob):
        out.add(_normalize_period(m.group(0)))
    return out


def detect_triggers(memo: StockMemoOut, mem: CompanyMemory) -> List[Dict[str, Any]]:
    """Inspect the memo + cache for delta events worth recording.

    Returns a list (possibly empty) of `{kind, label, detail}` dicts. An
    empty list means: no delta this run, do not write to memory.
    """
    triggers: List[Dict[str, Any]] = []

    # Filings — sources_used carries `filing:<accession>` pointers.
    seen_filings = _seen_accessions(mem)
    for src in memo.sources_used:
        if not src.startswith("filing:"):
            continue
        acc = src.split(":", 1)[1].strip()
        if not acc or acc in seen_filings or "-" not in acc:
            continue
        triggers.append({"kind": "filing", "label": f"filing:{acc}", "detail": acc})
        seen_filings.add(acc)

    # Earnings transcript period — sources_used has `transcript:<period>`.
    seen_periods = _seen_transcript_periods(mem)
    for src in memo.sources_used:
        if src.startswith("transcript:"):
            raw = src.split(":", 1)[1].strip()
            period = _normalize_period(raw)
            if period and period not in seen_periods:
                triggers.append({"kind": "earnings", "label": f"earnings:{period}",
                                 "detail": period})
                seen_periods.add(period)

    # Material / breaking news from the hot cache
    hot = cache_get(f"news_hot:{memo.ticker}", "news_hot")
    if hot and isinstance(hot.payload, dict):
        for a in (hot.payload.get("alerts") or []):
            sev = (a.get("severity") or "").lower()
            if sev in ("material", "breaking"):
                triggers.append({
                    "kind": "material_news",
                    "label": f"material_news:{a.get('title', '')[:80]}",
                    "detail": a.get("title") or "",
                })

    # De-duplicate by label so a single run never produces twin entries.
    out: List[Dict[str, Any]] = []
    seen_labels: set[str] = set()
    for t in triggers:
        if t["label"] in seen_labels:
            continue
        out.append(t)
        seen_labels.add(t["label"])
    return out


# ---------------------------------------------------------------------------
# Entry body composition
# ---------------------------------------------------------------------------

def _compose_company_entry(memo: StockMemoOut, triggers: List[Dict[str, Any]]) -> str:
    """LLM if available; deterministic fallback otherwise."""
    if settings.has_llm:
        prompt = (
            "You are an analyst keeping a long-term notebook on a single stock. "
            "From the memo below, write a 4–6 sentence entry capturing: what "
            "actually changed this period, what we got right or wrong vs. the "
            "prior take, what to watch next. Avoid generic boilerplate. "
            "Return JSON: {observation, update_to_thesis, watch_next}."
            f"\n\nTriggers: {json.dumps([t['label'] for t in triggers])}"
            f"\n\nMemo summary:\n{json.dumps(_compact_memo(memo))[:4500]}"
        )
        out = llm.chat_json(prompt, route="cheap")
        if isinstance(out, dict):
            obs = (out.get("observation") or "").strip()
            upd = (out.get("update_to_thesis") or "").strip()
            wn = (out.get("watch_next") or "").strip()
            if obs or upd or wn:
                lines = []
                lines.append(f"**Trigger:** {', '.join(t['label'] for t in triggers) or 'multiple'}")
                if obs:
                    lines.append(f"**Observation:** {obs}")
                if upd:
                    lines.append(f"**Update to thesis:** {upd}")
                if wn:
                    lines.append(f"**Watch next:** {wn}")
                return "\n\n".join(lines)
    # Deterministic
    sector_summary = (memo.sector_agent_view.summary or "").strip()
    val_summary = (memo.valuation_agent_view.summary or "").strip()
    risks = ", ".join(r.title for r in memo.thesis_breakers) or "none flagged"
    return (
        f"**Trigger:** {', '.join(t['label'] for t in triggers) or 'multiple'}\n\n"
        f"**Observation:** {sector_summary[:280]}\n\n"
        f"**Update to thesis:** Rating={memo.rating_label} "
        f"(confidence {int(memo.confidence_score)}). {memo.one_sentence_thesis}\n\n"
        f"**Valuation read:** {val_summary[:200]}\n\n"
        f"**Watch next:** {risks}"
    )


def _compose_sector_entry(memo: StockMemoOut, triggers: List[Dict[str, Any]]) -> str:
    """A sector-level reflection: did the prior heuristics work for THIS name?"""
    if settings.has_llm:
        prompt = (
            "You are a sector analyst maintaining a self-reflection journal. "
            "Look at the memo below and write a 3-4 sentence entry on what "
            "this run reinforced or contradicted in your sector heuristics. "
            "Be specific (regime, KPI, peer outlier). Return JSON: "
            "{reinforced, contradicted, takeaway}."
            f"\n\nMemo summary:\n{json.dumps(_compact_memo(memo))[:4500]}"
        )
        out = llm.chat_json(prompt, route="cheap")
        if isinstance(out, dict):
            r = (out.get("reinforced") or "").strip()
            c = (out.get("contradicted") or "").strip()
            t = (out.get("takeaway") or "").strip()
            if r or c or t:
                parts = [f"**Trigger:** {', '.join(x['label'] for x in triggers) or 'memo run'}"]
                if r:
                    parts.append(f"**Reinforced:** {r}")
                if c:
                    parts.append(f"**Contradicted:** {c}")
                if t:
                    parts.append(f"**Takeaway:** {t}")
                return "\n\n".join(parts)
    # Deterministic
    sector_data = memo.sector_agent_view.data or {}
    regime = sector_data.get("regime", "unknown")
    cross = ", ".join(sector_data.get("cross_sector_relevance", [])[:3]) or "none"
    return (
        f"**Trigger:** {', '.join(t['label'] for t in triggers) or 'memo run'}\n\n"
        f"**Subject:** {memo.ticker} ({memo.sector}) — rating {memo.rating_label}\n\n"
        f"**Regime read:** {regime}\n\n"
        f"**Cross-sector pull-through flagged:** {cross}\n\n"
        f"**Takeaway:** review next quarter whether this sector framing held."
    )


def _compose_cross_company_pattern(
    memo: StockMemoOut, triggers: List[Dict[str, Any]]
) -> Optional[CrossCompanyPattern]:
    """Distill a transferable lesson from this run.

    The pattern's `applies_to` is built from cohort peers + the
    sector agent's `cross_sector_relevance` list, since those are the names
    the lesson is most likely to inform next quarter. Returns None when no
    transferable lesson is identified (e.g., a routine quarter with no
    surprises) so we don't pollute sector memory with noise.
    """
    sector_data = memo.sector_agent_view.data or {}
    cohort = (sector_data.get("cohort") or {}).get("peers", []) or []
    cross = sector_data.get("cross_sector_relevance") or []
    applies_to = sorted({t.upper() for t in (cohort + cross) if t and t.upper() != memo.ticker.upper()})
    if not applies_to:
        return None

    lesson_text: Optional[str] = None
    if settings.has_llm:
        prompt = (
            "From this single-name memo, extract ONE generalizable lesson — "
            "something that would help an analyst reading it think more "
            "clearly about peer companies in the same sector. The lesson "
            "must be transferable (not company-specific facts). If nothing "
            "transferable surfaces, return {\"lesson\": null}. Otherwise: "
            "{\"lesson\": \"...\"} (1-3 sentences, plain text)."
            f"\n\nMemo summary:\n{json.dumps(_compact_memo(memo))[:4500]}"
        )
        out = llm.chat_json(prompt, route="cheap")
        if isinstance(out, dict):
            v = out.get("lesson")
            if isinstance(v, str) and v.strip():
                lesson_text = v.strip()
    if lesson_text is None:
        # Deterministic fallback: only emit a pattern when this run had a
        # real surprise (rating != Neutral OR a thesis-breaker fired).
        rating = (memo.rating_label or "").lower()
        if "neutral" in rating and not memo.thesis_breakers:
            return None
        regime = sector_data.get("regime", "unknown")
        breaker_text = (
            f"Thesis-breaker watch: {memo.thesis_breakers[0].title}. "
            if memo.thesis_breakers else ""
        )
        lesson_text = (
            f"In a '{regime}' regime, a {memo.rating_label} on {memo.ticker} "
            f"hinged on the same KPI mix as the cohort — apply the same "
            f"placement check (cohort quartile + multi-year delta) before "
            f"committing to a peer's rating. {breaker_text}"
            f"Re-examine the thesis on a peer's next earnings cycle."
        ).strip()
    return CrossCompanyPattern(
        date=date.today().isoformat(),
        source_company=memo.ticker.upper(),
        applies_to=applies_to[:8],
        lesson=lesson_text,
    )


def _compact_memo(memo: StockMemoOut) -> Dict[str, Any]:
    """Trim the memo to just the fields useful for reflection prompts."""
    return {
        "ticker": memo.ticker,
        "sector": memo.sector,
        "rating": memo.rating_label,
        "confidence": memo.confidence_score,
        "thesis": memo.one_sentence_thesis,
        "pm_view": memo.final_pm_view,
        "sector": memo.sector_agent_view.summary,
        "earnings": memo.earnings_agent_view.summary,
        "filing": memo.filing_agent_view.summary,
        "valuation": memo.valuation_agent_view.summary,
        "comps": memo.comps_agent_view.summary,
        "macro": memo.macro_sensitivity.summary,
        "key_risks": [r.title for r in memo.key_risks][:5],
        "thesis_breakers": [r.title for r in memo.thesis_breakers][:3],
        "degraded_agents": memo.degraded_agents,
    }


# ---------------------------------------------------------------------------
# LLM-backed condenser (used by CompanyMemory.condense_oldest)
# ---------------------------------------------------------------------------

def _llm_condenser(entries: List[MemoryEntry], existing: str) -> str:
    if not settings.has_llm:
        from ..memory.longterm import _deterministic_summary
        return _deterministic_summary(entries, existing)
    rendered = "\n\n".join(e.render() for e in entries)
    prompt = (
        "Condense the following old long-term-memory entries into 6–10 "
        "bullet lessons. Keep dates, preserve any thesis-breaker call-outs, "
        "and surface what we got wrong. Append to (don't replace) the "
        "existing condensed block. Return plain markdown."
        f"\n\nExisting condensed block:\n{existing or '(none yet)'}"
        f"\n\nOld entries:\n{rendered[:6000]}"
    )
    text = llm.chat_text(prompt, route="cheap")
    if text and text.strip():
        return text.strip()
    from ..memory.longterm import _deterministic_summary
    return _deterministic_summary(entries, existing)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(memo: StockMemoOut) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Inspect the memo, append memory entries when a delta fires.

    Returns `(triggers, written_files)` so callers can log telemetry.
    """
    if not settings.enable_long_term_memory:
        return [], []
    if not memo or not memo.ticker:
        return [], []
    written: List[str] = []
    company_mem = CompanyMemory.for_ticker(memo.ticker)
    triggers = detect_triggers(memo, company_mem)
    if not triggers:
        return [], []

    # Wave 3D: when filings/transcripts triggered, pull structured facts
    # from the Wave 2 history tables and attach them to the entry. The
    # backfill ensures the source rows exist even on first contact (the
    # rest of the pipeline will hit the same provider data so the
    # marginal cost of one more upsert is trivial). Wrapped defensively
    # so a missing source never blocks the memo from logging the trigger.
    structured_facts = None
    try:
        from .fact_extraction import collect_structured_facts
        from ..services.history_service import backfill_ticker
        # Best-effort backfill — quiet no-op if data_service has nothing.
        try:
            backfill_ticker(memo.ticker)
        except Exception as exc:  # pragma: no cover — diagnostic only
            log.debug("history backfill silenced for %s: %s", memo.ticker, exc)
        structured_facts = collect_structured_facts(memo.ticker, triggers)
    except Exception as exc:  # pragma: no cover — fact extraction must never block memory
        log.warning("structured fact extraction failed for %s: %s", memo.ticker, exc)
        structured_facts = None

    # Company entry
    company_body = _compose_company_entry(memo, triggers)
    company_mem.append_entry(MemoryEntry(
        date=date.today().isoformat(),
        trigger=", ".join(t["kind"] for t in triggers),
        body=company_body,
        structured_facts=structured_facts,
    ))
    company_mem.condense_oldest(summarizer=_llm_condenser)
    company_mem.save()
    written.append(str(company_mem.path))

    # Sector entry
    sector_mem = SectorMemory.for_sector(memo.sector or "unknown")
    sector_body = _compose_sector_entry(memo, triggers)
    sector_mem.append_entry(MemoryEntry(
        date=date.today().isoformat(),
        trigger=f"reflection:{memo.ticker}",
        body=sector_body,
    ))
    # Cross-company learning: if this run produced a transferable lesson,
    # drop a CrossCompanyPattern into the sector memory file. The next time
    # the sector agent runs on a peer, that lesson is surfaced in its prompt.
    pattern = _compose_cross_company_pattern(memo, triggers)
    if pattern is not None:
        sector_mem.add_pattern(pattern)
    sector_mem.condense_oldest(summarizer=_llm_condenser)
    sector_mem.save()
    written.append(str(sector_mem.path))

    return triggers, written
