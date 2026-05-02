"""Wave 3D — structured fact extraction from filings + transcripts.

When a filing/transcript trigger fires in the reflection step, we lift the
raw text out of the Wave 2 history tables and pull a small, opinionated
schema of structured facts (segment performance, guidance changes, capex
commentary, M&A, leadership changes). The facts ride alongside the
narrative entry on `MemoryEntry.structured_facts` so future memo runs can
read them back as deterministic context, without re-LLMing the same
filing every quarter.

Two layers, mirroring the rest of the platform:
- Deterministic regex-based extractor — always runs, very high precision /
  modest recall. Catches the easy stuff (capex dollar amounts, guidance
  ranges, "appointed", "M&A") so demo-mode memos still produce non-empty
  structured facts without an LLM bill.
- LLM enrichment — when keys are present, the same source text is passed
  to a JSON-strict prompt that supplements the regex finds with the
  qualitative context (which segment a guide change refers to, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import settings
from . import llm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _filing_text(ticker: str, accession: str) -> str:
    """Pull the raw filing text + concatenated parsed sections (whichever
    has content). Returns the empty string when the filing isn't in the
    Wave 2 store yet — caller treats that as "skip extraction"."""
    from ..services.history_service import get_filing_text
    rec = get_filing_text(ticker, accession)
    if not rec:
        return ""
    parts: List[str] = []
    if rec.get("raw_text"):
        parts.append(str(rec["raw_text"]))
    sections = rec.get("sections") or {}
    for k, v in sections.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"## {k}\n{v}")
        elif isinstance(v, list):
            parts.append(f"## {k}\n" + "\n".join(str(x) for x in v))
    return "\n\n".join(parts)


def _transcript_text(ticker: str, period: str) -> str:
    from ..services.history_service import get_transcript
    rec = get_transcript(ticker, period)
    if not rec:
        return ""
    if rec.get("full_text"):
        return str(rec["full_text"])
    blocks = rec.get("blocks") or []
    return "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict))


# ---------------------------------------------------------------------------
# Deterministic regex extractor
# ---------------------------------------------------------------------------

_GUIDANCE_PATTERNS = (
    r"guid(?:ance|ing)\s+(?:to\s+)?[\w\s,]{0,80}(?:\$[\d.]+[BMK]?|\d{1,3}%)",
    r"reaffirm(?:ed|ing)?\s+(?:full[- ]year\s+)?guid",
    r"raise[ds]?\s+(?:full[- ]year\s+)?guid",
    r"lower(?:ed|ing)?\s+(?:full[- ]year\s+)?guid",
)
_CAPEX_PATTERNS = (
    r"capex\s+(?:of\s+)?\$[\d.]+\s*(?:billion|million|B|M|bn|mn)?",
    r"capital\s+expenditures?\s+(?:of\s+)?\$[\d.]+",
    r"infrastructure\s+investment\s+(?:of\s+)?\$[\d.]+",
)
_MA_PATTERNS = (
    r"acqui(?:re|sition)\s+of\s+[A-Z][\w\s]{0,40}",
    r"agreed\s+to\s+acquire\s+[A-Z][\w\s]{0,40}",
    r"definitive\s+agreement\s+to\s+(?:acquire|merge)",
    r"divest(?:ed|iture)?\s+(?:of\s+)?[A-Z][\w\s]{0,40}",
)
_LEADERSHIP_PATTERNS = (
    r"(?:appointed|named|elected)\s+[A-Z][\w\s\.]{2,60}\s+(?:as|to)\s+(?:Chief|CEO|CFO|COO|CTO|President|Chair)",
    r"step(?:ped|ping)\s+down\s+as\s+(?:Chief|CEO|CFO|COO|CTO|President|Chair)",
    r"(?:CEO|CFO|COO|CTO)\s+transition",
    r"(?:retire|resign)\w*\s+(?:as|from)\s+(?:Chief|CEO|CFO|COO|CTO|President|Chair)",
)
_SEGMENT_PATTERNS = (
    r"(?:data\s+center|data-center|cloud|enterprise|gaming|consumer|advertising|automotive)\s+(?:revenue|growth|segment)\s+[\w\s,]{0,60}(?:\d{1,3}%|\$[\d.]+[BMK]?)",
)


def _find_snippets(text: str, patterns: tuple[str, ...], *, max_hits: int = 5,
                   max_len: int = 200) -> List[str]:
    """Return up to `max_hits` non-overlapping matches from `text`.

    Each hit is trimmed to `max_len` chars and includes a small window of
    surrounding context for downstream readability.
    """
    out: List[str] = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 80)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if len(snippet) > max_len:
                snippet = snippet[:max_len].rstrip() + "…"
            if snippet in seen:
                continue
            seen.add(snippet)
            out.append(snippet)
            if len(out) >= max_hits:
                return out
    return out


def deterministic_facts(text: str) -> Dict[str, List[str]]:
    """Regex-based fact pull. Returns `{kind: [snippet, ...]}`.

    Always returns a dict; empty lists when nothing matched. The key set
    is stable so callers can rely on `.get("guidance_changes", [])` etc.
    """
    if not text:
        return {
            "guidance_changes": [], "capex_commentary": [],
            "m_and_a": [], "leadership_changes": [], "segment_signals": [],
        }
    return {
        "guidance_changes": _find_snippets(text, _GUIDANCE_PATTERNS),
        "capex_commentary": _find_snippets(text, _CAPEX_PATTERNS),
        "m_and_a": _find_snippets(text, _MA_PATTERNS),
        "leadership_changes": _find_snippets(text, _LEADERSHIP_PATTERNS),
        "segment_signals": _find_snippets(text, _SEGMENT_PATTERNS),
    }


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

_FACT_PROMPT = (
    "You are extracting structured facts from a single SEC filing or "
    "earnings transcript. Output STRICT JSON with the exact keys below; "
    "values are short string lists. Quote phrases verbatim from the source "
    "when possible — do NOT paraphrase numbers. Empty list when no signal.\n\n"
    "Schema:\n"
    "{\n"
    '  "segment_signals": [...],         # per-segment growth / margin commentary\n'
    '  "guidance_changes": [...],        # raised/lowered/reaffirmed FY guidance, etc.\n'
    '  "capex_commentary": [...],        # capex dollars, infrastructure spend, AI capex\n'
    '  "m_and_a": [...],                 # acquisitions / divestitures / MOUs\n'
    '  "leadership_changes": [...]       # CEO/CFO/COO/board transitions\n'
    "}\n"
    "Return only the JSON object."
)


def _llm_enrich(
    text: str, *, ticker: str, kind: str, source_id: str,
) -> Optional[Dict[str, List[str]]]:
    if not settings.has_llm:
        return None
    prompt = (
        _FACT_PROMPT
        + f"\n\nTicker: {ticker}\nSource: {kind} {source_id}\n\nSource text (truncated):\n"
        + text[:6000]
    )
    try:
        out = llm.chat_json(prompt, route="cheap", model=settings.openai_tool_model)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Fact extraction LLM call failed for %s/%s: %s",
                    ticker, source_id, exc)
        return None
    if not isinstance(out, dict):
        return None
    cleaned: Dict[str, List[str]] = {}
    for k in ("segment_signals", "guidance_changes", "capex_commentary",
              "m_and_a", "leadership_changes"):
        v = out.get(k)
        if isinstance(v, list):
            cleaned[k] = [str(x) for x in v if x][:6]
        else:
            cleaned[k] = []
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _merge(deterministic: Dict[str, List[str]],
           llm_out: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Union deterministic + LLM hits per category, deduping by case-insensitive
    string. Cap each list at 8 entries so the structured block stays readable.
    """
    out: Dict[str, List[str]] = {}
    keys = set(deterministic) | set(llm_out or {})
    for k in keys:
        seen_norm = set()
        merged: List[str] = []
        for src in (deterministic.get(k, []), (llm_out or {}).get(k, [])):
            for s in src:
                norm = re.sub(r"\s+", " ", s.lower()).strip()
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    merged.append(s)
                if len(merged) >= 8:
                    break
        out[k] = merged
    return out


def extract_filing_facts(ticker: str, accession: str) -> Dict[str, Any]:
    """Run the deterministic + LLM extraction on a filing identified by accession."""
    text = _filing_text(ticker, accession)
    if not text.strip():
        return {"source_kind": "filing", "source_id": accession, "facts": {}, "skipped": True}
    base = deterministic_facts(text)
    enriched = _llm_enrich(text, ticker=ticker, kind="filing", source_id=accession)
    return {
        "source_kind": "filing",
        "source_id": accession,
        "facts": _merge(base, enriched),
        "had_llm": enriched is not None,
    }


def extract_transcript_facts(ticker: str, period: str) -> Dict[str, Any]:
    """Run the deterministic + LLM extraction on a transcript identified by period."""
    text = _transcript_text(ticker, period)
    if not text.strip():
        return {"source_kind": "transcript", "source_id": period, "facts": {}, "skipped": True}
    base = deterministic_facts(text)
    enriched = _llm_enrich(text, ticker=ticker, kind="transcript", source_id=period)
    return {
        "source_kind": "transcript",
        "source_id": period,
        "facts": _merge(base, enriched),
        "had_llm": enriched is not None,
    }


def collect_structured_facts(
    ticker: str, triggers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """For a memo's triggers, pull structured facts on every filing/transcript
    trigger and combine them into one dict suitable for `MemoryEntry.structured_facts`.

    Returns None when nothing is extractable (no filing/transcript triggers,
    or every source returned empty). Caller should `safe_call` this to keep
    a flaky LLM call from blocking the memory write.
    """
    pieces: List[Dict[str, Any]] = []
    for t in triggers or []:
        kind = t.get("kind")
        detail = t.get("detail") or t.get("label", "")
        if kind == "filing" and detail:
            pieces.append(extract_filing_facts(ticker, detail))
        elif kind == "earnings" and detail:
            pieces.append(extract_transcript_facts(ticker, detail))
    if not pieces:
        return None
    # Drop pieces that found nothing AND were skipped — but keep ones with
    # empty facts but non-skipped (they recorded that nothing matched, which
    # is still useful provenance).
    pieces = [p for p in pieces if not p.get("skipped")]
    if not pieces:
        return None
    return {"sources": pieces, "extractor_version": 1}
