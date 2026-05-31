"""Per-ticker geographic footprint resolution.

Three-tier lookup chain:
  1. Curated JSON seed at `data/company_geography.json` — ~30 sector-
     sensitive names. Editable; ground truth wins.
  2. Cache table (`ProviderCache` capability=`geography`) — LLM
     extractions from prior runs. 30-day TTL by default.
  3. LLM extraction from cached 10-K text (`filing_docs.sections.
     properties` first, then `business_description` and `risk_factors`).
     Result written back to (2) so subsequent runs are free.

Return shape:
    {
      "ticker": "PLD",
      "type": "REIT",
      "sub_type": "Industrial REIT",
      "metros": {"LA": 0.22, "NYC": 0.10, ...},
      "states": {"CA": 0.30, "TX": 0.12, ...},
      "notes": "...",
      "source": "seed" | "llm_extraction" | "cache",
    }

Geography is intentionally fuzzy. Weights are relative concentration
hints — the overlay engine renormalizes. None of this is meant as a
substitute for the issuer's own segment disclosure.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from . import provider_cache

log = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "company_geography.json"
_GEOGRAPHY_TTL = 30 * 86400  # 30 days


@lru_cache(maxsize=1)
def _load_seed() -> Dict[str, Dict[str, Any]]:
    try:
        with _SEED_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:  # pragma: no cover
        log.warning("Failed to parse %s: %s", _SEED_PATH, exc)
        return {}
    # Strip the leading `_doc` schema block.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def get_geography(
    ticker: str, *, allow_llm_fallback: bool = True,
) -> Optional[Dict[str, Any]]:
    """Resolve a ticker's geographic footprint via seed -> cache -> LLM."""
    if not ticker:
        return None
    sym = ticker.upper().strip()

    seed = _load_seed().get(sym)
    if seed:
        return {"ticker": sym, "source": "seed", **seed}

    cached = provider_cache.get(
        "geography", sym, ttl_seconds=_GEOGRAPHY_TTL,
    )
    if isinstance(cached, dict) and (cached.get("metros") or cached.get("states")):
        return {"ticker": sym, "source": "cache", **cached}

    if not allow_llm_fallback:
        return None

    extracted = _extract_from_filings(sym)
    if extracted is None:
        return None
    provider_cache.put("geography", sym, extracted)
    return {"ticker": sym, "source": "llm_extraction", **extracted}


def list_seeded_tickers() -> list[str]:
    return sorted(_load_seed().keys())


def _extract_from_filings(ticker: str) -> Optional[Dict[str, Any]]:
    """Pull recent 10-K text and ask the LLM to extract geographic weights.

    Returns the same {type, sub_type, metros, states, notes} shape the
    seed uses, minus the `ticker` / `source` keys (the caller adds those).
    Returns None on every failure path so the caller can fall through to
    "geography unavailable" without raising.
    """
    try:
        from .history_service import get_recent_filings, get_filing_text
        from ..agents import llm
    except Exception:  # pragma: no cover
        return None

    filings = get_recent_filings(ticker, filing_types=["10-K"], limit=1) or []
    if not filings:
        # Fall back to most recent 10-Q if no 10-K available.
        filings = get_recent_filings(ticker, filing_types=["10-Q"], limit=1) or []
    if not filings:
        return None

    accession = filings[0].get("accession_number")
    if not accession:
        return None
    full = get_filing_text(ticker, accession)
    if not full:
        return None

    sections: Dict[str, Any] = full.get("sections") or {}
    raw_text = full.get("raw_text") or ""

    # Pull the most-likely-relevant sections; cap at ~12K characters total
    # to keep the LLM call cheap.
    candidate_chunks = []
    for section_key in ("properties", "business", "business_description", "risk_factors", "mda"):
        text = sections.get(section_key) or ""
        if text:
            candidate_chunks.append(f"## {section_key}\n{text[:4000]}")
    if not candidate_chunks and raw_text:
        candidate_chunks.append(raw_text[:10000])
    if not candidate_chunks:
        return None

    payload = "\n\n".join(candidate_chunks)[:12000]

    prompt = (
        "You are reading a US public company's 10-K. Extract its geographic footprint "
        "as relative concentration weights. Output STRICT JSON with this shape:\n"
        "{\n"
        '  "type": "REIT" | "Retailer" | "Restaurant" | "Bank" | "Utility" | "Telecom" | "Energy" | "Insurance" | "Homebuilder" | "Other",\n'
        '  "sub_type": "<sub-industry phrase>",\n'
        '  "metros": {"<3-letter metro code>": <weight 0-1>, ...},\n'
        '  "states": {"<2-letter state code>": <weight 0-1>, ...},\n'
        '  "notes": "<1-2 sentence rationale>"\n'
        "}\n\n"
        "Rules:\n"
        "- Use metro codes: NYC, LA, SF, CHI, DCA, BOS, MIA, ATL, DAL, HOU, "
        "SEA, DEN, PHX, MIN, POR, LAS, DET, MIA, PHL, PIT, CLT, ORL, TPA, JAX, BHM, CIN.\n"
        "- Use 2-letter US state codes.\n"
        "- Weights need not sum to 1.0; they are relative shares. Omit a region if it is immaterial.\n"
        '- If the company has no meaningful geographic concentration, return empty objects: "metros": {}, "states": {}.\n'
        '- If the filing text does not disclose geography (e.g. a pure-IP software company), return empty objects with a notes string explaining this.\n'
        f"Ticker: {ticker}\n\n"
        f"Filing excerpts:\n{payload}\n"
    )

    try:
        result = llm.chat_json(prompt, route="cheap", max_tokens=1200)
    except Exception as exc:  # pragma: no cover
        log.debug("Geography extraction LLM call failed for %s: %s", ticker, exc)
        return None

    if not isinstance(result, dict):
        return None
    metros = result.get("metros") or {}
    states = result.get("states") or {}
    if not isinstance(metros, dict) or not isinstance(states, dict):
        return None
    return {
        "type": str(result.get("type") or "Other"),
        "sub_type": str(result.get("sub_type") or ""),
        "metros": {str(k).upper(): float(v) for k, v in metros.items()
                   if _coerce_weight(v) is not None},
        "states": {str(k).upper(): float(v) for k, v in states.items()
                   if _coerce_weight(v) is not None},
        "notes": str(result.get("notes") or ""),
    }


def _coerce_weight(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0 or v > 1.5:  # tolerate small over-1 from LLM rounding
        return None
    return v
