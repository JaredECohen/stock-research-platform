"""Wave 10 — per-company theme exposure scores.

Drives:
- The natural-language screener ("show me names with material AI
  exposure" gets a *real* exposure filter, not just sector tags).
- The cross-sector exposure peers in the comps agent (AMZN / GOOGL /
  MSFT all share AI exposure even though they live in different
  GICS sectors).

Source signals:
- Business description (from `companies.business_description`).
- Recent earnings transcripts (mentions of theme keywords in
  prepared remarks + Q&A).
- News headlines tagged to the ticker (when `EnableNewsThemeTagging`
  is on; today defers to keyword scan).

Wave 10i — scoring is now LLM-judged across the full theme
vocabulary in a single call per ticker. Keyword matching is kept
as a deterministic fallback (no API key, network error). The LLM
returns a score 0-100 per theme + a 1-sentence evidence excerpt;
the deterministic fallback uses keyword hit ratios + raw matches
as evidence.

Output lands in `theme_exposure(ticker, theme, score, evidence)`.
A monthly cron refreshes the universe.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import Company, EarningsTranscript, ThemeExposure

log = logging.getLogger(__name__)


# Curated theme vocabulary — start with the high-conviction ones.
# Add to this list as the screener evolves; per the design review,
# this should grow to ~30 investable themes over time.
THEME_KEYWORDS: Dict[str, List[str]] = {
    "ai_infrastructure": [
        "ai", "artificial intelligence", "machine learning", "data center",
        "gpu", "accelerator", "training", "inference", "foundation model",
        "language model", "generative",
    ],
    "ai_applications": [
        "ai assistant", "ai agent", "copilot", "rag", "embedding",
        "ai feature", "ai-powered", "generative ai", "ml-driven",
    ],
    "energy_transition": [
        "renewable", "solar", "wind", "battery", "ev", "electric vehicle",
        "decarbon", "emission", "clean energy", "grid",
    ],
    "glp1": [
        "glp-1", "glp1", "obesity", "diabetes", "semaglutide", "tirzepatide",
        "wegovy", "ozempic", "mounjaro", "zepbound",
    ],
    "china_consumer": [
        "china consumer", "shanghai", "shenzhen", "double 11", "tier-1 city",
    ],
    "data_center_buildout": [
        "data center", "hyperscaler", "colocation", "rack density", "power",
        "cooling", "interconnect",
    ],
    "cybersecurity": [
        "cybersecurity", "ransomware", "endpoint", "siem", "soc",
        "vulnerability", "zero trust",
    ],
    "weight_loss": [
        "obesity", "weight loss", "metabolic", "endocrinology",
    ],
    "long_rates_sensitivity": [
        "real estate", "reit", "duration", "long-duration", "mortgage",
    ],
    "consumer_credit": [
        "credit card", "buy now pay later", "consumer credit", "delinquency",
    ],
}


def _keyword_score(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """Naive weighted hit count, normalized to 0-100. Caps at one hit
    per keyword to avoid runaway scores from one mention spammed across
    a transcript."""
    if not text:
        return 0.0, []
    low = text.lower()
    hits: List[str] = []
    for kw in keywords:
        if kw in low:
            hits.append(kw)
    raw = min(len(hits) / max(1, len(keywords)) * 100.0, 100.0)
    return raw, hits


def _gather_text(ticker: str, char_cap: int = 30000) -> str:
    """Concatenate the highest-signal text for a ticker: business
    description + last 4 transcripts."""
    pieces: List[str] = []
    with SessionLocal() as db:
        c = db.get(Company, ticker.upper())
        if c is not None and c.business_description:
            pieces.append(c.business_description)
        rows = db.execute(
            select(EarningsTranscript)
            .where(EarningsTranscript.ticker == ticker.upper())
            .order_by(EarningsTranscript.fetched_at.desc())
            .limit(4)
        ).scalars().all()
        for r in rows:
            pieces.append(r.full_text or "")
    text = "\n\n".join(p for p in pieces if p)
    return text[:char_cap]


def _llm_theme_scores(
    ticker: str, text: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Score the ticker against the full theme vocabulary in one
    LLM call. Returns `{theme: {"score": float, "evidence": str}}`
    or None on any failure."""
    from ..config import settings
    if not getattr(settings, "openai_api_key", None) or not text.strip():
        return None
    schema_hint = {
        theme: {
            "score": "0-100 — material exposure (revenue, capex, "
                     "earnings sensitivity, or strategic identity)",
            "evidence": "1 sentence with a specific phrase / segment / "
                        "metric from the text supporting the score",
        }
        for theme in THEME_KEYWORDS.keys()
    }
    prompt = (
        f"Score the company {ticker.upper()}'s material exposure to each "
        "theme below on a 0-100 scale. Score reflects revenue / capex / "
        "earnings sensitivity OR strategic identity — not just keyword "
        "mentions.\n\n"
        "Reading rules:\n"
        "- 0-15: incidental / no exposure (a software firm mentioning "
        "  'AI' once in passing).\n"
        "- 30-50: partial exposure (one segment or driver tied to the "
        "  theme).\n"
        "- 60-80: material exposure (the theme moves earnings or is "
        "  central to capital allocation).\n"
        "- 85-100: defining exposure (the theme IS the company's "
        "  strategic identity, e.g. NVDA ↔ AI infrastructure).\n\n"
        f"Theme vocabulary + return shape:\n{json.dumps(schema_hint, indent=2)}\n\n"
        "Return strict JSON exactly matching the keys above. Each "
        "theme value must be `{score, evidence}`. If exposure is "
        "near-zero, score 0-10 and evidence 'no material exposure'.\n\n"
        f"Source text (business description + recent transcripts):\n{text[:18000]}"
    )
    try:
        from ..agents import llm
        out = llm.chat_json(
            prompt,
            system="You are a buy-side analyst. Be specific and honest.",
            route="cheap",
            model=getattr(settings, "openai_tool_model", None),
            max_tokens=1200,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("theme_exposure LLM call failed for %s: %s", ticker, exc)
        return None
    if not isinstance(out, dict):
        return None
    cleaned: Dict[str, Dict[str, Any]] = {}
    for theme in THEME_KEYWORDS.keys():
        raw = out.get(theme)
        if not isinstance(raw, dict):
            continue
        score = raw.get("score")
        if not isinstance(score, (int, float)):
            continue
        cleaned[theme] = {
            "score": max(0.0, min(100.0, float(score))),
            "evidence": str(raw.get("evidence") or "")[:300],
        }
    return cleaned or None


def compute_for_ticker(ticker: str) -> Dict[str, Any]:
    """Compute exposure scores across the theme vocabulary for one
    ticker. Persists to `theme_exposure`. Returns a summary dict.

    Wave 10i — LLM-judged scoring across all themes in a single call
    when an API key is available; falls back to keyword matching
    deterministically when not. Single LLM call per ticker × ~10
    themes scored ≈ ~$0.002-0.004 per ticker.
    """
    text = _gather_text(ticker)
    if not text.strip():
        return {"ticker": ticker.upper(), "themes_written": 0, "reason": "no_text"}

    llm_scores = _llm_theme_scores(ticker, text)
    used_llm = llm_scores is not None
    written = 0
    with SessionLocal() as db:
        for theme, keywords in THEME_KEYWORDS.items():
            if used_llm and theme in llm_scores:
                payload = llm_scores[theme]
                score = float(payload["score"])
                evidence = [
                    str(payload.get("evidence") or "no material exposure")
                ]
            else:
                score, hits = _keyword_score(text, keywords)
                evidence = [f"keyword: {h}" for h in hits[:5]] or ["no material exposure"]
            existing = db.execute(
                select(ThemeExposure).where(
                    ThemeExposure.ticker == ticker.upper(),
                    ThemeExposure.theme == theme,
                )
            ).scalars().first()
            if existing is None:
                db.add(ThemeExposure(
                    ticker=ticker.upper(),
                    theme=theme,
                    score=score,
                    evidence=evidence,
                    refreshed_at=datetime.utcnow(),
                ))
            else:
                existing.score = score
                existing.evidence = evidence
                existing.refreshed_at = datetime.utcnow()
            written += 1
        db.commit()
    return {
        "ticker": ticker.upper(),
        "themes_written": written,
        "scoring": "llm" if used_llm else "keyword_fallback",
    }


def refresh_universe(*, limit: Optional[int] = None) -> Dict[str, int]:
    """Recompute exposure for the curated screener universe."""
    with SessionLocal() as db:
        q = db.query(Company.ticker).filter(Company.universe_tier == "auto_analysis")
        tickers = [t for (t,) in q.all()]
    if limit is not None:
        tickers = tickers[:limit]
    total = 0
    for t in tickers:
        try:
            r = compute_for_ticker(t)
            total += int(r.get("themes_written") or 0)
        except Exception as exc:  # pragma: no cover
            log.debug("theme refresh failed for %s: %s", t, exc)
    return {"tickers": len(tickers), "rows_written": total}


def top_for_theme(
    theme: str, *, min_score: float = 25.0, limit: int = 25,
) -> List[Dict[str, Any]]:
    """Companies most exposed to a theme — drives the natural-language
    screener filter."""
    with SessionLocal() as db:
        rows = db.execute(
            select(ThemeExposure)
            .where(ThemeExposure.theme == theme, ThemeExposure.score >= min_score)
            .order_by(ThemeExposure.score.desc())
            .limit(limit)
        ).scalars().all()
    return [
        {
            "ticker": r.ticker, "theme": r.theme,
            "score": r.score, "evidence": r.evidence,
        }
        for r in rows
    ]
