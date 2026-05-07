"""PM context builder — Wave 10.

Assembles the markdown context block the PM reads on every synthesis
+ chat turn. Combines:

- The PM's own brain file (`memory/pm/notes.md`) — investing principles,
  recent macro takes, lessons.
- The relevant company memory file (`memory/companies/<TICKER>.md`)
  when a ticker is in scope.
- The relevant sector memory file (`memory/sectors/<slug>.md`) when a
  sector is in scope.
- Discretionary research notes routed to the PM agent
  (`research_notes/...` with `applies_to_agents: [pm]`).

Returns a single markdown string ready to splice into the PM system
prompt or user message. Empty string when nothing is loaded — callers
can unconditionally concatenate.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def build_pm_context(
    *,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    profile: Optional[dict] = None,
    max_chars_each: int = 3000,
) -> str:
    """Render the markdown context the PM should read.

    Defensive: every component caught individually so a missing file
    or malformed memory entry can't block a memo.
    """
    blocks: list[str] = []

    # 1) PM brain file — the persistent identity / principles file.
    try:
        from ..memory import PMMemory
        pm = PMMemory.load_pm()
        body = pm.as_prompt_context(max_chars=max_chars_each)
        if body and body.strip():
            blocks.append("## PM brain (memory/pm/notes.md)\n\n" + body.strip())
    except Exception as exc:  # pragma: no cover — never block on memory
        log.debug("PM memory read failed: %s", exc)

    # 2) Company memory for the ticker in scope.
    if ticker:
        try:
            from ..memory import CompanyMemory
            cm = CompanyMemory.for_ticker(ticker)
            body = cm.as_prompt_context(max_chars=max_chars_each)
            if body and body.strip():
                blocks.append(
                    f"## {ticker.upper()} memory (memory/companies/{ticker.upper()}.md)\n\n"
                    + body.strip()
                )
        except Exception as exc:  # pragma: no cover
            log.debug("company memory read failed for %s: %s", ticker, exc)

    # 3) Sector memory.
    if sector:
        try:
            from ..memory import SectorMemory
            sm = SectorMemory.for_sector(sector)
            body = (
                sm.as_prompt_context_for(ticker, max_chars=max_chars_each)
                if ticker else sm.as_prompt_context(max_chars=max_chars_each)
            )
            if body and body.strip():
                blocks.append(f"## {sector} sector memory\n\n" + body.strip())
        except Exception as exc:  # pragma: no cover
            log.debug("sector memory read failed: %s", exc)

    # 4) Discretionary research notes routed to the PM agent.
    try:
        from ..services.research_notes import build_notes_block_for_agent
        notes = build_notes_block_for_agent("pm", profile or {"ticker": ticker, "sector": sector})
        if notes and notes.strip():
            blocks.append("## Research notes (PM-tagged)\n\n" + notes.strip())
    except Exception as exc:  # pragma: no cover
        log.debug("research_notes read failed for pm: %s", exc)

    # 5) Wave 10 — PM self-improvement signals. Latest audit pattern
    # observation tells the PM what failure mode is most common in
    # recent memos (e.g. "you tend to write vague consensus_view
    # fields"). Regime-conditional accuracy tells the PM if the
    # current regime is one where it has historically been wrong
    # ("you've been wrong in recessions; be more cautious").
    try:
        from ..services.mispricing_audit import latest_pattern_observation
        obs = latest_pattern_observation(max_age_days=14)
        if obs and obs.strip():
            blocks.append(
                "## PM self-improvement (most recent audit)\n\n"
                f"_The most common failure mode in your recent memos:_ {obs.strip()}\n\n"
                "_Apply this lesson on the current memo. If the audit is "
                "wrong, you may explicitly disagree — but acknowledge it._"
            )
    except Exception as exc:  # pragma: no cover
        log.debug("mispricing audit read failed: %s", exc)

    try:
        from ..services.calibration_service import regime_conditional_accuracy
        # Only inject when there's a current regime tag worth conditioning on.
        from ..cache import cache_get
        broadcast = cache_get("macro:global", "macro_broadcast")
        current_regime = (
            (broadcast.payload or {}).get("regime")
            if broadcast and isinstance(broadcast.payload, dict) else None
        )
        if current_regime:
            stats = regime_conditional_accuracy(horizon_days=90)
            entry = (stats.get("regimes") or {}).get(current_regime.lower())
            if entry and entry.get("n", 0) >= 3:
                accuracy_pct = (entry.get("accuracy") or 0.0) * 100
                blocks.append(
                    "## PM track-record under current regime\n\n"
                    f"Current macro regime: **{current_regime}**.\n"
                    f"In this regime, your last {entry['n']} memo(s) had "
                    f"a **{accuracy_pct:.0f}%** hit rate at 90d "
                    f"(mean alpha: {(entry.get('mean_alpha') or 0.0)*100:+.1f}%).\n\n"
                    "_Use this calibration: if your regime accuracy is "
                    "low, lean toward lower-confidence ratings or more "
                    "explicit thesis-breakers._"
                )
    except Exception as exc:  # pragma: no cover
        log.debug("regime accuracy read failed: %s", exc)

    if not blocks:
        return ""
    header = (
        "# PM context\n\n"
        "_Read these before synthesizing. They are your second brain — "
        "your prior views, the company's history with you, sector lessons, "
        "and curated notes. Let them shape the synthesis; do not quote "
        "verbatim._"
    )
    return "\n\n---\n\n".join([header, *blocks])
