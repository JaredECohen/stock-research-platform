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
- Phase 6: the Fundamental Factor Scorecard block when the caller hands
  one in (`scorecard_block`, built by `scorecard_context.prompt_block`).
  Explicit rather than loaded here so the chat and orchestrator callers,
  which have no memo run in scope, add no DB read per turn.

- FEAT-003: the cross-industry snapshot the worker persisted for the PM
  (one indexed row read, rendered to ≤ 2,000 chars) plus a short excerpt
  of the latest ANALYST-WRITTEN Industry Analysis edition for the
  company's own group(s) — only when a snapshot exists; nothing is
  computed here. Template editions never reach the PM (owner decision 1),
  template-filled sections inside an analyst edition read "n/a", and a
  group whose newest week produced no analyst edition is marked
  "not updated this week".

Returns a single markdown string ready to splice into the PM system
prompt or user message. Empty string when nothing is loaded — callers
can unconditionally concatenate.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Budget for the FEAT-003 block: the snapshot render is capped by its own
# renderer; each group excerpt gets this many characters so the PM reads
# the analyst's view and what changed, not a whole edition.
INDUSTRY_EXCERPT_MAX_CHARS = 600
INDUSTRY_BLOCK_MAX_CHARS = 2000


def _clip(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _excerpt_from(code: str, report: dict[str, Any] | None, *, max_chars: int = INDUSTRY_EXCERPT_MAX_CHARS,
                  newer_period: str | None = None) -> dict[str, Any] | None:
    """The parts of the latest ANALYST edition the PM needs: the analyst
    view, what changed since the prior edition, and whether the edition
    carries degradations. ``None`` when the group has no publishable
    edition — the store's readers never return a template edition
    (owner decision 1), so the PM never reads template prose as analysis.

    Inside an analyst edition a section the template filled (the outlook
    view or the what-changed line) is returned as ``""`` — "n/a" in the
    block — never as the template's text. ``not_updated`` is the newer
    week whose refresh finished without a validated analyst edition
    (``newer_period``, from ``last_attempted_periods``), or ``None``."""
    if report is None:
        return None
    from ..services.industry_report_store import hidden_sections

    sections = (report.get("payload") or {}).get("sections") or {}
    hidden = set(hidden_sections(report))

    def text_of(section: str, *keys: str) -> str:
        if section in hidden:
            return ""
        interp = (sections.get(section) or {}).get("interpretation")
        if isinstance(interp, str):
            return interp
        if isinstance(interp, dict):
            for key in keys:
                if interp.get(key):
                    return str(interp[key])
        return ""

    per_field = max(80, max_chars // 2)
    return {
        "code": code,
        "version": report.get("version"),
        "period_key": report.get("period_key"),
        "as_of": report.get("as_of"),
        "analyst_view": _clip(text_of("outlook", "analyst_view", "text"), per_field),
        "what_changed": _clip(text_of("what_changed", "text"), per_field // 2),
        "hidden_sections": sorted(hidden),
        "degraded": list(report.get("degraded") or []),
        "status": report.get("status"),
        "not_updated": (
            newer_period if newer_period and newer_period > str(report.get("period_key") or "") else None
        ),
    }


def _report_excerpt(code: str, *, max_chars: int = INDUSTRY_EXCERPT_MAX_CHARS) -> dict[str, Any] | None:
    """One group's excerpt. Use ``_excerpt_from`` with the batched reads
    when several groups are in scope."""
    from ..services.industry_report_store import last_attempted_periods, latest_publishable
    return _excerpt_from(code, latest_publishable(code), max_chars=max_chars,
                         newer_period=last_attempted_periods([code]).get(code))


def industry_context_payload(
    *, tickers: list[str] | None = None, code: str | None = None,
) -> dict[str, Any]:
    """Stored artifacts only — the latest snapshot rows and report
    excerpts for the groups relevant to ``tickers`` (their own groups plus
    the capped dependency-linked ones) and/or the explicit ``code``. This
    is what the chat tool returns and what the block below renders;
    nothing here fetches prices, runs analytics or calls an LLM.

    The answer reaches a user through chat, so it is PUBLIC: groups are
    named by our labels and addressed by slug, and the whole payload goes
    through ``industry_labels.project_public`` (owner decision 2026-09-24).
    ``code`` accepts a slug or, for old callers, an internal group code."""
    from ..services import gics_registry, industry_labels
    from ..services.industry_report_store import access_policy, last_attempted_periods, latest_publishable_many
    from ..services.industry_snapshot import group_rows, latest_snapshot, relevant_groups_detail

    symbols = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    # The tier this answer is served under. The chat tool itself already
    # sits behind the `pm_chat` feature; carrying the policy means the UI
    # (and slice 4's read routes) read one answer rather than each
    # inventing its own from the setting.
    policy = access_policy()
    access = {
        "surface": "pm_chat",
        "tier": policy["surfaces"]["pm_chat"],
        "enforced": policy["enforced"],
        "latest_report_tier": policy["surfaces"]["latest"],
    }
    try:
        info = gics_registry.active_version()
    except Exception as exc:  # DB unavailable — say so, never guess
        log.debug("industry context: taxonomy read failed: %s", type(exc).__name__)
        info = None
    if info is None:
        public_none: dict[str, Any] = industry_labels.project_public(
            {"status": "taxonomy_not_imported", "tickers": symbols, "code": code, "groups": [], "access": access})
        return public_none

    detail = relevant_groups_detail(symbols, version=info) if symbols else {
        "tickers": [], "own": [], "linked": [], "unmapped": [], "cap": 0, "by_ticker": {},
    }
    codes: list[str] = []
    explicit: dict[str, Any] | None = None
    if code:
        try:
            internal = industry_labels.code_for(code)
        except industry_labels.UnknownLabel:
            internal = str(code)   # the registry words the refusal below
        try:
            node = gics_registry.group(internal, version=info)
            explicit = {"code": node.code, "name": industry_labels.label(node.code)}
            codes.append(node.code)
        except gics_registry.UnknownNode:
            explicit = {"code": str(code), "error": f"industry group {code!r} not found in taxonomy "
                                                    f"{industry_labels.public_version_key(info.version_key)}"}
    for c in detail["own"] + [item["code"] for item in detail["linked"]]:
        if c not in codes:
            codes.append(c)

    snapshot = latest_snapshot(version=info)
    rows = group_rows(snapshot, codes) if snapshot else []
    names = {n.code: industry_labels.label(n.code) for n in gics_registry.industry_groups(version=info)}
    # A constant number of queries for every group's edition — a portfolio
    # question can put a dozen groups in scope and this runs on a web
    # request. Analyst editions only; the attempted periods say which of
    # them a newer week failed to replace.
    editions = latest_publishable_many(codes, version=info) if codes else {}
    attempted = last_attempted_periods(codes, version=info) if codes else {}
    groups = []
    for c in codes:
        entry: dict[str, Any] = {
            "code": c,
            "name": names.get(c),
            "relation": ("requested" if explicit and explicit.get("code") == c else
                         "own" if c in detail["own"] else "linked"),
            "snapshot_row": next((r for r in rows if r.get("code") == c), None) if snapshot else None,
            "report": _excerpt_from(c, editions.get(c), newer_period=attempted.get(c)),
        }
        if entry["relation"] == "linked":
            entry["via"] = next((item["via"] for item in detail["linked"] if item["code"] == c), [])
        groups.append(entry)
    payload = {
        "status": "ok",
        "taxonomy_version": info.version_key,
        "tickers": symbols,
        "code": explicit,
        "by_ticker": detail["by_ticker"],
        "unmapped_tickers": detail["unmapped"],
        "snapshot": {
            "period_key": snapshot.get("period_key"),
            "as_of": snapshot.get("as_of"),
            "macro_regime": ((snapshot.get("payload") or {}).get("regime") or {}).get("macro_regime"),
            "n_missing_groups": len((snapshot.get("payload") or {}).get("missing_groups") or []),
        } if snapshot else {"status": "no_snapshot"},
        "groups": groups,
        "access": access,
        "attribution": industry_labels.PUBLIC_ATTRIBUTION,
        "mapping_caveat": industry_labels.PUBLIC_MAPPING_CAVEAT,
        "note": "stored weekly artifacts (observed statistics + labelled interpretation); scenarios, not recommendations",
    }
    public: dict[str, Any] = industry_labels.project_public(payload)
    return public


def industry_context_block(
    *, ticker: str | None = None, tickers: list[str] | None = None,
    max_chars: int = INDUSTRY_BLOCK_MAX_CHARS,
) -> str:
    """The FEAT-003 markdown block, or ``""`` when no snapshot exists.

    The snapshot render is capped at ``min(max_chars, 2000)``; report
    excerpts follow for the companies' own groups only (the linked
    groups are already lines in the snapshot), so the block is bounded
    by the render cap plus one excerpt per own group."""
    from ..services import industry_labels
    from ..services.industry_report_store import last_attempted_periods, latest_publishable_many
    from ..services.industry_snapshot import latest_snapshot, relevant_groups_detail, render_pm_block

    snapshot = latest_snapshot()
    if snapshot is None:
        return ""
    scope = [t for t in ([ticker] if ticker else []) + list(tickers or []) if t]
    render_cap = min(int(max_chars), INDUSTRY_BLOCK_MAX_CHARS)
    parts = [
        "## Cross-industry snapshot (weekly, persisted; observed data + rule-based reads)",
        render_pm_block(snapshot, max_chars=render_cap),
    ]
    if scope:
        detail = relevant_groups_detail(scope)
        editions = latest_publishable_many(detail["own"]) if detail["own"] else {}
        attempted = last_attempted_periods(detail["own"]) if detail["own"] else {}
        # Groups by OUR label only — no code, no edition number — because
        # the PM writes public memo prose from this block and cannot echo
        # an identifier it never saw (owner decision 2026-09-24). A legacy
        # analyst edition's own prose may still quote one, so the excerpt
        # texts are scrubbed too.
        for code in detail["own"]:
            group = industry_labels.label(code)
            excerpt = _excerpt_from(code, editions.get(code), newer_period=attempted.get(code))
            if excerpt is None:
                parts.append(f"Industry group {group}: no published Industry Analysis edition yet.")
                continue
            degraded = (f" (degraded edition: {', '.join(industry_labels.scrub_text(d) for d in excerpt['degraded'])})"
                        if excerpt["degraded"] else "")
            stale = (
                f" (not updated this week; newest analyst edition is {excerpt['period_key']}, "
                f"the {excerpt['not_updated']} refresh produced none)"
                if excerpt["not_updated"] else ""
            )
            parts.append(
                f"Industry group {group} — edition of {excerpt['period_key']}{degraded}{stale}. "
                f"Analyst view: {industry_labels.scrub_text(excerpt['analyst_view']) or 'n/a'} "
                f"What changed: {industry_labels.scrub_text(excerpt['what_changed']) or 'n/a'}"
            )
        if detail["linked"]:
            parts.append(
                "Linked groups (dependency graph, analyst hypotheses): "
                + "; ".join(f"{industry_labels.label(item['code'])} via {', '.join(v['id'] for v in item['via'])}"
                            for item in detail["linked"])
            )
        if detail["unmapped"]:
            parts.append(
                "No industry-group mapping for: "
                + ", ".join(f"{u['ticker']} ({u['state']})" for u in detail["unmapped"])
            )
    parts.append("_Industry statistics are observed data; analyst views are scenario input, not recommendations._")
    return "\n\n".join(parts)


def build_pm_context(
    *,
    ticker: str | None = None,
    sector: str | None = None,
    profile: dict | None = None,
    max_chars_each: int = 3000,
    scorecard_block: str | None = None,
    tickers: list[str] | None = None,
) -> str:
    """Render the markdown context the PM should read.

    Defensive: every component caught individually so a missing file
    or malformed memory entry can't block a memo.

    `scorecard_block` (Phase 6) is already capped at
    `scorecard_context.PROMPT_BLOCK_MAX_CHARS` (600), well inside
    `max_chars_each`; it is clipped here again so a caller cannot widen
    the budget by handing in a longer string.
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
        from ..services.mispricing_audit import (
            latest_pattern_observation,
        )
        from ..services.mispricing_audit import (
            prompt_fragment as _audit_fragment,
        )
        obs = latest_pattern_observation(max_age_days=14)
        if obs and obs.strip():
            blocks.append(
                "## PM self-improvement (most recent audit)\n\n"
                f"_The most common failure mode in your recent memos:_ {obs.strip()}\n\n"
                "_Apply this lesson on the current memo. If the audit is "
                "wrong, you may explicitly disagree — but acknowledge it._"
            )
        # Wave 10 — also surface a targeted dimension-specific
        # guidance fragment so the PM gets concrete instruction
        # ("be specific" / "differentiate from consensus" /
        # "falsifiers must be concrete") tied to the weakest dimension
        # in the latest audit.
        frag = _audit_fragment(max_age_days=14)
        if frag and frag.strip():
            blocks.append(frag.strip())
    except Exception as exc:  # pragma: no cover
        log.debug("mispricing audit read failed: %s", exc)

    # Specialist reliability — flagged specialists whose historical
    # pull has been correlated with WRONG calls. PM should demand
    # stronger evidence when their pull dominates the rating.
    try:
        from ..services.influence_feedback import reliability_prompt_block
        rel = reliability_prompt_block(lookback=30, threshold=-0.2)
        if rel and rel.strip():
            blocks.append(rel.strip())
    except Exception as exc:  # pragma: no cover
        log.debug("specialist reliability read failed: %s", exc)

    try:
        # Only inject when there's a current regime tag worth conditioning on.
        from ..cache import cache_get
        from ..services.calibration_service import regime_conditional_accuracy
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

    # 6) Phase 6 — the Fundamental Factor Scorecard read for the ticker in
    # scope. Observed rank and model read side by side; the synthesis
    # prompt asks for `scorecard_reconciliation` when they disagree.
    if scorecard_block and scorecard_block.strip():
        try:
            from .scorecard_context import PROMPT_BLOCK_MAX_CHARS
            budget = min(int(max_chars_each), PROMPT_BLOCK_MAX_CHARS)
            blocks.append(
                "## Fundamental scorecard (observed rank + model read)\n\n"
                + scorecard_block.strip()[:budget]
                + "\n\n_A cross-sectional model read is a scenario input. Reconcile "
                "a wide gap with observed figures or lower conviction; do not "
                "move the rating to match it._"
            )
        except Exception as exc:  # pragma: no cover
            log.debug("scorecard block render failed: %s", exc)

    # 7) FEAT-003 — the persisted cross-industry snapshot and the company's
    # own industry-group edition, only when a snapshot exists. `tickers`
    # widens the scope for portfolio-wide questions (chat, portfolio
    # brief); the memo path passes its one ticker. One indexed row read
    # for the snapshot, one SELECT for the classifications; nothing is
    # generated on this path.
    try:
        industry = industry_context_block(
            ticker=ticker, tickers=tickers, max_chars=min(int(max_chars_each), INDUSTRY_BLOCK_MAX_CHARS),
        )
        if industry and industry.strip():
            blocks.append(industry.strip())
    except Exception as exc:  # pragma: no cover — never block a memo on the snapshot
        log.debug("industry snapshot block failed: %s", type(exc).__name__)

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
