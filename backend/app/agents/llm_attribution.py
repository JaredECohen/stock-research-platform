"""Who called the LLM, to do what: the action registry, the runtime guard and
the per-call log line (owner, 2026-09-25: "log which agent/llm model does
what action (internal application logs for review)").

Design: `.claude/memory/proposals/design-llm-attribution-logging.md` §4, as
amended by the integration plan 2026-09-25 (slice B7-M1) and its critique.

Every model call names an `action=` registered here. The action carries a
default agent and role (so a call made under no specific context is still
attributed) and a routing *tier* (so the owner's model table is one lookup:
`llm.resolve_action_route`). The runtime guard records a call that names no
registered action; the static sweep (slice A2a) is the primary enforcement,
because most CI runs have no LLM client and never reach the guard's slow
path.

Import-light on purpose: stdlib only at module top, no app imports, so
`config.py`, `llm.py` and a later static sweep can all import it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

ROLES = frozenset({
    "pm", "analyst", "bull", "bear", "risk_reviewer", "critic", "chat", "news",
    "news_impact", "social", "postmortem", "judge", "writer", "utility", "embed",
})

# Routing tiers (integration plan P1). A tier resolves to a model and an
# effort through settings; a tier with nothing configured is today's
# route/model logic exactly, so code defaults change no request.
TIERS = frozenset({"research", "utility", "reviewer", "debate", "chat", "news", "embed"})

# Which effort setting an action reads. "pm" -> LLM_PM_EFFORT (owner item 3
# reserves "high" for PM synthesis and the advocates), "default" ->
# LLM_DEFAULT_EFFORT, the debate/reviewer/chat keys -> their own settings.
EFFORT_KEYS = frozenset({
    "pm", "default", "debate", "debate_research", "reviewer", "recheck", "chat",
})

ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$")
ACTION_MAX_LEN = 40


@dataclass(frozen=True)
class ActionSpec:
    agent: str                 # default agent when the context names none
    role: str                  # member of ROLES
    kind: str = "chat"         # "chat" | "embed" | "sdk"
    tier: str = "utility"      # member of TIERS
    effort_key: str | None = None
    # The failover hop's effort key when it differs from the primary's
    # (the routing table sends analyst.valuation's failover at "high").
    failover_effort_key: str | None = None
    # False for an action whose call site owns its route today and must
    # keep it (the legacy critic: REVIEWER_MODE=legacy is exactly today's
    # bytes). The tier is still reported, never applied.
    routed: bool = True


def _a(agent: str, role: str, tier: str, effort: str | None = None, **kw: Any) -> ActionSpec:
    return ActionSpec(agent=agent, role=role, tier=tier, effort_key=effort, **kw)


# The full registry for the 2026-09-25 program: every existing call site
# (design §8) plus the actions later slices add (debate, reviewer, PM
# revision/counterfactual, memo-time news fetch, model-access probe), so no
# later slice needs to edit this file. Tiers follow the plan's §2 routing
# table and its P2 no-downgrade rule: a call on today's strong route moves
# to research, a cheap one stays utility unless the owner named it.
ACTIONS: dict[str, ActionSpec] = {
    # --- PM -----------------------------------------------------------------
    "pm.intake": _a("PM Intake", "pm", "research", "default"),
    "pm.synthesis": _a("PM Synthesis", "pm", "research", "pm"),
    "pm.revision": _a("PM Revision", "pm", "research", "pm"),
    "pm.counterfactual": _a("PM Counterfactual", "pm", "research", "pm"),
    "pm.critique": _a("PM Critique", "pm", "research", "default"),
    "pm.dcf_adjust": _a("PM DCF Adjuster", "pm", "research", "default"),
    # --- analysts -------------------------------------------------------------
    "analyst.sector": _a("Sector Analyst", "analyst", "research", "default"),
    "analyst.industry_group": _a("Industry Group Analyst", "analyst", "research", "default"),
    "analyst.earnings": _a("Earnings Analyst", "analyst", "research", "default"),
    "analyst.earnings_qa": _a("Earnings Analyst", "analyst", "research", "default"),
    "analyst.filing": _a("Filing Analyst", "analyst", "research", "default"),
    "analyst.valuation": _a("Valuation Analyst", "analyst", "research", "default",
                            failover_effort_key="pm"),
    "analyst.comps": _a("Comps Analyst", "analyst", "research", "default"),
    "analyst.comps_followup": _a("Comps Analyst", "analyst", "research", "default"),
    "analyst.macro": _a("Macro Analyst", "analyst", "research", "default"),
    "analyst.macro_regime": _a("Macro Analyst", "analyst", "research", "default"),
    "analyst.risk_breakers": _a("Risk Analyst", "analyst", "research", "default"),
    "analyst.risk_followup": _a("Risk Analyst", "analyst", "research", "default"),
    "analyst.technical": _a("Technical Analyst", "analyst", "research", "default"),
    # --- research / other -----------------------------------------------------
    "news.impact": _a("News Impact", "news_impact", "research", "default"),
    "postmortem.review": _a("Postmortem", "postmortem", "research", "default"),
    "postmortem.lesson": _a("Postmortem", "postmortem", "research", "default"),
    "industry.report": _a("Industry Report Writer", "writer", "research", "default"),
    "dcf.update": _a("DCF Updater", "utility", "research", "default"),
    "chat.answer": _a("PM Chat", "chat", "research", "default"),
    # --- debate (B8-D3) -------------------------------------------------------
    "debate.bull_research": _a("Bull Advocate", "bull", "debate", "debate_research"),
    "debate.bear_research": _a("Bear Advocate", "bear", "debate", "debate_research"),
    "debate.bull_open": _a("Bull Advocate", "bull", "debate", "debate"),
    "debate.bear_open": _a("Bear Advocate", "bear", "debate", "debate"),
    "debate.bull_rebut": _a("Bull Advocate", "bull", "debate", "debate"),
    "debate.bear_rebut": _a("Bear Advocate", "bear", "debate", "debate"),
    # --- reviewer ---------------------------------------------------------------
    "risk.review": _a("Risk Reviewer", "risk_reviewer", "reviewer", "reviewer"),
    "review.recheck": _a("Risk Reviewer", "risk_reviewer", "reviewer", "recheck"),
    "risk.committee_review": _a("Risk Committee", "critic", "reviewer", routed=False),
    # --- chat -------------------------------------------------------------------
    "chat.sdk_turn": _a("PM Chat", "chat", "chat", "chat", kind="sdk"),
    "chat.classify": _a("PM Chat Router", "chat", "utility"),
    "memo.sdk_exchange": _a("SDK PM", "pm", "utility", kind="sdk", routed=False),
    # --- news / social (Gemini) -------------------------------------------------
    "news.search": _a("News Agent", "news", "news"),
    "news.memo_fetch": _a("News Agent", "news", "news"),
    # No LLM call in live mode (plan P13); registered for the demo path.
    "social.sentiment": _a("Social Agent", "social", "news"),
    # --- utility (today's cheap route) ------------------------------------------
    "macro.scenario": _a("Macro Scenario", "analyst", "utility"),
    "memo.long_form": _a("Long-form Writer", "utility", "utility"),
    "memo.earnings_qoq": _a("Earnings QoQ", "utility", "utility"),
    "memo.reflect_company": _a("Reflection", "utility", "utility"),
    "memo.reflect_sector": _a("Reflection", "utility", "utility"),
    "memo.reflect_pattern": _a("Reflection", "utility", "utility"),
    "memo.memory_condense": _a("Reflection", "utility", "utility"),
    "memo.fact_extract": _a("Fact Extraction", "utility", "utility"),
    "dcf.exposure_peers": _a("Valuation Service", "utility", "utility"),
    "dcf.sector_defaults": _a("DCF Defaults", "utility", "utility"),
    "dcf.scenarios": _a("Scenario Builder", "utility", "utility"),
    "screener.translate": _a("NL Screener", "utility", "utility"),
    "portfolio.brief": _a("Portfolio Brief", "utility", "utility"),
    "geography.extract": _a("Geography Extractor", "utility", "utility"),
    "filing.delta": _a("Filing Memory", "utility", "utility"),
    "filing.digest_weekly": _a("Filing Digest", "utility", "utility"),
    "filing.digest_sector": _a("Sector Digest", "utility", "utility"),
    "theme.exposure": _a("Theme Exposure", "utility", "utility"),
    "mispricing.audit": _a("Mispricing Audit", "utility", "utility"),
    "chart.commentary": _a("Chart Commentary", "utility", "utility"),
    "samples.commentary": _a("Sample Commentary", "utility", "utility"),
    "learning.judge": _a("Learning Judge", "judge", "utility"),
    "notes.summarize": _a("Notes Indexer", "utility", "utility"),
    "ops.validate_model_access": _a("Model Access Check", "utility", "utility"),
    # --- embeddings -------------------------------------------------------------
    "embed.index": _a("Embedder", "embed", "embed", kind="embed"),
    "embed.query": _a("Embedder", "embed", "embed", kind="embed"),
    "embed.repair": _a("Embedder", "embed", "embed", kind="embed"),
}

CHAT_ENTRY_POINTS = ("chat_json", "chat_text", "gemini_chat_json", "gemini_chat_text")
EMBED_ENTRY_POINTS = ("embed", "embed_one")
# Injected callables that stand for chat_json (checked by the A2a sweep).
INDIRECT_ENTRY_SITES = frozenset({("app/learning/ledger.py", "chat")})

# The only functions allowed to open `llm_call_context(agent_name=...)`
# (attribution critique #1). An umbrella context — a chat turn, a route, a
# loop, a worker, a script — sets origin/job/run_id only; if it named an
# agent, every specialist call nested under it would be credited to the
# umbrella. The A2a static sweep fails on any other agent_name site.
AGENT_CONTEXT_SITES = frozenset({
    ("app/agents/graph.py", "_run_analyst_round"),
    ("app/agents/graph.py", "_compose_memo"),
    ("app/agents/graph.py", "_review_memo"),
    ("app/agents/deep_research.py", "pm_critique"),
    ("app/agents/deep_research.py", "run_dialog_loop"),
    ("app/agents/dcf_pm_adjuster.py", "_propose_adjustments"),
    ("app/agents/industry_report_writer.py", "_llm_call"),
    ("app/services/industry_report_worker.py", "_run_group_report"),
    ("app/services/chart_commentary.py", "generate"),
    ("app/learning/ledger.py", "judge_due"),
    ("app/agents/long_form.py", "_enriched_long_form"),
})

# Context agent names that say nothing about who acted: the registry's
# default agent wins over them.
UMBRELLA_AGENTS = frozenset({"", "unknown", "run_stock_memo", "unattributed"})
UNATTRIBUTED = "unattributed"

ATTRIBUTION_MODES = ("strict", "warn", "off")


class LLMAttributionError(RuntimeError):
    """An LLM call named no registered action while the guard is strict."""


def spec_for(action: str | None) -> ActionSpec | None:
    return ACTIONS.get(action) if action else None


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------

# Bounded per-process record of unattributed call sites: site -> count.
# Never a per-call list (a 512 MiB worker runs for days), never served by an
# endpoint (it would describe only the process that answered); the DB rows
# with agent="unattributed" are the cross-process signal.
VIOLATIONS: dict[str, int] = {}
VIOLATIONS_MAX_SITES = 256
_WARNED: set[tuple[str, str]] = set()

# Files whose frames are the LLM layer itself, skipped when looking for the
# caller. contextlib because the public entries open their scope through a
# @contextmanager.
_LAYER_SUFFIXES = (
    "/app/agents/llm.py",
    "/app/agents/llm_attribution.py",
    "/app/services/embeddings.py",
    "/contextlib.py",
)
_TEST_SEGMENT = "/app/tests/"


def _rel(path: str) -> str:
    p = path.replace("\\", "/")
    i = p.rfind("/app/")
    return p[i + 1:] if i >= 0 else os.path.basename(p)


def caller_site() -> tuple[str, bool]:
    """(`<rel path>:<function>`, is_test) of the first frame outside the
    LLM layer. Only reached on the unattributed slow path."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = (frame.f_code.co_filename or "").replace("\\", "/")
        if not filename.endswith(_LAYER_SUFFIXES):
            return f"{_rel(filename)}:{frame.f_code.co_name}", _TEST_SEGMENT in filename
        frame = frame.f_back  # type: ignore[assignment]
    return "<unknown>:<unknown>", False


def check(entry: str, action: str | None, *, mode: str) -> bool:
    """Guard one public LLM entry. Returns True when the call is attributed
    (a registered action, or a direct unit test of the LLM layer).

    `mode` is the EFFECTIVE mode (`llm.attribution_mode()`, which forces
    "warn" in production): strict raises, warn logs one WARNING per
    (entry, site) per process, off records nothing.
    """
    if action and action in ACTIONS:
        return True
    site, is_test = caller_site()
    if is_test:
        # A test that calls chat_json directly is testing the LLM layer,
        # not a production call site.
        return True
    if mode == "off":
        return False
    if site in VIOLATIONS or len(VIOLATIONS) < VIOLATIONS_MAX_SITES:
        VIOLATIONS[site] = VIOLATIONS.get(site, 0) + 1
    shown = action if action else "-"
    if mode == "strict":
        raise LLMAttributionError(
            f"LLM call without a registered action: entry={entry} site={site} action={shown}"
        )
    key = (entry, site)
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(
            "llm_attribution_missing entry=%s site=%s action=%s",
            entry, sanitize(site), sanitize(shown),
        )
    return False


def reset_violations() -> None:
    """Test helper: forget recorded sites and the once-per-site warnings."""
    VIOLATIONS.clear()
    _WARNED.clear()


# ---------------------------------------------------------------------------
# The per-attempt log line (design §4.6)
# ---------------------------------------------------------------------------

LINE_FIELDS = (
    "call", "attempt", "outcome", "agent", "role", "action", "provider",
    "model_requested", "model", "model_served", "resolution", "failover_from",
    "failover_reason", "effort", "run_id", "ticker", "job", "origin", "feature",
    "route", "tokens_in", "tokens_out", "cache_read", "cache_write",
    "reasoning_tokens", "max_tokens", "finish", "cost_usd", "ms", "error", "proc",
)
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._:/()+\-]")
_VALUE_MAX = 64


def sanitize(value: Any) -> str:
    """A log-safe scalar: whitelisted characters only, capped at 64."""
    if value is None or value == "":
        return "-"
    s = _UNSAFE.sub("_", str(value))[:_VALUE_MAX]
    return s or "-"


def format_call_line(fields: dict[str, Any]) -> str:
    """`llm_call v=1 k=v ...` in the fixed LINE_FIELDS order.

    Built only from whitelisted scalars: never a prompt, system text,
    response, exception message, key, user id or email. Values with a space
    are double-quoted (logfmt); missing values render "-".
    """
    parts = ["llm_call", "v=1"]
    for key in LINE_FIELDS:
        v = sanitize(fields.get(key))
        if " " in v:
            v = f'"{v}"'
        parts.append(f"{key}={v}")
    return " ".join(parts)


def format_kv(pairs: list[tuple[str, Any]]) -> str:
    """The same sanitised `k=v` rendering for the failover/breaker lines."""
    out = []
    for key, value in pairs:
        v = sanitize(value)
        if " " in v:
            v = f'"{v}"'
        out.append(f"{key}={v}")
    return " ".join(out)
