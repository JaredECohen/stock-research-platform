"""Social monitoring loop — daily per-ticker sentiment scalar (demo only).

Outside demo mode there is no social data source (FIX-016 / L7), so a run
records "no social data source" on cron-health and does nothing else: no
focus query, no agent call, no LLM spend. The loop stays registered so
`KNOWN_LOOPS` and cron-health keep reporting it, and so a real source can
be wired into `social_agent` later without re-plumbing the scheduler.
Dropping the old daily Gemini call also stops it co-firing with the news
loop's Gemini traffic against the shared circuit breaker.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from ..agents import social_agent
from ..config import settings
from . import record_run
from .research_focus import select_focus

log = logging.getLogger(__name__)

# How often this loop runs. One constant for both the scheduler and
# `select_focus`, because they have to agree. This loop is the reason
# `rotation_period_hours` exists: the rotation offset comes off the wall
# clock, so with consecutive runs 24 hours apart it used to jump 24
# positions per run instead of one, and only `len(pool) / gcd(24,
# len(pool))` positions were ever reachable — at a pool of 12 that is two
# names covered and ten starved forever. `register` below hands this same
# number to APScheduler, so the interval and the rotation cannot drift.
_RUN_INTERVAL_HOURS = 24

# How many tickers one run may cover.
#
# Cost math: zero LLM calls a day. Live runs skip selection entirely (see
# `run_once`); demo runs cover this many tickers with the free
# deterministic stub. It used to bound one Gemini call per ticker per day
# (10 a day at 10); if a real, paid source is wired in later, calls/day =
# this budget again, and raising it is a spend decision for the owner.
SOCIAL_FOCUS_BUDGET = 10

# The cron-health note for a live run. Worded for a reader of
# `/api/admin/cron-health`, who otherwise sees a green loop and assumes it
# produced sentiment.
NO_SOURCE_NOTE = "no social data source; social sentiment unavailable"


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
    if not settings.use_demo_data_only:
        # success=True: the loop did what it can do. A failure flag would page
        # on every run for a known, owner-decided absence.
        record_run("social_loop", note=NO_SOURCE_NOTE)
        return []
    selection = None
    if tickers is None:
        # Relevance-ranked rather than an arbitrary universe slice — see
        # `research_focus` for why the old `list_tickers()[:10]` was wrong.
        #
        # No `require_memo` here, unlike `news_loop`. That flag exists
        # because a news alert on a memo-less ticker provably cannot do
        # anything (`on_news_alert` returns `no_prior_memo`). Nothing
        # equivalent is true of sentiment: `social_agent.run` computes a
        # standalone per-ticker scalar, and `sdk_runtime.run_social_agent`
        # calls that same function during memo generation against a 24h
        # cache — so this daily pass is a pre-warm for a pin's FIRST memo.
        # Withholding it from the pins that have no memo yet would remove
        # exactly the warm-up those first memos benefit from. (Demo-only
        # today; the reasoning holds for any real source wired in later.)
        selection = select_focus(
            budget=SOCIAL_FOCUS_BUDGET,
            rotation_period_hours=_RUN_INTERVAL_HOURS,
        )
        tickers = selection.tickers
    out: list[dict] = []
    for t in tickers:
        try:
            payload = social_agent.run(t, force_refresh=True)
            out.append({"ticker": t, "extremity": payload.get("sentiment_extremity")})
        except Exception as exc:  # pragma: no cover
            log.warning("social_agent failed for %s: %s", t, exc)
    note = f"{len(out)} tickers"
    if selection is not None:
        # Folded in so cron-health shows what this run covered and, more
        # importantly, which qualifying tickers the budget could not reach.
        note += f"; {selection.note()}"
    record_run("social_loop", note=note)
    return out


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "interval", hours=_RUN_INTERVAL_HOURS,
        id="social_loop", replace_existing=True,
    )
