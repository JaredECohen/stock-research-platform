"""Social monitoring loop — daily per-ticker sentiment scalar."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from ..agents import social_agent
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
# Cost math: this loop runs daily, and `social_agent.run(force_refresh=True)`
# makes one Gemini call per ticker (`settings.gemini_social_model`). So the
# steady-state spend is budget x 1 call/day — at 10, that is 10 Gemini calls
# a day. Raising it to ~25 (25/day) would cover every ticker that carries
# any research signal at all today: the 10 pins plus the 17 that have ever
# had a memo generated, minus the overlap.
#
# Deliberately left at 10, which is exactly what the old arbitrary
# `list_tickers()[:10]` slice spent. This change is about WHICH ten, not how
# many; raising it is a spend decision for the owner.
SOCIAL_FOCUS_BUDGET = 10


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
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
        # Withholding it from the five pins that have no memo yet would
        # remove exactly the warm-up those first memos benefit from, to
        # save five Gemini calls a day.
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
