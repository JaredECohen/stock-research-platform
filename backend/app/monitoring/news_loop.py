"""News monitoring loop.

For each ticker, throttled to once per hour, calls `news_agent.run` and
pushes the resulting `NewsAlert` records into the hot cache. If any alert
has severity `material` or `breaking`, we ping the relevant sector by
invalidating the sector's warm snapshot — the next sector run will pick up
the fresh news context.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from ..agents import news_agent
from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import get_data_service
from . import record_run
from .research_focus import select_focus

log = logging.getLogger(__name__)

_THROTTLE_SECONDS = 60 * 60  # 1 hour per ticker

# How often this loop runs. One constant for both the scheduler and
# `select_focus`, because they have to agree: the rotation advances one
# position per run, and it can only work out what "per run" means from the
# interval it is told. `register` below hands this to APScheduler, so the
# two cannot drift apart in a later edit.
_RUN_INTERVAL_HOURS = 1

# How many tickers one run may cover.
#
# Cost math: this loop runs hourly, and `news_agent.run(force_refresh=True)`
# makes one Gemini call per ticker (`settings.gemini_news_model`). So the
# steady-state spend is budget x 24 calls/day — at 25, that is 600 Gemini
# calls a day, up from 240 at the previous budget of 10.
#
# The previous 10 was not a judgement about coverage. It was exactly what
# the old arbitrary `list_tickers()[:10]` slice spent, held constant so that
# shipping the relevance ranking changed WHICH ten ran and not how many —
# spend being the owner's decision, not a side effect of a correctness fix.
# The owner has now taken it: 25.
#
# 25 is the size of the thing worth covering rather than a round number. The
# tickers that carry any research signal at all are the 10 curated pins plus
# the 17 that have ever had a memo generated, which overlap by a couple, so
# a budget of 25 reaches essentially all of them every hour. Above that the
# ranking has nothing left to rank: band 3 runs out, and the extra calls
# would buy news on tickers nobody has looked at.
#
# `research_focus`'s `LIVE_RESEARCH_RESERVE` and `ROTATING_SLOTS` are
# unchanged and deliberately so. Both are absolute slot counts, not
# fractions of the budget, and both exist for the *over*-budget case — the
# reserve so a ticker someone is researching right now outranks a full pin
# list, the rotation so the tail below the guaranteed prefix is covered in
# turn rather than never. A budget of 25 against a pool of ~25 mostly takes
# `_select`'s "everything fits" path, where neither fires; when the pool
# does grow past the budget again they do exactly what they did at 10.
# Raising either would not raise spend, only move slots between "covered
# every run" and "covered in turn".
#
# `SOCIAL_FOCUS_BUDGET` is untouched: the owner approved the news budget.
NEWS_FOCUS_BUDGET = 25


def _last_run_for(ticker: str) -> datetime | None:
    snap = cache_get(f"news_loop_throttle:{ticker}", "loop_throttle")
    if not snap or not isinstance(snap.payload, dict):
        return None
    ts = snap.payload.get("last_run_at")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _record_run_for(ticker: str) -> None:
    cache_put(
        f"news_loop_throttle:{ticker}", "loop_throttle",
        payload={"last_run_at": datetime.utcnow().isoformat()},
        sources_used=[f"throttle:{ticker}"],
        generated_by="news_loop", cost_tokens=0,
        ttl_seconds=_THROTTLE_SECONDS * 4,
    )


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
    """Run the news agent for each (un-throttled) ticker. Returns triggered events."""
    selection = None
    if tickers is None:
        # Relevance-ranked rather than an arbitrary universe slice — see
        # `research_focus` for why the old `list_tickers()[:10]` was wrong.
        #
        # On the rotation vs. `_THROTTLE_SECONDS`: the rotation window
        # advances one position per run, so a ticker below the guaranteed
        # prefix is selected on `research_focus.ROTATING_SLOTS` CONSECUTIVE
        # hourly runs before it rotates out. That matters because the
        # throttle and the interval are both exactly one hour: a run that
        # fires a few seconds early finds `elapsed < 3600` and skips the
        # ticker. With a window wider than one slot the next hour's run
        # picks it up — a ticker that appeared for only a single hour could
        # be throttled out of existence forever.
        #
        # `require_memo=True` because this loop's alerts feed exactly one
        # action path, `update_orchestrator.on_news_alert`, and its second
        # guard returns `no_prior_memo` when `memo_store.latest_memo` finds
        # nothing. A slot spent on a ticker with no memo on file is
        # therefore a guaranteed no-op — and five of the ten pins are in
        # that state, so this frees half the budget for names someone is
        # actually researching. The withheld tickers are named in the note,
        # and re-enter selection by themselves once a memo lands.
        selection = select_focus(
            budget=NEWS_FOCUS_BUDGET,
            require_memo=True,
            rotation_period_hours=_RUN_INTERVAL_HOURS,
        )
        tickers = selection.tickers

    events: list[dict] = []
    assessment_failures = 0
    agent_failures: list[str] = []
    update_failures: list[str] = []
    for t in tickers:
        last = _last_run_for(t)
        if last and (datetime.utcnow() - last).total_seconds() < _THROTTLE_SECONDS:
            continue
        try:
            alerts = news_agent.run(t, force_refresh=True)
        except Exception as exc:
            log.warning("news_agent failed for %s: %s", t, exc)
            agent_failures.append(t)
            continue
        _record_run_for(t)

        # Material or breaking → invalidate the sector warm snapshot for that
        # ticker's sector so the next sector pass re-incorporates the news.
        material = [a for a in alerts if a.severity in ("material", "breaking")]
        if material:
            ds = get_data_service()
            profile = ds.get_company_profile(t) or {}
            sector = profile.get("sector", "")
            sub_industry = profile.get("sub_industry") or profile.get("industry") or ""
            cache_key = f"{sector}:{sub_industry}:{t}"
            invalidate(cache_key, kind="sector_warm")
            events.append({"ticker": t, "severity_count": len(material)})

            # Wave 5B: hand each material/breaking alert to the update
            # orchestrator. It gates on prior-memo presence + daily patch
            # cap + news_impact_agent's materiality verdict, and only
            # writes a patched snapshot when all gates pass.
            try:
                from ..services.update_orchestrator import on_news_alert
                for alert in material:
                    res = on_news_alert(t, alert)
                    # A dead news-impact LLM used to read as "no material
                    # news"; the handler now reports it and the note counts it.
                    if isinstance(res, dict) and res.get("reason") == "assessment_error":
                        assessment_failures += 1
            except Exception as exc:  # pragma: no cover — diagnostic only
                log.warning("update_orchestrator failed for %s: %s", t, exc)
                update_failures.append(t)

    note = f"{len(events)} material events"
    if assessment_failures:
        note += f"; {assessment_failures} assessments failed"
    if agent_failures:
        note += f"; {len(agent_failures)} news agents failed: " + ", ".join(agent_failures)
    if update_failures:
        note += f"; {len(update_failures)} updates failed: " + ", ".join(update_failures)
    if selection is not None:
        # Folded in so cron-health shows what this run covered and, more
        # importantly, which qualifying tickers the budget could not reach.
        note += f"; {selection.note()}"
    log.info("news_loop: %s", note)
    record_run(
        "news_loop",
        success=assessment_failures == 0 and not agent_failures and not update_failures,
        note=note,
    )
    return events


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "interval", hours=_RUN_INTERVAL_HOURS,
        id="news_loop", replace_existing=True,
    )
