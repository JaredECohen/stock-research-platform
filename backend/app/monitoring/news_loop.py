"""News monitoring loop.

Scheduled hourly. For each focus ticker not fetched within the last hour it
calls `news_agent.run`, which writes the resulting `NewsAlert` records to the
hot cache (`news_hot:{T}`). Because the throttle equals the interval and the
stamp is written after the fetch, a run a few seconds short of the hour skips
the ticker, so each ticker is in practice fetched every 2 hours (production
shows a strict fetch / skip alternation). That cadence is the accepted one
(N5 not adopted, 2026-09-25); the note's `throttled=` count makes it visible.

Material or breaking alerts do two things. Each is handed to
`update_orchestrator.on_news_alert` (the news patch path). And the ticker's
`sector_warm` snapshot is invalidated. That snapshot holds cohort research,
not news: the sector analyst reads `news_hot` directly at run time, so the
invalidation does not deliver the news to anyone; it only forces the
sector-research recompute on the next sector run.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from datetime import datetime

from ..agents import news_agent
from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import get_data_service
from . import record_run
from .research_focus import select_focus

log = logging.getLogger(__name__)

# Minimum gap between fetches of one ticker. Equal to the run interval and
# stamped after the fetch, so the effective cadence is every 2 hours (see
# the module docstring); deliberately left so.
_THROTTLE_SECONDS = 60 * 60

# How often this loop runs. One constant for both the scheduler and
# `select_focus`, because they have to agree: the rotation advances one
# position per run, and it can only work out what "per run" means from the
# interval it is told. `register` below hands this to APScheduler, so the
# two cannot drift apart in a later edit.
_RUN_INTERVAL_HOURS = 1

# How many tickers one run may cover.
#
# Cost math: this loop runs hourly, and `news_agent.run(force_refresh=True)`
# can make one grounded Gemini call per ticker when Gemini is configured.
# The budget was approved as budget x 24 calls/day — at 25, up to 600
# instead of 240. The throttle halves that in practice: each ticker is
# fetched every 2 hours, so the real ceiling is budget x 12 — 300 a day at
# 25 (about 250 observed at 21 memo-bearing names, 2026-09-25). Actual calls
# depend on eligible names, throttle state and provider configuration;
# without Gemini this uses deterministic provider news.
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


def _origin_of(fetch: dict, alerts: list) -> str:
    """gemini / provider / empty for one fetched ticker. Falls back to the
    alerts themselves when `news_agent.run` did not fill the report (a
    stand-in in tests, or an older signature)."""
    origin = fetch.get("origin")
    if origin in ("gemini", "provider", "empty"):
        return str(origin)
    if not alerts:
        return "empty"
    if any(getattr(a, "source", "") == "gemini" for a in alerts):
        return "gemini"
    return "provider"


def _sources_note(sources: Counter[str]) -> str:
    return (
        f"sources gemini={sources['gemini']} provider={sources['provider']} "
        f"empty={sources['empty']} throttled={sources['throttled']} "
        f"gemini_breaker={sources['breaker_open']} grounding_cap={sources['grounding_cap']}"
    )


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


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
        # throttle and the interval are both exactly one hour and the stamp
        # lands after the fetch, so the next run finds `elapsed < 3600` and
        # skips the ticker — every ticker is fetched on alternate runs. With
        # a window wider than one slot the following run picks it up; a
        # ticker that appeared for only a single hour could be throttled
        # out of existence forever.
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
    assessment_failures: list[str] = []
    agent_failures: list[str] = []
    update_failures: list[str] = []
    # Where each fetched ticker's alerts came from, and why tickers were not
    # asked. Gemini returning nothing, Gemini never asked (breaker open,
    # grounding cap) and a throttled skip all used to read as "0 material
    # events"; the note now tells them apart.
    sources: Counter[str] = Counter()
    for t in tickers:
        last = _last_run_for(t)
        if last and (datetime.utcnow() - last).total_seconds() < _THROTTLE_SECONDS:
            sources["throttled"] += 1
            continue
        fetch: dict = {}
        try:
            alerts = news_agent.run(t, force_refresh=True, report=fetch)
        except Exception as exc:
            log.warning("news_agent failed ticker=%s error_type=%s", t, type(exc).__name__)
            agent_failures.append(t)
            continue
        _record_run_for(t)
        sources[_origin_of(fetch, alerts)] += 1
        skipped = fetch.get("gemini_skipped")
        if skipped in ("breaker_open", "grounding_cap"):
            sources[skipped] += 1

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
            for alert in material:
                # Per alert, not per ticker: one alert that raises (an
                # unreadable prior memo, a DB error on publish) used to drop
                # every remaining alert for the ticker.
                try:
                    from ..services.update_orchestrator import on_news_alert
                    res = on_news_alert(t, alert)
                except Exception as exc:
                    log.warning("news update failed ticker=%s error_type=%s", t, type(exc).__name__)
                    update_failures.append(t)
                    continue
                # A dead news-impact LLM used to read as "no material
                # news"; the handler now reports it and the note counts it.
                if isinstance(res, dict) and res.get("reason") == "assessment_error":
                    assessment_failures.append(t)

    note = f"{len(events)} material events"
    if assessment_failures:
        note += f"; {len(assessment_failures)} assessments failed: " + ", ".join(assessment_failures)
    if agent_failures:
        note += f"; {len(agent_failures)} news agents failed: " + ", ".join(agent_failures)
    if update_failures:
        note += f"; {len(update_failures)} updates failed: " + ", ".join(_unique(update_failures))
    if selection is not None:
        # Folded in so cron-health shows what this run covered and, more
        # importantly, which qualifying tickers the budget could not reach.
        note += f"; {selection.note()}"
    note += "; " + _sources_note(sources)
    log.info("news_loop: %s", note)
    record_run(
        "news_loop",
        success=not assessment_failures and not agent_failures and not update_failures,
        note=note,
    )
    return events


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "interval", hours=_RUN_INTERVAL_HOURS,
        id="news_loop", replace_existing=True,
    )
