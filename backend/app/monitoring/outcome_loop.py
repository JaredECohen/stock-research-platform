"""Wave 4A — daily outcome evaluator.

Runs once a day, scores every memo snapshot whose forward windows have
come of age, and writes reflection entries into long-term memory for
the long horizons. Idempotent: if a (snapshot, horizon) pair already
has a row in `memo_outcomes`, the evaluator skips it silently.
"""
from __future__ import annotations

import logging
from typing import Any

from ..services.outcome_service import evaluate_all_due
from . import record_run

log = logging.getLogger(__name__)


def run_once() -> dict[str, Any]:
    res = evaluate_all_due()
    # The note names each shortfall separately. Three of them are outages,
    # all three counted in `unavailable`, and `unavailable` drives the
    # failure flag: "the provider returned nothing for this ticker"
    # (no_prices), "the provider returned fewer bars than we asked for, so
    # the series begins after the memo" (short_history), and "the window
    # has a gap around the target date" (window_gap). Only "the memo is
    # older than the longest window we request" (unevaluable) is permanent,
    # and it deliberately does NOT turn the loop red — a nightly failure
    # nobody can act on is a failure everybody learns to ignore.
    #
    # Keeping short_history on the red side of that line is the whole
    # point: a truncated response is the shape a *partial* provider outage
    # takes, and filing it as permanent would show a green learning loop
    # that wrote nothing.
    note = (
        f"evaluated={res['evaluated']} due={res['due']} "
        f"written={res['written']} existing={res['already_recorded']} "
        f"unavailable={res['data_unavailable']} "
        f"no_prices={res.get('ticker_prices_unavailable', 0)} "
        f"short_history={res.get('price_history_too_short', 0)} "
        f"window_gap={res.get('price_window_incomplete', 0)} "
        f"unevaluable={res.get('unevaluable', 0)} "
        f"reflections={res['reflections']} errors={res['errors']}"
    )
    if res.get("unevaluable_pairs"):
        note += " unevaluable_pairs=" + ",".join(res["unevaluable_pairs"])
    success = res["errors"] == 0 and res["data_unavailable"] == 0
    record_run("outcome_loop", success=success, note=note)
    return res


def register(scheduler) -> None:
    # Daily at 02:30 UTC — runs after EDGAR poller (top-of-hour) and
    # before history backfill (03:15 UTC).
    scheduler.add_job(
        run_once, "cron", hour=2, minute=30,
        id="outcome_loop", replace_existing=True,
    )
