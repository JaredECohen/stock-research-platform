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
    # Missing or incomplete price coverage and evaluation errors remain
    # actionable failures. Keep legacy unevaluable counts compatible while
    # preserving every affected snapshot/horizon identity from the evaluator.
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
    for key in ("unevaluable_pairs", "unavailable_pairs", "error_pairs"):
        if res.get(key):
            note += f" {key}=" + ",".join(res[key])
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
