"""Capture ``frontend/src/test/fixtures/memo_quality.wire.json`` — the wire
fixture the W2b "Research checks" UI is tested against (S16).

Usage (from ``backend/``, against a THROWAWAY database, keys blank)::

    ENABLE_LIVE_DATA=false USE_DEMO_DATA=true \\
    OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" \\
    DATABASE_URL="sqlite:////tmp/memo-quality-fixture.db" \\
    python -m app.scripts.capture_memo_quality_fixture

The fixture is NOT hand-written. It is ``graph.run_stock_memo("NVDA")`` on
the demo dataset — the real source ledger, number check, rating
reconciliation and earned-confidence stages (S14/S15) — passed through
``memo_sections.present_memo`` and serialized the way
``GET /api/stocks/{t}/memo`` serializes it. A hand-written ``quality``
block would drift from its producer; ``test_memo_quality_fixture_contract``
re-validates this one against the models and compares key sets both ways,
and tells the reader to re-capture when it fails.

The model is scripted, because a keyless run has no LLM and a deterministic
memo has nothing for the checks to find (every figure it prints is written
by code from registered data — the CI invariant). ``has_llm`` is patched on
(keys stay blank, so no client exists and nothing can be spent) and
``llm.chat_json`` answers exactly two prompts; everything else answers
nothing, so those analysts fall back to their deterministic output as they
would in an outage and the presenter marks them unavailable:

* **PM synthesis** — a Bullish call with NO ``valuation_divergence_reason``,
  a PM view carrying one fabricated figure (``$123.45B``) and one declared
  forecast assumption (``18.5%`` revenue growth, ``basis_ref`` the
  financials, which the ledger registers), so the capture exercises an
  untraceable figure, a "PM assumption" label and the rating check;
* **Valuation analyst** — four key points, one of which
  (``Hidden optionality worth $987.65B``) traces to nothing, so the number
  check withholds it and the panel has a withheld point to disclose.

The critic is not scripted: it answers nothing, the rule-based critic
stands in, and the ``critic_not_live`` cap applies — a real outage shape.
``meta.scripted`` declares every scripted answer verbatim. Nothing in the
captured memo is edited afterwards; if a value looks wrong, it is what the
pipeline produced.

Run ids, timestamps and ``generated_at`` naturally differ between captures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Set BEFORE any app module is imported (they are imported lazily, inside
# the functions below, for exactly this reason).
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "frontend" / "src" / "test" / "fixtures" / "memo_quality.wire.json"
TICKER = "NVDA"

FABRICATED_PM_FIGURE = "$123.45B"
DECLARED_ASSUMPTION = "18.5%"
WITHHELD_POINT = "Hidden optionality worth $987.65B"

# The PM's answer. Bullish with no divergence reason: when the evidence
# verdict disagrees, S14 downgrades the rating and says so; when it does
# not, the reconciliation reads "consistent". Either is the real outcome.
PM_ANSWER: dict[str, Any] = {
    "rating_label": "Bullish",
    "confidence_score": 82,
    "one_sentence_thesis": (
        "NVDA compounds data-center share faster than the market prices, but the multiple "
        "already assumes much of it."
    ),
    "final_pm_view": (
        f"Data-center revenue of {FABRICATED_PM_FIGURE} anchors the call. We assume revenue "
        f"growth of {DECLARED_ASSUMPTION} through FY2027 as hyperscaler capex stays elevated."
    ),
    "mispricing_thesis": {
        "consensus_view": "The market treats the AI build-out as a one-cycle spike.",
        "our_view": "Share gains in networking and software extend the cycle.",
        "gap": "Duration of data-center demand.",
        "falsifiers": ["Two consecutive quarters of data-center revenue decline"],
    },
    "forecast_assumptions": [
        {"value": 18.5, "unit": "pct", "basis_ref": f"financials:{TICKER}", "horizon": "FY2027"},
    ],
}

VALUATION_ANSWER: dict[str, Any] = {
    "headline": "Valuation view",
    "summary": "Multiples look full against the DCF.",
    "key_points": ["P/E 46.8x", "EV/EBITDA 34.4x", WITHHELD_POINT, "FCF yield 2.0%"],
    "confidence": 0.7,
}

GENERATED_BY = (
    "python -m app.scripts.capture_memo_quality_fixture — graph.run_stock_memo on demo data "
    "with two scripted model answers, then memo_sections.present_memo, serialized as the API does"
)


def _refusal() -> str | None:
    """Why this capture must not run here, or None. It writes memo rows, and
    it patches `has_llm` on: with a real key set, any model call the script
    does not answer would reach a provider and spend money. Nothing here
    prints a key or a URL's credentials."""
    from app.config import settings
    from app.tests import dbguard

    if settings.enable_live_data or not settings.use_demo_data:
        return ("refusing to capture with live data enabled — rerun with "
                "ENABLE_LIVE_DATA=false USE_DEMO_DATA=true")
    if settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key:
        return ('refusing to capture with an LLM key set — rerun with OPENAI_API_KEY="" '
                'ANTHROPIC_API_KEY="" GEMINI_API_KEY=""')
    if not settings.database_url.startswith("sqlite"):
        return ("refusing to write memo rows outside sqlite — point DATABASE_URL at a throwaway "
                "file, e.g. sqlite:////tmp/memo-quality-fixture.db")
    return dbguard.refusal(settings.database_url)


def _scripted_chat_json(prompt: str, **_: Any) -> dict[str, Any] | None:
    from app.agents import prompts

    if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
        return json.loads(json.dumps(PM_ANSWER))
    if prompt.startswith(prompts.VALUATION_ANALYST_PROMPT):
        return json.loads(json.dumps(VALUATION_ANSWER))
    return None


def capture() -> dict[str, Any]:
    """Run the memo with the scripted answers and return the wire payload
    `{meta, memo}`, `memo` being what the memo route serves."""
    from app.agents import graph
    from app.agents import llm as llm_mod
    from app.config import Settings
    from app.database import init_db
    from app.services.data_service import get_data_service
    from app.services.memo_sections import PRESENTATION_VERSION, present_memo
    from app.tests.fixtures.demo_provider import DemoProvider

    init_db()
    # The demo dataset through the same in-memory provider the test suite
    # registers (`app/tests/conftest.py`), so the memo is built from the
    # data `test_memo_quality_pipeline` judges.
    ds = get_data_service()
    saved_provider = ds._test_provider
    saved_has_llm = Settings.__dict__["has_llm"]
    saved_chat = llm_mod.chat_json
    ds.register_test_provider(DemoProvider())
    Settings.has_llm = property(lambda self: True)  # type: ignore[assignment,method-assign]
    llm_mod.chat_json = _scripted_chat_json  # type: ignore[assignment]
    try:
        memo = graph.run_stock_memo(TICKER)
    finally:
        Settings.has_llm = saved_has_llm  # type: ignore[method-assign]
        llm_mod.chat_json = saved_chat  # type: ignore[assignment]
        ds.register_test_provider(saved_provider)
    if memo.quality is None or memo.quality.number_check is None:
        raise RuntimeError("the memo carries no number check; the W2b stages did not run")
    presented = present_memo(memo)
    return {
        "meta": {
            "generated_by": GENERATED_BY,
            "ticker": TICKER,
            "presentation_version": PRESENTATION_VERSION,
            "scripted": {
                "pm_synthesis": PM_ANSWER,
                "valuation_analyst": VALUATION_ANSWER,
                "everything_else": "no answer (deterministic fallback; the critic is rule-based)",
            },
            "expect": {
                "fabricated_pm_figure": FABRICATED_PM_FIGURE,
                "declared_assumption": DECLARED_ASSUMPTION,
                "withheld_point": WITHHELD_POINT,
            },
        },
        "memo": json.loads(presented.model_dump_json()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args(argv)
    refusal = _refusal()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    q = payload["memo"]["quality"]
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes); "
          f"counts={q['number_check']['counts']} "
          f"reconciliation={(q.get('rating_reconciliation') or {}).get('outcome')}")
    print("now run: npx vitest run in frontend/, and pytest "
          "app/tests/test_memo_quality_fixture_contract.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
