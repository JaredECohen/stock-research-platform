"""Capture ``frontend/src/test/fixtures/industry.wire.json`` — the wire
fixture every Industry Analysis UI test reads (FEAT-003 slice 6).

Usage (from ``backend/``, against a THROWAWAY database)::

    ENABLE_LIVE_DATA=false USE_DEMO_DATA=true \\
    OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" \\
    DATABASE_URL="sqlite:////tmp/industry-fixture.db" \\
    python -m app.scripts.capture_industry_ui_fixture

The fixture is NOT hand-written, and that is the whole point: it is what
``/api/industries/*`` served after the real weekly path (classify →
``enqueue_period`` → ``drain``) ran against the demo universe, read back
through ``TestClient``. ``app/tests/test_industry_ui_fixture_contract.py``
re-validates every stored body against the pydantic model that produced
it and compares field sets both ways, and when it fails it tells the
reader to re-capture — which is why this script lives in the repo rather
than in somebody's scratch directory.

What the capture is built to contain, because the UI has states that only
real data exercises:

* a group with MORE members than the statistics row could price (one
  ticker is deliberately left without a price series), so the table's
  "membership is not coverage" caption and its unpriced reason have
  something to say;
* a mixed provenance — one member assigned from the research map with an
  8-digit sub-industry, the rest derived from the provider's industry
  label;
* two adjacent editions: the first week has no price history at all (the
  honest ``insufficient_sample`` edition), the second has closes, so the
  diff carries facts that are missing on one side and must never be
  differenced to zero.

Two edits are made after the capture, and both are declared in ``meta``:
per-ticker ``weekly_closes`` arrays are emptied (each row keeps a
``weekly_closes_dropped`` count and ``meta.trimmed`` the total — a
truncated artifact has to count what it dropped), and two responses the
UI does not read are not stored at all (``meta.omitted_responses``).

Nothing else is edited. If a value in the fixture looks wrong, it is what
the API said.

Deterministic: prices come from a seeded PRNG over fixed weekday dates,
the period keys are fixed, and the group is chosen by membership size
with the code as the tie-break. Run ids, hashes and ``computed_at``
naturally differ between captures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Set BEFORE any app module is imported (they are all imported lazily,
# inside the functions below, for exactly this reason).
#
# `RATE_LIMIT_ENABLED=false` is what `backend/conftest.py` gives the test
# suite, and the capture needs it for the same reason: with slowapi live,
# every `/api/industries/*` GET raises inside `_inject_headers` because the
# handlers do not take the `response: Response` parameter slowapi's
# `headers_enabled=True` requires (`routes_chat.chat` does). That is a
# fault in the ROUTES, not in this script — it is why a limiter-enabled
# deployment answers 500 on this whole surface — and turning the limiter
# off here only keeps the capture from being blocked by it.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "frontend" / "src" / "test" / "fixtures" / "industry.wire.json"

# The two weeks the capture covers. The first is the no-price-history
# edition, the second the priced one; `as_of` is derived from the period
# key by the worker, so it is not a knob here.
PRIOR_PERIOD = "2026-W35"
PERIOD = "2026-W36"
LAST_CLOSE = date(2026, 9, 4)
SESSIONS = 400

# Left without a price series so the table has an unpriced row with a
# reason. Ignored when the chosen group does not contain it.
UNPRICED = "INTC"

OMITTED = [
    "GET /api/industries/snapshot",
    "GET /api/industries/{code}/report?version=1",
]
OMITTED_REASON = (
    "this slice's UI reads neither; the version-1 edition is derived in the fixture module"
)
TRIMMED_NOTE = (
    "weekly close points were dropped to keep the fixture readable; each row records how "
    "many it lost in `weekly_closes_dropped`. Nothing else was edited: every other value "
    "is what the API served."
)
GENERATED_BY = (
    "python -m app.scripts.capture_industry_ui_fixture — the real weekly path "
    "(classify -> enqueue_period -> drain) against the demo universe, captured through TestClient"
)


def _refuse_unsafe_database() -> str | None:
    """This script DELETES the report, stats and job rows for the active
    taxonomy version. That is fine on a scratch sqlite file and a disaster
    anywhere else, so it is checked rather than trusted."""
    from app.config import settings

    if settings.enable_live_data or not settings.use_demo_data:
        return (
            "refusing to capture with live data enabled — rerun with "
            "ENABLE_LIVE_DATA=false USE_DEMO_DATA=true"
        )
    if not settings.database_url.startswith("sqlite"):
        return (
            f"refusing to delete rows in {settings.database_url.split('://')[0]} — point "
            'DATABASE_URL at a throwaway sqlite file, e.g. sqlite:////tmp/industry-fixture.db'
        )
    return None


def seed_prices(tickers: list[str], *, skip: str) -> int:
    """Deterministic weekday closes in the provider cache, so the second
    edition has a real price history and the first does not."""
    from app.services import provider_cache

    days = sorted(
        d for d in (LAST_CLOSE - timedelta(days=i) for i in range(SESSIONS)) if d.weekday() < 5
    )
    seeded = 0
    for ticker in tickers:
        if ticker == skip:
            continue
        rng = random.Random(f"seed:{ticker}")
        px = 40.0 + rng.random() * 260.0
        rows = []
        for day in days:
            px = max(1.0, px * (1.0 + rng.gauss(0.0006, 0.016)))
            rows.append({"date": day.isoformat(), "close": round(px, 2), "adjusted_close": round(px, 2)})
        provider_cache.put("prices", f"{ticker.upper()}:252", rows)
        seeded += 1
    return seeded


def trim_weekly_closes(report: dict[str, Any]) -> dict[str, int]:
    """Empty the per-ticker weekly close series, COUNTING what each row
    lost. A row that loses nothing gains no count — an invented zero would
    be as dishonest as a silent truncation."""
    facts = report["payload"]["sections"]["companies"]["facts"]
    rows = facts.get("per_ticker")
    if not isinstance(rows, list):
        return {}
    total = 0
    for row in rows:
        closes = row.get("weekly_closes") or []
        if not closes:
            continue
        row["weekly_closes"] = []
        row["weekly_closes_dropped"] = len(closes)
        total += len(closes)
    if not total:
        return {}
    return {"report.payload.sections.companies.facts.per_ticker[].weekly_closes": total}


def capture() -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from sqlalchemy import delete

    from app.database import SessionLocal
    from app.main import app
    from app.models import IndustryReport, IndustryReportJob, IndustryStatSnapshot
    from app.services import gics_registry as reg
    from app.services import industry_classification as ic
    from app.services import industry_report_store as rs
    from app.services import industry_report_worker as jobs
    from app.tests.fixtures.seed_demo_data import run_full_seed

    run_full_seed()
    info = reg.ensure_taxonomy(activate=True)
    if info is None:
        raise SystemExit("the bundled taxonomy could not be imported; nothing to capture")
    counts = ic.classify_all(version=info)
    print("classified", counts.get("counts"))

    by_group = ic.constituents_by_group(version=info)
    if not by_group:
        raise SystemExit("no group has any classified member; the capture would be empty")
    # Biggest membership wins, code as the tie-break so two runs over the
    # same universe choose the same group.
    ranked = sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    code = ranked[0][0]
    # A group with members but no edition — the "the analysis has not been
    # written yet" state the page renders beside the membership table. A
    # group with NO members would exercise a different, emptier path.
    empty_code = ranked[-1][0] if ranked[-1][0] != code else ranked[-2][0]
    print("group", code, f"({len(by_group[code])} members)", "· empty group", empty_code)

    universe = sorted({t for tickers in by_group.values() for t in tickers})

    # A capture starts from no editions: a leftover row from an earlier
    # run would make the "first week has no prices" edition v2.
    with SessionLocal() as db:
        for model in (IndustryReportJob, IndustryReport, IndustryStatSnapshot):
            db.execute(delete(model).where(model.taxonomy_version_id == info.id))
        db.commit()

    for period in (PRIOR_PERIOD, PERIOD):
        if period == PERIOD:
            print("seeded prices for", seed_prices(universe, skip=UNPRICED), "tickers")
        res = jobs.enqueue_period(period, codes=[code], version=info, source="fixture")
        print("enqueued", period, res["enqueued"], res["cross_snapshot"])
        for drained in jobs.drain(limit=20):
            print(
                "drained",
                {k: drained.get(k) for k in ("id", "kind", "code", "status", "note", "error_type", "error_message")},
            )

    latest = rs.latest_good(code, version=info)
    if latest is None:
        raise SystemExit(
            f"no published edition for {code} after draining both periods — read the `drained` "
            "lines above; a validator rejection there is the bug to fix, not to capture around"
        )
    print("latest good edition:", latest["version"])

    client = TestClient(app)
    out: dict[str, Any] = {}
    for name, path in [
        ("taxonomy", "/api/industries/taxonomy"),
        ("report", f"/api/industries/{code}/report"),
        ("companies", f"/api/industries/{code}/companies"),
        ("history", f"/api/industries/{code}/history"),
        ("changes", f"/api/industries/{code}/changes"),
        ("report_missing", f"/api/industries/{empty_code}/report"),
    ]:
        r = client.get(path)
        expected = 404 if name == "report_missing" else 200
        if r.status_code != expected:
            raise SystemExit(f"{path} answered {r.status_code}, expected {expected}: {r.text[:400]}")
        print(name, r.status_code, path)
        out[name] = r.json()

    trimmed = trim_weekly_closes(out["report"])
    meta = {
        "generated_by": GENERATED_BY,
        "taxonomy_version": info.version_key,
        "code": code,
        "empty_code": empty_code,
        "trimmed": trimmed,
        "trimmed_note": TRIMMED_NOTE,
        "omitted_responses": OMITTED,
        "omitted_reason": OMITTED_REASON,
    }
    return {"meta": meta, **out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args(argv)

    refusal = _refuse_unsafe_database()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes; "
        f"dropped {sum(payload['meta']['trimmed'].values())} weekly closes)"
    )
    print("now run: npm test in frontend/, and pytest app/tests/test_industry_ui_fixture_contract.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
