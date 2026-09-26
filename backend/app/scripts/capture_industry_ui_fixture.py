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
  differenced to zero;
* **both ways a group can sit below the sample floor**, because the page
  has to word them differently. ``report_warming_up`` is the big group's
  first week — enough members, no prices yet, a state the weekly warm-up
  clears. ``report_universe_short`` is a group whose membership is itself
  below the floor: every member priced and still short, which no warm-up
  can fix. A hand-written stand-in for either would drift from the
  producer, so both are captured;
* **two reads of the same group that landed in different weeks.**
  ``companies_warming_up`` is ``/companies`` as it answered BEFORE the
  prices existed; the published ``report`` is the priced week. Serving
  them together is the race the page has to survive — the edition's own
  priced count and the live membership read disagreeing — and both bodies
  are what the API said, so neither has to be edited into disagreement.

* **the display rule's states** (owner decision 1: template editions are
  stored for audit only and never shown). ``report_not_updated`` is the
  structurally-short group after a later week whose only product was an
  audit-only template — the page serves the last analyst edition with a
  "not updated" banner; ``report_no_agentic`` is a group whose only
  edition is audit-only — the 404 that says no analyst edition has been
  published yet, and counts the withheld one.

The analyst model is the deterministic stand-in in
``app/tests/fixtures/industry_analyst_stub.py`` unless ``--no-stub-analyst``
(the capture must run with blank keys, and without a model every edition
would be an audit-only template with nothing to render). The stub returns
the writer's own template prose relabelled "Fixture analyst (stubbed
model)", so no analyst text in the fixture is invented; ``meta.analyst_stub``
says so. The two display-rule states are produced by running those weeks
with the stub OFF — a real model outage. The stub's one addition is a
registered forecast assumption in the outlook (FA1, holding the edition's
first rate anchor at its own observed value), so the page's assumptions
table renders from a real published edition; ``meta.analyst_stub
.forecast_assumption`` declares it.

Two edits are made after the capture, and both are declared in ``meta``:
per-ticker ``weekly_closes`` arrays are emptied (each row keeps a
``weekly_closes_dropped`` count and ``meta.trimmed`` the total — a
truncated artifact has to count what it dropped), and one response the
UI does not read is not stored at all (``meta.omitted_responses``).

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
# The week the model is "down" for the display-rule states.
OUTAGE_PERIOD = "2026-W37"
# Sections the stub analyst deliberately does not write, per week, so the
# published edition carries a real template-filled (hidden) section.
NOT_WRITTEN: dict[str, tuple[str, ...]] = {PERIOD: ("what_changed",)}
LAST_CLOSE = date(2026, 9, 4)
SESSIONS = 400

# Left without a price series so the table has an unpriced row with a
# reason. Ignored when the chosen group does not contain it.
UNPRICED = "INTC"

OMITTED = [
    "GET /api/industries/snapshot",
]
OMITTED_REASON = "this slice's UI does not read it"
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
    from app.tests import dbguard

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
    # The repo's shared database guard as well (it never prints credentials).
    return dbguard.refusal(settings.database_url)


def drain_through_retries(jobs: Any, *, attempts: int = 3) -> list[dict[str, Any]]:
    """Drain, stepping the claim clock past each retry's backoff, so a week
    in which the model is down runs every attempt (the non-final ones raise
    AnalystUnavailable and back off; the final one stores the audit-only
    template) instead of stopping at the first deferral."""
    from datetime import datetime

    done: list[dict[str, Any]] = []
    for step in range(attempts):
        now = datetime.utcnow() + timedelta(hours=2 * step)
        done.extend(jobs.drain(limit=40, now=now))
    return done


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
    return {"payload.sections.companies.facts.per_ticker[].weekly_closes": total}


def _get(client: Any, path: str, *, expect: int = 200, name: str = "") -> Any:
    """One captured response, or a refusal. A capture that stored a body
    the API answered with the wrong status would be a fixture of a bug."""
    r = client.get(path)
    if r.status_code != expect:
        raise SystemExit(f"{path} answered {r.status_code}, expected {expect}: {r.text[:400]}")
    print(name or path, r.status_code, path)
    return r.json()


def _reading_at(when: Any):
    """Serve the reads as if at ``when``. Staleness-by-age is ``now -
    as_of``, so a capture read on the wall clock would make the SAME
    fixture fresh one week and stale the next; the read clock is pinned to
    two days after the week's as-of instead (declared in ``meta.read_at``)."""
    from unittest import mock

    from app.services import industry_report_store as rs

    return mock.patch.object(rs, "_utcnow", lambda: when)


def _print_drained(done: list[dict[str, Any]]) -> None:
    for drained in done:
        print(
            "drained",
            {k: drained.get(k) for k in ("id", "kind", "code", "status", "note", "error_type", "error_message")},
        )


def capture(*, stub_analyst: bool = True) -> dict[str, Any]:
    import contextlib

    from fastapi.testclient import TestClient
    from sqlalchemy import delete

    from app.database import SessionLocal
    from app.main import app
    from app.models import IndustryReport, IndustryReportJob, IndustryStatSnapshot
    from app.services import gics_registry as reg
    from app.services import industry_analytics as ia
    from app.services import industry_classification as ic
    from app.services import industry_labels
    from app.services import industry_report_store as rs
    from app.services import industry_report_worker as jobs
    from app.tests.fixtures import industry_analyst_stub
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
    # A group this universe CANNOT cover: fewer classified constituents
    # than the sample floor, so every week is `insufficient_sample` however
    # long the price warm-up runs. The page has to say that in different
    # words from a group whose prices simply have not warmed up yet, and
    # only a real edition for a real thin group proves it does.
    floor = ia.sample_floor()
    thin = [c for c, members in ranked if c != code and len(members) < floor]
    if not thin:
        raise SystemExit(
            f"every group in this universe holds at least {floor} constituents, so the capture "
            "cannot contain a structurally-short edition — the fixture the UI needs for that "
            "state would have to be hand-written, which this repo does not do"
        )
    short_code = thin[-1]
    # A group with members but no edition — the "the analysis has not been
    # written yet" state the page renders beside the membership table. A
    # group with NO members would exercise a different, emptier path.
    empty_code = next(c for c, _ in reversed(ranked) if c not in (code, short_code))
    # A group whose ONLY edition is an audit-only template: the 404 that
    # says no analyst edition has been published and counts the withheld one.
    no_agentic_code = next(c for c, _ in reversed(ranked) if c not in (code, short_code, empty_code))
    print(
        "group", code, f"({len(by_group[code])} members)",
        "· structurally short", short_code, f"({len(by_group[short_code])} of {floor} needed)",
        "· empty group", empty_code, "· template-only group", no_agentic_code,
    )

    universe = sorted({t for tickers in by_group.values() for t in tickers})
    # The page requests a group by its public slug (the API also accepts the
    # internal code, for old links), so the capture does too.
    slug = industry_labels.slug

    # A capture starts from no editions: a leftover row from an earlier
    # run would make the "first week has no prices" edition v2.
    with SessionLocal() as db:
        for model in (IndustryReportJob, IndustryReport, IndustryStatSnapshot):
            db.execute(delete(model).where(model.taxonomy_version_id == info.id))
        db.commit()

    client = TestClient(app)
    out: dict[str, Any] = {}
    read_at = {p: jobs.as_of_for_period(p) + timedelta(days=2) for p in (PRIOR_PERIOD, PERIOD, OUTAGE_PERIOD)}

    for period in (PRIOR_PERIOD, PERIOD):
        # In the priced week the stub leaves `what_changed` unwritten, as a
        # model whose second call came back without it would: the published
        # edition then carries a REAL template-filled section, which the page
        # must hide ("unavailable in this version") and the diff must not
        # quote. Declared in meta.analyst_stub.sections_not_written.
        skip = NOT_WRITTEN.get(period, ())
        analyst = (
            industry_analyst_stub.stubbed_analyst(
                sections=[s for s in rs.INTERPRETED_SECTIONS if s not in skip]) if stub_analyst
            else contextlib.nullcontext()
        )
        with analyst:
            if period == PERIOD:
                print("seeded prices for", seed_prices(universe, skip=UNPRICED), "tickers")
            codes = [code] if period == PRIOR_PERIOD else [code, short_code]
            res = jobs.enqueue_period(period, codes=codes, version=info, source="fixture")
            print("enqueued", period, res["enqueued"], res["cross_snapshot"])
            _print_drained(drain_through_retries(jobs))
            if period == PRIOR_PERIOD:
                # `/companies` prices its rows from the LATEST statistics row,
                # so read once here, while the latest row is the un-priced
                # week's. Paired with the published edition below it is the
                # real race the page has to survive — two reads that landed in
                # different weeks — and neither body has to be hand-edited to
                # make the two counts disagree.
                with _reading_at(read_at[PRIOR_PERIOD]):
                    out["companies_warming_up"] = _get(client, f"/api/industries/{slug(code)}/companies")

    latest = rs.latest_good(code, version=info)
    if latest is None:
        raise SystemExit(
            f"no published edition for {code} after draining both periods — read the `drained` "
            "lines above; a validator rejection there is the bug to fix, not to capture around"
            + ("" if stub_analyst else " (with --no-stub-analyst and no model, nothing ever publishes)")
        )
    print("latest good edition:", latest["version"])

    for name, path in [
        ("report", f"/api/industries/{slug(code)}/report"),
        ("companies", f"/api/industries/{slug(code)}/companies"),
        ("history", f"/api/industries/{slug(code)}/history"),
        ("changes", f"/api/industries/{slug(code)}/changes"),
        # The two below-the-floor editions, so the UI can be tested against
        # the real shapes rather than a hand-edited copy of them: the first
        # week of the big group (membership clears the floor, no prices
        # yet — transient) and the thin group's priced week (every member
        # priced and still short — structural).
        ("report_warming_up", f"/api/industries/{slug(code)}/report?version=1"),
        ("report_universe_short", f"/api/industries/{slug(short_code)}/report"),
        ("report_missing", f"/api/industries/{slug(empty_code)}/report"),
    ]:
        with _reading_at(read_at[PERIOD]):
            out[name] = _get(client, path, expect=404 if name == "report_missing" else 200, name=name)

    # The model is DOWN for a week (no stub): every attempt falls back to the
    # template, the non-final ones are retried, the final one is stored
    # audit-only. The thin group keeps its analyst edition, marked not
    # updated; the template-only group has no analyst edition at all.
    res = jobs.enqueue_period(OUTAGE_PERIOD, codes=[short_code, no_agentic_code], version=info,
                              source="fixture", include_cross_snapshot=False)
    print("enqueued (model down)", OUTAGE_PERIOD, res["enqueued"])
    _print_drained(drain_through_retries(jobs))
    with _reading_at(read_at[OUTAGE_PERIOD]):
        out["report_not_updated"] = _get(client, f"/api/industries/{slug(short_code)}/report",
                                         name="report_not_updated")
        out["report_no_agentic"] = _get(client, f"/api/industries/{slug(no_agentic_code)}/report", expect=404,
                                        name="report_no_agentic")
        # Last, so the picker pointers describe the final state (the thin
        # group's pointer names its analyst edition and the newer attempt).
        out["taxonomy"] = _get(client, "/api/industries/taxonomy", name="taxonomy")

    trimmed: dict[str, int] = {}
    for name in ("report", "report_warming_up", "report_universe_short", "report_not_updated"):
        for path, n in trim_weekly_closes(out[name]).items():
            trimmed[f"{name}.{path}"] = n
    # The fixture is public data like every body in it: groups are named by
    # their public SLUG and the taxonomy by its public key (owner decision
    # 2026-09-24), so the UI's own fixture walk can assert that nothing in
    # the file is a taxonomy code.
    meta = {
        "generated_by": GENERATED_BY,
        "taxonomy_version": industry_labels.public_version_key(info.version_key),
        "code": industry_labels.slug(code),
        "short_code": industry_labels.slug(short_code),
        "empty_code": industry_labels.slug(empty_code),
        "no_agentic_code": industry_labels.slug(no_agentic_code),
        "outage_period": OUTAGE_PERIOD,
        "read_at": {p: t.isoformat() for p, t in read_at.items()},
        "min_sample": floor,
        "analyst_stub": {
            "enabled": bool(stub_analyst),
            "label": industry_analyst_stub.LABEL,
            "model": industry_analyst_stub.MODEL,
            "reason": industry_analyst_stub.REASON,
            "sections_not_written": {k: list(v) for k, v in NOT_WRITTEN.items()} if stub_analyst else {},
            # The outlook's registered assumption is the stub's only addition
            # to the template prose; declared so nobody reads it as a forecast.
            "forecast_assumption": (
                f"{industry_analyst_stub.ASSUMPTION_ID} holds the edition's first rate anchor at its own "
                "observed value; no number is invented" if stub_analyst else None
            ),
        },
        "trimmed": trimmed,
        "trimmed_note": TRIMMED_NOTE,
        "omitted_responses": OMITTED,
        "omitted_reason": OMITTED_REASON,
    }
    return {"meta": meta, **out}


ORIGIN = "script:capture_industry_ui_fixture"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument("--stub-analyst", action=argparse.BooleanOptionalAction, default=True,
                        help="use the deterministic stand-in analyst model (default on; see the module docstring)")
    args = parser.parse_args(argv)

    refusal = _refuse_unsafe_database()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    from app.agents.llm import llm_call_context

    # The drained industry jobs keep an outer origin, so their rows say this
    # script started them rather than the generic `worker:industry`.
    with llm_call_context(origin=ORIGIN):
        payload = capture(stub_analyst=args.stub_analyst)
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
