"""Capture ``frontend/src/test/fixtures/trackRecord.wire.json`` — the wire
fixture every Track Record UI test reads (W6 / FIX-007).

Usage (from ``backend/``)::

    ENABLE_LIVE_DATA=false USE_DEMO_DATA=true \\
    OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" \\
    DATABASE_URL="sqlite:////tmp/track-record-fixture.db" \\
    python -m app.scripts.capture_track_record_fixture

The fixture is NOT hand-written: it is what ``GET /api/admin/track-record``
served, read through ``TestClient``, after the real eligibility sweep ran
over a seeded throwaway sqlite database. The seed and the capture are
exposed as ``seed(db)`` and ``capture(client)`` so
``app/tests/test_track_record_fixture_contract.py`` re-runs them and fails,
with the re-capture command, the moment the producer's output and the
committed file differ.

The script never opens the configured database: it builds its own
temporary sqlite file and points ``outcome_service`` at it. It still
refuses to run with live data enabled or a non-local ``DATABASE_URL``, so
a mistaken environment cannot be half-used.

What the seed contains (fixed dates; ``track_record`` has no clock, so the
capture is deterministic):

* 40 companies, 2 of them ETFs (a 38-company universe);
* 36 companies with 3 live snapshots each from 2026-07-01, each with a
  90-day outcome: 108 directional calls on 36 companies, so 90d clears the
  provisional thresholds (30 companies, 100 directional calls);
* 30-day outcomes for only 8 of those companies, so 30d is provisional;
* rating sources that differ by company (keyword PM, LLM PM, degraded PM),
  one late-evaluation candidate at 90d;
* 3 dev-copy demo snapshots (enumerated evidence ids, 2026-05-03/04) with
  30d and 90d outcomes, and 1 snapshot with no generation mode: the
  exclusion note's rows;
* nothing at 365 days: the empty state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "frontend" / "src" / "test" / "fixtures" / "trackRecord.wire.json"

BODIES = (("established_90d", 90), ("provisional_30d", 30), ("empty_365d", 365))
GENERATED_BY = (
    "python -m app.scripts.capture_track_record_fixture — seed(db), the real eligibility sweep, "
    "then GET /api/admin/track-record through TestClient"
)

KEYWORD_VIEW = (
    "Research view: {rating}. Seeded thesis. Sector framing supports the cohort thesis; "
    "valuation-relative read is the main swing factor. The risk committee flagged the dominant "
    "downside scenarios; portfolio fit depends on macro view."
)
RATINGS = ("Bullish", "Very Bullish", "Bearish", "Bullish", "Very Bearish", "Bullish")
START = datetime(2026, 7, 1, 14, 0)


def _alpha(i: int, j: int) -> float:
    # Deterministic, spread either side of zero, no PRNG.
    return round(((i * 7 + j * 3) % 21 - 10) / 100, 4)


def seed(db: Any) -> dict[str, int]:
    """Populate an EMPTY database (tables already created). Returns counts."""
    from app.models import Company, MemoOutcome, MemoSnapshot
    from app.services.outcome_eligibility import classify_pending
    from app.services.outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS
    from app.services.outcome_service import _thesis_held

    def outcome(snap: Any, horizon: int, alpha: float, *, late: bool = False) -> None:
        fwd = round(alpha + 0.03, 4)
        rating = snap.memo_json["rating_label"]
        db.add(MemoOutcome(
            memo_snapshot_id=snap.id, ticker=snap.ticker, rating_at_memo=rating,
            confidence_at_memo=60.0, price_at_memo=100.0, horizon_days=horizon,
            evaluated_at=snap.generated_at + timedelta(days=(200 if late else horizon + 1)),
            forward_return=fwd, benchmark_return=round(fwd - alpha, 4), alpha=alpha,
            thesis_held=_thesis_held(rating, fwd), note="fixture seed",
        ))

    for i in range(40):
        ticker = f"CO{i:02d}"
        db.add(Company(
            ticker=ticker, company_name=f"Company {i:02d}", sector="Technology" if i % 2 else "Energy",
            industry="Seeded", is_etf=i >= 38,
        ))
    db.flush()
    # The dev copy first: its ids are the enumerated evidence ids (the sweep
    # refuses any other), and autoincrement then continues above them.
    for sid, ticker, generated in [row for row in DEV_COPY_SNAPSHOTS if row[1] in {"NVDA", "MSFT"}][:3]:
        at = datetime.fromisoformat(generated)
        dev = MemoSnapshot(
            id=sid, ticker=ticker, version=sid, trigger="full_reanalysis", revision_log=[], generated_at=at,
            memo_json={"ticker": ticker, "rating_label": "Bullish", "generation_mode": "demo",
                       "generated_at": generated, "sector": "Technology"},
        )
        db.add(dev)
        db.flush()
        outcome(dev, 30, 0.2)
        outcome(dev, 90, 0.3)
    for i in range(36):
        ticker = f"CO{i:02d}"
        for j in range(3):
            rating = RATINGS[(i + j) % len(RATINGS)]
            memo: dict[str, Any] = {
                "ticker": ticker, "rating_label": rating, "generation_mode": "live",
                "sector": "Technology" if i % 2 else "Energy",
            }
            if i % 3 == 0:
                memo["final_pm_view"] = KEYWORD_VIEW.format(rating=rating)
            elif i % 3 == 1:
                memo["final_pm_view"] = f"Seeded LLM view: {rating}."
            else:
                memo["final_pm_view"] = "Seeded view."
                memo["degraded_agents"] = ["PM Synthesis"]
            snap = MemoSnapshot(
                ticker=ticker, version=j + 1, trigger="full_reanalysis", memo_json=memo, revision_log=[],
                generated_at=START + timedelta(days=i, hours=j),
            )
            db.add(snap)
            db.flush()
            outcome(snap, 90, _alpha(i, j), late=(i == 5 and j == 0))
            if i < 8 and j == 0:
                outcome(snap, 30, _alpha(i, j + 1))
    legacy = MemoSnapshot(
        ticker="CO37", version=1, trigger="first_run", revision_log=[], generated_at=datetime(2026, 6, 20, 9),
        memo_json={"ticker": "CO37", "rating_label": "Bullish", "sector": "Energy"},
    )
    db.add(legacy)
    db.flush()
    outcome(legacy, 90, 0.05)
    db.commit()
    summary = classify_pending(db=db)
    return {"classified": summary["classified"]}


def capture(client: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "meta": {
            "generated_by": GENERATED_BY,
            "endpoint": "GET /api/admin/track-record",
            "bodies": {name: {"horizon_days": h} for name, h in BODIES},
            "note": "Nothing was edited after capture: every value is what the API served.",
        }
    }
    for name, horizon in BODIES:
        response = client.get(f"/api/admin/track-record?horizon_days={horizon}")
        if response.status_code != 200:
            raise RuntimeError(f"track-record {horizon}d answered {response.status_code}")
        out[name] = response.json()
    return out


def build(sessions: Any) -> dict[str, Any]:
    """Seed ``sessions``' empty database and capture through the real app."""
    from fastapi.testclient import TestClient

    from app.main import app

    with sessions() as db:
        seed(db)
    return capture(TestClient(app))


def _refuse_unsafe_environment() -> str | None:
    from app.config import settings
    from app.tests import dbguard

    if settings.enable_live_data or not settings.use_demo_data:
        return "refusing to capture with live data enabled — rerun with ENABLE_LIVE_DATA=false USE_DEMO_DATA=true"
    if not settings.database_url.startswith("sqlite"):
        return "refusing: point DATABASE_URL at a throwaway sqlite file, e.g. sqlite:////tmp/track-record-fixture.db"
    return dbguard.refusal(settings.database_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args(argv)
    refusal = _refuse_unsafe_environment()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.services import outcome_service

    with tempfile.TemporaryDirectory() as tmp:
        engine = create_engine(f"sqlite:///{Path(tmp) / 'track-record-fixture.db'}")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        # The route opens its session through outcome_service; nothing else
        # it calls touches the configured database.
        original = outcome_service.SessionLocal
        outcome_service.SessionLocal = sessions  # type: ignore[assignment]
        try:
            payload = build(sessions)
        finally:
            outcome_service.SessionLocal = original  # type: ignore[assignment]
            engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print("now run: npx vitest run in frontend/, and pytest app/tests/test_track_record_fixture_contract.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
