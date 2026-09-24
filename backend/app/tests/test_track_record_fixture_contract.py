"""W6 — the Track Record UI fixture is the API's own output, and stays so.

`frontend/src/test/fixtures/trackRecord.wire.json` was produced by
`app/scripts/capture_track_record_fixture.py`: a fixed seed, the real
eligibility sweep, then `GET /api/admin/track-record` through `TestClient`.
Every Vitest assertion about the Track Record page is made against that
file, which is only worth anything while it is still what the API serves.

Unlike the industry fixture (a demo-universe snapshot whose values drift),
this seed is fixed and `track_record` has no clock, so the comparison is
EXACT: re-run the seed and capture on a private engine and require
equality with the committed file. A renamed key, a changed denominator or a
new block fails here with the re-capture command, before the UI can keep
rendering a shape the API no longer sends.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.scripts import capture_track_record_fixture as producer
from app.services import outcome_service
from app.tests.eligibility_helpers import isolated_sessions

FIXTURE = Path(__file__).resolve().parents[3] / "frontend" / "src" / "test" / "fixtures" / "trackRecord.wire.json"
RECAPTURE = (
    "re-capture it: cd backend && ENABLE_LIVE_DATA=false USE_DEMO_DATA=true "
    'OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" '
    'DATABASE_URL="sqlite:////tmp/track-record-fixture.db" '
    "python -m app.scripts.capture_track_record_fixture"
)


def test_fixture_equals_producer(tmp_path, monkeypatch):
    sessions, engine = isolated_sessions(tmp_path, monkeypatch, outcome_service)
    try:
        produced = json.loads(json.dumps(producer.build(sessions)))
    finally:
        engine.dispose()
    committed = json.loads(FIXTURE.read_text())
    assert produced == committed, f"trackRecord.wire.json no longer matches the API; {RECAPTURE}"


def test_fixture_exercises_every_page_state():
    """The states the UI tests rely on are present in the capture itself."""
    body = json.loads(FIXTURE.read_text())
    established, provisional, empty = body["established_90d"], body["provisional_30d"], body["empty_365d"]
    assert established["provisional"]["is_provisional"] is False
    assert provisional["provisional"]["is_provisional"] is True
    assert empty["total"] == 0
    assert established["eligibility"]["excluded_by_reason"] == {
        "demo_dev_copy_2026_05_04": 3, "generation_mode_unrecorded": 1,
    }
    assert established["coverage"]["late_evaluation_candidates"] == 1
    assert set(established["rating_mix_by_source"]) == {"keyword_pm", "llm_pm"}
