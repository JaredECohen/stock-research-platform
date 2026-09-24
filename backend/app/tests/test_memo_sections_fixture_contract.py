"""The frontend's presented-memo fixture is the presenter's output.

`frontend/src/test/fixtures/memo-sections.wire.json` is what the memo
renderers are tested against (S12). A hand-written fixture drifts from its
producer; this one is `memo_sections.present_memo` over the committed,
minimized memo fixtures, serialized as the API serializes a memo
(`test_industry_ui_fixture_contract.py` is the precedent).
"""
from __future__ import annotations

import json
import re

from app.scripts import capture_memo_sections_fixture as capture
from app.services.memo_sections import SIG


def test_frontend_fixture_is_presenter_output():
    assert capture.FRONTEND_FIXTURE.exists(), (
        f"{capture.FRONTEND_FIXTURE} is missing; run "
        "`python -m app.scripts.capture_memo_sections_fixture` from backend/"
    )
    committed = capture.FRONTEND_FIXTURE.read_text()
    expected = capture.render(capture.capture())
    assert json.loads(committed) == json.loads(expected), (
        "memo-sections.wire.json is stale: rerun "
        "`python -m app.scripts.capture_memo_sections_fixture` from backend/"
    )
    assert committed == expected  # byte-stable too, so the diff is reviewable


def test_frontend_fixture_carries_every_scenario():
    payload = json.loads(capture.FRONTEND_FIXTURE.read_text())
    assert set(payload["memos"]) == set(capture.NAMES)
    for name, memo in payload["memos"].items():
        assert memo["section_availability"], name
        assert "Unavailable in this version." in json.dumps(memo) or name == "meta_v1", name


def test_frontend_pm_tail_constant_is_the_backend_signature():
    """The renderer tests assert the PDF/card never print `PM_TEMPLATE_TAIL`;
    a hand-copied string that drifted from the signature would make that
    assertion check for text the presenter no longer matches."""
    loader = capture.FRONTEND_FIXTURE.parent / "memoSections.ts"
    m = re.search(r'export const PM_TEMPLATE_TAIL =\s*"([^"]+)";', loader.read_text())
    assert m, f"PM_TEMPLATE_TAIL not found in {loader}"
    assert m.group(1) == SIG["pm_view_tail"].text
