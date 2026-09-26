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
from typing import get_args

from app.schemas.memo import SectionReason
from app.scripts import capture_memo_sections_fixture as capture
from app.services.memo_sections import SECTION_KEYS, SIG


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


MEMO_SECTIONS_TS = capture.FRONTEND_FIXTURE.parents[2] / "lib" / "memoSections.ts"


def test_frontend_section_keys_match_the_presenter():
    """D4: `memoSections.ts` keeps key-set parity with the presenter, in
    order, so a section the backend adds (the debate) is one the frontend
    knows the place of."""
    source = MEMO_SECTIONS_TS.read_text()
    m = re.search(r"export const SECTION_KEYS = \[(.*?)\] as const;", source, re.S)
    assert m, f"SECTION_KEYS not found in {MEMO_SECTIONS_TS}"
    assert tuple(re.findall(r'"([^"]+)"', m.group(1))) == SECTION_KEYS


def test_frontend_reason_sentences_cover_the_backend_vocabulary():
    """Every reason the presenter can emit has a sentence, and no more: the
    TypeScript `Record` catches a missing one only if the union was updated.

    A guard, not a regression (the two agreed before D4): it keeps the
    debate's new reasons, and any later one, worded on both sides."""
    source = MEMO_SECTIONS_TS.read_text()
    m = re.search(r"export const REASON_TEXT: Record<SectionReason, string> = \{(.*?)\n\};", source, re.S)
    assert m, f"REASON_TEXT not found in {MEMO_SECTIONS_TS}"
    words = set(re.findall(r"^\s+(\w+):", m.group(1), re.M))
    assert words == set(get_args(SectionReason))


def test_fixture_labels_every_legacy_review_as_not_live():
    """P12: the five legacy memos all carry a review that was not live; the
    served body labels each "not independently reviewed", and carries the
    debate entry as not produced."""
    payload = json.loads(capture.FRONTEND_FIXTURE.read_text())
    assert payload["presentation_version"] == 2
    for name, memo in payload["memos"].items():
        assert memo["risk_committee_challenge"]["review_status"] == "not_independent", name
        assert memo["section_availability"]["debate"]["reason"] == "not_produced", name
        assert memo["debate"] is None, name


def test_frontend_pm_tail_constant_is_the_backend_signature():
    """The renderer tests assert the PDF/card never print `PM_TEMPLATE_TAIL`;
    a hand-copied string that drifted from the signature would make that
    assertion check for text the presenter no longer matches."""
    loader = capture.FRONTEND_FIXTURE.parent / "memoSections.ts"
    m = re.search(r'export const PM_TEMPLATE_TAIL =\s*"([^"]+)";', loader.read_text())
    assert m, f"PM_TEMPLATE_TAIL not found in {loader}"
    assert m.group(1) == SIG["pm_view_tail"].text
