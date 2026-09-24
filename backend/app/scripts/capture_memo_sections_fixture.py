"""Regenerate the frontend's presented-memo fixture from the backend presenter.

    python -m app.scripts.capture_memo_sections_fixture [--check]

`frontend/src/test/fixtures/memo-sections.wire.json` is what the memo
renderers are tested against (S12). It is not hand-written: it is
`memo_sections.present_memo` over the committed, minimized fixtures in
`app/tests/fixtures/memo_sections/`, serialized as the API serializes a
memo. `test_memo_sections_fixture_contract.py` fails when the committed file
and this output differ — rerun this script after any presenter change.

Reads committed files and writes one file: no database, no provider, no LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..schemas import StockMemoOut
from ..services.memo_sections import PRESENTATION_VERSION, present_memo

BACKEND_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "memo_sections"
FRONTEND_FIXTURE = (
    Path(__file__).resolve().parents[3] / "frontend" / "src" / "test" / "fixtures"
    / "memo-sections.wire.json"
)
NAMES = ("aapl_demo", "abbv_v7_patch", "googl_live_prepflag", "meta_v1", "msft_live")


def capture() -> dict[str, Any]:
    """`{presentation_version, memos: {name: presented memo (JSON mode)}}`."""
    memos: dict[str, Any] = {}
    for name in NAMES:
        raw = json.loads((BACKEND_FIXTURES / f"{name}.json").read_text())
        presented = present_memo(StockMemoOut.model_validate(raw))
        memos[name] = json.loads(presented.model_dump_json())
    return {"presentation_version": PRESENTATION_VERSION, "memos": memos}


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    text = render(capture())
    if args.check:
        current = FRONTEND_FIXTURE.read_text() if FRONTEND_FIXTURE.exists() else ""
        if current != text:
            print(f"{FRONTEND_FIXTURE} is stale; rerun without --check")
            return 1
        return 0
    FRONTEND_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_FIXTURE.write_text(text)
    print(f"wrote {FRONTEND_FIXTURE} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
