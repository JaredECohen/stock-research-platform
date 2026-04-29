"""End-to-end smoke test for MarketMosaic demo mode.

Exercises every demo prompt against the in-process FastAPI TestClient.
Run via: `python -m scripts.smoke_test` from the backend directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


PROMPTS = [
    "Analyze NVDA as a long-term investment.",
    "Compare MSFT and GOOGL from a portfolio manager's perspective.",
    "Find 5 high-quality stocks that could benefit from falling rates.",
    "Build a 10-stock portfolio for a soft landing with falling rates and continued AI infrastructure spending.",
    "What sectors benefit if inflation stays sticky?",
    "Run a DCF for MSFT using base-case assumptions.",
    "Show me reasonable valuation growth stocks.",
]


def main() -> int:
    client = TestClient(app)
    health = client.get("/health").json()
    print("health:", health)
    status = client.get("/api/providers/status").json()
    print("mode:", status["mode"], "providers:", list(status["providers"].keys()))
    print("missing_keys:", status["missing_api_keys"])
    print("=" * 80)
    failures = 0
    for i, prompt in enumerate(PROMPTS, 1):
        r = client.post("/api/chat", json={"message": prompt, "history": []})
        if r.status_code != 200:
            print(f"[{i}] FAIL ({r.status_code}): {prompt}")
            failures += 1
            continue
        data = r.json()
        intent = data.get("intent")
        bits: list[str] = [f"intent={intent}"]
        if data.get("memo"):
            m = data["memo"]
            bits.append(f"memo={m['ticker']}/{m['rating_label']} ({int(m['confidence_score'])})")
        if data.get("portfolio"):
            bits.append(f"portfolio={len(data['portfolio']['holdings'])} holdings")
        if data.get("screener"):
            bits.append(f"screener={len(data['screener']['rows'])} rows")
        if data.get("macro"):
            bits.append(f"macro={data['macro']['scenario']}")
        if data.get("dcf"):
            bits.append(f"dcf base={data['dcf']['base']['implied_share_price']:.2f}")
        print(f"[{i}] OK :: {prompt}")
        print("    -", " | ".join(bits))
    print("=" * 80)
    print("failures:" if failures else "all prompts succeeded.", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
