"""Wave 8D — `/api/stocks/{t}/memory` endpoint test.

Surfaces long-term memory entries (Wave 3D structured_facts included)
to the UI without requiring filesystem access.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.memory import CompanyMemory, MemoryEntry


def test_stock_memory_endpoint_returns_entries_with_structured_facts(
    tmp_path, monkeypatch,
):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))

    cm = CompanyMemory.for_ticker("MEMTEST")
    cm.append_entry(MemoryEntry(
        date="2026-04-30", trigger="earnings",
        body="Earnings observation body.",
        structured_facts={
            "sources": [
                {
                    "source_kind": "transcript",
                    "source_id": "2024Q4",
                    "facts": {
                        "guidance_changes": ["raised FY guide to $14B"],
                        "capex_commentary": ["$35B AI capex"],
                    },
                },
            ],
            "extractor_version": 1,
        },
    ))
    cm.save()

    c = TestClient(app)
    r = c.get("/api/stocks/MEMTEST/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "MEMTEST"
    assert body["entry_count"] == 1
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["trigger"] == "earnings"
    assert entry["structured_facts"]
    sources = entry["structured_facts"]["sources"]
    assert sources[0]["source_id"] == "2024Q4"
    assert "guidance_changes" in sources[0]["facts"]


def test_stock_memory_endpoint_empty_for_unknown_ticker(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    c = TestClient(app)
    r = c.get("/api/stocks/NEVER_HEARD_OF_THIS/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert body["entry_count"] == 0
