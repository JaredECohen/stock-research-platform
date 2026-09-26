"""`retrieval_service.search_many`: the bull/bear debate's BM25 fallback
(slice B8-D3; design §5.3; critique #9)."""
from __future__ import annotations

import pytest

from app.services import retrieval_service as rs

MDA = " ".join(f"Paragraph {i} discusses ordinary operations and seasonal inventory." for i in range(60))
MDA += " The export license for the accelerator line was revoked by regulators in March. "
MDA += " ".join(f"Closing paragraph {i} restates segment results." for i in range(40))


def _docs(ticker: str) -> list[dict]:
    return [
        dict(ticker=ticker, source_type="filing", source_id="0001-25-000001", section="mda",
             title="10-K MD&A", url="", text=MDA),
        dict(ticker=ticker, source_type="filing", source_id="0001-25-000001", section="risk_factors",
             title="10-K risk factor", url="", text="Customers may design custom accelerators."),
        dict(ticker=ticker, source_type="filing", source_id="0001-25-000001", section="risk_factors",
             title="10-K risk factor", url="", text="Export rules may restrict sales to some regions."),
        dict(ticker=ticker, source_type="transcript", source_id="2025Q4", section="qa",
             title="Earnings call Q&A 2025Q4", url="", text="Backlog doubled and export demand is steady."),
        dict(ticker=ticker, source_type="news", source_id="https://n.example/1", section="article",
             title="Export curbs", url="https://n.example/1", text="Export curbs. New rules announced."),
    ]


def test_search_many_builds_one_index(monkeypatch):
    loads, builds = [], []
    monkeypatch.setattr(rs, "_chunks_for_ticker", lambda t: loads.append(t) or _docs(t))
    real_build = rs._build_index
    monkeypatch.setattr(rs, "_build_index", lambda chunks: builds.append(len(chunks)) or real_build(chunks))
    out = rs.search_many("ACME", ["export license revoked", "custom accelerators", "export curbs"],
                         limit=3, source_types=["filing", "filing", "news"])
    assert loads == ["ACME"] and len(builds) == 1, "one corpus load and ONE index for every query"
    assert len(out) == 3
    assert out[0] and all(h["source_type"] == "filing" for h in out[0])
    assert out[2] and all(h["source_type"] == "news" for h in out[2])
    extra = [{"id": "pack1", "source_type": "news", "title": "Pack item", "text": "Export ban widened."}]
    out = rs.search_many("ACME", ["export ban widened"], source_types=["news"], extra_chunks=extra)
    assert out[0][0]["id"] == "pack1", "the debate's news pack is indexed with the provider rows"
    with pytest.raises(ValueError):
        rs.search_many("", ["q"])
    with pytest.raises(ValueError):
        rs.search_many("ACME", ["q", "r"], source_types=["filing"])


def test_bm25_fallback_chunks_long_docs_stable_ids(monkeypatch):
    monkeypatch.setattr(rs, "_chunks_for_ticker", _docs)
    pieces = rs.split_long_documents(_docs("ACME"))
    mda = [p for p in pieces if p["section"] == "mda"]
    assert len(mda) > 3 and all(len(p["text"]) <= rs.WINDOW_CHARS for p in mda)
    ids = [p["id"] for p in pieces]
    assert len(ids) == len(set(ids)), "windows and same-section risk factors never collide"
    assert all(i.startswith("0001-25-000001:") for i in ids if "0001-25-000001" in i)
    risk = [p["id"] for p in pieces if p["section"] == "risk_factors"]
    assert len(risk) == 2 and risk[0] != risk[1]
    assert ids == [p["id"] for p in rs.split_long_documents(_docs("ACME"))], "stable across calls"
    # The passage returned is the span that matched, not the document's start.
    hit = rs.search_many("ACME", ["export license revoked regulators"], source_types=["filing"])[0][0]
    assert "export license" in hit["text"] and not hit["text"].startswith("Paragraph 0 ")
    # `search` (the filing analyst's fallback) still indexes whole documents,
    # so the analyst's prompt and refs are unchanged while the debate is off.
    whole = rs.search("ACME", "export license revoked regulators", limit=1)[0]
    assert whole["text"].startswith("Paragraph 0 ") and "id" not in whole
