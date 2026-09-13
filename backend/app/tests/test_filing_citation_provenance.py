import json
from datetime import date

import pytest

from app.agents import filing_agent
from app.services import research_notes, vector_store

FILINGS = [{
    "type": "10-K", "accession_number": "annual-accession",
    "period_end": "2025-12-31", "filing_date": "2026-01-30",
    "mda": "Annual operating discussion.", "risk_factors": ["Annual risk."],
}]


@pytest.fixture
def run(monkeypatch):
    prompts = []

    def response(prompt, **kwargs):
        prompts.append((prompt, kwargs))
        return {"headline": "Filing view", "summary": "Summary", "key_points": [], "confidence": 0.7}

    monkeypatch.setattr(filing_agent.llm, "chat_json", response)
    monkeypatch.setattr(research_notes, "build_notes_block_for_agent", lambda *a, **k: "")
    monkeypatch.setattr(filing_agent.retrieval_service, "search", lambda *a, **k: [])

    def call(hits):
        monkeypatch.setattr(vector_store, "search", lambda *a, **k: hits)
        return filing_agent.run_filing_agent({"ticker": "META"}, FILINGS)

    return call, prompts


def test_vector_passages_retain_their_own_filing_and_period(run):
    call, prompts = run
    hits = [
        {"id": 11, "source_id": 21, "text": "Quarterly revenue was 60.8 billion.",
         "section": "mda", "period_end": date(2026, 6, 30),
         "meta": {"accession": "quarter-accession", "filing_date": "2026-07-30", "filing_type": "10-Q"}},
        {"id": 12, "source_id": 22, "text": "A separate filing describes the new risk.",
         "section": "risk_factors", "meta": {"accession": "other-accession"}},
    ]
    finding = call(hits)
    refs = {c.excerpt: c.ref for c in finding.evidence}
    assert refs["Annual operating discussion."] == "annual-accession"
    assert refs[hits[0]["text"]] == "quarter-accession"
    assert refs[hits[1]["text"]] == "other-accession"
    assert finding.sources == ["filing:annual-accession", "filing:quarter-accession", "filing:other-accession"]
    assert finding.data["retrieved_sources"][0]["period_end"] == "2026-06-30"
    assert finding.data["retrieved_sources"][0]["source_id"] == 21
    assert finding.data["retrieved_sources"][0]["chunk_id"] == 11
    assert finding.data["retrieved_sources"][0]["filing_type"] == "10-Q"
    payload = json.loads(prompts[0][0].split("Filing context:\n")[1])
    assert payload["retrieved_chunks"] == [h["text"] for h in hits]
    # Only citation metadata changes: the analyst's request stays identical.
    hits[0]["meta"]["accession"] = "corrected-quarter-accession"
    corrected = call(hits)
    assert prompts[1] == prompts[0]
    assert corrected.evidence[2].ref == "corrected-quarter-accession"


def test_bm25_accession_is_not_replaced_with_primary(run, monkeypatch):
    call, _ = run
    monkeypatch.setattr(filing_agent.retrieval_service, "search", lambda *a, **k: [
        {"source_type": "filing", "source_id": "bm25-quarter-accession",
         "section": "mda", "text": "Quarterly operating discussion."},
        {"source_type": "news", "source_id": "news-url", "text": "Unrelated news."},
    ])
    finding = call([])
    assert finding.evidence[-1].ref == "bm25-quarter-accession"
    assert finding.sources == ["filing:annual-accession", "filing:bm25-quarter-accession"]
    assert len(finding.data["retrieved_sources"]) == 1


def test_missing_accession_uses_internal_identity_or_explicit_gap(run, caplog):
    call, _ = run
    finding = call([
        {"id": 90, "source_id": 2025, "text": "Known stored document."},
        {"id": 91, "text": "Unattributed first passage."},
        {"id": 92, "text": "Unattributed second passage."},
    ])
    assert [c.ref for c in finding.evidence[-3:]] == ["filing_doc:2025", "unattributed_chunk:91", "unattributed_chunk:92"]
    assert [c.kind for c in finding.evidence[-3:]] == ["filing", "other", "other"]
    assert finding.sources == ["filing:annual-accession", "filing:filing_doc:2025"]
    assert finding.data["unattributed_retrieved_chunks"] == ["unattributed_chunk:91", "unattributed_chunk:92"]
    assert "2 unattributed retrieved chunks: unattributed_chunk:91, unattributed_chunk:92" in caplog.text


def test_deterministic_finding_keeps_retrieval_provenance(run, monkeypatch):
    call, _ = run
    monkeypatch.setattr(filing_agent.llm, "chat_json", lambda *a, **k: None)
    finding = call([{"source_id": 22, "id": 31, "text": "Quarterly margin expanded on new product demand.",
                    "section": "mda", "meta": {"accession": "quarter-accession"}}])
    assert finding.sources == ["filing:annual-accession", "filing:quarter-accession"]
    assert finding.data["retrieved_sources"][0]["ref"] == "quarter-accession"
    assert finding.confidence == 0.6


def test_actual_bm25_missing_accession_placeholder_is_unattributed(run, monkeypatch):
    call, _ = run
    retrieval = filing_agent.retrieval_service
    monkeypatch.setattr(retrieval, "get_filings", lambda *a: [{"business_description": "Business without an accession."}])
    monkeypatch.setattr(retrieval, "get_transcripts", lambda *a: [])
    monkeypatch.setattr(retrieval, "get_news", lambda *a: [])
    chunks = retrieval._chunks_for_ticker("META")
    assert chunks[0]["source_id"] == "10K"
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: chunks)
    finding = call([])
    assert finding.evidence[-1].kind == "other"
    assert finding.evidence[-1].ref == "unattributed_chunk:1"
    assert finding.sources == ["filing:annual-accession"]
    assert finding.data["unattributed_retrieved_chunks"] == ["unattributed_chunk:1"]
