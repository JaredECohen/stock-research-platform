"""Real provider-shaped risk bullets must survive ingestion and post-processing."""
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.database import SessionLocal
from app.models import DocChunk, EarningsTranscript, FilingDoc
from app.monitoring import history_backfill, transcripts_poller
from app.providers.sec_edgar_provider import _extract_sections
from app.services import data_service, filing_memory, history_service, update_orchestrator

_REAL_LLM_DIFF = filing_memory._llm_diff


@pytest.fixture
def source(monkeypatch):
    ticker = "ZF" + uuid4().hex[:8].upper()
    risk = "Component shortages may reduce output and materially increase unit costs beyond the prices customers will accept."
    sections, risks = _extract_sections(
        "Item 1A. Risk Factors\n" + risk + "\n\nItem 2. Management's Discussion\nRevenue rose."
    )
    assert risks == [risk]
    filings = [{
        "accession_number": f"{ticker}-{n}", "type": "10-Q",
        "filing_date": d, "risk_factors": risks, "mda": sections["mda"],
    } for n, d in enumerate(["2026-07-31", "2026-04-30"])]
    provider = SimpleNamespace(
        get_financial_statements=lambda t: {},
        get_filings=lambda t, **kw: filings,
        get_earnings_transcripts=lambda t, **kw: [{
            "period": "2026Q2", "date": "2026-07-31",
            "blocks": [{"speaker": "CFO", "text": "Capex is twenty billion dollars."}],
        }],
        mode=lambda: "test",
    )
    monkeypatch.setattr(data_service, "get_data_service", lambda: provider)
    monkeypatch.setattr(filing_memory, "_llm_diff", lambda *a: None)
    monkeypatch.setattr(filing_memory, "_write_to_company_memory", lambda *a: None)
    monkeypatch.setattr(filing_memory, "_write_to_sector_memory", lambda *a: None)
    yield ticker, risk, filings
    with SessionLocal() as db:
        db.query(DocChunk).filter_by(ticker=ticker).delete(synchronize_session=False)
        db.query(FilingDoc).filter_by(ticker=ticker).delete(synchronize_session=False)
        db.query(EarningsTranscript).filter_by(ticker=ticker).delete(synchronize_session=False)
        db.get_transaction().commit()


def test_sec_risk_bullets_reach_index_and_deterministic_diff(source):
    ticker, risk, _ = source
    result = history_service.backfill_ticker(ticker)
    assert "post_pass_failures" not in result
    with SessionLocal() as db:
        rows = db.query(FilingDoc).filter_by(ticker=ticker).all()
        assert len(rows) == 2
        assert all(row.sections["risk_factors"] == [risk] for row in rows)
        chunks = db.query(DocChunk).filter_by(ticker=ticker, section="risk_factors").all()
        assert len(chunks) == 2
        assert all(risk in row.text for row in chunks)
        transcripts = db.query(DocChunk).filter_by(ticker=ticker, source_type="transcript").all()
        assert transcripts
    new = SimpleNamespace(sections={"risk_factors": [risk]})
    prior = SimpleNamespace(sections={})
    assert filing_memory._deterministic_diff(prior, new) == [
        f"New section disclosed: **risk_factors** ({len(risk.split())} words)."
    ]


def test_structured_segment_rows_are_preserved_as_text():
    assert filing_memory._section_text([{"name": "Cloud", "revenue_share": 0.4}]) == (
        '{"name": "Cloud", "revenue_share": 0.4}'
    )
    with pytest.raises(TypeError, match="unsupported filing section"):
        filing_memory._section_text([42])


@pytest.mark.parametrize("entry", [history_service.backfill_ticker, update_orchestrator._persist_raw_data_only])
def test_both_transaction_owners_commit_before_post_pass(source, monkeypatch, entry):
    ticker, _, _ = source
    checked = []
    real = filing_memory.post_pass

    def post_pass(row):
        with SessionLocal() as reader:
            assert reader.get(FilingDoc, row.id) is not None
        checked.append(row.id)
        return real(row)

    monkeypatch.setattr(filing_memory, "post_pass", post_pass)
    result = entry(ticker)
    assert "post_pass_failures" not in result
    assert len(checked) == 2
    with SessionLocal() as reader:
        assert reader.query(DocChunk).filter_by(ticker=ticker).count() > 0


def test_post_pass_failures_are_isolated_named_and_not_silently_retried(source, monkeypatch, caplog):
    ticker, _, _ = source
    visited = []

    def post_pass(row):
        visited.append(row.id)
        raise ValueError("private provider request must never reach logs")

    monkeypatch.setattr(filing_memory, "post_pass", post_pass)
    transcript_calls = []
    monkeypatch.setattr(filing_memory, "index_transcript", lambda row: transcript_calls.append(row.id))
    result = history_service.backfill_ticker(ticker)
    failures = result["post_pass_failures"]
    assert len(visited) == len(failures) == 2
    assert [f["id"] for f in failures] == visited
    assert all(f["ticker"] == ticker and f["error_type"] == "ValueError" for f in failures)
    assert len(transcript_calls) == 1
    assert "private provider request" not in caplog.text
    for row_id in visited:
        assert f"id={row_id}" in caplog.text
    assert history_service.backfill_ticker(ticker) == {
        "financial_periods": 0, "filings": 0, "transcripts": 0,
    }
    assert len(visited) == 2


def test_index_error_is_reported_even_when_diff_can_continue(source, monkeypatch):
    ticker, _, _ = source
    monkeypatch.setattr(filing_memory, "index_filing", lambda row: (_ for _ in ()).throw(RuntimeError("private")))
    result = history_service.backfill_ticker(ticker)
    failures = result["post_pass_failures"]
    assert len(failures) == 2
    assert all(f["stage"] == "index" and f["error_type"] == "RuntimeError" for f in failures)


def test_diff_never_selects_future_filing_in_committed_newest_first_batch(source):
    ticker, _, _ = source
    history_service.backfill_ticker(ticker)
    with SessionLocal() as db:
        rows = db.query(FilingDoc).filter_by(ticker=ticker).all()
        old = next(r for r in rows if r.filing_date == date(2026, 4, 30))
        new = next(r for r in rows if r.filing_date == date(2026, 7, 31))
    assert filing_memory._prior_filing_of_same_type(old) is None
    assert filing_memory._prior_filing_of_same_type(new).id == old.id


def test_failed_commit_never_runs_post_pass(source, monkeypatch):
    ticker, _, _ = source
    calls = []
    monkeypatch.setattr(filing_memory, "post_pass", lambda row: calls.append(row.id))
    with SessionLocal() as db:
        monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))
        with pytest.raises(RuntimeError, match="commit failed"):
            history_service.backfill_ticker(ticker, db=db)
        db.rollback()
    assert calls == []
    with SessionLocal() as reader:
        assert reader.query(FilingDoc).filter_by(ticker=ticker).count() == 0
        assert reader.query(EarningsTranscript).filter_by(ticker=ticker).count() == 0


def test_history_cron_names_all_post_pass_failures(source, monkeypatch):
    ticker, _, _ = source
    failures = [{"ticker": ticker, "kind": "filing", "id": i,
                 "stage": "index", "error_type": "TypeError"} for i in range(8)]
    monkeypatch.setattr(history_backfill, "backfill_ticker", lambda *a, **kw: {
        "filings": 8, "financial_periods": 0, "transcripts": 0, "post_pass_failures": failures,
    })
    notes = []
    monkeypatch.setattr(history_backfill, "record_run", lambda *a, **kw: notes.append(kw))
    result = history_backfill.run_once(ticker=ticker)
    assert result["filings"] == 8
    assert result["post_pass_errors"] == 8
    assert notes[-1]["success"] is False
    for item in failures:
        assert f"{ticker}:filing:{item['id']}:index:TypeError" in notes[-1]["note"]


@pytest.mark.parametrize("entry", [history_service.backfill_ticker, update_orchestrator._persist_raw_data_only])
def test_source_completeness_survives_storage_index_and_run_report(source, entry):
    ticker, _, filings = source
    metadata = {
        "text_truncated": True, "text_observed_chars": 700000,
        "text_retained_chars": 250000, "text_bytes_read": 900000,
        "html_oversized_tokens": 2,
    }
    filings[0].update(metadata)
    result = entry(ticker)
    assert "post_pass_failures" not in result
    assert result["truncated_filings"] == [{
        "ticker": ticker, "accession_number": filings[0]["accession_number"], **metadata,
    }]
    with SessionLocal() as db:
        row = db.query(FilingDoc).filter_by(accession_number=filings[0]["accession_number"]).one()
        assert row.sections["_source_metadata"] == metadata
        chunks = db.query(DocChunk).filter_by(source_type="filing", source_id=row.id).all()
        assert chunks
        assert all(c.section != "_source_metadata" for c in chunks)
        assert all(c.meta["text_truncated"] and c.meta["text_retained_chars"] == 250000 for c in chunks)


def test_history_cron_names_every_bounded_filing(source, monkeypatch):
    ticker, _, _ = source
    bounded = [{"ticker": ticker, "accession_number": str(i), "text_truncated": True,
                "text_retained_chars": 250000, "text_observed_chars": 800000} for i in range(8)]
    monkeypatch.setattr(history_backfill, "backfill_ticker", lambda *a, **kw: {
        "filings": 8, "financial_periods": 0, "transcripts": 0, "truncated_filings": bounded,
    })
    notes = []
    monkeypatch.setattr(history_backfill, "record_run", lambda *a, **kw: notes.append(kw))
    result = history_backfill.run_once(ticker=ticker)
    assert result["truncated_filings"] == 8
    assert notes[-1]["success"] is True
    for source in bounded:
        assert f"{ticker}:{source['accession_number']}(retained=250000,observed=800000" in notes[-1]["note"]


def test_transcript_cron_carries_raw_filing_post_pass_and_truncation_report(source, monkeypatch):
    ticker, _, _ = source
    failures = [{"ticker": ticker, "kind": "filing", "id": i,
                 "stage": "index", "error_type": "TypeError"} for i in range(7)]
    bounded = [{"ticker": ticker, "accession_number": "2026-000001",
                "text_retained_chars": 250000, "text_observed_chars": 700000}]
    monkeypatch.setattr(transcripts_poller, "get_transcripts", lambda t: [{"period": "2026Q2"}])
    monkeypatch.setattr(transcripts_poller, "_seen_periods", lambda t: {"2026Q1"})
    monkeypatch.setattr(transcripts_poller, "_save_seen_periods", lambda *a: None)
    monkeypatch.setattr(update_orchestrator, "on_transcript_event", lambda *a, **kw: {
        "kind": "skipped", "persisted": {"post_pass_failures": failures, "truncated_filings": bounded},
    })
    notes = []
    monkeypatch.setattr(transcripts_poller, "record_run", lambda *a, **kw: notes.append(kw))
    transcripts_poller.run_once([ticker])
    assert notes[-1]["success"] is False
    for row in failures:
        assert f"{ticker}:filing:{row['id']}:index:TypeError" in notes[-1]["note"]
    assert f"{ticker}:2026-000001(retained=250000,observed=700000" in notes[-1]["note"]


def test_vector_write_error_propagates_to_each_ingest_failure(source, monkeypatch, caplog):
    ticker, _, _ = source
    monkeypatch.setattr(filing_memory.vector_store.emb_svc, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("private request")))
    result = history_service.backfill_ticker(ticker)
    assert len(result["post_pass_failures"]) == 3
    assert all(f["error_type"] == "RuntimeError" for f in result["post_pass_failures"])
    assert "private request" not in caplog.text


def test_llm_diff_includes_provider_list_sections(source, monkeypatch):
    from app.agents import llm
    _, risk, _ = source
    monkeypatch.setattr(filing_memory.settings, "openai_api_key", "test-not-a-secret")
    captured = []
    monkeypatch.setattr(llm, "chat_json", lambda prompt, **kw: captured.append(prompt) or {})
    row = SimpleNamespace(filing_type="10-Q", filing_date=date(2026, 4, 30), sections={"risk_factors": [risk]})
    _REAL_LLM_DIFF(row, row)
    assert risk in captured[0]


@pytest.mark.parametrize("event", [update_orchestrator.on_filing_event, update_orchestrator.on_transcript_event])
def test_raw_provider_failure_is_visible_and_cannot_enqueue(source, monkeypatch, caplog, event):
    ticker, _, _ = source
    provider = data_service.get_data_service()
    provider.get_filings = lambda t: (_ for _ in ()).throw(ValueError("private request"))
    monkeypatch.setattr(update_orchestrator, "should_auto_regen", lambda t: pytest.fail("must not gate/enqueue"))
    result = event(ticker)
    assert result["kind"] == "persist_error"
    assert result["persisted"] == {"filings": 0, "transcripts": 0, "persist_error": {
        "ticker": ticker, "stage": "raw_ingest", "error_type": "ValueError",
    }}
    assert ticker in caplog.text and "ValueError" in caplog.text
    assert "private request" not in caplog.text


def test_raw_commit_failure_resets_uncommitted_counts(source, monkeypatch):
    from sqlalchemy.orm import Session
    ticker, _, _ = source
    monkeypatch.setattr(Session, "commit", lambda self: (_ for _ in ()).throw(RuntimeError("private commit")))
    result = update_orchestrator._persist_raw_data_only(ticker)
    assert result["filings"] == result["transcripts"] == 0
    assert result["persist_error"]["stage"] == "raw_ingest"
    with SessionLocal() as reader:
        assert reader.query(FilingDoc).filter_by(ticker=ticker).count() == 0


@pytest.mark.parametrize("entry", [history_service.backfill_ticker, update_orchestrator._persist_raw_data_only])
def test_failed_filing_body_is_named_durable_and_only_recovered_row_is_indexed(source, entry):
    ticker, risk, filings = source
    filings[0]["text_fetch_error"] = "http_status_504"
    result = entry(ticker)
    assert result["filing_fetch_failures"] == [{
        "ticker": ticker, "accession_number": filings[0]["accession_number"], "error_type": "http_status_504",
    }]
    with SessionLocal() as db:
        failed_row = db.query(FilingDoc).filter_by(accession_number=filings[0]["accession_number"]).one()
        row_id = failed_row.id
        assert failed_row.sections["_source_metadata"]["text_fetch_error"] == "http_status_504"
        assert db.query(DocChunk).filter_by(source_type="filing", source_id=row_id).count() == 0
    del filings[0]["text_fetch_error"]
    recovered = entry(ticker)
    assert "filing_fetch_failures" not in recovered
    assert recovered["filings"] == 1
    with SessionLocal() as db:
        assert db.query(DocChunk).filter_by(source_type="filing", source_id=row_id, section="risk_factors").one().text == risk
    filings[0] = {"accession_number": filings[0]["accession_number"], "text_fetch_error": "empty_document"}
    failed_refresh = entry(ticker)
    assert failed_refresh["filings"] == 0
    with SessionLocal() as db:
        assert db.get(FilingDoc, row_id).sections["risk_factors"] == [risk]


@pytest.mark.parametrize("kind", ["persist_error", "gate_error", "raise"])
def test_transcript_failure_preserves_unhandled_period_for_retry(source, monkeypatch, kind):
    ticker, _, _ = source
    monkeypatch.setattr(transcripts_poller, "get_transcripts", lambda t: [{"period": "2026Q2"}])
    monkeypatch.setattr(transcripts_poller, "_seen_periods", lambda t: {"2026Q1"})
    writes, notes = [], []
    monkeypatch.setattr(transcripts_poller, "_save_seen_periods", lambda t, p: writes.append(p))
    monkeypatch.setattr(transcripts_poller, "record_run", lambda *a, **kw: notes.append(kw))
    def handler(*a, **kw):
        if kind == "raise":
            raise ValueError("private request")
        return {"kind": kind, "persisted": {"persist_error": {
            "ticker": ticker, "stage": "raw_ingest", "error_type": "ValueError",
        }} if kind == "persist_error" else {}}
    monkeypatch.setattr(update_orchestrator, "on_transcript_event", handler)
    transcripts_poller.run_once([ticker])
    assert all("2026Q2" not in periods for periods in writes)
    assert notes[-1]["success"] is False
    assert ticker in notes[-1]["note"]
    assert "private request" not in notes[-1]["note"]
