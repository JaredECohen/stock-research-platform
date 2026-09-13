"""Filing memory status reflects real files and reaches ingest failure notes."""
import logging
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.agents import llm
from app.config import settings
from app.memory import CompanyMemory, SectorMemory
from app.models import Company, FilingDoc
from app.services import filing_memory as fm
from app.services import history_service, vector_store


@pytest.fixture
def source(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'filing-memory.db'}")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (database, fm, history_service, vector_store):
        monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")

    def no_chat(*args, **kwargs):
        raise AssertionError("memory writer tests must not call a model")

    monkeypatch.setattr(llm, "chat_json", no_chat)
    monkeypatch.setattr(fm, "_llm_diff", lambda *args: {
        "bullets": ["Costs increased."], "sector_relevant": True,
        "sector_pattern": "Component constraints affect peer margins.",
    })
    with sessions() as db:
        db.add(Company(ticker="FMTRUTH", company_name="Memory Test", sector="Technology",
                       industry="Semiconductors", universe_tier="auto_analysis"))
        rows = [FilingDoc(
            ticker="FMTRUTH", accession_number=f"FMTRUTH-{i}", filing_type="10-Q",
            filing_date=date.today() - timedelta(days=90 * (1 - i)),
            sections={"mda": "Revenue increased and costs rose."}, raw_text="original source",
        ) for i in range(2)]
        db.add_all(rows)
        db.commit()
    yield sessions, rows
    engine.dispose()


def test_ingest_routes_sector_pattern_using_only_the_stored_company(source):
    sessions, rows = source
    assert history_service.run_ingest_post_passes([rows[1].id], []) == []
    assert "Costs increased" in CompanyMemory.for_ticker("FMTRUTH").path.read_text()
    assert "Component constraints" in SectorMemory.for_sector("Technology").path.read_text()
    with sessions() as db:
        assert db.get(FilingDoc, rows[1].id).raw_text == "original source"


@pytest.mark.parametrize("target", ["company", "sector"])
def test_partial_save_failure_never_claims_the_failed_destination_was_written(source, monkeypatch, caplog, target):
    _, rows = source

    def fail_save(self):
        raise OSError("private file failure detail")

    monkeypatch.setattr(CompanyMemory if target == "company" else SectorMemory, "save", fail_save)
    with caplog.at_level(logging.INFO):
        report = fm.post_pass(rows[1])
    assert report["company_memory_written"] is (target != "company")
    assert report["sector_pattern_written"] is (target != "sector")
    assert report["errors"] == [{"stage": f"{target}_memory", "error_type": "OSError"}]
    assert report["memory_writes"][target] == {"status": "failed", "error_type": "OSError"}
    assert report["filing_id"] == rows[1].id
    assert report["accession_number"] in caplog.text
    assert "private file failure detail" not in caplog.text


def test_disabled_flag_blocks_both_files_and_reports_healthy_disabled_status(source, monkeypatch, caplog):
    _, rows = source
    monkeypatch.setattr(settings, "enable_long_term_memory", False)

    def no_open(*args):
        raise AssertionError("disabled memory must not open files")

    monkeypatch.setattr(CompanyMemory, "for_ticker", no_open)
    monkeypatch.setattr(SectorMemory, "for_sector", no_open)
    with caplog.at_level(logging.INFO):
        report = fm.post_pass(rows[1])
    assert not report["company_memory_written"] and not report["sector_pattern_written"]
    assert report["errors"] == []
    assert all(result["status"] == "disabled" for result in report["memory_writes"].values())
    assert "status=disabled" in caplog.text and rows[1].accession_number in caplog.text


def test_missing_stored_sector_is_explicit_after_company_save(source):
    sessions, rows = source
    with sessions() as db:
        db.delete(db.get(Company, "FMTRUTH"))
        db.commit()
    report = fm.post_pass(rows[1])
    assert report["company_memory_written"] and not report["sector_pattern_written"]
    assert report["errors"] == [{"stage": "sector_memory", "error_type": "sector_unavailable"}]


def test_every_memory_failure_reaches_existing_ingest_identity_and_note_contract(source, monkeypatch):
    _, rows = source

    def fail_save(self):
        raise PermissionError("private")

    monkeypatch.setattr(CompanyMemory, "save", fail_save)
    failures = history_service.run_ingest_post_passes([row.id for row in rows], [])
    assert {failure["id"] for failure in failures} == {row.id for row in rows}
    assert len(failures) == 2
    note = history_service.post_pass_failure_note(failures)
    for row in rows:
        assert f"FMTRUTH:filing:{row.id}:company_memory:PermissionError" in note


@pytest.mark.parametrize("enabled", [False, True])
def test_weekly_digest_writers_obey_flag_and_sector_import_works(source, monkeypatch, enabled):
    monkeypatch.setattr(settings, "enable_long_term_memory", enabled)
    company = fm.weekly_digest("FMTRUTH")
    sector = fm.weekly_sector_digest("Technology")
    assert company["wrote_memory"] is enabled
    assert sector["wrote_memory"] is enabled
    assert CompanyMemory.for_ticker("FMTRUTH").path.exists() is enabled
    assert SectorMemory.for_sector("Technology").path.exists() is enabled
