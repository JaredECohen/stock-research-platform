"""FIX-006 re-pull tooling: the review summary, and the worker-run ledger.

No agent holds production credentials, so the one-time FMP re-pull is in-app
code the deployed worker runs: a dry run by default that records the exact
plan, and execution only after the owner authorizes the reviewed digest.
These tests drive that ledger end to end against sqlite with a fake FMP
chain; nothing reaches a network.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Company, FinancialDataRepair, FinancialPeriod, MarketDataSync
from app.scripts import fundamentals_repull
from app.services import fmp_repull_ledger as ledger
from app.services import fundamental_history_service as svc
from app.services import fundamental_quarantine as fq
from app.services import market_data_backfill as backfill
from app.services import provider_cache

DAYTIME = datetime(2026, 9, 24, 14, 0)
BACKUP = "c" * 64  # the evidence-archive manifest SHA-256 the owner attests to


def _compact(**overrides):
    base = {"status": "complete", "success": True, "fundamentals_success": True, "rows_quarantined": 0,
            "quarantine_counts": {}, "entitlement_denied": [], "adoption_ids": [], "restatement_ids": []}
    return {**base, **overrides}


def test_summarize_review_list_rules():
    reports = {
        "MSFT": {"status": "complete", "success": True, "fundamentals": {"success": True, "committed": True, "quarantined": [
            {"id": i, "reason": "alias_superseded_by_primary"} for i in range(128)]}},
        "ZZZ": {"status": "complete", "success": True, "fundamentals": {"success": True, "committed": True, "quarantined": [
            {"id": i, "reason": "value_conflicts_with_primary"} for i in range(51)]}},
        "CUR": {"status": "complete", "success": True, "fundamentals": {"success": True, "committed": True, "quarantined": [
            {"id": 1, "reason": "currency_conflicts_with_primary"}]}},
        "BRK.B": {"status": "complete", "success": True, "fundamentals": {"success": True, "issues": [
            {"kind": "provider_entitlement_denied", "endpoint": "/income-statement", "cadence": "annual",
             "status": 402, "symbol": "BRK.B", "resolved": True, "resolved_by_symbol": "BRK-B"},
            {"kind": "provider_entitlement_denied", "endpoint": "/cash-flow-statement", "cadence": "quarterly",
             "status": 402, "symbol": "BRK-B"}]}},
        "DOWN": {"http_error": 502},
    }
    summary = fundamentals_repull.summarize(reports)
    review = {r["ticker"]: r for r in summary["review_list"]}
    # MSFT's 128 are a known case; an unknown ticker over 50, or any reason
    # outside the design's table, must be reviewed before execution.
    assert set(review) == {"ZZZ", "CUR"} and review["CUR"]["unexpected_reasons"] == ["currency_conflicts_with_primary"]
    assert summary["known_cases"]["MSFT"] == {"expected_quarantines": 128, "actual": 128}
    assert summary["rows_quarantined"] == 180
    assert summary["entitlement_denied_by_endpoint"] == {"/cash-flow-statement:quarterly:402": ["BRK.B"]}
    assert summary["per_ticker"]["BRK.B"]["entitlement_resolved_by_symbol"] == ["/income-statement:BRK.B->BRK-B"]
    assert [f["ticker"] for f in summary["failed"]] == ["DOWN"]
    assert "never evidenced" in summary["unexercised_endpoints_note"]


def test_summarize_is_pure_over_compacts():
    compacts = {"A": _compact(rows_quarantined=2, quarantine_counts={"value_conflicts_with_primary": 2}),
                "B": _compact(plan_mismatch={"kind": "repull_plan_mismatch"}, fundamentals_success=False)}
    summary = ledger.summarize(compacts)
    assert summary["plan_mismatch"] == ["B"] and summary["failed"] == [] and summary["review_list"] == []
    assert summary["quarantine_reasons"] == {"value_conflicts_with_primary": 2}


# ---------------------------------------------------------------------------
# The worker-run ledger
# ---------------------------------------------------------------------------

def _payload(value):
    rows = {s: [] for s in svc.LINES}
    ends = [(f"FY{y}", date(y, 12, 31)) for y in (2023, 2024, 2025)]
    ends += [(f"{y}Q{q}", date(y, q * 3, 30)) for y in (2024, 2025, 2026) for q in (1, 2, 3, 4)
             if date(2024, 6, 1) <= date(y, q * 3, 30) <= date(2026, 6, 30)]
    for period, end in ends:
        for statement, primary in svc.PRIMARY.items():
            rows[statement].append({"period": period, "period_end": end.isoformat(), "currency": "USD",
                                    "filing_date": (end + timedelta(days=30)).isoformat(), primary: value})
    return rows


@pytest.fixture
def world(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (ledger, svc, fq, backfill, provider_cache):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(svc, "_today", lambda: date(2026, 9, 13))
    # Pin the plan's clock too: its requested start must stay on or before
    # the pinned service "today" whatever the wall clock says.
    plan = backfill.backfill_plan
    monkeypatch.setattr(backfill, "backfill_plan", lambda today=None: plan(today=today or date(2026, 9, 13)))
    monkeypatch.setattr(ledger, "primary_history_capable", lambda: True)
    monkeypatch.setattr("app.cache.snapshots.invalidate", lambda *a, **k: 0)
    monkeypatch.delenv(ledger.ENABLE_ENV, raising=False)
    monkeypatch.delenv(ledger.EXECUTE_ENV, raising=False)
    monkeypatch.delenv(ledger.BACKUP_ENV, raising=False)
    calls = []
    values = {"ABC": 100.0, "DEF": 100.0}
    failures: dict[str, int] = {}

    def fetch(symbol, start):
        calls.append(symbol)
        if failures.get(symbol):
            failures[symbol] -= 1
            raise TimeoutError("FMP timed out")
        return _payload(values[symbol])

    fmp = SimpleNamespace(name="fmp", get_financial_history=fetch)
    monkeypatch.setattr(svc, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: [fmp]))
    with factory() as db:
        for ticker in ("ABC", "DEF"):
            db.add(Company(ticker=ticker, company_name=ticker, sector="Unknown", industry="Unknown"))
        # ABC holds drifted legacy values (90) FMP will contradict.
        for period, end in [(f"FY{y}", date(y, 12, 31)) for y in (2023, 2024, 2025)]:
            db.add(FinancialPeriod(ticker="ABC", period=period, period_end=end, fiscal_year=int(period[2:]),
                                   statement="income", line_item="revenue", value=90.0, currency="USD", source="live",
                                   available_at=end + timedelta(days=40), available_at_source="provider"))
        db.commit()
    return SimpleNamespace(factory=factory, calls=calls, values=values, failures=failures)


def _rows(factory):
    with factory() as db:
        return [{c.name: getattr(r, c.name) for c in FinancialPeriod.__table__.columns}
                for r in db.execute(select(FinancialPeriod).order_by(FinancialPeriod.id)).scalars()]


def _ledgers(factory):
    with factory() as db:
        return {r.id: r for r in db.execute(select(FinancialDataRepair)).scalars()}


def test_worker_dry_run_records_exact_plan_and_persists_no_financial_change(world, caplog):
    before = _rows(world.factory)
    legacy_ids = {r["id"] for r in before}
    with caplog.at_level("INFO", logger="app.services.fmp_repull_ledger"):
        first = ledger.run_pass(now=DAYTIME, max_tickers=1, sleep=lambda s: None)
        assert first["mode"] == "dry_run" and first["processed"] == ["ABC"] and first["remaining"] == 1
        second = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert second["processed"] == ["DEF"] and second["remaining"] == 0
    assert _rows(world.factory) == before
    rows = _ledgers(world.factory)
    assert set(rows) == {ledger.DRY_RUN_ID}  # the ledger only: no audit, no sync row
    with world.factory() as db:
        assert db.execute(select(MarketDataSync)).first() is None
    dry = rows[ledger.DRY_RUN_ID]
    assert dry.status == "complete" and dry.result["result_digest"]
    abc = dry.result["tickers"]["ABC"]
    assert {int(k) for k in abc["quarantine"]} == legacy_ids
    assert set(abc["quarantine"].values()) == {"value_conflicts_with_primary"}
    assert abc["quarantine_repair_id"] is None and abc["rows_inserted"] == 36
    assert dry.result["summary"]["rows_quarantined"] == 3
    assert "fundamentals repull dry_run ticker=ABC" in caplog.text and "complete:" in caplog.text
    status = ledger.repull_status(db=world.factory())
    assert status["dry_run"]["status"] == "complete" and status["execute"] is None
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == "awaiting_authorization"


def test_execution_requires_the_reviewed_digest_and_is_resumable(world):
    ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    digest = _ledgers(world.factory)[ledger.DRY_RUN_ID].result["result_digest"]
    before = {r["id"]: r for r in _rows(world.factory)}
    with pytest.raises(RuntimeError):
        ledger.authorize_execution("0" * 64, evidence_backup_sha256=BACKUP)
    # Owner decision 3: no execution without the evidence-archive backup.
    for missing in ("", "not-a-sha", "c" * 63):
        with pytest.raises(RuntimeError, match="evidence archive"):
            ledger.authorize_execution(digest, evidence_backup_sha256=missing)
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == "awaiting_authorization"
    assert _rows(world.factory) == list(before.values())
    authorized = ledger.authorize_execution(digest, evidence_backup_sha256=BACKUP.upper())
    assert authorized["mode"] == "execute" and authorized["dry_run_result_digest"] == digest
    assert authorized["evidence_backup_sha256"] == BACKUP
    with pytest.raises(RuntimeError):
        ledger.authorize_execution(digest, evidence_backup_sha256=BACKUP)  # one-shot
    fetched = len(world.calls)
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert done["remaining"] == 0 and done["result"]["summary"]["rows_quarantined"] == 3
    for row_id, row in before.items():
        moved = next(r for r in _rows(world.factory) if r["id"] == row_id)
        assert moved["ticker"].startswith("~Q") and {k: v for k, v in moved.items() if k != "ticker"} == {
            k: v for k, v in row.items() if k != "ticker"}
    execute = _ledgers(world.factory)[ledger.EXECUTE_ID]
    repair_id = execute.result["tickers"]["ABC"]["quarantine_repair_id"]
    assert repair_id == fq.repair_id_for(ledger.EXECUTE_ID, fq.QUARANTINE_KIND, "ABC")
    with world.factory() as db:
        assert db.get(MarketDataSync, "ABC").status == "complete"
    # Interrupted after ABC committed but before the ledger recorded it.
    with world.factory() as db:
        row = db.get(FinancialDataRepair, ledger.EXECUTE_ID)
        result = dict(row.result)
        result["tickers"] = {k: v for k, v in result["tickers"].items() if k != "ABC"}
        row.result, row.status = result, "running"
        db.commit()
    calls_before_resume = len(world.calls)
    resumed = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert resumed["processed"] == ["ABC"] and len(world.calls) == calls_before_resume
    assert _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["ABC"]["status"] == "already_applied"
    assert fetched < len(world.calls)
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == "complete"


def test_executed_ticker_that_drifted_since_review_is_aborted(world, monkeypatch):
    ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    digest = _ledgers(world.factory)[ledger.DRY_RUN_ID].result["result_digest"]
    with world.factory() as db:
        db.add(FinancialPeriod(ticker="ABC", period="2025Q1", period_end=date(2025, 3, 30), fiscal_year=2025,
                               fiscal_quarter=1, statement="income", line_item="revenue", value=1.0, currency="USD",
                               source="alpha_vantage"))
        db.commit()
    before = _rows(world.factory)
    monkeypatch.setenv(ledger.EXECUTE_ENV, digest[:16])  # the Render-dashboard path
    monkeypatch.setenv(ledger.BACKUP_ENV, BACKUP)
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert done["mode"] == "execute" and _ledgers(world.factory)[ledger.EXECUTE_ID].plan["authorized_by"] == "env"
    abc = _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["ABC"]
    assert abc["plan_mismatch"]["kind"] == "repull_plan_mismatch" and not abc["success"]
    assert [r for r in _rows(world.factory) if r["ticker"] == "ABC" or r["ticker"].startswith("~Q")] == \
        [r for r in before if r["ticker"] == "ABC"]
    assert done["result"]["summary"]["plan_mismatch"] == ["ABC"]


def test_pass_is_bounded_and_respects_the_nightly_window_and_switch(world, monkeypatch):
    assert ledger.run_pass(now=datetime(2026, 9, 24, 3, 10))["skipped_reason"] == "nightly loop window"
    clock = iter([0.0, 500.0, 1000.0])
    bounded = ledger.run_pass(now=DAYTIME, budget_seconds=100, sleep=lambda s: None, monotonic=lambda: next(clock))
    assert bounded["processed"] == ["ABC"] and bounded["remaining"] == 1
    monkeypatch.setenv(ledger.ENABLE_ENV, "off")
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == f"{ledger.ENABLE_ENV}=off"
    assert ledger.start_thread(SimpleNamespace()) is None


def test_demo_chain_never_starts_a_ledger(world, monkeypatch):
    monkeypatch.setattr(ledger, "primary_history_capable", lambda: False)
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"].startswith("no configured FMP")
    assert _ledgers(world.factory) == {}


def _dry_run_and_authorize(world):
    ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    digest = _ledgers(world.factory)[ledger.DRY_RUN_ID].result["result_digest"]
    ledger.authorize_execution(digest, evidence_backup_sha256=BACKUP)
    return digest


def test_unreviewed_restatement_aborts_the_executed_ticker(world):
    """The execute fence covers every change to a STORED row, not only the
    quarantine set: FMP restating DEF after review must not be applied."""
    svc.backfill_fundamentals("DEF", date(2024, 9, 13))  # DEF already holds FMP's 100s
    _dry_run_and_authorize(world)
    reviewed = _ledgers(world.factory)[ledger.DRY_RUN_ID].result["tickers"]["DEF"]
    assert reviewed["restatements"] == {} and reviewed["quarantine"] == {}
    before = [r for r in _rows(world.factory) if r["ticker"] == "DEF"]
    world.values["DEF"] = 105.0
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    record = _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["DEF"]
    assert record["plan_mismatch"]["differs"] == ["restatements"] and not record["success"]
    assert record["restatement_ids"] == [] and record["rows_quarantined"] == 0
    assert [r for r in _rows(world.factory) if r["ticker"] == "DEF"] == before
    assert done["result"]["summary"]["plan_mismatch"] == ["DEF"]


def test_failed_execution_records_no_rolled_back_changes(world, monkeypatch):
    """A fence error after the takeover rolls everything back; the ledger
    must not count the quarantines that never happened."""
    _dry_run_and_authorize(world)
    before = _rows(world.factory)

    def racing(db, row, **kwargs):
        raise fq.QuarantineFenceError("financial quarantine fence: row changed concurrently")

    monkeypatch.setattr(fq, "_move", racing)
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    abc = _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["ABC"]
    assert abc["rows_quarantined"] == 0 and abc["quarantine"] == {} and abc["adoption_ids"] == []
    assert done["result"]["summary"]["rows_quarantined"] == 0
    assert _rows(world.factory)[: len(before)] == before


def test_transient_fmp_failure_in_dry_run_is_retried_not_reviewed(world):
    world.failures["DEF"] = 2  # two timeouts, then FMP answers
    first = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert first["processed"] == ["ABC"] and first["remaining"] == 1
    assert ledger.run_pass(now=DAYTIME, sleep=lambda s: None)["remaining"] == 1
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    assert done["remaining"] == 0
    dry = _ledgers(world.factory)[ledger.DRY_RUN_ID]
    assert dry.result["tickers"]["DEF"]["primary_outcome"] == "received"
    assert dry.result["tickers"]["DEF"]["rows_inserted"] == 36
    assert dry.result["unreached_attempts"] == {"DEF": 2}


def test_ticker_the_dry_run_never_reached_is_skipped_at_execution(world):
    world.failures["DEF"] = ledger.MAX_DRY_RUN_ATTEMPTS
    for _ in range(ledger.MAX_DRY_RUN_ATTEMPTS):
        ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    dry = _ledgers(world.factory)[ledger.DRY_RUN_ID]
    assert dry.status == "complete"
    assert dry.result["tickers"]["DEF"]["primary_outcome"] == "error"
    assert dry.result["tickers"]["DEF"]["unreached_attempts"] == ledger.MAX_DRY_RUN_ATTEMPTS
    ledger.authorize_execution(dry.result["result_digest"], evidence_backup_sha256=BACKUP)
    calls = len(world.calls)
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    record = _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["DEF"]
    assert record["status"] == "skipped_unreviewed" and not record["success"]
    assert world.calls[calls:] == ["ABC"]  # DEF was never asked again
    assert not [r for r in _rows(world.factory) if r["ticker"] == "DEF"]
    assert done["result"]["summary"]["skipped_unreviewed"] == ["DEF"]


def test_execution_without_fmp_does_not_apply_a_plan_reviewed_with_fmp(world):
    """DEF's reviewed plan changes no stored row (inserts only), so only
    `primary_received` tells an FMP timeout at execution from the plan."""
    _dry_run_and_authorize(world)
    world.failures["DEF"] = 1
    done = ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    record = _ledgers(world.factory)[ledger.EXECUTE_ID].result["tickers"]["DEF"]
    assert record["plan_mismatch"]["differs"] == ["primary_received"] and not record["success"]
    assert done["result"]["summary"]["plan_mismatch"] == ["DEF"]


def test_authorization_refuses_an_incomplete_dry_run_and_a_short_env_prefix(world, monkeypatch):
    ledger.run_pass(now=DAYTIME, max_tickers=1, sleep=lambda s: None)
    running = _ledgers(world.factory)[ledger.DRY_RUN_ID]
    assert running.status == "running"
    with pytest.raises(RuntimeError, match="not completed"):
        ledger.authorize_execution("a" * 64, evidence_backup_sha256=BACKUP)
    assert ledger.EXECUTE_ID not in _ledgers(world.factory)
    ledger.run_pass(now=DAYTIME, sleep=lambda s: None)
    digest = _ledgers(world.factory)[ledger.DRY_RUN_ID].result["result_digest"]
    monkeypatch.setenv(ledger.BACKUP_ENV, BACKUP)
    monkeypatch.setenv(ledger.EXECUTE_ENV, digest[: ledger.MIN_DIGEST_PREFIX - 1])  # correct but too short
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == "awaiting_authorization"
    monkeypatch.setenv(ledger.EXECUTE_ENV, digest)
    monkeypatch.delenv(ledger.BACKUP_ENV)  # the env path needs the backup attestation too
    assert ledger.run_pass(now=DAYTIME)["skipped_reason"] == "awaiting_authorization"
    assert ledger.EXECUTE_ID not in _ledgers(world.factory)


def test_status_reports_the_workers_environment_not_the_web_process(world, monkeypatch):
    from app.providers import fmp_provider

    monkeypatch.setattr(fmp_provider, "_ENTITLEMENT_DENIALS", {})
    # The GET is served by web; its environment says nothing about the worker.
    monkeypatch.setenv(ledger.EXECUTE_ENV, "a" * 64)
    status = ledger.repull_status(db=world.factory())
    assert status["worker_view"] is None and "execute_env_set" not in status and "enabled" not in status
    assert "evidence archive" in status["authorize"]
    # The worker records its own view, including every FMP endpoint's refusals.
    monkeypatch.setenv(ledger.ENABLE_ENV, "off")
    monkeypatch.delenv(ledger.EXECUTE_ENV)
    fmp_provider._record_entitlement_denial("/key-metrics", 402, "ACME")
    assert ledger.start_thread(SimpleNamespace()) is None
    view = ledger.repull_status(db=world.factory())["worker_view"]
    assert view["enabled"] is False and view["execute_env_set"] is False
    assert [(d["endpoint"], d["status"], d["symbols"]) for d in view["fmp_entitlement_denials"]] == [
        ("/key-metrics", 402, ["ACME"])]
    monkeypatch.delenv(ledger.ENABLE_ENV)
    monkeypatch.setenv(ledger.EXECUTE_ENV, "b" * 64)
    ledger.observe_worker_env({"mode": "dry_run", "remaining": 3})
    view = ledger.repull_status(db=world.factory())["worker_view"]
    assert view["enabled"] is True and view["execute_env_set"] is True
    assert view["last_pass"] == {"mode": "dry_run", "remaining": 3}
    # A record row, not an applicable plan.
    assert fq.plan_kind(ledger.WORKER_VIEW_ID) == ledger.LEDGER_KIND


# ---------------------------------------------------------------------------
# The admin-path client script
# ---------------------------------------------------------------------------

def _script_world(monkeypatch, outcomes):
    calls = []
    plan = {"targets": [{"ticker": t, "fundamentals_required": True} for t in ("ABC", "DEF")]}

    def sync(ticker, dry_run):
        calls.append((ticker, dry_run))
        outcome = outcomes[ticker].pop(0) if outcomes.get(ticker) else "ok"
        if outcome == "timeout":
            raise TimeoutError("read timed out")
        if outcome == "running":
            return {"ticker": ticker, "status": "running", "success": False}
        return {"ticker": ticker, "status": "dry_run" if dry_run else "complete", "success": True, "dry_run": dry_run,
                "fundamentals": {"success": True, "committed": not dry_run, "quarantined": []}}

    monkeypatch.setattr(fundamentals_repull, "_in_process", lambda: (lambda: plan, sync))
    monkeypatch.setattr(fundamentals_repull.time, "sleep", lambda s: None)
    return calls


def test_script_retries_failed_tickers_on_resume(tmp_path, monkeypatch):
    calls = _script_world(monkeypatch, {"DEF": ["timeout"], "ABC": ["running"]})
    out = tmp_path / "dry"
    assert fundamentals_repull.main(["--in-process", "--out", str(out), "--dry-run"]) == 1
    assert not (out / "DEF.json").exists() and (out / "failures" / "DEF.json").exists()
    assert not (out / "ABC.json").exists()  # a held claim is not an answer either
    calls.clear()
    assert fundamentals_repull.main(["--in-process", "--out", str(out), "--dry-run"]) == 0
    assert calls == [("ABC", True), ("DEF", True)]
    assert (out / "DEF.json").exists() and not (out / "failures" / "DEF.json").exists()


def test_script_refuses_to_resume_across_modes(tmp_path, monkeypatch):
    calls = _script_world(monkeypatch, {})
    out = tmp_path / "shared"
    assert fundamentals_repull.main(["--in-process", "--out", str(out), "--dry-run"]) == 0
    calls.clear()
    # The real run pointed at the dry run's directory must not "resume" from
    # dry-run files, execute nothing and exit 0.
    assert fundamentals_repull.main(["--in-process", "--out", str(out)]) == 2
    assert calls == []
    assert fundamentals_repull.main(["--in-process", "--out", str(tmp_path / "real")]) == 0
    assert calls == [("ABC", False), ("DEF", False)]


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from app.config import settings
    from app.main import app
    monkeypatch.setattr(settings, "admin_api_token", "test-repull-token")
    return TestClient(app), {"Authorization": "Bearer test-repull-token"}


def test_repull_routes_require_admin_and_bind_the_digest(client, monkeypatch):
    http, headers = client
    calls = []
    monkeypatch.setattr(ledger, "repull_status", lambda: {"dry_run": {"status": "complete"}})

    def authorize(digest, evidence_backup_sha256, authorized_by):
        calls.append((digest, evidence_backup_sha256, authorized_by))
        if digest.startswith("b"):
            raise RuntimeError("Digest does not match the reviewed dry run")
        return {"id": ledger.EXECUTE_ID}

    monkeypatch.setattr(ledger, "authorize_execution", authorize)
    base = "/api/admin/market-data/fmp-repull"
    assert http.get(base).status_code == 401
    body = {"result_digest": "a" * 64, "evidence_backup_sha256": BACKUP}
    assert http.post(base + "/authorize", json=body).status_code == 401
    assert calls == []
    assert http.get(base, headers=headers).json() == {"dry_run": {"status": "complete"}}
    assert http.post(base + "/authorize", headers=headers, json={**body, "result_digest": "short"}).status_code == 422
    # The backup attestation is required, not optional.
    assert http.post(base + "/authorize", headers=headers, json={"result_digest": "a" * 64}).status_code == 422
    assert calls == []
    assert http.post(base + "/authorize", headers=headers, json={**body, "result_digest": "b" * 64}).status_code == 409
    ok = http.post(base + "/authorize", headers=headers, json=body)
    assert ok.status_code == 200 and calls[-1] == ("a" * 64, BACKUP, "admin")


def test_backfill_route_accepts_scope_and_rejects_a_dry_run_with_prices(client, monkeypatch):
    http, headers = client
    seen = []

    def sync(ticker, **kwargs):
        seen.append(kwargs)
        if kwargs["dry_run"] and kwargs["scope"] != "fundamentals":
            raise ValueError("dry_run requires scope=fundamentals")
        return {"ticker": ticker, "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(backfill, "sync_ticker", sync)
    url = "/api/admin/market-data/backfill"
    ok = http.post(url, headers=headers, params={"ticker": "ABC", "scope": "fundamentals", "dry_run": "true"})
    assert ok.status_code == 200 and seen[-1] == {"force_refresh": False, "scope": "fundamentals", "dry_run": True}
    assert http.post(url, headers=headers, params={"ticker": "ABC", "dry_run": "true"}).status_code == 400
    assert http.post(url, headers=headers, params={"ticker": "ABC", "scope": "prices"}).status_code == 422
