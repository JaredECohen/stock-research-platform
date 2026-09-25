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


def test_memory_trail_suppresses_template_reflection(tmp_path, monkeypatch):
    """W2a (critique delta 3): the memory trail is a memo exit too. An entry
    the no-LLM reflection branch wrote, or one quoting a section the latest
    memo hides, is left out of the response; the file is untouched."""
    import json
    from pathlib import Path

    from app.agents import reflection_agent
    from app.config import settings
    from app.schemas import StockMemoOut
    from app.services import memo_store
    from app.tests.gating_helpers import purge_memos

    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    raw = json.loads((Path(__file__).parent / "fixtures" / "memo_sections"
                      / "googl_live_prepflag.json").read_text())
    memo = StockMemoOut.model_validate(raw).model_copy(update={"ticker": "ZZMEMR"})
    purge_memos("ZZMEMR")
    memo_store.save_memo(memo)
    try:
        template = reflection_agent._compose_company_entry(memo, [{"label": "new filing"}])
        quoting = ("**Trigger:** new filing\n\n**Observation:** The call was "
                   + memo.one_sentence_thesis + "\n\n**Watch next:** margins.")
        real = "**Trigger:** earnings\n\n**Observation:** Cloud grew faster than the model assumed."
        cm = CompanyMemory.for_ticker("ZZMEMR")
        for i, body in enumerate((template, quoting, real)):
            cm.append_entry(MemoryEntry(date=f"2026-09-0{i + 1}", trigger="t", body=body))
        cm.save()
        before = cm.path.read_text()

        body = TestClient(app).get("/api/stocks/ZZMEMR/memory").json()
        assert body["entry_count"] == 3
        assert body["suppressed_count"] == 2
        assert [e["body"] for e in body["entries"]] == [real]
        assert cm.path.read_text() == before
    finally:
        purge_memos("ZZMEMR")


def test_memory_trail_template_rule_stands_alone(tmp_path, monkeypatch):
    """The reflection-template rule on its own: the latest memo hides no
    prose (META v1), and the template entry was composed from an OLDER,
    different memo, so no hidden-field probe can match it. It is still left
    out, and so is its line in the condensed history block."""
    import json
    from pathlib import Path

    from app.agents import reflection_agent
    from app.config import settings
    from app.schemas import StockMemoOut
    from app.services import memo_sections, memo_store
    from app.tests.gating_helpers import purge_memos

    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    fixtures = Path(__file__).parent / "fixtures" / "memo_sections"
    latest = StockMemoOut.model_validate(json.loads((fixtures / "meta_v1.json").read_text()))
    latest = latest.model_copy(update={"ticker": "ZZMEMS"})
    older = StockMemoOut.model_validate(json.loads((fixtures / "googl_live_prepflag.json").read_text()))
    older = older.model_copy(update={"ticker": "ZZMEMS"})
    purge_memos("ZZMEMS")
    memo_store.save_memo(latest)
    try:
        shown = memo_store.present_snapshot(memo_store.latest_memo("ZZMEMS"))
        assert not {k for k, v in shown.section_availability.items()
                    if v.status == "unavailable" and v.reason != "not_produced"
                    and k in ("final_pm_view", "one_sentence_thesis", "final_verdict")}
        template = reflection_agent._compose_company_entry(older, [{"label": "memo run"}])
        assert memo_sections.is_reflection_template(template)
        cm = CompanyMemory.for_ticker("ZZMEMS")
        cm.append_entry(MemoryEntry(date="2026-09-01", trigger="memo_run", body=template))
        real_line = "- 2026-08-01 (earnings): Ads grew faster than the model assumed."
        folded = "- 2026-08-02 (memo_run): " + template[:160].strip().replace("\n", " ")
        cm.historical_context = "\n".join(
            ["**Condensed 2026-08-01 → 2026-08-02** (2 entries)", real_line, folded])
        cm.save()

        body = TestClient(app).get("/api/stocks/ZZMEMS/memory").json()
        assert body["suppressed_count"] == 1 and body["entries"] == []
        assert real_line in body["historical_context"]
        assert folded not in body["historical_context"]
        assert "regime read" not in body["historical_context"]
        assert body["historical_context_suppressed"] == 1
    finally:
        purge_memos("ZZMEMS")


def test_stock_memory_endpoint_empty_for_unknown_ticker(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    c = TestClient(app)
    r = c.get("/api/stocks/NEVER_HEARD_OF_THIS/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert body["entry_count"] == 0


# ---------------------------------------------------------------------------
# W7 (S19): the ledger branch is public only once priors are injected
# ---------------------------------------------------------------------------

def test_public_memory_in_shadow_returns_legacy_shape_and_never_detail(tmp_path, monkeypatch):
    """Until the owner promotes learning to inject, the page shows the legacy
    trail exactly as before, whatever the ledger holds. In inject it shows
    company-scope observations and non-suppressed lessons, every lesson
    labelled a provisional hypothesis — never the audit-only `detail`, never
    a peer scope, never a suppressed item — in the same response shape."""
    from datetime import date, datetime

    from app.config import settings
    from app.learning import context, control, ledger
    from app.models import LearningControlEvent, LearningEvidence
    from app.tests.learning_helpers import learning_db

    sessions, engine = learning_db(tmp_path, monkeypatch, context)
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    monkeypatch.setattr(settings, "learning_mode_max", "inject")
    now = datetime.utcnow()
    cm = CompanyMemory.for_ticker("ZZLRN")
    cm.append_entry(MemoryEntry(date="2026-05-01", trigger="earnings", body="Legacy file entry."))
    cm.save()
    with sessions() as s:
        def item(kind: str, text: str, ref: str, *, scope_type: str = "company", scope_key: str = "ZZLRN",
                 status: str = "active", filed: date = date(2026, 8, 1)) -> int:
            row = ledger._new_item(
                s, kind=kind, scope_type=scope_type, scope_key=scope_key, text=text,
                detail="SECRET-DETAIL full postmortem narrative",
                origin_kind="postmortem" if kind == "lesson" else "filing_delta",
                origin_ref=ref, origin_ticker="ZZLRN", source_date=filed, now=now,
            )
            row.status = status
            s.flush()
            return int(row.id)

        tested = item("lesson", "When bookings accelerate, expect the stock to outperform the benchmark over 90 days.",
                      "pm-1", filed=date(2026, 6, 1))
        for n, verdict in enumerate(("held", "held", "held", "held")):
            s.add(LearningEvidence(item_id=tested, verdict=verdict, ticker="ZZLRN", horizon_days=90,
                                   independence_key=f"ZZLRN:90:{n}", observed_at=now))
        item("lesson", "SUPPRESSED-TEXT should never be public.", "pm-2", status="suppressed")
        item("lesson", "When churn rises, expect the stock to underperform the benchmark over 90 days.", "pm-3",
             status="retired", filed=date(2026, 5, 1))
        item("observation", "What's new in 10-Q filed 2026-08-01: bookings up.", "acc-1")
        item("lesson", "PEER-SCOPE lesson stays off the company trail.", "pm-4",
             scope_type="industry_group", scope_key="4510")
        s.commit()

    try:
        client = TestClient(app)
        # Shadow (the DB default after deploy): the legacy trail, unchanged.
        control._cache_clear()
        assert control.effective_mode() == "shadow"
        legacy = client.get("/api/stocks/ZZLRN/memory").json()
        assert legacy["path"].endswith("ZZLRN.md")
        assert [e["body"] for e in legacy["entries"]] == ["Legacy file entry."]
        assert "SECRET-DETAIL" not in str(legacy)

        # Inject: the ledger, in the same shape.
        with sessions() as s:
            s.add(LearningControlEvent(mode="inject", actor="admin", reason="test", gates={}, created_at=now))
            s.commit()
        control._cache_clear()
        body = client.get("/api/stocks/ZZLRN/memory").json()
        assert set(body) == set(legacy)
        assert body["path"] == "database: learning ledger"
        text = str(body)
        assert "SECRET-DETAIL" not in text and "SUPPRESSED-TEXT" not in text and "PEER-SCOPE" not in text
        assert "4510" not in text
        assert body["entry_count"] == 3 and body["suppressed_count"] == 1
        triggers = [e["trigger"] for e in body["entries"]]
        assert triggers == ["filing observation", "provisional hypothesis · supported 4 of 4 later outcomes",
                            "provisional hypothesis · untested hypothesis · retired"]
        lessons = [e for e in body["entries"] if e["trigger"].startswith("provisional hypothesis")]
        assert all(e["body"].endswith("Provisional hypothesis — not investment advice.") for e in lessons)
        assert all(e["structured_facts"] is None for e in body["entries"])
        assert "not investment advice" in body["historical_context"]
        assert body["entries"][0]["date"] == "2026-08-01"

        # The env ceiling is the emergency stop: off serves the legacy trail.
        monkeypatch.setattr(settings, "learning_mode_max", "off")
        assert client.get("/api/stocks/ZZLRN/memory").json()["path"].endswith("ZZLRN.md")
    finally:
        control._cache_clear()
        engine.dispose()
