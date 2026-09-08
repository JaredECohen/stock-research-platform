"""Cron-health notes must count gate/assessment errors, not hide them.

`update_orchestrator` now distinguishes a gate that *crashed* (``kind ==
"gate_error"``, e.g. a DB error inside ``should_auto_regen``) or a news
assessment whose LLM failed (``reason == "assessment_error"``) from a
deliberate skip. Before this, each loop recorded "N new filings" /
"N material events" with ``success=True`` either way, so a dead gate read
as "nothing to do" — the exact silent-failure shape that hid the
memo_outcomes outage for weeks. These tests pin that the notes count the
errors and flip the loop to unhealthy.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.monitoring import edgar_poller, macro_loop, news_loop, transcripts_poller


def _capture(monkeypatch, module) -> list[tuple[tuple, dict[str, Any]]]:
    calls: list[tuple[tuple, dict[str, Any]]] = []
    monkeypatch.setattr(module, "record_run", lambda *a, **k: calls.append((a, k)))
    return calls


def test_news_loop_counts_failed_assessments(monkeypatch):
    calls = _capture(monkeypatch, news_loop)
    monkeypatch.setattr(news_loop, "_last_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "invalidate", lambda *a, **k: None)
    monkeypatch.setattr(
        news_loop.news_agent, "run",
        lambda t, **k: [SimpleNamespace(severity="material")],
    )
    import app.services.update_orchestrator as uo
    monkeypatch.setattr(
        uo, "on_news_alert",
        lambda t, alert: {"patched": False, "ticker": t, "reason": "assessment_error"},
    )

    news_loop.run_once(["NVDA"])

    (args, kwargs), = calls
    assert args == ("news_loop",)
    assert kwargs["success"] is False
    assert "1 material events; 1 assessments failed" == kwargs["note"]


def test_news_loop_is_healthy_when_assessments_succeed(monkeypatch):
    calls = _capture(monkeypatch, news_loop)
    monkeypatch.setattr(news_loop, "_last_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "invalidate", lambda *a, **k: None)
    monkeypatch.setattr(
        news_loop.news_agent, "run",
        lambda t, **k: [SimpleNamespace(severity="material")],
    )
    import app.services.update_orchestrator as uo
    monkeypatch.setattr(uo, "on_news_alert", lambda t, alert: {"patched": False, "reason": "not_material"})

    news_loop.run_once(["NVDA"])

    (_, kwargs), = calls
    assert kwargs["success"] is True
    assert kwargs["note"] == "1 material events"


def test_edgar_poller_counts_gate_errors(monkeypatch):
    calls = _capture(monkeypatch, edgar_poller)
    monkeypatch.setattr(
        edgar_poller, "get_filings",
        lambda t: [
            {"type": "10-K", "accession_number": "0001"},
            {"type": "10-Q", "accession_number": "0002"},
        ],
    )
    monkeypatch.setattr(edgar_poller, "_seen_accessions", lambda t: {"0001"})
    monkeypatch.setattr(edgar_poller, "_save_seen_accessions", lambda t, acc: None)
    monkeypatch.setattr(edgar_poller, "invalidate", lambda *a, **k: None)
    import app.services.update_orchestrator as uo
    monkeypatch.setattr(uo, "on_filing_event", lambda t, **k: {"ticker": t, "kind": "gate_error"})

    events = edgar_poller.run_once(["NVDA"])

    assert events and events[0]["new_accessions"] == ["0002"]
    (_, kwargs), = calls
    assert kwargs["success"] is False
    assert kwargs["note"] == "1 new filings; gate errors on 1: NVDA"


def test_transcripts_poller_counts_gate_errors(monkeypatch):
    calls = _capture(monkeypatch, transcripts_poller)
    monkeypatch.setattr(
        transcripts_poller, "get_transcripts",
        lambda t: [{"period": "2025Q1"}, {"period": "2025Q2"}],
    )
    monkeypatch.setattr(transcripts_poller, "_seen_periods", lambda t: {"2025Q1"})
    monkeypatch.setattr(transcripts_poller, "_save_seen_periods", lambda t, p: None)
    import app.services.update_orchestrator as uo
    monkeypatch.setattr(
        uo, "on_transcript_event",
        lambda t, period="": {"ticker": t, "period": period, "kind": "gate_error"},
    )

    events = transcripts_poller.run_once(["NVDA"])

    assert [e["period"] for e in events] == ["2025Q2"]
    (_, kwargs), = calls
    assert kwargs["success"] is False
    assert kwargs["note"] == "1 new transcripts; gate errors on 1: NVDA"


def test_macro_loop_counts_gate_errors_on_regime_shift(monkeypatch):
    calls = _capture(monkeypatch, macro_loop)
    monkeypatch.setattr(macro_loop.macro_service, "macro_snapshot", lambda: {})
    monkeypatch.setattr(macro_loop, "_detect_regime", lambda snap: "soft_landing")
    monkeypatch.setattr(
        macro_loop, "cache_get",
        lambda *a, **k: SimpleNamespace(payload={"regime": "credit_stress"}),
    )
    monkeypatch.setattr(macro_loop, "cache_put", lambda *a, **k: None)
    import app.services.update_orchestrator as uo
    monkeypatch.setattr(
        uo, "on_regime_shift",
        lambda prior, new: {"prior": prior, "new": new, "refreshed": [], "gate_errors": ["NVDA"]},
    )

    macro_loop.run_once()

    (_, kwargs), = calls
    assert kwargs["success"] is False
    assert kwargs["note"].startswith("regime=soft_landing")
    assert "gate errors on 1: NVDA" in kwargs["note"]
