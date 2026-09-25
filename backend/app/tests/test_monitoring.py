"""Phase 5 — monitoring loops.

Each loop is unit-tested in isolation (no scheduler needed). Verifies:
- `news_loop.run_once` writes a NewsAlert to the hot cache.
- `social_loop.run_once` writes a sentiment scalar.
- `macro_loop.run_once` produces a regime label and broadcasts it.
- `edgar_poller.run_once` is a no-op on first run (initialization) but
  invalidates company_cold once a new accession appears.
"""
from __future__ import annotations

from app.cache import cache_get, cache_put
from app.monitoring import edgar_poller, macro_loop, news_loop, social_loop


def test_news_loop_writes_news_hot_to_cache():
    news_loop.run_once(["NVDA"])
    snap = cache_get("news_hot:NVDA", "news_hot")
    assert snap is not None
    assert "alerts" in snap.payload


def test_social_loop_writes_sentiment_scalar():
    social_loop.run_once(["MSFT"])
    # Today's bucket
    from datetime import date
    snap = cache_get(f"social_hot:MSFT:{date.today().isoformat()}", "social_hot")
    assert snap is not None
    assert "sentiment_extremity" in snap.payload


def test_macro_loop_produces_broadcast_with_regime():
    out = macro_loop.run_once()
    assert "regime" in out
    snap = cache_get("macro:global", "macro_broadcast")
    assert snap is not None
    assert snap.payload.get("regime") == out["regime"]


def test_edgar_poller_invalidates_company_cold_on_new_accession():
    # Wave 5B: the poller now also enqueues a `full_reanalysis` via the
    # update orchestrator on a new filing. Patch that side-effect out so
    # this test stays focused on the poller's own invalidate call.
    from unittest.mock import patch
    with patch("app.services.update_orchestrator.on_filing_event") as fe:
        fe.return_value = {"kind": "full_reanalysis", "ticker": "NVDA"}
        # First run primes the bookkeeping store; should produce no events.
        events = edgar_poller.run_once(["NVDA"])
        assert events == []

        # Simulate a new accession by clobbering the bookkeeping snapshot to a
        # subset of what we'll see. The next call should detect the diff and
        # invalidate the company_cold snapshot.
        seen = cache_get("NVDA", "edgar_seen_accessions")
        if not seen:
            # Demo provider may have no filings — exit early in that case.
            return
        accessions = list(seen.payload.get("accessions") or [])
        if len(accessions) < 1:
            return
        # Drop one accession from the bookkeeping so the next poll sees it as new.
        cache_put(
            "NVDA", "edgar_seen_accessions",
            payload={"accessions": accessions[:-1]},
            sources_used=["edgar:NVDA:bookkeeping"],
            generated_by="test", cost_tokens=0,
        )
        # Pre-populate company_cold so we can prove invalidation marks it stale.
        cache_put(
            "NVDA", "company_cold",
            payload={"profile": {"ticker": "NVDA"}, "income": [{"period": "2023-12-31", "revenue": 1}]},
            sources_used=["filing:NVDA:000001"],
            generated_by="test", cost_tokens=0,
        )
        events = edgar_poller.run_once(["NVDA"])
        assert events  # at least one new accession event
        assert cache_get("NVDA", "company_cold") is None  # invalidated
        # Wave 5B: orchestrator's filing handler was called for the new accession.
        fe.assert_called_once_with("NVDA")


# ---------------------------------------------------------------------------
# N34 — per-alert isolation and a note that says where news came from
# ---------------------------------------------------------------------------

def _quiet_loop(monkeypatch, *, throttled=()):
    from datetime import datetime
    notes: list[dict] = []
    monkeypatch.setattr(news_loop, "record_run", lambda *a, **k: notes.append(k))
    monkeypatch.setattr(
        news_loop, "_last_run_for", lambda t: datetime.utcnow() if t in throttled else None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "invalidate", lambda *a, **k: None)
    return notes


def test_first_alert_raises_second_still_assessed(monkeypatch):
    from types import SimpleNamespace

    import app.services.update_orchestrator as uo
    notes = _quiet_loop(monkeypatch)
    alerts = [SimpleNamespace(severity="material", title="first", source="news_service"),
              SimpleNamespace(severity="breaking", title="second", source="news_service")]
    monkeypatch.setattr(news_loop.news_agent, "run", lambda t, **k: alerts)
    seen: list[str] = []

    def on_news_alert(ticker, alert):
        seen.append(alert.title)
        if alert.title == "first":
            raise ValueError("unreadable prior memo")
        return {"patched": False, "reason": "not_material"}
    monkeypatch.setattr(uo, "on_news_alert", on_news_alert)

    news_loop.run_once(["NVDA"])

    assert seen == ["first", "second"]
    (rec,) = notes
    assert rec["success"] is False
    assert "1 updates failed: NVDA" in rec["note"]


def test_news_loop_note_counts_sources(monkeypatch):
    from types import SimpleNamespace
    notes = _quiet_loop(monkeypatch, throttled=("THR",))

    def run(ticker, *, force_refresh=False, report=None):
        report = report if report is not None else {}
        if ticker == "GEM":
            report["origin"] = "gemini"
            return [SimpleNamespace(severity="advisory", source="gemini")]
        if ticker == "PROV":
            report["origin"] = "provider"
            return [SimpleNamespace(severity="advisory", source="news_service")]
        if ticker == "BRK":  # breaker open: Gemini never asked, feed empty too
            report.update(origin="empty", gemini_skipped="breaker_open")
            return []
        report.update(origin="provider", gemini_skipped="grounding_cap")
        return [SimpleNamespace(severity="advisory", source="news_service")]
    monkeypatch.setattr(news_loop.news_agent, "run", run)

    news_loop.run_once(["GEM", "PROV", "THR", "BRK", "CAP"])

    (rec,) = notes
    assert rec["note"].endswith(
        "sources gemini=1 provider=2 empty=1 throttled=1 gemini_breaker=1 grounding_cap=1")


def test_news_loop_note_counts_sources_without_a_report(monkeypatch):
    # A news_agent.run that fills no report (a stand-in) is classified from
    # the alerts it returned.
    from types import SimpleNamespace
    notes = _quiet_loop(monkeypatch)
    by_ticker = {"A": [SimpleNamespace(severity="advisory", source="gemini")], "B": []}
    monkeypatch.setattr(news_loop.news_agent, "run", lambda t, **k: by_ticker[t])
    news_loop.run_once(["A", "B"])
    assert "sources gemini=1 provider=0 empty=1 throttled=0" in notes[0]["note"]
