"""FEAT-001 S3 — `POST /api/fundamentals/commentary` and `services/chart_commentary`.

Everything here is deterministic and network-free: series rows are
seeded for throwaway tickers, memos are factory-built, the clock is the
two modules' `_utcnow()` seams, and `llm.chat_json` is either patched
with canned JSON or proven never to have been called. When a test needs
the LLM to look *available* it sets a fake OpenAI key — the patched
`chat_json` is the only thing that would ever read it.

Covered, per the slice brief: no keys → degraded observed-only body
with no charge; canned output → out-of-selection items dropped and
counted, excerpts truncated, `memos_used` None without a snapshot,
`memo_stale` from a newer filing and from "predates the last displayed
period"; cache hits without an LLM call; 409 on fingerprint mismatch;
charge before the call and release on None; the prompt budget for the
maximal 5 × 4 × 30 selection; anonymous with
`FUNDAMENTALS_ANON_COMMENTARY=false` → the account reason and no call.
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from app.agents import llm
from app.auth import features, ratelimit, usage
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import ChartCommentary, FilingDoc, FinancialPeriod, MemoSnapshot, UsageEvent
from app.prompts import chart_commentary as P
from app.rate_limit import limiter
from app.schemas import CommentaryOut
from app.services import chart_commentary as cc
from app.services import fundamentals_series_service as fss
from app.services import history_service
from app.services.fundamentals_series_service import build_series
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.factories import make_memo, seed_annual_periods
from app.tests.gating_helpers import (
    assert_structured,
    free_user,
    purge_memos,
    purge_rate_windows,
    store_memo,
    usage_events,
    user_id_for,
)

NOW = datetime(2025, 6, 1, 12, 0, 0)
FRESH = datetime(2025, 5, 30)
TICKERS = ("FCMA", "FCMB", "FCMC", "FCMD", "FCME")
WITH_MEMO = ("FCMA", "FCMB")           # FCMC–FCME never get a memo
FULL = {
    "revenue": 1000.0, "gross_profit": 600.0, "operating_income": 250.0, "net_income": 180.0,
    "cash_from_operations": 320.0, "capex": -70.0, "free_cash_flow": 250.0,
    "stock_based_compensation": 40.0, "weighted_avg_shares_diluted": 100.0,
}
FAKE_OPENAI = "sk-test-chart-commentary-not-a-real-key-0000"

CANNED = {
    "memo_view": [
        {"ticker": "FCMA", "text": "The memo's thesis leans on margin expansion, which the displayed data shows."},
        {"ticker": "fcma", "text": "A second sentence for the same memo."},
        {"ticker": "FCMC", "text": "FCMC has no stored memo, so this must be dropped."},
        {"ticker": "MSFT", "text": "MSFT is outside the selection, so this must be dropped."},
    ],
    "caveats": ["The comparison spans companies with different fiscal year ends."],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    purge_rate_windows()
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fss, "_utcnow", lambda: NOW)
    monkeypatch.setattr(cc, "_utcnow", lambda: NOW)


@pytest.fixture(autouse=True)
def _clean_rows():
    """The commentary cache is keyed by fingerprint, so a row left by an
    earlier test would turn a "no LLM call" assertion into a cache hit.
    The database is per-run, so wiping the table is safe."""
    def wipe():
        with SessionLocal() as db:
            db.query(ChartCommentary).delete(synchronize_session=False)
            db.query(FilingDoc).filter(FilingDoc.ticker.in_(TICKERS)).delete(synchronize_session=False)
            # The two-in-flight test seeds leases it never releases; they
            # would otherwise sit in `active_actions` for their TTL and
            # leak into any later test that counts leases or runs the GC.
            db.query(ratelimit.ActiveAction).filter(ratelimit.ActiveAction.feature == "chart_commentary").delete(
                synchronize_session=False,
            )
            db.commit()
        purge_memos(*TICKERS)
    wipe()
    yield
    wipe()


@pytest.fixture(autouse=True)
def _breakers():
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


def _wipe_periods(db) -> None:
    db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_(TICKERS)).delete(synchronize_session=False)
    db.commit()


@pytest.fixture()
def seeded():
    """FY2017–FY2024 for every throwaway ticker; values scale per ticker
    so rankings are deterministic."""
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        _wipe_periods(db)
        for i, t in enumerate(TICKERS):
            seed_annual_periods(
                db, t, {fy: {k: v * (1 + 0.1 * i) * (1 + 0.05 * (fy - 2017)) for k, v in FULL.items()}
                        for fy in range(2017, 2025)},
                fetched_at=FRESH,
            )
        db.commit()
    yield
    with SessionLocal() as db:
        _wipe_periods(db)


def _set_generated_at(snap: MemoSnapshot, when: datetime) -> None:
    with SessionLocal() as db:
        db.execute(update(MemoSnapshot).where(MemoSnapshot.id == snap.id).values(generated_at=when))
        db.commit()


@pytest.fixture()
def memos():
    """Memos for FCMA and FCMB generated *after* FY2024 closed, so
    neither is stale by the predates rule unless a test moves it."""
    snaps = {}
    for t in WITH_MEMO:
        snap = store_memo(t)
        _set_generated_at(snap, datetime(2025, 3, 1))
        snaps[t] = snap
    return snaps


@pytest.fixture()
def anon_ok(monkeypatch):
    monkeypatch.setattr(settings, "fundamentals_anon_commentary", True)


@pytest.fixture()
def llm_available(monkeypatch):
    """An LLM that *looks* configured. Every test using this also patches
    `llm.chat_json`, so the key is never read by a client."""
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "llm_provider", "auto")
    assert settings.active_llm_provider == "openai"


def _fp(tickers, metrics, *, years=None, normalize="none") -> str:
    with SessionLocal() as db:
        return build_series(list(tickers), list(metrics), years=years, normalize=normalize, db=db).fingerprint


def _post(client, token=None, **body):
    return client.post("/api/fundamentals/commentary", json=body, headers=bearer(token) if token else {})


def _body(tickers=("FCMA", "FCMB"), metrics=("revenue", "gross_margin"), years=None, fingerprint=None):
    return {
        "tickers": list(tickers), "metrics": list(metrics), "years": years,
        "fingerprint": fingerprint or _fp(tickers, metrics, years=years),
    }


def _rows() -> list[ChartCommentary]:
    with SessionLocal() as db:
        rows = db.query(ChartCommentary).order_by(ChartCommentary.id).all()
        db.expunge_all()
        return rows


def _event_count() -> int:
    """`chart_commentary` usage events on file. Tests compare before/after
    rather than asserting zero: the sqlite file is reused across runs and
    keeps the committed events of earlier auth-on tests."""
    with SessionLocal() as db:
        return db.query(UsageEvent).filter(UsageEvent.feature == "chart_commentary").count()


# ---------------------------------------------------------------------------
# Wall off — the anonymous default
# ---------------------------------------------------------------------------

def test_anonymous_default_is_degraded_with_the_account_reason_and_no_llm_call(client, seeded, memos):
    before = _event_count()
    assert settings.fundamentals_anon_commentary is False
    with patch.object(llm, "chat_json") as call:
        resp = _post(client, **_body())
    call.assert_not_called()
    assert resp.status_code == 200, resp.text
    out = CommentaryOut.model_validate(resp.json())
    assert out.degraded is True and out.degraded_reason == cc.REASON_NO_ACCOUNT
    assert out.observed and out.memo_view == [] and out.cache_hit is False
    assert out.commentary_id is None and out.model is None
    assert out.fingerprint == _fp(("FCMA", "FCMB"), ("revenue", "gross_margin"))
    assert _rows() == []      # nothing stored, nothing to charge
    assert _event_count() == before       # nothing charged


def test_no_keys_is_degraded_observed_only_with_no_charge(client, seeded, memos, anon_ok):
    before = _event_count()
    assert settings.active_llm_provider == "none"
    with patch.object(llm, "chat_json") as call:
        resp = _post(client, **_body())
    call.assert_not_called()
    out = CommentaryOut.model_validate(resp.json())
    assert out.degraded is True and out.degraded_reason == cc.REASON_NO_LLM
    assert out.observed and out.memo_view == []
    assert any("FCMA" in o.text and "Revenue" in o.text for o in out.observed)
    # Observations cite the points they are about, inside the selection.
    for o in out.observed:
        for r in o.refs:
            assert r.ticker in ("FCMA", "FCMB") and r.metric in ("revenue", "gross_margin")
            assert r.period.startswith("FY")
    assert _rows() == []
    assert _event_count() == before       # nothing charged


def test_fingerprint_mismatch_is_409_with_the_current_fingerprint(client, seeded, memos, anon_ok):
    with patch.object(llm, "chat_json") as call:
        resp = _post(client, **_body(fingerprint="0" * 64))
    call.assert_not_called()
    detail = assert_structured(resp, code="series_changed", status=409)
    assert detail["feature"] == "chart_commentary"
    assert detail["extra"]["fingerprint"] == _fp(("FCMA", "FCMB"), ("revenue", "gross_margin"))
    assert detail["extra"]["requested"] == "0" * 64


def test_indexed_view_fingerprint_is_accepted(client, seeded, memos):
    """`normalize` is not in the request; the indexed chart's fingerprint
    must still be recognised as the displayed data."""
    fp = _fp(("FCMA", "FCMB"), ("revenue", "gross_margin"), normalize="indexed")
    resp = _post(client, **_body(fingerprint=fp))
    assert resp.status_code == 200, resp.text
    out = CommentaryOut.model_validate(resp.json())
    assert out.fingerprint == fp
    assert any("indexed" in o.text for o in out.observed)


def test_no_stored_memos_is_degraded_and_free(client, seeded, anon_ok, llm_available):
    with patch.object(llm, "chat_json") as call:
        resp = _post(client, **_body(tickers=("FCMC", "FCMD")))
    call.assert_not_called()
    out = CommentaryOut.model_validate(resp.json())
    assert out.degraded is True and out.degraded_reason == cc.REASON_NO_MEMOS
    assert out.observed and any("No stored memo for FCMC" in c for c in out.caveats)
    assert _rows() == []


# ---------------------------------------------------------------------------
# Canned model output
# ---------------------------------------------------------------------------

def test_canned_output_is_validated_against_the_selection(client, seeded, memos, anon_ok, llm_available):
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        resp = _post(client, **_body(tickers=("FCMA", "FCMB", "FCMC")))
    assert resp.status_code == 200, resp.text
    out = CommentaryOut.model_validate(resp.json())
    assert out.degraded is False and out.degraded_reason is None and out.cache_hit is False
    # FCMA twice (the lower-case ticker is normalised); FCMC has no memo
    # and MSFT is outside the selection — both dropped and counted.
    assert [m.ticker for m in out.memo_view] == ["FCMA", "FCMA"]
    assert out.memo_view[0].memo_version == memos["FCMA"].version
    assert out.memo_view[0].memo_generated_at == datetime(2025, 3, 1)
    assert out.memo_view[0].memo_stale is False and out.memo_view[0].memo_stale_reason is None
    assert any("2 model sentence(s)" in c for c in out.caveats)
    assert any(c.startswith("Model note: ") for c in out.caveats)
    assert any("No stored memo for FCMC" in c for c in out.caveats)
    assert out.commentary_id is not None
    # One cheap-route call with the fixed system prompt and the bounded user prompt.
    assert call.call_count == 1
    kwargs = call.call_args.kwargs
    assert kwargs["route"] == "cheap" and kwargs["system"] == P.SYSTEM_PROMPT and kwargs["model"] is None
    prompt = call.call_args.args[0]
    assert "FCMA" in prompt and "FCMB" in prompt and "MSFT" not in prompt
    assert "[FCMC] no stored memo" in prompt
    assert "gross_profit / revenue" in prompt        # formula_text from the catalog
    assert len(prompt) <= P.MAX_PROMPT_CHARS
    rows = _rows()
    assert len(rows) == 1 and rows[0].degraded is False and rows[0].fingerprint == out.fingerprint
    assert rows[0].memo_versions["FCMA"]["version"] == memos["FCMA"].version
    assert rows[0].memo_versions["FCMC"] is None      # memos_used: None without a snapshot
    assert rows[0].user_id is None and rows[0].tickers == ["FCMA", "FCMB", "FCMC"]


def test_cache_hit_returns_the_stored_row_without_an_llm_call(client, seeded, memos, anon_ok, llm_available):
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        first = CommentaryOut.model_validate(_post(client, **_body()).json())
        second = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert call.call_count == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert second.commentary_id == first.commentary_id
    assert [m.text for m in second.memo_view] == [m.text for m in first.memo_view]
    assert second.observed == first.observed
    assert len(_rows()) == 1


def test_a_new_memo_version_invalidates_the_cache(client, seeded, memos, anon_ok, llm_available):
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        _post(client, **_body())
        snap = store_memo("FCMA")
        _set_generated_at(snap, datetime(2025, 3, 2))
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert call.call_count == 2 and out.cache_hit is False
    assert out.memo_view[0].memo_version == snap.version
    assert len(_rows()) == 2


def test_invalid_output_is_degraded_and_not_a_future_cache_hit(client, seeded, memos, anon_ok, llm_available):
    with patch.object(llm, "chat_json", return_value={"garbage": 1}) as call:
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
        assert out.degraded is True and out.degraded_reason == cc.REASON_INVALID
        assert out.memo_view == [] and out.observed
        rows = _rows()
        assert len(rows) == 1 and rows[0].degraded is True and rows[0].degraded_reason == cc.REASON_INVALID
        assert out.commentary_id == rows[0].id
        # A degraded row is auditable but never served: the next call tries again.
        again = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert call.call_count == 2 and again.cache_hit is False
    assert len(_rows()) == 1                         # refreshed in place, not duplicated


def test_only_dropped_items_is_invalid_output(client, seeded, memos, anon_ok, llm_available):
    canned = {"memo_view": [{"ticker": "MSFT", "text": "outside"}], "caveats": []}
    with patch.object(llm, "chat_json", return_value=canned):
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert out.degraded is True and out.degraded_reason == cc.REASON_INVALID and out.memo_view == []


# ---------------------------------------------------------------------------
# Memo staleness
# ---------------------------------------------------------------------------

def test_memo_stale_when_it_predates_the_last_displayed_period(client, seeded, memos, anon_ok, llm_available):
    _set_generated_at(memos["FCMA"], datetime(2024, 6, 1))   # before FY2024 ended 2024-12-31
    with patch.object(llm, "chat_json", return_value=CANNED):
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
    item = out.memo_view[0]
    assert item.memo_stale is True
    assert "predates the last displayed period FY2024 (ended 2024-12-31)" in (item.memo_stale_reason or "")
    assert any("FCMA memo v" in c and "is stale" in c for c in out.caveats)


def test_memo_stale_when_a_newer_filing_exists(client, seeded, memos, anon_ok, llm_available):
    with SessionLocal() as db:
        db.add(FilingDoc(ticker="FCMA", accession_number="0000000000-25-000001", filing_type="10-K",
                         filing_date=date(2025, 3, 15)))
        db.commit()
    with patch.object(llm, "chat_json", return_value=CANNED):
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
    item = out.memo_view[0]
    assert item.memo_stale is True and "new 10-K on 2025-03-15" in (item.memo_stale_reason or "")
    assert "predates" not in (item.memo_stale_reason or "")


def test_cache_hit_reflects_memo_staleness_now_not_at_generation(client, seeded, memos, anon_ok, llm_available):
    """A filing that lands after the row was written leaves the memo
    version — and so the cache key — unchanged. The hit must still carry
    today's verdict: `memo_store.memo_freshness` says stale, so does the
    served row, without an LLM call and without a charge."""
    before = _event_count()
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        first = CommentaryOut.model_validate(_post(client, **_body()).json())
        assert first.cache_hit is False
        assert all(m.memo_stale is False and m.memo_stale_reason is None for m in first.memo_view)
        assert not any("is stale" in c for c in first.caveats)
        with SessionLocal() as db:
            db.add(FilingDoc(ticker="FCMA", accession_number="0000000000-25-000003", filing_type="10-K",
                             filing_date=date(2025, 4, 15)))
            db.commit()
        second = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert call.call_count == 1 and second.cache_hit is True
    assert second.commentary_id == first.commentary_id and len(_rows()) == 1
    assert _event_count() == before
    fcma = [m for m in second.memo_view if m.ticker == "FCMA"]
    assert fcma and all(m.memo_stale is True for m in fcma)
    assert all("new 10-K on 2025-04-15" in (m.memo_stale_reason or "") for m in fcma)
    stale = [c for c in second.caveats if c.startswith(f"FCMA memo v{memos['FCMA'].version} is stale: ")]
    assert len(stale) == 1 and "new 10-K on 2025-04-15" in stale[0]
    assert not any(c.startswith("FCMB memo v") for c in second.caveats)
    # The stored row itself is untouched: the overlay is per request.
    stored = CommentaryOut.model_validate(_rows()[0].output)
    assert all(m.memo_stale is False for m in stored.memo_view)
    # The sentences, versions and generation time are what was generated.
    assert [m.text for m in second.memo_view] == [m.text for m in first.memo_view]
    assert [m.memo_version for m in second.memo_view] == [m.memo_version for m in first.memo_view]
    assert second.generated_at == first.generated_at


def test_cache_hit_keeps_generation_caveats_without_duplicating_stale_ones(
    client, seeded, memos, anon_ok, llm_available,
):
    """The memo is already stale when the row is written (predates rule),
    so the stored caveats carry a stale caveat. The hit rebuilds the list
    from today's verdict: one stale caveat, not two, and the
    generation-time caveats (drops, model notes) survive after it."""
    _set_generated_at(memos["FCMA"], datetime(2024, 6, 1))
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        first = CommentaryOut.model_validate(_post(client, **_body()).json())
        second = CommentaryOut.model_validate(_post(client, **_body()).json())
    assert call.call_count == 1 and second.cache_hit is True
    stale = [c for c in second.caveats if "FCMA memo v" in c and "is stale" in c]
    assert len(stale) == 1 and "predates the last displayed period FY2024" in stale[0]
    assert second.caveats == first.caveats
    assert any(c.startswith("Model note:") for c in second.caveats)
    assert any("were dropped" in c for c in second.caveats)
    assert second.caveats.index(stale[0]) < second.caveats.index(next(c for c in second.caveats if "were dropped" in c))
    assert second.memo_view[0].memo_stale is True


def test_memo_staleness_combines_both_rules(seeded, memos):
    _set_generated_at(memos["FCMA"], datetime(2024, 6, 1))
    with SessionLocal() as db:
        db.add(FilingDoc(ticker="FCMA", accession_number="0000000000-25-000002", filing_type="10-Q",
                         filing_date=date(2024, 8, 1)))
        db.commit()
        out = build_series(["FCMA"], ["revenue"], db=db)
        snap = db.get(MemoSnapshot, memos["FCMA"].id)
        stale, reason = cc.memo_staleness(snap, out, db)
    assert stale is True and "new 10-Q" in reason and "predates" in reason


def test_predates_rule_needs_a_stored_period_end(memos):
    """Rows without a `period_end` cannot say when the period closed, so
    the rule stays silent rather than guessing December."""
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        _wipe_periods(db)
        seed_annual_periods(db, "FCMA", {2024: FULL}, period_end_month=None, fetched_at=FRESH)
        db.commit()
        _set_generated_at(memos["FCMA"], datetime(2020, 1, 1))
        out = build_series(["FCMA"], ["revenue"], db=db)
        snap = db.get(MemoSnapshot, memos["FCMA"].id)
        stale, reason = cc.memo_staleness(snap, out, db)
        _wipe_periods(db)
    assert stale is False and reason is None


# ---------------------------------------------------------------------------
# Excerpts and the prompt budget
# ---------------------------------------------------------------------------

def test_excerpts_are_truncated_and_disclosed(client, seeded, memos, anon_ok, llm_available):
    long_thesis = "margin expansion " * 80          # ~1.3k chars
    snap = store_memo("FCMA")
    with SessionLocal() as db:
        row = db.get(MemoSnapshot, snap.id)
        row.memo_json = {**row.memo_json, "one_sentence_thesis": long_thesis}
        row.generated_at = datetime(2025, 3, 1)
        db.commit()
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        out = CommentaryOut.model_validate(_post(client, **_body()).json())
    prompt = call.call_args.args[0]
    assert long_thesis.strip() not in prompt
    assert "thesis: margin expansion" in prompt and "…" in prompt
    assert any("FCMA memo excerpts were shortened" in c for c in out.caveats)
    assert not any("FCMB memo excerpts were shortened" in c for c in out.caveats)


def test_memos_used_is_none_without_a_snapshot(seeded, memos):
    with SessionLocal() as db:
        out = build_series(["FCMA", "FCMC"], ["revenue"], db=db)
        excerpts = {t: cc.memo_excerpt(t, out, db) for t in ("FCMA", "FCMC")}
    assert excerpts["FCMC"] is None
    used = cc.memos_used(excerpts)
    assert used["FCMC"] is None
    assert used["FCMA"] == {"version": memos["FCMA"].version, "generated_at": datetime(2025, 3, 1).isoformat()}
    # A memo that no longer validates is treated as missing, not quoted blind.
    with SessionLocal() as db:
        row = db.get(MemoSnapshot, memos["FCMB"].id)
        row.memo_json = {"ticker": "FCMB"}
        db.commit()
        assert cc.memo_excerpt("FCMB", out, db) is None


def test_prompt_budget_holds_for_the_maximal_selection(client, anon_ok, llm_available):
    """5 companies × 4 metrics × 30 fiscal years, every ticker with a memo:
    the user prompt stays ≤ MAX_PROMPT_CHARS (≈6k tokens) without cutting
    a single displayed point."""
    metrics = ("revenue", "gross_margin", "fcf_after_sbc", "pe_ttm")
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        _wipe_periods(db)
        for i, t in enumerate(TICKERS):
            seed_annual_periods(
                db, t, {fy: {k: v * (1 + 0.1 * i) * (1 + 0.03 * (fy - 1995)) for k, v in FULL.items()}
                        for fy in range(1995, 2025)},
                fetched_at=FRESH,
            )
        db.commit()
    for t in TICKERS:
        _set_generated_at(store_memo(t), datetime(2025, 3, 1))
    try:
        body = _body(tickers=TICKERS, metrics=metrics)
        with patch.object(llm, "chat_json", return_value=CANNED) as call:
            resp = _post(client, **body)
        assert resp.status_code == 200, resp.text
        prompt = call.call_args.args[0]
        assert len(prompt) <= P.MAX_PROMPT_CHARS
        assert len(prompt) // 4 <= 6000
        assert len(P.SYSTEM_PROMPT) // 4 <= 600
        for fy in range(1995, 2025):
            assert f"FY{fy}=" in prompt
        for t in TICKERS:
            assert f"[{t}] stored memo v" in prompt
        out = CommentaryOut.model_validate(resp.json())
        assert not any("did not fit" in c for c in out.caveats)
    finally:
        with SessionLocal() as db:
            _wipe_periods(db)


def test_build_prompt_shrinks_memos_before_dropping_them(seeded, memos):
    with SessionLocal() as db:
        out = build_series(["FCMA", "FCMB"], ["revenue", "gross_margin"], db=db)
        excerpts = {t: cc.memo_excerpt(t, out, db) for t in ("FCMA", "FCMB")}
    from app.schemas.fundamentals import CommentaryRequest
    req = CommentaryRequest(tickers=["FCMA", "FCMB"], metrics=["revenue", "gross_margin"], fingerprint=out.fingerprint)
    observed = cc.observed_items(out)
    full = cc.build_prompt_for(req, out, observed, excerpts)
    assert full.memos_dropped is False and full.field_chars == P.MEMO_FIELD_CHARS
    metrics = [P.MetricLine(id=m, label=m, unit_type="currency", kind="reported", formula_text=m) for m in req.metrics]
    tight = P.build_prompt(
        tickers=req.tickers, metrics=metrics, periods=out.periods, normalize="none",
        series=cc._series_lines(out), observed=[o.text for o in observed],
        memos={t: e.block for t, e in excerpts.items() if e}, unavailable={},
        max_chars=len(full.text) - 50,
    )
    assert tight.field_chars < P.MEMO_FIELD_CHARS and len(tight.text) <= len(full.text) - 50
    hopeless = P.build_prompt(
        tickers=req.tickers, metrics=metrics, periods=out.periods, normalize="none",
        series=cc._series_lines(out), observed=[o.text for o in observed],
        memos={t: e.block for t, e in excerpts.items() if e}, unavailable={},
        max_chars=1200,
    )
    assert hopeless.memos_dropped is True and len(hopeless.text) <= 1200


# ---------------------------------------------------------------------------
# Observed in the data — deterministic
# ---------------------------------------------------------------------------

def test_observed_items_are_unit_aware_and_reasoned(seeded):
    with SessionLocal() as db:
        out = build_series(["FCMA", "FCMB"], ["revenue", "gross_margin", "pe_ttm"], db=db)
    items = cc.observed_items(out)
    texts = [i.text for i in items]
    rev = next(t for t in texts if t.startswith("FCMA Revenue:"))
    assert "FY2017 USD 1.00K → FY2024 USD 1.35K (+35.0%, 4.4% CAGR over 7 years)" in rev
    gm = next(t for t in texts if t.startswith("FCMA Gross margin:"))
    assert "60.0% → FY2024 60.0% (+0.0 pp)" in gm
    # No prices for throwaway tickers: the market metric says why, never zero.
    pe = next(t for t in texts if t.startswith("FCMA P/E (at fiscal year end): no value"))
    assert "no_price ×8" in pe
    rank = next(t for t in texts if t.startswith("Revenue at FY2024, highest to lowest"))
    assert rank.index("FCMB") < rank.index("FCMA")
    assert all(len(i.refs) >= 1 for i in items if not i.text.startswith("Revenue at"))
    assert cc.fmt_value(0.1234, "percent", None) == "12.3%"
    assert cc.fmt_value(12.34, "multiple", None) == "12.3x"
    assert cc.fmt_value(2.5e9, "currency", "EUR") == "EUR 2.50B"
    assert cc.fmt_value(None, "currency", "USD") == "n/a"


def test_observed_ranking_refuses_mixed_currencies():
    from app.schemas.fundamentals import MetricSeries, SeriesPoint
    series = [
        MetricSeries(ticker="AAA", metric="revenue", unit_type="currency", kind="reported", currency="USD",
                     points=[SeriesPoint(period="FY2024", value=10.0)]),
        MetricSeries(ticker="BBB", metric="revenue", unit_type="currency", kind="reported", currency="EUR",
                     points=[SeriesPoint(period="FY2024", value=20.0)]),
    ]
    item = cc._ranking_observation("revenue", "Revenue", series, "FY2024")
    assert item is not None and "different currencies (EUR, USD)" in item.text


# ---------------------------------------------------------------------------
# Wall on — charge, release, meters, leases, shape
# ---------------------------------------------------------------------------

def test_charge_happens_before_the_call_and_is_released_on_none(auth_on, client, seeded, memos, llm_available):
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    pk = usage.period_key(NOW)
    seen: list[int] = []

    def during_call(*_a, **_kw):
        with SessionLocal() as db:
            seen.append(usage.used(db, uid, "chart_commentary", pk))
        return None

    with patch.object(llm, "chat_json", side_effect=during_call) as call:
        resp = _post(client, tok, **_body(years=5))
    assert call.call_count == 1 and seen == [1]          # reserved before the call
    out = CommentaryOut.model_validate(resp.json())
    assert out.degraded is True and out.degraded_reason == cc.REASON_INVALID
    events = usage_events(uid, "chart_commentary")
    assert [e.status for e in events] == ["released"]
    with SessionLocal() as db:
        assert usage.used(db, uid, "chart_commentary", pk) == 0
        assert db.query(ratelimit.ActiveAction).filter(ratelimit.ActiveAction.user_id == uid).count() == 0


def test_success_commits_once_and_a_cache_hit_is_free(auth_on, client, seeded, memos, llm_available):
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    pk = usage.period_key(NOW)
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        first = CommentaryOut.model_validate(_post(client, tok, **_body(years=5)).json())
        second = CommentaryOut.model_validate(_post(client, tok, **_body(years=5)).json())
    assert call.call_count == 1 and first.degraded is False and second.cache_hit is True
    events = usage_events(uid, "chart_commentary")
    assert [e.status for e in events] == ["committed"]
    with SessionLocal() as db:
        assert usage.used(db, uid, "chart_commentary", pk) == 1
        row = db.query(ChartCommentary).one()
        assert row.user_id == uid


def test_quota_exceeded_is_a_structured_402(auth_on, client, seeded, memos, llm_available, monkeypatch):
    monkeypatch.setattr(settings, "entitlement_overrides_json", '{"chart_commentary": {"free": 1}}')
    features._parse_overrides.cache_clear()
    try:
        _sub, tok = free_user(auth_on)
        with patch.object(llm, "chat_json", return_value=CANNED) as call:
            assert _post(client, tok, **_body(years=5)).status_code == 200
            resp = _post(client, tok, **_body(metrics=("revenue", "net_margin"), years=5))
        assert call.call_count == 1
        detail = assert_structured(resp, code="quota_exceeded", status=402)
        assert detail["feature"] == "chart_commentary" and detail["plan"] == "free"
        assert detail["used"] == 1 and detail["limit"] == 1 and detail["remaining"] == 0
        assert detail["upgrade_url"] == "/pricing" and detail["resets_at"]
    finally:
        features._parse_overrides.cache_clear()


def test_two_in_flight_is_a_structured_429(auth_on, client, seeded, memos, llm_available):
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        for _ in range(2):
            assert ratelimit.lease(db, user_id=uid, feature="chart_commentary", max_concurrent=2)
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        resp = _post(client, tok, **_body(years=5))
    call.assert_not_called()
    detail = assert_structured(resp, code="concurrency_limited", status=429)
    assert detail["scope"] == "concurrency:chart_commentary" and resp.headers["retry-after"]
    assert usage_events(uid, "chart_commentary") == []


def test_free_shape_applies_to_commentary_too(auth_on, client, seeded, memos):
    _sub, tok = free_user(auth_on)
    fp = _fp(("FCMA", "FCMB", "FCMC"), ("revenue",), years=5)
    resp = _post(client, tok, tickers=["FCMA", "FCMB", "FCMC"], metrics=["revenue"], fingerprint=fp)
    detail = assert_structured(resp, code="plan_required", status=402)
    assert detail["extra"]["limits"] == {"max_companies": 2, "max_metrics": 2, "max_years": 5}
    # Free's default range is the plan's 5 years — the fingerprint of a
    # 5-year chart is the one that matches.
    resp = _post(client, tok, tickers=["FCMA"], metrics=["revenue"], fingerprint=_fp(("FCMA",), ("revenue",), years=5))
    assert resp.status_code == 200, resp.text


def test_anonymous_under_the_wall_is_401(auth_on, client, seeded, memos):
    assert_structured(_post(client, **_body()), code="auth_required", status=401)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_route_is_mounted_with_the_ip_limit():
    assert "/api/fundamentals/commentary" in app.openapi()["paths"]
    limits = limiter._route_limits["app.api.routes_fundamentals_commentary.post_commentary"]
    assert [str(lim.limit) for lim in limits] == ["10 per 1 minute"]


def test_unknown_metric_is_a_structured_422_not_a_500(seeded, memos, anon_ok, llm_available):
    """The catalog check runs before anything is recomputed, in the same
    envelope the series route uses; `raise_server_exceptions=False` so a
    regression shows up as the 500 a client would see, not a traceback."""
    quiet = TestClient(app, raise_server_exceptions=False)
    fp = "f" * 64
    with patch.object(llm, "chat_json", return_value=CANNED) as call:
        resp = _post(quiet, tickers=["FCMA"], metrics=["bogus_metric", "revenue"], fingerprint=fp)
    detail = assert_structured(resp, code="invalid_request", status=422)
    assert detail["message"] == "unknown metric id(s): bogus_metric"
    assert detail["extra"]["unknown_metrics"] == ["bogus_metric"]
    assert "revenue" in detail["extra"]["known_metrics"]
    assert call.call_count == 0 and _rows() == []
    # Same body, same answer on the series route: one error shape to switch on.
    series = quiet.post("/api/fundamentals/series", json={"tickers": ["FCMA"], "metrics": ["bogus_metric", "revenue"]})
    assert series.status_code == 422 and series.json()["detail"]["message"] == detail["message"]


def test_service_value_error_is_a_structured_422(client, seeded, monkeypatch):
    """The honest fallback: a `ValueError` out of the service (a future
    rule the schema does not mirror) is the caller's mistake, never a 500."""
    quiet = TestClient(app, raise_server_exceptions=False)
    monkeypatch.setattr(cc, "generate", lambda *a, **k: (_ for _ in ()).throw(ValueError("years must be >= 1")))
    resp = _post(quiet, tickers=["FCMA"], metrics=["revenue"], fingerprint="f" * 64)
    detail = assert_structured(resp, code="invalid_request", status=422)
    assert detail["message"] == "years must be >= 1"


def test_request_validation_422s(client, seeded):
    fp = "f" * 64
    assert _post(client, tickers=[], metrics=["revenue"], fingerprint=fp).status_code == 422
    assert _post(client, tickers=["FCMA"], metrics=[], fingerprint=fp).status_code == 422
    assert _post(client, tickers=["FCMA"], metrics=["revenue"], fingerprint="abc").status_code == 422
    assert _post(client, tickers=[*TICKERS, "FCMF"], metrics=["revenue"], fingerprint=fp).status_code == 422


def test_cache_key_moves_with_every_input():
    base = dict(tickers=["A"], metrics=["revenue"], years=None, normalize="none", fingerprint="f" * 64,
                memo_versions={"A": {"version": 1, "generated_at": "x"}})
    k = cc.cache_key(**base)
    assert k == cc.cache_key(**base)
    assert k != cc.cache_key(**{**base, "memo_versions": {"A": {"version": 2, "generated_at": "x"}}})
    assert k != cc.cache_key(**{**base, "memo_versions": {"A": None}})
    assert k != cc.cache_key(**{**base, "years": 5})
    assert k != cc.cache_key(**{**base, "normalize": "indexed"})
    assert k != cc.cache_key(**{**base, "fingerprint": "e" * 64})


def test_commentary_cache_key_includes_presentation_version(monkeypatch):
    """W2a: the excerpt is the PRESENTED memo, so a change to the rules that
    hide sections must miss the cache like a prompt change does."""
    from app.services import memo_sections
    base = dict(tickers=["A"], metrics=["revenue"], years=None, normalize="none", fingerprint="f" * 64,
                memo_versions={"A": {"version": 1, "generated_at": "x"}})
    k = cc.cache_key(**base)
    monkeypatch.setattr(memo_sections, "PRESENTATION_VERSION", memo_sections.PRESENTATION_VERSION + 1)
    assert cc.cache_key(**base) != k


def test_presentation_version_2_and_commentary_cache_key():
    """D4 (plan P12): the presenter labels reviews and adds the debate, so
    the version is 2, and every commentary row cached under version 1 misses
    once (and is regenerated) because the key reads the constant at call
    time rather than a copy taken at import."""
    import hashlib
    import json as _json

    from app.services import memo_sections
    assert memo_sections.PRESENTATION_VERSION == 2
    base = dict(tickers=["A"], metrics=["revenue"], years=None, normalize="none", fingerprint="f" * 64,
                memo_versions={"A": {"version": 1, "generated_at": "x"}})

    def key_at(version: int) -> str:
        payload = {
            "catalog_version": cc.catalog.CATALOG_VERSION, "prompt_version": P.PROMPT_VERSION,
            "presentation_version": version, "tickers": ["A"], "metrics": ["revenue"], "years": None,
            "normalize": "none", "fingerprint": "f" * 64, "memo_versions": {"A": 1},
        }
        blob = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    assert cc.cache_key(**base) == key_at(2)
    assert cc.cache_key(**base) != key_at(1)


def test_memo_excerpt_drops_hidden_fields(seeded):
    """The model never relates the chart to text the memo page withholds:
    the template thesis, the fallback mispricing card and the template case
    headlines are left out; the rating and the computed verdict stay."""
    import json as _json
    from pathlib import Path

    from app.schemas import StockMemoOut
    from app.services import memo_sections, memo_store
    raw = _json.loads((Path(__file__).parent / "fixtures" / "memo_sections"
                       / "googl_live_prepflag.json").read_text())
    memo = StockMemoOut.model_validate(raw).model_copy(update={"ticker": "FCMB"})
    snap = memo_store.save_memo(memo)
    _set_generated_at(snap, datetime(2025, 3, 1))
    with SessionLocal() as db:
        out = build_series(["FCMB"], ["revenue"], db=db)
        e = cc.memo_excerpt("FCMB", out, db)
    assert e is not None
    names = [n for n, _ in e.block.fields]
    assert names == ["rating", "valuation verdict"]
    text, _ = P.render_memo_block(e.block, field_chars=P.MEMO_FIELD_CHARS, block_chars=P.MEMO_BLOCK_CHARS)
    assert memo_sections.SIG["pm_view_tail"].text not in text
    assert memo.one_sentence_thesis[:40] not in text and memo_sections.UNAVAILABLE_TEXT not in text


def test_make_memo_round_trips_into_an_excerpt(seeded):
    """The excerpt quotes the memo fields the contract names, nothing else."""
    from app.services import memo_store
    snap = memo_store.save_memo(make_memo(
        ticker="FCMB", company_name="FCMB Corp", one_sentence_thesis="Cash conversion is the story.",
    ))
    _set_generated_at(snap, datetime(2025, 3, 1))
    with SessionLocal() as db:
        out = build_series(["FCMB"], ["revenue"], db=db)
        e = cc.memo_excerpt("FCMB", out, db)
    assert e is not None and e.version == snap.version
    names = [n for n, _ in e.block.fields]
    assert names == ["rating", "thesis", "consensus view", "our view", "gap", "valuation verdict", "bull case", "bear case"]
    text, truncated = P.render_memo_block(e.block, field_chars=P.MEMO_FIELD_CHARS, block_chars=P.MEMO_BLOCK_CHARS)
    assert "thesis: Cash conversion is the story." in text and truncated is False
