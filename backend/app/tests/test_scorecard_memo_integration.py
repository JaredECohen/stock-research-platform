"""Phase 6, slice D — the scorecard inside the memo pipeline.

What is pinned here (plan §8-D as amended by the orchestrator decisions):

* `scorecard_context.prompt_block` is <= 600 chars, names the version,
  as-of, percentiles and the top contributors, and is "" without a score.
* `scorecard_context.summarize` / `detect_disagreement`: material / watch
  / none thresholds, the valuation contradiction, the direction, and the
  rule that coverage below 0.6 is never material. A missing percentile is
  n/a — never a flag, never "agrees".
* The compounder / inflection profile rule (z >= 0.5) is computed
  server-side and carried on the summary's notes so the memo block and the
  page agree.
* A demo memo with no scorecard row: `memo.scorecard is None`,
  "Fundamental Scorecard" is a SOFT degradation, the memo persists and an
  old snapshot without the field still validates.
* `ENABLE_SCORECARD=false`: the field is None and NOTHING is recorded.
* A demo memo with a seeded score row: `memo.scorecard` is populated, the
  valuation analyst's payload and the PM context carry the block, the
  disagreement row is written when material, and the rating blend output
  is identical with and without the row (the scorecard informs; it does
  not move ratings in this phase).
* `update_orchestrator.handle_scorecard_disagreements`: flag off means no
  enqueue; the daily cap; the `should_auto_regen` gate; never re-queues a
  (ticker, version, as_of) that already has a review.

Every test is deterministic and network-free: blank LLM keys, the demo
provider, synthetic score rows, `_utcnow` seams pinned where a clock
matters. Rows are purged by the test's own requester tag.
"""
from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import Any

import pytest

from app.agents import graph, scorecard_context
from app.agents import valuation_agent as va
from app.agents.pm_context import build_pm_context
from app.config import settings
from app.database import SessionLocal
from app.models import MemoSnapshot, RegenJob, ScorecardDisagreement, ScorecardRun, ScorecardScore
from app.schemas import (
    ScorecardCategory,
    ScorecardContribution,
    ScorecardSummary,
    StockMemoOut,
    score_from_rating_label,
)
from app.services import memo_store, update_orchestrator
from app.services.scorecard_service import VERSION_KEY

# Two requester tags: per-test rows are purged around every test; the
# module-scoped memo fixtures own their rows until the module ends (a
# per-test purge would delete the row a later test asserts against).
REQUESTED_BY = "test-scorecard-memo"
REQUESTED_BY_MODULE = "test-scorecard-memo-m"
AS_OF = date(2026, 6, 30)
NOW = datetime(2026, 7, 1, 12, 0, 0)


@pytest.fixture(autouse=True)
def _blank_keys_guard():
    # Not `assert not settings.has_llm`: pytest prints a failing assert's
    # operands, here the Settings object, and this guard fails precisely
    # when a developer .env with live keys is loaded.
    if settings.has_llm:
        pytest.fail(
            "test_scorecard_memo_integration must run with blank LLM keys "
            "(OPENAI_API_KEY='' ANTHROPIC_API_KEY='' GEMINI_API_KEY='')",
            pytrace=False,
        )
    yield


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _cat(z: float | None, pct: float | None, *, weight: float = 0.125) -> ScorecardCategory:
    return ScorecardCategory(
        z=z, score=None if z is None else 50 + 20 * z, percentile=pct, weight=weight,
        coverage=1.0 if z is not None else 0.0, n_features=4, n_available=4 if z is not None else 0,
    )


def _summary(
    *, pct: float | None = 22.0, val_pct: float | None = 18.0, coverage: float = 0.93,
    compounder: float | None = 0.8, inflection: float | None = -0.3, run_id: str = "",
) -> ScorecardSummary:
    return ScorecardSummary(
        version_key=VERSION_KEY, as_of=AS_OF, run_id=run_id, overall_z=-0.62, overall_score=37.6,
        universe_percentile=pct, sector_percentile=30.0, coverage=coverage,
        categories={
            "valuation": _cat(-1.1, val_pct), "quality": _cat(0.9, 80.0), "growth": _cat(-0.4, 40.0),
            "profitability": _cat(0.7, 74.0), "efficiency": _cat(None, None), "leverage": _cat(-0.2, 45.0),
            "capital_allocation": _cat(0.6, 70.0), "earnings_quality": _cat(-1.8, 8.0),
        },
        top_positive=[
            ScorecardContribution(feature="roic", family="quality", z=1.4, contribution=0.044),
            ScorecardContribution(feature="fcf_margin", family="profitability", z=1.1, contribution=0.034),
        ],
        top_negative=[
            ScorecardContribution(feature="accruals_ratio", family="earnings_quality", z=-1.8, contribution=-0.056),
            ScorecardContribution(feature="ebitda_ev_yield", family="valuation", z=-1.3, contribution=-0.041),
        ],
        profiles={"compounder": compounder, "inflection": inflection},
        latest_period="FY2025", data_available_at=date(2026, 3, 16), price_date=AS_OF, stale=False,
    )


def _memo_stub(rating: str, verdict: str = "fairly_priced") -> Any:
    """Just the fields the detector reads; a full StockMemoOut is not needed."""
    from types import SimpleNamespace

    from app.schemas import ValuationVerdict
    return SimpleNamespace(
        ticker="NVDA", rating_label=rating,
        valuation_verdict=ValuationVerdict(verdict=verdict, summary="stub"),
    )


def _seed_score_row(
    ticker: str, summary: ScorecardSummary, *, requested_by: str = REQUESTED_BY,
) -> tuple[int, int, str]:
    """A succeeded run + one score row for `ticker` shaped like the service
    writes it (the `_profiles` meta rides inside `category_z`)."""
    run_uuid = str(uuid.uuid4())
    with SessionLocal() as db:
        for model in (ScorecardRun, ScorecardScore, ScorecardDisagreement):
            model.__table__.create(bind=db.get_bind(), checkfirst=True)
        run = ScorecardRun(
            run_id=run_uuid, version_key=VERSION_KEY, as_of=AS_OF, run_kind="month_end", status="succeeded",
            attempts=1, requested_by=requested_by, enqueued_at=NOW, started_at=NOW, finished_at=NOW,
            params={"spec_hash": "test"},
        )
        db.add(run)
        db.flush()
        cz = {k: v.z for k, v in summary.categories.items()}
        cz["_profiles"] = dict(summary.profiles)
        cz["_coverage"] = {k: v.coverage for k, v in summary.categories.items()}
        row = ScorecardScore(
            run_id=run.id, version_key=VERSION_KEY, as_of=AS_OF, ticker=ticker, sector="Information Technology",
            overall_z=summary.overall_z, overall_score=summary.overall_score,
            universe_percentile=summary.universe_percentile, sector_percentile=summary.sector_percentile,
            coverage=summary.coverage, category_z=cz,
            category_percentile={k: v.percentile for k, v in summary.categories.items()},
            feature_raw={"roic": 0.31}, feature_z={"roic": 1.4, "accruals_ratio": -1.8},
            top_positive=[c.model_dump() for c in summary.top_positive],
            top_negative=[c.model_dump() for c in summary.top_negative],
            latest_period=summary.latest_period, data_available_at=summary.data_available_at,
            price_date=summary.price_date, inputs_hash="test", is_month_end=True, created_at=NOW,
        )
        db.add(row)
        db.commit()
        return int(run.id), int(row.id), run_uuid


def _purge(requested_by: str = REQUESTED_BY) -> None:
    with SessionLocal() as db:
        for model in (ScorecardRun, ScorecardScore, ScorecardDisagreement):
            model.__table__.create(bind=db.get_bind(), checkfirst=True)
        run_ids = [r[0] for r in db.query(ScorecardRun.id).filter(ScorecardRun.requested_by == requested_by).all()]
        if run_ids:
            score_ids = [r[0] for r in db.query(ScorecardScore.id).filter(ScorecardScore.run_id.in_(run_ids)).all()]
            if score_ids:
                db.query(ScorecardDisagreement).filter(
                    ScorecardDisagreement.scorecard_score_id.in_(score_ids)
                ).delete(synchronize_session=False)
            db.query(ScorecardScore).filter(ScorecardScore.run_id.in_(run_ids)).delete(synchronize_session=False)
            db.query(ScorecardRun).filter(ScorecardRun.id.in_(run_ids)).delete(synchronize_session=False)
        db.query(ScorecardDisagreement).filter(ScorecardDisagreement.ticker.like("ZSD%")).delete(synchronize_session=False)
        db.query(RegenJob).filter(RegenJob.ticker.like("ZSD%")).delete(synchronize_session=False)
        db.commit()


@pytest.fixture(autouse=True)
def _isolate_rows():
    _purge()
    yield
    _purge()


# ---------------------------------------------------------------------------
# 1. prompt_block
# ---------------------------------------------------------------------------

def test_prompt_block_is_empty_without_a_score():
    assert scorecard_context.prompt_block(None) == ""


def test_prompt_block_fits_the_budget_and_names_the_read():
    block = scorecard_context.prompt_block(_summary())
    assert len(block) <= scorecard_context.PROMPT_BLOCK_MAX_CHARS
    assert VERSION_KEY in block and AS_OF.isoformat() in block
    assert "22th pct" in block or "22nd pct" in block or "universe 22" in block
    assert "roic" in block and "accruals_ratio" in block
    assert "not a recommendation" in block
    # Observed and model layers are both labelled.
    assert "Observed rank" in block and "Model read" in block


def test_prompt_block_truncates_long_contributor_names_without_losing_the_caveat():
    s = _summary()
    s.top_positive = [
        ScorecardContribution(feature="x" * 200, family="quality", z=1.0, contribution=0.01) for _ in range(3)
    ]
    block = scorecard_context.prompt_block(s)
    assert len(block) <= scorecard_context.PROMPT_BLOCK_MAX_CHARS
    assert "not a recommendation" in block


def test_prompt_block_renders_missing_percentiles_as_na_not_zero():
    block = scorecard_context.prompt_block(_summary(pct=None, val_pct=None))
    assert "universe n/a" in block and "valuation family n/a" in block
    assert not re.search(r"(?<!\d)0th pct", block), block


# ---------------------------------------------------------------------------
# 2. Profile rule and disagreement thresholds
# ---------------------------------------------------------------------------

def test_profile_rule_uses_the_shared_threshold():
    assert scorecard_context.PROFILE_THRESHOLD_Z == 0.5
    reads = scorecard_context.profile_reads(_summary(compounder=0.5, inflection=0.49))
    assert reads == {"compounder": "reads", "inflection": "does_not_read"}
    assert scorecard_context.profile_reads(_summary(compounder=None, inflection=None)) == {
        "compounder": "n/a", "inflection": "n/a",
    }
    head = scorecard_context.profile_headline(_summary(compounder=0.8, inflection=-0.3))
    assert head.startswith("Reads as a compounder") and "+0.80" in head and "-0.30" in head
    assert "n/a" in scorecard_context.profile_headline(_summary(compounder=None, inflection=None))


def test_for_memo_carries_the_profile_read_and_reconciliation():
    out = scorecard_context.for_memo(_summary(), reconciliation="  Narrative overrides on FY26 margin step-up. ")
    assert out is not None
    assert out.reconciliation == "Narrative overrides on FY26 margin step-up."
    assert any(n.startswith(f"Profile read ({VERSION_KEY}): Reads as a compounder") for n in out.notes)
    assert scorecard_context.for_memo(None) is None


@pytest.mark.parametrize("rating, pct, expected", [
    ("Bullish", 22.0, "material"),        # gap +48 >= 40
    ("Very Bullish", 22.0, "material"),   # gap +68
    ("Neutral", 22.0, "watch"),           # gap +28 >= 25
    ("Bearish", 22.0, None),              # gap +8
    ("Very Bearish", 22.0, None),         # gap -12
    ("Bearish", 75.0, "material"),        # gap -45
    ("Neutral", 78.0, "watch"),           # gap -28
])
def test_detect_disagreement_thresholds(rating, pct, expected):
    flag = scorecard_context.detect_disagreement(_summary(pct=pct, val_pct=50.0), _memo_stub(rating))
    if expected is None:
        assert flag is None
    else:
        assert flag is not None
        assert flag.severity == expected
        assert flag.dimension == "overall"
        assert flag.gap == pytest.approx(score_from_rating_label(rating) - pct, abs=0.01)
        assert flag.direction == ("narrative_above_quant" if flag.gap > 0 else "narrative_below_quant")
        assert rating in flag.note and f"{pct:.1f}" in flag.note


def test_low_coverage_is_never_material():
    flag = scorecard_context.detect_disagreement(_summary(pct=5.0, coverage=0.45), _memo_stub("Very Bullish"))
    assert flag is not None
    assert flag.severity == "watch"
    assert "held at watch" in flag.note and "45%" in flag.note
    val = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=12.0, coverage=0.45), _memo_stub("Neutral", verdict="undervalued"),
    )
    assert val is not None and val.severity == "watch" and val.dimension == "valuation"


def test_valuation_contradiction_flags_on_the_verdict_word():
    under = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=12.0), _memo_stub("Neutral", verdict="undervalued"),
    )
    assert under is not None
    assert (under.severity, under.dimension, under.direction) == ("material", "valuation", "narrative_above_quant")
    assert "undervalued" in under.note and "12.0" in under.note
    over = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=88.0), _memo_stub("Neutral", verdict="overvalued"),
    )
    assert over is not None and over.direction == "narrative_below_quant"
    # The verdict agreeing with the family rank is not a contradiction.
    assert scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=88.0), _memo_stub("Neutral", verdict="undervalued"),
    ) is None
    # An overall material gap takes precedence over the valuation dimension.
    both = scorecard_context.detect_disagreement(
        _summary(pct=10.0, val_pct=12.0), _memo_stub("Bullish", verdict="undervalued"),
    )
    assert both is not None and both.dimension == "overall"


def test_valuation_contradiction_is_material_even_when_the_overall_gap_only_reaches_watch():
    """Plan §5.5 verbatim: material = coverage >= 0.6 AND (|gap| >= 40 OR a
    valuation contradiction). The overall gap sitting in the watch band
    must not downgrade a contradiction the plan calls material (review
    finding: pct 40 / val 20 / Bullish+undervalued returned 'watch')."""
    flag = scorecard_context.detect_disagreement(
        _summary(pct=40.0, val_pct=20.0, coverage=0.9), _memo_stub("Bullish", verdict="undervalued"),
    )
    assert flag is not None
    assert (flag.severity, flag.dimension, flag.direction) == ("material", "valuation", "narrative_above_quant")
    assert flag.gap == pytest.approx(70.0 - 20.0)
    # The note names both triggers so a reviewer sees why it is material.
    assert "undervalued" in flag.note and "20.0" in flag.note
    assert "overall gap +30.0" in flag.note and "40.0" in flag.note
    # Coverage below the floor still holds it at watch — never material.
    held = scorecard_context.detect_disagreement(
        _summary(pct=40.0, val_pct=20.0, coverage=0.45), _memo_stub("Bullish", verdict="undervalued"),
    )
    assert held is not None and held.severity == "watch" and held.dimension == "valuation"
    assert "held at watch" in held.note
    # The mirror case (Bearish + overvalued against a cheap valuation family).
    mirror = scorecard_context.detect_disagreement(
        _summary(pct=60.0, val_pct=80.0, coverage=0.9), _memo_stub("Bearish", verdict="overvalued"),
    )
    assert mirror is not None
    assert (mirror.severity, mirror.dimension, mirror.direction) == ("material", "valuation", "narrative_below_quant")


def _evidence_stub(rating: str, verdict: str) -> Any:
    """A W2b memo: its verdict is an evidence read (`basis="evidence"`)."""
    from types import SimpleNamespace

    from app.schemas import ValuationVerdict
    return SimpleNamespace(
        ticker="NVDA", rating_label=rating,
        valuation_verdict=ValuationVerdict(verdict=verdict, basis="evidence", summary="evidence"),
    )


def test_valuation_labels_read_the_rating_direction_on_evidence_verdicts():
    """W2b: an evidence verdict already counts the valuation-family rank, so
    comparing it with that rank would pit evidence against evidence and the
    `narrative_*_quant` label would describe no narrative. On those memos
    the narrative's valuation stance is the RATING's direction."""
    above = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=12.0), _evidence_stub("Bullish", "fairly_priced"))
    assert above is not None
    assert (above.severity, above.dimension, above.direction) == ("material", "valuation", "narrative_above_quant")
    assert above.note.startswith("Rating Bullish (reads undervalued) vs valuation-family universe percentile 12.0")
    below = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=88.0), _evidence_stub("Bearish", "fairly_priced"))
    assert below is not None and below.direction == "narrative_below_quant"
    # A Neutral memo takes no valuation stance, whatever the evidence reads
    # (the rating-derived rule would have flagged "undervalued" here).
    assert scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=12.0), _evidence_stub("Neutral", "undervalued")) is None
    # Legacy memos (verdict derived from the rating) keep the verdict rule.
    legacy = scorecard_context.detect_disagreement(
        _summary(pct=50.0, val_pct=12.0), _memo_stub("Neutral", verdict="undervalued"))
    assert legacy is not None and legacy.note.startswith("Valuation verdict 'undervalued'")


def test_missing_percentile_is_na_not_a_flag():
    assert scorecard_context.detect_disagreement(_summary(pct=None), _memo_stub("Very Bullish")) is None
    assert scorecard_context.summarize(None, _memo_stub("Bullish")) is None


def test_summarize_returns_a_copy_with_the_flag():
    s = _summary()
    out = scorecard_context.summarize(s, _memo_stub("Bullish"))
    assert out is not None and out is not s
    assert s.disagreement is None and out.disagreement is not None
    assert out.disagreement.severity == "material"


def test_seed_question_names_the_observed_figures():
    s = _summary()
    flag = scorecard_context.detect_disagreement(s, _memo_stub("Bullish"))
    assert flag is not None
    q = scorecard_context.seed_question("NVDA", s, flag)
    assert q.target_agent == "valuation"
    assert "NVDA" in q.question and "accruals_ratio" in q.question and "falsify" in q.question
    assert len(q.question) <= 600


def test_seed_questions_target_valuation_and_earnings():
    """Plan §5.5: the seeded question targets `valuation` AND `earnings`.
    One stored text, one CritiqueQuestion per target, in that order."""
    s = _summary()
    flag = scorecard_context.detect_disagreement(s, _memo_stub("Bullish"))
    assert flag is not None
    qs = scorecard_context.seed_questions("NVDA", s, flag)
    assert [q.target_agent for q in qs] == list(scorecard_context.SEED_TARGETS) == ["valuation", "earnings"]
    assert len({q.question for q in qs}) == 1
    assert qs[0].question == scorecard_context.seed_question("NVDA", s, flag).question
    assert all(q.why_it_matters for q in qs) and qs[0].why_it_matters != qs[1].why_it_matters


# ---------------------------------------------------------------------------
# 3. Memo runs (demo mode)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def memo_without_row():
    """Demo memo with no scorecard row on file (the CI default)."""
    _purge()
    _purge(REQUESTED_BY_MODULE)
    return graph.run_stock_memo("NVDA", force_refresh=True)


def test_memo_without_row_says_na_and_degrades_softly(memo_without_row):
    memo = memo_without_row
    assert memo.scorecard is None
    assert scorecard_context.AGENT_NAME in memo.degraded_agents
    ev = next(e for e in memo.degradation_events if e["agent"] == scorecard_context.AGENT_NAME)
    assert ev["error_type"] == "DataUnavailable"
    assert "no scorecard row on file" in ev["message"]
    # Persisted, and round-trips through the store with the field null.
    snap = memo_store.latest_memo("NVDA")
    assert snap is not None
    assert memo_store.memo_to_pydantic(snap).scorecard is None


def test_old_snapshot_without_the_field_still_validates(memo_without_row):
    payload = memo_without_row.model_dump(mode="json")
    payload.pop("scorecard", None)
    hydrated = StockMemoOut.model_validate(payload)
    assert hydrated.scorecard is None


def test_kill_switch_off_omits_silently(monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard", False)
    calls: list[str] = []
    monkeypatch.setattr(
        scorecard_context, "load_for_memo",
        lambda *a, **k: calls.append("read") or None,
    )
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    assert memo.scorecard is None
    assert scorecard_context.AGENT_NAME not in memo.degraded_agents
    assert not any(e["agent"] == scorecard_context.AGENT_NAME for e in memo.degradation_events)
    # The gather stage never even called the reader.
    assert calls == []


def test_read_crash_is_a_hard_degradation(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated scorecard read failure")

    monkeypatch.setattr(scorecard_context, "load_for_memo", boom)
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    assert memo.scorecard is None
    ev = next(e for e in memo.degradation_events if e["agent"] == scorecard_context.AGENT_NAME)
    assert ev["error_type"] == "RuntimeError"


@pytest.fixture(scope="module")
def memo_with_row():
    """Demo memo with a seeded score row; thresholds lowered so the demo
    rating (whatever it is) sits a material distance from a 5th-percentile
    row. Returns (memo, score_row_id)."""
    _purge(REQUESTED_BY_MODULE)
    _run_id, score_id, _run_uuid = _seed_score_row(
        "NVDA", _summary(pct=5.0, val_pct=5.0), requested_by=REQUESTED_BY_MODULE,
    )
    old_material, old_watch = settings.scorecard_disagreement_material, settings.scorecard_disagreement_watch
    settings.scorecard_disagreement_material = 3.0
    settings.scorecard_disagreement_watch = 1.0
    try:
        memo = graph.run_stock_memo("NVDA", force_refresh=True)
    finally:
        settings.scorecard_disagreement_material = old_material
        settings.scorecard_disagreement_watch = old_watch
    yield memo, score_id
    _purge(REQUESTED_BY_MODULE)


def test_memo_with_row_carries_the_summary(memo_with_row):
    memo, _score_id = memo_with_row
    sc = memo.scorecard
    assert sc is not None
    assert sc.version_key == VERSION_KEY and sc.as_of == AS_OF
    assert sc.universe_percentile == 5.0
    assert sc.top_negative[0].feature == "accruals_ratio"
    assert sc.profiles == {"compounder": 0.8, "inflection": -0.3}
    assert any(n.startswith("Profile read") for n in sc.notes)
    assert scorecard_context.AGENT_NAME not in memo.degraded_agents
    # Round-trips through the store.
    snap = memo_store.latest_memo("NVDA")
    assert snap is not None
    assert memo_store.memo_to_pydantic(snap).scorecard is not None


def test_memo_with_row_flags_and_persists_the_disagreement(memo_with_row):
    memo, score_id = memo_with_row
    flag = memo.scorecard.disagreement
    assert flag is not None and flag.severity == "material"
    # The flag is a finding, not an outage.
    assert "Scorecard" not in " ".join(memo.degraded_agents)
    snap = memo_store.latest_memo("NVDA")
    with SessionLocal() as db:
        rows = db.query(ScorecardDisagreement).filter(ScorecardDisagreement.scorecard_score_id == score_id).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.memo_snapshot_id == snap.id
        assert row.status == "open" and row.severity == "material"
        assert row.memo_rating == memo.rating_label
        assert row.memo_rating_score == score_from_rating_label(memo.rating_label)
        assert "NVDA" in row.seed_question and "falsify" in row.seed_question


def test_rating_blend_is_unchanged_with_and_without_a_row(memo_without_row, memo_with_row):
    with_row, _ = memo_with_row
    without = memo_without_row
    assert with_row.rating_label == without.rating_label
    for key in ("factor_pm_score", "llm_rating_score", "llm_rating_weight", "blended_pm_score"):
        assert with_row.scores.get(key) == without.scores.get(key), key
    assert with_row.valuation_verdict.verdict == without.valuation_verdict.verdict


def test_valuation_agent_payload_carries_the_block_only_when_present(monkeypatch):
    seen: list[str] = []

    def capture(prompt, **kw):
        seen.append(prompt)
        return None  # deterministic fallback path

    monkeypatch.setattr(va.llm, "chat_json", capture)
    from app.services.fundamentals_service import get_full_financials
    fin = get_full_financials("NVDA")
    block = scorecard_context.prompt_block(_summary())
    va.run_valuation_agent(fin["profile"], fin["ratios"], None, scorecard_block=block)
    va.run_valuation_agent(fin["profile"], fin["ratios"], None)
    va.run_valuation_agent(fin["profile"], fin["ratios"], None, scorecard_block="")
    assert len(seen) == 3
    # The JSON payload key (the prompt text itself names the block in prose).
    assert '"fundamental_scorecard"' in seen[0] and "accruals_ratio" in seen[0]
    assert '"fundamental_scorecard"' not in seen[1] and '"fundamental_scorecard"' not in seen[2]
    # Without a block the context is byte-identical to the pre-Phase-6 payload.
    assert seen[1].split("Context:\n")[1] == seen[2].split("Context:\n")[1]


def test_pm_context_includes_the_block_within_budget():
    block = scorecard_context.prompt_block(_summary())
    ctx = build_pm_context(ticker="NVDA", sector="Information Technology", scorecard_block=block)
    assert "## Fundamental scorecard" in ctx
    assert "accruals_ratio" in ctx and "scenario input" in ctx
    assert "## Fundamental scorecard" not in build_pm_context(ticker="NVDA", scorecard_block="")
    # A caller cannot widen the budget past the block cap.
    wide = build_pm_context(ticker="NVDA", scorecard_block="y" * 5000)
    section = wide.split("## Fundamental scorecard")[1]
    assert section.count("y") <= scorecard_context.PROMPT_BLOCK_MAX_CHARS


def test_pm_synthesis_threads_the_scorecard_into_the_pm_context(monkeypatch):
    got: dict[str, Any] = {}

    def fake_ctx(**kw):
        got.update(kw)
        return ""

    import app.agents.pm_context as pmc
    monkeypatch.setattr(pmc, "build_pm_context", fake_ctx)
    graph._pm_synthesis({"ticker": "NVDA", "sector": "Tech"}, {}, None, scorecard=_summary())
    assert "accruals_ratio" in got["scorecard_block"]
    graph._pm_synthesis({"ticker": "NVDA", "sector": "Tech"}, {}, None)
    assert got["scorecard_block"] == ""


# ---------------------------------------------------------------------------
# 4. Review seeds through the pipeline
# ---------------------------------------------------------------------------

def test_queued_review_row_seeds_round_one_and_is_marked_reviewed(monkeypatch):
    _run_id, score_id, _ = _seed_score_row("NVDA", _summary(pct=5.0, val_pct=5.0))
    with SessionLocal() as db:
        db.add(ScorecardDisagreement(
            ticker="NVDA", memo_snapshot_id=None, scorecard_score_id=score_id, version_key=VERSION_KEY,
            as_of=AS_OF, memo_rating="Bullish", memo_rating_score=70.0, scorecard_percentile=5.0, gap=65.0,
            severity="material", dimension="overall", status="queued_review",
            seed_question="Which observed figures justify the Bullish call against a 5th-percentile rank?",
            created_at=NOW,
        ))
        db.commit()
    monkeypatch.setattr(scorecard_context, "_utcnow", lambda: NOW)
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    # Round 1 re-fired on the seed even though the PM critique (no LLM)
    # returned no questions.
    r1 = next(r for r in memo.round_findings if r.round == 1)
    assert r1.early_exit is False
    # One row seeds one question per target (plan §5.5: valuation + earnings).
    assert [q.target_agent for q in r1.pm_questions][:2] == ["valuation", "earnings"]
    assert all("5th-percentile" in q.question for q in r1.pm_questions[:2])
    assert "valuation" in r1.findings and "earnings" in r1.findings
    assert "seeded 2 review question" in r1.pm_rationale
    snap = memo_store.latest_memo("NVDA")
    with SessionLocal() as db:
        reviewed = db.query(ScorecardDisagreement).filter(
            ScorecardDisagreement.scorecard_score_id == score_id, ScorecardDisagreement.status == "reviewed",
        ).all()
        assert len(reviewed) == 1
        assert reviewed[0].resolved_at == NOW
        assert f"memo snapshot {snap.id}" in reviewed[0].note


def test_pending_seed_questions_respects_the_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard", False)
    assert scorecard_context.pending_seed_questions("NVDA") == []


def _queued_review_row(score_id: int, *, ticker: str = "NVDA", text: str = "seed?") -> None:
    with SessionLocal() as db:
        db.add(ScorecardDisagreement(
            ticker=ticker, memo_snapshot_id=None, scorecard_score_id=score_id, version_key=VERSION_KEY,
            as_of=AS_OF, memo_rating="Bullish", memo_rating_score=70.0, scorecard_percentile=5.0, gap=65.0,
            severity="material", dimension="overall", status="queued_review", seed_question=text, created_at=NOW,
        ))
        db.commit()


def test_pending_seed_questions_emit_one_per_target_and_dedupe_per_target():
    _run_id, score_id, _ = _seed_score_row("NVDA", _summary(pct=5.0, val_pct=5.0))
    _queued_review_row(score_id, text="Same question?")
    _queued_review_row(score_id, text="Same question?")   # a duplicate row must not double the re-fires
    _queued_review_row(score_id, text="Another question?")
    qs = scorecard_context.pending_seed_questions("NVDA")
    assert [(q.target_agent, q.question) for q in qs] == [
        ("valuation", "Same question?"), ("earnings", "Same question?"),
        ("valuation", "Another question?"), ("earnings", "Another question?"),
    ]
    assert all("queued for review" in q.why_it_matters for q in qs)


def test_review_rows_are_closed_even_when_the_scorecard_read_returns_none(monkeypatch):
    """Review finding: `mark_reviewed` used to sit under `memo.scorecard is
    not None`, so a review regen whose read came back None (row GC'd, DB
    hiccup) consumed the seeds but left the `queued_review` rows open — and
    every later memo re-asked the stale seed, spending LLM budget each
    time. The seeds were asked; the row is reviewed regardless."""
    _run_id, score_id, _ = _seed_score_row("NVDA", _summary(pct=5.0, val_pct=5.0))
    _queued_review_row(score_id, text="Which observed figures justify the Bullish call?")
    monkeypatch.setattr(scorecard_context, "load_for_memo", lambda *_a, **_k: None)
    monkeypatch.setattr(scorecard_context, "_utcnow", lambda: NOW)
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    assert memo.scorecard is None
    assert scorecard_context.AGENT_NAME in memo.degraded_agents      # soft: no row on file
    r1 = next(r for r in memo.round_findings if r.round == 1)
    assert r1.early_exit is False and "seeded 2 review question" in r1.pm_rationale
    with SessionLocal() as db:
        rows = db.query(ScorecardDisagreement).filter(ScorecardDisagreement.scorecard_score_id == score_id).all()
        assert [r.status for r in rows] == ["reviewed"]
        assert rows[0].resolved_at == NOW
    # Nothing left to re-fire on the next run.
    assert scorecard_context.pending_seed_questions("NVDA") == []


def test_backtest_memo_carries_the_flag_but_writes_no_disagreement_row():
    """Review finding: a reproduced historical memo (`as_of_date` set) used
    to write an `open` row like a live one, and `handle_scorecard_disagreements`
    cannot tell them apart — so a two-year-old disagreement could enqueue a
    present-day review regen. The flag stays on the memo (content); the
    finding row (what drives regen) is live-only."""
    _run_id, score_id, _ = _seed_score_row("NVDA", _summary(pct=5.0, val_pct=5.0))

    def _nvda_rows() -> int:
        with SessionLocal() as db:
            return db.query(ScorecardDisagreement).filter(ScorecardDisagreement.ticker == "NVDA").count()

    before = _nvda_rows()
    old_material, old_watch = settings.scorecard_disagreement_material, settings.scorecard_disagreement_watch
    settings.scorecard_disagreement_material = 3.0
    settings.scorecard_disagreement_watch = 1.0
    try:
        memo = graph.run_stock_memo("NVDA", force_refresh=True, as_of_date=date(2026, 7, 15))
    finally:
        settings.scorecard_disagreement_material = old_material
        settings.scorecard_disagreement_watch = old_watch
    assert memo.scorecard is not None and memo.scorecard.as_of == AS_OF
    assert memo.scorecard.disagreement is not None
    assert _nvda_rows() == before
    with SessionLocal() as db:
        assert db.query(ScorecardDisagreement).filter(ScorecardDisagreement.scorecard_score_id == score_id).count() == 0


# ---------------------------------------------------------------------------
# 5. handle_scorecard_disagreements
# ---------------------------------------------------------------------------

def _open_row(ticker: str, score_id: int, *, as_of: date = AS_OF, severity: str = "material", status: str = "open",
              created_at: datetime = NOW) -> int:
    with SessionLocal() as db:
        row = ScorecardDisagreement(
            ticker=ticker, memo_snapshot_id=None, scorecard_score_id=score_id, version_key=VERSION_KEY, as_of=as_of,
            memo_rating="Bullish", memo_rating_score=70.0, scorecard_percentile=5.0, gap=65.0, severity=severity,
            dimension="overall", status=status, seed_question="seed?", created_at=created_at,
        )
        db.add(row)
        db.commit()
        return int(row.id)


def _statuses(ticker_prefix: str = "ZSD") -> dict[int, str]:
    with SessionLocal() as db:
        return {
            r.id: r.status for r in db.query(ScorecardDisagreement)
            .filter(ScorecardDisagreement.ticker.like(f"{ticker_prefix}%")).all()
        }


def _regen_jobs(ticker: str) -> list[RegenJob]:
    with SessionLocal() as db:
        rows = db.query(RegenJob).filter(RegenJob.ticker == ticker).all()
        db.expunge_all()
        return rows


def test_handle_disagreements_is_inert_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard_disagreement_regen", False)
    _run, score_id, _ = _seed_score_row("ZSDA", _summary(pct=5.0))
    rid = _open_row("ZSDA", score_id)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(update_orchestrator, "should_auto_regen", lambda t, **k: {"should": t == "ZSDA", "reason": "stub"})
        out = update_orchestrator.handle_scorecard_disagreements()
    assert out["enabled"] is False and out["queued"] == 0
    assert _statuses()[rid] == "open"
    assert _regen_jobs("ZSDA") == []


def test_handle_disagreements_queues_material_rows_under_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard_disagreement_regen", True)
    monkeypatch.setattr(update_orchestrator, "_utcnow", lambda: NOW)
    ids: dict[str, int] = {}
    for i, t in enumerate(("ZSDB", "ZSDC", "ZSDD", "ZSDE")):
        _run, score_id, _ = _seed_score_row(t, _summary(pct=5.0))
        ids[t] = _open_row(t, score_id, created_at=datetime(2026, 7, 1, 0, 0, i))
    _run, watch_score, _ = _seed_score_row("ZSDW", _summary(pct=40.0))
    watch_id = _open_row("ZSDW", watch_score, severity="watch")
    # Pin the regen queue's clock too so the enqueue waypoint counts as "today".
    from app.services import regen_worker
    monkeypatch.setattr(regen_worker, "_utcnow", lambda: NOW)

    # Rows other tests own (the module fixtures' NVDA row) must not spend
    # this test's budget: gate them off and count only the ZSD outcomes.
    monkeypatch.setattr(
        update_orchestrator, "should_auto_regen",
        lambda t, **k: {"should": t.startswith("ZSD"), "reason": "stub"},
    )
    out = update_orchestrator.handle_scorecard_disagreements(cap=2)
    assert out["enabled"] is True and out["cap"] == 2
    assert out["queued"] == 2 and out["open_material"] >= 4
    st = _statuses()
    assert st[ids["ZSDB"]] == "queued_review" and st[ids["ZSDC"]] == "queued_review"
    assert st[ids["ZSDD"]] == "open" and st[ids["ZSDE"]] == "open"
    assert st[watch_id] == "open"  # watch rows never trigger spend
    jobs = _regen_jobs("ZSDB")
    assert len(jobs) == 1 and jobs[0].progress[0]["source"] == "scorecard_disagreement"

    # Second call the same day: the cap is already used up.
    again = update_orchestrator.handle_scorecard_disagreements(cap=2)
    assert again["used_today"] == 2 and again["queued"] == 0
    assert _statuses()[ids["ZSDD"]] == "open"


def test_handle_disagreements_respects_the_gate_and_never_requeues_a_reviewed_key(monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard_disagreement_regen", True)
    monkeypatch.setattr(update_orchestrator, "_utcnow", lambda: NOW)
    from app.services import regen_worker
    monkeypatch.setattr(regen_worker, "_utcnow", lambda: NOW)
    _run, gated_score, _ = _seed_score_row("ZSDG", _summary(pct=5.0))
    gated = _open_row("ZSDG", gated_score)
    _run, rev_score, _ = _seed_score_row("ZSDR", _summary(pct=5.0))
    _open_row("ZSDR", rev_score, status="reviewed")
    reopened = _open_row("ZSDR", rev_score)

    monkeypatch.setattr(
        update_orchestrator, "should_auto_regen",
        lambda t, **k: {"should": t == "ZSDR", "reason": "stale_memo_60d_old" if t == "ZSDG" else "stub"},
    )
    out = update_orchestrator.handle_scorecard_disagreements(cap=5)
    assert out["queued"] == 0
    assert out["skipped_gate"] >= 1 and out["skipped_reviewed"] == 1
    st = _statuses()
    assert st[gated] == "open" and st[reopened] == "open"
    with SessionLocal() as db:
        note = db.get(ScorecardDisagreement, gated).note
    assert "stale_memo_60d_old" in note
    assert _regen_jobs("ZSDG") == [] and _regen_jobs("ZSDR") == []


def test_memo_snapshot_linkage_uses_the_returned_snapshot():
    """`_persist_memo_snapshot` returns the row it wrote (Phase 6), which
    is what keys `scorecard_disagreements.memo_snapshot_id`."""
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    snap = graph._persist_memo_snapshot(memo)
    assert isinstance(snap, MemoSnapshot)
    assert snap.id is not None and snap.ticker == "NVDA"
    latest = memo_store.latest_memo("NVDA")
    assert latest is not None and latest.id == snap.id
