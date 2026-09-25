"""W7 (S19): the learned-priors block is bounded, audited, framed as
hypotheses, and reaches nobody outside a live memo run.

Owner decision 9: memory is priors that update with evidence and must never
over-index. So the block has a fixed character budget and per-kind caps, is
packed from whole lines only, says on its face that it is provisional and
not a source, carries the evidence count on every lesson, and is shown only
after its audit row is written. Off does no I/O at all; shadow audits what
WOULD be shown and shows nothing; backtests, demo/no-LLM runs and anything
outside a memo run (chat's `ask_sector`) get nothing and log nothing.
"""
from __future__ import annotations

import itertools
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event

from app.agents.llm import llm_call_context
from app.agents.safe_runner import DegradationLog
from app.config import settings
from app.database import engine as global_engine
from app.learning import context, control, ledger
from app.models import LearningControlEvent, LearningEvidence, LearningItem, LearningRender
from app.services.data_service import as_of_context
from app.tests.learning_helpers import add_company, classify, learning_db

NOW = datetime(2026, 11, 20, 12, 0)
TICKER = "CTXA"
GROUP = "4510"
_REFS = itertools.count(1)


@pytest.fixture
def env(tmp_path, monkeypatch):
    sessions, engine = learning_db(tmp_path, monkeypatch, context)
    monkeypatch.setattr(settings, "learning_mode_max", "inject")
    monkeypatch.setattr(context, "_live_generation", lambda: True)
    with sessions() as s:
        add_company(s, TICKER, "Technology")
        classify(s, TICKER, group=GROUP)
    control._cache_clear()
    yield sessions, engine
    control._cache_clear()
    engine.dispose()


def _mode(sessions, mode: str) -> None:
    with sessions() as s:
        s.add(LearningControlEvent(mode=mode, actor="admin", reason="test", gates={}, created_at=NOW))
        s.commit()
    control._cache_clear()


def _lesson(sessions, text: str, *, scope_type: str = "company", scope_key: str = TICKER,
            verdicts: tuple[str, ...] = (), source_date: date = date(2026, 6, 1),
            status: str = "active") -> int:
    with sessions() as s:
        item = ledger._new_item(
            s, kind="lesson", scope_type=scope_type, scope_key=scope_key, text=text,
            detail="FULL POSTMORTEM NARRATIVE (audit only)", origin_kind="postmortem",
            origin_ref=f"pm-{next(_REFS)}", origin_ticker=TICKER, source_date=source_date, now=NOW,
        )
        item.status = status
        s.flush()
        for n, verdict in enumerate(verdicts):
            s.add(LearningEvidence(
                item_id=item.id, verdict=verdict, ticker=TICKER, horizon_days=90,
                independence_key=f"{scope_key}:90:{n}", observed_at=NOW,
            ))
        s.commit()
        return int(item.id)


def _observation(sessions, text: str, *, filed: date = date(2026, 10, 30), scope_type: str = "company",
                 scope_key: str = TICKER, expires_at: datetime | None = None, status: str = "active") -> int:
    with sessions() as s:
        item = ledger._new_item(
            s, kind="observation", scope_type=scope_type, scope_key=scope_key, text=text,
            origin_kind="filing_delta", origin_ref=f"acc-{next(_REFS)}", origin_ticker=TICKER,
            source_date=filed, expires_at=expires_at or NOW + timedelta(days=300), now=NOW,
        )
        item.status = status
        s.commit()
        return int(item.id)


def _renders(sessions) -> list[LearningRender]:
    with sessions() as s:
        return list(s.query(LearningRender).order_by(LearningRender.id).all())


@contextmanager
def _memo_run(run_id: str = "run-ctx-1"):
    with DegradationLog().activate(), llm_call_context(agent_name="test", run_id=run_id):
        yield


@contextmanager
def _count_statements(*engines: Any):
    seen: list[str] = []

    def hook(conn, cursor, statement, *a):
        seen.append(statement)

    for e in engines:
        event.listen(e, "before_cursor_execute", hook)
    try:
        yield seen
    finally:
        for e in engines:
            event.remove(e, "before_cursor_execute", hook)


def _long(tag: str, n: int = 200) -> str:
    base = f"When {tag} margins widen while capex falls, expect the stock to outperform the benchmark"
    return (base + " " + "x" * n)[:230].rstrip() + "."


def test_budgets_match_the_promotion_gate():
    """G3 checks audited renders against control's copy of the caps."""
    assert {k: b.max_chars for k, b in context.BUDGETS.items()} == control.RENDER_BUDGETS


def test_off_returns_empty_without_db_access(env, monkeypatch):
    sessions, engine = env
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    monkeypatch.setattr(settings, "learning_mode_max", "off")
    with _memo_run(), _count_statements(engine, global_engine) as seen:
        for consumer in context.BUDGETS:
            assert context.render_for(consumer, ticker=TICKER, sector="Technology") == context.RenderOutcome("off", "")
        assert context.link_run("run-ctx-1", 5) == 0
    assert seen == []
    assert _renders(sessions) == []


def test_shadow_logs_row_returns_empty(env):
    sessions, _ = env
    lid = _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    oid = _observation(sessions, "What's new in 10-Q filed 2026-10-30: bundles lifted retention.")
    with _memo_run("run-shadow"):
        out = context.render_for("pm_memo", ticker=TICKER, sector="Technology", now=NOW)
    assert out == context.RenderOutcome("shadow", "")
    (row,) = _renders(sessions)
    assert row.mode == "shadow" and row.consumer == "pm_memo" and row.run_id == "run-shadow"
    assert row.ticker == TICKER and row.error_type is None and row.memo_snapshot_id is None
    assert [i["ref"] for i in row.items] == [f"L-{lid}", f"O-{oid}"]
    # What WOULD have been shown is measured, within budget.
    assert 0 < row.chars <= context.BUDGETS["pm_memo"].max_chars


def test_inject_respects_char_and_count_caps_whole_lines(env):
    sessions, _ = env
    _mode(sessions, "inject")
    supported = [_lesson(sessions, _long(f"s{i}"), verdicts=("held",) * 5) for i in range(3)]
    untested = [_lesson(sessions, _long(f"u{i}"), source_date=date(2026, 7, i + 1)) for i in range(5)]
    weakened = [_lesson(sessions, _long(f"w{i}"), verdicts=("failed", "failed")) for i in range(2)]
    group = _lesson(sessions, _long("g"), scope_type="industry_group", scope_key=GROUP)
    obs = [_observation(sessions, f"What's new in 10-Q filed 2026-10-{i + 1:02d}: " + "y" * 250,
                        filed=date(2026, 10, i + 1)) for i in range(4)]
    texts: dict[str, str] = {}
    with sessions() as s:
        for item in s.query(LearningItem).all():
            texts[ledger.ref(item.id, item.kind)] = item.text

    for consumer, budget in context.BUDGETS.items():
        with _memo_run(f"run-caps-{consumer}"):
            out = context.render_for(consumer, ticker=TICKER, sector="Technology", now=NOW)
        assert out.mode == "inject"
        assert 0 < len(out.text) <= budget.max_chars, consumer
        lines = [ln for ln in out.text.split("\n") if ln.startswith("- [")]
        refs = [re.match(r"- \[([LO]-\d+)\]", ln).group(1) for ln in lines]  # type: ignore[union-attr]
        lessons = [r for r in refs if r.startswith("L-")]
        observations = [r for r in refs if r.startswith("O-")]
        assert len(lessons) <= budget.max_lessons and len(observations) <= budget.max_observations
        assert len([r for r in lessons if int(r[2:]) in untested]) <= budget.max_untested
        assert len([r for r in lessons if int(r[2:]) in weakened]) <= 1
        for ref, line in zip(refs, lines, strict=True):
            assert line.endswith(texts[ref]), f"{consumer}: {ref} was cut"   # whole lines only
        row = _renders(sessions)[-1]
        assert row.chars == len(out.text) and [i["ref"] for i in row.items] == refs
        assert {d["reason"] for d in row.dropped} <= {"budget", "cap_untested", "cap_weakened", "cap_kind"}
        if consumer == "pm_memo":
            # Stance first: every supported lesson precedes every untested one.
            assert lessons[: len(supported)] == [f"L-{i}" for i in sorted(supported, reverse=True)][: len(lessons)]
            assert out.text.endswith(context.PM_FOOTER)
            assert any(d["reason"] == "budget" for d in row.dropped)
        if consumer == "critic":
            assert f"L-{group}" not in refs            # company scope only
    assert obs  # seeded


def test_evidence_counts_ride_on_every_lesson(env):
    sessions, _ = env
    _mode(sessions, "inject")
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.",
            verdicts=("held", "held", "held", "held", "failed"))
    _lesson(sessions, "When capex rises, expect the stock to underperform the benchmark over 90 days.",
            verdicts=("failed", "failed"))
    _lesson(sessions, "When bundles ship, expect the stock to outperform the benchmark over 90 days.")
    with _memo_run():
        text = context.render_for("pm_memo", ticker=TICKER, now=NOW).text
    assert "(this company; supported 4 of 5 later outcomes)" in text
    assert "(this company; has not held: 0 of 2)" in text
    assert "(this company; untested hypothesis)" in text


def test_retired_suppressed_superseded_and_expired_never_selected(env):
    sessions, _ = env
    _mode(sessions, "inject")
    for status in ("retired", "suppressed", "superseded"):
        _lesson(sessions, f"When {status} happens, expect the stock to outperform the benchmark over 90 days.",
                status=status)
    _observation(sessions, "What's new in 10-K filed 2025-01-01: stale.", expires_at=NOW - timedelta(days=1))
    _lesson(sessions, "When a future fact is known, expect the stock to outperform the benchmark over 90 days.",
            source_date=date(2027, 1, 1))
    with _memo_run():
        out = context.render_for("pm_memo", ticker=TICKER, now=NOW)
    assert out == context.RenderOutcome("inject", "")
    assert _renders(sessions)[-1].items == []


def test_backtest_and_demo_runs_get_nothing_and_log_nothing(env, monkeypatch):
    sessions, engine = env
    _mode(sessions, "inject")
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    with _count_statements(engine) as seen:
        # A backtest (as_of_date set) inside a memo run.
        with _memo_run(), as_of_context(date(2026, 3, 1)):
            assert context.render_for("pm_memo", ticker=TICKER, now=NOW) == context.RenderOutcome("off", "")
        # Outside any memo run (chat, the orchestrator, a direct call).
        assert context.render_for("sector", ticker=TICKER, now=NOW) == context.RenderOutcome("off", "")
        # A demo / no-LLM run.
        monkeypatch.setattr(context, "_live_generation", lambda: False)
        with _memo_run():
            assert context.render_for("critic", ticker=TICKER, now=NOW) == context.RenderOutcome("off", "")
    assert seen == []
    assert _renders(sessions) == []


def test_live_generation_follows_llm_enabled(monkeypatch):
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    assert context._live_generation() is False


def test_audit_insert_failure_injects_nothing(env):
    sessions, engine = env
    _mode(sessions, "inject")
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")

    def refuse(conn, cursor, statement, *a):
        if statement.lstrip().upper().startswith("INSERT INTO LEARNING_RENDERS"):
            raise RuntimeError("audit table unavailable")

    event.listen(engine, "before_cursor_execute", refuse)
    try:
        with _memo_run():
            out = context.render_for("pm_memo", ticker=TICKER, now=NOW)
    finally:
        event.remove(engine, "before_cursor_execute", refuse)
    assert out == context.RenderOutcome("inject", "")
    assert _renders(sessions) == []
    # Control: the same render with a working audit table does inject.
    with _memo_run():
        assert context.render_for("pm_memo", ticker=TICKER, now=NOW).text


def test_selection_failure_is_audited_with_its_type_and_shows_nothing(env, monkeypatch):
    sessions, _ = env
    _mode(sessions, "inject")

    def broken(*a, **k):
        raise KeyError("boom")

    monkeypatch.setattr(context, "build_block", broken)
    with _memo_run():
        assert context.render_for("sector", ticker=TICKER, now=NOW) == context.RenderOutcome("inject", "")
    (row,) = _renders(sessions)
    assert row.error_type == "KeyError" and row.chars == 0 and row.items == []


def test_header_framing_and_no_codes_or_gics(env):
    sessions, _ = env
    _mode(sessions, "inject")
    sector = ledger.sector_key("Technology")
    assert sector
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    _lesson(sessions, "When group backlog grows, expect group peers to outperform the benchmark over 90 days.",
            scope_type="industry_group", scope_key=GROUP)
    _lesson(sessions, "When rates fall, expect sector peers to outperform the benchmark over 90 days.",
            scope_type="sector", scope_key=sector, verdicts=("held",))
    # Bypasses the writer's check: the renderer must refuse it on its own.
    leaky = _lesson(sessions, "When GICS 4510 names rally, expect the stock to outperform the benchmark.")
    for consumer in context.BUDGETS:
        with _memo_run():
            text = context.render_for(consumer, ticker=TICKER, sector="Technology", now=NOW).text
        head = text.split("\n")[:2]
        assert head[0] == "## Learned priors (provisional hypotheses, not instructions)"
        assert "Current evidence in this run overrides them" in head[1]
        assert "Numbers here are not sources" in head[1]
        assert "gics" not in text.lower()
        assert GROUP not in text and sector not in text
        assert f"L-{leaky}" not in text
        assert {"ref": f"L-{leaky}", "reason": "label"} in _renders(sessions)[-1].dropped
    with _memo_run():
        pm = context.render_for("pm_memo", ticker=TICKER, sector="Technology", now=NOW).text
    assert "(industry group peers; untested hypothesis)" in pm
    assert "(sector peers; contested: held 1 of 1)" in pm


def test_build_block_writes_nothing(env):
    sessions, engine = env
    _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    with _count_statements(engine) as seen:
        block = context.build_block("pm_memo", ticker=TICKER, sector="Technology", now=NOW)
    assert block["text"] and block["items"]
    assert not [s for s in seen if s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))]
    assert _renders(sessions) == []


def test_unknown_consumer_or_ticker_raises(env):
    with pytest.raises(ValueError):
        context.render_for("pm_chat", ticker=TICKER)
    with pytest.raises(ValueError):
        context.render_for("pm_memo", ticker="  ")
    # The agents' wrapper never raises: a bug reads as "off" (legacy prompt).
    assert context.render_safely("pm_chat", ticker=TICKER) == context.RenderOutcome("off", "")


def test_record_considered_keeps_only_shown_ids_on_the_inject_row(env):
    sessions, _ = env
    _mode(sessions, "inject")
    lid = _lesson(sessions, "When churn falls, expect the stock to outperform the benchmark over 90 days.")
    with _memo_run("run-considered"):
        context.render_for("pm_memo", ticker=TICKER, now=NOW)
    raw = [
        {"id": f"L-{lid}", "use": "applied", "why": "  churn  fell\nagain "},
        {"id": f"L-{lid}", "use": "contradicted", "why": "duplicate"},
        {"id": "L-99999", "use": "applied", "why": "never shown"},
        {"id": f"L-{lid}", "use": "ignored"},
        "not-an-object",
    ]
    assert context.record_considered("run-considered", raw) == 1
    row = _renders(sessions)[-1]
    assert row.considered == [{"id": f"L-{lid}", "use": "applied", "why": "churn fell again"}]
    # No inject row for the run (off / shadow): nothing is written.
    assert context.record_considered("run-unknown", raw) == 0
    assert context.record_considered("run-considered", None) == 0


def _shown(block: dict[str, Any]) -> list[str]:
    return [i["ref"] for i in block["items"]]


def _dropped(block: dict[str, Any], reason: str) -> list[str]:
    return [d["ref"] for d in block["dropped"] if d["reason"] == reason]


def test_count_caps_bind_before_the_character_budget(env):
    """Owner decision 9's over-indexing guard is the per-kind caps, not only
    the character budget. Short lines here, so every cap binds while the
    budget never does: at most `max_untested` unproven hypotheses, at most
    one weakened lesson, at most `max_observations` observations (newest
    first), and at most `max_lessons` lessons in all."""
    sessions, _ = env
    untested = [_lesson(sessions, f"When u{i} rises, expect outperformance.", source_date=date(2026, 7, i + 1))
                for i in range(5)]
    weakened = [_lesson(sessions, f"When w{i} rises, expect underperformance.", verdicts=("failed", "failed"))
                for i in range(3)]
    obs = [_observation(sessions, f"10-Q filed 2026-10-{i + 1:02d}: item {i}.", filed=date(2026, 10, i + 1))
           for i in range(4)]
    u = [f"L-{i}" for i in reversed(untested)]      # newest information first
    w = [f"L-{i}" for i in reversed(weakened)]      # same date: newest id first
    o = [f"O-{i}" for i in reversed(obs)]

    for consumer, budget in context.BUDGETS.items():
        block = context.build_block(consumer, ticker=TICKER, sector="Technology", now=NOW)
        n_u, n_o = budget.max_untested, budget.max_observations
        assert _shown(block) == u[:n_u] + w[:1] + o[:n_o], consumer
        assert _dropped(block, "cap_untested") == u[n_u:], consumer
        assert _dropped(block, "cap_weakened") == w[1:], consumer
        assert _dropped(block, "cap_kind") == o[n_o:], consumer
        assert _dropped(block, "budget") == [], consumer   # the caps did the work
    assert [b.max_untested for b in context.BUDGETS.values()] == [2, 2, 1, 1]
    assert [b.max_observations for b in context.BUDGETS.values()] == [2, 1, 1, 2]

    # Enough supported lessons to fill every consumer: exactly max_lessons
    # are shown, and nothing weaker gets in behind them.
    supported = [_lesson(sessions, f"When s{i} rises, expect outperformance.", verdicts=("held",) * 5)
                 for i in range(5)]
    s = [f"L-{i}" for i in reversed(supported)]
    for consumer, budget in context.BUDGETS.items():
        block = context.build_block(consumer, ticker=TICKER, sector="Technology", now=NOW)
        lessons = [r for r in _shown(block) if r.startswith("L-")]
        assert lessons == s[: budget.max_lessons], consumer
        assert set(_dropped(block, "cap_kind")) >= set(s[budget.max_lessons:] + u + w[:1]), consumer
        assert _dropped(block, "budget") == [], consumer
    assert [b.max_lessons for b in context.BUDGETS.values()] == [4, 3, 2, 2]


def test_consumer_scopes_decide_which_peers_a_prompt_sees(env):
    """The critic reviews this company only; the industry-group analyst sees
    its group and the company, never sector peers. Here scope is the only
    thing that can exclude a line: every lesson is supported and all of
    them fit every cap and budget."""
    sessions, _ = env
    sector = ledger.sector_key("Technology")
    assert sector
    group = _lesson(sessions, "When group backlog grows, expect peers to outperform.",
                    scope_type="industry_group", scope_key=GROUP, verdicts=("held",) * 5)
    sect = _lesson(sessions, "When rates fall, expect sector peers to outperform.",
                   scope_type="sector", scope_key=sector, verdicts=("held",) * 5)
    own = _observation(sessions, "10-Q filed 2026-10-30: bundles lifted retention.")
    peer = _observation(sessions, "Sector filing 2026-10-01: pricing firmed.", filed=date(2026, 10, 1),
                        scope_type="sector", scope_key=sector)
    expected = {
        "pm_memo": [f"L-{group}", f"L-{sect}", f"O-{own}", f"O-{peer}"],
        "sector": [f"L-{group}", f"L-{sect}", f"O-{own}"],
        "industry_group": [f"L-{group}", f"O-{own}"],
        "critic": [f"O-{own}"],
    }
    for consumer, refs in expected.items():
        block = context.build_block(consumer, ticker=TICKER, sector="Technology", now=NOW)
        assert _shown(block) == refs, consumer
        assert _dropped(block, "budget") == [], consumer
        if consumer in ("industry_group", "critic"):
            assert "sector peers" not in block["text"], consumer
        if consumer == "critic":
            assert "peers" not in block["text"]
            assert block["dropped"] == []      # never selected at all, not capped out
    assert "(sector peers; supported 5 of 5 later outcomes)" in context.build_block(
        "sector", ticker=TICKER, sector="Technology", now=NOW)["text"]
