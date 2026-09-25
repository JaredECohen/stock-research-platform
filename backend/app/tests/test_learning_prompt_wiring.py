"""W7 (S19): where the learned-priors block plugs into the memo's prompts.

The rollback property is the one that matters most: with learning off, or in
shadow (the mode production deploys in), every prompt the four consumers
send — PM synthesis, sector analyst, Industry Group analyst, critic — is
byte-identical to the legacy prompt, even though shadow renders and audits a
block. Only inject changes a prompt, and then only by replacing the legacy
memory blocks: the PM's regime and specialist-reliability blocks stay
exactly as they were (decision 7(d): calibration is frozen). Chat's
`ask_sector` re-fires the sector analyst outside a memo run and never gets
a prior or writes an audit row. The PM's `priors_considered` is kept only
for ids it was shown, and `_persist` links every render of a live run to
its snapshot (never a backtest's).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import app.cache as app_cache
from app.agents import critic_agent, graph, pm_context, sector_agents
from app.agents import industry_analysts as ia
from app.agents.llm import llm_call_context
from app.agents.memo_context import MemoInputs
from app.agents.safe_runner import DegradationLog
from app.config import settings
from app.learning import context, control, ledger
from app.memory import CompanyMemory, SectorMemory
from app.models import LearningControlEvent, LearningRender
from app.services import calibration_service, influence_feedback
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.tests.factories import make_memo
from app.tests.gating_helpers import seed_demo_universe
from app.tests.learning_helpers import add_company, classify, learning_db

NOW = datetime.utcnow()
TICKER = "MSFT"
LEGACY = "LEGACY-FILE-MEMORY-SENTINEL"
PROFILE = {"ticker": TICKER, "company_name": "Microsoft", "sector": "Technology"}


@pytest.fixture(scope="module")
def classification():
    seed_demo_universe()
    assert reg.ensure_taxonomy(activate=True) is not None
    ic.classify_all(tickers=[TICKER])
    row = ic.current_for([TICKER])[TICKER]
    assert row.get("industry_group_code")
    return row


class _Mem:
    """A legacy file-memory stand-in whose text is unmistakable."""

    def as_prompt_context(self, **kw: Any) -> str:
        return f"{LEGACY} company"

    def as_prompt_context_for(self, *a: Any, **kw: Any) -> str:
        return f"{LEGACY} sector"


@pytest.fixture
def env(tmp_path, monkeypatch, classification):
    ia.clear_cache()
    sessions, engine = learning_db(tmp_path, monkeypatch, context)
    monkeypatch.setattr(context, "_live_generation", lambda: True)
    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    monkeypatch.setattr(settings, "enable_agent_critic", True)
    monkeypatch.setattr(settings, "learning_mode_max", "off")
    monkeypatch.setattr(CompanyMemory, "for_ticker", classmethod(lambda cls, t: _Mem()))
    monkeypatch.setattr(SectorMemory, "for_sector", classmethod(lambda cls, s: _Mem()))
    with sessions() as s:
        add_company(s, TICKER, "Technology")
        classify(s, TICKER, group=str(classification["industry_group_code"]))
        ledger._new_item(s, kind="lesson", scope_type="company", scope_key=TICKER,
                         text="When cloud bookings accelerate, expect the stock to outperform the benchmark over 90 days.",
                         detail="audit-only narrative", origin_kind="postmortem", origin_ref="pm-1",
                         origin_ticker=TICKER, source_date=date(2026, 6, 1), now=NOW)
        ledger._new_item(s, kind="lesson", scope_type="industry_group",
                         scope_key=str(classification["industry_group_code"]),
                         text="When seat growth slows, expect group peers to underperform the benchmark over 90 days.",
                         origin_kind="postmortem_sector", origin_ref="pm-1", origin_ticker=TICKER,
                         source_date=date(2026, 6, 1), now=NOW)
        ledger._new_item(s, kind="observation", scope_type="company", scope_key=TICKER,
                         text="What's new in 10-Q filed 2026-07-30: commercial bookings up on large renewals.",
                         origin_kind="filing_delta", origin_ref="acc-1", origin_ticker=TICKER,
                         source_date=date(2026, 7, 30), expires_at=NOW + timedelta(days=300), now=NOW)
        s.commit()
    control._cache_clear()
    yield sessions, classification
    control._cache_clear()
    ia.clear_cache()
    engine.dispose()


def _set_mode(monkeypatch, sessions, mode: str) -> None:
    """off: the env ceiling; shadow: the DB default; inject: a mode event."""
    monkeypatch.setattr(settings, "learning_mode_max", "off" if mode == "off" else "inject")
    if mode == "inject":
        with sessions() as s:
            s.add(LearningControlEvent(mode="inject", actor="admin", reason="test", gates={},
                                       created_at=datetime.utcnow()))
            s.commit()
    control._cache_clear()
    assert control.effective_mode() == mode


def _renders(sessions, **where: Any) -> list[LearningRender]:
    with sessions() as s:
        return list(s.query(LearningRender).filter_by(**where).order_by(LearningRender.id).all())


@contextmanager
def _memo_run(run_id: str):
    with DegradationLog().activate(), llm_call_context(agent_name="test", run_id=run_id):
        yield


def _capture(monkeypatch, module: Any, reply: Any = None) -> list[str]:
    seen: list[str] = []

    def fake(prompt, *a, **kw):
        seen.append(prompt)
        return reply

    monkeypatch.setattr(module.llm, "chat_json", fake)
    return seen


# --- the four consumers, each captured in one mode -----------------------------

def _pm_context(run_id: str) -> str:
    with _memo_run(run_id):
        return pm_context.build_pm_context(ticker=TICKER, sector="Technology", profile=PROFILE,
                                           learning_consumer="pm_memo")


def _sector_prompt(monkeypatch, run_id: str, *, in_run: bool = True, question: str | None = None) -> str:
    seen = _capture(monkeypatch, sector_agents)
    if in_run:
        with _memo_run(run_id):
            sector_agents.run_sector_agent(dict(PROFILE), {}, prior_round_critique=question)
    else:
        sector_agents.run_sector_agent(dict(PROFILE), {}, prior_round_critique=question)
    return seen[-1]


def _group_prompt(monkeypatch, run_id: str, row: dict) -> str:
    seen = _capture(monkeypatch, ia)
    with _memo_run(run_id):
        ia.run_industry_group_agent(dict(PROFILE), {}, classification=row)
    return seen[-1]


def _critic_prompt(monkeypatch, run_id: str) -> str:
    seen = _capture(monkeypatch, critic_agent)
    with _memo_run(run_id):
        critic_agent.run_critic({"ticker": TICKER, "sector": "Technology", "rating_label": "Neutral"})
    return seen[-1]


def _all_prompts(monkeypatch, row: dict, tag: str) -> dict[str, str]:
    """Each consumer's prompt, captured in a run whose id is `{tag}-{consumer}`."""
    return {
        "pm_memo": _pm_context(f"{tag}-pm_memo"),
        "sector": _sector_prompt(monkeypatch, f"{tag}-sector"),
        "industry_group": _group_prompt(monkeypatch, f"{tag}-industry_group", row),
        "critic": _critic_prompt(monkeypatch, f"{tag}-critic"),
    }


def test_off_and_shadow_prompts_byte_identical_to_legacy(env, monkeypatch):
    sessions, row = env
    # Legacy: the PM context as chat builds it (no learning consumer at all),
    # and the three agents with learning off.
    legacy_pm = pm_context.build_pm_context(ticker=TICKER, sector="Technology", profile=PROFILE)
    _set_mode(monkeypatch, sessions, "off")
    off = _all_prompts(monkeypatch, row, "off")
    assert off["pm_memo"] == legacy_pm
    assert _renders(sessions) == []                        # off writes nothing

    _set_mode(monkeypatch, sessions, "shadow")
    shadow = _all_prompts(monkeypatch, row, "shadow")
    for consumer in context.BUDGETS:
        assert shadow[consumer] == off[consumer], consumer
        assert LEGACY in off[consumer], consumer            # the legacy branch really ran
        # ...while shadow really rendered and audited what it would show.
        (audit,) = _renders(sessions, run_id=f"shadow-{consumer}")
        assert audit.mode == "shadow" and audit.consumer == consumer and audit.items

    # Control: inject does change every one of them, replacing legacy memory.
    _set_mode(monkeypatch, sessions, "inject")
    inject = _all_prompts(monkeypatch, row, "inject")
    for consumer in context.BUDGETS:
        assert context.HEADER in inject[consumer], consumer
        assert LEGACY not in inject[consumer], consumer
        assert "audit-only narrative" not in inject[consumer]


def test_inject_header_reworded_and_critic_reframed(env, monkeypatch):
    sessions, row = env
    _set_mode(monkeypatch, sessions, "inject")
    pm = _pm_context("hdr-pm")
    assert pm.startswith(pm_context.PM_CONTEXT_HEADER_INJECT)
    assert "Let them shape the synthesis" not in pm
    critic = _critic_prompt(monkeypatch, "hdr-critic")
    assert critic_agent.CRITIC_PRIORS_INSTRUCTION in critic
    assert "CONTRADICTS prior recorded lessons" not in critic
    sector = _sector_prompt(monkeypatch, "hdr-sector")
    assert "Prior context from long-term memory" not in sector
    group = _group_prompt(monkeypatch, "hdr-group", row)
    assert "Prior context from the group's long-term memory" not in group


def test_inject_keeps_regime_and_reliability_blocks_identical(env, monkeypatch):
    """7(d): same thresholds (n=4 still renders), same wording, outside the
    learned block's budget, in every mode."""
    sessions, _ = env
    reliability = "## Specialist reliability\n\nDiscount the Sector Analyst's pull (sentinel)."
    monkeypatch.setattr(influence_feedback, "reliability_prompt_block", lambda **kw: reliability)
    monkeypatch.setattr(app_cache, "cache_get",
                        lambda *a, **k: SimpleNamespace(payload={"regime": "Expansion"}))
    monkeypatch.setattr(calibration_service, "regime_conditional_accuracy",
                        lambda **kw: {"regimes": {"expansion": {"n": 4, "accuracy": 0.5, "mean_alpha": 0.012}}})

    def blocks(text: str) -> list[str]:
        return [b for b in text.split("\n\n---\n\n")
                if b.startswith(("## Specialist reliability", "## PM track-record under current regime"))]

    _set_mode(monkeypatch, sessions, "off")
    off = blocks(_pm_context("td-off"))
    _set_mode(monkeypatch, sessions, "inject")
    injected = _pm_context("td-inject")
    assert len(off) == 2
    assert blocks(injected) == off
    learned = next(b for b in injected.split("\n\n---\n\n") if b.startswith(context.HEADER))
    assert "Expansion" not in learned and "sentinel" not in learned
    assert len(learned) <= context.BUDGETS["pm_memo"].max_chars


def test_ask_sector_gets_no_priors_and_writes_no_render(env, monkeypatch):
    """Chat's `ask_sector` calls `run_sector_agent(profile, ratios,
    prior_round_critique=question)` outside any memo run."""
    sessions, _ = env
    _set_mode(monkeypatch, sessions, "off")
    legacy = _sector_prompt(monkeypatch, "unused", in_run=False, question="What if rates fall 100bps?")
    _set_mode(monkeypatch, sessions, "inject")
    chat = _sector_prompt(monkeypatch, "unused", in_run=False, question="What if rates fall 100bps?")
    assert chat == legacy
    assert context.HEADER not in chat
    assert _renders(sessions) == []
    # Chat's PM context never asks for priors either (no learning consumer).
    with _memo_run("chat-pm"):
        assert context.HEADER not in pm_context.build_pm_context(ticker=TICKER, sector="Technology")
    assert _renders(sessions) == []


def test_priors_considered_recorded_and_restricted_to_rendered_ids(env, monkeypatch):
    sessions, _ = env
    _set_mode(monkeypatch, sessions, "inject")
    shown: list[str] = []

    def fake(prompt, *a, **kw):
        shown.extend(ref for ref in ("L-1", "L-2", "O-3") if f"[{ref}]" in prompt)
        return {
            "rating_label": "Neutral", "confidence_score": 55, "final_pm_view": "v", "one_sentence_thesis": "t",
            "priors_considered": [
                {"id": "L-1", "use": "applied", "why": "bookings did accelerate"},
                {"id": "L-404", "use": "applied", "why": "never shown"},
                {"id": "L-2", "use": "overrode", "why": "bad use"},
            ],
        }

    monkeypatch.setattr(graph.llm, "chat_json", fake)
    with _memo_run("pm-considered"):
        synth = graph._pm_synthesis(dict(PROFILE), {}, None)
    assert synth["rating_label"] == "Neutral"
    assert "L-1" in shown
    (row,) = _renders(sessions, run_id="pm-considered", consumer="pm_memo")
    assert row.mode == "inject"
    assert row.considered == [{"id": "L-1", "use": "applied", "why": "bookings did accelerate"}]

    # Shadow: the PM was shown nothing, so nothing it claims is recorded.
    with sessions() as s:
        s.add(LearningControlEvent(mode="shadow", actor="admin", reason="test", gates={},
                                   created_at=datetime.utcnow() + timedelta(seconds=1)))
        s.commit()
    control._cache_clear()
    assert control.effective_mode() == "shadow"
    with _memo_run("pm-shadow"):
        graph._pm_synthesis(dict(PROFILE), {}, None)
    (shadow_row,) = _renders(sessions, run_id="pm-shadow", consumer="pm_memo")
    assert shadow_row.mode == "shadow" and shadow_row.considered is None


def _inputs(run_id: str, *, as_of: date | None = None) -> MemoInputs:
    return MemoInputs(
        ticker=TICKER, run_id=run_id, scenario="soft_landing", force_refresh=False, as_of_date=as_of,
        fin={}, profile=dict(PROFILE), ratios={}, earnings={}, transcript=None, filings=[], dcf=None,
        comps=None, degradation=DegradationLog(),
    )


def test_link_run_attaches_snapshot_and_skips_backtests(env, monkeypatch):
    sessions, row = env
    _set_mode(monkeypatch, sessions, "shadow")
    for run_id in ("live-run", "backtest-run"):
        with _memo_run(run_id):
            context.render_for("pm_memo", ticker=TICKER, sector="Technology")
            context.render_for("critic", ticker=TICKER, sector="Technology")
    ids = iter((501, 502))
    monkeypatch.setattr(graph, "_persist_memo_snapshot",
                        lambda memo, as_of_date=None: SimpleNamespace(id=next(ids)))
    memo = make_memo(ticker=TICKER)
    graph._persist(memo, _inputs("live-run"))
    graph._persist(memo, _inputs("backtest-run", as_of=date(2026, 3, 1)))

    assert {r.memo_snapshot_id for r in _renders(sessions, run_id="live-run")} == {501}
    assert {r.memo_snapshot_id for r in _renders(sessions, run_id="backtest-run")} == {None}
    # Idempotent: a second link of the same run changes nothing.
    assert context.link_run("live-run", 999) == 0
    assert {r.memo_snapshot_id for r in _renders(sessions, run_id="live-run")} == {501}
