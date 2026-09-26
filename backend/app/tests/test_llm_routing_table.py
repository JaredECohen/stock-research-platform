"""The owner's model table, as routing (slice B7-M1; integration plan P1,
§2 and §7; DEVPLAN "Owner decisions — 2026-09-25" items 2, 3, 7, 8).

`test_routing_table_matches_owner_decisions` is the pinned encoding of the
table under the §7 PRODUCTION env (waves H and I). The code defaults are
blank, so `test_blank_tiers_are_legacy_routing` pins that a deployment
setting nothing sends exactly today's requests. Base note: production's
Gemini key cannot call gemini-2.5-flash (hotfix ca08521), so
gemini-3.5-flash-lite is BOTH the code default and the production news model.
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents import llm, llm_attribution
from app.config import Settings, settings
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, anthropic_response, openai_response

# Integration plan §7, production column (waves H and I), both services.
PROD_ENV = {
    "llm_research_model": "claude-opus-5-5",
    "llm_research_failover_model": "gpt-6-sol",
    "llm_pm_effort": "high",
    "llm_default_effort": "medium",
    "llm_failover_pm_effort": "high",
    "llm_failover_default_effort": "medium",
    "anthropic_strong_model": "claude-opus-5-5",
    "anthropic_critic_model": "claude-opus-5-5",
    "openai_strong_model": "gpt-6-sol",
    "openai_pm_model": "gpt-6-sol",
    "gemini_news_model": "gemini-3.5-flash-lite",
    "reviewer_mode": "full",
    "risk_reviewer_provider": "openai",
    "risk_reviewer_model": "gpt-6-astra",
    "risk_reviewer_effort": "high",
    "risk_reviewer_max_tokens": 25000,
    "review_recheck_effort": "medium",
    "chat_agents_sdk": True,
    "chat_model": "gpt-6-sol",
    "chat_effort": "medium",
    "debate_mode": "on",
    "debate_provider": "anthropic",
    "debate_model": "claude-opus-5-5",
    "debate_effort": "high",
    "debate_research_effort": "low",
    "debate_counterfactual_sample": 20,
}

OPUS, SOL, ASTRA = "claude-opus-5-5", "gpt-6-sol", "gpt-6-astra"
HAIKU, MINI = "claude-haiku-4-5", "gpt-4.1-mini"

# action -> (configured, provider, model, effort, failover provider, model, effort)
_RESEARCH_HIGH = (True, "anthropic", OPUS, "high", "openai", SOL, "high")
_RESEARCH_MED = (True, "anthropic", OPUS, "medium", "openai", SOL, "medium")
_UTILITY = (False, "anthropic", HAIKU, None, "openai", MINI, None)
_NEWS = (True, "gemini", "gemini-3.5-flash-lite", "minimal", None, None, None)
_EMBED = (False, "openai", "text-embedding-3-small", None, None, None, None)
_LEGACY = (False, None, None, None, None, None, None)

EXPECTED: dict[str, tuple] = {
    # PM synthesis / revision / counterfactual: effort HIGH (owner item 3).
    "pm.synthesis": _RESEARCH_HIGH, "pm.revision": _RESEARCH_HIGH,
    "pm.counterfactual": _RESEARCH_HIGH,
    # Intake, deep-research critique, DCF adjuster: Opus at medium (P2, P3).
    "pm.intake": _RESEARCH_MED, "pm.critique": _RESEARCH_MED, "pm.dcf_adjust": _RESEARCH_MED,
    # Every analyst: Opus at medium; valuation fails over at HIGH.
    **{a: _RESEARCH_MED for a in (
        "analyst.sector", "analyst.industry_group", "analyst.earnings", "analyst.earnings_qa",
        "analyst.filing", "analyst.comps", "analyst.comps_followup", "analyst.macro",
        "analyst.macro_regime", "analyst.risk_breakers", "analyst.risk_followup",
        "analyst.technical")},
    "analyst.valuation": (True, "anthropic", OPUS, "medium", "openai", SOL, "high"),
    # Research / other (item 7; P2's no-downgrade rule for dcf.update/chat.answer).
    **{a: _RESEARCH_MED for a in (
        "news.impact", "postmortem.review", "postmortem.lesson", "industry.report",
        "dcf.update", "chat.answer")},
    # Debate: one model for both sides; research plan LOW, openings/rebuttals HIGH;
    # the pair fails over to gpt-6-sol with the effort re-resolved.
    **{f"debate.{s}_research": (True, "anthropic", OPUS, "low", "openai", SOL, "low")
       for s in ("bull", "bear")},
    **{f"debate.{s}_{p}": (True, "anthropic", OPUS, "high", "openai", SOL, "high")
       for s in ("bull", "bear") for p in ("open", "rebut")},
    # Reviewer: gpt-6-astra high, failover Opus 5.5 high; re-check medium.
    "risk.review": (True, "openai", ASTRA, "high", "anthropic", OPUS, "high"),
    "review.recheck": (True, "openai", ASTRA, "medium", "anthropic", OPUS, "medium"),
    # The legacy critic keeps its own route (ANTHROPIC_CRITIC_MODEL).
    "risk.committee_review": _LEGACY,
    # Chat on the Agents SDK: gpt-6-sol medium, failover to Opus (legacy chat).
    "chat.sdk_turn": (True, "openai", SOL, "medium", "anthropic", OPUS, "medium"),
    "memo.sdk_exchange": _LEGACY,
    # News (and the demo-only social path) on Gemini 3.5 Flash-Lite, minimal.
    "news.search": _NEWS, "news.memo_fetch": _NEWS, "social.sentiment": _NEWS,
    # Utility: today's cheap route (Haiku 4.5, failover gpt-4.1-mini), effort
    # never sent.
    **{a: _UTILITY for a in (
        "chat.classify", "macro.scenario", "memo.long_form", "memo.earnings_qoq",
        "memo.reflect_company", "memo.reflect_sector", "memo.reflect_pattern",
        "memo.memory_condense", "memo.fact_extract", "dcf.exposure_peers",
        "dcf.sector_defaults", "dcf.scenarios", "screener.translate", "portfolio.brief",
        "geography.extract", "filing.delta", "filing.digest_weekly", "filing.digest_sector",
        "theme.exposure", "mispricing.audit", "chart.commentary", "samples.commentary",
        "learning.judge", "notes.summarize", "ops.validate_model_access")},
    "embed.index": _EMBED, "embed.query": _EMBED, "embed.repair": _EMBED,
}


@pytest.fixture
def prod_env(monkeypatch):
    """The §7 production values, validated by Settings itself, applied to
    the live settings with stub keys for all three providers (Render sets
    no VERTEX_*)."""
    validated = Settings(_env_file=None, **PROD_ENV)
    for key in PROD_ENV:
        monkeypatch.setattr(settings, key, getattr(validated, key))
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "anthropic_cheap_model", HAIKU)
    monkeypatch.setattr(settings, "openai_cheap_model", MINI)
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-anthropic")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-gemini")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "vertex_model", "")
    monkeypatch.setattr(settings, "llm_action_tier_overrides", "")
    return settings


@pytest.fixture(autouse=True)
def _fresh_breakers():
    """Breakers and failover counters are process-wide: a test that opens
    one must not route the next test's call."""
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


def _row(route: llm.ActionRoute) -> tuple:
    return (route.configured, route.provider, route.model, route.effort,
            route.failover_provider, route.failover_model, route.failover_effort)


def test_routing_table_matches_owner_decisions(prod_env):
    assert set(EXPECTED) == set(llm_attribution.ACTIONS), (
        "every registered action must have a pinned route here"
    )
    wrong = {a: _row(llm.resolve_action_route(a)) for a in EXPECTED
             if _row(llm.resolve_action_route(a)) != EXPECTED[a]}
    assert not wrong, wrong
    # Always-thinking floors come with the route.
    assert llm.resolve_action_route("pm.synthesis").floor == 16000
    assert llm.resolve_action_route("risk.review").floor == 25000
    # The legacy critic runs on ANTHROPIC_CRITIC_MODEL, which wave H moves.
    assert llm.resolve_role_model("critic") == OPUS
    # Gemini news: code default AND production value (hotfix ca08521).
    assert Settings.model_fields["gemini_news_model"].default == "gemini-3.5-flash-lite"
    assert llm.model_summary()["chat"] == f"sdk:{SOL}"


def test_prod_env_dispatches_the_tier_and_fails_over_by_model(prod_env, monkeypatch):
    partner = FakeClient(openai_response(model="gpt-6-sol-2026-08-01"))
    primary = FakeClient(anthropic_response("not json", model="claude-opus-5-5-20260901"))
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: primary)
    monkeypatch.setattr(llm, "_openai_client", lambda: partner)
    llm.reset_circuit_breaker()
    run_id = f"route-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        # The call site still passes today's per-role knob; the tier wins.
        out = llm.chat_json("p", route="strong", model="gpt-5.5", action="pm.synthesis")
    assert out == {"ok": True}
    (sent,) = primary.requests
    assert sent["model"] == OPUS and sent["output_config"] == {"effort": "high"}
    assert sent["max_tokens"] == 16000 and "temperature" not in sent
    (hop,) = partner.requests
    assert hop["model"] == SOL and hop["reasoning_effort"] == "high"
    assert hop["max_completion_tokens"] == 25000
    first, second = llm_fakes.rows_for(run_id)
    assert first.model_resolution == "tier" and first.requested_model == OPUS
    assert second.model_resolution == "failover_mapped" and second.effort == "high"
    assert second.served_model == "gpt-6-sol-2026-08-01"


def test_prod_env_reviewer_goes_to_astra(prod_env, monkeypatch):
    client = FakeClient(openai_response())
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    llm.chat_json("p", route="strong", provider_override="anthropic", action="risk.review")
    (sent,) = client.requests
    assert sent["model"] == ASTRA and sent["reasoning_effort"] == "high"
    assert sent["max_completion_tokens"] == 25000


def test_blank_tiers_are_legacy_routing(monkeypatch):
    """Code defaults: every tier blank, so a call with an action sends
    exactly what the same call without one sends today."""
    for field in ("llm_research_model", "debate_model", "debate_provider",
                  "risk_reviewer_provider", "chat_model"):
        assert Settings.model_fields[field].default == "", field
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()),
                   anthropic=FakeClient(anthropic_response()), active="anthropic")
    for action in llm_attribution.ACTIONS:
        route = llm.resolve_action_route(action)
        assert route.configured == (route.tier == "news"), action
    with_action = FakeClient(anthropic_response())
    without = FakeClient(anthropic_response())
    monkeypatch.setattr(llm, "_anthropic_client", lambda: with_action)
    llm.chat_json("p", route="strong", model="claude-opus-4-8", action="pm.synthesis")
    monkeypatch.setattr(llm, "_anthropic_client", lambda: without)
    llm.chat_json("p", route="strong", model="claude-opus-4-8")
    assert with_action.requests == without.requests
    assert "output_config" not in with_action.requests[0]


def test_action_tier_override(prod_env):
    prod_env.llm_action_tier_overrides = "industry.report:legacy,news.impact:utility"
    assert llm.tier_overrides() == {"industry.report": "legacy", "news.impact": "utility"}
    assert llm.resolve_action_route("industry.report").configured is False
    assert llm.resolve_action_route("industry.report").reason == "override:legacy"
    assert _row(llm.resolve_action_route("news.impact")) == _UTILITY
    assert llm.resolve_action_route("pm.synthesis").configured is True
    prod_env.llm_action_tier_overrides = "no.such_action:legacy"
    with pytest.raises(ValueError, match="unknown action"):
        llm.resolve_action_route("pm.synthesis")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_action_tier_overrides="industry.report")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_action_tier_overrides="industry.report:turbo")


def test_unknown_override_action_is_rejected_at_boot():
    """A typo in the rollback lever must stop the process at boot. Raised
    later, inside chat_json, it is swallowed by the memo pipeline's
    safe_call and every routed call silently becomes a stub finding."""
    with pytest.raises(ValidationError, match="unknown action 'industry.reprot'"):
        Settings(_env_file=None, llm_action_tier_overrides="industry.reprot:legacy")
    ok = Settings(_env_file=None,
                  llm_action_tier_overrides="industry.report:legacy, news.impact:utility")
    assert ok.llm_action_tier_overrides == "industry.report:legacy, news.impact:utility"


def test_unregistered_action_is_an_error():
    with pytest.raises(ValueError, match="unregistered"):
        llm.resolve_action_route("made.up")


def test_a_tier_without_a_key_stays_legacy(prod_env):
    prod_env.anthropic_api_key = ""
    route = llm.resolve_action_route("pm.synthesis")
    assert route.configured is False and "no anthropic key" in route.reason


def test_astra_effort_none_rejected():
    with pytest.raises(ValidationError, match="gpt-6-astra"):
        Settings(_env_file=None, risk_reviewer_model="gpt-6-astra", risk_reviewer_effort="none")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, risk_reviewer_effort="turbo")
    ok = Settings(_env_file=None, risk_reviewer_model="gpt-6-astra", risk_reviewer_effort="HIGH")
    assert ok.risk_reviewer_effort == "high"


def test_rebuttal_rounds_clamped():
    """Protocol v1 has exactly one rebuttal round."""
    assert Settings(_env_file=None, debate_rebuttal_rounds=3).debate_rebuttal_rounds == 1
    assert Settings(_env_file=None, debate_rebuttal_rounds=-1).debate_rebuttal_rounds == 0
    assert Settings(_env_file=None).debate_rebuttal_rounds == 1


def test_mode_enums_and_the_debate_without_reviewer_warning(caplog):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, reviewer_mode="partial")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, debate_mode="maybe")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_attribution_mode="loud")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        Settings(_env_file=None, debate_mode="on", reviewer_mode="legacy")
    assert any("DEBATE_MODE=on with REVIEWER_MODE=legacy" in r.getMessage() for r in caplog.records)


def test_settings_manifest_code_defaults():
    """Integration plan §7, code-default column: blank or off everywhere a
    value would change a request, the program's caps as specified."""
    defaults = {name: f.default for name, f in Settings.model_fields.items()}
    expected = {
        "llm_research_model": "", "llm_research_failover_model": "",
        "llm_pm_effort": "", "llm_default_effort": "",
        "llm_failover_pm_effort": "", "llm_failover_default_effort": "",
        "anthropic_strong_model": "claude-opus-4-8", "anthropic_critic_model": "claude-opus-4-8",
        "openai_strong_model": "gpt-5.5", "openai_pm_model": "gpt-5.5",
        "gemini_news_model": "gemini-3.5-flash-lite",
        "gemini_longdoc_model": "gemini-3.1-pro-preview",
        "gemini_grounded_max_per_day": 250, "news_fetch_at_memo_time": True,
        "llm_action_tier_overrides": "", "reviewer_mode": "legacy",
        "risk_reviewer_provider": "", "risk_reviewer_model": "", "risk_reviewer_effort": "",
        "risk_reviewer_max_tokens": 25000, "review_recheck_effort": "",
        "review_revision_max_usd": 1.00, "reviewer_caps_enabled": True,
        "chat_agents_sdk": False, "chat_model": "", "chat_effort": "",
        "debate_mode": "off", "debate_provider": "", "debate_model": "",
        "debate_effort": "", "debate_research_effort": "", "debate_rebuttal_rounds": 1,
        "debate_max_tokens_research": 8000, "debate_max_tokens_opening": 16000,
        "debate_max_tokens_rebuttal": 16000, "debate_pm_max_tokens": 4000,
        "debate_pm_block_max_chars": 14000, "debate_queries_per_side": 3,
        "debate_pool_max": 16, "debate_max_claims": 5, "debate_horizon": "12 months",
        "debate_max_usd_per_memo": 1.50, "debate_max_calls": 10, "memo_max_usd": 5.00,
        "debate_parallel": True, "debate_counterfactual_sample": 0,
        "debate_counterfactual_max_usd": 0.40,
        "llm_thinking_max_tokens_floor": 16000, "llm_anthropic_nonstream_max_tokens": 16000,
        "llm_attribution_mode": "warn", "llm_call_log_enabled": True,
    }
    wrong = {k: (defaults.get(k), v) for k, v in expected.items() if defaults.get(k) != v}
    assert not wrong, wrong


def test_committed_env_files_document_every_program_setting():
    """config.env and example.env name every §7 setting (commented where
    the code default applies) so an operator can find each knob."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    for name in ("config.env", "example.env"):
        text = (root / name).read_text()
        missing = [k.upper() for k in PROD_ENV if k.upper() not in text]
        missing += [k for k in ("LLM_ACTION_TIER_OVERRIDES", "GEMINI_GROUNDED_MAX_PER_DAY",
                                "LLM_ATTRIBUTION_MODE", "LLM_CALL_LOG_ENABLED", "MEMO_MAX_USD",
                                "DEBATE_PM_BLOCK_MAX_CHARS", "LLM_THINKING_MAX_TOKENS_FLOOR",
                                "LLM_ANTHROPIC_NONSTREAM_MAX_TOKENS") if k not in text]
        assert not missing, (name, missing)


def test_model_summary_reports_the_tiers(prod_env):
    summary = llm.model_summary()
    assert summary["tiers"]["research_pm"] == f"anthropic:{OPUS}@high failover=openai:{SOL}@high"
    assert summary["tiers"]["reviewer"] == f"openai:{ASTRA}@high failover=anthropic:{OPUS}@high"
    assert summary["tiers"]["news"] == "gemini:gemini-3.5-flash-lite@minimal"
    assert summary["tiers"]["utility"].startswith("legacy")
    assert summary["gemini"]["backend"] == "api"
    assert summary["failover_map"][OPUS] == f"openai:{SOL}"
    assert summary["attribution_mode"] == "warn"
    text = repr(summary)
    assert "stub-" not in text


def test_model_access_report_lists_only(prod_env, monkeypatch, caplog):
    """`models.list` only: never a generation (plan §8.1 — the owner reads
    this line before wave H)."""
    def _lister(ids):
        generated: list = []
        client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: [SimpleNamespace(id=i) for i in ids],
                                   generate_content=lambda **k: generated.append(k)),
            messages=SimpleNamespace(create=lambda **k: generated.append(k)),
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: generated.append(k))),
        )
        return client, generated

    anthropic_client, a_gen = _lister([f"{OPUS}-20260901", "claude-haiku-4-5-20251001"])
    openai_client, o_gen = _lister([SOL, MINI])            # astra missing
    gemini_client = SimpleNamespace(models=SimpleNamespace(
        list=lambda: [SimpleNamespace(name="models/gemini-3.5-flash-lite")]))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: anthropic_client)
    monkeypatch.setattr(llm, "_openai_client", lambda: openai_client)
    monkeypatch.setattr(llm, "_gemini_client", lambda: gemini_client)
    with caplog.at_level(logging.INFO, logger="app.agents.llm"):
        report = llm.model_access_report()
    assert report[OPUS] == "ok" and report[HAIKU] == "ok"
    assert report[SOL] == "ok" and report[ASTRA] == "missing"
    assert report["gemini-3.5-flash-lite"] == "ok"
    assert report["gemini-3.1-pro-preview"] == "missing"
    assert a_gen == [] and o_gen == [], "no generation, ever"
    (line,) = [r for r in caplog.records if r.getMessage().startswith("model_access ")]
    assert f"{ASTRA}=missing" in line.getMessage() and line.levelno == logging.WARNING


def test_model_access_accepts_only_the_id_or_its_dated_snapshot(prod_env, monkeypatch):
    """A listed id that merely STARTS with the configured one is a
    different model. gemini-3.1-pro is the id DEVPLAN 2026-09-25 item 4
    records as unlisted for the production key; reporting it `ok` because
    gemini-3.1-pro-preview is listed would pass the §8.1 access check for
    exactly the model it exists to catch."""
    monkeypatch.setattr(prod_env, "gemini_longdoc_model", "gemini-3.1-pro")
    monkeypatch.setattr(prod_env, "gemini_social_model", "gemini-3.5-flash")
    prod_env.anthropic_cheap_model = "claude-opus-5"
    prod_env.openai_cheap_model = "gpt-6"

    def _client(ids, attr="id"):
        return SimpleNamespace(models=SimpleNamespace(
            list=lambda: [SimpleNamespace(**{attr: i}) for i in ids]))

    monkeypatch.setattr(llm, "_anthropic_client", lambda: _client([f"{OPUS}-20260901"]))
    monkeypatch.setattr(llm, "_openai_client", lambda: _client(
        ["gpt-6-sol-2026-08-01", ASTRA, "gpt-6-mini"]))
    monkeypatch.setattr(llm, "_gemini_client", lambda: _client(
        ["models/gemini-3.1-pro-preview", "models/gemini-3.5-flash-lite"], attr="name"))
    report = llm.model_access_report()
    # Different models sharing a prefix: missing.
    assert report["gemini-3.1-pro"] == "missing"
    assert report["gemini-3.5-flash"] == "missing"
    assert report["claude-opus-5"] == "missing"
    assert report["gpt-6"] == "missing"
    # The id itself, or a dated snapshot of it (both date forms): ok.
    assert report[OPUS] == "ok" and report[SOL] == "ok" and report[ASTRA] == "ok"
    assert report["gemini-3.5-flash-lite"] == "ok"


def test_reviewer_provider_is_inferred_from_the_model(prod_env):
    """RISK_REVIEWER_MODEL alone routes the reviewer, as DEBATE_MODEL alone
    routes the debate. A silent legacy fallback here would keep risk.review
    off the owner's item-8 reviewer with nothing in the routing line but
    `legacy(reviewer tier blank)`."""
    prod_env.risk_reviewer_provider = ""
    assert _row(llm.resolve_action_route("risk.review")) == EXPECTED["risk.review"]
    assert _row(llm.resolve_action_route("review.recheck")) == EXPECTED["review.recheck"]


def _tier_failover_setup(monkeypatch):
    partner = FakeClient(openai_response())
    primary = FakeClient(anthropic_response("not json"))
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: primary)
    monkeypatch.setattr(llm, "_openai_client", lambda: partner)
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    return primary, partner


def test_tier_failover_honours_the_kill_switch(prod_env, monkeypatch):
    """The configured-tier hop (`_failover_hop` with a mapped route) has
    its own LLM_FAILOVER_ENABLED check; from wave H it carries every
    research, debate, reviewer and chat call."""
    primary, partner = _tier_failover_setup(monkeypatch)
    monkeypatch.setattr(prod_env, "llm_failover_enabled", False)
    assert llm.chat_json("p", route="strong", action="pm.synthesis") is None
    assert len(primary.requests) == 1
    assert partner.requests == [] and llm.get_failover_state()["count"] == 0


def test_tier_failover_needs_the_partner_key(prod_env, monkeypatch):
    primary, partner = _tier_failover_setup(monkeypatch)
    monkeypatch.setattr(prod_env, "llm_failover_enabled", True)
    prod_env.openai_api_key = ""
    assert llm.chat_json("p", route="strong", action="pm.synthesis") is None
    assert len(primary.requests) == 1
    assert partner.requests == [] and llm.get_failover_state()["count"] == 0


def test_skip_rows_record_the_skipped_attempts_own_budget(prod_env, monkeypatch):
    """A skipped attempt's row carries the effort and max_tokens THAT
    attempt would have sent, not the previous attempt's: a skipped
    gpt-6-sol hop is 25,000 tokens at its failover effort, and a skipped
    Opus primary is the 16k floor, not the caller's 1,600."""
    primary, partner = _tier_failover_setup(monkeypatch)
    monkeypatch.setattr(prod_env, "llm_failover_enabled", True)
    for _ in range(3):
        llm._record_failure("openai")
    assert llm.breaker_open("openai")
    run_id = f"skip-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", route="strong", max_tokens=1600,
                             action="analyst.valuation") is None
    first, skipped = llm_fakes.rows_for(run_id)
    assert (first.provider, first.effort, first.max_tokens) == ("anthropic", "medium", 16000)
    assert skipped.error_type == "skipped:partner_breaker_open"
    # analyst.valuation fails over at HIGH (owner item 3's table).
    assert (skipped.provider, skipped.model) == ("openai", SOL)
    assert (skipped.effort, skipped.max_tokens) == ("high", 25000)

    llm.reset_circuit_breaker()
    for _ in range(3):
        llm._record_failure("anthropic")
    run_id = f"skip-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", route="strong", max_tokens=1600,
                             action="pm.synthesis") == {"ok": True}
    skipped, hop = llm_fakes.rows_for(run_id)
    assert skipped.error_type == "skipped:breaker_open" and skipped.model == OPUS
    assert (skipped.effort, skipped.max_tokens) == ("high", 16000)
    assert (hop.model, hop.effort, hop.max_tokens) == (SOL, "high", 25000)


def test_an_explicit_effort_beats_the_tier(prod_env, monkeypatch):
    client = FakeClient(anthropic_response())
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    monkeypatch.setattr(llm, "_openai_client", lambda: None)
    assert llm.chat_json("p", route="strong", action="pm.synthesis", effort="low") == {"ok": True}
    (sent,) = client.requests
    assert sent["model"] == OPUS and sent["output_config"] == {"effort": "low"}


def test_an_action_on_the_context_routes_the_call(prod_env, monkeypatch):
    """Call sites wrapped in `llm_call_context(action=...)` pass no action=
    of their own; the tier must still apply to them."""
    anthropic = FakeClient(anthropic_response())
    openai = FakeClient(openai_response())
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: anthropic)
    monkeypatch.setattr(llm, "_openai_client", lambda: openai)
    with llm.llm_call_context(action="pm.synthesis"):
        llm.chat_json("p", route="strong", provider_override="openai", model="gpt-5.5")
    assert openai.requests == []
    (sent,) = anthropic.requests
    assert sent["model"] == OPUS and sent["output_config"] == {"effort": "high"}


def test_a_gemini_tier_is_never_dispatched_through_chat_json(prod_env, monkeypatch):
    """The news tier is Gemini's own entry points; an Anthropic/OpenAI call
    site that happens to carry a news action keeps its provider."""
    anthropic = FakeClient(anthropic_response())
    openai = FakeClient(openai_response())
    gemini = FakeClient(llm_fakes.gemini_response())
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: anthropic)
    monkeypatch.setattr(llm, "_openai_client", lambda: openai)
    monkeypatch.setattr(llm, "_gemini_client", lambda: gemini)
    assert llm.chat_json("p", action="news.search", provider_override="anthropic") == {"ok": True}
    assert len(anthropic.requests) == 1 and gemini.requests == [] and openai.requests == []


def test_boot_rejects_an_unknown_override_action_before_the_agents_import():
    """Production imports app.config before the agents package, so the
    registry is read by FILE PATH there — the branch an in-process test
    never reaches (llm_attribution is already in sys.modules)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("import sys\n"
            "try:\n"
            "    import app.config\n"
            "except Exception as exc:\n"
            "    assert 'app.agents.llm_attribution' not in sys.modules\n"
            "    print(type(exc).__name__, exc)\n"
            "    sys.exit(3)\n")
    env = {**os.environ, "LLM_ACTION_TIER_OVERRIDES": "industry.reprot:legacy"}
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        env[key] = ""
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120,
                          cwd=Path(__file__).resolve().parents[2])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "ValidationError" in proc.stdout and "industry.reprot" in proc.stdout
