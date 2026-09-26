"""Guards against price-table drift in `services/llm_metrics.py`.

Until 2026-09-19 the table carried a placeholder for Opus 4.8 (3x the list
price), half the Haiku 4.5 rate, and no row at all for the configured cheap
OpenAI and long-document Gemini models, which silently priced at a provider
default 7x and 1.5x off. Nothing failed, because nothing checked. These tests
make the next drift a failing test rather than a wrong dashboard: every model
`Settings` routes to must have an exact row, dated snapshots must price as
their alias, and a fallback must announce itself.
"""
from __future__ import annotations

import logging

import pytest

from app.config import Settings
from app.services import llm_metrics


def _configured_models() -> dict[str, str]:
    """Every `*_model` setting with a non-empty default, by field name.

    Field defaults, not the loaded `settings`, so a developer `.env` cannot
    make the test pass or fail; a model set only through the environment in
    production is covered by the runtime warning instead."""
    out: dict[str, str] = {}
    for name, field in Settings.model_fields.items():
        default = field.default
        if name.endswith("_model") and isinstance(default, str) and default:
            out[name] = default
    return out


def test_every_configured_model_has_an_exact_price_row():
    configured = _configured_models()
    assert configured, "no *_model settings found — did Settings change shape?"
    missing = {
        name: model for name, model in configured.items()
        if llm_metrics.price_key(model) not in llm_metrics.MODEL_PRICES_PER_MTOK
    }
    assert not missing, (
        "models routed by Settings with no row in llm_metrics.MODEL_PRICES_PER_MTOK "
        f"(they would price at a provider default): {missing}"
    )


def test_price_key_prices_dated_snapshots_as_their_alias():
    assert llm_metrics.price_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert llm_metrics.price_key("gpt-5.5-2026-04-23") == "gpt-5.5"
    assert llm_metrics.price_key(" GPT-5.4 ") == "gpt-5.4"
    assert llm_metrics.price_key("gemini-2.5-pro") == "gemini-2.5-pro"
    assert llm_metrics.price_key("") == ""
    assert llm_metrics.estimate_cost_usd(
        "anthropic", "claude-haiku-4-5-20251001", 1_000_000, 0,
    ) == llm_metrics.estimate_cost_usd("anthropic", "claude-haiku-4-5", 1_000_000, 0)


def test_price_source_distinguishes_row_default_and_unpriced():
    assert llm_metrics.price_source("openai", "gpt-5.5") == "model"
    assert llm_metrics.price_source("Anthropic", "claude-opus-4-8") == "model"
    assert llm_metrics.price_source("openai", "gpt-9-preview") == "provider_default"
    assert llm_metrics.price_source("acme", "whatever") == "unpriced"
    assert llm_metrics.price_source("", "") == "unpriced"


def test_provider_default_is_logged_once_per_model(caplog):
    llm_metrics._warned_models.clear()
    with caplog.at_level(logging.WARNING, logger="app.services.llm_metrics"):
        first = llm_metrics.estimate_cost_usd("openai", "gpt-9-preview", 1_000_000, 0)
        second = llm_metrics.estimate_cost_usd("openai", "gpt-9-preview", 1_000_000, 0)
    assert first == second > 0, "the fallback still prices, it just says so"
    hits = [r for r in caplog.records if "gpt-9-preview" in r.getMessage()]
    assert len(hits) == 1
    assert "provider default" in hits[0].getMessage()


def test_verified_on_is_a_date():
    from datetime import date
    date.fromisoformat(llm_metrics.PRICES_VERIFIED_ON)


# Models Google has withdrawn from new API keys, or that never existed. A key
# created after the withdrawal gets 404 on every call, so a default naming one
# silently drops news to the provider-headline fallback in production.
_UNAVAILABLE_GEMINI_MODELS = {"gemini-2.5-flash", "gemini-3.1-pro"}


def test_gemini_defaults_name_models_a_new_key_can_call():
    configured = {
        name: model for name, model in _configured_models().items()
        if name.startswith("gemini_")
    }
    assert set(configured) >= {"gemini_news_model", "gemini_social_model", "gemini_longdoc_model"}
    stale = {n: m for n, m in configured.items() if m in _UNAVAILABLE_GEMINI_MODELS}
    assert not stale, f"Gemini defaults name unavailable models: {stale}"


def test_committed_config_env_matches_the_gemini_defaults():
    from pathlib import Path

    env = Path(__file__).resolve().parents[3] / "config.env"
    values = dict(
        line.split("=", 1) for line in env.read_text().splitlines()
        if line.startswith("GEMINI_") and "=" in line
    )
    for key in ("GEMINI_NEWS_MODEL", "GEMINI_SOCIAL_MODEL", "GEMINI_LONGDOC_MODEL"):
        field = Settings.model_fields[key.lower()].default
        assert values.get(key) == field, f"config.env {key}={values.get(key)!r} but Settings default is {field!r}"


# ---------------------------------------------------------------------------
# 2026-09-25 program (slice B7-M1): the owner's models and per-model cache rates
# ---------------------------------------------------------------------------

# $/MTok (in, out, cache read, cache write or None) from DEVPLAN owner
# decision 7 (verified official sources, 2026-09-25) and the plan's table.
_PROGRAM_PRICES = {
    "claude-opus-5-5":        (4.00, 20.00, 0.20, 5.00),
    "gpt-6-astra":            (10.00, 50.00, 1.00, 12.50),
    "gpt-6-sol":              (2.00, 10.00, 0.20, 2.50),
    "gpt-6-luna":             (0.10, 0.50, 0.01, 0.125),
    "gpt-4.1-mini":           (0.40, 1.60, 0.10, None),
    "gpt-5.5":                (5.00, 30.00, 0.50, None),
    "gemini-3.5-flash-lite":  (0.30, 2.50, 0.03, None),
    "gemini-3.8-flash":       (0.75, 3.75, 0.075, None),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20, None),
    "text-embedding-3-small": (0.02, 0.00, 0.02, None),
}


def test_price_rows_and_cache_rates():
    for model, (p_in, p_out, read, write) in _PROGRAM_PRICES.items():
        assert llm_metrics.MODEL_PRICES_PER_MTOK[model] == (p_in, p_out), model
        assert llm_metrics.MODEL_CACHE_PRICES_PER_MTOK[model] == (read, write), model
    # Every priced model has its own cache row: no hard-coded multiplier.
    assert set(llm_metrics.MODEL_CACHE_PRICES_PER_MTOK) == set(llm_metrics.MODEL_PRICES_PER_MTOK)
    assert llm_metrics.PRICES_VERIFIED_ON >= "2026-09-25"


def test_cache_reads_price_at_the_model_rate_not_a_multiplier():
    # Opus 5.5 reads at 0.05x input (the old code billed 0.1x); writes 1.25x.
    assert llm_metrics.estimate_cost_usd(
        "anthropic", "claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.20)
    assert llm_metrics.estimate_cost_usd(
        "anthropic", "claude-opus-5-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(5.00)
    # OpenAI cached tokens are inside prompt_tokens: 600k fresh + 400k cached.
    assert llm_metrics.estimate_cost_usd(
        "openai", "gpt-6-sol", 1_000_000, 0, cache_read_tokens=400_000,
    ) == pytest.approx(0.6 * 2.00 + 0.4 * 0.20)
    # gpt-4.1-mini reads at 0.25x (the old 0.5x overstated it).
    assert llm_metrics.estimate_cost_usd(
        "openai", "gpt-4.1-mini", 1_000_000, 0, cache_read_tokens=1_000_000,
    ) == pytest.approx(0.10)
    # Gemini cached content is inside prompt_token_count too.
    assert llm_metrics.estimate_cost_usd(
        "gemini", "gemini-3.5-flash-lite", 1_000_000, 0, cache_read_tokens=1_000_000,
    ) == pytest.approx(0.03)


def test_dated_price_applies_from_its_day():
    from datetime import date
    before = llm_metrics.estimate_cost_usd("gemini", "gemini-3.8-flash", 1_000_000, 1_000_000,
                                           on=date(2026, 12, 31))
    after = llm_metrics.estimate_cost_usd("gemini", "gemini-3.8-flash", 1_000_000, 1_000_000,
                                          on=date(2027, 1, 1))
    assert before == pytest.approx(0.75 + 3.75)
    assert after == pytest.approx(1.50 + 7.50)


def test_unpriced_configured_model_warns(monkeypatch, caplog):
    """A model set only through the environment (wave H sets tier models
    in render.yaml) is invisible to the defaults test above; the startup
    routing summary names it instead of silently pricing at a guess."""
    from app.agents import llm
    from app.config import settings
    monkeypatch.setattr(settings, "openai_strong_model", "gpt-9-unpriced")
    monkeypatch.setattr(llm, "_UNPRICED_WARNED", set())
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        assert "gpt-9-unpriced" in llm.unpriced_configured_models()
        llm.model_summary()
    hits = [r for r in caplog.records if "gpt-9-unpriced" in r.getMessage()]
    assert len(hits) == 1 and "OPENAI_STRONG_MODEL" in hits[0].getMessage()
