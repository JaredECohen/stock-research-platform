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
