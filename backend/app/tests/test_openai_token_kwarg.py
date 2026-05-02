"""Param-name selection for OpenAI's chat.completions endpoint.

GPT-5.x and the o-series reasoning models reject `max_tokens` and require
`max_completion_tokens`. Older / non-reasoning chat models still accept
`max_tokens`. The `_openai_token_kwarg` helper picks the right one based
on the model name; this test pins that contract so a future refactor
can't accidentally regress live mode by sending the wrong kwarg.
"""
from __future__ import annotations

import pytest

from app.agents.llm import _openai_token_kwarg


@pytest.mark.parametrize("model", [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4",
])
def test_uses_max_completion_tokens_for_gpt5_and_o_series(model):
    out = _openai_token_kwarg(model, 100)
    assert out == {"max_completion_tokens": 100}


@pytest.mark.parametrize("model", [
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-opus-4-7",   # not OpenAI but the helper should still pick max_tokens
])
def test_uses_max_tokens_for_legacy_chat_models(model):
    out = _openai_token_kwarg(model, 100)
    assert out == {"max_tokens": 100}


def test_handles_empty_and_none_model():
    # Defensive — an empty model name shouldn't crash; pick the safer default.
    assert _openai_token_kwarg("", 50) == {"max_tokens": 50}
    assert _openai_token_kwarg(None, 50) == {"max_tokens": 50}  # type: ignore[arg-type]


def test_case_insensitive_model_name():
    # In case some caller passes uppercased model IDs.
    assert _openai_token_kwarg("GPT-5.5", 10) == {"max_completion_tokens": 10}
    assert _openai_token_kwarg("GPT-4.1-MINI", 10) == {"max_tokens": 10}


def test_n_coerced_to_int():
    # The helper always emits an int even when caller hands it a float (some
    # callers compute max_tokens dynamically as a fraction of context budget).
    out = _openai_token_kwarg("gpt-5.4", 50.7)  # type: ignore[arg-type]
    assert out == {"max_completion_tokens": 50}
