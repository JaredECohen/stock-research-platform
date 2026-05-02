"""Vertex AI as an alternative Gemini backend.

Pin the contract for `_gemini_client` and `_resolve_gemini_model`:
- When `VERTEX_PROJECT_ID` is set, Vertex backend is selected (auth via ADC).
- When only `GEMINI_API_KEY` is set, direct API backend is selected.
- When both are set, Vertex wins.
- When neither is set, returns None and callers fall back to deterministic stub.
- `VERTEX_MODEL` overrides per-agent Gemini model envs across all Gemini
  calls when set; per-call `model=` arg always wins over both.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents import llm as llm_mod
from app.agents.llm import _gemini_client, _resolve_gemini_model
from app.config import settings


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_vertex_wins_when_both_keys_configured(monkeypatch):
    """Vertex backend selected even when GEMINI_API_KEY is also set —
    production should not silently fall back to API-key auth."""
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_location", "us-central1")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-key")
    monkeypatch.setattr(settings, "vertex_model", "")

    with patch.object(llm_mod, "_genai") as fake_genai:
        fake_genai.Client.return_value = "vertex-client"
        client = _gemini_client()
        assert client == "vertex-client"
        fake_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )


def test_direct_api_when_only_gemini_key_set(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-key")

    with patch.object(llm_mod, "_genai") as fake_genai:
        fake_genai.Client.return_value = "api-client"
        client = _gemini_client()
        assert client == "api-client"
        fake_genai.Client.assert_called_once_with(api_key="stub-key")


def test_returns_none_when_nothing_configured(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")

    with patch.object(llm_mod, "_genai") as fake_genai:
        client = _gemini_client()
        assert client is None
        fake_genai.Client.assert_not_called()


def test_returns_none_when_genai_package_not_installed(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-key")
    with patch.object(llm_mod, "_genai", None):
        assert _gemini_client() is None


def test_vertex_location_defaults_when_unset(monkeypatch):
    """If VERTEX_LOCATION is somehow blanked, fall back to us-central1
    rather than passing an empty string to the SDK."""
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_location", "")

    with patch.object(llm_mod, "_genai") as fake_genai:
        _gemini_client()
        fake_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )


# ---------------------------------------------------------------------------
# has_vertex / has_gemini convenience properties
# ---------------------------------------------------------------------------

def test_has_vertex_true_when_project_id_set(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    assert settings.has_vertex is True


def test_has_vertex_false_when_project_id_empty(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "")
    assert settings.has_vertex is False


def test_has_gemini_true_when_either_path_configured(monkeypatch):
    # Vertex only
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert settings.has_gemini is True

    # API key only
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-key")
    assert settings.has_gemini is True

    # Neither
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert settings.has_gemini is False


# ---------------------------------------------------------------------------
# Model resolution precedence
# ---------------------------------------------------------------------------

def test_caller_model_wins_over_vertex_model(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_model", "vertex-default")
    out = _resolve_gemini_model("explicit-from-caller", "default-fallback")
    assert out == "explicit-from-caller"


def test_vertex_model_wins_over_default_when_vertex_active(monkeypatch):
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_model", "vertex-3.1-pro")
    out = _resolve_gemini_model(None, "gemini-2.5-flash")
    assert out == "vertex-3.1-pro"


def test_vertex_model_ignored_when_vertex_not_active(monkeypatch):
    """VERTEX_MODEL is set but VERTEX_PROJECT_ID isn't → not actually
    on Vertex, so per-agent default wins."""
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "vertex_model", "vertex-3.1-pro")
    out = _resolve_gemini_model(None, "gemini-2.5-flash")
    assert out == "gemini-2.5-flash"


def test_default_used_when_vertex_active_but_vertex_model_unset(monkeypatch):
    """Vertex is configured but VERTEX_MODEL is empty → per-agent env
    flows through unchanged. Caller is responsible for ensuring the
    model name exists on Vertex."""
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_model", "")
    out = _resolve_gemini_model(None, "gemini-2.5-flash")
    assert out == "gemini-2.5-flash"


def test_caller_empty_string_treated_as_none(monkeypatch):
    """An empty-string `model=` shouldn't override; should fall through to
    the resolver's other tiers (matches existing chat_json semantics)."""
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(settings, "vertex_model", "vertex-3.1-pro")
    out = _resolve_gemini_model("", "default-fallback")
    assert out == "vertex-3.1-pro"
