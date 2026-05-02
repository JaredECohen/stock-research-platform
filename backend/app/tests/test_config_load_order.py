"""Two-file env split: load order + override semantics.

Verifies the contract documented in `_project_env_files`:
    1. config.env  ← committed defaults (model assignments, flags)
    2. .env        ← gitignored secrets + per-deployment overrides
    3. process env ← wins over both

Use a transient tmp dir to construct synthetic config + env files and load
a `Settings` with `model_config.env_file` pointed at them. The real repo
files are not touched.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


def _make_settings(env_files):
    """Build a Settings subclass with the given env files in order."""
    class _S(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=tuple(env_files),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )
        openai_pm_model: str = "default"
        openai_api_key: str = ""

    return _S()


def test_config_env_provides_default_when_dot_env_absent(tmp_path):
    cfg = tmp_path / "config.env"
    cfg.write_text("OPENAI_PM_MODEL=from-config\n")
    s = _make_settings([str(cfg), str(tmp_path / ".env-missing")])
    assert s.openai_pm_model == "from-config"


def test_dot_env_overrides_config_env(tmp_path):
    cfg = tmp_path / "config.env"
    cfg.write_text("OPENAI_PM_MODEL=from-config\n")
    env = tmp_path / ".env"
    env.write_text("OPENAI_PM_MODEL=from-dot-env\n")
    s = _make_settings([str(cfg), str(env)])
    assert s.openai_pm_model == "from-dot-env"


def test_process_env_overrides_both(tmp_path, monkeypatch):
    cfg = tmp_path / "config.env"
    cfg.write_text("OPENAI_PM_MODEL=from-config\n")
    env = tmp_path / ".env"
    env.write_text("OPENAI_PM_MODEL=from-dot-env\n")
    monkeypatch.setenv("OPENAI_PM_MODEL", "from-process-env")
    s = _make_settings([str(cfg), str(env)])
    assert s.openai_pm_model == "from-process-env"


def test_dot_env_carries_secrets_config_env_carries_models(tmp_path):
    """The whole point of the split: secrets in .env, models in config.env."""
    cfg = tmp_path / "config.env"
    cfg.write_text("OPENAI_PM_MODEL=from-config\n")
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-dot-env\n")
    s = _make_settings([str(cfg), str(env)])
    assert s.openai_pm_model == "from-config"
    assert s.openai_api_key == "sk-from-dot-env"


def test_repo_config_env_is_loaded_at_app_startup():
    """Smoke check: `app.config.settings` actually reads our committed
    config.env (verified by the per-agent model defaults landing on the
    spec values)."""
    from app.config import settings as live_settings
    # These come from config.env — if the file weren't being loaded the
    # field defaults in Settings would still match because we set them
    # there too. So instead test that the env_file list contains config.env.
    from app.config import _project_env_files
    paths = _project_env_files()
    assert any(p.endswith("config.env") for p in paths), (
        f"config.env should appear in env_file list: {paths}"
    )
    # And ordering: config.env appears BEFORE .env so .env wins for collisions.
    cfg_idx = next(i for i, p in enumerate(paths) if p.endswith("config.env"))
    env_idx = next(
        (i for i, p in enumerate(paths) if p.endswith("/.env") or p.endswith("\\.env")),
        len(paths),
    )
    assert cfg_idx < env_idx, (
        f"config.env (idx {cfg_idx}) must load before .env (idx {env_idx})"
    )
    # And the spec defaults are reachable (sanity).
    assert live_settings.openai_pm_model
