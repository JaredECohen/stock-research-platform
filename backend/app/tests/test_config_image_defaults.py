"""Exercise settings in the image layout without developer files or secrets."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("model_override", [None, "deployment-pm-override"])
def test_image_defaults_keep_pm_model_and_bounded_research(tmp_path, model_override):
    from app import config

    # Docker copies backend but not the repository's config.env. Loading the
    # real module from this staged layout avoids masking defaults with local
    # .env/config.env files or the already-imported singleton.
    image_app = tmp_path / "image" / "backend" / "app"
    image_app.mkdir(parents=True)
    image_config = image_app / "config.py"
    shutil.copyfile(config.__file__, image_config)
    script = """
import importlib.util
import json
import pathlib
import sys
spec = importlib.util.spec_from_file_location("image_config", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
settings = module.settings
print(json.dumps({
    "existing_env_files": [p for p in module._project_env_files() if pathlib.Path(p).exists()],
    "pm": settings.openai_pm_model,
    "rounds": settings.deep_research_max_rounds,
    "questions": settings.deep_research_max_questions_per_round,
    "sdk": settings.use_agents_sdk,
    "vertex_project": settings.vertex_project_id,
    "vertex_model": settings.vertex_model,
    "has_gemini": settings.has_gemini,
}))
"""
    process_env = {} if model_override is None else {"OPENAI_PM_MODEL": model_override}
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(image_config)],
        env=process_env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout) == {
        "existing_env_files": [],
        "pm": model_override or "gpt-5.5",
        "rounds": 1,
        "questions": 2,
        "sdk": False,
        "vertex_project": "",
        "vertex_model": "",
        "has_gemini": False,
    }
