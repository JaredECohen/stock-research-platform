"""Static checks on render.yaml against the Dockerfile's actual layout.

These exist because of a real production failure: the worker service
shipped with `dockerCommand: python -m app.worker`, which exits 1 with
"No module named 'app'". The image's WORKDIR is /app while the backend
is copied to /app/backend, so the `app` package is not importable from
the default working directory. It was only ever tested from the
backend/ directory locally, where it works, so nothing caught it until
Render emailed about the crash.

Deploy config is code that runs exactly once per deploy, in an
environment nothing else reproduces. Cheap static assertions are the
only practical guard short of building the image in CI.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RENDER_YAML = REPO_ROOT / "render.yaml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

pytestmark = pytest.mark.skipif(
    not RENDER_YAML.exists() or not DOCKERFILE.exists(),
    reason="deploy config not present (e.g. installed package, not a checkout)",
)


def _services() -> list[dict]:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(RENDER_YAML.read_text())["services"]


def _backend_dest() -> str:
    """Where the Dockerfile copies the backend to, e.g. /app/backend."""
    m = re.search(r"^COPY\s+backend\s+(\S+)", DOCKERFILE.read_text(), re.MULTILINE)
    assert m, "Dockerfile no longer has a `COPY backend <dest>` line"
    return m.group(1)


def test_worker_command_can_import_the_app_package():
    """`python -m app.worker` must resolve the `app` package.

    Two acceptable mechanisms: PYTHONPATH pointing at the backend (what
    we use), or a command that cd's there. Either satisfies the import;
    neither present is the production failure this test exists for.
    """
    backend_dest = _backend_dest()
    workers = [s for s in _services() if s.get("type") == "worker"]
    assert workers, "no worker service defined in render.yaml"
    for svc in workers:
        cmd = svc.get("dockerCommand", "")
        assert "app.worker" in cmd, f"{svc['name']}: unexpected worker command {cmd!r}"
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        on_pythonpath = backend_dest in (env.get("PYTHONPATH") or "")
        assert on_pythonpath or backend_dest in cmd, (
            f"{svc['name']}: neither PYTHONPATH nor dockerCommand {cmd!r} puts "
            f"{backend_dest} on sys.path, so the `app` package is not importable "
            f"and the service will exit 1 with ModuleNotFoundError"
        )


def test_worker_command_execs_so_sigterm_reaches_python():
    """A wrapping shell must hand off via `exec`.

    Without it the shell is PID 1 and absorbs the SIGTERM Render sends on
    deploy; Python never sees it, the graceful shutdown in app/worker.py
    never runs, and the worker is SIGKILLed mid-job instead.
    """
    for svc in _services():
        cmd = svc.get("dockerCommand", "")
        if not cmd.startswith(("sh ", "bash ", "/bin/sh ", "/bin/bash ")):
            continue
        assert " exec " in f" {cmd} ", (
            f"{svc['name']}: shell-wrapped dockerCommand {cmd!r} must `exec` "
            f"the process so it receives SIGTERM directly"
        )


def test_web_and_worker_own_disjoint_background_work():
    """Exactly one service may run the regen queue + monitoring loops.

    Both on would re-couple the memory profiles the split exists to
    separate; both off would silently stop memo regeneration entirely.
    """
    owners = []
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        if env.get("ENABLE_REGEN_WORKER") == "true" or env.get("ENABLE_MONITORING") == "true":
            owners.append(svc["name"])
    assert len(owners) == 1, (
        f"expected exactly one service running regen/monitoring, got {owners}"
    )


def test_every_service_caps_malloc_arenas():
    """MALLOC_ARENA_MAX is the cheapest guard against RSS ratcheting in
    these thread-heavy processes; a new service silently omitting it
    would regress the 2026-08-12 memory work."""
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        assert env.get("MALLOC_ARENA_MAX"), f"{svc['name']} is missing MALLOC_ARENA_MAX"
