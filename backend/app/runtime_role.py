"""Which process is this: the worker, the web service, or a one-off script.

Extracted from `app.monitoring` (which re-exports it) so the LLM layer can
label every call row with the process that made it without importing every
monitoring loop. Stdlib only: `agents/llm.py` imports this lazily from its
single row writer, and `app.monitoring` imports it at module top.
"""
from __future__ import annotations

import os
import sys

# Env override, read first so an operator can settle the question without
# depending on any inference below. Unset in production today, which is why
# the fallbacks have to work on their own.
PROCESS_ROLE_ENV = "MM_PROCESS_ROLE"
_ROLES = ("worker", "web")

# What `python -m app.worker` names the running module. Checked against
# `__main__.__spec__`, which runpy sets to the spec of the module it is
# executing — the only signal that survives `-m` intact.
_WORKER_MODULE = "app.worker"

# ASGI servers that host the web service. Render's web command runs uvicorn;
# anything else that is neither the worker nor one of these is a script.
_WEB_SERVERS = ("uvicorn", "gunicorn", "hypercorn")


def _process_role() -> str:
    """Which process is reporting a cron run: "worker" or "web".

    This is `/api/admin/cron-health`'s `reported_by`, and it was wrong for
    every row. The whole check used to be `any("app.worker" in a for a in
    sys.argv)`, which is never true in production: `python -m app.worker`
    rewrites `sys.argv[0]` to the module's *file path*, `/app/app/worker.py`.
    The literal "app.worker" — with a dot — appears nowhere in it. So every
    loop, `worker_heartbeat` included, was labelled "web", and the endpoint's
    one cross-process signal said the opposite of the truth.

    Three signals, most authoritative first:

    1. `MM_PROCESS_ROLE`, when an operator sets it. Nothing in Render sets it
       today, which is exactly why it cannot be the only signal.
    2. `__main__.__spec__.name` — runpy sets this to "app.worker" under
       `python -m app.worker`, dots intact, whatever it did to argv.
    3. The basename of `sys.argv[0]`, for `python app/worker.py`, plus the
       original substring check for a wrapper that really does carry the
       dotted name on its command line.

    Anything else is the web service: uvicorn, pytest, a shell.
    """
    explicit = (os.environ.get(PROCESS_ROLE_ENV) or "").strip().lower()
    if explicit in _ROLES:
        return explicit

    main = sys.modules.get("__main__")
    if getattr(getattr(main, "__spec__", None), "name", "") == _WORKER_MODULE:
        return "worker"

    argv = list(sys.argv or [])
    if argv and os.path.basename(argv[0].replace("\\", "/")) == "worker.py":
        return "worker"
    if any(_WORKER_MODULE in a for a in argv):
        return "worker"
    return "web"


def process_role() -> str:
    """Public name for `_process_role` (the monitoring alias stays for callers
    and tests that already use it)."""
    return _process_role()


def _main_module_name() -> str:
    """Best-effort dotted/base name of what is running as `__main__`."""
    main = sys.modules.get("__main__")
    spec_name = getattr(getattr(main, "__spec__", None), "name", "") or ""
    if spec_name:
        name = spec_name
        if name.endswith(".__main__"):
            name = name[: -len(".__main__")]
        return name.rsplit(".", 1)[-1]
    argv = list(sys.argv or [])
    path = getattr(main, "__file__", None) or (argv[0] if argv else "")
    base = os.path.basename(str(path).replace("\\", "/"))
    return base[:-3] if base.endswith(".py") else base


def default_origin() -> str:
    """The `origin` an LLM call gets when nothing upstream set one.

    Attribution critique #16: several scripts and threads reach the LLM
    layer with no origin (postmortem backfill, corpus repair, the fixture
    capture script, the worker's seed thread), and "-" in production tells a
    reviewer nothing. The worker says `worker:other`, the web server
    `web:other`, and anything else `script:<main module basename>`; explicit
    origins set by loops, workers and routes refine this.
    """
    role = _process_role()
    if role == "worker":
        return "worker:other"
    name = _main_module_name()
    if name in _WEB_SERVERS or any(s in name for s in _WEB_SERVERS):
        return "web:other"
    explicit = (os.environ.get(PROCESS_ROLE_ENV) or "").strip().lower()
    if explicit == "web":
        return "web:other"
    return f"script:{name or 'unknown'}"
