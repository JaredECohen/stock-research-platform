"""`reported_by` must say "worker" for the process that is the worker.

`/api/admin/cron-health` exists because the loops moved off the web service:
they run in `marketmosaic-worker`, the endpoint is served by web, and the two
processes share nothing but Postgres. `reported_by` is the column that says
which side wrote a row — the endpoint's only evidence that the split is
working at all.

It was wrong for every row. The entire check was

    any("app.worker" in a for a in sys.argv)

and production runs `python -m app.worker`, which rewrites `sys.argv[0]` to the
module's *file path*: `/app/app/worker.py`. That string contains "app/worker",
never "app.worker", so the condition was constant False. Every loop, including
`worker_heartbeat`, was labelled "web" — the endpoint confidently reported the
opposite of the truth, which is worse than reporting nothing.

An env override is the primary signal now, but it cannot be the only one: the
Render services are not configured with it and this fix has to work without a
config change. So the tests below drive the fallbacks with the argv production
actually has, and the first of them fails against the old implementation.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app import monitoring
from app.database import SessionLocal
from app.models import CronLoopRun


@pytest.fixture()
def no_env_override(monkeypatch):
    monkeypatch.delenv(monitoring.PROCESS_ROLE_ENV, raising=False)


def _argv(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)


def _no_main_spec(monkeypatch) -> None:
    """Isolate the argv fallback from the `__main__` one.

    Under pytest `__main__` is the pytest entry point, so this only matters
    for keeping each test about a single signal.
    """
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__spec__=None))


def test_python_dash_m_argv_is_recognised_as_the_worker(monkeypatch, no_env_override):
    """The production case, and the exact string that broke it.

    `python -m app.worker` on Render leaves `sys.argv == ['/app/app/worker.py']`.
    The old substring check returns False here; this test is the regression.
    """
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["/app/app/worker.py"])

    assert monitoring._process_role() == "worker"


def test_the_dash_m_module_spec_is_recognised(monkeypatch, no_env_override):
    """The signal `-m` actually preserves: runpy sets `__main__.__spec__`."""
    _argv(monkeypatch, ["/somewhere/else/entirely"])
    monkeypatch.setitem(
        sys.modules, "__main__", SimpleNamespace(__spec__=SimpleNamespace(name="app.worker")),
    )

    assert monitoring._process_role() == "worker"


def test_a_direct_script_invocation_is_recognised(monkeypatch, no_env_override):
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["app/worker.py"])
    assert monitoring._process_role() == "worker"


def test_a_wrapper_carrying_the_dotted_name_is_still_recognised(
    monkeypatch, no_env_override,
):
    """The original signal is kept — it was too narrow, not wrong."""
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["/bin/sh", "-c", "python -m app.worker"])
    assert monitoring._process_role() == "worker"


@pytest.mark.parametrize("argv", [
    ["/usr/local/bin/uvicorn", "app.main:app", "--host", "0.0.0.0"],
    ["/usr/local/bin/pytest", "app/tests"],
    [],
])
def test_everything_else_is_the_web_process(monkeypatch, no_env_override, argv):
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, argv)
    assert monitoring._process_role() == "web"


@pytest.mark.parametrize("value,expected", [
    ("worker", "worker"),
    ("web", "web"),
    ("  WORKER ", "worker"),
])
def test_the_env_override_wins(monkeypatch, value, expected):
    """An operator can settle it outright, whatever the process looks like."""
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["/app/app/worker.py"])
    monkeypatch.setenv(monitoring.PROCESS_ROLE_ENV, value)

    assert monitoring._process_role() == expected


def test_a_meaningless_override_falls_through_to_detection(monkeypatch):
    """A typo must not silently pin every row to a garbage label."""
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["/app/app/worker.py"])
    monkeypatch.setenv(monitoring.PROCESS_ROLE_ENV, "wroker")

    assert monitoring._process_role() == "worker"


def test_record_run_persists_the_worker_label(monkeypatch, no_env_override):
    """End to end: the row `/api/admin/cron-health` reads says "worker".

    The unit assertions above are about the predicate; this is about the
    column, which is what the endpoint actually shows and what was wrong.
    """
    _no_main_spec(monkeypatch)
    _argv(monkeypatch, ["/app/app/worker.py"])
    loop_name = "zz_process_role_probe"

    monitoring.record_run(loop_name, success=True, note="probe")
    try:
        with SessionLocal() as db:
            row = db.query(CronLoopRun).filter(
                CronLoopRun.loop_name == loop_name,
            ).one()
            assert row.reported_by == "worker"
        assert monitoring.status_snapshot()[loop_name]["note"] == "probe"
    finally:
        with SessionLocal() as db:
            db.query(CronLoopRun).filter(CronLoopRun.loop_name == loop_name).delete()
            db.commit()
        monitoring._LAST_RUNS.pop(loop_name, None)
