"""FIX-003: tests and seeding scripts refuse a non-local database.

Fixture tickers reached production because the suite and the
sqlite-to-postgres migration shared one database file and nothing checked
where a run was pointed. These tests pin the refusal at every entry point
that writes: the pytest session and the CI smoke test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.tests.dbguard import OPT_IN, refusal

BACKEND = Path(__file__).resolve().parents[2]

# RFC 2606 reserves .invalid, so even an unguarded run cannot reach a server.
REMOTE = "postgresql+psycopg2://fixture_user:fixture-s3cret@remote-db.invalid:5432/prod?connect_timeout=1"


def _child_env(**overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (OPT_IN, "MM_MIGRATE_OVERWRITE_TARGET")}
    env.update({"ENABLE_LIVE_DATA": "false", "USE_DEMO_DATA": "true",
                "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""})
    env.update(overrides)
    return env


def test_sqlite_urls_pass():
    for url in ("sqlite:///./marketmosaic.db", "sqlite:////tmp/x.db", "sqlite://"):
        assert refusal(url, {}) is None, url


def test_loopback_postgres_passes():
    for url in (
        "postgresql+psycopg2://u:p@localhost:5432/mm",
        "postgresql+psycopg2://u:p@127.0.0.1/mm",
        "postgresql+psycopg2://u:p@[::1]/mm",
        "postgresql:///mm",
        "postgresql://u:p@/mm?host=/var/run/postgresql",
    ):
        assert refusal(url, {}) is None, url


def test_remote_host_refused():
    reason = refusal("postgresql+psycopg2://user:s3cret@dpg-abc.oregon-postgres.render.com/marketmosaic", {})
    assert isinstance(reason, str)
    assert "dpg-abc.oregon-postgres.render.com" in reason
    assert OPT_IN in reason
    # libpq also takes the host from the query string when the netloc has none.
    assert isinstance(refusal("postgresql://u:p@/mm?host=db.example.com", {}), str)


def test_opt_in_allows_remote():
    assert refusal(REMOTE, {OPT_IN: "1"}) is None
    assert isinstance(refusal(REMOTE, {OPT_IN: "0"}), str)


def test_refusal_redacts_credentials():
    reason = refusal("postgresql+psycopg2://fixture_user:fixture-s3cret@db.example.com:5432/prod?password=hunter2", {})
    assert reason is not None
    assert "***@db.example.com:5432/prod" in reason
    for secret in ("fixture_user", "fixture-s3cret", "hunter2"):
        assert secret not in reason
    # An unparseable value is refused without being echoed back.
    unparseable = refusal("::not a url s3cret", {})
    assert unparseable is not None and "s3cret" not in unparseable


def _assert_redacted_refusal(output: str) -> None:
    assert "remote-db.invalid" in output
    assert "fixture-s3cret" not in output
    assert "fixture_user" not in output


def test_suite_refuses_remote_database_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "app/tests/test_db_isolation_guard.py::test_sqlite_urls_pass"],
        cwd=BACKEND, env=_child_env(DATABASE_URL=REMOTE),
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 4, output
    assert OPT_IN in output
    _assert_redacted_refusal(output)


def test_smoke_test_refuses_remote_database():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test"],
        cwd=BACKEND, env=_child_env(DATABASE_URL=REMOTE),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    _assert_redacted_refusal(result.stderr)
