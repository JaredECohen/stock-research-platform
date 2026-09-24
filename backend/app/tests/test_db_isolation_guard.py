"""FIX-003: tests and seeding scripts refuse a non-local database.

Fixture tickers reached production because the suite and the
sqlite-to-postgres migration shared one database file and nothing checked
where a run was pointed. These tests pin the refusal at every entry point
that writes: the pytest session, the CI smoke test, the test-fixture seeder
(and measure_live_cost, which runs it) and the migration.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select

from app.database import Base
from app.models import Company
from app.tests.dbguard import OPT_IN, refusal

BACKEND = Path(__file__).resolve().parents[2]

# RFC 2606 reserves .invalid, so even an unguarded run cannot reach a server.
REMOTE = "postgresql+psycopg2://fixture_user:fixture-s3cret@remote-db.invalid:5432/prod?connect_timeout=1"


# Stripped from every child: the opt-ins would switch the guards off, the
# migrate inputs would override the cases under test, and the libpq PG*
# fallbacks would change where a host-less URL points.
_INHERITED_OFF = (OPT_IN, "MM_MIGRATE_OVERWRITE_TARGET", "SOURCE_SQLITE", "TARGET_POSTGRES_URL",
                  "PGHOST", "PGHOSTADDR", "PGSERVICE")


def _child_env(**overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _INHERITED_OFF}
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
        "postgresql+psycopg2://u:p@127.0.0.5/mm",
        "postgresql:///mm",
        "postgresql://u:p@/mm?host=/var/run/postgresql",
        "postgresql://u:p@/mm?host=localhost:5433&host=127.0.0.1:5434",
        "postgresql://u:p@localhost/mm?hostaddr=127.0.0.1",
    ):
        assert refusal(url, {}) is None, url
    # A host-less URL is local only when libpq's env fallbacks are too.
    assert refusal("postgresql:///mm", {"PGHOST": "/tmp"}) is None
    assert refusal("postgresql:///mm", {"PGHOST": "localhost", "PGHOSTADDR": "127.0.0.1"}) is None


def test_remote_host_refused():
    reason = refusal("postgresql+psycopg2://user:s3cret@dpg-abc.oregon-postgres.render.com/marketmosaic", {})
    assert isinstance(reason, str)
    assert "dpg-abc.oregon-postgres.render.com" in reason
    assert OPT_IN in reason
    # libpq also takes the host from the query string when the netloc has none.
    assert isinstance(refusal("postgresql://u:p@/mm?host=db.example.com", {}), str)


@pytest.mark.parametrize("url", [
    # A ?host= overrides the netloc host in what libpq receives.
    "postgresql+psycopg2://u:p@localhost/mm?host=remote-db.invalid",
    # libpq tries a host list in order and falls through to the remote one.
    "postgresql+psycopg2://u:p@/mm?host=localhost&host=remote-db.invalid",
    "postgresql+psycopg2://u:p@/mm?host=localhost,remote-db.invalid",
    # hostaddr is the address actually dialled, whatever host says.
    "postgresql+psycopg2://u:p@localhost/mm?hostaddr=10.0.0.5",
    # A DNS name that merely starts with 127. is not loopback.
    "postgresql+psycopg2://u:p@127.remote-db.invalid/mm",
    # Targets the guard cannot see into.
    "postgresql+psycopg2://u:p@/mm?service=prod",
    "postgresql+psycopg2://u:p@localhost/mm?dsn=host%3Dremote-db.invalid",
])
def test_libpq_query_targets_refused(url):
    assert isinstance(refusal(url, {}), str), url


@pytest.mark.parametrize("env", [
    {"PGHOST": "remote-db.invalid"},
    {"PGHOST": "localhost,remote-db.invalid"},
    {"PGHOSTADDR": "10.1.2.3"},
    {"PGSERVICE": "prod"},
])
def test_libpq_env_fallbacks_refused(env):
    # With no host in the URL, libpq reads these; the guard must too.
    reason = refusal("postgresql+psycopg2://u:p@/mm", env)
    assert isinstance(reason, str), env
    assert next(iter(env)) in reason


def test_query_host_refusal_names_the_host():
    reason = refusal("postgresql+psycopg2://fixture_user:fixture-s3cret@/mm?host=remote-db.invalid", {})
    assert reason is not None
    assert "***@remote-db.invalid/mm" in reason
    assert "fixture-s3cret" not in reason and "fixture_user" not in reason


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


class _StubConfig:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def addinivalue_line(self, name: str, line: str) -> None:
        self.lines.append((name, line))


def test_conftest_guard_reads_resolved_settings(monkeypatch, request):
    """The realistic leak is a DATABASE_URL in .env, which reaches `settings`
    but not os.environ. The end-to-end test below sets the shell variable, so
    only this one tells the two apart."""
    from app.config import settings

    root = next(
        plugin for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == BACKEND / "conftest.py"
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv(OPT_IN, raising=False)
    monkeypatch.setattr(settings, "database_url", REMOTE)
    with pytest.raises(pytest.exit.Exception) as exc:
        root.pytest_configure(_StubConfig())
    assert exc.value.returncode == pytest.ExitCode.USAGE_ERROR
    assert "fixture-s3cret" not in str(exc.value)

    monkeypatch.setattr(settings, "database_url", "sqlite://")
    config = _StubConfig()
    root.pytest_configure(config)
    assert config.lines  # got past the guard to the marker registration


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


def test_fixture_seeder_refuses_remote_database():
    # run_full_seed overwrites company rows and deletes screener scores; its
    # __main__ is a way in that pytest's conftest never sees.
    result = subprocess.run(
        [sys.executable, "-m", "app.tests.fixtures.seed_demo_data"],
        cwd=BACKEND, env=_child_env(DATABASE_URL=REMOTE),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    _assert_redacted_refusal(result.stderr)


def test_fixture_seeder_raises_for_library_callers(monkeypatch):
    from app.config import settings
    from app.tests.dbguard import NonLocalDatabase
    from app.tests.fixtures import seed_demo_data

    monkeypatch.delenv(OPT_IN, raising=False)
    monkeypatch.setattr(settings, "database_url", REMOTE)
    with pytest.raises(NonLocalDatabase):
        seed_demo_data.run_full_seed()


def test_measure_live_cost_refuses_remote_database(tmp_path):
    # It seeds the test fixtures and runs smoke twice; the refusal has to
    # come before any of that, and before it unlinks ./marketmosaic.db.
    sentinel = tmp_path / "marketmosaic.db"
    sentinel.write_text("untouched")
    result = subprocess.run(
        [sys.executable, str(BACKEND / "scripts" / "measure_live_cost.py")],
        cwd=tmp_path, env=_child_env(DATABASE_URL=REMOTE),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    _assert_redacted_refusal(result.stderr)
    assert sentinel.read_text() == "untouched"


def test_measure_live_cost_fails_when_smoke_fails(monkeypatch, tmp_path):
    # A failed smoke run writes no cost rows, and 0/0 used to read as a
    # 0.0% ratio and exit 0.
    import scripts.measure_live_cost as measure
    import scripts.smoke_test as smoke

    monkeypatch.chdir(tmp_path)  # it unlinks ./marketmosaic.db
    monkeypatch.setattr(smoke, "main", lambda: 1)
    assert measure.main() == 1


def _tickers(url: str) -> list[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return sorted(conn.execute(select(Company.ticker)).scalars())
    finally:
        engine.dispose()


def _seed(url: str, ticker: str) -> None:
    engine = create_engine(url)
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(Company.__table__.insert(), [{
                "ticker": ticker, "company_name": ticker, "sector": "Tech", "industry": "Software",
            }])
    finally:
        engine.dispose()


def test_migrate_refuses_nonempty_target_and_requires_source(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    target_url = f"sqlite:///{target}"
    _seed(f"sqlite:///{source}", "SRCONLY")
    _seed(target_url, "KEEPME")
    # The script imports app.* (and so builds the app engine): keep that on a
    # scratch file rather than whatever DATABASE_URL this session inherited.
    base = _child_env(DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}", TARGET_POSTGRES_URL=target_url)

    def run(**overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "scripts.migrate_sqlite_to_postgres"],
            cwd=BACKEND, env={**base, **overrides}, capture_output=True, text=True, timeout=120,
        )

    refused = run(SOURCE_SQLITE=str(source))
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "not empty" in refused.stderr
    assert _tickers(target_url) == ["KEEPME"]

    no_source = run()
    assert no_source.returncode == 1
    assert "SOURCE_SQLITE must be set" in no_source.stderr
    assert _tickers(target_url) == ["KEEPME"]

    # A source a test run has written into is refused even for an empty
    # target: that is exactly how the fixtures reached production.
    dirty, empty = tmp_path / "dirty.db", tmp_path / "empty.db"
    _seed(f"sqlite:///{dirty}", "TSTONE")
    fixture_source = run(SOURCE_SQLITE=str(dirty), TARGET_POSTGRES_URL=f"sqlite:///{empty}")
    assert fixture_source.returncode == 1, fixture_source.stdout + fixture_source.stderr
    assert "TSTONE" in fixture_source.stderr
    assert not inspect(create_engine(f"sqlite:///{empty}")).get_table_names()

    # The deliberate overwrite still works.
    allowed = run(SOURCE_SQLITE=str(source), MM_MIGRATE_OVERWRITE_TARGET="1")
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert _tickers(target_url) == ["SRCONLY"]
