"""Database-isolation guard for the test session and the seeding scripts.

Fixture tickers (TSTONE, AUDA, RTRIP, ...) and the outcome rows scored on
them sit in production Postgres (FIX-003) because the test suite and the
sqlite-to-postgres migration once shared one database file, and nothing
checked where a run was pointed. The suite is not read-only: several
modules wipe whole tables without a filter (test_history_service clears
financial_periods, test_dcf_persistence clears dcf_models, ...), so a run
whose DATABASE_URL resolves to a shared database destroys and plants rows
there.

netguard cannot catch this. It patches Python's socket module, and psycopg2
connects through libpq in C, so a remote Postgres connection sails past it.
This guard therefore checks the resolved URL itself and refuses anything
that is not sqlite or a loopback/Unix-socket Postgres. `MM_ALLOW_REMOTE_TEST_DB=1`
opts in for a database that is known to be disposable (e.g. a compose
service reached by its container name).

The refusal message never carries the URL's credentials: it is printed to
terminals and CI logs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

OPT_IN = "MM_ALLOW_REMOTE_TEST_DB"

_LOOPBACK_NAMES = {"localhost", "::1", "[::1]"}


def redacted(url: URL) -> str:
    """Render `url` without user, password or query string.

    `URL.render_as_string(hide_password=True)` is not enough: it keeps the
    username and any `?password=` query parameter.
    """
    creds = "***@" if (url.username or url.password) else ""
    port = f":{url.port}" if url.port else ""
    database = f"/{url.database}" if url.database else ""
    return f"{url.drivername}://{creds}{url.host or ''}{port}{database}"


def _host(url: URL) -> str:
    """The host libpq will reach: the URL host, else a `?host=` parameter."""
    if url.host:
        return url.host
    host = url.query.get("host", "")
    if isinstance(host, tuple):
        host = host[0] if host else ""
    return host


def _is_local(host: str) -> bool:
    # An empty host means libpq's default Unix socket; a leading "/" is an
    # explicit socket directory. Both are on this machine by construction.
    if host == "" or host.startswith("/"):
        return True
    return host.lower() in _LOOPBACK_NAMES or host.startswith("127.")


def refusal(database_url: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return why `database_url` must not be used by tests, or None if it may."""
    env = os.environ if environ is None else environ
    if env.get(OPT_IN) == "1":
        return None
    try:
        url = make_url(database_url)
    except (ArgumentError, ValueError):
        return (
            "refusing to run: DATABASE_URL could not be parsed, so its target "
            "cannot be shown to be local. The value is withheld because it may "
            f"carry credentials. Fix it, or set {OPT_IN}=1 if the database is disposable."
        )
    if url.get_backend_name() == "sqlite" or _is_local(_host(url)):
        return None
    return (
        f"refusing to run against a non-local database ({redacted(url)}). "
        "The suite deletes and rewrites rows (e.g. test_history_service wipes "
        "financial_periods; test_dcf_persistence wipes dcf_models) and netguard "
        "cannot see libpq connections. DATABASE_URL resolves from the shell, then "
        "backend/.env, then the repo .env. Point it at a unique sqlite file "
        "(DATABASE_URL=sqlite:////tmp/<unique>.db) or set "
        f"{OPT_IN}=1 if this database is disposable."
    )
