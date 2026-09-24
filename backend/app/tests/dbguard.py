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

"Where libpq connects" is not just the URL's netloc. SQLAlchemy passes every
query parameter to libpq, a `?host=` there overrides the netloc host, a host
list is tried in order, `hostaddr` beats `host` for the actual address, and
when the URL names no host at all libpq falls back to PGHOST / PGHOSTADDR /
PGSERVICE. So every one of those is checked, and anything the guard cannot
see into (a service-file entry, a raw `dsn`) is refused.

The refusal message never carries the URL's credentials: it is printed to
terminals and CI logs.
"""
from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

OPT_IN = "MM_ALLOW_REMOTE_TEST_DB"

# Symbols the suite writes and never cleans up afterwards, so any database
# a test run has touched holds some of them. migrate_sqlite_to_postgres
# refuses a source containing one. Exact names, not a pattern: CHAIN, AUDA,
# AUDB and AUDC also reached production (FIX-003) but are left out because
# they could be, or are, real listings (AUDC is AudioCodes).
FIXTURE_TICKERS = frozenset({
    "TSTONE", "TSTONE2", "TSTAA", "TSTBB", "TSTNEU", "TSTRFL", "TSTADM",
    "TSTSHORT", "TSTHALF", "TSTPATCH", "TSTNOMA", "TSTCAP", "RTRIP",
    "AUDN0", "AUDN1", "AUDN2", "AUDN3", "AUDN4", "ASOFT1",
})

# Query keys whose target libpq resolves somewhere the guard cannot read:
# a pg_service.conf entry, or a whole connection string of its own.
_OPAQUE_KEYS = ("service", "dsn")


class NonLocalDatabase(RuntimeError):
    """Raised by `ensure_local` for a database the guard refuses."""


def _values(raw: str | tuple[str, ...] | None) -> list[str]:
    """Flatten one URL-query / env value into libpq's comma-separated entries."""
    if raw is None:
        return []
    items = (raw,) if isinstance(raw, str) else raw
    return [part.strip() for item in items for part in item.split(",")]


def _strip_port(entry: str) -> str:
    # SQLAlchemy accepts `?host=db:5432`; a bracketed IPv6 literal may carry
    # one too. A bare IPv6 literal has several colons and no port.
    if entry.startswith("["):
        return entry[1:entry.find("]")] if "]" in entry else entry
    if entry.count(":") == 1:
        return entry.split(":", 1)[0]
    return entry


def _is_local(host: str) -> bool:
    # An empty entry is libpq's default Unix socket; a leading "/" is an
    # explicit socket directory. Either is this machine.
    if host == "" or host.startswith("/"):
        return True
    if host.lower() == "localhost":
        return True
    # Only a literal loopback address counts: a DNS name such as
    # 127.example.com resolves wherever its owner likes.
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _targets(url: URL, env: Mapping[str, str]) -> list[tuple[str, str]]:
    """Every (source, host) libpq may connect to for `url`.

    Netloc and `?host=` are both checked rather than modelling which one
    wins: refusing on either is never less safe than libpq's choice.
    """
    out: list[tuple[str, str]] = []
    if url.host:
        out.append(("URL host", url.host))
    out += [("?host=", _strip_port(h)) for h in _values(url.query.get("host"))]
    if not out and url.get_backend_name() == "postgresql":
        out += [("PGHOST", _strip_port(h)) for h in _values(env.get("PGHOST"))]
    hostaddrs = [("?hostaddr=", h) for h in _values(url.query.get("hostaddr"))]
    if not hostaddrs and url.get_backend_name() == "postgresql":
        hostaddrs = [("PGHOSTADDR", h) for h in _values(env.get("PGHOSTADDR"))]
    return out + hostaddrs


def redacted(url: URL) -> str:
    """Render `url` without user, password or query string.

    `URL.render_as_string(hide_password=True)` is not enough: it keeps the
    username and any `?password=` query parameter. When the netloc has no
    host the `?host=` value is shown instead, so the message still says
    where the run was pointed.
    """
    creds = "***@" if (url.username or url.password) else ""
    host = url.host or ",".join(_values(url.query.get("host")))
    port = f":{url.port}" if url.port else ""
    database = f"/{url.database}" if url.database else ""
    return f"{url.drivername}://{creds}{host}{port}{database}"


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
    if url.get_backend_name() == "sqlite":
        return None
    unseen = [f"?{k}=" for k in _OPAQUE_KEYS if k in url.query]
    if url.get_backend_name() == "postgresql" and env.get("PGSERVICE"):
        unseen.append("PGSERVICE")
    remote = [f"{host} (from {source})" for source, host in _targets(url, env) if not _is_local(host)]
    if not unseen and not remote:
        return None
    where = "; ".join(
        ([f"non-local target {', '.join(remote)}"] if remote else [])
        + ([f"target set by {', '.join(unseen)}, which this guard cannot inspect"] if unseen else [])
    )
    return (
        f"refusing to run against a non-local database ({redacted(url)}: {where}). "
        "The suite deletes and rewrites rows (e.g. test_history_service wipes "
        "financial_periods; test_dcf_persistence wipes dcf_models) and netguard "
        "cannot see libpq connections. DATABASE_URL resolves from the shell, then "
        "backend/.env, then the repo .env. Point it at a unique sqlite file "
        "(DATABASE_URL=sqlite:////tmp/<unique>.db) or set "
        f"{OPT_IN}=1 if this database is disposable."
    )


def ensure_local(database_url: str, environ: Mapping[str, str] | None = None) -> None:
    """`refusal` for library code: raise instead of returning the reason."""
    reason = refusal(database_url, environ)
    if reason:
        raise NonLocalDatabase(reason)
