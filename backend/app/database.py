"""Database engine + session helpers.

Defaults to SQLite for zero-config local. Switch to Postgres by setting
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


def _build_engine():
    url = settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(url, future=True, connect_args=connect_args)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Imperative session context for scripts (seeders, tests)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables from imported models. Called at app startup."""
    from . import models  # noqa: F401  -- ensures models register on Base
    Base.metadata.create_all(bind=engine)
    _ensure_added_columns()


# Lightweight in-place migrations. We don't run Alembic — instead each
# new column lands here as an idempotent ALTER TABLE that no-ops when
# the column already exists. The DDL templates use `{bool_false}` /
# `{bool_true}` placeholders so the per-dialect renderer below can emit
# `false` (Postgres) or `0` (SQLite) — Postgres rejects integer literals
# as BOOLEAN defaults and the failure silently no-ops the migration,
# leaving the column missing while the ORM thinks it exists.
_ADDITIVE_COLUMNS = [
    # (table, column, ddl_template) — DDL is the column part of ALTER
    # TABLE, not the full statement.
    ("companies", "auto_update_memo",
     "BOOLEAN NOT NULL DEFAULT {bool_false}"),
]


def _ensure_added_columns() -> None:
    """Apply any additive ALTER TABLE migrations the model expects.

    Per-dialect DDL rendering — Postgres BOOLEAN columns need `false`
    as the default literal; SQLite accepts `0`. Catches both 'column
    already exists' (re-run, normal) and 'table does not exist'
    (initial boot before create_all races) — neither is fatal. Real
    schema errors are logged at ERROR level (not warning) so prod log
    scraping catches them.
    """
    import logging
    from sqlalchemy import text
    log = logging.getLogger(__name__)
    dialect = engine.dialect.name  # "postgresql", "sqlite", "mysql", ...
    if dialect == "postgresql":
        params = {"bool_false": "false", "bool_true": "true"}
    else:
        params = {"bool_false": "0", "bool_true": "1"}
    with engine.begin() as conn:
        for table, column, ddl_template in _ADDITIVE_COLUMNS:
            ddl = ddl_template.format(**params)
            stmt = f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
            try:
                conn.execute(text(stmt))
                log.info("init_db: added column %s.%s (%s)", table, column, dialect)
            except Exception as exc:
                msg = str(exc).lower()
                # SQLite: "duplicate column name"; Postgres: "already exists"
                if "duplicate column" in msg or "already exists" in msg:
                    continue
                if "no such table" in msg or "does not exist" in msg:
                    continue
                # Anything else IS a real problem — log at ERROR so it
                # surfaces in prod monitoring instead of silently leaving
                # the column unmigrated.
                log.error(
                    "init_db: ALTER TABLE %s.%s failed (%s): %s",
                    table, column, dialect, exc,
                )
