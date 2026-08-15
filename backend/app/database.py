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
    reconcile_missing_columns()
    ensure_pgvector()


def reconcile_missing_columns() -> list[str]:
    """Add ORM-declared columns that are absent from existing tables.

    `create_all` creates missing TABLES but never touches an existing
    one, so a column added to a model after its table was first created
    simply never appears in a long-lived database. Every query naming it
    then fails — not at deploy, but whenever that code path next runs.

    Found in production 2026-08-15: `postmortem_loop` had been raising
    every night at 03:00 UTC with

        psycopg2.errors.UndefinedColumn:
        column memo_outcomes.regime_at_memo does not exist

    `regime_at_memo` was added to `MemoOutcome` in Wave 10, long after
    `memo_outcomes` was created in Wave 4A. It was invisible twice over:
    the loop raises before reaching `record_run`, and `/api/admin/
    cron-health` only listed loops that had reported at least once, so a
    loop that never succeeded didn't appear at all.

    `_ADDITIVE_COLUMNS` above is the hand-maintained version of this, and
    the drift is exactly what it misses — someone has to remember to add
    an entry. This reconciles automatically against the ORM metadata,
    which is the thing that actually changed.

    Only adds columns that are safe to add to a populated table: nullable,
    or carrying a default. A missing NOT NULL column with no default
    cannot be added without inventing values, so it is logged at ERROR
    and left alone. Primary keys likewise.

    Returns the list of `table.column` strings added.
    """
    import logging
    from sqlalchemy import inspect as sa_inspect, text
    log = logging.getLogger(__name__)

    from . import models  # noqa: F401
    added: list[str] = []
    try:
        insp = sa_inspect(engine)
        existing_tables = set(insp.get_table_names())
    except Exception as exc:  # pragma: no cover — inspection unavailable
        log.warning("schema reconcile skipped (inspection failed): %s", exc)
        return added

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all just made it, with every column
        try:
            have = {c["name"] for c in insp.get_columns(table.name)}
        except Exception as exc:  # pragma: no cover
            log.warning("schema reconcile skipped for %s: %s", table.name, exc)
            continue
        for col in table.columns:
            if col.name in have:
                continue
            if col.primary_key:
                log.error(
                    "schema drift: %s.%s is a missing PRIMARY KEY — cannot be "
                    "added automatically; needs a manual migration",
                    table.name, col.name,
                )
                continue
            if not col.nullable and col.default is None and col.server_default is None:
                log.error(
                    "schema drift: %s.%s is NOT NULL with no default — cannot be "
                    "added to a populated table automatically; needs a manual "
                    "migration or a default",
                    table.name, col.name,
                )
                continue
            try:
                col_type = col.type.compile(dialect=engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}"
                    ))
                added.append(f"{table.name}.{col.name}")
                # WARNING, not INFO: drift means a deploy shipped a model
                # change without one, and that is worth noticing even
                # though it just got repaired.
                log.warning(
                    "schema drift repaired: added %s.%s (%s)",
                    table.name, col.name, col_type,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    continue  # raced with another process; fine
                log.error(
                    "schema drift: failed to add %s.%s: %s",
                    table.name, col.name, exc,
                )

    if added:
        # Recreate indexes for the repaired columns — the column alone
        # restores correctness, but an indexed column silently losing its
        # index is a performance cliff that would be hard to attribute
        # later. `checkfirst` makes this a no-op for existing indexes.
        for table in Base.metadata.sorted_tables:
            for index in table.indexes:
                try:
                    index.create(bind=engine, checkfirst=True)
                except Exception as exc:  # pragma: no cover — best effort
                    log.warning("could not ensure index %s: %s", index.name, exc)
    return added


def ensure_pgvector() -> bool:
    """Add the pgvector column + HNSW index to `doc_chunks` (Postgres only).

    Kept out of `_ADDITIVE_COLUMNS` because it is not a plain ALTER: it
    needs `CREATE EXTENSION` first, and an index build after. Structured
    as a *derived* column rather than a conversion of the existing JSON
    `embedding` column, which matters for three reasons:

      - The ORM model stays dialect-agnostic, so SQLite (local dev, CI)
        needs no pgvector import and no divergent mapping.
      - Nothing is destructive or one-way — the JSON column remains the
        source of truth and is still what the streaming search path and
        the embedding-dim mismatch check read.
      - Every failure mode (extension unavailable, no CREATE privilege,
        index build OOM) degrades to the streaming path instead of
        breaking retrieval.

    Returns True when the column exists and is usable afterwards.
    Idempotent — safe on every boot.
    """
    import logging
    from sqlalchemy import text
    log = logging.getLogger(__name__)
    if engine.dialect.name != "postgresql":
        return False
    # Imported here (not at module scope) so this module keeps working
    # when the embeddings service is unavailable for any reason.
    from .services.embeddings import EMBEDDING_DIM

    # Startup does the CHEAP DDL only. Both of these are effectively
    # instant: `CREATE EXTENSION` is metadata, and adding a nullable
    # column does not rewrite the table.
    #
    # The HNSW index is deliberately NOT built here. `CREATE INDEX ...
    # USING hnsw` over a large `doc_chunks` takes minutes and holds the
    # transaction the whole time — and this runs inside `init_db()`, on
    # the startup path, before the service can answer `/health`. Render
    # fails a deploy whose health check doesn't come up in time, so
    # building the index here would have turned the first post-deploy
    # boot into a failed deploy. It lives in
    # `scripts/backfill_pgvector.py` instead, which also runs it *after*
    # the rows are populated (much faster than building on an empty
    # table and inserting into it).
    #
    # Search is correct without the index — Postgres just does an exact
    # scan, which still returns `top_k` rows and still avoids pulling the
    # corpus into Python. The index is a latency optimisation, not a
    # correctness dependency.
    steps = [
        ("extension", "CREATE EXTENSION IF NOT EXISTS vector"),
        ("column",
         f"ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({EMBEDDING_DIM})"),
    ]
    ok = True
    for label, stmt in steps:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg:
                continue
            # A missing extension is an expected, recoverable state on a
            # managed Postgres without pgvector — warn, don't error.
            log.warning("pgvector %s step skipped: %s", label, exc)
            ok = False
            # No extension means the column cannot be added either.
            break
    if ok:
        log.info(
            "init_db: pgvector column ready on doc_chunks "
            "(run scripts/backfill_pgvector to populate + build the HNSW index)"
        )
    return ok


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
