"""A model column added after its table exists must still reach the DB.

Found in production 2026-08-15 by reading worker logs: `postmortem_loop`
had been raising every night at 03:00 UTC with

    psycopg2.errors.UndefinedColumn:
    column memo_outcomes.regime_at_memo does not exist

`regime_at_memo` was added to `MemoOutcome` in Wave 10; `memo_outcomes`
was created in Wave 4A. `Base.metadata.create_all` creates missing
TABLES and never alters an existing one, so the column simply never
appeared in a long-lived database — and nightly postmortems had been
silently dead ever since.

Doubly invisible: the loop raises before reaching `record_run`, and
`/api/admin/cron-health` only listed loops that had reported at least
once, so a loop that never succeeded was absent rather than flagged.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from app.database import Base, engine, init_db, reconcile_missing_columns


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa_inspect(engine).get_columns(table)}


def _simulate_drift() -> None:
    """Drop `memo_outcomes.regime_at_memo`, reproducing the production state.

    The index has to go first: SQLite refuses to drop a column an index
    still references ("error in index ... after drop column"). Postgres
    cascades, so this is a test-harness detail, not a difference in what
    is being reproduced — the end state (table exists, column absent) is
    identical either way.
    """
    with engine.begin() as conn:
        try:
            conn.execute(text("DROP INDEX IF EXISTS ix_memo_outcomes_regime_at_memo"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE memo_outcomes DROP COLUMN regime_at_memo"))
        except Exception:
            pass  # already absent


def test_missing_nullable_column_is_restored():
    """The production case: a nullable column added to an existing table."""
    init_db()
    assert "regime_at_memo" in _columns("memo_outcomes")

    _simulate_drift()
    assert "regime_at_memo" not in _columns("memo_outcomes"), "setup failed"

    added = reconcile_missing_columns()

    assert "memo_outcomes.regime_at_memo" in added
    assert "regime_at_memo" in _columns("memo_outcomes")


def test_the_repaired_column_is_actually_queryable():
    """Restoring the name is not enough — the query that was crashing in
    production has to work again."""
    _simulate_drift()
    reconcile_missing_columns()

    from app.database import SessionLocal
    from app.models import MemoOutcome
    with SessionLocal() as db:
        # Selecting the column is exactly what postmortem_service._due_memos
        # did when it blew up.
        db.query(MemoOutcome.regime_at_memo).limit(1).all()


def test_reconcile_is_idempotent():
    """Runs on every boot; a second pass must be a silent no-op."""
    init_db()
    assert reconcile_missing_columns() == []


def test_no_drift_between_models_and_a_freshly_created_schema():
    """Guards the whole model layer, not just memo_outcomes: after
    init_db every ORM column must exist. Catches a model change that
    create_all cannot express (and would otherwise only surface as a
    runtime UndefinedColumn on some rarely-hit path)."""
    init_db()
    insp = sa_inspect(engine)
    existing = set(insp.get_table_names())
    missing: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            missing.append(f"{table.name} (whole table)")
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        missing += [f"{table.name}.{c.name}" for c in table.columns if c.name not in have]
    assert not missing, "ORM columns absent from the database:\n  " + "\n  ".join(missing)


def test_index_is_restored_with_the_column():
    """`regime_at_memo` is `index=True`. Restoring the column but not its
    index would trade a crash for a silent performance cliff."""
    _simulate_drift()
    reconcile_missing_columns()

    indexed = {
        col
        for idx in sa_inspect(engine).get_indexes("memo_outcomes")
        for col in (idx.get("column_names") or [])
    }
    assert "regime_at_memo" in indexed


# ---------------------------------------------------------------------------
# FEAT-002 — the additive account columns and tables
# ---------------------------------------------------------------------------

FEAT_002_TABLES = (
    "users", "subscriptions", "usage_counters", "usage_events", "admin_overrides",
    "billing_webhook_events", "rate_limit_windows", "active_actions",
    "public_samples", "analytics_events",
)

# (table, column, index name or None) — columns added to tables that
# already exist in every long-lived database.
FEAT_002_ADDED_COLUMNS = (
    ("regen_jobs", "requested_by_user_id", "ix_regen_jobs_requested_by_user_id"),
    ("regen_jobs", "usage_event_id", None),
    ("llm_call_logs", "user_id", "ix_llm_call_logs_user_id"),
    ("llm_call_logs", "feature", None),
)


def test_feat_002_tables_are_created():
    init_db()
    existing = set(sa_inspect(engine).get_table_names())
    missing = [t for t in FEAT_002_TABLES if t not in existing]
    assert not missing, missing


def _drop_column(table: str, column: str, index: str | None) -> None:
    with engine.begin() as conn:
        if index:
            try:
                conn.execute(text(f"DROP INDEX IF EXISTS {index}"))
            except Exception:
                pass
        try:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        except Exception:
            pass


@pytest.mark.parametrize("table,column,index", FEAT_002_ADDED_COLUMNS)
def test_feat_002_added_columns_are_repaired_on_a_live_table(table, column, index):
    """The production scenario for this feature: `regen_jobs` and
    `llm_call_logs` exist in every deployment; the new nullable columns
    must be added by `reconcile_missing_columns`, not by a migration
    nobody runs."""
    init_db()
    assert column in _columns(table)
    _drop_column(table, column, index)
    assert column not in _columns(table), "setup failed"

    added = reconcile_missing_columns()
    assert f"{table}.{column}" in added
    assert column in _columns(table)
    if index:
        indexed = {
            col for idx in sa_inspect(engine).get_indexes(table)
            for col in (idx.get("column_names") or [])
        }
        assert column in indexed


def test_feat_002_columns_are_all_nullable_or_defaulted():
    """Nothing added to an existing table may be NOT NULL without a
    default — that is the one shape `reconcile_missing_columns` refuses."""
    from app.models import LLMCallLog, RegenJob
    for model, names in ((RegenJob, ("requested_by_user_id", "usage_event_id")),
                         (LLMCallLog, ("user_id", "feature"))):
        for name in names:
            col = model.__table__.columns[name]
            assert col.nullable or col.default is not None or col.server_default is not None, f"{model.__tablename__}.{name}"


# ---------------------------------------------------------------------------
# FEAT-001 — the chart-commentary cache table
# ---------------------------------------------------------------------------

def test_feat_001_chart_commentaries_is_created():
    init_db()
    assert "chart_commentaries" in set(sa_inspect(engine).get_table_names())
    have = _columns("chart_commentaries")
    assert {"cache_key", "fingerprint", "user_id", "output", "degraded", "degraded_reason", "created_at"} <= have


# The row's identity, set on every insert and created with the table —
# the same standing as `users.external_id`. Everything else must be
# addable to a live table later by `reconcile_missing_columns`, the only
# migration path this repo has.
_CHART_COMMENTARY_IDENTITY = {"id", "cache_key", "fingerprint"}


def test_feat_001_chart_commentaries_columns_are_all_nullable_or_defaulted():
    from app.models import ChartCommentary
    names = {c.name for c in ChartCommentary.__table__.columns}
    assert _CHART_COMMENTARY_IDENTITY <= names
    for col in ChartCommentary.__table__.columns:
        if col.name in _CHART_COMMENTARY_IDENTITY:
            continue
        assert col.nullable or col.default is not None or col.server_default is not None, col.name


def test_feat_001_chart_commentaries_columns_are_repaired_on_a_live_table():
    """The production scenario once the table exists: a column added
    later must be restored, with its index, by the boot-time reconcile."""
    init_db()
    assert "degraded_reason" in _columns("chart_commentaries")
    _drop_column("chart_commentaries", "user_id", "ix_chart_commentaries_user_id")
    assert "user_id" not in _columns("chart_commentaries"), "setup failed"

    added = reconcile_missing_columns()
    assert "chart_commentaries.user_id" in added
    assert "user_id" in _columns("chart_commentaries")
    indexed = {
        col for idx in sa_inspect(engine).get_indexes("chart_commentaries")
        for col in (idx.get("column_names") or [])
    }
    assert {"user_id", "fingerprint", "cache_key"} <= indexed
