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

from sqlalchemy import inspect as sa_inspect, text

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
