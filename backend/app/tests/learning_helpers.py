"""Test helpers for the W7 learning ledger.

Ledger tests assert exact counts, so each one gets a private sqlite file
(`learning_db`), with every module that opens its own session pointed at it —
the same approach as `eligibility_helpers.isolated_sessions`. Memos are real,
valid `StockMemoOut` bodies (the ledger presents them through W2a), and
snapshots are marked eligible or not in the W6 ledger explicitly.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import Company, CompanyIndustryClassification, MemoOutcome, MemoSnapshot, TaxonomyVersion
from app.tests.eligibility_helpers import isolated_sessions, mark
from app.tests.factories import make_memo


def learning_db(tmp_path: Any, monkeypatch: Any, *extra: Any) -> Any:
    from app.api import routes_learning_admin
    from app.learning import control, ledger
    from app.services import filing_memory, industry_classification, memo_store
    from app.services import postmortem_service as pm

    sessions, engine = isolated_sessions(
        tmp_path, monkeypatch, ledger, control, pm, industry_classification, filing_memory,
        memo_store, routes_learning_admin, *extra,
    )
    control._cache_clear()
    return sessions, engine


def memo_body(ticker: str, *, rating: str = "Bullish", sector: str = "Technology",
              thesis: str | None = None, pm_view: str | None = None, **overrides: Any) -> dict[str, Any]:
    memo = make_memo(
        ticker=ticker, company_name=f"{ticker} Corp", sector=sector, rating_label=rating,
        one_sentence_thesis=thesis if thesis is not None else f"{ticker} is cheap versus growth as churn falls.",
        final_pm_view=pm_view if pm_view is not None else "Margins expand as bundles roll out; churn keeps falling.",
        **overrides,
    )
    body = memo.model_dump(mode="json")
    body["generation_mode"] = "live"
    return body


def add_snapshot(
    db: Session, ticker: str, *, generated_at: datetime, version: int = 1, eligible: bool | None = True,
    body: dict[str, Any] | None = None, **memo: Any,
) -> MemoSnapshot:
    snap = MemoSnapshot(
        ticker=ticker, version=version, trigger="full_reanalysis",
        memo_json=body if body is not None else memo_body(ticker, **memo),
        revision_log=[], generated_at=generated_at,
    )
    db.add(snap)
    db.commit()
    if eligible is not None:
        mark(db, snap.id, eligible=eligible,
             reason="live_generation" if eligible else "demo_dev_copy_2026_05_04")
    return snap


def add_outcome(db: Session, snap: MemoSnapshot, *, horizon: int = 90, alpha: float | None = 0.1,
                evaluated_at: datetime | None = None) -> MemoOutcome:
    row = MemoOutcome(
        memo_snapshot_id=snap.id, ticker=snap.ticker, rating_at_memo="Bullish", confidence_at_memo=60.0,
        price_at_memo=100.0, horizon_days=horizon,
        evaluated_at=evaluated_at or snap.generated_at + timedelta(days=horizon + 1),
        forward_return=None if alpha is None else alpha + 0.05, benchmark_return=0.05, alpha=alpha,
        thesis_held=None, note="seeded",
    )
    db.add(row)
    db.commit()
    return row


def add_company(db: Session, ticker: str, sector: str = "Technology") -> Company:
    row = Company(ticker=ticker, company_name=f"{ticker} Corp", sector=sector, industry="Software")
    db.add(row)
    db.commit()
    return row


def classify(db: Session, ticker: str, *, state: str = "mapped", group: str | None = "4510") -> None:
    """A current classification under an active taxonomy version."""
    version = db.query(TaxonomyVersion).filter_by(is_active=True).first()
    if version is None:
        version = TaxonomyVersion(version_key="test-2026", is_active=True, effective_from=date(2023, 1, 1))
        db.add(version)
        db.flush()
    db.add(CompanyIndustryClassification(
        ticker=ticker, taxonomy_version_id=version.id, sector_code=(group or "45")[:2],
        industry_group_code=group, state=state, is_current=True,
    ))
    db.commit()
