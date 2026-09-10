"""Shared seeding for the Phase 6 slice-C tests (service, queue, loop,
routes, evaluation).

Everything is synthetic and deterministic (`random.Random(seed)`): a small
universe of test companies tagged `data_only` (so they never enter the
real `auto_analysis` universe another suite might read), four fiscal
years of every statement line the feature engine knows, availability
dates from the annual lag rule, and month-end closes in `price_month_ends`.
Tests pass the tickers explicitly to `run_scorecard`, purge before and
after, and never touch a provider.
"""
from __future__ import annotations

import random
import uuid
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any

from app.config import settings
from app.database import SessionLocal
from app.models import (
    Company,
    FinancialPeriod,
    PriceMonthEnd,
    ScorecardEvaluation,
    ScorecardRun,
    ScorecardScore,
    ScorecardVersion,
)

REQUESTED_BY = "test-scorecard"

# Plausible magnitudes for one "unit" company; every line the engine reads.
BASE_LINES: dict[str, dict[str, float]] = {
    "income": {
        "revenue": 1000.0, "cost_of_revenue": 600.0, "gross_profit": 400.0, "r_and_d": 50.0, "sga": 100.0,
        "operating_income": 250.0, "ebit": 250.0, "ebitda": 300.0, "net_income": 180.0, "eps_diluted": 1.8,
        "weighted_avg_shares_diluted": 100.0, "interest_expense": 10.0, "pretax_income": 230.0, "tax_expense": 50.0,
    },
    "balance": {
        "total_assets": 2000.0, "total_liabilities": 1200.0, "shareholders_equity": 800.0,
        "cash_and_equivalents": 200.0, "short_term_investments": 50.0, "short_term_debt": 100.0,
        "long_term_debt": 400.0, "total_debt": 500.0, "goodwill": 150.0, "current_assets": 600.0,
        "current_liabilities": 400.0,
    },
    "cash": {
        "cash_from_operations": 260.0, "capex": -80.0, "free_cash_flow": 180.0,
        "depreciation_and_amortization": 50.0, "dividends_paid": -40.0, "share_repurchases": -30.0,
        "stock_based_compensation": 20.0,
    },
}
FETCHED_AT = datetime(2026, 9, 1, 12, 0, 0)


def month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def prev_month_end(d: date, back: int = 1) -> date:
    y, m = d.year, d.month
    for _ in range(back):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return date(y, m, monthrange(y, m)[1])


def next_month_end(d: date) -> date:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(y, m, monthrange(y, m)[1])


def seed_universe(
    tickers_by_sector: dict[str, list[str]],
    *,
    fiscal_years: tuple[int, ...] = (2022, 2023, 2024, 2025),
    price_through: date = date(2026, 6, 30),
    price_months: int = 20,
    seed: int = 0,
) -> list[str]:
    """Companies + four fiscal years of statements + month-end prices.
    Returns the tickers seeded (sorted)."""
    rng = random.Random(seed)
    lag = timedelta(days=int(settings.scorecard_pit_lag_annual_days))
    seeded: list[str] = []
    with SessionLocal() as db:
        for sector, tickers in tickers_by_sector.items():
            for t in tickers:
                seeded.append(t)
                scale = rng.uniform(0.5, 4.0)
                growth = rng.uniform(0.92, 1.25)
                db.merge(Company(
                    ticker=t, company_name=f"{t} Test Co", exchange="TEST", sector=sector, industry="Test",
                    universe_tier="data_only", beta=rng.uniform(0.6, 1.6), shares_outstanding=None,
                ))
                for i, fy in enumerate(sorted(fiscal_years)):
                    g = growth ** i
                    pe = date(fy, 12, 31)
                    for statement, lines in BASE_LINES.items():
                        for line, base in lines.items():
                            jitter = 1.0 + rng.uniform(-0.15, 0.15)
                            if line == "eps_diluted":
                                value = base * g * jitter
                            elif line == "weighted_avg_shares_diluted":
                                value = base * scale * (1.0 + 0.01 * i)
                            else:
                                value = base * scale * g * jitter
                            db.add(FinancialPeriod(
                                ticker=t, period=f"FY{fy}", period_end=pe, fiscal_year=fy, fiscal_quarter=None,
                                statement=statement, line_item=line, value=value, source="test",
                                fetched_at=FETCHED_AT, available_at=pe + lag, available_at_source="lag_rule",
                            ))
                price = rng.uniform(20.0, 200.0)
                me = month_end(price_through)
                for k in range(price_months):
                    m = prev_month_end(me, k)
                    close = price * (1.0 + rng.uniform(-0.06, 0.06)) ** k
                    db.add(PriceMonthEnd(
                        ticker=t, month_end=m, price_date=m, close=close, adjusted_close=close,
                        source="test", fetched_at=FETCHED_AT,
                    ))
        db.commit()
    return sorted(seeded)


def purge(tickers: list[str] | tuple[str, ...] = (), *, versions: tuple[str, ...] = (),
          requested_by: tuple[str, ...] = (REQUESTED_BY,)) -> None:
    """Remove everything a test seeded: rows by ticker, runs by requester,
    and (optionally) every row under a synthetic version key."""
    tickers = list(tickers)
    with SessionLocal() as db:
        for model in (ScorecardVersion, ScorecardRun, ScorecardScore, ScorecardEvaluation, PriceMonthEnd,
                      FinancialPeriod, Company):
            model.__table__.create(bind=db.get_bind(), checkfirst=True)
        if tickers:
            for model in (ScorecardScore, PriceMonthEnd, FinancialPeriod):
                db.query(model).filter(model.ticker.in_(tickers)).delete(synchronize_session=False)
            db.query(Company).filter(Company.ticker.in_(tickers)).delete(synchronize_session=False)
        if versions:
            for model in (ScorecardScore, ScorecardEvaluation, ScorecardRun, ScorecardVersion):
                db.query(model).filter(model.version_key.in_(versions)).delete(synchronize_session=False)
        if requested_by:
            ids = [r[0] for r in db.query(ScorecardRun.id).filter(ScorecardRun.requested_by.in_(requested_by)).all()]
            if ids:
                db.query(ScorecardScore).filter(ScorecardScore.run_id.in_(ids)).delete(synchronize_session=False)
                db.query(ScorecardRun).filter(ScorecardRun.id.in_(ids)).delete(synchronize_session=False)
        db.commit()


def purge_queue() -> None:
    """Every queued/running run row, whoever made it — queue tests claim
    "the oldest queued row" and must not inherit another test's."""
    with SessionLocal() as db:
        ScorecardRun.__table__.create(bind=db.get_bind(), checkfirst=True)
        ids = [r[0] for r in db.query(ScorecardRun.id).filter(ScorecardRun.status.in_(("queued", "running"))).all()]
        if ids:
            db.query(ScorecardScore).filter(ScorecardScore.run_id.in_(ids)).delete(synchronize_session=False)
            db.query(ScorecardRun).filter(ScorecardRun.id.in_(ids)).delete(synchronize_session=False)
        db.commit()


def insert_run(
    *, version_key: str, as_of: date, status: str = "succeeded", kind: str = "month_end",
    params: dict[str, Any] | None = None, requested_by: str = REQUESTED_BY, finished_at: datetime | None = None,
) -> int:
    with SessionLocal() as db:
        row = ScorecardRun(
            run_id=str(uuid.uuid4()), version_key=version_key, as_of=as_of, run_kind=kind, status=status,
            attempts=1, requested_by=requested_by, enqueued_at=FETCHED_AT, started_at=FETCHED_AT,
            finished_at=finished_at or FETCHED_AT, params=dict(params or {}),
        )
        db.add(row)
        db.commit()
        return int(row.id)


def score_rows_for_run(run_row_id: int) -> list[ScorecardScore]:
    with SessionLocal() as db:
        rows = db.query(ScorecardScore).filter(ScorecardScore.run_id == run_row_id).order_by(ScorecardScore.ticker).all()
        db.expunge_all()
        return rows


def run_row(run_row_id: int) -> ScorecardRun | None:
    with SessionLocal() as db:
        row = db.get(ScorecardRun, run_row_id)
        if row is not None:
            db.expunge(row)
        return row
