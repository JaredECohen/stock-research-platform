"""Phase 6 — Fundamental Factor Scorecard: versions, runs, scores, prices,
evaluations and memo disagreements.

Everything here is additive: new tables only, created by
`Base.metadata.create_all` at boot. The two point-in-time columns the
scorecard needs on an EXISTING table (`financial_periods.available_at`,
`available_at_source`) live on `FinancialPeriod` in `documents.py`, both
nullable so `reconcile_missing_columns` can add them to a populated table.

Why separate tables instead of extending `screener_scores`: the screener
row is a snapshot that is deleted and rebuilt nightly. The scorecard is a
versioned, point-in-time product — every score row carries the methodology
version and the availability date of the data it was computed from, and
month-end rows are kept forever so the evaluation can be re-run against the
exact numbers a reader saw at the time. Nothing on the web process computes
a score; runs are queued in `scorecard_runs` and drained by the worker
(cross-process state goes in Postgres, never in a module dict).

Two-layer convention carried into the row shape: `feature_raw` is the
observed layer (the ratio as measured, `null` when the input is missing),
`feature_z` is the interpretation layer (the normalised read). A missing
input stays `null` in both — it is never scored as zero or neutral.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class ScorecardVersion(Base):
    """Methodology registry — one row per scorecard spec version (`fs-v1`).

    `spec_hash` is the sha256 of the canonical JSON of the in-code feature
    spec, so a weight change that forgets to bump `version_key` is still
    detectable. `is_active` marks the version the daily loop scores with;
    the web process falls back to the in-code spec when no active row
    exists yet (the worker registers it on its first tick, and the web
    service can boot first after a deploy).
    """
    __tablename__ = "scorecard_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    spec_hash: Mapped[str] = mapped_column(String(64), default="")
    spec_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")


class ScorecardRun(Base):
    """Durable job + lineage record for one scoring / evaluation pass.

    Mirrors `RegenJob`: the row is the queue entry AND the audit trail.
    `run_kind` ∈ {scheduled, month_end, backfill, manual, pit_prepare,
    evaluate}; `status` ∈ {queued, running, succeeded, failed, skipped}.
    Identity for coalescing is `(version_key, as_of, run_kind)`; a run
    whose `inputs_hash` matches an already-succeeded run for the same
    `(version_key, as_of)` ends as `skipped` without writing rows. Orphaned
    `running` rows are marked failed at the start of the loop tick, not
    requeued — a scorecard run is cheap and atomic, unlike a memo regen.
    """
    __tablename__ = "scorecard_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    version_key: Mapped[str] = mapped_column(String(32), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    run_kind: Mapped[str] = mapped_column(String(16), default="scheduled")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    requested_by: Mapped[str] = mapped_column(String(32), default="")
    universe_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scored_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inputs_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_type: Mapped[str] = mapped_column(String(64), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    # Free-form `key=value` diagnostics ("written=… pit_excluded=…") so an
    # empty run is distinguishable from "nothing due" in cron-health.
    note: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("ix_scorecard_runs_version_asof_status", "version_key", "as_of", "status"),
    )


class ScorecardScore(Base):
    """One scored ticker in one run.

    `run_id` is the integer PK of `scorecard_runs` (not its string
    `run_id`), so readers resolve "latest succeeded run for (version,
    as_of)" and join on the integer. `overall_score` maps `overall_z` to
    0–100 the same way `factor_scores._z_to_100` does; percentiles are
    rank-based. `data_available_at` is the latest `available_at` among the
    fundamentals used — the point-in-time proof for the row. Contributions
    beyond the top ±3 are derivable from `feature_z × weight` and not
    stored, to keep rows small (170 tickers × (45 dailies + 60 month-ends)).
    """
    __tablename__ = "scorecard_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("scorecard_runs.id"), index=True)
    version_key: Mapped[str] = mapped_column(String(32))
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    sector: Mapped[str] = mapped_column(String(64), default="")
    overall_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    universe_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Share of applicable features that had an observed value, 0–1.
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    category_z: Mapped[dict] = mapped_column(JSON, default=dict)
    category_percentile: Mapped[dict] = mapped_column(JSON, default=dict)
    # Observed layer: `{feature: value | null}`.
    feature_raw: Mapped[dict] = mapped_column(JSON, default=dict)
    # Interpretation layer: `{feature: z | null}`.
    feature_z: Mapped[dict] = mapped_column(JSON, default=dict)
    top_positive: Mapped[list] = mapped_column(JSON, default=list)
    top_negative: Mapped[list] = mapped_column(JSON, default=list)
    latest_period: Mapped[str] = mapped_column(String(16), default="")
    data_available_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    price_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    inputs_hash: Mapped[str] = mapped_column(String(64), default="")
    is_month_end: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("run_id", "ticker", name="uq_scorecard_scores_run_ticker"),
        Index("ix_scorecard_scores_ticker_asof", "ticker", "as_of"),
        Index("ix_scorecard_scores_version_asof_month_end", "version_key", "as_of", "is_month_end"),
    )


class PriceMonthEnd(Base):
    """Minimal persisted price store: the last close of each calendar month.

    Filled from the SAME 252-day series the rest of the app already caches
    (`data_service.get_price_history(days=252)`), so month-end sync costs no
    extra provider calls. `month_end` is the calendar month end (the join
    key for as-of queries); `price_date` is the trading day the close is
    from. Only complete months are written — see
    `scorecard_pit.sync_price_month_ends`.

    Caveat carried into every evaluation result: FMP `/stable/` returns
    `adjusted_close == close` (no split adjustment, `fmp_provider.py`), so
    `adjusted_close` is only genuinely adjusted when another provider in
    the chain supplied the row. Returns across a split are wrong for
    FMP-sourced rows; this is accepted and flagged, not silently fixed.
    """
    __tablename__ = "price_month_ends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    month_end: Mapped[date] = mapped_column(Date, index=True)
    price_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    close: Mapped[float] = mapped_column(Float)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "month_end", name="uq_price_month_ends_ticker_month"),
    )


class ScorecardEvaluation(Base):
    """Append-only evaluation results (quintile spread, FF5+MOM regression,
    double-selection LASSO). `result` carries the caveats verbatim —
    unadjusted prices, current constituents, current sector labels,
    restated values at original availability — so no rendering layer can
    drop them. Model outputs, not recommendations.
    """
    __tablename__ = "scorecard_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_key: Mapped[str] = mapped_column(String(32), index=True)
    eval_kind: Mapped[str] = mapped_column(String(24), index=True)  # quintile_ls | ff6_regression | double_lasso
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    sample_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    n_obs: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ScorecardDisagreement(Base):
    """A memo rating that contradicts the scorecard read for the same name.

    A finding, not an outage: it never enters `degraded_agents`. The row
    links the memo snapshot (`memo_snapshot_id`, returned by the compose
    stage's persist) to the score row it was compared against, so the
    reviewer sees exactly which numbers disagreed. `status` ∈ {open,
    queued_review, reviewed, dismissed}; `seed_question` is the question a
    flag-gated deep-research regen would re-fire with.
    """
    __tablename__ = "scorecard_disagreements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    memo_snapshot_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("memo_snapshots.id"), nullable=True, index=True,
    )
    scorecard_score_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("scorecard_scores.id"), index=True,
    )
    version_key: Mapped[str] = mapped_column(String(32))
    as_of: Mapped[date] = mapped_column(Date)
    memo_rating: Mapped[str] = mapped_column(String(32), default="")
    memo_rating_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    scorecard_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), default="watch")  # material | watch
    dimension: Mapped[str] = mapped_column(String(24), default="overall")  # overall | valuation
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    seed_question: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        UniqueConstraint(
            "memo_snapshot_id", "scorecard_score_id", "dimension",
            name="uq_scorecard_disagreements_snapshot_score_dim",
        ),
    )
