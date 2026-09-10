"""FEAT-003 — GICS taxonomy registry, company classification audit, and the
weekly Industry Analysis artifacts (stats, cross-industry snapshot, reports,
report jobs).

Seven tables, all new. Nothing here alters an existing table, so
``create_all`` handles a fresh database and ``reconcile_missing_columns``
handles every later additive column (each one is nullable or defaulted).

Design rules these tables encode:

* **Counts come from the registry, never from code.** ``gics_nodes`` holds
  whatever the imported knowledge JSON holds, at four levels (sector 2 /
  group 4 / industry 6 / sub-industry 8), with retired rows kept as
  ``is_active = False`` so an old classification can still be named.
* **Classification is append-only with a current flag.** A company is
  re-classified by superseding its current row and inserting a new one;
  the history answers "why did this ticker move groups on that date".
  Every row carries its provenance (source, author, as_of) and an explicit
  state — ``mapped | fallback | missing | stale | conflict`` — so the audit
  never has to infer why a company is where it is.
* **Two processes, one database.** The active taxonomy is a DB flag
  (``gics_taxonomy_versions.is_active``), report jobs are a durable queue
  mirroring ``regen_jobs``, and nothing is coordinated through memory.
* **Facts and narrative are stored apart.** ``industry_stats`` is the
  canonical analytics row; ``industry_reports.payload`` copies facts from
  it and keeps the analyst's interpretation in separate objects, so a
  report can always be checked against the numbers it was written from.
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

# Levels of the GICS hierarchy as stored in `gics_nodes.level`, in order.
NODE_LEVELS: tuple[str, ...] = ("sector", "industry_group", "industry", "sub_industry")
LEVEL_BY_CODE_LENGTH: dict[int, str] = {2: "sector", 4: "industry_group", 6: "industry", 8: "sub_industry"}

# `company_industry_classifications.state` — every row names one of these.
#   mapped    — resolved to an industry group (and usually deeper)
#   fallback  — only the sector could be resolved (provider sector alias)
#   missing   — nothing resolved; the audit lists the unmapped labels
#   stale     — the inputs this row was computed from no longer match the
#               company row; awaiting re-classification (a transient state
#               that persists only if re-classification fails)
#   conflict  — the research map and the provider alias disagree on the
#               GROUP; both are recorded and the research map wins routing
CLASSIFICATION_STATES: tuple[str, ...] = ("mapped", "fallback", "missing", "stale", "conflict")
# `company_industry_classifications.source` — where the winning codes came from.
CLASSIFICATION_SOURCES: tuple[str, ...] = ("research_map", "provider_alias", "none")


class TaxonomyVersion(Base):
    """One imported GICS structure (e.g. ``gics-2026-04``).

    Exactly one row is active at a time — enforced by
    ``gics_registry.activate_version`` inside one transaction, not by the
    database, because "which one is active" is a deployment decision that
    must be readable by both processes and flippable without a deploy.
    ``checksum`` is the sha256 of the canonical node list, so re-importing
    the same file is a no-op and importing a *different* structure under an
    existing key is refused rather than silently rewriting history.
    """
    __tablename__ = "gics_taxonomy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(64), default="")
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    # Provenance of the file the nodes came from (encyclopedia + map hashes,
    # map as-of) and the per-level counts at import — for the audit
    # endpoint and for a test that asserts counts come from the data.
    node_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    attribution: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GicsNode(Base):
    """One node of one taxonomy version — sector, industry group, industry
    or sub-industry. Immutable after import (a new structure is a new
    version), which is what makes the per-version node cache safe."""
    __tablename__ = "gics_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    level: Mapped[str] = mapped_column(String(16), index=True)
    code: Mapped[str] = mapped_column(String(8))
    name: Mapped[str] = mapped_column(String(160))
    parent_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # Free-form, e.g. {"status": "discontinued"} on retired rows. The
    # display label for `internal_labels` mode would live here if that
    # mode is ever built; nothing reads it today.
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("taxonomy_version_id", "level", "code", name="uq_gics_nodes_version_level_code"),
        Index("ix_gics_nodes_version_parent", "taxonomy_version_id", "parent_code"),
    )


class CompanyIndustryClassification(Base):
    """A company's placement in the taxonomy — one current row per
    (ticker, taxonomy version), older rows kept with ``superseded_at`` set.

    ``ticker`` is deliberately not a foreign key: an on-demand symbol may be
    classified in the same request that introduces it, and a company that
    later leaves the universe keeps its history.

    Resolution precedence (``industry_classification``): (1) the map's
    security reference by exact symbol → ``research_map``; (2) the
    provider label alias map → ``provider_alias``; (3) ``missing``. When
    (1) and (2) disagree on the GROUP the row is ``conflict`` with both
    codes in ``candidates`` and (1) wins routing. ``source_*`` are the
    company labels the row was computed from and ``inputs_fingerprint``
    hashes them together with the alias-map version and the map's as-of,
    so drift is detected by comparison, never by re-resolving on read.
    """
    __tablename__ = "company_industry_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    sector_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    industry_group_code: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    industry_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sub_industry_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Every sub-industry code the research map records for the symbol
    # (the first is `sub_industry_code`); [] when the map has no entry.
    sub_industry_codes: Mapped[list] = mapped_column(JSON, default=list)
    state: Mapped[str] = mapped_column(String(16), index=True)
    source: Mapped[str] = mapped_column(String(24), default="none")
    method: Mapped[str] = mapped_column(String(32), default="")
    author: Mapped[str] = mapped_column(String(128), default="")
    source_as_of: Mapped[str] = mapped_column(String(32), default="")
    # Ordinal, not calibrated: research_map > alias industry > alias sector.
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_sub_industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    inputs_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    # conflict: [{"source", "industry_group_code", "industry_code", ...}]
    candidates: Mapped[list] = mapped_column(JSON, default=list)
    # alias key hit, previous state on staleness, unmapped labels, …
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    classified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    superseded_reason: Mapped[str] = mapped_column(String(32), default="")

    __table_args__ = (
        Index(
            "ix_company_industry_class_current",
            "ticker", "taxonomy_version_id", "is_current",
        ),
        Index(
            "ix_company_industry_class_group_current",
            "taxonomy_version_id", "industry_group_code", "is_current",
        ),
    )


# Frozen-contract alias: the implementation plan named the class
# `CompanyClassification`; the table gained the `industry` qualifier so it
# cannot be mistaken for a future scorecard/universe classification.
CompanyClassification = CompanyIndustryClassification


class IndustryStatSnapshot(Base):
    """Canonical weekly analytics for one industry group — persisted apart
    from any narrative so the numbers a report was written from are
    always recoverable. ``method`` states weighting, benchmarks, sample
    floor and missing-data policy; ``sample`` states who was in and who
    was excluded and why; ``per_ticker`` keeps each constituent's weekly
    closes so later horizons reuse them instead of re-reading 252-day
    price windows. ``inputs_hash`` makes a recompute over identical inputs
    a no-op (determinism is a tested property)."""
    __tablename__ = "industry_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    industry_group_code: Mapped[str] = mapped_column(String(8), index=True)
    period_key: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime, index=True)
    method: Mapped[dict] = mapped_column(JSON, default=dict)
    sample: Mapped[dict] = mapped_column(JSON, default=dict)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    per_ticker: Mapped[dict] = mapped_column(JSON, default=dict)
    inputs_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    compute_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "taxonomy_version_id", "industry_group_code", "period_key", "inputs_hash",
            name="uq_industry_stats_identity",
        ),
    )


class CrossIndustrySnapshot(Base):
    """The Portfolio Manager's compact cross-industry view for one period —
    computed deterministically from that period's stats rows, the macro
    broadcast, catalyst/news events and the checked-in dependency edges.
    Persisted separately (with the taxonomy and report versions it saw) so
    the PM block can be re-rendered without recomputing."""
    __tablename__ = "industry_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    period_key: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime, index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    stats_ids: Mapped[list] = mapped_column(JSON, default=list)
    report_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "taxonomy_version_id", "period_key", "as_of", name="uq_industry_snapshots_identity",
        ),
    )


class IndustryReport(Base):
    """A versioned weekly Industry Analysis edition for one group.

    Failures never create a row: ``is_latest_good`` stays on the prior
    edition and the read API composes ``stale`` + ``last_attempt`` from the
    jobs table. A succeeded edition with a non-empty ``degraded`` list is a
    labelled deterministic/partial edition, not a failure.
    ``status`` ∈ succeeded | pending_review | superseded (``pending_review``
    is reachable only under ``INDUSTRY_REPORTS_REQUIRE_REVIEW``, which has
    no publish endpoint yet — owner decision 4).
    """
    __tablename__ = "industry_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    industry_group_code: Mapped[str] = mapped_column(String(8), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_report_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("industry_reports.id"), nullable=True,
    )
    period_key: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="succeeded", index=True)
    is_latest_good: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    report_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    stats_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("industry_stats.id"), nullable=True,
    )
    source_manifest: Mapped[list] = mapped_column(JSON, default=list)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)
    freshness: Mapped[dict] = mapped_column(JSON, default=dict)
    generation: Mapped[dict] = mapped_column(JSON, default=dict)
    degraded: Mapped[list] = mapped_column(JSON, default=list)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint(
            "taxonomy_version_id", "industry_group_code", "version",
            name="uq_industry_reports_version",
        ),
        Index("ix_industry_reports_group_latest", "industry_group_code", "is_latest_good"),
    )


class IndustryReportJob(Base):
    """Durable queue for report generation — mirrors ``RegenJob``.

    ``kind`` ∈ group_report | cross_snapshot; a cross_snapshot job has no
    group code and a lower priority (higher number) so it drains after the
    period's group reports. ``not_before`` carries the retry backoff
    (15 min × attempts); ``max_attempts`` defaults to 3 with the final
    attempt run deterministic. ``heartbeat_at`` is refreshed at each
    waypoint so a hung job is distinguishable from a killed one. Restart
    recovery and the 14-day ``QueueExpired`` rule live in the drainer.
    """
    __tablename__ = "industry_report_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), default="group_report", index=True)
    taxonomy_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("gics_taxonomy_versions.id"), index=True,
    )
    industry_group_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    period_key: Mapped[str] = mapped_column(String(16), index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    not_before: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(24), default="weekly_cron")
    force: Mapped[bool] = mapped_column(Boolean, default=False)
    report_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str] = mapped_column(String(64), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    traceback_tail: Mapped[str] = mapped_column(Text, default="")
    progress: Mapped[list] = mapped_column(JSON, default=list)

    __table_args__ = (
        Index("ix_industry_report_jobs_claim", "status", "priority", "id"),
        Index(
            "ix_industry_report_jobs_identity",
            "taxonomy_version_id", "industry_group_code", "period_key", "kind",
        ),
    )
