"""FEAT-003 — the Industry Analysis API contracts (slice 5).

Shapes for `api/routes_industries.py` (public reads) and
`api/routes_industries_admin.py` (ops). Three rules run through all of
them, and every field name is chosen to keep them visible:

* **Observed vs interpretation.** A report's `payload.sections[*]` keeps
  server-computed `facts` apart from the analyst's `interpretation`; this
  layer never flattens the two. `stats` carries the observed numbers with
  the `method` that produced them — the two travel together so a reader
  can never quote a benchmark-relative return without the cohort basis or
  a breadth number without its session window.
* **Membership is not coverage.** `constituent_count` / `membership` is
  what the classification says belongs to a group; `n_priced` is how many
  of those had a price series this period. They are different numbers and
  are reported as different fields, never collapsed.
* **Missing is missing.** Absent values are `None` with a stated reason
  (`stale_reason`, `exclusion`, `reason`), never zero and never a bare
  `n/a`. A truncated list reports how many entries it dropped.

Research and education only: every forward-looking line in a report is a
scenario, not a recommendation.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Access (the Phase 4 entitlement seam)
# ---------------------------------------------------------------------------


class IndustryAccessOut(BaseModel):
    """What this surface costs and whether that is being enforced.

    `tier` is the policy; `enforced` is whether `AUTH_ENABLED` makes it
    bite; `route_gated` is whether the route that returned this block
    applied it. With the login wall off every surface answers to everyone,
    and saying so is the difference between describing a price list and
    claiming a customer paid it.
    """
    surface: str
    tier: Literal["public", "pro"]
    required_tier: str | None = None
    allowed: bool = True
    enforced: bool = False
    # Did THIS route apply the tier? `/taxonomy` answers to everyone by
    # design — it is how a signed-out visitor learns the reports behind it
    # are Pro — so it reports the `latest` tier with `route_gated: false`.
    # A UI that read `allowed` alone would think the gate had been cleared.
    route_gated: bool = True
    setting: str = "public"
    surfaces: dict[str, str] = Field(default_factory=dict)
    plan: str | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TaxonomyVersionOut(BaseModel):
    key: str
    checksum: str = ""
    source: str = ""
    is_active: bool = False
    effective_from: str | None = None
    effective_to: str | None = None
    node_counts: dict[str, int] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    attribution: str = ""
    display_mode: str = "codes_and_names"
    mapping_caveat: str = ""
    imported_at: str | None = None
    activated_at: str | None = None


class LatestReportOut(BaseModel):
    """The pointer the picker renders. `stale_by_age` is arithmetic on
    `as_of` only — a failed refresh is the other half of staleness and
    costs a query per group, so it is reported by
    `GET /api/industries/{code}/report` and named here rather than
    silently folded in."""
    version: int
    period_key: str = ""
    as_of: str | None = None
    status: str = ""
    generated_at: str | None = None
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    stale_by_age: bool = False
    age_days: int | None = None
    stale_basis: str = "as_of age only; failed-refresh staleness is on the report endpoint"


class IndustryGroupNodeOut(BaseModel):
    code: str
    name: str
    sector_code: str
    is_active: bool = True
    effective_from: str | None = None
    effective_to: str | None = None
    industry_count: int
    sub_industry_count: int
    constituent_count: int
    latest_report: LatestReportOut | None = None


class SectorNodeOut(BaseModel):
    code: str
    name: str
    industry_groups: list[IndustryGroupNodeOut] = Field(default_factory=list)


class TaxonomyOut(BaseModel):
    """Counts come from the registry and the classification table on every
    call — never from a literal, at any level."""
    taxonomy_version: TaxonomyVersionOut
    sectors: list[SectorNodeOut] = Field(default_factory=list)
    node_counts: dict[str, int] = Field(default_factory=dict)
    constituents: dict[str, Any] = Field(default_factory=dict)
    reports: dict[str, int] = Field(default_factory=dict)
    access: IndustryAccessOut
    attribution: str = ""
    mapping_caveat: str = ""
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class LastAttemptOut(BaseModel):
    job_id: int
    status: str
    at: str | None = None
    period_key: str = ""
    attempts: int = 0
    max_attempts: int = 0
    error_type: str = ""
    error_message: str = ""
    report_id: int | None = None
    source: str = ""


class IndustryStatsOut(BaseModel):
    """The observed layer. `method` always ships with `payload`: the
    benchmark cohort deliberately excludes constituents the group's own
    numbers include, and the breadth window is a session count the reader
    cannot otherwise check. Per-ticker rows are NOT here — they are the
    `/companies` response, which is where membership and price coverage
    are reconciled."""
    id: int
    period_key: str = ""
    as_of: str | None = None
    method: dict[str, Any] = Field(default_factory=dict)
    sample: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    inputs_hash: str = ""
    computed_at: str | None = None
    per_ticker_note: str = "per-ticker rows are served by GET /api/industries/{code}/companies"


class IndustryReportOut(BaseModel):
    code: str
    name: str = ""
    sector_code: str = ""
    taxonomy_version: str = ""
    version: int
    parent_report_id: int | None = None
    period_key: str = ""
    as_of: str | None = None
    status: str = ""
    is_latest_good: bool = False
    report_schema_version: int = 1
    stale: bool | None = None
    stale_reason: str | None = None
    last_attempt: LastAttemptOut | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    stats: IndustryStatsOut | None = None
    stats_unavailable_reason: str | None = None
    coverage: dict[str, Any] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    llm_cost_usd: float | None = None
    generated_at: str | None = None
    access: IndustryAccessOut
    disclaimer: str = ""
    attribution: str = ""
    mapping_caveat: str = ""


class IndustryReportHistoryItemOut(BaseModel):
    version: int
    period_key: str = ""
    as_of: str | None = None
    status: str = ""
    is_latest_good: bool = False
    degraded: list[str] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    stats_id: int | None = None
    llm_cost_usd: float | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None


class IndustryHistoryOut(BaseModel):
    code: str
    name: str = ""
    taxonomy_version: str = ""
    count: int = 0
    limit: int = 26
    truncated: int = 0
    items: list[IndustryReportHistoryItemOut] = Field(default_factory=list)
    last_attempt: LastAttemptOut | None = None
    access: IndustryAccessOut
    disclaimer: str = ""


class FactDeltaOut(BaseModel):
    """One fact, both sides. `delta` is `None` with a `reason` whenever
    either side is missing — a missing fact is never differenced to zero."""
    from_: Any = Field(None, alias="from")
    to: Any = None
    delta: float | None = None
    reason: str | None = None

    model_config = {"populate_by_name": True}


class IndustryChangesOut(BaseModel):
    code: str
    name: str = ""
    taxonomy_version: str = ""
    from_: dict[str, Any] = Field(default_factory=dict, alias="from")
    to: dict[str, Any] = Field(default_factory=dict)
    adjacent: bool = False
    facts_delta: dict[str, FactDeltaOut] = Field(default_factory=dict)
    constituents: dict[str, Any] = Field(default_factory=dict)
    leaders_laggards: dict[str, Any] = Field(default_factory=dict)
    analyst_view: dict[str, Any] = Field(default_factory=dict)
    access: IndustryAccessOut
    disclaimer: str = ""

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


class CompanyClassificationOut(BaseModel):
    """Where a company's group assignment came from, with the caveat the
    map's author insisted on. `source_label` is what a UI shows."""
    state: str
    source: str = ""
    source_label: str = ""
    method: str = ""
    author: str = ""
    as_of: str = ""
    confidence: float | None = None
    classified_at: str | None = None
    mapping_caveat: str = ""


class IndustryCompanyRowOut(BaseModel):
    ticker: str
    company_name: str | None = None
    is_active: bool = True
    industry_code: str | None = None
    industry_name: str | None = None
    sub_industry_code: str | None = None
    sub_industry_name: str | None = None
    sub_industry_codes: list[str] = Field(default_factory=list)
    classification: CompanyClassificationOut
    market_cap: float | None = None
    weight_mcw: float | None = None
    last_close: float | None = None
    last_date: str | None = None
    price_source: str | None = None
    returns: dict[str, float | None] = Field(default_factory=dict)
    return_reasons: dict[str, str] = Field(default_factory=dict)
    above_50d_mean: bool | None = None
    priced: bool = False
    unpriced_reason: str | None = None


class IndustryCompaniesOut(BaseModel):
    """Membership first, price coverage second, and both counted.

    `items` is the classified membership of the group. `n_priced` is how
    many of those the latest stats row could price; the rest carry
    `priced=false` with `unpriced_reason`. A caller that wants "the names
    behind the statistics" uses `n_priced`, not `count`.
    """
    code: str
    name: str = ""
    sector_code: str = ""
    taxonomy_version: str = ""
    as_of: str | None = None
    count: int = 0
    n_priced: int = 0
    membership_source: str = ""
    membership_states: dict[str, int] = Field(default_factory=dict)
    stats: dict[str, Any] | None = None
    stats_unavailable_reason: str | None = None
    limit: int = 500
    truncated: int = 0
    items: list[IndustryCompanyRowOut] = Field(default_factory=list)
    excluded: list[dict[str, Any]] = Field(default_factory=list)
    access: IndustryAccessOut
    attribution: str = ""
    mapping_caveat: str = ""
    security_reference_caveat: str = ""
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Cross-industry snapshot
# ---------------------------------------------------------------------------


class IndustrySnapshotOut(BaseModel):
    id: int
    taxonomy_version: str = ""
    period_key: str = ""
    as_of: str | None = None
    schema_version: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)
    stats_ids: list[int] = Field(default_factory=list)
    report_versions: dict[str, int] = Field(default_factory=dict)
    computed_at: str | None = None
    access: IndustryAccessOut
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TaxonomyImportRequest(BaseModel):
    version_key: str | None = Field(
        None, description="Key to import the bundled knowledge JSON under; default the payload's own taxonomy_version.",
    )
    activate: bool = Field(False, description="Make this version the single active one.")
    notes: str = ""


class TaxonomyImportOut(BaseModel):
    version_key: str
    version_id: int
    imported: bool
    nodes_inserted: int
    node_counts: dict[str, int] = Field(default_factory=dict)
    checksum: str
    activated: bool
    drift: dict[str, Any] | None = None


class ClassifyRequest(BaseModel):
    tickers: list[str] | None = Field(
        None, description="Restrict to these tickers; default every companies row.",
    )
    reclassify: bool = Field(True, description="Re-resolve rows whose inputs drifted (False = detection only).")
    force: bool = Field(False, description="Supersede every row even when the outcome is unchanged.")


class ClassifyOut(BaseModel):
    taxonomy_version: str
    classified: int = 0
    inserted: int = 0
    reclassified: int = 0
    restamped: int = 0
    unchanged: int = 0
    stale_detected: int = 0
    stale_fixed: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    sources: dict[str, int] = Field(default_factory=dict)
    changed: list[dict[str, Any]] = Field(default_factory=list)
    changed_total: int = 0
    changed_truncated: int = 0
    unmapped_labels: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_labels_total: int = 0
    unmapped_labels_truncated: int = 0
    mapping_caveat: str = ""


class RegenerateRequest(BaseModel):
    codes: list[str] | None = Field(None, description="Industry group codes; default every active group.")
    period_key: str | None = Field(None, description="ISO week; default the week of the most recent as-of weekday.")
    force: bool = Field(False, description="Re-generate even when the period already published.")


class RegenerateOut(BaseModel):
    period_key: str
    taxonomy_version: str
    requested: int = 0
    enqueued: list[dict[str, Any]] = Field(default_factory=list)
    coalesced: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class IndustryJobOut(BaseModel):
    id: int
    kind: str
    code: str | None = None
    period_key: str = ""
    run_id: str = ""
    status: str = ""
    attempts: int = 0
    max_attempts: int = 0
    priority: int = 0
    not_before: str | None = None
    enqueued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    source: str = ""
    force: bool = False
    report_id: int | None = None
    error_type: str = ""
    error_message: str = ""
    progress_waypoints: int = 0


class IndustryJobsOut(BaseModel):
    count: int = 0
    limit: int = 50
    truncated: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    taxonomy_version: str = ""
    drainer: dict[str, Any] = Field(default_factory=dict)
    jobs: list[IndustryJobOut] = Field(default_factory=list)
