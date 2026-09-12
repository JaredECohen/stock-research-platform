"""Fundamental Factor Scorecard — API and memo contracts (Phase 6).

Two layers, kept apart in every shape below so a reader can never mistake
one for the other:

* **observed** — the raw feature value as measured from reported
  statements and the price on the as-of date (`raw`, `latest_period`,
  `data_available_at`, `price_date`). A missing input is `None` with a
  reason string, never a zero.
* **model read** — the normalised interpretation under a named
  methodology version (`z`, `score`, percentiles, contributions). The
  score scale is 0–100 with 50 = a z of 0, i.e. the sector (or
  universe-fallback) mean of the winsorised composite — not the median;
  only the rank-based percentiles are median-anchored.

`ScorecardSummary` is the compact form the memo carries
(`StockMemoOut.scorecard`, additive and optional); the rest are the
responses of `api/routes_scorecard.py` and the admin enqueue endpoints.
Research and education only: a scorecard read is a scenario input to the
memo, not a recommendation.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Building blocks shared by the memo summary and the detail response
# ---------------------------------------------------------------------------


class ScorecardCategory(BaseModel):
    """One feature family's model read."""
    z: float | None = None
    score: float | None = None            # 0-100, 50 = z of 0 (mean)
    percentile: float | None = None       # rank-based across the universe, (0, 100]
    sector_percentile: float | None = None
    weight: float
    coverage: float | None = None         # available / applicable features in the family
    n_features: int = 0                   # applicable features in the family
    n_available: int = 0                  # of those, with an observed value


class ScorecardContribution(BaseModel):
    """A feature's share of the overall z (contributions sum to overall_z)."""
    feature: str
    family: str
    z: float | None = None
    contribution: float


class ScorecardDisagreementFlag(BaseModel):
    """A memo rating that contradicts the scorecard read. A finding, not an
    outage — it never enters `degraded_agents`. Filled by the memo slice."""
    severity: Literal["material", "watch"]
    dimension: Literal["overall", "valuation"] = "overall"
    direction: Literal["narrative_above_quant", "narrative_below_quant"]
    gap: float
    note: str = ""


class ScorecardSummary(BaseModel):
    """What the memo carries: enough to say where the name ranks, why, and
    whether the narrative disagrees with the quant read."""
    version_key: str
    as_of: date
    run_id: str = ""
    overall_z: float | None = None
    overall_score: float | None = None
    universe_percentile: float | None = None
    sector_percentile: float | None = None
    coverage: float = 0.0
    categories: dict[str, ScorecardCategory] = Field(default_factory=dict)
    top_positive: list[ScorecardContribution] = Field(default_factory=list)
    top_negative: list[ScorecardContribution] = Field(default_factory=list)
    # Research-process sub-composites (never weighted into overall).
    profiles: dict[str, float | None] = Field(default_factory=dict)
    latest_period: str = ""
    data_available_at: date | None = None
    price_date: date | None = None
    stale: bool = False
    notes: list[str] = Field(default_factory=list)
    disagreement: ScorecardDisagreementFlag | None = None
    reconciliation: str = ""


# ---------------------------------------------------------------------------
# Public read responses
# ---------------------------------------------------------------------------


class ScorecardFeatureOut(BaseModel):
    """Observed value and model read for one feature, side by side."""
    name: str
    family: str
    sign: int
    weight: float
    formula: str
    description: str
    applicable: bool
    raw: float | None = None              # observed
    reason: str | None = None             # why `raw` is None (n/a because ...)
    z: float | None = None                # model read (sector-neutral or fallback)
    z_universe: float | None = None       # model read, universe-relative
    basis: str = ""                       # which sample z was taken against
    contribution: float | None = None


class ScorecardHistoryPoint(BaseModel):
    as_of: date
    is_month_end: bool
    overall_z: float | None = None
    overall_score: float | None = None
    universe_percentile: float | None = None
    sector_percentile: float | None = None
    coverage: float = 0.0
    category_z: dict[str, float | None] = Field(default_factory=dict)


class ScorecardDetailOut(ScorecardSummary):
    ticker: str
    company_name: str = ""
    sector: str = ""
    sector_raw: str | None = None
    is_month_end: bool = False
    spec_hash: str = ""
    inputs_hash: str = ""
    features: list[ScorecardFeatureOut] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    history: list[ScorecardHistoryPoint] = Field(default_factory=list)


class ScorecardUniverseRow(BaseModel):
    rank: int
    ticker: str
    company_name: str = ""
    sector: str = ""
    overall_z: float | None = None
    overall_score: float | None = None
    universe_percentile: float | None = None
    sector_percentile: float | None = None
    coverage: float = 0.0
    category_score: dict[str, float | None] = Field(default_factory=dict)
    category_z: dict[str, float | None] = Field(default_factory=dict)
    top_positive: list[ScorecardContribution] = Field(default_factory=list)
    top_negative: list[ScorecardContribution] = Field(default_factory=list)
    latest_period: str = ""
    notes: list[str] = Field(default_factory=list)


class ScorecardUniverseOut(BaseModel):
    version_key: str
    spec_hash: str = ""
    as_of: date
    run_id: str
    is_month_end: bool = False
    universe_size: int
    scored: int
    insufficient: int
    sort_by: str
    order: str
    stale: bool = False
    rows: list[ScorecardUniverseRow]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ScorecardEvaluationItem(BaseModel):
    kind: str
    run_id: str = ""
    created_at: datetime | None = None
    sample_start: date | None = None
    sample_end: date | None = None
    n_obs: int = 0
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class ScorecardEvaluationOut(BaseModel):
    version_key: str
    evaluations: dict[str, ScorecardEvaluationItem] = Field(default_factory=dict)
    # Rendered verbatim by every UI: the evaluation is honest only with these.
    caveats: list[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# Admin enqueue contracts
# ---------------------------------------------------------------------------


class ScorecardRunOut(BaseModel):
    id: int
    run_id: str
    version_key: str
    as_of: date
    run_kind: str
    status: str
    attempts: int = 0
    requested_by: str = ""
    universe_size: int | None = None
    scored_count: int | None = None
    inputs_hash: str | None = None
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_type: str = ""
    error_message: str = ""
    note: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


def _not_in_the_future(value: date | None) -> date | None:
    """Readers resolve "latest succeeded run" by `as_of DESC`, so one
    mistyped future date would become the cross-section every read and
    the memo's summary return until a later-dated run existed. Refuse it
    at the edge (422) rather than let a typo pin the scorecard."""
    if value is not None and value > datetime.utcnow().date():
        raise ValueError("as_of may not be in the future")
    return value


class ScorecardRefreshRequest(BaseModel):
    as_of: date | None = None
    version_key: str | None = None
    tickers: list[str] | None = Field(None, max_length=600)
    kind: Literal["manual", "month_end"] = "manual"

    _as_of_bounded = field_validator("as_of")(_not_in_the_future)


class ScorecardEvaluateRequest(BaseModel):
    version_key: str | None = None
    as_of: date | None = None

    _as_of_bounded = field_validator("as_of")(_not_in_the_future)


class ScorecardBackfillRequest(BaseModel):
    version_key: str | None = None
    months: int | None = Field(None, ge=1, le=240)
    end: date | None = None

    _end_bounded = field_validator("end")(_not_in_the_future)


class ScorecardEnqueueOut(BaseModel):
    run: ScorecardRunOut
    created: bool
    note: str = ""


class ScorecardBackfillOut(BaseModel):
    pit_prepare: ScorecardRunOut
    enqueued: list[ScorecardRunOut] = Field(default_factory=list)
    skipped_existing: list[date] = Field(default_factory=list)
    note: str = ""


__all__ = [
    "ScorecardBackfillOut",
    "ScorecardBackfillRequest",
    "ScorecardCategory",
    "ScorecardContribution",
    "ScorecardDetailOut",
    "ScorecardDisagreementFlag",
    "ScorecardEnqueueOut",
    "ScorecardEvaluateRequest",
    "ScorecardEvaluationItem",
    "ScorecardEvaluationOut",
    "ScorecardFeatureOut",
    "ScorecardHistoryPoint",
    "ScorecardRefreshRequest",
    "ScorecardRunOut",
    "ScorecardSummary",
    "ScorecardUniverseOut",
    "ScorecardUniverseRow",
]
