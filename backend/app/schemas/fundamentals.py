"""FEAT-001 Fundamentals Explorer — wire contracts.

Three surfaces: the metric catalog, the series response (the exact
data a chart displays), and the commentary response (an LLM's reading
of *that displayed data* plus bounded stored-memo excerpts, kept in two
labelled sections so observed data and interpretation never blend).

A leaf of the schema DAG: imports only pydantic. `StructuredError`
(`schemas/accounts.py`) is the 402/429 body these routes raise, so this
module does not redefine it.

Every "no value" on the wire is `value: null` plus a closed-set `reason`
— never zero, never omitted — which is how the research process's
"missing evidence stays visibly unknown" rule reaches the chart.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

UnitType = Literal["currency", "percent", "ratio", "multiple", "count"]
MetricKind = Literal["reported", "derived", "market"]
PointReason = Literal[
    "base_nonpositive", "denominator_nonpositive", "no_price", "no_shares",
    "missing_line", "not_backfilled",
]
NormalizeMode = Literal["none", "indexed"]
UnavailableReason = Literal["not_backfilled"]

# Absolute request ceilings. Plan limits (Free 2×2×5y, Pro 5×4×max) are
# enforced by the route through the entitlement seam; these bound what
# the service will ever be asked to compute.
MAX_TICKERS = 5
MAX_METRICS = 4


def _normalise_tickers(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        t = (v or "").strip().upper()
        if t:
            seen.setdefault(t, None)
    return list(seen)


def _normalise_metrics(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        m = (v or "").strip().lower()
        if m:
            seen.setdefault(m, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class MetricSpecOut(BaseModel):
    id: str
    label: str
    family: str
    unit_type: UnitType
    kind: MetricKind
    formula_text: str
    inputs: list[str] = Field(default_factory=list)
    sign_note: str | None = None
    provenance: str
    requires_price: bool = False
    requires_prior: bool = False
    frequency: Literal["annual"] = "annual"


class CatalogOut(BaseModel):
    catalog_version: str
    as_of: datetime
    frequency: Literal["annual"] = "annual"
    families: list[str] = Field(default_factory=list)
    metrics: list[MetricSpecOut] = Field(default_factory=list)
    # Named so the frontend can reserve legend/table columns; empty in v1.
    reserved_source_kinds: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

class SeriesRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=MAX_TICKERS)
    metrics: list[str] = Field(min_length=1, max_length=MAX_METRICS)
    # Number of most-recent fiscal years; None = every year on record
    # (the route caps this to the plan before calling the service).
    years: int | None = Field(default=None, ge=1, le=60)
    normalize: NormalizeMode = "none"

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: list[str]) -> list[str]:
        out = _normalise_tickers(v)
        if not out:
            raise ValueError("at least one ticker is required")
        if len(out) > MAX_TICKERS:
            raise ValueError(f"at most {MAX_TICKERS} tickers")
        return out

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: list[str]) -> list[str]:
        out = _normalise_metrics(v)
        if not out:
            raise ValueError("at least one metric is required")
        if len(out) > MAX_METRICS:
            raise ValueError(f"at most {MAX_METRICS} metrics")
        return out


class SeriesPoint(BaseModel):
    period: str                      # "FY2024"
    period_end: date | None = None   # null when the provider gave no date
    value: float | None = None
    reason: PointReason | None = None  # set iff value is null
    estimated: bool = False          # a documented fallback was used


class SeriesCoverage(BaseModel):
    first: str | None = None   # first period with a value
    last: str | None = None    # last period with a value
    n: int = 0                 # periods with a value
    expected: int = 0          # periods on the shared axis


class SeriesProvenance(BaseModel):
    source: str = ""                  # provider name(s) of the raw rows
    fetched_at: datetime | None = None
    stale: bool = False
    stale_reason: str | None = None


class MetricSeries(BaseModel):
    ticker: str
    metric: str
    unit_type: UnitType
    kind: MetricKind
    currency: str | None = None
    # True when `normalize=indexed` was applied to this series (base = 100
    # at its first valued period). Percent/ratio/multiple series are never
    # indexed — indexing a ratio is misleading — and report False.
    indexed: bool = False
    points: list[SeriesPoint] = Field(default_factory=list)
    coverage: SeriesCoverage = Field(default_factory=SeriesCoverage)
    provenance: SeriesProvenance = Field(default_factory=SeriesProvenance)


class AppliedLimits(BaseModel):
    companies: int
    metrics: int
    years: int | None = None   # None = full history


class SeriesLimits(BaseModel):
    applied: AppliedLimits
    capped_by_plan: bool = False


class UnavailableTicker(BaseModel):
    ticker: str
    reason: UnavailableReason
    # What fixes it. For `not_backfilled`: run research on the company —
    # the memo job backfills its history.
    remedy: str = ""


class SeriesResponse(BaseModel):
    catalog_version: str
    as_of: datetime
    frequency: Literal["annual"] = "annual"
    normalize: NormalizeMode = "none"
    # The shared x-axis every series is aligned to, oldest first.
    periods: list[str] = Field(default_factory=list)
    series: list[MetricSeries] = Field(default_factory=list)
    limits: SeriesLimits
    unavailable: list[UnavailableTicker] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # sha256 over the displayed values; the commentary request echoes it
    # so the server can refuse to explain data the client is not showing.
    fingerprint: str


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------

class CommentaryRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=MAX_TICKERS)
    metrics: list[str] = Field(min_length=1, max_length=MAX_METRICS)
    years: int | None = Field(default=None, ge=1, le=60)
    fingerprint: str = Field(min_length=8, max_length=64)

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: list[str]) -> list[str]:
        out = _normalise_tickers(v)
        if not out:
            raise ValueError("at least one ticker is required")
        return out

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: list[str]) -> list[str]:
        out = _normalise_metrics(v)
        if not out:
            raise ValueError("at least one metric is required")
        return out


class CommentaryRef(BaseModel):
    ticker: str
    metric: str
    period: str


class ObservedItem(BaseModel):
    """A sentence about the displayed data, each citing the points it
    is about. Numbers here are recomputed from the series, never authored
    by the model."""
    text: str
    refs: list[CommentaryRef] = Field(default_factory=list)


class MemoViewItem(BaseModel):
    """A sentence attributed to one stored memo version. `memo_stale`
    combines memo_store freshness (a newer filing/transcript exists) with
    "the memo predates the last displayed fiscal period end"."""
    text: str
    ticker: str
    memo_version: int
    memo_generated_at: datetime
    memo_stale: bool = False
    memo_stale_reason: str | None = None


class CommentaryOut(BaseModel):
    commentary_id: int | None = None   # null on the uncached degraded path
    fingerprint: str
    cache_hit: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    observed: list[ObservedItem] = Field(default_factory=list)
    memo_view: list[MemoViewItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    generated_at: datetime
    model: str | None = None
    disclaimer: str = (
        "Research and education only. Observations are recomputed from the displayed data; "
        "memo excerpts are scenarios from stored research, not recommendations."
    )
