"""FEAT-003 — the public Industry Analysis read API (slice 5).

Every route here is a row fetch. Nothing computes statistics, nothing
calls a provider or an LLM, and nothing enqueues: the weekly worker owns
generation and a page view must never start it (the repo's "no expensive
work in a page request" rule, and the reason `/industries` can be public
at all).

What the shapes are careful about:

* **Counts come from the registry and the classification table**, on
  every call. There is no literal 24 / 25 / 74 / 163 anywhere in this
  file, and none may be added: a taxonomy revision changes the numbers
  and the API must follow it without a deploy.
* **Membership is not price coverage.** `/companies` lists the classified
  membership of a group and marks which names the latest statistics row
  could price, with a reason for each one it could not. `n_priced` and
  `count` are different fields on purpose.
* **Numbers travel with their method.** The report response carries the
  statistics row's `method` and `sample` whenever it carries `payload`,
  because `benchmark_relative` without `method.benchmark_cohort_basis`
  and breadth without `method.breadth_mean_window.sessions` are numbers
  a reader cannot check.
* **Staleness is composed, not guessed.** `stale` is the store's answer:
  an as-of older than `INDUSTRY_REPORT_STALE_AFTER_DAYS`, or a refresh
  attempt that failed after this edition was generated — with
  `last_attempt` naming the failure. A failed week never blanks the page.
* **Truncation is counted.** Any list this API caps reports how many
  entries it dropped.

Access is the `entitlements_industry` seam: `latest` follows
`INDUSTRY_ANALYSIS_ACCESS`, `history` / `changes` are Pro, the snapshot
rides the PM tier, and `/taxonomy` always answers — including when the
taxonomy has not been imported — so the UI can explain the gate instead
of showing an error.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select

from ..config import settings
from ..database import SessionLocal
from ..models import Company, CompanyIndustryClassification, IndustryReport
from ..rate_limit import LIMITS, limiter
from ..schemas.industry import (
    IndustryAccessOut,
    IndustryChangesOut,
    IndustryCompaniesOut,
    IndustryHistoryOut,
    IndustryReportOut,
    IndustrySnapshotOut,
    TaxonomyOut,
)
from ..services import gics_registry, industry_analytics, industry_classification, industry_snapshot
from ..services import industry_knowledge as knowledge
from ..services import industry_report_store as store
from .entitlements_industry import (
    SURFACE_CHANGES,
    SURFACE_HISTORY,
    SURFACE_LATEST,
    SURFACE_SNAPSHOT,
    describe_surface,
    require_surface,
)
from .gating import rate_scope

log = logging.getLogger(__name__)
router = APIRouter()

# The membership states that make a company a constituent. A `fallback`
# row knows its sector, not its group, and must never inflate a cohort;
# `conflict` rows are members (the research map wins routing) and are
# labelled as such on every row.
MEMBER_STATES = (industry_classification.STATE_MAPPED, industry_classification.STATE_CONFLICT)

COMPANIES_LIMIT = 500
HISTORY_LIMIT = 26

_SOURCE_LABELS = {
    industry_classification.SOURCE_RESEARCH_MAP:
        "research map (economic research examples; not licensed issuer GICS mapping)",
    industry_classification.SOURCE_PROVIDER_ALIAS:
        "provider label crosswalk (derived from provider classification)",
    industry_classification.SOURCE_NONE: "unresolved",
}


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _error(status: int, error_code: str, message: str, **extra: Any) -> HTTPException:
    """A structured refusal: `code` + `message` plus whatever names the
    state (the group, the remedy, the failed attempt). The first argument
    is `error_code`, not `code`, so a caller can pass the industry-group
    `code=` in `extra` without colliding with it."""
    return HTTPException(status_code=status, detail={"code": error_code, "message": message, **extra})


def _taxonomy_or_503(access: dict[str, Any]) -> gics_registry.VersionInfo:
    """The active taxonomy, or a 503 that still carries the access policy
    and the operator's remedy. "Not imported" is a deployment state, not
    a bug, and the page says so.

    Takes the block the access dependency already built rather than
    rebuilding one, so the refusal cannot describe a different tier than
    a success on the same route would — with `allowed` flipped, because
    nothing was served.
    """
    try:
        return gics_registry.require_active_version()
    except gics_registry.TaxonomyNotImported:
        raise _error(
            503, "taxonomy_not_imported",
            "the GICS taxonomy has not been imported on this deployment yet",
            access={**access, "allowed": False},
            remedy="POST /api/admin/industries/taxonomy/import (or python -m app.scripts.import_gics_taxonomy --activate)",
        ) from None


def _group_or_404(code: str, info: gics_registry.VersionInfo) -> gics_registry.NodeInfo:
    try:
        return gics_registry.group(code, version=info)
    except gics_registry.UnknownNode as exc:
        raise _error(404, "unknown_industry_group", str(exc), industry_group_code=str(code),
                     taxonomy_version=info.version_key) from None


def _age_days(as_of: str | None, now: datetime) -> int | None:
    if not as_of:
        return None
    try:
        return max(0, (now - datetime.fromisoformat(as_of)).days)
    except ValueError:  # pragma: no cover — stored values are isoformat
        return None


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@router.get("/api/industries/taxonomy", response_model=TaxonomyOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_taxonomy(
    request: Request,
    access: dict = Depends(describe_surface(SURFACE_LATEST)),
    _rate: None = Depends(rate_scope("data")),
) -> TaxonomyOut:
    """The active structure: sectors → industry groups, with each group's
    industry and sub-industry counts, its constituent count, and the
    pointer to its latest published edition.

    Four reads: the active version, the version's nodes (cached per
    version id — they are immutable after import), one constituent count
    query and one bulk latest-edition query. Never a query per group.
    """
    info = _taxonomy_or_503(access)
    nodes = gics_registry.nodes(version=info)
    by_group = industry_classification.constituents_by_group(version=info)

    groups = [n for n in nodes if n.level == "industry_group"]
    latest = store.latest_good_many([g.code for g in groups], version=info)
    industries_by_group: dict[str, int] = {}
    subs_by_group: dict[str, int] = {}
    for node in nodes:
        if node.level == "industry":
            industries_by_group[node.code[:4]] = industries_by_group.get(node.code[:4], 0) + 1
        elif node.level == "sub_industry":
            subs_by_group[node.code[:4]] = subs_by_group.get(node.code[:4], 0) + 1

    now = _utcnow()
    stale_after = int(settings.industry_report_stale_after_days)
    groups_by_sector: dict[str, list[dict[str, Any]]] = {}
    for node in groups:
        edition = latest.get(node.code)
        pointer = None
        if edition is not None:
            age = _age_days(edition.get("as_of"), now)
            pointer = {
                "version": edition["version"],
                "period_key": edition.get("period_key") or "",
                "as_of": edition.get("as_of"),
                "status": edition.get("status") or "",
                "generated_at": edition.get("generated_at"),
                "degraded": bool(edition.get("degraded")),
                "degraded_reasons": list(edition.get("degraded") or []),
                "stale_by_age": bool(age is not None and age > stale_after),
                "age_days": age,
            }
        groups_by_sector.setdefault(node.code[:2], []).append({
            "code": node.code,
            "name": node.name,
            "sector_code": node.code[:2],
            "is_active": node.is_active,
            "effective_from": node.effective_from.isoformat() if node.effective_from else None,
            "effective_to": node.effective_to.isoformat() if node.effective_to else None,
            "industry_count": industries_by_group.get(node.code, 0),
            "sub_industry_count": subs_by_group.get(node.code, 0),
            "constituent_count": len(by_group.get(node.code, [])),
            "latest_report": pointer,
        })

    sectors = [
        {"code": n.code, "name": n.name, "industry_groups": groups_by_sector.get(n.code, [])}
        for n in nodes if n.level == "sector"
    ]
    return TaxonomyOut(
        taxonomy_version=info.as_dict(),
        sectors=sectors,
        node_counts=gics_registry.counts(version=info),
        constituents={
            "classified": sum(len(v) for v in by_group.values()),
            "groups_with_constituents": len(by_group),
            "member_states": list(MEMBER_STATES),
            "note": (
                "constituents are companies whose current classification maps to the group; "
                "a `fallback` row knows only its sector and is not counted here"
            ),
        },
        reports={"groups_with_a_published_edition": len(latest), "groups": len(groups)},
        access=IndustryAccessOut(**access),
        attribution=gics_registry.ATTRIBUTION,
        mapping_caveat=gics_registry.MAPPING_CAVEAT,
        disclaimer=store.DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _stats_projection(stats_id: int | None) -> tuple[dict[str, Any] | None, str | None]:
    """The observed layer for an edition: `method` + `sample` + `payload`,
    without `per_ticker` (that is `/companies`). Returns the reason when
    there is nothing to show, never an empty dict pretending to be data."""
    if stats_id is None:
        return None, "this edition stored no statistics row (insufficient sample, or a deterministic edition)"
    row = industry_analytics.stats_by_id(int(stats_id))
    if row is None:
        return None, f"statistics row {stats_id} is no longer on file"
    row.pop("per_ticker", None)
    return row, None


@router.get("/api/industries/{code}/report", response_model=IndustryReportOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_report(
    request: Request,
    code: str,
    version: str = Query("latest", description="`latest` (the published edition) or an edition number."),
    access: dict = Depends(require_surface(SURFACE_LATEST)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryReportOut:
    """One edition of a group's report, with its statistics row, its
    freshness and — when a later refresh failed — the attempt that
    failed. A failed week leaves the previous edition in place and is
    reported as `stale` with `last_attempt`, never as a blank page.
    """
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    if version != "latest" and not version.isdigit():
        raise _error(422, "bad_version", "version must be `latest` or an edition number", version=version)

    report = store.get(node.code, version, version=info)
    if report is None:
        attempt = store.last_attempt(node.code, version=info)
        raise _error(
            404, "no_report",
            f"no {'published edition' if version == 'latest' else f'edition {version}'} for "
            f"{node.name} ({node.code}) in taxonomy {info.version_key}",
            industry_group_code=node.code, name=node.name, taxonomy_version=info.version_key,
            last_attempt=attempt,
        )

    fresh = store.freshness(node.code, version=info, report=report)
    stats, stats_reason = _stats_projection(report.get("stats_id"))
    return IndustryReportOut(
        code=node.code,
        name=node.name,
        sector_code=node.code[:2],
        taxonomy_version=info.version_key,
        version=report["version"],
        parent_report_id=report.get("parent_report_id"),
        period_key=report.get("period_key") or "",
        as_of=report.get("as_of"),
        status=report.get("status") or "",
        is_latest_good=bool(report.get("is_latest_good")),
        report_schema_version=int(report.get("report_schema_version") or 1),
        stale=fresh["stale"],
        stale_reason=fresh["stale_reason"],
        last_attempt=fresh["last_attempt"],
        payload=report.get("payload") or {},
        stats=stats,
        stats_unavailable_reason=stats_reason,
        coverage=report.get("coverage") or {},
        freshness=report.get("freshness") or {},
        generation=report.get("generation") or {},
        degraded=report.get("degraded") or [],
        errors=report.get("errors") or [],
        sources=report.get("sources") or [],
        llm_cost_usd=report.get("llm_cost_usd"),
        generated_at=report.get("generated_at"),
        access=IndustryAccessOut(**access),
        disclaimer=report.get("disclaimer") or store.DISCLAIMER,
        attribution=report.get("attribution") or gics_registry.ATTRIBUTION,
        mapping_caveat=report.get("mapping_caveat") or gics_registry.MAPPING_CAVEAT,
    )


@router.get("/api/industries/{code}/history", response_model=IndustryHistoryOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_history(
    request: Request,
    code: str,
    limit: int = Query(HISTORY_LIMIT, ge=1, le=104),
    access: dict = Depends(require_surface(SURFACE_HISTORY)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryHistoryOut:
    """Prior editions, newest first, metadata only (no payloads). Pro when
    the login wall is on."""
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    items = store.history(node.code, limit=limit, version=info)
    # A truncated list must say how much it dropped, and "len(items) ==
    # limit, so probably more" is a guess. One COUNT gives the real number.
    with SessionLocal() as db:
        total = int(db.execute(
            select(func.count(IndustryReport.id)).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == node.code,
            )
        ).scalar() or 0)
    return IndustryHistoryOut(
        code=node.code,
        name=node.name,
        taxonomy_version=info.version_key,
        count=len(items),
        limit=limit,
        truncated=max(0, total - len(items)),
        items=items,
        last_attempt=store.last_attempt(node.code, version=info),
        access=IndustryAccessOut(**access),
        disclaimer=store.DISCLAIMER,
    )


@router.get("/api/industries/{code}/changes", response_model=IndustryChangesOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_changes(
    request: Request,
    code: str,
    from_version: int | None = Query(None, alias="from", description="Edition to compare from; default the current edition's parent."),
    to: str = Query("latest", description="`latest` or an edition number."),
    access: dict = Depends(require_surface(SURFACE_CHANGES)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryChangesOut:
    """What changed between two editions: the facts delta (pure arithmetic
    over the stored statistics rows), constituent additions and removals,
    leaders/laggards, and each edition's analyst view. A fact missing on
    either side is `null` with the side named — never differenced to zero.
    """
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    if to != "latest" and not to.isdigit():
        raise _error(422, "bad_version", "`to` must be `latest` or an edition number", to=to)

    target = store.get(node.code, to, version=info)
    if target is None:
        raise _error(404, "no_report", f"no edition {to!r} for {node.name} ({node.code})",
                     industry_group_code=node.code, taxonomy_version=info.version_key)
    source_version = from_version
    if source_version is None:
        parent = target.get("parent_report_id")
        if parent is None:
            raise _error(
                404, "no_prior_edition",
                f"{node.name} ({node.code}) edition {target['version']} is the first on file; "
                "there is no prior edition to compare it with",
                industry_group_code=node.code, version=target["version"],
            )
        source_version = max(1, int(target["version"]) - 1)
    try:
        delta = store.diff(node.code, source_version, to, version=info)
    except store.ReportNotFound as exc:
        raise _error(404, "no_report", str(exc), industry_group_code=node.code) from None
    return IndustryChangesOut(
        code=node.code,
        name=node.name,
        taxonomy_version=delta["taxonomy_version"],
        **{"from": delta["from"]},
        to=delta["to"],
        adjacent=bool(delta["adjacent"]),
        facts_delta=delta["facts_delta"],
        constituents=delta["constituents"],
        leaders_laggards=delta["leaders_laggards"],
        analyst_view=delta["analyst_view"],
        access=IndustryAccessOut(**access),
        disclaimer=delta["disclaimer"],
    )


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def _membership(code: str, info: gics_registry.VersionInfo) -> list[tuple[Any, Any]]:
    """The group's classified membership, joined to `companies`, in ONE
    statement.

    Deliberately not `constituents()` + `current_for()` + a companies
    read: three round-trips for one page, which is exactly the N+1 shape
    the analytics review flagged. Ordered by ticker so the response is
    stable between calls.
    """
    q = (
        select(CompanyIndustryClassification, Company)
        .join(Company, Company.ticker == CompanyIndustryClassification.ticker)
        .where(
            CompanyIndustryClassification.taxonomy_version_id == info.id,
            CompanyIndustryClassification.is_current.is_(True),
            CompanyIndustryClassification.industry_group_code == code,
            CompanyIndustryClassification.state.in_(MEMBER_STATES),
            Company.is_active.is_(True),
        )
        .order_by(CompanyIndustryClassification.ticker)
    )
    with SessionLocal() as db:
        # Unpacked rather than `list(...)`: the rows are detached after the
        # session closes, and the pair is what every caller wants.
        return [(row[0], row[1]) for row in db.execute(q).all()]


@router.get("/api/industries/{code}/companies", response_model=IndustryCompaniesOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_companies(
    request: Request,
    code: str,
    limit: int = Query(COMPANIES_LIMIT, ge=1, le=COMPANIES_LIMIT),
    access: dict = Depends(require_surface(SURFACE_LATEST)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryCompaniesOut:
    """The group's constituents: who is in it, how each one got there, and
    which of them the latest statistics row could price.

    Three queries: the active version, one membership join, one statistics
    read. Sub-industry names come from the cached registry nodes, so the
    8-digit layer costs nothing extra.
    """
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    rows = _membership(node.code, info)
    stats = industry_analytics.latest_stats(node.code, version=info)
    per_ticker: dict[str, Any] = (stats or {}).get("per_ticker") or {}
    stats_reason = None if stats else (
        "no statistics row for this group yet — membership is shown without prices"
    )

    states: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    n_priced = 0
    for classification, company in rows[:limit]:
        states[classification.state] = states.get(classification.state, 0) + 1
        stat = per_ticker.get(classification.ticker) or {}
        priced = bool(stat) and stat.get("last_close") is not None and not stat.get("exclusion")
        if priced:
            n_priced += 1
        sub_code = classification.sub_industry_code
        sub_node = gics_registry.node(sub_code, version=info, include_inactive=True) if sub_code else None
        industry_node = (
            gics_registry.node(classification.industry_code, version=info, include_inactive=True)
            if classification.industry_code else None
        )
        items.append({
            "ticker": classification.ticker,
            "company_name": company.company_name,
            "is_active": bool(company.is_active),
            "industry_code": classification.industry_code,
            "industry_name": industry_node.name if industry_node else None,
            "sub_industry_code": sub_code,
            "sub_industry_name": sub_node.name if sub_node else None,
            "sub_industry_codes": list(classification.sub_industry_codes or []),
            "classification": {
                "state": classification.state,
                "source": classification.source,
                "source_label": _SOURCE_LABELS.get(classification.source, classification.source),
                "method": classification.method,
                "author": classification.author or "",
                "as_of": classification.source_as_of or "",
                "confidence": classification.confidence,
                "classified_at": classification.classified_at.isoformat() if classification.classified_at else None,
                "mapping_caveat": gics_registry.MAPPING_CAVEAT,
            },
            "market_cap": stat.get("market_cap"),
            "weight_mcw": stat.get("weight_mcw"),
            "last_close": stat.get("last_close"),
            "last_date": stat.get("last_date"),
            "price_source": stat.get("price_source"),
            "returns": stat.get("returns") or {},
            "return_reasons": stat.get("return_reasons") or {},
            "above_50d_mean": stat.get("above_50d_mean"),
            "priced": priced,
            # Never a blank: a name the statistics row never saw is as
            # unpriced as one it excluded, and both say why.
            "unpriced_reason": None if priced else (
                stat.get("exclusion") or (
                    "not in the latest statistics row" if not stat else "no close at the as-of date"
                )
            ),
        })
    return IndustryCompaniesOut(
        code=node.code,
        name=node.name,
        sector_code=node.code[:2],
        taxonomy_version=info.version_key,
        as_of=(stats or {}).get("as_of"),
        count=len(rows),
        n_priced=n_priced,
        membership_source=f"company_industry_classifications (states: {', '.join(MEMBER_STATES)})",
        membership_states=states,
        stats=None if stats is None else {
            "id": stats["id"], "period_key": stats["period_key"], "as_of": stats["as_of"],
            "sample": stats.get("sample") or {}, "method": stats.get("method") or {},
        },
        stats_unavailable_reason=stats_reason,
        limit=limit,
        truncated=max(0, len(rows) - len(items)),
        items=items,
        excluded=list(((stats or {}).get("sample") or {}).get("excluded") or []),
        access=IndustryAccessOut(**access),
        attribution=gics_registry.ATTRIBUTION,
        mapping_caveat=gics_registry.MAPPING_CAVEAT,
        security_reference_caveat=knowledge.SECURITY_REFERENCE_CAVEAT,
        disclaimer=store.DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Cross-industry snapshot
# ---------------------------------------------------------------------------


@router.get("/api/industries/snapshot", response_model=IndustrySnapshotOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_snapshot(
    request: Request,
    period_key: str | None = Query(None, description="ISO week; default the newest snapshot."),
    access: dict = Depends(require_surface(SURFACE_SNAPSHOT)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustrySnapshotOut:
    """The compact cross-industry view the PM reads, as stored. One row;
    the worker computed it, this only reads it."""
    info = _taxonomy_or_503(access)
    row = (
        industry_snapshot.snapshot_for_period(period_key, version=info) if period_key
        else industry_snapshot.latest_snapshot(version=info)
    )
    if row is None:
        raise _error(
            404, "no_snapshot",
            f"no cross-industry snapshot on file for taxonomy {info.version_key}"
            + (f" period {period_key}" if period_key else ""),
            taxonomy_version=info.version_key, period_key=period_key,
        )
    return IndustrySnapshotOut(
        id=row["id"],
        taxonomy_version=info.version_key,
        period_key=row["period_key"],
        as_of=row["as_of"],
        schema_version=row["schema_version"],
        payload=row["payload"],
        stats_ids=row["stats_ids"],
        report_versions=row["report_versions"],
        computed_at=row["computed_at"],
        access=IndustryAccessOut(**access),
        disclaimer=store.DISCLAIMER,
    )
