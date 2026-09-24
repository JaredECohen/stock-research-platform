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
  `count` are different fields on purpose — and the same distinction says
  why a group is below the sample floor: `/taxonomy`'s `universe_coverage`
  names the groups this universe holds too few CLASSIFIED constituents to
  ever cover, which is a different statement from a week whose prices have
  not warmed up yet, and (via `uncounted`) a different statement again
  from a universe short of companies.
* **Numbers travel with their method.** The report response carries the
  statistics row's `method` and `sample` whenever it carries `payload`,
  because `benchmark_relative` without `method.benchmark_cohort_basis`
  and breadth without `method.breadth_mean_window.sessions` are numbers
  a reader cannot check.
* **Staleness is composed, not guessed.** `stale` is the store's answer:
  an as-of older than `INDUSTRY_REPORT_STALE_AFTER_DAYS`, or a refresh
  attempt that failed after this edition was generated — with
  `last_attempt` naming the failure. A failed week never blanks the page.
* **Only analyst-written editions are served** (owner decision 1,
  2026-09-24; the rule is `industry_report_store.is_publishable`). A week
  that produced only an audit-only template leaves the last analyst
  edition up, `stale` with "not updated this week" and
  `display.not_updated` naming the week; `last_attempt` never quotes the
  rejected model prose a validator message carries.
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
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select

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

# Why a group with too few constituents is a different statement from a
# group whose prices have not warmed up yet. Stated once, on the
# response, because the page must not compose this claim itself.
UNIVERSE_COVERAGE_BASIS = (
    "a group whose classified constituent count is below the sample floor cannot reach the floor "
    "however many weeks the price warm-up runs — it is short of CLASSIFIED CONSTITUENTS, not of "
    "prices, and its statistics will read insufficient_sample every week until more companies are "
    "classified into it. That is not the same as being short of companies: `uncounted` counts the "
    "companies already in this universe that no group counts (a `fallback` row knows its sector and "
    "not its group, a `missing` row carries a label the alias map does not recognise, a `stale` row "
    "awaits re-classification), and classifying those closes the same gap without widening the "
    "universe. Groups counted here are every industry group in this taxonomy version; membership "
    "comes from the classification table on every call."
)


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """``1 company`` / ``4 companies`` — these strings are read by a
    person, and "1 companies" reads as a fault in the number."""
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def _uncounted_phrase(by_state: dict[str, int]) -> str:
    """``3 fallback, 1 stale`` — the states, so a reader knows which fix
    each row needs rather than being told a bare total."""
    return ", ".join(f"{n} {state}" for state, n in sorted(by_state.items()))


def _group_universe_coverage(
    constituent_count: int, floor: int, uncounted: dict[str, Any], group_code: str,
) -> dict[str, Any]:
    """The structural half of the sample-floor story for one group.

    Deliberately NOT the three-state classifier the analytics row carries:
    this endpoint knows the membership and not the week's price coverage,
    and a response that guessed the other half would be inventing it.
    `industry_analytics.universe_covers` is the shared definition of
    structural, so the two layers cannot disagree.

    The shortfall is counted in CLASSIFIED CONSTITUENTS, never in
    companies: a universe can hold plenty of companies for a group and
    still count none of them, and telling an operator to add companies
    when the alias map is what is short sends them the wrong way.
    """
    short_by = max(floor - constituent_count, 0)
    coverable = industry_analytics.universe_covers(constituent_count, floor)
    classified = _plural(constituent_count, "classified constituent")
    uncounted_total = int(uncounted.get("total") or 0)
    for_group = int((uncounted.get("by_group_code") or {}).get(group_code) or 0)
    if coverable:
        explanation = (
            f"{classified} in this universe, at or above the sample floor of {floor}; a week below the "
            f"floor is about prices, not about the size of this universe."
        )
    else:
        if uncounted_total:
            names_group = f"; {for_group} of them already name this group" if for_group else ""
            remedy = (
                f"which can come from companies added to the universe, or from the "
                f"{_plural(uncounted_total, 'company', 'companies')} already in this universe that no "
                f"group counts ({_uncounted_phrase(uncounted.get('by_state') or {})}{names_group}) — "
                f"classifying those adds no companies."
            )
        else:
            remedy = (
                "and every company in this universe already counts towards a group, so only adding "
                "companies can supply them."
            )
        explanation = (
            f"this universe holds {classified} for the group and the sample floor is {floor}. No amount of "
            f"price warm-up can cover it — the group needs "
            f"{_plural(short_by, 'more classified constituent', 'more classified constituents')}, {remedy}"
        )
    return {
        "min_sample": floor,
        "constituent_count": constituent_count,
        "coverable": coverable,
        "constituents_short_by": short_by,
        # The two numbers that keep "short of constituents" from being
        # read as "short of companies".
        "uncounted_for_group": for_group,
        "uncounted_in_universe": uncounted_total,
        "explanation": explanation,
    }


def _universe_coverage_explanation(
    *, groups: int, not_coverable: int, needed: int, floor: int, uncounted: dict[str, Any],
) -> str:
    """The taxonomy-level sentence the index page prints verbatim. Written
    here for the same reason every per-group explanation is: a page that
    composes it is making a claim the server never made."""
    uncounted_total = int(uncounted.get("total") or 0)
    if not not_coverable:
        return (
            f"Every one of the {_plural(groups, 'industry group')} in this taxonomy holds at least "
            f"{floor} classified constituents, so a group below the floor in a given week is waiting on "
            f"prices, not on the universe."
        )
    if uncounted_total:
        tail = (
            f"The weekly price warm-up cannot supply them. "
            f"{_plural(uncounted_total, 'company', 'companies')} in this universe carry a classification "
            f"no group counts ({_uncounted_phrase(uncounted.get('by_state') or {})}); classifying those "
            f"counts towards the gap without widening the universe."
        )
    else:
        tail = (
            "The weekly price warm-up cannot supply them, and every company in this universe already "
            "counts towards a group, so only adding companies can."
        )
    return (
        f"{not_coverable} of {groups} industry groups hold fewer than {floor} classified constituents "
        f"each and will report insufficient_sample every week until that changes — "
        f"{_plural(needed, 'more classified constituent', 'more classified constituents')} in total. {tail}"
    )


_SOURCE_LABELS = {
    industry_classification.SOURCE_RESEARCH_MAP:
        "research map (economic research examples; not licensed issuer GICS mapping)",
    industry_classification.SOURCE_PROVIDER_ALIAS:
        "provider label crosswalk (derived from provider classification)",
    industry_classification.SOURCE_NONE: "unresolved",
}


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time.

    It defers to the store's clock rather than reading its own, because
    `/taxonomy`'s `stale_by_age` and `/report`'s `stale` are two renderings
    of ONE verdict. Two independent seams could be patched apart in a test
    (and skewed apart by two processes' clocks in production), which is
    exactly the drift that let the picker call an edition fresh while its
    own page called it stale.
    """
    return store._utcnow()


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


def _age(as_of: str | None, now: datetime) -> timedelta | None:
    """How old an edition's as-of is, or `None` when it has none."""
    if not as_of:
        return None
    try:
        return max(timedelta(0), now - datetime.fromisoformat(as_of))
    except ValueError:  # pragma: no cover — stored values are isoformat
        return None


def _stale_by_age(age: timedelta | None, stale_after_days: int) -> bool:
    """The SAME comparison `industry_report_store.freshness` makes.

    It must be the full timedelta, not `age.days`: flooring to whole days
    first made an edition 10 days and 12 hours old read as `age_days: 10`
    and therefore fresh on `/taxonomy`, while `/report` — comparing
    `now - as_of > timedelta(days=10)` — called the same edition stale.
    The picker and the page disagreed for everything in the [N, N+1) day
    window. `age_days` stays a floored whole number because that is what
    a UI renders; only the verdict is computed from the real age.
    """
    return age is not None and age > timedelta(days=stale_after_days)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@router.get("/api/industries/taxonomy", response_model=TaxonomyOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_taxonomy(
    request: Request,
    response: Response,
    access: dict = Depends(describe_surface(SURFACE_LATEST)),
    _rate: None = Depends(rate_scope("data")),
) -> TaxonomyOut:
    """The active structure: sectors → industry groups, with each group's
    industry and sub-industry counts, its constituent count, whether this
    universe can cover the group at the sample floor, and the pointer to
    its latest published edition.

    `universe_coverage` answers, in one place, "how much of this taxonomy
    can the current universe never report on": a group whose membership is
    below the floor is short of CLASSIFIED CONSTITUENTS, and no number of
    price warm-up weeks changes that. It counts the rows no group counts
    (`uncounted`) alongside it, because "add companies" and "classify the
    companies already here" are different instructions and the shortfall
    alone does not say which one an operator needs. It is the STRUCTURAL
    half only — this endpoint reads no statistics row and so cannot know
    which groups are merely un-warmed; that half is
    `stats.sample.sample_floor` on the report.

    Reads: the active version, the version's nodes (cached per version id
    — they are immutable after import), one constituent count query, one
    uncounted-rows query, the two bulk latest-publishable-edition queries
    and one grouped attempted-periods query. Never a query per group.
    """
    info = _taxonomy_or_503(access)
    nodes = gics_registry.nodes(version=info)
    by_group = industry_classification.constituents_by_group(version=info)
    # The other half of the membership picture: companies in this universe
    # that no group's constituent count includes. Without it a group short
    # of CLASSIFIED constituents gets reported as a universe short of
    # COMPANIES, and an operator widens the universe when the alias map is
    # what needs extending.
    uncounted = industry_classification.uncounted_rows(version=info)

    groups = [n for n in nodes if n.level == "industry_group"]
    # Publishable editions only (owner decision 1): a group whose newest
    # edition is an audit-only template points at its last ANALYST edition,
    # and the attempted-periods read (one grouped query) says it is older
    # than the week that was attempted.
    latest = store.latest_publishable_many([g.code for g in groups], version=info)
    attempted = store.last_attempted_periods([g.code for g in groups], version=info)
    industries_by_group: dict[str, int] = {}
    subs_by_group: dict[str, int] = {}
    for node in nodes:
        if node.level == "industry":
            industries_by_group[node.code[:4]] = industries_by_group.get(node.code[:4], 0) + 1
        elif node.level == "sub_industry":
            subs_by_group[node.code[:4]] = subs_by_group.get(node.code[:4], 0) + 1

    now = _utcnow()
    stale_after = int(settings.industry_report_stale_after_days)
    floor = industry_analytics.sample_floor()
    not_coverable: list[str] = []
    constituents_needed = 0
    groups_by_sector: dict[str, list[dict[str, Any]]] = {}
    for node in groups:
        edition = latest.get(node.code)
        pointer = None
        if edition is not None:
            age = _age(edition.get("as_of"), now)
            newer = attempted.get(node.code)
            newer = newer if newer and newer > str(edition.get("period_key") or "") else None
            pointer = {
                "version": edition["version"],
                "period_key": edition.get("period_key") or "",
                "as_of": edition.get("as_of"),
                "status": edition.get("status") or "",
                "generated_at": edition.get("generated_at"),
                "degraded": bool(edition.get("degraded")),
                "degraded_reasons": list(edition.get("degraded") or []),
                "stale_by_age": _stale_by_age(age, stale_after),
                "age_days": None if age is None else age.days,
                "not_updated": newer is not None,
                "newer_attempt_period_key": newer,
            }
        constituent_count = len(by_group.get(node.code, []))
        coverage = _group_universe_coverage(constituent_count, floor, uncounted, node.code)
        if not coverage["coverable"]:
            not_coverable.append(node.code)
            constituents_needed += coverage["constituents_short_by"]
        groups_by_sector.setdefault(node.code[:2], []).append({
            "code": node.code,
            "name": node.name,
            "sector_code": node.code[:2],
            "is_active": node.is_active,
            "effective_from": node.effective_from.isoformat() if node.effective_from else None,
            "effective_to": node.effective_to.isoformat() if node.effective_to else None,
            "industry_count": industries_by_group.get(node.code, 0),
            "sub_industry_count": subs_by_group.get(node.code, 0),
            "constituent_count": constituent_count,
            "universe_coverage": coverage,
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
        universe_coverage={
            "min_sample": floor,
            "setting": "INDUSTRY_STATS_MIN_SAMPLE",
            "groups": len(groups),
            "coverable": len(groups) - len(not_coverable),
            "not_coverable": len(not_coverable),
            "not_coverable_codes": sorted(not_coverable),
            "constituents_needed": constituents_needed,
            "uncounted": uncounted,
            "explanation": _universe_coverage_explanation(
                groups=len(groups), not_coverable=len(not_coverable),
                needed=constituents_needed, floor=floor, uncounted=uncounted,
            ),
            "basis": UNIVERSE_COVERAGE_BASIS,
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
    response: Response,
    code: str,
    version: str = Query("latest", description="`latest` (the published edition) or an edition number."),
    access: dict = Depends(require_surface(SURFACE_LATEST)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryReportOut:
    """One edition of a group's report, with its statistics row, its
    freshness and — when a later refresh failed — the attempt that
    failed. A failed week leaves the previous edition in place and is
    reported as `stale` with `last_attempt`, never as a blank page.

    Only analyst-written editions are served (owner decision 1). `latest`
    is the newest publishable one; a group that has never had one answers
    404 `no_report` with `reason: no_validated_analyst_edition` and a count
    of the audit-only editions it does hold. An explicit `version=N` that
    names an audit-only edition answers 404 `edition_withheld`. Sections a
    template filled inside an analyst edition come back with a null
    `interpretation` and are listed in `display.hidden_sections`.
    """
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    if version != "latest" and not version.isdigit():
        raise _error(422, "bad_version", "version must be `latest` or an edition number", version=version)

    try:
        report = store.get(node.code, version, version=info)
    except store.EditionWithheld as exc:
        raise _error(
            404, "edition_withheld",
            f"edition {exc.version} of {node.name} ({node.code}) is kept for audit only and is not published",
            industry_group_code=node.code, name=node.name, taxonomy_version=info.version_key,
            version=exc.version,
        ) from None
    if report is None:
        attempt = store.public_attempt(store.last_attempt(node.code, version=info))
        if version == "latest":
            raise _error(
                404, "no_report",
                f"no analyst-written edition has been published for {node.name} ({node.code}) in "
                f"taxonomy {info.version_key}",
                industry_group_code=node.code, name=node.name, taxonomy_version=info.version_key,
                reason="no_validated_analyst_edition",
                withheld_editions=store.withheld_count(node.code, version=info),
                last_attempt=attempt,
            )
        raise _error(
            404, "no_report",
            f"no edition {version} for {node.name} ({node.code}) in taxonomy {info.version_key}",
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
        last_attempt=store.public_attempt(fresh["last_attempt"]),
        payload=store.public_payload(report),
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
        display=store.display_block(report, fresh),
    )


@router.get("/api/industries/{code}/history", response_model=IndustryHistoryOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_history(
    request: Request,
    response: Response,
    code: str,
    limit: int = Query(HISTORY_LIMIT, ge=1, le=104),
    access: dict = Depends(require_surface(SURFACE_HISTORY)),
    _rate: None = Depends(rate_scope("data")),
) -> IndustryHistoryOut:
    """Prior editions, newest first, metadata only (no payloads). Pro when
    the login wall is on.

    Audit-only editions are not listed (owner decision 1); `withheld`
    counts them, because they are why the version numbers can skip. A
    truncated list must say how much it dropped, and "len(items) == limit,
    so probably more" is a guess: `truncated` comes from the same filtered
    metadata read as the items, so withheld rows never inflate it."""
    info = _taxonomy_or_503(access)
    node = _group_or_404(code, info)
    page = store.history_page(node.code, limit=limit, version=info)
    items = page["items"]
    return IndustryHistoryOut(
        code=node.code,
        name=node.name,
        taxonomy_version=info.version_key,
        count=len(items),
        limit=limit,
        truncated=max(0, page["total"] - len(items)),
        withheld=page["withheld"],
        items=items,
        last_attempt=store.public_attempt(store.last_attempt(node.code, version=info)),
        access=IndustryAccessOut(**access),
        disclaimer=store.DISCLAIMER,
    )


def _parent_version_or_404(
    node: gics_registry.NodeInfo, info: gics_registry.VersionInfo, target: dict[str, Any],
) -> int:
    """The edition `target` actually replaced, resolved through
    `parent_report_id` — never `version - 1`.

    `save_report` points `parent_report_id` at the latest *good* edition
    at save time and increments `version` unconditionally, so the two are
    not the same thing the moment an edition is not published. With
    `INDUSTRY_REPORTS_REQUIRE_REVIEW=true` a group can hold v1
    (published), v2 (pending_review, never served) and v3 (published,
    parent v1); `version - 1` diffed v3 against the edition no reader has
    ever seen. And when the parent is None it is not proof the target is
    first on file — the earlier editions may simply all be unpublished —
    so that refusal counts them before making the claim.

    A parent that is kept for audit only (a legacy template that was
    latest-good when this edition was saved) is never a basis — its prose
    is exactly what decision 1 keeps off the page — so the newest
    publishable edition older than the target stands in for it. Audit-only
    editions are counted apart (`withheld_editions`) and never offered as
    an explicit `?from=` either.
    """
    parent_id = target.get("parent_report_id")
    parent = None
    if parent_id is not None:
        parent = store.edition_meta_by_id(int(parent_id), code=node.code, version=info)
        if parent is None:
            raise _error(
                404, "no_prior_edition",
                f"{node.name} ({node.code}) edition {target['version']} records edition id "
                f"{parent_id} as the one it replaced, but that row is no longer on file; "
                "pass ?from=<version> to choose a basis explicitly",
                industry_group_code=node.code, version=target["version"],
                parent_report_id=int(parent_id),
            )
        if not parent["withheld"]:
            return int(parent["version"])
        fallback = store.newest_publishable_below(node.code, int(target["version"]), version=info)
        if fallback is not None:
            return fallback
    with SessionLocal() as db:
        metas = db.execute(
            select(IndustryReport.status, IndustryReport.generation, IndustryReport.degraded).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == node.code,
                IndustryReport.version < int(target["version"]),
            )
        ).all()
    withheld = sum(1 for m in metas if store.is_withheld(m))
    earlier = len(metas) - withheld
    if earlier:
        raise _error(
            404, "no_prior_edition",
            f"{node.name} ({node.code}) edition {target['version']} replaced no published "
            f"edition ({earlier} earlier edition(s) exist but none was published); "
            "pass ?from=<version> to compare with one of them anyway",
            industry_group_code=node.code, version=target["version"],
            earlier_editions=earlier, withheld_editions=withheld,
        )
    if withheld:
        raise _error(
            404, "no_prior_edition",
            f"{node.name} ({node.code}) edition {target['version']} is the first analyst-written "
            f"edition; the {withheld} earlier edition(s) are kept for audit only and are not published",
            industry_group_code=node.code, version=target["version"], earlier_editions=0,
            withheld_editions=withheld,
        )
    raise _error(
        404, "no_prior_edition",
        f"{node.name} ({node.code}) edition {target['version']} is the first on file; "
        "there is no prior edition to compare it with",
        industry_group_code=node.code, version=target["version"], earlier_editions=0,
        withheld_editions=0,
    )


@router.get("/api/industries/{code}/changes", response_model=IndustryChangesOut)
@limiter.limit(LIMITS["industry_read"])
def get_industry_changes(
    request: Request,
    response: Response,
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

    def withheld_404(exc: store.EditionWithheld) -> HTTPException:
        return _error(
            404, "edition_withheld",
            f"edition {exc.version} of {node.name} ({node.code}) is kept for audit only and is not published",
            industry_group_code=node.code, version=exc.version,
        )

    try:
        target = store.get(node.code, to, version=info)
    except store.EditionWithheld as exc:
        raise withheld_404(exc) from None
    if target is None:
        raise _error(404, "no_report", f"no edition {to!r} for {node.name} ({node.code})",
                     industry_group_code=node.code, taxonomy_version=info.version_key)
    source_version = from_version
    if source_version is None:
        source_version = _parent_version_or_404(node, info, target)
    try:
        delta = store.diff(node.code, source_version, to, version=info)
    except store.EditionWithheld as exc:
        raise withheld_404(exc) from None
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


def _is_priced(stat: dict[str, Any] | None) -> bool:
    """Did the latest statistics row actually price this name? One
    definition, used by both the page rows and the `n_priced` counter, so
    the two can never drift apart."""
    if not stat:
        return False
    return stat.get("last_close") is not None and not stat.get("exclusion")


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
    response: Response,
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

    # The counters describe the WHOLE membership, the items describe the
    # page. Counting them in one loop over `rows[:limit]` made `count`
    # (all members) and `n_priced` / `membership_states` (this page)
    # describe different populations, so a capped call reported a
    # coverage figure that was simply wrong — `3 of 4 priced` became
    # `0 of 4 priced` at `?limit=1`.
    states: dict[str, int] = {}
    n_priced = 0
    for classification, _company in rows:
        states[classification.state] = states.get(classification.state, 0) + 1
        if _is_priced(per_ticker.get(classification.ticker)):
            n_priced += 1

    items: list[dict[str, Any]] = []
    for classification, company in rows[:limit]:
        stat = per_ticker.get(classification.ticker) or {}
        priced = _is_priced(stat)
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
    response: Response,
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
