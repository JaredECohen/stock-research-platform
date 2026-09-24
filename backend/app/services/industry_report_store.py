"""FEAT-003 — the versioned store for weekly Industry Analysis editions.

An edition is an ``industry_reports`` row. The rules the store enforces:

* **Versions are monotonic per (taxonomy version, group)** and every
  edition links to the edition it replaced (``parent_report_id``), so
  "what changed since last week" always has a well-defined "last week".
* **Only analyst-written editions are ever displayed** (owner decision 1,
  2026-09-24; contract C8). An edition is *publishable* when its status is
  ``succeeded``/``superseded``, its ``generation.generation_mode`` is
  ``llm``, ``drivers`` and ``outlook`` were written by the model, and at
  least ``MIN_MODEL_SECTIONS`` of the interpreted sections were. Anything
  else — the deterministic template, the ``llm_unavailable`` fallback, an
  analyst edition too thin to stand on its own — is stored with status
  ``audit_only`` and is served by no public read. ``save_report`` refuses
  to publish one even when a caller passes an explicit status.
* **Every reader applies that rule; none reads ``is_latest_good``.** Rows
  written before the rule (a template flagged latest-good, the analyst
  edition it superseded) are therefore read correctly the moment the code
  deploys, with no backfill. The flag is still maintained so admin views
  and the one-shot reclassification (below) agree with the rule, and the
  public ``status`` / ``is_latest_good`` a response carries are DERIVED
  from the rule rather than copied from the row.
* **Failures never write here.** A failed refresh leaves the last good
  edition in place; the read API composes ``stale`` and ``last_attempt``
  from the jobs table (``last_attempt``/``freshness`` below), so a reader
  sees a dated edition with the failure named — never a blank week and
  never a silently old page. A week whose only product was an audit-only
  template is reported the same way: ``freshness`` says the group was
  *not updated* and names the period.
* **Diffs are arithmetic over stored facts.** ``diff`` compares the two
  editions' stats rows (returns, breadth, dispersion, valuation medians,
  sample) and constituent sets, and quotes the newer edition's own
  ``what_changed`` text unless a template wrote it; it never re-generates
  anything.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import defer

from ..config import settings
from ..database import SessionLocal
from ..models import IndustryReport, IndustryReportJob, IndustryStatSnapshot
from . import gics_registry
from .gics_registry import ATTRIBUTION, MAPPING_CAVEAT, VersionInfo

log = logging.getLogger(__name__)

STATUS_SUCCEEDED = "succeeded"
STATUS_PENDING_REVIEW = "pending_review"
STATUS_SUPERSEDED = "superseded"
# A template (or too-thin analyst) edition: kept for audit, never served.
# Fits `industry_reports.status String(16)`, so no schema change.
STATUS_AUDIT_ONLY = "audit_only"
REPORT_STATUSES: tuple[str, ...] = (
    STATUS_SUCCEEDED, STATUS_PENDING_REVIEW, STATUS_SUPERSEDED, STATUS_AUDIT_ONLY,
)
# The statuses a publishable edition can have; `pending_review` is an
# analyst edition nobody has released yet, `audit_only` never publishes.
PUBLISHED_STATUSES: tuple[str, ...] = (STATUS_SUCCEEDED, STATUS_SUPERSEDED)
REPORT_SCHEMA_VERSION = 1

# --- the display rule (contract C8) -------------------------------------------
#
# `generation.generation_mode` is the writer's final mode for the edition
# (`industry_report_writer.write_report` stamps it next to
# `payload.analyst_narrative`, which always carries the same value). The
# filter reads `generation` rather than `payload` because it is a few
# hundred bytes against a payload of tens of kilobytes, and the candidate
# scan below reads it for every edition of a group.
AGENTIC_MODE = "llm"
EDITION_AGENTIC = "agentic"
EDITION_TEMPLATE = "template"
# The interpreted sections, in the writer's order. A copy of
# `industry_report_validator.INTERPRETED_SECTIONS` rather than an import:
# this module is on the web request path and importing `app.agents` would
# load the memo orchestrator into it. `test_industry_report_store` pins the
# two tuples equal.
INTERPRETED_SECTIONS: tuple[str, ...] = (
    "overview", "drivers", "kpis", "performance", "companies", "themes",
    "cross_industry", "outlook", "risks", "what_changed",
)
# An analyst edition in which the model wrote neither the causal chain nor
# the forward view is a template with a few paragraphs of model prose on
# top; it is not published as the analyst's.
REQUIRED_MODEL_SECTIONS: tuple[str, ...] = ("drivers", "outlook")
MIN_MODEL_SECTIONS = 6
_TEMPLATE_MARKER_PREFIX = "analyst_narrative:"
_TEMPLATE_MARKER_SUFFIX = ":deterministic"

HIDDEN_SECTION_REASON = "Analyst interpretation unavailable in this version."

# The one-shot legacy reclassification's ledger. It is an
# `industry_report_jobs` row that is written already finished, so no
# drainer ever claims it (claims select `status == 'queued'`), no report
# read sees it (they filter `kind == 'group_report'`), and it needs no new
# table. Its `progress` holds the manifest: the audit trail and what a
# revert reads.
RECLASSIFY_KIND = "reclassify"
RECLASSIFY_RUN_ID = "reclassify_industry_editions:v1"
RECLASSIFY_REVERT_RUN_ID = "reclassify_industry_editions:v1:revert"
RECLASSIFY_PERIOD_KEY = "RECLASSIFY-V1"

DISCLAIMER = (
    "Research and education only. Industry statistics are observed data; the "
    "analyst narrative is interpretation and every forward view is a scenario, "
    "not a recommendation or personalised advice."
)

# The facts a diff walks, as (label, path into stats_dict) pairs. Kept
# explicit so a report reader can name what "facts_delta" covers.
_FACT_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    [(f"returns.{h}.ew", ("payload", "returns", h, "equal_weight")) for h in ("1w", "1m", "qtd", "ytd", "1y")]
    + [(f"returns.{h}.mcw", ("payload", "returns", h, "market_cap_weight")) for h in ("1w", "1m", "qtd", "ytd", "1y")]
    + [
        ("breadth.1m", ("payload", "breadth", "1m", "pct_positive")),
        ("breadth.1w", ("payload", "breadth", "1w", "pct_positive")),
        ("dispersion.1m.stdev", ("payload", "dispersion", "stdev")),
        ("valuation.ev_ebitda.median", ("payload", "valuation", "ev_ebitda", "median")),
        ("valuation.pe_ttm.median", ("payload", "valuation", "pe_ttm", "median")),
        ("valuation.ev_revenue.median", ("payload", "valuation", "ev_revenue", "median")),
        ("fundamentals.revenue_growth_yoy.median", ("payload", "fundamentals", "revenue_growth_yoy", "median")),
        ("fundamentals.op_margin.median", ("payload", "fundamentals", "op_margin", "median")),
        ("rel_1m_vs_universe", ("payload", "benchmark_relative", "universe_ew", "1m", "value")),
        ("sample.n_with_prices", ("sample", "n_with_prices")),
        ("sample.n_constituents", ("sample", "n_constituents")),
    ]
)


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


class ReportNotFound(LookupError):
    """No edition matches — a route answers 404 with this."""


class EditionWithheld(ReportNotFound):
    """The edition exists and is kept for audit only (a template, or an
    analyst edition below the minimum). A subclass of ``ReportNotFound`` so
    a caller that only knows "not found" still refuses it, while a route
    can say which of the two it is."""

    def __init__(self, code: str, version: int, reason: str) -> None:
        super().__init__(f"industry report {code} edition {version} is kept for audit only ({reason})")
        self.code = code
        self.version = int(version)
        self.reason = reason


class ReclassifyRefused(RuntimeError):
    """The legacy reclassification will not run now (a report job is
    queued or running, and its save must not interleave with this one)."""


class ReclassifyAborted(RuntimeError):
    """A compare-and-set or a post-write invariant did not hold; the whole
    transaction was rolled back and nothing changed."""


# ---------------------------------------------------------------------------
# The display rule
# ---------------------------------------------------------------------------


def _get_field(row: Any, name: str) -> Any:
    """``row.name`` for an ORM row / named tuple, ``row[name]`` for a dict."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def edition_kind(generation: Any) -> str:
    """``agentic`` iff ``generation['generation_mode'] == 'llm'``.

    Everything else — ``deterministic``, ``llm_unavailable``, an empty or
    missing mode, a non-dict — is a template edition. Unknown values fail
    closed: a mode this code has never seen is not evidence that a model
    wrote the edition. ``agentic_clause`` is the SQL twin of this function
    and the two are pinned to agree on legacy rows."""
    mode = generation.get("generation_mode") if isinstance(generation, dict) else None
    return EDITION_AGENTIC if mode == AGENTIC_MODE else EDITION_TEMPLATE


def agentic_clause() -> Any:
    """``edition_kind == 'agentic'`` as a WHERE clause (``json_extract`` on
    sqlite, ``->>`` on Postgres). A JSON null or a missing key is SQL NULL,
    which the comparison treats as false — the same fail-closed answer."""
    return IndustryReport.generation["generation_mode"].as_string() == AGENTIC_MODE


def template_sections(degraded: Any, payload: dict[str, Any] | None = None) -> list[str]:
    """The interpreted sections a template filled, in the writer's order.

    The writer lists every section the model did not write as
    ``analyst_narrative:<section>:deterministic`` in ``degraded`` (a row
    column, so it is readable without the payload). When the payload is to
    hand its ``narrative_by_section`` is read too, and a section either
    source calls a template is a template — the two are written together,
    and if they ever disagreed the stricter answer is the safe one.
    Meaningful for an agentic edition; a template edition is a template
    everywhere (``model_written_sections`` says so)."""
    found: set[str] = set()
    for item in degraded or []:
        if (isinstance(item, str) and item.startswith(_TEMPLATE_MARKER_PREFIX)
                and item.endswith(_TEMPLATE_MARKER_SUFFIX)):
            found.add(item[len(_TEMPLATE_MARKER_PREFIX):-len(_TEMPLATE_MARKER_SUFFIX)])
    by_section = (payload or {}).get("narrative_by_section") if isinstance(payload, dict) else None
    if isinstance(by_section, dict):
        found.update(name for name, mode in by_section.items() if mode != AGENTIC_MODE)
    return [name for name in INTERPRETED_SECTIONS if name in found]


def model_written_sections(generation: Any, degraded: Any,
                           payload: dict[str, Any] | None = None) -> list[str]:
    """The interpreted sections the model wrote. Empty for a template."""
    if edition_kind(generation) != EDITION_AGENTIC:
        return []
    templated = set(template_sections(degraded, payload))
    return [name for name in INTERPRETED_SECTIONS if name not in templated]


def withheld_reason(generation: Any, degraded: Any, payload: dict[str, Any] | None = None) -> str | None:
    """Why this edition's CONTENT may not be published, or ``None`` when
    it may. Status is a separate question (``is_publishable``)."""
    if edition_kind(generation) != EDITION_AGENTIC:
        mode = generation.get("generation_mode") if isinstance(generation, dict) else None
        return f"template edition (generation_mode={mode or 'not recorded'})"
    written = model_written_sections(generation, degraded, payload)
    missing = [name for name in REQUIRED_MODEL_SECTIONS if name not in written]
    if missing or len(written) < MIN_MODEL_SECTIONS:
        parts = [f"{len(written)} of {len(INTERPRETED_SECTIONS)} interpreted sections model-written "
                 f"(minimum {MIN_MODEL_SECTIONS})"]
        if missing:
            parts.append(f"not model-written: {', '.join(missing)}")
        return "analyst edition below the publication minimum: " + "; ".join(parts)
    return None


def content_publishable(generation: Any, degraded: Any, payload: dict[str, Any] | None = None) -> bool:
    """The content half of C8: agentic, drivers + outlook model-written and
    at least ``MIN_MODEL_SECTIONS`` interpreted sections model-written."""
    return withheld_reason(generation, degraded, payload) is None


def is_withheld(row: Any) -> bool:
    """Kept for audit only: an ``audit_only`` row, or any row whose content
    fails the rule — including a legacy template still marked
    ``succeeded``. ``pending_review`` analyst editions are NOT withheld;
    they are unreleased, which is a different state."""
    if _get_field(row, "status") == STATUS_AUDIT_ONLY:
        return True
    return not content_publishable(_get_field(row, "generation"), _get_field(row, "degraded"))


def is_publishable(row: Any) -> bool:
    """C8: status ``succeeded``/``superseded`` AND publishable content.
    Accepts an ORM row, a metadata row or a dict with the same fields."""
    return (_get_field(row, "status") in PUBLISHED_STATUSES
            and content_publishable(_get_field(row, "generation"), _get_field(row, "degraded")))


# ---------------------------------------------------------------------------
# Access policy (owner decision 1) — the Phase 4 seam
# ---------------------------------------------------------------------------

# The surfaces this store and the PM integration expose, and the tier each
# one needs. `latest` follows INDUSTRY_ANALYSIS_ACCESS (default "public");
# the rest are Pro — the deeper reads are the paid part of the feature.
PUBLIC = "public"
PRO = "pro"
_SURFACE_TIERS: dict[str, str] = {
    "latest": PUBLIC,      # the current edition for one group
    "history": PRO,        # prior editions
    "changes": PRO,        # changes-since-prior (`diff`)
    "pm_chat": PRO,        # the industry block in Ask-the-PM / the chat tool
}


def access_policy() -> dict[str, Any]:
    """What each Industry Analysis surface costs, and whether that is being
    enforced right now.

    Two different questions, kept apart on purpose. ``surfaces`` is the
    policy — stable, and what a UI shows when it explains gating. ``enforced``
    is whether ``AUTH_ENABLED`` is on; with the login wall off every surface
    answers to everyone, and a caller that reported the policy as the served
    tier would be telling the user they paid for something they did not.

    ``INDUSTRY_ANALYSIS_ACCESS`` moves the ``latest`` surface only: set it to
    ``pro`` and the whole feature is Pro. No route enforces this yet (there is
    no read API until slice 4); this is the single place that answer lives, so
    the routes and the chat tool agree rather than each inventing one."""
    setting = str(settings.industry_analysis_access or PUBLIC).strip().lower()
    if setting not in (PUBLIC, PRO):
        log.warning("unknown INDUSTRY_ANALYSIS_ACCESS %r — falling back to %r", setting, PRO)
        setting = PRO  # an unreadable policy fails closed, never open
    surfaces = dict(_SURFACE_TIERS)
    surfaces["latest"] = setting
    return {
        "setting": setting,
        "surfaces": surfaces,
        "auth_enabled": bool(settings.auth_enabled),
        "enforced": bool(settings.auth_enabled),
        "note": (
            "tiers are the policy; with AUTH_ENABLED off nothing is gated and every surface "
            "answers publicly. Pro surfaces meter through the pm_chat / portfolio features."
        ),
    }


def surface_tier(surface: str) -> str:
    """The tier one surface needs. Unknown surfaces fail closed at Pro."""
    return access_policy()["surfaces"].get(str(surface), PRO)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def report_dict(row: IndustryReport, *, include_payload: bool = True,
                latest_publishable_version: int | None = None) -> dict[str, Any]:
    """Detached projection of an edition. ``include_payload=False`` is the
    history shape (metadata only).

    ``status`` and ``is_latest_good`` are the PUBLIC flags, derived from
    the display rule and the group's newest publishable version (pass it as
    ``latest_publishable_version``): a legacy analyst edition that a
    template superseded reads ``succeeded``/latest, and a withheld row reads
    ``audit_only`` whatever its stored status says. ``stored_status`` keeps
    the row's own value for admin callers and tests."""
    withheld = is_withheld(row)
    if withheld:
        status, latest = STATUS_AUDIT_ONLY, False
    elif row.status == STATUS_PENDING_REVIEW:
        status, latest = STATUS_PENDING_REVIEW, False
    elif latest_publishable_version is not None and row.version == latest_publishable_version:
        status, latest = STATUS_SUCCEEDED, True
    else:
        status, latest = STATUS_SUPERSEDED, False
    out: dict[str, Any] = {
        "id": row.id,
        "taxonomy_version_id": row.taxonomy_version_id,
        "code": row.industry_group_code,
        "version": row.version,
        "parent_report_id": row.parent_report_id,
        "period_key": row.period_key,
        "as_of": row.as_of.isoformat() if row.as_of else None,
        "status": status,
        "is_latest_good": latest,
        "stored_status": row.status,
        "withheld": withheld,
        "edition_kind": edition_kind(row.generation),
        "report_schema_version": row.report_schema_version,
        "stats_id": row.stats_id,
        "coverage": dict(row.coverage or {}),
        "freshness": dict(row.freshness or {}),
        "generation": dict(row.generation or {}),
        "degraded": list(row.degraded or []),
        "errors": list(row.errors or []),
        "job_id": row.job_id,
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "llm_cost_usd": (row.generation or {}).get("cost_usd"),
        "disclaimer": DISCLAIMER,
        "attribution": ATTRIBUTION,
        "mapping_caveat": MAPPING_CAVEAT,
    }
    if include_payload:
        out["payload"] = dict(row.payload or {})
        out["sources"] = list(row.source_manifest or [])
    return out


def hidden_sections(report: dict[str, Any]) -> list[str]:
    """The template-filled sections of an agentic edition — what a reader
    is told is unavailable rather than shown (decision 2's principle,
    applied to reports). Reads the row's markers and the payload's
    per-section record."""
    return template_sections(report.get("degraded"), report.get("payload"))


def public_payload(report: dict[str, Any]) -> dict[str, Any]:
    """The payload with every hidden section's ``interpretation`` replaced
    by ``None`` — in the RESPONSE only. Facts stay: they are the server's
    observations, not template prose. The stored payload is never touched
    (this copies the two levels it edits)."""
    payload = dict(report.get("payload") or {})
    hidden = set(hidden_sections(report))
    if not hidden:
        return payload
    sections = {name: dict(sec) if isinstance(sec, dict) else sec
                for name, sec in (payload.get("sections") or {}).items()}
    for name in hidden:
        if isinstance(sections.get(name), dict):
            sections[name]["interpretation"] = None
    payload["sections"] = sections
    return payload


def display_block(report: dict[str, Any], fresh: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the page needs to present an edition honestly: its kind, the
    sections it must not show, and — when a newer week produced no
    validated analyst edition — which week that was. ``not_updated`` is set
    only for a FINISHED newer attempt; one still in flight is not news yet."""
    not_updated = (fresh or {}).get("not_updated")
    if not_updated and not_updated.get("outcome") not in ("withheld_template", "failed"):
        not_updated = None
    return {
        "edition_kind": report.get("edition_kind") or edition_kind(report.get("generation")),
        "hidden_sections": hidden_sections(report),
        "hidden_reason": HIDDEN_SECTION_REASON,
        "not_updated": dict(not_updated) if not_updated else None,
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _next_version(db, version_id: int, code: str) -> int:
    current = db.execute(
        select(func.max(IndustryReport.version)).where(
            IndustryReport.taxonomy_version_id == version_id,
            IndustryReport.industry_group_code == code,
        )
    ).scalar()
    return int(current or 0) + 1


def _candidate_meta(db, version_id: int, codes: list[str] | None) -> list[Any]:
    """Metadata of every edition that COULD be publishable: status and the
    agentic JSON clause in SQL, the section-coverage half in Python
    (``is_publishable``). Small columns only — never the payload."""
    q = select(
        IndustryReport.id, IndustryReport.industry_group_code, IndustryReport.version,
        IndustryReport.status, IndustryReport.generation, IndustryReport.degraded,
    ).where(
        IndustryReport.taxonomy_version_id == version_id,
        IndustryReport.status.in_(PUBLISHED_STATUSES),
        agentic_clause(),
    )
    if codes is not None:
        q = q.where(IndustryReport.industry_group_code.in_(codes))
    return list(db.execute(q).all())


def _latest_publishable_ids(db, version_id: int, codes: list[str] | None) -> dict[str, tuple[int, int]]:
    """``{code: (id, version)}`` of each group's newest publishable edition.
    One query per 200 codes on the ``IN`` list — never one per group."""
    best: dict[str, tuple[int, int]] = {}
    chunks: list[list[str] | None] = [None] if codes is None else [
        codes[i:i + 200] for i in range(0, len(codes), 200)
    ]
    for chunk in chunks:
        for meta in _candidate_meta(db, version_id, chunk):
            if not is_publishable(meta):
                continue
            code = str(meta.industry_group_code)
            if code not in best or int(meta.version) > best[code][1]:
                best[code] = (int(meta.id), int(meta.version))
    return best


def _latest_publishable_row(db, version_id: int, code: str) -> IndustryReport | None:
    """The group's newest publishable edition (C8), never by ``is_latest_good``:
    a legacy template still flagged latest-good is skipped and the analyst
    edition it superseded is found, before any backfill has run."""
    found = _latest_publishable_ids(db, version_id, [str(code)]).get(str(code))
    return db.get(IndustryReport, found[0]) if found else None


def save_report(
    *, code: str, period_key: str, as_of: datetime, payload: dict[str, Any],
    version: VersionInfo | str | int | None = None, stats_id: int | None = None,
    status: str | None = None, source_manifest: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None, freshness: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None, degraded: list[str] | None = None,
    errors: list[Any] | None = None, job_id: int | None = None,
    report_schema_version: int = REPORT_SCHEMA_VERSION,
) -> IndustryReport:
    """Persist a validated edition and, unless review is required, make it
    the latest-good one — all in one transaction.

    An edition whose content fails the display rule (a template, or an
    analyst edition below the minimum) is stored with status ``audit_only``
    and flips nothing: the previous analyst edition stays the one readers
    see. Passing an explicit ``status`` for such an edition raises
    ``ValueError`` before any write — no caller can publish a template by
    asking for it.

    Otherwise ``status`` defaults to ``succeeded``, or ``pending_review``
    when ``INDUSTRY_REPORTS_REQUIRE_REVIEW`` is on; an explicit
    ``succeeded`` overrides the flag (the caller has reviewed). The parent
    link always points at the latest publishable edition at the time of
    the save, whether or not this edition replaces it. Returns the
    detached row.
    """
    info = gics_registry.resolve_version(version)
    gics_registry.group(code, version=info)  # UnknownNode for a bad code, before any write
    reason = withheld_reason(generation, degraded, payload)
    if reason is not None:
        if status is not None:
            raise ValueError(
                f"template editions are stored for audit only and cannot be saved as {status!r} ({reason})"
            )
        effective = STATUS_AUDIT_ONLY
    else:
        effective = status or (STATUS_PENDING_REVIEW if settings.industry_reports_require_review else STATUS_SUCCEEDED)
        if effective not in (STATUS_SUCCEEDED, STATUS_PENDING_REVIEW):
            raise ValueError(f"unknown report status {effective!r}; allowed: succeeded, pending_review")
    now = _utcnow()
    from . import industry_lease

    with SessionLocal() as db:
        job = industry_lease.assert_current(db=db, lock=True)
        industry_lease.assert_identity(job, kind="group_report", taxonomy_version_id=info.id,
                                       period_key=period_key, code=code)
        if job is not None:
            if job_id != job.id:
                raise industry_lease.LeaseLost(f"Industry job {job.id} publication job ID differs")
            if job.report_id is not None:
                saved = db.get(IndustryReport, job.report_id)
                if saved is None or saved.job_id != job.id:
                    raise RuntimeError("Industry report publication receipt is invalid")
                db.expunge(saved)
                return saved
        previous = _latest_publishable_row(db, info.id, code)
        row = IndustryReport(
            taxonomy_version_id=info.id,
            industry_group_code=code,
            version=_next_version(db, info.id, code),
            parent_report_id=previous.id if previous is not None else None,
            period_key=period_key,
            as_of=as_of,
            status=effective,
            is_latest_good=(effective == STATUS_SUCCEEDED),
            report_schema_version=int(report_schema_version),
            payload=dict(payload or {}),
            stats_id=stats_id,
            source_manifest=list(source_manifest or []),
            coverage=dict(coverage or {}),
            freshness=dict(freshness or {}),
            generation=dict(generation or {}),
            degraded=list(degraded or []),
            errors=list(errors or []),
            job_id=job_id,
            generated_at=now,
        )
        if effective == STATUS_SUCCEEDED:
            # One statement over every currently-latest row for the group,
            # not just `previous`, so a duplicated flag can never survive a save.
            db.execute(
                update(IndustryReport)
                .where(
                    IndustryReport.taxonomy_version_id == info.id,
                    IndustryReport.industry_group_code == code,
                    IndustryReport.is_latest_good.is_(True),
                )
                .values(is_latest_good=False, status=STATUS_SUPERSEDED)
            )
        db.add(row)
        db.flush()
        if job is not None:
            job.report_id = row.id
        db.commit()
        db.refresh(row)
        db.expunge(row)
        if effective == STATUS_AUDIT_ONLY:
            log.warning("industry report %s v%d (%s) stored AUDIT ONLY, not displayed: %s",
                        code, row.version, period_key, reason)
        return row


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def latest_publishable(code: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """The newest publishable edition as a dict, or ``None`` when the group
    has never had one. Never raises for an unknown code — the analysts'
    context block asks for groups that may have no edition yet."""
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return None
    with SessionLocal() as db:
        row = _latest_publishable_row(db, info.id, str(code))
        return report_dict(row, latest_publishable_version=row.version) if row is not None else None


def latest_publishable_many(
    codes: list[str] | tuple[str, ...], *, version: VersionInfo | str | int | None = None,
) -> dict[str, dict[str, Any]]:
    """``{code: edition}`` for several groups in TWO queries, whatever the
    number of groups: the candidates' metadata (the rule's coverage half is
    applied in Python), then exactly the winning rows.

    The chat tool and the PM block ask about a whole portfolio's groups at
    once; a SELECT per group on a web request is what this avoids. Codes
    with no publishable edition are simply absent from the map — the caller
    reports that as "no edition yet", never as an empty report.
    """
    wanted = sorted({str(c) for c in codes if str(c)})
    if not wanted:
        return {}
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with SessionLocal() as db:
        best = _latest_publishable_ids(db, info.id, wanted)
        ids = [rid for rid, _ in best.values()]
        for i in range(0, len(ids), 200):
            for row in db.execute(select(IndustryReport).where(IndustryReport.id.in_(ids[i:i + 200]))).scalars():
                out[row.industry_group_code] = report_dict(row, latest_publishable_version=row.version)
    return out


# The names every caller used before the rule existed. Same functions: the
# "latest good" edition IS the latest publishable one now.
latest_good = latest_publishable
latest_good_many = latest_publishable_many


def _row_withheld_reason(row: Any) -> str:
    reason = withheld_reason(_get_field(row, "generation"), _get_field(row, "degraded"))
    return reason or "status audit_only"


def get(code: str, version_number: int | str, *, version: VersionInfo | str | int | None = None,
        include_withheld: bool = False) -> dict[str, Any] | None:
    """One edition by version number (``"latest"`` → latest publishable).

    ``None`` when no such edition exists. A withheld (audit-only) edition
    raises ``EditionWithheld`` unless ``include_withheld`` — "not on file"
    and "on file, kept for audit" are different answers."""
    if str(version_number).lower() == "latest":
        return latest_publishable(code, version=version)
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryReport).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
                IndustryReport.version == int(version_number),
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if is_withheld(row) and not include_withheld:
            raise EditionWithheld(str(code), row.version, _row_withheld_reason(row))
        best = _latest_publishable_ids(db, info.id, [str(code)]).get(str(code))
        return report_dict(row, latest_publishable_version=best[1] if best else None)


def history_page(code: str, *, limit: int = 26, version: VersionInfo | str | int | None = None,
                 include_withheld: bool = False) -> dict[str, Any]:
    """Editions newest first (metadata only), plus the counts that make a
    capped list honest: ``total`` editions the filter admits and how many
    were ``withheld`` (audit only, never listed publicly). Two queries: the
    group's metadata, then the page's rows with the payload deferred."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        metas = db.execute(
            select(IndustryReport.id, IndustryReport.version, IndustryReport.status,
                   IndustryReport.generation, IndustryReport.degraded).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
            ).order_by(IndustryReport.version.desc())
        ).all()
        withheld_ids = {int(m.id) for m in metas if is_withheld(m)}
        admitted = [m for m in metas if include_withheld or int(m.id) not in withheld_ids]
        latest = next((int(m.version) for m in metas if is_publishable(m)), None)
        page_ids = [int(m.id) for m in admitted[: max(1, int(limit))]]
        rows = db.execute(
            select(IndustryReport)
            .options(defer(IndustryReport.payload), defer(IndustryReport.source_manifest))
            .where(IndustryReport.id.in_(page_ids)).order_by(IndustryReport.version.desc())
        ).scalars().all() if page_ids else []
        items = [report_dict(r, include_payload=False, latest_publishable_version=latest) for r in rows]
    return {"items": items, "total": len(admitted), "withheld": len(withheld_ids)}


def history(code: str, *, limit: int = 26, version: VersionInfo | str | int | None = None,
            include_withheld: bool = False) -> list[dict[str, Any]]:
    """Editions newest first, metadata only. Withheld (audit-only) editions
    are excluded unless ``include_withheld`` (admin and tests)."""
    return history_page(code, limit=limit, version=version, include_withheld=include_withheld)["items"]


def withheld_count(code: str, *, version: VersionInfo | str | int | None = None) -> int:
    """How many of the group's editions are kept for audit only — what a
    404 for "no analyst edition yet" counts, so that answer is not read as
    "never attempted"."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        metas = db.execute(
            select(IndustryReport.status, IndustryReport.generation, IndustryReport.degraded).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
            )
        ).all()
    return sum(1 for m in metas if is_withheld(m))


def newest_publishable_below(code: str, below_version: int, *,
                             version: VersionInfo | str | int | None = None) -> int | None:
    """The newest publishable edition older than ``below_version`` — the
    default "changes since" basis when the recorded parent is withheld."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        versions = [int(m.version) for m in _candidate_meta(db, info.id, [str(code)])
                    if is_publishable(m) and int(m.version) < int(below_version)]
    return max(versions) if versions else None


def edition_meta_by_id(report_id: int, *, code: str,
                       version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """``{version, status, withheld}`` of one of ``code``'s editions by row
    id, or ``None`` when that row is not on file for this group."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        meta = db.execute(
            select(IndustryReport.version, IndustryReport.status, IndustryReport.generation,
                   IndustryReport.degraded).where(
                IndustryReport.id == int(report_id),
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
            )
        ).first()
    if meta is None:
        return None
    return {"version": int(meta.version), "status": meta.status, "withheld": is_withheld(meta)}


# The public error text for a validator rejection. The raw message quotes
# the rejected, never-validated model sentences ("unsupported causal claim
# 'The …'") — exactly the text decision 1 keeps off the page.
_REJECTED_PUBLIC = "the analyst draft did not pass validation ({n} problem{s})"


def _attempt_outcome(db, job: IndustryReportJob) -> str:
    if job.status in ("queued", "running"):
        return "in_progress"
    if job.status != "succeeded" or job.report_id is None:
        return "failed"
    row = db.execute(
        select(IndustryReport.status, IndustryReport.generation, IndustryReport.degraded)
        .where(IndustryReport.id == job.report_id)
    ).first()
    return "published" if row is not None and not is_withheld(row) else "withheld_template"


def last_attempt(code: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """The most recent report job for the group, any status — what the
    read API shows next to a stale edition. ``outcome`` names what the
    attempt produced: ``published``, ``withheld_template`` (an audit-only
    edition), ``failed`` or ``in_progress``. The raw error text is kept
    here for admin callers; the public routes serve ``public_attempt``."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        job = db.execute(
            select(IndustryReportJob).where(
                IndustryReportJob.taxonomy_version_id == info.id,
                IndustryReportJob.industry_group_code == str(code),
                IndustryReportJob.kind == "group_report",
            ).order_by(IndustryReportJob.id.desc())
        ).scalars().first()
        if job is None:
            return None
        at = job.finished_at or job.started_at or job.enqueued_at
        return {
            "job_id": job.id,
            "status": job.status,
            "outcome": _attempt_outcome(db, job),
            "at": at.isoformat() if at else None,
            "period_key": job.period_key,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "error_type": job.error_type or "",
            "error_message": (job.error_message or "")[:500],
            "report_id": job.report_id,
            "source": job.source,
        }


def public_attempt(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    """``last_attempt`` as a reader may see it: a ``ReportRejected``
    message becomes a count of problems (the raw one quotes rejected model
    prose), and ``report_id`` is dropped when it names an audit-only row."""
    if attempt is None:
        return None
    out = dict(attempt)
    if out.get("error_type") == "ReportRejected":
        head = str(out.get("error_message") or "").split(" ", 1)[0]
        n = int(head) if head.isdigit() else None
        out["error_message"] = (
            _REJECTED_PUBLIC.format(n=n, s="" if n == 1 else "s") if n is not None
            else "the analyst draft did not pass validation"
        )
    if out.get("outcome") == "withheld_template":
        out["report_id"] = None
    return out


def last_attempted_periods(codes: list[str] | tuple[str, ...], *,
                           version: VersionInfo | str | int | None = None) -> dict[str, str]:
    """``{code: newest period_key}`` over FINISHED group-report jobs — one
    grouped query for any number of groups. The picker pointer and the PM
    block compare it with an edition's own period to say "not updated"."""
    wanted = sorted({str(c) for c in codes if str(c)})
    if not wanted:
        return {}
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return {}
    out: dict[str, str] = {}
    with SessionLocal() as db:
        for i in range(0, len(wanted), 200):
            for code, period in db.execute(
                select(IndustryReportJob.industry_group_code, func.max(IndustryReportJob.period_key)).where(
                    IndustryReportJob.taxonomy_version_id == info.id,
                    IndustryReportJob.kind == "group_report",
                    IndustryReportJob.status.in_(("succeeded", "failed")),
                    IndustryReportJob.industry_group_code.in_(wanted[i:i + 200]),
                ).group_by(IndustryReportJob.industry_group_code)
            ).all():
                if code and period:
                    out[str(code)] = str(period)
    return out


def freshness(code: str, *, version: VersionInfo | str | int | None = None,
              report: dict[str, Any] | None = None) -> dict[str, Any]:
    """``stale`` / ``stale_reason`` / ``last_attempt`` / ``not_updated``
    for the latest publishable edition (or the ``report`` handed in).

    Stale when the as-of is older than ``INDUSTRY_REPORT_STALE_AFTER_DAYS``,
    when a refresh attempt failed after this edition was generated, or when
    a newer week FINISHED without a validated analyst edition (its only
    product was an audit-only template). ``not_updated`` names that newer
    week and its outcome — ``withheld_template``, ``failed``, or
    ``in_progress`` for a newer attempt still running, which is reported
    but does not by itself make the edition stale. ``stale_codes`` lists
    the machine-readable reasons (``as_of_age``, ``refresh_failed``,
    ``not_updated``) beside the sentence."""
    edition = report if report is not None else latest_publishable(code, version=version)
    attempt = last_attempt(code, version=version)
    if edition is None:
        return {"stale": None, "stale_reason": "no_report", "last_attempt": attempt,
                "not_updated": None, "stale_codes": []}
    now = _utcnow()
    reasons: list[str] = []
    stale_codes: list[str] = []
    as_of = datetime.fromisoformat(edition["as_of"]) if edition.get("as_of") else None
    if as_of is not None and now - as_of > timedelta(days=int(settings.industry_report_stale_after_days)):
        reasons.append(f"as_of older than {settings.industry_report_stale_after_days} days")
        stale_codes.append("as_of_age")
    generated_at = datetime.fromisoformat(edition["generated_at"]) if edition.get("generated_at") else None
    if attempt and attempt["status"] == "failed":
        attempted_at = datetime.fromisoformat(attempt["at"]) if attempt.get("at") else None
        if generated_at is None or attempted_at is None or attempted_at >= generated_at:
            reasons.append(f"latest refresh attempt failed ({attempt.get('error_type') or 'error'})")
            stale_codes.append("refresh_failed")
    not_updated: dict[str, Any] | None = None
    # ISO-week keys ("2026-W39") order lexicographically, year turns included.
    if attempt is not None and str(attempt.get("period_key") or "") > str(edition.get("period_key") or ""):
        outcome = attempt.get("outcome")
        if outcome in ("withheld_template", "failed", "in_progress"):
            not_updated = {"period_key": attempt["period_key"], "outcome": outcome}
        if outcome == "withheld_template":
            reasons.append(
                f"not updated this week: the {attempt['period_key']} refresh produced no validated "
                "analyst edition"
            )
            stale_codes.append("not_updated")
        elif outcome == "failed":
            if "refresh_failed" not in stale_codes:
                reasons.append(f"latest refresh attempt failed ({attempt.get('error_type') or 'error'})")
                stale_codes.append("refresh_failed")
            stale_codes.append("not_updated")
    return {
        "stale": bool(reasons),
        "stale_reason": "; ".join(reasons) if reasons else None,
        "last_attempt": attempt,
        "not_updated": not_updated,
        "stale_codes": stale_codes,
    }


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _stats_dict_by_id(db, stats_id: int | None) -> dict[str, Any] | None:
    if stats_id is None:
        return None
    row = db.get(IndustryStatSnapshot, int(stats_id))
    if row is None:
        return None
    from .industry_analytics import stats_dict
    return stats_dict(row)


def _dig(doc: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    cur: Any = doc
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _section_text(payload: dict[str, Any], section: str, *keys: str) -> Any:
    interp = ((payload.get("sections") or {}).get(section) or {}).get("interpretation")
    if not isinstance(interp, dict):
        return interp if isinstance(interp, str) else None
    for key in keys:
        if interp.get(key) is not None:
            return interp[key]
    return None


def _analyst_view(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Each edition's outlook view and the newer edition's what-changed
    line — except where a template wrote them. A hidden side is ``None``
    with its reason in ``reasons``, never the template's prose (that text
    is exactly what decision 1 keeps off the page)."""
    hidden_a, hidden_b = set(hidden_sections(a)), set(hidden_sections(b))
    reason = f"template-filled in this edition; {HIDDEN_SECTION_REASON[0].lower()}{HIDDEN_SECTION_REASON[1:]}"
    reasons: dict[str, str | None] = {
        "from": reason if "outlook" in hidden_a else None,
        "to": reason if "outlook" in hidden_b else None,
        "what_changed": reason if "what_changed" in hidden_b else None,
    }
    return {
        "from": None if reasons["from"] else _section_text(a.get("payload") or {}, "outlook", "analyst_view", "text"),
        "to": None if reasons["to"] else _section_text(b.get("payload") or {}, "outlook", "analyst_view", "text"),
        "what_changed": None if reasons["what_changed"] else _section_text(b.get("payload") or {}, "what_changed", "text"),
        "reasons": reasons,
    }


def diff(code: str, from_version: int | str, to_version: int | str = "latest",
         *, version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """Facts delta, constituent changes, leaders/laggards and the analyst
    views of two editions (any two — not only adjacent ones). Pure
    arithmetic over stored rows; a fact missing on either side is
    reported as ``null`` with the side named, never as zero. A withheld
    edition on either side raises ``EditionWithheld``."""
    info = gics_registry.resolve_version(version)
    a = get(code, from_version, version=info)
    b = get(code, to_version, version=info)
    if a is None or b is None:
        missing = "from" if a is None else "to"
        raise ReportNotFound(f"industry report {code} version {from_version if a is None else to_version!r} not found ({missing})")
    with SessionLocal() as db:
        sa = _stats_dict_by_id(db, a.get("stats_id"))
        sb = _stats_dict_by_id(db, b.get("stats_id"))

    facts_delta: dict[str, Any] = {}
    for label, path in _FACT_PATHS:
        va = _dig(sa, path)
        vb = _dig(sb, path)
        entry: dict[str, Any] = {"from": va, "to": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            entry["delta"] = round(float(vb) - float(va), 6)
        else:
            entry["delta"] = None
            entry["reason"] = (
                "no stats on either edition" if sa is None and sb is None else
                # Both-missing first: the old chain fell through to
                # "missing in from-edition" and blamed one side for a fact
                # that neither edition carried.
                "missing in both editions" if va is None and vb is None else
                "missing in from-edition" if va is None else
                "missing in to-edition"
            )
        facts_delta[label] = entry

    members_a = set((sa or {}).get("per_ticker") or {})
    members_b = set((sb or {}).get("per_ticker") or {})
    constituents = {
        "added": sorted(members_b - members_a),
        "removed": sorted(members_a - members_b),
        "n_from": len(members_a) if sa else None,
        "n_to": len(members_b) if sb else None,
    }
    leaders = {
        "from": {"leaders": ((sa or {}).get("payload") or {}).get("leaders"),
                 "laggards": ((sa or {}).get("payload") or {}).get("laggards")},
        "to": {"leaders": ((sb or {}).get("payload") or {}).get("leaders"),
               "laggards": ((sb or {}).get("payload") or {}).get("laggards")},
    }
    return {
        "code": str(code),
        "taxonomy_version": info.version_key,
        "from": {"version": a["version"], "period_key": a["period_key"], "as_of": a["as_of"], "stats_id": a.get("stats_id")},
        "to": {"version": b["version"], "period_key": b["period_key"], "as_of": b["as_of"], "stats_id": b.get("stats_id")},
        "adjacent": (b.get("parent_report_id") == a.get("id")),
        "facts_delta": facts_delta,
        "constituents": constituents,
        "leaders_laggards": leaders,
        "analyst_view": _analyst_view(a, b),
        "disclaimer": DISCLAIMER,
    }


def latest_versions(*, version: VersionInfo | str | int | None = None) -> dict[str, int]:
    """``{code: version}`` of every group's latest PUBLISHABLE edition —
    what the cross-industry snapshot records as the report versions a
    reader could see. One metadata query."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        return {code: v for code, (_, v) in _latest_publishable_ids(db, info.id, None).items()}


# ---------------------------------------------------------------------------
# Legacy reclassification (one-shot; rows written before the display rule)
# ---------------------------------------------------------------------------
#
# Reads are already correct without this (none of them reads the flags). It
# normalises `status` / `is_latest_good` on legacy rows so admin views and
# the flags agree with the rule. Never deletes; every change is a
# compare-and-set; the manifest of before/after states is the audit trail
# and what `revert_reclassification` replays backwards.


def _target_state(meta: Any, newest_publishable_id: int | None) -> tuple[str, bool] | None:
    """The (status, is_latest_good) the rule gives a row, or ``None`` for a
    row the reclassification never touches (``pending_review``: a human's
    queue, not a legacy artefact)."""
    if meta.status == STATUS_PENDING_REVIEW:
        return None
    if meta.status == STATUS_AUDIT_ONLY or not content_publishable(meta.generation, meta.degraded):
        return STATUS_AUDIT_ONLY, False
    if int(meta.id) == newest_publishable_id:
        return STATUS_SUCCEEDED, True
    return STATUS_SUPERSEDED, False


def _reclassification_plan(db, version_id: int, *, lock: bool) -> list[dict[str, Any]]:
    q = select(
        IndustryReport.id, IndustryReport.industry_group_code, IndustryReport.version,
        IndustryReport.status, IndustryReport.is_latest_good,
        IndustryReport.generation, IndustryReport.degraded,
    ).where(IndustryReport.taxonomy_version_id == version_id).order_by(
        IndustryReport.industry_group_code, IndustryReport.version,
    )
    if lock:
        # Row locks on Postgres (a no-op on sqlite): a concurrent
        # `save_report` flip waits for this transaction instead of
        # interleaving with it.
        q = q.with_for_update()
    metas = list(db.execute(q).all())
    newest: dict[str, tuple[int, int]] = {}
    for m in metas:
        if is_publishable(m):
            code = str(m.industry_group_code)
            if code not in newest or int(m.version) > newest[code][1]:
                newest[code] = (int(m.id), int(m.version))
    plan: list[dict[str, Any]] = []
    for m in metas:
        target = _target_state(m, newest.get(str(m.industry_group_code), (None, None))[0])
        before = (str(m.status), bool(m.is_latest_good))
        if target is None or before == target:
            continue
        plan.append({
            "row_id": int(m.id), "code": str(m.industry_group_code), "version": int(m.version),
            "before": {"status": before[0], "is_latest_good": before[1]},
            "after": {"status": target[0], "is_latest_good": target[1]},
            "edition_kind": edition_kind(m.generation),
        })
    return plan


def _active_group_jobs(db, version_id: int, *, lock: bool) -> list[int]:
    q = select(IndustryReportJob.id).where(
        IndustryReportJob.taxonomy_version_id == version_id,
        IndustryReportJob.kind == "group_report",
        IndustryReportJob.status.in_(("queued", "running")),
    )
    if lock:
        q = q.with_for_update()
    return [int(i) for i in db.execute(q).scalars().all()]


def _plan_counts(plan: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"changed": len(plan)}
    for entry in plan:
        key = f"to_{entry['after']['status']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def reclassification_ledger(*, version: VersionInfo | str | int | None = None,
                            run_id: str = RECLASSIFY_RUN_ID) -> dict[str, Any] | None:
    """The newest ledger row for ``run_id`` (the apply by default), as a
    dict with its manifest, or ``None`` when it has never run."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryReportJob).where(
                IndustryReportJob.taxonomy_version_id == info.id,
                IndustryReportJob.kind == RECLASSIFY_KIND,
                IndustryReportJob.run_id == run_id,
                IndustryReportJob.status == "succeeded",
            ).order_by(IndustryReportJob.id.desc())
        ).scalars().first()
        if row is None:
            return None
        entry = (list(row.progress or []) or [{}])[-1]
        return {"ledger_id": row.id, "source": row.source,
                "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                "manifest": list(entry.get("manifest") or []), "counts": dict(entry.get("counts") or {})}


def _write_ledger(db, version_id: int, *, run_id: str, source: str, step: str,
                  manifest: list[dict[str, Any]], counts: dict[str, int]) -> int:
    now = _utcnow()
    row = IndustryReportJob(
        kind=RECLASSIFY_KIND, taxonomy_version_id=version_id, industry_group_code=None,
        period_key=RECLASSIFY_PERIOD_KEY, run_id=run_id, status="succeeded",
        attempts=1, max_attempts=1, priority=0, enqueued_at=now, started_at=now,
        finished_at=now, source=str(source)[:24],
        progress=[{"step": step, "at": now.isoformat(), "counts": counts, "manifest": manifest}],
    )
    db.add(row)
    db.flush()
    return int(row.id)


def reclassify_legacy_editions(*, apply: bool = False, source: str = "cli",
                               version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """Plan (and with ``apply``, perform) the flag normalisation.

    Template rows in ``succeeded``/``superseded`` → ``audit_only``; each
    group's newest publishable row → ``succeeded`` + latest-good; other
    publishable rows → ``superseded``; ``pending_review`` untouched. Refuses
    (``ReclassifyRefused``) while any group-report job is queued or running,
    checked under ``FOR UPDATE`` inside the same transaction as the writes.
    Every change is a compare-and-set against the planned before-state, and
    the plan is recomputed after the writes and must be empty; either check
    failing rolls the whole transaction back (``ReclassifyAborted``). An
    apply writes a ledger row carrying the manifest in the same commit.
    Idempotent: a second apply changes nothing."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        active = _active_group_jobs(db, info.id, lock=apply)
        if active:
            db.rollback()
            raise ReclassifyRefused(
                f"{len(active)} group_report job(s) queued or running for {info.version_key} "
                f"(ids {active[:20]}); a save_report flip must not interleave with the reclassification"
            )
        plan = _reclassification_plan(db, info.id, lock=apply)
        counts = _plan_counts(plan)
        result: dict[str, Any] = {
            "taxonomy_version": info.version_key, "applied": False, "counts": counts, "manifest": plan,
        }
        if not apply:
            db.rollback()
            return result
        mismatched: list[int] = []
        for entry in plan:
            changed = db.execute(
                update(IndustryReport).where(
                    IndustryReport.id == entry["row_id"],
                    IndustryReport.status == entry["before"]["status"],
                    IndustryReport.is_latest_good.is_(entry["before"]["is_latest_good"]),
                ).values(status=entry["after"]["status"], is_latest_good=entry["after"]["is_latest_good"])
                .execution_options(synchronize_session=False)
            ).rowcount
            if changed != 1:
                mismatched.append(entry["row_id"])
        leftover = _reclassification_plan(db, info.id, lock=False) if not mismatched else []
        if mismatched or leftover:
            db.rollback()
            raise ReclassifyAborted(
                f"reclassification rolled back: compare-and-set missed rows {mismatched[:20]}"
                if mismatched else
                f"reclassification rolled back: {len(leftover)} row(s) still disagree with the rule after the writes"
            )
        result["ledger_id"] = _write_ledger(db, info.id, run_id=RECLASSIFY_RUN_ID, source=source,
                                            step="reclassified", manifest=plan, counts=counts)
        db.commit()
        result["applied"] = True
    log.warning("industry reclassification applied (%s, source=%s): %s", info.version_key, source, counts)
    return result


def run_legacy_reclassification_once(*, source: str,
                                     version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """The deployed worker's entry point: apply once, recorded by the ledger.

    ``already_done`` when the ledger row exists (from this runner or from
    the owner's shell). ``ReclassifyRefused`` propagates so the caller can
    retry later; ``ReclassifyAborted`` propagates so the caller can stop."""
    info = gics_registry.resolve_version(version)
    done = reclassification_ledger(version=info)
    if done is not None:
        return {"status": "already_done", "ledger_id": done["ledger_id"], "counts": done["counts"]}
    out = reclassify_legacy_editions(apply=True, source=source, version=info)
    return {"status": "applied", "ledger_id": out.get("ledger_id"), "counts": out["counts"],
            "manifest": out["manifest"]}


def revert_reclassification(manifest: list[dict[str, Any]], *, apply: bool = False, source: str = "cli",
                            version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """Replay a manifest backwards: each row goes from its ``after`` state
    back to its ``before`` state, compare-and-set, all or nothing. A row
    that has moved on since (a newer save flipped it) aborts the revert
    rather than being overwritten. Same active-job refusal as the apply."""
    info = gics_registry.resolve_version(version)
    entries = [e for e in manifest if isinstance(e, dict)]
    for e in entries:
        if not {"row_id", "before", "after"} <= set(e):
            raise ValueError(f"manifest entry is missing row_id/before/after: {e!r}")
    with SessionLocal() as db:
        active = _active_group_jobs(db, info.id, lock=apply)
        if active:
            db.rollback()
            raise ReclassifyRefused(
                f"{len(active)} group_report job(s) queued or running for {info.version_key} (ids {active[:20]})"
            )
        result: dict[str, Any] = {"taxonomy_version": info.version_key, "applied": False,
                                  "counts": {"reverted": len(entries)}, "manifest": entries}
        if not apply:
            db.rollback()
            return result
        mismatched: list[int] = []
        for e in entries:
            changed = db.execute(
                update(IndustryReport).where(
                    IndustryReport.id == int(e["row_id"]),
                    IndustryReport.taxonomy_version_id == info.id,
                    IndustryReport.status == e["after"]["status"],
                    IndustryReport.is_latest_good.is_(bool(e["after"]["is_latest_good"])),
                ).values(status=e["before"]["status"], is_latest_good=bool(e["before"]["is_latest_good"]))
                .execution_options(synchronize_session=False)
            ).rowcount
            if changed != 1:
                mismatched.append(int(e["row_id"]))
        if mismatched:
            db.rollback()
            raise ReclassifyAborted(f"revert rolled back: rows no longer in their manifest state {mismatched[:20]}")
        result["ledger_id"] = _write_ledger(db, info.id, run_id=RECLASSIFY_REVERT_RUN_ID, source=source,
                                            step="reverted", manifest=entries, counts=result["counts"])
        db.commit()
        result["applied"] = True
    log.warning("industry reclassification reverted (%s, source=%s): %d rows", info.version_key, source, len(entries))
    return result
