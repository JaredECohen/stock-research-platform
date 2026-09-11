"""FEAT-003 — the versioned store for weekly Industry Analysis editions.

An edition is an ``industry_reports`` row. The rules the store enforces:

* **Versions are monotonic per (taxonomy version, group)** and every
  edition links to the edition it replaced (``parent_report_id``), so
  "what changed since last week" always has a well-defined "last week".
* **Exactly one latest-good edition per group.** ``save_report`` assigns
  the version, links the parent and flips ``is_latest_good`` inside one
  transaction; the previous latest becomes ``superseded``. Under
  ``INDUSTRY_REPORTS_REQUIRE_REVIEW`` (owner decision 4, default off) a
  new edition lands as ``pending_review`` and nothing flips — the
  publish step does not exist yet and this store does not pretend it
  does.
* **Failures never write here.** A failed refresh leaves the last good
  edition in place; the read API composes ``stale`` and ``last_attempt``
  from the jobs table (``last_attempt``/``freshness`` below), so a reader
  sees a dated edition with the failure named — never a blank week and
  never a silently old page.
* **Diffs are arithmetic over stored facts.** ``diff`` compares the two
  editions' stats rows (returns, breadth, dispersion, valuation medians,
  sample) and constituent sets, and quotes the newer edition's own
  ``what_changed`` text; it never re-generates anything.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update

from ..config import settings
from ..database import SessionLocal
from ..models import IndustryReport, IndustryReportJob, IndustryStatSnapshot
from . import gics_registry
from .gics_registry import ATTRIBUTION, MAPPING_CAVEAT, VersionInfo

log = logging.getLogger(__name__)

STATUS_SUCCEEDED = "succeeded"
STATUS_PENDING_REVIEW = "pending_review"
STATUS_SUPERSEDED = "superseded"
REPORT_STATUSES: tuple[str, ...] = (STATUS_SUCCEEDED, STATUS_PENDING_REVIEW, STATUS_SUPERSEDED)
REPORT_SCHEMA_VERSION = 1

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


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def report_dict(row: IndustryReport, *, include_payload: bool = True) -> dict[str, Any]:
    """Detached projection of an edition. ``include_payload=False`` is the
    history shape (metadata only)."""
    out: dict[str, Any] = {
        "id": row.id,
        "taxonomy_version_id": row.taxonomy_version_id,
        "code": row.industry_group_code,
        "version": row.version,
        "parent_report_id": row.parent_report_id,
        "period_key": row.period_key,
        "as_of": row.as_of.isoformat() if row.as_of else None,
        "status": row.status,
        "is_latest_good": bool(row.is_latest_good),
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


def _latest_good_row(db, version_id: int, code: str) -> IndustryReport | None:
    return db.execute(
        select(IndustryReport).where(
            IndustryReport.taxonomy_version_id == version_id,
            IndustryReport.industry_group_code == code,
            IndustryReport.is_latest_good.is_(True),
        ).order_by(IndustryReport.version.desc())
    ).scalars().first()


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

    ``status`` defaults to ``succeeded``, or ``pending_review`` when
    ``INDUSTRY_REPORTS_REQUIRE_REVIEW`` is on; an explicit ``succeeded``
    overrides the flag (the caller has reviewed). The parent link always
    points at the latest-good edition at the time of the save, whether or
    not this edition flips it. Returns the detached row.
    """
    info = gics_registry.resolve_version(version)
    gics_registry.group(code, version=info)  # UnknownNode for a bad code, before any write
    effective = status or (STATUS_PENDING_REVIEW if settings.industry_reports_require_review else STATUS_SUCCEEDED)
    if effective not in (STATUS_SUCCEEDED, STATUS_PENDING_REVIEW):
        raise ValueError(f"unknown report status {effective!r}; allowed: succeeded, pending_review")
    now = _utcnow()
    with SessionLocal() as db:
        previous = _latest_good_row(db, info.id, code)
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
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def latest_good(code: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """The latest-good edition as a dict, or ``None`` when the group has
    never published. Never raises for an unknown code — the analysts'
    context block asks for groups that may have no edition yet."""
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return None
    with SessionLocal() as db:
        row = _latest_good_row(db, info.id, str(code))
        return report_dict(row) if row is not None else None


def latest_good_many(
    codes: list[str] | tuple[str, ...], *, version: VersionInfo | str | int | None = None,
) -> dict[str, dict[str, Any]]:
    """``{code: edition}`` for several groups in ONE query.

    The chat tool and the PM block ask about a whole portfolio's groups at
    once; calling ``latest_good`` per code would put a SELECT per group on
    a web request. Codes with no edition are simply absent from the map —
    the caller reports that as "no edition yet", never as an empty report.
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
        for i in range(0, len(wanted), 200):
            rows = db.execute(
                select(IndustryReport).where(
                    IndustryReport.taxonomy_version_id == info.id,
                    IndustryReport.industry_group_code.in_(wanted[i:i + 200]),
                    IndustryReport.is_latest_good.is_(True),
                ).order_by(IndustryReport.version)
            ).scalars().all()
            for row in rows:
                out[row.industry_group_code] = report_dict(row)
    return out


def get(code: str, version_number: int | str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """One edition by version number (``"latest"`` → latest-good)."""
    if str(version_number).lower() == "latest":
        return latest_good(code, version=version)
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(IndustryReport).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
                IndustryReport.version == int(version_number),
            )
        ).scalar_one_or_none()
        return report_dict(row) if row is not None else None


def history(code: str, *, limit: int = 26, version: VersionInfo | str | int | None = None) -> list[dict[str, Any]]:
    """Editions newest first, metadata only."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        rows = db.execute(
            select(IndustryReport).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == str(code),
            ).order_by(IndustryReport.version.desc()).limit(max(1, int(limit)))
        ).scalars().all()
        return [report_dict(r, include_payload=False) for r in rows]


def last_attempt(code: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """The most recent report job for the group, any status — what the
    read API shows next to a stale edition."""
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
            "at": at.isoformat() if at else None,
            "period_key": job.period_key,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "error_type": job.error_type or "",
            "error_message": (job.error_message or "")[:500],
            "report_id": job.report_id,
            "source": job.source,
        }


def freshness(code: str, *, version: VersionInfo | str | int | None = None,
              report: dict[str, Any] | None = None) -> dict[str, Any]:
    """``stale`` / ``stale_reason`` / ``last_attempt`` for the latest-good
    edition (or the ``report`` handed in): stale when the as-of is older
    than ``INDUSTRY_REPORT_STALE_AFTER_DAYS`` or when a refresh attempt
    failed after this edition was generated."""
    edition = report if report is not None else latest_good(code, version=version)
    attempt = last_attempt(code, version=version)
    if edition is None:
        return {"stale": None, "stale_reason": "no_report", "last_attempt": attempt}
    now = _utcnow()
    reasons: list[str] = []
    as_of = datetime.fromisoformat(edition["as_of"]) if edition.get("as_of") else None
    if as_of is not None and now - as_of > timedelta(days=int(settings.industry_report_stale_after_days)):
        reasons.append(f"as_of older than {settings.industry_report_stale_after_days} days")
    generated_at = datetime.fromisoformat(edition["generated_at"]) if edition.get("generated_at") else None
    if attempt and attempt["status"] == "failed":
        attempted_at = datetime.fromisoformat(attempt["at"]) if attempt.get("at") else None
        if generated_at is None or attempted_at is None or attempted_at >= generated_at:
            reasons.append(f"latest refresh attempt failed ({attempt.get('error_type') or 'error'})")
    return {
        "stale": bool(reasons),
        "stale_reason": "; ".join(reasons) if reasons else None,
        "last_attempt": attempt,
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


def diff(code: str, from_version: int | str, to_version: int | str = "latest",
         *, version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """Facts delta, constituent changes, leaders/laggards and the analyst
    views of two editions (any two — not only adjacent ones). Pure
    arithmetic over stored rows; a fact missing on either side is
    reported as ``null`` with the side named, never as zero."""
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
    analyst_view = {
        "from": _section_text(a.get("payload") or {}, "outlook", "analyst_view", "text"),
        "to": _section_text(b.get("payload") or {}, "outlook", "analyst_view", "text"),
        "what_changed": _section_text(b.get("payload") or {}, "what_changed", "text"),
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
        "analyst_view": analyst_view,
        "disclaimer": DISCLAIMER,
    }


def latest_versions(*, version: VersionInfo | str | int | None = None) -> dict[str, int]:
    """``{code: version}`` of every latest-good edition — what the
    cross-industry snapshot records as the report versions it saw."""
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        rows = db.execute(
            select(IndustryReport.industry_group_code, IndustryReport.version).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.is_latest_good.is_(True),
            )
        ).all()
        return {str(code): int(v) for code, v in rows}
