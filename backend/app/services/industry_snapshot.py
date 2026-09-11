"""FEAT-003 — the Portfolio Manager's cross-industry snapshot.

One deterministic row per period (``industry_snapshots``), computed from
that period's ``industry_stats`` rows — never from prices, never from an
LLM — plus three cheap context reads: the hourly macro broadcast, the
forward catalyst calendar for the constituents, and the checked-in
dependency graph (``atlas_dependencies.json``: the Atlas's analyst causal
edges and the universe map's themed relationships, each labelled with
its source). The PM reads it as a ≤ 2,000-character block; the chat
tool and the portfolio builder read the same row.

What the snapshot refuses to do:

* invent a number for a group without stats — such groups are listed
  under ``missing_groups`` with a reason, and the block says how many;
* present a dependency edge as a measured correlation — every spillover
  line carries the graph's own status text ("analyst causal hypothesis")
  and its source label;
* re-render anything expensive on a page view — ``latest_snapshot`` is
  one indexed row read, ``render_pm_block`` is string formatting.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import CatalystEvent, CrossIndustrySnapshot
from . import gics_registry, industry_analytics, industry_classification
from .gics_registry import VersionInfo

log = logging.getLogger(__name__)

ATLAS_PATH = Path(__file__).resolve().parent.parent / "data" / "industry_knowledge" / "atlas_dependencies.json"
SNAPSHOT_SCHEMA_VERSION = 1
PM_BLOCK_MAX_CHARS = 2000
EVENT_WINDOW_DAYS = 14
MAX_MAJOR_EVENTS = 25
MAX_SPILLOVERS = 40
# A linked group's 1M equal-weight move at or beyond this is worth a
# spillover line; below it the edge is listed as dormant.
SPILLOVER_MOVE = 0.05
_CHUNK = 200

_MATERIALITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Dependency graph (checked-in JSON; generated at dev time)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def dependency_graph() -> dict[str, Any]:
    """The Atlas edges and map relationships, each resolved to industry
    group codes by the build script. An edge with no resolved group codes
    is unlabelled and is dropped from ``links`` — it can never name a
    spillover. Missing file → empty graph (the snapshot still computes;
    ``spillovers`` says the graph was unavailable)."""
    try:
        with ATLAS_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        log.warning("atlas_dependencies.json missing at %s — snapshot spillovers unavailable", ATLAS_PATH)
        return {"edges": [], "relationships": [], "links": [], "available": False,
                "atlas_snapshot_date": None, "unlabelled_edges": 0}
    links: list[dict[str, Any]] = []
    unlabelled = 0
    for edge in raw.get("edges") or []:
        codes = sorted({str(c) for c in (edge.get("industry_group_codes") or []) if c})
        if not codes:
            unlabelled += 1
            continue
        links.append({
            "kind": "edge",
            "id": str(edge.get("edge_id") or ""),
            "label": f"{edge.get('origin', '')} → {edge.get('destination', '')}".strip(" →"),
            "mechanism": edge.get("transmission") or "",
            "invalidation": edge.get("invalidation") or "",
            "status": edge.get("status") or "",
            "source": edge.get("source") or raw.get("edge_source") or "atlas_dependencies_sheet",
            "codes": codes,
        })
    for i, rel in enumerate(raw.get("relationships") or []):
        codes = sorted({str(c) for c in (rel.get("industry_group_codes") or []) if c})
        if not codes:
            unlabelled += 1
            continue
        links.append({
            "kind": "relationship",
            "id": f"R{i + 1:02d}",
            "label": rel.get("theme") or "",
            "mechanism": rel.get("mechanism") or "",
            "invalidation": rel.get("failure_of_inference") or "",
            "status": "themed cross-industry relationship (analyst framework, not an estimated correlation)",
            "source": rel.get("source") or raw.get("relationship_source") or "universe_map_cross_industry_relationships",
            "codes": codes,
        })
    return {
        "edges": list(raw.get("edges") or []),
        "relationships": list(raw.get("relationships") or []),
        "links": links,
        "available": True,
        "atlas_snapshot_date": raw.get("atlas_snapshot_date"),
        "knowledge_map_as_of": raw.get("knowledge_map_as_of"),
        "unlabelled_edges": unlabelled,
    }


def links_for(code: str) -> list[dict[str, Any]]:
    """Every labelled link touching ``code``."""
    return [link for link in dependency_graph()["links"] if code in link["codes"]]


def linked_group_codes(code: str) -> dict[str, list[dict[str, Any]]]:
    """``{other_code: [links]}`` — the groups a dependency link joins to
    ``code``, with the links that do it."""
    out: dict[str, list[dict[str, Any]]] = {}
    for link in links_for(code):
        for other in link["codes"]:
            if other != code:
                out.setdefault(other, []).append(link)
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _db_macro() -> dict[str, Any] | None:
    from ..cache import cache_get
    snap = cache_get("macro:global", "macro_broadcast")
    if snap is None or not isinstance(snap.payload, dict):
        return None
    payload = snap.payload
    return {
        "regime": payload.get("regime"),
        "favored_sectors": list(payload.get("favored_sectors") or []),
        "pressured_sectors": list(payload.get("pressured_sectors") or []),
        "note": payload.get("note") or "",
        "as_of": snap.generated_at.isoformat() if getattr(snap, "generated_at", None) else None,
    }


def _db_events(tickers: list[str], cutoff: date, days: int) -> list[dict[str, Any]]:
    """Forward catalysts within ``days`` of the cutoff — chunked ``IN``."""
    out: list[dict[str, Any]] = []
    if not tickers:
        return out
    end = cutoff + timedelta(days=days)
    symbols = sorted({t.upper() for t in tickers})
    with SessionLocal() as db:
        for i in range(0, len(symbols), _CHUNK):
            chunk = symbols[i:i + _CHUNK]
            rows = db.execute(
                select(CatalystEvent).where(
                    CatalystEvent.ticker.in_(chunk),
                    CatalystEvent.event_date >= cutoff,
                    CatalystEvent.event_date <= end,
                ).order_by(CatalystEvent.event_date, CatalystEvent.ticker)
            ).scalars().all()
            for r in rows:
                out.append({
                    "ticker": r.ticker, "event_type": r.event_type,
                    "event_date": r.event_date.isoformat() if r.event_date else None,
                    "title": r.title or "", "materiality": r.materiality or "medium",
                    "source": r.source or "",
                })
    return out


def _db_report_versions(version: VersionInfo) -> dict[str, int]:
    from .industry_report_store import latest_versions
    return latest_versions(version=version)


@dataclass
class SnapshotLoaders:
    stats: Callable[[str, VersionInfo], dict[str, dict[str, Any]]] = (
        lambda period_key, version: industry_analytics.period_stats(period_key, version=version)
    )
    macro: Callable[[], dict[str, Any] | None] = _db_macro
    events: Callable[[list[str], date, int], list[dict[str, Any]]] = _db_events
    report_versions: Callable[[VersionInfo], dict[str, int]] = _db_report_versions


# ---------------------------------------------------------------------------
# Regime rules — deterministic, over the group's own stats
# ---------------------------------------------------------------------------


def regime_label(stats: dict[str, Any] | None) -> str:
    """A one-phrase read of the group's month versus the universe. Rule
    thresholds are stated in the payload's ``regime_rules``; every path
    that lacks the inputs says so rather than defaulting to neutral."""
    if not stats:
        return "no_stats"
    payload = stats.get("payload") or {}
    if payload.get("status") == industry_analytics.REASON_INSUFFICIENT:
        return "insufficient_sample"
    ret = ((payload.get("returns") or {}).get("1m") or {}).get("equal_weight")
    rel = (((payload.get("benchmark_relative") or {}).get("universe_ew") or {}).get("1m") or {}).get("value")
    breadth = ((payload.get("breadth") or {}).get("1m") or {}).get("pct_positive")
    if ret is None:
        return "no_return_data"
    if rel is None:
        return "universe_benchmark_unavailable"
    if rel > 0.02:
        if breadth is None:
            return "leading (breadth unknown)"
        return "broad leadership" if breadth >= 0.6 else "narrow leadership"
    if rel < -0.02:
        if breadth is None:
            return "lagging (breadth unknown)"
        return "broad weakness" if breadth <= 0.4 else "lagging (mixed breadth)"
    return "in line with universe"


REGIME_RULES = {
    "leading": "1M equal-weight return more than +2pp above the universe equal-weight cohort",
    "lagging": "1M equal-weight return more than −2pp below the universe equal-weight cohort",
    "broad": "share of constituents positive over 1M ≥ 60% (leadership) / ≤ 40% (weakness)",
    "in_line": "within ±2pp of the universe",
}


def _pick(stats: dict[str, Any], *path: str) -> Any:
    cur: Any = stats
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _group_row(node: gics_registry.NodeInfo, stats: dict[str, Any] | None, events_count: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": node.code,
        "name": node.name,
        "sector_code": node.sector_code,
        "status": None,
        "n": None,
        "ret_1w_ew": None, "ret_1m_ew": None, "ret_ytd_ew": None,
        "ret_1m_mcw": None,
        "rel_1m_vs_universe": None, "rel_1m_vs_sector": None, "rel_1m_vs_market_factor": None,
        "breadth_1m": None, "dispersion_1m": None,
        "val_median_ev_ebitda": None, "val_median_pe_ttm": None,
        "fund_momentum": None,
        "regime_label": regime_label(stats),
        "events_14d": events_count,
        "stats_id": None,
        "reasons": {},
    }
    if not stats:
        row["status"] = "no_stats"
        return row
    payload = stats.get("payload") or {}
    row["status"] = payload.get("status")
    row["n"] = _pick(stats, "sample", "n_with_prices")
    row["stats_id"] = stats.get("id")
    reasons: dict[str, str] = {}

    def take(label: str, *path: str, reason_key: str = "reason") -> Any:
        entry = _pick(payload, *path[:-1])
        value = _pick(payload, *path)
        if value is None and isinstance(entry, dict) and entry.get(reason_key):
            reasons[label] = str(entry[reason_key])
        return value

    row["ret_1w_ew"] = take("ret_1w_ew", "returns", "1w", "equal_weight")
    row["ret_1m_ew"] = take("ret_1m_ew", "returns", "1m", "equal_weight")
    row["ret_ytd_ew"] = take("ret_ytd_ew", "returns", "ytd", "equal_weight")
    row["ret_1m_mcw"] = take("ret_1m_mcw", "returns", "1m", "market_cap_weight", reason_key="market_cap_weight_reason")
    row["rel_1m_vs_universe"] = take("rel_1m_vs_universe", "benchmark_relative", "universe_ew", "1m", "value")
    row["rel_1m_vs_sector"] = take("rel_1m_vs_sector", "benchmark_relative", "sector_ew", "1m", "value")
    row["rel_1m_vs_market_factor"] = take(
        "rel_1m_vs_market_factor", "benchmark_relative", industry_analytics.MARKET_FACTOR_ID, "1m", "value",
    )
    row["breadth_1m"] = take("breadth_1m", "breadth", "1m", "pct_positive")
    row["dispersion_1m"] = take("dispersion_1m", "dispersion", "stdev")
    row["val_median_ev_ebitda"] = take("val_median_ev_ebitda", "valuation", "ev_ebitda", "median")
    row["val_median_pe_ttm"] = take("val_median_pe_ttm", "valuation", "pe_ttm", "median")
    row["fund_momentum"] = take("fund_momentum", "fundamental_momentum", "value")
    row["reasons"] = reasons
    return row


def compute_cross_snapshot(
    period_key: str, as_of: datetime, *, version: VersionInfo | str | int | None = None,
    persist: bool = True, loaders: SnapshotLoaders | None = None,
) -> CrossIndustrySnapshot:
    """Build (and by default upsert) the snapshot row for ``period_key``.

    Groups come from the registry, stats from the period's rows; a group
    without a row is listed under ``missing_groups``. Dependency
    spillovers are evaluated over the linked groups' observed 1M moves
    and labelled with the link's source. Returns the detached row.
    """
    info = gics_registry.resolve_version(version)
    ld = loaders or SnapshotLoaders()
    cutoff = as_of.date()
    groups = gics_registry.industry_groups(version=info)
    stats = ld.stats(period_key, info)

    members: dict[str, str] = {}
    for code, row in stats.items():
        for ticker in (row.get("per_ticker") or {}):
            members[str(ticker).upper()] = code
    try:
        events = ld.events(sorted(members), cutoff, EVENT_WINDOW_DAYS) if members else []
    except Exception as exc:
        log.debug("catalyst read failed for snapshot: %s", type(exc).__name__)
        events = []
    events_by_group: dict[str, int] = {}
    for ev in events:
        code = members.get(str(ev.get("ticker") or "").upper())
        if code:
            ev["industry_group_code"] = code
            events_by_group[code] = events_by_group.get(code, 0) + 1
    major = sorted(
        events,
        key=lambda e: (_MATERIALITY_RANK.get(str(e.get("materiality")), 3), e.get("event_date") or "", e.get("ticker") or ""),
    )[:MAX_MAJOR_EVENTS]

    rows = [_group_row(node, stats.get(node.code), events_by_group.get(node.code, 0)) for node in groups]
    missing = [
        {"code": node.code, "name": node.name, "reason": "no_stats_for_period"}
        for node in groups if node.code not in stats
    ]
    insufficient = [r["code"] for r in rows if r["status"] == industry_analytics.REASON_INSUFFICIENT]

    try:
        macro = ld.macro()
    except Exception as exc:
        log.debug("macro broadcast read failed: %s", type(exc).__name__)
        macro = None

    by_code = {r["code"]: r for r in rows}
    graph = dependency_graph()
    spillovers: list[dict[str, Any]] = []
    for link in graph["links"]:
        moves = []
        for code in link["codes"]:
            r = by_code.get(code)
            if r is None:
                continue
            moves.append({"code": code, "name": r["name"], "ret_1m_ew": r["ret_1m_ew"],
                          "rel_1m_vs_universe": r["rel_1m_vs_universe"], "breadth_1m": r["breadth_1m"],
                          "status": r["status"]})
        observed = [m for m in moves if m["ret_1m_ew"] is not None]
        active = [m for m in observed if abs(float(m["ret_1m_ew"])) >= SPILLOVER_MOVE]
        spillovers.append({
            "id": link["id"], "kind": link["kind"], "label": link["label"], "source": link["source"],
            "mechanism": link["mechanism"], "invalidation": link["invalidation"], "status": link["status"],
            "codes": link["codes"],
            "moves": moves,
            "signal": (
                "active" if active else
                "dormant" if observed else
                "unobserved"
            ),
            "n_observed": len(observed),
            "n_unobserved": len(link["codes"]) - len(observed),
        })
    spillovers.sort(key=lambda s: ({"active": 0, "dormant": 1, "unobserved": 2}[s["signal"]], s["id"]))
    spillovers = spillovers[:MAX_SPILLOVERS]

    try:
        report_versions = ld.report_versions(info)
    except Exception as exc:
        log.debug("report versions read failed: %s", type(exc).__name__)
        report_versions = {}

    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "period_key": period_key,
        "as_of": as_of.isoformat(),
        "taxonomy_version": info.version_key,
        "groups": rows,
        "missing_groups": missing,
        "insufficient_sample_groups": insufficient,
        "coverage": {
            "n_groups": len(groups),
            "n_with_stats": len(groups) - len(missing),
            "n_insufficient_sample": len(insufficient),
        },
        "regime": {
            "macro_regime": (macro or {}).get("regime"),
            "macro_as_of": (macro or {}).get("as_of"),
            "favored_sectors": (macro or {}).get("favored_sectors") or [],
            "pressured_sectors": (macro or {}).get("pressured_sectors") or [],
            "source": "macro_broadcast" if macro else None,
            "reason": None if macro else "no_macro_broadcast",
            "group_rules": REGIME_RULES,
        },
        "spillovers": spillovers,
        "dependency_graph": {
            "available": graph["available"],
            "atlas_snapshot_date": graph.get("atlas_snapshot_date"),
            "knowledge_map_as_of": graph.get("knowledge_map_as_of"),
            "n_links": len(graph["links"]),
            "unlabelled_edges": graph.get("unlabelled_edges", 0),
            "caveat": "analyst causal hypotheses and themed relationships, not estimated correlations",
        },
        "major_events": major,
        "events_window_days": EVENT_WINDOW_DAYS,
        "n_events": len(events),
        "observed_vs_interpretation": (
            "groups[], missing_groups[], major_events[] are observed data from stored rows; "
            "regime_label and spillover signal are rule-based reads stated in regime.group_rules"
        ),
    }
    row = CrossIndustrySnapshot(
        taxonomy_version_id=info.id,
        period_key=period_key,
        as_of=as_of,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        payload=payload,
        stats_ids=sorted(int(s["id"]) for s in stats.values() if s.get("id") is not None),
        report_versions=dict(report_versions),
        computed_at=_utcnow(),
    )
    if not persist:
        return row
    return _upsert(row)


def _upsert(row: CrossIndustrySnapshot) -> CrossIndustrySnapshot:
    with SessionLocal() as db:
        existing = db.execute(
            select(CrossIndustrySnapshot).where(
                CrossIndustrySnapshot.taxonomy_version_id == row.taxonomy_version_id,
                CrossIndustrySnapshot.period_key == row.period_key,
                CrossIndustrySnapshot.as_of == row.as_of,
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.payload = row.payload
            existing.stats_ids = row.stats_ids
            existing.report_versions = row.report_versions
            existing.schema_version = row.schema_version
            existing.computed_at = row.computed_at
            db.commit()
            db.refresh(existing)
            db.expunge(existing)
            return existing
        db.add(row)
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def snapshot_dict(row: CrossIndustrySnapshot) -> dict[str, Any]:
    return {
        "id": row.id,
        "taxonomy_version_id": row.taxonomy_version_id,
        "period_key": row.period_key,
        "as_of": row.as_of.isoformat() if row.as_of else None,
        "schema_version": row.schema_version,
        "payload": dict(row.payload or {}),
        "stats_ids": list(row.stats_ids or []),
        "report_versions": dict(row.report_versions or {}),
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
    }


def latest_snapshot(*, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    """The newest snapshot for the active (or given) taxonomy — one row.
    ``None`` when nothing has been computed or no taxonomy is active."""
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return None
    with SessionLocal() as db:
        row = db.execute(
            select(CrossIndustrySnapshot).where(
                CrossIndustrySnapshot.taxonomy_version_id == info.id,
            ).order_by(CrossIndustrySnapshot.as_of.desc(), CrossIndustrySnapshot.id.desc())
        ).scalars().first()
        return snapshot_dict(row) if row is not None else None


def snapshot_for_period(period_key: str, *, version: VersionInfo | str | int | None = None) -> dict[str, Any] | None:
    info = gics_registry.resolve_version(version)
    with SessionLocal() as db:
        row = db.execute(
            select(CrossIndustrySnapshot).where(
                CrossIndustrySnapshot.taxonomy_version_id == info.id,
                CrossIndustrySnapshot.period_key == period_key,
            ).order_by(CrossIndustrySnapshot.as_of.desc(), CrossIndustrySnapshot.id.desc())
        ).scalars().first()
        return snapshot_dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Rendering — the PM block
# ---------------------------------------------------------------------------


def _pct(value: Any, *, pp: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:+.1f}{'pp' if pp else '%'}"


def _share(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _mult(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}x"
    except (TypeError, ValueError):
        return "n/a"


def _short(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def render_pm_block(snapshot: CrossIndustrySnapshot | dict[str, Any] | None, max_chars: int = PM_BLOCK_MAX_CHARS) -> str:
    """The compact cross-industry block the PM reads — never longer than
    ``max_chars``, and never quietly shorter than it claims.

    The block is assembled from four parts in a fixed priority order: a
    one-line head (period, coverage, macro regime) that is always
    present; the column legend; one line per group with stats, sorted by
    1M relative return (unknowns last); and a tail of context lines
    (groups without stats, active dependency links, catalyst count).
    Whatever does not fit is DROPPED AS WHOLE LINES and counted in a
    closing note — the block never ends mid-sentence and never presents
    a partial table as a complete one. Only a budget too small for the
    head itself (under ~120 characters) falls back to clipping, because
    at that size there is nothing honest left to say.
    """
    if snapshot is None:
        return ""
    doc = snapshot_dict(snapshot) if isinstance(snapshot, CrossIndustrySnapshot) else dict(snapshot)
    payload = doc.get("payload") or {}
    budget = int(max_chars)
    if budget <= 0:
        return ""
    regime = (payload.get("regime") or {}).get("macro_regime") or "n/a"
    cov = payload.get("coverage") or {}
    head = (
        f"Cross-industry snapshot {doc.get('period_key')} (as of {str(doc.get('as_of') or '')[:10]}; "
        f"{cov.get('n_with_stats', 0)}/{cov.get('n_groups', 0)} groups with stats; macro regime: {regime})."
    )
    legend = (
        f"Observed data from stored rows (taxonomy {payload.get('taxonomy_version')}); n = constituents with "
        "prices; regime labels are rule-based reads. Columns: code name | 1W/1M/YTD EW | rel-1M vs universe "
        "| breadth-1M | EV/EBITDA median | regime | n"
    )
    groups = list(payload.get("groups") or [])

    def sort_key(r: dict[str, Any]) -> tuple[int, float, str]:
        rel = r.get("rel_1m_vs_universe")
        return (0 if rel is not None else 1, -(float(rel) if rel is not None else 0.0), str(r.get("code")))

    lines: list[str] = []
    for r in sorted(groups, key=sort_key):
        if r.get("status") == "no_stats":
            # Named in the tail with the count; a line of n/a cells would
            # spend budget saying the same thing.
            continue
        lines.append(
            f"{r.get('code')} {_short(str(r.get('name') or ''))} | "
            f"{_pct(r.get('ret_1w_ew'))}/{_pct(r.get('ret_1m_ew'))}/{_pct(r.get('ret_ytd_ew'))} | "
            f"{_pct(r.get('rel_1m_vs_universe'), pp=True)} | {_share(r.get('breadth_1m'))} | "
            f"{_mult(r.get('val_median_ev_ebitda'))} | {r.get('regime_label')} | n={r.get('n') if r.get('n') is not None else 'n/a'}"
        )
    missing = payload.get("missing_groups") or []
    tail: list[str] = []
    if missing:
        codes = ", ".join(m["code"] for m in missing[:8]) + (" …" if len(missing) > 8 else "")
        tail.append(f"No stats this period for {len(missing)} group(s): {codes}.")
    active = [s for s in (payload.get("spillovers") or []) if s.get("signal") == "active"]
    if active:
        parts = []
        for s in active[:4]:
            moved = ", ".join(
                f"{m['code']} {_pct(m.get('ret_1m_ew'))}" for m in s.get("moves", []) if m.get("ret_1m_ew") is not None
            )
            parts.append(f"{s['id']} {_short(s.get('label') or '', 40)} [{s.get('source')}]: {moved}")
        tail.append("Dependency links with a ≥5% 1M move (analyst hypotheses, not correlations): " + "; ".join(parts) + ".")
    n_events = payload.get("n_events")
    if n_events:
        tail.append(f"{n_events} catalyst event(s) in the next {payload.get('events_window_days', EVENT_WINDOW_DAYS)} days across covered constituents.")

    def assemble(n_lines: int, keep_legend: bool, n_tail: int, *, compact: bool = False) -> str:
        dropped_lines = len(lines) - n_lines
        dropped_context = (0 if keep_legend else 1) + (len(tail) - n_tail)
        parts = [head]
        if keep_legend:
            parts.append(legend)
        parts.extend(lines[:n_lines])
        parts.extend(tail[:n_tail])
        if dropped_lines or dropped_context:
            if compact:
                # Budget too small even for the counted note: still say that
                # the block is partial rather than read as complete.
                parts.append("… truncated for length.")
            else:
                bits = []
                if dropped_lines:
                    bits.append(f"{dropped_lines} of {len(lines)} group line(s)")
                if dropped_context:
                    bits.append(f"{dropped_context} context line(s)")
                parts.append("… omitted for length: " + " and ".join(bits) + ".")
        return "\n".join(parts)

    # Preference order: the whole spine, then shed context from the least
    # load-bearing end (catalyst count, spillovers), then the legend, then
    # the missing-groups line. Within each spine, keep as many group lines
    # as fit. The first configuration that fits wins.
    configs = ([(True, n) for n in range(len(tail), -1, -1)]
               + [(False, n) for n in range(len(tail), -1, -1)])
    for compact in (False, True):
        for keep_legend, n_tail in configs:
            for k in range(len(lines), -1, -1):
                text = assemble(k, keep_legend, n_tail, compact=compact)
                if len(text) <= budget:
                    return text
    # Nothing fits around the head itself — the caller's budget is smaller
    # than one sentence. Clip, and say so with the ellipsis.
    return head[: budget - 1] + "…" if budget > 1 else head[:budget]


# ---------------------------------------------------------------------------
# Relevance — which groups a ticker (or a portfolio) should read
# ---------------------------------------------------------------------------


def relevant_groups_detail(
    tickers: str | Iterable[str], *, cap: int | None = None,
    version: VersionInfo | str | int | None = None,
) -> dict[str, Any]:
    """The companies' own groups plus at most ``cap`` groups linked by the
    dependency graph (``INDUSTRY_PM_MAX_LINKED_GROUPS`` by default),
    ranked by how many links join them and then by code. Unmapped
    tickers are named, not dropped silently."""
    symbols = [tickers] if isinstance(tickers, str) else list(tickers)
    symbols = [str(t).strip().upper() for t in symbols if str(t).strip()]
    limit = settings.industry_pm_max_linked_groups if cap is None else int(cap)
    limit = max(0, limit)
    own: list[str] = []
    by_ticker: dict[str, dict[str, Any]] = {}
    unmapped: list[dict[str, Any]] = []
    if symbols:
        try:
            info = gics_registry.resolve_version(version)
            current = industry_classification.current_for(symbols, version=info)
        except gics_registry.TaxonomyNotImported:
            current = {}
        for t in symbols:
            row = current.get(t)
            code = row.get("industry_group_code") if row else None
            state = row.get("state") if row else "unclassified"
            if code and state in (industry_classification.STATE_MAPPED, industry_classification.STATE_CONFLICT):
                by_ticker[t] = {"code": code, "state": state}
                if code not in own:
                    own.append(code)
            else:
                unmapped.append({"ticker": t, "state": state})
    link_counts: dict[str, list[dict[str, Any]]] = {}
    for code in own:
        for other, links in linked_group_codes(code).items():
            if other in own:
                continue
            link_counts.setdefault(other, []).extend(links)
    ranked = sorted(link_counts.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    linked = [
        {"code": code, "via": [{"id": link["id"], "kind": link["kind"], "source": link["source"], "label": link["label"]}
                               for link in links]}
        for code, links in ranked[:limit]
    ]
    return {
        "tickers": symbols,
        "own": own,
        "linked": linked,
        "unmapped": unmapped,
        "cap": limit,
        "n_linked_candidates": len(link_counts),
        "by_ticker": by_ticker,
    }


def relevant_groups_for(
    tickers: str | Iterable[str], *, cap: int | None = None,
    version: VersionInfo | str | int | None = None,
) -> list[str]:
    """Own group codes first, then the capped linked groups."""
    detail = relevant_groups_detail(tickers, cap=cap, version=version)
    return list(detail["own"]) + [item["code"] for item in detail["linked"]]


def group_rows(snapshot: dict[str, Any] | None, codes: Iterable[str]) -> list[dict[str, Any]]:
    """The snapshot's rows for ``codes``, in the order given; a code the
    snapshot does not carry yields a row that says so."""
    wanted = [str(c) for c in codes]
    rows = {r.get("code"): r for r in ((snapshot or {}).get("payload") or {}).get("groups", [])}
    out = []
    for code in wanted:
        row = rows.get(code)
        out.append(row if row is not None else {"code": code, "status": "not_in_snapshot"})
    return out
