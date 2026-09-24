"""FEAT-003 — the versioned GICS taxonomy registry.

One taxonomy source. The registry is populated from the knowledge JSON
(``data/industry_knowledge/gics_industries_2026.json``, built by
``scripts/build_industry_knowledge`` from the encyclopedia and the universe
map) and nothing else: no second seed file, no hand-typed node list, no
literal group count anywhere. Four levels are imported — sector (2 digits)
/ industry group (4) / industry (6) / sub-industry (8) — plus the retired
sub-industries as inactive nodes so a historical classification can still
be named.

Two processes share only Postgres, so:

* ``active_version()`` is a database read on EVERY call. Import and
  activation happen on the web process (admin route / CLI) or the worker;
  a process-local "active" flag would leave the other process classifying
  and reporting against a stale structure until its next restart — the
  module-dict-across-processes failure this repo has already paid for.
* Only the immutable node lists are cached, keyed by
  ``taxonomy_version.id``. Nodes never change after import (a different
  structure is a different version with a new id), so a per-id cache can
  never be wrong, in either process.

Display: internally, codes and names with ``ATTRIBUTION``; company
mappings are labelled with ``MAPPING_CAVEAT``. Publicly, MarketMosaic's own
labels (owner decision 2026-09-24, licensing): ``display()`` returns the
``industry_labels`` label in ``internal_labels`` mode, and every public
surface projects through ``industry_labels`` whatever the mode says. No
S&P/MSCI constituent files are ever imported.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select, update

from ..config import settings
from ..database import SessionLocal
from ..models import GicsNode, TaxonomyVersion
from ..models.industry import LEVEL_BY_CODE_LENGTH, NODE_LEVELS

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "GICS® industry codes and names are the public structure published by "
    "MSCI and S&P Global. MarketMosaic displays them with attribution and "
    "holds no licensed constituent or issuer-assignment data."
)
MAPPING_CAVEAT = (
    "derived from provider classification, not licensed GICS security assignments"
)
DISPLAY_MODES: tuple[str, ...] = ("codes_and_names", "internal_labels")
SOURCE_LABEL = "knowledge_json"


class TaxonomyNotImported(RuntimeError):
    """No active taxonomy version exists — read paths answer 503 with this."""


class TaxonomyChecksumMismatch(ValueError):
    """The same ``version_key`` already holds a different node set. A
    structure change is a new version, never a rewrite of an old one."""


class UnknownTaxonomyVersion(LookupError):
    """``version`` did not resolve to an imported taxonomy version."""


class UnknownNode(LookupError):
    """A code that is not in the requested taxonomy version."""


@dataclass(frozen=True)
class VersionInfo:
    """Detached snapshot of a ``gics_taxonomy_versions`` row."""

    id: int
    version_key: str
    checksum: str
    source: str
    is_active: bool
    effective_from: date | None
    effective_to: date | None
    node_counts: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    attribution: str = ATTRIBUTION
    imported_at: datetime | None = None
    activated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.version_key,
            "checksum": self.checksum,
            "source": self.source,
            "is_active": self.is_active,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "node_counts": dict(self.node_counts),
            "provenance": dict(self.provenance),
            "attribution": self.attribution,
            "display_mode": display_mode(),
            "mapping_caveat": MAPPING_CAVEAT,
        }


@dataclass(frozen=True)
class NodeInfo:
    """Detached snapshot of a ``gics_nodes`` row."""

    code: str
    name: str
    level: str
    parent_code: str | None
    is_active: bool
    effective_from: date | None
    effective_to: date | None
    sort_order: int
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def sector_code(self) -> str:
        return self.code[:2]

    @property
    def industry_group_code(self) -> str | None:
        return self.code[:4] if len(self.code) >= 4 else None

    @property
    def industry_code(self) -> str | None:
        return self.code[:6] if len(self.code) >= 6 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "level": self.level,
            "parent_code": self.parent_code,
            "is_active": self.is_active,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "display": display(self),
        }


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def display_mode() -> str:
    """The configured display mode, validated; an unknown value is reported
    as the default rather than acted on."""
    mode = (settings.gics_display_mode or "").strip()
    return mode if mode in DISPLAY_MODES else DISPLAY_MODES[0]


# One warning per process, not per call: `display()` can run per row.
_CODES_AND_NAMES_WARNED = False


def display(node: NodeInfo) -> str:
    """How a node is named.

    ``internal_labels``: MarketMosaic's own label (``industry_labels``). A
    sector or group gets its own label; an industry or sub-industry is
    never named publicly, so it is shown as the label of the group it rolls
    up to.

    ``codes_and_names``: ``"Semiconductors & Semiconductor Equipment (4530)"``,
    with a one-time WARNING — that rendering carries licensed names and codes
    and is for internal/admin use only; the public projection runs regardless
    of this setting (licensing is not a toggle). A display mode never changes
    what is stored."""
    global _CODES_AND_NAMES_WARNED
    if display_mode() == "internal_labels":
        # Imported here: `industry_labels` reads this module's constants
        # for its scrub table, so a top-level import would be circular.
        from . import industry_labels

        return industry_labels.label(node.code if len(node.code) <= 4 else node.code[:4])
    if not _CODES_AND_NAMES_WARNED:
        _CODES_AND_NAMES_WARNED = True
        log.warning(
            "gics_display_mode=codes_and_names renders taxonomy codes and names; "
            "public surfaces must project through industry_labels (owner decision 2026-09-24)"
        )
    return f"{node.name} ({node.code})"


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def nodes_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The canonical node list for a knowledge payload, in document order.

    Every active level's ``effective_from`` is the structure's official
    effective date from the map metadata; retired sub-industries are
    inactive with that date as ``effective_to`` (the structure that retired
    them is the one that took effect then) and no ``effective_from`` (the
    map does not say when they began). Nothing here counts anything.
    """
    meta = payload.get("map_metadata") or {}
    effective = _parse_date(meta.get("taxonomy_structure_effective"))
    rows: list[dict[str, Any]] = []
    order = 0

    def add(level: str, code: str, name: str, parent: str | None, *, active: bool = True,
            attributes: dict[str, Any] | None = None) -> None:
        nonlocal order
        order += 1
        rows.append({
            "level": level,
            "code": str(code),
            "name": str(name),
            "parent_code": parent,
            # A retired row's own start date is unknown to the map; only the
            # structure that discontinued it is dated.
            "effective_from": effective if active else None,
            "effective_to": None if active else effective,
            "is_active": active,
            "sort_order": order,
            "attributes": attributes or {},
        })

    for sector in payload.get("sectors", []):
        add("sector", sector["code"], sector["name"], None)
        for group in sector.get("industry_groups", []):
            add("industry_group", group["code"], group["name"], sector["code"])
            for industry in group.get("industries", []):
                add("industry", industry["code"], industry["name"], group["code"])
                for sub in industry.get("sub_industries", []):
                    add("sub_industry", sub["code"], sub["name"], industry["code"])
    for retired in payload.get("retired_sub_industries", []):
        add(
            "sub_industry", retired["code"], retired["name"], retired.get("parent_code"),
            active=False,
            attributes={"status": retired.get("status", "discontinued"),
                        "parent_name": retired.get("parent_name", "")},
        )
    return rows


def checksum_for(nodes: list[dict[str, Any]]) -> str:
    """sha256 over the canonical (level, code, name, parent, active, dates)
    tuples — the identity of a structure, independent of the prose that
    travels with it in the knowledge JSON."""
    canonical = sorted(
        (
            n["level"], n["code"], n["name"], n["parent_code"] or "", bool(n["is_active"]),
            n["effective_from"].isoformat() if n["effective_from"] else "",
            n["effective_to"].isoformat() if n["effective_to"] else "",
        )
        for n in nodes
    )
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode("utf-8")).hexdigest()


def node_counts_for(nodes: list[dict[str, Any]]) -> dict[str, int]:
    """Active nodes per level plus ``inactive`` — the shape stored on the
    version row and reported by the drift check."""
    counts: dict[str, int] = {level: 0 for level in NODE_LEVELS}
    inactive = 0
    for n in nodes:
        if n["is_active"]:
            counts[n["level"]] += 1
        else:
            inactive += 1
    counts["inactive"] = inactive
    return counts


def _load_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        from .industry_knowledge import load_industry_knowledge
        return load_industry_knowledge()
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _version_info(row: TaxonomyVersion) -> VersionInfo:
    return VersionInfo(
        id=row.id,
        version_key=row.version_key,
        checksum=row.checksum or "",
        source=row.source or "",
        is_active=bool(row.is_active),
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        node_counts=dict(row.node_counts or {}),
        provenance=dict(row.provenance or {}),
        attribution=row.attribution or ATTRIBUTION,
        imported_at=row.imported_at,
        activated_at=row.activated_at,
    )


def import_from_knowledge_json(
    path: Path | None = None,
    *,
    payload: dict[str, Any] | None = None,
    version_key: str | None = None,
    activate: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Import the taxonomy in the knowledge JSON as one version.

    Idempotent by checksum: a version with the same key and the same node
    checksum is left untouched (``imported=False``); the same key with a
    different checksum raises ``TaxonomyChecksumMismatch`` — the caller
    must choose a new key for a changed structure. Nodes are written in
    one bulk insert. ``activate=True`` makes this version the active one
    (whether or not it was just imported).
    """
    doc = payload if payload is not None else _load_payload(path)
    key = (version_key or doc.get("taxonomy_version") or "").strip()
    if not key:
        raise ValueError("knowledge payload carries no taxonomy_version and no version_key was given")
    nodes = nodes_from_payload(doc)
    if not nodes:
        raise ValueError("knowledge payload holds no taxonomy nodes")
    checksum = checksum_for(nodes)
    counts = node_counts_for(nodes)
    meta = doc.get("map_metadata") or {}
    provenance = {
        "generated_from": doc.get("generated_from"),
        "source_sha256": doc.get("source_sha256"),
        "map_generated_from": doc.get("map_generated_from"),
        "map_source_sha256": doc.get("map_source_sha256"),
        "map_as_of": doc.get("map_as_of"),
        "taxonomy_structure_effective": meta.get("taxonomy_structure_effective"),
        "official_workbook_sha256": meta.get("official_workbook_sha256"),
    }

    with SessionLocal() as db:
        existing = db.execute(
            select(TaxonomyVersion).where(TaxonomyVersion.version_key == key)
        ).scalar_one_or_none()
        imported = False
        if existing is not None:
            if existing.checksum != checksum:
                raise TaxonomyChecksumMismatch(
                    f"taxonomy version {key!r} is already imported with checksum "
                    f"{existing.checksum[:12]}…; the payload hashes to {checksum[:12]}…. "
                    "A changed structure needs a new version key."
                )
            version = existing
        else:
            version = TaxonomyVersion(
                version_key=key,
                source=SOURCE_LABEL,
                effective_from=_parse_date(meta.get("taxonomy_structure_effective")),
                effective_to=None,
                is_active=False,
                checksum=checksum,
                node_counts=counts,
                provenance=provenance,
                attribution=ATTRIBUTION,
                notes=notes or "",
                imported_at=_utcnow(),
            )
            db.add(version)
            db.flush()  # assigns the id the nodes reference
            db.execute(
                insert(GicsNode),
                [{"taxonomy_version_id": version.id, **n} for n in nodes],
            )
            imported = True
        if activate:
            _activate_in_session(db, version)
        db.commit()
        info = _version_info(version)
    invalidate_cache()
    log.info(
        "gics taxonomy %s: %s (%s nodes, active=%s)",
        key, "imported" if imported else "already present", len(nodes), info.is_active,
    )
    return {
        "version_key": key,
        "version_id": info.id,
        "imported": imported,
        "nodes_inserted": len(nodes) if imported else 0,
        "node_counts": dict(counts),
        "checksum": checksum,
        "activated": info.is_active,
    }


def _activate_in_session(db, version: TaxonomyVersion) -> None:
    """Exactly one active row, flipped inside the caller's transaction."""
    db.execute(
        update(TaxonomyVersion)
        .where(TaxonomyVersion.is_active.is_(True), TaxonomyVersion.id != version.id)
        .values(is_active=False)
    )
    if not version.is_active:
        version.is_active = True
        version.activated_at = _utcnow()


def activate_version(version_key: str) -> VersionInfo:
    """Make ``version_key`` the single active taxonomy. Both processes see
    the flip on their next ``active_version()`` read."""
    with SessionLocal() as db:
        row = db.execute(
            select(TaxonomyVersion).where(TaxonomyVersion.version_key == version_key)
        ).scalar_one_or_none()
        if row is None:
            raise UnknownTaxonomyVersion(f"taxonomy version {version_key!r} is not imported")
        _activate_in_session(db, row)
        db.commit()
        return _version_info(row)


def bundled_drift(
    active: VersionInfo | None = None, *, payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Does the bundled knowledge JSON describe the structure the registry
    is serving? ``None`` when the active version's checksum equals the
    bundled payload's; otherwise a dict naming both sides and the remedy.

    This is the "ONE taxonomy source" invariant made observable. The
    knowledge JSON is regenerated whenever the user-authored map evolves,
    so a deploy can ship a changed structure under the SAME version key —
    ``import_from_knowledge_json`` refuses to rewrite it, which is right,
    but refusing silently leaves the worker classifying against the old
    node set. The loop reads this every day and reports it. Two kinds:

    * ``same_key_changed_structure`` — the bundled file hashes differently
      under the active key; the operator re-imports under a new key
      (``--version-key``) and activates it.
    * ``bundled_version_not_active`` — the bundled file carries another
      version key; import/activate it, or keep the pin deliberately.
    """
    if active is None:
        active = active_version()
    if active is None:
        return None
    if payload is None:
        payload = _load_payload(None)
    bundled_key = str(payload.get("taxonomy_version") or settings.gics_taxonomy_version)
    bundled_nodes = nodes_from_payload(payload)
    bundled_checksum = checksum_for(bundled_nodes)
    if bundled_checksum == active.checksum:
        return None
    same_key = bundled_key == active.version_key
    return {
        "kind": "same_key_changed_structure" if same_key else "bundled_version_not_active",
        "active_version_key": active.version_key,
        "active_checksum": active.checksum,
        "bundled_version_key": bundled_key,
        "bundled_checksum": bundled_checksum,
        "active_node_counts": dict(active.node_counts),
        "bundled_node_counts": node_counts_for(bundled_nodes),
        "remedy": (
            "re-import the bundled JSON under a new key "
            "(python -m app.scripts.import_gics_taxonomy --version-key <key> --activate)"
            if same_key else
            f"import and activate {bundled_key!r} "
            "(python -m app.scripts.import_gics_taxonomy --activate), or keep the pin deliberately"
        ),
    }


def ensure_taxonomy(*, activate: bool = True) -> VersionInfo | None:
    """Bootstrap: import the bundled knowledge JSON when its version is not
    yet in the database, and activate it when nothing is active.

    Called by the classification loop and the CLI. Never changes an
    existing active version — activation of a NEW structure is a
    deliberate operator action (``activate_version``). Returns the active
    version, or ``None`` when ``activate`` is False and nothing is active.

    When a version IS active its checksum is compared with the bundled
    JSON's and a mismatch is logged as a warning — the registry must never
    diverge from the one taxonomy source without saying so. The loop turns
    the same comparison (``bundled_drift``) into a red health row.
    """
    active = active_version()
    if active is not None:
        drift = bundled_drift(active)
        if drift is not None:
            log.warning(
                "gics taxonomy drift: active %s (checksum %s…) differs from the bundled "
                "knowledge JSON %s (checksum %s…) [%s]; %s",
                drift["active_version_key"], drift["active_checksum"][:12],
                drift["bundled_version_key"], drift["bundled_checksum"][:12],
                drift["kind"], drift["remedy"],
            )
        return active
    payload = _load_payload(None)
    key = payload.get("taxonomy_version") or settings.gics_taxonomy_version
    if key != settings.gics_taxonomy_version:
        log.warning(
            "bundled knowledge JSON is taxonomy %s but GICS_TAXONOMY_VERSION=%s; "
            "importing the bundled version", key, settings.gics_taxonomy_version,
        )
    import_from_knowledge_json(payload=payload, version_key=key, activate=activate)
    return active_version()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def active_version() -> VersionInfo | None:
    """The active taxonomy — a database read on every call, by design."""
    with SessionLocal() as db:
        row = db.execute(
            select(TaxonomyVersion)
            .where(TaxonomyVersion.is_active.is_(True))
            .order_by(TaxonomyVersion.id.desc())
        ).scalars().first()
        return _version_info(row) if row is not None else None


def require_active_version() -> VersionInfo:
    info = active_version()
    if info is None:
        raise TaxonomyNotImported("taxonomy not imported")
    return info


def list_versions() -> list[VersionInfo]:
    with SessionLocal() as db:
        rows = db.execute(select(TaxonomyVersion).order_by(TaxonomyVersion.id)).scalars().all()
        return [_version_info(r) for r in rows]


def resolve_version(version: VersionInfo | str | int | None = None) -> VersionInfo:
    """``None`` → the active version; a key or an id → that version."""
    if isinstance(version, VersionInfo):
        return version
    if version is None:
        return require_active_version()
    with SessionLocal() as db:
        if isinstance(version, int):
            row = db.get(TaxonomyVersion, version)
        else:
            row = db.execute(
                select(TaxonomyVersion).where(TaxonomyVersion.version_key == str(version))
            ).scalar_one_or_none()
        if row is None:
            raise UnknownTaxonomyVersion(f"taxonomy version {version!r} is not imported")
        return _version_info(row)


@lru_cache(maxsize=8)
def _node_rows(version_id: int) -> tuple[NodeInfo, ...]:
    """Every node of one version, in import order. Cached per version id —
    node rows are immutable after import, so this can never go stale."""
    with SessionLocal() as db:
        rows = db.execute(
            select(GicsNode)
            .where(GicsNode.taxonomy_version_id == version_id)
            .order_by(GicsNode.sort_order, GicsNode.id)
        ).scalars().all()
        return tuple(
            NodeInfo(
                code=r.code, name=r.name, level=r.level, parent_code=r.parent_code,
                is_active=bool(r.is_active), effective_from=r.effective_from,
                effective_to=r.effective_to, sort_order=r.sort_order or 0,
                attributes=dict(r.attributes or {}),
            )
            for r in rows
        )


@lru_cache(maxsize=8)
def _node_index(version_id: int) -> dict[tuple[str, str], NodeInfo]:
    """``(level, code) -> node`` for one version, including inactive rows —
    the classifier probes hundreds of provider-derived codes per run."""
    return {(n.level, n.code): n for n in _node_rows(version_id)}


def invalidate_cache() -> None:
    _node_rows.cache_clear()
    _node_index.cache_clear()


def nodes(
    level: str | None = None, *, version: VersionInfo | str | int | None = None,
    include_inactive: bool = False,
) -> list[NodeInfo]:
    info = resolve_version(version)
    return [
        n for n in _node_rows(info.id)
        if (level is None or n.level == level) and (include_inactive or n.is_active)
    ]


def node(code: str, *, version: VersionInfo | str | int | None = None,
         include_inactive: bool = False) -> NodeInfo | None:
    """One node by code (its level follows from the code length); ``None``
    when unknown. Never raises for a bad code — the classifier probes with
    provider-derived codes and must be able to say "not in this version"."""
    key = str(code or "").strip()
    level = LEVEL_BY_CODE_LENGTH.get(len(key))
    if level is None or not key.isdigit():
        return None
    info = resolve_version(version)
    found = _node_index(info.id).get((level, key))
    if found is None or not (include_inactive or found.is_active):
        return None
    return found


def group(code: str, version: VersionInfo | str | int | None = None) -> NodeInfo:
    """One industry group; raises ``UnknownNode`` so a route can answer 404."""
    found = node(code, version=version)
    if found is None or found.level != "industry_group":
        info = resolve_version(version)
        raise UnknownNode(f"industry group {code!r} not found in taxonomy {info.version_key}")
    return found


def industry_groups(version: VersionInfo | str | int | None = None) -> list[NodeInfo]:
    return nodes("industry_group", version=version)


def sectors(version: VersionInfo | str | int | None = None) -> list[NodeInfo]:
    return nodes("sector", version=version)


def children(code: str, *, version: VersionInfo | str | int | None = None,
             include_inactive: bool = False) -> list[NodeInfo]:
    """Direct children of ``code`` at the next level down."""
    key = str(code or "").strip()
    info = resolve_version(version)
    return [
        n for n in _node_rows(info.id)
        if n.parent_code == key and (include_inactive or n.is_active)
    ]


def industries_of_group(code: str, version: VersionInfo | str | int | None = None) -> list[NodeInfo]:
    return [n for n in children(code, version=version) if n.level == "industry"]


def sub_industries_of(code: str, version: VersionInfo | str | int | None = None,
                      *, include_inactive: bool = False) -> list[NodeInfo]:
    """Sub-industries under an industry (6) or, via prefix, a group (4) or
    sector (2)."""
    key = str(code or "").strip()
    info = resolve_version(version)
    return [
        n for n in _node_rows(info.id)
        if n.level == "sub_industry" and n.code.startswith(key)
        and (include_inactive or n.is_active)
    ]


def group_code_for(code: str, *, version: VersionInfo | str | int | None = None) -> str | None:
    """The active industry-group code a code of any level rolls up to, or
    ``None`` when the prefix is not an active group in this version."""
    key = str(code or "").strip()
    if len(key) < 4:
        return None
    found = node(key[:4], version=version)
    return found.code if found is not None and found.level == "industry_group" else None


def counts(version: VersionInfo | str | int | None = None) -> dict[str, int]:
    """Active node counts per level, plus ``inactive`` — always from rows."""
    info = resolve_version(version)
    out: dict[str, int] = {level: 0 for level in NODE_LEVELS}
    out["inactive"] = 0
    for n in _node_rows(info.id):
        if n.is_active:
            out[n.level] += 1
        else:
            out["inactive"] += 1
    return out


def tree(version: VersionInfo | str | int | None = None) -> dict[str, Any]:
    """Nested sectors → industry groups → industries → sub-industries, with
    the version's attribution — the shape the taxonomy endpoint serves."""
    info = resolve_version(version)
    by_parent: dict[str | None, list[NodeInfo]] = {}
    for n in _node_rows(info.id):
        if n.is_active:
            by_parent.setdefault(n.parent_code, []).append(n)

    def nest(parent: str | None, child_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        out = []
        for n in by_parent.get(parent, []):
            item = n.as_dict()
            if child_keys:
                item[child_keys[0]] = nest(n.code, child_keys[1:])
            out.append(item)
        return out

    return {
        "taxonomy_version": info.as_dict(),
        "sectors": nest(None, ("industry_groups", "industries", "sub_industries")),
    }
