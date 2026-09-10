"""FEAT-003 — placing companies in the GICS taxonomy, with an audit trail.

We hold no GICS codes for any company: ``Company.sector / industry /
sub_industry`` are the provider's (FMP's) own labels, and the demo fixtures
use their own. So a company's place in the registry is a *derived*
crosswalk, and every row says how it was derived:

1. ``research_map`` — the universe map's economic security reference,
   matched by exact symbol. It records 8-digit sub-industry codes and is
   the author's own research; the industry group is the code's 4-digit
   prefix. First code wins routing when several are recorded; all are
   kept.
2. ``provider_alias`` — ``data/industry_knowledge/provider_aliases.json``,
   a hand-maintained label crosswalk. An industry label that resolves
   gives ``mapped``; only a sector label resolving gives ``fallback``.
3. Nothing resolves → ``missing`` (the unmapped labels are reported so the
   alias map can be extended).

When (1) and (2) both name an industry GROUP and disagree, the row is
``conflict``: both candidates are stored and (1) wins routing. ``stale``
marks a current row whose inputs (company labels, alias-map version, map
edition) no longer match what it was computed from; ``classify_all``
detects that by fingerprint and re-classifies in the same pass,
superseding the old row so history is preserved. Every row carries
``source``, ``author`` and ``source_as_of`` and is labelled, wherever it is
displayed, as derived from provider classification — not a licensed GICS
security assignment.

Two processes, one database, and the web process must never loop:
``classify_all`` is two bulk reads and one bulk insert; ``current_for`` is
one SELECT joined to the active version; ``classify_ticker`` (the
on-demand hook) is one company read, one current-row read, one insert.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update

from ..database import SessionLocal
from ..models import Company, CompanyIndustryClassification, TaxonomyVersion
from ..models.industry import CLASSIFICATION_STATES
from . import gics_registry, industry_knowledge
from .gics_registry import MAPPING_CAVEAT, VersionInfo

log = logging.getLogger(__name__)

ALIAS_PATH = Path(__file__).resolve().parent.parent / "data" / "industry_knowledge" / "provider_aliases.json"

STATE_MAPPED = "mapped"
STATE_FALLBACK = "fallback"
STATE_MISSING = "missing"
STATE_STALE = "stale"
STATE_CONFLICT = "conflict"
assert set(CLASSIFICATION_STATES) == {
    STATE_MAPPED, STATE_FALLBACK, STATE_MISSING, STATE_STALE, STATE_CONFLICT,
}

SOURCE_RESEARCH_MAP = "research_map"
SOURCE_PROVIDER_ALIAS = "provider_alias"
SOURCE_NONE = "none"

# Ordinal confidence, documented on the model: research map > alias
# industry > alias sector. Not calibrated probabilities.
CONFIDENCE = {
    ("research_map", STATE_MAPPED): 0.9,
    ("research_map", STATE_CONFLICT): 0.6,
    ("provider_alias", STATE_MAPPED): 0.8,
    ("provider_alias", STATE_FALLBACK): 0.5,
    ("none", STATE_MISSING): 0.0,
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Alias map
# ---------------------------------------------------------------------------


def normalize_label(text: Any) -> str:
    """``"Software - Application "`` → ``"software application"``; ``"Oil &
    Gas E&P"`` → ``"oil and gas e and p"``. Case, punctuation and spacing
    differences between provider editions must not create new labels."""
    value = str(text or "").casefold().replace("&", " and ")
    return _NON_ALNUM.sub(" ", value).strip()


@dataclass(frozen=True)
class AliasMap:
    version: str
    as_of: str
    provider: str
    sectors: dict[str, str]
    industries: dict[str, str]

    @property
    def author(self) -> str:
        return f"provider_aliases.json@{self.version}"


@lru_cache(maxsize=1)
def alias_map() -> AliasMap:
    """The crosswalk, normalised once per process. It is checked-in data
    (a new entry is a commit), so a process-level cache cannot go stale."""
    with ALIAS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    sectors: dict[str, str] = {}
    industries: dict[str, str] = {}
    for label, code in (raw.get("sectors") or {}).items():
        key = normalize_label(label)
        if key and str(code).isdigit() and len(str(code)) == 2:
            sectors[key] = str(code)
    for label, code in (raw.get("industries") or {}).items():
        key = normalize_label(label)
        if key and str(code).isdigit() and len(str(code)) in (4, 6):
            industries[key] = str(code)
    return AliasMap(
        version=str(raw.get("version") or ""),
        as_of=str(raw.get("as_of") or ""),
        provider=str(raw.get("provider") or ""),
        sectors=sectors,
        industries=industries,
    )


# ---------------------------------------------------------------------------
# Resolution (pure, given the registry's node index)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    state: str
    source: str
    method: str
    author: str
    source_as_of: str
    confidence: float
    sector_code: str | None = None
    industry_group_code: str | None = None
    industry_code: str | None = None
    sub_industry_code: str | None = None
    sub_industry_codes: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.sector_code, self.industry_group_code, self.industry_code, self.sub_industry_code)


def _codes_from(code: str) -> dict[str, str | None]:
    return {
        "sector_code": code[:2],
        "industry_group_code": code[:4] if len(code) >= 4 else None,
        "industry_code": code[:6] if len(code) >= 6 else None,
        "sub_industry_code": code if len(code) == 8 else None,
    }


def _research_map_candidate(ticker: str, version: VersionInfo) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The map's entry for ``ticker`` resolved against this version, or
    ``None`` plus the evidence for why not."""
    ref = industry_knowledge.security_reference(ticker)
    if ref is None:
        return None, {}
    usable = [c for c in ref["codes"] if gics_registry.node(c, version=version) is not None]
    if not usable:
        return None, {"research_map_codes_unknown_in_version": list(ref["codes"])}
    primary = usable[0]
    # `author` names the map EDITION (its metadata as-of — the thing that
    # changes when the author re-issues the file); `source_as_of` keeps the
    # entry's own provider-snapshot date. They differ, and both matter.
    return (
        {
            "source": SOURCE_RESEARCH_MAP,
            "method": "security_reference",
            "author": f"Investment_Universe_163_Map.json@{industry_knowledge.security_reference_as_of()}",
            "source_as_of": ref["as_of"],
            "codes": list(usable),
            **_codes_from(primary),
        },
        {
            "research_map_source_id": ref["source_id"],
            "research_map_caveat": ref["caveat"],
            # Present only when the provider spells a share class differently
            # from the map (BRK.B vs BRK-B) — the row still says which entry.
            **({"research_map_symbol": ref["matched_symbol"]}
               if ref.get("matched_symbol", ref["symbol"]) != ref["symbol"] else {}),
        },
    )


def _alias_candidate(
    sector: str | None, industry: str | None, sub_industry: str | None, version: VersionInfo,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    aliases = alias_map()
    evidence: dict[str, Any] = {}
    sector_key = normalize_label(sector)
    sector_code = aliases.sectors.get(sector_key)
    if sector_code is not None and gics_registry.node(sector_code, version=version) is None:
        evidence["alias_sector_code_unknown_in_version"] = sector_code
        sector_code = None
    for label_kind, label in (("industry", industry), ("sub_industry", sub_industry)):
        key = normalize_label(label)
        if not key:
            continue
        code = aliases.industries.get(key)
        if code is None:
            continue
        if gics_registry.node(code, version=version) is None:
            evidence[f"alias_{label_kind}_code_unknown_in_version"] = code
            continue
        codes = _codes_from(code)
        if sector_code is not None and codes["sector_code"] != sector_code:
            # Not a conflict by the FEAT-003 definition (that is map vs
            # alias on the GROUP); recorded so the audit can see it.
            evidence["alias_sector_disagrees_with_industry"] = {
                "sector_code": sector_code, "industry_sector_code": codes["sector_code"],
            }
        return (
            {
                "source": SOURCE_PROVIDER_ALIAS,
                "method": f"alias_{label_kind}",
                "author": aliases.author,
                "source_as_of": aliases.as_of,
                "alias_key": key,
                **codes,
            },
            evidence,
        )
    if sector_code is not None:
        return (
            {
                "source": SOURCE_PROVIDER_ALIAS,
                "method": "alias_sector",
                "author": aliases.author,
                "source_as_of": aliases.as_of,
                "alias_key": sector_key,
                "sector_code": sector_code,
                "industry_group_code": None,
                "industry_code": None,
                "sub_industry_code": None,
            },
            evidence,
        )
    return None, evidence


def resolve(
    ticker: str, sector: str | None, industry: str | None, sub_industry: str | None,
    *, version: VersionInfo,
) -> Resolution:
    """Pure resolution for one company against one taxonomy version.

    No database writes; the registry's cached node index is the only
    lookup. ``classify_all`` calls this per company after its two bulk
    reads, ``classify_ticker`` once.
    """
    symbol = str(ticker or "").strip().upper()
    evidence: dict[str, Any] = {
        "labels": {"sector": sector, "industry": industry, "sub_industry": sub_industry},
        "mapping_caveat": MAPPING_CAVEAT,
    }
    research, ev = _research_map_candidate(symbol, version)
    evidence.update(ev)
    alias, ev = _alias_candidate(sector, industry, sub_industry, version)
    evidence.update(ev)

    if research is not None:
        state = STATE_MAPPED
        candidates: tuple[dict[str, Any], ...] = ()
        if (
            alias is not None
            and alias.get("industry_group_code")
            and alias["industry_group_code"] != research["industry_group_code"]
        ):
            state = STATE_CONFLICT
            candidates = (
                {k: research[k] for k in ("source", "method", "author", "source_as_of",
                                          "sector_code", "industry_group_code", "industry_code",
                                          "sub_industry_code")},
                {k: alias[k] for k in ("source", "method", "author", "source_as_of",
                                       "sector_code", "industry_group_code", "industry_code",
                                       "sub_industry_code")}
                | {"alias_key": alias.get("alias_key")},
            )
            evidence["conflict"] = "research map and provider alias disagree on the industry group; research map wins routing"
        elif alias is not None:
            evidence["provider_alias_agrees"] = alias.get("industry_group_code") == research["industry_group_code"]
            evidence["alias_key"] = alias.get("alias_key")
        return Resolution(
            state=state,
            source=SOURCE_RESEARCH_MAP,
            method=research["method"],
            author=research["author"],
            source_as_of=research["source_as_of"],
            confidence=CONFIDENCE[(SOURCE_RESEARCH_MAP, state)],
            sector_code=research["sector_code"],
            industry_group_code=research["industry_group_code"],
            industry_code=research["industry_code"],
            sub_industry_code=research["sub_industry_code"],
            sub_industry_codes=tuple(research["codes"]),
            candidates=candidates,
            evidence=evidence,
        )

    if alias is not None:
        state = STATE_MAPPED if alias.get("industry_group_code") else STATE_FALLBACK
        evidence["alias_key"] = alias.get("alias_key")
        if state == STATE_FALLBACK:
            evidence["unmapped_industry_label"] = industry
        return Resolution(
            state=state,
            source=SOURCE_PROVIDER_ALIAS,
            method=alias["method"],
            author=alias["author"],
            source_as_of=alias["source_as_of"],
            confidence=CONFIDENCE[(SOURCE_PROVIDER_ALIAS, state)],
            sector_code=alias["sector_code"],
            industry_group_code=alias["industry_group_code"],
            industry_code=alias["industry_code"],
            sub_industry_code=alias["sub_industry_code"],
            evidence=evidence,
        )

    evidence["unmapped_sector_label"] = sector
    evidence["unmapped_industry_label"] = industry
    return Resolution(
        state=STATE_MISSING,
        source=SOURCE_NONE,
        method="none",
        author="",
        source_as_of="",
        confidence=CONFIDENCE[(SOURCE_NONE, STATE_MISSING)],
        evidence=evidence,
    )


def inputs_fingerprint(sector: str | None, industry: str | None, sub_industry: str | None) -> str:
    """Identity of everything a resolution depends on besides the taxonomy
    version: the company's labels, the alias-map edition and the universe
    map edition. A row whose fingerprint differs from the current one is
    stale by definition — no re-resolution needed to know that."""
    aliases = alias_map()
    parts = [
        str(sector or ""), str(industry or ""), str(sub_industry or ""),
        aliases.version, industry_knowledge.security_reference_as_of(),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def row_dict(row: CompanyIndustryClassification) -> dict[str, Any]:
    """Detached snapshot of a classification row."""
    return {
        "id": row.id,
        "ticker": row.ticker,
        "taxonomy_version_id": row.taxonomy_version_id,
        "sector_code": row.sector_code,
        "industry_group_code": row.industry_group_code,
        "industry_code": row.industry_code,
        "sub_industry_code": row.sub_industry_code,
        "sub_industry_codes": list(row.sub_industry_codes or []),
        "state": row.state,
        "source": row.source,
        "method": row.method,
        "author": row.author,
        "source_as_of": row.source_as_of,
        "confidence": row.confidence,
        "source_sector": row.source_sector,
        "source_industry": row.source_industry,
        "source_sub_industry": row.source_sub_industry,
        "candidates": list(row.candidates or []),
        "evidence": dict(row.evidence or {}),
        "is_current": bool(row.is_current),
        "classified_at": row.classified_at.isoformat() if row.classified_at else None,
        "superseded_at": row.superseded_at.isoformat() if row.superseded_at else None,
        "superseded_reason": row.superseded_reason or "",
        "mapping_caveat": MAPPING_CAVEAT,
    }


def _new_row(
    ticker: str, version_id: int, res: Resolution,
    sector: str | None, industry: str | None, sub_industry: str | None, now: datetime,
) -> CompanyIndustryClassification:
    return CompanyIndustryClassification(
        ticker=ticker,
        taxonomy_version_id=version_id,
        sector_code=res.sector_code,
        industry_group_code=res.industry_group_code,
        industry_code=res.industry_code,
        sub_industry_code=res.sub_industry_code,
        sub_industry_codes=list(res.sub_industry_codes),
        state=res.state,
        source=res.source,
        method=res.method,
        author=res.author,
        source_as_of=res.source_as_of,
        confidence=res.confidence,
        source_sector=sector,
        source_industry=industry,
        source_sub_industry=sub_industry,
        inputs_fingerprint=inputs_fingerprint(sector, industry, sub_industry),
        candidates=list(res.candidates),
        evidence=dict(res.evidence),
        is_current=True,
        classified_at=now,
    )


def _same_outcome(row: CompanyIndustryClassification, res: Resolution) -> bool:
    return (
        row.state == res.state
        and row.source == res.source
        and (row.sector_code, row.industry_group_code, row.industry_code, row.sub_industry_code) == res.codes
        and tuple(row.sub_industry_codes or []) == tuple(res.sub_industry_codes)
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def classify_ticker(
    ticker: str, *, version: VersionInfo | str | int | None = None, force: bool = False,
) -> dict[str, Any] | None:
    """Classify one company (the on-demand hook and the CLI single-ticker path).

    Returns the current row as a dict, or ``None`` when there is no active
    taxonomy or no ``companies`` row — both are "nothing to do", not
    errors, because the request that introduces a symbol must not fail on
    its classification. A current row with an identical outcome and
    fingerprint is left alone (``changed=False``); otherwise the old row is
    superseded and a new one inserted.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    try:
        info = gics_registry.resolve_version(version)
    except gics_registry.TaxonomyNotImported:
        return None
    now = _utcnow()
    with SessionLocal() as db:
        company = db.execute(
            select(Company.ticker, Company.sector, Company.industry, Company.sub_industry)
            .where(Company.ticker == symbol)
        ).first()
        if company is None:
            return None
        _, sector, industry, sub_industry = company
        current = db.execute(
            select(CompanyIndustryClassification).where(
                CompanyIndustryClassification.ticker == symbol,
                CompanyIndustryClassification.taxonomy_version_id == info.id,
                CompanyIndustryClassification.is_current.is_(True),
            ).order_by(CompanyIndustryClassification.id.desc())
        ).scalars().first()
        res = resolve(symbol, sector, industry, sub_industry, version=info)
        fingerprint = inputs_fingerprint(sector, industry, sub_industry)
        if (
            current is not None and not force
            and current.inputs_fingerprint == fingerprint and _same_outcome(current, res)
        ):
            out = row_dict(current)
            out["changed"] = False
            return out
        if current is not None:
            current.is_current = False
            current.superseded_at = now
            current.superseded_reason = (
                "forced" if force else
                "inputs_changed" if current.inputs_fingerprint != fingerprint else "outcome_changed"
            )
        row = _new_row(symbol, info.id, res, sector, industry, sub_industry, now)
        db.add(row)
        db.commit()
        out = row_dict(row)
        out["changed"] = True
        return out


def classify_all(
    *, version: VersionInfo | str | int | None = None, tickers: list[str] | None = None,
    reclassify: bool = True, force: bool = False,
) -> dict[str, Any]:
    """Classify every ``companies`` row (or ``tickers``) against one version.

    Bulk by construction: one read of the companies, one read of the
    current classification rows, pure resolution in memory, then one bulk
    ``UPDATE`` of superseded rows and one bulk ``INSERT`` of new ones.

    Staleness: a current row whose ``inputs_fingerprint`` differs from the
    company's current labels (or from the current alias-map / map
    editions) is flagged ``stale`` in place first — with its previous
    state kept in ``evidence`` — then, when ``reclassify`` is on (the
    default), re-resolved and superseded in the same pass. With
    ``reclassify=False`` only the detection happens, which is what an
    audit wants. The summary reports the counts the loop writes into
    ``record_run``.
    """
    info = gics_registry.resolve_version(version)
    now = _utcnow()
    wanted = {str(t).strip().upper() for t in tickers} if tickers else None
    summary: dict[str, Any] = {
        "taxonomy_version": info.version_key,
        "classified": 0,
        "inserted": 0,
        "reclassified": 0,
        "stale_detected": 0,
        "stale_fixed": 0,
        "unchanged": 0,
        "counts": {state: 0 for state in CLASSIFICATION_STATES},
        "sources": {},
        "changed": [],
        "unmapped_labels": [],
    }

    with SessionLocal() as db:
        company_q = select(Company.ticker, Company.sector, Company.industry, Company.sub_industry)
        if wanted:
            company_q = company_q.where(Company.ticker.in_(sorted(wanted)))
        companies = db.execute(company_q).all()
        current_q = select(CompanyIndustryClassification).where(
            CompanyIndustryClassification.taxonomy_version_id == info.id,
            CompanyIndustryClassification.is_current.is_(True),
        )
        if wanted:
            current_q = current_q.where(CompanyIndustryClassification.ticker.in_(sorted(wanted)))
        current_by_ticker: dict[str, CompanyIndustryClassification] = {}
        for row in db.execute(current_q).scalars().all():
            # Two current rows for one ticker cannot happen through this
            # module; keep the newest if it ever does, and supersede the rest.
            prev = current_by_ticker.get(row.ticker)
            if prev is None or row.id > prev.id:
                current_by_ticker[row.ticker] = row

        new_rows: list[CompanyIndustryClassification] = []
        supersede_ids: dict[str, list[int]] = {}
        unmapped: dict[tuple[str, str], int] = {}

        for ticker, sector, industry, sub_industry in companies:
            symbol = str(ticker).upper()
            fingerprint = inputs_fingerprint(sector, industry, sub_industry)
            current = current_by_ticker.get(symbol)
            drifted = current is not None and current.inputs_fingerprint != fingerprint
            if drifted and current.state != STATE_STALE:
                # Flip in place so the audit can see WHY the row is about to
                # be replaced even if re-classification fails below.
                evidence = dict(current.evidence or {})
                evidence["previous_state"] = current.state
                evidence["stale_reason"] = "inputs_changed"
                evidence["stale_detected_at"] = now.isoformat()
                current.evidence = evidence
                current.state = STATE_STALE
                summary["stale_detected"] += 1
            elif drifted:
                summary["stale_detected"] += 1
            if current is not None and not drifted and not force and not reclassify:
                summary["unchanged"] += 1
                continue
            if current is not None and drifted and not reclassify:
                continue
            res = resolve(symbol, sector, industry, sub_industry, version=info)
            if current is not None and not drifted and not force and _same_outcome(current, res):
                summary["unchanged"] += 1
                continue
            if current is not None:
                reason = "forced" if force else ("inputs_changed" if drifted else "outcome_changed")
                supersede_ids.setdefault(reason, []).append(current.id)
                summary["reclassified"] += 1
                if drifted:
                    summary["stale_fixed"] += 1
                summary["changed"].append({
                    "ticker": symbol,
                    "from": current.industry_group_code,
                    "to": res.industry_group_code,
                    "from_state": current.evidence.get("previous_state", current.state)
                    if current.state == STATE_STALE else current.state,
                    "to_state": res.state,
                })
            else:
                summary["inserted"] += 1
            new_rows.append(_new_row(symbol, info.id, res, sector, industry, sub_industry, now))
            if res.state in (STATE_MISSING, STATE_FALLBACK):
                key = (str(sector or ""), str(industry or ""))
                unmapped[key] = unmapped.get(key, 0) + 1

        for reason, ids in supersede_ids.items():
            db.execute(
                update(CompanyIndustryClassification)
                .where(CompanyIndustryClassification.id.in_(ids))
                .values(is_current=False, superseded_at=now, superseded_reason=reason)
            )
        if new_rows:
            db.add_all(new_rows)
        db.commit()

        summary["classified"] = len(companies)
        # Final tallies from the table, not from the loop above, so the note
        # describes what is actually current after the commit.
        tally_q = (
            select(
                CompanyIndustryClassification.state,
                CompanyIndustryClassification.source,
                func.count(CompanyIndustryClassification.id),
            )
            .where(
                CompanyIndustryClassification.taxonomy_version_id == info.id,
                CompanyIndustryClassification.is_current.is_(True),
            )
        )
        if wanted:
            tally_q = tally_q.where(CompanyIndustryClassification.ticker.in_(sorted(wanted)))
        for state, source, n in db.execute(tally_q.group_by(
            CompanyIndustryClassification.state, CompanyIndustryClassification.source,
        )).all():
            summary["counts"][state] = summary["counts"].get(state, 0) + int(n)
            summary["sources"][source] = summary["sources"].get(source, 0) + int(n)
        if not reclassify:
            # Detection-only: unmapped labels come from the current rows.
            for row in current_by_ticker.values():
                if row.state in (STATE_MISSING, STATE_FALLBACK):
                    key = (str(row.source_sector or ""), str(row.source_industry or ""))
                    unmapped[key] = unmapped.get(key, 0) + 1

    summary["unmapped_labels"] = [
        {"sector": s, "industry": i, "count": n}
        for (s, i), n in sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return summary


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _current_join(version: VersionInfo | None):
    """Current rows of the active version (or ``version``) — the join to
    ``gics_taxonomy_versions.is_active`` keeps this ONE statement, which is
    what lets a memo run ask for its company's group without a second
    round-trip for the active version."""
    q = select(CompanyIndustryClassification).where(
        CompanyIndustryClassification.is_current.is_(True),
    )
    if version is None:
        q = q.join(
            TaxonomyVersion, TaxonomyVersion.id == CompanyIndustryClassification.taxonomy_version_id,
        ).where(TaxonomyVersion.is_active.is_(True))
    else:
        q = q.where(CompanyIndustryClassification.taxonomy_version_id == version.id)
    return q


def current_for(
    tickers: list[str] | tuple[str, ...] | set[str], *, version: VersionInfo | None = None,
) -> dict[str, dict[str, Any]]:
    """``{ticker: row_dict}`` for the tickers that have a current row under
    the active taxonomy — one SELECT. Tickers without a row are absent
    (the caller decides between "missing" and "not in the universe")."""
    symbols = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not symbols:
        return {}
    with SessionLocal() as db:
        rows = db.execute(
            _current_join(version).where(CompanyIndustryClassification.ticker.in_(symbols))
        ).scalars().all()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            prev = out.get(row.ticker)
            if prev is None or row.id > prev["id"]:
                out[row.ticker] = row_dict(row)
        return out


def constituents(
    code: str, *, version: VersionInfo | None = None, active_only: bool = True,
    states: tuple[str, ...] = (STATE_MAPPED, STATE_CONFLICT),
) -> list[str]:
    """Tickers whose current classification rolls up to ``code`` (a group,
    industry or sub-industry code, by prefix on the matching column) —
    one SELECT joined to ``companies`` for the active filter. Only
    ``mapped`` and ``conflict`` rows count: a ``fallback`` row knows its
    sector, not its group, and must not inflate a cohort."""
    key = str(code or "").strip()
    if not key.isdigit() or len(key) not in (4, 6, 8):
        return []
    column = {
        4: CompanyIndustryClassification.industry_group_code,
        6: CompanyIndustryClassification.industry_code,
        8: CompanyIndustryClassification.sub_industry_code,
    }[len(key)]
    q = _current_join(version).where(column == key, CompanyIndustryClassification.state.in_(states))
    if active_only:
        q = q.join(Company, Company.ticker == CompanyIndustryClassification.ticker).where(
            Company.is_active.is_(True),
        )
    with SessionLocal() as db:
        return sorted({r.ticker for r in db.execute(q).scalars().all()})


def constituents_by_group(
    *, version: VersionInfo | None = None, active_only: bool = True,
    states: tuple[str, ...] = (STATE_MAPPED, STATE_CONFLICT),
) -> dict[str, list[str]]:
    """``{industry_group_code: [tickers]}`` for every group with at least
    one constituent — one SELECT, for the analytics and the taxonomy
    endpoint's constituent counts. Counts come from here, never from a
    literal."""
    q = _current_join(version).where(
        CompanyIndustryClassification.industry_group_code.is_not(None),
        CompanyIndustryClassification.state.in_(states),
    )
    if active_only:
        q = q.join(Company, Company.ticker == CompanyIndustryClassification.ticker).where(
            Company.is_active.is_(True),
        )
    out: dict[str, set[str]] = {}
    with SessionLocal() as db:
        for row in db.execute(q).scalars().all():
            out.setdefault(row.industry_group_code, set()).add(row.ticker)
    return {code: sorted(tickers) for code, tickers in sorted(out.items())}


def history(ticker: str, *, version: VersionInfo | None = None) -> list[dict[str, Any]]:
    """Every classification row for ``ticker`` under the active (or given)
    version, oldest first — the audit trail."""
    symbol = str(ticker or "").strip().upper()
    q = select(CompanyIndustryClassification).where(CompanyIndustryClassification.ticker == symbol)
    if version is None:
        q = q.join(
            TaxonomyVersion, TaxonomyVersion.id == CompanyIndustryClassification.taxonomy_version_id,
        ).where(TaxonomyVersion.is_active.is_(True))
    else:
        q = q.where(CompanyIndustryClassification.taxonomy_version_id == version.id)
    with SessionLocal() as db:
        return [row_dict(r) for r in db.execute(q.order_by(CompanyIndustryClassification.id)).scalars().all()]


def audit(*, version: VersionInfo | None = None, limit: int = 200) -> dict[str, Any]:
    """Counts by state and source, the tickers in each non-mapped state,
    and the unmapped provider labels by frequency — what an operator needs
    to extend the alias map. Aggregate first; per-ticker lists are capped."""
    counts = {state: 0 for state in CLASSIFICATION_STATES}
    sources: dict[str, int] = {}
    by_state: dict[str, list[str]] = {s: [] for s in CLASSIFICATION_STATES if s != STATE_MAPPED}
    unmapped: dict[tuple[str, str], int] = {}
    with SessionLocal() as db:
        rows = db.execute(_current_join(version)).scalars().all()
        for row in rows:
            counts[row.state] = counts.get(row.state, 0) + 1
            sources[row.source] = sources.get(row.source, 0) + 1
            if row.state != STATE_MAPPED and len(by_state.setdefault(row.state, [])) < limit:
                by_state[row.state].append(row.ticker)
            if row.state in (STATE_MISSING, STATE_FALLBACK):
                key = (str(row.source_sector or ""), str(row.source_industry or ""))
                unmapped[key] = unmapped.get(key, 0) + 1
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "sources": sources,
        "tickers_by_state": {s: sorted(t) for s, t in by_state.items()},
        "unmapped_labels": [
            {"sector": s, "industry": i, "count": n}
            for (s, i), n in sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "alias_map": {"version": alias_map().version, "as_of": alias_map().as_of},
        "research_map_as_of": industry_knowledge.security_reference_as_of(),
        "mapping_caveat": MAPPING_CAVEAT,
    }
