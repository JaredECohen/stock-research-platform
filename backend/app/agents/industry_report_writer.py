"""FEAT-003 — the weekly Industry Analysis report writer.

``write_report`` turns one group's computed statistics, the cross-industry
snapshot, the prior edition and the period's events into the versioned
report payload (``industry_reports.payload``, schema 1). Thirteen sections,
each ``{facts, interpretation}``:

    overview, drivers, kpis, performance, companies, statistics, themes,
    cross_industry, outlook, risks, what_changed, sources, metadata

``facts`` are built HERE, by the server, from the inputs — the LLM never
writes into them. ``interpretation`` is the analyst layer: at most
``INDUSTRY_REPORT_MAX_LLM_CALLS`` bounded ``chat_json`` calls under the
group's analyst prompt, or — with no LLM, an open breaker, or
``deterministic=True`` on a job's final attempt — templates over the
mandate and the facts, labelled ``analyst_narrative: llm_unavailable`` /
``deterministic`` and listed in ``degraded`` so a reader knows which
edition they hold. Degradation is per section as well as per edition:
``narrative_by_section`` says who wrote each one, because a call budget
that reaches only some sections still produces templates for the rest.
The drivers section is ordered by the eight-stage
causal order loaded from the knowledge base; the deterministic edition
says ``n/a`` for a stage it cannot support rather than inventing one.

Missing evidence renders as ``n/a`` plus a reason, never as zero. The
themes section reuses ``sector_research_service.aggregate_cohort_filing_themes``
and the weekly sector-digest memory rather than duplicating either. Cost
comes from ``services.llm_metrics.estimate_cost_usd`` aggregated by
``run_id``.

Inputs are duck-typed (``IndustryStatSnapshot`` rows or dicts with the
same keys) so slices 3 and 4 can hand in either; ``_field`` reads both.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import settings
from ..memory import SectorMemory
from ..prompts import load_prompt
from ..services import industry_classification, industry_knowledge
from ..services.industry_group_knowledge import primary_sources, thesis_stages
from ..services.llm_metrics import cost_per_run, estimate_cost_usd
from . import llm, prompts
from .industry_analysts import IndustryAnalyst
from .industry_report_validator import CAUSAL_MARKERS, INTERPRETED_SECTIONS, SECTION_ORDER, _sentences
from .log_safety import log_safely, redact

log = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1
NARRATIVE_LLM = "llm"
NARRATIVE_LLM_UNAVAILABLE = "llm_unavailable"
NARRATIVE_DETERMINISTIC = "deterministic"
HORIZONS: tuple[str, ...] = ("1W", "1M", "QTD", "YTD", "1Y")
_ATLAS_PATH = Path(__file__).resolve().parent.parent / "data" / "industry_knowledge" / "atlas_dependencies.json"
_MAX_PER_TICKER_ROWS = 40
_MAX_EVENTS = 20


def _utcnow() -> datetime:
    """Clock seam — tests pin it."""
    return datetime.utcnow()


@dataclass
class WriterResult:
    payload: dict[str, Any]
    generation: dict[str, Any]
    degraded: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --- input access --------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or key, so ORM rows and plain dicts are both accepted."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _na(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


@lru_cache(maxsize=1)
def _atlas() -> dict[str, Any]:
    """The checked-in dependency graph; `{}` (and a degraded entry) when the
    file is absent rather than an exception — the report can stand without
    spillover context, and it says so in the section."""
    try:
        with _ATLAS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# --- facts ---------------------------------------------------------------------


def _ranked(mandate: Any, name: str, limit: int = 8) -> list[dict[str, Any]]:
    return [r.as_dict() for r in mandate.lists.get(name, ())[:limit]]


def _overview_facts(analyst: IndustryAnalyst, stats: Any, sample: dict[str, Any],
                    constituents: list[str]) -> dict[str, Any]:
    m = analyst.mandate
    # The stats row's own count when it has one, otherwise the membership
    # list's length — the two sections must never disagree about how many
    # companies are in the group.
    n_constituents = sample.get("n_constituents")
    if not isinstance(n_constituents, int) or isinstance(n_constituents, bool):
        n_constituents = len(constituents) or None
    return {
        "code": analyst.code,
        "name": analyst.name,
        "sector": {"code": analyst.sector_code, "name": analyst.sector_name},
        "taxonomy_version": analyst.taxonomy_version_key,
        "boundaries": [{"code": i["code"], "name": i["name"]} for i in m.industries],
        "sub_industries": [{"code": s.code, "name": s.name, "industry_code": s.industry_code}
                           for s in m.sub_industries],
        "n_constituents": n_constituents,
        "n_with_prices": sample.get("n_with_prices"),
        "period_key": _field(stats, "period_key"),
        "as_of": _iso(_field(stats, "as_of")),
        "research_priority": m.research_priority,
        "research_priority_source": m.research_priority_source,
        "preferred_archetypes": [r.as_dict() for r in m.preferred_archetypes],
        "attribution": m.attribution,
        "mapping_caveat": industry_classification.MAPPING_CAVEAT,
    }


def _drivers_facts(analyst: IndustryAnalyst, snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    m = analyst.mandate
    regime = snapshot_payload.get("regime") if isinstance(snapshot_payload, dict) else None
    macro_regime = None
    if isinstance(regime, dict):
        macro_regime = regime.get("macro_regime")
    elif isinstance(regime, str):
        macro_regime = regime
    relationships = [
        r for r in (_atlas().get("relationships") or [])
        if analyst.code in (r.get("industry_group_codes") or [])
    ]
    return {
        "stage_order": thesis_stages(),
        "economic_engine": _ranked(m, "economic_engine"),
        "capital_cycle": _ranked(m, "capital_cycle_supply_response"),
        "leading_indicators": _ranked(m, "leading_indicators"),
        "cadence": [r.as_dict() for r in m.cadence],
        "sensitivities": {
            "macro_regime": macro_regime if macro_regime else _na("no macro broadcast in the snapshot"),
            "relationships": [
                {"theme": r.get("theme"), "mechanism": r.get("mechanism"), "monitor": r.get("monitor"),
                 "source": r.get("source")}
                for r in relationships[:6]
            ],
        },
    }


def _kpis_facts(analyst: IndustryAnalyst) -> dict[str, Any]:
    m = analyst.mandate
    return {
        "core_kpis": _ranked(m, "core_kpis", 12),
        "leading_indicators": _ranked(m, "leading_indicators", 8),
        "valuation_lenses": _ranked(m, "valuation_lenses", 8),
        "highest_evi_questions": [{"industry_code": c, "question": q} for c, q in m.highest_evi_questions],
        "sub_industry_monitoring": [
            {"code": s.code, "name": s.name, "advantage_test": s.advantage_test, "source_ids": list(s.source_ids)}
            for s in m.sub_industries
        ],
    }


def _performance_facts(stats: Any, payload: dict[str, Any], sample: dict[str, Any],
                       method: dict[str, Any]) -> dict[str, Any]:
    returns = payload.get("returns") if isinstance(payload, dict) else None
    status = payload.get("status") if isinstance(payload, dict) else None
    facts: dict[str, Any] = {
        "as_of": _iso(_field(stats, "as_of")),
        "period_key": _field(stats, "period_key"),
        "horizons": list(method.get("horizons") or HORIZONS),
        "weighting": list(method.get("weighting") or ["equal", "market_cap"]),
        "benchmarks": list(method.get("benchmarks") or []),
        "sample": {k: sample.get(k) for k in ("n_constituents", "n_with_prices", "excluded") if k in sample},
    }
    if not returns:
        facts["status"] = status or "unavailable"
        facts["reason"] = (payload.get("reason") if isinstance(payload, dict) else None) or (
            "insufficient_sample" if status == "insufficient_sample" else "no statistics row for this period"
        )
        facts["returns"] = {h: _na(facts["reason"]) for h in facts["horizons"]}
        return facts
    facts["status"] = status or "ok"
    facts["returns"] = returns
    facts["benchmark_relative"] = payload.get("benchmark_relative") or {}
    return facts


def _companies_facts(analyst: IndustryAnalyst, payload: dict[str, Any], per_ticker: dict[str, Any],
                     events: list[dict[str, Any]], constituents: list[str],
                     sample: dict[str, Any], membership_source: str) -> dict[str, Any]:
    rows = []
    for ticker in sorted(per_ticker)[:_MAX_PER_TICKER_ROWS]:
        row = per_ticker.get(ticker)
        rows.append({"ticker": ticker, **(row if isinstance(row, dict) else {"value": row})})
    # A constituent with no price row is n/a with a reason, not absent.
    excluded_reason = {
        str(e.get("ticker")): str(e.get("reason") or "excluded")
        for e in (sample.get("excluded") or []) if isinstance(e, dict) and e.get("ticker")
    }
    unpriced = [
        {"ticker": t, "reason": excluded_reason.get(t, "no price series for this period")}
        for t in sorted(constituents) if t not in per_ticker
    ]
    return {
        "constituents": sorted(constituents),
        "n_constituents": len(constituents),
        "membership_source": membership_source,
        "n_priced": len(per_ticker),
        "unpriced": unpriced,
        "largest": list(payload.get("largest") or [])[:10],
        "leaders": list(payload.get("leaders") or [])[:5],
        "laggards": list(payload.get("laggards") or [])[:5],
        "per_ticker": rows,
        "per_ticker_truncated": max(0, len(per_ticker) - _MAX_PER_TICKER_ROWS),
        "events": [dict(e) for e in events[:_MAX_EVENTS] if isinstance(e, dict)],
        "checklists": analyst.mandate.as_checklists(),
        "mapping_caveat": industry_classification.MAPPING_CAVEAT,
    }


def _statistics_facts(stats: Any, payload: dict[str, Any], sample: dict[str, Any],
                      method: dict[str, Any]) -> dict[str, Any]:
    keys = ("breadth", "dispersion", "valuation", "fundamentals", "status", "reason")
    return {
        **{k: payload.get(k) for k in keys if k in payload},
        "sample": dict(sample),
        "method": dict(method),
        "inputs_hash": _field(stats, "inputs_hash"),
        "computed_at": _iso(_field(stats, "computed_at")),
        "stats_id": _field(stats, "id"),
    }


def _sector_labels(sector_code: str, sector_name: str) -> list[str]:
    """The provider labels a sector's digest memory may be filed under
    (`SectorMemory` is keyed by `Company.sector`, an FMP label), from the
    alias map — plus the GICS name itself."""
    labels = [sector_name] if sector_name else []
    try:
        aliases = industry_classification.alias_map()
        for label, code in aliases.sectors.items():
            if code == sector_code and label not in labels:
                labels.append(label)
    except Exception:  # pragma: no cover — the alias map is checked in
        pass
    return labels


def _themes_facts(analyst: IndustryAnalyst, constituents: list[str], errors: list[str]) -> dict[str, Any]:
    from ..services.sector_research_service import aggregate_cohort_filing_themes

    filing_themes: list[dict[str, Any]] | dict[str, Any]
    try:
        filing_themes = aggregate_cohort_filing_themes(constituents[:_MAX_PER_TICKER_ROWS]) if constituents else []
    except Exception as exc:
        errors.append(f"themes: filing theme aggregation failed: {redact(exc)}")
        filing_themes = _na("filing theme aggregation failed")
    digests: dict[str, str] = {}
    for label in _sector_labels(analyst.sector_code, analyst.sector_name):
        try:
            excerpt = SectorMemory.for_sector(label).as_prompt_context(max_chars=800)
        except Exception as exc:  # pragma: no cover — memory must never block a report
            errors.append(f"themes: sector digest memory read failed: {redact(exc)}")
            continue
        if excerpt:
            digests[label] = excerpt
    return {
        "cohort_filing_themes": filing_themes,
        "n_cohort": min(len(constituents), _MAX_PER_TICKER_ROWS),
        "sector_digest_memory": digests if digests else _na("no weekly sector digest on file"),
        "sources": ["sector_research_service.aggregate_cohort_filing_themes", "weekly sector digest memory"],
    }


def _cross_industry_facts(analyst: IndustryAnalyst, snapshot_row: Any, snapshot_payload: dict[str, Any],
                          degraded: list[str]) -> dict[str, Any]:
    atlas = _atlas()
    if not atlas:
        degraded.append("cross_industry:atlas_dependencies:missing")
    edges = [
        {k: e.get(k) for k in ("edge_id", "origin", "destination", "transmission", "invalidation",
                               "next_evidence", "status", "source", "industry_group_codes")}
        for e in (atlas.get("edges") or [])
        if analyst.code in (e.get("industry_group_codes") or [])
    ]
    relationships = [
        {k: r.get(k) for k in ("theme", "mechanism", "monitor", "failure_of_inference", "source",
                               "industry_group_codes")}
        for r in (atlas.get("relationships") or [])
        if analyst.code in (r.get("industry_group_codes") or [])
    ]
    spillovers = [
        s for s in (snapshot_payload.get("spillovers") or [])
        if isinstance(s, dict) and analyst.code in (s.get("origin_code"), s.get("destination_code"))
    ]
    return {
        "edges": edges,
        "relationships": relationships,
        "spillovers": spillovers,
        "snapshot": {
            "as_of": _iso(_field(snapshot_row, "as_of")),
            "period_key": _field(snapshot_row, "period_key"),
            "id": _field(snapshot_row, "id"),
        } if snapshot_row is not None else _na("no cross-industry snapshot for this period"),
        "atlas_snapshot_date": atlas.get("atlas_snapshot_date"),
    }


def _outlook_facts(payload: dict[str, Any], snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    valuation = payload.get("valuation") if isinstance(payload, dict) else None
    price_implied: dict[str, Any]
    if isinstance(valuation, dict) and valuation:
        price_implied = {"value": valuation, "basis": "cohort valuation distribution (observed multiples)"}
    else:
        price_implied = _na("no valuation distribution for this period")
    regime = snapshot_payload.get("regime") if isinstance(snapshot_payload, dict) else None
    return {
        "expectations_ledger": {
            "reported_consensus": _na("no licensed consensus tape"),
            "management_guidance": _na("guidance is company-level; not aggregated at group level"),
            "price_implied": price_implied,
            "our_forecast": _na("scenarios are analyst interpretation, recorded there, not a forecast"),
        },
        "macro_regime": regime if regime else _na("no macro broadcast in the snapshot"),
        "scenario_policy": "Scenarios are labelled scenarios, not recommendations.",
    }


def _risks_facts(analyst: IndustryAnalyst) -> dict[str, Any]:
    m = analyst.mandate
    return {
        "common_failure_modes": _ranked(m, "common_failure_modes", 10),
        "accounting_data_traps": _ranked(m, "accounting_data_traps", 10),
        "typical_moats": _ranked(m, "typical_moats", 8),
    }


def _returns_flat(perf: dict[str, Any]) -> dict[str, float]:
    """`returns.<horizon>.<weighting>` → value for the delta arithmetic."""
    out: dict[str, float] = {}
    returns = perf.get("returns") if isinstance(perf, dict) else None
    if not isinstance(returns, dict):
        return out
    for horizon, cell in returns.items():
        if not isinstance(cell, dict):
            continue
        for key, value in cell.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"returns.{horizon}.{key}"] = float(value)
    return out


def _what_changed_facts(prior_report: Any, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if prior_report is None:
        return {
            "prior_version": None,
            "prior_as_of": None,
            "facts_delta": _na("first edition; no prior report to compare"),
            "constituents": {"added": [], "removed": []},
        }
    prior_payload = _field(prior_report, "payload", {}) or {}
    prior_sections = prior_payload.get("sections") or {}
    prior_perf = (prior_sections.get("performance") or {}).get("facts") or {}
    prior_companies = (prior_sections.get("companies") or {}).get("facts") or {}
    before = _returns_flat(prior_perf)
    after = _returns_flat(current["performance"])
    delta = {
        key: {"from": before.get(key), "to": after.get(key),
              "change": (round(after[key] - before[key], 6) if key in before and key in after else None)}
        for key in sorted(set(before) | set(after))
    }
    prev_c = set(prior_companies.get("constituents") or [])
    now_c = set(current["companies"].get("constituents") or [])
    return {
        "prior_version": _field(prior_report, "version"),
        "prior_as_of": _iso(_field(prior_report, "as_of")),
        "prior_period_key": _field(prior_report, "period_key"),
        "facts_delta": delta if delta else _na("no comparable return facts on either edition"),
        "constituents": {"added": sorted(now_c - prev_c), "removed": sorted(prev_c - now_c)},
    }


def _sources_facts(analyst: IndustryAnalyst, stats: Any, method: dict[str, Any]) -> dict[str, Any]:
    knowledge = industry_knowledge.load_industry_knowledge()
    source_ids = sorted({sid for s in analyst.mandate.sub_industries for sid in s.source_ids})
    register = {s.get("id"): s for s in industry_knowledge.sources()}
    manifest: list[dict[str, Any]] = [
        {"kind": "knowledge_base", "ref": str(knowledge.get("generated_from", "")),
         "as_of": str(knowledge.get("taxonomy_version", "")), "provider": "encyclopedia",
         "sha256": str(knowledge.get("source_sha256", ""))},
        {"kind": "universe_map", "ref": str(knowledge.get("map_generated_from", "")),
         "as_of": str(knowledge.get("map_as_of", "")), "provider": "research_map",
         "sha256": str(knowledge.get("map_source_sha256", ""))},
        {"kind": "statistics", "ref": f"industry_stats:{_field(stats, 'id')}",
         "as_of": _iso(_field(stats, "as_of")), "provider": "industry_analytics",
         "inputs_hash": _field(stats, "inputs_hash")},
    ]
    for b in method.get("benchmarks") or []:
        manifest.append({"kind": "benchmark", "ref": (b.get("id") if isinstance(b, dict) else str(b)),
                         "as_of": _iso(_field(stats, "as_of")), "provider": "data_catalog"})
    atlas = _atlas()
    if atlas:
        manifest.append({"kind": "atlas_dependencies", "ref": str(atlas.get("generated_from", "")),
                         "as_of": str(atlas.get("atlas_snapshot_date", "")), "provider": "atlas"})
    return {
        "manifest": manifest,
        "map_sources": [
            {k: register[sid].get(k) for k in ("id", "title", "url", "purpose", "accessed")}
            for sid in source_ids if sid in register
        ],
        "primary_sources": primary_sources(),
        "knowledge_version": str(knowledge.get("taxonomy_version", "")),
        "map_as_of": str(knowledge.get("map_as_of", "")),
        "attribution": analyst.mandate.attribution,
        "security_reference_caveat": industry_knowledge.SECURITY_REFERENCE_CAVEAT,
    }


def _metadata_facts(analyst: IndustryAnalyst, stats: Any, *, run_id: str, mode: str) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _utcnow().isoformat(),
        "run_id": run_id,
        "generation_mode": mode,
        "taxonomy_version": analyst.taxonomy_version_key,
        "taxonomy_version_id": analyst.taxonomy_version_id,
        "knowledge_version": analyst.mandate.knowledge_version,
        "analyst": analyst.display_name,
        "period_key": _field(stats, "period_key"),
        "as_of": _iso(_field(stats, "as_of")),
        "section_order": list(SECTION_ORDER),
    }


def build_facts(
    analyst: IndustryAnalyst, stats: Any, snapshot_row: Any, prior_report: Any,
    events: list[dict[str, Any]], *, run_id: str, mode: str,
    degraded: list[str], errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Every section's `facts`, server-computed. Exposed so a job can hand
    the same dict to the validator after the writer returns."""
    payload = _field(stats, "payload", {}) or {}
    sample = _field(stats, "sample", {}) or {}
    method = _field(stats, "method", {}) or {}
    per_ticker = _field(stats, "per_ticker", {}) or {}
    snapshot_payload = _field(snapshot_row, "payload", {}) or {}
    if stats is None:
        degraded.append("statistics:missing")
    # Membership is the classification's constituent list, NEVER the priced
    # subset: a name that merely lost price coverage has not left the group,
    # and reporting it as "removed" would turn missing price evidence into a
    # membership fact (and make the report state two constituent counts).
    # The priced names are a last resort, labelled as such.
    constituents = [str(t) for t in (sample.get("tickers") or [])]
    membership_source = "statistics.sample.tickers"
    if not constituents:
        try:
            constituents = industry_classification.constituents(analyst.code)
            membership_source = "industry_classification.constituents"
        except Exception as exc:
            errors.append(f"companies: constituent lookup failed: {redact(exc)}")
            constituents = []
    if not constituents and per_ticker:
        constituents = sorted(per_ticker)
        membership_source = "statistics.per_ticker (priced names only; membership list unavailable)"

    facts: dict[str, dict[str, Any]] = {
        "overview": _overview_facts(analyst, stats, sample, constituents),
        "drivers": _drivers_facts(analyst, snapshot_payload),
        "kpis": _kpis_facts(analyst),
        "performance": _performance_facts(stats, payload, sample, method),
        "companies": _companies_facts(analyst, payload, per_ticker, events or [], constituents,
                                      sample, membership_source),
        "statistics": _statistics_facts(stats, payload, sample, method),
        "themes": _themes_facts(analyst, constituents, errors),
        "cross_industry": _cross_industry_facts(analyst, snapshot_row, snapshot_payload, degraded),
        "outlook": _outlook_facts(payload, snapshot_payload),
        "risks": _risks_facts(analyst),
    }
    facts["what_changed"] = _what_changed_facts(prior_report, facts)
    facts["sources"] = _sources_facts(analyst, stats, method)
    facts["metadata"] = _metadata_facts(analyst, stats, run_id=run_id, mode=mode)
    if facts["performance"].get("status") not in (None, "ok"):
        degraded.append(f"performance:{facts['performance']['status']}")
    return facts


# --- deterministic interpretation ----------------------------------------------


def _claim(text: str, kind: str, basis: list[str], falsifier: str = "") -> dict[str, Any]:
    return {"type": kind, "text": text, "basis": basis, "falsifier": falsifier}


def _register_causal(text: str, claims: list[dict[str, Any]], basis: list[str]) -> None:
    """Every sentence in `text` with a causal marker becomes a registered
    causal_inference claim (basis = where it came from) so the deterministic
    edition satisfies the same gate the LLM edition must."""
    for sentence in _sentences(text):
        low = f" {sentence.lower()} "
        if any(f" {m} " in low or f" {m}," in low for m in CAUSAL_MARKERS):
            claims.append(_claim(
                sentence, "causal_inference", basis,
                "n/a: mandate-level mechanism quoted from the knowledge base; no dated falsifier in this edition",
            ))


def _horizon_reason(cell: Any) -> str:
    """Why a horizon carries no return — never a bare "n/a".

    The statistics payload already answers this for a horizon it computed
    and could not anchor: `{"value": null, "reason": "history_window"}`.
    The case that used to print "n/a (n/a)" is the OTHER one — a horizon
    `method.horizons` declares that the payload never carried at all,
    where the absence itself is the reason and the reader is entitled to
    be told which of the two it is looking at. A cell that is present but
    silent about why is reported as exactly that, so a payload that stops
    recording reasons shows up as a defect instead of as "n/a".
    """
    if not isinstance(cell, dict):
        return "declared in method.horizons, absent from the statistics returns payload"
    reason = str(cell.get("reason") or "").strip()
    return reason or "no return and no reason recorded on the statistics row"


def _fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def _items_text(items: list[dict[str, Any]], limit: int = 4) -> str:
    return "; ".join(f"{i['text']} [{','.join(i.get('industry_codes') or [])}]" for i in items[:limit])


def _deterministic_interpretation(facts: dict[str, dict[str, Any]], analyst: IndustryAnalyst,
                                  mode: str) -> dict[str, dict[str, Any]]:
    m = analyst.mandate
    ov, dr, kp, pf, co, th, ci, ou, rk, wc = (facts[s] for s in (
        "overview", "drivers", "kpis", "performance", "companies", "themes", "cross_industry",
        "outlook", "risks", "what_changed",
    ))
    label = ("Deterministic edition (no analyst narrative)" if mode == NARRATIVE_LLM_UNAVAILABLE
             else "Deterministic edition (deterministic mode)")
    out: dict[str, dict[str, Any]] = {}

    # overview
    claims: list[dict[str, Any]] = []
    n = ov.get("n_constituents")
    text = (
        f"{label}: {analyst.code} {analyst.name} spans {len(ov['boundaries'])} industries and "
        f"{len(ov['sub_industries'])} sub-industries in sector {analyst.sector_code} {analyst.sector_name}. "
        + (f"Constituents in the sample: {n}." if isinstance(n, int) else "Constituent count: n/a (no statistics row).")
        + (f" Research priority {m.research_priority}/5 (set by {m.research_priority_source})."
           if m.research_priority is not None else "")
    )
    claims.append(_claim(text, "observed_fact", ["overview.boundaries", "overview.n_constituents"]))
    out["overview"] = {"text": text, "claims": claims}

    # drivers — the eight stages, in order, honest about what is not asserted
    claims = []
    stages: list[dict[str, str]] = []
    engine = dr.get("economic_engine") or []
    capital = dr.get("capital_cycle") or []
    leading = dr.get("leading_indicators") or []
    regime = dr.get("sensitivities", {}).get("macro_regime")
    regime_text = regime if isinstance(regime, str) else "n/a (no macro broadcast in the snapshot)"
    stage_text = {
        "world_change": (
            "n/a: this edition asserts no dated external change; the observed macro regime is "
            f"{regime_text}, and the leading indicators to watch are {_items_text(leading, 3) or 'n/a'}."
        ),
        "industry_consequences": "Mandate economic engine: " + (_items_text(engine, 3) or "n/a") + ".",
        "durable_company_advantage": "Typical moats per the mandate: " + (_items_text(rk.get("typical_moats") or [], 3) or "n/a") + ".",
        "value_capture": "Capital cycle / supply response per the mandate: " + (_items_text(capital, 3) or "n/a") + ".",
        "reinvestment_runway": "n/a: reinvestment runway is company-level evidence; not asserted at group level in this edition.",
        "long_term_per_share_compounding": "n/a: per-share compounding is not asserted without company-level cash reconciliation.",
        "financial_evidence_falsification": "KPIs that test the chain: " + (_items_text(kp.get("core_kpis") or [], 4) or "n/a") + ".",
        "valuation": "Valuation lenses per the mandate: " + (_items_text(kp.get("valuation_lenses") or [], 3) or "n/a") + ".",
    }
    for s in dr.get("stage_order") or []:
        txt = stage_text.get(s["id"], "n/a: stage not covered in this edition")
        stages.append({"id": s["id"], "text": txt})
        _register_causal(txt, claims, [f"drivers.{s['id']}", f"mandate:{analyst.code}"])
    text = " ".join(f"{i + 1}. {s['text']}" for i, s in enumerate(stages))
    claims.insert(0, _claim(
        "Stage texts quote the group mandate; nothing here is a company-level thesis.",
        "observed_fact", ["drivers.stage_order", "drivers.economic_engine"],
    ))
    out["drivers"] = {"text": text, "stages": stages, "claims": claims}

    # kpis
    claims = []
    text = (
        "KPIs the mandate tests first: " + (_items_text(kp.get("core_kpis") or [], 5) or "n/a") + ". "
        "Leading indicators: " + (_items_text(kp.get("leading_indicators") or [], 4) or "n/a") + ". "
        + ("Highest-EVI questions: " + " | ".join(
            f"[{q['industry_code']}] {q['question']}" for q in (kp.get("highest_evi_questions") or [])[:2]
        ) + "." if kp.get("highest_evi_questions") else "")
    )
    _register_causal(text, claims, ["kpis.core_kpis", f"mandate:{analyst.code}"])
    claims.append(_claim(text, "observed_fact", ["kpis.core_kpis", "kpis.leading_indicators"]))
    out["kpis"] = {"text": text, "claims": claims}

    # performance
    claims = []
    if pf.get("status") == "ok" and isinstance(pf.get("returns"), dict):
        bits = []
        for h in pf.get("horizons") or []:
            cell = pf["returns"].get(h)
            if isinstance(cell, dict) and isinstance(cell.get("equal_weight"), (int, float)):
                bits.append(f"{h} equal-weight {_fmt_pct(cell['equal_weight'])} (n={cell.get('n', 'n/a')})")
            else:
                bits.append(f"{h} n/a ({_horizon_reason(cell)})")
        text = "Observed group returns: " + "; ".join(bits) + ". Benchmarks: " + (
            ", ".join(b.get("id") if isinstance(b, dict) else str(b) for b in pf.get("benchmarks") or []) or "n/a"
        ) + "."
    else:
        text = (
            f"Performance: {pf.get('status', 'unavailable')} — {pf.get('reason', 'n/a')}. "
            "No return figure is quoted for this period."
        )
    claims.append(_claim(text, "observed_fact", ["performance.returns", "performance.sample"]))
    out["performance"] = {"text": text, "claims": claims}

    # companies
    claims = []
    n_c = co.get("n_constituents", 0)
    unpriced = co.get("unpriced") or []
    # Price coverage is reported as coverage, never as membership: a name
    # without a price series is still a constituent, and says why it has no
    # figure this period.
    coverage = (
        f"Price coverage: {co.get('n_priced', 0)} of {n_c} constituents have a price series this period"
        + ("; all constituents priced. " if not unpriced else
           "; without prices: " + ", ".join(f"{u['ticker']} ({u['reason']})" for u in unpriced[:8])
           + (" (further names in companies.unpriced). " if len(unpriced) > 8 else ". "))
    )
    text = (
        f"Constituents on file: {n_c}, from {co.get('membership_source')} (mapping derived from provider "
        "classification, not licensed GICS security assignments). "
        + coverage
        + (f"Largest: {', '.join(str(x.get('ticker', x)) if isinstance(x, dict) else str(x) for x in co['largest'][:5])}. "
           if co.get("largest") else "Largest / leaders / laggards: n/a (no statistics row). ")
        + f"Events in window: {len(co.get('events') or [])}."
    )
    claims.append(_claim(text, "observed_fact", ["companies.constituents", "companies.unpriced",
                                                 "companies.largest", "companies.events"]))
    out["companies"] = {"text": text, "claims": claims}

    # themes
    claims = []
    themes = th.get("cohort_filing_themes")
    if isinstance(themes, list) and themes:
        text = "Cohort filing themes (mentions across the cohort): " + ", ".join(
            f"{t.get('theme')} ({t.get('cohort_mentions')})" for t in themes[:5]
        ) + "."
    elif isinstance(themes, dict):
        text = f"Cohort filing themes: n/a ({themes.get('reason')})."
    else:
        text = "Cohort filing themes: n/a (no filings on file for the cohort)."
    digest = th.get("sector_digest_memory")
    text += (" Weekly sector digest memory: on file for " + ", ".join(sorted(digest)) + "."
             if isinstance(digest, dict) and "reason" not in digest else
             " Weekly sector digest memory: n/a (none on file).")
    claims.append(_claim(text, "observed_fact", ["themes.cohort_filing_themes", "themes.sector_digest_memory"]))
    out["themes"] = {"text": text, "claims": claims}

    # cross_industry
    claims = []
    edges = ci.get("edges") or []
    rels = ci.get("relationships") or []
    spill = ci.get("spillovers") or []
    text = (
        f"Dependency context: {len(edges)} Atlas edges and {len(rels)} map relationships touch this group"
        + (f" (e.g. {edges[0].get('origin')} → {edges[0].get('destination')} [{edges[0].get('source')}])" if edges else "")
        + f"; {len(spill)} spillover signals in the snapshot"
        + (" (n/a: no cross-industry snapshot for this period)" if isinstance(ci.get("snapshot"), dict)
           and "reason" in ci["snapshot"] else "") + ". Edges are analyst causal hypotheses, not estimated correlations."
    )
    _register_causal(text, claims, ["cross_industry.edges"])
    claims.append(_claim(text, "observed_fact", ["cross_industry.edges", "cross_industry.relationships",
                                                  "cross_industry.spillovers"]))
    out["cross_industry"] = {"text": text, "claims": claims}

    # outlook — scenarios as scenarios
    claims = []
    ledger = ou.get("expectations_ledger") or {}
    text = (
        "Expectations ledger: reported consensus n/a (" + str(ledger.get("reported_consensus", {}).get("reason")) + "); "
        "management guidance n/a (" + str(ledger.get("management_guidance", {}).get("reason")) + "); "
        + ("price-implied: cohort valuation distribution on file; " if "value" in ledger.get("price_implied", {})
           and ledger["price_implied"].get("value") else "price-implied n/a; ")
        + "our forecast n/a (deterministic edition). Scenarios below are mandate templates, not forecasts."
    )
    compounder = _items_text(m.as_checklists()["compounder"], 2) or "n/a"
    inflection = _items_text(m.as_checklists()["inflection"], 2) or "n/a"
    scenarios = {
        "base": {"text": "Base scenario: the group's KPIs track the mandate's cadence with no regime change asserted.",
                 "falsifiers": ["A dated break in the mandate's leading indicators (n/a in this edition)."]},
        "bull": {"text": f"Bull scenario (compounder mandate): {compounder}.",
                 "falsifiers": [f"Failure mode observed: {r['text']}" for r in (rk.get('common_failure_modes') or [])[:1]] or ["n/a"]},
        "bear": {"text": f"Bear scenario (inflection mandate breaks): {inflection} fails to materialise.",
                 "falsifiers": ["Leading indicators turn before the KPI does (n/a: not dated in this edition)."]},
    }
    for sc in scenarios.values():
        _register_causal(sc["text"], claims, ["outlook.expectations_ledger", f"mandate:{analyst.code}"])
    claims.append(_claim(text, "observed_fact", ["outlook.expectations_ledger"]))
    claims.append(_claim("Scenarios are mandate templates, not forecasts.", "forecast_assumption",
                         ["outlook.scenario_policy"], "n/a: template scenario carries no dated forecast"))
    out["outlook"] = {"text": text, "claims": claims, "scenarios": scenarios}

    # risks
    claims = []
    text = (
        "Common failure modes per the mandate: " + (_items_text(rk.get("common_failure_modes") or [], 4) or "n/a") + ". "
        "Accounting and data traps: " + (_items_text(rk.get("accounting_data_traps") or [], 4) or "n/a") + "."
    )
    _register_causal(text, claims, ["risks.common_failure_modes", f"mandate:{analyst.code}"])
    claims.append(_claim(text, "observed_fact", ["risks.common_failure_modes", "risks.accounting_data_traps"]))
    out["risks"] = {"text": text, "claims": claims}

    # what_changed
    claims = []
    delta = wc.get("facts_delta")
    if isinstance(delta, dict) and "reason" in delta and delta.get("value") is None:
        text = f"What changed: n/a ({delta['reason']})."
    else:
        moved = [k for k, v in (delta or {}).items() if isinstance(v, dict) and v.get("change") not in (None, 0)]
        text = (
            f"Versus edition {wc.get('prior_version')} (as of {wc.get('prior_as_of')}): "
            f"{len(moved)} return facts moved; constituents added {len(wc['constituents']['added'])}, "
            f"removed {len(wc['constituents']['removed'])}."
        )
    claims.append(_claim(text, "observed_fact", ["what_changed.facts_delta", "what_changed.constituents"]))
    out["what_changed"] = {"text": text, "claims": claims}
    return out


# --- LLM interpretation ----------------------------------------------------------


def _coerce_section(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"].strip():
        return None
    out: dict[str, Any] = {"text": raw["text"].strip()}
    claims = []
    for c in raw.get("claims") or []:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            claims.append({
                "type": str(c.get("type") or "observed_fact"),
                "text": c["text"],
                "basis": [str(b) for b in (c.get("basis") or [])],
                "falsifier": str(c.get("falsifier") or ""),
            })
    out["claims"] = claims
    if isinstance(raw.get("stages"), list):
        out["stages"] = [
            {"id": str(s.get("id", "")), "text": str(s.get("text", ""))}
            for s in raw["stages"] if isinstance(s, dict)
        ]
    if isinstance(raw.get("scenarios"), dict):
        out["scenarios"] = {
            str(k): {"text": str(v.get("text", "")), "falsifiers": [str(f) for f in (v.get("falsifiers") or [])]}
            for k, v in raw["scenarios"].items() if isinstance(v, dict)
        }
    return out


def _llm_call(analyst: IndustryAnalyst, facts: dict[str, dict[str, Any]], sections: tuple[str, ...],
              *, run_id: str, generation: dict[str, Any]) -> dict[str, Any] | None:
    """One bounded chat_json call for `sections`; records tokens and cost."""
    report_rules = load_prompt("industry_report") or ""
    system = analyst.system_prompt() + ("\n\n" + report_rules if report_rules else "")
    prompt = (
        f"Sections to interpret now: {', '.join(sections)}.\n"
        "Facts payload (observed; do not restate numbers that are not here):\n"
        + json.dumps({s: facts[s] for s in sections}, default=str)[: settings.max_agent_context_chars]
    )
    started = time.monotonic()
    with llm.llm_call_context(agent_name=analyst.display_name, run_id=run_id, route="cheap"):
        out = llm.chat_json(prompt, system=system, route="cheap", model=llm.resolve_role_model("sector"),
                            max_tokens=3200)
    generation["llm_calls"] += 1
    generation["latency_ms"] += int((time.monotonic() - started) * 1000)
    usage = llm.last_usage() or {}
    if usage:
        generation["provider"] = usage.get("provider") or generation["provider"]
        generation["model"] = usage.get("model") or generation["model"]
        generation["prompt_tokens"] += int(usage.get("input_tokens") or 0)
        generation["completion_tokens"] += int(usage.get("output_tokens") or 0)
        generation["cost_usd"] = round(generation["cost_usd"] + estimate_cost_usd(
            str(usage.get("provider") or ""), str(usage.get("model") or ""),
            int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
        ), 6)
    return out if isinstance(out, dict) else None


# --- entry point -----------------------------------------------------------------


def write_report(
    analyst: IndustryAnalyst, stats: Any, snapshot_row: Any, prior_report: Any,
    events: list[dict[str, Any]] | None, *, run_id: str, deterministic: bool = False,
) -> WriterResult:
    """Build one edition. Never raises on the interpretation layer: a failed
    or absent LLM degrades to the deterministic edition and says so."""
    degraded: list[str] = []
    errors: list[str] = []
    generation: dict[str, Any] = {
        "provider": settings.active_llm_provider, "model": "", "route": "cheap",
        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
        "latency_ms": 0, "run_id": run_id, "generation_mode": "",
        "max_llm_calls": int(settings.industry_report_max_llm_calls),
        # Which sections the call budget planned to send to the analyst at
        # all; empty with no LLM. The rest are templates by construction.
        "llm_planned_sections": [],
    }
    if deterministic:
        mode = NARRATIVE_DETERMINISTIC
    elif not settings.has_llm or llm._demo_only():
        mode = NARRATIVE_LLM_UNAVAILABLE
    else:
        mode = NARRATIVE_LLM

    facts = build_facts(analyst, stats, snapshot_row, prior_report, events or [],
                        run_id=run_id, mode=mode, degraded=degraded, errors=errors)

    interpretation: dict[str, dict[str, Any]] = {}
    if mode == NARRATIVE_LLM:
        # Two bounded calls at the default budget. A budget of 1 drops the
        # second batch — the sections it would have written then come from
        # the deterministic templates and are listed in `degraded` below,
        # never passed off as the analyst's.
        plan = (
            tuple(s for s in INTERPRETED_SECTIONS if s not in ("outlook", "what_changed")),
            ("outlook", "what_changed"),
        )[: max(1, int(settings.industry_report_max_llm_calls))]
        generation["llm_planned_sections"] = sorted({s for batch in plan for s in batch})
        for sections in plan:
            try:
                out = _llm_call(analyst, facts, sections, run_id=run_id, generation=generation)
            except Exception as exc:
                log_safely(log, f"industry report LLM call failed for {analyst.code}", exc)
                errors.append(f"llm: {redact(exc)}")
                out = None
            for name in sections:
                section = _coerce_section((out or {}).get(name))
                if section is not None:
                    interpretation[name] = section
        if not interpretation:
            mode = NARRATIVE_LLM_UNAVAILABLE
        else:
            # Every section the analyst did not write falls back to a
            # template — whether the model returned nothing for it or the
            # call budget (`INDUSTRY_REPORT_MAX_LLM_CALLS`) never reached
            # it. Both are the same thing to a reader, and neither may be
            # published silently under `analyst_narrative: "llm"`.
            degraded.extend(f"analyst_narrative:{name}:deterministic"
                            for name in INTERPRETED_SECTIONS if name not in interpretation)

    if mode != NARRATIVE_LLM:
        degraded.append(f"analyst_narrative:{mode}" if mode == NARRATIVE_LLM_UNAVAILABLE
                        else "analyst_narrative:deterministic_mode")
    written_by_llm = set(interpretation)
    fallback = _deterministic_interpretation(facts, analyst, mode)
    for name in INTERPRETED_SECTIONS:
        interpretation.setdefault(name, fallback[name])

    facts["metadata"]["generation_mode"] = mode
    generation["generation_mode"] = mode
    if run_id and generation["llm_calls"]:
        try:
            run_cost = cost_per_run(run_id)
            generation["cost_usd_run"] = run_cost.get("cost_usd_total", 0.0)
            generation["llm_calls_run"] = run_cost.get("n_calls", 0)
        except Exception as exc:  # pragma: no cover — telemetry must not fail a report
            errors.append(f"cost aggregation: {redact(exc)}")

    sections = {
        name: {"facts": facts[name], "interpretation": interpretation.get(name)}
        for name in SECTION_ORDER
    }
    payload = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "section_order": list(SECTION_ORDER),
        "sections": sections,
        "analyst_narrative": mode,
        # Per section, who wrote the interpretation. `analyst_narrative`
        # alone cannot say "llm" for an edition in which the call budget
        # only reached some of the sections.
        "narrative_by_section": {
            name: (NARRATIVE_LLM if name in written_by_llm else NARRATIVE_DETERMINISTIC)
            for name in INTERPRETED_SECTIONS
        },
        "disclaimer": prompts.DISCLAIMER,
        "attribution": analyst.mandate.attribution,
        "mapping_caveat": industry_classification.MAPPING_CAVEAT,
    }
    return WriterResult(payload=payload, generation=generation, degraded=degraded, errors=errors)
