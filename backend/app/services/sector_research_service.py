"""Sector research service — does the heavy lifting that makes the sector
analyst a *researcher* and not a rubric look-up.

Given a target ticker, this builds the sub-industry cohort, computes
distributional stats per KPI, places the target within those distributions,
detects multi-year sector trends, and aggregates filing themes across the
cohort. The sector agent consumes this output to write a grounded narrative.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from .data_service import get_data_service
from .filings_service import get_filings
from .fundamentals_service import get_full_financials


_SECTOR_CONFIG_CACHE: Optional[Dict[str, Dict]] = None


def _sector_config() -> Dict[str, Dict]:
    global _SECTOR_CONFIG_CACHE
    if _SECTOR_CONFIG_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "data" / "sector_configs.json"
        with open(path) as f:
            _SECTOR_CONFIG_CACHE = json.load(f)
    return _SECTOR_CONFIG_CACHE


def _resolve_sector_block(sector: str) -> Dict:
    cfg = _sector_config()
    if sector in cfg:
        return cfg[sector]
    for k, v in cfg.items():
        if k.lower() in sector.lower() or sector.lower() in k.lower():
            return v
    return next(iter(cfg.values()))


def _resolve_subindustry_overrides(sector_block: Dict, industry: str) -> Dict:
    overrides = sector_block.get("subindustry_overrides", {}) or {}
    if industry in overrides:
        return overrides[industry]
    for k, v in overrides.items():
        if k.lower() in (industry or "").lower() or (industry or "").lower() in k.lower():
            return v
    return {}


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def build_cohort(target_ticker: str) -> List[str]:
    """Return peers in the same sector + (preferably) sub-industry as the target.

    Sub-industry is preferred so semis aren't lumped with software when both
    are 'Technology'. Falls back to sector-level peers if sub-industry has too
    few names.
    """
    ds = get_data_service()
    target = ds.get_company_profile(target_ticker) or {}
    if not target:
        return []
    sector = target.get("sector")
    industry = target.get("industry")
    sub_ind = target.get("sub_industry")

    universe = ds.list_tickers()
    same_sub: List[str] = []
    same_industry: List[str] = []
    same_sector: List[str] = []
    for t in universe:
        if t == target_ticker:
            continue
        p = ds.get_company_profile(t)
        if not p:
            continue
        if sub_ind and p.get("sub_industry") == sub_ind:
            same_sub.append(t)
        if industry and p.get("industry") == industry:
            same_industry.append(t)
        if sector and p.get("sector") == sector:
            same_sector.append(t)

    if len(same_sub) >= 3:
        return same_sub
    if len(same_industry) >= 3:
        return same_industry
    return same_sector


# ---------------------------------------------------------------------------
# Distributional placement
# ---------------------------------------------------------------------------

def _quartile(values: List[float], target: float) -> int:
    """Return 1..4 quartile of `target` within `values`. 4 = top quartile."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return 0
    below = sum(1 for v in clean if v < target)
    pct = below / len(clean)
    if pct < 0.25:
        return 1
    if pct < 0.50:
        return 2
    if pct < 0.75:
        return 3
    return 4


def _distribution(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    s = sorted(clean)
    n = len(s)
    return {
        "n": n,
        "min": s[0],
        "p25": s[max(0, n // 4 - 1)] if n >= 4 else s[0],
        "median": median(clean),
        "p75": s[min(n - 1, (3 * n) // 4)] if n >= 4 else s[-1],
        "max": s[-1],
        "mean": mean(clean),
        "stdev": pstdev(clean) if n > 1 else 0.0,
    }


def compute_kpi_placements(
    target_ratios: Dict, cohort_ratios: List[Dict], kpi_groups: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """For each KPI in the sector framework, compute cohort distribution and
    target's quartile placement.
    """
    placements: Dict[str, Dict] = {}
    for group_name, kpis in kpi_groups.items():
        for kpi in kpis:
            target_val = target_ratios.get(kpi)
            cohort_vals = [r.get(kpi) for r in cohort_ratios]
            dist = _distribution(cohort_vals)
            if not dist:
                continue
            entry = {
                "group": group_name,
                "target": target_val,
                "distribution": dist,
            }
            if target_val is not None:
                entry["quartile"] = _quartile(cohort_vals, target_val)
                # Higher is better for some, lower for valuation
                higher_is_better = group_name not in ("valuation", "capital_intensity")
                if entry["quartile"]:
                    if higher_is_better:
                        entry["interpretation"] = (
                            "top quartile" if entry["quartile"] == 4
                            else "above median" if entry["quartile"] == 3
                            else "below median" if entry["quartile"] == 2
                            else "bottom quartile"
                        )
                    else:
                        entry["interpretation"] = (
                            "richest in cohort" if entry["quartile"] == 4
                            else "above-median multiple" if entry["quartile"] == 3
                            else "below-median multiple" if entry["quartile"] == 2
                            else "cheapest in cohort"
                        )
            placements[kpi] = entry
    return placements


# ---------------------------------------------------------------------------
# Cohort outliers
# ---------------------------------------------------------------------------

def _argmax(rows: List[Tuple[str, Optional[float]]]) -> Optional[str]:
    clean = [(t, v) for t, v in rows if v is not None]
    return max(clean, key=lambda r: r[1])[0] if clean else None


def _argmin(rows: List[Tuple[str, Optional[float]]]) -> Optional[str]:
    clean = [(t, v) for t, v in rows if v is not None]
    return min(clean, key=lambda r: r[1])[0] if clean else None


def detect_outliers(cohort_with_target: List[Dict]) -> Dict[str, Optional[str]]:
    """Return tickers leading the cohort on growth, margin, ROIC, and valuation."""
    if not cohort_with_target:
        return {}
    growth = [(r["ticker"], r["ratios"].get("revenue_growth")) for r in cohort_with_target]
    op_margin = [(r["ticker"], r["ratios"].get("operating_margin")) for r in cohort_with_target]
    roic = [(r["ticker"], r["ratios"].get("ROIC")) for r in cohort_with_target]
    fcf_yield = [(r["ticker"], r["ratios"].get("FCF_yield")) for r in cohort_with_target]
    ev_ebitda = [(r["ticker"], r["ratios"].get("EV_EBITDA")) for r in cohort_with_target]
    return {
        "growth_leader": _argmax(growth),
        "margin_leader": _argmax(op_margin),
        "roic_leader": _argmax(roic),
        "fcf_yield_leader": _argmax(fcf_yield),
        "valuation_cheapest": _argmin(ev_ebitda),
    }


# ---------------------------------------------------------------------------
# Multi-year sector trends
# ---------------------------------------------------------------------------

def _trend_pct(prev: Optional[float], cur: Optional[float]) -> Optional[float]:
    if not prev or prev == 0 or cur is None:
        return None
    return (cur - prev) / abs(prev)


def compute_sector_trends(cohort_fins: List[Dict]) -> Dict[str, Any]:
    """Multi-year cohort-aggregated trends.

    Aggregates across (latest, latest-2) to detect whether margins are expanding
    industry-wide, capex is intensifying, growth is decelerating, etc.
    """
    if not cohort_fins:
        return {}
    op_margins_now: List[float] = []
    op_margins_then: List[float] = []
    capex_now: List[float] = []
    capex_then: List[float] = []
    growth_recent: List[float] = []

    for fin in cohort_fins:
        income = sorted(fin.get("income", []), key=lambda r: r.get("period", ""))
        cash = sorted(fin.get("cash", []), key=lambda r: r.get("period", ""))
        if len(income) < 2:
            continue
        latest, prior = income[-1], income[0]
        if latest.get("revenue") and latest.get("operating_income"):
            op_margins_now.append(latest["operating_income"] / latest["revenue"])
        if prior.get("revenue") and prior.get("operating_income"):
            op_margins_then.append(prior["operating_income"] / prior["revenue"])
        if cash and len(cash) >= 1 and latest.get("revenue"):
            capex_now.append(abs(cash[-1].get("capex", 0)) / latest["revenue"])
        if cash and len(cash) >= 1 and prior.get("revenue"):
            capex_then.append(abs(cash[0].get("capex", 0)) / prior["revenue"])
        prev_rev = income[-2].get("revenue") if len(income) >= 2 else None
        if prev_rev:
            growth_recent.append(_trend_pct(prev_rev, latest.get("revenue")) or 0.0)

    def _safe_mean(xs: List[float]) -> Optional[float]:
        return mean(xs) if xs else None

    op_now = _safe_mean(op_margins_now)
    op_then = _safe_mean(op_margins_then)
    cx_now = _safe_mean(capex_now)
    cx_then = _safe_mean(capex_then)

    return {
        "cohort_op_margin_now": op_now,
        "cohort_op_margin_then": op_then,
        "cohort_op_margin_delta": (op_now - op_then) if (op_now is not None and op_then is not None) else None,
        "cohort_capex_intensity_now": cx_now,
        "cohort_capex_intensity_then": cx_then,
        "cohort_capex_delta": (cx_now - cx_then) if (cx_now is not None and cx_then is not None) else None,
        "cohort_revenue_growth_recent": _safe_mean(growth_recent),
    }


# ---------------------------------------------------------------------------
# Filing-theme aggregation across the cohort
# ---------------------------------------------------------------------------

_RISK_KEYWORDS = [
    ("export restrictions", "Geopolitical export risk"),
    ("china", "China exposure"),
    ("regulator", "Regulatory pressure"),
    ("antitrust", "Antitrust scrutiny"),
    ("competition", "Competitive intensity"),
    ("interest rate", "Rate sensitivity"),
    ("supply chain", "Supply chain risk"),
    ("customer concentration", "Customer concentration"),
    ("talent", "Talent retention"),
    ("cyclical", "Cyclical demand risk"),
    ("inflation", "Input-cost inflation"),
    ("liquidity", "Liquidity risk"),
    ("credit", "Credit risk"),
]


def aggregate_cohort_filing_themes(cohort_tickers: List[str]) -> List[Dict[str, Any]]:
    """Cluster risk-factor language across the cohort into named themes."""
    counter: Counter = Counter()
    n_filings = 0
    for t in cohort_tickers:
        for f in get_filings(t):
            risks = f.get("risk_factors") or []
            n_filings += 1
            text = " ".join(risks).lower() + " " + (f.get("mda", "") or "").lower()
            for needle, label in _RISK_KEYWORDS:
                if needle in text:
                    counter[label] += 1
    if not counter:
        return []
    top = counter.most_common(8)
    return [
        {"theme": label, "cohort_mentions": cnt, "share": round(cnt / max(1, len(cohort_tickers)), 2)}
        for label, cnt in top
    ]


# ---------------------------------------------------------------------------
# Industry structure (light-weight; enhances with HHI when revenue available)
# ---------------------------------------------------------------------------

def industry_structure(cohort_with_target: List[Dict]) -> Dict[str, Any]:
    """Approximate concentration via revenue-share Herfindahl on the cohort."""
    revs = [(r["ticker"], r.get("ratios", {}).get("PS") and (r.get("market_cap") or 0)) for r in cohort_with_target]
    revenues = []
    for r in cohort_with_target:
        income = sorted(r.get("financials", {}).get("income", []) or [], key=lambda x: x.get("period", ""))
        if income:
            revenues.append((r["ticker"], income[-1].get("revenue") or 0))
    total = sum(v for _, v in revenues) or 0.0
    if not total:
        return {}
    shares = [(t, v / total) for t, v in revenues]
    hhi = sum(s * s for _, s in shares)
    shares.sort(key=lambda x: x[1], reverse=True)
    top3 = sum(s for _, s in shares[:3])
    return {
        "hhi_revenue": round(hhi, 4),
        "top_3_revenue_share": round(top3, 4),
        "n_in_cohort": len(shares),
        "concentration_label": (
            "highly concentrated" if hhi > 0.25
            else "moderately concentrated" if hhi > 0.15
            else "fragmented"
        ),
    }


# ---------------------------------------------------------------------------
# Sector regime detection (heuristic; opinionated)
# ---------------------------------------------------------------------------

def _kpi_fingerprint_inputs(entry: Dict[str, Any]) -> Optional[str]:
    """Wave 6B: build a stable, KPI-only fingerprint string for a cohort member.

    Returns a compact `kpi:<ticker>:<rounded values>` token that goes into
    `sources_used` so the sector_warm fingerprint hashes only on the values
    that actually move the cohort math. Compared to the previous "every
    filing invalidates" approach, this keeps the warm snapshot fresh
    through irrelevant peer filings (legal disclosures, shelf registrations,
    dividend declarations).

    Rounding is intentional — micro-jitter from currency translation or
    one-off charges shouldn't trigger a recompute. Round revenue / op
    income / capex to nearest $1M, shares to nearest 1M.
    """
    ticker = (entry or {}).get("ticker")
    fin = (entry or {}).get("financials") or {}
    if not ticker:
        return None
    income_rows = fin.get("income") or []
    cash_rows = fin.get("cash") or []
    if not income_rows:
        return None
    # Most recent period (rows are typically chronological — defend either way).
    income = max(income_rows, key=lambda r: r.get("period", "")) if income_rows else {}
    cash = max(cash_rows, key=lambda r: r.get("period", "")) if cash_rows else {}
    profile = fin.get("profile") or {}

    def _round_m(v: Any, scale: float = 1e6) -> str:
        try:
            return f"{round(float(v) / scale):.0f}"
        except (TypeError, ValueError):
            return "_"

    rev = _round_m(income.get("revenue"))
    op_inc = _round_m(income.get("operating_income"))
    capex = _round_m(cash.get("capex"))
    shares = _round_m(profile.get("shares_outstanding"))
    return f"kpi:{ticker}:rev={rev}:op={op_inc}:cx={capex}:sh={shares}"


def detect_sector_regime(sector: str, trends: Dict[str, Any], industry: str) -> str:
    growth = trends.get("cohort_revenue_growth_recent") or 0.0
    capex_delta = trends.get("cohort_capex_delta") or 0.0
    margin_delta = trends.get("cohort_op_margin_delta") or 0.0

    if "Semiconductor" in (industry or "") and growth > 0.20 and capex_delta > 0.01:
        return "AI capex expansion"
    if growth < 0 and margin_delta < -0.01:
        return "Cyclical downturn / digestion phase"
    if growth > 0.10 and margin_delta > 0:
        return "Operating leverage cycle"
    if margin_delta < -0.01 and capex_delta > 0:
        return "Capex re-investment phase (margin headwind)"
    if abs(growth) < 0.05 and abs(margin_delta) < 0.01:
        return "Mid-cycle / mature"
    return "Mixed regime"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_sector_research(target_ticker: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Build the deep sector research payload consumed by the sector agent.

    Cache-aware. When `force_refresh=False` (default) the function consults the
    snapshot store first under kind="sector_warm" and key
    f"{sector}:{sub_industry}:{target_ticker}". A 7-day TTL bounds staleness,
    and the warm snapshot's lineage points at each cohort member's
    company_cold snapshot so a 10-K refresh ripples through automatically.
    """
    # Lazy import to avoid circular import: cache → models → cache
    from ..cache import cache_get, cache_put

    target_fin = get_full_financials(target_ticker)
    profile = target_fin.get("profile") or {}
    target_ratios = target_fin.get("ratios") or {}
    sector = profile.get("sector", "")
    industry = profile.get("industry", "")
    sub_industry = profile.get("sub_industry") or industry

    cache_key = f"{sector}:{sub_industry}:{target_ticker}"
    if not force_refresh:
        cached = cache_get(cache_key, "sector_warm", max_age_seconds=7 * 24 * 3600)
        if cached and isinstance(cached.payload, dict):
            payload = dict(cached.payload)
            payload.pop("schema_version", None)
            return payload

    sector_block = _resolve_sector_block(sector)
    sub_overrides = _resolve_subindustry_overrides(sector_block, industry)
    kpi_groups = sector_block.get("kpi_groups", {})
    if sub_overrides.get("additional_kpis"):
        kpi_groups = {
            **kpi_groups,
            "_subindustry_extra": sub_overrides["additional_kpis"],
        }

    cohort_tickers = build_cohort(target_ticker)
    cohort_with_target: List[Dict] = []
    cohort_ratios: List[Dict] = []
    cohort_fins: List[Dict] = []
    for t in cohort_tickers:
        fin = get_full_financials(t)
        if not fin or not fin.get("ratios"):
            continue
        ratios = fin["ratios"]
        cohort_ratios.append(ratios)
        cohort_fins.append(fin)
        cohort_with_target.append({
            "ticker": t,
            "ratios": ratios,
            "market_cap": (fin.get("profile") or {}).get("market_cap"),
            "financials": fin,
        })
    # Include target itself for outlier detection
    cohort_with_target.append({
        "ticker": target_ticker,
        "ratios": target_ratios,
        "market_cap": profile.get("market_cap"),
        "financials": target_fin,
    })
    cohort_fins.append(target_fin)

    placements = compute_kpi_placements(target_ratios, cohort_ratios, kpi_groups)
    outliers = detect_outliers(cohort_with_target)
    trends = compute_sector_trends(cohort_fins)
    structure = industry_structure(cohort_with_target)
    filing_themes = aggregate_cohort_filing_themes(cohort_tickers)
    regime = detect_sector_regime(sector, trends, industry)

    payload = {
        "target_ticker": target_ticker,
        "sector": sector,
        "industry": industry,
        "sub_industry": sub_industry,
        "cohort": {
            "peers": cohort_tickers,
            "size": len(cohort_tickers),
            "selection_basis": "sub_industry" if len(cohort_tickers) > 0 else "sector",
        },
        "sector_drivers": sector_block.get("key_drivers", []),
        "sector_secular_trends": sector_block.get("secular_trends", []),
        "subindustry_themes": sub_overrides.get("structural_themes", ""),
        "subindustry_watch": sub_overrides.get("watch", ""),
        "valuation_lens": sector_block.get("valuation_lens", ""),
        "macro_sensitivities": sector_block.get("macro_sensitivities", []),
        "common_sector_risks": sector_block.get("common_risks", []),
        "industry_structure": structure,
        "kpi_placements": placements,
        "outliers": outliers,
        "trends": trends,
        "regime": regime,
        "cohort_filing_themes": filing_themes,
    }

    # Wave 6B: tighten cohort invalidation. Previously every peer-side
    # 10-K invalidated this sector_warm entry — even when the filing
    # didn't move any of the KPIs the sector math actually consumes.
    # Now we fingerprint the *KPI inputs* (revenue / operating_income /
    # capex / shares) per cohort member. A peer's irrelevant 10-K (e.g.
    # legal disclosures, shelf registration) leaves the fingerprint
    # untouched and the warm snapshot stays fresh — saving a bunch of
    # spurious recomputes.
    sources_used: List[Any] = [f"cohort:{','.join(sorted(cohort_tickers))}"]
    for entry in cohort_with_target:
        kpis = _kpi_fingerprint_inputs(entry)
        if kpis is not None:
            sources_used.append(kpis)

    # Lineage: parent ids = each cohort member's most recent company_cold
    # snapshot. Lineage cascades remain useful for cohort-membership
    # changes (a name dropped from the universe, replaced by another),
    # which the per-KPI fingerprint above wouldn't catch.
    parent_ids: List[int] = []
    for t in cohort_tickers + [target_ticker]:
        cold = cache_get(t, "company_cold")
        if cold:
            parent_ids.append(cold.id)

    # Phase C: in live mode `resolved_cost_tokens` adds any LLM tokens just
    # consumed; in demo mode it returns the baseline only.
    from ..cache import resolved_cost_tokens
    cache_put(
        cache_key, "sector_warm",
        payload=payload, sources_used=sources_used,
        generated_by="sector_research_service",
        parent_snapshots=parent_ids,
        ttl_seconds=7 * 24 * 3600,
        cost_tokens=resolved_cost_tokens(400),
    )
    return payload
