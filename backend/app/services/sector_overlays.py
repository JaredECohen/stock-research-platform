"""Per-ticker sector-relevant data overlays.

Each `compute_*_overlay` produces a small, JSON-friendly bundle that
the sector agent (or any downstream consumer) can hand directly to an
LLM as context. The bundles include:
  - the raw `SeriesSnapshot.to_dict()` for each series consulted
  - aggregated read-outs ("Sun Belt HPI +5.2% YoY") when geography is
    available
  - a `narrative_hints` list of short, declarative phrases the LLM
    can lift verbatim into the memo when relevant.

The overlays are deliberately conservative — they always degrade to
`{"available": False, "reason": "..."}` rather than throwing, so the
sector agent can call them speculatively.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import data_catalog_service, company_geography, factor_analytics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real estate / housing
# ---------------------------------------------------------------------------

_REIT_LIKE_SECTORS = {"Real Estate", "Financials"}
_NATIONAL_HOUSING_SERIES = [
    "CSUSHPISA", "MORTGAGE30US", "HOUST", "PERMIT", "EXHOSLUSM495S", "MSPUS",
    "RRVRUSQ156N", "CUUR0000SEHA",
]
_METRO_HPI_SERIES = {
    "SF": "SFXRSA", "NYC": "NYXRSA", "LA": "LXXRSA", "ATL": "ATXRSA",
    "DAL": "DAXRSA", "PHX": "PHXRSA", "MIA": "MIXRSA", "CHI": "CHXRSA",
    "BOS": "BOXRSA", "SEA": "SEXRSA", "DCA": "WDXRSA", "DEN": "DNXRSA",
    "MIN": "MNXRSA", "POR": "POXRSA", "LAS": "LVXRSA", "DET": "DEXRSA",
}


def compute_real_estate_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """National housing snapshot + REIT-specific metro overlay when geography is known."""
    sector = (profile or {}).get("sector") or ""
    sub_industry = (profile or {}).get("sub_industry") or ""

    # Always pull the national housing readings — they touch most sectors.
    national = data_catalog_service.fetch_snapshots(_NATIONAL_HOUSING_SERIES)
    national_payload = [snap.to_dict() for snap in national]

    geo = company_geography.get_geography(ticker, allow_llm_fallback=False)

    metro_block: List[Dict[str, Any]] = []
    weighted_yoy: Optional[float] = None
    if geo and geo.get("metros"):
        weights = geo["metros"]
        total_weight = 0.0
        weighted_sum = 0.0
        for metro_code, weight in weights.items():
            series_id = _METRO_HPI_SERIES.get(str(metro_code).upper())
            if not series_id:
                continue
            snap = data_catalog_service.fetch_series(series_id)
            if snap is None or snap.error:
                continue
            metro_block.append({
                **snap.to_dict(),
                "metro_code": metro_code,
                "footprint_weight": weight,
            })
            if snap.yoy_pct is not None:
                weighted_sum += weight * snap.yoy_pct
                total_weight += weight
        if total_weight > 0:
            weighted_yoy = weighted_sum / total_weight

    narrative: List[str] = []
    case_shiller = next((s for s in national if s.series_id == "CSUSHPISA"), None)
    mortgage = next((s for s in national if s.series_id == "MORTGAGE30US"), None)
    starts = next((s for s in national if s.series_id == "HOUST"), None)
    if case_shiller and case_shiller.yoy_pct is not None:
        narrative.append(
            f"National Case-Shiller home prices {case_shiller.yoy_pct:+.1%} YoY."
        )
    if mortgage and mortgage.latest:
        narrative.append(
            f"30-year mortgage at {mortgage.latest.get('value'):.2f}%; "
            f"{mortgage.change_3m:+.2f}pp over last 3 months."
            if mortgage.change_3m is not None else
            f"30-year mortgage at {mortgage.latest.get('value'):.2f}%."
        )
    if starts and starts.yoy_pct is not None:
        narrative.append(f"Housing starts {starts.yoy_pct:+.1%} YoY.")
    if weighted_yoy is not None and metro_block:
        metro_names = ", ".join(m.get("metro_code", "") for m in metro_block[:4])
        narrative.append(
            f"Footprint-weighted home-price growth across {metro_names}: "
            f"{weighted_yoy:+.1%} YoY."
        )

    return {
        "available": bool(national_payload),
        "ticker": ticker,
        "sector": sector,
        "sub_industry": sub_industry,
        "national_series": national_payload,
        "geography": geo,
        "metro_overlay": metro_block,
        "footprint_weighted_home_price_yoy": weighted_yoy,
        "narrative_hints": narrative,
    }


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

_CORE_ENERGY_SERIES = [
    "PET.WCESTUS1.W", "PET.WGTSTUS1.W", "PET.WDISTUS1.W",
    "NG.NW2_EPG0_SWO_R48_BCF.W", "PET.RWTC.D", "NG.RNGWHHD.D",
]


def compute_energy_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Petroleum + natural-gas storage + WTI + Henry Hub snapshot."""
    sector = (profile or {}).get("sector") or ""
    sub_industry = (profile or {}).get("sub_industry") or ""

    series = data_catalog_service.fetch_snapshots(_CORE_ENERGY_SERIES)
    series_payload = [snap.to_dict() for snap in series]

    # EIA-specific high-signal snapshots (week-over-week and YoY)
    try:
        from .data_service import get_data_service
        ds = get_data_service()
        eia = getattr(ds, "eia", None)
    except Exception:
        eia = None

    petroleum_snap = eia.get_petroleum_storage_snapshot() if eia else None
    natgas_snap = eia.get_natgas_storage_snapshot() if eia else None

    narrative: List[str] = []
    wti = next((s for s in series if s.series_id == "PET.RWTC.D"), None)
    hh = next((s for s in series if s.series_id == "NG.RNGWHHD.D"), None)
    crude = next((s for s in series if s.series_id == "PET.WCESTUS1.W"), None)
    if wti and wti.latest:
        narrative.append(
            f"WTI at ${wti.latest.get('value'):.2f}/bbl; {wti.change_3m:+.2f}/bbl over 3 months."
            if wti.change_3m is not None else
            f"WTI at ${wti.latest.get('value'):.2f}/bbl."
        )
    if hh and hh.latest:
        narrative.append(f"Henry Hub natural gas at ${hh.latest.get('value'):.2f}/MMBtu.")
    if crude and crude.latest and crude.change_1m is not None:
        delta_kbpd = crude.change_1m
        direction = "build" if delta_kbpd > 0 else "draw"
        narrative.append(
            f"US crude inventories {direction} of {abs(delta_kbpd):,.0f} thousand barrels over 4 weeks."
        )
    if petroleum_snap and petroleum_snap.get("year_over_year_delta") is not None:
        delta = petroleum_snap["year_over_year_delta"]
        side = "above" if delta > 0 else "below"
        narrative.append(
            f"Crude stocks {abs(delta):,.0f} kbbls {side} same week prior year."
        )

    return {
        "available": bool(series_payload),
        "ticker": ticker,
        "sector": sector,
        "sub_industry": sub_industry,
        "core_series": series_payload,
        "petroleum_snapshot": petroleum_snap,
        "natgas_snapshot": natgas_snap,
        "narrative_hints": narrative,
    }


# ---------------------------------------------------------------------------
# Consumer / retail
# ---------------------------------------------------------------------------

_NATIONAL_RETAIL_SERIES = [
    "RSAFS", "RSXFS", "RRSFS", "DSPIC96", "PSAVERT", "TOTALSL",
    "DRCCLACBS", "UMCSENT", "VMTD11",
]
_CENSUS_RETAIL_SERIES = [
    "MARTS_44X72", "MARTS_445", "MARTS_448", "MARTS_454", "MARTS_447", "MARTS_722",
]


def compute_consumer_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Consumer health + retail trade by category + footprint state unemployment."""
    sector = (profile or {}).get("sector") or ""
    sub_industry = (profile or {}).get("sub_industry") or ""

    national_consumer = data_catalog_service.fetch_snapshots(_NATIONAL_RETAIL_SERIES)
    retail_categories = data_catalog_service.fetch_snapshots(_CENSUS_RETAIL_SERIES)

    # State-level unemployment for the ticker's footprint
    state_block: List[Dict[str, Any]] = []
    weighted_unemployment: Optional[float] = None
    geo = company_geography.get_geography(ticker, allow_llm_fallback=False)
    if geo and geo.get("states"):
        state_to_series = {
            "CA": "LAUST060000000000003",
            "TX": "LAUST480000000000003",
            "NY": "LAUST360000000000003",
            "FL": "LAUST120000000000003",
        }
        weights = geo["states"]
        total_weight = 0.0
        weighted_sum = 0.0
        for state_code, weight in weights.items():
            series_id = state_to_series.get(str(state_code).upper())
            if not series_id:
                continue
            snap = data_catalog_service.fetch_series(series_id)
            if snap is None or snap.error or snap.latest is None:
                continue
            state_block.append({
                **snap.to_dict(),
                "state_code": state_code,
                "footprint_weight": weight,
            })
            value = snap.latest.get("value")
            if value is not None:
                weighted_sum += weight * value
                total_weight += weight
        if total_weight > 0:
            weighted_unemployment = weighted_sum / total_weight

    narrative: List[str] = []
    retail_total = next((s for s in national_consumer if s.series_id == "RSAFS"), None)
    saving = next((s for s in national_consumer if s.series_id == "PSAVERT"), None)
    delinq = next((s for s in national_consumer if s.series_id == "DRCCLACBS"), None)
    if retail_total and retail_total.yoy_pct is not None:
        narrative.append(f"Headline retail sales {retail_total.yoy_pct:+.1%} YoY.")
    if saving and saving.latest:
        narrative.append(
            f"Personal saving rate at {saving.latest.get('value'):.1f}%, "
            f"{saving.change_3m:+.2f}pp over 3 months."
            if saving.change_3m is not None else
            f"Personal saving rate at {saving.latest.get('value'):.1f}%."
        )
    if delinq and delinq.latest:
        narrative.append(f"Credit-card delinquency rate at {delinq.latest.get('value'):.2f}%.")

    nonstore = next((s for s in retail_categories if s.series_id == "MARTS_454"), None)
    if nonstore and nonstore.yoy_pct is not None:
        narrative.append(f"E-commerce (nonstore retail) sales {nonstore.yoy_pct:+.1%} YoY.")
    if weighted_unemployment is not None and state_block:
        states = ", ".join(s.get("state_code", "") for s in state_block[:4])
        narrative.append(
            f"Footprint-weighted unemployment ({states}): {weighted_unemployment:.2f}%."
        )

    return {
        "available": bool(national_consumer or retail_categories),
        "ticker": ticker,
        "sector": sector,
        "sub_industry": sub_industry,
        "national_consumer": [s.to_dict() for s in national_consumer],
        "retail_category_sales": [s.to_dict() for s in retail_categories],
        "geography": geo,
        "footprint_state_unemployment": state_block,
        "footprint_weighted_unemployment": weighted_unemployment,
        "narrative_hints": narrative,
    }


# ---------------------------------------------------------------------------
# Inflation / pricing power
# ---------------------------------------------------------------------------

_INFLATION_SERIES = [
    "CPIAUCSL", "PCEPI", "CORESTICKM159SFRBATL",
    "CUUR0000SAF1", "CUUR0000SETA01", "CUUR0000SETA02",
    "CUUR0000SAM", "CUUR0000SETB01",
]


def compute_inflation_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Headline + category inflation prints. Drives pricing-power narrative."""
    series = data_catalog_service.fetch_snapshots(_INFLATION_SERIES)
    payload = [s.to_dict() for s in series]
    narrative: List[str] = []
    sticky = next((s for s in series if s.series_id == "CORESTICKM159SFRBATL"), None)
    pce = next((s for s in series if s.series_id == "PCEPI"), None)
    if sticky and sticky.latest:
        narrative.append(f"Sticky core CPI at {sticky.latest.get('value'):.1f}% YoY.")
    if pce and pce.yoy_pct is not None:
        narrative.append(f"PCE inflation {pce.yoy_pct:+.1%} YoY.")
    return {
        "available": bool(payload),
        "ticker": ticker,
        "series": payload,
        "narrative_hints": narrative,
    }


# ---------------------------------------------------------------------------
# Credit
# ---------------------------------------------------------------------------

_CREDIT_SERIES = [
    "BAMLH0A0HYM2", "T10Y2Y", "TOTALSL", "DRCCLACBS",
]


def compute_credit_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    series = data_catalog_service.fetch_snapshots(_CREDIT_SERIES)
    payload = [s.to_dict() for s in series]
    narrative: List[str] = []
    hy = next((s for s in series if s.series_id == "BAMLH0A0HYM2"), None)
    curve = next((s for s in series if s.series_id == "T10Y2Y"), None)
    if hy and hy.latest:
        narrative.append(
            f"High-yield credit spread at {hy.latest.get('value'):.2f}%, "
            f"{hy.change_3m:+.2f}pp over 3 months."
            if hy.change_3m is not None else
            f"High-yield credit spread at {hy.latest.get('value'):.2f}%."
        )
    if curve and curve.latest:
        spread = curve.latest.get("value")
        direction = "inverted" if spread is not None and spread < 0 else "positive"
        narrative.append(f"10Y-2Y term spread {spread:+.2f}% ({direction}).")
    return {
        "available": bool(payload),
        "ticker": ticker,
        "series": payload,
        "narrative_hints": narrative,
    }


# ---------------------------------------------------------------------------
# Sector-aware dispatcher
# ---------------------------------------------------------------------------

_OVERLAY_PRIORITY_BY_SECTOR: Dict[str, List[str]] = {
    "Real Estate":              ["real_estate", "credit", "factor"],
    "Energy":                   ["energy", "credit", "factor"],
    "Utilities":                ["energy", "credit", "factor"],
    "Financials":               ["credit", "real_estate", "factor"],
    "Consumer Discretionary":   ["consumer", "credit", "factor"],
    "Consumer Staples":         ["consumer", "inflation", "factor"],
    "Materials":                ["real_estate", "energy", "factor"],
    "Industrials":              ["energy", "real_estate", "factor"],
    "Health Care":              ["inflation", "credit", "factor"],
    "Information Technology":   ["credit", "factor", "inflation"],
    "Communication Services":   ["consumer", "credit", "factor"],
}


def compute_sector_overlays(
    ticker: str, *,
    profile: Optional[Dict[str, Any]] = None,
    explicit_overlays: Optional[List[str]] = None,
    max_overlays: int = 3,
) -> Dict[str, Any]:
    """Dispatch the right overlays for a ticker's sector.

    Returns a dict keyed by overlay name. `available=False` overlays are
    still included so the agent can see what it tried and what came back
    empty.
    """
    sector = (profile or {}).get("sector") or ""
    chosen = list(explicit_overlays) if explicit_overlays else (
        _OVERLAY_PRIORITY_BY_SECTOR.get(sector, ["credit", "inflation"])[:max_overlays]
    )
    bundles: Dict[str, Any] = {}
    for name in chosen:
        fn = _OVERLAY_FUNCS.get(name)
        if fn is None:
            continue
        try:
            bundles[name] = fn(ticker, profile=profile)
        except Exception as exc:
            log.warning("Overlay %s failed for %s: %s", name, ticker, exc)
            bundles[name] = {"available": False, "reason": str(exc)}
    return {
        "ticker": ticker,
        "sector": sector,
        "overlays_run": list(bundles.keys()),
        "bundles": bundles,
    }


def compute_factor_overlay(
    ticker: str, *, profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fama-French 5 + momentum decomposition of the ticker's recent excess returns.

    Returns the same dict shape every other overlay does so the sector
    agent can lift `narrative_hints` straight into key_points. Falls
    back to `available: False` when Ken French data hasn't been fetched
    yet or the price history overlap is too short.
    """
    sector = (profile or {}).get("sector") or ""
    sub_industry = (profile or {}).get("sub_industry") or ""
    try:
        result = factor_analytics.compute_for_ticker(ticker)
    except Exception as exc:
        log.warning("Factor overlay failed for %s: %s", ticker, exc)
        return {"available": False, "reason": str(exc)}
    if result is None:
        return {
            "available": False,
            "reason": "Insufficient price history or Fama-French factor data unavailable.",
        }
    return {
        "available": True,
        "ticker": ticker,
        "sector": sector,
        "sub_industry": sub_industry,
        "factor_profile": result,
        "narrative_hints": result.get("narrative_hints", []),
    }


_OVERLAY_FUNCS = {
    "real_estate": compute_real_estate_overlay,
    "energy": compute_energy_overlay,
    "consumer": compute_consumer_overlay,
    "inflation": compute_inflation_overlay,
    "credit": compute_credit_overlay,
    "factor": compute_factor_overlay,
}
