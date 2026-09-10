"""Fundamental Factor Scorecard — methodology spec, version ``fs-v1``.

Pure constants. This module is the single place where the scorecard's
*methodology* lives: which features exist, how each one is defined, which
direction is "good", how families are weighted, which sectors a feature
does not apply to, and the normalization parameters. Everything else in
the scorecard (feature engine, normalization, persistence, API) reads
from here, and ``spec_hash()`` fingerprints it so a stored score row can
always be traced back to the exact definitions that produced it.

Bumping the methodology means a new ``VERSION_KEY`` (``fs-v2`` …), never a
silent edit of ``fs-v1``: rows already persisted under ``fs-v1`` were
computed with these definitions and must stay comparable to each other.

Where this sits in the research process
---------------------------------------
The owner's process asks four questions: what will happen, who captures
the economic benefit, what is already priced in, and what evidence will
reveal the gap. The scorecard is *observed data*, not interpretation:

* The **Valuation** family is the "what is already priced in" leg — every
  feature there is a yield (earnings / FCF / EBITDA / sales over what the
  market pays), so a high score means the market is paying little for
  observed cash flows. It says nothing about whether those cash flows
  persist; that is the job of the memo's expectations ledger.
* Quality, Profitability, Capital Allocation and Earnings Quality
  together describe the "compounder" profile; Growth plus the two
  inflection features (``revenue_growth_accel``,
  ``operating_margin_change_1y``) describe the "inflection" profile.
  ``scorecard_normalize`` exposes both as sub-composites that are
  computed but never weighted into the overall score.

Explicit non-goals for ``fs-v1`` (hooks named so the next version has a
place to land):

* **Expectations ledger** (reported consensus / management guidance /
  price-implied / our forecast). The scorecard uses reported financials
  only. The ledger lives in the memo's ``mispricing_thesis`` block; a
  consensus-based feature family would read
  ``providers.fmp_provider.get_estimates`` and must obey the
  ``docs/research/Framework_Update_2026-09-07.md`` same-basis rule.
  Hook: add a ``Expectations`` family here under a new VERSION_KEY.
* **Industry-group neutralisation.** ``fs-v1`` neutralises by the
  provider's ``Company.sector`` string through ``SECTOR_ALIASES`` below.
  The GICS registry already on this branch
  (``app.services.industry_knowledge.list_industry_groups``) is the
  ``fs-v2`` key: neutralise at the 4-digit industry-group level and
  attach industry-specific applicability masks from the encyclopedia.
  Hook: ``normalize_sector`` is the only function the normaliser calls
  to bucket a name, so ``fs-v2`` swaps that one function.

Sign convention
---------------
``sign = +1`` means a higher raw value is *better* (scores up);
``sign = -1`` means a higher raw value is *worse*. The feature engine
always emits the raw ratio; the normaliser multiplies the z-score by
``sign`` so every stored z reads "higher = better".

Missing inputs and percentiles (``RULES``)
------------------------------------------
Two methodology rules that are not visible in a formula string are
frozen in ``RULES`` and hashed with the rest of the spec:

* **Missing inputs are null, never zero.** Every line a formula names
  must be present in the point-in-time snapshot; an absent one makes the
  feature null with a ``missing:<line>`` reason. No partner-line
  zero-fill: ``r_and_d`` is not 0 because ``sga`` is present,
  ``share_repurchases`` is not 0 because ``dividends_paid`` is present,
  ``short_term_investments`` is not 0 because ``cash_and_equivalents`` is
  present, and one half of the debt split is not 0 because the other is.
  The persistence slice stores a provider null as a null row, so by the
  time a formula runs "absent" and "unknown" are the same thing, and a 0
  invented here would be standardised and ranked as if it had been
  observed. (``ratios.net_debt`` and the screener do zero-fill; the
  scorecard deliberately does not.)
* **Percentiles are rank-based** (average rank for ties, ``(0, 100]``,
  100 = best). The overall percentile ranks rows with a non-null overall
  z; each family percentile ranks rows with a non-null z for *that*
  family, so a name whose overall is "insufficient data" can still carry
  a valuation percentile — which is what the memo's valuation
  disagreement rule reads.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

VERSION_KEY = "fs-v1"

# ---------------------------------------------------------------------------
# Sector taxonomy (fs-v1: provider sector strings → canonical GICS sector)
# ---------------------------------------------------------------------------

# Canonical names follow the 11 GICS sectors exactly as the industry
# registry spells them (`gics_industries_2026.json`), so that the fs-v2
# switch to industry groups is a refinement of this bucketing, not a
# rename of it.
SECTOR_ENERGY = "Energy"
SECTOR_MATERIALS = "Materials"
SECTOR_INDUSTRIALS = "Industrials"
SECTOR_CONSUMER_DISCRETIONARY = "Consumer Discretionary"
SECTOR_CONSUMER_STAPLES = "Consumer Staples"
SECTOR_HEALTH_CARE = "Health Care"
SECTOR_FINANCIALS = "Financials"
SECTOR_INFORMATION_TECHNOLOGY = "Information Technology"
SECTOR_COMMUNICATION_SERVICES = "Communication Services"
SECTOR_UTILITIES = "Utilities"
SECTOR_REAL_ESTATE = "Real Estate"

CANONICAL_SECTORS: tuple[str, ...] = (
    SECTOR_ENERGY,
    SECTOR_MATERIALS,
    SECTOR_INDUSTRIALS,
    SECTOR_CONSUMER_DISCRETIONARY,
    SECTOR_CONSUMER_STAPLES,
    SECTOR_HEALTH_CARE,
    SECTOR_FINANCIALS,
    SECTOR_INFORMATION_TECHNOLOGY,
    SECTOR_COMMUNICATION_SERVICES,
    SECTOR_UTILITIES,
    SECTOR_REAL_ESTATE,
)

# Provider vocabularies seen in this codebase: FMP profiles ("Financial
# Services", "Consumer Cyclical", "Basic Materials", "Technology",
# "Healthcare"), the app's own `sector_configs.json` ("Technology",
# "Healthcare", "Financials") and GICS proper. Keys are already in the
# `_sector_key` form (lower-case, alphanumerics only) so lookups are
# insensitive to case, punctuation and spacing.
SECTOR_ALIASES: dict[str, str] = {
    "energy": SECTOR_ENERGY,
    "oilgas": SECTOR_ENERGY,
    "materials": SECTOR_MATERIALS,
    "basicmaterials": SECTOR_MATERIALS,
    "industrials": SECTOR_INDUSTRIALS,
    "industrial": SECTOR_INDUSTRIALS,
    "industrialgoods": SECTOR_INDUSTRIALS,
    "consumerdiscretionary": SECTOR_CONSUMER_DISCRETIONARY,
    "consumercyclical": SECTOR_CONSUMER_DISCRETIONARY,
    "consumerstaples": SECTOR_CONSUMER_STAPLES,
    "consumerdefensive": SECTOR_CONSUMER_STAPLES,
    "healthcare": SECTOR_HEALTH_CARE,
    "health": SECTOR_HEALTH_CARE,
    "financials": SECTOR_FINANCIALS,
    "financial": SECTOR_FINANCIALS,
    "financialservices": SECTOR_FINANCIALS,
    "banks": SECTOR_FINANCIALS,
    "informationtechnology": SECTOR_INFORMATION_TECHNOLOGY,
    "technology": SECTOR_INFORMATION_TECHNOLOGY,
    "tech": SECTOR_INFORMATION_TECHNOLOGY,
    "communicationservices": SECTOR_COMMUNICATION_SERVICES,
    "communication": SECTOR_COMMUNICATION_SERVICES,
    "communications": SECTOR_COMMUNICATION_SERVICES,
    "telecommunicationservices": SECTOR_COMMUNICATION_SERVICES,
    "telecommunications": SECTOR_COMMUNICATION_SERVICES,
    "telecom": SECTOR_COMMUNICATION_SERVICES,
    "utilities": SECTOR_UTILITIES,
    "utility": SECTOR_UTILITIES,
    "realestate": SECTOR_REAL_ESTATE,
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _sector_key(raw: str) -> str:
    return _NON_ALNUM.sub("", raw.strip().lower())


def normalize_sector(raw: str | None) -> str | None:
    """Map a provider sector string onto a canonical GICS sector name.

    Returns None for empty input or a string the alias table does not
    know. The caller decides what an unmatched sector means (the
    normaliser falls back to universe-relative z and records the raw
    string), so this never guesses: an unknown label is *unknown*, not
    "Industrials".
    """
    if raw is None:
        return None
    key = _sector_key(str(raw))
    if not key:
        return None
    return SECTOR_ALIASES.get(key)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

FAMILY_VALUATION = "valuation"
FAMILY_QUALITY = "quality"
FAMILY_GROWTH = "growth"
FAMILY_PROFITABILITY = "profitability"
FAMILY_EFFICIENCY = "efficiency"
FAMILY_LEVERAGE = "leverage"
FAMILY_CAPITAL_ALLOCATION = "capital_allocation"
FAMILY_EARNINGS_QUALITY = "earnings_quality"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    weight: float
    description: str


# Equal weights in v1 (owner decision). The evaluation job is what earns a
# family a bigger weight; until it has run, an uneven prior would be a
# guess dressed up as methodology.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        FAMILY_VALUATION, 1.0 / 8,
        "What is already priced in: observed cash flows and earnings per "
        "dollar of market value or enterprise value. High = the market pays "
        "little for what the company has reported.",
    ),
    FamilySpec(
        FAMILY_QUALITY, 1.0 / 8,
        "Returns on the capital the business employs and how stable its "
        "gross economics have been.",
    ),
    FamilySpec(
        FAMILY_GROWTH, 1.0 / 8,
        "Reported growth in revenue, operating income and earnings, plus "
        "whether revenue growth is accelerating (an inflection signal).",
    ),
    FamilySpec(
        FAMILY_PROFITABILITY, 1.0 / 8,
        "Margin structure and its one-year change.",
    ),
    FamilySpec(
        FAMILY_EFFICIENCY, 1.0 / 8,
        "How much revenue each dollar of assets and operating spend produces.",
    ),
    FamilySpec(
        FAMILY_LEVERAGE, 1.0 / 8,
        "Balance-sheet risk: net debt against earnings, equity and interest.",
    ),
    FamilySpec(
        FAMILY_CAPITAL_ALLOCATION, 1.0 / 8,
        "Whether cash goes back to owners or leaks through dilution, "
        "stock compensation and acquired goodwill.",
    ),
    FamilySpec(
        FAMILY_EARNINGS_QUALITY, 1.0 / 8,
        "Whether reported earnings are backed by cash.",
    ),
)

FAMILY_WEIGHTS: dict[str, float] = {f.name: f.weight for f in FAMILIES}
FAMILY_NAMES: tuple[str, ...] = tuple(f.name for f in FAMILIES)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """One scorecard feature.

    ``inputs`` names the statement line items (and the ``price``/``shares``
    context keys) the formula consumes — it is documentation for the
    ``/api/scorecard/spec`` endpoint and the CSV data dictionary, not a
    dependency graph the engine walks. ``exclude_sectors`` holds canonical
    sector names (see ``CANONICAL_SECTORS``); an excluded feature is
    reported as not applicable and does not count against coverage.
    """
    name: str
    family: str
    sign: int          # +1 higher raw is better, -1 higher raw is worse
    weight: float      # within-family weight (equal in v1)
    formula: str       # human-readable definition, frozen with the version
    description: str
    inputs: tuple[str, ...]
    exclude_sectors: tuple[str, ...] = ()


_FIN = (SECTOR_FINANCIALS,)
_FIN_RE = (SECTOR_FINANCIALS, SECTOR_REAL_ESTATE)
_FIN_UTIL = (SECTOR_FINANCIALS, SECTOR_UTILITIES)

# `_ttm` in the formulas below is the latest annual row available at the
# as-of date (fs-v1 ingests annual statements only); `_prior` is the row
# exactly one fiscal year earlier. Cash-flow outflows (capex, dividends,
# buybacks) are stored negative by FMP, hence the abs().
FEATURE_SPEC: tuple[FeatureSpec, ...] = (
    # --- Valuation: "what is already priced in" ---------------------------
    FeatureSpec(
        "earnings_yield", FAMILY_VALUATION, +1, 1.0,
        "net_income_ttm / market_cap",
        "Trailing earnings per dollar of market value.",
        ("net_income", "price", "shares"),
    ),
    FeatureSpec(
        "fcf_yield", FAMILY_VALUATION, +1, 1.0,
        "free_cash_flow_ttm / market_cap",
        "Trailing free cash flow per dollar of market value.",
        ("free_cash_flow", "price", "shares"),
    ),
    FeatureSpec(
        "ebitda_ev_yield", FAMILY_VALUATION, +1, 1.0,
        "ebitda_ttm / enterprise_value (EV > 0)",
        "Trailing EBITDA per dollar of enterprise value. Enterprise value "
        "is not meaningful for banks, insurers or REITs.",
        ("ebitda", "operating_income", "depreciation_and_amortization", "price", "shares",
         "total_debt", "cash_and_equivalents", "short_term_investments"),
        _FIN_RE,
    ),
    FeatureSpec(
        "sales_ev_yield", FAMILY_VALUATION, +1, 1.0,
        "revenue_ttm / enterprise_value (EV > 0)",
        "Trailing revenue per dollar of enterprise value.",
        ("revenue", "price", "shares", "total_debt", "cash_and_equivalents", "short_term_investments"),
        _FIN,
    ),
    # --- Quality ----------------------------------------------------------
    FeatureSpec(
        "roic", FAMILY_QUALITY, +1, 1.0,
        "operating_income_ttm * (1 - tax_rate) / (total_debt + shareholders_equity); "
        "tax_rate = effective rate when credible, statutory fallback only for "
        "profitable names (finance.ratios.roic_with_provenance)",
        "Return on invested capital, with the shared tax-rate provenance rules.",
        ("operating_income", "pretax_income", "tax_expense", "total_debt",
         "short_term_debt", "long_term_debt", "shareholders_equity"),
    ),
    FeatureSpec(
        "roa", FAMILY_QUALITY, +1, 1.0,
        "net_income_ttm / avg(total_assets_latest, total_assets_prior)",
        "Return on average assets.",
        ("net_income", "total_assets"),
    ),
    FeatureSpec(
        "roe", FAMILY_QUALITY, +1, 1.0,
        "net_income_ttm / shareholders_equity_latest (equity > 0)",
        "Return on equity; undefined for negative-equity balance sheets.",
        ("net_income", "shareholders_equity"),
    ),
    FeatureSpec(
        "gross_margin_stability", FAMILY_QUALITY, +1, 1.0,
        "-stdev(gross_margin over the last <= 5 annual periods; >= 3 required)",
        "Negated sample standard deviation of annual gross margin: stable "
        "gross economics score high.",
        ("gross_profit", "revenue"),
    ),
    # --- Growth -----------------------------------------------------------
    FeatureSpec(
        "revenue_growth_1y", FAMILY_GROWTH, +1, 1.0,
        "revenue_ttm / revenue_prior - 1 (prior > 0)",
        "One-year revenue growth.",
        ("revenue",),
    ),
    FeatureSpec(
        "revenue_cagr_3y", FAMILY_GROWTH, +1, 1.0,
        "(revenue_ttm / revenue_3y_ago) ** (1/3) - 1 (both > 0)",
        "Three-year revenue CAGR.",
        ("revenue",),
    ),
    FeatureSpec(
        "operating_income_growth_1y", FAMILY_GROWTH, +1, 1.0,
        "operating_income_ttm / operating_income_prior - 1 (prior > 0)",
        "One-year operating income growth.",
        ("operating_income",),
    ),
    FeatureSpec(
        "eps_growth_1y", FAMILY_GROWTH, +1, 1.0,
        "eps_diluted_ttm / eps_diluted_prior - 1 (prior > 0); net_income fallback",
        "One-year diluted EPS growth.",
        ("eps_diluted", "net_income"),
    ),
    FeatureSpec(
        "revenue_growth_accel", FAMILY_GROWTH, +1, 1.0,
        "revenue_growth_1y - revenue_growth_1y_prior",
        "Change in revenue growth versus the prior year (inflection signal).",
        ("revenue",),
    ),
    # --- Profitability ----------------------------------------------------
    FeatureSpec(
        "gross_margin", FAMILY_PROFITABILITY, +1, 1.0,
        "gross_profit_ttm / revenue_ttm (revenue > 0)",
        "Gross margin. Not meaningful for financials.",
        ("gross_profit", "revenue"),
        _FIN,
    ),
    FeatureSpec(
        "operating_margin", FAMILY_PROFITABILITY, +1, 1.0,
        "operating_income_ttm / revenue_ttm (revenue > 0)",
        "Operating margin.",
        ("operating_income", "revenue"),
    ),
    FeatureSpec(
        "fcf_margin", FAMILY_PROFITABILITY, +1, 1.0,
        "free_cash_flow_ttm / revenue_ttm (revenue > 0)",
        "Free-cash-flow margin.",
        ("free_cash_flow", "revenue"),
    ),
    FeatureSpec(
        "operating_margin_change_1y", FAMILY_PROFITABILITY, +1, 1.0,
        "operating_margin_ttm - operating_margin_prior",
        "One-year change in operating margin (inflection signal).",
        ("operating_income", "revenue"),
    ),
    # --- Efficiency -------------------------------------------------------
    FeatureSpec(
        "asset_turnover", FAMILY_EFFICIENCY, +1, 1.0,
        "revenue_ttm / avg(total_assets_latest, total_assets_prior)",
        "Revenue per dollar of average assets. Not meaningful for financials.",
        ("revenue", "total_assets"),
        _FIN,
    ),
    FeatureSpec(
        "opex_ratio", FAMILY_EFFICIENCY, -1, 1.0,
        "(sga_ttm + r_and_d_ttm) / revenue_ttm (revenue > 0)",
        "Operating spend per dollar of revenue; lower is better.",
        ("sga", "r_and_d", "revenue"),
    ),
    FeatureSpec(
        "capex_intensity", FAMILY_EFFICIENCY, -1, 1.0,
        "abs(capex_ttm) / revenue_ttm (revenue > 0)",
        "Capital spend per dollar of revenue; lower is better.",
        ("capex", "revenue"),
    ),
    # --- Leverage ---------------------------------------------------------
    FeatureSpec(
        "net_debt_to_ebitda", FAMILY_LEVERAGE, -1, 1.0,
        "(total_debt - cash_and_equivalents - short_term_investments) / ebitda_ttm (ebitda > 0)",
        "Net debt in years of EBITDA; lower is better.",
        ("total_debt", "short_term_debt", "long_term_debt", "cash_and_equivalents",
         "short_term_investments", "ebitda", "operating_income", "depreciation_and_amortization"),
        _FIN,
    ),
    FeatureSpec(
        "debt_to_equity", FAMILY_LEVERAGE, -1, 1.0,
        "total_debt / shareholders_equity (equity > 0)",
        "Gross debt to book equity; lower is better.",
        ("total_debt", "short_term_debt", "long_term_debt", "shareholders_equity"),
        _FIN,
    ),
    FeatureSpec(
        "interest_coverage", FAMILY_LEVERAGE, +1, 1.0,
        "min(ebit_ttm / abs(interest_expense_ttm), 50) (interest != 0)",
        "Times interest earned, capped at 50 so near-zero interest does not dominate.",
        ("ebit", "operating_income", "interest_expense"),
        _FIN,
    ),
    FeatureSpec(
        "current_ratio", FAMILY_LEVERAGE, +1, 1.0,
        "current_assets / current_liabilities (liabilities > 0)",
        "Current ratio. Not meaningful for financials or utilities.",
        ("current_assets", "current_liabilities"),
        _FIN_UTIL,
    ),
    # --- Capital allocation ----------------------------------------------
    FeatureSpec(
        "shareholder_yield", FAMILY_CAPITAL_ALLOCATION, +1, 1.0,
        "(abs(dividends_paid_ttm) + abs(share_repurchases_ttm)) / market_cap",
        "Cash returned to owners per dollar of market value.",
        ("dividends_paid", "share_repurchases", "price", "shares"),
    ),
    FeatureSpec(
        "net_share_change_1y", FAMILY_CAPITAL_ALLOCATION, -1, 1.0,
        "weighted_avg_shares_diluted_ttm / weighted_avg_shares_diluted_prior - 1",
        "Dilution (positive) or net buybacks (negative); lower is better.",
        ("weighted_avg_shares_diluted",),
    ),
    FeatureSpec(
        "sbc_to_revenue", FAMILY_CAPITAL_ALLOCATION, -1, 1.0,
        "stock_based_compensation_ttm / revenue_ttm (revenue > 0)",
        "Stock compensation per dollar of revenue; lower is better.",
        ("stock_based_compensation", "revenue"),
    ),
    FeatureSpec(
        "goodwill_to_assets", FAMILY_CAPITAL_ALLOCATION, -1, 1.0,
        "goodwill / total_assets (assets > 0)",
        "Share of the balance sheet that is acquired goodwill; lower is better.",
        ("goodwill", "total_assets"),
    ),
    # --- Earnings quality -------------------------------------------------
    FeatureSpec(
        "accruals_ratio", FAMILY_EARNINGS_QUALITY, -1, 1.0,
        "(net_income_ttm - cash_from_operations_ttm) / avg(total_assets)",
        "Earnings not backed by operating cash, scaled by assets; lower is better.",
        ("net_income", "cash_from_operations", "total_assets"),
    ),
    FeatureSpec(
        "cash_conversion", FAMILY_EARNINGS_QUALITY, +1, 1.0,
        "clip(cash_from_operations_ttm / net_income_ttm, -3, 3) (net_income > 0)",
        "Operating cash per dollar of net income.",
        ("cash_from_operations", "net_income"),
    ),
    FeatureSpec(
        "fcf_to_net_income", FAMILY_EARNINGS_QUALITY, +1, 1.0,
        "clip(free_cash_flow_ttm / net_income_ttm, -3, 3) (net_income > 0)",
        "Free cash flow per dollar of net income.",
        ("free_cash_flow", "net_income"),
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_SPEC)
FEATURES_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURE_SPEC}

# Sub-composites for the research process (computed by the normaliser,
# never weighted into the overall score).
COMPOUNDER_PROFILE_FAMILIES: tuple[str, ...] = (
    FAMILY_QUALITY, FAMILY_PROFITABILITY, FAMILY_CAPITAL_ALLOCATION, FAMILY_EARNINGS_QUALITY,
)
INFLECTION_PROFILE_FEATURES: tuple[str, ...] = ("revenue_growth_accel", "operating_margin_change_1y")
INFLECTION_PROFILE_FAMILIES: tuple[str, ...] = (FAMILY_GROWTH,)


def features_by_family(spec: tuple[FeatureSpec, ...] = FEATURE_SPEC) -> dict[str, tuple[FeatureSpec, ...]]:
    out: dict[str, list[FeatureSpec]] = {name: [] for name in FAMILY_NAMES}
    for f in spec:
        out.setdefault(f.family, []).append(f)
    return {k: tuple(v) for k, v in out.items()}


def get_feature(name: str) -> FeatureSpec:
    return FEATURES_BY_NAME[name]


def applicable_features(sector: str | None, spec: tuple[FeatureSpec, ...] = FEATURE_SPEC) -> frozenset[str]:
    """Feature names that apply to a name in ``sector`` (canonical name or
    None). An unmatched / unknown sector gets every feature: we cannot
    exclude on the basis of a label we could not read, and the row's
    notes say the sector was unmatched."""
    if sector is None:
        return frozenset(f.name for f in spec)
    return frozenset(f.name for f in spec if sector not in f.exclude_sectors)


# ---------------------------------------------------------------------------
# Normalization parameters (methodology, part of the hash)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizationParams:
    """Cross-sectional normalization settings.

    * ``winsor_pct`` — each feature is clamped to its [pct, 1-pct]
      percentiles across the applicable universe before z-scoring.
    * ``clip_z`` — z-scores are clipped to ±clip_z after standardization.
    * ``sector_neutral`` / ``min_sector_n`` — z is taken against the
      sector's own mean/stdev when the sector has at least ``min_sector_n``
      scored names, else against the universe (recorded per row). Sized
      for a 100-600 name universe: the curated list is ~170 names today,
      so 5 keeps most of the 11 sectors neutralised rather than pushing
      half of them to universe-relative.
    * ``min_feature_coverage`` — a family composite is null when fewer
      than this share of its *applicable* features are available.
    * ``min_categories`` — the overall composite is null when fewer than
      this many family composites are available.
    """
    winsor_pct: float = 0.025
    clip_z: float = 3.0
    sector_neutral: bool = True
    min_sector_n: int = 5
    min_feature_coverage: float = 0.5
    min_categories: int = 5


NORMALIZATION = NormalizationParams()


# ---------------------------------------------------------------------------
# Methodology rules that no formula string shows (part of the hash)
# ---------------------------------------------------------------------------

# Plain data so it serialises as-is into ``spec_as_dict`` and reaches the
# ``/api/scorecard/spec`` reader verbatim. Changing a rule here changes the
# hash, which is the point: a row scored under a different null policy or
# ranking population is not comparable to one scored under this one.
RULES: dict[str, Any] = {
    "missing_inputs": {
        "policy": "null_with_reason",
        "partner_line_zero_fill": False,
        "detail": (
            "Every line a formula names must be present in the point-in-time "
            "snapshot; an absent line makes the feature null with a "
            "missing:<line> reason. No line is assumed to be zero because a "
            "sibling line is present (r_and_d/sga, share_repurchases/"
            "dividends_paid, short_term_investments/cash_and_equivalents, "
            "short_term_debt/long_term_debt). A provider that reports 0 "
            "stores 0 and scores as 0; a provider that omits the line scores "
            "as n/a."
        ),
        "total_debt_fallback": "short_term_debt + long_term_debt only when both are reported",
    },
    "percentiles": {
        "method": "average_rank",
        "scale": "(0, 100], 100 = best",
        "overall_population": "rows with a non-null overall z",
        "category_population": "rows with a non-null z for that family",
        "sector_population": "same rule within the canonical sector, at least 2 names",
    },
}


# ---------------------------------------------------------------------------
# Serialisation and fingerprint
# ---------------------------------------------------------------------------

def spec_as_dict(
    *,
    features: tuple[FeatureSpec, ...] = FEATURE_SPEC,
    families: tuple[FamilySpec, ...] = FAMILIES,
    normalization: NormalizationParams = NORMALIZATION,
    version_key: str = VERSION_KEY,
) -> dict[str, Any]:
    """JSON-able description of the methodology, grouped by family in
    declaration order. This is what ``/api/scorecard/spec`` returns and
    what ``spec_hash`` fingerprints."""
    by_family = features_by_family(features)
    return {
        "version_key": version_key,
        "families": [
            {
                "name": fam.name,
                "weight": fam.weight,
                "description": fam.description,
                "features": [
                    {
                        "name": f.name,
                        "sign": f.sign,
                        "weight": f.weight,
                        "formula": f.formula,
                        "description": f.description,
                        "inputs": list(f.inputs),
                        "applicability": {"exclude_sectors": list(f.exclude_sectors)},
                    }
                    for f in by_family.get(fam.name, ())
                ],
            }
            for fam in families
        ],
        "normalization": asdict(normalization),
        "rules": json.loads(json.dumps(RULES)),   # a copy; callers may mutate the dict
        "sectors": {
            "canonical": list(CANONICAL_SECTORS),
            "aliases": dict(SECTOR_ALIASES),
        },
        "profiles": {
            "compounder": {"families": list(COMPOUNDER_PROFILE_FAMILIES)},
            "inflection": {
                "families": list(INFLECTION_PROFILE_FAMILIES),
                "features": list(INFLECTION_PROFILE_FEATURES),
            },
        },
    }


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN. Dict
    insertion order never reaches the hash, so two builds of the same
    spec agree byte-for-byte."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def hash_spec_dict(spec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def spec_hash(spec: dict[str, Any] | None = None) -> str:
    """sha256 of the canonical JSON of the spec (the in-code one by default)."""
    return hash_spec_dict(spec if spec is not None else spec_as_dict())


# ---------------------------------------------------------------------------
# Self-checks (cheap; run at import so a bad edit fails loudly)
# ---------------------------------------------------------------------------

def _validate() -> None:
    names = [f.name for f in FEATURE_SPEC]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in FEATURE_SPEC")
    for f in FEATURE_SPEC:
        if f.family not in FAMILY_WEIGHTS:
            raise ValueError(f"feature {f.name!r} names unknown family {f.family!r}")
        if f.sign not in (1, -1):
            raise ValueError(f"feature {f.name!r} sign must be +1 or -1")
        if f.weight <= 0:
            raise ValueError(f"feature {f.name!r} weight must be positive")
        for s in f.exclude_sectors:
            if s not in CANONICAL_SECTORS:
                raise ValueError(f"feature {f.name!r} excludes non-canonical sector {s!r}")
    for fam in FAMILY_NAMES:
        if not any(f.family == fam for f in FEATURE_SPEC):
            raise ValueError(f"family {fam!r} has no features")
    for alias, canon in SECTOR_ALIASES.items():
        if canon not in CANONICAL_SECTORS or alias != _sector_key(alias):
            raise ValueError(f"bad sector alias {alias!r} -> {canon!r}")


_validate()
