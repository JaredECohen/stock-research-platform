"""Fundamental Factor Scorecard — cross-sectional normalization (``fs-v1``).

Pure, stdlib ``statistics`` only: the universe is a few hundred names, so
numpy buys nothing here and keeping it out means the web process can
import this module without paying for numpy.

Pipeline per as-of run (``normalize_universe``):

1. **Winsorize** each feature's applicable, non-null raw values at the
   ``[pct, 1-pct]`` percentiles of the universe (linear interpolation on
   the sorted sample). Tails are clamped, not dropped, so a name in the
   tail still gets the extreme z rather than disappearing.
2. **Sector-neutral z**: within each canonical sector that has at least
   ``min_sector_n`` scored names, ``z = (v - mean_sector) / pstdev_sector``;
   smaller sectors and names whose sector string did not match the alias
   table fall back to the universe mean/stdev, and the row records which
   basis was used. An unneutralised universe z is kept alongside for the
   UI toggle.
3. **Clip** to ``±clip_z`` and apply the feature's **sign** so every stored
   z reads "higher = better".
4. **Composites**: family z = weighted mean of the family's available
   feature z (null when fewer than ``min_feature_coverage`` of the
   *applicable* features are available); overall z = weighted mean of the
   available family z (null when fewer than ``min_categories`` families are
   available). Excluded features never count against coverage.
5. **Contributions**: each available feature's share of the overall z,
   with the weights renormalised over what was actually available so the
   contributions sum exactly to the overall z. This is what lets the memo
   say "the score is where it is because of X, Y and Z" honestly.
6. **Percentiles**: rank-based across rows with a non-null overall z
   (universe) and within the canonical sector; ties share the average
   rank so the result does not depend on input order.

Nothing here invents a value: a feature with no raw value has no z, a
family with too few features has no z, and a row with too few families
has no overall — it is still returned (with its coverage) so the UI can
say "insufficient data" instead of silently dropping the name.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

# `_z_to_100` is the app's one z→0-100 mapping (50 = median, ±2.5 z = 0/100).
# Sharing it keeps the scorecard's `overall_score` on the same scale as the
# memo's factor scores rather than introducing a second, drifting mapping.
from app.finance.factor_scores import _z_to_100
from app.finance.scorecard_spec import (
    COMPOUNDER_PROFILE_FAMILIES,
    FAMILY_NAMES,
    FAMILY_WEIGHTS,
    FEATURE_SPEC,
    INFLECTION_PROFILE_FAMILIES,
    INFLECTION_PROFILE_FEATURES,
    NORMALIZATION,
    FeatureSpec,
    NormalizationParams,
    applicable_features,
    normalize_sector,
)

BASIS_SECTOR = "sector"                    # sector:<canonical name>
BASIS_UNIVERSE = "universe"
BASIS_UNIVERSE_SMALL_SECTOR = "universe:sector_small"
BASIS_UNIVERSE_UNMATCHED = "universe:sector_unmatched"
BASIS_UNIVERSE_NOT_NEUTRAL = "universe:sector_neutral_off"
BASIS_NA = "n/a"                           # n/a:<reason>


# ---------------------------------------------------------------------------
# Primitive statistics
# ---------------------------------------------------------------------------

def _finite_values(values: Iterable[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def interpolated_percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default / Hyndman-Fan type 7)
    on an already-sorted sample; ``p`` in [0, 1]."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of an empty sample")
    if n == 1:
        return sorted_values[0]
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def winsor_bounds(values: Iterable[float], pct: float) -> tuple[float, float]:
    """The ``[pct, 1-pct]`` interpolated percentiles of ``values``."""
    if not 0.0 <= pct < 0.5:
        raise ValueError("winsor pct must be in [0, 0.5)")
    s = sorted(_finite_values(values))
    if not s:
        raise ValueError("winsor bounds of an empty sample")
    return interpolated_percentile(s, pct), interpolated_percentile(s, 1.0 - pct)


def winsorize(values: Sequence[float | None], pct: float = NORMALIZATION.winsor_pct) -> list[float | None]:
    """Clamp every non-null value to the sample's ``[pct, 1-pct]``
    percentiles. Nulls pass through as nulls (they are not in the sample
    and they get no value)."""
    clean = _finite_values(values)
    if not clean:
        return [None for _ in values]
    lo, hi = winsor_bounds(clean, pct)
    return [None if v is None or not math.isfinite(v) else min(hi, max(lo, float(v))) for v in values]


def zscores(values: Sequence[float | None]) -> list[float | None]:
    """Population z-scores against the non-null sample. Fewer than two
    observations cannot be standardised (None); a zero-dispersion sample
    puts every name at its mean (0.0), which is the truth, not a filler."""
    clean = _finite_values(values)
    if len(clean) < 2:
        return [None for _ in values]
    m, s = mean(clean), pstdev(clean)
    out: list[float | None] = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(None)
        elif s == 0:
            out.append(0.0)
        else:
            out.append((float(v) - m) / s)
    return out


def clip_z(z: float | None, limit: float = NORMALIZATION.clip_z) -> float | None:
    if z is None:
        return None
    return max(-limit, min(limit, z))


def sector_neutral_z(
    feature_by_ticker: Mapping[str, float | None],
    sector_by_ticker: Mapping[str, str | None],
    *,
    min_n: int = NORMALIZATION.min_sector_n,
    sector_neutral: bool = True,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Standardise one feature across the universe.

    ``sector_by_ticker`` holds *canonical* sector names (None = unmatched).
    Returns ``(z_by_ticker, basis_by_ticker)`` where the basis says which
    sample the name was standardised against:

    * ``sector:<name>`` — the sector had ``>= min_n`` non-null values;
    * ``universe:sector_small`` — the sector had fewer, universe stats used;
    * ``universe:sector_unmatched`` — no canonical sector, universe used;
    * ``universe:sector_neutral_off`` — neutralisation disabled by params;
    * ``n/a:<reason>`` — no value, or the universe sample is too small.
    """
    tickers = list(feature_by_ticker.keys())
    universe_vals = _finite_values(feature_by_ticker[t] for t in tickers)
    universe_ok = len(universe_vals) >= 2
    u_mean = mean(universe_vals) if universe_ok else 0.0
    u_sd = pstdev(universe_vals) if universe_ok else 0.0

    # Sector samples over the same non-null population.
    by_sector: dict[str, list[float]] = {}
    for t in tickers:
        v = feature_by_ticker[t]
        s = sector_by_ticker.get(t)
        if s is not None and v is not None and math.isfinite(v):
            by_sector.setdefault(s, []).append(float(v))
    sector_stats: dict[str, tuple[float, float]] = {
        s: (mean(vals), pstdev(vals)) for s, vals in by_sector.items() if len(vals) >= min_n
    }

    z_out: dict[str, float | None] = {}
    basis: dict[str, str] = {}
    for t in tickers:
        v = feature_by_ticker[t]
        if v is None or not math.isfinite(v):
            z_out[t] = None
            basis[t] = f"{BASIS_NA}:no_value"
            continue
        s = sector_by_ticker.get(t)
        if sector_neutral and s is not None and s in sector_stats:
            m, sd = sector_stats[s]
            z_out[t] = 0.0 if sd == 0 else (float(v) - m) / sd
            basis[t] = f"{BASIS_SECTOR}:{s}"
            continue
        if not universe_ok:
            z_out[t] = None
            basis[t] = f"{BASIS_NA}:universe_n<2"
            continue
        z_out[t] = 0.0 if u_sd == 0 else (float(v) - u_mean) / u_sd
        if not sector_neutral:
            basis[t] = BASIS_UNIVERSE_NOT_NEUTRAL
        elif s is None:
            basis[t] = BASIS_UNIVERSE_UNMATCHED
        else:
            basis[t] = BASIS_UNIVERSE_SMALL_SECTOR
    return z_out, basis


def rank_percentiles(values_by_key: Mapping[str, float | None]) -> dict[str, float | None]:
    """Rank-based percentile in (0, 100]: ``100 * average_rank / N`` over the
    non-null values, so the best name is 100, the worst is ``100 / N`` and
    tied values share the same percentile regardless of input order.
    Nulls stay null."""
    items = [(k, float(v)) for k, v in values_by_key.items() if v is not None and math.isfinite(v)]
    out: dict[str, float | None] = {k: None for k in values_by_key}
    n = len(items)
    if n == 0:
        return out
    ordered = sorted(items, key=lambda kv: kv[1])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0   # 1-based ranks i+1 .. j+1
        pct = 100.0 * avg_rank / n
        for k in range(i, j + 1):
            out[ordered[k][0]] = pct
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# Composites and contributions
# ---------------------------------------------------------------------------

@dataclass
class CompositeResult:
    category_z: dict[str, float | None]
    category_coverage: dict[str, float | None]   # available / applicable per family; None when nothing applies
    overall_z: float | None
    coverage: float                              # available / applicable across all families
    n_available: int
    n_applicable: int
    contributions: dict[str, float] = field(default_factory=dict)


def composites(
    feature_z: Mapping[str, float | None],
    *,
    applicable: Iterable[str] | None = None,
    spec: tuple[FeatureSpec, ...] = FEATURE_SPEC,
    family_weights: Mapping[str, float] = FAMILY_WEIGHTS,
    params: NormalizationParams = NORMALIZATION,
) -> CompositeResult:
    """Family and overall composites from signed, clipped feature z.

    ``applicable`` is the set of feature names that apply to this name
    (excluded features are left out of both numerator and denominator).
    Defaults to every feature in ``spec``.
    """
    applicable_set = frozenset(applicable) if applicable is not None else frozenset(f.name for f in spec)
    families: dict[str, list[FeatureSpec]] = {}
    for f in spec:
        families.setdefault(f.family, []).append(f)

    category_z: dict[str, float | None] = {}
    category_coverage: dict[str, float | None] = {}
    available_by_family: dict[str, list[FeatureSpec]] = {}
    n_available = n_applicable = 0
    for fam in family_weights:
        feats = [f for f in families.get(fam, ()) if f.name in applicable_set]
        avail = [f for f in feats if feature_z.get(f.name) is not None]
        n_applicable += len(feats)
        n_available += len(avail)
        if not feats:
            category_z[fam] = None
            category_coverage[fam] = None
            continue
        cov = len(avail) / len(feats)
        category_coverage[fam] = cov
        if cov < params.min_feature_coverage or not avail:
            category_z[fam] = None
            continue
        available_by_family[fam] = avail
        w_total = sum(f.weight for f in avail)
        category_z[fam] = sum(f.weight * float(feature_z[f.name] or 0.0) for f in avail) / w_total

    scored = [fam for fam, z in category_z.items() if z is not None]
    coverage = (n_available / n_applicable) if n_applicable else 0.0
    if len(scored) < params.min_categories:
        return CompositeResult(category_z, category_coverage, None, coverage, n_available, n_applicable, {})

    w_cat_total = sum(family_weights[fam] for fam in scored)
    overall = sum(family_weights[fam] * float(category_z[fam] or 0.0) for fam in scored) / w_cat_total

    # Renormalised weights: (w_c / W_C) * (w_f / W_Vc) * z_f sums to overall
    # exactly, so a reader can add the contributions back up.
    contributions: dict[str, float] = {}
    for fam in scored:
        avail = available_by_family[fam]
        w_feat_total = sum(f.weight for f in avail)
        cat_share = family_weights[fam] / w_cat_total
        for f in avail:
            contributions[f.name] = cat_share * (f.weight / w_feat_total) * float(feature_z[f.name] or 0.0)
    return CompositeResult(category_z, category_coverage, overall, coverage, n_available, n_applicable, contributions)


def top_contributors(contributions: Mapping[str, float], k: int = 3) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """``(top_positive, top_negative)``: the ``k`` largest positive and the
    ``k`` most negative contributions, each as ``(feature, value)``. Ties
    break on feature name so the output is deterministic."""
    items = sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0]))
    pos = [(n, v) for n, v in items if v > 0][:k]
    neg = sorted(((n, v) for n, v in items if v < 0), key=lambda kv: (kv[1], kv[0]))[:k]
    return pos, neg


def profiles(
    category_z: Mapping[str, float | None],
    feature_z: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Research-process sub-composites (never weighted into overall):

    * ``compounder`` — mean of the quality, profitability, capital
      allocation and earnings-quality family z (needs at least 2 of 4);
    * ``inflection`` — mean of the growth family z and the two inflection
      features (needs at least 2 of 3).
    """
    comp = [category_z.get(f) for f in COMPOUNDER_PROFILE_FAMILIES]
    comp_vals = [v for v in comp if v is not None]
    infl = [category_z.get(f) for f in INFLECTION_PROFILE_FAMILIES] + [feature_z.get(f) for f in INFLECTION_PROFILE_FEATURES]
    infl_vals = [v for v in infl if v is not None]
    return {
        "compounder": mean(comp_vals) if len(comp_vals) >= 2 else None,
        "inflection": mean(infl_vals) if len(infl_vals) >= 2 else None,
    }


# ---------------------------------------------------------------------------
# Universe-level driver
# ---------------------------------------------------------------------------

@dataclass
class TickerNormalization:
    ticker: str
    sector_raw: str | None
    sector: str | None                         # canonical, None when unmatched
    feature_z: dict[str, float | None]         # sector-neutral (or fallback), signed, clipped
    feature_z_universe: dict[str, float | None]  # unneutralised, signed, clipped
    feature_basis: dict[str, str]
    category_z: dict[str, float | None]
    category_z_universe: dict[str, float | None]
    category_coverage: dict[str, float | None]
    overall_z: float | None
    overall_z_universe: float | None
    overall_score: float | None                # 0-100 via factor_scores._z_to_100
    coverage: float
    n_available: int
    n_applicable: int
    contributions: dict[str, float]
    percentile_universe: float | None = None
    percentile_sector: float | None = None
    profiles: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class UniverseNormalization:
    rows: dict[str, TickerNormalization]
    notes: dict[str, Any]
    params: NormalizationParams


def normalize_universe(
    raw_by_ticker: Mapping[str, Mapping[str, float | None]],
    sector_by_ticker: Mapping[str, str | None],
    *,
    applicable_by_ticker: Mapping[str, Iterable[str]] | None = None,
    spec: tuple[FeatureSpec, ...] = FEATURE_SPEC,
    family_weights: Mapping[str, float] = FAMILY_WEIGHTS,
    params: NormalizationParams = NORMALIZATION,
) -> UniverseNormalization:
    """Run the whole normalisation for one as-of cross-section.

    ``raw_by_ticker`` holds the raw feature values from
    ``scorecard_features.compute_features`` (None where not computable);
    ``sector_by_ticker`` holds the provider sector strings, normalised here.
    ``applicable_by_ticker`` may carry the per-ticker applicable sets from
    ``compute_features_detailed``; when omitted they are derived from the
    canonical sector, which gives the same answer.
    """
    tickers = list(raw_by_ticker.keys())
    sector_raw = {t: sector_by_ticker.get(t) for t in tickers}
    sector_canon = {t: normalize_sector(sector_raw[t]) for t in tickers}
    applicable: dict[str, frozenset[str]] = {}
    for t in tickers:
        if applicable_by_ticker is not None and t in applicable_by_ticker:
            applicable[t] = frozenset(applicable_by_ticker[t])
        else:
            applicable[t] = applicable_features(sector_canon[t], spec)

    unmatched: dict[str, int] = {}
    for t in tickers:
        raw = sector_raw[t]
        if sector_canon[t] is None:
            key = raw if raw else ""
            unmatched[key] = unmatched.get(key, 0) + 1

    # Per-feature: winsorize → z (sector-neutral and universe) → clip → sign.
    feature_z: dict[str, dict[str, float | None]] = {t: {} for t in tickers}
    feature_z_u: dict[str, dict[str, float | None]] = {t: {} for t in tickers}
    feature_basis: dict[str, dict[str, str]] = {t: {} for t in tickers}
    feature_n: dict[str, int] = {}
    sectors_small: dict[str, int] = {}
    for f in spec:
        vals: dict[str, float | None] = {}
        for t in tickers:
            v = raw_by_ticker[t].get(f.name) if f.name in applicable[t] else None
            vals[t] = float(v) if v is not None and math.isfinite(v) else None
        wins = winsorize([vals[t] for t in tickers], params.winsor_pct)
        wins_by_t = dict(zip(tickers, wins))
        feature_n[f.name] = sum(1 for v in wins if v is not None)
        z_sn, basis = sector_neutral_z(
            wins_by_t, sector_canon, min_n=params.min_sector_n, sector_neutral=params.sector_neutral,
        )
        z_un, _ = sector_neutral_z(wins_by_t, sector_canon, min_n=params.min_sector_n, sector_neutral=False)
        for t in tickers:
            zs, zu = clip_z(z_sn[t], params.clip_z), clip_z(z_un[t], params.clip_z)
            feature_z[t][f.name] = None if zs is None else f.sign * zs
            feature_z_u[t][f.name] = None if zu is None else f.sign * zu
            b = basis[t]
            if f.name not in applicable[t]:
                b = f"{BASIS_NA}:excluded"
            feature_basis[t][f.name] = b
            canon = sector_canon[t]
            if b == BASIS_UNIVERSE_SMALL_SECTOR and canon is not None:
                sectors_small[canon] = sectors_small.get(canon, 0) + 1

    rows: dict[str, TickerNormalization] = {}
    for t in tickers:
        comp = composites(feature_z[t], applicable=applicable[t], spec=spec, family_weights=family_weights, params=params)
        comp_u = composites(feature_z_u[t], applicable=applicable[t], spec=spec, family_weights=family_weights, params=params)
        notes: list[str] = []
        if sector_raw[t] and sector_canon[t] is None:
            notes.append(f"sector_unmatched:{sector_raw[t]}")
        elif not sector_raw[t]:
            notes.append("sector_missing")
        if comp.overall_z is None:
            scored = sum(1 for z in comp.category_z.values() if z is not None)
            notes.append(f"insufficient_data:categories={scored}<{params.min_categories}")
        rows[t] = TickerNormalization(
            ticker=t,
            sector_raw=sector_raw[t],
            sector=sector_canon[t],
            feature_z=feature_z[t],
            feature_z_universe=feature_z_u[t],
            feature_basis=feature_basis[t],
            category_z=comp.category_z,
            category_z_universe=comp_u.category_z,
            category_coverage=comp.category_coverage,
            overall_z=comp.overall_z,
            overall_z_universe=comp_u.overall_z,
            overall_score=None if comp.overall_z is None else _z_to_100(comp.overall_z),
            coverage=comp.coverage,
            n_available=comp.n_available,
            n_applicable=comp.n_applicable,
            contributions=comp.contributions,
            profiles=profiles(comp.category_z, feature_z[t]),
            notes=notes,
        )

    # Percentiles among rows that actually have an overall z.
    pct_u = rank_percentiles({t: rows[t].overall_z for t in tickers})
    by_sector: dict[str, dict[str, float | None]] = {}
    for t in tickers:
        s = sector_canon[t]
        if s is not None and rows[t].overall_z is not None:
            by_sector.setdefault(s, {})[t] = rows[t].overall_z
    pct_s: dict[str, float | None] = {}
    for group in by_sector.values():
        if len(group) >= 2:
            pct_s.update(rank_percentiles(group))
    for t in tickers:
        rows[t].percentile_universe = pct_u.get(t)
        rows[t].percentile_sector = pct_s.get(t)

    universe_notes: dict[str, Any] = {
        "n_tickers": len(tickers),
        "n_scored": sum(1 for r in rows.values() if r.overall_z is not None),
        "n_insufficient": sum(1 for r in rows.values() if r.overall_z is None),
        "sectors_unmatched": unmatched,
        # feature-rows that fell back to universe stats because the sector
        # had fewer than min_sector_n names for that feature
        "sector_fallback_feature_rows": sectors_small,
        "feature_n": feature_n,
        "sector_counts": _counts(sector_canon.values()),
    }
    return UniverseNormalization(rows=rows, notes=universe_notes, params=params)


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = v if v is not None else ""
        out[key] = out.get(key, 0) + 1
    return out


__all__ = [
    "BASIS_NA",
    "BASIS_SECTOR",
    "BASIS_UNIVERSE",
    "BASIS_UNIVERSE_NOT_NEUTRAL",
    "BASIS_UNIVERSE_SMALL_SECTOR",
    "BASIS_UNIVERSE_UNMATCHED",
    "CompositeResult",
    "FAMILY_NAMES",
    "TickerNormalization",
    "UniverseNormalization",
    "clip_z",
    "composites",
    "interpolated_percentile",
    "normalize_universe",
    "profiles",
    "rank_percentiles",
    "sector_neutral_z",
    "top_contributors",
    "winsor_bounds",
    "winsorize",
    "zscores",
]
