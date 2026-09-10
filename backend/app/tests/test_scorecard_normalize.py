"""Scorecard normalization (fs-v1): winsor exactness, sector-neutral z with
the min-n fallback, clipping, rank percentiles, composites with coverage
thresholds and the contributions-sum-to-overall property.

Pure module — no DB, no numpy. Random universes use ``random.Random(0)``.
"""
from __future__ import annotations

import math
import random
from statistics import mean, pstdev

import pytest

from app.finance import scorecard_normalize as N
from app.finance import scorecard_spec as S
from app.finance.scorecard_spec import NormalizationParams

# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


def test_interpolated_percentile_matches_type7_definition():
    s = [1.0, 2.0, 3.0, 4.0]
    assert N.interpolated_percentile(s, 0.0) == 1.0
    assert N.interpolated_percentile(s, 1.0) == 4.0
    assert N.interpolated_percentile(s, 0.5) == 2.5
    assert N.interpolated_percentile(s, 0.25) == 1.75     # pos = 0.75 → 1 + 0.75
    assert N.interpolated_percentile([7.0], 0.9) == 7.0
    with pytest.raises(ValueError):
        N.interpolated_percentile([], 0.5)


def test_winsorize_clamps_exactly_at_the_interpolated_percentiles():
    values = [float(i) for i in range(1, 41)]          # 1..40, n = 40
    # pos_lo = 0.025 * 39 = 0.975 → 1 + 0.975 = 1.975; pos_hi = 38.025 → 39.025
    lo, hi = N.winsor_bounds(values, 0.025)
    assert lo == pytest.approx(1.975)
    assert hi == pytest.approx(39.025)
    out = N.winsorize(values, 0.025)
    assert out[0] == pytest.approx(1.975)
    assert out[-1] == pytest.approx(39.025)
    assert out[1:-1] == values[1:-1]                    # interior untouched


def test_winsorize_passes_nulls_through_and_ignores_them_in_the_sample():
    values = [None, 100.0, 1.0, 2.0, 3.0, None, 4.0]
    out = N.winsorize(values, 0.25)
    assert out[0] is None and out[5] is None
    lo, hi = N.winsor_bounds([100.0, 1.0, 2.0, 3.0, 4.0], 0.25)
    assert out[1] == pytest.approx(hi)
    assert out[2] == pytest.approx(lo)
    assert N.winsorize([None, None]) == [None, None]


def test_winsorize_rejects_bad_pct():
    with pytest.raises(ValueError):
        N.winsor_bounds([1.0, 2.0], 0.5)


# ---------------------------------------------------------------------------
# zscores / clip
# ---------------------------------------------------------------------------


def test_zscores_hand_computed_and_null_safe():
    vals = [1.0, 2.0, None, 3.0, 4.0]
    z = N.zscores(vals)
    sd = pstdev([1, 2, 3, 4])
    assert z[2] is None
    assert z[0] == pytest.approx((1 - 2.5) / sd)
    assert z[4] == pytest.approx((4 - 2.5) / sd)
    assert N.zscores([5.0]) == [None]                   # cannot standardise one point
    assert N.zscores([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]  # zero dispersion = at the mean


def test_clip_z():
    assert N.clip_z(4.2) == 3.0
    assert N.clip_z(-9.0) == -3.0
    assert N.clip_z(1.5) == 1.5
    assert N.clip_z(None) is None
    assert N.clip_z(4.2, limit=2.0) == 2.0


# ---------------------------------------------------------------------------
# sector_neutral_z
# ---------------------------------------------------------------------------


def _universe(n_tech=6, n_fin=3):
    feat = {}
    sector = {}
    for i in range(n_tech):
        feat[f"T{i}"] = float(i)                         # 0..5
        sector[f"T{i}"] = S.SECTOR_INFORMATION_TECHNOLOGY
    for i in range(n_fin):
        feat[f"F{i}"] = 10.0 + i                          # 10, 11, 12
        sector[f"F{i}"] = S.SECTOR_FINANCIALS
    feat["U0"] = 100.0
    sector["U0"] = None                                    # unmatched sector
    feat["X0"] = None
    sector["X0"] = S.SECTOR_INFORMATION_TECHNOLOGY         # no value
    return feat, sector


def test_sector_neutral_uses_sector_stats_when_n_at_least_min():
    feat, sector = _universe()
    z, basis = N.sector_neutral_z(feat, sector, min_n=5)
    tech = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert z["T5"] == pytest.approx((5.0 - mean(tech)) / pstdev(tech))
    assert basis["T5"] == "sector:Information Technology"


def test_small_sector_and_unmatched_fall_back_to_universe_stats():
    feat, sector = _universe()
    z, basis = N.sector_neutral_z(feat, sector, min_n=5)
    universe = [0, 1, 2, 3, 4, 5, 10, 11, 12, 100]
    assert basis["F0"] == N.BASIS_UNIVERSE_SMALL_SECTOR
    assert z["F0"] == pytest.approx((10.0 - mean(universe)) / pstdev(universe))
    assert basis["U0"] == N.BASIS_UNIVERSE_UNMATCHED
    assert z["U0"] == pytest.approx((100.0 - mean(universe)) / pstdev(universe))
    assert z["X0"] is None and basis["X0"] == "n/a:no_value"


def test_min_n_boundary_is_inclusive():
    feat, sector = _universe(n_fin=5)
    z, basis = N.sector_neutral_z(feat, sector, min_n=5)
    assert basis["F0"] == "sector:Financials"
    _, basis6 = N.sector_neutral_z(feat, sector, min_n=6)
    assert basis6["F0"] == N.BASIS_UNIVERSE_SMALL_SECTOR


def test_sector_neutral_off_uses_universe_for_everyone():
    feat, sector = _universe()
    z, basis = N.sector_neutral_z(feat, sector, min_n=5, sector_neutral=False)
    assert basis["T5"] == N.BASIS_UNIVERSE_NOT_NEUTRAL
    universe = [0, 1, 2, 3, 4, 5, 10, 11, 12, 100]
    assert z["T5"] == pytest.approx((5.0 - mean(universe)) / pstdev(universe))


def test_universe_too_small_is_na_not_zero():
    z, basis = N.sector_neutral_z({"A": 1.0, "B": None}, {"A": None, "B": None})
    assert z == {"A": None, "B": None}
    assert basis["A"] == "n/a:universe_n<2"


# ---------------------------------------------------------------------------
# rank_percentiles
# ---------------------------------------------------------------------------


def test_rank_percentiles_no_ties():
    assert N.rank_percentiles({"a": 3.0, "b": 1.0, "c": 4.0, "d": 2.0}) == {
        "a": 75.0, "b": 25.0, "c": 100.0, "d": 50.0,
    }


def test_rank_percentiles_ties_share_average_rank_and_ignore_order():
    p1 = N.rank_percentiles({"a": 1.0, "b": 2.0, "c": 2.0, "d": 5.0})
    p2 = N.rank_percentiles({"d": 5.0, "c": 2.0, "a": 1.0, "b": 2.0})
    assert p1 == p2
    assert p1["b"] == p1["c"] == pytest.approx(100.0 * 2.5 / 4)
    assert p1["a"] == 25.0 and p1["d"] == 100.0


def test_rank_percentiles_nulls_stay_null():
    p = N.rank_percentiles({"a": None, "b": 1.0, "c": 2.0})
    assert p == {"a": None, "b": 50.0, "c": 100.0}
    assert N.rank_percentiles({"a": None}) == {"a": None}
    assert N.rank_percentiles({}) == {}


# ---------------------------------------------------------------------------
# composites / contributions
# ---------------------------------------------------------------------------


def _full_z(value=1.0):
    return {name: value for name in S.FEATURE_NAMES}


def test_composites_equal_weights_everything_available():
    z = _full_z(1.0)
    z["opex_ratio"] = -1.0
    res = N.composites(z)
    by_family = S.features_by_family()
    # efficiency has 3 features: (1 - 1 + 1) / 3
    assert res.category_z["efficiency"] == pytest.approx(1 / 3)
    for fam in by_family:
        if fam != "efficiency":
            assert res.category_z[fam] == pytest.approx(1.0), fam
    assert res.overall_z == pytest.approx((7 * 1.0 + 1 / 3) / 8)
    assert res.coverage == 1.0 and res.n_available == res.n_applicable == 31
    assert res.contributions["roic"] == pytest.approx((1 / 8) * (1 / 4) * 1.0)
    assert res.contributions["opex_ratio"] == pytest.approx((1 / 8) * (1 / 3) * -1.0)
    assert sum(res.contributions.values()) == pytest.approx(res.overall_z)


def test_composites_ignore_nulls_and_renormalise():
    z = _full_z(2.0)
    z["roic"] = None                      # quality: 3 of 4 available → still scored
    z["cash_conversion"] = None           # earnings quality: 2 of 3 → scored
    res = N.composites(z)
    assert res.category_z["quality"] == pytest.approx(2.0)
    assert res.category_coverage["quality"] == pytest.approx(0.75)
    assert res.overall_z == pytest.approx(2.0)
    assert "roic" not in res.contributions
    assert res.contributions["roa"] == pytest.approx((1 / 8) * (1 / 3) * 2.0)
    assert res.coverage == pytest.approx(29 / 31)


def test_category_null_below_feature_coverage_threshold():
    z = _full_z(1.0)
    for name in ("earnings_yield", "fcf_yield", "ebitda_ev_yield"):
        z[name] = None                    # valuation: 1 of 4 = 25% < 50%
    res = N.composites(z)
    assert res.category_z["valuation"] is None
    assert res.category_coverage["valuation"] == pytest.approx(0.25)
    assert res.overall_z == pytest.approx(1.0)        # 7 categories remain
    assert "sales_ev_yield" not in res.contributions   # its family is not scored
    assert sum(res.contributions.values()) == pytest.approx(res.overall_z)
    # exactly at the threshold (2 of 4) counts as covered
    z["ebitda_ev_yield"] = 1.0
    assert N.composites(z).category_z["valuation"] == pytest.approx(1.0)


def test_overall_null_below_min_categories_but_coverage_recorded():
    z = _full_z(1.0)
    for fam in ("valuation", "quality", "growth", "profitability"):
        for f in S.features_by_family()[fam]:
            z[f.name] = None
    res = N.composites(z)                                 # 4 categories < 5
    assert res.overall_z is None
    assert res.contributions == {}
    assert res.n_available == 31 - 17 and res.n_applicable == 31
    assert res.category_z["leverage"] == pytest.approx(1.0)
    res5 = N.composites(z, params=NormalizationParams(min_categories=4))
    assert res5.overall_z == pytest.approx(1.0)


def test_excluded_features_do_not_count_against_coverage():
    applicable = S.applicable_features(S.SECTOR_FINANCIALS)
    z = {name: (1.0 if name in applicable else None) for name in S.FEATURE_NAMES}
    res = N.composites(z, applicable=applicable)
    assert res.coverage == 1.0
    assert res.n_applicable == len(applicable) == 23
    assert res.category_z["leverage"] is None            # nothing applies
    assert res.category_coverage["leverage"] is None
    assert res.category_z["valuation"] == pytest.approx(1.0)
    assert res.overall_z == pytest.approx(1.0)
    # ... whereas the same nulls without the mask read as missing data
    plain = N.composites(z)
    assert plain.coverage == pytest.approx(23 / 31)
    assert plain.category_coverage["leverage"] == 0.0


def test_contributions_sum_to_overall_z_random_property():
    rng = random.Random(0)
    for _ in range(200):
        z = {name: (rng.gauss(0, 1.5) if rng.random() > 0.3 else None) for name in S.FEATURE_NAMES}
        sector = rng.choice([None, S.SECTOR_FINANCIALS, S.SECTOR_UTILITIES, S.SECTOR_REAL_ESTATE])
        res = N.composites(z, applicable=S.applicable_features(sector))
        if res.overall_z is None:
            assert res.contributions == {}
            continue
        assert math.isclose(sum(res.contributions.values()), res.overall_z, rel_tol=0, abs_tol=1e-9)
        assert set(res.contributions) <= {n for n, v in z.items() if v is not None}


def test_top_contributors_deterministic():
    contrib = {"a": 0.3, "b": -0.2, "c": 0.3, "d": 0.1, "e": -0.5, "f": 0.0}
    pos, neg = N.top_contributors(contrib, k=2)
    assert pos == [("a", 0.3), ("c", 0.3)]                # tie broken by name
    assert neg == [("e", -0.5), ("b", -0.2)]
    assert N.top_contributors({}) == ([], [])


def test_profiles_need_at_least_two_inputs():
    cat = {"quality": 1.0, "profitability": 2.0, "capital_allocation": None, "earnings_quality": None, "growth": 0.5}
    feat = {"revenue_growth_accel": 1.5, "operating_margin_change_1y": None}
    p = N.profiles(cat, feat)
    assert p["compounder"] == pytest.approx(1.5)
    assert p["inflection"] == pytest.approx(1.0)
    p2 = N.profiles({"quality": 1.0}, {})
    assert p2 == {"compounder": None, "inflection": None}


# ---------------------------------------------------------------------------
# normalize_universe (end to end, pure)
# ---------------------------------------------------------------------------


def _random_universe(rng: random.Random, n: int, sectors, null_rate=0.15):
    raw = {}
    sector = {}
    for i in range(n):
        t = f"T{i:03d}"
        sector[t] = sectors[i % len(sectors)]
        raw[t] = {f.name: (rng.gauss(0, 1) if rng.random() > null_rate else None) for f in S.FEATURE_SPEC}
    return raw, sector


def test_normalize_universe_applies_sign_after_z():
    rng = random.Random(0)
    raw, sector = _random_universe(rng, 30, ["Technology"], null_rate=0.0)
    u = N.normalize_universe(raw, sector)
    # opex_ratio has sign -1: the name with the highest raw must have the lowest z
    worst = max(raw, key=lambda t: raw[t]["opex_ratio"])
    assert u.rows[worst].feature_z["opex_ratio"] == min(r.feature_z["opex_ratio"] for r in u.rows.values())
    best = max(raw, key=lambda t: raw[t]["roic"])
    assert u.rows[best].feature_z["roic"] == max(r.feature_z["roic"] for r in u.rows.values())


def test_normalize_universe_z_is_winsorized_clipped_and_sector_based():
    raw = {f"T{i}": {name: float(i) for name in S.FEATURE_NAMES} for i in range(40)}
    raw["T0"]["roic"] = -1000.0                          # tail: winsorized, not dropped
    sector = {t: "Technology" for t in raw}
    u = N.normalize_universe(raw, sector)
    zs = [u.rows[t].feature_z["roic"] for t in raw]
    assert all(-3.0 <= z <= 3.0 for z in zs)
    assert u.rows["T0"].feature_z["roic"] == min(zs)
    assert u.rows["T0"].feature_basis["roic"] == "sector:Information Technology"
    assert u.notes["feature_n"]["roic"] == 40
    assert u.notes["sectors_unmatched"] == {}


def test_normalize_universe_sector_fallback_and_unmatched_notes():
    rng = random.Random(0)
    raw, sector = _random_universe(rng, 24, ["Technology", "Technology", "Financial Services", "Bogus"], null_rate=0.0)
    # 12 tech, 6 financials, 6 bogus; min_sector_n=5 keeps financials neutralised…
    u = N.normalize_universe(raw, sector)
    fin = next(t for t in raw if sector[t] == "Financial Services")
    assert u.rows[fin].sector == S.SECTOR_FINANCIALS
    assert u.rows[fin].feature_basis["earnings_yield"] == "sector:Financials"
    assert u.rows[fin].feature_basis["net_debt_to_ebitda"] == "n/a:excluded"
    bogus = next(t for t in raw if sector[t] == "Bogus")
    assert u.rows[bogus].sector is None
    assert u.rows[bogus].feature_basis["roic"] == N.BASIS_UNIVERSE_UNMATCHED
    assert u.rows[bogus].notes == ["sector_unmatched:Bogus"]
    assert u.rows[bogus].percentile_sector is None
    assert u.notes["sectors_unmatched"] == {"Bogus": 6}
    assert u.notes["sector_fallback_feature_rows"] == {}
    # … and a higher min_n pushes them to the universe with a note
    u8 = N.normalize_universe(raw, sector, params=NormalizationParams(min_sector_n=8))
    assert u8.rows[fin].feature_basis["earnings_yield"] == N.BASIS_UNIVERSE_SMALL_SECTOR
    assert u8.notes["sector_fallback_feature_rows"][S.SECTOR_FINANCIALS] == 6 * 23


def test_normalize_universe_property_over_random_universes():
    rng = random.Random(0)
    sectors = ["Technology", "Financial Services", "Healthcare", "Utilities", "Real Estate", "Energy", "Bogus", None]
    for trial in range(12):
        n = rng.randint(12, 90)
        raw, sector = _random_universe(rng, n, sectors, null_rate=rng.choice([0.0, 0.15, 0.4]))
        u = N.normalize_universe(raw, sector)
        assert set(u.rows) == set(raw)
        assert u.notes["n_scored"] + u.notes["n_insufficient"] == n
        scored = [r for r in u.rows.values() if r.overall_z is not None]
        assert u.notes["n_scored"] == len(scored)
        for r in u.rows.values():
            for name, z in r.feature_z.items():
                if z is not None:
                    assert -3.0 <= z <= 3.0
                    assert raw[r.ticker][name] is not None
                elif raw[r.ticker][name] is not None and name in S.applicable_features(r.sector):
                    # a raw value that got no z means the universe sample was too small
                    assert r.feature_basis[name] == "n/a:universe_n<2"
            if r.overall_z is None:
                assert r.overall_score is None and r.percentile_universe is None
                assert r.contributions == {}
                assert any(nt.startswith("insufficient_data:") for nt in r.notes)
                continue
            assert math.isclose(sum(r.contributions.values()), r.overall_z, rel_tol=0, abs_tol=1e-9)
            assert 0.0 <= r.overall_score <= 100.0
            assert 0.0 < r.percentile_universe <= 100.0
            assert 0.0 < r.coverage <= 1.0
            if r.percentile_sector is not None:
                assert 0.0 < r.percentile_sector <= 100.0
        if scored:
            assert max(r.percentile_universe for r in scored) == 100.0, trial


def test_normalize_universe_returns_insufficient_rows_instead_of_dropping_them():
    rng = random.Random(0)
    raw, sector = _random_universe(rng, 20, ["Technology"], null_rate=0.0)
    raw["T000"] = {name: None for name in S.FEATURE_NAMES}
    raw["T000"]["roic"] = 0.1                              # one feature only
    u = N.normalize_universe(raw, sector)
    row = u.rows["T000"]
    assert row.overall_z is None and row.overall_score is None
    assert row.coverage == pytest.approx(1 / 31)
    assert row.notes == ["insufficient_data:categories=0<5"]
    assert row.percentile_universe is None
    assert u.notes["n_insufficient"] == 1
    others = [r.percentile_universe for t, r in u.rows.items() if t != "T000"]
    assert sorted(others) == [pytest.approx(100.0 * k / 19) for k in range(1, 20)]


def test_normalize_universe_percentiles_and_score_scale():
    raw = {f"T{i}": {name: float(i) for name in S.FEATURE_NAMES} for i in range(10)}
    sector = {t: "Healthcare" for t in raw}
    u = N.normalize_universe(raw, sector)
    # monotone inputs → monotone overall z (signs flip 8 of 31 features, but
    # the positive majority wins) and a perfectly ranked percentile ladder
    order = sorted(raw, key=lambda t: u.rows[t].overall_z)
    assert [u.rows[t].percentile_universe for t in order] == [pytest.approx(10.0 * k) for k in range(1, 11)]
    assert [u.rows[t].percentile_sector for t in order] == [pytest.approx(10.0 * k) for k in range(1, 11)]
    top = u.rows[order[-1]]
    assert top.overall_score > 50.0 > u.rows[order[0]].overall_score
    assert top.overall_z_universe == pytest.approx(top.overall_z)   # single sector = universe
    assert top.category_z_universe == pytest.approx(top.category_z)
    assert top.profiles["compounder"] is not None and top.profiles["inflection"] is not None
    assert u.notes["sector_counts"] == {S.SECTOR_HEALTH_CARE: 10}


def test_normalize_universe_accepts_precomputed_applicable_sets():
    raw = {f"T{i}": {name: float(i) for name in S.FEATURE_NAMES} for i in range(10)}
    sector = {t: "Financial Services" for t in raw}
    fin = S.applicable_features(S.SECTOR_FINANCIALS)
    u = N.normalize_universe(raw, sector, applicable_by_ticker={t: fin for t in raw})
    assert u.rows["T3"].n_applicable == 23
    assert u.rows["T3"].feature_z["net_debt_to_ebitda"] is None
    assert u.rows["T3"].coverage == 1.0
